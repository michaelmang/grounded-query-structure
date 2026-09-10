"""OpenAI Chat Completions provider for bounded topic-map extension."""

from __future__ import annotations

import json
import os
from typing import Any

from .heuristic import normalize_query
from .models import QueryStructure, StructureError, TopicMatch
from .prompts import RESPONSE_SCHEMA, build_messages
from .topic_map import TopicMap

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TIMEOUT = 20.0


def _clean_strings(values: Any, *, limit: int = 8) -> tuple[str, ...]:
    if not isinstance(values, list):
        return ()
    cleaned: list[str] = []
    for item in values:
        if not isinstance(item, str):
            continue
        text = " ".join(item.split()).strip()
        if text and text not in cleaned:
            cleaned.append(text[:200])
        if len(cleaned) >= limit:
            break
    return tuple(cleaned)


def parse_structure_payload(
    original: str,
    payload: dict[str, Any],
    *,
    model: str,
    matched_topic_ids: tuple[str, ...] = (),
) -> QueryStructure:
    intent = payload.get("intent")
    if not isinstance(intent, str) or not intent.strip():
        raise StructureError("OpenAI response missing intent")
    intent_text = " ".join(intent.split()).strip()[:300]
    return QueryStructure(
        original=original,
        intent=intent_text,
        emphasize=_clean_strings(payload.get("emphasize"), limit=8),
        de_emphasize=_clean_strings(payload.get("de_emphasize"), limit=8),
        historical_expressions=_clean_strings(
            payload.get("historical_expressions"), limit=8
        ),
        contrasts=_clean_strings(payload.get("contrasts"), limit=8),
        added_concepts=_clean_strings(payload.get("added_concepts"), limit=4),
        matched_topic_ids=matched_topic_ids,
        source="openai",
        model=model,
    )


class OpenAIStructurer:
    """Call a cheap OpenAI model with a constrained JSON schema."""

    source = "openai"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = DEFAULT_TIMEOUT,
        http_client: Any | None = None,
        match_limit: int = 5,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._http_client = http_client
        self.match_limit = max(1, match_limit)

    def available(self) -> bool:
        return bool(self.api_key)

    def structure(
        self,
        query: str,
        *,
        topic_map: TopicMap | None = None,
        matches: tuple[TopicMatch, ...] | None = None,
        domain_context: str = "",
    ) -> QueryStructure:
        original = normalize_query(query)
        resolved_map = topic_map or TopicMap()
        resolved_matches = matches
        if resolved_matches is None:
            resolved_matches = resolved_map.match(original, limit=self.match_limit)
        topic_ids = tuple(match.entry.id for match in resolved_matches)

        if not original:
            return QueryStructure(
                original="",
                intent="",
                emphasize=(),
                de_emphasize=(),
                historical_expressions=(),
                contrasts=(),
                added_concepts=(),
                matched_topic_ids=(),
                source=self.source,
                model=self.model,
            )
        if not self.api_key:
            raise StructureError("OPENAI_API_KEY is not set")

        body = {
            "model": self.model,
            "temperature": 0,
            "messages": build_messages(
                original,
                topic_map=resolved_map,
                matches=resolved_matches,
                domain_context=domain_context,
            ),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "query_structure",
                    "strict": True,
                    "schema": RESPONSE_SCHEMA,
                },
            },
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        client = self._http_client
        owns_client = client is None
        if owns_client:
            try:
                import httpx
            except ImportError as exc:  # pragma: no cover - optional extra
                raise StructureError(
                    "Install grounded-query-structure[openai] to use OpenAIStructurer"
                ) from exc
            client = httpx.Client(timeout=self.timeout)
        try:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=body,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise StructureError(f"OpenAI request failed: {exc}") from exc
        finally:
            if owns_client:
                client.close()

        try:
            content = data["choices"][0]["message"]["content"]
            payload = json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise StructureError("OpenAI response was not valid structured JSON") from exc
        if not isinstance(payload, dict):
            raise StructureError("OpenAI response JSON must be an object")
        return parse_structure_payload(
            original,
            payload,
            model=self.model,
            matched_topic_ids=topic_ids,
        )

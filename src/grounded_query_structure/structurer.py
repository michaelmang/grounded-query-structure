"""Facade: topic-map match, then LLM extension with heuristic fallback."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from threading import Lock
from typing import Protocol

from .cache import FileStructureCache
from .heuristic import HeuristicStructurer, normalize_query
from .models import QueryStructure, StructureError, TopicMatch
from .openai_provider import DEFAULT_MODEL, OpenAIStructurer
from .topic_map import TopicMap


class StructureProvider(Protocol):
    def structure(
        self,
        query: str,
        *,
        topic_map: TopicMap | None = None,
        matches: tuple[TopicMatch, ...] | None = None,
        domain_context: str = "",
    ) -> QueryStructure: ...


def _map_fingerprint(topic_map: TopicMap) -> str:
    payload = json.dumps(topic_map.to_dict(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class QueryStructurer:
    """Topic-map match + bounded expansion for hybrid search applications.

    Flow:
    1. Match the query against a static topic map.
    2. Ask an LLM (when configured) to extend only that matched subset.
    3. Fall back to copying map fields via the heuristic planner.

    This class never searches a corpus and never generates an answer.
    """

    def __init__(
        self,
        *,
        topic_map: TopicMap | None = None,
        provider: StructureProvider | None = None,
        fallback: StructureProvider | None = None,
        domain_context: str = "",
        match_limit: int = 5,
        cache_size: int = 256,
        file_cache: FileStructureCache | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        self.topic_map = topic_map or TopicMap()
        self.provider = provider
        self.fallback = fallback or HeuristicStructurer(match_limit=match_limit)
        self.domain_context = domain_context
        self.match_limit = max(1, match_limit)
        self.cache_size = max(0, cache_size)
        self.file_cache = file_cache
        self.on_error = on_error
        self._cache: dict[str, QueryStructure] = {}
        self._lock = Lock()
        self._map_fp = _map_fingerprint(self.topic_map)

    @classmethod
    def from_env(
        cls,
        *,
        topic_map: TopicMap | None = None,
        domain_context: str = "",
        model: str | None = None,
        api_key: str | None = None,
        fallback: StructureProvider | None = None,
        match_limit: int = 5,
        cache_size: int = 256,
        cache_path: str | None = None,
        file_cache_max_entries: int = 2000,
        enabled: bool = True,
        on_error: Callable[[Exception], None] | None = None,
    ) -> QueryStructurer:
        provider: StructureProvider | None = None
        if enabled:
            openai = OpenAIStructurer(
                api_key=api_key,
                model=model or DEFAULT_MODEL,
                match_limit=match_limit,
            )
            if openai.available():
                provider = openai
        file_cache = (
            FileStructureCache(cache_path, max_entries=file_cache_max_entries)
            if cache_path
            else None
        )
        return cls(
            topic_map=topic_map,
            provider=provider,
            fallback=fallback,
            domain_context=domain_context,
            match_limit=match_limit,
            cache_size=cache_size,
            file_cache=file_cache,
            on_error=on_error,
        )

    def match(self, query: str) -> tuple[TopicMatch, ...]:
        return self.topic_map.match(normalize_query(query), limit=self.match_limit)

    def structure(self, query: str) -> QueryStructure:
        original = normalize_query(query)
        matches = self.topic_map.match(original, limit=self.match_limit)
        cache_key = f"{self._map_fp}\n{self.domain_context}\n{original}"
        if self.cache_size:
            with self._lock:
                cached = self._cache.get(cache_key)
            if cached is not None:
                return QueryStructure(
                    original=cached.original,
                    intent=cached.intent,
                    emphasize=cached.emphasize,
                    de_emphasize=cached.de_emphasize,
                    historical_expressions=cached.historical_expressions,
                    contrasts=cached.contrasts,
                    added_concepts=cached.added_concepts,
                    matched_topic_ids=cached.matched_topic_ids,
                    source="cache",
                    model=cached.model,
                )

        if self.file_cache is not None:
            durable = self.file_cache.get(cache_key)
            if durable is not None:
                if self.cache_size:
                    with self._lock:
                        self._cache[cache_key] = durable
                        while len(self._cache) > self.cache_size:
                            self._cache.pop(next(iter(self._cache)))
                return durable

        result: QueryStructure | None = None
        if self.provider is not None:
            try:
                result = self.provider.structure(
                    original,
                    topic_map=self.topic_map,
                    matches=matches,
                    domain_context=self.domain_context,
                )
            except Exception as exc:
                if self.on_error is not None:
                    self.on_error(exc)
                if self.fallback is None:
                    raise StructureError(str(exc)) from exc
                result = None

        if result is None:
            result = self.fallback.structure(
                original,
                topic_map=self.topic_map,
                matches=matches,
                domain_context=self.domain_context,
            )

        if self.cache_size and result.original:
            with self._lock:
                self._cache[cache_key] = result
                while len(self._cache) > self.cache_size:
                    self._cache.pop(next(iter(self._cache)))
        if self.file_cache is not None and result.original and result.source != "cache":
            self.file_cache.set(cache_key, result)
        return result

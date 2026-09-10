"""Dependency-free heuristic expansion from topic-map matches."""

from __future__ import annotations

import re

from .models import QueryStructure, TopicMatch
from .topic_map import TopicMap

_FLUFF = re.compile(
    r"\b(?:what|who|whom|which|where|when|why|how|did|does|do|is|are|was|were|"
    r"the|a|an|please|explain|describe|tell|me|us|about|concerning|regarding|"
    r"according|to|of|on|for|with|from|into|their|they|them|this|that)\b",
    re.I,
)


def normalize_query(query: str) -> str:
    cleaned = " ".join(query.split()).strip()
    return cleaned[:500]


def strip_question_fluff(query: str) -> str:
    stripped = _FLUFF.sub(" ", query)
    stripped = re.sub(r"[^\w\s\-']+", " ", stripped, flags=re.UNICODE)
    return re.sub(r"\s+", " ", stripped).strip()


def _dedupe(values: list[str], *, limit: int) -> tuple[str, ...]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = " ".join(item.split()).strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return tuple(cleaned)


class HeuristicStructurer:
    """Map-grounded planner: copy matched topic fields; no invented concepts."""

    source = "heuristic"

    def __init__(self, *, match_limit: int = 5) -> None:
        self.match_limit = max(1, match_limit)

    def structure(
        self,
        query: str,
        *,
        topic_map: TopicMap | None = None,
        matches: tuple[TopicMatch, ...] | None = None,
        domain_context: str = "",
    ) -> QueryStructure:
        del domain_context
        original = normalize_query(query)
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
            )

        resolved_map = topic_map or TopicMap()
        resolved_matches = matches
        if resolved_matches is None:
            resolved_matches = resolved_map.match(original, limit=self.match_limit)

        topical = strip_question_fluff(original) or original
        emphasize: list[str] = []
        historical: list[str] = []
        contrasts: list[str] = []
        topic_ids: list[str] = []
        for match in resolved_matches:
            topic_ids.append(match.entry.id)
            emphasize.extend(match.entry.core_concepts)
            emphasize.extend(match.matched_labels)
            historical.extend(match.entry.historical_expressions)
            contrasts.extend(match.entry.contrasts)

        if not emphasize:
            emphasize = [topical]

        intent = topical
        if resolved_matches:
            primary = resolved_matches[0].entry
            label = primary.labels[0] if primary.labels else primary.id
            intent = f"passages about {label}"

        return QueryStructure(
            original=original,
            intent=intent,
            emphasize=_dedupe(emphasize, limit=8),
            de_emphasize=(),
            historical_expressions=_dedupe(historical, limit=8),
            contrasts=_dedupe(contrasts, limit=8),
            added_concepts=(),
            matched_topic_ids=tuple(topic_ids),
            source=self.source,
        )

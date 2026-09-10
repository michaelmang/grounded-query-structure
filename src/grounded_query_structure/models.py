"""Typed structures produced by topic-map match + bounded LLM extension."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


class StructureError(RuntimeError):
    """Raised when a structure provider fails and no fallback is configured."""


@dataclass(frozen=True, slots=True)
class TopicEntry:
    """One node in a static corpus topic map."""

    id: str
    labels: tuple[str, ...]
    core_concepts: tuple[str, ...] = ()
    historical_expressions: tuple[str, ...] = ()
    contrasts: tuple[str, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TopicMatch:
    """A topic-map entry that matched the user query."""

    entry: TopicEntry
    score: float
    matched_labels: tuple[str, ...]

    @property
    def id(self) -> str:
        return self.entry.id

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.entry.id,
            "score": self.score,
            "matched_labels": list(self.matched_labels),
            "entry": self.entry.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class QueryStructure:
    """Bounded expansion plan for expanded semantic search.

    Produced by matching a static topic map, then optionally extending that
    match with an LLM that may only invent a small number of added concepts.
    This object never contains answers, citations, or retrieved passages.
    """

    original: str
    intent: str
    emphasize: tuple[str, ...]
    de_emphasize: tuple[str, ...]
    historical_expressions: tuple[str, ...]
    contrasts: tuple[str, ...]
    added_concepts: tuple[str, ...]
    matched_topic_ids: tuple[str, ...]
    source: str
    model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["expanded_semantic_queries"] = list(self.expanded_semantic_queries())
        payload["lexical_phrases"] = list(self.lexical_phrases())
        return payload

    def lexical_phrases(self) -> tuple[str, ...]:
        """Corpus-native phrases useful for optional FTS alongside expansion."""
        return _dedupe(
            (
                *self.emphasize,
                *self.historical_expressions,
                *self.added_concepts,
            ),
            limit=12,
        )

    def lexical_or(self) -> str:
        """Join lexical phrases for Postgres ``websearch_to_tsquery`` OR search."""
        phrases = self.lexical_phrases()
        if not phrases:
            return self.original
        return " OR ".join(phrases)

    def expanded_semantic_queries(self, *, limit: int = 8) -> tuple[str, ...]:
        """Derive multiple short semantic-search strings from the expansion fields."""
        queries: list[str] = []
        if self.intent.strip():
            queries.append(self.intent.strip())
        for phrase in self.emphasize:
            queries.append(phrase)
        for phrase in self.historical_expressions:
            queries.append(phrase)
        for phrase in self.contrasts:
            if self.intent.strip():
                queries.append(f"{self.intent.strip()} contrasted with {phrase}")
            else:
                queries.append(phrase)
        for phrase in self.added_concepts:
            queries.append(phrase)
        return _dedupe(queries, limit=limit)


def _dedupe(values: tuple[str, ...] | list[str], *, limit: int) -> tuple[str, ...]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = " ".join(str(item).split()).strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text[:300])
        if len(cleaned) >= limit:
            break
    return tuple(cleaned)

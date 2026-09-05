"""Typed structures produced by the query planner."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


class StructureError(RuntimeError):
    """Raised when a structure provider fails and no fallback is configured."""


@dataclass(frozen=True, slots=True)
class QueryStructure:
    """Corpus-oriented rewrite of a user search question.

    The planner extracts concepts and retrieval forms only. It must not invent
    answers, citations, or passage text.
    """

    original: str
    concepts: tuple[str, ...]
    lexical_phrases: tuple[str, ...]
    semantic_query: str
    source: str
    model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def lexical_or(self) -> str:
        """Join lexical phrases for Postgres ``websearch_to_tsquery`` OR search."""
        phrases = [p.strip() for p in self.lexical_phrases if p and p.strip()]
        if not phrases:
            return self.original
        return " OR ".join(phrases)

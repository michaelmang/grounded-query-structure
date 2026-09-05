"""Dependency-free heuristic structuring when no LLM is available."""

from __future__ import annotations

import re

from .models import QueryStructure

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


class HeuristicStructurer:
    """Passthrough planner: strip question glue, keep the user's topical words."""

    source = "heuristic"

    def structure(self, query: str, *, domain_context: str = "") -> QueryStructure:
        del domain_context  # reserved for domain-specific subclasses / wrappers
        original = normalize_query(query)
        if not original:
            return QueryStructure(
                original="",
                concepts=(),
                lexical_phrases=(),
                semantic_query="",
                source=self.source,
            )
        topical = strip_question_fluff(original) or original
        concepts = tuple(part for part in topical.split() if part)[:8]
        return QueryStructure(
            original=original,
            concepts=concepts or (original,),
            lexical_phrases=(topical,),
            semantic_query=topical,
            source=self.source,
        )

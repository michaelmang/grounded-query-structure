"""Facade: prefer LLM structuring, fall back to a heuristic planner."""

from __future__ import annotations

from collections.abc import Callable
from threading import Lock
from typing import Protocol

from .heuristic import HeuristicStructurer, normalize_query
from .models import QueryStructure, StructureError
from .openai_provider import DEFAULT_MODEL, OpenAIStructurer


class StructureProvider(Protocol):
    def structure(self, query: str, *, domain_context: str = "") -> QueryStructure: ...


class QueryStructurer:
    """First-pass query planner for hybrid search applications.

    The LLM (when configured) extracts concepts and retrieval forms only. It
    never searches a corpus and never generates an answer.
    """

    def __init__(
        self,
        *,
        provider: StructureProvider | None = None,
        fallback: StructureProvider | None = None,
        domain_context: str = "",
        cache_size: int = 256,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        self.provider = provider
        self.fallback = fallback or HeuristicStructurer()
        self.domain_context = domain_context
        self.cache_size = max(0, cache_size)
        self.on_error = on_error
        self._cache: dict[str, QueryStructure] = {}
        self._lock = Lock()

    @classmethod
    def from_env(
        cls,
        *,
        domain_context: str = "",
        model: str | None = None,
        api_key: str | None = None,
        fallback: StructureProvider | None = None,
        cache_size: int = 256,
        enabled: bool = True,
        on_error: Callable[[Exception], None] | None = None,
    ) -> QueryStructurer:
        provider: StructureProvider | None = None
        if enabled:
            openai = OpenAIStructurer(api_key=api_key, model=model or DEFAULT_MODEL)
            if openai.available():
                provider = openai
        return cls(
            provider=provider,
            fallback=fallback,
            domain_context=domain_context,
            cache_size=cache_size,
            on_error=on_error,
        )

    def structure(self, query: str) -> QueryStructure:
        original = normalize_query(query)
        cache_key = f"{self.domain_context}\n{original}"
        if self.cache_size:
            with self._lock:
                cached = self._cache.get(cache_key)
            if cached is not None:
                return QueryStructure(
                    original=cached.original,
                    concepts=cached.concepts,
                    lexical_phrases=cached.lexical_phrases,
                    semantic_query=cached.semantic_query,
                    source="cache",
                    model=cached.model,
                )

        result: QueryStructure | None = None
        if self.provider is not None:
            try:
                result = self.provider.structure(
                    original, domain_context=self.domain_context
                )
            except Exception as exc:
                if self.on_error is not None:
                    self.on_error(exc)
                if self.fallback is None:
                    raise StructureError(str(exc)) from exc
                result = None

        if result is None:
            result = self.fallback.structure(
                original, domain_context=self.domain_context
            )

        if self.cache_size and result.original:
            with self._lock:
                self._cache[cache_key] = result
                while len(self._cache) > self.cache_size:
                    self._cache.pop(next(iter(self._cache)))
        return result

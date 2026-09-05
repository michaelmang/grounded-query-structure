from __future__ import annotations

import json
import unittest
from typing import Any

from grounded_query_structure import (
    HeuristicStructurer,
    OpenAIStructurer,
    QueryStructure,
    QueryStructurer,
    StructureError,
)
from grounded_query_structure.openai_provider import parse_structure_payload


class FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeClient:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, headers: dict[str, str], json: dict[str, Any]) -> FakeResponse:
        self.calls.append({"url": url, "headers": headers, "json": json})
        return FakeResponse(self.payload)

    def close(self) -> None:
        return None


class StructureTests(unittest.TestCase):
    def test_heuristic_strips_question_fluff(self) -> None:
        result = HeuristicStructurer().structure("What did they say about the Logos?")
        self.assertEqual(result.source, "heuristic")
        self.assertIn("Logos", result.semantic_query)
        self.assertNotIn("What", result.semantic_query)

    def test_openai_parses_constrained_json(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "concepts": ["veneration of martyrs", "relics"],
                                "lexical_phrases": [
                                    "veneration of the martyrs",
                                    "relics of the saints",
                                ],
                                "semantic_query": (
                                    "veneration of the martyrs and relics of the saints"
                                ),
                            }
                        )
                    }
                }
            ]
        }
        client = FakeClient(payload)
        provider = OpenAIStructurer(api_key="test-key", http_client=client, model="gpt-4o-mini")
        result = provider.structure(
            "cult of the saints",
            domain_context="Early Church Fathers English translations.",
        )
        self.assertEqual(result.source, "openai")
        self.assertEqual(result.concepts[0], "veneration of martyrs")
        self.assertIn("veneration of the martyrs", result.lexical_or())
        self.assertEqual(len(client.calls), 1)
        body = client.calls[0]["json"]
        self.assertEqual(body["model"], "gpt-4o-mini")
        self.assertEqual(body["response_format"]["type"], "json_schema")

    def test_facade_falls_back_when_provider_fails(self) -> None:
        class Broken:
            def structure(self, query: str, *, domain_context: str = "") -> QueryStructure:
                raise StructureError("boom")

        errors: list[str] = []
        structurer = QueryStructurer(
            provider=Broken(),
            fallback=HeuristicStructurer(),
            on_error=lambda exc: errors.append(str(exc)),
        )
        result = structurer.structure("incarnation of the Word")
        self.assertEqual(result.source, "heuristic")
        self.assertEqual(errors, ["boom"])

    def test_cache_marks_source(self) -> None:
        calls = {"n": 0}

        class Counting:
            def structure(self, query: str, *, domain_context: str = "") -> QueryStructure:
                calls["n"] += 1
                return QueryStructure(
                    original=query,
                    concepts=("a",),
                    lexical_phrases=("a",),
                    semantic_query="a",
                    source="openai",
                    model="gpt-4o-mini",
                )

        structurer = QueryStructurer(provider=Counting(), cache_size=8)
        first = structurer.structure("trinity")
        second = structurer.structure("trinity")
        self.assertEqual(first.source, "openai")
        self.assertEqual(second.source, "cache")
        self.assertEqual(calls["n"], 1)

    def test_parse_rejects_empty_semantic(self) -> None:
        with self.assertRaises(StructureError):
            parse_structure_payload(
                "q",
                {"concepts": ["a"], "lexical_phrases": ["a"], "semantic_query": "  "},
                model="gpt-4o-mini",
            )


if __name__ == "__main__":
    unittest.main()

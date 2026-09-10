from __future__ import annotations

import json
import unittest
from typing import Any

from grounded_query_structure import (
    FileStructureCache,
    HeuristicStructurer,
    OpenAIStructurer,
    QueryStructure,
    QueryStructurer,
    StructureError,
    TopicEntry,
    TopicMap,
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


def sample_map() -> TopicMap:
    return TopicMap(
        [
            TopicEntry(
                id="cult_of_saints",
                labels=("cult of the saints", "veneration of martyrs"),
                core_concepts=("veneration of the martyrs", "relics of the saints"),
                historical_expressions=("cultus sanctorum", "honor of the martyrs"),
                contrasts=("latria versus dulia", "worship due to God alone"),
            ),
            TopicEntry(
                id="trinity",
                labels=("Trinity", "three persons"),
                core_concepts=("Father", "Son", "Holy Spirit"),
                historical_expressions=("homoousios", "three hypostases"),
                contrasts=("modalism", "Arianism"),
            ),
        ]
    )


class StructureTests(unittest.TestCase):
    def test_topic_map_match_finds_label(self) -> None:
        matches = sample_map().match("What about the cult of the saints?")
        self.assertEqual(matches[0].id, "cult_of_saints")
        self.assertGreater(matches[0].score, 0.9)

    def test_heuristic_copies_map_fields(self) -> None:
        result = HeuristicStructurer().structure(
            "What did they say about the cult of the saints?",
            topic_map=sample_map(),
        )
        self.assertEqual(result.source, "heuristic")
        self.assertEqual(result.matched_topic_ids, ("cult_of_saints",))
        self.assertIn("veneration of the martyrs", result.emphasize)
        self.assertIn("cultus sanctorum", result.historical_expressions)
        self.assertIn("latria versus dulia", result.contrasts)
        self.assertEqual(result.added_concepts, ())
        queries = result.expanded_semantic_queries()
        self.assertTrue(queries)
        self.assertIn("cultus sanctorum", queries)

    def test_openai_parses_bounded_expansion(self) -> None:
        payload = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "intent": "patristic teaching on veneration of the saints",
                                "emphasize": [
                                    "veneration of the martyrs",
                                    "relics of the saints",
                                ],
                                "de_emphasize": ["modern popular piety"],
                                "historical_expressions": [
                                    "cultus sanctorum",
                                    "honor of the martyrs",
                                ],
                                "contrasts": ["latria versus dulia"],
                                "added_concepts": ["intercession of the saints"],
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
            topic_map=sample_map(),
            domain_context="Early Church Fathers English translations.",
        )
        self.assertEqual(result.source, "openai")
        self.assertEqual(result.matched_topic_ids, ("cult_of_saints",))
        self.assertEqual(result.emphasize[0], "veneration of the martyrs")
        self.assertIn("intercession of the saints", result.added_concepts)
        self.assertIn("cultus sanctorum", result.lexical_or())
        self.assertEqual(len(client.calls), 1)
        body = client.calls[0]["json"]
        self.assertEqual(body["model"], "gpt-4o-mini")
        self.assertEqual(body["response_format"]["type"], "json_schema")
        user = body["messages"][1]["content"]
        self.assertIn("cult_of_saints", user)
        self.assertIn("Matched topic-map subset", user)

    def test_facade_falls_back_when_provider_fails(self) -> None:
        class Broken:
            def structure(
                self,
                query: str,
                *,
                topic_map: TopicMap | None = None,
                matches: tuple | None = None,
                domain_context: str = "",
            ) -> QueryStructure:
                raise StructureError("boom")

        errors: list[str] = []
        structurer = QueryStructurer(
            topic_map=sample_map(),
            provider=Broken(),
            fallback=HeuristicStructurer(),
            on_error=lambda exc: errors.append(str(exc)),
        )
        result = structurer.structure("cult of the saints")
        self.assertEqual(result.source, "heuristic")
        self.assertEqual(result.matched_topic_ids, ("cult_of_saints",))
        self.assertEqual(errors, ["boom"])

    def test_cache_marks_source(self) -> None:
        calls = {"n": 0}

        class Counting:
            def structure(
                self,
                query: str,
                *,
                topic_map: TopicMap | None = None,
                matches: tuple | None = None,
                domain_context: str = "",
            ) -> QueryStructure:
                calls["n"] += 1
                return QueryStructure(
                    original=query,
                    intent="about Trinity",
                    emphasize=("Trinity",),
                    de_emphasize=(),
                    historical_expressions=("homoousios",),
                    contrasts=("modalism",),
                    added_concepts=(),
                    matched_topic_ids=("trinity",),
                    source="openai",
                    model="gpt-4o-mini",
                )

        structurer = QueryStructurer(
            topic_map=sample_map(),
            provider=Counting(),
            cache_size=8,
        )
        first = structurer.structure("trinity")
        second = structurer.structure("trinity")
        self.assertEqual(first.source, "openai")
        self.assertEqual(second.source, "cache")
        self.assertEqual(calls["n"], 1)

    def test_file_cache_survives_new_structurer(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        calls = {"n": 0}

        class Counting:
            def structure(
                self,
                query: str,
                *,
                topic_map: TopicMap | None = None,
                matches: tuple | None = None,
                domain_context: str = "",
            ) -> QueryStructure:
                calls["n"] += 1
                return QueryStructure(
                    original=query,
                    intent="Trinity three persons",
                    emphasize=("Trinity",),
                    de_emphasize=(),
                    historical_expressions=("homoousios",),
                    contrasts=("Arianism",),
                    added_concepts=(),
                    matched_topic_ids=("trinity",),
                    source="openai",
                    model="gpt-4o-mini",
                )

        with TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            first = QueryStructurer(
                topic_map=sample_map(),
                provider=Counting(),
                cache_size=0,
                file_cache=FileStructureCache(path),
            )
            first.structure("trinity")
            second = QueryStructurer(
                topic_map=sample_map(),
                provider=Counting(),
                cache_size=0,
                file_cache=FileStructureCache(path),
            )
            hit = second.structure("trinity")
            self.assertEqual(hit.source, "cache")
            self.assertEqual(calls["n"], 1)

    def test_parse_rejects_empty_intent(self) -> None:
        with self.assertRaises(StructureError):
            parse_structure_payload(
                "q",
                {
                    "intent": "  ",
                    "emphasize": [],
                    "de_emphasize": [],
                    "historical_expressions": [],
                    "contrasts": [],
                    "added_concepts": [],
                },
                model="gpt-4o-mini",
            )

    def test_topic_map_roundtrip_json(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        original = sample_map()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "map.json"
            path.write_text(json.dumps(original.to_dict()), encoding="utf-8")
            loaded = TopicMap.from_json_file(path)
        self.assertEqual(loaded.entries[0].id, "cult_of_saints")
        self.assertEqual(loaded.entries[1].historical_expressions[0], "homoousios")


if __name__ == "__main__":
    unittest.main()

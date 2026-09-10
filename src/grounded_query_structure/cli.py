"""CLI for inspecting topic-map grounded query expansions (no search)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .structurer import QueryStructurer
from .topic_map import TopicMap


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="grounded-query-structure",
        description=(
            "Match a query to a static topic map and produce a bounded "
            "retrieval expansion (no search)."
        ),
    )
    parser.add_argument("query", help="Natural-language search input")
    parser.add_argument(
        "--topic-map",
        required=True,
        help="Path to a topic-map JSON file (entries with labels + expansion fields)",
    )
    parser.add_argument(
        "--domain-file",
        help="Optional text file with domain context appended to the system prompt",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="OpenAI model (default: gpt-4o-mini when OPENAI_API_KEY is set)",
    )
    parser.add_argument(
        "--heuristic-only",
        action="store_true",
        help="Skip OpenAI even if OPENAI_API_KEY is set",
    )
    args = parser.parse_args(argv)

    topic_map = TopicMap.from_json_file(args.topic_map)
    domain_context = ""
    if args.domain_file:
        domain_context = Path(args.domain_file).read_text(encoding="utf-8")

    structurer = QueryStructurer.from_env(
        topic_map=topic_map,
        domain_context=domain_context,
        model=args.model,
        enabled=not args.heuristic_only,
    )
    result = structurer.structure(args.query)
    json.dump(result.to_dict(), sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

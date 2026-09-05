"""CLI for inspecting query structures without running search."""

from __future__ import annotations

import argparse
import json
import sys

from .structurer import QueryStructurer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="grounded-query-structure",
        description="Extract retrieval concepts from a search query (no search).",
    )
    parser.add_argument("query", help="Natural-language search input")
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

    domain_context = ""
    if args.domain_file:
        from pathlib import Path

        domain_context = Path(args.domain_file).read_text(encoding="utf-8")

    structurer = QueryStructurer.from_env(
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

"""Default system prompt and JSON shape for structured query extraction."""

from __future__ import annotations

DEFAULT_SYSTEM_PROMPT = """\
You prepare natural-language search questions for hybrid lexical + semantic
retrieval over a document corpus.

Your job is ONLY to extract retrieval structure. You must NOT answer the
question, invent citations, quote sources, or rewrite corpus text.

Return a JSON object with:
- concepts: short atomic topics (1–8 strings)
- lexical_phrases: corpus-native phrases for full-text OR search (1–8 strings).
  Prefer vocabulary the corpus is likely to use, not modern jargon alone.
- semantic_query: one short topical sentence for an embedding model

Keep the user's intent. Expand modern synonyms into older or domain-native
phrasing when that helps retrieval. Prefer OR-friendly phrases over a single
AND of every token.
"""

RESPONSE_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["concepts", "lexical_phrases", "semantic_query"],
    "properties": {
        "concepts": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 8,
        },
        "lexical_phrases": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 8,
        },
        "semantic_query": {
            "type": "string",
            "minLength": 1,
            "maxLength": 500,
        },
    },
}


def build_messages(query: str, *, domain_context: str = "") -> list[dict[str, str]]:
    system = DEFAULT_SYSTEM_PROMPT
    if domain_context.strip():
        system = f"{system}\n\nDomain context:\n{domain_context.strip()}"
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                "Extract retrieval structure for this search input. "
                "Respond with JSON only.\n\n"
                f"{query.strip()}"
            ),
        },
    ]

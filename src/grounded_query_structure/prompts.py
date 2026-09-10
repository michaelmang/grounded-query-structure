"""Default system prompt and JSON shape for bounded topic-map extension."""

from __future__ import annotations

from .models import TopicMatch
from .topic_map import TopicMap

DEFAULT_SYSTEM_PROMPT = """\
You extend a static corpus topic-map match into a bounded retrieval expansion.

Your job is ONLY to produce expansion fields for hybrid search. You must NOT
answer the question, invent citations, quote sources, or rewrite corpus text.

You are given the user query and the matched topic-map subset only. Ground
emphasize, historical_expressions, and contrasts primarily in that map.
You may add at most a few optional concepts, and only when clearly implied by
the query and compatible with the matched topics.

Return a JSON object with:
- intent: one short sentence naming what the user is looking for
- emphasize: phrases/topics that should weigh more in retrieval (0–8)
- de_emphasize: nearby topics to down-rank or avoid conflating (0–8)
- historical_expressions: corpus-native / older phrasings (0–8)
- contrasts: distinctions or opposing categories that clarify intent (0–8)
- added_concepts: optional extra concepts not already in the map (0–4)

Prefer map vocabulary. Do not invent doctrines, authors, or claims.
If the map is empty, keep lists sparse and added_concepts empty unless the
query itself supplies an unavoidable topical phrase.
"""

RESPONSE_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "intent",
        "emphasize",
        "de_emphasize",
        "historical_expressions",
        "contrasts",
        "added_concepts",
    ],
    "properties": {
        "intent": {
            "type": "string",
            "minLength": 1,
            "maxLength": 300,
        },
        "emphasize": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 0,
            "maxItems": 8,
        },
        "de_emphasize": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 0,
            "maxItems": 8,
        },
        "historical_expressions": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 0,
            "maxItems": 8,
        },
        "contrasts": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 0,
            "maxItems": 8,
        },
        "added_concepts": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 0,
            "maxItems": 4,
        },
    },
}


def build_messages(
    query: str,
    *,
    topic_map: TopicMap,
    matches: tuple[TopicMatch, ...] | list[TopicMatch],
    domain_context: str = "",
) -> list[dict[str, str]]:
    system = DEFAULT_SYSTEM_PROMPT
    if domain_context.strip():
        system = f"{system}\n\nDomain context:\n{domain_context.strip()}"
    bound = topic_map.bound_context(matches)
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                "Extend this topic-map match into a bounded retrieval expansion. "
                "Respond with JSON only.\n\n"
                f"User query:\n{query.strip()}\n\n"
                f"Matched topic-map subset:\n{bound}"
            ),
        },
    ]

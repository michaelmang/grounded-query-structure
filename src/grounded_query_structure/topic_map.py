"""Static topic map loading and lexical matching."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .models import TopicEntry, TopicMatch

_WORD = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)?", re.I)


def _tokens(text: str) -> set[str]:
    return {match.group(0).casefold() for match in _WORD.finditer(text)}


def _clean_strings(values: Any) -> tuple[str, ...]:
    if not isinstance(values, list):
        return ()
    cleaned: list[str] = []
    for item in values:
        if not isinstance(item, str):
            continue
        text = " ".join(item.split()).strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return tuple(cleaned)


def parse_topic_entry(raw: dict[str, Any]) -> TopicEntry:
    topic_id = str(raw.get("id") or "").strip()
    if not topic_id:
        raise ValueError("topic map entry requires a non-empty id")
    labels = _clean_strings(raw.get("labels"))
    if not labels:
        raise ValueError(f"topic {topic_id!r} requires at least one label")
    return TopicEntry(
        id=topic_id,
        labels=labels,
        core_concepts=_clean_strings(raw.get("core_concepts")),
        historical_expressions=_clean_strings(raw.get("historical_expressions")),
        contrasts=_clean_strings(raw.get("contrasts")),
        notes=str(raw.get("notes") or "").strip(),
    )


class TopicMap:
    """Curated static map of corpus topics used to ground query expansion."""

    def __init__(self, entries: tuple[TopicEntry, ...] | list[TopicEntry] = ()) -> None:
        self.entries = tuple(entries)
        seen: set[str] = set()
        for entry in self.entries:
            if entry.id in seen:
                raise ValueError(f"duplicate topic id: {entry.id!r}")
            seen.add(entry.id)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TopicMap:
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, list):
            raise ValueError("topic map JSON must contain an 'entries' array")
        entries = [parse_topic_entry(item) for item in raw_entries if isinstance(item, dict)]
        return cls(entries)

    @classmethod
    def from_json_file(cls, path: str | Path) -> TopicMap:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("topic map JSON root must be an object")
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return {"entries": [entry.to_dict() for entry in self.entries]}

    def get(self, topic_id: str) -> TopicEntry | None:
        for entry in self.entries:
            if entry.id == topic_id:
                return entry
        return None

    def match(self, query: str, *, limit: int = 5) -> tuple[TopicMatch, ...]:
        """Return ranked topic matches using label substring / token overlap."""
        query_norm = " ".join(query.split()).strip().casefold()
        if not query_norm or limit <= 0:
            return ()
        query_tokens = _tokens(query_norm)
        scored: list[TopicMatch] = []
        for entry in self.entries:
            matched_labels: list[str] = []
            best = 0.0
            for label in entry.labels:
                label_norm = " ".join(label.split()).strip().casefold()
                if not label_norm:
                    continue
                label_tokens = _tokens(label_norm)
                score = 0.0
                if label_norm in query_norm or query_norm in label_norm:
                    score = 1.0
                elif label_tokens and query_tokens:
                    overlap = len(label_tokens & query_tokens) / len(label_tokens)
                    if overlap >= 0.5:
                        score = 0.5 + (0.5 * overlap)
                if score > 0:
                    matched_labels.append(label)
                    best = max(best, score)
            if best > 0:
                scored.append(
                    TopicMatch(
                        entry=entry,
                        score=best,
                        matched_labels=tuple(matched_labels),
                    )
                )
        scored.sort(key=lambda item: (-item.score, item.entry.id))
        return tuple(scored[:limit])

    def bound_context(self, matches: tuple[TopicMatch, ...] | list[TopicMatch]) -> str:
        """Serialize matched topics for a bounded LLM system/user prompt."""
        if not matches:
            return "No topic-map matches. Stay extremely conservative; prefer empty added_concepts."
        blocks: list[str] = []
        for match in matches:
            entry = match.entry
            lines = [
                f"id: {entry.id}",
                f"matched_labels: {', '.join(match.matched_labels)}",
                f"core_concepts: {', '.join(entry.core_concepts) or '(none)'}",
                f"historical_expressions: {', '.join(entry.historical_expressions) or '(none)'}",
                f"contrasts: {', '.join(entry.contrasts) or '(none)'}",
            ]
            if entry.notes:
                lines.append(f"notes: {entry.notes}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

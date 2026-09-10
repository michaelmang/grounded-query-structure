"""Optional durable cache for structured expansions (avoids repeat LLM calls)."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .models import QueryStructure


class FileStructureCache:
    """Simple JSON file cache keyed by normalized query + map fingerprint."""

    def __init__(self, path: str | Path, *, max_entries: int = 2000) -> None:
        self.path = Path(path)
        self.max_entries = max(1, max_entries)
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return
        if isinstance(payload, dict):
            self._data = {
                str(key): value
                for key, value in payload.items()
                if isinstance(value, dict)
            }

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=0, sort_keys=True),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def get(self, key: str) -> QueryStructure | None:
        with self._lock:
            raw = self._data.get(key)
        if not raw:
            return None
        try:
            original = str(raw.get("original") or "")
            intent = str(raw.get("intent") or "")
            if not original or not intent:
                return None
            return QueryStructure(
                original=original,
                intent=intent,
                emphasize=tuple(str(item) for item in raw.get("emphasize") or ()),
                de_emphasize=tuple(str(item) for item in raw.get("de_emphasize") or ()),
                historical_expressions=tuple(
                    str(item) for item in raw.get("historical_expressions") or ()
                ),
                contrasts=tuple(str(item) for item in raw.get("contrasts") or ()),
                added_concepts=tuple(
                    str(item) for item in raw.get("added_concepts") or ()
                ),
                matched_topic_ids=tuple(
                    str(item) for item in raw.get("matched_topic_ids") or ()
                ),
                source="cache",
                model=raw.get("model"),
            )
        except Exception:
            return None

    def set(self, key: str, value: QueryStructure) -> None:
        if not value.original or not value.intent:
            return
        record = {
            "original": value.original,
            "intent": value.intent,
            "emphasize": list(value.emphasize),
            "de_emphasize": list(value.de_emphasize),
            "historical_expressions": list(value.historical_expressions),
            "contrasts": list(value.contrasts),
            "added_concepts": list(value.added_concepts),
            "matched_topic_ids": list(value.matched_topic_ids),
            "model": value.model,
            "provider_source": value.source,
        }
        with self._lock:
            self._data.pop(key, None)
            self._data[key] = record
            while len(self._data) > self.max_entries:
                self._data.pop(next(iter(self._data)))
            self._save()

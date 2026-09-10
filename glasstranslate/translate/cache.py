"""Thread-safe LRU cache for translations keyed on ``(text, src, tgt, context)``.

The pipeline consults the cache before every translation call so unchanged
segments never hit the model twice.  ``context`` is the translator's
:meth:`~glasstranslate.core.interfaces.Translator.context_key` (series name /
prompt template digest for prompt-driven backends, ``""`` otherwise) so a line
translated for one series is never served for another.  Optional JSON
persistence lets the cache survive restarts; version-1 files (no context) load
with ``context=""``.
"""
from __future__ import annotations

import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

Key = Tuple[str, str, str, str]

CACHE_FORMAT_VERSION = 2


class TranslationCache:
    """Least-recently-used mapping ``(text, src, tgt, context) -> translation``.

    All public methods hold an internal :class:`threading.Lock`, so a single
    instance may be shared between the pipeline thread and the UI thread.
    """

    def __init__(self, max_size: int = 5000) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self._data: "OrderedDict[Key, str]" = OrderedDict()
        self._lock = threading.Lock()

    # ------------------------------------------------------------ core API
    def get(self, text: str, src: str, tgt: str, context: str = "") -> Optional[str]:
        """Return the cached translation or ``None``; a hit marks the entry as
        most recently used."""
        key = (text, src, tgt, context)
        with self._lock:
            value = self._data.get(key)
            if value is None:
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return value

    def put(self, text: str, src: str, tgt: str, translation: str, context: str = "") -> None:
        """Insert or refresh an entry, evicting the least recently used one
        when the cache is full."""
        key = (text, src, tgt, context)
        with self._lock:
            self._data[key] = translation
            self._data.move_to_end(key)
            while len(self._data) > self.max_size:
                self._data.popitem(last=False)

    def clear(self) -> None:
        """Drop all entries (counters are kept)."""
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def __contains__(self, key: object) -> bool:
        if isinstance(key, tuple) and len(key) == 3:
            key = (*key, "")
        with self._lock:
            return key in self._data

    def stats(self) -> Dict[str, Union[int, float]]:
        """Snapshot of counters: ``size``, ``max_size``, ``hits``, ``misses``,
        ``hit_rate`` (0..1, 0 when nothing was looked up yet)."""
        with self._lock:
            lookups = self.hits + self.misses
            return {
                "size": len(self._data),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": (self.hits / lookups) if lookups else 0.0,
            }

    # --------------------------------------------------------- persistence
    def save(self, path: Union[str, Path]) -> Path:
        """Write the entries (oldest first) as JSON.  The file is written to a
        temporary sibling and renamed so a crash never leaves a torn file."""
        path = Path(path)
        with self._lock:
            entries = [
                {"text": t, "src": s, "tgt": g, "context": c, "translation": v}
                for (t, s, g, c), v in self._data.items()
            ]
        payload = {"version": CACHE_FORMAT_VERSION, "max_size": self.max_size, "entries": entries}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        return path

    def load(self, path: Union[str, Path]) -> int:
        """Merge entries from a file written by :meth:`save`.  Returns the
        number of entries loaded; a missing or corrupt file loads nothing.
        Version-1 entries (no ``context``) load with ``context=""``."""
        path = Path(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            entries = payload["entries"]
        except (OSError, ValueError, KeyError, TypeError):
            return 0
        loaded = 0
        with self._lock:
            for entry in entries:
                try:
                    key = (
                        str(entry["text"]),
                        str(entry["src"]),
                        str(entry["tgt"]),
                        str(entry.get("context", "") or ""),
                    )
                    self._data[key] = str(entry["translation"])
                except (KeyError, TypeError, AttributeError):
                    continue
                self._data.move_to_end(key)
                loaded += 1
            while len(self._data) > self.max_size:
                self._data.popitem(last=False)
        return loaded

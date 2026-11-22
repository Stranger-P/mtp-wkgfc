"""Tiny persistent key-value cache (SQLite). Every SPARQL query, search and LLM call goes
through it, so re-running an experiment costs nothing and results are reproducible."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Callable


class Cache:
    def __init__(self, directory: str, name: str):
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.path = Path(directory) / f"{name}.sqlite"
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")
        self._conn.commit()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(*parts: Any) -> str:
        raw = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str):
        with self._lock:
            row = self._conn.execute("SELECT v FROM kv WHERE k=?", (key,)).fetchone()
        return None if row is None else json.loads(row[0])

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)",
                (key, json.dumps(value, ensure_ascii=False)),
            )
            self._conn.commit()

    def get_or_compute(self, key: str, fn: Callable[[], Any]):
        val = self.get(key)
        if val is not None:
            self.hits += 1
            return val
        self.misses += 1
        val = fn()
        self.set(key, val)
        return val


class NullCache(Cache):
    """Cache that stores nothing (used in tests)."""

    def __init__(self):  # noqa: D401 - no DB
        self.hits = self.misses = 0

    def get(self, key):
        return None

    def set(self, key, value):
        pass

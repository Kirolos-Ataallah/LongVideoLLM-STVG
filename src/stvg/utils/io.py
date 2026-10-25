from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)
    tmp.replace(path)


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


class JsonlStore:
    """Append-only JSONL keyed store used for resumable inference and judging.

    Later records override earlier ones with the same key, so retries simply append.
    """

    def __init__(self, path: str | Path, key: str = "qid") -> None:
        self.path = Path(path)
        self.key = key
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            for r in iter_jsonl(self.path):
                self._records[r[key]] = r

    def __contains__(self, key: str) -> bool:
        return key in self._records

    def __len__(self) -> int:
        return len(self._records)

    def get(self, key: str) -> dict[str, Any] | None:
        return self._records.get(key)

    def records(self) -> dict[str, dict[str, Any]]:
        return dict(self._records)

    def add(self, record: dict[str, Any]) -> None:
        self._records[record[self.key]] = record
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()


def load_jsonl_shards(paths: Iterable[Path], key: str = "qid") -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for p in sorted(paths):
        for r in iter_jsonl(p):
            merged[r[key]] = r
    return merged

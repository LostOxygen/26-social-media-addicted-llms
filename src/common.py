"""Shared paths and helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
META_DIR = DATA / "meta"
VIDEO_DIR = DATA / "videos"
RAW_DIR = VIDEO_DIR / "raw"
FRAMES_DIR = VIDEO_DIR / "frames"
PROCESSED_DIR = VIDEO_DIR / "processed"
POOL_FILE = VIDEO_DIR / "pool.jsonl"
MANIFEST_FILE = VIDEO_DIR / "manifest.jsonl"
RUNS = ROOT / "runs"


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]], mode: str = "w") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode) as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    write_jsonl(path, [row], mode="a")

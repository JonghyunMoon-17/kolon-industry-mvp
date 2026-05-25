from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def dump_json(data: Any) -> str:
    if hasattr(data, "model_dump"):
        data = data.model_dump()
    return json.dumps(data, ensure_ascii=False, indent=2)


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def split_text(text: str, max_chars: int = 900, overlap: int = 120) -> list[str]:
    text = normalize_ws(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        window = text[start:end]
        cut = max(window.rfind(". "), window.rfind("\n"), window.rfind("다. "))
        if cut > max_chars * 0.45:
            end = start + cut + 1
            window = text[start:end]
        chunks.append(window.strip())
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return [chunk for chunk in chunks if chunk]

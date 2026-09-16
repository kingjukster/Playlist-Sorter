from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


def normalized(value: str | None) -> str:
    text = unicodedata.normalize("NFKC", value or "")
    return " ".join(text.casefold().split())


def stable_id(prefix: str, *parts: str) -> str:
    payload = "\x1f".join(normalized(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:20]}"


def apple_id_from_url(url: str | None, kind: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    match = re.search(rf"/{re.escape(kind)}/[^/]+/(\d+)$", parsed.path)
    if kind == "album":
        match = re.search(r"/album/[^/]+/(\d+)$", parsed.path)
    if kind == "song" and not match:
        query_id = parse_qs(parsed.query).get("i", [""])[0]
        return query_id if query_id.isdigit() else ""
    return match.group(1) if match else ""


def duration_to_ms(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, int | float):
        return int(value)
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    parts = text.split(":")
    if not parts or any(not part.isdigit() for part in parts):
        return None
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + int(part)
    return seconds * 1000


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )

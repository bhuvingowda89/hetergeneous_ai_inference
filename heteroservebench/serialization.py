"""Deterministic serialization and hashing utilities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_dumps(value: Any) -> str:
    """Return a deterministic JSON representation of a JSON-compatible value."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(text: str) -> str:
    """Return a SHA-256 hex digest for UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_hash(value: Any) -> str:
    """Hash a JSON-compatible value after canonical serialization."""
    return sha256_text(canonical_dumps(value))


def write_json(path: Path, value: Any) -> None:
    """Write pretty JSON for auditability."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    """Read a JSON document."""
    return json.loads(path.read_text(encoding="utf-8"))

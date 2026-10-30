"""Explicit, hashed workflow experiment profiles."""
import hashlib
import json
from pathlib import Path

from .common import ROOT


PROFILE_ROOT = ROOT / "profiles"


def load_profile(name):
    path = PROFILE_ROOT / f"{name}.json"
    if not path.is_file():
        raise ValueError(f"unknown workflow profile: {name}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("name") != name:
        raise ValueError("workflow profile name mismatch")
    canonical = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return {**data, "sha256": hashlib.sha256(canonical).hexdigest(), "path": str(path.resolve())}

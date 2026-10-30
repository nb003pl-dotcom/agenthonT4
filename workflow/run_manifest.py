"""Reproducibility metadata for workflow experiments."""
import hashlib
import subprocess

from .common import ROOT


def _sha256(data):
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()


def git_identity():
    try:
        return _git_identity()
    except (OSError, subprocess.CalledProcessError):
        # A source distribution does not require Git or a surrounding checkout.
        return {"commit": None, "dirty": None, "available": False}


def _git_identity():
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip())
    return {"commit": commit, "dirty": dirty}


def build_run_manifest(*, profile, query_plan_path=None, query_plan_bytes=None, prompt_texts=None):
    return {
        "workflow_profile": profile["name"],
        "profile_sha256": profile["sha256"],
        "profile_path": profile.get("path"),
        "git": git_identity(),
        "query_plan_path": query_plan_path,
        "query_plan_sha256": _sha256(query_plan_bytes) if query_plan_bytes is not None else None,
        "prompt_sha256": {name: _sha256(text) for name, text in sorted((prompt_texts or {}).items())},
    }

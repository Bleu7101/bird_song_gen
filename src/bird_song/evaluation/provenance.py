"""Shared provenance helpers for evaluation artifacts."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(cwd: Path) -> str:
    """Return the current Git revision, or ``unknown`` outside a checkout."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"

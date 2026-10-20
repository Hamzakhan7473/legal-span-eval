"""Refuse to start runs from an uncommitted working tree."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


class DirtyTreeError(RuntimeError):
    pass


def dirty_paths(repo_root: Path = REPO_ROOT) -> list[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo_root, capture_output=True, text=True, check=True
    )
    return [line[3:] for line in result.stdout.splitlines() if line.strip()]


def ensure_clean_tree(*, allow_dirty: bool = False, repo_root: Path = REPO_ROOT) -> bool:
    """Raise unless the tree is clean or `allow_dirty` is set. Returns whether it was dirty.

    Experiment runs (E1-E6) must never pass allow_dirty; only smoke tests may.
    """
    paths = dirty_paths(repo_root)
    if paths and not allow_dirty:
        listing = "\n  ".join(paths[:20])
        raise DirtyTreeError(
            "refusing to start: the working tree has uncommitted changes.\n  "
            f"{listing}\nCommit them first so the run manifest's git SHA describes the code that ran."
        )
    return bool(paths)

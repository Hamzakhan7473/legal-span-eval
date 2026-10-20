"""Per-run call log (data/runs/<run_id>/calls.jsonl) and run manifest."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import threading
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNS_DIR = REPO_ROOT / "data" / "runs"
DEFAULT_CONFIG_PATHS = (
    REPO_ROOT / "configs" / "models.yaml",
    REPO_ROOT / "configs" / "pricing.yaml",
    REPO_ROOT / "configs" / "prompts" / "clause_v1.yaml",
)
DEFAULT_EVAL_SET = REPO_ROOT / "data" / "eval_set" / "eval_set.jsonl"
TRACKED_PACKAGES = (
    "legal-span-eval",
    "boto3",
    "botocore",
    "httpx",
    "numpy",
    "pandas",
    "pydantic",
    "scikit-learn",
    "scipy",
    "matplotlib",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return path.name


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain")
    return {"sha": run("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None}


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def build_manifest(
    run_id: str,
    *,
    config_paths: tuple[Path, ...] = DEFAULT_CONFIG_PATHS,
    eval_set_path: Path | None = DEFAULT_EVAL_SET,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "created_utc": datetime.now(UTC).isoformat(),
        "git": _git_state(),
        "configs": {_relative(p): _sha256(p) for p in config_paths if p.exists()},
        "eval_set": (
            {"path": _relative(eval_set_path), "sha256": _sha256(eval_set_path)}
            if eval_set_path is not None and eval_set_path.exists()
            else None
        ),
        "python": sys.version.split()[0],
        "platform": platform.platform(terse=True),
        "packages": _package_versions(),
    }
    if extra:
        manifest.update(extra)
    return manifest


class RunLog:
    """Appends one JSON line per router call. Thread-safe."""

    def __init__(
        self,
        run_id: str,
        *,
        runs_dir: Path = DEFAULT_RUNS_DIR,
        config_paths: tuple[Path, ...] = DEFAULT_CONFIG_PATHS,
        eval_set_path: Path | None = DEFAULT_EVAL_SET,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.run_id = run_id
        self.dir = runs_dir / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.calls_path = self.dir / "calls.jsonl"
        self.manifest_path = self.dir / "manifest.json"
        self._lock = threading.Lock()
        if not self.manifest_path.exists():
            manifest = build_manifest(
                run_id, config_paths=config_paths, eval_set_path=eval_set_path, extra=extra
            )
            self.manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    def append(self, row: dict[str, Any]) -> None:
        line = json.dumps({"run_id": self.run_id, **row}, sort_keys=True, ensure_ascii=False)
        with self._lock, self.calls_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

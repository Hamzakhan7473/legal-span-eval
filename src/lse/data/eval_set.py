"""Frozen eval set: paths, hashing, and the SHA-256 manifest."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_SET_DIR = REPO_ROOT / "data" / "eval_set"
EVAL_SET_PATH = EVAL_SET_DIR / "eval_set.jsonl"
CUAD_IDS_PATH = EVAL_SET_DIR / "frozen_eval_ids.txt"
MANIFEST_PATH = EVAL_SET_DIR / "MANIFEST.json"
CONFIG_PATH = REPO_ROOT / "configs" / "eval_set.yaml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_eval_set(path: Path = EVAL_SET_PATH) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_cuad_ids(path: Path = CUAD_IDS_PATH) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_manifest() -> dict[str, Any]:
    """Describe the committed eval-set files. All values are computed from the files."""
    examples = read_eval_set()
    cuad_ids = read_cuad_ids()
    by_category = Counter((e["category"], e["present"]) for e in examples)
    categories = sorted({e["category"] for e in examples})
    config = load_config()
    return {
        "files": {
            "eval_set.jsonl": {"sha256": sha256_file(EVAL_SET_PATH), "n_lines": len(examples)},
            "frozen_eval_ids.txt": {"sha256": sha256_file(CUAD_IDS_PATH), "n_lines": len(cuad_ids)},
        },
        "n_examples": len(examples),
        "n_present": sum(1 for e in examples if e["present"]),
        "n_absent": sum(1 for e in examples if not e["present"]),
        "categories": {
            c: {"present": by_category[(c, True)], "absent": by_category[(c, False)]}
            for c in categories
        },
        "n_unique_contract_titles": len({e["contract_title"] for e in examples}),
        "source": config["source"],
        "construction": config["construction"],
    }


def verify_manifest(path: Path = MANIFEST_PATH) -> list[str]:
    """Return mismatches between the committed manifest and the files (empty = ok)."""
    recorded = json.loads(path.read_text(encoding="utf-8"))
    current = build_manifest()
    return [
        f"{key}: manifest={recorded.get(key)!r} current={current[key]!r}"
        for key in current
        if recorded.get(key) != current[key]
    ]


def write_manifest(path: Path = MANIFEST_PATH) -> Path:
    path.write_text(json.dumps(build_manifest(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path

"""Thinking-config pinning, dirty-tree guard, and run provenance."""

from __future__ import annotations

import dataclasses
import json
import subprocess

import pytest

from lse.router import RunLog
from lse.router.providers import GoogleProvider
from lse.router.router import load_model_configs, models_manifest
from lse.router.types import Message
from lse.run.guard import DirtyTreeError, ensure_clean_tree

MESSAGES = [Message("system", "s"), Message("user", "u")]


def test_gemini_is_pinned_to_api_default_thinking():
    cfg = load_model_configs()["google"]
    assert cfg.thinking == "api_default"
    request = GoogleProvider().build_request(cfg, MESSAGES, None, 0.0, None)
    assert "thinkingConfig" not in request["generationConfig"]


def test_explicit_thinking_budget_is_sent():
    cfg = dataclasses.replace(load_model_configs()["google"], thinking=512)
    request = GoogleProvider().build_request(cfg, MESSAGES, None, 0.0, None)
    assert request["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 512}


def test_unknown_thinking_setting_is_rejected():
    cfg = dataclasses.replace(load_model_configs()["google"], thinking="max")
    with pytest.raises(ValueError):
        GoogleProvider().build_request(cfg, MESSAGES, None, 0.0, None)


def test_models_manifest_records_bedrock_profile_and_gemini_thinking():
    manifest = models_manifest(load_model_configs(), ["google", "openai", "bedrock_claude"])
    assert manifest["google"]["thinking"] == "api_default"
    claude = manifest["bedrock_claude"]
    assert claude["inference_profile_id"] == "us.anthropic.claude-sonnet-4-6"
    assert claude["inference_profile_scope"] == "us"
    assert claude["region"] == "us-east-1"
    assert claude["model_version_returned_by_api"] is False


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "--allow-empty", "-m", "init")
    return tmp_path


def test_clean_tree_passes(repo):
    assert ensure_clean_tree(repo_root=repo) is False


def test_dirty_tree_is_refused(repo):
    (repo / "change.txt").write_text("x")
    with pytest.raises(DirtyTreeError):
        ensure_clean_tree(repo_root=repo)


def test_allow_dirty_permits_and_reports(repo):
    (repo / "change.txt").write_text("x")
    assert ensure_clean_tree(allow_dirty=True, repo_root=repo) is True


def test_finalize_records_access_windows(tmp_path):
    run_log = RunLog("r", runs_dir=tmp_path)
    for stamp in ("2026-10-20T10:00:00+00:00", "2026-10-20T10:05:00+00:00"):
        run_log.append({"logged_utc": stamp, "model": "bedrock_claude"})
    manifest = run_log.finalize()
    window = manifest["access_windows"]["bedrock_claude"]
    assert window == {
        "first_call_utc": "2026-10-20T10:00:00+00:00",
        "last_call_utc": "2026-10-20T10:05:00+00:00",
        "n_calls": 2,
    }
    assert "completed_utc" in json.loads(run_log.manifest_path.read_text())

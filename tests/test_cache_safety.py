"""Committed cache and run files must not contain credentials, headers, or request IDs."""

from pathlib import Path

import pytest
from dotenv import dotenv_values

from lse.router.safety import FORBIDDEN_PATTERNS, scan_paths, scan_text

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_OUTPUT_DIRS = [REPO_ROOT / "data" / "cache", REPO_ROOT / "data" / "runs"]


def test_committed_cache_and_runs_contain_no_secrets_or_request_ids():
    findings = scan_paths(COMMITTED_OUTPUT_DIRS)
    report = "\n".join(f"{f.path.relative_to(REPO_ROOT)}:{f.line}: {f.pattern}" for f in findings)
    assert not findings, f"forbidden content found:\n{report}"


def test_local_env_secret_values_never_appear_in_committed_outputs():
    """Catches credentials of any format, not only those matching known key patterns."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        pytest.skip("no local .env")
    secrets = {
        name: value
        for name, value in dotenv_values(env_path).items()
        if value and len(value) >= 12 and any(tag in name for tag in ("KEY", "SECRET", "TOKEN"))
    }
    leaked = []
    for root in COMMITTED_OUTPUT_DIRS:
        for path in (p for p in root.rglob("*") if p.is_file()):
            text = path.read_text(encoding="utf-8", errors="replace")
            leaked.extend(
                f"{path.relative_to(REPO_ROOT)}: value of {name}" for name, value in secrets.items() if value in text
            )
    assert not leaked, "secret values found:\n" + "\n".join(leaked)


# Samples are assembled at runtime so this file never contains a literal secret.
SAMPLES = {
    "openai_key": "sk-" + "proj-" + "a" * 24,
    "google_api_key": "AIza" + "B" * 35,
    "aws_access_key_id": "AKIA" + "C" * 16,
    "aws_secret_key_field": "aws_secret_access_key = x",
    "aws_session_token_field": "AWS_SESSION_TOKEN=x",
    "private_key_block": "-----BEGIN " + "RSA PRIVATE KEY-----",
    "bearer_token": "Bearer " + "d" * 24,
    "auth_header_key": '{"Authorization": "x"}',
    "http_headers_block": '{"HTTPHeaders": {}}',
    "aws_response_metadata": '{"ResponseMetadata": {}}',
    "google_response_id": '{"responseId": "x"}',
    "openai_completion_id": '{"id": "chatcmpl-abc"}',
    "bedrock_tool_use_id": '{"toolUseId": "tooluse_abc"}',
    "openai_request_id": '{"x-request-id": "x"}',
}


def test_every_pattern_has_a_sample():
    assert set(SAMPLES) == set(FORBIDDEN_PATTERNS)


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_pattern_detects_its_sample(name):
    assert (1, name) in scan_text(SAMPLES[name])


@pytest.mark.parametrize(
    "text",
    [
        "MMT may not grant sublicenses without the prior written authorization of SIGA.",
        '{"present": true, "span": "Licensee shall keep accurate books", "confidence": 0.9}',
        '{"model": "gpt-5.4-mini", "usage": {"prompt_tokens": 10}}',
    ],
)
def test_ordinary_content_is_not_flagged(text):
    assert scan_text(text) == []

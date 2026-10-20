"""Patterns that must never appear in committed cache or run files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Credentials, auth headers, and per-request identifiers. Header names are matched as
# JSON keys so ordinary contract prose (e.g. "prior written authorization") is not flagged.
FORBIDDEN_PATTERNS: dict[str, re.Pattern[str]] = {
    "openai_key": re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    "google_api_key": re.compile(r"AIza[0-9A-Za-z_-]{35}"),
    "aws_access_key_id": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "aws_secret_key_field": re.compile(r"aws_secret_access_key", re.IGNORECASE),
    "aws_session_token_field": re.compile(r"aws_session_token", re.IGNORECASE),
    "private_key_block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "bearer_token": re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]{16,}"),
    "auth_header_key": re.compile(
        r'"(?:authorization|x-api-key|x-goog-api-key|api-key|cookie|set-cookie)"\s*:',
        re.IGNORECASE,
    ),
    "http_headers_block": re.compile(r'"(?:HTTPHeaders|headers)"\s*:'),
    "aws_response_metadata": re.compile(r'"(?:ResponseMetadata|RequestId|x-amzn-requestid)"\s*:'),
    "google_response_id": re.compile(r'"responseId"\s*:'),
    "openai_completion_id": re.compile(r'"id"\s*:\s*"chatcmpl-'),
    "openai_request_id": re.compile(r'"(?:x-request-id|openai-organization|openai-project)"\s*:', re.IGNORECASE),
}


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    pattern: str


def scan_text(text: str) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for name, pattern in FORBIDDEN_PATTERNS.items():
            if pattern.search(line):
                hits.append((line_no, name))
    return hits


def scan_paths(roots: list[Path]) -> list[Finding]:
    """Scan every file under the given roots. Reports pattern names, never matched text."""
    findings: list[Finding] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            text = path.read_text(encoding="utf-8", errors="replace")
            findings.extend(Finding(path, line, name) for line, name in scan_text(text))
    return findings

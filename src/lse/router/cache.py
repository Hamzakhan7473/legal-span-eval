"""Content-addressed on-disk response cache.

The key is the SHA-256 of the canonical JSON of {provider, model_id, request}, where
`request` is exactly the payload sent to the provider (messages, parameters, schema).
Credentials travel in headers or the AWS signer and never enter the payload.

Each entry stores only the request payload, the sanitized response body, and timing
metadata. Writes are atomic and deterministic (sorted keys), so a committed cache
reproduces model outputs byte for byte.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def cache_key(provider: str, model_id: str, request: dict[str, Any]) -> str:
    payload = {"provider": provider, "model_id": model_id, "request": request}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CacheEntry:
    key: str
    provider: str
    model_id: str
    request: dict[str, Any]
    body: dict[str, Any]
    latency_ms: float
    provider_attempts: int


class ResponseCache:
    def __init__(self, root: Path = DEFAULT_CACHE_DIR, *, read_only: bool = False) -> None:
        self.root = root
        self.read_only = read_only

    def path_for(self, provider: str, key: str) -> Path:
        return self.root / provider / key[:2] / f"{key}.json"

    def get(self, provider: str, key: str) -> CacheEntry | None:
        path = self.path_for(provider, key)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return CacheEntry(
            key=data["key"],
            provider=data["provider"],
            model_id=data["model_id"],
            request=data["request"],
            body=data["response"]["body"],
            latency_ms=data["meta"]["latency_ms"],
            provider_attempts=data["meta"]["provider_attempts"],
        )

    def put(self, entry: CacheEntry) -> Path:
        if self.read_only:
            raise PermissionError("cache is read-only")
        path = self.path_for(entry.provider, entry.key)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "key": entry.key,
            "provider": entry.provider,
            "model_id": entry.model_id,
            "request": entry.request,
            "response": {"body": entry.body},
            "meta": {"latency_ms": entry.latency_ms, "provider_attempts": entry.provider_attempts},
        }
        text = json.dumps(record, sort_keys=True, indent=1, ensure_ascii=False) + "\n"
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return path

"""Allowlist response bodies before they are cached or logged.

Only fields needed to reproduce the model output, its token usage, and the model version
are kept. Per-request identifiers (OpenAI `id`, Google `responseId`, Bedrock
`ResponseMetadata` with its RequestId and HTTP headers, Bedrock `toolUseId`) are dropped.
"""

from __future__ import annotations

from typing import Any

_OPENAI_TOP = ("model", "system_fingerprint", "choices", "usage")
_OPENAI_CHOICE = ("index", "message", "finish_reason")
_OPENAI_MESSAGE = ("role", "content", "refusal")

_GOOGLE_TOP = ("candidates", "usageMetadata", "modelVersion")
_GOOGLE_CANDIDATE = ("content", "finishReason", "index")

_BEDROCK_TOP = ("output", "stopReason", "usage", "metrics")


def _pick(source: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {k: source[k] for k in keys if k in source}


def sanitize_openai(body: dict[str, Any]) -> dict[str, Any]:
    out = _pick(body, _OPENAI_TOP)
    out["choices"] = [
        {**_pick(choice, _OPENAI_CHOICE), "message": _pick(choice.get("message") or {}, _OPENAI_MESSAGE)}
        for choice in body.get("choices") or []
    ]
    return out


def sanitize_google(body: dict[str, Any]) -> dict[str, Any]:
    out = _pick(body, _GOOGLE_TOP)
    out["candidates"] = [_pick(c, _GOOGLE_CANDIDATE) for c in body.get("candidates") or []]
    return out


def sanitize_bedrock(body: dict[str, Any]) -> dict[str, Any]:
    out = _pick(body, _BEDROCK_TOP)
    message = (out.get("output") or {}).get("message")
    if message is not None:
        content = []
        for block in message.get("content") or []:
            if "toolUse" in block:
                tool_use = {k: v for k, v in block["toolUse"].items() if k != "toolUseId"}
                block = {**block, "toolUse": tool_use}
            content.append(block)
        out["output"] = {**out["output"], "message": {**message, "content": content}}
    return out


SANITIZERS = {
    "openai": sanitize_openai,
    "google": sanitize_google,
    "bedrock": sanitize_bedrock,
}

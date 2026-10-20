"""Defensive JSON extraction and schema validation.

The fence-stripping and object-extraction logic is copied from the original harness
(models/prompt.py) so that parse outcomes match the original runs.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from lse.router.types import OutputSchema

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _FENCE_RE.sub("", stripped).strip()
    return stripped


def extract_json_object(text: str) -> str:
    stripped = _strip_code_fences(text)
    if stripped.startswith("{"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]
    return stripped


def parse_structured(text: str | None, schema: OutputSchema) -> tuple[dict[str, Any] | None, str | None]:
    """Return (parsed, error). Exactly one of the two is None."""
    if text is None:
        return None, "no text to parse"
    try:
        payload: Any = json.loads(extract_json_object(text))
        instance = schema.model.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
        return None, str(exc)
    if schema.check is not None:
        problem = schema.check(instance)
        if problem is not None:
            return None, problem
    return instance.model_dump(), None

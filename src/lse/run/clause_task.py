"""Clause presence + span extraction task: prompts, output schema, and validation rules.

Prompts and the tool schema are loaded from configs/prompts/clause_v1.yaml, which was
exported verbatim from the original harness. The prediction model and the two
cross-field rules are copied from the original harness (models/prompt.py).
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from lse.router.types import Message, OutputSchema

PROMPT_PATH = Path(__file__).resolve().parents[3] / "configs" / "prompts" / "clause_v1.yaml"


class ClausePrediction(BaseModel):
    present: bool
    span: str | None
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


def check_prediction(prediction: BaseModel) -> str | None:
    assert isinstance(prediction, ClausePrediction)
    if prediction.present and not prediction.span:
        return "present=true requires a non-null span"
    if not prediction.present and prediction.span:
        return "present=false requires span=null"
    return None


def normalize_tool_input(payload: dict[str, Any]) -> dict[str, Any]:
    """The tool schema uses an empty string for 'no span'; map it to null."""
    out = dict(payload)
    if out.get("span") == "":
        out["span"] = None
    return out


@lru_cache(maxsize=1)
def load_prompts(path: Path = PROMPT_PATH) -> dict[str, Any]:
    prompts = yaml.safe_load(path.read_text(encoding="utf-8"))
    digests = {
        "system": prompts["system"],
        "user_template": prompts["user_template"],
        "tool_description": prompts["tool"]["description"],
    }
    for field, text in digests.items():
        actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if actual != prompts["sha256"][field]:
            raise ValueError(f"{path.name}: {field} does not match its recorded sha256")
    return prompts


def clause_schema() -> OutputSchema:
    tool = load_prompts()["tool"]
    return OutputSchema(
        name=tool["name"],
        description=tool["description"],
        model=ClausePrediction,
        json_schema=tool["input_schema"],
        check=check_prediction,
        normalize_tool_input=normalize_tool_input,
    )


def build_messages(example: dict[str, Any]) -> list[Message]:
    prompts = load_prompts()
    user = prompts["user_template"].format(
        contract_title=example["contract_title"],
        category=example["category"],
        contract_excerpt=example["contract_excerpt"],
    )
    return [Message("system", prompts["system"]), Message("user", user)]

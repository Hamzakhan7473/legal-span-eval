"""Provider adapters: build the request payload, send it, and read the response.

Request shapes reproduce the original harness (models/runner.py and models/bedrock.py)
so that runs through this router are comparable with the original runs:

- OpenAI: POST /v1/chat/completions with max_completion_tokens, temperature, and
  response_format {"type": "json_object"}.
- Google: POST /v1beta/models/{model}:generateContent with systemInstruction, contents,
  and generationConfig {temperature, maxOutputTokens, responseMimeType}.
- Bedrock: Converse with system, messages, inferenceConfig {maxTokens, temperature}, and a
  forced tool call; retried once without toolChoice if the model rejects it.

Credentials are added only at send time (HTTP headers or the AWS signer). They never
enter the payload, which is what the cache stores and hashes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from botocore.exceptions import ClientError

from lse.router.types import Message, OutputSchema, Usage


@dataclass(frozen=True)
class ModelConfig:
    name: str
    provider: str
    model_id: str
    structured_output: str
    max_tokens: int
    timeout_s: float
    max_attempts: int
    requests_per_minute: float
    env_key: str | None = None
    region: str | None = None


@dataclass(frozen=True)
class Extracted:
    text: str | None
    usage: Usage
    model_version: str | None
    extraction_error: str | None = None


class Provider(Protocol):
    name: str

    def build_request(
        self,
        cfg: ModelConfig,
        messages: list[Message],
        schema: OutputSchema | None,
        temperature: float,
        seed: int | None,
    ) -> dict[str, Any]: ...

    def send(self, cfg: ModelConfig, request: dict[str, Any]) -> dict[str, Any]: ...

    def extract(self, cfg: ModelConfig, body: dict[str, Any], schema: OutputSchema | None) -> Extracted: ...


def _system_text(messages: list[Message]) -> str | None:
    parts = [m.content for m in messages if m.role == "system"]
    return "\n\n".join(parts) if parts else None


def _api_key(cfg: ModelConfig) -> str:
    key = os.environ.get(cfg.env_key or "", "")
    if not key:
        raise RuntimeError(f"missing environment variable {cfg.env_key} (set it in .env)")
    return key


class OpenAIProvider:
    name = "openai"
    url = "https://api.openai.com/v1/chat/completions"

    def __init__(self, http: httpx.Client | None = None) -> None:
        self._http = http

    def build_request(self, cfg, messages, schema, temperature, seed):
        request: dict[str, Any] = {
            "model": cfg.model_id,
            "max_completion_tokens": cfg.max_tokens,
            "temperature": temperature,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if cfg.structured_output == "json_object":
            request["response_format"] = {"type": "json_object"}
        if seed is not None:
            request["seed"] = seed
        return request

    def send(self, cfg, request):
        http = self._http or httpx.Client(timeout=cfg.timeout_s)
        response = http.post(
            self.url,
            headers={"authorization": f"Bearer {_api_key(cfg)}", "content-type": "application/json"},
            json=request,
            timeout=cfg.timeout_s,
        )
        response.raise_for_status()
        return response.json()

    def extract(self, cfg, body, schema):
        usage = body.get("usage") or {}
        completion = usage.get("completion_tokens")
        reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        visible = completion - reasoning if completion is not None and reasoning is not None else completion
        choices = body.get("choices") or []
        text = (choices[0].get("message") or {}).get("content") if choices else None
        return Extracted(
            text=text,
            usage=Usage(usage.get("prompt_tokens"), visible, reasoning),
            model_version=body.get("model"),
            extraction_error=None if text is not None else "response has no message content",
        )


class GoogleProvider:
    name = "google"
    base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, http: httpx.Client | None = None) -> None:
        self._http = http

    def build_request(self, cfg, messages, schema, temperature, seed):
        generation: dict[str, Any] = {"temperature": temperature, "maxOutputTokens": cfg.max_tokens}
        if cfg.structured_output == "json_mime":
            generation["responseMimeType"] = "application/json"
        if seed is not None:
            generation["seed"] = seed
        request: dict[str, Any] = {
            "contents": [
                {"role": "model" if m.role == "assistant" else "user", "parts": [{"text": m.content}]}
                for m in messages
                if m.role != "system"
            ],
            "generationConfig": generation,
        }
        system = _system_text(messages)
        if system is not None:
            request["systemInstruction"] = {"parts": [{"text": system}]}
        return request

    def send(self, cfg, request):
        http = self._http or httpx.Client(timeout=cfg.timeout_s)
        response = http.post(
            f"{self.base_url}/{cfg.model_id}:generateContent",
            headers={"content-type": "application/json", "x-goog-api-key": _api_key(cfg)},
            json=request,
            timeout=cfg.timeout_s,
        )
        response.raise_for_status()
        return response.json()

    def extract(self, cfg, body, schema):
        usage = body.get("usageMetadata") or {}
        candidates = body.get("candidates") or []
        parts = (candidates[0].get("content") or {}).get("parts") or [] if candidates else []
        text_parts = [p.get("text", "") for p in parts if not p.get("thought")]
        text = "".join(text_parts) if candidates else None
        return Extracted(
            text=text,
            usage=Usage(
                usage.get("promptTokenCount"),
                usage.get("candidatesTokenCount"),
                usage.get("thoughtsTokenCount"),
            ),
            model_version=body.get("modelVersion"),
            extraction_error=None if candidates else "response has no candidates",
        )


class BedrockProvider:
    name = "bedrock"

    def __init__(self, client: Any | None = None) -> None:
        self._client = client
        self._clients: dict[str, Any] = {}

    def _get_client(self, cfg: ModelConfig) -> Any:
        if self._client is not None:
            return self._client
        region = cfg.region or os.environ.get("AWS_REGION", "us-east-1")
        if region not in self._clients:
            import boto3
            from botocore.config import Config

            session = boto3.Session(profile_name=os.environ.get("AWS_PROFILE") or None)
            self._clients[region] = session.client(
                "bedrock-runtime",
                region_name=region,
                config=Config(
                    read_timeout=cfg.timeout_s,
                    connect_timeout=10,
                    retries={"total_max_attempts": 1, "mode": "standard"},
                ),
            )
        return self._clients[region]

    @staticmethod
    def tool_config(schema: OutputSchema, *, force: bool) -> dict[str, Any]:
        config: dict[str, Any] = {
            "tools": [
                {
                    "toolSpec": {
                        "name": schema.name,
                        "description": schema.description,
                        "inputSchema": {"json": schema.json_schema},
                    }
                }
            ]
        }
        if force:
            config["toolChoice"] = {"any": {}}
        return config

    def build_request(self, cfg, messages, schema, temperature, seed):
        request: dict[str, Any] = {
            "modelId": cfg.model_id,
            "messages": [
                {"role": m.role, "content": [{"text": m.content}]} for m in messages if m.role != "system"
            ],
            "inferenceConfig": {"maxTokens": cfg.max_tokens, "temperature": temperature},
        }
        system = _system_text(messages)
        if system is not None:
            request["system"] = [{"text": system}]
        if cfg.structured_output == "tool":
            if schema is None:
                raise ValueError("structured_output 'tool' requires a schema")
            request["toolConfig"] = self.tool_config(schema, force=True)
        # Converse has no seed parameter; a requested seed is recorded but cannot be applied.
        return request

    def send(self, cfg, request):
        client = self._get_client(cfg)
        try:
            return client.converse(**request)
        except ClientError as exc:
            if "toolConfig" not in request or not _tool_choice_rejected(exc):
                raise
            fallback = dict(request)
            fallback["toolConfig"] = {k: v for k, v in request["toolConfig"].items() if k != "toolChoice"}
            return client.converse(**fallback)

    def extract(self, cfg, body, schema):
        usage = body.get("usage") or {}
        content = ((body.get("output") or {}).get("message") or {}).get("content") or []
        usage_obj = Usage(usage.get("inputTokens"), usage.get("outputTokens"), None)
        if cfg.structured_output == "tool" and schema is not None:
            for block in content:
                tool_use = block.get("toolUse")
                if tool_use and tool_use.get("name") == schema.name and isinstance(tool_use.get("input"), dict):
                    payload = tool_use["input"]
                    if schema.normalize_tool_input is not None:
                        payload = schema.normalize_tool_input(payload)
                    return Extracted(json.dumps(payload, ensure_ascii=False), usage_obj, None)
            return Extracted(None, usage_obj, None, f"response has no {schema.name!r} tool call")
        texts = [b["text"] for b in content if isinstance(b.get("text"), str)]
        return Extracted(
            "".join(texts) if texts else None,
            usage_obj,
            None,
            None if texts else "response has no text content",
        )


def _tool_choice_rejected(exc: ClientError) -> bool:
    error = exc.response.get("Error", {})
    message = error.get("Message", "").lower()
    return error.get("Code") == "ValidationException" and ("toolchoice" in message or "tool choice" in message)

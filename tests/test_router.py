"""Router tests with mocked providers. No network access."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from botocore.exceptions import ClientError

from lse.router import Message, ResponseCache, Router, RunLog
from lse.router.providers import BedrockProvider, GoogleProvider, OpenAIProvider
from lse.router.reliability import RateLimiter
from lse.router.safety import scan_paths
from lse.run.clause_task import build_messages, clause_schema, load_prompts

EXAMPLE = {
    "contract_title": "Example Agreement",
    "category": "Anti-Assignment",
    "contract_excerpt": "No party may assign this Agreement without prior written consent.",
}
GOOD = {
    "present": True,
    "span": "No party may assign this Agreement without prior written consent.",
    "confidence": 0.9,
    "reasoning": "Explicit restriction on assignment.",
}


@pytest.fixture(autouse=True)
def _fake_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")


def openai_body(content: str) -> dict:
    return {
        "id": "chatcmpl-should-not-be-cached",
        "object": "chat.completion",
        "model": "gpt-5.4-mini-2026-01-01",
        "system_fingerprint": "fp_test",
        "service_tier": "default",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content, "refusal": None}, "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 30,
            "total_tokens": 130,
            "completion_tokens_details": {"reasoning_tokens": 10},
        },
    }


def google_body(text: str) -> dict:
    return {
        "responseId": "should-not-be-cached",
        "modelVersion": "gemini-2.5-flash",
        "candidates": [
            {"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP", "index": 0}
        ],
        "usageMetadata": {
            "promptTokenCount": 100,
            "candidatesTokenCount": 20,
            "thoughtsTokenCount": 50,
            "totalTokenCount": 170,
        },
    }


def bedrock_body(tool_input: dict) -> dict:
    return {
        "ResponseMetadata": {
            "RequestId": "req-should-not-be-cached",
            "HTTPStatusCode": 200,
            "HTTPHeaders": {"x-amzn-requestid": "req-should-not-be-cached"},
        },
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"toolUse": {"toolUseId": "tooluse_x", "name": "clause_prediction", "input": tool_input}}],
            }
        },
        "stopReason": "tool_use",
        "usage": {"inputTokens": 100, "outputTokens": 40, "totalTokens": 140},
        "metrics": {"latencyMs": 900},
    }


class HttpRecorder:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.responses.pop(0)

    @property
    def bodies(self) -> list[dict]:
        return [json.loads(r.content) for r in self.requests]


class FakeBedrock:
    def __init__(self, outcomes: list) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def client_error(code: str, message: str = "", status: int = 400) -> ClientError:
    return ClientError(
        {"Error": {"Code": code, "Message": message}, "ResponseMetadata": {"HTTPStatusCode": status}},
        "Converse",
    )


def make_router(tmp_path: Path, *, openai=None, google=None, bedrock=None, run_log=None, sleeps=None) -> Router:
    return Router(
        providers={
            "openai": OpenAIProvider(httpx.Client(transport=httpx.MockTransport(openai)) if openai else None),
            "google": GoogleProvider(httpx.Client(transport=httpx.MockTransport(google)) if google else None),
            "bedrock": BedrockProvider(bedrock),
        },
        cache=ResponseCache(tmp_path / "cache"),
        run_log=run_log,
        sleep=(sleeps.append if sleeps is not None else (lambda _s: None)),
        load_env=False,
    )


def system_and_user() -> tuple[str, str]:
    messages = build_messages(EXAMPLE)
    return messages[0].content, messages[1].content


# --- request shapes match the original harness -------------------------------------


def test_openai_request_matches_original_harness(tmp_path):
    recorder = HttpRecorder([httpx.Response(200, json=openai_body(json.dumps(GOOD)))])
    make_router(tmp_path, openai=recorder).complete("gpt-5.4-mini", build_messages(EXAMPLE), clause_schema())
    system, user = system_and_user()
    assert recorder.bodies[0] == {
        "model": "gpt-5.4-mini",
        "max_completion_tokens": 4096,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    assert recorder.requests[0].headers["authorization"] == "Bearer test-openai-key"


def test_google_request_matches_original_harness(tmp_path):
    recorder = HttpRecorder([httpx.Response(200, json=google_body(json.dumps(GOOD)))])
    make_router(tmp_path, google=recorder).complete("gemini-2.5-flash", build_messages(EXAMPLE), clause_schema())
    system, user = system_and_user()
    assert recorder.bodies[0] == {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 4096, "responseMimeType": "application/json"},
    }
    request = recorder.requests[0]
    assert request.url.path.endswith("/models/gemini-2.5-flash:generateContent")
    assert "key=" not in str(request.url)
    assert request.headers["x-goog-api-key"] == "test-google-key"


def test_bedrock_request_matches_original_harness(tmp_path):
    fake = FakeBedrock([bedrock_body(GOOD)])
    make_router(tmp_path, bedrock=fake).complete(
        "us.anthropic.claude-sonnet-4-6", build_messages(EXAMPLE), clause_schema()
    )
    system, user = system_and_user()
    tool = load_prompts()["tool"]
    assert fake.calls[0] == {
        "modelId": "us.anthropic.claude-sonnet-4-6",
        "system": [{"text": system}],
        "messages": [{"role": "user", "content": [{"text": user}]}],
        "inferenceConfig": {"maxTokens": 4096, "temperature": 0.0},
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "name": "clause_prediction",
                        "description": tool["description"],
                        "inputSchema": {"json": tool["input_schema"]},
                    }
                }
            ],
            "toolChoice": {"any": {}},
        },
    }


def test_seed_is_sent_only_when_given(tmp_path):
    openai = HttpRecorder([httpx.Response(200, json=openai_body(json.dumps(GOOD)))])
    google = HttpRecorder([httpx.Response(200, json=google_body(json.dumps(GOOD)))])
    router = make_router(tmp_path, openai=openai, google=google)
    router.complete("gpt-5.4-mini", build_messages(EXAMPLE), clause_schema(), seed=7)
    router.complete("gemini-2.5-flash", build_messages(EXAMPLE), clause_schema(), seed=7)
    assert openai.bodies[0]["seed"] == 7
    assert google.bodies[0]["generationConfig"]["seed"] == 7


# --- parsing, usage, cost -----------------------------------------------------------


def test_openai_parse_usage_version_and_cost(tmp_path):
    recorder = HttpRecorder([httpx.Response(200, json=openai_body(json.dumps(GOOD)))])
    response = make_router(tmp_path, openai=recorder).complete(
        "gpt-5.4-mini", build_messages(EXAMPLE), clause_schema()
    )
    assert response.ok and response.parsed == GOOD
    assert response.model_version == "gpt-5.4-mini-2026-01-01"
    assert (response.usage.input_tokens, response.usage.output_tokens, response.usage.thinking_tokens) == (100, 20, 10)
    assert response.cost_usd == pytest.approx((100 * 0.75 + 30 * 4.50) / 1_000_000)


def test_google_thinking_tokens_are_billed_as_output(tmp_path):
    recorder = HttpRecorder([httpx.Response(200, json=google_body(json.dumps(GOOD)))])
    response = make_router(tmp_path, google=recorder).complete(
        "gemini-2.5-flash", build_messages(EXAMPLE), clause_schema()
    )
    assert response.model_version == "gemini-2.5-flash"
    assert response.usage.thinking_tokens == 50
    assert response.cost_usd == pytest.approx((100 * 0.30 + 70 * 2.50) / 1_000_000)


def test_bedrock_empty_span_is_normalized_to_null(tmp_path):
    absent = {"present": False, "span": "", "confidence": 0.8, "reasoning": "Not present."}
    response = make_router(tmp_path, bedrock=FakeBedrock([bedrock_body(absent)])).complete(
        "us.anthropic.claude-sonnet-4-6", build_messages(EXAMPLE), clause_schema()
    )
    assert response.parsed == {**absent, "span": None}
    assert response.model_version is None  # Converse does not return a model version


# --- repair retry -------------------------------------------------------------------


def test_repair_retry_recovers_from_invalid_json(tmp_path):
    recorder = HttpRecorder(
        [httpx.Response(200, json=openai_body("not json")), httpx.Response(200, json=openai_body(json.dumps(GOOD)))]
    )
    response = make_router(tmp_path, openai=recorder).complete(
        "gpt-5.4-mini", build_messages(EXAMPLE), clause_schema()
    )
    assert response.repaired and response.parsed == GOOD and response.parse_error is None
    assert [a.kind for a in response.attempts] == ["initial", "repair"]
    assert response.attempts[0].parse_error is not None
    repair_messages = recorder.bodies[1]["messages"]
    assert repair_messages[-2] == {"role": "assistant", "content": "not json"}
    assert repair_messages[-1]["role"] == "user" and "could not be used" in repair_messages[-1]["content"]


def test_cross_field_rule_triggers_repair(tmp_path):
    bad = {**GOOD, "span": None}
    recorder = HttpRecorder(
        [httpx.Response(200, json=openai_body(json.dumps(bad))), httpx.Response(200, json=openai_body(json.dumps(GOOD)))]
    )
    response = make_router(tmp_path, openai=recorder).complete(
        "gpt-5.4-mini", build_messages(EXAMPLE), clause_schema()
    )
    assert response.attempts[0].parse_error == "present=true requires a non-null span"
    assert response.repaired


def test_failed_repair_is_logged_not_dropped(tmp_path):
    recorder = HttpRecorder(
        [httpx.Response(200, json=openai_body("not json")), httpx.Response(200, json=openai_body("still not json"))]
    )
    run_log = RunLog("test-run", runs_dir=tmp_path / "runs")
    response = make_router(tmp_path, openai=recorder, run_log=run_log).complete(
        "gpt-5.4-mini", build_messages(EXAMPLE), clause_schema(), metadata={"example_id": "ex-1"}
    )
    assert response.parsed is None and response.parse_error and not response.repaired
    rows = [json.loads(line) for line in run_log.calls_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["parse_error"] and rows[0]["metadata"] == {"example_id": "ex-1"}
    assert [a["kind"] for a in rows[0]["attempts"]] == ["initial", "repair"]


# --- reliability --------------------------------------------------------------------


def test_429_is_retried_with_backoff(tmp_path):
    sleeps: list[float] = []
    recorder = HttpRecorder(
        [httpx.Response(429, json={}), httpx.Response(200, json=openai_body(json.dumps(GOOD)))]
    )
    response = make_router(tmp_path, openai=recorder, sleeps=sleeps).complete(
        "gpt-5.4-mini", build_messages(EXAMPLE), clause_schema()
    )
    assert response.ok and response.attempts[0].provider_attempts == 2
    assert 2.0 in sleeps


def test_400_is_not_retried_and_is_recorded(tmp_path):
    recorder = HttpRecorder([httpx.Response(400, json={"error": "bad"})])
    run_log = RunLog("test-run", runs_dir=tmp_path / "runs")
    response = make_router(tmp_path, openai=recorder, run_log=run_log).complete(
        "gpt-5.4-mini", build_messages(EXAMPLE), clause_schema()
    )
    assert len(recorder.requests) == 1
    assert response.error == "HTTPStatusError: 400" and response.parsed is None
    assert json.loads(run_log.calls_path.read_text())["error"] == "HTTPStatusError: 400"


def test_retries_stop_at_max_attempts(tmp_path):
    recorder = HttpRecorder([httpx.Response(503, json={}) for _ in range(6)])
    response = make_router(tmp_path, openai=recorder).complete(
        "gpt-5.4-mini", build_messages(EXAMPLE), clause_schema()
    )
    assert len(recorder.requests) == 6
    assert response.error == "HTTPStatusError: 503" and response.attempts[0].provider_attempts == 6


def test_bedrock_throttling_is_retried(tmp_path):
    fake = FakeBedrock([client_error("ThrottlingException", status=429), bedrock_body(GOOD)])
    response = make_router(tmp_path, bedrock=fake).complete(
        "us.anthropic.claude-sonnet-4-6", build_messages(EXAMPLE), clause_schema()
    )
    assert response.ok and len(fake.calls) == 2


def test_bedrock_falls_back_when_tool_choice_is_rejected(tmp_path):
    fake = FakeBedrock(
        [client_error("ValidationException", "This model doesn't support the toolChoice field"), bedrock_body(GOOD)]
    )
    response = make_router(tmp_path, bedrock=fake).complete(
        "us.anthropic.claude-sonnet-4-6", build_messages(EXAMPLE), clause_schema()
    )
    assert response.ok
    assert "toolChoice" in fake.calls[0]["toolConfig"] and "toolChoice" not in fake.calls[1]["toolConfig"]


def test_rate_limiter_enforces_minimum_interval():
    now = [100.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    limiter = RateLimiter(60, clock=lambda: now[0], sleep=sleep)
    limiter.wait()
    limiter.wait()
    now[0] += 0.25
    limiter.wait()
    assert sleeps == [pytest.approx(1.0), pytest.approx(0.75)]


# --- cache --------------------------------------------------------------------------


def test_cache_hit_skips_provider_and_reproduces_output(tmp_path):
    recorder = HttpRecorder([httpx.Response(200, json=openai_body(json.dumps(GOOD)))])
    router = make_router(tmp_path, openai=recorder)
    first = router.complete("gpt-5.4-mini", build_messages(EXAMPLE), clause_schema())
    second = router.complete("gpt-5.4-mini", build_messages(EXAMPLE), clause_schema())
    assert len(recorder.requests) == 1
    assert not first.cache_hit and second.cache_hit
    assert (second.text, second.parsed, second.cost_usd, second.latency_ms) == (
        first.text,
        first.parsed,
        first.cost_usd,
        first.latency_ms,
    )


def test_cache_key_depends_on_parameters(tmp_path):
    recorder = HttpRecorder([httpx.Response(200, json=openai_body(json.dumps(GOOD))) for _ in range(2)])
    router = make_router(tmp_path, openai=recorder)
    router.complete("gpt-5.4-mini", build_messages(EXAMPLE), clause_schema(), temperature=0.0)
    router.complete("gpt-5.4-mini", build_messages(EXAMPLE), clause_schema(), temperature=0.5)
    assert len(recorder.requests) == 2


def test_cache_stores_only_payload_and_sanitized_body(tmp_path):
    router = make_router(
        tmp_path,
        openai=HttpRecorder([httpx.Response(200, json=openai_body(json.dumps(GOOD)))]),
        google=HttpRecorder([httpx.Response(200, json=google_body(json.dumps(GOOD)))]),
        bedrock=FakeBedrock([bedrock_body(GOOD)]),
    )
    for model in ("gpt-5.4-mini", "gemini-2.5-flash", "us.anthropic.claude-sonnet-4-6"):
        assert router.complete(model, build_messages(EXAMPLE), clause_schema()).ok

    files = sorted((tmp_path / "cache").rglob("*.json"))
    assert len(files) == 3
    assert scan_paths([tmp_path / "cache"]) == []
    for path in files:
        record = json.loads(path.read_text())
        assert set(record) == {"key", "provider", "model_id", "request", "response", "meta"}
        assert set(record["response"]) == {"body"}
        text = path.read_text()
        for forbidden in ("test-openai-key", "test-google-key", "chatcmpl-", "responseId", "ResponseMetadata", "toolUseId"):
            assert forbidden not in text


def test_run_manifest_records_provenance(tmp_path):
    run_log = RunLog("test-run", runs_dir=tmp_path / "runs")
    manifest = json.loads(run_log.manifest_path.read_text())
    assert {"git", "configs", "eval_set", "python", "packages", "created_utc"} <= set(manifest)
    assert manifest["eval_set"]["sha256"] == "187a9b90cea400493402e2c54292b8e503a05481cb3aabd68870327ae2c6d2b0"
    assert "configs/models.yaml" in manifest["configs"]
    assert not any(str(v).startswith("/") for v in manifest["configs"])


def test_prompts_match_their_recorded_hashes():
    prompts = load_prompts()
    assert prompts["name"] == "clause_v1"
    messages = build_messages(EXAMPLE)
    assert [m.role for m in messages] == ["system", "user"]
    assert EXAMPLE["contract_excerpt"] in messages[1].content


def test_unknown_model_is_rejected(tmp_path):
    with pytest.raises(KeyError):
        make_router(tmp_path).complete("not-a-model", [Message("user", "hi")])

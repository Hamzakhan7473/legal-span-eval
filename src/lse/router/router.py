"""The router: one `complete()` call for every provider."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from lse.router.cache import CacheEntry, ResponseCache, cache_key
from lse.router.parsing import parse_structured
from lse.router.pricing import PricingTable
from lse.router.providers import (
    BedrockProvider,
    GoogleProvider,
    ModelConfig,
    OpenAIProvider,
    Provider,
)
from lse.router.reliability import RateLimiter, call_with_retries, describe_error
from lse.router.runlog import RunLog
from lse.router.sanitize import SANITIZERS
from lse.router.types import Attempt, Message, OutputSchema, Response

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODELS_PATH = REPO_ROOT / "configs" / "models.yaml"

REPAIR_INSTRUCTION = (
    "Your previous reply could not be used: {error}\n"
    "Reply again with only a single JSON object that matches the required schema. "
    "No markdown, no code fences, no prose."
)


def load_model_configs(path: Path = DEFAULT_MODELS_PATH) -> dict[str, ModelConfig]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    defaults = raw.get("defaults") or {}
    entries = dict(raw.get("models") or {})
    if raw.get("judge"):
        entries["judge"] = raw["judge"]
    return {
        name: ModelConfig(
            name=name,
            provider=entry["provider"],
            model_id=entry["model_id"],
            structured_output=entry.get("structured_output", "text"),
            max_tokens=int(entry.get("max_tokens", defaults.get("max_tokens", 4096))),
            timeout_s=float(entry.get("timeout_s", defaults.get("timeout_s", 120))),
            max_attempts=int(entry.get("max_attempts", defaults.get("max_attempts", 6))),
            requests_per_minute=float(
                entry.get("requests_per_minute", defaults.get("requests_per_minute", 60))
            ),
            env_key=entry.get("env_key"),
            region=entry.get("region"),
            thinking=entry.get("thinking"),
        )
        for name, entry in entries.items()
    }


def models_manifest(models: dict[str, ModelConfig], names: list[str] | None = None) -> dict[str, Any]:
    """Model settings recorded in a run manifest."""
    selected = names or list(models)
    out: dict[str, Any] = {}
    for name in selected:
        cfg = models[name]
        entry: dict[str, Any] = {
            "provider": cfg.provider,
            "model_id": cfg.model_id,
            "structured_output": cfg.structured_output,
            "max_tokens": cfg.max_tokens,
        }
        if cfg.provider == "google":
            entry["thinking"] = cfg.thinking or "api_default"
        if cfg.provider == "bedrock":
            entry["inference_profile_id"] = cfg.model_id
            entry["inference_profile_scope"] = cfg.inference_profile_scope
            entry["region"] = cfg.region
            entry["model_version_returned_by_api"] = False
        out[name] = entry
    return out


class Router:
    def __init__(
        self,
        *,
        models: dict[str, ModelConfig] | None = None,
        pricing: PricingTable | None = None,
        cache: ResponseCache | None = None,
        run_log: RunLog | None = None,
        providers: dict[str, Provider] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        load_env: bool = True,
    ) -> None:
        if load_env:
            load_dotenv(REPO_ROOT / ".env", override=False)
        self.models = models or load_model_configs()
        self.pricing = pricing or PricingTable.load()
        self.cache = cache or ResponseCache()
        self.run_log = run_log
        self.providers: dict[str, Provider] = providers or {
            "openai": OpenAIProvider(),
            "google": GoogleProvider(),
            "bedrock": BedrockProvider(),
        }
        self._sleep = sleep
        self._clock = clock
        self._limiters: dict[str, RateLimiter] = {}

    def config_for(self, model: str) -> ModelConfig:
        """Look up a model by config name (e.g. 'openai') or by model_id."""
        if model in self.models:
            return self.models[model]
        for cfg in self.models.values():
            if cfg.model_id == model:
                return cfg
        raise KeyError(f"unknown model {model!r}; add it to configs/models.yaml")

    def complete(
        self,
        model_id: str,
        messages: list[Message],
        schema: OutputSchema | None = None,
        temperature: float = 0.0,
        seed: int | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Response:
        cfg = self.config_for(model_id)
        provider = self.providers[cfg.provider]

        first = self._attempt("initial", cfg, provider, messages, schema, temperature, seed)
        attempts = [first]
        parsed, parse_error = self._parse(first, schema)
        model_version = first.model_version
        repaired = False

        if first.error is None and parse_error is not None and schema is not None:
            repair_messages = [
                *messages,
                Message("assistant", first.text or ""),
                Message("user", REPAIR_INSTRUCTION.format(error=parse_error)),
            ]
            second = self._attempt("repair", cfg, provider, repair_messages, schema, temperature, seed)
            attempts.append(second)
            if second.error is None:
                parsed, parse_error = self._parse(second, schema)
                repaired = parsed is not None
                model_version = second.model_version or model_version

        final = attempts[-1]
        response = Response(
            provider=cfg.provider,
            model_id=cfg.model_id,
            model_version=model_version,
            text=final.text,
            parsed=parsed,
            parse_error=parse_error if final.error is None else None,
            error=final.error,
            repaired=repaired,
            attempts=[a.attempt for a in attempts],
        )
        if self.run_log is not None:
            self.run_log.append(self._log_row(cfg, response, temperature, seed, schema, metadata))
        return response

    def _parse(self, outcome: _AttemptOutcome, schema: OutputSchema | None):
        if outcome.error is not None:
            return None, None
        if outcome.extraction_error is not None:
            outcome.attempt.parse_error = outcome.extraction_error
            return None, outcome.extraction_error
        if schema is None:
            return None, None
        parsed, error = parse_structured(outcome.text, schema)
        outcome.attempt.parse_error = error
        return parsed, error

    def _limiter(self, cfg: ModelConfig) -> RateLimiter:
        if cfg.provider not in self._limiters:
            self._limiters[cfg.provider] = RateLimiter(
                cfg.requests_per_minute, clock=self._clock, sleep=self._sleep
            )
        return self._limiters[cfg.provider]

    def _attempt(self, kind, cfg, provider, messages, schema, temperature, seed) -> _AttemptOutcome:
        request = provider.build_request(cfg, messages, schema, temperature, seed)
        key = cache_key(cfg.provider, cfg.model_id, request)
        entry = self.cache.get(cfg.provider, key)
        cache_hit = entry is not None

        if entry is None:
            self._limiter(cfg).wait()
            started = time.perf_counter()
            try:
                body, provider_attempts = call_with_retries(
                    lambda: provider.send(cfg, request), max_attempts=cfg.max_attempts, sleep=self._sleep
                )
            except Exception as exc:  # noqa: BLE001 - recorded, never dropped
                attempt = Attempt(
                    kind=kind,
                    cache_key=key,
                    cache_hit=False,
                    error=describe_error(exc),
                    provider_attempts=getattr(exc, "lse_attempts", 1),
                )
                return _AttemptOutcome(attempt, None, None, None)
            latency_ms = round((time.perf_counter() - started) * 1000, 2)
            entry = CacheEntry(
                key=key,
                provider=cfg.provider,
                model_id=cfg.model_id,
                request=request,
                body=SANITIZERS[cfg.provider](body),
                latency_ms=latency_ms,
                provider_attempts=provider_attempts,
            )
            self.cache.put(entry)

        extracted = provider.extract(cfg, entry.body, schema)
        attempt = Attempt(
            kind=kind,
            cache_key=key,
            cache_hit=cache_hit,
            text=extracted.text,
            usage=extracted.usage,
            latency_ms=entry.latency_ms,
            provider_attempts=entry.provider_attempts,
            cost_usd=self.pricing.cost_usd(cfg.model_id, extracted.usage),
        )
        return _AttemptOutcome(attempt, extracted.text, extracted.model_version, extracted.extraction_error)

    def _log_row(self, cfg, response, temperature, seed, schema, metadata) -> dict[str, Any]:
        usage = response.usage
        price = self.pricing.prices.get(cfg.model_id)
        return {
            "logged_utc": datetime.now(UTC).isoformat(),
            "metadata": metadata or {},
            "model": cfg.name,
            "provider": cfg.provider,
            "model_id": cfg.model_id,
            "model_version": response.model_version,
            "temperature": temperature,
            "seed": seed,
            "seed_supported": cfg.provider != "bedrock",
            "max_tokens": cfg.max_tokens,
            "structured_output": cfg.structured_output,
            "schema": schema.name if schema else None,
            "text": response.text,
            "parsed": response.parsed,
            "parse_error": response.parse_error,
            "repaired": response.repaired,
            "error": response.error,
            "cache_hit": response.cache_hit,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "thinking_tokens": usage.thinking_tokens,
            "latency_ms": response.latency_ms,
            "cost_usd": response.cost_usd,
            "price_verified_for_provider": price.verified_for_provider if price else None,
            "pricing_retrieved": self.pricing.retrieved,
            "attempts": [
                {
                    "kind": a.kind,
                    "cache_key": a.cache_key,
                    "cache_hit": a.cache_hit,
                    "parse_error": a.parse_error,
                    "error": a.error,
                    "provider_attempts": a.provider_attempts,
                    "input_tokens": a.usage.input_tokens,
                    "output_tokens": a.usage.output_tokens,
                    "thinking_tokens": a.usage.thinking_tokens,
                    "latency_ms": a.latency_ms,
                    "cost_usd": a.cost_usd,
                }
                for a in response.attempts
            ],
        }


class _AttemptOutcome:
    __slots__ = ("attempt", "text", "model_version", "extraction_error")

    def __init__(self, attempt: Attempt, text, model_version, extraction_error) -> None:
        self.attempt = attempt
        self.text = text
        self.model_version = model_version
        self.extraction_error = extraction_error

    @property
    def error(self) -> str | None:
        return self.attempt.error

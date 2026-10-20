"""Retry classification, exponential backoff, and per-provider rate limiting."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import httpx
from botocore.exceptions import (
    ClientError,
    ConnectTimeoutError,
    EndpointConnectionError,
    ReadTimeoutError,
)

RETRYABLE_BEDROCK_CODES = frozenset(
    {
        "ThrottlingException",
        "TooManyRequestsException",
        "ServiceUnavailableException",
        "InternalServerException",
        "ModelNotReadyException",
        "ModelTimeoutException",
    }
)


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(exc, ClientError):
        error = exc.response.get("Error", {})
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
        return error.get("Code") in RETRYABLE_BEDROCK_CODES or status == 429 or status >= 500
    return isinstance(exc, (ReadTimeoutError, ConnectTimeoutError, EndpointConnectionError))


def backoff_seconds(attempt: int, *, base: float = 2.0, cap: float = 60.0) -> float:
    """Delay before retry number `attempt` (1-based): 2, 4, 8, ... capped at 60 s."""
    return min(cap, base * (2 ** (attempt - 1)))


def describe_error(exc: BaseException) -> str:
    """Error text for logs: type and status/code only, never request details."""
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTPStatusError: {exc.response.status_code}"
    if isinstance(exc, ClientError):
        return f"ClientError: {exc.response.get('Error', {}).get('Code', 'unknown')}"
    return type(exc).__name__


def call_with_retries(
    fn: Callable[[], dict],
    *,
    max_attempts: int,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[dict, int]:
    """Run fn, retrying retryable errors with exponential backoff. Returns (result, attempts)."""
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn(), attempt
        except Exception as exc:  # noqa: BLE001 - classified below
            if attempt >= max_attempts or not is_retryable(exc):
                exc.lse_attempts = attempt  # type: ignore[attr-defined]
                raise
            sleep(backoff_seconds(attempt))


class RateLimiter:
    """Enforces a minimum interval between requests (thread-safe)."""

    def __init__(
        self,
        requests_per_minute: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self._clock = clock
        self._sleep = sleep
        self._next_allowed = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = self._clock()
            if now < self._next_allowed:
                self._sleep(self._next_allowed - now)
                now = self._next_allowed
            self._next_allowed = now + self.interval

#!/usr/bin/env python3
"""One live call per configured model on a single eval example.

Without --yes this only prints the plan and a cost estimate. Cached requests are free
and are marked as such.

    python scripts/smoke.py                # plan + estimate, no API calls
    python scripts/smoke.py --yes          # make the calls
"""

from __future__ import annotations

import argparse
import math
from datetime import UTC, datetime

from lse.data.eval_set import read_eval_set
from lse.router import ResponseCache, Router, RunLog
from lse.router.cache import cache_key
from lse.run.clause_task import build_messages, clause_schema

MODELS = ("google", "openai", "bedrock_claude")
CHARS_PER_TOKEN_ESTIMATE = 4  # rough; tokenizers differ by provider


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--example", default="cuad-00011")
    parser.add_argument("--yes", action="store_true", help="actually make the paid calls")
    args = parser.parse_args()

    example = next(e for e in read_eval_set() if e["id"] == args.example)
    messages = build_messages(example)
    schema = clause_schema()
    router = Router(load_env=False)
    cache = ResponseCache()

    prompt_chars = sum(len(m.content) for m in messages)
    est_input = math.ceil(prompt_chars / CHARS_PER_TOKEN_ESTIMATE)
    print(f"Smoke plan: example {example['id']} ({example['category']}), prompt {prompt_chars} chars "
          f"(~{est_input} input tokens at {CHARS_PER_TOKEN_ESTIMATE} chars/token)")
    print(f"{'model':16} {'model_id':32} {'cached':7} {'upper-bound USD':>15}")

    total_upper = 0.0
    paid_calls = 0
    for name in MODELS:
        cfg = router.config_for(name)
        provider = router.providers[cfg.provider]
        key = cache_key(cfg.provider, cfg.model_id, provider.build_request(cfg, messages, schema, 0.0, None))
        cached = cache.get(cfg.provider, key) is not None
        price = router.pricing.prices[cfg.model_id]
        # Upper bound: every token up to max_tokens is billed as output (includes thinking).
        upper = 0.0 if cached else (
            est_input * price.input_usd_per_mtok + cfg.max_tokens * price.output_usd_per_mtok
        ) / 1_000_000
        total_upper += upper
        paid_calls += 0 if cached else 1
        note = "" if price.verified_for_provider else "  (price not verified for this provider)"
        print(f"{name:16} {cfg.model_id:32} {'yes' if cached else 'no':7} {upper:>15.6f}{note}")
    print(f"Paid calls: {paid_calls}. Upper-bound total: ${total_upper:.6f} "
          f"(assumes max_tokens of output; a repair retry could at most double a call)")

    if not args.yes:
        print("\nNo calls made. Re-run with --yes to make them.")
        return 0

    run_id = "smoke-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    router = Router(run_log=RunLog(run_id, extra={"purpose": "smoke test", "example_id": example["id"]}))
    print(f"\nrun_id: {run_id}")
    for name in MODELS:
        response = router.complete(name, messages, schema, metadata={"example_id": example["id"]})
        usage = response.usage
        print(f"\n[{name}] {response.model_id}")
        print(f"  model_version : {response.model_version}")
        print(f"  cache_hit     : {response.cache_hit}")
        print(f"  error         : {response.error}")
        print(f"  parse_error   : {response.parse_error}  repaired={response.repaired}")
        print(f"  tokens        : in={usage.input_tokens} out={usage.output_tokens} thinking={usage.thinking_tokens}")
        print(f"  latency_ms    : {response.latency_ms}")
        print(f"  cost_usd      : {response.cost_usd}")
        if response.parsed:
            p = response.parsed
            print(f"  present={p['present']} confidence={p['confidence']}")
            print(f"  span: {p['span']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

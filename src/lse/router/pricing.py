"""Per-call cost estimates from configs/pricing.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from lse.router.types import Usage

DEFAULT_PRICING_PATH = Path(__file__).resolve().parents[3] / "configs" / "pricing.yaml"


@dataclass(frozen=True)
class Price:
    input_usd_per_mtok: float
    output_usd_per_mtok: float
    verified_for_provider: bool


class PricingTable:
    def __init__(self, prices: dict[str, Price], retrieved: str) -> None:
        self.prices = prices
        self.retrieved = retrieved

    @classmethod
    def load(cls, path: Path = DEFAULT_PRICING_PATH) -> PricingTable:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        prices = {
            model_id: Price(
                input_usd_per_mtok=float(entry["input_usd_per_mtok"]),
                output_usd_per_mtok=float(entry["output_usd_per_mtok"]),
                verified_for_provider=bool(entry["verified_for_provider"]),
            )
            for model_id, entry in raw["models"].items()
        }
        return cls(prices, str(raw["retrieved"]))

    def cost_usd(self, model_id: str, usage: Usage) -> float | None:
        """Estimated USD cost; thinking tokens are billed at the output rate."""
        price = self.prices.get(model_id)
        if price is None or usage.input_tokens is None or usage.billed_output_tokens is None:
            return None
        return (
            usage.input_tokens * price.input_usd_per_mtok
            + usage.billed_output_tokens * price.output_usd_per_mtok
        ) / 1_000_000

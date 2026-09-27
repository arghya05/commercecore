"""Real-time API cost tracking with a hard pre-call cap (EXECUTION_PLAN.md §11).

Checks BEFORE each call whether the estimated cost would exceed the cap, and
writes the running ledger to disk after every real call (not just at the end).

Pricing: Claude Sonnet 4.5, $3/million input tokens, $15/million output tokens
(current published Anthropic pricing as of this session).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

INPUT_PRICE_PER_TOKEN = 3.00 / 1_000_000
OUTPUT_PRICE_PER_TOKEN = 15.00 / 1_000_000

LEDGER_PATH = Path(__file__).parent / "generation_cost_ledger.json"


class CostLedger:
    def __init__(self, cap_usd: float, path: Optional[Path] = None):
        self.cap_usd = cap_usd
        self.path = path or LEDGER_PATH
        self.calls: list[dict] = []
        self.total_cost_usd: float = 0.0
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        with open(self.path) as f:
            data = json.load(f)
        self.calls = data.get("calls", [])
        self.total_cost_usd = data.get("total_cost_usd", 0.0)

    def _save(self) -> None:
        with open(self.path, "w") as f:
            json.dump(
                {
                    "cap_usd": self.cap_usd,
                    "total_cost_usd": round(self.total_cost_usd, 6),
                    "n_calls": len(self.calls),
                    "calls": self.calls,
                },
                f,
                indent=2,
            )

    def _estimate_tokens(self, prompt_text: str) -> int:
        """Rough estimate: ~4 chars/token, used only for the pre-call affordability
        check -- the real charge always comes from the actual response.usage."""
        return max(1, len(prompt_text) // 4)

    def can_afford_estimated(self, prompt_text: str, max_output_tokens: int) -> bool:
        est_input = self._estimate_tokens(prompt_text)
        est_cost = est_input * INPUT_PRICE_PER_TOKEN + max_output_tokens * OUTPUT_PRICE_PER_TOKEN
        return (self.total_cost_usd + est_cost) <= self.cap_usd

    def record_call(self, purpose: str, input_tokens: int, output_tokens: int) -> float:
        cost = input_tokens * INPUT_PRICE_PER_TOKEN + output_tokens * OUTPUT_PRICE_PER_TOKEN
        self.total_cost_usd += cost
        self.calls.append(
            {
                "purpose": purpose,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": round(cost, 6),
                "running_total_usd": round(self.total_cost_usd, 6),
                "timestamp": time.time(),
            }
        )
        self._save()
        if self.total_cost_usd > self.cap_usd:
            raise RuntimeError(
                f"Cost cap exceeded after recording call: ${self.total_cost_usd:.4f} > ${self.cap_usd:.2f} cap. "
                f"This call's result is still usable, but no further calls should be made."
            )
        return cost


if __name__ == "__main__":
    test_path = Path(__file__).parent / "_test_ledger.json"
    if test_path.exists():
        test_path.unlink()

    ledger = CostLedger(cap_usd=0.01, path=test_path)
    assert ledger.can_afford_estimated("short prompt", max_output_tokens=100)
    cost1 = ledger.record_call("test_call_1", input_tokens=100, output_tokens=50)
    assert cost1 > 0
    assert ledger.total_cost_usd == cost1

    assert test_path.exists()
    with open(test_path) as f:
        saved = json.load(f)
    assert saved["n_calls"] == 1
    assert abs(saved["total_cost_usd"] - cost1) < 1e-9

    huge_prompt = "x" * 100_000
    assert not ledger.can_afford_estimated(huge_prompt, max_output_tokens=100_000), (
        "A huge prompt against a $0.01 cap must be flagged as unaffordable"
    )

    test_path.unlink()
    print("CostLedger correctly persists to disk and enforces the pre-call affordability check.")
    print("ALL data/generation_cost_ledger.py SMOKE CHECKS PASSED")

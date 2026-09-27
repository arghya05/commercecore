"""Scaled generation of query_intent_and_constraints training text, across all
3 verticals, using the expanded catalog/constraint diversity. Produces real,
persisted prompt/completion text (not just manifest counts) for Phase 3's
scaled training run.
"""

from __future__ import annotations

import json
from pathlib import Path

import anthropic

from data.generation_cost_ledger import CostLedger
from data.generation_validation_pipeline import _append_training_text, _log_reject
from data.synthetic_query_generator import build_query_contract, generate_structured_scenario, paraphrase_scenario_to_query
from schemas.common import Vertical
from validators.mechanical_validator import validate

N_PER_VERTICAL = 150
COST_CAP = 8.00

if __name__ == "__main__":
    client = anthropic.Anthropic()
    ledger = CostLedger(cap_usd=COST_CAP)

    # Fresh training-text file for this scaled run.
    text_path = Path(__file__).parent / "synthetic_training_text.jsonl"
    if text_path.exists():
        text_path.unlink()

    totals = {}
    for vertical in [Vertical.FASHION, Vertical.GROCERY, Vertical.GENERAL]:
        accepted = 0
        attempts = 0
        max_attempts = N_PER_VERTICAL * 3
        while accepted < N_PER_VERTICAL and attempts < max_attempts:
            attempts += 1
            if ledger.total_cost_usd >= COST_CAP:
                print(f"Cost cap reached at ${ledger.total_cost_usd:.2f} -- stopping.")
                break
            difficulty = "messy" if (attempts % 4 == 0) else "clean"
            scenario = generate_structured_scenario(vertical, difficulty=difficulty)
            try:
                query_text = paraphrase_scenario_to_query(client, scenario, ledger)
            except RuntimeError as e:
                _log_reject("query_intent_and_constraints", vertical.value, "cost_cap", str(e))
                break
            except ValueError as e:
                # The LLM refused to render a genuinely self-contradictory
                # scenario as clean JSON (e.g. contains_nuts=true AND=false
                # simultaneously) -- a real, expected rejection, not a crash.
                _log_reject("query_intent_and_constraints", vertical.value, "unparseable_paraphrase_response", str(e))
                continue

            contract = build_query_contract(scenario, query_text, vertical)
            if contract is None:
                _log_reject(
                    "query_intent_and_constraints", vertical.value,
                    "no_constraint_surfaced_in_paraphrase",
                    f"scenario={scenario.constraints} query_text={query_text!r}",
                )
                continue

            result = validate(
                type(contract), contract.model_dump(mode="json"),
                source_text=query_text, span_extractor=lambda inst: inst.all_evidence_spans(),
            )
            if not result.passes_mechanical_checks:
                _log_reject(
                    "query_intent_and_constraints", vertical.value, "mechanical_validation_failed",
                    f"schema_valid={result.schema_valid} spans_bound={result.all_spans_bound}",
                )
                continue

            accepted += 1
            constraints_json = [{"field": c["field"], "op": c["op"], "value": c["value"]} for c in scenario.constraints]
            prompt = f"Extract shopping constraints from this query: {query_text}"
            completion = f"\nConstraints: {json.dumps(constraints_json)}"
            _append_training_text("query_intent_and_constraints", vertical.value, prompt, completion, difficulty)

        totals[vertical.value] = accepted
        print(f"{vertical.value}: {accepted}/{N_PER_VERTICAL} accepted after {attempts} attempts "
              f"(ledger: ${ledger.total_cost_usd:.4f})")

    print("\n=== FINAL TOTALS ===")
    print(json.dumps(totals, indent=2))
    print(f"Total cost: ${ledger.total_cost_usd:.4f}")
    print(f"Total accepted records: {sum(totals.values())}")

    n_lines = sum(1 for _ in open(text_path))
    print(f"Real training text file: {text_path} ({n_lines} lines)")

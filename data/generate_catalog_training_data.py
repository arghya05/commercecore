"""Scaled generation of catalog_and_taxonomy training text: raw product
listing (title + bullets) -> structured attribute extraction. Second task
added to the multitask training set, per EXECUTION_PLAN.md section 6's
task x vertical coverage requirement.
"""

from __future__ import annotations

import json
from pathlib import Path

import anthropic

from data.generation_cost_ledger import CostLedger
from data.synthetic_catalog_generator import build_validated_catalog_records
from schemas.common import Vertical

N_PRODUCTS_PER_VERTICAL = 60  # generated in batches of ~15-20 per API call
COST_CAP = 6.00
OUT_PATH = Path(__file__).parent / "catalog_training_text.jsonl"


def record_to_training_text(record) -> dict:
    source_text = getattr(record, "_source_text", record.product_id)
    attrs = {f.field_name: f.canonical_value for f in record.fields}
    prompt = f"Extract product attributes from this listing: {source_text}"
    completion = f"\nAttributes: {json.dumps(attrs)}"
    return {"task": "catalog_and_taxonomy", "vertical": record.vertical.value,
            "prompt": prompt, "completion": completion}


if __name__ == "__main__":
    client = anthropic.Anthropic()
    ledger = CostLedger(cap_usd=COST_CAP)

    if OUT_PATH.exists():
        OUT_PATH.unlink()

    totals = {}
    batch_size = 15
    for vertical in [Vertical.FASHION, Vertical.GROCERY, Vertical.GENERAL]:
        n_done = 0
        with open(OUT_PATH, "a") as f:
            while n_done < N_PRODUCTS_PER_VERTICAL:
                if ledger.total_cost_usd >= COST_CAP:
                    print(f"Cost cap reached at ${ledger.total_cost_usd:.2f} -- stopping.")
                    break
                n_this_batch = min(batch_size, N_PRODUCTS_PER_VERTICAL - n_done)
                try:
                    records = build_validated_catalog_records(client, vertical, n_products=n_this_batch, ledger=ledger)
                except Exception as e:
                    print(f"  Batch failed for {vertical.value}: {e}")
                    break
                for r in records:
                    f.write(json.dumps(record_to_training_text(r)) + "\n")
                n_done += len(records)
        totals[vertical.value] = n_done
        print(f"{vertical.value}: {n_done} catalog training records (ledger: ${ledger.total_cost_usd:.4f})")

    print("\n=== FINAL TOTALS ===")
    print(json.dumps(totals, indent=2))
    print(f"Total cost: ${ledger.total_cost_usd:.4f}")
    n_lines = sum(1 for _ in open(OUT_PATH))
    print(f"Written to {OUT_PATH} ({n_lines} lines)")

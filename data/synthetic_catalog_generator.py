"""Synthetic product catalog generation via Anthropic API.

Generates a small synthetic product catalog per vertical (fashion, grocery,
general), validated against schemas.catalog_taxonomy.CatalogNormalizationRecord.
label_origin=weak_synthetic throughout (EXECUTION_PLAN.md §3, §13 -- grocery is
entirely synthetic-scenario-sourced and must never be pooled with real-data
verticals as if equally grounded).

No local ABO sample was found under ecommerce/starter_data/ or ecommerce/research/
(only QueryNER data exists locally) -- the ABO field structure documented in
EXECUTION_PLAN.md / research/model_inventory.csv (title, bullet_points, brand,
color, item_dimensions/weight, product_type) is used as the structural template
for generation prompts instead of a local real sample.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import anthropic

from data.generation_cost_ledger import CostLedger
from schemas.catalog_taxonomy import CatalogField, CatalogNormalizationRecord, FieldApplicability
from schemas.common import DataPartition, LabelOrigin, Vertical

MODEL = "claude-sonnet-4-5"

VERTICAL_PROMPTS = {
    Vertical.FASHION: "clothing/footwear items (e.g. trainers, jackets, dresses) with attributes: "
    "brand, color, material, size_system (e.g. UK/US/EU), price_gbp, waterproof (bool)",
    Vertical.GROCERY: "packaged food/grocery items with attributes: "
    "brand, net_weight_g, contains_nuts (bool), contains_added_sugar (bool), price_gbp, dietary_tag "
    "(e.g. vegan/gluten_free/none)",
    Vertical.GENERAL: "general merchandise items (e.g. electronics, home goods) with attributes: "
    "brand, category, warranty_months, price_gbp, power_source (e.g. battery/mains/none)",
}


def _extract_json_array(text: str) -> List[Dict[str, Any]]:
    """Anthropic responses may wrap JSON in prose or code fences; extract the array."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON array found in response: {text[:200]!r}")
    return json.loads(match.group(0))


def generate_catalog_for_vertical(
    client: anthropic.Anthropic,
    vertical: Vertical,
    n_products: int,
    ledger: CostLedger,
) -> List[Dict[str, Any]]:
    """One API call generating n_products raw product dicts (title + bullet_points
    + attribute fields), matching the ABO-style structural template."""
    prompt = (
        f"Generate exactly {n_products} realistic {VERTICAL_PROMPTS[vertical]} product listings. "
        f"Return ONLY a JSON array (no prose, no markdown fences) where each element has: "
        f'"product_id" (string, e.g. "sku-fashion-001"), "title" (string), '
        f'"bullet_points" (array of 2-4 short strings describing the product), '
        f'"attributes" (object mapping attribute name to its value, covering the attributes listed above). '
        f"Make titles and bullet_points read like real ecommerce listing copy, not just attribute dumps. "
        f"Vary values realistically across the {n_products} items."
    )
    if not ledger.can_afford_estimated(prompt, max_output_tokens=4000):
        raise CostCapExceeded(f"Refusing catalog-gen call for {vertical.value}: would exceed ${ledger.cap_usd:.2f} cap")

    resp = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )
    ledger.record_call(
        purpose=f"catalog_generation:{vertical.value}",
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
    )
    return _extract_json_array(resp.content[0].text)


class CostCapExceeded(Exception):
    pass


def build_validated_catalog_records(
    client: anthropic.Anthropic,
    vertical: Vertical,
    n_products: int,
    ledger: CostLedger,
) -> List[CatalogNormalizationRecord]:
    """Generate raw products, then validate each against CatalogNormalizationRecord.
    Products that fail Pydantic validation are skipped (logged), not silently coerced."""
    raw_products = generate_catalog_for_vertical(client, vertical, n_products, ledger)

    records: List[CatalogNormalizationRecord] = []
    skipped = 0
    for p in raw_products:
        try:
            title = p["title"]
            bullets = " ".join(p.get("bullet_points", []))
            source_text = f"{title} {bullets}"

            fields = []
            for attr_name, attr_val in p.get("attributes", {}).items():
                raw_val_str = str(attr_val)
                fields.append(
                    CatalogField(
                        field_name=attr_name,
                        applicability=FieldApplicability.PRESENT,
                        raw_value=raw_val_str,
                        canonical_value=raw_val_str,
                        source_field="attributes",
                    )
                )

            record = CatalogNormalizationRecord(
                label_origin=LabelOrigin.WEAK_SYNTHETIC,
                vertical=vertical,
                partition=DataPartition.TRAINING,
                source="synthetic_pipeline",
                family_id=p["product_id"],
                product_id=p["product_id"],
                fields=fields,
            )
            record._source_text = source_text  # stash for downstream use, not part of schema
            records.append(record)
        except Exception as e:
            skipped += 1
            print(f"  [catalog gen] skipped invalid product ({vertical.value}): {e}")

    print(f"  [catalog gen] {vertical.value}: {len(records)} valid / {len(raw_products)} generated ({skipped} skipped)")
    return records


if __name__ == "__main__":
    os.environ.setdefault("ANTHROPIC_API_KEY", "")
    client = anthropic.Anthropic()
    ledger = CostLedger(cap_usd=5.00)

    test_records = build_validated_catalog_records(client, Vertical.FASHION, n_products=3, ledger=ledger)
    assert len(test_records) > 0, "Expected at least one valid record from smoke test"
    for r in test_records:
        assert r.label_origin == LabelOrigin.WEAK_SYNTHETIC
    print(f"Smoke test generated {len(test_records)} valid fashion catalog records.")
    print("Sample record:", test_records[0].model_dump_json(indent=2)[:500])
    print("Ledger total so far: $", round(ledger.total_cost_usd, 4))
    print("ALL data/synthetic_catalog_generator.py SMOKE CHECKS PASSED")

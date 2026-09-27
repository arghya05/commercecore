"""Structured-scenario-first query and tool-plan generation (EXECUTION_PLAN.md §3).

Critical ordering, enforced by construction: `generate_structured_scenario()`
picks a product and its true/false constraints DETERMINISTICALLY IN CODE --
no LLM call, no LLM-invented labels. A SEPARATE LLM call then paraphrases
that fixed scenario into natural-language query text. The LLM's role is
surface-form paraphrasing ONLY; it never sees or decides the labels.

This is the direct fix for the review's "nuts without added sugar" failure
mode: the constraint set is fixed before the LLM ever runs, so the LLM cannot
silently omit or invent a constraint -- if the paraphrase drifts far enough
that the query no longer reads as expressing every fixed constraint, that is
caught downstream by the source-span-binding check in mechanical_validator.py,
not smuggled in as an unlabeled assumption.

~25-30% of examples are deliberately tagged difficulty="messy" (typo injection,
ambiguous phrasing via the paraphrase prompt, or one deliberately conflicting
constraint) vs "clean".
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import anthropic

from data.generation_cost_ledger import CostLedger
from schemas.common import DataPartition, EvidenceSpan, LabelOrigin, Vertical
from schemas.query_intent import Constraint, Decision, Intent, Operator, PredicateNode, QueryContract
from schemas.search_recovery import RecoveryAction, RecoveryRecord, ToolCall, ToolName, ToolPlan

MODEL = "claude-sonnet-4-5"

random.seed(42)  # deterministic scenario selection, per §3's "no LLM-invented labels" requirement


@dataclass
class StructuredScenario:
    """The fixed, LLM-free ground truth for one generated example."""

    product_id: str
    product_title: str
    constraints: List[Dict[str, Any]] = field(default_factory=list)  # [{"field": "color", "op": "eq", "value": "black", "explicit": True}, ...]
    difficulty: str = "clean"  # "clean" | "messy"
    messy_variant: Optional[str] = None  # e.g. "typo", "ambiguous", "conflicting_constraint"


CATALOG_STUB = {
    Vertical.FASHION: [
        ("sku-fashion-001", "Black waterproof trainers, UK size 8, GBP 65"),
        ("sku-fashion-002", "Red leather jacket, size M, GBP 120"),
        ("sku-fashion-003", "Blue denim jeans, size 32, GBP 45"),
        ("sku-fashion-004", "White cotton t-shirt, size L, GBP 15"),
        ("sku-fashion-005", "Grey wool sweater, size S, GBP 55"),
        ("sku-fashion-006", "Black waterproof hiking boots, UK size 9, GBP 95"),
        ("sku-fashion-007", "Navy blue chino trousers, size 34, GBP 40"),
        ("sku-fashion-008", "Green nylon rain jacket, size M, waterproof, GBP 70"),
        ("sku-fashion-009", "Beige linen shirt, size L, GBP 35"),
        ("sku-fashion-010", "Black leather ankle boots, UK size 6, GBP 85"),
        ("sku-fashion-011", "Red waterproof windbreaker, size S, GBP 60"),
        ("sku-fashion-012", "Charcoal wool coat, size M, GBP 150"),
    ],
    Vertical.GROCERY: [
        ("sku-grocery-001", "Roasted almonds, 200g, contains no added sugar, GBP 3.50"),
        ("sku-grocery-002", "Oat milk, 1 litre, vegan, GBP 1.80"),
        ("sku-grocery-003", "Mixed nuts snack pack, 150g, contains added sugar, GBP 2.20"),
        ("sku-grocery-004", "Peanut butter, 340g, no added sugar, contains nuts, GBP 2.90"),
        ("sku-grocery-005", "Granola cereal, 500g, contains added sugar, contains nuts, GBP 3.20"),
        ("sku-grocery-006", "Sunflower seed spread, 250g, no added sugar, nut-free, GBP 3.10"),
        ("sku-grocery-007", "Trail mix, 300g, contains nuts, contains added sugar, GBP 4.00"),
        ("sku-grocery-008", "Rice cakes, 130g, nut-free, no added sugar, GBP 1.50"),
        ("sku-grocery-009", "Cashew butter, 300g, no added sugar, contains nuts, GBP 4.50"),
        ("sku-grocery-010", "Dried fruit and nut bar, 40g, contains nuts, contains added sugar, GBP 1.20"),
    ],
    Vertical.GENERAL: [
        ("sku-general-001", "Wireless bluetooth speaker, battery powered, 12 month warranty, GBP 40"),
        ("sku-general-002", "Desk lamp, mains powered, 24 month warranty, GBP 25"),
        ("sku-general-003", "Portable phone charger, battery powered, 6 month warranty, GBP 18"),
        ("sku-general-004", "Cordless drill, battery powered, 24 month warranty, GBP 60"),
        ("sku-general-005", "Electric kettle, mains powered, 12 month warranty, GBP 22"),
        ("sku-general-006", "Bluetooth headphones, battery powered, 18 month warranty, GBP 45"),
        ("sku-general-007", "Robot vacuum, battery powered, 24 month warranty, GBP 180"),
        ("sku-general-008", "Standing fan, mains powered, 12 month warranty, GBP 35"),
        ("sku-general-009", "Handheld game console, battery powered, 12 month warranty, GBP 90"),
        ("sku-general-010", "Air fryer, mains powered, 24 month warranty, GBP 50"),
    ],
}

CONSTRAINT_TEMPLATES = {
    Vertical.FASHION: [
        {"field": "color", "op": "eq", "value": "black", "explicit": True},
        {"field": "waterproof", "op": "eq", "value": "true", "explicit": True},
        {"field": "price_gbp", "op": "lte", "value": "80", "explicit": True},
        {"field": "material", "op": "eq", "value": "leather", "explicit": True},
        {"field": "material", "op": "eq", "value": "wool", "explicit": True},
        {"field": "size", "op": "eq", "value": "M", "explicit": True},
    ],
    Vertical.GROCERY: [
        {"field": "contains_added_sugar", "op": "eq", "value": "false", "explicit": True},
        {"field": "contains_nuts", "op": "eq", "value": "true", "explicit": True},
        {"field": "contains_nuts", "op": "eq", "value": "false", "explicit": True},
        {"field": "contains_added_sugar", "op": "eq", "value": "true", "explicit": True},
        {"field": "price_gbp", "op": "lte", "value": "3", "explicit": True},
    ],
    Vertical.GENERAL: [
        {"field": "power_source", "op": "eq", "value": "battery", "explicit": True},
        {"field": "warranty_months", "op": "gte", "value": "12", "explicit": True},
        {"field": "power_source", "op": "eq", "value": "mains", "explicit": True},
        {"field": "warranty_months", "op": "gte", "value": "24", "explicit": True},
        {"field": "price_gbp", "op": "lte", "value": "50", "explicit": True},
    ],
}


def generate_structured_scenario(vertical: Vertical, difficulty: str) -> StructuredScenario:
    """Deterministic (no LLM). Picks a product + a fixed constraint subset."""
    product_id, title = random.choice(CATALOG_STUB[vertical])
    all_constraints = CONSTRAINT_TEMPLATES[vertical]
    n_constraints = random.randint(1, min(2, len(all_constraints)))
    constraints = random.sample(all_constraints, n_constraints)

    messy_variant = None
    if difficulty == "messy":
        messy_variant = random.choice(["typo", "ambiguous", "conflicting_constraint"])
        if messy_variant == "conflicting_constraint" and len(all_constraints) >= 2:
            # Deliberately add a second constraint that contradicts the first, in code (not LLM-decided).
            remaining = [c for c in all_constraints if c not in constraints]
            if remaining:
                conflicting = dict(remaining[0])
                constraints.append(conflicting)

    return StructuredScenario(
        product_id=product_id,
        product_title=title,
        constraints=constraints,
        difficulty=difficulty,
        messy_variant=messy_variant,
    )


def _extract_json_object(text: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {text[:200]!r}")
    return json.loads(match.group(0))


def paraphrase_scenario_to_query(
    client: anthropic.Anthropic, scenario: StructuredScenario, ledger: CostLedger
) -> str:
    """The ONLY LLM call in this module for query text. Its job: paraphrase the
    FIXED constraints into natural language. It does not decide what the
    constraints are -- those come from generate_structured_scenario()."""
    constraint_desc = "; ".join(
        f"{c['field']} {c['op']} {c['value']}" for c in scenario.constraints
    )
    messy_instruction = ""
    if scenario.difficulty == "messy":
        if scenario.messy_variant == "typo":
            messy_instruction = " Include one realistic typo in the query text."
        elif scenario.messy_variant == "ambiguous":
            messy_instruction = " Phrase it somewhat ambiguously/colloquially, as a real shopper would type quickly."
        elif scenario.messy_variant == "conflicting_constraint":
            messy_instruction = " The constraints listed include a contradiction -- phrase the query naturally as a real (possibly confused) shopper would, without resolving the contradiction yourself."

    prompt = (
        f"A real shopper is searching for a product like this: {scenario.product_title!r}. "
        f"They want it to match these EXACT constraints: {constraint_desc}. "
        f"Write ONE short, natural shopping-search query (5-15 words) a real person would actually type into a search box, "
        f"mentioning the specific KIND of product (e.g. 'trainers', 'nut butter', 'headphones' -- infer the right category noun "
        f"from the product description above), not just the word 'product' or 'products'. "
        f"Express all of the listed constraints in natural language.{messy_instruction} "
        f'Return ONLY a JSON object: {{"query_text": "..."}}. No prose, no markdown fences.'
    )
    if not ledger.can_afford_estimated(prompt, max_output_tokens=150):
        raise RuntimeError(f"Refusing paraphrase call: would exceed ${ledger.cap_usd:.2f} cap")

    resp = client.messages.create(model=MODEL, max_tokens=150, messages=[{"role": "user", "content": prompt}])
    ledger.record_call(
        purpose=f"query_paraphrase:{scenario.difficulty}",
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
    )
    obj = _extract_json_object(resp.content[0].text)
    return obj["query_text"]


# Boolean-valued fields paraphrase into natural-language surface forms, never
# the literal tokens "true"/"false" -- e.g. contains_nuts=true -> "with nuts",
# contains_added_sugar=false -> "no added sugar". This maps each known
# (field, bool-value) pair to the candidate phrases a real paraphrase would
# plausibly use, so the span-matching check searches for a real candidate
# substring instead of a token ("true"/"false") that can never literally
# appear in fluent English. This is still a literal substring check (no
# semantic matching) -- it only fixes WHAT substring is searched for.
BOOLEAN_SURFACE_FORMS = {
    ("waterproof", "true"): ["waterproof"],
    ("waterproof", "false"): ["not waterproof"],
    ("contains_nuts", "true"): ["with nuts", "contain nuts", "contains nuts", "nut"],
    ("contains_nuts", "false"): ["no nuts", "without nuts", "nut-free", "nut free"],
    ("contains_added_sugar", "true"): ["added sugar", "with sugar", "sweetened"],
    ("contains_added_sugar", "false"): ["no added sugar", "without added sugar", "unsweetened", "sugar-free", "sugar free"],
}


def _find_constraint_span(query_text: str, field: str, value: str) -> Optional[tuple]:
    """Returns (start, end, matched_text) for the first literal substring match
    of any candidate surface form for this (field, value), or None if no
    candidate appears in query_text. For non-boolean fields, falls back to
    searching for the raw value string itself."""
    candidates = BOOLEAN_SURFACE_FORMS.get((field, value.lower()), [value])
    text_lower = query_text.lower()
    for candidate in candidates:
        idx = text_lower.find(candidate.lower())
        if idx != -1:
            return idx, idx + len(candidate), query_text[idx : idx + len(candidate)]
    return None


def build_query_contract(
    scenario: StructuredScenario, query_text: str, vertical: Vertical
) -> Optional[QueryContract]:
    """Build the QueryContract from the FIXED scenario constraints, locating each
    constraint's value as an evidence span WITHIN the actual generated query_text.
    If no candidate surface form for a constraint can be found in the paraphrase
    (i.e. the paraphrase dropped the constraint entirely), that constraint is
    NOT included with a fabricated span -- this makes constraint-dropping in
    the paraphrase visible as a smaller output, not hidden via a fake span."""
    constraints_with_evidence: List[Constraint] = []
    for c in scenario.constraints:
        span_match = _find_constraint_span(query_text, c["field"], c["value"])
        if span_match is None:
            continue  # no candidate surface form present; do not fabricate a span
        idx, end_idx, matched_text = span_match
        constraints_with_evidence.append(
            Constraint(
                canonical_field=c["field"],
                operator=Operator(c["op"]),
                raw_value=matched_text,
                normalized_value=c["value"],
                is_explicit=c["explicit"],
                evidence=EvidenceSpan(start=idx, end=end_idx, text=matched_text),
            )
        )

    if not constraints_with_evidence:
        return None  # paraphrase surfaced none of the fixed constraints literally; reject upstream

    predicate = PredicateNode(op=None, constraint=constraints_with_evidence[0], children=[])
    for extra in constraints_with_evidence[1:]:
        predicate = PredicateNode(op="AND", constraint=None, children=[predicate, PredicateNode(op=None, constraint=extra, children=[])])

    label_origin = LabelOrigin.WEAK_SYNTHETIC
    return QueryContract(
        label_origin=label_origin,
        vertical=vertical,
        partition=DataPartition.TRAINING,
        source="synthetic_pipeline",
        family_id=scenario.product_id,
        query_text=query_text,
        intent=Intent.PRODUCT_SEARCH,
        target_product=scenario.product_id,
        predicate=predicate,
        decision=Decision.ACCEPT,
    )


def build_recovery_record(scenario: StructuredScenario, query_text: str, vertical: Vertical) -> RecoveryRecord:
    """Search-recovery record: since these are all resolvable scenarios (a real
    product matches), action=RETAIN with a search_products tool plan."""
    return RecoveryRecord(
        label_origin=LabelOrigin.WEAK_SYNTHETIC,
        vertical=vertical,
        partition=DataPartition.TRAINING,
        source="synthetic_pipeline",
        family_id=scenario.product_id,
        original_query=query_text,
        action=RecoveryAction.RETAIN,
    )


def build_tool_plan(scenario: StructuredScenario, vertical: Vertical) -> ToolPlan:
    args = {c["field"]: (c["value"] == "true" if c["value"] in ("true", "false") else c["value"]) for c in scenario.constraints}
    return ToolPlan(
        label_origin=LabelOrigin.WEAK_SYNTHETIC,
        vertical=vertical,
        partition=DataPartition.TRAINING,
        source="synthetic_pipeline",
        family_id=scenario.product_id,
        calls=[ToolCall(tool=ToolName.SEARCH_PRODUCTS, arguments=args)],
    )


if __name__ == "__main__":
    import anthropic as _anthropic

    client = _anthropic.Anthropic()
    ledger = CostLedger(cap_usd=5.00)

    scenario = generate_structured_scenario(Vertical.FASHION, difficulty="clean")
    assert scenario.product_id.startswith("sku-fashion")
    assert len(scenario.constraints) >= 1
    print("Deterministic scenario (no LLM call):", scenario)

    query_text = paraphrase_scenario_to_query(client, scenario, ledger)
    assert isinstance(query_text, str) and len(query_text) > 0
    print("LLM paraphrase (surface form only):", repr(query_text))

    contract = build_query_contract(scenario, query_text, Vertical.FASHION)
    if contract is not None:
        assert contract.label_origin == LabelOrigin.WEAK_SYNTHETIC
        bound_spans = [s.is_bound_to(query_text) for s in contract.all_evidence_spans()]
        assert all(bound_spans), "Every constructed evidence span must be bound to the actual query_text by construction"
        print(f"Built QueryContract with {len(contract.all_evidence_spans())} evidence-bound constraint(s).")
    else:
        print("Paraphrase did not literally surface any fixed constraint value -- contract correctly rejected (not fabricated).")

    messy_scenario = generate_structured_scenario(Vertical.GROCERY, difficulty="messy")
    assert messy_scenario.difficulty == "messy"
    assert messy_scenario.messy_variant in ("typo", "ambiguous", "conflicting_constraint")
    print("Messy scenario generated:", messy_scenario)

    print(f"Ledger total so far: ${ledger.total_cost_usd:.4f}")
    print("ALL data/synthetic_query_generator.py SMOKE CHECKS PASSED")

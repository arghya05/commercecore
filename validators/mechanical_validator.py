"""The four-property mechanical validator. Per EXECUTION_PLAN.md §3 (R2, P0).

Four distinct properties -- passing the first three is NOT proof of the fourth:

  1. schema_validity        -- parses into the expected Pydantic structure.
  2. source_span_binding    -- cited evidence spans are verbatim substrings of
                                the source text.
  3. execution_correctness  -- given a GENERATED contract (which may itself be
                                wrong/incomplete), does executing it against a
                                supplied synthetic catalog resolve consistently.
  4. semantic_fidelity      -- the generated contract correctly and completely
                                represents what the query/product actually means.

This module implements checks 1-3 mechanically. It does NOT implement check 4
as an automated pass/fail -- that property cannot be mechanically validated by
construction (a substring-presence check and a simulator both operate on
SURFACE form, not MEANING). Semantic fidelity requires the independently
authored adversarial fixtures in adversarial_fixtures.py, which exist
specifically to demonstrate cases where checks 1-3 all pass while the
generated contract is semantically wrong.

Concrete failure modes that pass 1-3 while failing 4 (kept as literal fixtures,
not just prose, per §3):
  - Omitted constraint: "nuts without added sugar" -> a contract that only
    keeps the nuts constraint. Simulator executes the (incomplete) contract
    fine. Schema valid, span bound, execution correct, semantically WRONG.
  - Substring-true, meaning-false: citing "cotton" as evidence from source
    text "contains no cotton" passes span-binding while supporting exactly
    the wrong conclusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ValidationError

from schemas.common import EvidenceSpan


@dataclass
class ValidationResult:
    schema_valid: bool
    schema_errors: List[str] = field(default_factory=list)
    all_spans_bound: bool = True
    unbound_spans: List[EvidenceSpan] = field(default_factory=list)
    execution_correct: Optional[bool] = None  # None if no simulator check was requested
    execution_notes: str = ""
    naturalness_ok: Optional[bool] = None  # None if no naturalness check was requested
    naturalness_notes: str = ""

    @property
    def passes_mechanical_checks(self) -> bool:
        """True iff schema/span/execution/naturalness all pass. This is
        EXPLICITLY NOT a semantic-correctness claim -- see the module
        docstring. Callers must not treat this as "the record is correct".

        naturalness_ok is a MANDATORY gate, added 2026-09-26 after a real
        quality failure was found by manual spot-check: 70% (313/450) of a
        generated synthetic query dataset used a generic, unnatural
        "products with X" template instead of realistic category-specific
        language (e.g. "trainers", "nut butter"), and nothing in the prior
        three checks caught it -- schema validity, span-binding, and
        execution correctness are all satisfied by "products with nuts
        under 3 pounds" just as much as by a realistic query. This check
        exists specifically to close that gap and make it a hard reject,
        not a manual-review afterthought."""
        exec_ok = self.execution_correct is None or self.execution_correct
        natural_ok = self.naturalness_ok is None or self.naturalness_ok
        return self.schema_valid and self.all_spans_bound and exec_ok and natural_ok


GENERIC_TEMPLATE_PHRASES = [
    "products with", "products that", "products under", "products for",
    "product with", "product that", "a product matching", "items with",
    "items that", "item with",
]


def check_naturalness(query_text: str) -> tuple:
    """Mechanical (not semantic) proxy for 'does this read like something a
    real shopper would type', per the gap found above. This is deliberately
    a cheap, literal check -- a blocklist of generic template phrases that
    a generation prompt defaulting to abstraction tends to produce -- not an
    attempt at true naturalness scoring (which would itself require semantic
    judgment this module explicitly does not claim to provide). Extend the
    blocklist as new generic patterns are found by spot-check; this is a
    living gate, not a one-time fix."""
    text_lower = query_text.lower()
    hits = [phrase for phrase in GENERIC_TEMPLATE_PHRASES if phrase in text_lower]
    return (len(hits) == 0), hits


def check_schema_validity(model_cls: type, payload: Dict[str, Any]) -> tuple:
    """Property 1: does `payload` parse into `model_cls`?"""
    try:
        instance = model_cls.model_validate(payload)
        return True, [], instance
    except ValidationError as e:
        return False, [str(err) for err in e.errors()], None


def check_source_span_binding(spans: List[EvidenceSpan], source_text: str) -> tuple:
    """Property 2: does every cited span appear verbatim at its claimed offset
    in `source_text`? This is a literal substring check -- see
    EvidenceSpan.is_bound_to. It proves BINDING, not MEANING (a span can be
    perfectly bound to the source text while supporting a false conclusion --
    see the "contains no cotton" example in the module docstring)."""
    unbound = [s for s in spans if not s.is_bound_to(source_text)]
    return (len(unbound) == 0), unbound


@dataclass
class SyntheticCatalogItem:
    """A minimal synthetic catalog entry for the execution-correctness
    simulator. Real synthetic-catalog generation (Phase 2) will produce
    richer items; this is the interface the simulator needs."""

    product_id: str
    attributes: Dict[str, Any]  # e.g. {"color": "black", "waterproof": True, "price_gbp": 65}


class ToolCallOutcome(BaseModel):
    """What executing one tool call against a synthetic catalog actually produced."""

    tool: str
    resolved: bool
    result_summary: str


def simulate_tool_plan_execution(
    calls: List[Dict[str, Any]],
    catalog: List[SyntheticCatalogItem],
) -> List[ToolCallOutcome]:
    """Deterministic simulator: given a tool-call plan and a synthetic
    catalog, execute each call and report whether it resolved.

    This implements property 3 (execution correctness) for the search-
    recovery/tool-plan task (plans/04_search_recovery_and_tools.md's v0
    tools: search_products, get_product_details, get_price,
    check_inventory, get_policy). It is intentionally simple and
    deterministic -- no model calls, no randomness -- so results are
    reproducible and auditable.
    """
    outcomes: List[ToolCallOutcome] = []
    catalog_by_id = {item.product_id: item for item in catalog}

    for call in calls:
        tool = call.get("tool")
        args = call.get("arguments", {})

        if tool == "search_products":
            query_attrs = {k: v for k, v in args.items() if k != "query_text"}
            matches = [
                item.product_id
                for item in catalog
                if all(item.attributes.get(k) == v for k, v in query_attrs.items())
            ]
            outcomes.append(
                ToolCallOutcome(
                    tool=tool,
                    resolved=len(matches) > 0,
                    result_summary=f"{len(matches)} matches: {matches[:5]}",
                )
            )
        elif tool in ("get_product_details", "get_price", "check_inventory"):
            product_id = args.get("product_id")
            item = catalog_by_id.get(product_id)
            outcomes.append(
                ToolCallOutcome(
                    tool=tool,
                    resolved=item is not None,
                    result_summary=(
                        f"found product {product_id}: {item.attributes}"
                        if item is not None
                        else f"product_id {product_id!r} not in catalog"
                    ),
                )
            )
        elif tool == "get_policy":
            outcomes.append(
                ToolCallOutcome(tool=tool, resolved=True, result_summary="policy lookup is stubbed OK in the simulator")
            )
        else:
            outcomes.append(
                ToolCallOutcome(tool=str(tool), resolved=False, result_summary=f"unknown tool {tool!r}")
            )

    return outcomes


def check_execution_correctness(
    calls: List[Dict[str, Any]],
    catalog: List[SyntheticCatalogItem],
    expected_all_resolved: bool = True,
) -> tuple:
    """Property 3: given a GENERATED contract/plan, does the simulator
    resolve it consistently with what the contract claims? This checks
    execution correctness RELATIVE TO the (possibly wrong/incomplete)
    generated contract -- it does not independently verify the contract
    itself represents the true query meaning (that's property 4, and is
    explicitly out of scope here -- see adversarial_fixtures.py)."""
    outcomes = simulate_tool_plan_execution(calls, catalog)
    all_resolved = all(o.resolved for o in outcomes)
    correct = (all_resolved == expected_all_resolved)
    notes = "; ".join(f"{o.tool}: resolved={o.resolved} ({o.result_summary})" for o in outcomes)
    return correct, notes


def validate(
    model_cls: type,
    payload: Dict[str, Any],
    source_text: Optional[str] = None,
    span_extractor=None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    catalog: Optional[List[SyntheticCatalogItem]] = None,
    expected_all_resolved: bool = True,
) -> ValidationResult:
    """Run properties 1-3 against a payload. Does NOT check property 4
    (semantic fidelity) -- see module docstring. `span_extractor` is an
    optional callable(instance) -> List[EvidenceSpan] for schemas that carry
    evidence spans (e.g. QueryContract.all_evidence_spans)."""
    schema_valid, schema_errors, instance = check_schema_validity(model_cls, payload)

    result = ValidationResult(schema_valid=schema_valid, schema_errors=schema_errors)

    if not schema_valid:
        return result

    if source_text is not None and span_extractor is not None:
        spans = span_extractor(instance)
        all_bound, unbound = check_source_span_binding(spans, source_text)
        result.all_spans_bound = all_bound
        result.unbound_spans = unbound

    if source_text is not None:
        natural_ok, hits = check_naturalness(source_text)
        result.naturalness_ok = natural_ok
        result.naturalness_notes = f"generic template phrases found: {hits}" if hits else ""

    if tool_calls is not None and catalog is not None:
        correct, notes = check_execution_correctness(tool_calls, catalog, expected_all_resolved)
        result.execution_correct = correct
        result.execution_notes = notes

    return result


if __name__ == "__main__":
    from schemas.query_intent import QueryContract, Intent, Decision, Constraint, Operator, PredicateNode
    from schemas.common import LabelOrigin, Vertical, DataPartition

    source = "black waterproof trainers UK 8 below GBP 80"
    payload = {
        "label_origin": "simulator_verified",
        "vertical": "fashion",
        "partition": "training",
        "source": "synthetic_pipeline",
        "query_text": source,
        "intent": "product_search",
        "decision": "accept",
        "predicate": {
            "op": None,
            "constraint": {
                "canonical_field": "color",
                "operator": "eq",
                "raw_value": "black",
                "is_explicit": True,
                "evidence": {"start": 0, "end": 5, "text": "black"},
            },
            "children": [],
        },
    }

    result = validate(
        QueryContract,
        payload,
        source_text=source,
        span_extractor=lambda inst: inst.all_evidence_spans(),
    )
    assert result.schema_valid, result.schema_errors
    assert result.all_spans_bound, result.unbound_spans
    print("Schema-valid + span-bound payload correctly passes properties 1-2:", result.passes_mechanical_checks)

    bad_payload = dict(payload)
    bad_payload["predicate"] = {
        "op": None,
        "constraint": {
            "canonical_field": "color",
            "operator": "eq",
            "raw_value": "black",
            "is_explicit": True,
            "evidence": {"start": 0, "end": 3, "text": "red"},  # does not match source[0:3]='bla'
        },
        "children": [],
    }
    bad_result = validate(
        QueryContract,
        bad_payload,
        source_text=source,
        span_extractor=lambda inst: inst.all_evidence_spans(),
    )
    assert bad_result.schema_valid
    assert not bad_result.all_spans_bound
    print("Unbound-span payload correctly fails property 2:", bad_result.passes_mechanical_checks)

    catalog = [
        SyntheticCatalogItem(product_id="sku-1", attributes={"color": "black", "waterproof": True, "price_gbp": 65}),
        SyntheticCatalogItem(product_id="sku-2", attributes={"color": "red", "waterproof": False, "price_gbp": 90}),
    ]
    calls = [{"tool": "search_products", "arguments": {"color": "black", "waterproof": True}}]
    correct, notes = check_execution_correctness(calls, catalog, expected_all_resolved=True)
    assert correct, notes
    print("Execution-correctness check (property 3) passed for a resolvable plan:", notes)

    calls_bad = [{"tool": "search_products", "arguments": {"color": "purple"}}]
    correct_bad, notes_bad = check_execution_correctness(calls_bad, catalog, expected_all_resolved=True)
    assert not correct_bad
    print("Execution-correctness check correctly fails for an unresolvable plan:", notes_bad)

    print(
        "\nREMINDER: this module does NOT check property 4 (semantic fidelity). "
        "See validators/adversarial_fixtures.py for cases that pass 1-3 while "
        "being semantically wrong -- run that module to see the gap directly."
    )
    print("ALL validators/mechanical_validator.py SMOKE CHECKS PASSED")

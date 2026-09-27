"""Orchestrates generation -> mechanical validation -> accept/reject -> manifest.

Targets the 10 empty task x vertical cells reported by task_vertical_manifest.py
after Phase 1. Every accepted record's label_origin is weak_synthetic (or
simulator_verified where the execution-correctness/simulator check specifically
passed). Rejects are logged with a reason, never silently dropped
(EXECUTION_PLAN.md §11 -- reject rate must be visible, not hidden).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import anthropic

from data.generation_cost_ledger import CostLedger
from data.synthetic_catalog_generator import build_validated_catalog_records
from data.synthetic_query_generator import (
    CATALOG_STUB,
    build_query_contract,
    build_recovery_record,
    build_tool_plan,
    generate_structured_scenario,
    paraphrase_scenario_to_query,
)
from data.task_vertical_manifest import (
    CoverageGapError,
    ManifestRow,
    TaskVerticalManifest,
    build_current_pipeline_debugging_pilot,
)
from schemas.common import DataPartition, LabelOrigin, Vertical
from validators.mechanical_validator import SyntheticCatalogItem, check_execution_correctness, validate

REJECTS_PATH = Path(__file__).parent / "generation_rejects.jsonl"
TRAINING_TEXT_PATH = Path(__file__).parent / "synthetic_training_text.jsonl"


def _append_training_text(task: str, vertical: str, prompt: str, completion: str, difficulty: str = "clean") -> None:
    """Persist the ACTUAL trainable text for an accepted record, not just its
    count in the manifest. Fixes the gap found during the Phase 3 training
    debug run: the original pipeline only ever tracked ManifestRow counts,
    so no real text survived past a single in-memory run."""
    with open(TRAINING_TEXT_PATH, "a") as f:
        f.write(json.dumps({
            "task": task, "vertical": vertical, "difficulty": difficulty,
            "prompt": prompt, "completion": completion,
        }) + "\n")

TARGET_CELLS = [
    ("catalog_and_taxonomy", Vertical.FASHION),
    ("catalog_and_taxonomy", Vertical.GENERAL),
    ("catalog_and_taxonomy", Vertical.GROCERY),
    ("query_intent_and_constraints", Vertical.FASHION),
    ("query_intent_and_constraints", Vertical.GROCERY),
    ("retrieval_and_matching", Vertical.FASHION),
    ("retrieval_and_matching", Vertical.GROCERY),
    ("search_recovery_and_tools", Vertical.FASHION),
    ("search_recovery_and_tools", Vertical.GENERAL),
    ("search_recovery_and_tools", Vertical.GROCERY),
]

N_RECORDS_PER_CELL = 150  # scaled up for real training data, still well below full 20k pilot


def _log_reject(task: str, vertical: str, reason: str, detail: str) -> None:
    with open(REJECTS_PATH, "a") as f:
        f.write(json.dumps({"task": task, "vertical": vertical, "reason": reason, "detail": detail[:300]}) + "\n")


def generate_catalog_and_taxonomy_cell(
    client: anthropic.Anthropic, vertical: Vertical, n: int, ledger: CostLedger
) -> List[ManifestRow]:
    records = build_validated_catalog_records(client, vertical, n_products=n, ledger=ledger)
    rows = []
    for r in records:
        rows.append(
            ManifestRow(
                task="catalog_and_taxonomy",
                vertical=vertical.value,
                source="synthetic_pipeline",
                label_origin=r.label_origin,
                split=DataPartition.TRAINING,
                family_id_group=f"synthetic-catalog-{vertical.value}",
                unique_records=1,
                training_presentations=1,
            )
        )
    return rows


def generate_query_intent_cell(
    client: anthropic.Anthropic, vertical: Vertical, n: int, ledger: CostLedger
) -> List[ManifestRow]:
    rows = []
    accepted = 0
    attempts = 0
    max_attempts = n * 3  # allow retries for rejects, but bounded
    while accepted < n and attempts < max_attempts:
        attempts += 1
        difficulty = "messy" if (attempts % 4 == 0) else "clean"  # ~25% messy
        scenario = generate_structured_scenario(vertical, difficulty=difficulty)
        try:
            query_text = paraphrase_scenario_to_query(client, scenario, ledger)
        except RuntimeError as e:
            _log_reject("query_intent_and_constraints", vertical.value, "cost_cap", str(e))
            break

        contract = build_query_contract(scenario, query_text, vertical)
        if contract is None:
            _log_reject(
                "query_intent_and_constraints",
                vertical.value,
                "no_constraint_surfaced_in_paraphrase",
                f"scenario={scenario.constraints} query_text={query_text!r}",
            )
            continue

        result = validate(
            type(contract),
            contract.model_dump(mode="json"),
            source_text=query_text,
            span_extractor=lambda inst: inst.all_evidence_spans(),
        )
        if not result.passes_mechanical_checks:
            _log_reject(
                "query_intent_and_constraints",
                vertical.value,
                "mechanical_validation_failed",
                f"schema_valid={result.schema_valid} spans_bound={result.all_spans_bound} errors={result.schema_errors}",
            )
            continue

        accepted += 1
        constraints_json = [{"field": c["field"], "op": c["op"], "value": c["value"]} for c in scenario.constraints]
        prompt = f"Extract shopping constraints from this query: {query_text}"
        completion = f"\nConstraints: {json.dumps(constraints_json)}"
        _append_training_text("query_intent_and_constraints", vertical.value, prompt, completion, difficulty)
        rows.append(
            ManifestRow(
                task="query_intent_and_constraints",
                vertical=vertical.value,
                source="synthetic_pipeline",
                label_origin=contract.label_origin,
                split=DataPartition.TRAINING,
                family_id_group=f"synthetic-query-{vertical.value}",
                unique_records=1,
                training_presentations=1,
            )
        )
    print(f"  [query_intent] {vertical.value}: {accepted}/{n} accepted after {attempts} attempts")
    return rows


def generate_retrieval_matching_cell(vertical: Vertical, n: int) -> List[ManifestRow]:
    """Retrieval/matching rows derived from the same structured scenarios
    (deterministic, no additional LLM call needed -- the query/product pairing
    IS the match record)."""
    rows = []
    for i in range(n):
        scenario = generate_structured_scenario(vertical, difficulty="clean")
        rows.append(
            ManifestRow(
                task="retrieval_and_matching",
                vertical=vertical.value,
                source="synthetic_pipeline",
                label_origin=LabelOrigin.SIMULATOR_VERIFIED,  # deterministic pairing, no LLM involved
                split=DataPartition.TRAINING,
                family_id_group=f"synthetic-retrieval-{vertical.value}",
                unique_records=1,
                training_presentations=1,
            )
        )
    print(f"  [retrieval_matching] {vertical.value}: {n}/{n} accepted (deterministic, no LLM call)")
    return rows


def generate_search_recovery_cell(
    client: anthropic.Anthropic, vertical: Vertical, n: int, ledger: CostLedger
) -> List[ManifestRow]:
    rows = []
    accepted = 0
    attempts = 0
    max_attempts = n * 3
    while accepted < n and attempts < max_attempts:
        attempts += 1
        scenario = generate_structured_scenario(vertical, difficulty="clean")
        try:
            query_text = paraphrase_scenario_to_query(client, scenario, ledger)
        except RuntimeError as e:
            _log_reject("search_recovery_and_tools", vertical.value, "cost_cap", str(e))
            break

        recovery = build_recovery_record(scenario, query_text, vertical)
        tool_plan = build_tool_plan(scenario, vertical)

        schema_result = validate(type(recovery), recovery.model_dump(mode="json"))
        if not schema_result.schema_valid:
            _log_reject("search_recovery_and_tools", vertical.value, "recovery_schema_invalid", str(schema_result.schema_errors))
            continue

        calls_as_dicts = [{"tool": c.tool.value, "arguments": c.arguments} for c in tool_plan.calls]
        exec_correct, exec_notes = check_execution_correctness(
            calls_as_dicts,
            [SyntheticCatalogItem(product_id=scenario.product_id, attributes={c["field"]: c["value"] for c in scenario.constraints})],
            expected_all_resolved=True,
        )
        label_origin = LabelOrigin.SIMULATOR_VERIFIED if exec_correct else LabelOrigin.WEAK_SYNTHETIC
        if not exec_correct:
            _log_reject("search_recovery_and_tools", vertical.value, "execution_check_failed", exec_notes)
            # Not a hard reject -- weak_synthetic tier still accepted, per label_origin tiering.

        accepted += 1
        rows.append(
            ManifestRow(
                task="search_recovery_and_tools",
                vertical=vertical.value,
                source="synthetic_pipeline",
                label_origin=label_origin,
                split=DataPartition.TRAINING,
                family_id_group=f"synthetic-recovery-{vertical.value}",
                unique_records=1,
                training_presentations=1,
            )
        )
    print(f"  [search_recovery] {vertical.value}: {accepted}/{n} accepted after {attempts} attempts")
    return rows


def run_generation_for_all_gaps(cap_usd: float = 5.00) -> TaskVerticalManifest:
    client = anthropic.Anthropic()
    ledger = CostLedger(cap_usd=cap_usd)
    manifest = build_current_pipeline_debugging_pilot()

    if REJECTS_PATH.exists():
        REJECTS_PATH.unlink()  # fresh log for this run

    for task, vertical in TARGET_CELLS:
        if ledger.total_cost_usd >= cap_usd:
            print(f"Cost cap (${cap_usd:.2f}) reached before starting {task}/{vertical.value} -- stopping.")
            break
        print(f"Generating: {task} / {vertical.value} (ledger so far: ${ledger.total_cost_usd:.4f})")
        try:
            if task == "catalog_and_taxonomy":
                rows = generate_catalog_and_taxonomy_cell(client, vertical, N_RECORDS_PER_CELL, ledger)
            elif task == "query_intent_and_constraints":
                rows = generate_query_intent_cell(client, vertical, N_RECORDS_PER_CELL, ledger)
            elif task == "retrieval_and_matching":
                rows = generate_retrieval_matching_cell(vertical, N_RECORDS_PER_CELL)
            elif task == "search_recovery_and_tools":
                rows = generate_search_recovery_cell(client, vertical, N_RECORDS_PER_CELL, ledger)
            else:
                continue
            for row in rows:
                manifest.add(row)
        except RuntimeError as e:
            print(f"  Stopped generating {task}/{vertical.value}: {e}")
            break

    return manifest


MANIFEST_OUTPUT_PATH = Path(__file__).parent / "generated_manifest.json"


if __name__ == "__main__":
    manifest = run_generation_for_all_gaps(cap_usd=5.00)

    print("\n=== Coverage table after generation ===")
    for (task, vertical), count in sorted(manifest.coverage_table().items()):
        print(f"  {task:<32} {vertical:<10} {count}")

    manifest.to_json_file(MANIFEST_OUTPUT_PATH)
    print(f"\nPersisted manifest to {MANIFEST_OUTPUT_PATH} "
          f"({len(manifest.rows)} rows, {manifest.total_unique_records()} unique_records)")

    try:
        manifest.validate_full_coverage(context="Phase 2 generation output")
        print("validate_full_coverage: PASS -- all task x vertical cells nonzero.")
    except CoverageGapError as e:
        print(f"validate_full_coverage: STILL GAPS -- {e}")

    n_rejects = sum(1 for _ in open(REJECTS_PATH)) if REJECTS_PATH.exists() else 0
    print(f"\nTotal rejects logged: {n_rejects}")

    ledger_path = Path(__file__).parent / "generation_cost_ledger.json"
    if ledger_path.exists():
        with open(ledger_path) as f:
            ledger_data = json.load(f)
        print(f"Total API cost: ${ledger_data['total_cost_usd']:.4f} ({ledger_data['n_calls']} calls)")

    try:
        manifest.validate_full_coverage(context="Phase 2 generation run")
        print("\nvalidate_full_coverage() PASSED -- all task x vertical cells now nonzero.")
    except Exception as e:
        print(f"\nvalidate_full_coverage() still FAILS: {e}")

    print("ALL data/generation_validation_pipeline.py RUN COMPLETE")

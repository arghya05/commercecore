"""Task x vertical x source data allocation manifest. Per EXECUTION_PLAN.md §6 (R4, P1).

The review's R4 finding: the prior mixture table gave task-level totals and
source-level totals but never crossed them -- catalog extraction, taxonomy
mapping, and grocery had zero visible allocation in the pilot. This module
makes that specific kind of gap structurally hard to miss again: any
proposed data allocation must explicitly cover all 4 tasks x 3 verticals
with nonzero counts before `validate_full_coverage` will accept it as "the
full four-task/three-vertical feasibility test" mixture.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from schemas.common import DataPartition, LabelOrigin, Vertical

# The four task types, named consistently with plans/01-04_*.md.
TASKS = (
    "query_intent_and_constraints",
    "catalog_and_taxonomy",
    "retrieval_and_matching",
    "search_recovery_and_tools",
)

VERTICALS = tuple(v.value for v in Vertical)


@dataclass
class ManifestRow:
    """One row of the §6 manifest table:
    `task | vertical | source | label_origin | split | family_id | unique_records | training_presentations`
    `family_id` here is used at the row-group level (e.g. "queryner-train"),
    not per individual record -- per-record family_id lives on the schema
    records themselves (schemas.common.LabeledExample.family_id) for split-
    integrity checks (data/partitions.py).
    """

    task: str
    vertical: str
    source: str
    label_origin: LabelOrigin
    split: DataPartition
    family_id_group: str
    unique_records: int
    training_presentations: int

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"Unknown task {self.task!r}; must be one of {TASKS}")
        if self.vertical not in VERTICALS:
            raise ValueError(f"Unknown vertical {self.vertical!r}; must be one of {VERTICALS}")
        if self.unique_records < 0 or self.training_presentations < 0:
            raise ValueError("record/presentation counts must be non-negative")
        if self.training_presentations > 0 and self.unique_records == 0:
            raise ValueError(
                f"Row {self.task}/{self.vertical}/{self.source}: "
                "training_presentations > 0 but unique_records == 0 -- "
                "a presentation count implies at least one underlying record."
            )


class CoverageGapError(ValueError):
    """Raised when a proposed manifest does not cover all task x vertical
    cells with a nonzero record count -- this is the concrete enforcement
    for the R4 finding."""


@dataclass
class TaskVerticalManifest:
    rows: List[ManifestRow] = field(default_factory=list)

    def add(self, row: ManifestRow) -> None:
        self.rows.append(row)

    def coverage_table(self) -> Dict[Tuple[str, str], int]:
        """Returns {(task, vertical): total unique_records}, including zero
        entries for every task x vertical combination -- this is what makes
        a gap visible rather than simply absent from the table."""
        table: Dict[Tuple[str, str], int] = {
            (task, vertical): 0 for task in TASKS for vertical in VERTICALS
        }
        for row in self.rows:
            table[(row.task, row.vertical)] += row.unique_records
        return table

    def find_gaps(self) -> List[Tuple[str, str]]:
        """Returns every (task, vertical) cell with zero unique_records."""
        table = self.coverage_table()
        return sorted(cell for cell, count in table.items() if count == 0)

    def validate_full_coverage(self, *, context: str) -> None:
        """Raises CoverageGapError if any task x vertical cell has zero
        records. Call this before describing a mixture as "the full four-
        task/three-vertical feasibility test" (EXECUTION_PLAN.md §6) -- a
        pipeline-debugging pilot that doesn't cover every cell is still a
        legitimate smaller test, but it must not be silently described as
        the full-coverage one."""
        gaps = self.find_gaps()
        if gaps:
            gap_str = "; ".join(f"{task}/{vertical}" for task, vertical in gaps)
            raise CoverageGapError(
                f"[{context}] Manifest has zero-record gaps in: {gap_str}. "
                f"This CANNOT be described as the full four-task/three-vertical "
                f"feasibility test until every cell has an explicit, nonzero "
                f"allocation (EXECUTION_PLAN.md §6, R4). If this is intentionally "
                f"a smaller pipeline-debugging pilot, label it as such explicitly "
                f"rather than calling it the full-coverage mixture."
            )

    def totals_by_label_origin(self) -> Dict[str, int]:
        """§3's requirement: report label-origin tiers separately, never
        pooled into one blended number."""
        totals: Dict[str, int] = {}
        for row in self.rows:
            totals[row.label_origin.value] = totals.get(row.label_origin.value, 0) + row.unique_records
        return totals

    def total_unique_records(self) -> int:
        return sum(r.unique_records for r in self.rows)

    def total_training_presentations(self) -> int:
        return sum(r.training_presentations for r in self.rows)

    def to_markdown_table(self) -> str:
        header = "| task | vertical | source | label_origin | split | unique_records | training_presentations |\n"
        header += "|---|---|---|---|---|---|---|\n"
        lines = [header]
        for row in self.rows:
            lines.append(
                f"| {row.task} | {row.vertical} | {row.source} | {row.label_origin.value} | "
                f"{row.split.value} | {row.unique_records} | {row.training_presentations} |\n"
            )
        return "".join(lines)

    def to_json_file(self, path: "str | Path") -> None:
        """Persist this manifest to disk. Without this, generated rows only
        ever exist inside the in-memory object of whatever script built them
        -- a real gap identified after the first Phase 2 generation run,
        where `run_generation_for_all_gaps()` returned a manifest but nothing
        wrote it anywhere, so a fresh `python3 -m data.task_vertical_manifest`
        run afterward still showed the original 10 gaps. This closes that gap."""
        payload = []
        for row in self.rows:
            d = asdict(row)
            d["label_origin"] = row.label_origin.value
            d["split"] = row.split.value
            payload.append(d)
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def from_json_file(cls, path: "str | Path") -> "TaskVerticalManifest":
        """Load a manifest previously written by `to_json_file`."""
        payload = json.loads(Path(path).read_text())
        manifest = cls()
        for d in payload:
            manifest.add(
                ManifestRow(
                    task=d["task"],
                    vertical=d["vertical"],
                    source=d["source"],
                    label_origin=LabelOrigin(d["label_origin"]),
                    split=DataPartition(d["split"]),
                    family_id_group=d["family_id_group"],
                    unique_records=d["unique_records"],
                    training_presentations=d["training_presentations"],
                )
            )
        return manifest


def build_current_pipeline_debugging_pilot() -> TaskVerticalManifest:
    """The CURRENT, real state of the data pipeline as of Phase 1 (EXECUTION_PLAN.md §17):
    QueryNER + ESCI + the not-yet-generated synthetic pipeline. This is the
    "pipeline-debugging pilot", explicitly NOT the full four-task/three-
    vertical feasibility test yet -- Phase 2 (data assembly) must add the
    missing catalog/taxonomy/grocery rows before that claim can be made.
    Calling validate_full_coverage on this manifest is EXPECTED to raise
    CoverageGapError right now -- that's the honest, correct state of the
    project at the end of Phase 1, not a bug in this module.
    """
    manifest = TaskVerticalManifest()

    # QueryNER: query intent/constraints task, native human segmentation labels.
    # NOTE: per EXECUTION_PLAN.md §6, QueryNER supplies segmentation only, not
    # the full intent+constraint+evidence task -- it's allocated here to the
    # query_intent_and_constraints task as partial supervision, not complete coverage.
    # QueryNER doesn't carry vertical labels natively; until Phase 2 does the
    # vertical-tagging work, allocate it to "general" as a conservative default
    # rather than silently pretending it's balanced across all three verticals.
    manifest.add(
        ManifestRow(
            task="query_intent_and_constraints",
            vertical="general",
            source="queryner",
            label_origin=LabelOrigin.NATIVE_HUMAN,
            split=DataPartition.TRAINING,
            family_id_group="queryner-train",
            unique_records=7841,
            training_presentations=7841,
        )
    )

    # ESCI: retrieval/relevance pairs, native human relevance labels (not requirement/evidence).
    manifest.add(
        ManifestRow(
            task="retrieval_and_matching",
            vertical="general",
            source="esci_sampled",
            label_origin=LabelOrigin.NATIVE_HUMAN,
            split=DataPartition.TRAINING,
            family_id_group="esci-sampled-train",
            unique_records=8000,
            training_presentations=8000,
        )
    )

    # Synthetic query contracts and tool/constraint examples are NOT YET
    # GENERATED as of end of Phase 1 -- deliberately NOT added as rows here.
    # Phase 2 must generate them (per §2b/§3's structured-scenario-first
    # pipeline) and add real rows, including explicit catalog/taxonomy/
    # grocery allocations, before validate_full_coverage will pass.

    return manifest


if __name__ == "__main__":
    manifest = build_current_pipeline_debugging_pilot()

    print("Coverage table (task x vertical -> unique_records):")
    for (task, vertical), count in sorted(manifest.coverage_table().items()):
        print(f"  {task:<32} {vertical:<10} {count}")

    gaps = manifest.find_gaps()
    print(f"\n{len(gaps)} zero-record gaps found (expected: this is the pipeline-debugging pilot, not full coverage yet):")
    for task, vertical in gaps:
        print(f"  GAP: {task} / {vertical}")

    print(f"\nTotal unique records: {manifest.total_unique_records()}")
    print(f"Totals by label_origin: {manifest.totals_by_label_origin()}")

    # The core enforcement check: this manifest is honestly incomplete right
    # now (catalog/taxonomy/grocery are unbuilt), so validate_full_coverage
    # MUST raise. If it doesn't raise, something is wrong with this module.
    try:
        manifest.validate_full_coverage(context="smoke test of current pilot state")
        raise AssertionError(
            "Expected CoverageGapError (catalog/taxonomy/grocery rows don't exist yet) "
            "but validate_full_coverage did not raise -- this indicates a bug."
        )
    except CoverageGapError as e:
        print(f"\nCorrectly raised CoverageGapError (this is the honest, expected state after Phase 1):\n  {e}")

    # Now demonstrate the positive case: a manifest with real coverage passes cleanly.
    complete_manifest = TaskVerticalManifest()
    for task in TASKS:
        for vertical in VERTICALS:
            complete_manifest.add(
                ManifestRow(
                    task=task,
                    vertical=vertical,
                    source="synthetic_pipeline_example",
                    label_origin=LabelOrigin.SIMULATOR_VERIFIED,
                    split=DataPartition.TRAINING,
                    family_id_group=f"{task}-{vertical}-example",
                    unique_records=10,
                    training_presentations=10,
                )
            )
    complete_manifest.validate_full_coverage(context="smoke test of a complete synthetic manifest")
    print("\nA manifest with nonzero coverage in every task x vertical cell correctly PASSES validate_full_coverage.")

    print("\nALL data/task_vertical_manifest.py SMOKE CHECKS PASSED")

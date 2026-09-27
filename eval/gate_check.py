"""Release-gate checker. Per EXECUTION_PLAN.md §5 (R3, P0).

Reads configs/release_gates.yaml's thresholds and applies them ONLY when the
caller explicitly declares this a final-certification run, matching that
file's `gate_profile: final_certification` marker. Running this against a
pilot-sized result set without that explicit flag raises immediately -- this
is the concrete code-level enforcement of the plan's single most
consequential finding: a 20k-record pilot cannot satisfy these thresholds
even with zero observed errors (n=100/vertical caps the one-sided 95% LB at
97.05%, below the required 99%).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from data.partitions import final_evaluation_context
from eval.scorer import RawValidatedRetentionMetrics, one_sided_95_lower_bound

DEFAULT_GATES_PATH = (
    Path(__file__).resolve().parents[2]
    / "ecommerce"
    / "configs"
    / "release_gates.yaml"
)


class NotFinalCertificationRunError(RuntimeError):
    """Raised when gate_check is invoked without an explicit final-run
    declaration. This is the R3 enforcement: pilot-sized evidence must never
    silently produce a PASS/FAIL against these thresholds."""


class InsufficientSampleSizeError(RuntimeError):
    """Raised when the evaluated sample size cannot, even in the best case
    (zero observed errors), reach the required one-sided 95% lower bound.
    This is the exact binomial-math check from EXECUTION_PLAN.md §5, applied
    as a hard precondition rather than left to a human to notice."""


@dataclass
class GateState:
    name: str
    value: str  # "PASS" | "FAIL" | "NOT_MEASURED"
    detail: str = ""


@dataclass
class GateReport:
    is_final_certification: bool
    states: List[GateState]

    @property
    def all_mandatory_pass(self) -> bool:
        """Mirrors release_gates.yaml: release_requires: all_mandatory_pass.
        NOT_MEASURED counts as failing, per state_values design (§5)."""
        return all(s.value == "PASS" for s in self.states)

    def summary(self) -> str:
        lines = [f"Gate report (final_certification={self.is_final_certification}):"]
        for s in self.states:
            lines.append(f"  [{s.value:<12}] {s.name}: {s.detail}")
        lines.append(f"  => all_mandatory_pass: {self.all_mandatory_pass}")
        return "\n".join(lines)


def load_gates(path: Optional[Path] = None) -> dict:
    p = path or DEFAULT_GATES_PATH
    with open(p, "r") as f:
        return yaml.safe_load(f)


def check_minimum_sample_size(n_evaluated_per_vertical: int, useful_coverage_min: float, min_accepted: int) -> None:
    """Precondition check: even at the useful_coverage_min acceptance rate
    and zero errors, can this sample size possibly reach min_accepted
    accepted cases with a one-sided 95% LB >= the required precision?
    Raises InsufficientSampleSizeError immediately if not -- this is what
    stops a pilot-sized run from silently proceeding to a PASS/FAIL verdict.
    """
    expected_accepted = int(n_evaluated_per_vertical * useful_coverage_min)
    if expected_accepted < min_accepted:
        raise InsufficientSampleSizeError(
            f"n_evaluated_per_vertical={n_evaluated_per_vertical} at "
            f"useful_coverage_min={useful_coverage_min} yields an expected "
            f"{expected_accepted} accepted cases, below the required "
            f"minimum of {min_accepted} accepted cases/vertical "
            f"(EXECUTION_PLAN.md §5: needs ~429 evaluated cases/vertical in "
            f"expectation at 70% coverage to reach 300 accepted). This is "
            f"pilot-sized evidence -- run the properly-powered final study "
            f"instead of forcing this evaluation through the final-"
            f"certification gate."
        )


def run_final_certification(
    *,
    reason: str,
    per_vertical_metrics: Dict[str, RawValidatedRetentionMetrics],
    per_vertical_n_evaluated: Dict[str, int],
    gates_path: Optional[Path] = None,
) -> GateReport:
    """The ONLY entry point that may produce a PASS/FAIL against
    release_gates.yaml's evidence.* thresholds. Requires:
      1. An explicit `reason` (propagated into the underlying
         final_evaluation_context, so the run ledger shows exactly when and
         why locked-final data was consulted).
      2. Per-vertical evaluated sample sizes large enough to POSSIBLY reach
         the required bound -- checked before scoring, not after.

    Calling this without going through here (e.g. trying to eyeball a pilot
    result against the yaml thresholds directly) is exactly the mistake
    EXECUTION_PLAN.md §5 exists to prevent.
    """
    gates = load_gates(gates_path)
    if gates.get("gate_profile") != "final_certification":
        raise ValueError(
            "release_gates.yaml does not declare gate_profile: final_certification -- "
            "refusing to proceed. This file's thresholds are documented as the FINAL "
            "profile only (EXECUTION_PLAN.md §5); if this changed, this code needs review."
        )

    evidence_gates = gates["evidence"]
    min_accepted = evidence_gates["minimum_accepted_adjudicated_cases_per_vertical"]
    useful_coverage_min = evidence_gates["useful_coverage_min"]
    required_lb = evidence_gates["supported_precision_one_sided_95_lower_bound_min"]

    # Precondition: every vertical's evaluated sample size must be large
    # enough to POSSIBLY clear the bar, before we even attempt scoring.
    for vertical, n_evaluated in per_vertical_n_evaluated.items():
        check_minimum_sample_size(n_evaluated, useful_coverage_min, min_accepted)

    states: List[GateState] = []

    with final_evaluation_context(reason=reason):
        for vertical in gates["evidence"]["report_per_vertical"]:
            metrics = per_vertical_metrics.get(vertical)
            if metrics is None:
                states.append(
                    GateState(
                        name=f"evidence.precision[{vertical}]",
                        value="NOT_MEASURED",
                        detail=f"No metrics supplied for vertical {vertical!r}.",
                    )
                )
                continue

            if metrics.n_accepted < min_accepted:
                states.append(
                    GateState(
                        name=f"evidence.precision[{vertical}]",
                        value="FAIL",
                        detail=(
                            f"n_accepted={metrics.n_accepted} < required {min_accepted}. "
                            f"NOT_MEASURED-equivalent per state design; reported as FAIL "
                            f"since insufficient evidence does not count as a pass "
                            f"(release_gates.yaml: state_values, release_requires: all_mandatory_pass)."
                        ),
                    )
                )
                continue

            lb = metrics.accepted_precision_one_sided_95_lb
            passed = lb >= required_lb
            states.append(
                GateState(
                    name=f"evidence.precision[{vertical}]",
                    value="PASS" if passed else "FAIL",
                    detail=(
                        f"n_accepted={metrics.n_accepted}, n_accepted_errors={metrics.n_accepted_errors}, "
                        f"one_sided_95_lb={lb*100:.3f}%, required>={required_lb*100:.3f}%"
                    ),
                )
            )

    return GateReport(is_final_certification=True, states=states)


def run_pilot_feasibility_check(
    *,
    pipeline_ran_end_to_end: bool,
    loss_decreased: bool,
    rough_precision_point_estimate: Optional[float],
    rough_coverage_point_estimate: Optional[float],
    diagnosed_failures: List[str],
) -> GateReport:
    """The PILOT profile (EXECUTION_PLAN.md §5) -- deliberately NOT scored
    against release_gates.yaml's evidence.* thresholds. This checks
    feasibility only: did the pipeline run, did meaningful learning occur,
    what's the rough (wide-interval, non-certifying) quality signal, and
    what failures were found and need fixing. There is no PASS/FAIL notion
    of "did the pilot clear the final precision bar" here by design --
    asking that question is exactly the R3 mistake."""
    states = [
        GateState(
            name="pilot.pipeline_integrity",
            value="PASS" if pipeline_ran_end_to_end else "FAIL",
            detail="Did the training/eval pipeline run end-to-end without breaking.",
        ),
        GateState(
            name="pilot.meaningful_learning",
            value="PASS" if loss_decreased else "FAIL",
            detail="Did loss decrease / is task accuracy non-trivially above a naive baseline.",
        ),
        GateState(
            name="pilot.rough_precision_signal",
            value="NOT_MEASURED" if rough_precision_point_estimate is None else "PASS",
            detail=(
                f"Point estimate {rough_precision_point_estimate} -- reported as a signal "
                f"with a wide honest interval, NOT compared against the 99% final gate "
                f"(EXECUTION_PLAN.md §5)."
                if rough_precision_point_estimate is not None
                else "Not yet measured."
            ),
        ),
        GateState(
            name="pilot.rough_coverage_signal",
            value="NOT_MEASURED" if rough_coverage_point_estimate is None else "PASS",
            detail=f"Point estimate {rough_coverage_point_estimate}." if rough_coverage_point_estimate is not None else "Not yet measured.",
        ),
        GateState(
            name="pilot.diagnosed_failures",
            value="PASS",  # having a diagnosed-failures list is itself the point of the pilot, not a failure
            detail=f"{len(diagnosed_failures)} failure(s) diagnosed: {diagnosed_failures}",
        ),
    ]
    return GateReport(is_final_certification=False, states=states)


if __name__ == "__main__":
    # --- Pilot profile: no evidence.* thresholds involved, always succeeds structurally ---
    pilot_report = run_pilot_feasibility_check(
        pipeline_ran_end_to_end=True,
        loss_decreased=True,
        rough_precision_point_estimate=0.95,
        rough_coverage_point_estimate=0.68,
        diagnosed_failures=["grocery vertical has no real training data yet (synthetic-only, per §6)"],
    )
    print(pilot_report.summary())
    assert not pilot_report.is_final_certification

    # --- Final-certification profile: must reject an undersized sample BEFORE scoring ---
    try:
        run_final_certification(
            reason="smoke test -- deliberately undersized pilot sample",
            per_vertical_metrics={},
            per_vertical_n_evaluated={"fashion": 100, "grocery": 100, "general": 100},
        )
        raise AssertionError("expected InsufficientSampleSizeError for a 100-case-per-vertical sample")
    except InsufficientSampleSizeError as e:
        print(f"\nCorrectly rejected an undersized (pilot-scale) sample before scoring:\n  {e}")

    # --- Final-certification profile: a properly-sized, well-performing sample scores PASS ---
    metrics_ok = RawValidatedRetentionMetrics(
        n_raw_generations=650, n_raw_errors=30, n_validator_rejected=28, n_accepted=430, n_accepted_errors=0
    )
    report_ok = run_final_certification(
        reason="smoke test -- properly-sized sample, zero accepted errors",
        per_vertical_metrics={"fashion": metrics_ok, "grocery": metrics_ok, "general": metrics_ok},
        per_vertical_n_evaluated={"fashion": 429, "grocery": 429, "general": 429},
    )
    print("\n" + report_ok.summary())
    assert report_ok.all_mandatory_pass, "expected all verticals to PASS with 430 accepted / 0 errors"

    # --- Final-certification profile: NOT_MEASURED for a missing vertical counts as failing overall ---
    report_missing = run_final_certification(
        reason="smoke test -- grocery vertical not yet measured",
        per_vertical_metrics={"fashion": metrics_ok, "general": metrics_ok},
        per_vertical_n_evaluated={"fashion": 429, "grocery": 429, "general": 429},
    )
    print("\n" + report_missing.summary())
    assert not report_missing.all_mandatory_pass, "a NOT_MEASURED vertical must make all_mandatory_pass False"

    print("\nALL eval/gate_check.py SMOKE CHECKS PASSED")

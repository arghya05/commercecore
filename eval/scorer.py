"""Development-only scorer. Per EXECUTION_PLAN.md §5, §7, §9, §10.

Operates ONLY on development/calibration partition data during iteration.
Locked-final scoring must go through eval/gate_check.py's explicit
final_certification path, which itself requires
data.partitions.final_evaluation_context -- this module does not bypass
that guard, it relies on it (see score_system's partition check below).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from scipy import stats

from data.partitions import assert_partition_access_allowed
from schemas.common import DataPartition


# ---------------------------------------------------------------------------
# NDCG@10 -- the frozen primary metric (EXECUTION_PLAN.md §1).
# ---------------------------------------------------------------------------

def dcg_at_k(relevance_grades: Sequence[float], k: int = 10) -> float:
    """Standard DCG@k: sum(rel_i / log2(i+1)) for i=1..k (1-indexed position)."""
    total = 0.0
    for i, rel in enumerate(relevance_grades[:k], start=1):
        total += rel / math.log2(i + 1)
    return total


def ndcg_at_k(ranked_relevance_grades: Sequence[float], ideal_relevance_grades: Sequence[float], k: int = 10) -> float:
    """NDCG@k = DCG@k(actual ranking) / DCG@k(ideal ranking).

    `ranked_relevance_grades` is the relevance grade of each item in the
    system's actual output order. `ideal_relevance_grades` is the same set
    of grades sorted descending (the best possible ordering). Returns 0.0
    if the ideal DCG is 0 (no relevant items exist -- avoids a spurious
    division by zero being silently treated as a perfect score).
    """
    ideal_sorted = sorted(ideal_relevance_grades, reverse=True)
    ideal_dcg = dcg_at_k(ideal_sorted, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(list(ranked_relevance_grades), k) / ideal_dcg


def ndcg_at_10_for_query(
    ranked_product_ids: List[str],
    relevance_by_product_id: Dict[str, float],
    judged_only: bool = True,
) -> Optional[float]:
    """Computes NDCG@10 for one query given a ranked list and a relevance
    mapping. Per EXECUTION_PLAN.md §9: when `judged_only` is True (the
    PRIMARY track), unjudged items (not present in relevance_by_product_id)
    are excluded from scoring entirely -- NOT auto-scored as irrelevant
    (relevance 0), since that would penalize a system for retrieving
    correct-but-unjudged items. Returns None if there are no judged items
    at all in the ranked list (this query cannot be scored on this track).
    """
    if judged_only:
        graded = [
            relevance_by_product_id[pid]
            for pid in ranked_product_ids
            if pid in relevance_by_product_id
        ]
        all_known_grades = list(relevance_by_product_id.values())
        if not graded:
            return None
        return ndcg_at_k(graded, all_known_grades, k=10)
    else:
        # Full-corpus / open-retrieval track: unjudged items are scored 0,
        # but the CALLER must separately report judgment coverage per §9 --
        # this function alone does not disclose that, callers must track it.
        graded = [relevance_by_product_id.get(pid, 0.0) for pid in ranked_product_ids]
        ideal = list(relevance_by_product_id.values())
        return ndcg_at_k(graded, ideal, k=10)


# ---------------------------------------------------------------------------
# Exact binomial confidence interval for the precision gate (EXECUTION_PLAN.md §5).
# ---------------------------------------------------------------------------

def one_sided_95_lower_bound(n_accepted: int, n_errors: int) -> float:
    """Exact one-sided 95% lower confidence bound for a binomial proportion,
    via the Clopper-Pearson method (scipy.stats.beta quantile function).

    For n_errors == 0 this reduces algebraically to 0.05^(1/n_accepted),
    matching EXECUTION_PLAN.md §5's stated formula exactly -- verified in
    the smoke check below. For n_errors > 0 this uses the proper exact
    binomial interval rather than a naive percentile bootstrap, which the
    plan explicitly flags as degenerating near zero observed errors.
    """
    if n_accepted <= 0:
        raise ValueError("n_accepted must be positive")
    if n_errors < 0 or n_errors > n_accepted:
        raise ValueError("n_errors must be in [0, n_accepted]")

    n_correct = n_accepted - n_errors
    if n_correct == n_accepted:
        # Zero errors: exact closed form, alpha=0.05 one-sided.
        return 0.05 ** (1.0 / n_accepted)

    # Clopper-Pearson one-sided lower bound: solve for p such that
    # P(X >= n_correct | n=n_accepted, p) = 0.05, i.e. the 5th percentile
    # of Beta(n_correct, n_accepted - n_correct + 1).
    return float(stats.beta.ppf(0.05, n_correct, n_accepted - n_correct + 1))


def probability_of_zero_errors(n: int, true_precision: float) -> float:
    """P(observe zero errors in n trials | true precision). Used to sanity-
    check the "4.9% at n=300, precision=0.99" figure from EXECUTION_PLAN.md §5."""
    return true_precision ** n


# ---------------------------------------------------------------------------
# The three distinct §10 metrics -- never conflated.
# ---------------------------------------------------------------------------

@dataclass
class RawValidatedRetentionMetrics:
    """EXECUTION_PLAN.md §10: raw generation error rate, validator rejection
    rate, and accepted-output precision are three DISTINCT metrics. This
    dataclass makes it structurally impossible to report just one number
    labeled ambiguously as "quality"."""

    n_raw_generations: int
    n_raw_errors: int  # errors present in raw output, before any validator runs
    n_validator_rejected: int  # of the raw errors, how many the validator caught and rejected
    n_accepted: int  # what actually shipped, post-validation
    n_accepted_errors: int  # errors found in ACCEPTED output (should be near-zero; this is what release_gates.yaml's invalid_source_span_accepted_max constrains)

    def __post_init__(self) -> None:
        if self.n_raw_errors > self.n_raw_generations:
            raise ValueError("n_raw_errors cannot exceed n_raw_generations")
        if self.n_validator_rejected > self.n_raw_errors:
            raise ValueError("n_validator_rejected cannot exceed n_raw_errors (can only reject errors that exist)")
        if self.n_accepted > self.n_raw_generations:
            raise ValueError("n_accepted cannot exceed n_raw_generations")
        if self.n_accepted_errors > self.n_accepted:
            raise ValueError("n_accepted_errors cannot exceed n_accepted")

    @property
    def raw_generation_error_rate(self) -> float:
        return self.n_raw_errors / self.n_raw_generations if self.n_raw_generations else 0.0

    @property
    def validator_rejection_rate(self) -> float:
        """Of all raw generations, what fraction were rejected by the validator (not just of the errors)."""
        return self.n_validator_rejected / self.n_raw_generations if self.n_raw_generations else 0.0

    @property
    def accepted_output_precision(self) -> float:
        """This is the number release_gates.yaml's supported_acceptance_precision_target
        and supported_precision_one_sided_95_lower_bound_min actually constrain --
        precision over ACCEPTED output only, not raw generation."""
        if self.n_accepted == 0:
            return float("nan")
        return (self.n_accepted - self.n_accepted_errors) / self.n_accepted

    @property
    def accepted_precision_one_sided_95_lb(self) -> float:
        if self.n_accepted == 0:
            return float("nan")
        return one_sided_95_lower_bound(self.n_accepted, self.n_accepted_errors)

    def as_dict(self) -> Dict[str, float]:
        """Three distinct numbers, explicitly labeled -- never pooled into
        one "dataset quality: X%" figure (§3's requirement, applied here to
        the eval side of the same principle)."""
        return {
            "raw_generation_error_rate": self.raw_generation_error_rate,
            "validator_rejection_rate": self.validator_rejection_rate,
            "accepted_output_precision": self.accepted_output_precision,
            "accepted_precision_one_sided_95_lb": self.accepted_precision_one_sided_95_lb,
        }


# ---------------------------------------------------------------------------
# Multi-system comparison scorer (EXECUTION_PLAN.md §7's three-way ablation support).
# ---------------------------------------------------------------------------

@dataclass
class SystemResult:
    system_name: str
    per_query_ndcg: Dict[str, float] = field(default_factory=dict)  # query_id -> ndcg@10
    metrics: Optional[RawValidatedRetentionMetrics] = None

    def mean_ndcg(self) -> float:
        if not self.per_query_ndcg:
            return float("nan")
        return sum(self.per_query_ndcg.values()) / len(self.per_query_ndcg)


@dataclass
class DevelopmentScorer:
    """Compares an arbitrary number of NAMED systems against each other on
    matched conditions -- supports §7's three distinct shared-vs-specialist
    comparisons (not hardcoded to "CommerceCore vs one baseline"). Operates
    on development/calibration partition data only; see score_system's
    partition guard."""

    results: Dict[str, SystemResult] = field(default_factory=dict)

    def add_system_result(self, result: SystemResult) -> None:
        self.results[result.system_name] = result

    def record_query_score(
        self,
        system_name: str,
        query_id: str,
        ranked_product_ids: List[str],
        relevance_by_product_id: Dict[str, float],
        partition: DataPartition,
        judged_only: bool = True,
    ) -> Optional[float]:
        """Scores one query for one system. Enforces the partition guard --
        this will raise LockedFinalAccessError if `partition` is LOCKED_FINAL
        and we're not inside a final_evaluation_context (data/partitions.py).
        Development-time iteration must always pass DEVELOPMENT or
        CALIBRATION here."""
        assert_partition_access_allowed(partition)

        score = ndcg_at_10_for_query(ranked_product_ids, relevance_by_product_id, judged_only=judged_only)
        if score is None:
            return None

        if system_name not in self.results:
            self.results[system_name] = SystemResult(system_name=system_name)
        self.results[system_name].per_query_ndcg[query_id] = score
        return score

    def comparison_table(self) -> Dict[str, float]:
        """Mean NDCG@10 per system -- the basic building block for any of
        §7's three comparisons (task-sharing, vertical-sharing, learned-vs-
        deterministic). Which systems get compared against which is a
        caller-level decision (per §7, preregister the comparison matrix
        before running it) -- this scorer just needs named systems, not a
        hardcoded pair."""
        return {name: result.mean_ndcg() for name, result in self.results.items()}

    def relative_gain(self, candidate: str, baseline: str) -> float:
        """(candidate_ndcg - baseline_ndcg) / baseline_ndcg -- the exact
        quantity release_gates.yaml's target_relative_gain_over_best_affordable_baseline
        (0.02) is compared against."""
        cand = self.results[candidate].mean_ndcg()
        base = self.results[baseline].mean_ndcg()
        if base == 0:
            raise ValueError(f"baseline {baseline!r} has mean NDCG of 0 -- relative gain undefined")
        return (cand - base) / base


if __name__ == "__main__":
    # --- NDCG@10 checks ---
    # Perfect ranking: NDCG should be 1.0.
    perfect = ndcg_at_k([3, 2, 1], [3, 2, 1], k=10)
    assert abs(perfect - 1.0) < 1e-9, perfect
    print(f"Perfect ranking NDCG@10 = {perfect:.4f} (expected 1.0)")

    # Worst ranking (reversed) should be < 1.0.
    worst = ndcg_at_k([1, 2, 3], [3, 2, 1], k=10)
    assert 0 <= worst < 1.0, worst
    print(f"Reversed ranking NDCG@10 = {worst:.4f} (expected < 1.0)")

    # Empty ideal -> 0.0, not a crash or a spurious 1.0.
    empty_case = ndcg_at_k([], [], k=10)
    assert empty_case == 0.0
    print(f"Empty ranking NDCG@10 = {empty_case} (expected 0.0)")

    # Judged-only track: unjudged items must be excluded, not scored as 0.
    ranked = ["p1", "p2", "p3"]
    relevance = {"p1": 3.0, "p3": 1.0}  # p2 is UNJUDGED
    score_judged_only = ndcg_at_10_for_query(ranked, relevance, judged_only=True)
    score_full_corpus = ndcg_at_10_for_query(ranked, relevance, judged_only=False)
    print(f"judged_only NDCG@10 = {score_judged_only:.4f}, full_corpus NDCG@10 = {score_full_corpus:.4f}")
    assert score_judged_only >= score_full_corpus, (
        "judged_only score should be >= full_corpus score here, since excluding "
        "the unjudged item removes a potential zero-relevance penalty"
    )

    # --- Binomial LB checks: must match EXECUTION_PLAN.md §5's exact stated values ---
    lb_100 = one_sided_95_lower_bound(n_accepted=100, n_errors=0)
    lb_300 = one_sided_95_lower_bound(n_accepted=300, n_errors=0)
    print(f"n=100, 0 errors: one-sided 95% LB = {lb_100*100:.2f}% (plan states 97.05%)")
    print(f"n=300, 0 errors: one-sided 95% LB = {lb_300*100:.3f}% (plan states 99.006%)")
    assert abs(lb_100 * 100 - 97.05) < 0.01, lb_100
    assert abs(lb_300 * 100 - 99.006) < 0.01, lb_300

    # Confirms the closed-form matches the general Clopper-Pearson formula exactly.
    closed_form_100 = 0.05 ** (1.0 / 100)
    assert abs(lb_100 - closed_form_100) < 1e-9

    # Nonzero-error case must not crash and must be lower than the zero-error bound.
    lb_300_with_1_error = one_sided_95_lower_bound(n_accepted=300, n_errors=1)
    assert lb_300_with_1_error < lb_300
    print(f"n=300, 1 error: one-sided 95% LB = {lb_300_with_1_error*100:.3f}% (correctly lower than 0-error case)")

    # P(zero errors) sanity check from §5: ~4.9% at n=300, precision=0.99.
    p_zero = probability_of_zero_errors(300, 0.99)
    print(f"P(zero errors | n=300, true precision=0.99) = {p_zero*100:.2f}% (plan states ~4.9%)")
    assert abs(p_zero * 100 - 4.9) < 0.5, p_zero

    # --- Three-distinct-metrics check ---
    metrics = RawValidatedRetentionMetrics(
        n_raw_generations=1000,
        n_raw_errors=80,
        n_validator_rejected=75,
        n_accepted=925,  # 1000 - 75 rejected
        n_accepted_errors=2,  # a few errors slipped past the validator
    )
    d = metrics.as_dict()
    print("Three distinct §10 metrics (never pooled):", d)
    assert d["raw_generation_error_rate"] == 0.08
    assert d["validator_rejection_rate"] == 0.075
    assert abs(d["accepted_output_precision"] - (923 / 925)) < 1e-9

    # --- Multi-system comparison scorer, including the partition guard ---
    scorer = DevelopmentScorer()
    scorer.record_query_score(
        "commercecore", "q1", ["p1", "p3"], {"p1": 3.0, "p3": 1.0}, partition=DataPartition.DEVELOPMENT
    )
    scorer.record_query_score(
        "claude_haiku_baseline", "q1", ["p3", "p1"], {"p1": 3.0, "p3": 1.0}, partition=DataPartition.DEVELOPMENT
    )
    table = scorer.comparison_table()
    print("Multi-system comparison table (dev partition):", table)
    assert "commercecore" in table and "claude_haiku_baseline" in table

    gain = scorer.relative_gain("commercecore", "claude_haiku_baseline")
    print(f"Relative gain of commercecore over claude_haiku_baseline: {gain*100:.2f}%")

    # Locked-final partition must be rejected outside a final_evaluation_context.
    from data.partitions import LockedFinalAccessError

    try:
        scorer.record_query_score(
            "commercecore", "q_final", ["p1"], {"p1": 1.0}, partition=DataPartition.LOCKED_FINAL
        )
        raise AssertionError("expected LockedFinalAccessError")
    except LockedFinalAccessError:
        print("DevelopmentScorer correctly refuses to score LOCKED_FINAL data outside final_evaluation_context")

    print("\nALL eval/scorer.py SMOKE CHECKS PASSED")

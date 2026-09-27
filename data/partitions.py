"""Data partition enforcement. Per EXECUTION_PLAN.md §2 (R1, P0).

This is the single most important mechanism in Phase 1: the review's R1
finding was that WANDS (and the final retention suite) got called
"untouched" in one section while another section proposed using WANDS
failures to steer synthetic-data generation. That is test-informed
development, and confidence intervals computed afterward do not repair it
-- the fix has to be a hard runtime barrier, not a documentation note.

Four partitions:
  - TRAINING: fits model weights.
  - DEVELOPMENT: all iteration -- prompts, mixture reweighting, hyperparameters,
    architecture choices, diagnosing pilot underperformance.
  - CALIBRATION: freezes acceptance thresholds / baseline selection before
    final scoring. Separate from DEVELOPMENT so threshold-picking doesn't
    quietly become test-set mining either.
  - LOCKED_FINAL: WANDS in full, the final joint-task eval suite, the final
    retention audit. Used exactly once, after every other decision is frozen.

The enforcement mechanism: reading LOCKED_FINAL data requires an explicit,
named "final evaluation" context manager. Any attempt to read locked-final
data outside that context raises LockedFinalAccessError immediately.
"""

from __future__ import annotations

import hashlib
import json
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from schemas.common import DataPartition


class LockedFinalAccessError(RuntimeError):
    """Raised when code attempts to read the locked-final partition outside
    an explicit final-evaluation context. This is the P0 enforcement for
    EXECUTION_PLAN.md §2/R1 -- it must be loud and immediate, not a warning."""


class _PartitionGuardState(threading.local):
    def __init__(self) -> None:
        self.final_evaluation_active: bool = False
        self.final_evaluation_reason: Optional[str] = None


_guard_state = _PartitionGuardState()


@contextmanager
def final_evaluation_context(reason: str) -> Iterator[None]:
    """The ONLY way locked-final data may be read. `reason` is logged so a
    reviewer of the run ledger (EXECUTION_PLAN.md §11) can see exactly when
    and why locked-final was touched -- this should happen once per study.
    """
    if not reason or not reason.strip():
        raise ValueError("final_evaluation_context requires a non-empty reason string")
    prev = _guard_state.final_evaluation_active
    prev_reason = _guard_state.final_evaluation_reason
    _guard_state.final_evaluation_active = True
    _guard_state.final_evaluation_reason = reason
    try:
        yield
    finally:
        _guard_state.final_evaluation_active = prev
        _guard_state.final_evaluation_reason = prev_reason


def assert_partition_access_allowed(partition: DataPartition) -> None:
    """Call this before reading any record. Raises if `partition` is
    LOCKED_FINAL and we are not inside `final_evaluation_context`."""
    if partition == DataPartition.LOCKED_FINAL and not _guard_state.final_evaluation_active:
        raise LockedFinalAccessError(
            "Attempted to read LOCKED_FINAL partition data outside an explicit "
            "final_evaluation_context. Per EXECUTION_PLAN.md §2 (R1, P0), the "
            "locked-final partition (WANDS, the final joint-task suite, the "
            "final retention audit) must never inform iteration, prompt "
            "changes, mixture reweighting, or architecture choices. If this "
            "is genuinely the one-time final confirmatory evaluation, wrap "
            "the call in `with final_evaluation_context('...'): ...`."
        )


@dataclass
class PartitionRecord:
    """One record's partition assignment plus the metadata needed for the
    completion-evidence requirements in EXECUTION_PLAN.md §2: split hashes,
    family-id exclusion checks, and a frozen final-run manifest."""

    record_id: str
    partition: DataPartition
    family_id: Optional[str] = None
    source: str = ""


@dataclass
class PartitionManifest:
    """A versioned, hashable record of every partition assignment made for
    a study. This is the "split hashes" completion evidence from §2 --
    write one of these to disk before any run and treat it as immutable."""

    records: List[PartitionRecord] = field(default_factory=list)
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def add(self, record: PartitionRecord) -> None:
        self.records.append(record)

    def partition_of(self, record_id: str) -> DataPartition:
        for r in self.records:
            if r.record_id == record_id:
                return r.partition
        raise KeyError(f"record_id {record_id!r} not found in manifest")

    def family_ids_in_partition(self, partition: DataPartition) -> set:
        return {r.family_id for r in self.records if r.partition == partition and r.family_id}

    def check_no_family_overlap(
        self, a: DataPartition, b: DataPartition
    ) -> List[str]:
        """Returns list of family_ids present in both partitions -- should be
        empty. A nonempty result means a query/product family leaked across
        partitions (e.g. a paraphrase of a locked-final query ended up in
        training), which is exactly the kind of leakage §2 requires excluding."""
        overlap = self.family_ids_in_partition(a) & self.family_ids_in_partition(b)
        return sorted(overlap)

    def counts_by_partition(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in self.records:
            counts[r.partition.value] = counts.get(r.partition.value, 0) + 1
        return counts

    def compute_split_hash(self, partition: DataPartition) -> str:
        """Deterministic hash of the sorted record_ids in a partition --
        the "split hashes" completion evidence from §2. Changing which
        records are in a partition changes this hash, making silent
        partition drift detectable."""
        ids = sorted(r.record_id for r in self.records if r.partition == partition)
        payload = json.dumps(ids, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict:
        return {
            "created_at_utc": self.created_at_utc,
            "counts_by_partition": self.counts_by_partition(),
            "split_hashes": {
                p.value: self.compute_split_hash(p) for p in DataPartition
            },
            "records": [
                {
                    "record_id": r.record_id,
                    "partition": r.partition.value,
                    "family_id": r.family_id,
                    "source": r.source,
                }
                for r in self.records
            ],
        }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: Path) -> "PartitionManifest":
        data = json.loads(path.read_text())
        manifest = cls(created_at_utc=data["created_at_utc"])
        for rec in data["records"]:
            manifest.add(
                PartitionRecord(
                    record_id=rec["record_id"],
                    partition=DataPartition(rec["partition"]),
                    family_id=rec.get("family_id"),
                    source=rec.get("source", ""),
                )
            )
        return manifest


if __name__ == "__main__":
    # Smoke check.
    manifest = PartitionManifest()
    manifest.add(PartitionRecord(record_id="q1", partition=DataPartition.TRAINING, family_id="fam-a", source="synthetic"))
    manifest.add(PartitionRecord(record_id="q2", partition=DataPartition.DEVELOPMENT, family_id="fam-b", source="synthetic"))
    manifest.add(PartitionRecord(record_id="q3", partition=DataPartition.LOCKED_FINAL, family_id="fam-c", source="wands"))

    assert manifest.partition_of("q3") == DataPartition.LOCKED_FINAL
    overlap = manifest.check_no_family_overlap(DataPartition.TRAINING, DataPartition.LOCKED_FINAL)
    assert overlap == [], f"unexpected family overlap: {overlap}"
    print("Partition manifest basic ops OK:", manifest.counts_by_partition())

    # The core P0 enforcement: locked-final access must raise outside the context.
    try:
        assert_partition_access_allowed(DataPartition.LOCKED_FINAL)
        raise AssertionError("expected LockedFinalAccessError to be raised")
    except LockedFinalAccessError:
        print("LockedFinalAccessError correctly raised outside final_evaluation_context")

    # And must succeed inside the context.
    with final_evaluation_context(reason="smoke test of the enforcement mechanism itself"):
        assert_partition_access_allowed(DataPartition.LOCKED_FINAL)
        print("Locked-final access correctly allowed inside final_evaluation_context")

    # And must revert to raising once the context exits.
    try:
        assert_partition_access_allowed(DataPartition.LOCKED_FINAL)
        raise AssertionError("expected LockedFinalAccessError after context exit")
    except LockedFinalAccessError:
        print("LockedFinalAccessError correctly raised again after context exit")

    # Non-locked partitions are always allowed.
    assert_partition_access_allowed(DataPartition.TRAINING)
    assert_partition_access_allowed(DataPartition.DEVELOPMENT)
    assert_partition_access_allowed(DataPartition.CALIBRATION)
    print("Non-locked partition access correctly always allowed")

    # Save/load round-trip.
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "manifest.json"
        manifest.save(p)
        reloaded = PartitionManifest.load(p)
        assert reloaded.counts_by_partition() == manifest.counts_by_partition()
        assert reloaded.compute_split_hash(DataPartition.TRAINING) == manifest.compute_split_hash(DataPartition.TRAINING)
        print("Manifest save/load round-trip OK")

    print("ALL data/partitions.py SMOKE CHECKS PASSED")

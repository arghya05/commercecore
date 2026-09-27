"""Shared types used across all four task schemas.

Per EXECUTION_PLAN.md §3: every labeled example must carry a `label_origin`
so label-authority tiers are never pooled into one blended "quality" number.
Per plans/00_shared_core.md: evidence spans are half-open character offsets
`[start, end)` against immutable source text; matching `source[start:end]`
against the claimed span checks binding only, not semantic entailment
(EXECUTION_PLAN.md §3's schema-validity / source-span-binding / execution-
correctness / semantic-fidelity distinction applies here directly).
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class LabelOrigin(str, Enum):
    """Label-authority tier. Never pool these into one quality number (EXECUTION_PLAN.md §3)."""

    NATIVE_HUMAN = "native_human"  # e.g. QueryNER's native segmentation labels
    SIMULATOR_VERIFIED = "simulator_verified"  # programmatically-scored controlled scenario
    WEAK_SYNTHETIC = "weak_synthetic"  # LLM-paraphrase label, not independently verified


class Vertical(str, Enum):
    FASHION = "fashion"
    GROCERY = "grocery"
    GENERAL = "general"


class DataPartition(str, Enum):
    """EXECUTION_PLAN.md §2 — training / development / calibration / locked-final."""

    TRAINING = "training"
    DEVELOPMENT = "development"
    CALIBRATION = "calibration"
    LOCKED_FINAL = "locked_final"


class EvidenceSpan(BaseModel):
    """Half-open character span `[start, end)` against immutable source text.

    `source[start:end] == text` checks binding, not semantic entailment
    (plans/00_shared_core.md, "Contracts" section).
    """

    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: str

    @model_validator(mode="after")
    def _check_span_order(self) -> "EvidenceSpan":
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be > start ({self.start})")
        return self

    def is_bound_to(self, source: str) -> bool:
        """Source-span-binding check only (EXECUTION_PLAN.md §3, property 2).

        This is NOT a semantic-fidelity check. A span can be bound (this
        returns True) while still supporting a false claim -- e.g. citing
        "cotton" as evidence for a "contains cotton" claim when the source
        text actually reads "contains no cotton". See
        validators/mechanical_validator.py and
        validators/adversarial_fixtures.py for why this check alone is
        insufficient.
        """
        if self.start < 0 or self.end > len(source):
            return False
        return source[self.start : self.end] == self.text


class MatchState(str, Enum):
    """plans/00_shared_core.md: match states are pass/fail/unknown/not_applicable per requirement."""

    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class LabeledExample(BaseModel):
    """Mixin-style base carrying the required provenance fields.

    Every training/eval record for every task must carry these fields --
    this is what makes the §3/§6 label-origin and task/vertical/source
    tracking enforceable in code, not just in documentation.
    """

    label_origin: LabelOrigin
    vertical: Vertical
    partition: DataPartition
    source: str = Field(description="e.g. 'queryner', 'esci', 'synthetic_pipeline'")
    family_id: Optional[str] = Field(
        default=None,
        description="Query/product family grouping for split-integrity checks (EXECUTION_PLAN.md §2/§5).",
    )

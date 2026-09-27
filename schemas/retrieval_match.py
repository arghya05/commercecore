"""Use case 3 -- retrieval and requirement-aware matching. Per plans/03_retrieval_and_matching.md."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from .common import LabeledExample, MatchState


class RelationClass(str, Enum):
    """plans/03: distinguish exact matches, substitutes, complements, irrelevant items."""

    EXACT_MATCH = "exact_match"
    SUBSTITUTE = "substitute"
    COMPLEMENT = "complement"
    IRRELEVANT = "irrelevant"


class RequirementCheck(BaseModel):
    """One explicit requirement evaluated against one candidate product.

    plans/00_shared_core.md: match states are pass/fail/unknown/not_applicable.
    A hard contradiction always blocks a claim that the item satisfies that
    requirement; a missing field is not itself a negative factual label
    (i.e. absence of evidence is `unknown`, not `fail`).
    """

    requirement_field: str
    state: MatchState
    evidence_span_text: Optional[str] = None


class MatchRecord(LabeledExample):
    """A single query-product pair evaluation, feeding the shared decoder's ~12k relation/match records."""

    query_id: str
    product_id: str
    relation: RelationClass
    requirement_checks: List[RequirementCheck] = Field(default_factory=list)

    def has_unsupported_pass(self) -> bool:
        """True if any requirement is marked PASS with no evidence -- a validator-catchable error.

        This corresponds to EXECUTION_PLAN.md §10's "accepted-output precision"
        gate: a PASS with no evidence_span_text should be rejected before
        acceptance, not silently counted as a correct match.
        """
        return any(
            rc.state == MatchState.PASS and not rc.evidence_span_text
            for rc in self.requirement_checks
        )


class RetrievalJudgment(BaseModel):
    """A single (query, product) relevance judgment from a judged pool (ESCI/WANDS style).

    EXECUTION_PLAN.md §9: primary comparison runs on the existing judged
    candidate pool. `judged` distinguishes a real judgment from an unjudged
    item that must NOT be auto-scored as irrelevant in full-corpus mode.
    """

    query_id: str
    product_id: str
    judged: bool
    relevance_grade: Optional[int] = Field(
        default=None, description="e.g. ESCI E/S/C/I mapped to an ordinal grade, or WANDS grade"
    )


class RankedList(BaseModel):
    """One system's ranked output for one query, for NDCG@10/Recall@100/MRR scoring."""

    query_id: str
    system_name: str
    ranked_product_ids: List[str] = Field(max_length=1000)
    retrieval_track: str = Field(description="'judged_pool' or 'full_corpus' (EXECUTION_PLAN.md §9)")

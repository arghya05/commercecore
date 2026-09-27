"""Use case 1 -- query intent and constraints. Per plans/01_query_intent_and_constraints.md."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, Field

from .common import EvidenceSpan, LabeledExample


class Intent(str, Enum):
    PRODUCT_SEARCH = "product_search"
    PRODUCT_COMPARISON = "product_comparison"
    PRODUCT_INFORMATION = "product_information"
    ORDER_SUPPORT = "order_support"
    POLICY_SUPPORT = "policy_support"
    UNKNOWN = "unknown"


class Operator(str, Enum):
    EQ = "eq"
    NEQ = "neq"
    LT = "lt"
    LTE = "lte"
    GT = "gt"
    GTE = "gte"
    IN = "in"
    NOT_IN = "not_in"
    RANGE = "range"


class Decision(str, Enum):
    ACCEPT = "accept"
    CLARIFY = "clarify"
    FALLBACK = "fallback"


class Constraint(BaseModel):
    """plans/01: canonical field, operator, raw/normalized value, unit, explicit/soft, evidence."""

    canonical_field: str
    operator: Operator
    raw_value: str
    normalized_value: Optional[str] = None
    unit_or_currency_or_size_system: Optional[str] = None
    is_explicit: bool = Field(description="True = explicit constraint, False = soft preference")
    evidence: EvidenceSpan


class PredicateNode(BaseModel):
    """Bounded predicate tree node -- boolean AND/OR/NOT over constraints.

    plans/01: "red or blue, not leather" must be `(red OR blue) AND NOT leather`,
    not a flat conjunction of all three. Depth/size overflow must be rejected
    by the caller before this is accepted (see MAX_PREDICATE_DEPTH below).
    """

    op: Optional[str] = Field(default=None, description="'AND' | 'OR' | 'NOT' | None (leaf)")
    constraint: Optional[Constraint] = Field(default=None, description="Set only for leaf nodes")
    children: List["PredicateNode"] = Field(default_factory=list)


PredicateNode.model_rebuild()

MAX_PREDICATE_DEPTH = 8
MAX_PREDICATE_NODES = 64


def predicate_tree_depth(node: PredicateNode, _depth: int = 0) -> int:
    if not node.children:
        return _depth
    return max(predicate_tree_depth(c, _depth + 1) for c in node.children)


def predicate_tree_size(node: PredicateNode) -> int:
    return 1 + sum(predicate_tree_size(c) for c in node.children)


class QueryContract(LabeledExample):
    """POST /v1/query/understand response, per plans/01_query_intent_and_constraints.md.

    This is the object the mechanical validator's "schema validity" check
    (EXECUTION_PLAN.md §3, property 1) parses against, and whose evidence
    spans the "source-span binding" check (property 2) verifies.
    """

    query_text: str
    intent: Intent
    target_product: Optional[str] = None
    candidate_categories: List[str] = Field(default_factory=list)
    predicate: Optional[PredicateNode] = None
    soft_preferences: List[Constraint] = Field(default_factory=list)
    ambiguities: List[str] = Field(default_factory=list)
    decision: Decision

    def is_within_bounds(self) -> bool:
        if self.predicate is None:
            return True
        return (
            predicate_tree_depth(self.predicate) <= MAX_PREDICATE_DEPTH
            and predicate_tree_size(self.predicate) <= MAX_PREDICATE_NODES
        )

    def all_evidence_spans(self) -> List[EvidenceSpan]:
        spans: List[EvidenceSpan] = []

        def _walk(node: PredicateNode) -> None:
            if node.constraint is not None:
                spans.append(node.constraint.evidence)
            for c in node.children:
                _walk(c)

        if self.predicate is not None:
            _walk(self.predicate)
        for sp in self.soft_preferences:
            spans.append(sp.evidence)
        return spans

"""Use case 4 -- controlled recovery and read-only tool plans. Per plans/04_search_recovery_and_tools.md."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .common import LabeledExample


class RecoveryAction(str, Enum):
    RETAIN = "retain"
    REWRITE = "rewrite"
    CLARIFY = "clarify"
    NO_VERIFIED_MATCH = "no_verified_match"
    SUGGEST_RELAXATION = "suggest_relaxation"


class ToolName(str, Enum):
    """plans/04 v0 allowlist -- no checkout/refund/return/order-modification tools exist yet."""

    SEARCH_PRODUCTS = "search_products"
    GET_PRODUCT_DETAILS = "get_product_details"
    GET_PRICE = "get_price"
    CHECK_INVENTORY = "check_inventory"
    GET_POLICY = "get_policy"


class ToolCall(BaseModel):
    tool: ToolName
    arguments: Dict[str, Any] = Field(default_factory=dict)


MAX_TOOL_CALLS_PER_PLAN = 5  # plans/04: "maximum call budget" -- enforced, not left open-ended


class ToolPlan(LabeledExample):
    """POST /v1/tools/plan response. Validated arguments only; execution is server code, not the model."""

    calls: List[ToolCall] = Field(default_factory=list)

    def is_within_call_budget(self) -> bool:
        return len(self.calls) <= MAX_TOOL_CALLS_PER_PLAN

    def uses_only_allowed_tools(self) -> bool:
        """Trivially true given the ToolName enum, but kept explicit as the
        security release gate (release_gates.yaml: security.arbitrary_tool_execution_test_failures_max: 0)
        this check exists to satisfy -- any future tool addition must update
        both the enum and this check deliberately, not implicitly.
        """
        return all(isinstance(c.tool, ToolName) for c in self.calls)


class RecoveryRecord(LabeledExample):
    """POST /v1/search/recover response."""

    original_query: str
    action: RecoveryAction
    rewritten_query: Optional[str] = Field(
        default=None, description="At most one alternate text query with unchanged explicit predicates"
    )
    clarification_question: Optional[str] = None
    relaxation_proposal: Optional[str] = None
    explanation: Optional[str] = None

    def silently_changes_constraints(self, original_predicate_hash: str, new_predicate_hash: str) -> bool:
        """Placeholder check: a rewrite/relaxation must not silently alter explicit predicates.

        Real implementation needs the actual predicate-tree hash from
        schemas.query_intent.QueryContract -- this stub documents the
        release-gate requirement (no mutated hard constraints) so Phase 2
        wires it to the real predicate comparison once query contracts flow
        through this record type.
        """
        if self.action in (RecoveryAction.RETAIN,):
            return False
        return original_predicate_hash != new_predicate_hash

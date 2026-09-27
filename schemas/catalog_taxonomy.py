"""Use case 2 -- catalog understanding and taxonomy mapping. Per plans/02_catalog_and_taxonomy.md."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from .common import EvidenceSpan, LabeledExample


class FieldApplicability(str, Enum):
    """plans/02: distinguish unavailable information -- not every field applies to every product."""

    PRESENT = "present"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class CatalogField(BaseModel):
    """plans/02: typed field with raw value, canonical value, unit, applicability, source, span.

    `2 x 500 ml` -> count=2, per_unit=500ml, total=1000ml -- these are three
    distinct facts, not interchangeable with "one 1-litre container".
    """

    field_name: str
    applicability: FieldApplicability
    raw_value: Optional[str] = None
    canonical_value: Optional[str] = None
    unit: Optional[str] = None
    conversion_rule_version: Optional[str] = None
    source_field: Optional[str] = Field(
        default=None, description="Which input field this was extracted from (title, bullet_points, ...)"
    )
    evidence: Optional[EvidenceSpan] = None


class CatalogNormalizationRecord(LabeledExample):
    """POST /v1/catalog/normalize response."""

    product_id: str
    fields: List[CatalogField]


class TaxonomyCandidate(BaseModel):
    category_id: str
    path: str
    description: Optional[str] = None


class TaxonomyMappingRecord(LabeledExample):
    """POST /v1/taxonomy/map response.

    plans/02: candidate IDs must come from the request's candidate set (or
    be `none_of_candidates`) -- no invented leaves. Candidate recall (did
    the retrieval step even include the right category) must be measured
    separately from selection accuracy (did the model pick correctly among
    what it was given) -- a selector cannot recover a category the
    candidate-retrieval step omitted.
    """

    product_id: str
    candidates: List[TaxonomyCandidate] = Field(max_length=20)
    selected_category_id: Optional[str] = Field(
        default=None, description="None means none_of_candidates was selected"
    )
    none_of_candidates: bool = False

    def is_valid_selection(self) -> bool:
        """Schema-validity check (EXECUTION_PLAN.md §3, property 1): no invented leaves."""
        if self.none_of_candidates:
            return self.selected_category_id is None
        if self.selected_category_id is None:
            return False
        return self.selected_category_id in {c.category_id for c in self.candidates}

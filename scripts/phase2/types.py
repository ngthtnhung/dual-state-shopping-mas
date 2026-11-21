from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class HardConstraint:
    attribute: str
    operator: str
    value: float
    unit: str | None
    source_text: str


@dataclass
class QueryState:
    text: str
    hard_constraints: list[HardConstraint]
    target_attributes: list[str]
    uncertainty: dict[str, float]
    vague_terms: list[str]


@dataclass
class CandidateProfile:
    product: dict[str, Any]
    specs: dict[str, str]
    spec_evidence: list[dict[str, Any]]
    retrieval_score: float

    @property
    def parent_asin(self) -> str:
        return str(self.product.get("parent_asin", ""))


@dataclass
class CatalogState:
    candidates: list[CandidateProfile]
    preliminary_candidate_count: int
    completeness: dict[str, float]
    entropy: dict[str, float]


@dataclass
class Conflict:
    kind: str
    parent_asin: str
    attribute: str
    summary: str
    evidence: list[dict[str, Any]]
    confidence: float


@dataclass
class Decision:
    action: str
    rationale: list[str]
    recommendations: list[dict[str, Any]] = field(default_factory=list)
    clarification: dict[str, Any] | None = None
    conflicts: list[Conflict] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)


@dataclass
class RunTrace:
    query_state: QueryState
    catalog_state: CatalogState
    actions_taken: list[str]
    spec_agent_calls: int
    review_agent_calls: int
    review_source_records_scanned: int
    evidence: list[dict[str, Any]]
    conflicts: list[Conflict]
    decision: Decision

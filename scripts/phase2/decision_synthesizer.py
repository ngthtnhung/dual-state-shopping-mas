from __future__ import annotations

from typing import Any

from scripts.phase2.query_analyzer import SUPPORTED_ATTRIBUTES
from scripts.phase2.types import CandidateProfile, CatalogState, Conflict, Decision, QueryState


class DecisionSynthesizer:
    """Produce cited recommendations or an explicit clarification/disclosure."""

    def synthesize(
        self,
        query: QueryState,
        catalog: CatalogState,
        evidence: list[dict[str, Any]],
        conflicts: list[Conflict],
        *,
        clarification_attributes: list[str] | None = None,
        unresolved_attributes: list[str] | None = None,
        evidence_agents_available: bool = True,
    ) -> Decision:
        clarification_attributes = clarification_attributes or []
        unresolved_attributes = unresolved_attributes or []
        if clarification_attributes:
            return Decision(
                action="ASK_CLARIFICATION",
                rationale=[
                    "The query is vague on an attribute whose candidate-catalog values are sufficiently covered and differentiated."
                ],
                clarification={
                    "attributes": clarification_attributes,
                    "question": (
                        "Which preference matters most for "
                        + ", ".join(clarification_attributes)
                        + "?"
                    ),
                    "probe_waste_prevented": True,
                },
            )
        if not catalog.candidates:
            return Decision(
                action="ABSTAIN_DISCLOSE",
                rationale=["No catalog candidates satisfy the parsed hard constraints and retrieval terms."],
                limitations=["No unsupported product recommendation was generated."],
            )

        evidence_by_parent: dict[str, list[dict[str, Any]]] = {}
        for row in evidence:
            parent = str(row.get("parent_asin") or "")
            if parent:
                evidence_by_parent.setdefault(parent, []).append(row)
        missing_attrs = [
            attribute for attribute in query.target_attributes
            if attribute in SUPPORTED_ATTRIBUTES
            and not any(
                candidate.specs.get(attribute)
                or any(
                    row.get("attribute") == attribute
                    or row.get("aspect_category") == attribute
                    for row in evidence_by_parent.get(candidate.parent_asin, [])
                )
                for candidate in catalog.candidates
            )
        ]
        conflict_parents = {conflict.parent_asin for conflict in conflicts}
        recommendations: list[dict[str, Any]] = []
        for candidate in catalog.candidates[:5]:
            parent = candidate.parent_asin
            cited = [
                *candidate.spec_evidence,
                *evidence_by_parent.get(parent, []),
            ]
            if not cited:
                continue
            recommendations.append(
                {
                    "parent_asin": parent,
                    "title": candidate.product.get("title"),
                    "price": candidate.product.get("price"),
                    "average_rating": candidate.product.get("average_rating"),
                    "specs": candidate.specs,
                    "retrieval_score": round(candidate.retrieval_score, 4),
                    "citations": cited,
                    "conflict_disclosure_required": parent in conflict_parents,
                }
            )

        if conflicts:
            return Decision(
                action="ABSTAIN_DISCLOSE",
                rationale=["The evidence contains material spec/review or variant sentiment conflicts."],
                recommendations=recommendations,
                conflicts=conflicts,
                limitations=["Conflicting review sentiment is disclosed and is not treated as verified product fact."],
            )

        if unresolved_attributes:
            return Decision(
                action="ABSTAIN_DISCLOSE",
                rationale=["High-uncertainty requested attributes could not be clarified from sufficiently covered, differentiated catalog evidence."],
                recommendations=recommendations,
                limitations=[
                    f"Unresolved ambiguous attributes: {', '.join(unresolved_attributes)}.",
                    "No definitive recommendation is made for these unresolved preferences.",
                ],
            )

        if missing_attrs:
            limitations = [
                f"Evidence is missing for: {', '.join(missing_attrs)}."
            ]
            if not evidence_agents_available:
                return Decision(
                    action="ACTIVATE_EVIDENCE_AGENTS",
                    rationale=["Requested attributes remain unsupported; specialized evidence agents are unavailable."],
                    recommendations=recommendations,
                    limitations=limitations,
                )
            return Decision(
                action="ABSTAIN_DISCLOSE",
                rationale=["Evidence agents could not recover all requested attributes; avoid a fully supported recommendation."],
                recommendations=recommendations,
                conflicts=conflicts,
                limitations=limitations,
            )

        if not recommendations:
            return Decision(
                action="ABSTAIN_DISCLOSE",
                rationale=["Retrieved candidates lack citable evidence for a grounded recommendation."],
                limitations=["No unsupported product recommendation was generated."],
            )

        limitations: list[str] = []
        unsupported = sorted(set(query.target_attributes) - SUPPORTED_ATTRIBUTES)
        if unsupported:
            limitations.append(
                f"These requested attributes are not supported by current extractors: {', '.join(unsupported)}."
            )
        if missing_attrs:
            limitations.append(
                f"Evidence remains incomplete for: {', '.join(missing_attrs)}."
            )
        if any(
            row.get("kind") == "review_sentiment" or "sentiment" in row
            for row in evidence
        ):
            limitations.append(
                "Review sentiment is experiential evidence, not a verified specification value."
            )
        return Decision(
            action="RECOMMEND",
            rationale=["Recommendations are limited to retrieved catalog candidates and carry source citations."],
            recommendations=recommendations,
            limitations=limitations,
        )

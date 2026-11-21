from __future__ import annotations

import re
from typing import Any

from scripts.phase2.catalog_profiler import CatalogProfiler
from scripts.phase2.conflict_evaluator import ConflictEvaluator
from scripts.phase2.decision_synthesizer import DecisionSynthesizer
from scripts.phase2.evidence_agents import ReviewEvidenceAgent, SpecDescriptionAgent
from scripts.phase2.query_analyzer import QueryAnalyzer, SUPPORTED_ATTRIBUTES
from scripts.phase2.types import CatalogState, QueryState, RunTrace


class AdaptiveOrchestrator:
    """Coordinate query-state and evidence-state gates with early exits."""

    def __init__(
        self,
        catalog_profiler: CatalogProfiler,
        *,
        query_analyzer: QueryAnalyzer | None = None,
        spec_agent: SpecDescriptionAgent | None = None,
        review_agent: Any | None = None,
        conflict_evaluator: ConflictEvaluator | None = None,
        decision_synthesizer: DecisionSynthesizer | None = None,
        ambiguity_threshold: float = 0.5,
        minimum_clarification_coverage: float = 0.5,
        minimum_catalog_entropy: float = 0.3,
    ) -> None:
        if not 0 <= ambiguity_threshold <= 1:
            raise ValueError("ambiguity_threshold must be between zero and one")
        if not 0 <= minimum_clarification_coverage <= 1:
            raise ValueError("minimum_clarification_coverage must be between zero and one")
        if not 0 <= minimum_catalog_entropy <= 1:
            raise ValueError("minimum_catalog_entropy must be between zero and one")
        self.catalog_profiler = catalog_profiler
        self.query_analyzer = query_analyzer or QueryAnalyzer()
        self.spec_agent = spec_agent or SpecDescriptionAgent()
        self.review_agent = review_agent
        self.conflict_evaluator = conflict_evaluator or ConflictEvaluator()
        self.decision_synthesizer = decision_synthesizer or DecisionSynthesizer()
        self.ambiguity_threshold = ambiguity_threshold
        self.minimum_clarification_coverage = minimum_clarification_coverage
        self.minimum_catalog_entropy = minimum_catalog_entropy

    def run(self, text: str) -> RunTrace:
        query_state = self.query_analyzer.analyze(text)
        catalog_state = self.catalog_profiler.retrieve(query_state)
        return self.run_states(query_state, catalog_state)

    def run_states(
        self, query_state: QueryState, catalog_state: CatalogState
    ) -> RunTrace:
        """Route precomputed states, useful when evaluating many threshold settings."""
        actions: list[str] = ["ANALYZE_QUERY", "RETRIEVE_AND_PROFILE"]
        clarification_attributes = [
            attribute
            for attribute in query_state.target_attributes
            if attribute in SUPPORTED_ATTRIBUTES
            and query_state.uncertainty.get(attribute, 0) >= self.ambiguity_threshold
            and catalog_state.completeness.get(attribute, 0)
            >= self.minimum_clarification_coverage
            and catalog_state.entropy.get(attribute, 0)
            >= self.minimum_catalog_entropy
        ]
        unresolved_attributes = [
            attribute
            for attribute in query_state.target_attributes
            if attribute in SUPPORTED_ATTRIBUTES
            and query_state.uncertainty.get(attribute, 0) >= self.ambiguity_threshold
            and attribute not in clarification_attributes
        ]
        if clarification_attributes:
            actions.append("ASK_CLARIFICATION")
            decision = self.decision_synthesizer.synthesize(
                query_state,
                catalog_state,
                [],
                [],
                clarification_attributes=clarification_attributes,
            )
            return RunTrace(
                query_state, catalog_state, actions, 0, 0, 0, [], [], decision
            )

        requested = set(query_state.target_attributes) & SUPPORTED_ATTRIBUTES
        missing = {
            attribute for attribute in requested
            if catalog_state.completeness.get(attribute, 0) < 1
        }
        lower_query = query_state.text.casefold()
        experiential_intent = bool(
            re.search(
                r"\b(?:experience|feels|comfortable|reliable|reviews?)\b",
                lower_query,
            )
            or re.search(
                r"\b(?:battery|charge)\b.{0,24}\b(?:lasts?|hours?)\b",
                lower_query,
            )
        )
        evidence: list[dict[str, Any]] = []
        spec_calls = 0
        review_calls = 0
        scanned = 0
        if missing:
            actions.append("ACTIVATE_EVIDENCE_AGENTS")
            spec_calls = 1
            for candidate in catalog_state.candidates:
                evidence.extend(self.spec_agent.extract(candidate, missing))
        if (missing or experiential_intent) and self.review_agent is not None:
            review_calls = 1
            review_rows, scanned = self.review_agent.collect(
                catalog_state.candidates, requested or missing
            )
            evidence.extend(review_rows)
            actions.append("ACTIVATE_REVIEW_ABSA_AGENT")

        conflicts = self.conflict_evaluator.evaluate(catalog_state.candidates, evidence)
        if conflicts:
            actions.append("DISCLOSE_CONFLICT")
        else:
            actions.append("SYNTHESIZE_DECISION")
        decision = self.decision_synthesizer.synthesize(
            query_state,
            catalog_state,
            evidence,
            conflicts,
            unresolved_attributes=unresolved_attributes,
            evidence_agents_available=self.review_agent is not None,
        )
        return RunTrace(
            query_state,
            catalog_state,
            actions,
            spec_calls,
            review_calls,
            scanned,
            evidence,
            conflicts,
            decision,
        )

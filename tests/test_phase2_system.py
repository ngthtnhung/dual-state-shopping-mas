from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from scripts.phase2.catalog_profiler import CatalogProfiler
from scripts.phase2.conflict_evaluator import ConflictEvaluator
from scripts.phase2.evidence_agents import SpecDescriptionAgent
from scripts.phase2.orchestrator import AdaptiveOrchestrator
from scripts.phase2.query_analyzer import QueryAnalyzer
from scripts.phase2.types import CandidateProfile


class FakeReviewAgent:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.called = False

    def collect(self, candidates: list[CandidateProfile], attributes: set[str]) -> tuple[list[dict[str, Any]], int]:
        self.called = True
        return self.rows, 17


class Phase2SystemTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.catalog_path = Path(self.temporary_directory.name) / "catalog.jsonl"
        products = [
            {
                "parent_asin": "P1",
                "title": "Lightweight gaming laptop 15.6 inch 16 GB RAM 512 GB SSD",
                "price": 799.0,
                "average_rating": 4.5,
                "features": ["16 GB RAM", "15.6 inch display", "512 GB SSD"],
                "description": ["Battery life up to 9 hours", "Item weight 3.2 pounds"],
                "details": {
                    "RAM Memory Installed Size": "16 GB",
                    "Screen Size": "15.6 Inches",
                    "Item Weight": "3.2 pounds",
                    "Hard Disk Size": "512 GB",
                },
            },
            {
                "parent_asin": "P2",
                "title": "Gaming laptop 17 inch 32 GB RAM 1 TB SSD",
                "price": 1299.0,
                "average_rating": 4.2,
                "features": ["32 GB RAM", "17 inch display", "1 TB SSD"],
                "description": ["Battery life up to 6 hours", "Weight 6 pounds"],
                "details": {
                    "RAM Memory Installed Size": "32 GB",
                    "Screen Size": "17 Inches",
                    "Item Weight": "6 pounds",
                    "Hard Disk Size": "1 TB",
                },
            },
        ]
        self.catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in products),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_query_analyzer_extracts_constraints_and_scores_vague_attribute(self) -> None:
        state = QueryAnalyzer().analyze("Find a good laptop with at least 16 GB RAM under $1000")
        self.assertIn("ram", state.target_attributes)
        self.assertIn("price", state.target_attributes)
        self.assertEqual(
            [(constraint.attribute, constraint.operator, constraint.value) for constraint in state.hard_constraints],
            [("ram", "gte", 16.0), ("price", "lte", 1000.0)],
        )
        self.assertEqual(state.uncertainty["ram"], 0.0)

    def test_profiler_filters_hard_constraints_and_measures_candidate_entropy(self) -> None:
        query = QueryAnalyzer().analyze("laptop at least 16 GB RAM under $1000")
        state = CatalogProfiler(self.catalog_path).retrieve(query)
        self.assertEqual([candidate.parent_asin for candidate in state.candidates], ["P1"])
        self.assertEqual(state.completeness["ram"], 1.0)

    def test_numeric_filters_normalize_battery_weight_and_display_units(self) -> None:
        analyzer = QueryAnalyzer()
        battery = analyzer.analyze("laptop with at least 8 hours of battery life")
        weight = analyzer.analyze("laptop under 4 pounds")
        display = analyzer.analyze("laptop with 15 inch screen")
        self.assertEqual(
            [(item.attribute, item.operator, item.value, item.unit) for item in battery.hard_constraints],
            [("battery", "gte", 8.0, "HOUR")],
        )
        self.assertEqual(
            [(item.attribute, item.operator, item.value, item.unit) for item in weight.hard_constraints],
            [("weight_portability", "lte", 4 * 453.59237, "G")],
        )
        self.assertEqual(display.hard_constraints[0].unit, "IN")
        self.assertEqual(display.hard_constraints[0].value, 15.0)

        candidates = CatalogProfiler(self.catalog_path).retrieve(weight)
        self.assertEqual([item.parent_asin for item in candidates.candidates], ["P1"])

    def test_query_gate_asks_only_when_catalog_has_coverage_and_diversity(self) -> None:
        query = QueryAnalyzer().analyze("good gaming laptop screen")
        state = CatalogProfiler(self.catalog_path).retrieve(query)
        self.assertGreater(state.entropy["display"], 0)
        system = AdaptiveOrchestrator(CatalogProfiler(self.catalog_path))
        self.assertEqual(system.minimum_catalog_entropy, 0.3)
        trace = system.run("good gaming laptop screen")
        self.assertEqual(trace.decision.action, "ASK_CLARIFICATION")
        self.assertIn("ASK_CLARIFICATION", trace.actions_taken)

    def test_ambiguous_query_does_not_probe_when_catalog_has_no_attribute_data(self) -> None:
        sparse_path = Path(self.temporary_directory.name) / "sparse.jsonl"
        sparse_path.write_text(
            json.dumps({"parent_asin": "SPARSE", "title": "Laptop", "price": 500}) + "\n",
            encoding="utf-8",
        )
        trace = AdaptiveOrchestrator(CatalogProfiler(sparse_path)).run("good laptop battery")
        self.assertNotEqual(trace.decision.action, "ASK_CLARIFICATION")
        self.assertIn("ACTIVATE_EVIDENCE_AGENTS", trace.actions_taken)

    def test_clear_complete_query_early_exits_without_agents(self) -> None:
        review_agent = FakeReviewAgent([])
        system = AdaptiveOrchestrator(
            CatalogProfiler(self.catalog_path),
            review_agent=review_agent,
        )
        trace = system.run("laptop at least 16 GB RAM under $1000")
        self.assertEqual(trace.decision.action, "RECOMMEND")
        self.assertEqual(trace.spec_agent_calls, 0)
        self.assertEqual(trace.review_agent_calls, 0)
        self.assertFalse(review_agent.called)
        self.assertTrue(trace.decision.recommendations[0]["citations"])

    def test_missing_metadata_activates_specialized_agents(self) -> None:
        missing_path = Path(self.temporary_directory.name) / "missing-battery.jsonl"
        missing_path.write_text(
            json.dumps({
                "parent_asin": "P1",
                "title": "gaming laptop",
                "price": 799,
                "average_rating": 4.0,
            }) + "\n",
            encoding="utf-8",
        )
        agent = FakeReviewAgent(
            [{
                "aspect_category": "battery",
                "sentiment": "positive",
                "parent_asin": "P1",
                "sentence_text": "The battery lasts a long time.",
                "asin": "V1",
            }]
        )
        system = AdaptiveOrchestrator(
            CatalogProfiler(missing_path),
            review_agent=agent,
        )
        trace = system.run("gaming laptop battery experience")
        self.assertIn("ACTIVATE_EVIDENCE_AGENTS", trace.actions_taken)
        self.assertEqual(trace.spec_agent_calls, 1)
        self.assertEqual(trace.review_agent_calls, 1)
        self.assertTrue(agent.called)

    def test_spec_agent_emits_source_citations(self) -> None:
        query = QueryAnalyzer().analyze("gaming laptop")
        candidate = CatalogProfiler(self.catalog_path).retrieve(query).candidates[0]
        rows = SpecDescriptionAgent().extract(candidate, {"ram"})
        self.assertTrue(rows)
        self.assertTrue(all(row["source"].startswith("metadata.") for row in rows))

    def test_conflict_evaluator_detects_experiential_drift_and_variant_conflict(self) -> None:
        candidate = CandidateProfile(
            {"parent_asin": "P1"},
            {"battery": "9 hours"},
            [],
            1.0,
        )
        rows = [
            {"parent_asin": "P1", "asin": "V1", "aspect_category": "battery", "sentiment": "negative"},
            {"parent_asin": "P1", "asin": "V1", "aspect_category": "battery", "sentiment": "negative"},
            {"parent_asin": "P1", "asin": "V1", "aspect_category": "battery", "sentiment": "negative"},
            {"parent_asin": "P1", "asin": "V1", "aspect_category": "battery", "sentiment": "negative"},
            {"parent_asin": "P1", "asin": "V1", "aspect_category": "battery", "sentiment": "negative"},
            {"parent_asin": "P1", "asin": "V1", "aspect_category": "battery", "sentiment": "negative"},
            {"parent_asin": "P1", "asin": "V1", "aspect_category": "battery", "sentiment": "negative"},
            {"parent_asin": "P1", "asin": "V2", "aspect_category": "battery", "sentiment": "positive"},
            {"parent_asin": "P1", "asin": "V2", "aspect_category": "battery", "sentiment": "positive"},
            {"parent_asin": "P1", "asin": "V2", "aspect_category": "battery", "sentiment": "positive"},
        ]
        conflicts = ConflictEvaluator().evaluate([candidate], rows)
        self.assertEqual(
            {conflict.kind for conflict in conflicts},
            {"EXPERIENTIAL_DRIFT", "VARIANT_CONFLICT"},
        )


if __name__ == "__main__":
    unittest.main()

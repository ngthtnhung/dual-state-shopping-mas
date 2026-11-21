from __future__ import annotations

import unittest

from scripts.phase2.evaluate import (
    CountingReviewAgent,
    EvaluationCase,
    _build_stratified_scenarios,
    _binary_metrics,
    _calibrate,
    _run_evaluation,
    _select_sample,
    _split_calibration_evaluation,
)
from scripts.phase2.query_analyzer import QueryAnalyzer
from scripts.phase2.catalog_profiler import CatalogProfiler
from scripts.phase2.types import CandidateProfile, CatalogState


class Phase2EvaluationTests(unittest.TestCase):
    def test_select_sample_balances_annotated_vague_and_clear_queries(self) -> None:
        records = [
            {"id": 1, "text": "Laptop with 16 GB RAM", "vague_annotations": []},
            {"id": 2, "text": "Find a laptop with 32 GB RAM", "vague_annotations": []},
            {
                "id": 3,
                "text": "A lightweight laptop",
                "vague_annotations": [{"word": "lightweight", "start index": 2}],
            },
            {
                "id": 4,
                "text": "Laptop with good battery life",
                "vague_annotations": [{"word": "good", "start index": 12}],
            },
        ]
        sample = _select_sample(records, 4, 17)
        self.assertEqual(len(sample), 4)
        self.assertEqual(sum(bool(row[2]) for row in sample), 2)

    def test_calibration_reports_false_probe_rate_and_selects_grid_point(self) -> None:
        analyzer = QueryAnalyzer()
        cases = []
        for index, (text, ambiguous_attributes, uncertainty, completeness, entropy) in enumerate(
            [
                ("good screen laptop", ["display"], 0.65, 0.8, 0.6),
                ("15 inch screen laptop", [], 0.0, 0.9, 0.6),
                ("good screen laptop", ["display"], 0.65, 0.0, 0.0),
                ("screen laptop", [], 0.35, 0.8, 0.6),
            ]
        ):
            state = analyzer.analyze(text)
            for attribute in state.target_attributes:
                state.uncertainty[attribute] = uncertainty
            cases.append(
                EvaluationCase(
                    index,
                    text,
                    ambiguous_attributes,
                    state,
                    CatalogState([], 0, {"display": completeness}, {"display": entropy}),
                    0.0,
                )
            )

        chosen, scores = _calibrate(cases, (0.3, 0.5, 0.7), (0.25, 0.75), (0.1, 0.7))
        self.assertEqual(len(scores), 12)
        self.assertEqual(chosen["ambiguity_threshold"], 0.5)
        best = next(
            row for row in scores
            if row["uncertainty_threshold"] == chosen["ambiguity_threshold"]
            and row["coverage_threshold"] == chosen["minimum_clarification_coverage"]
            and row["entropy_threshold"] == chosen["minimum_catalog_entropy"]
        )
        self.assertEqual(best["probe_waste_rate"], 0.0)
        self.assertEqual(best["f1"], 1.0)

    def test_binary_metrics_return_zero_when_no_clarification_is_predicted(self) -> None:
        metrics = _binary_metrics([True, False], [False, False])
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["probe_waste_rate"], 0.0)

    def test_calibration_and_heldout_sets_are_disjoint_and_balanced(self) -> None:
        analyzer = QueryAnalyzer()
        cases = []
        for index, (text, ambiguous) in enumerate(
            [
                ("16 GB RAM laptop", False),
                ("32 GB RAM laptop", False),
                ("good screen laptop", True),
                ("great battery laptop", True),
            ]
        ):
            state = analyzer.analyze(text)
            cases.append(
                EvaluationCase(
                    index,
                    text,
                    ["display"] if ambiguous and "display" in state.target_attributes else
                    ["battery"] if ambiguous else [],
                    state,
                    CatalogState([], 0, {}, {}),
                    0.0,
                )
            )
        calibration, evaluation = _split_calibration_evaluation(cases, 42)
        self.assertEqual(len(calibration), 2)
        self.assertEqual(len(evaluation), 2)
        self.assertEqual(
            {case.record_id for case in calibration}
            & {case.record_id for case in evaluation},
            set(),
        )
        self.assertEqual(sum(case.gold_ambiguous for case in calibration), 1)
        self.assertEqual(sum(case.gold_ambiguous for case in evaluation), 1)

    def test_four_strata_routes_early_asks_escalates_and_abstains(self) -> None:
        analyzer = QueryAnalyzer()
        clear_state = analyzer.analyze("laptop with 16 GB RAM")
        vague_state = analyzer.analyze("good screen laptop")
        product = {
            "parent_asin": "P1",
            "title": "Laptop 16 GB RAM 15 inch screen",
            "features": ["16 GB RAM and 15 inch screen"],
            "details": {
                "RAM Memory Installed Size": "16 GB",
                "Screen Size": "15 Inches",
            },
            "price": 500,
            "average_rating": 4.5,
        }
        candidate = CandidateProfile(
            product,
            {
                "ram": "16 GB",
                "display": "15 Inches",
                "price": "500 USD",
            },
            [
                {"kind": "specification", "attribute": "ram", "source": "metadata.details.RAM"},
                {"kind": "specification", "attribute": "display", "source": "metadata.details.Screen Size"},
            ],
            1.0,
        )
        clear_case = EvaluationCase(
            1,
            clear_state.text,
            [],
            clear_state,
            CatalogState([candidate], 1, {"ram": 1.0}, {"ram": 0.0}),
            0.0,
        )
        vague_case = EvaluationCase(
            2,
            vague_state.text,
            ["display"],
            vague_state,
            CatalogState([candidate], 1, {"display": 1.0}, {"display": 0.8}),
            0.0,
        )
        thresholds = {
            "ambiguity_threshold": 0.5,
            "minimum_clarification_coverage": 0.5,
            "minimum_catalog_entropy": 0.2,
        }
        scenarios = _build_stratified_scenarios(
            [clear_case, vague_case], thresholds
        )
        profiler = CatalogProfiler("unused-catalog.jsonl")
        review_agent = CountingReviewAgent()
        rows, metrics = _run_evaluation(
            scenarios, thresholds, profiler=profiler, review_agent=review_agent
        )
        profiler.close()

        by_scenario = {row["scenario"]: row for row in rows}
        self.assertEqual(by_scenario["S1"]["action"], "RECOMMEND")
        self.assertEqual(by_scenario["S1"]["review_agent_calls"], 0)
        self.assertEqual(by_scenario["S2"]["action"], "ASK_CLARIFICATION")
        self.assertEqual(by_scenario["S3"]["review_agent_calls"], 1)
        self.assertEqual(by_scenario["S4"]["action"], "ABSTAIN_DISCLOSE")
        self.assertEqual(by_scenario["S4"]["review_agent_calls"], 1)
        self.assertEqual(metrics["overall_scenario_behavior_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()

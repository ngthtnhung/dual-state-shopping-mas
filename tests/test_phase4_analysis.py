from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.phase4.analyze_annotations import analyze_annotations


FIELDS = [
    "System",
    "Query ID",
    "Stratum",
    "System action",
    "Detected conflicts",
    "Expected-route proxy (0/1)",
    "Prompt tokens",
    "Output tokens",
    "Latency seconds",
    "CSR (0/1)",
    "Recommendation coverage (0/1)",
    "UCR: any unsupported claim (0/1)",
    "Factual claims total",
    "Evidence-supported factual claims",
    "Unsupported/contradicted factual claims",
    "EGP",
    "Probe waste (0/1)",
    "Disambiguation success (0/1)",
    "Abstain quality (0/1)",
    "Conflict disclosure quality (0/1)",
]


def _row(system: str, action: str, **overrides: str) -> dict[str, str]:
    row = {
        "System": system,
        "Query ID": "q1",
        "Stratum": "S4",
        "System action": action,
        "Detected conflicts": "[]",
        "Expected-route proxy (0/1)": "1",
        "Prompt tokens": "10",
        "Output tokens": "5",
        "Latency seconds": "1.5",
        "CSR (0/1)": "",
        "Recommendation coverage (0/1)": "0",
        "UCR: any unsupported claim (0/1)": "0",
        "Factual claims total": "2",
        "Evidence-supported factual claims": "2",
        "Unsupported/contradicted factual claims": "0",
        "EGP": "1.0",
        "Probe waste (0/1)": "",
        "Disambiguation success (0/1)": "",
        "Abstain quality (0/1)": "1",
        "Conflict disclosure quality (0/1)": "",
    }
    row.update(overrides)
    return row


class Phase4AnalysisTests(unittest.TestCase):
    def test_summarizes_pwr_only_over_formal_clarification_actions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "annotations.csv"
            rows = [
                _row(
                    "B4_query_adaptive_mas",
                    "ASK_CLARIFICATION",
                    **{"Probe waste (0/1)": "1", "Disambiguation success (0/1)": "0"},
                ),
                _row(
                    "B4_query_adaptive_mas",
                    "RECOMMEND",
                    **{"Probe waste (0/1)": "0", "Disambiguation success (0/1)": "0"},
                ),
            ]
            with path.open("w", encoding="utf-8", newline="") as destination:
                writer = csv.DictWriter(destination, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)

            result = analyze_annotations(path)
            group = next(
                row
                for row in result["by_system_and_stratum"]
                if row["system"] == "B4_query_adaptive_mas" and row["stratum"] == "S4"
            )
            self.assertEqual(group["probe_waste_formal_ask_n"], 1)
            self.assertEqual(group["probe_waste_formal_ask_mean"], 1.0)
            self.assertEqual(group["factual_claims_total"], 4)
            self.assertEqual(group["egp_micro"], 1.0)
            self.assertEqual(
                result["annotation_audit"]["review_flags"][
                    "probe_labels_outside_formal_ask_action"
                ]["count"],
                1,
            )

    def test_paired_mcnemar_and_holm_adjustment_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "annotations.csv"
            rows = [
                _row(
                    "B4_query_adaptive_mas",
                    "ASK_CLARIFICATION",
                    **{"Query ID": "q1", "UCR: any unsupported claim (0/1)": "1"},
                ),
                _row(
                    "Ours_dual_conditioned_mas",
                    "ABSTAIN_DISCLOSE",
                    **{"Query ID": "q1", "UCR: any unsupported claim (0/1)": "0"},
                ),
            ]
            with path.open("w", encoding="utf-8", newline="") as destination:
                writer = csv.DictWriter(destination, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)

            result = analyze_annotations(path)
            test = next(
                item
                for item in result["paired_tests"]
                if item["baseline"] == "B4_query_adaptive_mas"
                and item["stratum"] == "S4"
                and item["metric"] == "ucr"
            )
            self.assertEqual(test["paired_n"], 1)
            self.assertEqual(test["exact_mcnemar_p"], 1.0)
            self.assertGreaterEqual(test["holm_adjusted_p"], test["exact_mcnemar_p"])


if __name__ == "__main__":
    unittest.main()

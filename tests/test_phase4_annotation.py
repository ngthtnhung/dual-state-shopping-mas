from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from openpyxl import load_workbook

from scripts.phase4.prepare_annotation_workbook import (
    ANNOTATION_HEADERS,
    create_annotation_workbook,
)


def _sample_report() -> dict[str, object]:
    case = {
        "query_id": 1,
        "scenario": "S4",
        "query": "Find a light laptop.",
        "target_attributes": ["weight"],
        "hard_constraints": [],
        "uncertainty": {"weight": 0.7},
        "completeness": {"weight": 0.0},
        "entropy": {"weight": 0.0},
        "candidate_count": 1,
        "candidate_parent_asins": ["P1"],
        "synthetic_catalog_mask": True,
    }
    return {
        "model": "local-test",
        "seed": 42,
        "case_count": 1,
        "baselines": ["Ours_dual_conditioned_mas"],
        "frozen_cases": [case],
        "frozen_data": {"case_manifest_sha256": "test-hash"},
        "results": [
            {
                "baseline": "Ours_dual_conditioned_mas",
                "query_id": 1,
                "scenario": "S4",
                "query": "Find a light laptop.",
                "decision": {
                    "action": "ABSTAIN_DISCLOSE",
                    "rationale": ["Insufficient evidence."],
                    "recommendations": [],
                    "conflicts": [],
                    "limitations": ["Weight is not verifiable."],
                },
                "system_action": "ABSTAIN_DISCLOSE",
                "response": "I cannot verify laptop weight from the available data.",
                "followup_response": None,
                "expected_behavior_met": True,
                "agent_calls": 1,
                "spec_agent_calls": 1,
                "review_agent_calls": 0,
                "tool_calls": 1,
                "generation_calls": 1,
                "prompt_tokens": 10,
                "output_tokens": 8,
                "latency_seconds": 1.25,
            }
        ],
        "metrics": {
            "Ours_dual_conditioned_mas": {
                "scenario_metrics": {
                    "S4": {
                        "count": 1,
                        "expected_behavior_rate": 1.0,
                        "clarification_rate": 0.0,
                        "abstain_disclose_rate": 1.0,
                    },
                    "S1": {
                        "count": 0,
                        "expected_behavior_rate": None,
                        "clarification_rate": None,
                        "abstain_disclose_rate": None,
                    },
                    "S2": {
                        "count": 0,
                        "expected_behavior_rate": None,
                        "clarification_rate": None,
                        "abstain_disclose_rate": None,
                    },
                    "S3": {
                        "count": 0,
                        "expected_behavior_rate": None,
                        "clarification_rate": None,
                        "abstain_disclose_rate": None,
                    },
                },
                "total_agent_calls": 1,
                "prompt_tokens": 10,
                "output_tokens": 8,
                "mean_latency_seconds": 1.25,
            }
        },
    }


class Phase4AnnotationTests(unittest.TestCase):
    def test_exports_response_context_rubric_and_ablation_notes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "benchmark.json"
            output_path = root / "nested" / "annotation.xlsx"
            input_path.write_text(
                json.dumps(_sample_report()), encoding="utf-8"
            )

            row_count = create_annotation_workbook(input_path, output_path)

            self.assertEqual(row_count, 1)
            workbook = load_workbook(output_path, data_only=False)
            self.assertEqual(
                workbook.sheetnames,
                [
                    "Annotation",
                    "Metric Summary",
                    "Ablation Comparison",
                    "Annotation Rubric",
                ],
            )
            annotation = workbook["Annotation"]
            self.assertEqual(annotation.max_row, 2)
            self.assertEqual(
                tuple(cell.value for cell in annotation[1]),
                ANNOTATION_HEADERS,
            )
            row = {
                cell.value: annotation.cell(2, cell.column).value
                for cell in annotation[1]
            }
            self.assertEqual(row["Catalog completeness"], '{"weight": 0.0}')
            self.assertEqual(row["System action"], "ABSTAIN_DISCLOSE")
            self.assertEqual(row["Initial response"], "I cannot verify laptop weight from the available data.")
            self.assertIsNone(row["CSR (0/1)"])
            self.assertEqual(row["EGP"], '=IFERROR(IF(AH2=0,"",AI2/AH2),"")')
            self.assertGreater(len(annotation.data_validations.dataValidation), 0)
            self.assertEqual(workbook["Ablation Comparison"].max_row, 6)
            workbook.close()

    def test_rejects_missing_case_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "benchmark.json"
            report = _sample_report()
            report["frozen_cases"] = []
            report["case_count"] = 0
            input_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "No frozen case"):
                create_annotation_workbook(input_path, root / "annotation.xlsx")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.worksheet.table import Table, TableStyleInfo
from openpyxl.utils import get_column_letter


ANNOTATION_COLUMNS = (
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
    "Annotation notes",
)

BINARY_ANNOTATION_COLUMNS = (
    "CSR (0/1)",
    "Recommendation coverage (0/1)",
    "UCR: any unsupported claim (0/1)",
    "Probe waste (0/1)",
    "Disambiguation success (0/1)",
    "Abstain quality (0/1)",
    "Conflict disclosure quality (0/1)",
)

ANNOTATION_HEADERS = (
    "System",
    "Query ID",
    "Stratum",
    "Query",
    "Target attributes",
    "Hard constraints",
    "Query uncertainty",
    "Catalog completeness",
    "Catalog entropy",
    "Candidate count",
    "Candidate parent ASINs",
    "Synthetic catalog mask",
    "System action",
    "Decision rationale",
    "Decision recommendations",
    "Detected conflicts",
    "Decision limitations",
    "Evidence available to annotator",
    "Simulated user reply",
    "Initial response",
    "Follow-up response",
    "Expected-route proxy (0/1)",
    "Agent calls",
    "Spec agent calls",
    "Review agent calls",
    "Tool calls",
    "Generation calls",
    "Prompt tokens",
    "Output tokens",
    "Latency seconds",
    *ANNOTATION_COLUMNS,
)

ANNOTATION_RUBRIC = (
    (
        "CSR (0/1)",
        "1 if every explicit hard constraint in the query is satisfied by the "
        "recommended product(s); 0 if any is violated or a recommendation is "
        "made despite a known violation. Leave blank when the query has no "
        "explicit hard constraint.",
    ),
    (
        "Recommendation coverage (0/1)",
        "1 if the response offers at least one identifiable product "
        "recommendation; otherwise 0. This is availability, not correctness.",
    ),
    (
        "UCR: any unsupported claim (0/1)",
        "1 if the initial or follow-up answer contains at least one material "
        "product fact that is unsupported or contradicted by the supplied "
        "evidence; otherwise 0.",
    ),
    (
        "Factual claim counts / EGP",
        "Count factual product claims across the initial and follow-up "
        "responses. Count evidence-supported claims separately. EGP is "
        "supported claims divided by total factual claims; the sheet computes "
        "this ratio. Do not count opinions or recommendations as product facts "
        "unless they assert a factual property.",
    ),
    (
        "Probe waste (0/1)",
        "1 if a clarification asks about a preference that the available "
        "catalog/evidence cannot act on or verify; otherwise 0. Leave blank "
        "when no clarification is asked.",
    ),
    (
        "Disambiguation success (0/1)",
        "1 if the clarification and user reply resolve the relevant ambiguity "
        "and the follow-up uses that information; otherwise 0. Leave blank "
        "when no clarification is asked.",
    ),
    (
        "Abstain quality (0/1)",
        "1 if the system appropriately discloses uncertainty/abstains when "
        "evidence is insufficient or conflicting, and does not abstain when a "
        "grounded answer is supported; otherwise 0.",
    ),
    (
        "Conflict disclosure quality (0/1)",
        "1 if a material source conflict is accurately disclosed without "
        "presenting review sentiment as verified specification; otherwise 0. "
        "Leave blank if no material conflict is present in the supplied "
        "evidence.",
    ),
    (
        "Single-annotator reliability",
        "This workbook supports expert annotation, but one annotator alone "
        "does not establish inter-annotator reliability or an error-free gold "
        "standard. Record the rubric/version and consider a second annotator "
        "or a blinded re-annotation subset.",
    ),
)


def _json_cell(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _citation_evidence(decision: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not decision:
        return []
    evidence: list[dict[str, Any]] = []
    for recommendation in decision.get("recommendations", []):
        evidence.extend(recommendation.get("citations", []))
    for conflict in decision.get("conflicts", []):
        evidence.extend(conflict.get("evidence", []))
    unique: dict[str, dict[str, Any]] = {}
    for item in evidence:
        unique[_json_cell(item)] = item
    return list(unique.values())


def _validate_report(report: dict[str, Any]) -> None:
    results = report.get("results")
    cases = report.get("frozen_cases")
    if not isinstance(results, list) or not isinstance(cases, list):
        raise ValueError("Benchmark report must contain results and frozen_cases lists")
    if len(cases) != report.get("case_count"):
        raise ValueError("frozen_cases count does not match case_count")
    seen_results: set[tuple[str, str, str]] = set()
    for row in results:
        key = (str(row["baseline"]), str(row["query_id"]), str(row["scenario"]))
        if key in seen_results:
            raise ValueError(f"Duplicate result row for {key}")
        seen_results.add(key)
    if len(seen_results) != len(results):
        raise ValueError("Benchmark result rows are not unique")


def _annotation_rows(report: dict[str, Any]) -> list[list[Any]]:
    case_by_key = {
        (str(case["query_id"]), str(case["scenario"])): case
        for case in report["frozen_cases"]
    }
    rows: list[list[Any]] = []
    for result in report["results"]:
        case_key = (str(result["query_id"]), str(result["scenario"]))
        case = case_by_key.get(case_key)
        if case is None:
            raise ValueError(f"No frozen case found for result {case_key}")
        decision = result.get("decision")
        evidence = _citation_evidence(decision)
        row = [
            result["baseline"],
            result["query_id"],
            result["scenario"],
            result["query"],
            _json_cell(case.get("target_attributes")),
            _json_cell(case.get("hard_constraints")),
            _json_cell(case.get("uncertainty")),
            _json_cell(case.get("completeness")),
            _json_cell(case.get("entropy")),
            case.get("candidate_count"),
            _json_cell(case.get("candidate_parent_asins")),
            int(bool(case.get("synthetic_catalog_mask"))),
            result["system_action"],
            _json_cell(decision.get("rationale") if decision else None),
            _json_cell(decision.get("recommendations") if decision else None),
            _json_cell(decision.get("conflicts") if decision else None),
            _json_cell(decision.get("limitations") if decision else None),
            _json_cell(evidence),
            _json_cell(result.get("user_reply")),
            result.get("response", ""),
            result.get("followup_response") or "",
            (
                int(bool(result["expected_behavior_met"]))
                if result.get("expected_behavior_met") is not None
                else None
            ),
            result.get("agent_calls", 0),
            result.get("spec_agent_calls", 0),
            result.get("review_agent_calls", 0),
            result.get("tool_calls", 0),
            result.get("generation_calls", 0),
            result.get("prompt_tokens", 0),
            result.get("output_tokens", 0),
            result.get("latency_seconds", 0),
        ]
        row.extend([None] * len(ANNOTATION_COLUMNS))
        rows.append(row)
    return rows


def _style_sheet(sheet: Any) -> None:
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    sheet.row_dimensions[1].height = 42
    for cell in sheet[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    for row in sheet.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
    for index, cell in enumerate(sheet[1], start=1):
        header = str(cell.value)
        if header in {"Query", "Initial response", "Follow-up response"}:
            width = 64
        elif header in {
            "Decision recommendations",
            "Evidence available to annotator",
            "Candidate parent ASINs",
            "Hard constraints",
        }:
            width = 48
        elif header in ANNOTATION_COLUMNS:
            width = 24
        else:
            width = min(max(len(header) + 2, 14), 28)
        sheet.column_dimensions[get_column_letter(index)].width = width


def _add_annotation_sheet(workbook: Workbook, rows: list[list[Any]]) -> None:
    sheet = workbook.active
    sheet.title = "Annotation"
    sheet.append(ANNOTATION_HEADERS)
    for row in rows:
        sheet.append(row)
    header_index = {name: index + 1 for index, name in enumerate(ANNOTATION_HEADERS)}
    for row_number in range(2, sheet.max_row + 1):
        total_col = get_column_letter(header_index["Factual claims total"])
        supported_col = get_column_letter(
            header_index["Evidence-supported factual claims"]
        )
        egp_col = header_index["EGP"]
        sheet.cell(
            row_number,
            egp_col,
            f'=IFERROR(IF({total_col}{row_number}=0,"",'
            f'{supported_col}{row_number}/{total_col}{row_number}),"")',
        )
    for name in BINARY_ANNOTATION_COLUMNS:
        column = get_column_letter(header_index[name])
        validation = DataValidation(type="list", formula1='"0,1"', allow_blank=True)
        validation.error = "Enter 0, 1, or leave blank when not applicable."
        validation.errorTitle = "Binary annotation required"
        validation.showErrorMessage = True
        sheet.add_data_validation(validation)
        validation.add(f"{column}2:{column}{sheet.max_row}")
    for name in (
        "Factual claims total",
        "Evidence-supported factual claims",
        "Unsupported/contradicted factual claims",
    ):
        column = get_column_letter(header_index[name])
        validation = DataValidation(
            type="whole", operator="greaterThanOrEqual", formula1="0", allow_blank=True
        )
        validation.error = "Enter a non-negative whole number."
        validation.showErrorMessage = True
        sheet.add_data_validation(validation)
        validation.add(f"{column}2:{column}{sheet.max_row}")
    if rows:
        table = Table(displayName="Phase4Annotations", ref=sheet.dimensions)
        table.tableStyleInfo = TableStyleInfo(
            name="TableStyleMedium2",
            showFirstColumn=False,
            showLastColumn=False,
            showRowStripes=True,
            showColumnStripes=False,
        )
        sheet.add_table(table)
    _style_sheet(sheet)


def _add_metric_summary(workbook: Workbook, report: dict[str, Any]) -> None:
    sheet = workbook.create_sheet("Metric Summary")
    sheet.append(
        [
            "System",
            "Stratum",
            "Rows",
            "Annotated rows",
            "CSR mean",
            "Recommendation coverage",
            "UCR any-claim rate",
            "EGP (supported / total claims)",
            "Probe-waste rate",
            "Disambiguation success",
            "Abstain quality",
            "Conflict disclosure quality",
        ]
    )
    annotation = workbook["Annotation"]
    headers = {
        str(cell.value): cell.column for cell in annotation[1]
    }
    system_col = get_column_letter(headers["System"])
    stratum_col = get_column_letter(headers["Stratum"])
    labels = [
        "CSR (0/1)",
        "Recommendation coverage (0/1)",
        "UCR: any unsupported claim (0/1)",
        "Probe waste (0/1)",
        "Disambiguation success (0/1)",
        "Abstain quality (0/1)",
        "Conflict disclosure quality (0/1)",
    ]
    last_row = annotation.max_row
    for system in report["baselines"]:
        for stratum in ("S1", "S2", "S3", "S4"):
            sheet.append([system, stratum, f'=COUNTIFS(Annotation!${system_col}$2:${system_col}${last_row},A{sheet.max_row + 1},Annotation!${stratum_col}$2:${stratum_col}${last_row},B{sheet.max_row + 1})'])
            current_row = sheet.max_row
            sheet.cell(
                current_row,
                4,
                f'=COUNTIFS(Annotation!${system_col}$2:${system_col}${last_row},A{current_row},'
                f'Annotation!${stratum_col}$2:${stratum_col}${last_row},B{current_row},'
                f'Annotation!${get_column_letter(headers[labels[2]])}$2:${get_column_letter(headers[labels[2]])}${last_row},">=0")',
            )
            for output_column, label in (
                (5, labels[0]),
                (6, labels[1]),
                (7, labels[2]),
                (9, labels[3]),
                (10, labels[4]),
                (11, labels[5]),
                (12, labels[6]),
            ):
                label_col = get_column_letter(headers[label])
                sheet.cell(
                    current_row,
                    output_column,
                    f'=IFERROR(AVERAGEIFS(Annotation!${label_col}$2:${label_col}${last_row},'
                    f'Annotation!${system_col}$2:${system_col}${last_row},A{current_row},'
                    f'Annotation!${stratum_col}$2:${stratum_col}${last_row},B{current_row}),"")',
                )
            total_col = get_column_letter(headers["Factual claims total"])
            supported_col = get_column_letter(
                headers["Evidence-supported factual claims"]
            )
            sheet.cell(
                current_row,
                8,
                f'=IFERROR(SUMIFS(Annotation!${supported_col}$2:${supported_col}${last_row},'
                f'Annotation!${system_col}$2:${system_col}${last_row},A{current_row},'
                f'Annotation!${stratum_col}$2:${stratum_col}${last_row},B{current_row})/'
                f'SUMIFS(Annotation!${total_col}$2:${total_col}${last_row},'
                f'Annotation!${system_col}$2:${system_col}${last_row},A{current_row},'
                f'Annotation!${stratum_col}$2:${stratum_col}${last_row},B{current_row}),"")',
            )
    note_row = sheet.max_row + 2
    sheet.cell(
        note_row,
        1,
        "Formulas recalculate in Excel/LibreOffice after annotation; blank labels are excluded.",
    )
    _style_sheet(sheet)


def _add_ablation_comparison(workbook: Workbook, report: dict[str, Any]) -> None:
    sheet = workbook.create_sheet("Ablation Comparison")
    sheet.append(
        [
            "Proposed ablation",
            "Existing comparison",
            "Cohort",
            "Expected-route rate",
            "Clarification rate",
            "Abstain/disclose rate",
            "Mean agent calls / case",
            "Mean prompt + output tokens / case",
            "Mean latency seconds",
            "Interpretation / limitation",
        ]
    )
    metrics = report["metrics"]
    comparisons = (
        (
            "Ablation 1: disable Catalog Profiler",
            "B4 query-only proxy",
            "S4",
            "B4_query_adaptive_mas",
            "Comparable policy baseline; it is not an exact switch-off run.",
        ),
        (
            "Full dual-conditioned system",
            "Ours",
            "S4",
            "Ours_dual_conditioned_mas",
            "Reference for Ablation 1; route score is policy compliance, not human-rated quality.",
        ),
        (
            "Ablation 2: disable Review Agent and Conflict Evaluator",
            "Not available in current run",
            "S3",
            None,
            "S3 only masks metadata; no curated manufacturer-versus-review contradiction labels exist. Do not infer this ablation from current results.",
        ),
        (
            "Ablation 3: disable early exit / fixed pipeline",
            "B3 fixed-sequential proxy",
            "S1",
            "B3_fixed_sequential_mas",
            "Comparable fixed-path baseline; not an exact switch-off run of Ours.",
        ),
        (
            "Full dual-conditioned system",
            "Ours",
            "S1",
            "Ours_dual_conditioned_mas",
            "Reference for Ablation 3. Compare calls, tokens, and latency; latency includes local LLM generation.",
        ),
    )
    for ablation, system_label, stratum, baseline, note in comparisons:
        if baseline is None or baseline not in metrics:
            values: list[Any] = ["", "", "", "", "", ""]
        else:
            system = metrics[baseline]
            scenario = system["scenario_metrics"][stratum]
            cohort = [
                row
                for row in report["results"]
                if row["baseline"] == baseline and row["scenario"] == stratum
            ]
            count = len(cohort)
            values = [
                scenario["expected_behavior_rate"],
                scenario["clarification_rate"],
                scenario["abstain_disclose_rate"],
                (
                    round(sum(row["agent_calls"] for row in cohort) / count, 4)
                    if count
                    else ""
                ),
                (
                    round(
                        sum(
                            row["prompt_tokens"] + row["output_tokens"]
                            for row in cohort
                        )
                        / count,
                        2,
                    )
                    if count
                    else ""
                ),
                (
                    round(
                        sum(row["latency_seconds"] for row in cohort) / count,
                        4,
                    )
                    if count
                    else ""
                ),
            ]
        sheet.append([ablation, system_label, stratum, *values, note])
    _style_sheet(sheet)


def _add_rubric(workbook: Workbook, report: dict[str, Any]) -> None:
    sheet = workbook.create_sheet("Annotation Rubric")
    sheet.append(["Field", "Guidance"])
    for row in ANNOTATION_RUBRIC:
        sheet.append(list(row))
    sheet.append(
        [
            "Run provenance",
            (
                f"Model={report.get('model')}; seed={report.get('seed')}; "
                f"cases={report.get('case_count')}; rows={len(report.get('results', []))}; "
                f"case_manifest_sha256={report.get('frozen_data', {}).get('case_manifest_sha256')}"
            ),
        ]
    )
    sheet.append(
        [
            "Important",
            "S3/S4 are controlled metadata-masking counterfactuals, not naturally observed missing/conflicting catalog records.",
        ]
    )
    sheet.column_dimensions["A"].width = 38
    sheet.column_dimensions["B"].width = 120
    for row in sheet.iter_rows():
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
    sheet.row_dimensions[1].height = 28
    for cell in sheet[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")
    sheet.freeze_panes = "A2"


def create_annotation_workbook(
    input_path: Path, output_path: Path
) -> int:
    with input_path.open("r", encoding="utf-8") as source:
        report = json.load(source)
    _validate_report(report)
    rows = _annotation_rows(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    workbook.calculation.fullCalcOnLoad = True
    workbook.calculation.forceFullCalc = True
    _add_annotation_sheet(workbook, rows)
    _add_metric_summary(workbook, report)
    _add_ablation_comparison(workbook, report)
    _add_rubric(workbook, report)
    workbook.save(output_path)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Phase 3 responses and frozen metadata for Phase 4 annotation."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("reports/phase3/phase3_baseline_results.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/phase4/phase4_annotation.xlsx"),
    )
    args = parser.parse_args()
    row_count = create_annotation_workbook(args.input, args.output)
    print(f"Annotation rows: {row_count}")
    print(f"Workbook: {args.output}")


if __name__ == "__main__":
    main()

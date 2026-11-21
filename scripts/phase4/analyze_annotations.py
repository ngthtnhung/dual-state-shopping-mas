from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


SYSTEMS = (
    "B1_single_llm",
    "B2_single_agent_retrieval",
    "B3_fixed_sequential_mas",
    "B4_query_adaptive_mas",
    "Ours_dual_conditioned_mas",
)
STRATA = ("S1", "S2", "S3", "S4")
METRIC_COLUMNS = {
    "csr": "CSR (0/1)",
    "recommendation_coverage": "Recommendation coverage (0/1)",
    "ucr": "UCR: any unsupported claim (0/1)",
    "probe_waste": "Probe waste (0/1)",
    "disambiguation_success": "Disambiguation success (0/1)",
    "abstain_quality": "Abstain quality (0/1)",
    "conflict_disclosure_quality": "Conflict disclosure quality (0/1)",
}
COUNT_COLUMNS = (
    "Factual claims total",
    "Evidence-supported factual claims",
    "Unsupported/contradicted factual claims",
)
BINARY_COLUMNS = tuple(METRIC_COLUMNS.values())


def _number(row: dict[str, str], column: str) -> float | None:
    value = row.get(column, "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"Invalid numeric value in {column}: {value!r}") from error


def _is_present(value: str | None) -> bool:
    return bool(value and value.strip() not in ("", "[]", "{}", "null", "None"))


def _json_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _exact_mcnemar_pvalue(ours: list[int], baseline: list[int]) -> float:
    discordant_ours = sum(a == 1 and b == 0 for a, b in zip(ours, baseline, strict=True))
    discordant_baseline = sum(a == 0 and b == 1 for a, b in zip(ours, baseline, strict=True))
    discordant = discordant_ours + discordant_baseline
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(discordant_ours, discordant_baseline) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _holm_adjust(pvalues: list[float]) -> list[float]:
    if not pvalues:
        return []
    ordered = sorted(enumerate(pvalues), key=lambda item: item[1])
    adjusted = [1.0] * len(pvalues)
    running_max = 0.0
    count = len(pvalues)
    for rank, (index, pvalue) in enumerate(ordered):
        running_max = max(running_max, (count - rank) * pvalue)
        adjusted[index] = min(1.0, running_max)
    return adjusted


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _summarize_group(
    rows: list[dict[str, str]], system: str, stratum: str
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "system": system,
        "stratum": stratum,
        "row_count": len(rows),
        "unique_query_ids": len({row["Query ID"] for row in rows}),
    }
    for metric, column in METRIC_COLUMNS.items():
        values = [value for row in rows if (value := _number(row, column)) is not None]
        summary[f"{metric}_n"] = len(values)
        summary[f"{metric}_mean"] = _mean(values)

    clarification_rows = [
        row for row in rows if row.get("System action") == "ASK_CLARIFICATION"
    ]
    summary["formal_clarification_count"] = len(clarification_rows)
    for metric in ("probe_waste", "disambiguation_success"):
        column = METRIC_COLUMNS[metric]
        annotated_values = [
            value for row in rows if (value := _number(row, column)) is not None
        ]
        summary[f"{metric}_annotated_n"] = len(annotated_values)
        summary[f"{metric}_annotated_mean"] = _mean(annotated_values)
        values = [
            value
            for row in clarification_rows
            if (value := _number(row, column)) is not None
        ]
        summary[f"{metric}_formal_ask_n"] = len(values)
        summary[f"{metric}_formal_ask_mean"] = _mean(values)

    total_claims = sum(_number(row, COUNT_COLUMNS[0]) or 0 for row in rows)
    supported_claims = sum(_number(row, COUNT_COLUMNS[1]) or 0 for row in rows)
    unsupported_claims = sum(_number(row, COUNT_COLUMNS[2]) or 0 for row in rows)
    summary["factual_claims_total"] = total_claims
    summary["evidence_supported_claims"] = supported_claims
    summary["unsupported_claims"] = unsupported_claims
    summary["egp_micro"] = supported_claims / total_claims if total_claims else None
    row_egp = [
        value for row in rows if (value := _number(row, "EGP")) is not None
    ]
    summary["egp_macro_n"] = len(row_egp)
    summary["egp_macro"] = _mean(row_egp)

    token_counts = [
        (_number(row, "Prompt tokens") or 0) + (_number(row, "Output tokens") or 0)
        for row in rows
    ]
    latencies = [_number(row, "Latency seconds") for row in rows]
    summary["mean_tokens_per_case"] = _mean(token_counts)
    summary["mean_latency_seconds"] = _mean(
        [value for value in latencies if value is not None]
    )

    labeled_conflicts = [
        row
        for row in rows
        if _number(row, METRIC_COLUMNS["conflict_disclosure_quality"]) is not None
    ]
    summary["conflict_label_n"] = len(labeled_conflicts)
    summary["rows_with_structured_detected_conflict"] = sum(
        _is_present(row.get("Detected conflicts")) for row in rows
    )
    summary["conflict_labels_without_structured_conflict"] = sum(
        not _is_present(row.get("Detected conflicts")) for row in labeled_conflicts
    )
    return summary


def _audit_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    violations: dict[str, list[dict[str, Any]]] = {
        "invalid_binary_values": [],
        "negative_claim_counts": [],
        "claim_count_mismatch": [],
        "ucr_claim_count_mismatch": [],
        "egp_mismatch": [],
    }
    review_flags: dict[str, list[dict[str, Any]]] = {
        "probe_labels_outside_formal_ask_action": [],
        "conflict_labels_without_structured_conflict": [],
    }
    for line_number, row in enumerate(rows, start=2):
        identity = {
            "line": line_number,
            "system": row.get("System"),
            "query_id": row.get("Query ID"),
            "stratum": row.get("Stratum"),
        }
        for column in BINARY_COLUMNS:
            value = row.get(column, "").strip()
            if value and value not in ("0", "1"):
                violations["invalid_binary_values"].append(
                    {**identity, "column": column, "value": value}
                )

        total = _number(row, COUNT_COLUMNS[0])
        supported = _number(row, COUNT_COLUMNS[1])
        unsupported = _number(row, COUNT_COLUMNS[2])
        if any(value is not None and value < 0 for value in (total, supported, unsupported)):
            violations["negative_claim_counts"].append(identity)
        if None not in (total, supported, unsupported) and total != supported + unsupported:
            violations["claim_count_mismatch"].append(
                {**identity, "total": total, "supported": supported, "unsupported": unsupported}
            )

        ucr = _number(row, METRIC_COLUMNS["ucr"])
        if unsupported is not None and ucr is not None and ucr != int(unsupported > 0):
            violations["ucr_claim_count_mismatch"].append(
                {**identity, "ucr": ucr, "unsupported": unsupported}
            )

        egp = _number(row, "EGP")
        if total and supported is not None and egp is not None:
            if abs(egp - supported / total) > 1e-8:
                violations["egp_mismatch"].append(
                    {**identity, "egp": egp, "expected": supported / total}
                )

        if (
            _number(row, METRIC_COLUMNS["probe_waste"]) is not None
            and row.get("System action") != "ASK_CLARIFICATION"
        ):
            review_flags["probe_labels_outside_formal_ask_action"].append(identity)

        if (
            _number(row, METRIC_COLUMNS["conflict_disclosure_quality"]) is not None
            and not _is_present(row.get("Detected conflicts"))
        ):
            review_flags["conflict_labels_without_structured_conflict"].append(identity)

    duplicate_keys: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (row["System"], row["Query ID"], row["Stratum"])
        if key in seen:
            duplicate_keys.append(
                {"system": key[0], "query_id": key[1], "stratum": key[2]}
            )
        seen.add(key)

    return {
        "row_count": len(rows),
        "system_counts": dict(Counter(row["System"] for row in rows)),
        "stratum_counts": dict(Counter(row["Stratum"] for row in rows)),
        "unique_query_ids": len({row["Query ID"] for row in rows}),
        "duplicate_system_query_stratum_keys": duplicate_keys,
        "violations": {
            name: {"count": len(items), "examples": items[:10]}
            for name, items in violations.items()
        },
        "review_flags": {
            name: {"count": len(items), "examples": items[:10]}
            for name, items in review_flags.items()
        },
        "notes": [
            "Probe labels outside ASK_CLARIFICATION are retained: a generated response may ask a natural-language clarification without a formal route action.",
            "Conflict labels without a structured detected-conflict record are retained: the annotator may have judged a conflict from the supplied source evidence even when the system did not detect it.",
        ],
    }


def _paired_tests(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str], dict[str, dict[str, str]]] = {}
    for row in rows:
        key = (row["Stratum"], row["Query ID"], row["System"])
        by_key.setdefault(key, {})[row["System"]] = row

    tests: list[dict[str, Any]] = []
    ours_name = "Ours_dual_conditioned_mas"
    for baseline in SYSTEMS:
        if baseline == ours_name:
            continue
        for stratum in STRATA:
            for metric, column in (
                ("ucr", METRIC_COLUMNS["ucr"]),
                ("recommendation_coverage", METRIC_COLUMNS["recommendation_coverage"]),
            ):
                ours_values: list[int] = []
                baseline_values: list[int] = []
                case_keys = {(key[0], key[1]) for key in by_key}
                for case_stratum, query_id in case_keys:
                    if case_stratum != stratum:
                        continue
                    ours_row = by_key.get((stratum, query_id, ours_name), {}).get(ours_name)
                    baseline_row = by_key.get((stratum, query_id, baseline), {}).get(baseline)
                    if ours_row is None or baseline_row is None:
                        continue
                    ours_value = _number(ours_row, column)
                    baseline_value = _number(baseline_row, column)
                    if ours_value is None or baseline_value is None:
                        continue
                    ours_values.append(int(ours_value))
                    baseline_values.append(int(baseline_value))
                if not ours_values:
                    continue
                tests.append(
                    {
                        "baseline": baseline,
                        "stratum": stratum,
                        "metric": metric,
                        "paired_n": len(ours_values),
                        "ours_rate": statistics.mean(ours_values),
                        "baseline_rate": statistics.mean(baseline_values),
                        "ours_1_baseline_0": sum(
                            ours == 1 and other == 0
                            for ours, other in zip(ours_values, baseline_values, strict=True)
                        ),
                        "ours_0_baseline_1": sum(
                            ours == 0 and other == 1
                            for ours, other in zip(ours_values, baseline_values, strict=True)
                        ),
                        "exact_mcnemar_p": _exact_mcnemar_pvalue(
                            ours_values, baseline_values
                        ),
                    }
                )
    adjusted = _holm_adjust([float(test["exact_mcnemar_p"]) for test in tests])
    for test, adjusted_p in zip(tests, adjusted, strict=True):
        test["holm_adjusted_p"] = adjusted_p
    return tests


def analyze_annotations(input_path: Path) -> dict[str, Any]:
    with input_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError("Annotation CSV has no header row")
        required = {
            "System",
            "Query ID",
            "Stratum",
            "System action",
            "Detected conflicts",
            "Expected-route proxy (0/1)",
            "Prompt tokens",
            "Output tokens",
            "Latency seconds",
            "EGP",
            *METRIC_COLUMNS.values(),
            *COUNT_COLUMNS,
        }
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"Annotation CSV is missing required columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError("Annotation CSV contains no annotation rows")
    groups = [
        _summarize_group(
            [
                row
                for row in rows
                if row["System"] == system and row["Stratum"] == stratum
            ],
            system,
            stratum,
        )
        for system in SYSTEMS
        for stratum in STRATA
    ]
    all_systems = [
        _summarize_group(
            [row for row in rows if row["System"] == system], system, "ALL"
        )
        for system in SYSTEMS
    ]
    return {
        "source_file": input_path.name,
        "annotation_audit": _audit_rows(rows),
        "all_systems": all_systems,
        "by_system_and_stratum": groups,
        "paired_tests": _paired_tests(rows),
        "interpretation_limits": [
            "S3/S4 are counterfactual catalog-mask conditions and some query IDs recur across strata.",
            "Exact paired tests are exploratory, run within each stratum, and compare Ours with each baseline for UCR and recommendation coverage; Holm correction covers all such tests.",
            "PWR is reported among formally labeled ASK_CLARIFICATION actions; additionally labeled natural-language questions outside that action require manual applicability review.",
            "Conflict-disclosure labels are not interpreted as a complete conflict metric until applicability is checked against the row-level evidence.",
        ],
    }


def _write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    columns = [
        "system",
        "stratum",
        "row_count",
        "unique_query_ids",
        "recommendation_coverage_n",
        "recommendation_coverage_mean",
        "ucr_n",
        "ucr_mean",
        "csr_n",
        "csr_mean",
        "egp_micro",
        "egp_macro_n",
        "egp_macro",
        "formal_clarification_count",
        "probe_waste_formal_ask_n",
        "probe_waste_formal_ask_mean",
        "probe_waste_annotated_n",
        "probe_waste_annotated_mean",
        "disambiguation_success_formal_ask_mean",
        "disambiguation_success_annotated_n",
        "disambiguation_success_annotated_mean",
        "abstain_quality_n",
        "abstain_quality_mean",
        "conflict_label_n",
        "rows_with_structured_detected_conflict",
        "mean_tokens_per_case",
        "mean_latency_seconds",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            {column: _json_value(group.get(column)) for column in columns}
            for group in summary["by_system_and_stratum"]
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize and quality-check Phase 4 expert annotations."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("reports/phase4/phase4_annotation_completed.csv"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/phase4/phase4_annotation_analysis.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("reports/phase4/phase4_annotation_summary.csv"),
    )
    args = parser.parse_args()
    summary = analyze_annotations(args.input)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_summary_csv(args.output_csv, summary)
    print(
        json.dumps(
            {
                "rows": summary["annotation_audit"]["row_count"],
                "systems": summary["annotation_audit"]["system_counts"],
                "strata": summary["annotation_audit"]["stratum_counts"],
                "json": str(args.output_json),
                "csv": str(args.output_csv),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

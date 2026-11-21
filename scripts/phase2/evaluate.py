from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import random
import re
import statistics
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.phase2.catalog_profiler import (
    CatalogProfiler,
    _DETAIL_KEYS,
    _SPEC_PATTERNS,
)
from scripts.phase2.evidence_agents import ReviewEvidenceAgent
from scripts.phase2.orchestrator import AdaptiveOrchestrator
from scripts.phase2.query_analyzer import (
    ATTRIBUTE_ALIASES,
    QueryAnalyzer,
    SUPPORTED_ATTRIBUTES,
)
from scripts.phase2.types import CandidateProfile, CatalogState, QueryState


CORE_ATTRIBUTES = {"display", "ram", "battery", "weight_portability"}
DEFAULT_UNCERTAINTY_GRID = (0.3, 0.5, 0.7)
DEFAULT_COVERAGE_GRID = (0.25, 0.5, 0.75)
DEFAULT_ENTROPY_GRID = (0.1, 0.2, 0.3)


@dataclass
class EvaluationCase:
    record_id: Any
    text: str
    gold_ambiguous_attributes: list[str]
    query_state: QueryState
    catalog_state: CatalogState
    profile_seconds: float
    scenario_override: str | None = None
    synthetic_catalog_mask: bool = False

    @property
    def gold_ambiguous(self) -> bool:
        return bool(self.gold_ambiguous_attributes)

    @property
    def attributes(self) -> list[str]:
        return [
            attribute
            for attribute in self.query_state.target_attributes
            if attribute in CORE_ATTRIBUTES
        ]

    @property
    def supported_attributes(self) -> list[str]:
        return [
            attribute
            for attribute in self.query_state.target_attributes
            if attribute in SUPPORTED_ATTRIBUTES
        ]


class CountingReviewAgent:
    """Count routed review calls without loading or running the ABSA model."""

    def __init__(self) -> None:
        self.calls = 0
        self.model_input_tokens = 0

    def collect(self, candidates: list[Any], attributes: set[str]) -> tuple[list[dict[str, Any]], int]:
        self.calls += 1
        return [], 0


def _read_queries(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            if not isinstance(record.get("text"), str):
                raise ValueError(f"Missing string query text at {path}:{line_number}")
            records.append(record)
    return records


def _vague_attributes(
    record: dict[str, Any], query_state: QueryState
) -> list[str]:
    annotations = record.get("vague_annotations", [])
    if not isinstance(annotations, list):
        raise ValueError(f"Query {record.get('id')} has invalid vague_annotations")
    text = query_state.text.casefold()
    vague_spans: list[tuple[int, int]] = []
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        word = annotation.get("word")
        start = annotation.get("start index")
        if not isinstance(word, str) or not isinstance(start, int):
            continue
        vague_spans.append((start, start + len(word)))

    attributes: list[str] = []
    for attribute in query_state.target_attributes:
        if attribute not in CORE_ATTRIBUTES:
            continue
        alias_spans = [
            (match.start(), match.end())
            for alias in ATTRIBUTE_ALIASES.get(attribute, ())
            for match in re.finditer(re.escape(alias), text)
        ]
        if any(
            max(alias_start - vague_end, vague_start - alias_end, 0) <= 24
            for vague_start, vague_end in vague_spans
            for alias_start, alias_end in alias_spans
        ):
            attributes.append(attribute)
    return attributes


def _select_sample(
    records: list[dict[str, Any]], sample_size: int, seed: int
) -> list[tuple[dict[str, Any], QueryState, list[str]]]:
    analyzer = QueryAnalyzer()
    groups: dict[bool, list[tuple[dict[str, Any], QueryState, list[str]]]] = {
        False: [],
        True: [],
    }
    for record in records:
        query_state = analyzer.analyze(record["text"])
        if not (set(query_state.target_attributes) & CORE_ATTRIBUTES):
            continue
        vague_attributes = _vague_attributes(record, query_state)
        groups[bool(vague_attributes)].append(
            (record, query_state, vague_attributes)
        )

    requested_per_group = {
        False: sample_size // 2,
        True: sample_size - sample_size // 2,
    }
    if any(
        len(groups[ambiguous]) < required
        for ambiguous, required in requested_per_group.items()
    ):
        raise ValueError(
            f"Cannot draw a balanced {sample_size}-query sample: "
            f"clear={len(groups[False])}, ambiguous={len(groups[True])}"
        )
    rng = random.Random(seed)
    sample = [
        *rng.sample(groups[False], sample_size // 2),
        *rng.sample(groups[True], sample_size - sample_size // 2),
    ]
    rng.shuffle(sample)
    return sample


def _binary_metrics(expected: list[bool], predicted: list[bool]) -> dict[str, float | int]:
    tp = sum(actual and output for actual, output in zip(expected, predicted))
    fp = sum(not actual and output for actual, output in zip(expected, predicted))
    fn = sum(actual and not output for actual, output in zip(expected, predicted))
    tn = sum(not actual and not output for actual, output in zip(expected, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    beta_squared = 0.25
    f0_5 = (
        (1 + beta_squared) * precision * recall
        / (beta_squared * precision + recall)
        if beta_squared * precision + recall
        else 0.0
    )
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "f0_5": round(f0_5, 4),
        "probe_waste_rate": round(fp / (tp + fp), 4) if tp + fp else 0.0,
    }


def _calibrate(
    cases: list[EvaluationCase],
    uncertainty_grid: tuple[float, ...],
    coverage_grid: tuple[float, ...],
    entropy_grid: tuple[float, ...],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    examples = [
        (case, attribute)
        for case in cases
        for attribute in case.attributes
    ]
    gold = [
        attribute in case.gold_ambiguous_attributes
        and case.catalog_state.completeness.get(attribute, 0.0) > 0
        and case.catalog_state.entropy.get(attribute, 0.0) > 0
        for case, attribute in examples
    ]
    scores: list[dict[str, Any]] = []
    for uncertainty, coverage, entropy in itertools.product(
        uncertainty_grid, coverage_grid, entropy_grid
    ):
        predicted = [
            (
                case.query_state.uncertainty.get(attribute, 0.0) >= uncertainty
                and case.catalog_state.completeness.get(attribute, 0.0) >= coverage
                and case.catalog_state.entropy.get(attribute, 0.0) >= entropy
            )
            for case, attribute in examples
        ]
        scores.append(
            {
                "uncertainty_threshold": uncertainty,
                "coverage_threshold": coverage,
                "entropy_threshold": entropy,
                **_binary_metrics(gold, predicted),
            }
        )
    best = max(
        scores,
        key=lambda row: (
            row["f0_5"],
            -row["probe_waste_rate"],
            row["precision"],
            row["uncertainty_threshold"],
            row["coverage_threshold"],
            row["entropy_threshold"],
        ),
    )
    chosen = {
        "ambiguity_threshold": float(best["uncertainty_threshold"]),
        "minimum_clarification_coverage": float(best["coverage_threshold"]),
        "minimum_catalog_entropy": float(best["entropy_threshold"]),
    }
    return chosen, scores


def _split_calibration_evaluation(
    cases: list[EvaluationCase], seed: int
) -> tuple[list[EvaluationCase], list[EvaluationCase]]:
    groups = {
        ambiguous: [case for case in cases if case.gold_ambiguous is ambiguous]
        for ambiguous in (False, True)
    }
    if any(len(group) < 2 or len(group) % 2 for group in groups.values()):
        raise ValueError(
            "The selected sample must contain an even number of clear and "
            "ambiguous queries to create balanced calibration/evaluation splits"
        )
    rng = random.Random(seed)
    calibration: list[EvaluationCase] = []
    evaluation: list[EvaluationCase] = []
    for group in groups.values():
        rng.shuffle(group)
        midpoint = len(group) // 2
        calibration.extend(group[:midpoint])
        evaluation.extend(group[midpoint:])
    rng.shuffle(calibration)
    rng.shuffle(evaluation)
    return calibration, evaluation


def _scenario(case: EvaluationCase, thresholds: dict[str, float]) -> str:
    if case.scenario_override is not None:
        return case.scenario_override
    if not case.gold_ambiguous:
        return (
            "S1"
            if all(
                case.catalog_state.completeness.get(attribute, 0.0) >= 1.0
                for attribute in case.attributes
            )
            else "S3"
        )
    rich = any(
        attribute in case.gold_ambiguous_attributes
        and case.catalog_state.completeness.get(attribute, 0.0)
        >= thresholds["minimum_clarification_coverage"]
        and case.catalog_state.entropy.get(attribute, 0.0)
        >= thresholds["minimum_catalog_entropy"]
        for attribute in case.attributes
    )
    return "S2" if rich else "S4"


def _requests_experiential_evidence(text: str) -> bool:
    lower = text.casefold()
    return bool(
        re.search(r"\b(?:experience|feels|comfortable|reliable|reviews?)\b", lower)
        or re.search(r"\b(?:battery|charge)\b.{0,24}\b(?:lasts?|hours?)\b", lower)
    )


def _mask_catalog_attributes(
    case: EvaluationCase, attributes: set[str], scenario: str
) -> EvaluationCase:
    masked_candidates = []
    for candidate in case.catalog_state.candidates:
        product = copy.deepcopy(candidate.product)
        details = product.get("details")
        if isinstance(details, dict):
            masked_keys = {
                key.casefold()
                for attribute in attributes
                for key in _DETAIL_KEYS.get(attribute, ())
            }
            product["details"] = {
                key: value for key, value in details.items()
                if str(key).casefold() not in masked_keys
            }

        def strip_values(value: Any) -> Any:
            if isinstance(value, str):
                for attribute in attributes:
                    for pattern in _SPEC_PATTERNS.get(attribute, ()):
                        value = pattern.sub("", value)
                return value
            if isinstance(value, list):
                return [strip_values(item) for item in value]
            if isinstance(value, dict):
                return {key: strip_values(item) for key, item in value.items()}
            return value

        product["features"] = strip_values(product.get("features"))
        product["description"] = strip_values(product.get("description"))
        product["title"] = strip_values(product.get("title"))
        if "price" in attributes:
            product["price"] = None
        masked_candidates.append(
            CandidateProfile(
                product=product,
                specs={
                    key: value for key, value in candidate.specs.items()
                    if key not in attributes
                },
                spec_evidence=[
                    row for row in candidate.spec_evidence
                    if row.get("attribute") not in attributes
                ],
                retrieval_score=candidate.retrieval_score,
            )
        )
    completeness = dict(case.catalog_state.completeness)
    entropy = dict(case.catalog_state.entropy)
    for attribute in attributes:
        completeness[attribute] = 0.0
        entropy[attribute] = 0.0
    masked_state = CatalogState(
        masked_candidates,
        case.catalog_state.preliminary_candidate_count,
        completeness,
        entropy,
    )
    return EvaluationCase(
        record_id=case.record_id,
        text=case.text,
        gold_ambiguous_attributes=case.gold_ambiguous_attributes,
        query_state=case.query_state,
        catalog_state=masked_state,
        profile_seconds=case.profile_seconds,
        scenario_override=scenario,
        synthetic_catalog_mask=True,
    )


def _build_stratified_scenarios(
    cases: list[EvaluationCase], thresholds: dict[str, float]
) -> list[EvaluationCase]:
    explicit_complete = [
        case
        for case in cases
        if not case.gold_ambiguous
        and case.supported_attributes
        and not _requests_experiential_evidence(case.text)
        and all(
            case.catalog_state.completeness.get(attribute, 0.0) >= 1.0
            and case.query_state.uncertainty.get(attribute, 0.0)
            < thresholds["ambiguity_threshold"]
            for attribute in case.supported_attributes
        )
    ]
    ambiguous_supported = [
        case
        for case in cases
        if any(
            attribute in case.gold_ambiguous_attributes
            and case.query_state.uncertainty.get(attribute, 0.0)
            >= thresholds["ambiguity_threshold"]
            and case.catalog_state.completeness.get(attribute, 0.0)
            >= thresholds["minimum_clarification_coverage"]
            and case.catalog_state.entropy.get(attribute, 0.0)
            >= thresholds["minimum_catalog_entropy"]
            for attribute in case.attributes
        )
    ]
    if not explicit_complete:
        raise ValueError(
            "Held-out VACOS sample contains no clear, fully covered queries for S1"
        )
    if not ambiguous_supported:
        raise ValueError(
            "Held-out VACOS sample contains no vague, evidence-supported queries for S2"
        )
    scenarios = [
        EvaluationCase(
            record_id=case.record_id,
            text=case.text,
            gold_ambiguous_attributes=case.gold_ambiguous_attributes,
            query_state=case.query_state,
            catalog_state=case.catalog_state,
            profile_seconds=case.profile_seconds,
            scenario_override="S1",
        )
        for case in explicit_complete
    ]
    scenarios.extend(
        _mask_catalog_attributes(case, set(case.supported_attributes), "S3")
        for case in explicit_complete
    )
    scenarios.extend(
        EvaluationCase(
            record_id=case.record_id,
            text=case.text,
            gold_ambiguous_attributes=case.gold_ambiguous_attributes,
            query_state=case.query_state,
            catalog_state=case.catalog_state,
            profile_seconds=case.profile_seconds,
            scenario_override="S2",
        )
        for case in ambiguous_supported
    )
    scenarios.extend(
        _mask_catalog_attributes(
            case, set(case.supported_attributes), "S4"
        )
        for case in ambiguous_supported
    )
    return scenarios


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _run_evaluation(
    cases: list[EvaluationCase],
    thresholds: dict[str, float],
    *,
    profiler: CatalogProfiler,
    review_agent: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scenario_counts: Counter[str] = Counter()
    scenario_correct: Counter[str] = Counter()
    scenario_early_exit: Counter[str] = Counter()
    scenario_ask: Counter[str] = Counter()
    scenario_review: Counter[str] = Counter()
    scenario_abstain: Counter[str] = Counter()
    latencies: list[float] = []
    probe_waste = 0

    for case in cases:
        scenario = _scenario(case, thresholds)
        scenario_counts[scenario] += 1
        tokens_before = int(getattr(review_agent, "model_input_tokens", 0))
        system = AdaptiveOrchestrator(
            profiler,
            review_agent=review_agent,
            ambiguity_threshold=thresholds["ambiguity_threshold"],
            minimum_clarification_coverage=thresholds[
                "minimum_clarification_coverage"
            ],
            minimum_catalog_entropy=thresholds["minimum_catalog_entropy"],
        )
        started = time.perf_counter()
        trace = system.run_states(case.query_state, case.catalog_state)
        route_seconds = time.perf_counter() - started
        total_seconds = case.profile_seconds + route_seconds
        latencies.append(total_seconds)
        input_tokens = int(getattr(review_agent, "model_input_tokens", 0)) - tokens_before
        asked = trace.decision.action == "ASK_CLARIFICATION"
        review_called = trace.review_agent_calls > 0
        early_exit = (
            scenario == "S1"
            and not asked
            and trace.spec_agent_calls == 0
            and trace.review_agent_calls == 0
        )
        safe_abstain = (
            scenario == "S4"
            and trace.decision.action == "ABSTAIN_DISCLOSE"
            and not asked
        )
        if asked and scenario != "S2":
            probe_waste += 1
        scenario_early_exit[scenario] += int(early_exit)
        scenario_ask[scenario] += int(asked)
        scenario_review[scenario] += int(review_called)
        scenario_abstain[scenario] += int(safe_abstain)
        correct = {
            "S1": early_exit,
            "S2": asked,
            "S3": review_called,
            "S4": safe_abstain,
        }[scenario]
        scenario_correct[scenario] += int(correct)
        rows.append(
            {
                "query_id": case.record_id,
                "scenario": scenario,
                "gold_ambiguous": case.gold_ambiguous,
                "query": case.text,
                "target_attributes": case.attributes,
                "completeness": {
                    key: case.catalog_state.completeness.get(key, 0.0)
                    for key in case.attributes
                },
                "entropy": {
                    key: case.catalog_state.entropy.get(key, 0.0)
                    for key in case.attributes
                },
                "candidate_count": case.catalog_state.preliminary_candidate_count,
                "action": trace.decision.action,
                "actions_taken": trace.actions_taken,
                "spec_agent_calls": trace.spec_agent_calls,
                "review_agent_calls": trace.review_agent_calls,
                "review_source_records_scanned": trace.review_source_records_scanned,
                "external_llm_tokens": 0,
                "review_model_input_tokens": input_tokens,
                "profile_seconds": round(case.profile_seconds, 6),
                "route_seconds": round(route_seconds, 6),
                "total_seconds": round(total_seconds, 6),
                "expected_behavior_met": correct,
                "synthetic_catalog_mask": case.synthetic_catalog_mask,
            }
        )

    scenario_metrics = {
        scenario: {
            "count": scenario_counts[scenario],
            "behavior_rate": round(
                scenario_correct[scenario] / scenario_counts[scenario], 4
            ) if scenario_counts[scenario] else 0.0,
            "early_exit_rate": round(
                scenario_early_exit[scenario] / scenario_counts[scenario], 4
            ) if scenario_counts[scenario] else 0.0,
            "ask_clarification_rate": round(
                scenario_ask[scenario] / scenario_counts[scenario], 4
            ) if scenario_counts[scenario] else 0.0,
            "review_activation_rate": round(
                scenario_review[scenario] / scenario_counts[scenario], 4
            ) if scenario_counts[scenario] else 0.0,
            "safe_abstain_rate": round(
                scenario_abstain[scenario] / scenario_counts[scenario], 4
            ) if scenario_counts[scenario] else 0.0,
        }
        for scenario in ("S1", "S2", "S3", "S4")
    }
    total = len(rows)
    metrics = {
        "scenario_metrics": scenario_metrics,
        "overall_scenario_behavior_rate": round(
            sum(scenario_correct.values()) / total, 4
        ) if total else 0.0,
        "probe_waste_rate": round(probe_waste / sum(scenario_ask.values()), 4)
        if sum(scenario_ask.values()) else 0.0,
        "total_spec_agent_calls": sum(row["spec_agent_calls"] for row in rows),
        "total_review_agent_calls": sum(row["review_agent_calls"] for row in rows),
        "total_agent_calls": sum(
            row["spec_agent_calls"] + row["review_agent_calls"] for row in rows
        ),
        "total_reviews_scanned": sum(
            row["review_source_records_scanned"] for row in rows
        ),
        "external_llm_tokens": 0,
        "review_model_input_tokens": sum(
            row["review_model_input_tokens"] for row in rows
        ),
        "mean_latency_seconds": round(statistics.mean(latencies), 6) if latencies else 0.0,
        "p50_latency_seconds": round(_percentile(latencies, 0.50), 6),
        "p95_latency_seconds": round(_percentile(latencies, 0.95), 6),
        "latency_includes_index_build": False,
    }
    return rows, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate Phase 2 routing thresholds and evaluate four VACOS strata."
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("dataset/processed/vacos_laptops_bio.jsonl"),
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("dataset/processed/amazon-laptops/laptops_catalog.jsonl"),
    )
    parser.add_argument(
        "--reviews",
        type=Path,
        default=Path("dataset/processed/amazon-laptops/laptops_reviews.jsonl.gz"),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/setfit-absa-laptops"),
    )
    parser.add_argument("--sample-size", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/phase2-evaluation"))
    parser.add_argument("--with-review-model", action="store_true")
    parser.add_argument("--review-scan-limit", type=int, default=5000)
    parser.add_argument("--uncertainty-grid", nargs="+", type=float, default=DEFAULT_UNCERTAINTY_GRID)
    parser.add_argument("--coverage-grid", nargs="+", type=float, default=DEFAULT_COVERAGE_GRID)
    parser.add_argument("--entropy-grid", nargs="+", type=float, default=DEFAULT_ENTROPY_GRID)
    args = parser.parse_args()

    if args.sample_size < 4 or args.sample_size > 240 or args.sample_size % 4:
        parser.error(
            "--sample-size must be a multiple of four between 4 and 240 "
            "to form balanced calibration and evaluation splits"
        )
    if args.top_k < 1:
        parser.error("--top-k must be greater than zero")
    if args.with_review_model and args.sample_size > 32:
        parser.error(
            "Real Review ABSA inference is bounded to 32 cases per run; "
            "use --sample-size 32 or fewer."
        )
    for grid_name, grid in (
        ("uncertainty", args.uncertainty_grid),
        ("coverage", args.coverage_grid),
        ("entropy", args.entropy_grid),
    ):
        if not grid or any(value < 0 or value > 1 for value in grid):
            parser.error(f"--{grid_name}-grid values must be between zero and one")

    selected = _select_sample(
        _read_queries(args.queries), args.sample_size, args.seed
    )
    profiler = CatalogProfiler(args.catalog, max_candidates=args.top_k)
    profiler.prepare()
    cases: list[EvaluationCase] = []
    for record, query_state, ambiguous in selected:
        started = time.perf_counter()
        catalog_state = profiler.retrieve(query_state)
        profile_seconds = time.perf_counter() - started
        cases.append(
            EvaluationCase(
                record.get("id"),
                query_state.text,
                ambiguous,
                query_state,
                catalog_state,
                profile_seconds,
            )
        )

    calibration_cases, evaluation_cases = _split_calibration_evaluation(
        cases, args.seed + 1
    )
    chosen, calibration_scores = _calibrate(
        calibration_cases,
        tuple(args.uncertainty_grid),
        tuple(args.coverage_grid),
        tuple(args.entropy_grid),
    )
    review_agent: Any
    if args.with_review_model:
        review_agent = ReviewEvidenceAgent(
            args.reviews,
            model_dir=args.model_dir,
            max_source_records=args.review_scan_limit,
        )
    else:
        review_agent = CountingReviewAgent()
    scenario_cases = _build_stratified_scenarios(
        evaluation_cases, chosen
    )
    evaluation_rows, metrics = _run_evaluation(
        scenario_cases, chosen, profiler=profiler, review_agent=review_agent
    )

    strata_counts = Counter(row["scenario"] for row in evaluation_rows)
    report = {
        "dataset": str(args.queries),
        "catalog": str(args.catalog),
        "sample_size": len(cases),
        "calibration_sample_size": len(calibration_cases),
        "evaluation_sample_size": len(evaluation_cases),
        "evaluation_scenario_runs": len(scenario_cases),
        "seed": args.seed,
        "sample_design": (
            "Balanced random sample from VACOS laptop queries with at least one "
            "recognized core attribute; half have attribute-linked vague spans and half do not."
        ),
        "evaluation_mode": (
            "real_review_absa" if args.with_review_model else "routing_only_no_model_inference"
        ),
        "catalog_index_build_seconds": round(profiler.index_build_seconds, 6),
        "strata_counts": {key: strata_counts.get(key, 0) for key in ("S1", "S2", "S3", "S4")},
        "calibration_ground_truth": (
            "A vague annotation is linked to a core attribute only when its character "
            "span is within 24 characters of that attribute's query mention. It is "
            "eligible for clarification when that attribute has nonzero metadata "
            "completeness and entropy. This is a proxy label, not human judgment."
        ),
        "calibration_selection": (
            "Maximum F0.5 (precision-weighted clarification score); ties prefer "
            "lower Probe-Waste Rate, higher precision, then conservative thresholds."
        ),
        "calibration_evaluation_split": (
            "Balanced 50/50 split by annotated core-attribute vagueness; thresholds "
            "are selected only on calibration queries and metrics below use held-out queries."
        ),
        "chosen_thresholds": chosen,
        "calibration_metrics": {},
        "calibration_grid": calibration_scores,
        "evaluation_metrics": metrics,
        "limitations": [
            "Vague spans are associated with recognized core attributes within a 24-character window; this heuristic may misattribute a vague phrase.",
            "The held-out query set is reused in paired S3/S4 counterfactuals by masking metadata; scenario rows are not independent queries.",
            "A query is not an independent test of model quality; this sample is for routing calibration only.",
            "Routing-only mode counts review-agent dispatches but does not run SetFit or scan review records.",
            "External LLM token use is zero. Review-model input tokens are reported only in real_review_absa mode.",
            "VACOS does not provide gold product recommendations or conflict labels; S1-S4 strata are operational proxies.",
        ],
        "cases": evaluation_rows,
    }
    report["calibration_metrics"] = next(
        (
            {
                key: value
                for key, value in row.items()
                if key in {"precision", "recall", "f1", "f0_5", "probe_waste_rate"}
            }
            for row in calibration_scores
            if (
                row["uncertainty_threshold"] == chosen["ambiguity_threshold"]
                and row["coverage_threshold"] == chosen["minimum_clarification_coverage"]
                and row["entropy_threshold"] == chosen["minimum_catalog_entropy"]
            )
        ),
        {},
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "phase2_evaluation.json"
    csv_path = args.output_dir / "phase2_evaluation_cases.csv"
    json_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        fieldnames = [
            "query_id", "scenario", "gold_ambiguous", "target_attributes",
            "candidate_count", "action", "actions_taken", "spec_agent_calls",
            "review_agent_calls", "review_source_records_scanned",
            "external_llm_tokens", "review_model_input_tokens",
            "synthetic_catalog_mask",
            "profile_seconds", "route_seconds", "total_seconds",
            "expected_behavior_met", "query",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in evaluation_rows:
            flattened = {
                **row,
                "target_attributes": ",".join(row["target_attributes"]),
                "actions_taken": ",".join(row["actions_taken"]),
            }
            writer.writerow({key: flattened.get(key) for key in fieldnames})

    summary = {
        "sample_size": len(cases),
        "calibration_sample_size": len(calibration_cases),
        "evaluation_sample_size": len(evaluation_cases),
        "strata_counts": report["strata_counts"],
        "chosen_thresholds": chosen,
        "calibration_metrics": report["calibration_metrics"],
        "evaluation_metrics": metrics,
        "catalog_index_build_seconds": report["catalog_index_build_seconds"],
        "output_json": str(json_path),
        "output_csv": str(csv_path),
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    profiler.close()


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import random
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scripts.phase1.review_absa_agent import ReviewABSAAgent
from scripts.phase2.catalog_profiler import CatalogProfiler
from scripts.phase2.conflict_evaluator import ConflictEvaluator
from scripts.phase2.decision_synthesizer import DecisionSynthesizer
from scripts.phase2.evaluate import (
    EvaluationCase,
    _build_stratified_scenarios,
    _read_queries,
    _select_sample,
    _split_calibration_evaluation,
)
from scripts.phase2.evidence_agents import SpecDescriptionAgent
from scripts.phase2.orchestrator import AdaptiveOrchestrator
from scripts.phase2.query_analyzer import QueryAnalyzer, SUPPORTED_ATTRIBUTES
from scripts.phase2.types import CandidateProfile, CatalogState, Decision, QueryState
from scripts.phase3.ollama_client import Generation, OllamaClient
from scripts.phase3.simulated_users import ReviewGroundedUserSimulator


BASELINES = (
    "B1_single_llm",
    "B2_single_agent_retrieval",
    "B3_fixed_sequential_mas",
    "B4_query_adaptive_mas",
    "Ours_dual_conditioned_mas",
)
SYSTEM_PROMPT = (
    "You are a laptop shopping assistant. Answer the user's request clearly. "
    "Use only facts in the supplied evidence; do not invent specifications, "
    "availability, prices, or review findings. Cite evidence with its source "
    "identifier in square brackets. Distinguish manufacturer metadata from "
    "buyer-review opinions. If the decision says to clarify, ask one concise "
    "question instead of recommending. If evidence is insufficient or "
    "conflicting, disclose that limitation rather than guessing."
)


@dataclass
class IndexedReviewAgent:
    evidence: list[dict[str, Any]]
    reviews_scanned: int

    def collect(
        self, candidates: list[CandidateProfile], attributes: set[str]
    ) -> tuple[list[dict[str, Any]], int]:
        parents = {candidate.parent_asin for candidate in candidates}
        rows = [
            row
            for row in self.evidence
            if str(row.get("parent_asin", "")) in parents
            and row.get("aspect_category") in attributes
        ]
        return rows, self.reviews_scanned


@dataclass
class BaselineRun:
    name: str
    decision: Decision | None
    tool_calls: int
    spec_calls: int
    review_calls: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_reviews_for_candidates(
    path: Path,
    parent_asins: set[str],
    *,
    scan_limit: int,
    per_product_limit: int = 6,
) -> tuple[list[dict[str, Any]], int]:
    opener = gzip.open if path.suffix.casefold() == ".gz" else open
    selected: list[dict[str, Any]] = []
    counts = dict.fromkeys(parent_asins, 0)
    scanned = 0
    with opener(path, "rt", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            scanned += 1
            review = json.loads(line)
            parent = review.get("parent_asin")
            if parent in counts and counts[parent] < per_product_limit:
                counts[parent] += 1
                selected.append(review)
            if scanned >= scan_limit or all(
                count >= per_product_limit for count in counts.values()
            ):
                break
    return selected, scanned


def _classify_reviews(
    reviews: list[dict[str, Any]], model_dir: Path
) -> tuple[list[dict[str, Any]], int]:
    if not reviews:
        return [], 0
    classifier = ReviewABSAAgent(model_dir=model_dir, batch_size=64)
    evidence: list[dict[str, Any]] = []
    for batch in classifier.analyze_reviews(reviews, batch_size=64):
        evidence.extend(batch)
    for row in evidence:
        row["source"] = "amazon_reviews_2023"
    return evidence, classifier.input_token_count


def _ensure_absa_dependencies() -> None:
    missing = [
        module
        for module in ("setfit", "spacy")
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        raise RuntimeError(
            "Phase 3 review-grounded simulation requires "
            f"{', '.join(missing)} in the active interpreter "
            f"({sys.executable}). Install them with:\n"
            f'  "{sys.executable}" -m pip install -r requirements.txt'
        )


def _prepare_cases(
    queries_path: Path,
    catalog_path: Path,
    *,
    sample_size: int,
    seed: int,
    top_k: int,
) -> tuple[list[EvaluationCase], list[EvaluationCase], CatalogProfiler]:
    selected = _select_sample(_read_queries(queries_path), sample_size, seed)
    profiler = CatalogProfiler(catalog_path, max_candidates=top_k)
    try:
        profiler.prepare()
        cases: list[EvaluationCase] = []
        for record, query_state, vague_attributes in selected:
            started = time.perf_counter()
            catalog_state = profiler.retrieve(query_state)
            cases.append(
                EvaluationCase(
                    record.get("id"),
                    query_state.text,
                    vague_attributes,
                    query_state,
                    catalog_state,
                    time.perf_counter() - started,
                )
            )
        calibration, heldout = _split_calibration_evaluation(cases, seed + 1)
        thresholds = {
            "ambiguity_threshold": 0.5,
            "minimum_clarification_coverage": 0.5,
            "minimum_catalog_entropy": 0.3,
        }
        scenarios = _build_stratified_scenarios(heldout, thresholds)
    except Exception:
        profiler.close()
        raise
    return calibration, scenarios, profiler


def _limit_scenarios(
    scenarios: list[EvaluationCase], per_stratum: int, seed: int
) -> list[EvaluationCase]:
    rng = random.Random(seed)
    selected: list[EvaluationCase] = []
    for scenario in ("S1", "S2", "S3", "S4"):
        group = [
            case for case in scenarios if case.scenario_override == scenario
        ]
        if len(group) < per_stratum:
            raise ValueError(
                f"Only {len(group)} eligible {scenario} cases; "
                f"requested {per_stratum}"
            )
        selected.extend(rng.sample(group, per_stratum))
    return selected


def _source_quote(row: dict[str, Any]) -> str:
    source = str(row.get("source", "metadata"))
    quote = str(row.get("quote", row.get("sentence_text", "")))
    parent = str(row.get("parent_asin", "unknown"))
    return f"{source} parent_asin={parent}: {quote[:400]}"


def _route_baseline(
    name: str,
    query: QueryState,
    catalog: CatalogState,
    review_agent: IndexedReviewAgent,
    *,
    selected_tools: set[str] | None = None,
) -> BaselineRun:
    synthesizer = DecisionSynthesizer()
    conflict_evaluator = ConflictEvaluator()
    spec_agent = SpecDescriptionAgent()
    candidates = catalog.candidates
    requested = set(query.target_attributes) & SUPPORTED_ATTRIBUTES
    all_attributes = requested or set(SUPPORTED_ATTRIBUTES)
    metadata_evidence = [
        {**row, "parent_asin": candidate.parent_asin}
        for candidate in candidates
        for row in candidate.spec_evidence
        if row.get("attribute") in all_attributes
    ]

    if name == "B1_single_llm":
        return BaselineRun(name, None, 0, 0, 0)

    if name == "B2_single_agent_retrieval":
        if selected_tools is None or "catalog_search" not in selected_tools:
            raise ValueError("B2 ReAct planner must select catalog_search")
        reviews: list[dict[str, Any]] = []
        if "review_search" in selected_tools:
            reviews, _ = review_agent.collect(candidates, all_attributes)
        evidence = [*metadata_evidence, *reviews]
        conflicts = conflict_evaluator.evaluate(candidates, evidence)
        decision = synthesizer.synthesize(query, catalog, evidence, conflicts)
        return BaselineRun(
            name,
            decision,
            len(selected_tools),
            0,
            int("review_search" in selected_tools),
        )

    if name == "B3_fixed_sequential_mas":
        spec_evidence = [
            row
            for candidate in candidates
            for row in spec_agent.extract(candidate, all_attributes)
        ]
        reviews, _ = review_agent.collect(candidates, all_attributes)
        evidence = [*metadata_evidence, *spec_evidence, *reviews]
        conflicts = conflict_evaluator.evaluate(candidates, evidence)
        decision = synthesizer.synthesize(query, catalog, evidence, conflicts)
        return BaselineRun(name, decision, 4, 1, 1)

    if name == "B4_query_adaptive_mas":
        difficulty = (
            len(query.hard_constraints)
            + 2 * len(query.vague_terms)
            + max(0, len(query.target_attributes) - 1)
        )
        if query.vague_terms:
            vague_targets = [
                attribute
                for attribute in query.target_attributes
                if attribute in SUPPORTED_ATTRIBUTES
                and query.uncertainty.get(attribute, 0.0) >= 0.5
            ]
            decision = synthesizer.synthesize(
                query,
                catalog,
                [],
                [],
                clarification_attributes=vague_targets,
            )
            if decision.action == "ASK_CLARIFICATION":
                return BaselineRun(name, decision, 1, 0, 0)
        if difficulty >= 2:
            spec_evidence = [
                row
                for candidate in candidates
                for row in spec_agent.extract(candidate, requested)
            ]
            reviews, _ = review_agent.collect(candidates, requested)
            evidence = [*metadata_evidence, *spec_evidence, *reviews]
            conflicts = conflict_evaluator.evaluate(candidates, evidence)
            decision = synthesizer.synthesize(query, catalog, evidence, conflicts)
            return BaselineRun(name, decision, 3, 1, 1)
        decision = synthesizer.synthesize(query, catalog, metadata_evidence, [])
        return BaselineRun(name, decision, 1, 0, 0)

    if name == "Ours_dual_conditioned_mas":
        orchestrator = AdaptiveOrchestrator(
            catalog_profiler=_NoOpProfiler(),
            review_agent=review_agent,
        )
        trace = orchestrator.run_states(query, catalog)
        return BaselineRun(
            name,
            trace.decision,
            trace.spec_agent_calls + trace.review_agent_calls,
            trace.spec_agent_calls,
            trace.review_agent_calls,
        )
    raise ValueError(f"Unknown baseline: {name}")


class _NoOpProfiler:
    """Placeholder: run_states consumes the already frozen catalog state."""


def _format_decision(decision: Decision | None) -> str:
    if decision is None:
        return "Route: direct single-LLM response; no retrieval or evidence tools."
    payload = asdict(decision)
    # Keep prompts bounded while retaining product identifiers and citations.
    return json.dumps(payload, ensure_ascii=True, default=str)[:12_000]


def _generate_response(
    client: OllamaClient,
    *,
    query: str,
    baseline: BaselineRun,
    catalog: CatalogState,
    user_answer: str | None = None,
) -> Generation:
    evidence_lines = []
    if baseline.decision is not None:
        for candidate in catalog.candidates[:5]:
            evidence_lines.extend(
                _source_quote({**row, "parent_asin": candidate.parent_asin})
                for row in candidate.spec_evidence[:3]
            )
        for recommendation in baseline.decision.recommendations[:5]:
            for row in recommendation.get("citations", [])[:3]:
                evidence_lines.append(_source_quote(row))
    prompt = (
        f"User request:\n{query}\n\n"
        f"System decision and route:\n{_format_decision(baseline.decision)}\n\n"
        f"Retrieved evidence:\n{chr(10).join(dict.fromkeys(evidence_lines))[:8000]}\n"
    )
    if user_answer is not None:
        prompt += f"\nSimulated user's answer to clarification:\n{user_answer}\n"
        prompt += (
            "\nProvide the best grounded next response. If the new answer still "
            "does not support a safe recommendation, disclose that."
        )
    else:
        prompt += "\nProvide the response to the user."
    return client.generate(prompt, system=SYSTEM_PROMPT)


def _plan_retrieval_tools(
    client: OllamaClient, query: str
) -> tuple[dict[str, Any], Generation]:
    generation = client.generate(
        (
            "Choose tools for this laptop request. catalog_search is required "
            "to retrieve matching products and their specifications. Add "
            "review_search only if buyer experience, subjective quality, or "
            "battery longevity matters to the request. Return JSON with keys "
            '"tools" (array of allowed tool names) and "reason" (short string). '
            'Allowed tools: "catalog_search", "review_search".\n\n'
            f"Request: {query}"
        ),
        system=(
            "You are a ReAct agent planning one retrieval step. Output only a "
            "valid JSON object and always include catalog_search."
        ),
        format_json=True,
    )
    try:
        plan = json.loads(generation.text)
    except json.JSONDecodeError as error:
        raise RuntimeError("Ollama returned invalid JSON for the B2 tool plan") from error
    tools = plan.get("tools") if isinstance(plan, dict) else None
    if (
        not isinstance(tools, list)
        or "catalog_search" not in tools
        or any(tool not in {"catalog_search", "review_search"} for tool in tools)
    ):
        raise RuntimeError(
            "B2 ReAct plan must select catalog_search and only supported tools"
        )
    reason = plan.get("reason", "")
    if not isinstance(reason, str):
        raise RuntimeError("B2 ReAct plan reason must be a string")
    return {"tools": list(dict.fromkeys(tools)), "reason": reason}, generation


def _expected_behavior(case: EvaluationCase, run: BaselineRun) -> bool | None:
    if run.decision is None:
        return None
    action = run.decision.action
    scenario = case.scenario_override
    if scenario == "S1":
        return action == "RECOMMEND" and run.spec_calls == 0 and run.review_calls == 0
    if scenario == "S2":
        return action == "ASK_CLARIFICATION"
    if scenario == "S3":
        return run.spec_calls > 0 or run.review_calls > 0
    if scenario == "S4":
        return action == "ABSTAIN_DISCLOSE"
    return False


def _serialize_case(case: EvaluationCase) -> dict[str, Any]:
    return {
        "query_id": case.record_id,
        "scenario": case.scenario_override,
        "query": case.text,
        "target_attributes": case.query_state.target_attributes,
        "hard_constraints": [asdict(item) for item in case.query_state.hard_constraints],
        "uncertainty": case.query_state.uncertainty,
        "candidate_parent_asins": [
            candidate.parent_asin for candidate in case.catalog_state.candidates
        ],
        "candidate_count": case.catalog_state.preliminary_candidate_count,
        "completeness": case.catalog_state.completeness,
        "entropy": case.catalog_state.entropy,
        "synthetic_catalog_mask": case.synthetic_catalog_mask,
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    queries_path = Path(args.queries)
    catalog_path = Path(args.catalog)
    reviews_path = Path(args.reviews)
    model_dir = Path(args.model_dir)
    _ensure_absa_dependencies()
    calibration, scenarios, profiler = _prepare_cases(
        queries_path,
        catalog_path,
        sample_size=args.sample_size,
        seed=args.seed,
        top_k=args.top_k,
    )
    profiler.close()
    cases = _limit_scenarios(scenarios, args.cases_per_stratum, args.seed + 2)
    parent_asins = {
        candidate.parent_asin
        for case in cases
        for candidate in case.catalog_state.candidates
    }
    raw_reviews, review_scan_count = _read_reviews_for_candidates(
        reviews_path,
        parent_asins,
        scan_limit=args.review_scan_limit,
    )
    review_evidence, review_model_tokens = _classify_reviews(raw_reviews, model_dir)
    review_agent = IndexedReviewAgent(review_evidence, review_scan_count)
    simulator = ReviewGroundedUserSimulator(review_evidence, seed=args.seed)
    client = OllamaClient(
        model=args.model,
        host=args.ollama_host,
        timeout_seconds=args.timeout,
    )

    query_records = {
        str(row.get("id")): row for row in _read_queries(queries_path)
    }
    outputs: list[dict[str, Any]] = []
    for case in cases:
        source = query_records.get(str(case.record_id), {})
        user = simulator.create(
            case.record_id,
            case.query_state.target_attributes,
            source.get("vague_annotations", []),
        )
        for baseline_name in BASELINES:
            started = time.perf_counter()
            tool_plan: dict[str, Any] | None = None
            planner_generation: Generation | None = None
            selected_tools: set[str] | None = None
            if baseline_name == "B2_single_agent_retrieval":
                tool_plan, planner_generation = _plan_retrieval_tools(
                    client, case.text
                )
                selected_tools = set(tool_plan["tools"])
            run = _route_baseline(
                baseline_name,
                case.query_state,
                case.catalog_state,
                review_agent,
                selected_tools=selected_tools,
            )
            first = _generate_response(
                client,
                query=case.text,
                baseline=run,
                catalog=case.catalog_state,
            )
            user_reply: dict[str, Any] | None = None
            followup: Generation | None = None
            if (
                run.decision is not None
                and run.decision.action == "ASK_CLARIFICATION"
            ):
                asked = (run.decision.clarification or {}).get("attributes", [])
                user_reply = simulator.respond(user, asked)
                if user_reply["responded"]:
                    followup = _generate_response(
                        client,
                        query=case.text,
                        baseline=run,
                        catalog=case.catalog_state,
                        user_answer=user_reply["answer"],
                    )
            latency = time.perf_counter() - started
            outputs.append(
                {
                    "query_id": case.record_id,
                    "scenario": case.scenario_override,
                    "baseline": baseline_name,
                    "query": case.text,
                    "decision": (
                        asdict(run.decision) if run.decision is not None else None
                    ),
                    "system_action": (
                        run.decision.action
                        if run.decision is not None
                        else "DIRECT_GENERATION"
                    ),
                    "simulated_user": {
                        "profile_id": user.profile_id,
                        "ambiguity_profile": user.ambiguity_profile,
                        "cooperation_profile": user.cooperation_profile,
                        "review_preferences": [asdict(row) for row in user.preferences],
                    },
                    "user_reply": user_reply,
                    "response": first.text,
                    "followup_response": followup.text if followup else None,
                    "tool_plan": tool_plan,
                    "tool_calls": run.tool_calls,
                    "spec_agent_calls": run.spec_calls,
                    "review_agent_calls": run.review_calls,
                    "agent_calls": run.spec_calls + run.review_calls,
                    "generation_calls": (
                        1
                        + int(planner_generation is not None)
                        + int(followup is not None)
                    ),
                    "prompt_tokens": first.prompt_tokens
                    + (planner_generation.prompt_tokens if planner_generation else 0)
                    + (followup.prompt_tokens if followup else 0),
                    "output_tokens": first.output_tokens
                    + (planner_generation.output_tokens if planner_generation else 0)
                    + (followup.output_tokens if followup else 0),
                    "latency_seconds": latency,
                    "expected_behavior_met": _expected_behavior(case, run),
                    "synthetic_catalog_mask": case.synthetic_catalog_mask,
                }
            )

    metrics: dict[str, Any] = {}
    for baseline_name in BASELINES:
        baseline_rows = [
            row for row in outputs if row["baseline"] == baseline_name
        ]
        scenario_metrics: dict[str, Any] = {}
        for scenario in ("S1", "S2", "S3", "S4"):
            group = [row for row in baseline_rows if row["scenario"] == scenario]
            scored = [
                row for row in group if row["expected_behavior_met"] is not None
            ]
            has_explicit_actions = baseline_name != "B1_single_llm"
            scenario_metrics[scenario] = {
                "count": len(group),
                "expected_behavior_rate": round(
                    sum(row["expected_behavior_met"] for row in scored) / len(scored), 4
                ) if scored else None,
                "expected_behavior_scored_cases": len(scored),
                "clarification_rate": round(
                    sum(row["system_action"] == "ASK_CLARIFICATION" for row in group)
                    / len(group), 4
                ) if group and has_explicit_actions else None,
                "abstain_disclose_rate": round(
                    sum(row["system_action"] == "ABSTAIN_DISCLOSE" for row in group)
                    / len(group), 4
                ) if group and has_explicit_actions else None,
                "review_agent_activation_rate": round(
                    sum(row["review_agent_calls"] > 0 for row in group) / len(group), 4
                ) if group else 0.0,
            }
        latencies = [row["latency_seconds"] for row in baseline_rows]
        scored = [
            row for row in baseline_rows if row["expected_behavior_met"] is not None
        ]
        metrics[baseline_name] = {
            "scenario_metrics": scenario_metrics,
            "expected_behavior_rate": round(
                sum(row["expected_behavior_met"] for row in scored) / len(scored),
                4,
            ) if scored else None,
            "expected_behavior_scored_cases": len(scored),
            "total_agent_calls": sum(row["agent_calls"] for row in baseline_rows),
            "total_tool_calls": sum(row["tool_calls"] for row in baseline_rows),
            "generation_calls": sum(row["generation_calls"] for row in baseline_rows),
            "prompt_tokens": sum(row["prompt_tokens"] for row in baseline_rows),
            "output_tokens": sum(row["output_tokens"] for row in baseline_rows),
            "mean_latency_seconds": round(statistics.mean(latencies), 4)
            if latencies
            else 0.0,
        }

    queries_hash = _sha256(queries_path)
    catalog_hash = _sha256(catalog_path)
    reviews_hash = _sha256(reviews_path)
    frozen_cases = [_serialize_case(case) for case in cases]
    frozen_digest = hashlib.sha256(
        json.dumps(frozen_cases, sort_keys=True, ensure_ascii=True).encode()
    ).hexdigest()
    report = {
        "phase": 3,
        "model": args.model,
        "ollama_host": args.ollama_host,
        "evaluation_mode": "local_ollama_generation",
        "seed": args.seed,
        "sample_size": args.sample_size,
        "calibration_queries_not_used_for_routing": len(calibration),
        "cases_per_stratum": args.cases_per_stratum,
        "case_count": len(cases),
        "strata_counts": dict(Counter(case.scenario_override for case in cases)),
        "thresholds": {
            "ambiguity_threshold": 0.5,
            "minimum_clarification_coverage": 0.5,
            "minimum_catalog_entropy": 0.3,
        },
        "frozen_data": {
            "queries": {"path": str(queries_path), "sha256": queries_hash},
            "catalog": {"path": str(catalog_path), "sha256": catalog_hash},
            "reviews": {"path": str(reviews_path), "sha256": reviews_hash},
            "case_manifest_sha256": frozen_digest,
        },
        "review_grounded_simulation": {
            "review_records_scanned": review_scan_count,
            "candidate_reviews_selected": len(raw_reviews),
            "aspect_evidence_rows": len(review_evidence),
            "setfit_input_tokens": review_model_tokens,
            "profile_ids_are_synthetic": True,
            "cooperation_profiles": ["cooperative", "guarded", "non_cooperative"],
            "preference_basis": (
                "Deterministically selected aspect/sentiment evidence from real "
                "Amazon review sentences. Simulated preferences are synthetic "
                "personas, not claims about the review authors."
            ),
        },
        "baselines": list(BASELINES),
        "metrics": metrics,
        "limitations": [
            "The held-out VACOS cases are frozen query/catalog conditions; S3 and S4 use controlled metadata masking.",
            "VACOS annotations do not provide ground-truth recommendation quality; expected-behavior rates measure policy routing only.",
            "B1 is intentionally retrieval-free and therefore cannot provide verified catalog citations.",
            "B4 routes only from query complexity and does not use catalog completeness to select its workflow.",
            "Ollama generation quality, unsupported claims, and citation correctness require human review; these automatic metrics do not establish factual accuracy.",
            "Simulated users are synthetic personas seeded by real review excerpts and do not replace participants or a validated user simulator from cited studies.",
        ],
        "frozen_cases": frozen_cases,
        "results": outputs,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 3 baselines with review-grounded simulated users and local Ollama."
    )
    parser.add_argument(
        "--queries", type=Path, default=Path("dataset/processed/vacos_laptops_bio.jsonl")
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
        "--model-dir", type=Path, default=Path("models/setfit-absa-laptops")
    )
    parser.add_argument("--model", default="llama3:latest")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--sample-size", type=int, default=160)
    parser.add_argument("--cases-per-stratum", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--review-scan-limit", type=int, default=50_000)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/phase3"))
    args = parser.parse_args()
    if args.sample_size < 4 or args.sample_size > 240 or args.sample_size % 4:
        parser.error("--sample-size must be a multiple of four between 4 and 240")
    if args.cases_per_stratum < 1:
        parser.error("--cases-per-stratum must be positive")
    if args.top_k < 1 or args.review_scan_limit < 1:
        parser.error("--top-k and --review-scan-limit must be positive")

    report = run_benchmark(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "phase3_baseline_results.json"
    csv_path = args.output_dir / "phase3_baseline_summary.csv"
    frozen_path = args.output_dir / "phase3_frozen_cases.jsonl"
    json_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    with frozen_path.open("w", encoding="utf-8", newline="\n") as destination:
        for case in report["frozen_cases"]:
            destination.write(json.dumps(case, ensure_ascii=True) + "\n")
    with csv_path.open("w", encoding="utf-8", newline="") as destination:
        fields = [
            "baseline",
            "scenario",
            "query_id",
            "system_action",
            "expected_behavior_met",
            "tool_calls",
            "agent_calls",
            "generation_calls",
            "prompt_tokens",
            "output_tokens",
            "latency_seconds",
        ]
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for row in report["results"]:
            writer.writerow({field: row.get(field) for field in fields})
    print(
        json.dumps(
            {
                "case_count": report["case_count"],
                "strata_counts": report["strata_counts"],
                "baselines": report["baselines"],
                "metrics": report["metrics"],
                "output_json": str(json_path),
                "output_csv": str(csv_path),
                "frozen_cases": str(frozen_path),
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

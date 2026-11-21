from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scripts.phase2.catalog_profiler import CatalogProfiler
from scripts.phase2.evidence_agents import ReviewEvidenceAgent
from scripts.phase2.orchestrator import AdaptiveOrchestrator


def _jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the offline dual-conditioned laptop recommendation pipeline."
    )
    parser.add_argument("query", help="Natural-language laptop query")
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
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--review-scan-limit", type=int, default=50_000)
    parser.add_argument("--disable-review-agent", action="store_true")
    args = parser.parse_args()

    review_agent = None
    if not args.disable_review_agent:
        review_agent = ReviewEvidenceAgent(
            args.reviews,
            model_dir=args.model_dir,
            max_source_records=args.review_scan_limit,
        )
    profiler = CatalogProfiler(args.catalog, max_candidates=args.top_k)
    try:
        orchestrator = AdaptiveOrchestrator(
            profiler,
            review_agent=review_agent,
        )
        trace = orchestrator.run(args.query)
        print(json.dumps(asdict(trace), ensure_ascii=True, indent=2, default=_jsonable))
    finally:
        profiler.close()


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from scripts.phase1.review_absa_agent import ReviewABSAAgent
from scripts.phase2.catalog_profiler import _extract_specs, _iter_records
from scripts.phase2.types import CandidateProfile


class SpecDescriptionAgent:
    """Recover attribute evidence from metadata descriptions and detail fields."""

    def extract(
        self, candidate: CandidateProfile, attributes: Iterable[str]
    ) -> list[dict[str, Any]]:
        requested = set(attributes)
        specs, evidence = _extract_specs(candidate.product)
        return [
            {**row, "parent_asin": candidate.parent_asin}
            for row in evidence
            if row["attribute"] in requested and specs.get(row["attribute"])
        ]


class ReviewEvidenceAgent:
    """Run the Phase 1 ABSA agent on a bounded scan of candidate laptop reviews."""

    def __init__(
        self,
        review_path: str | Path,
        *,
        classifier: Any | None = None,
        model_dir: str | Path = "models/setfit-absa-laptops",
        max_source_records: int = 50_000,
        max_reviews_per_product: int = 6,
        inference_batch_size: int = 64,
    ) -> None:
        if max_source_records < 1 or max_reviews_per_product < 1:
            raise ValueError("review scan and per-product limits must be positive")
        self.review_path = Path(review_path)
        self.classifier = classifier
        self.model_dir = Path(model_dir)
        self.max_source_records = max_source_records
        self.max_reviews_per_product = max_reviews_per_product
        self.inference_batch_size = inference_batch_size
        self.model_input_tokens = 0

    def collect(
        self,
        candidates: list[CandidateProfile],
        attributes: set[str],
    ) -> tuple[list[dict[str, Any]], int]:
        parent_asins = {candidate.parent_asin for candidate in candidates}
        if not parent_asins or not attributes:
            return [], 0
        reviews: list[dict[str, Any]] = []
        counts = {parent: 0 for parent in parent_asins}
        scanned = 0
        for review in _iter_records(self.review_path):
            scanned += 1
            parent = review.get("parent_asin")
            if parent in counts and counts[parent] < self.max_reviews_per_product:
                reviews.append(review)
                counts[parent] += 1
            if scanned >= self.max_source_records:
                break
            if all(count >= self.max_reviews_per_product for count in counts.values()):
                break

        if not reviews:
            return [], scanned
        if self.classifier is None:
            self.classifier = ReviewABSAAgent(model_dir=self.model_dir)
        aspect_categories = {
            "display", "ram", "battery", "weight_portability", "storage"
        }
        relevant = attributes & aspect_categories
        evidence: list[dict[str, Any]] = []
        for rows in self.classifier.analyze_reviews(
            reviews, batch_size=self.inference_batch_size
        ):
            evidence.extend(
                row for row in rows
                if row["aspect_category"] in relevant
            )
        self.model_input_tokens = int(
            getattr(self.classifier, "input_token_count", self.model_input_tokens)
        )
        return evidence, scanned

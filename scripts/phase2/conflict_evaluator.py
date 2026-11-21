from __future__ import annotations

from collections import defaultdict
from typing import Any

from scripts.phase2.types import CandidateProfile, Conflict


class ConflictEvaluator:
    """Detect review/spec experiential drift and disagreement across review variants."""

    def __init__(self, *, min_reviews: int = 3, negative_ratio_threshold: float = 0.67) -> None:
        if min_reviews < 2:
            raise ValueError("min_reviews must be at least two")
        if not 0.5 < negative_ratio_threshold <= 1:
            raise ValueError("negative_ratio_threshold must be in (0.5, 1]")
        self.min_reviews = min_reviews
        self.negative_ratio_threshold = negative_ratio_threshold

    def evaluate(
        self,
        candidates: list[CandidateProfile],
        evidence: list[dict[str, Any]],
    ) -> list[Conflict]:
        products = {candidate.parent_asin: candidate for candidate in candidates}
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        variants: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in evidence:
            parent = str(row.get("parent_asin") or "")
            attribute = str(row.get("aspect_category") or row.get("attribute") or "")
            if parent and attribute:
                grouped[(parent, attribute)].append(row)
                asin = str(row.get("asin") or "")
                if asin:
                    variants[(parent, attribute, asin)].append(row)

        conflicts: list[Conflict] = []
        for (parent, attribute), rows in grouped.items():
            product = products.get(parent)
            sentiments = [str(row.get("sentiment", "")).casefold() for row in rows]
            negative_count = sentiments.count("negative")
            positive_count = sentiments.count("positive")
            if (
                product is not None
                and product.specs.get(attribute)
                and len(rows) >= self.min_reviews
                and negative_count / len(rows) >= self.negative_ratio_threshold
            ):
                conflicts.append(
                    Conflict(
                        kind="EXPERIENTIAL_DRIFT",
                        parent_asin=parent,
                        attribute=attribute,
                        summary=(
                            f"Metadata lists {product.specs[attribute]!r}, while "
                            f"{negative_count}/{len(rows)} aspect reviews are negative."
                        ),
                        evidence=rows,
                        confidence=round(negative_count / len(rows), 3),
                    )
                )

            by_variant = [
                rows_for_variant
                for (variant_parent, variant_attribute, _), rows_for_variant in variants.items()
                if variant_parent == parent and variant_attribute == attribute
                and len(rows_for_variant) >= self.min_reviews
            ]
            if len(by_variant) >= 2:
                polarities = []
                selected_rows: list[dict[str, Any]] = []
                for variant_rows in by_variant:
                    labels = [str(row.get("sentiment", "")).casefold() for row in variant_rows]
                    pos = labels.count("positive") / len(labels)
                    neg = labels.count("negative") / len(labels)
                    polarities.append("positive" if pos > neg else "negative" if neg > pos else "mixed")
                    selected_rows.extend(variant_rows)
                if "positive" in polarities and "negative" in polarities:
                    conflicts.append(
                        Conflict(
                            kind="VARIANT_CONFLICT",
                            parent_asin=parent,
                            attribute=attribute,
                            summary="Review sentiment differs by asin variant for this attribute.",
                            evidence=selected_rows,
                            confidence=0.75,
                        )
                    )
        return conflicts

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any


ASPECT_TO_ATTRIBUTE = {
    "display": "display",
    "ram": "ram",
    "battery": "battery",
    "weight_portability": "weight_portability",
}


@dataclass(frozen=True)
class ReviewPreference:
    attribute: str
    sentiment: str
    quote: str
    parent_asin: str
    asin: str | None
    source: str


@dataclass(frozen=True)
class SimulatedUser:
    profile_id: str
    ambiguity_profile: str
    cooperation_profile: str
    preferences: tuple[ReviewPreference, ...]


class ReviewGroundedUserSimulator:
    """Create reproducible preference personas from real ABSA review evidence."""

    def __init__(self, evidence: list[dict[str, Any]], *, seed: int = 42) -> None:
        self.seed = seed
        self._by_attribute: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in evidence:
            attribute = ASPECT_TO_ATTRIBUTE.get(str(row.get("aspect_category", "")))
            if (
                attribute
                and row.get("sentiment") in {"positive", "negative", "neutral", "conflict"}
                and isinstance(row.get("sentence_text"), str)
                and row.get("parent_asin")
            ):
                self._by_attribute[attribute].append(row)

    def create(
        self,
        query_id: Any,
        query_attributes: list[str],
        vague_annotations: list[dict[str, Any]],
    ) -> SimulatedUser:
        identity = str(query_id)
        rng = random.Random(f"{self.seed}:{identity}")
        ambiguity_profile = "ambiguous" if vague_annotations else "explicit"
        cooperation_profile = rng.choice(
            ["cooperative", "guarded", "non_cooperative"]
        )
        preferences: list[ReviewPreference] = []
        for attribute in dict.fromkeys(query_attributes):
            options = self._by_attribute.get(attribute, [])
            if not options:
                continue
            row = rng.choice(options)
            preferences.append(
                ReviewPreference(
                    attribute=attribute,
                    sentiment=str(row["sentiment"]),
                    quote=str(row["sentence_text"]),
                    parent_asin=str(row["parent_asin"]),
                    asin=str(row["asin"]) if row.get("asin") else None,
                    source=str(row.get("source", "amazon_review")),
                )
            )
        return SimulatedUser(
            profile_id=f"vacos-{identity}",
            ambiguity_profile=ambiguity_profile,
            cooperation_profile=cooperation_profile,
            preferences=tuple(preferences),
        )

    def respond(
        self, user: SimulatedUser, asked_attributes: list[str]
    ) -> dict[str, Any]:
        if user.cooperation_profile == "non_cooperative":
            return {
                "responded": False,
                "answer": None,
                "reason": "simulated user declines to answer follow-up questions",
                "supporting_reviews": [],
            }
        preferences = {
            preference.attribute: preference for preference in user.preferences
        }
        available = [
            preferences[attribute]
            for attribute in asked_attributes
            if attribute in preferences
        ]
        if not available:
            return {
                "responded": False,
                "answer": None,
                "reason": "no review-grounded preference for the asked attribute",
                "supporting_reviews": [],
            }
        answer_parts = []
        if user.cooperation_profile == "cooperative":
            for preference in available:
                direction = (
                    "prioritize a similar positive experience"
                    if preference.sentiment == "positive"
                    else "avoid the issue described"
                    if preference.sentiment == "negative"
                    else "consider this experience as one factor"
                )
                answer_parts.append(f"For {preference.attribute}, I would {direction}.")
        else:
            answer_parts.append("I am not sure; please show me a few options.")
        return {
            "responded": True,
            "answer": " ".join(answer_parts),
            "reason": user.cooperation_profile,
            "supporting_reviews": [
                {
                    "attribute": preference.attribute,
                    "sentiment": preference.sentiment,
                    "parent_asin": preference.parent_asin,
                    "asin": preference.asin,
                    "source": preference.source,
                }
                for preference in available
            ],
        }


def stable_dataset_id(dataset_hash: str, record_id: Any) -> str:
    digest = hashlib.sha256(f"{dataset_hash}:{record_id}".encode()).hexdigest()[:16]
    return f"sim-{digest}"

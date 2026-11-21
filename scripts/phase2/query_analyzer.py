from __future__ import annotations

import re
from collections.abc import Iterable

from scripts.phase2.types import HardConstraint, QueryState


ATTRIBUTE_ALIASES: dict[str, tuple[str, ...]] = {
    "display": ("display", "screen", "monitor", "resolution", "touchscreen", "oled"),
    "ram": ("ram", "memory", "ddr4", "ddr5"),
    "battery": ("battery", "charge", "charging", "battery life", "hours"),
    "weight_portability": (
        "weight",
        "weighs",
        "lightweight",
        "portable",
        "portability",
        "heavy",
        "light",
        "compact",
        "small",
        "pound",
        "pounds",
        "lb",
        "lbs",
        "ounce",
        "ounces",
        "oz",
        "kg",
        "kilogram",
        "kilograms",
        "gram",
        "grams",
    ),
    "storage": ("storage", "ssd", "hard drive", "hard disk", "hdd", "emmc"),
    "performance": ("processor", "cpu", "gpu", "graphics", "gaming", "fast"),
    "operating_system": ("operating system", "windows", "macos", "linux", "chromeos"),
    "price": ("price", "cost", "budget", "affordable", "cheap", "$"),
}
SUPPORTED_ATTRIBUTES = {
    "display",
    "ram",
    "battery",
    "weight_portability",
    "storage",
    "price",
}

_VAGUE_TERMS = (
    "good",
    "great",
    "best",
    "nice",
    "better",
    "enough",
    "reasonable",
    "affordable",
    "cheap",
    "fast",
    "powerful",
    "long",
    "small",
    "light",
    "lightweight",
    "compact",
    "sleek",
    "not too",
    "a lot",
)
_STOP_WORDS = {
    "a", "an", "and", "are", "at", "below", "for", "find", "get", "good", "great",
    "i", "in", "is", "it", "laptop", "laptops", "least", "less", "maximum", "max",
    "me", "minimum", "min", "more", "my", "no", "of", "on", "or", "please",
    "the", "than", "that", "to", "under", "up", "want", "with", "would",
    "gb", "tb", "inch", "inches",
}
_NON_RETRIEVAL_TERMS = set(_VAGUE_TERMS) | {
    "experience", "experiences", "review", "reviews", "want", "looking",
    "recommend", "recommendation", "prefer", "please",
}

_OPERATOR_PATTERNS = (
    (re.compile(r"\b(?:at least|minimum|min|no less than|over|above)\b", re.I), "gte"),
    (re.compile(r"\b(?:at most|maximum|max|no more than|under|below|less than|up to)\b", re.I), "lte"),
)


def _operator_before(text: str, position: int) -> str:
    prefix = text[max(0, position - 32):position]
    matches = [
        (match.end(), operator)
        for pattern, operator in _OPERATOR_PATTERNS
        for match in pattern.finditer(prefix)
    ]
    return max(matches, default=(0, "eq"))[1]


def _extract_constraints(text: str) -> list[HardConstraint]:
    constraints: list[HardConstraint] = []
    patterns = (
        (
            "ram",
            re.compile(r"(?P<number>\d+(?:\.\d+)?)\s*(?P<unit>gb|tb)\s*(?:of\s*)?(?:ram|memory)\b", re.I),
        ),
        (
            "storage",
            re.compile(r"(?P<number>\d+(?:\.\d+)?)\s*(?P<unit>gb|tb)\s*(?:ssd|hdd|emmc|storage|hard\s*drive|hard\s*disk)\b", re.I),
        ),
        (
            "display",
            re.compile(
                r"(?P<number>\d+(?:\.\d+)?)\s*-?\s*"
                r"(?P<unit>inch(?:es)?|in\.|\")",
                re.I,
            ),
        ),
        (
            "price",
            re.compile(r"\$\s*(?P<number>\d+(?:,\d{3})*(?:\.\d{1,2})?)", re.I),
        ),
        (
            "battery",
            re.compile(
                r"(?P<number>\d+(?:\.\d+)?)\s*(?P<unit>hours?|hrs?)"
                r"(?:\s+of)?\s+(?:battery\s*)?(?:life|runtime)\b",
                re.I,
            ),
        ),
        (
            "weight_portability",
            re.compile(
                r"(?P<number>\d+(?:\.\d+)?)\s*(?P<unit>pounds?|lbs?|ounces?|oz|"
                r"kilograms?|kg|grams?|g)\b",
                re.I,
            ),
        ),
    )
    for attribute, pattern in patterns:
        for match in pattern.finditer(text):
            number = float(match.group("number").replace(",", ""))
            unit = match.groupdict().get("unit")
            if unit:
                folded_unit = unit.casefold()
                if folded_unit == "tb":
                    number *= 1024
                    unit = "GB"
                elif attribute in {"ram", "storage"}:
                    unit = "GB"
                elif attribute == "display":
                    unit = "IN"
                elif attribute == "battery":
                    unit = "HOUR"
                elif attribute == "weight_portability":
                    weight_in_grams = {
                        "pound": 453.59237, "pounds": 453.59237,
                        "lb": 453.59237, "lbs": 453.59237,
                        "ounce": 28.349523125, "ounces": 28.349523125,
                        "oz": 28.349523125, "kilogram": 1000.0,
                        "kilograms": 1000.0, "kg": 1000.0,
                        "gram": 1.0, "grams": 1.0, "g": 1.0,
                    }
                    number *= weight_in_grams[folded_unit]
                    unit = "G"
            constraints.append(
                HardConstraint(
                    attribute=attribute,
                    operator=_operator_before(text, match.start()),
                    value=number,
                    unit=unit.upper() if unit else None,
                    source_text=match.group(0),
                )
            )
    return constraints


def _find_terms(text: str, terms: Iterable[str]) -> list[str]:
    return [
        term
        for term in terms
        if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text, re.IGNORECASE)
    ]


class QueryAnalyzer:
    """Extract explicit product criteria and estimate attribute-level vagueness."""

    def analyze(self, text: str) -> QueryState:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("query must be a non-empty string")
        constraints = _extract_constraints(text)
        lower = text.casefold()
        targets = [
            attribute
            for attribute, aliases in ATTRIBUTE_ALIASES.items()
            if _find_terms(lower, aliases)
        ]
        for constraint in constraints:
            if constraint.attribute not in targets:
                targets.append(constraint.attribute)

        vague_terms = _find_terms(lower, _VAGUE_TERMS)
        uncertainty: dict[str, float] = {}
        for attribute in targets:
            aliases = ATTRIBUTE_ALIASES.get(attribute, ())
            positions = [
                match.start()
                for alias in aliases
                for match in re.finditer(re.escape(alias), lower)
            ]
            near_vague = any(
                abs(vague_match.start() - position) <= 24
                for term in vague_terms
                for vague_match in re.finditer(re.escape(term), lower)
                for position in positions
            )
            has_explicit_constraint = any(
                item.attribute == attribute for item in constraints
            )
            if near_vague:
                score = 0.65
            elif has_explicit_constraint:
                score = 0.0
            else:
                score = 0.35
            uncertainty[attribute] = score

        return QueryState(
            text=text.strip(),
            hard_constraints=constraints,
            target_attributes=targets,
            uncertainty=uncertainty,
            vague_terms=vague_terms,
        )

    @staticmethod
    def search_terms(text: str) -> list[str]:
        terms = re.findall(r"[a-z0-9]+", text.casefold())
        content_terms = [
                term for term in terms
                if (
                    term not in _STOP_WORDS
                    and term not in _NON_RETRIEVAL_TERMS
                    and len(term) > 1
                    and not term.isdigit()
                )
        ]
        lower = text.casefold()
        attribute_terms = [
                token
                for aliases in ATTRIBUTE_ALIASES.values()
                for alias in aliases
                if re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", lower)
                for token in re.findall(r"[a-z0-9]+", alias)
        ]
        return list(dict.fromkeys([*content_terms, *attribute_terms]))

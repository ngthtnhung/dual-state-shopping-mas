from __future__ import annotations

import gzip
import heapq
import json
import math
import re
import sqlite3
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from scripts.phase2.query_analyzer import QueryAnalyzer, SUPPORTED_ATTRIBUTES
from scripts.phase2.types import (
    CandidateProfile,
    CatalogState,
    HardConstraint,
    QueryState,
)


_SPEC_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "display": (
        re.compile(r"(?P<value>\d+(?:\.\d+)?\s*(?:inches|inch|in\.|\"))", re.I),
        re.compile(r"(?P<value>\d{3,4}\s*[x×]\s*\d{3,4}(?:\s*(?:pixels|px))?)", re.I),
        re.compile(r"(?P<value>(?:oled|ips|lcd|fhd|uhd|4k)\s*(?:display|screen)?)", re.I),
    ),
    "ram": (
        re.compile(r"(?:ram|memory(?: installed size)?)\s*(?:is|:|-)?\s*(?P<value>\d+(?:\.\d+)?\s*(?:gb|tb)(?:\s*(?:ddr[345]|lpddr[345]))?)", re.I),
        re.compile(r"(?P<value>\d+(?:\.\d+)?\s*(?:gb|tb))\s*(?:of\s*)?(?:ram|memory)\b", re.I),
    ),
    "battery": (
        re.compile(r"(?P<value>\d+(?:\.\d+)?\s*(?:hours?|hrs?))\s*(?:of\s*)?(?:battery\s*)?(?:life|runtime)?", re.I),
        re.compile(r"(?P<value>\d+(?:\.\d+)?\s*(?:mah|wh|watt[- ]?hours?))", re.I),
    ),
    "weight_portability": (
        re.compile(r"(?:item\s*)?weight\s*(?:is|:|-)?\s*(?P<value>\d+(?:\.\d+)?\s*(?:pounds?|lbs?|ounces?|oz|kilograms?|kg|grams?|g))", re.I),
        re.compile(r"(?P<value>\d+(?:\.\d+)?\s*(?:pounds?|lbs?|ounces?|oz|kilograms?|kg|grams?|g))", re.I),
    ),
    "storage": (
        re.compile(r"(?:hard (?:drive|disk) size|storage|ssd|hdd|emmc)\s*(?:is|:|-)?\s*(?P<value>\d+(?:\.\d+)?\s*(?:gb|tb))", re.I),
        re.compile(r"(?P<value>\d+(?:\.\d+)?\s*(?:gb|tb))\s*(?:ssd|hdd|emmc|storage)\b", re.I),
    ),
}

_DETAIL_KEYS: dict[str, tuple[str, ...]] = {
    "display": ("screen size", "standing screen display size", "max screen resolution"),
    "ram": ("ram memory installed size", "computer memory type", "ram", "memory"),
    "battery": ("battery", "battery life"),
    "weight_portability": ("item weight", "weight"),
    "storage": ("hard disk size", "hard drive", "storage"),
}


def _iter_records(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix.casefold() == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            yield record


def _flatten(value: Any, prefix: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(value, str):
        yield prefix, value
    elif isinstance(value, dict):
        for key, nested in value.items():
            key_text = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten(nested, key_text)
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            yield from _flatten(nested, f"{prefix}[{index}]")
    elif value is not None and isinstance(value, (int, float)):
        yield prefix, str(value)


def _normalize_value(value: str) -> str:
    value = re.sub(r"\s+", " ", value.casefold()).strip()
    value = value.replace("inches", "inch").replace("lbs", "lb")
    return value


def _diversity_key(attribute: str, value: str) -> str:
    if attribute == "price":
        match = re.search(r"\d+(?:\.\d+)?", value)
        return f"{float(match.group(0)):g} usd" if match else _normalize_value(value)
    if attribute in {"ram", "storage"}:
        match = re.search(r"(\d+(?:\.\d+)?)\s*(tb|gb)", value, re.I)
        if match:
            number = float(match.group(1))
            if match.group(2).casefold() == "tb":
                number *= 1024
            return f"{number:g} gb"
    if attribute == "display":
        match = re.search(r"(\d+(?:\.\d+)?)\s*(?:inches|inch|in\.|\")", value, re.I)
        if match:
            return f"{float(match.group(1)):g} inch"
        match = re.search(r"\d{3,4}\s*[x×]\s*\d{3,4}", value, re.I)
        if match:
            return re.sub(r"\s+", "", match.group(0).casefold().replace("×", "x"))
    if attribute in {"battery", "weight_portability"}:
        match = re.search(
            r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|mah|wh|pounds?|lbs?|ounces?|oz|kilograms?|kg|grams?|g)\b",
            value,
            re.I,
        )
        if match:
            unit = match.group(2).casefold()
            unit = {"hours": "hour", "hrs": "hour", "lb": "pound", "lbs": "pound",
                    "pounds": "pound", "ounces": "ounce", "oz": "ounce",
                    "kilograms": "kg", "grams": "g"}.get(unit, unit)
            return f"{float(match.group(1)):g} {unit}"
    return _normalize_value(value.split(";")[0])


def _extract_specs(product: dict[str, Any]) -> tuple[dict[str, str], list[dict[str, Any]]]:
    fields = list(_flatten(product.get("details"), "details"))
    fields.extend(_flatten(product.get("features"), "features"))
    fields.extend(_flatten(product.get("description"), "description"))
    fields.extend(_flatten(product.get("title"), "title"))
    specs: dict[str, list[str]] = {}
    evidence: list[dict[str, Any]] = []
    details = product.get("details")
    if isinstance(details, dict):
        for attribute, keys in _DETAIL_KEYS.items():
            for key, value in details.items():
                if str(key).casefold() in keys:
                    value_text = str(value)
                    if value_text.strip():
                        specs.setdefault(attribute, []).append(value_text)
                        evidence.append(
                            {
                                "kind": "specification",
                                "attribute": attribute,
                                "source": f"metadata.details.{key}",
                                "quote": value_text,
                            }
                        )

    for field_name, text in fields:
        for attribute, patterns in _SPEC_PATTERNS.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    value = match.group("value")
                    if value:
                        specs.setdefault(attribute, []).append(value)
                        evidence.append(
                            {
                                "kind": "specification",
                                "attribute": attribute,
                                "source": f"metadata.{field_name}",
                                "quote": text[max(0, match.start() - 70):match.end() + 70],
                                "value": value,
                            }
                        )

    result = {
        attribute: "; ".join(dict.fromkeys(value.strip() for value in values if value.strip()))
        for attribute, values in specs.items()
    }
    price = product.get("price")
    if isinstance(price, (int, float)):
        result["price"] = f"{price:g} USD"
        evidence.append(
            {
                "kind": "specification",
                "attribute": "price",
                "source": "metadata.price",
                "quote": str(price),
                "value": str(price),
            }
        )
    return result, evidence


def _numeric_spec(candidate: CandidateProfile, attribute: str) -> float | None:
    if attribute == "price":
        value = candidate.product.get("price")
        return float(value) if isinstance(value, (int, float)) else None
    text = candidate.specs.get(attribute, "")
    if attribute in {"ram", "storage"}:
        match = re.search(r"(\d+(?:\.\d+)?)\s*(tb|gb)\b", text, re.I)
        if not match:
            return None
        number = float(match.group(1))
        return number * 1024 if match.group(2).casefold() == "tb" else number
    if attribute == "display":
        match = re.search(r"(\d+(?:\.\d+)?)\s*(?:inches|inch|in\.|\")", text, re.I)
        return float(match.group(1)) if match else None
    if attribute == "battery":
        match = re.search(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)\b", text, re.I)
        return float(match.group(1)) if match else None
    if attribute == "weight_portability":
        match = re.search(
            r"(\d+(?:\.\d+)?)\s*(pounds?|lbs?|ounces?|oz|kilograms?|kg|grams?|g)\b",
            text,
            re.I,
        )
        if not match:
            return None
        number = float(match.group(1))
        factors = {
            "pound": 453.59237, "pounds": 453.59237, "lb": 453.59237,
            "lbs": 453.59237, "ounce": 28.349523125, "ounces": 28.349523125,
            "oz": 28.349523125, "kilogram": 1000.0, "kilograms": 1000.0,
            "kg": 1000.0, "gram": 1.0, "grams": 1.0, "g": 1.0,
        }
        return number * factors[match.group(2).casefold()]
    return None


def _satisfies(candidate: CandidateProfile, constraint: HardConstraint) -> bool:
    value = _numeric_spec(candidate, constraint.attribute)
    if value is None:
        return False
    if constraint.operator == "gte":
        return value >= constraint.value
    if constraint.operator == "lte":
        return value <= constraint.value
    return math.isclose(value, constraint.value, rel_tol=0.03, abs_tol=0.1)


def _product_text(product: dict[str, Any]) -> str:
    return " ".join(value for _, value in _flatten(product)).casefold()


class CatalogProfiler:
    """Retrieve products and profile metadata coverage and value diversity."""

    def __init__(self, catalog_path: str | Path, *, max_candidates: int = 40) -> None:
        self.catalog_path = Path(catalog_path)
        if max_candidates < 1:
            raise ValueError("max_candidates must be greater than zero")
        self.max_candidates = max_candidates
        self._index_directory: tempfile.TemporaryDirectory[str] | None = None
        self._index: sqlite3.Connection | None = None
        self.index_build_seconds = 0.0

    def close(self) -> None:
        if self._index is not None:
            self._index.close()
            self._index = None
        if self._index_directory is not None:
            self._index_directory.cleanup()
            self._index_directory = None

    def __del__(self) -> None:
        self.close()

    def _ensure_index(self) -> sqlite3.Connection:
        if self._index is not None:
            return self._index
        started = time.perf_counter()
        if self.catalog_path.stat().st_size >= 16 * 1024 * 1024:
            self._index_directory = tempfile.TemporaryDirectory(prefix="phase2-catalog-")
            index_path = Path(self._index_directory.name) / "catalog.sqlite"
            database = sqlite3.connect(index_path)
        else:
            database = sqlite3.connect(":memory:")
        database.execute("PRAGMA journal_mode = OFF")
        database.execute("PRAGMA synchronous = OFF")
        database.execute(
            "CREATE TABLE products ("
            "id INTEGER PRIMARY KEY, parent_asin TEXT, product_json TEXT NOT NULL, "
            "specs_json TEXT NOT NULL, search_text TEXT NOT NULL, "
            "rating REAL, price REAL)"
        )
        database.execute(
            "CREATE VIRTUAL TABLE products_fts USING fts5(search_text, tokenize='unicode61')"
        )
        rows: list[tuple[Any, ...]] = []
        try:
            for product in _iter_records(self.catalog_path):
                specs, _ = _extract_specs(product)
                search_text = _product_text(product)
                rows.append(
                    (
                        product.get("parent_asin"),
                        json.dumps(product, ensure_ascii=True),
                        json.dumps(specs, ensure_ascii=True),
                        search_text,
                        product.get("average_rating"),
                        product.get("price") if isinstance(product.get("price"), (int, float)) else None,
                    )
                )
                if len(rows) >= 256:
                    self._insert_index_rows(database, rows)
                    rows.clear()
            if rows:
                self._insert_index_rows(database, rows)
            database.commit()
        except Exception:
            database.close()
            if self._index_directory is not None:
                self._index_directory.cleanup()
                self._index_directory = None
            raise
        self._index = database
        self.index_build_seconds = time.perf_counter() - started
        return database

    def prepare(self) -> None:
        """Build the reusable search index before timing repeated query runs."""
        self._ensure_index()

    @staticmethod
    def _insert_index_rows(
        database: sqlite3.Connection, rows: list[tuple[Any, ...]]
    ) -> None:
        database.executemany(
            "INSERT INTO products "
            "(parent_asin, product_json, specs_json, search_text, rating, price) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        last_id = database.execute("SELECT last_insert_rowid()").fetchone()[0]
        first_id = last_id - len(rows) + 1
        database.executemany(
            "INSERT INTO products_fts(rowid, search_text) VALUES (?, ?)",
            [
                (first_id + offset, row[3])
                for offset, row in enumerate(rows)
            ],
        )

    def retrieve(self, query: QueryState) -> CatalogState:
        terms = QueryAnalyzer.search_terms(query.text)
        if not terms:
            raise RuntimeError(
                "Catalog retrieval requires at least one recognized query term"
            )
        database = self._ensure_index()
        match_query = " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)
        match_count_sql = " + ".join(
            "CASE WHEN instr(p.search_text, ?) > 0 THEN 1 ELSE 0 END"
            for _ in terms
        )
        best: list[tuple[float, str, int]] = []
        matched_count = 0
        cursor = database.execute(
            "SELECT p.id, p.parent_asin, p.specs_json, p.rating, p.price, "
            f"({match_count_sql}) AS matched_terms "
            "FROM products_fts AS f "
            "JOIN products AS p ON p.id = f.rowid "
            "WHERE products_fts MATCH ?",
            [*terms, match_query],
        )
        for row_id, parent_asin, specs_json, rating, price, matched_terms in cursor:
            if not matched_terms:
                continue
            candidate = CandidateProfile(
                {"price": price, "parent_asin": parent_asin},
                json.loads(specs_json),
                [],
                0.0,
            )
            if not all(_satisfies(candidate, item) for item in query.hard_constraints):
                continue
            matched_count += 1
            lexical = matched_terms / len(terms) if terms else 0.0
            rating_score = float(rating) / 5 if isinstance(rating, (int, float)) else 0.0
            retrieval_score = lexical * 0.85 + rating_score * 0.15
            heap_item = (retrieval_score, str(parent_asin or ""), int(row_id))
            if len(best) < self.max_candidates:
                heapq.heappush(best, heap_item)
            elif heap_item > best[0]:
                heapq.heapreplace(best, heap_item)

        candidates: list[CandidateProfile] = []
        for retrieval_score, _, row_id in sorted(best, reverse=True):
            row = database.execute(
                "SELECT product_json FROM products WHERE id = ?", (row_id,)
            ).fetchone()
            if row is None:
                raise RuntimeError(f"Catalog index lost product row {row_id}")
            product = json.loads(row[0])
            specs, evidence = _extract_specs(product)
            candidates.append(CandidateProfile(product, specs, evidence, retrieval_score))
        attributes = set(query.target_attributes) & SUPPORTED_ATTRIBUTES
        attributes.update(item.attribute for item in query.hard_constraints)
        completeness: dict[str, float] = {}
        entropy: dict[str, float] = {}
        for attribute in attributes:
            present = [candidate.specs.get(attribute, "") for candidate in candidates]
            count = sum(bool(value) for value in present)
            completeness[attribute] = count / len(candidates) if candidates else 0.0
            values = [value for value in present if value]
            frequencies = Counter(
                _diversity_key(attribute, value) for value in values
            )
            if len(frequencies) <= 1:
                entropy[attribute] = 0.0
            else:
                total = sum(frequencies.values())
                raw_entropy = -sum(
                    (frequency / total) * math.log(frequency / total)
                    for frequency in frequencies.values()
                )
                entropy[attribute] = raw_entropy / math.log(len(frequencies))
        return CatalogState(candidates, matched_count, completeness, entropy)

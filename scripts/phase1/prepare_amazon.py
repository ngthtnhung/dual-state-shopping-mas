from __future__ import annotations

import argparse
import gzip
import io
import json
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, IO, Iterator


ATTRIBUTE_PATTERNS = {
    "display": re.compile(
        r"\b(?:screen|display|panel|resolution|refresh rate|oled|ips)\b",
        re.IGNORECASE,
    ),
    "ram": re.compile(
        r"\b(?:ram|memory|ddr[345]\w*|lpddr[345]\w*|sdram)\b",
        re.IGNORECASE,
    ),
    "battery": re.compile(
        r"\b(?:battery|batteries|mah|watt[- ]?hours?|wh)\b",
        re.IGNORECASE,
    ),
    "weight_portability": re.compile(
        r"\b(?:weight|weighs|lightweight|light weight|portable|portability|"
        r"lbs?|pounds?|ounces?|kg|grams?)\b",
        re.IGNORECASE,
    ),
}


def _open_text(path: Path) -> IO[str]:
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


@contextmanager
def _atomic_text_output(
    path: Path, *, compressed: bool = False
) -> Generator[io.TextIOWrapper, None, None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    try:
        if compressed:
            handle = io.TextIOWrapper(
                gzip.open(temporary_path, "wb"), encoding="utf-8"
            )
        else:
            handle = temporary_path.open("w", encoding="utf-8", newline="\n")
        with handle:
            yield handle
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with _open_text(path) as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object in {path} at line {line_number}")
            yield record


def _normalize_category(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def is_laptop_product(record: dict[str, Any]) -> bool:
    categories = record.get("categories")
    if not isinstance(categories, list):
        return False
    accepted = {"laptop", "laptops", "laptop computers"}
    return any(
        isinstance(category, str) and _normalize_category(category) in accepted
        for category in categories
    )


def _flatten_text(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, nested_value in value.items():
            yield str(key)
            yield from _flatten_text(nested_value)
    elif isinstance(value, list):
        for nested_value in value:
            yield from _flatten_text(nested_value)


def catalog_signals(record: dict[str, Any]) -> dict[str, bool]:
    source_text = " ".join(
        _flatten_text(
            {
                key: record.get(key)
                for key in ("title", "features", "description", "details")
            }
        )
    )
    return {
        attribute: bool(pattern.search(source_text))
        for attribute, pattern in ATTRIBUTE_PATTERNS.items()
    }


def prepare_amazon_data(
    metadata_path: Path,
    reviews_path: Path,
    output_dir: Path,
    *,
    max_metadata_records: int | None = None,
    max_review_records: int | None = None,
) -> dict[str, Any]:
    parent_asins: set[str] = set()
    covered_parent_asins = {name: set() for name in ATTRIBUTE_PATTERNS}
    metadata_records_read = 0
    metadata_records_selected = 0
    metadata_records_missing_parent = 0
    catalog_path = output_dir / "laptops_catalog.jsonl"

    with _atomic_text_output(catalog_path) as catalog_output:
        for record in iter_jsonl(metadata_path):
            if (
                max_metadata_records is not None
                and metadata_records_read >= max_metadata_records
            ):
                break
            metadata_records_read += 1
            if not is_laptop_product(record):
                continue
            parent_asin = record.get("parent_asin")
            if not isinstance(parent_asin, str) or not parent_asin:
                metadata_records_missing_parent += 1
                continue

            signals = catalog_signals(record)
            for attribute, present in signals.items():
                if present:
                    covered_parent_asins[attribute].add(parent_asin)

            if parent_asin in parent_asins:
                continue
            parent_asins.add(parent_asin)
            metadata_records_selected += 1
            catalog_output.write(json.dumps(record, ensure_ascii=False) + "\n")

    parent_asins_path = output_dir / "laptop_parent_asins.txt"
    with _atomic_text_output(parent_asins_path) as parent_output:
        for parent_asin in sorted(parent_asins):
            parent_output.write(parent_asin + "\n")

    reviews_path_out = output_dir / "laptops_reviews.jsonl.gz"
    reviews_read = 0
    reviews_selected = 0
    reviews_missing_parent = 0
    reviews_missing_asin = 0
    with _atomic_text_output(reviews_path_out, compressed=True) as review_output:
        for review in iter_jsonl(reviews_path):
            if (
                max_review_records is not None
                and reviews_read >= max_review_records
            ):
                break
            reviews_read += 1
            parent_asin = review.get("parent_asin")
            if not isinstance(parent_asin, str) or not parent_asin:
                reviews_missing_parent += 1
                continue
            if parent_asin not in parent_asins:
                continue
            if not review.get("asin"):
                reviews_missing_asin += 1
            review_output.write(json.dumps(review, ensure_ascii=False) + "\n")
            reviews_selected += 1

    product_count = len(parent_asins)
    coverage = {
        attribute: {
            "products_with_signal": len(covered_parent_asins[attribute]),
            "product_count": product_count,
            "coverage_percent": (
                round(100 * len(covered_parent_asins[attribute]) / product_count, 2)
                if product_count
                else 0.0
            ),
        }
        for attribute in ATTRIBUTE_PATTERNS
    }
    summary = {
        "selection_rule": (
            "An exact category path component equal to Laptop, Laptops, "
            "or Laptop Computers (case-insensitive)."
        ),
        "coverage_method": (
            "Keyword evidence in title, features, description, or details; "
            "this is metadata text coverage, not verified value extraction."
        ),
        "metadata_records_read": metadata_records_read,
        "unique_laptop_parent_asins": product_count,
        "catalog_rows_written": metadata_records_selected,
        "metadata_records_missing_parent_asin": metadata_records_missing_parent,
        "reviews_read": reviews_read,
        "reviews_linked_by_parent_asin": reviews_selected,
        "reviews_missing_parent_asin": reviews_missing_parent,
        "linked_reviews_missing_asin": reviews_missing_asin,
        "catalog_coverage": coverage,
        "outputs": {
            "catalog": str(catalog_path),
            "parent_asins": str(parent_asins_path),
            "reviews": str(reviews_path_out),
        },
    }
    summary_path = output_dir / "catalog_coverage.json"
    with _atomic_text_output(summary_path) as summary_output:
        json.dump(summary, summary_output, ensure_ascii=False, indent=2)
        summary_output.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter Amazon Reviews 2023 Electronics data to laptop products."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("dataset/amazon-review/meta_Electronics.jsonl"),
    )
    parser.add_argument(
        "--reviews",
        type=Path,
        default=Path("dataset/amazon-review/review_Electronics.jsonl.gz"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/processed/amazon-laptops"))
    parser.add_argument("--max-metadata-records", type=int)
    parser.add_argument("--max-review-records", type=int)
    args = parser.parse_args()

    summary = prepare_amazon_data(
        args.metadata,
        args.reviews,
        args.output_dir,
        max_metadata_records=args.max_metadata_records,
        max_review_records=args.max_review_records,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _bio_label(scheme: str, category: str) -> str:
    if scheme not in {"B", "I"}:
        raise ValueError(f"Unexpected BIO scheme: {scheme!r}")
    normalized_category = re.sub(r"[^A-Za-z0-9]+", "_", category).strip("_")
    if not normalized_category:
        normalized_category = "UNKNOWN"
    return f"{scheme}-{normalized_category}"


def prepare_query(record: dict[str, Any]) -> dict[str, Any]:
    tokens_with_offsets = record.get("text token")
    if not isinstance(tokens_with_offsets, list):
        raise ValueError(f"Query {record.get('ID')} has no token-offset list")
    tokens: list[str] = []
    offsets: list[int] = []
    for token_record in tokens_with_offsets:
        if (
            not isinstance(token_record, list)
            or len(token_record) != 2
            or not isinstance(token_record[0], int)
            or not isinstance(token_record[1], str)
        ):
            raise ValueError(f"Query {record.get('ID')} has an invalid token record")
        offsets.append(token_record[0])
        tokens.append(token_record[1])

    offset_to_index = {offset: index for index, offset in enumerate(offsets)}
    if len(offset_to_index) != len(offsets):
        raise ValueError(f"Query {record.get('ID')} has duplicate token offsets")

    key_fact_annotations = (record.get("key facts") or {}).get("annotations") or []
    bio_tags = ["O"] * len(tokens)
    for annotation in key_fact_annotations:
        offset = annotation.get("start index")
        if offset not in offset_to_index:
            raise ValueError(
                f"Query {record.get('ID')} key fact offset {offset!r} "
                "does not align with a token"
            )
        index = offset_to_index[offset]
        label = _bio_label(
            annotation.get("BIO scheme", ""),
            annotation.get("attribute category", ""),
        )
        if bio_tags[index] != "O":
            raise ValueError(
                f"Query {record.get('ID')} has overlapping key-fact labels "
                f"at token offset {offset}"
            )
        bio_tags[index] = label

    vague_annotations = (record.get("vague words") or {}).get("annotations") or []
    vague_bio_tags = ["O"] * len(tokens)
    text = record.get("text", "")
    for annotation in vague_annotations:
        offset = annotation.get("start index")
        word = annotation.get("word")
        if not isinstance(offset, int) or not isinstance(word, str) or not word:
            raise ValueError(f"Query {record.get('ID')} has an invalid vague-word annotation")
        if offset not in offset_to_index or text[offset : offset + len(word)].casefold() != word.casefold():
            raise ValueError(
                f"Query {record.get('ID')} vague word {word!r} at {offset} "
                "does not align with its text token"
            )
        span_end = offset + len(word)
        span_indices = [
            index
            for index, token_offset in enumerate(offsets)
            if offset <= token_offset < span_end
        ]
        if not span_indices or offsets[span_indices[0]] != offset:
            raise ValueError(
                f"Query {record.get('ID')} vague word {word!r} has no aligned tokens"
            )
        for position, index in enumerate(span_indices):
            if vague_bio_tags[index] != "O":
                raise ValueError(
                    f"Query {record.get('ID')} has overlapping vague-word labels "
                    f"at token offset {offsets[index]}"
                )
            vague_bio_tags[index] = "B-VAGUE" if position == 0 else "I-VAGUE"

    return {
        "id": record["ID"],
        "text": text,
        "tokens": tokens,
        "token_offsets": offsets,
        "bio_tags": bio_tags,
        "vague_bio_tags": vague_bio_tags,
        "key_fact_annotations": key_fact_annotations,
        "vague_annotations": vague_annotations,
        "negations": record.get("negations", []),
        "uncategorized_key_fact_count": sum(
            not annotation.get("attribute category")
            for annotation in key_fact_annotations
        ),
    }


def prepare_vacos_queries(input_path: Path, output_path: Path) -> dict[str, Any]:
    with input_path.open(encoding="utf-8") as source:
        records = json.load(source)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON array in {input_path}")

    laptop_records = [record for record in records if record.get("domain") == "laptop"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    bio_tag_counts: dict[str, int] = {}
    vague_query_count = 0
    uncategorized_key_fact_count = 0
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as destination:
            for record in laptop_records:
                prepared = prepare_query(record)
                for tag in prepared["bio_tags"]:
                    if tag != "O":
                        bio_tag_counts[tag] = bio_tag_counts.get(tag, 0) + 1
                uncategorized_key_fact_count += prepared[
                    "uncategorized_key_fact_count"
                ]
                if any(tag != "O" for tag in prepared["vague_bio_tags"]):
                    vague_query_count += 1
                destination.write(json.dumps(prepared, ensure_ascii=False) + "\n")
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    summary = {
        "source_records": len(records),
        "laptop_queries": len(laptop_records),
        "queries_with_vague_annotations": vague_query_count,
        "uncategorized_key_fact_annotations": uncategorized_key_fact_count,
        "bio_tag_counts": bio_tag_counts,
        "output": str(output_path),
        "source_of_bio_tags": "VACOS_NLQ_v2 key-fact BIO scheme annotations",
        "vague_bio_tags": "Derived from the annotated vague-word spans",
    }
    summary_path = output_path.with_name(f"{output_path.stem}.summary.json")
    summary["summary_output"] = str(summary_path)
    temporary_summary_path = summary_path.with_name(summary_path.name + ".tmp")
    try:
        with temporary_summary_path.open(
            "w", encoding="utf-8", newline="\n"
        ) as destination:
            json.dump(summary, destination, ensure_ascii=False, indent=2)
            destination.write("\n")
        temporary_summary_path.replace(summary_path)
    except Exception:
        temporary_summary_path.unlink(missing_ok=True)
        raise
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert VACOS_NLQ v2 laptop annotations to token-level BIO JSONL."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "dataset/chiir21_naturallanguagequeries/"
            "VACOS_NLQ_v2/VACOS_NLQ_data.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/processed/vacos_laptops_bio.jsonl"),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_vacos_queries(args.input, args.output),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

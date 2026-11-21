from __future__ import annotations

import argparse
import gzip
import json
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any


DEFAULT_ASPECT_DICTIONARY: dict[str, tuple[str, ...]] = {
    "display": (
        "display",
        "screen",
        "monitor",
        "panel",
        "resolution",
        "refresh rate",
        "brightness",
        "touchscreen",
        "oled",
        "ips",
    ),
    "ram": (
        "ram",
        "memory",
        "ddr3",
        "ddr4",
        "ddr5",
        "lpddr3",
        "lpddr4",
        "lpddr5",
    ),
    "battery": (
        "battery",
        "batteries",
        "battery life",
        "charge",
        "charging",
        "charger",
        "power",
        "mah",
        "wh",
        "watt hour",
        "watt-hour",
    ),
    "weight_portability": (
        "weight",
        "weighs",
        "heavy",
        "light",
        "lightweight",
        "portable",
        "portability",
        "pound",
        "pounds",
        "lb",
        "lbs",
        "ounce",
        "ounces",
        "oz",
        "kilogram",
        "kilograms",
        "kg",
        "gram",
        "grams",
    ),
}

_ABBREVIATION_END = re.compile(
    r"(?:\b(?:e\.g|i\.e|etc|vs|mr|mrs|ms|dr|prof|approx|inc|no)\.)$",
    re.IGNORECASE,
)


def _build_nlp() -> Any:
    try:
        import spacy
    except ImportError as error:
        raise RuntimeError(
            "spaCy is required. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from error

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    return nlp


def _sentence_ranges(text: str, nlp: Any) -> list[tuple[int, int]]:
    document = nlp(text)
    segments = list(document.sents)
    if not segments:
        return []

    ranges: list[tuple[int, int]] = []
    current_start = segments[0].start_char
    current_end = segments[0].end_char
    for segment in segments[1:]:
        preceding_text = text[current_start:current_end].rstrip()
        if _ABBREVIATION_END.search(preceding_text):
            current_end = segment.end_char
            continue
        ranges.append((current_start, current_end))
        current_start = segment.start_char
        current_end = segment.end_char
    ranges.append((current_start, current_end))
    return ranges


class ReviewABSAAgent:
    """Extract aspect spans from review sentences and classify their sentiment."""

    def __init__(
        self,
        model_dir: str | Path = "models/setfit-absa-laptops",
        *,
        sentiment_model: Any | None = None,
        nlp: Any | None = None,
        aspect_dictionary: Mapping[str, Sequence[str]] | None = None,
        batch_size: int = 128,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be greater than zero")
        try:
            from spacy.matcher import PhraseMatcher
        except ImportError as error:
            raise RuntimeError(
                "spaCy is required. Install dependencies with "
                "`python -m pip install -r requirements.txt`."
            ) from error

        if sentiment_model is None:
            try:
                from setfit import SetFitModel
            except ImportError as error:
                raise RuntimeError(
                    "SetFit is required. Install dependencies with "
                    "`python -m pip install -r requirements.txt`."
                ) from error
            sentiment_model = SetFitModel.from_pretrained(str(model_dir))

        self._model: Any = sentiment_model
        self.input_token_count = 0
        self._nlp = nlp if nlp is not None else _build_nlp()
        dictionary = (
            DEFAULT_ASPECT_DICTIONARY
            if aspect_dictionary is None
            else aspect_dictionary
        )
        if not dictionary:
            raise ValueError("aspect_dictionary must contain at least one aspect")

        self._matcher = PhraseMatcher(self._nlp.vocab, attr="LOWER")
        self._categories: dict[int, str] = {}
        for index, (category, aliases) in enumerate(dictionary.items()):
            unique_aliases = tuple(
                dict.fromkeys(alias.strip() for alias in aliases if alias.strip())
            )
            if not unique_aliases:
                raise ValueError(f"Aspect {category!r} must have at least one non-empty alias")
            label = f"ASPECT_{index}"
            self._matcher.add(label, [self._nlp.make_doc(alias) for alias in unique_aliases])
            self._categories[self._nlp.vocab.strings[label]] = category
        self._batch_size = batch_size

    def extract_spans(self, text: str) -> list[dict[str, Any]]:
        if not isinstance(text, str):
            raise TypeError("Review text must be a string")
        candidates: list[dict[str, Any]] = []
        for sentence_index, (start, end) in enumerate(_sentence_ranges(text, self._nlp)):
            sentence_text = text[start:end]
            sentence_doc = self._nlp.make_doc(sentence_text)
            if not sentence_doc:
                continue
            matches_by_category: dict[str, list[Any]] = {}
            for match_id, token_start, token_end in self._matcher(sentence_doc):
                category = self._categories[match_id]
                matches_by_category.setdefault(category, []).append(
                    sentence_doc[token_start:token_end]
                )

            for category, matches in matches_by_category.items():
                selected: list[Any] = []
                occupied_until = -1
                for span in sorted(
                    matches,
                    key=lambda item: (item.start, -(item.end - item.start)),
                ):
                    if span.start < occupied_until:
                        continue
                    selected.append(span)
                    occupied_until = span.end
                for span in selected:
                    candidates.append(
                        {
                            "aspect_category": category,
                            "aspect_span": span.text,
                            "sentence_text": sentence_text,
                            "sentence_index": sentence_index,
                            "span_start": start + span.start_char,
                            "span_end": start + span.end_char,
                            "sentence_start": start,
                            "sentence_end": end,
                        }
                    )
        return sorted(
            candidates,
            key=lambda item: (item["span_start"], item["span_end"]),
        )

    def analyze_review(self, review: Mapping[str, Any]) -> list[dict[str, Any]]:
        return next(self.analyze_reviews([review], batch_size=1))

    def analyze_reviews(
        self,
        reviews: Iterable[Mapping[str, Any]],
        *,
        batch_size: int | None = None,
    ) -> Iterator[list[dict[str, Any]]]:
        effective_batch_size = self._batch_size if batch_size is None else batch_size
        if effective_batch_size < 1:
            raise ValueError("batch_size must be greater than zero")

        batch: list[tuple[Mapping[str, Any], list[dict[str, Any]]]] = []
        for review in reviews:
            if not isinstance(review, Mapping):
                raise TypeError("Each review must be a mapping")
            text = review.get("text")
            if not isinstance(text, str):
                raise ValueError("Each review must have a string 'text' field")
            candidates = self.extract_spans(text)
            batch.append((review, candidates))
            if len(batch) >= effective_batch_size:
                yield from self._classify_batch(batch)
                batch = []
        if batch:
            yield from self._classify_batch(batch)

    def _classify_batch(
        self,
        batch: Sequence[tuple[Mapping[str, Any], list[dict[str, Any]]]],
    ) -> Iterator[list[dict[str, Any]]]:
        all_candidates = [candidate for _, candidates in batch for candidate in candidates]
        if not all_candidates:
            for _ in batch:
                yield []
            return

        model_inputs = [
            f"Review: {candidate['sentence_text']}\n"
            f"Aspect: {candidate['aspect_span']}"
            for candidate in all_candidates
        ]
        model_body = getattr(self._model, "model_body", None)
        tokenizer = getattr(model_body, "tokenizer", None)
        if tokenizer is not None:
            tokenized = tokenizer(
                model_inputs,
                truncation=True,
                max_length=getattr(model_body, "max_seq_length", None),
            )
            self.input_token_count += sum(
                sum(attention) for attention in tokenized["attention_mask"]
            )
        predictions = self._model.predict(model_inputs)
        if len(predictions) != len(all_candidates):
            raise RuntimeError(
                "SetFit returned a different number of predictions than aspect spans"
            )

        prediction_iter = iter(predictions)
        for review, candidates in batch:
            evidence: list[dict[str, Any]] = []
            for candidate in candidates:
                sentiment = str(next(prediction_iter))
                if sentiment not in {"positive", "negative", "neutral", "conflict"}:
                    raise ValueError(f"SetFit returned an unsupported label: {sentiment!r}")
                evidence.append(
                    {
                        **candidate,
                        "sentiment": sentiment,
                        "asin": review.get("asin"),
                        "parent_asin": review.get("parent_asin"),
                        "rating": review.get("rating"),
                        "review_title": review.get("title"),
                        "timestamp": review.get("timestamp"),
                    }
                )
            yield evidence


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            yield record


def _batches(
    records: Iterable[dict[str, Any]], batch_size: int
) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for record in records:
        batch.append(record)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _run_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Extract aspect evidence and classify sentiment in laptop reviews."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset/processed/amazon-laptops/laptops_reviews.jsonl.gz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/processed/amazon-laptops/review_aspect_evidence.jsonl"),
    )
    parser.add_argument(
        "--model-dir", type=Path, default=Path("models/setfit-absa-laptops")
    )
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    agent = ReviewABSAAgent(model_dir=args.model_dir, batch_size=args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output.with_name(args.output.name + ".tmp")
    review_count = 0
    evidence_count = 0
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as destination:
            for review_batch in _batches(_iter_jsonl(args.input), args.batch_size):
                evidence_batch = agent.analyze_reviews(
                    review_batch, batch_size=args.batch_size
                )
                for evidence_rows in evidence_batch:
                    review_count += 1
                    for evidence in evidence_rows:
                        destination.write(json.dumps(evidence, ensure_ascii=False) + "\n")
                        evidence_count += 1
        temporary_path.replace(args.output)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    print(
        json.dumps(
            {
                "reviews_processed": review_count,
                "aspect_evidence_rows": evidence_count,
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    _run_cli()

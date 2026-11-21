from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from scripts.phase1.prepare_amazon import (
    catalog_signals,
    is_laptop_product,
    prepare_amazon_data,
)
from scripts.phase1.prepare_vacos_queries import prepare_query
from scripts.phase1.train_setfit_absa import _split_indices, _training_text


class AmazonPreparationTests(unittest.TestCase):
    def test_laptop_filter_requires_a_laptop_category_component(self) -> None:
        self.assertTrue(
            is_laptop_product(
                {"categories": ["Electronics", "Computers & Tablets", "Laptop Computers"]}
            )
        )
        self.assertFalse(
            is_laptop_product(
                {"categories": ["Electronics", "Computers", "Laptop Decals"]}
            )
        )

    def test_coverage_signals_are_attribute_specific(self) -> None:
        signals = catalog_signals(
            {
                "features": ["16GB DDR5 RAM", "Battery: 70Wh"],
                "details": {"Item Weight": "1.5 kg"},
                "description": ["15-inch OLED display"],
            }
        )
        self.assertEqual(
            signals,
            {
                "display": True,
                "ram": True,
                "battery": True,
                "weight_portability": True,
            },
        )

    def test_reviews_join_by_parent_asin_and_keep_variant_asin(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            metadata_path = root / "meta.jsonl"
            reviews_path = root / "reviews.jsonl.gz"
            output_dir = root / "output"
            metadata = [
                {
                    "parent_asin": "PARENT-1",
                    "title": "Laptop",
                    "categories": ["Electronics", "Laptops"],
                    "features": ["16GB RAM"],
                },
                {
                    "parent_asin": "PARENT-2",
                    "title": "Laptop sleeve",
                    "categories": ["Electronics", "Laptop Accessories"],
                },
            ]
            metadata_path.write_text(
                "".join(json.dumps(row) + "\n" for row in metadata),
                encoding="utf-8",
            )
            with gzip.open(reviews_path, "wt", encoding="utf-8") as destination:
                destination.write(
                    json.dumps(
                        {
                            "parent_asin": "PARENT-1",
                            "asin": "VARIANT-1",
                            "text": "Good laptop",
                        }
                    )
                    + "\n"
                )
                destination.write(
                    json.dumps(
                        {
                            "parent_asin": "PARENT-2",
                            "asin": "VARIANT-2",
                            "text": "Sleeve review",
                        }
                    )
                    + "\n"
                )

            summary = prepare_amazon_data(metadata_path, reviews_path, output_dir)
            self.assertEqual(summary["unique_laptop_parent_asins"], 1)
            self.assertEqual(summary["reviews_linked_by_parent_asin"], 1)
            with gzip.open(
                output_dir / "laptops_reviews.jsonl.gz", "rt", encoding="utf-8"
            ) as source:
                review = json.loads(next(source))
            self.assertEqual(review["parent_asin"], "PARENT-1")
            self.assertEqual(review["asin"], "VARIANT-1")


class VacosPreparationTests(unittest.TestCase):
    def test_builds_fact_and_vagueness_bio_tags_from_offsets(self) -> None:
        record = {
            "ID": 1,
            "text": "Need 16GB RAM and lightweight design",
            "text token": [[0, "Need"], [5, "16GB"], [10, "RAM"], [14, "and"], [18, "lightweight"], [29, "design"]],
            "key facts": {
                "annotations": [
                    {
                        "start index": 5,
                        "BIO scheme": "B",
                        "attribute category": "hardware_core_ram",
                    },
                    {
                        "start index": 10,
                        "BIO scheme": "I",
                        "attribute category": "hardware_core_ram",
                    },
                ]
            },
            "vague words": {
                "annotations": [{"start index": 18, "word": "lightweight"}]
            },
            "negations": [],
        }
        prepared = prepare_query(record)
        self.assertEqual(prepared["bio_tags"], ["O", "B-hardware_core_ram", "I-hardware_core_ram", "O", "O", "O"])
        self.assertEqual(prepared["vague_bio_tags"], ["O", "O", "O", "O", "B-VAGUE", "O"])

    def test_missing_source_category_is_kept_as_unknown(self) -> None:
        record = {
            "ID": 2,
            "text": "Good laptop",
            "text token": [[0, "Good"], [5, "laptop"]],
            "key facts": {
                "annotations": [
                    {
                        "start index": 0,
                        "BIO scheme": "B",
                        "attribute category": "",
                    }
                ]
            },
            "vague words": {"annotations": []},
            "negations": [],
        }
        prepared = prepare_query(record)
        self.assertEqual(prepared["bio_tags"], ["B-UNKNOWN", "O"])
        self.assertEqual(prepared["uncategorized_key_fact_count"], 1)


class SetFitInputTests(unittest.TestCase):
    def test_training_input_conditions_sentiment_on_the_aspect(self) -> None:
        self.assertEqual(
            _training_text("Battery life is excellent.", "Battery life"),
            "Review: Battery life is excellent.\nAspect: Battery life",
        )

    def test_validation_split_keeps_aspects_from_each_sentence_together(self) -> None:
        source_texts = [
            f"sentence {sentence_id}"
            for sentence_id in range(20)
            for _ in range(3)
        ]
        labels = ["positive", "negative", "neutral"] * 20
        train_indices, validation_indices = _split_indices(
            labels, source_texts, seed=42
        )
        train_texts = {source_texts[index] for index in train_indices}
        validation_texts = {source_texts[index] for index in validation_indices}
        self.assertFalse(train_texts & validation_texts)
        self.assertGreaterEqual(len(validation_indices), 10)


if __name__ == "__main__":
    unittest.main()

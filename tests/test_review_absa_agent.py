from __future__ import annotations

import unittest
from typing import Any

from scripts.phase1.review_absa_agent import ReviewABSAAgent


class FakeSentimentModel:
    def __init__(self, labels: list[str] | None = None) -> None:
        self.labels = labels or ["positive"]
        self.inputs: list[str] = []

    def predict(self, texts: list[str]) -> list[str]:
        self.inputs.extend(texts)
        if len(self.labels) == 1:
            return self.labels * len(texts)
        return self.labels


class ReviewABSAAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            import spacy
        except ImportError as error:
            raise RuntimeError(
                "Install spaCy with `python -m pip install -r requirements.txt`"
            ) from error
        cls.nlp: Any = spacy.blank("en")
        cls.nlp.add_pipe("sentencizer")

    def make_agent(self, labels: list[str] | None = None) -> tuple[ReviewABSAAgent, FakeSentimentModel]:
        model = FakeSentimentModel(labels)
        return ReviewABSAAgent(sentiment_model=model, nlp=self.nlp), model

    def test_extracts_aspects_and_merges_common_abbreviation_boundaries(self) -> None:
        agent, _ = self.make_agent()
        text = (
            "Good battery life, e.g. 10 hours. "
            "The screen is bright; RAM is 16GB. "
            "It feels lightweight."
        )

        candidates = agent.extract_spans(text)

        self.assertEqual(
            [candidate["aspect_category"] for candidate in candidates],
            ["battery", "display", "ram", "weight_portability"],
        )
        self.assertEqual(candidates[0]["aspect_span"], "battery life")
        self.assertEqual(candidates[0]["sentence_text"], "Good battery life, e.g. 10 hours.")
        for candidate in candidates:
            self.assertEqual(
                text[candidate["span_start"] : candidate["span_end"]],
                candidate["aspect_span"],
            )

    def test_analyze_review_returns_linkable_evidence_and_predictions(self) -> None:
        agent, model = self.make_agent(["positive", "negative"])
        review = {
            "asin": "VARIANT-1",
            "parent_asin": "PARENT-1",
            "rating": 4.0,
            "title": "Good laptop",
            "timestamp": 123,
            "text": "Battery life is great. The screen is dim.",
        }

        evidence = agent.analyze_review(review)

        self.assertEqual([row["sentiment"] for row in evidence], ["positive", "negative"])
        self.assertEqual({row["asin"] for row in evidence}, {"VARIANT-1"})
        self.assertEqual({row["parent_asin"] for row in evidence}, {"PARENT-1"})
        self.assertEqual(len(model.inputs), 2)
        self.assertIn("aspect: battery", model.inputs[0].lower())

    def test_reviews_without_matching_aspects_produce_no_model_call(self) -> None:
        agent, model = self.make_agent()

        evidence = agent.analyze_review({"text": "The package arrived yesterday."})

        self.assertEqual(evidence, [])
        self.assertEqual(model.inputs, [])

    def test_batch_processing_preserves_review_order(self) -> None:
        agent, model = self.make_agent()
        reviews = [
            {"asin": "A", "text": "The display is sharp."},
            {"asin": "B", "text": "RAM is sufficient."},
            {"asin": "C", "text": "No aspect term here."},
        ]

        results = list(agent.analyze_reviews(reviews, batch_size=3))

        self.assertEqual([rows[0]["asin"] for rows in results[:2]], ["A", "B"])
        self.assertEqual(results[2], [])
        self.assertEqual(len(model.inputs), 2)

    def test_missing_review_text_is_reported(self) -> None:
        agent, _ = self.make_agent()
        with self.assertRaisesRegex(ValueError, "string 'text'"):
            agent.analyze_review({"asin": "NO-TEXT"})

    def test_empty_aspect_dictionary_is_rejected(self) -> None:
        model = FakeSentimentModel()
        with self.assertRaisesRegex(ValueError, "at least one aspect"):
            ReviewABSAAgent(
                sentiment_model=model,
                nlp=self.nlp,
                aspect_dictionary={},
            )

    def test_unsupported_model_label_is_reported(self) -> None:
        agent, _ = self.make_agent(["other"])
        with self.assertRaisesRegex(ValueError, "unsupported label"):
            agent.analyze_review({"text": "The screen is bright."})


if __name__ == "__main__":
    unittest.main()

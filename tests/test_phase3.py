from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.phase2.query_analyzer import QueryAnalyzer
from scripts.phase2.types import CandidateProfile, CatalogState
from scripts.phase3.benchmark import (
    BASELINES,
    IndexedReviewAgent,
    BaselineRun,
    _ensure_absa_dependencies,
    _expected_behavior,
    _prepare_cases,
    _plan_retrieval_tools,
    _route_baseline,
)
from scripts.phase3.ollama_client import Generation, OllamaClient
from scripts.phase3.simulated_users import ReviewGroundedUserSimulator


class Phase3Tests(unittest.TestCase):
    def test_only_local_ollama_hosts_are_allowed(self) -> None:
        with self.assertRaisesRegex(ValueError, "this machine"):
            OllamaClient(host="https://ollama.example")

    def test_missing_setfit_dependency_gives_active_interpreter_install_command(self) -> None:
        def find_spec(module: str) -> object | None:
            return None if module == "setfit" else object()

        with patch("scripts.phase3.benchmark.importlib.util.find_spec", side_effect=find_spec):
            with self.assertRaisesRegex(RuntimeError, "pip install -r requirements.txt"):
                _ensure_absa_dependencies()

    def test_catalog_index_is_closed_when_case_preparation_fails(self) -> None:
        class FakeProfiler:
            closed = False

            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def prepare(self) -> None:
                return None

            def close(self) -> None:
                self.closed = True

        fake_profiler = FakeProfiler()
        with (
            patch("scripts.phase3.benchmark._read_queries", return_value=[]),
            patch("scripts.phase3.benchmark._select_sample", return_value=[]),
            patch("scripts.phase3.benchmark.CatalogProfiler", return_value=fake_profiler),
            patch(
                "scripts.phase3.benchmark._split_calibration_evaluation",
                side_effect=ValueError("test failure"),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "test failure"):
                _prepare_cases(
                    queries_path=Path("queries.jsonl"),
                    catalog_path=Path("catalog.jsonl"),
                    sample_size=4,
                    seed=42,
                    top_k=10,
                )
        self.assertTrue(fake_profiler.closed)

    def test_local_ollama_generation_returns_text_tokens_and_latency(self) -> None:
        response_body = json.dumps(
            {
                "response": "A grounded answer.",
                "prompt_eval_count": 12,
                "eval_count": 5,
            }
        ).encode()

        class FakeResponse:
            def __enter__(self) -> FakeResponse:
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return response_body

        client = OllamaClient()
        with patch("scripts.phase3.ollama_client.urlopen", return_value=FakeResponse()):
            result = client.generate("prompt", system="system")
        self.assertEqual(result.text, "A grounded answer.")
        self.assertEqual(result.prompt_tokens, 12)
        self.assertEqual(result.output_tokens, 5)
        self.assertGreaterEqual(result.elapsed_seconds, 0)

    def test_react_tool_plan_requires_catalog_search(self) -> None:
        class FakeClient:
            def generate(self, prompt: str, *, system: str, format_json: bool) -> Generation:
                self.assert_format = format_json
                return Generation(
                    '{"tools":["catalog_search","review_search"],"reason":"Check battery experience."}',
                    10,
                    8,
                    0.1,
                )

        fake = FakeClient()
        plan, generation = _plan_retrieval_tools(fake, "laptop with good battery")
        self.assertTrue(fake.assert_format)
        self.assertEqual(plan["tools"], ["catalog_search", "review_search"])
        self.assertEqual(generation.output_tokens, 8)

        class InvalidClient:
            def generate(self, prompt: str, *, system: str, format_json: bool) -> Generation:
                return Generation('{"tools":["review_search"],"reason":"Reviews"}', 1, 1, 0.1)

        with self.assertRaisesRegex(RuntimeError, "must select catalog_search"):
            _plan_retrieval_tools(InvalidClient(), "any request")

    def test_review_grounded_persona_has_cooperative_and_noncooperative_paths(self) -> None:
        evidence = [
            {
                "aspect_category": "battery",
                "sentiment": "positive",
                "sentence_text": "The battery lasted through my workday.",
                "parent_asin": "P1",
                "asin": "V1",
                "source": "amazon_reviews_2023",
            }
        ]
        simulator = ReviewGroundedUserSimulator(evidence, seed=3)
        cooperative = simulator.create(
            "q-coop", ["battery"], [{"word": "good", "start index": 0}]
        )
        cooperative = type(cooperative)(
            cooperative.profile_id,
            cooperative.ambiguity_profile,
            "cooperative",
            cooperative.preferences,
        )
        reply = simulator.respond(cooperative, ["battery"])
        self.assertTrue(reply["responded"])
        self.assertIn("positive", reply["answer"])
        self.assertEqual(reply["supporting_reviews"][0]["parent_asin"], "P1")

        noncooperative = type(cooperative)(
            cooperative.profile_id,
            cooperative.ambiguity_profile,
            "non_cooperative",
            cooperative.preferences,
        )
        declined = simulator.respond(noncooperative, ["battery"])
        self.assertFalse(declined["responded"])

    def test_benchmark_defines_five_baselines_and_ours_uses_dual_state(self) -> None:
        self.assertEqual(len(BASELINES), 5)
        self.assertIsNone(_expected_behavior(None, BaselineRun("B1", None, 0, 0, 0)))
        query = QueryAnalyzer().analyze("good battery laptop")
        candidate = CandidateProfile(
            product={
                "parent_asin": "P1",
                "title": "Laptop",
                "details": {"Battery Life": "8 hours"},
            },
            specs={"battery": "8 hours"},
            spec_evidence=[
                {
                    "kind": "specification",
                    "attribute": "battery",
                    "source": "metadata.details.Battery Life",
                    "quote": "8 hours",
                    "parent_asin": "P1",
                }
            ],
            retrieval_score=1.0,
        )
        catalog = CatalogState(
            [candidate],
            1,
            {"battery": 1.0},
            {"battery": 0.8},
        )
        review_agent = IndexedReviewAgent([], 0)
        run = _route_baseline(
            "Ours_dual_conditioned_mas", query, catalog, review_agent
        )
        self.assertIsNotNone(run.decision)
        self.assertEqual(run.decision.action, "ASK_CLARIFICATION")


if __name__ == "__main__":
    unittest.main()

from dataclasses import dataclass
from unittest.mock import MagicMock, create_autospec

import pytest
from vllm import LLM

from track_llm_apis.sampling.analyze_bi import (
    bi_tv_statistic_vectorized,
    bi_two_sample_test,
)
from track_llm_apis.sampling.bi_phase_1 import (
    BIPhase1ModelResult,
)
from track_llm_apis.sampling.common import CompressedOutputRow, References
from track_llm_apis.sampling.vllm_sampling import vllm_inference_bi


@dataclass
class MockCompletionOutput:
    text: str
    token_ids: list[int]


@dataclass
class MockRequestOutput:
    outputs: list[MockCompletionOutput]
    prompt_token_ids: list[int]


def create_mock_llm(output_fn=None):
    """Create a mock vLLM instance that satisfies beartype and returns configurable outputs."""
    output_fn = output_fn or (lambda prompt: prompt[0] if prompt else "x")
    mock_llm = create_autospec(LLM, instance=True)

    # Create nested mock structure for llm_engine.model_config.model
    mock_llm.llm_engine = MagicMock()
    mock_llm.llm_engine.model_config.model = "mock-model"

    mock_llm._generate_calls = []
    token_id_counter = [0]
    token_to_id: dict[str, int] = {}

    def get_token_id(text: str) -> int:
        if text not in token_to_id:
            token_to_id[text] = token_id_counter[0]
            token_id_counter[0] += 1
        return token_to_id[text]

    mock_tokenizer = MagicMock()
    mock_tokenizer.decode.side_effect = lambda ids: next(
        t for t, tid in token_to_id.items() if tid == ids[0]
    )
    mock_llm.get_tokenizer.return_value = mock_tokenizer

    def mock_generate(prompts, sampling_params, use_tqdm=False):
        mock_llm._generate_calls.append(list(prompts))
        results = []
        for p in prompts:
            text = output_fn(p)
            token_id = get_token_id(text)
            results.append(
                MockRequestOutput(
                    outputs=[MockCompletionOutput(text=text, token_ids=[token_id])],
                    prompt_token_ids=[0] * len(p),
                )
            )
        return results

    mock_llm.generate.side_effect = mock_generate
    return mock_llm


class TestBIPhase1ModelResult:
    def test_add_result(self):
        result = BIPhase1ModelResult(model_name="test", bi_config={}, phase="1a")
        result.add_result("a", "x")
        result.add_result("a", "x")
        result.add_result("a", "y")
        assert result.results == {"a": {"x": 2, "y": 1}}

    def test_get_border_inputs(self):
        result = BIPhase1ModelResult(model_name="test", bi_config={}, phase="1a")
        result.add_result("a", "x")
        result.add_result("a", "y")
        result.add_result("b", "x")
        result.add_result("b", "x")
        result.add_result("c", "x")
        result.add_result("c", "y")
        result.add_result("c", "z")
        assert set(result.get_border_inputs()) == {"a", "c"}

    def test_get_border_input_ratio(self):
        result = BIPhase1ModelResult(model_name="test", bi_config={}, phase="1a")
        result.add_result("a", "x")
        result.add_result("a", "y")
        result.add_result("b", "x")
        assert result.get_border_input_ratio() == 0.5


def get_border_inputs(results: dict[str, dict[str, int]]) -> list[str]:
    """Return inputs that produced at least 2 different outputs."""
    return [inp for inp, outputs in results.items() if len(outputs) >= 2]


class TestVllmInferenceBi:
    def test_basic_inference(self):
        llm = create_mock_llm()
        inputs = ["inp1", "inp2", "inp3"]
        traffic = ["t1", "t2", "t3", "t4", "t5"]

        results = vllm_inference_bi(
            llm=llm,
            inputs=inputs,
            traffic_prompts=traffic,
            n_samples=2,
            temperature=1.0,
            input_fraction=0.5,
            batch_size=4,
            max_tokens=1,
        )

        assert set(results.counts.keys()) == set(inputs)
        for inp in inputs:
            total_samples = sum(results.counts[inp].values())
            assert total_samples == 2

    def test_batch_composition(self):
        """Verify batches contain correct mix of inputs and traffic."""
        llm = create_mock_llm()
        inputs = ["inp1", "inp2"]
        traffic = ["t1", "t2", "t3"]

        vllm_inference_bi(
            llm=llm,
            inputs=inputs,
            traffic_prompts=traffic,
            n_samples=1,
            temperature=1.0,
            input_fraction=0.5,
            batch_size=4,
            max_tokens=1,
        )

        for batch in llm._generate_calls:
            assert len(batch) == 4
            input_count = sum(1 for p in batch if p in inputs)
            traffic_count = sum(1 for p in batch if p in traffic)
            assert input_count == 2
            assert traffic_count == 2

    def test_border_input_detection(self):
        """Test that inputs producing multiple outputs are detected as border inputs."""
        call_count = {"inp1": 0, "inp2": 0}

        def varying_output(prompt):
            if prompt == "inp1":
                call_count["inp1"] += 1
                return "a" if call_count["inp1"] % 2 == 0 else "b"
            return "constant"

        llm = create_mock_llm(output_fn=varying_output)
        inputs = ["inp1", "inp2"]
        traffic = ["t1", "t2"]

        results = vllm_inference_bi(
            llm=llm,
            inputs=inputs,
            traffic_prompts=traffic,
            n_samples=4,
            temperature=1.0,
            input_fraction=0.5,
            batch_size=4,
            max_tokens=1,
        )

        border_inputs = get_border_inputs(results.counts)
        assert "inp1" in border_inputs
        assert "inp2" not in border_inputs

    def test_input_groups_with_many_inputs(self):
        """Test that inputs are correctly split into groups when there are many."""
        llm = create_mock_llm()
        inputs = [f"inp{i}" for i in range(10)]
        traffic = ["t1", "t2", "t3"]

        results = vllm_inference_bi(
            llm=llm,
            inputs=inputs,
            traffic_prompts=traffic,
            n_samples=2,
            temperature=1.0,
            input_fraction=0.5,
            batch_size=4,  # 2 inputs per batch
            max_tokens=1,
        )

        assert set(results.counts.keys()) == set(inputs)
        for inp in inputs:
            assert sum(results.counts[inp].values()) == 2


def make_rows(refs: References, prompt: str, outputs: list[str]) -> list[CompressedOutputRow]:
    """Helper to create CompressedOutputRow objects for testing."""
    return [
        CompressedOutputRow.from_values(
            references=refs,
            source=0,
            variant="test",
            prompt=(prompt, 1),
            text=(output, 1),
        )
        for output in outputs
    ]


class TestBiTwoSampleTest:
    def test_identical_distributions_tv_zero(self):
        """TV distance should be 0 when both samples have identical distributions."""
        refs = References()
        rows1 = {"p1": make_rows(refs, "p1", ["a", "a", "b"])}
        rows2 = {"p1": make_rows(refs, "p1", ["a", "a", "b"])}

        result = bi_two_sample_test(rows1, rows2, pvalue_b=0)
        assert result.statistic == pytest.approx(0.0)

    def test_completely_different_distributions_tv_one(self):
        """TV distance should be 1 when distributions have no overlap."""
        refs = References()
        rows1 = {"p1": make_rows(refs, "p1", ["a", "a", "a"])}
        rows2 = {"p1": make_rows(refs, "p1", ["b", "b", "b"])}

        result = bi_two_sample_test(rows1, rows2, pvalue_b=0)
        assert result.statistic == pytest.approx(1.0)

    def test_partial_overlap(self):
        """TV distance for partially overlapping distributions."""
        refs = References()
        # Sample 1: P(a)=1.0, Sample 2: P(a)=0.5, P(b)=0.5
        # TV = 0.5 * (|1-0.5| + |0-0.5|) = 0.5 * (0.5 + 0.5) = 0.5
        rows1 = {"p1": make_rows(refs, "p1", ["a", "a"])}
        rows2 = {"p1": make_rows(refs, "p1", ["a", "b"])}

        result = bi_two_sample_test(rows1, rows2, pvalue_b=0)
        assert result.statistic == pytest.approx(0.5)

    def test_averaging_across_prompts(self):
        """TV should be averaged across multiple prompts."""
        refs = References()
        # Prompt 1: identical distributions -> TV = 0
        # Prompt 2: completely different -> TV = 1
        # Average TV = 0.5
        rows1 = {
            "p1": make_rows(refs, "p1", ["a", "a"]),
            "p2": make_rows(refs, "p2", ["x", "x"]),
        }
        rows2 = {
            "p1": make_rows(refs, "p1", ["a", "a"]),
            "p2": make_rows(refs, "p2", ["y", "y"]),
        }

        result = bi_two_sample_test(rows1, rows2, pvalue_b=0)
        assert result.statistic == pytest.approx(0.5)

    def test_multiple_output_tokens(self):
        """Test with more than 2 output tokens."""
        refs = References()
        # Sample 1: P(a)=0.5, P(b)=0.25, P(c)=0.25
        # Sample 2: P(a)=0.25, P(b)=0.5, P(c)=0.25
        # TV = 0.5 * (|0.5-0.25| + |0.25-0.5| + |0.25-0.25|) = 0.5 * (0.25 + 0.25 + 0) = 0.25
        rows1 = {"p1": make_rows(refs, "p1", ["a", "a", "b", "c"])}
        rows2 = {"p1": make_rows(refs, "p1", ["a", "b", "b", "c"])}

        result = bi_two_sample_test(rows1, rows2, pvalue_b=0)
        assert result.statistic == pytest.approx(0.25)

    def test_mismatched_keys_raises(self):
        """Should raise AssertionError if prompt keys don't match."""
        refs = References()
        rows1 = {"p1": make_rows(refs, "p1", ["a"])}
        rows2 = {"p2": make_rows(refs, "p2", ["a"])}

        with pytest.raises(AssertionError):
            bi_two_sample_test(rows1, rows2, pvalue_b=0)

    def test_single_sample_per_prompt(self):
        """Edge case: single sample per prompt per group."""
        refs = References()
        rows1 = {"p1": make_rows(refs, "p1", ["a"])}
        rows2 = {"p1": make_rows(refs, "p1", ["a"])}

        result = bi_two_sample_test(rows1, rows2, pvalue_b=0)
        assert result.statistic == pytest.approx(0.0)

        rows3 = {"p1": make_rows(refs, "p1", ["b"])}
        result2 = bi_two_sample_test(rows1, rows3, pvalue_b=0)
        assert result2.statistic == pytest.approx(1.0)


class TestBiTvStatisticVectorized:
    def test_identical_counts(self):
        """TV should be 0 for identical count distributions."""
        import torch

        counts1 = torch.tensor([[2.0, 3.0, 5.0]])
        counts2 = torch.tensor([[2.0, 3.0, 5.0]])
        assert bi_tv_statistic_vectorized(counts1, counts2) == pytest.approx(0.0)

    def test_disjoint_counts(self):
        """TV should be 1 for disjoint distributions."""
        import torch

        counts1 = torch.tensor([[10.0, 0.0]])
        counts2 = torch.tensor([[0.0, 10.0]])
        assert bi_tv_statistic_vectorized(counts1, counts2) == pytest.approx(1.0)

    def test_multiple_prompts_averaged(self):
        """TV should be averaged across prompts."""
        import torch

        # Prompt 1: TV = 0, Prompt 2: TV = 1 -> Average = 0.5
        counts1 = torch.tensor([[5.0, 5.0], [10.0, 0.0]])
        counts2 = torch.tensor([[5.0, 5.0], [0.0, 10.0]])
        assert bi_tv_statistic_vectorized(counts1, counts2) == pytest.approx(0.5)

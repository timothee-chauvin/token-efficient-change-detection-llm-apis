"""BI (Border Inputs) analysis using Total Variation distance as the test statistic."""

import random

import torch
from torch import Tensor

from track_llm_apis.config import config
from track_llm_apis.sampling.common import CompressedOutputRow, TwoSampleTestResult


def bi_two_sample_test(
    rows_subset: dict[str, list[CompressedOutputRow]],
    unchanged_rows_subset: dict[str, list[CompressedOutputRow]],
    pvalue_b: int = 1000,
    **kwargs,
) -> TwoSampleTestResult:
    """
    Two-sample test for BI data using TV distance as the statistic.

    Args:
        rows_subset: dict mapping input_token (prompt) -> list of rows for sample 1
        unchanged_rows_subset: dict mapping input_token (prompt) -> list of rows for sample 2
        pvalue_b: number of permutations for computing the p-value (0 to skip)

    The TV distance is computed per prompt (input token), then averaged across prompts.
    For each prompt, we compute the empirical distribution over output tokens from each sample,
    then compute TV(P1, P2) = 0.5 * sum_x |P1(x) - P2(x)|.
    """
    assert set(rows_subset.keys()) == set(unchanged_rows_subset.keys())
    prompts = list(rows_subset.keys())

    counts1, counts2 = build_count_tensors(rows_subset, unchanged_rows_subset, prompts)
    statistic = bi_tv_statistic_vectorized(counts1, counts2)

    if pvalue_b > 0:
        perm_stats = bi_permutation_pvalue(counts1, counts2, b=pvalue_b)
        pvalue = (perm_stats >= statistic).float().mean().item()
        return TwoSampleTestResult(pvalue=pvalue, statistic=statistic)
    return TwoSampleTestResult(statistic=statistic)


def build_count_tensors(
    rows1: dict[str, list[CompressedOutputRow]],
    rows2: dict[str, list[CompressedOutputRow]],
    prompts: list[str],
) -> tuple[Tensor, Tensor]:
    """
    Build padded count tensors for all prompts.

    Returns:
        counts1: shape (n_prompts, max_vocab_size), counts for sample 1
        counts2: shape (n_prompts, max_vocab_size), counts for sample 2
    """
    device = config.analysis.device
    n_prompts = len(prompts)

    # First pass: build vocabularies and find max vocab size
    vocabs: list[dict[str, int]] = []
    for prompt in prompts:
        outputs1 = [row.text[0] for row in rows1[prompt]]
        outputs2 = [row.text[0] for row in rows2[prompt]]
        all_tokens = sorted(set(outputs1) | set(outputs2))
        vocabs.append({tok: i for i, tok in enumerate(all_tokens)})

    max_vocab = max(len(v) for v in vocabs)

    # Second pass: build count tensors
    counts1 = torch.zeros((n_prompts, max_vocab), dtype=torch.float32, device=device)
    counts2 = torch.zeros((n_prompts, max_vocab), dtype=torch.float32, device=device)

    for i, prompt in enumerate(prompts):
        vocab = vocabs[i]
        for row in rows1[prompt]:
            counts1[i, vocab[row.text[0]]] += 1
        for row in rows2[prompt]:
            counts2[i, vocab[row.text[0]]] += 1

    return counts1, counts2


def bi_tv_statistic_vectorized(counts1: Tensor, counts2: Tensor) -> float:
    """
    Compute the average TV distance across prompts from padded count tensors.

    Args:
        counts1: shape (n_prompts, max_vocab_size), counts for sample 1
        counts2: shape (n_prompts, max_vocab_size), counts for sample 2
    """
    p1 = counts1 / counts1.sum(dim=1, keepdim=True)
    p2 = counts2 / counts2.sum(dim=1, keepdim=True)
    tv_distances = 0.5 * (p1 - p2).abs().sum(dim=1)
    return tv_distances.mean().item()


def bi_tv_statistic_from_counts(counts1: list[Tensor], counts2: list[Tensor]) -> float:
    """
    Compute the average TV distance across prompts from count tensors.
    Legacy API kept for backward compatibility with tests.

    Args:
        counts1: list of count tensors, one per prompt, shape (n_output_tokens,) for sample 1
        counts2: list of count tensors, one per prompt, shape (n_output_tokens,) for sample 2
    """
    max_len = max(max(c.shape[0] for c in counts1), max(c.shape[0] for c in counts2))
    device = counts1[0].device

    padded1 = torch.zeros((len(counts1), max_len), dtype=torch.float32, device=device)
    padded2 = torch.zeros((len(counts2), max_len), dtype=torch.float32, device=device)

    for i, (c1, c2) in enumerate(zip(counts1, counts2)):
        padded1[i, : c1.shape[0]] = c1
        padded2[i, : c2.shape[0]] = c2

    return bi_tv_statistic_vectorized(padded1, padded2)


def bi_permutation_pvalue(counts1: Tensor, counts2: Tensor, b: int = 1000) -> Tensor:
    """
    Compute permutation test statistics for BI TV distance.

    For each prompt, we pool all samples from both groups and randomly assign
    them to two new groups of the same sizes as the original.
    """
    raise NotImplementedError()


class PrecomputedBICounts:
    """
    Precomputed per-sample count tensors for efficient batched testing.

    For each prompt, we store a tensor of shape (n_samples, vocab_size) where each row
    is a one-hot encoding of that sample's output token.
    """

    def __init__(
        self,
        rows: dict[str, list[CompressedOutputRow]],
        prompts: list[str],
        vocabs: list[dict[str, int]] | None,
        max_vocab: int | None,
    ):
        self.device = config.analysis.device
        self.prompts = prompts
        self.prompt_to_idx = {p: i for i, p in enumerate(prompts)}

        if vocabs is not None:
            self.vocabs = vocabs
            self.max_vocab = max_vocab
        else:
            self.vocabs: list[dict[str, int]] = []
            for prompt in prompts:
                all_tokens = sorted(set(row.text[0] for row in rows[prompt]))
                self.vocabs.append({tok: i for i, tok in enumerate(all_tokens)})
            self.max_vocab = max(len(v) for v in self.vocabs) if self.vocabs else 1

        # Build per-sample one-hot tensors for each prompt
        # sample_counts[prompt_idx] has shape (n_samples_for_prompt, max_vocab)
        self.sample_counts: list[Tensor] = []
        for i, prompt in enumerate(prompts):
            n_samples = len(rows[prompt])
            counts = torch.zeros(
                (n_samples, self.max_vocab), dtype=torch.float32, device=self.device
            )
            vocab = self.vocabs[i]
            for j, row in enumerate(rows[prompt]):
                counts[j, vocab[row.text[0]]] = 1.0
            self.sample_counts.append(counts)

    def get_batched_counts(
        self,
        prompt_indices: Tensor,
        sample_indices: Tensor,
    ) -> Tensor:
        """
        Get count tensors for a batch of tests.

        Args:
            prompt_indices: shape (n_tests, n_prompts), which prompts to use per test
            sample_indices: shape (n_tests, n_prompts, samples_per_prompt), which samples to use

        Returns:
            counts: shape (n_tests, n_prompts, max_vocab), summed counts per test per prompt
        """
        n_tests, n_prompts_per_test, samples_per_prompt = sample_indices.shape

        # Initialize output tensor
        counts = torch.zeros(
            (n_tests, n_prompts_per_test, self.max_vocab),
            dtype=torch.float32,
            device=self.device,
        )

        # For each prompt position in the test, gather and sum the samples
        for test_idx in range(n_tests):
            for prompt_pos in range(n_prompts_per_test):
                prompt_idx = prompt_indices[test_idx, prompt_pos].item()
                sample_idxs = sample_indices[test_idx, prompt_pos]
                # Sum the one-hot vectors for selected samples
                counts[test_idx, prompt_pos] = self.sample_counts[prompt_idx][sample_idxs].sum(
                    dim=0
                )

        return counts


def bi_tv_statistic_batched(counts1: Tensor, counts2: Tensor) -> Tensor:
    """
    Compute TV distances for a batch of tests.

    Args:
        counts1: shape (n_tests, n_prompts, max_vocab)
        counts2: shape (n_tests, n_prompts, max_vocab)

    Returns:
        stats: shape (n_tests,), average TV distance per test
    """
    # Normalize to probabilities: (n_tests, n_prompts, max_vocab)
    p1 = counts1 / counts1.sum(dim=2, keepdim=True)
    p2 = counts2 / counts2.sum(dim=2, keepdim=True)
    # TV per prompt: (n_tests, n_prompts)
    tv_per_prompt = 0.5 * (p1 - p2).abs().sum(dim=2)
    # Average across prompts: (n_tests,)
    return tv_per_prompt.mean(dim=1)


def generate_batch_indices(
    n_tests: int,
    n_prompts: int,
    n_total_prompts: int,
    reference_samples_per_prompt: int,
    detection_samples_per_prompt: int,
    n_available_per_prompt1: list[int],
    n_available_per_prompt2: list[int] | None,
    same: bool,
    device: str,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Pre-generate all random indices for batched BI tests.

    Returns:
        prompt_indices: (n_tests, n_prompts) which prompts to use per test
        sample_indices1: (n_tests, n_prompts, reference_samples_per_prompt)
        sample_indices2: (n_tests, n_prompts, detection_samples_per_prompt)
    """
    prompt_indices = torch.zeros((n_tests, n_prompts), dtype=torch.long, device=device)
    for t in range(n_tests):
        selected = random.sample(range(n_total_prompts), n_prompts)
        prompt_indices[t] = torch.tensor(selected, device=device)

    sample_indices1 = torch.zeros(
        (n_tests, n_prompts, reference_samples_per_prompt), dtype=torch.long, device=device
    )
    sample_indices2 = torch.zeros(
        (n_tests, n_prompts, detection_samples_per_prompt), dtype=torch.long, device=device
    )

    for t in range(n_tests):
        for p in range(n_prompts):
            prompt_idx = prompt_indices[t, p].item()
            n_available = n_available_per_prompt1[prompt_idx]

            if same:
                total_needed = reference_samples_per_prompt + detection_samples_per_prompt
                selected = random.sample(range(n_available), total_needed)
                sample_indices1[t, p] = torch.tensor(
                    selected[:reference_samples_per_prompt], device=device
                )
                sample_indices2[t, p] = torch.tensor(
                    selected[reference_samples_per_prompt:], device=device
                )
            else:
                n_available2 = n_available_per_prompt2[prompt_idx]
                sample_indices1[t, p] = torch.tensor(
                    random.sample(range(n_available), reference_samples_per_prompt), device=device
                )
                sample_indices2[t, p] = torch.tensor(
                    random.sample(range(n_available2), detection_samples_per_prompt), device=device
                )

    return prompt_indices, sample_indices1, sample_indices2


def bi_batched_tests_from_precomputed(
    precomputed1: PrecomputedBICounts,
    precomputed2: PrecomputedBICounts | None,
    n_prompts: int,
    reference_samples_per_prompt: int,
    detection_samples_per_prompt: int,
    n_tests: int,
    n_available_per_prompt1: list[int],
    n_available_per_prompt2: list[int] | None,
    same: bool,
) -> list[float]:
    """
    Run batched BI tests using precomputed per-sample counts.
    """
    device = precomputed1.device
    n_total_prompts = len(precomputed1.prompts)

    if same:
        precomputed2 = precomputed1

    prompt_indices, sample_indices1, sample_indices2 = generate_batch_indices(
        n_tests=n_tests,
        n_prompts=n_prompts,
        n_total_prompts=n_total_prompts,
        reference_samples_per_prompt=reference_samples_per_prompt,
        detection_samples_per_prompt=detection_samples_per_prompt,
        n_available_per_prompt1=n_available_per_prompt1,
        n_available_per_prompt2=n_available_per_prompt2,
        same=same,
        device=device,
    )

    counts1 = precomputed1.get_batched_counts(prompt_indices, sample_indices1)
    counts2 = precomputed2.get_batched_counts(prompt_indices, sample_indices2)

    stats = bi_tv_statistic_batched(counts1, counts2)
    return stats.tolist()


def build_shared_vocabs(
    rows1: dict[str, list[CompressedOutputRow]],
    rows2: dict[str, list[CompressedOutputRow]] | None,
    all_prompts: list[str],
) -> tuple[list[dict[str, int]], int]:
    """
    Build shared vocabulary from one or two datasets.

    Returns:
        vocabs: list of vocab dicts, one per prompt
        max_vocab: maximum vocabulary size across prompts
    """
    vocabs: list[dict[str, int]] = []
    for prompt in all_prompts:
        tokens1 = set(row.text[0] for row in rows1[prompt])
        tokens2 = set(row.text[0] for row in rows2[prompt]) if rows2 is not None else set()
        all_tokens = sorted(tokens1 | tokens2)
        vocabs.append({tok: i for i, tok in enumerate(all_tokens)})
    max_vocab = max(len(v) for v in vocabs) if vocabs else 1
    return vocabs, max_vocab


def bi_batched_tests(
    rows1: dict[str, list[CompressedOutputRow]],
    rows2: dict[str, list[CompressedOutputRow]],
    all_prompts: list[str],
    n_prompts: int,
    reference_samples_per_prompt: int,
    detection_samples_per_prompt: int,
    n_tests: int,
    same: bool,
) -> list[float]:
    """
    Run multiple two-sample tests with full batching across n_tests.

    Precomputes per-sample counts once, then uses tensor indexing to efficiently
    compute all test statistics in parallel.
    """
    vocabs, max_vocab = build_shared_vocabs(rows1, None if same else rows2, all_prompts)

    precomputed1 = PrecomputedBICounts(rows1, all_prompts, vocabs, max_vocab)
    precomputed2 = None if same else PrecomputedBICounts(rows2, all_prompts, vocabs, max_vocab)

    n_available1 = [len(rows1[p]) for p in all_prompts]
    n_available2 = None if same else [len(rows2[p]) for p in all_prompts]

    return bi_batched_tests_from_precomputed(
        precomputed1=precomputed1,
        precomputed2=precomputed2,
        n_prompts=n_prompts,
        reference_samples_per_prompt=reference_samples_per_prompt,
        detection_samples_per_prompt=detection_samples_per_prompt,
        n_tests=n_tests,
        n_available_per_prompt1=n_available1,
        n_available_per_prompt2=n_available2,
        same=same,
    )

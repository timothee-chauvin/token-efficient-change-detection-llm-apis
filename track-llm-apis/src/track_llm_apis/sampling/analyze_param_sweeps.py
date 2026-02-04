import os
import random
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

import orjson
import plotly.express as px
import plotly.graph_objects as go
from pydantic import BaseModel
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoTokenizer

from track_llm_apis.config import config, logger
from track_llm_apis.sampling.analyze_bi import (
    PrecomputedBICounts,
    bi_batched_tests_from_precomputed,
    build_shared_vocabs,
)
from track_llm_apis.sampling.analyze_gao2025 import gao2025_two_sample_test
from track_llm_apis.sampling.analyze_logprobs import (
    logprob_two_sample_test_from_compressed_output_row,
)
from track_llm_apis.sampling.analyze_mmlu import mmlu_two_sample_test
from track_llm_apis.sampling.common import (
    CompressedOutput,
    CompressedOutputRow,
    DataSource,
    SingleVariantResult,
    TwoSampleMultiTestResult,
    get_analysis_dir,
    get_variant_filename,
    load_single_variant_results,
)
from track_llm_apis.tinychange import TinyChange
from track_llm_apis.util import ci, compute_yearly_cost

T = TypeVar("T")


def chunk_param_combinations(
    all_combinations: list[T],
    chunk_index: int | None,
    chunk_size: int,
) -> list[T]:
    """Slice parameter combinations for SLURM-based parallelization.

    Args:
        all_combinations: Complete list of parameter combinations to sweep
        chunk_index: Which chunk this job handles (0-indexed). If None, return all.
        chunk_size: Number of combinations per chunk

    Returns:
        The subset of combinations for this chunk, or all if chunk_index is None
    """
    if chunk_index is None:
        return all_combinations

    start = chunk_index * chunk_size
    end = start + chunk_size
    chunked = all_combinations[start:end]

    if not chunked:
        logger.warning(
            f"Chunk {chunk_index} is empty (total combinations: {len(all_combinations)}, "
            f"chunk_size: {chunk_size}). Nothing to process."
        )
    else:
        logger.info(
            f"Processing chunk {chunk_index}: combinations {start}-{min(end, len(all_combinations)) - 1} "
            f"of {len(all_combinations)} total"
        )

    return chunked


@dataclass
class SweepPreparedData:
    """Common data prepared for parameter sweep computation."""

    filtered_data: CompressedOutput
    rows_by_variant: dict[str, list[CompressedOutputRow]]
    unchanged_rows: list[CompressedOutputRow]
    unchanged_rows_by_prompt: dict[str, list[CompressedOutputRow]]
    variants: list[str]
    all_prompts: list[str]
    # Number of tokens of each prompt in the data
    prompt_length: dict[str, int]
    variant_rows_by_prompt: dict[str, dict[str, list[CompressedOutputRow]]]
    analysis_dir: Path


def prepare_sweep_data(
    directory: Path,
    data: CompressedOutput,
    source: DataSource,
) -> SweepPreparedData:
    """Prepare common data structures for parameter sweep computation."""
    filtered_data = data.filter(datasource=source)
    rows_by_variant = filtered_data.get_rows_by_variant()

    unchanged_str = TinyChange.unchanged_str()
    unchanged_rows = rows_by_variant[unchanged_str]
    unchanged_rows_by_prompt = filtered_data.get_rows_by_prompt(unchanged_rows)

    variants = [v for v in rows_by_variant.keys() if v != unchanged_str]
    all_prompts = list(unchanged_rows_by_prompt.keys())

    prompt_length = {
        prompt: tokens
        for prompt, tokens in filtered_data.references.prompts.keys()
        if prompt in unchanged_rows_by_prompt
    }

    variant_rows_by_prompt = {
        v: filtered_data.get_rows_by_prompt(rows_by_variant[v]) for v in variants
    }

    analysis_dir = get_analysis_dir(directory, source)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    return SweepPreparedData(
        filtered_data=filtered_data,
        rows_by_variant=rows_by_variant,
        unchanged_rows=unchanged_rows,
        unchanged_rows_by_prompt=unchanged_rows_by_prompt,
        variants=variants,
        all_prompts=all_prompts,
        prompt_length=prompt_length,
        variant_rows_by_prompt=variant_rows_by_prompt,
        analysis_dir=analysis_dir,
    )


def save_variant_result(
    analysis_dir: Path,
    model_name: str,
    variant: str,
    source: DataSource,
    n_tests: int,
    h0_stats: list[float],
    h1_stats: list[float],
    input_token_avg: int | float,
    output_token_avg: int | float,
    params: dict[str, Any],
) -> None:
    """Save a SingleVariantResult to disk."""
    merged_h0 = TwoSampleMultiTestResult(
        stats=h0_stats,
        pvalues=None,
        input_token_avg=input_token_avg,
        output_token_avg=output_token_avg,
    )
    h1_result = TwoSampleMultiTestResult(
        stats=h1_stats,
        pvalues=None,
        input_token_avg=input_token_avg,
        output_token_avg=output_token_avg,
    )

    result = SingleVariantResult(
        model_name=model_name,
        variant=variant,
        method=source,
        n_tests=n_tests,
        pvalue_b=0,
        h0_stats=merged_h0,
        h1_stats=h1_result,
        params=params,
    )

    filename = get_variant_filename(variant, params)
    with open(analysis_dir / filename, "wb") as f:
        f.write(orjson.dumps(result.model_dump(mode="json")))


def compute_auc_from_stats(h0_stats: list[float], h1_stats: list[float], sampling: bool) -> float:
    if sampling:
        h0_stats = random.choices(h0_stats, k=len(h0_stats))
        h1_stats = random.choices(h1_stats, k=len(h1_stats))
    y_true = [0] * len(h0_stats) + [1] * len(h1_stats)
    y_pred = h0_stats + h1_stats
    return roc_auc_score(y_true, y_pred)


def _bootstrap_one_hierarchical(
    individual_stats: list[tuple[list[float], list[float]]],
) -> float:
    """Single bootstrap iteration with hierarchical structure.

    For each (model, variant) pair, compute AUC with within-pair resampling,
    then return the average across all pairs.
    """
    aucs = [compute_auc_from_stats(h0, h1, sampling=True) for h0, h1 in individual_stats]
    return sum(aucs) / len(aucs)


class ParamSweepResultBase(BaseModel):
    """Base class for parameter sweep results."""

    n_prompts: int
    input_token_avg: float
    output_token_avg: float
    auc: float | None = None
    auc_lower: float | None = None
    auc_upper: float | None = None
    bootstrap_aucs: list[float] | None = None
    yearly_cost: float | None = None


class BIParamSweepResult(ParamSweepResultBase):
    """Result for a single BI parameter combination."""

    detection_samples_per_prompt: int


class METParamSweepResult(ParamSweepResultBase):
    """Result for a single MET parameter combination (T>0 and T=0)."""

    output_tokens: int
    detection_samples_per_prompt: int


class MMLUParamSweepResult(ParamSweepResultBase):
    """Result for a single MMLU parameter combination (T>0 and T=0)."""

    detection_samples_per_prompt: int


class LTParamSweepResult(ParamSweepResultBase):
    """Result for a single LT parameter combination."""

    detection_samples_per_prompt: int


class BIParamSweepPlotData(BaseModel):
    """Plot data for BIParameterSweep."""

    results: list[BIParamSweepResult]
    reference_samples_per_prompt: int


class METParamSweepPlotData(BaseModel):
    """Plot data for METParameterSweep."""

    results: list[METParamSweepResult]
    reference_samples_per_prompt: int


class MMLUParamSweepPlotData(BaseModel):
    """Plot data for MMLUParameterSweep."""

    results: list[MMLUParamSweepResult]
    reference_samples_per_prompt: int


class LTParamSweepPlotData(BaseModel):
    """Plot data for LTParameterSweep."""

    results: list[LTParamSweepResult]
    reference_samples_per_prompt: int


# Type variables for generic parameter sweep classes
TResult = TypeVar("TResult", bound=BaseModel)
TPlotData = TypeVar("TPlotData", bound=BaseModel)


class BaseParameterSweep(ABC, Generic[TResult, TPlotData]):
    """Abstract base class for parameter sweep analyses.

    Provides template methods for the common workflow and requires subclasses
    to implement abstract hooks for method-specific logic.

    Type Parameters:
        TResult: The result type (e.g., BIParamSweepResult)
        TPlotData: The plot data type (e.g., BIParamSweepPlotData)
    """

    stats_filename: str
    plot_dir: Path
    plot_data_path: Path

    @classmethod
    @abstractmethod
    def get_data_source(cls) -> DataSource:
        """Return the DataSource enum for this sweep type."""
        ...

    @classmethod
    @abstractmethod
    def get_param_keys(cls) -> list[str]:
        """Return the parameter keys used for grouping results.

        Example: ["n_prompts", "detection_samples"] for BI
        """
        ...

    @classmethod
    @abstractmethod
    def create_result(
        cls,
        param_values: tuple,
        avg_input_tokens: float,
        avg_output_tokens: float,
    ) -> TResult:
        """Create a result object for a parameter combination."""
        ...

    @classmethod
    @abstractmethod
    def create_plot_data(cls, results: list[TResult]) -> TPlotData:
        """Create plot data object from list of results."""
        ...

    @classmethod
    @abstractmethod
    def load_plot_data(cls, data: dict) -> TPlotData:
        """Load and validate plot data from dict."""
        ...

    @classmethod
    def _compute_yearly_cost(cls, result: TResult) -> float:
        input_tokens_per_sample = result.input_token_avg / result.detection_samples_per_prompt
        output_tokens_per_sample = result.output_token_avg / result.detection_samples_per_prompt
        return compute_yearly_cost(
            input_tokens_per_sample=input_tokens_per_sample,
            output_tokens_per_sample=output_tokens_per_sample,
            n_samples=result.detection_samples_per_prompt,
        )

    @classmethod
    @abstractmethod
    def get_plot_config(cls) -> dict[str, Any]:
        """Return plot configuration dict.

        Should include:
            - title: Plot title
            - group_by: Key to group results by for different colored lines (or None)
            - group_label: Label format for grouped lines (e.g., "n_prompts={}")
            - line_color: Single color if not grouping (optional)
            - line_name: Single name if not grouping (optional)
        """
        ...

    # --- Template methods with shared implementation ---

    @classmethod
    def gen_plot_data(cls) -> None:
        """Aggregate results from all models into a single plot data file with precomputed bootstrap."""
        sampling_dirs = [
            config.sampling_data_dir / dirname for dirname in config.analysis.sampling_dirnames
        ]

        source = cls.get_data_source()
        all_results = load_single_variant_results(sampling_dirs, source)
        if not all_results:
            logger.warning(f"No {source.value} results found")
            return

        param_keys = cls.get_param_keys()
        param_to_results: dict[tuple, list[SingleVariantResult]] = {}
        for result in all_results:
            if result.params:
                key = tuple(result.params[k] for k in param_keys)
                if key not in param_to_results:
                    param_to_results[key] = []
                param_to_results[key].append(result)

        logger.info(
            f"Computing bootstrap AUC for {len(param_to_results)} parameter combinations..."
        )
        n_bootstrap = config.analysis.n_bootstrap

        results: list[TResult] = []
        for param_values, individual_results in tqdm(param_to_results.items(), desc="bootstrap"):
            avg_input = sum(r.h0_stats.input_token_avg for r in individual_results) / len(
                individual_results
            )
            avg_output = sum(r.h0_stats.output_token_avg for r in individual_results) / len(
                individual_results
            )

            result = cls.create_result(param_values, avg_input, avg_output)

            # Point estimate: average of individual (model, variant) AUCs
            individual_aucs = [
                compute_auc_from_stats(r.h0_stats.stats, r.h1_stats.stats, sampling=False)
                for r in individual_results
            ]
            result.auc = sum(individual_aucs) / len(individual_aucs)

            # Bootstrap at the (model, variant) level: each iteration computes AUC per pair
            # with within-pair resampling, then averages
            individual_stats = [(r.h0_stats.stats, r.h1_stats.stats) for r in individual_results]
            with ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(_bootstrap_one_hierarchical, individual_stats)
                    for _ in range(n_bootstrap)
                ]
                bootstrap_aucs = [f.result() for f in futures]

            result.bootstrap_aucs = bootstrap_aucs
            auc_ci = ci(bootstrap_aucs, config.analysis.results_alpha)
            result.auc_lower = auc_ci[0]
            result.auc_upper = auc_ci[1]

            results.append(result)

        plot_data = cls.create_plot_data(results)

        for result in plot_data.results:
            result.yearly_cost = cls._compute_yearly_cost(result)

        os.makedirs(cls.plot_dir, exist_ok=True)
        with open(cls.plot_data_path, "wb") as f:
            f.write(orjson.dumps(plot_data.model_dump(mode="json")))
        logger.info(f"Saved plot data to {cls.plot_data_path}")

    # Marker sizes for secondary grouping (linear scale based on rank)
    MIN_MARKER_SIZE = 8
    MAX_MARKER_SIZE = 16

    @classmethod
    def plot(cls) -> None:
        """Generate performance vs cost plot.

        Supports two-level visual encoding:
        - `group_by`: Parameter encoded by COLOR (primary grouping, e.g. n_prompts)
        - `size_by`: Parameter encoded by MARKER SIZE (secondary grouping, e.g. n_samples)

        When both are specified, a custom two-part legend is created:
        1. Colored lines showing the n_prompts values
        2. Empty circles of increasing size showing the n_samples values
        """
        with open(cls.plot_data_path, "rb") as f:
            plot_data = cls.load_plot_data(orjson.loads(f.read()))

        results = plot_data.results
        plot_config = cls.get_plot_config()

        group_by = plot_config.get("group_by")  # Color encoding (n_prompts)
        size_by = plot_config.get("size_by")  # Size encoding (n_samples)

        if group_by and size_by:
            # Two-level encoding: color for group_by, size for size_by
            fig = go.Figure()
            group_values = sorted(set(getattr(r, group_by) for r in results))
            size_values = sorted(set(getattr(r, size_by) for r in results))

            # Create color palette for group_by
            turbo_samples = px.colors.sample_colorscale(
                "Turbo",
                [i / max(1, len(group_values) - 1) for i in range(len(group_values))],
            )

            # Map size_by values to marker sizes (linear by rank, not proportional to value)
            size_map = {
                val: cls.MIN_MARKER_SIZE
                + (cls.MAX_MARKER_SIZE - cls.MIN_MARKER_SIZE) * i / max(1, len(size_values) - 1)
                for i, val in enumerate(size_values)
            }

            group_label = plot_config.get("group_label", "{}")
            size_label = plot_config.get("size_label", "{}")

            # Add traces grouped by n_prompts (with lines connecting points)
            for i, group_val in enumerate(group_values):
                color = turbo_samples[i]
                # Get all results for this group_val, sorted by cost
                group_results = sorted(
                    [r for r in results if getattr(r, group_by) == group_val],
                    key=lambda r: r.yearly_cost,
                )

                x_vals = [r.yearly_cost for r in group_results]
                y_vals = [r.auc for r in group_results]
                # Marker sizes as array based on each point's size_by value
                marker_sizes = [size_map[getattr(r, size_by)] for r in group_results]

                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode="lines+markers",
                        showlegend=False,  # We'll add custom legend traces
                        line=dict(color=color, width=2),
                        marker=dict(
                            size=marker_sizes,
                            color=color,
                            opacity=1,
                            line=dict(width=0),  # Remove default marker outline
                        ),
                    )
                )

            # --- Custom two-part legend ---
            # Part 1: n_prompts (color legend using short lines)
            for i, group_val in enumerate(group_values):
                color = turbo_samples[i]
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="lines",
                        name=f"{group_label}={group_val}",
                        line=dict(color=color, width=4),
                        legendgroup="color",
                        legendgrouptitle=dict(text=f"<b>{group_label}</b>"),
                    )
                )

            # Part 2: n_samples (size legend using empty circles)
            for size_val in size_values:
                marker_size = size_map[size_val]
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="markers",
                        name=f"{size_label}={size_val}",
                        marker=dict(
                            size=marker_size,
                            color="white",
                            symbol="circle",
                            line=dict(color="black", width=1.5),
                        ),
                        legendgroup="size",
                        legendgrouptitle=dict(text=f"<b>{size_label}</b>"),
                    )
                )

        elif group_by:
            # Single-level encoding: color for group_by, with marker size for varying param
            fig = go.Figure()
            group_values = sorted(set(getattr(r, group_by) for r in results))

            # Map group values to marker sizes (linear by rank)
            size_map = {
                val: cls.MIN_MARKER_SIZE
                + (cls.MAX_MARKER_SIZE - cls.MIN_MARKER_SIZE) * i / max(1, len(group_values) - 1)
                for i, val in enumerate(group_values)
            }

            group_label = plot_config.get("group_label", "{}")

            # All points on a single line, with varying sizes
            results_sorted = sorted(results, key=lambda r: r.yearly_cost)
            x_vals = [r.yearly_cost for r in results_sorted]
            y_vals = [r.auc for r in results_sorted]
            marker_sizes = [size_map[getattr(r, group_by)] for r in results_sorted]

            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="lines+markers",
                    showlegend=False,
                    line=dict(color="gray", width=1),
                    marker=dict(
                        size=marker_sizes,
                        color="gray",
                        symbol="circle",
                        opacity=1,
                        line=dict(width=0),
                    ),
                )
            )

            # Custom legend with colored circles of increasing size
            for i, group_val in enumerate(group_values):
                marker_size = size_map[group_val]
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="markers",
                        name=f"{group_label}={group_val}",
                        marker=dict(
                            size=marker_size,
                            color="white",
                            symbol="circle",
                            line=dict(color="black", width=1.5),
                        ),
                        legendgroup="param",
                        legendgrouptitle=dict(text=f"<b>{group_label}</b>"),
                    )
                )
        else:
            fig = go.Figure()
            results_sorted = sorted(results, key=lambda r: r.yearly_cost)
            x_vals = [r.yearly_cost for r in results_sorted]
            y_vals = [r.auc for r in results_sorted]
            line_color = plot_config.get("line_color", config.plotting.color_map["1"])
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="lines+markers",
                    name=plot_config.get("line_name", cls.get_data_source().value),
                    line=dict(color=line_color),
                    marker=dict(size=10, color=line_color, line=dict(color="black", width=2)),
                )
            )

        fig.update_layout(
            font_family="Spectral",
            font_size=20,
            template="plotly_white",
            xaxis=dict(title="Cost per year ($)", type="log"),
            yaxis_title="Average ROC AUC",
            showlegend=True,
            margin=dict(l=0, r=0, t=0, b=70),
        )

        fig_path = cls.plot_dir / f"{cls.plot_dir.name}.pdf"
        os.makedirs(cls.plot_dir, exist_ok=True)
        fig.write_image(fig_path)
        logger.info(f"Saved plot to {fig_path}")

    @classmethod
    def gen_plot_data_and_plot(cls) -> None:
        """Generate plot data and create plot."""
        cls.gen_plot_data()
        cls.plot()


class BIParameterSweep(BaseParameterSweep[BIParamSweepResult, BIParamSweepPlotData]):
    """Sweep over B3IT parameters (n_prompts, detection_samples_per_prompt) to analyze performance vs cost."""

    stats_filename = "bi_param_sweep.json"
    plot_dir = config.plots_dir / "paper" / "bi_param_sweep"
    plot_data_path = plot_dir / "bi_param_sweep.json"

    @staticmethod
    def compute_stats(
        directory: Path,
        data: CompressedOutput,
        n_prompts_values: list[int],
        detection_samples_values: list[int],
        reference_samples_per_prompt: int,
        n_tests: int,
    ):
        """Saves per-variant-per-param files to analysis/bi/."""
        source = DataSource.BI
        sd = prepare_sweep_data(directory, data, source)

        logger.info("B3IT: Building shared vocabulary across all variants...")
        all_rows_combined: dict[str, list[CompressedOutputRow]] = {p: [] for p in sd.all_prompts}
        for p in sd.all_prompts:
            all_rows_combined[p].extend(sd.unchanged_rows_by_prompt[p])
            for v in sd.variants:
                all_rows_combined[p].extend(sd.variant_rows_by_prompt[v][p])
        vocabs, max_vocab = build_shared_vocabs(all_rows_combined, None, sd.all_prompts)

        logger.info("B3IT: Precomputing counts for unchanged model...")
        precomputed_unchanged = PrecomputedBICounts(
            sd.unchanged_rows_by_prompt, sd.all_prompts, vocabs, max_vocab
        )
        n_available_unchanged = [len(sd.unchanged_rows_by_prompt[p]) for p in sd.all_prompts]

        logger.info("B3IT: Precomputing counts for all variants...")
        precomputed_variants: dict[str, PrecomputedBICounts] = {}
        n_available_variants: dict[str, list[int]] = {}
        for v in sd.variants:
            precomputed_variants[v] = PrecomputedBICounts(
                sd.variant_rows_by_prompt[v], sd.all_prompts, vocabs, max_vocab
            )
            n_available_variants[v] = [len(sd.variant_rows_by_prompt[v][p]) for p in sd.all_prompts]

        avg_prompt_length = sum(sd.prompt_length[p] for p in sd.all_prompts) / len(sd.all_prompts)

        # Generate all parameter combinations and chunk if configured
        all_param_combos = [(n, d) for d in detection_samples_values for n in n_prompts_values]
        param_combos = chunk_param_combinations(
            all_param_combos,
            config.analysis.param_chunk_index,
            config.analysis.param_chunk_size,
        )

        for n_prompts, detection_samples in param_combos:
            logger.info(f"Computing stats for {n_prompts=}, {detection_samples=}...")
            params = {"n_prompts": n_prompts, "detection_samples": detection_samples}

            unchanged_h0_stats = bi_batched_tests_from_precomputed(
                precomputed1=precomputed_unchanged,
                precomputed2=None,
                n_prompts=n_prompts,
                reference_samples_per_prompt=reference_samples_per_prompt,
                detection_samples_per_prompt=detection_samples,
                n_tests=n_tests,
                n_available_per_prompt1=n_available_unchanged,
                n_available_per_prompt2=None,
                same=True,
            )

            input_tokens = n_prompts * detection_samples * avg_prompt_length
            output_tokens = n_prompts * detection_samples

            for variant in sd.variants:
                logger.info(
                    f"[{n_prompts}, {detection_samples}] Computing stats for variant {variant}..."
                )

                variant_h0_stats = bi_batched_tests_from_precomputed(
                    precomputed1=precomputed_variants[variant],
                    precomputed2=None,
                    n_prompts=n_prompts,
                    reference_samples_per_prompt=reference_samples_per_prompt,
                    detection_samples_per_prompt=detection_samples,
                    n_tests=n_tests,
                    n_available_per_prompt1=n_available_variants[variant],
                    n_available_per_prompt2=None,
                    same=True,
                )

                h1_stats = bi_batched_tests_from_precomputed(
                    precomputed1=precomputed_unchanged,
                    precomputed2=precomputed_variants[variant],
                    n_prompts=n_prompts,
                    reference_samples_per_prompt=reference_samples_per_prompt,
                    detection_samples_per_prompt=detection_samples,
                    n_tests=n_tests,
                    n_available_per_prompt1=n_available_unchanged,
                    n_available_per_prompt2=n_available_variants[variant],
                    same=False,
                )

                save_variant_result(
                    sd.analysis_dir,
                    data.model_name,
                    variant,
                    source,
                    n_tests,
                    unchanged_h0_stats + variant_h0_stats,
                    h1_stats,
                    input_tokens,
                    output_tokens,
                    params,
                )

    # --- Abstract method implementations ---

    @classmethod
    def get_data_source(cls) -> DataSource:
        return DataSource.BI

    @classmethod
    def get_param_keys(cls) -> list[str]:
        return ["n_prompts", "detection_samples"]

    @classmethod
    def create_result(
        cls,
        param_values: tuple,
        avg_input_tokens: float,
        avg_output_tokens: float,
    ) -> BIParamSweepResult:
        n_prompts, detection_samples = param_values
        return BIParamSweepResult(
            n_prompts=n_prompts,
            detection_samples_per_prompt=detection_samples,
            input_token_avg=avg_input_tokens,
            output_token_avg=avg_output_tokens,
        )

    @classmethod
    def create_plot_data(cls, results: list[BIParamSweepResult]) -> BIParamSweepPlotData:
        reference_samples = config.sampling.bi.reference_samples_per_prompt
        return BIParamSweepPlotData(
            results=results,
            reference_samples_per_prompt=reference_samples,
        )

    @classmethod
    def load_plot_data(cls, data: dict) -> BIParamSweepPlotData:
        return BIParamSweepPlotData.model_validate(data)

    @classmethod
    def compute_yearly_cost(
        cls, result: BIParamSweepResult, plot_data: BIParamSweepPlotData
    ) -> float:
        input_tokens_per_sample = result.input_token_avg / result.detection_samples_per_prompt
        output_tokens_per_sample = result.output_token_avg / result.detection_samples_per_prompt
        return compute_yearly_cost(
            input_tokens_per_sample=input_tokens_per_sample,
            output_tokens_per_sample=output_tokens_per_sample,
            n_samples=result.detection_samples_per_prompt,
        )

    @classmethod
    def get_plot_config(cls) -> dict[str, Any]:
        return {
            "group_by": "n_prompts",
            "group_label": "n_prompts",
            "size_by": "detection_samples_per_prompt",
            "size_label": "n_samples",
        }


class METBaseSweep(BaseParameterSweep[METParamSweepResult, METParamSweepPlotData]):
    """Shared base for MET and METT0 parameter sweeps."""

    @classmethod
    def get_param_keys(cls) -> list[str]:
        return ["n_prompts", "output_tokens"]

    @classmethod
    def create_plot_data(cls, results: list[METParamSweepResult]) -> METParamSweepPlotData:
        return METParamSweepPlotData(
            results=results,
            reference_samples_per_prompt=config.sampling.gao2025.reference_samples_per_prompt,
        )

    @classmethod
    def load_plot_data(cls, data: dict) -> METParamSweepPlotData:
        return METParamSweepPlotData.model_validate(data)


class METParameterSweep(METBaseSweep):
    """Sweep over MET parameters (n_prompts, output_tokens) to analyze performance vs cost."""

    stats_filename = "met_param_sweep.json"
    plot_dir = config.plots_dir / "paper" / "met_param_sweep"
    plot_data_path = plot_dir / "met_param_sweep.json"

    @staticmethod
    def compute_stats(
        directory: Path,
        data: CompressedOutput,
        n_prompts_values: list[int],
        output_tokens_values: list[int],
        reference_samples_per_prompt: int,
        detection_samples_per_prompt: int,
        n_tests: int,
    ):
        """Saves per-variant-per-param files to analysis/met/."""
        source = DataSource.GAO2025
        sd = prepare_sweep_data(directory, data, source)

        tokenizer = AutoTokenizer.from_pretrained(data.model_name)
        if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        avg_prompt_length = sum(sd.prompt_length[p] for p in sd.all_prompts) / len(sd.all_prompts)

        # Generate all parameter combinations and chunk if configured
        all_param_combos = [(n, o) for o in output_tokens_values for n in n_prompts_values]
        param_combos = chunk_param_combinations(
            all_param_combos,
            config.analysis.param_chunk_index,
            config.analysis.param_chunk_size,
        )

        for n_prompts, output_tokens in param_combos:
            logger.info(
                f"Computing stats for n_prompts={n_prompts}, output_tokens={output_tokens}..."
            )
            params = {"n_prompts": n_prompts, "output_tokens": output_tokens}

            # H0 test: reference samples vs detection samples (asymmetric)
            total_needed = reference_samples_per_prompt + detection_samples_per_prompt
            unchanged_h0_stats: list[float] = []
            for _ in range(n_tests):
                selected_prompts = random.sample(
                    sd.all_prompts, min(n_prompts, len(sd.all_prompts))
                )
                unchanged_subset1, unchanged_subset2 = {}, {}
                for p in selected_prompts:
                    rows = sd.unchanged_rows_by_prompt[p]
                    sampled = random.sample(rows, min(total_needed, len(rows)))
                    unchanged_subset1[p] = sampled[:reference_samples_per_prompt]
                    unchanged_subset2[p] = sampled[reference_samples_per_prompt:total_needed]

                h0_result = gao2025_two_sample_test(
                    unchanged_subset1,
                    unchanged_subset2,
                    tokenizer=tokenizer,
                    pvalue_b=0,
                    max_tokens=output_tokens,
                )
                unchanged_h0_stats.append(h0_result.statistic)

            input_tokens_avg = n_prompts * detection_samples_per_prompt * avg_prompt_length
            output_tokens_avg = n_prompts * detection_samples_per_prompt * output_tokens

            for variant in sd.variants:
                logger.info(
                    f"[{n_prompts}, {output_tokens}] Computing stats for variant {variant}..."
                )
                variant_h0_stats: list[float] = []
                h1_stats: list[float] = []

                for _ in range(n_tests):
                    selected_prompts = random.sample(
                        sd.all_prompts, min(n_prompts, len(sd.all_prompts))
                    )

                    # H1 test: reference samples from unchanged vs detection samples from variant
                    unchanged_subset = {
                        p: random.sample(
                            sd.unchanged_rows_by_prompt[p],
                            min(reference_samples_per_prompt, len(sd.unchanged_rows_by_prompt[p])),
                        )
                        for p in selected_prompts
                    }
                    variant_subset = {
                        p: random.sample(
                            sd.variant_rows_by_prompt[variant][p],
                            min(
                                detection_samples_per_prompt,
                                len(sd.variant_rows_by_prompt[variant][p]),
                            ),
                        )
                        for p in selected_prompts
                    }

                    h1_result = gao2025_two_sample_test(
                        unchanged_subset,
                        variant_subset,
                        tokenizer=tokenizer,
                        pvalue_b=0,
                        max_tokens=output_tokens,
                    )
                    h1_stats.append(h1_result.statistic)

                    # Variant H0: reference vs detection from variant
                    variant_subset1, variant_subset2 = {}, {}
                    for p in selected_prompts:
                        rows = sd.variant_rows_by_prompt[variant][p]
                        sampled = random.sample(rows, min(total_needed, len(rows)))
                        variant_subset1[p] = sampled[:reference_samples_per_prompt]
                        variant_subset2[p] = sampled[reference_samples_per_prompt:total_needed]

                    h0_variant_result = gao2025_two_sample_test(
                        variant_subset1,
                        variant_subset2,
                        tokenizer=tokenizer,
                        pvalue_b=0,
                        max_tokens=output_tokens,
                    )
                    variant_h0_stats.append(h0_variant_result.statistic)

                save_variant_result(
                    sd.analysis_dir,
                    data.model_name,
                    variant,
                    source,
                    n_tests,
                    unchanged_h0_stats + variant_h0_stats,
                    h1_stats,
                    input_tokens_avg,
                    output_tokens_avg,
                    params,
                )

    @classmethod
    def get_data_source(cls) -> DataSource:
        return DataSource.GAO2025

    @classmethod
    def create_result(
        cls,
        param_values: tuple,
        avg_input_tokens: float,
        avg_output_tokens: float,
    ) -> METParamSweepResult:
        n_prompts, output_tokens = param_values
        return METParamSweepResult(
            n_prompts=n_prompts,
            output_tokens=output_tokens,
            detection_samples_per_prompt=config.analysis.get_detection_samples("GAO2025"),
            input_token_avg=avg_input_tokens,
            output_token_avg=avg_output_tokens,
        )

    @classmethod
    def get_plot_config(cls) -> dict[str, Any]:
        return {
            "group_by": "n_prompts",
            "group_label": "n_prompts",
            "size_by": "output_tokens",
            "size_label": "output_tokens",
        }


class MMLUBaseSweep(BaseParameterSweep[MMLUParamSweepResult, MMLUParamSweepPlotData]):
    """Shared base for MMLU and MMLUT0 parameter sweeps."""

    @classmethod
    def get_param_keys(cls) -> list[str]:
        return ["n_prompts"]

    @classmethod
    def create_plot_data(cls, results: list[MMLUParamSweepResult]) -> MMLUParamSweepPlotData:
        return MMLUParamSweepPlotData(
            results=results,
            reference_samples_per_prompt=config.sampling.mmlu.reference_samples_per_prompt,
        )

    @classmethod
    def load_plot_data(cls, data: dict) -> MMLUParamSweepPlotData:
        return MMLUParamSweepPlotData.model_validate(data)


class MMLUParameterSweep(MMLUBaseSweep):
    """Sweep over MMLU parameters (n_prompts) to analyze performance vs cost."""

    stats_filename = "mmlu_param_sweep.json"
    plot_dir = config.plots_dir / "paper" / "mmlu_param_sweep"
    plot_data_path = plot_dir / "mmlu_param_sweep.json"

    @staticmethod
    def compute_stats(
        directory: Path,
        data: CompressedOutput,
        n_prompts_values: list[int],
        reference_samples_per_prompt: int,
        detection_samples_per_prompt: int,
        n_tests: int,
    ):
        """Saves per-variant-per-param files to analysis/mmlu/."""
        source = DataSource.MMLU
        sd = prepare_sweep_data(directory, data, source)

        avg_prompt_length = sum(sd.prompt_length[p] for p in sd.all_prompts) / len(sd.all_prompts)

        # Generate all parameter combinations and chunk if configured
        # MMLU only has n_prompts as a sweep parameter
        all_param_combos = [(n,) for n in n_prompts_values]
        param_combos = chunk_param_combinations(
            all_param_combos,
            config.analysis.param_chunk_index,
            config.analysis.param_chunk_size,
        )

        for (n_prompts,) in param_combos:
            logger.info(f"Computing stats for n_prompts={n_prompts}...")
            params = {"n_prompts": n_prompts}

            # H0 test: reference samples vs detection samples (asymmetric)
            total_needed = reference_samples_per_prompt + detection_samples_per_prompt
            unchanged_h0_stats: list[float] = []
            for _ in range(n_tests):
                selected_prompts = random.sample(
                    sd.all_prompts, min(n_prompts, len(sd.all_prompts))
                )
                unchanged_subset1, unchanged_subset2 = {}, {}
                for p in selected_prompts:
                    rows = sd.unchanged_rows_by_prompt[p]
                    sampled = random.sample(rows, min(total_needed, len(rows)))
                    unchanged_subset1[p] = sampled[:reference_samples_per_prompt]
                    unchanged_subset2[p] = sampled[reference_samples_per_prompt:total_needed]

                h0_result = mmlu_two_sample_test(unchanged_subset1, unchanged_subset2, pvalue_b=0)
                unchanged_h0_stats.append(h0_result.statistic)

            input_tokens_avg = n_prompts * detection_samples_per_prompt * avg_prompt_length
            output_tokens_avg = (
                n_prompts * detection_samples_per_prompt * config.sampling.mmlu.max_tokens
            )

            for variant in sd.variants:
                logger.info(f"[{n_prompts}] Computing stats for variant {variant}...")
                variant_h0_stats: list[float] = []
                h1_stats: list[float] = []

                for _ in range(n_tests):
                    selected_prompts = random.sample(
                        sd.all_prompts, min(n_prompts, len(sd.all_prompts))
                    )

                    # H1 test: reference samples from unchanged vs detection samples from variant
                    unchanged_subset = {
                        p: random.sample(
                            sd.unchanged_rows_by_prompt[p],
                            min(reference_samples_per_prompt, len(sd.unchanged_rows_by_prompt[p])),
                        )
                        for p in selected_prompts
                    }
                    variant_subset = {
                        p: random.sample(
                            sd.variant_rows_by_prompt[variant][p],
                            min(
                                detection_samples_per_prompt,
                                len(sd.variant_rows_by_prompt[variant][p]),
                            ),
                        )
                        for p in selected_prompts
                    }

                    h1_result = mmlu_two_sample_test(unchanged_subset, variant_subset, pvalue_b=0)
                    h1_stats.append(h1_result.statistic)

                    # Variant H0: reference vs detection from variant
                    variant_subset1, variant_subset2 = {}, {}
                    for p in selected_prompts:
                        rows = sd.variant_rows_by_prompt[variant][p]
                        sampled = random.sample(rows, min(total_needed, len(rows)))
                        variant_subset1[p] = sampled[:reference_samples_per_prompt]
                        variant_subset2[p] = sampled[reference_samples_per_prompt:total_needed]

                    h0_variant_result = mmlu_two_sample_test(
                        variant_subset1, variant_subset2, pvalue_b=0
                    )
                    variant_h0_stats.append(h0_variant_result.statistic)

                save_variant_result(
                    sd.analysis_dir,
                    data.model_name,
                    variant,
                    source,
                    n_tests,
                    unchanged_h0_stats + variant_h0_stats,
                    h1_stats,
                    input_tokens_avg,
                    output_tokens_avg,
                    params,
                )

    @classmethod
    def get_data_source(cls) -> DataSource:
        return DataSource.MMLU

    @classmethod
    def create_result(
        cls,
        param_values: tuple,
        avg_input_tokens: float,
        avg_output_tokens: float,
    ) -> MMLUParamSweepResult:
        (n_prompts,) = param_values
        return MMLUParamSweepResult(
            n_prompts=n_prompts,
            detection_samples_per_prompt=config.analysis.get_detection_samples("MMLU"),
            input_token_avg=avg_input_tokens,
            output_token_avg=avg_output_tokens,
        )

    @classmethod
    def get_plot_config(cls) -> dict[str, Any]:
        return {
            "group_by": "n_prompts",
            "group_label": "n_prompts",
        }


class METT0ParameterSweep(METBaseSweep):
    """Sweep over MET-T0 parameters (n_prompts, output_tokens) to analyze performance vs cost.

    Unlike METParameterSweep, T0 has a single sample per prompt (temperature=0 gives
    deterministic output), so there's a single stat per (variant, params).
    """

    stats_filename = "met_t0_param_sweep.json"
    plot_dir = config.plots_dir / "paper" / "met_t0_param_sweep"
    plot_data_path = plot_dir / "met_t0_param_sweep.json"

    @staticmethod
    def compute_stats(
        directory: Path,
        data: CompressedOutput,
        n_prompts_values: list[int],
        output_tokens_values: list[int],
    ):
        """Saves per-variant-per-param files to analysis/met_t0/.

        With T0, n_samples_per_prompt=1 (deterministic output), so we get a single
        statistic per (variant, params) combination.
        """
        source = DataSource.GAO2025_T0
        sd = prepare_sweep_data(directory, data, source)

        tokenizer = AutoTokenizer.from_pretrained(data.model_name)
        if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Generate all parameter combinations and chunk if configured
        all_param_combos = [(n, o) for o in output_tokens_values for n in n_prompts_values]
        param_combos = chunk_param_combinations(
            all_param_combos,
            config.analysis.param_chunk_index,
            config.analysis.param_chunk_size,
        )

        for n_prompts, output_tokens in param_combos:
            logger.info(
                f"Computing stats for n_prompts={n_prompts}, output_tokens={output_tokens}..."
            )
            params = {"n_prompts": n_prompts, "output_tokens": output_tokens}

            selected_prompts = (
                sd.all_prompts[:n_prompts] if len(sd.all_prompts) >= n_prompts else sd.all_prompts
            )

            unchanged_subset = {
                p: sd.unchanged_rows_by_prompt[p][:1]
                for p in selected_prompts
                if sd.unchanged_rows_by_prompt[p]
            }

            h0_result = gao2025_two_sample_test(
                unchanged_subset,
                unchanged_subset,
                tokenizer=tokenizer,
                pvalue_b=0,
                max_tokens=output_tokens,
            )
            unchanged_h0_stat = h0_result.statistic

            avg_prompt_length = sum(sd.prompt_length[p] for p in selected_prompts) / len(
                selected_prompts
            )
            # Cost: only count detection samples (1 sample per prompt for T=0)
            input_tokens_avg = n_prompts * avg_prompt_length
            output_tokens_avg = n_prompts * output_tokens

            for variant in sd.variants:
                logger.info(
                    f"[{n_prompts}, {output_tokens}] Computing stats for variant {variant}..."
                )

                variant_subset = {
                    p: sd.variant_rows_by_prompt[variant][p][:1]
                    for p in selected_prompts
                    if sd.variant_rows_by_prompt[variant].get(p)
                }

                variant_h0_result = gao2025_two_sample_test(
                    variant_subset,
                    variant_subset,
                    tokenizer=tokenizer,
                    pvalue_b=0,
                    max_tokens=output_tokens,
                )

                h1_result = gao2025_two_sample_test(
                    unchanged_subset,
                    variant_subset,
                    tokenizer=tokenizer,
                    pvalue_b=0,
                    max_tokens=output_tokens,
                )

                save_variant_result(
                    sd.analysis_dir,
                    data.model_name,
                    variant,
                    source,
                    1,
                    [unchanged_h0_stat, variant_h0_result.statistic],
                    [h1_result.statistic],
                    input_tokens_avg,
                    output_tokens_avg,
                    params,
                )

    @classmethod
    def get_data_source(cls) -> DataSource:
        return DataSource.GAO2025_T0

    @classmethod
    def create_result(
        cls,
        param_values: tuple,
        avg_input_tokens: float,
        avg_output_tokens: float,
    ) -> METParamSweepResult:
        n_prompts, output_tokens = param_values
        return METParamSweepResult(
            n_prompts=n_prompts,
            output_tokens=output_tokens,
            detection_samples_per_prompt=1,
            input_token_avg=avg_input_tokens,
            output_token_avg=avg_output_tokens,
        )

    @classmethod
    def get_plot_config(cls) -> dict[str, Any]:
        return {
            "group_by": "n_prompts",
            "group_label": "n_prompts",
            "size_by": "output_tokens",
            "size_label": "output_tokens",
        }


class MMLUT0ParameterSweep(MMLUBaseSweep):
    """Sweep over MMLU-T0 parameters (n_prompts) to analyze performance vs cost.

    Unlike MMLUParameterSweep, T0 has a single sample per prompt (temperature=0 gives
    deterministic output), so there's a single stat per (variant, params).
    """

    stats_filename = "mmlu_t0_param_sweep.json"
    plot_dir = config.plots_dir / "paper" / "mmlu_t0_param_sweep"
    plot_data_path = plot_dir / "mmlu_t0_param_sweep.json"

    @staticmethod
    def compute_stats(
        directory: Path,
        data: CompressedOutput,
        n_prompts_values: list[int],
    ):
        """Saves per-variant-per-param files to analysis/mmlu_t0/.

        With T0, n_samples_per_prompt=1 (deterministic output), so we get a single
        statistic per (variant, params) combination.
        """
        source = DataSource.MMLU_T0
        sd = prepare_sweep_data(directory, data, source)

        # Generate all parameter combinations and chunk if configured
        # MMLU-T0 only has n_prompts as a sweep parameter
        all_param_combos = [(n,) for n in n_prompts_values]
        param_combos = chunk_param_combinations(
            all_param_combos,
            config.analysis.param_chunk_index,
            config.analysis.param_chunk_size,
        )

        for (n_prompts,) in param_combos:
            logger.info(f"Computing stats for n_prompts={n_prompts}...")
            params = {"n_prompts": n_prompts}

            selected_prompts = (
                sd.all_prompts[:n_prompts] if len(sd.all_prompts) >= n_prompts else sd.all_prompts
            )

            unchanged_subset = {
                p: sd.unchanged_rows_by_prompt[p][:1]
                for p in selected_prompts
                if sd.unchanged_rows_by_prompt[p]
            }

            h0_result = mmlu_two_sample_test(unchanged_subset, unchanged_subset, pvalue_b=0)
            unchanged_h0_stat = h0_result.statistic

            avg_prompt_length = sum(sd.prompt_length[p] for p in selected_prompts) / len(
                selected_prompts
            )
            input_tokens_avg = n_prompts * avg_prompt_length
            output_tokens_avg = n_prompts * config.sampling.mmlu.max_tokens

            for variant in sd.variants:
                logger.info(f"[{n_prompts}] Computing stats for variant {variant}...")

                variant_subset = {
                    p: sd.variant_rows_by_prompt[variant][p][:1]
                    for p in selected_prompts
                    if sd.variant_rows_by_prompt[variant].get(p)
                }

                variant_h0_result = mmlu_two_sample_test(variant_subset, variant_subset, pvalue_b=0)
                h1_result = mmlu_two_sample_test(unchanged_subset, variant_subset, pvalue_b=0)

                save_variant_result(
                    sd.analysis_dir,
                    data.model_name,
                    variant,
                    source,
                    1,
                    [unchanged_h0_stat, variant_h0_result.statistic],
                    [h1_result.statistic],
                    input_tokens_avg,
                    output_tokens_avg,
                    params,
                )

    @classmethod
    def get_data_source(cls) -> DataSource:
        return DataSource.MMLU_T0

    @classmethod
    def create_result(
        cls,
        param_values: tuple,
        avg_input_tokens: float,
        avg_output_tokens: float,
    ) -> MMLUParamSweepResult:
        (n_prompts,) = param_values
        return MMLUParamSweepResult(
            n_prompts=n_prompts,
            detection_samples_per_prompt=1,
            input_token_avg=avg_input_tokens,
            output_token_avg=avg_output_tokens,
        )

    @classmethod
    def get_plot_config(cls) -> dict[str, Any]:
        return {
            "group_by": "n_prompts",
            "group_label": "n_prompts",
        }


class LTParameterSweep(BaseParameterSweep[LTParamSweepResult, LTParamSweepPlotData]):
    """Sweep over LT parameters (n_samples_per_prompt) to analyze performance vs cost."""

    stats_filename = "lt_param_sweep.json"
    plot_dir = config.plots_dir / "paper" / "lt_param_sweep"
    plot_data_path = plot_dir / "lt_param_sweep.json"

    @staticmethod
    def compute_stats(
        directory: Path,
        data: CompressedOutput,
        n_samples_values: list[int],
        reference_samples_per_prompt: int,
        n_tests: int,
    ):
        """Saves per-variant-per-param files to analysis/lt/."""
        source = DataSource.LT
        sd = prepare_sweep_data(directory, data, source)

        default_prompt = config.sampling.logprob.default_prompt
        if default_prompt not in sd.unchanged_rows_by_prompt:
            logger.warning(f"Default prompt '{default_prompt}' not found in LT data")
            return

        prompt_length = sd.prompt_length.get(default_prompt, 0)
        unchanged_rows = sd.unchanged_rows_by_prompt[default_prompt]

        all_param_combos = [(n,) for n in n_samples_values]
        param_combos = chunk_param_combinations(
            all_param_combos,
            config.analysis.param_chunk_index,
            config.analysis.param_chunk_size,
        )

        for (detection_samples,) in param_combos:
            logger.info(f"Computing stats for {detection_samples=}...")
            params = {"detection_samples_per_prompt": detection_samples}

            # H0 test: reference samples vs detection samples (asymmetric)
            total_needed = reference_samples_per_prompt + detection_samples
            unchanged_h0_stats: list[float] = []
            for _ in range(n_tests):
                sampled = random.sample(unchanged_rows, min(total_needed, len(unchanged_rows)))
                sample1 = {default_prompt: sampled[:reference_samples_per_prompt]}
                sample2 = {default_prompt: sampled[reference_samples_per_prompt:total_needed]}

                result = logprob_two_sample_test_from_compressed_output_row(
                    sample1, sample2, pvalue_b=0
                )
                unchanged_h0_stats.append(result.statistic)

            sample_output_tokens = (
                sum(r.text[1] for r in unchanged_rows[:detection_samples]) / detection_samples
            )
            input_tokens_avg = detection_samples * prompt_length
            output_tokens_avg = detection_samples * sample_output_tokens

            for variant in sd.variants:
                logger.info(f"[{detection_samples}] Computing stats for variant {variant}...")

                variant_rows = sd.variant_rows_by_prompt[variant].get(default_prompt, [])
                if not variant_rows:
                    logger.warning(f"No LT rows for variant {variant}, prompt {default_prompt}")
                    continue

                variant_h0_stats: list[float] = []
                h1_stats: list[float] = []

                for _ in range(n_tests):
                    # H1 test: reference samples from unchanged vs detection samples from variant
                    unchanged_sample = random.sample(
                        unchanged_rows, min(reference_samples_per_prompt, len(unchanged_rows))
                    )
                    variant_sample = random.sample(
                        variant_rows, min(detection_samples, len(variant_rows))
                    )

                    h1_result = logprob_two_sample_test_from_compressed_output_row(
                        {default_prompt: unchanged_sample},
                        {default_prompt: variant_sample},
                        pvalue_b=0,
                    )
                    h1_stats.append(h1_result.statistic)

                    # Variant H0: reference vs detection from variant
                    sampled = random.sample(variant_rows, min(total_needed, len(variant_rows)))
                    v_sample1 = {default_prompt: sampled[:reference_samples_per_prompt]}
                    v_sample2 = {default_prompt: sampled[reference_samples_per_prompt:total_needed]}

                    h0_variant_result = logprob_two_sample_test_from_compressed_output_row(
                        v_sample1, v_sample2, pvalue_b=0
                    )
                    variant_h0_stats.append(h0_variant_result.statistic)

                save_variant_result(
                    sd.analysis_dir,
                    data.model_name,
                    variant,
                    source,
                    n_tests,
                    unchanged_h0_stats + variant_h0_stats,
                    h1_stats,
                    input_tokens_avg,
                    output_tokens_avg,
                    params,
                )

    @classmethod
    def get_data_source(cls) -> DataSource:
        return DataSource.LT

    @classmethod
    def get_param_keys(cls) -> list[str]:
        return ["detection_samples_per_prompt"]

    @classmethod
    def create_result(
        cls,
        param_values: tuple,
        avg_input_tokens: float,
        avg_output_tokens: float,
    ) -> LTParamSweepResult:
        (detection_samples,) = param_values
        return LTParamSweepResult(
            n_prompts=1,  # LT always uses single prompt
            detection_samples_per_prompt=detection_samples,
            input_token_avg=avg_input_tokens,
            output_token_avg=avg_output_tokens,
        )

    @classmethod
    def create_plot_data(cls, results: list[LTParamSweepResult]) -> LTParamSweepPlotData:
        return LTParamSweepPlotData(
            results=results,
            reference_samples_per_prompt=config.sampling.logprob.reference_samples_per_prompt,
        )

    @classmethod
    def load_plot_data(cls, data: dict) -> LTParamSweepPlotData:
        return LTParamSweepPlotData.model_validate(data)

    @classmethod
    def get_plot_config(cls) -> dict[str, Any]:
        return {
            "group_by": "detection_samples_per_prompt",
            "group_label": "n_samples",
        }

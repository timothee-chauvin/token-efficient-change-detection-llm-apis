import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Literal

import orjson
import plotly.express as px
import plotly.graph_objects as go
from pydantic import BaseModel, field_validator
from tqdm import tqdm

from track_llm_apis.config import config
from track_llm_apis.sampling.analysis.lt_prompt_ablation import PromptAblation
from track_llm_apis.sampling.analyze_param_sweeps import (
    BIParameterSweep,
    LTParameterSweep,
    METParameterSweep,
    METT0ParameterSweep,
    MMLUParameterSweep,
    MMLUT0ParameterSweep,
    compute_auc_from_stats,
)
from track_llm_apis.sampling.common import (
    CIResult,
    CompressedOutput,
    DataSource,
    SingleVariantResult,
    Variant,
    _variant_preslug,
    load_single_variant_results,
)
from track_llm_apis.util import ci, slugify

logger = config.logger


def get_baseline_params(source: DataSource) -> dict[str, int]:
    """Get the baseline parameters for a given DataSource."""
    return {
        DataSource.BI: config.analysis.baseline_bi_params,
        DataSource.GAO2025: config.analysis.baseline_met_params,
        DataSource.GAO2025_T0: config.analysis.baseline_met_t0_params,
        DataSource.MMLU: config.analysis.baseline_mmlu_params,
        DataSource.MMLU_T0: config.analysis.baseline_mmlu_t0_params,
        DataSource.LT: config.analysis.baseline_lt_params,
    }[source]


def filter_results_by_baseline_params(
    results: list[SingleVariantResult],
) -> list[SingleVariantResult]:
    """Filter results to only include those matching baseline params for each method."""
    filtered = []
    for r in results:
        baseline_params = get_baseline_params(r.method)
        if r.params == baseline_params:
            filtered.append(r)
    return filtered


class PlotData(BaseModel):
    experiment: Literal["baseline", "ablation_prompt"]
    model_name: str
    variant: str | None
    roc_curves: dict[DataSource | str, list[tuple[list[float], list[float]]]]
    roc_auc_ci: dict[DataSource | str, CIResult]

    @field_validator("roc_curves", "roc_auc_ci", mode="before")
    @classmethod
    def validate_roc_curves(cls, v, info):
        if info.data.get("experiment") == "baseline":
            if isinstance(v, dict) and v:
                first_key = next(iter(v))
                if isinstance(first_key, str):
                    return {DataSource(int(k)): v for k, v in v.items()}
        return v


def get_plot_dir(sampling_directory: Path) -> Path:
    directory = config.plots_dir / "roc_curves" / sampling_directory.name
    os.makedirs(directory, exist_ok=True)
    return directory


def plot_roc_curve_with_fs_cache(plot_data: PlotData, out_dir: Path):
    os.makedirs(out_dir, exist_ok=True)
    variant_slug = slugify(_variant_preslug(plot_data.variant), max_length=200, hash_length=0)
    data_path = out_dir / f"{variant_slug}.json"
    with open(data_path, "wb") as f:
        f.write(orjson.dumps(plot_data.model_dump(mode="json"), option=orjson.OPT_NON_STR_KEYS))
    plot_roc_curve(data_path)


def plot_roc_curve(
    plot_data_path: Path,
):
    """From the path of a data file containing the necessary data, plot the ROC curves in the parent directory,
    with the same name except for the extension."""
    plot_data = PlotData.model_validate_json(plot_data_path.read_text())
    conditions = list(plot_data.roc_curves.keys())
    fig = go.Figure()

    # Define colors for each data source
    colors = px.colors.qualitative.Plotly

    for i, condition in enumerate(conditions):
        match plot_data.experiment:
            case "baseline":
                # Conditions are DataSource objects
                condition_name = condition.name
            case "ablation_prompt":
                # Conditions are prompts
                condition_name = repr(condition)

        color = colors[i % len(colors)]
        display_name = f"{condition_name} (AUC: {plot_data.roc_auc_ci[condition].avg:.4f})"

        for j, (fpr, tpr) in enumerate(plot_data.roc_curves[condition]):
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    name=display_name,
                    line=dict(color=color),
                    showlegend=(j == 0),  # Only show legend for first curve of each condition
                    legendgroup=condition_name,  # Group all curves from same condition
                )
            )

    # Random chance diagonal
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random chance",
            line=dict(dash="dash", color="black"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    variant = plot_data.variant
    title = f"ROC Curves on model {plot_data.model_name}"
    if variant:
        title += f", on variant: {_variant_preslug(variant)}"
    else:
        title += " across all variants"
    fig.update_layout(
        font_family="Spectral",
        template="plotly_white",
        title=title,
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    plot_dir = plot_data_path.parent.parent
    filename_base = plot_data_path.stem
    fig.write_html(plot_dir / f"{filename_base}.html")
    logger.info(f"Saved ROC curve to {plot_dir / f'{filename_base}.html'}")


class BAPlotData(BaseModel):
    token_count: dict[DataSource, tuple[float, float]]
    # ROC AUCs averaged across models, by variant and DataSource
    v_point_estimates: dict[Variant, dict[DataSource, float]]
    v_bootstrap_results: dict[Variant, dict[DataSource, list[float]]]
    # ROC AUCs averaged across variants and models, by DataSource
    t_point_estimates: dict[DataSource, float]
    t_bootstrap_results: dict[DataSource, list[float]]

    @field_validator(
        "token_count",
        "v_point_estimates",
        "v_bootstrap_results",
        "t_point_estimates",
        "t_bootstrap_results",
        mode="before",
    )
    @classmethod
    def validate_datasource_keys(cls, value):
        """Convert string keys back to DataSource enums during deserialization"""
        if isinstance(value, dict):
            result = {}
            for key, val in value.items():
                if isinstance(val, dict) and all(
                    isinstance(k, str) and k.isdigit() for k in val.keys()
                ):
                    # Nested dict with string/int DataSource keys
                    result[key] = {DataSource.from_str(ds): v for ds, v in val.items()}
                elif isinstance(key, str) and key.isdigit():
                    # Direct string/int DataSource key
                    result[DataSource.from_str(key)] = val
                else:
                    result[key] = val
            return result
        return value

    def sources(self) -> list[DataSource]:
        return list(self.token_count.keys())


StatsPair = tuple[list[float], list[float]]
StatsByVariantSource = dict[str, dict[DataSource, list[StatsPair]]]
StatsBySource = dict[DataSource, list[StatsPair]]


def _score_by_variant_from_stats(
    stats_by_variant_source: StatsByVariantSource,
) -> dict[str, dict[DataSource, float]]:
    """Compute AUC by variant and source from pre-extracted stats, with bootstrap sampling."""
    results = {}
    for v, source_stats in stats_by_variant_source.items():
        results[v] = {}
        for s, model_stats in source_stats.items():
            aucs = [compute_auc_from_stats(h0, h1, sampling=True) for h0, h1 in model_stats]
            results[v][s] = sum(aucs) / len(aucs)
    return results


def _score_by_variant_bootstrap_chunk(
    stats_by_variant_source: StatsByVariantSource,
    n_iterations: int,
) -> list[dict[str, dict[DataSource, float]]]:
    """Run multiple bootstrap iterations for score_by_variant."""
    return [_score_by_variant_from_stats(stats_by_variant_source) for _ in range(n_iterations)]


def _overall_score_from_stats(stats_by_source: StatsBySource) -> dict[DataSource, float]:
    """Compute overall AUC per source from pre-extracted stats, with bootstrap sampling."""
    results = {}
    for s, pair_stats in stats_by_source.items():
        aucs = [compute_auc_from_stats(h0, h1, sampling=True) for h0, h1 in pair_stats]
        results[s] = sum(aucs) / len(aucs)
    return results


def _overall_score_bootstrap_chunk(
    stats_by_source: StatsBySource,
    n_iterations: int,
) -> list[dict[DataSource, float]]:
    """Run multiple bootstrap iterations for overall_score."""
    return [_overall_score_from_stats(stats_by_source) for _ in range(n_iterations)]


def _group_results(
    results: list[SingleVariantResult],
) -> dict[str, dict[str, dict[DataSource, SingleVariantResult]]]:
    """Group results by model -> variant -> source."""
    grouped: dict[str, dict[str, dict[DataSource, SingleVariantResult]]] = {}
    for r in results:
        if r.model_name not in grouped:
            grouped[r.model_name] = {}
        if r.variant not in grouped[r.model_name]:
            grouped[r.model_name][r.variant] = {}
        grouped[r.model_name][r.variant][r.method] = r
    return grouped


class BaselineAnalysis:
    plot_dir = config.plots_dir / "paper" / "baseline"
    overall_performance_path = plot_dir / "overall_performance.json"
    plot_data_path = plot_dir / "baseline.json"

    @staticmethod
    def score_by_variant(
        grouped: dict[str, dict[str, dict[DataSource, SingleVariantResult]]],
        sources: list[DataSource],
    ) -> dict[Variant, dict[DataSource, float]]:
        """Compute AUC by variant and source, averaged across models (point estimate)."""
        models = list(grouped.keys())
        variants = list(grouped[models[0]].keys())
        results = {v: {s: 0.0 for s in sources} for v in variants}
        for v in variants:
            for s in sources:
                aucs = [
                    compute_auc_from_stats(
                        grouped[m][v][s].h0_stats.stats,
                        grouped[m][v][s].h1_stats.stats,
                        sampling=False,
                    )
                    for m in models
                ]
                results[v][s] = sum(aucs) / len(aucs)
        return results

    @staticmethod
    def score_by_variant_bootstrap(
        grouped: dict[str, dict[str, dict[DataSource, SingleVariantResult]]],
        sources: list[DataSource],
    ) -> dict[Variant, dict[DataSource, list[float]]]:
        """Bootstrap by sampling with replacement from statistics. Models fixed."""
        n_bootstrap = config.analysis.n_bootstrap
        models = list(grouped.keys())
        variants = list(grouped[models[0]].keys())

        # Extract just the stats to avoid pickling the full grouped structure
        stats_by_variant_source: StatsByVariantSource = {}
        for v in variants:
            stats_by_variant_source[v] = {}
            for s in sources:
                stats_by_variant_source[v][s] = [
                    (grouped[m][v][s].h0_stats.stats, grouped[m][v][s].h1_stats.stats)
                    for m in models
                ]

        results = {v: {s: [] for s in sources} for v in variants}
        n_workers = os.cpu_count() or 4
        chunk_size = (n_bootstrap + n_workers - 1) // n_workers

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(
                    _score_by_variant_bootstrap_chunk,
                    stats_by_variant_source,
                    min(chunk_size, n_bootstrap - i * chunk_size),
                )
                for i in range(n_workers)
                if i * chunk_size < n_bootstrap
            ]
            for future in tqdm(futures, desc="score by variant bootstrap"):
                for result in future.result():
                    for v in variants:
                        for s in sources:
                            results[v][s].append(result[v][s])
        return results

    @staticmethod
    def overall_score(
        grouped: dict[str, dict[str, dict[DataSource, SingleVariantResult]]],
        variants: list[Variant],
        sources: list[DataSource],
    ) -> dict[DataSource, float]:
        """Compute overall AUC for each source, averaged across models and variants (point estimate)."""
        models = list(grouped.keys())
        results = {s: 0.0 for s in sources}
        for s in sources:
            aucs = [
                compute_auc_from_stats(
                    grouped[m][v][s].h0_stats.stats,
                    grouped[m][v][s].h1_stats.stats,
                    sampling=False,
                )
                for m in models
                for v in variants
            ]
            results[s] = sum(aucs) / len(aucs)
        return results

    @staticmethod
    def overall_score_bootstrap(
        grouped: dict[str, dict[str, dict[DataSource, SingleVariantResult]]],
        sources: list[DataSource],
    ) -> dict[DataSource, list[float]]:
        """Bootstrap by sampling with replacement from statistics. Models and variants fixed."""
        n_bootstrap = config.analysis.n_bootstrap
        models = list(grouped.keys())
        variants = list(grouped[models[0]].keys())

        # Extract just the stats to avoid pickling the full grouped structure
        stats_by_source: StatsBySource = {}
        for s in sources:
            stats_by_source[s] = [
                (grouped[m][v][s].h0_stats.stats, grouped[m][v][s].h1_stats.stats)
                for m in models
                for v in variants
            ]

        results = {s: [] for s in sources}
        n_workers = os.cpu_count() or 4
        chunk_size = (n_bootstrap + n_workers - 1) // n_workers

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(
                    _overall_score_bootstrap_chunk,
                    stats_by_source,
                    min(chunk_size, n_bootstrap - i * chunk_size),
                )
                for i in range(n_workers)
                if i * chunk_size < n_bootstrap
            ]
            for future in tqdm(futures, desc="overall score bootstrap"):
                for result in future.result():
                    for s in sources:
                        results[s].append(result[s])
        return results

    @staticmethod
    def gen_plot_data_and_plot():
        BaselineAnalysis.gen_plot_data()
        BaselineAnalysis.plot()

    @staticmethod
    def gen_plot_data():
        sampling_dirs = [
            config.sampling_data_dir / dirname for dirname in config.analysis.sampling_dirnames
        ]
        all_sources = list(DataSource)

        all_results: list[SingleVariantResult] = []
        for source in all_sources:
            logger.info(f"Loading results for {source}...")
            all_results.extend(load_single_variant_results(sampling_dirs, source))

        all_results = filter_results_by_baseline_params(all_results)
        logger.info(f"Filtered to {len(all_results)} results matching baseline params")

        grouped = _group_results(all_results)
        models = list(grouped.keys())
        variants = list(grouped[models[0]].keys())
        sources = list(grouped[models[0]][variants[0]].keys())

        token_count: dict[DataSource, tuple[float, float]] = {}
        for s in sources:
            input_avg = sum(
                grouped[m][v][s].h0_stats.input_token_avg for m in models for v in variants
            ) / (len(models) * len(variants))
            output_avg = sum(
                grouped[m][v][s].h0_stats.output_token_avg for m in models for v in variants
            ) / (len(models) * len(variants))
            token_count[s] = (input_avg / s.hourly_samples(), output_avg / s.hourly_samples())

        v_point_estimates = BaselineAnalysis.score_by_variant(grouped, sources)
        v_bootstrap_results = BaselineAnalysis.score_by_variant_bootstrap(grouped, sources)
        t_point_estimates = BaselineAnalysis.overall_score(grouped, variants, sources)
        t_bootstrap_results = BaselineAnalysis.overall_score_bootstrap(grouped, sources)
        plot_data = BAPlotData(
            token_count=token_count,
            v_point_estimates=v_point_estimates,
            v_bootstrap_results=v_bootstrap_results,
            t_point_estimates=t_point_estimates,
            t_bootstrap_results=t_bootstrap_results,
        )
        os.makedirs(BaselineAnalysis.plot_dir, exist_ok=True)
        with open(BaselineAnalysis.plot_data_path, "wb") as f:
            f.write(orjson.dumps(plot_data.model_dump(mode="json")))

    @staticmethod
    def plot():
        with open(BaselineAnalysis.plot_data_path, "rb") as f:
            plot_data = BAPlotData.model_validate(orjson.loads(f.read()))

        BaselineAnalysis.plot_pareto()

        difficulty_scales = {
            "finetune_no_lora": {
                "title": "Finetuning",
                "match_fn": lambda v: v["type"] == "finetune" and v["lora"] is False,
                "scale_attr": "n_samples",
                "xaxis_title": "Number of steps of finetuning",
            },
            "finetune_lora": {
                "title": "LoRA finetuning",
                "match_fn": lambda v: v["type"] == "finetune" and v["lora"] is True,
                "scale_attr": "n_samples",
                "xaxis_title": "Number of steps of finetuning",
            },
            "random_noise": {
                "title": "Random noise",
                "match_fn": lambda v: v["type"] == "random_noise",
                "scale_attr": "scale",
                "xaxis_title": "Standard deviation of the gaussian noise added to each weight",
            },
            "weight_pruning_magnitude": {
                "title": "Weight pruning, selection by magnitude",
                "match_fn": lambda v: v["type"] == "weight_pruning" and v["method"] == "magnitude",
                "scale_attr": "scale",
                "xaxis_title": "Fraction of the weights to prune",
            },
            "weight_pruning_random": {
                "title": "Weight pruning, random selection",
                "match_fn": lambda v: v["type"] == "weight_pruning" and v["method"] == "random",
                "scale_attr": "scale",
                "xaxis_title": "Fraction of the weights to prune",
            },
        }

        for scale_name, scale_info in difficulty_scales.items():
            BaselineAnalysis.plot_difficulty_scale(plot_data, scale_name, scale_info)

    @staticmethod
    def plot_difficulty_scale(plot_data: BAPlotData, scale_name: str, scale_info: dict):
        logger.info(f"Plotting difficulty scale: {scale_name}...")

        source_order = [DataSource.GAO2025, DataSource.MMLU, DataSource.LT, DataSource.BI]
        sources = [s for s in source_order]
        variant_description_subset = []
        for k in plot_data.v_point_estimates.keys():
            variant = orjson.loads(k)
            if scale_info["match_fn"](variant):
                variant_description_subset.append(k)
        scale_attr = scale_info["scale_attr"]
        variant_description_subset.sort(key=lambda k: orjson.loads(k)[scale_attr], reverse=True)
        xaxis_values = [orjson.loads(k)[scale_attr] for k in variant_description_subset]
        print(xaxis_values)

        fig = go.Figure()
        for source in sources:
            y_values = [plot_data.v_point_estimates[v][source] for v in variant_description_subset]
            y_bootstrap_values = [
                plot_data.v_bootstrap_results[v][source] for v in variant_description_subset
            ]
            y_cis = [ci(b, config.analysis.results_alpha) for b in y_bootstrap_values]
            y_upper = [y_ci[1] for y_ci in y_cis]
            y_lower = [y_ci[0] for y_ci in y_cis]
            fig.add_trace(
                go.Scatter(
                    x=xaxis_values,
                    y=y_values,
                    name=config.plotting.source_name[source.value],
                    line_color=config.plotting.color_map[source.to_str()],
                )
            )
            # https://plotly.com/python/continuous-error-bars/
            fig.add_trace(
                go.Scatter(
                    x=xaxis_values + xaxis_values[::-1],
                    y=y_upper + y_lower[::-1],
                    fill="toself",
                    fillcolor=config.plotting.color_map[source.to_str()],
                    line=dict(color="rgba(255,255,255,0)"),
                    opacity=0.2,
                    showlegend=False,
                )
            )

        # Add horizontal line for random guessing
        fig.add_hline(
            y=0.5,
            line_dash="dash",
            line_color="gray",
            annotation_text="Random guessing",
            annotation_position="bottom right",
        )
        if xaxis_values[-1] < 1:
            xaxis_ticktext = [f"2<sup>{round(math.log(x, 2))}</sup>" for x in xaxis_values]
        else:
            xaxis_ticktext = xaxis_values

        fig.update_layout(
            font_family="Spectral",
            font_size=20,
            template="plotly_white",
            title=f"{scale_info['title']}",
            xaxis=dict(
                title=scale_info["xaxis_title"],
                type="log",
                autorange="reversed",
                tickmode="array",
                tickvals=xaxis_values,
                ticktext=xaxis_ticktext,
            ),
            yaxis_title="ROC AUC",
        )
        fig_path = BaselineAnalysis.plot_dir / f"{scale_name}.pdf"
        fig.write_image(fig_path)
        logger.info(f"Saved plot to {fig_path}")

    @staticmethod
    def plot_pareto():
        with open(BaselineAnalysis.plot_data_path, "rb") as f:
            plot_data = BAPlotData.model_validate(orjson.loads(f.read()))

        sources = plot_data.sources()
        logger.info(f"Pareto plot sources: {sources}")

        fig = go.Figure()

        BaselineAnalysis._add_param_sweep_points(fig)

        fig.add_hline(
            y=0.5,
            line_dash="dash",
            line_color="gray",
            annotation_text="Random guessing",
            annotation_position="bottom right",
        )

        fig.update_layout(
            font_family="Spectral",
            font_size=20,
            template="plotly_white",
            xaxis=dict(title="Cost per year ($)", type="log"),
            yaxis_title="Average ROC AUC",
            showlegend=True,
        )

        fig_path = BaselineAnalysis.plot_dir / "pareto.pdf"
        fig.write_image(fig_path)
        logger.info(f"Saved Pareto plot to {fig_path}")

    @staticmethod
    def _add_param_sweep_points(fig: go.Figure):
        sweep_configs = [
            (METParameterSweep, DataSource.GAO2025, "MET(T=1)"),
            (METT0ParameterSweep, DataSource.GAO2025_T0, "MET(T=0)"),
            (MMLUParameterSweep, DataSource.MMLU, "MMLU-ALG(T=0.1)"),
            (MMLUT0ParameterSweep, DataSource.MMLU_T0, "MMLU-ALG(T=0)"),
            (LTParameterSweep, DataSource.LT, "LT"),
            (BIParameterSweep, DataSource.BI, "B3IT (Ours)"),
        ]

        for sweep_cls, data_source, display_name in sweep_configs:
            with open(sweep_cls.plot_data_path, "rb") as f:
                plot_data = sweep_cls.load_plot_data(orjson.loads(f.read()))

            x_vals = []
            y_vals = []
            logger.info(f"\n{display_name} coordinates:")
            for result in plot_data.results:
                x_vals.append(result.yearly_cost)
                y_vals.append(result.auc)
                params = {
                    k: getattr(result, k)
                    for k in ["n_prompts", "detection_samples_per_prompt", "output_tokens"]
                    if hasattr(result, k)
                }
                logger.info(f"  x={result.yearly_cost:.2f}, y={result.auc:.4f}, {params}")

            color = config.plotting.color_map[str(data_source.value)]
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="markers",
                    name=display_name,
                    legendgroup="param-sweep",
                    marker=dict(size=8, color=color),
                )
            )


if __name__ == "__main__":
    analysis_config = config.analysis

    DEBUG = False
    if DEBUG:
        # analysis_config.n_bootstrap = 10000
        analysis_config.n_tests = 1000
        analysis_config.sampling_dirnames = [
            "keep/icml/deepseek-r1-7B",
            "keep/icml/gemma-3-1b",
            "keep/icml/llama-3.1-8B",
            "keep/icml/mistral-7B",
            "keep/icml/olmo-2-7B",
            "keep/icml/phi-4-mini",
            "keep/icml/qwen2.5-0.5B",
            "keep/icml/qwen2.5-7B",
            "keep/icml/gemma-2-9b-it",
        ]
        analysis_config.sampling_dirname = analysis_config.sampling_dirnames[0]
        analysis_config.experiment = "met_param_sweep"
        analysis_config.task = "plot"

    if analysis_config.task == "compute_stats":
        output_dir = config.sampling_data_dir / analysis_config.sampling_dirname
        compressed_output = CompressedOutput.from_json_dir(output_dir)
        logger.info(compressed_output.model_name)
        logger.info(f"number of rows: {len(compressed_output.rows)}")
        for ref_attr in compressed_output.references.__dict__.keys():
            ref = getattr(compressed_output.references, ref_attr)
            logger.info(f"length of field '{ref_attr}': {len(ref)}")

        if analysis_config.experiment == "ablation_prompt":
            PromptAblation.compute_stats(
                directory=output_dir,
                data=compressed_output,
                n_tests=analysis_config.n_tests,
                pvalue_b=analysis_config.pvalue_b,
            )
        elif analysis_config.experiment == "bi_param_sweep":
            BIParameterSweep.compute_stats(
                directory=output_dir,
                data=compressed_output,
                n_prompts_values=analysis_config.bi_n_prompts_values,
                detection_samples_values=analysis_config.bi_detection_samples_values,
                reference_samples_per_prompt=config.sampling.bi.reference_samples_per_prompt,
                n_tests=analysis_config.n_tests,
            )
        elif analysis_config.experiment == "met_param_sweep":
            METParameterSweep.compute_stats(
                directory=output_dir,
                data=compressed_output,
                n_prompts_values=analysis_config.met_n_prompts_values,
                output_tokens_values=analysis_config.met_output_tokens_values,
                reference_samples_per_prompt=config.sampling.gao2025.reference_samples_per_prompt,
                detection_samples_per_prompt=analysis_config.get_detection_samples("GAO2025"),
                n_tests=analysis_config.n_tests,
            )
        elif analysis_config.experiment == "mmlu_param_sweep":
            MMLUParameterSweep.compute_stats(
                directory=output_dir,
                data=compressed_output,
                n_prompts_values=analysis_config.mmlu_n_prompts_values,
                reference_samples_per_prompt=config.sampling.mmlu.reference_samples_per_prompt,
                detection_samples_per_prompt=analysis_config.get_detection_samples("MMLU"),
                n_tests=analysis_config.n_tests,
            )
        elif analysis_config.experiment == "met_t0_param_sweep":
            METT0ParameterSweep.compute_stats(
                directory=output_dir,
                data=compressed_output,
                n_prompts_values=analysis_config.met_t0_n_prompts_values,
                output_tokens_values=analysis_config.met_t0_output_tokens_values,
            )
        elif analysis_config.experiment == "mmlu_t0_param_sweep":
            MMLUT0ParameterSweep.compute_stats(
                directory=output_dir,
                data=compressed_output,
                n_prompts_values=analysis_config.mmlu_t0_n_prompts_values,
            )
        elif analysis_config.experiment == "lt_param_sweep":
            LTParameterSweep.compute_stats(
                directory=output_dir,
                data=compressed_output,
                n_samples_values=analysis_config.lt_n_samples_values,
                reference_samples_per_prompt=config.sampling.logprob.reference_samples_per_prompt,
                n_tests=analysis_config.n_tests,
            )

    elif analysis_config.task == "plot":
        if analysis_config.experiment == "ablation_prompt":
            PromptAblation.gen_plot_data_and_plot()
        elif analysis_config.experiment == "baseline":
            BaselineAnalysis.gen_plot_data_and_plot()
        elif analysis_config.experiment == "bi_param_sweep":
            BIParameterSweep.gen_plot_data_and_plot()
        elif analysis_config.experiment == "met_param_sweep":
            METParameterSweep.gen_plot_data_and_plot()
        elif analysis_config.experiment == "mmlu_param_sweep":
            MMLUParameterSweep.gen_plot_data_and_plot()
        elif analysis_config.experiment == "met_t0_param_sweep":
            METT0ParameterSweep.gen_plot_data_and_plot()
        elif analysis_config.experiment == "mmlu_t0_param_sweep":
            MMLUT0ParameterSweep.gen_plot_data_and_plot()
        elif analysis_config.experiment == "lt_param_sweep":
            LTParameterSweep.gen_plot_data_and_plot()

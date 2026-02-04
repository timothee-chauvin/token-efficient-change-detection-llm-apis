import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Literal

import orjson
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pydantic import BaseModel
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from track_llm_apis.config import config, logger
from track_llm_apis.sampling.analyze_logprobs import (
    logprob_two_sample_test_from_compressed_output_row,
)
from track_llm_apis.sampling.common import (
    CompressedOutput,
    CompressedOutputRow,
    Condition,
    DataSource,
    TwoSampleMultiTestResult,
    Variant,
)
from track_llm_apis.tinychange import TinyChange


class AnalysisResult(BaseModel):
    """Legacy class for prompt ablation analysis. Not for use in new code."""

    experiment: Literal["baseline", "ablation_prompt"]
    model_name: str
    n_tests: int
    pvalue_b: int

    h0_stats: dict[Condition, TwoSampleMultiTestResult] = {}
    variants: dict[Variant, dict[Condition, TwoSampleMultiTestResult]] = {}

    def _token_avg(self, attr: str) -> dict[Condition, float]:
        result = {}
        for condition in self.conditions:
            result[condition] = sum(
                getattr(self.variants[variant][condition], attr) for variant in self.variants
            ) / len(self.variants)
        return result

    @property
    def input_token_avg(self) -> dict[Condition, float]:
        return self._token_avg("input_token_avg")

    @property
    def output_token_avg(self) -> dict[Condition, float]:
        return self._token_avg("output_token_avg")

    @property
    def conditions(self) -> list[Condition]:
        return list(self.h0_stats.keys())

    @property
    def variant_names(self) -> list[Variant]:
        return list(self.variants.keys())

    def avg_auc_across_variants(
        self, sampling: bool = False, centered: bool = False
    ) -> dict[Condition, float]:
        """If centered is True, subtract the average to the AUC of each condition.
        If sampling is True, sample with replacement from each list of statistics.
        """
        result = {}
        for condition in self.conditions:
            result[condition] = sum(
                self.auc(variant=variant, condition=condition, sampling=sampling)
                for variant in self.variants
            ) / len(self.variants)

        if centered:
            avg_auc = sum(result.values()) / len(result)
            result = {condition: result[condition] - avg_auc for condition in result}

        return result

    def auc(self, variant: Variant, condition: Condition, sampling: bool) -> float:
        h0_stats = self.h0_stats[condition].stats
        variant_stats = self.variants[variant][condition].stats
        if sampling:
            h0_stats = random.choices(h0_stats, k=len(h0_stats))
            variant_stats = random.choices(variant_stats, k=len(variant_stats))
        y_true = [0] * len(h0_stats) + [1] * len(variant_stats)
        y_pred = h0_stats + variant_stats
        return roc_auc_score(y_true, y_pred)

    @staticmethod
    def multianalysis_input_token_avg(analyses: list["AnalysisResult"]) -> dict[Condition, float]:
        return {
            condition: sum(analysis.input_token_avg[condition] for analysis in analyses)
            / len(analyses)
            for condition in analyses[0].conditions
        }

    @staticmethod
    def multianalysis_output_token_avg(analyses: list["AnalysisResult"]) -> dict[Condition, float]:
        return {
            condition: sum(analysis.output_token_avg[condition] for analysis in analyses)
            / len(analyses)
            for condition in analyses[0].conditions
        }


def evaluate_lt_prompt(
    rows1: dict[str, list[CompressedOutputRow]],
    rows2: dict[str, list[CompressedOutputRow]],
    same: bool,
    prompt_length: dict[str, int],
    logprob_prompt: str,
    n_tests: int,
    pvalue_b: int,
) -> TwoSampleMultiTestResult:
    """Evaluate the logprob detector on a single prompt."""
    logger.info(f"Evaluating prompt {repr(logprob_prompt)}...")
    samples_per_prompt = config.analysis.get_detection_samples("LT")

    rows1 = {logprob_prompt: rows1[logprob_prompt]}
    rows2 = {logprob_prompt: rows2[logprob_prompt]}

    token_count = {"i": [], "o": []}
    stats = []
    pvalues = []
    for _ in range(n_tests):
        if same:
            subset = {p: random.sample(rows, 2 * samples_per_prompt) for p, rows in rows1.items()}
            sample1 = {p: subset[p][:samples_per_prompt] for p in rows1.keys()}
            sample2 = {p: subset[p][samples_per_prompt:] for p in rows2.keys()}
        else:
            sample1 = {p: random.sample(rows, samples_per_prompt) for p, rows in rows1.items()}
            sample2 = {p: random.sample(rows, samples_per_prompt) for p, rows in rows2.items()}

        result = logprob_two_sample_test_from_compressed_output_row(
            sample1, sample2, pvalue_b=pvalue_b
        )
        stats.append(result.statistic)
        if pvalue_b > 0:
            pvalues.append(result.pvalue)
        token_count["i"].append(
            sum(prompt_length[p] * len(r) for p, r in sample1.items())
            + sum(prompt_length[p] * len(r) for p, r in sample2.items())
        )
        token_count["o"].append(
            sum(r.text[1] for rows in sample1.values() for r in rows)
            + sum(r.text[1] for rows in sample2.values() for r in rows)
        )

    return TwoSampleMultiTestResult(
        stats=stats,
        pvalues=pvalues if pvalue_b > 0 else None,
        input_token_avg=sum(token_count["i"]) / len(token_count["i"]),
        output_token_avg=sum(token_count["o"]) / len(token_count["o"]),
    )


class PromptAblation:
    stats_filename = "prompt_ablation_analysis.json"
    plot_data_path = config.plots_dir / "paper" / "prompt_ablation.json"
    plot_path = config.plots_dir / "paper" / "prompt_ablation.html"

    @staticmethod
    def compute_stats(
        directory: Path,
        data: CompressedOutput,
        n_tests: int = 1000,
        pvalue_b: int = 1000,
    ):
        """Test the influence of the prompt choice on detection performance for the logprob method."""
        filtered_data = data.filter(datasource=DataSource.LT)
        rows_by_variant = filtered_data.get_rows_by_variant()
        unchanged_rows = rows_by_variant[TinyChange.unchanged_str()]
        unchanged_rows_by_prompt = filtered_data.get_rows_by_prompt(unchanged_rows)

        prompt_length = {
            prompt: tokens
            for prompt, tokens in filtered_data.references.prompts.keys()
            if prompt in unchanged_rows_by_prompt
        }
        prompts = list(prompt_length.keys())
        analysis_results = AnalysisResult(
            experiment="ablation_prompt",
            model_name=data.model_name,
            n_tests=n_tests,
            pvalue_b=pvalue_b,
        )
        variants = [v for v in data.references.variants.keys() if v != TinyChange.unchanged_str()]
        # Initialize H0 stats with original/original comparisons
        analysis_results.h0_stats = {
            prompt: evaluate_lt_prompt(
                unchanged_rows_by_prompt,
                unchanged_rows_by_prompt,
                same=True,
                prompt_length=prompt_length,
                logprob_prompt=prompt,
                n_tests=n_tests,
                pvalue_b=pvalue_b,
            )
            for prompt in prompts
        }
        for variant_idx, variant in enumerate(variants):
            logger.info(f"Variant {variant_idx + 1}/{len(variants)}: {variant}")
            rows_by_prompt = filtered_data.get_rows_by_prompt(rows_by_variant[variant])
            analysis_results.variants[variant] = {}
            for prompt in prompts:
                # H1: original vs variant
                analysis_results.variants[variant][prompt] = evaluate_lt_prompt(
                    unchanged_rows_by_prompt,
                    rows_by_prompt,
                    same=False,
                    prompt_length=prompt_length,
                    logprob_prompt=prompt,
                    n_tests=n_tests,
                    pvalue_b=pvalue_b,
                )
                # H0: variant vs variant
                variant_h0 = evaluate_lt_prompt(
                    rows_by_prompt,
                    rows_by_prompt,
                    same=True,
                    prompt_length=prompt_length,
                    logprob_prompt=prompt,
                    n_tests=n_tests,
                    pvalue_b=pvalue_b,
                )
                analysis_results.h0_stats[prompt] = analysis_results.h0_stats[prompt].merge(
                    variant_h0
                )
            with open(directory / PromptAblation.stats_filename, "wb") as f:
                f.write(orjson.dumps(analysis_results.model_dump(mode="json")))

    @staticmethod
    def compute_prompt_score(analyses: list[AnalysisResult], sampling: bool) -> dict[str, float]:
        """For each prompt, return the average over analyses (models) of its AUC minus the model average."""
        prompts = list(analyses[0].h0_stats.keys())
        scores = []
        for analysis in analyses:
            scores.append(analysis.avg_auc_across_variants(sampling=sampling, centered=True))
        return {prompt: sum(score[prompt] for score in scores) / len(scores) for prompt in prompts}

    @staticmethod
    def bootstrap_iteration(analyses: list[AnalysisResult]):
        """Single bootstrap iteration"""
        analyses_bootstrap = random.choices(analyses, k=len(analyses))
        return PromptAblation.compute_prompt_score(analyses_bootstrap, sampling=True)

    @staticmethod
    def compute_prompt_score_bootstrap(analyses: list[AnalysisResult]) -> dict[str, list[float]]:
        """Bootstrap by sampling with replacement both from the analyses, then from the statistics for each analysis."""
        n_bootstrap = config.analysis.n_bootstrap
        prompts = list(analyses[0].h0_stats.keys())
        results = {prompt: [] for prompt in prompts}

        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(PromptAblation.bootstrap_iteration, analyses)
                for _ in range(n_bootstrap)
            ]
            for future in tqdm(futures, desc="bootstrap"):
                result = future.result()
                for prompt, score in result.items():
                    results[prompt].append(score)

        return results

    @staticmethod
    def gen_plot_data_and_plot():
        PromptAblation.gen_plot_data()
        PromptAblation.plot()

    @staticmethod
    def gen_plot_data():
        sampling_dirs = [
            config.sampling_data_dir / dirname for dirname in config.analysis.sampling_dirnames
        ]
        analyses = []
        for sampling_dir in sampling_dirs:
            p = Path(sampling_dir) / PromptAblation.stats_filename
            with open(p) as f:
                analyses.append(AnalysisResult.model_validate(orjson.loads(f.read())))

        prompt_length_avg = AnalysisResult.multianalysis_input_token_avg(analyses)
        for prompt in prompt_length_avg.keys():
            prompt_length_avg[prompt] /= 2 * config.analysis.get_detection_samples("LT")

        point_estimates = PromptAblation.compute_prompt_score(analyses, sampling=False)

        bootstrap_results = PromptAblation.compute_prompt_score_bootstrap(analyses)

        all_results = {
            "prompt_length_avg": prompt_length_avg,
            "point_estimates": point_estimates,
            "bootstrap_results": bootstrap_results,
        }
        plot_data_path = PromptAblation.plot_data_path
        with open(plot_data_path, "wb") as f:
            f.write(orjson.dumps(all_results))
        logger.info(f"Prompt ablation analysis data saved to {plot_data_path}")

    @staticmethod
    def plot():
        with open(PromptAblation.plot_data_path, "rb") as f:
            all_results = orjson.loads(f.read())
        prompt_length_avg = all_results["prompt_length_avg"]
        bootstrap_results = all_results["bootstrap_results"]

        prompt_legend_names = {
            "Let's generate random words! Only output the words, no other text. Continue the list: Underpay\nPolicy\nRisotto\nIdealist": "Let's generate random words! [...] Underpay\nPolicy\nRisotto\nIdealist",
            "Let's generate random words! Only output the words, no other text. Continue the list: Sinuous\nCornbread\nStipulate\nOverreact": "Let's generate random words! [...] Sinuous\nCornbread\nStipulate\nOverreact",
        }

        def get_display_name(prompt):
            return prompt_legend_names.get(prompt, prompt)

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        sorted_prompts = sorted(bootstrap_results.keys(), key=lambda x: prompt_length_avg[x])

        # Violin plots for AUC advantages
        for prompt in sorted_prompts:
            scores = bootstrap_results[prompt]
            fig.add_trace(
                go.Violin(
                    x=[get_display_name(prompt)] * len(scores),
                    y=scores,
                    name=get_display_name(prompt),
                    box_visible=True,
                    points="all",
                    showlegend=True,
                    legend="legend",
                ),
                secondary_y=False,
            )

        # Scatter plot for prompt lengths
        fig.add_trace(
            go.Scatter(
                x=[get_display_name(prompt) for prompt in sorted_prompts],
                y=[prompt_length_avg[prompt] for prompt in sorted_prompts],
                mode="markers+lines+text",
                name="Average Token Length of Prompts Across Models",
                marker=dict(size=10, color="black", line_width=2),
                text=[f"{prompt_length_avg[prompt]:.1f}" for prompt in sorted_prompts],
                textposition="bottom right",
                textfont=dict(size=40),
                showlegend=True,
                legend="legend2",
            ),
            secondary_y=True,
        )

        fig.update_layout(
            font_family="Spectral",
            font_size=40,
            template="plotly_white",
            xaxis_showticklabels=False,
            legend=dict(
                x=0.05,
                y=1.1,
                xanchor="left",
                yanchor="top",
                traceorder="reversed",
                bgcolor="rgba(255,255,255,0)",
            ),
            legend2=dict(x=0.5, y=0.1, xanchor="left", yanchor="bottom"),
        )

        fig.update_yaxes(
            title_text="Overall AUC Advantage", tickformat=".0%", dtick=0.01, secondary_y=False
        )
        fig.update_yaxes(showgrid=False, showticklabels=False, secondary_y=True)

        plot_path = PromptAblation.plot_path
        # Something in this figure triggers a plotly.js bug when trying to save as pdf or svg.
        fig.write_html(plot_path)
        logger.info(f"Prompt ablation analysis plot saved to {plot_path}")

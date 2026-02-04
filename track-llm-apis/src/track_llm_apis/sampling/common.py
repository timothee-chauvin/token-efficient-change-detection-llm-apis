import gzip
import json
import os
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Self

import numpy as np
import orjson
import rapidgzip
from pydantic import BaseModel
from sklearn.metrics import roc_auc_score, roc_curve
from vllm import RequestOutput

from track_llm_apis.config import config
from track_llm_apis.util import slugify

logger = config.logger

# Type aliases for clarity
Condition = str
Variant = str
ROCCurve = tuple[list[float], list[float]]


def get_methods_suffix(methods: list[str] | None) -> str:
    """Return a suffix string for output filenames based on methods being run."""
    if methods is None:
        return ""
    return "_" + "-".join(sorted(methods))


class DataSource(Enum):
    LT = 0
    MMLU = 1
    GAO2025 = 2
    BI = 3
    GAO2025_T0 = 4
    MMLU_T0 = 5

    @classmethod
    def from_name(cls, name: str) -> "DataSource":
        return cls[name]

    @classmethod
    def all_names(cls) -> list[str]:
        return [ds.name for ds in cls]

    def get_config(self) -> BaseModel:
        return {
            DataSource.LT: config.sampling.logprob,
            DataSource.MMLU: config.sampling.mmlu,
            DataSource.MMLU_T0: config.sampling.mmlu,
            DataSource.GAO2025: config.sampling.gao2025,
            DataSource.GAO2025_T0: config.sampling.gao2025,
            DataSource.BI: config.sampling.bi,
        }[self]

    def hourly_samples(self) -> int:
        """Number of detection samples collected per hour (for cost calculation).

        Returns the baseline detection samples for this source from AnalysisConfig.
        """
        return config.analysis.get_detection_samples(self.name)

    def to_str(self) -> str:
        return str(self.value)

    @classmethod
    def from_str(cls, s: str) -> Self:
        """s contains the integer value of the DataSource"""
        return cls(int(s))


class References:
    def __init__(self):
        # Dictionaries mapping element => index of the element in the dictionary
        # which preserves insertion order
        self.variants: dict[str, int] = {}
        # (prompt, input_tokens)
        self.prompts: dict[tuple[str, int], int] = {}
        # (text, output_tokens)
        self.texts: dict[tuple[str, int], int] = {}
        # logprobs are a list of dictionaries mapping tokens to floats.
        # They are stored here in a JSON string representation,
        # in order to be used as dictionary keys.
        self.logprobs: dict[str, int] = {}
        # Cache for the ordered keys of the dictionaries, for lookup by index
        self._cache = {}

    def to_json(self) -> dict[str, Any]:
        return {
            "variants": list(self.variants.keys()),
            "prompts": list(self.prompts.keys()),
            "texts": list(self.texts.keys()),
            "logprobs": list(self.logprobs.keys()),
        }

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> Self:
        instance = cls.__new__(cls)
        instance.variants = {variant: i for i, variant in enumerate(json_data["variants"])}
        instance.prompts = {tuple(prompt): i for i, prompt in enumerate(json_data["prompts"])}
        instance.texts = {tuple(text): i for i, text in enumerate(json_data["texts"])}
        instance.logprobs = {
            logprobs if isinstance(logprobs, str) else json.dumps(logprobs): i
            for i, logprobs in enumerate(json_data["logprobs"])
        }
        instance._cache = {}
        return instance

    def _get_keys(self, attr_name: str) -> list[Any]:
        if attr_name not in self._cache:
            self._cache[attr_name] = list(getattr(self, attr_name).keys())
        return self._cache[attr_name]

    def _invalidate_cache(self, attr_name: str):
        self._cache.pop(attr_name, None)

    def get_variant(self, variant_idx: int) -> str:
        return self._get_keys("variants")[variant_idx]

    def get_prompt(self, prompt_idx: int) -> tuple[str, int]:
        return self._get_keys("prompts")[prompt_idx]

    def get_text(self, text_idx: int) -> tuple[str, int]:
        return self._get_keys("texts")[text_idx]

    def get_logprobs(self, logprobs_idx: int) -> list[dict[int, float]]:
        raw = json.loads(self._get_keys("logprobs")[logprobs_idx])
        return [{int(k): v for k, v in lp.items()} for lp in raw]

    def add_variant(self, variant: str) -> int:
        if variant not in self.variants:
            self.variants[variant] = len(self.variants)
            self._invalidate_cache("variants")
        return self.variants[variant]

    def add_prompt(self, prompt: tuple[str, int]) -> int:
        if prompt not in self.prompts:
            self.prompts[prompt] = len(self.prompts)
            self._invalidate_cache("prompts")
        return self.prompts[prompt]

    def add_text(self, text: tuple[str, int]) -> int:
        if text not in self.texts:
            self.texts[text] = len(self.texts)
            self._invalidate_cache("texts")
        return self.texts[text]

    def add_logprobs(self, logprobs: list[dict[int, float]]) -> int:
        logprobs_str = json.dumps(logprobs)
        if logprobs_str not in self.logprobs:
            self.logprobs[logprobs_str] = len(self.logprobs)
            self._invalidate_cache("logprobs")
        return self.logprobs[logprobs_str]


class CompressedOutputRow:
    def __init__(
        self,
        references: References,
        source: int,
        variant_idx: int,
        prompt_idx: int,
        text_idx: int,
        logprobs_idx: int | None = None,
    ):
        """Initialization from indices"""
        self.references = references
        self.source = source
        self.variant_idx = variant_idx
        self.prompt_idx = prompt_idx
        self.text_idx = text_idx
        self.logprobs_idx = logprobs_idx

    @classmethod
    def from_values(
        cls,
        references: References,
        source: int,
        variant: str,
        prompt: tuple[str, int],
        text: tuple[str, int],
        logprobs: list[dict[int, float]] | None = None,
    ):
        """Initialization from values, adding to the references if necessary."""
        instance = cls.__new__(cls)
        instance.references = references
        instance.source = source
        instance.variant_idx = references.add_variant(variant)
        instance.prompt_idx = references.add_prompt(prompt)
        instance.text_idx = references.add_text(text)
        instance.logprobs_idx = references.add_logprobs(logprobs) if logprobs is not None else None
        return instance

    def to_json(self) -> tuple[int, int, int, int, int | None]:
        return (self.source, self.variant_idx, self.prompt_idx, self.text_idx, self.logprobs_idx)

    @classmethod
    def from_json(cls, references: References, json_data: Sequence[int | None]) -> Self:
        """This function assumes that the references are already up-to-date."""
        return cls(
            references=references,
            source=json_data[0],
            variant_idx=json_data[1],
            prompt_idx=json_data[2],
            text_idx=json_data[3],
            logprobs_idx=json_data[4],
        )

    @property
    def variant(self) -> str:
        return self.references.get_variant(self.variant_idx)

    @property
    def prompt(self) -> tuple[str, int]:
        return self.references.get_prompt(self.prompt_idx)

    @property
    def text(self) -> tuple[str, int]:
        return self.references.get_text(self.text_idx)

    @property
    def logprobs(self) -> list[dict[int, float]] | None:
        if self.logprobs_idx is None:
            return None
        return self.references.get_logprobs(self.logprobs_idx)

    @property
    def first_token_logprobs(self) -> dict[int, float]:
        """
        Return the logprob dictionary for the first token of the output.

        If there are no logprobs, raise a ValueError.
        """
        if self.logprobs_idx is None:
            raise ValueError("logprobs_idx is None")
        logprobs_list = self.references.get_logprobs(self.logprobs_idx)
        return logprobs_list[0]


class CompressedOutput:
    def __init__(self, model_name: str, gpus: list[str] | None = None):
        self.model_name: str = model_name
        # GPUs used during sampling
        self.gpus: list[str] | None = gpus
        self.rows: list[CompressedOutputRow] = []
        self.references: References = References()

    def add_batch_from_request_output(
        self, request_output: RequestOutput, variant: str, source: DataSource
    ):
        prompt_length = (
            len(request_output.prompt_token_ids)
            if request_output.prompt_token_ids is not None
            else 0
        )
        prompt = (request_output.prompt or "", prompt_length)
        for output in request_output.outputs:
            if output.logprobs is None:
                logprobs_dicts = None
            else:
                logprobs_dicts = [
                    {int(k): v.logprob for k, v in logprobs.items()} for logprobs in output.logprobs
                ]
            self.rows.append(
                CompressedOutputRow.from_values(
                    references=self.references,
                    source=source.value,
                    variant=variant,
                    prompt=prompt,
                    text=(output.text, len(output.token_ids)),
                    logprobs=logprobs_dicts,
                )
            )

    def add_bi_results(
        self,
        counts: dict[str, dict[str, int]],
        prompt_lengths: dict[str, int],
        variant: str,
        output_tokens: int,
    ):
        """Add BI results from vllm_inference_bi output.

        Args:
            counts: dict mapping input_token -> {output_token -> count}
            prompt_lengths: dict mapping input_token -> prompt token count (after chat template)
            variant: variant name
            output_tokens: number of tokens generated (for cost tracking; only last token is stored)
        """
        for input_token, output_counts in counts.items():
            prompt_length = prompt_lengths[input_token]
            for output_token, count in output_counts.items():
                for _ in range(count):
                    self.rows.append(
                        CompressedOutputRow.from_values(
                            references=self.references,
                            source=DataSource.BI.value,
                            variant=variant,
                            prompt=(input_token, prompt_length),
                            text=(output_token, output_tokens),
                            logprobs=None,
                        )
                    )

    def to_json(self) -> dict[str, Any]:
        return {
            "version": 1,
            "model_name": self.model_name,
            "gpus": self.gpus,
            "rows": [row.to_json() for row in self.rows],
            "references": self.references.to_json(),
        }

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> Self:
        if "version" not in json_data.keys():
            # v0
            logger.info("Loading compressed output v0...")
            return cls._from_json_v0(json_data)
        if json_data["version"] == 1:
            logger.info("Loading compressed output v1...")
            return cls._from_json_v1(json_data)
        raise ValueError(f"Unknown version: {json_data['version']}")

    @classmethod
    def _from_json_v0(cls, json_data: dict[str, Any]) -> Self:
        json_references = json_data["references"]
        references = References.__new__(References)
        references.variants = {variant: i for i, variant in enumerate(json_references["variant"])}
        references.prompts = {
            tuple(prompt): i for i, prompt in enumerate(json_references["prompt"])
        }
        references.texts = {tuple(text): i for i, text in enumerate(json_references["text"])}
        references.logprobs = {
            json.dumps(logprobs): i for i, logprobs in enumerate(json_references["logprobs"])
        }
        references._cache = {}

        result = cls.__new__(cls)
        result.model_name = json_data["model_name"]
        result.gpus = json_data.get("gpus", None)
        result.rows = [CompressedOutputRow.from_json(references, row) for row in json_data["rows"]]
        result.references = references
        return result

    @classmethod
    def _from_json_v1(cls, json_data: dict[str, Any]) -> Self:
        references = References.from_json(json_data["references"])
        result = cls.__new__(cls)
        result.model_name = json_data["model_name"]
        result.gpus = json_data.get("gpus", None)
        result.rows = [CompressedOutputRow.from_json(references, row) for row in json_data["rows"]]
        result.references = references
        return result

    def dump_json(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        json_filename = f"{slugify(self.model_name, max_length=200, hash_length=0)}.json.gz"
        json_path = output_dir / json_filename
        json_dict = self.to_json()
        with gzip.open(json_path, "wb") as f:
            f.write(orjson.dumps(json_dict, option=orjson.OPT_NON_STR_KEYS))

    @classmethod
    def from_json_dir(cls, json_dir: Path) -> Self:
        logger.info(f"Searching for a .json.gz file in {json_dir}...")
        # Find the only .json.gz file in the directory
        json_paths = list(json_dir.glob("*.json.gz"))
        if len(json_paths) != 1:
            raise ValueError(f"Expected 1 .json.gz file in {json_dir}, got {len(json_paths)}")
        json_path = json_paths[0]
        logger.info(f"Decompressing {json_path}...")
        with rapidgzip.open(json_path, parallelization=os.cpu_count()) as f:
            logger.info("Loading JSON data from contents of the file...")
            json_dict = orjson.loads(f.read())
        return cls.from_json(json_dict)

    def filter(self, datasource: DataSource, keep_references: bool = True) -> Self:
        """Return a new CompressedOutput with only the rows for the given datasource.
        If keep_references is True, the References object will be kept in the new CompressedOutput, and potentially shared by multiple CompressedOutput objects. Filtering will be faster.
        """
        if keep_references:
            new_compressed_output = self.__class__(model_name=self.model_name, gpus=self.gpus)
            new_compressed_output.references = self.references
            for row in self.rows:
                if row.source == datasource.value:
                    new_compressed_output.rows.append(row)
            return new_compressed_output
        else:
            new_compressed_output = self.__class__(model_name=self.model_name, gpus=self.gpus)
            for row in self.rows:
                if row.source == datasource.value:
                    new_compressed_output.rows.append(
                        CompressedOutputRow.from_values(
                            references=new_compressed_output.references,
                            source=row.source,
                            variant=row.variant,
                            prompt=row.prompt,
                            text=row.text,
                            logprobs=row.logprobs,
                        )
                    )
            return new_compressed_output

    def get_rows_by_variant(self) -> dict[str, list[CompressedOutputRow]]:
        rows_by_variant = defaultdict(list)
        for row in self.rows:
            rows_by_variant[row.variant].append(row)
        return dict(rows_by_variant)

    @staticmethod
    def get_rows_by_prompt(rows: list[CompressedOutputRow]) -> dict[str, list[CompressedOutputRow]]:
        rows_by_prompt = defaultdict(list)
        for row in rows:
            rows_by_prompt[row.prompt[0]].append(row)
        return dict(rows_by_prompt)

    def merge(self, other: "CompressedOutput") -> "CompressedOutput":
        """Merge another CompressedOutput into this one.

        The other CompressedOutput should have the same model_name.
        Rows from `other` are re-indexed to use this instance's References.
        """
        if self.model_name != other.model_name:
            raise ValueError(
                f"Cannot merge: model names differ ({self.model_name} vs {other.model_name})"
            )
        for row in other.rows:
            self.rows.append(
                CompressedOutputRow.from_values(
                    references=self.references,
                    source=row.source,
                    variant=row.variant,
                    prompt=row.prompt,
                    text=row.text,
                    logprobs=row.logprobs,
                )
            )
        return self


class CIResult(BaseModel):
    lower: float
    avg: float
    upper: float

    def __str__(self) -> str:
        return f"({self.lower}, {self.avg}, {self.upper})"


class TwoSampleTestResult(BaseModel):
    """Result of a single two-sample test, that returns a single statistic and a p-value."""

    pvalue: float | None = None
    statistic: float


class TwoSampleTestResultWithDate(BaseModel):
    """Like TwoSampleTestResult, but with the date."""

    date: datetime
    pvalue: float | None = None
    statistic: float


class TwoSampleMultiTestResult(BaseModel):
    """Result of multiple two-sample tests on the same variant."""

    stats: list[float]
    pvalues: list[float] | None = None
    input_token_avg: float
    output_token_avg: float

    def power(self, alpha: float) -> list[float]:
        if self.pvalues is None:
            raise ValueError("pvalues is None, can't compute power")
        return sum(pvalue < alpha for pvalue in self.pvalues) / len(self.pvalues)

    def roc_curve(self, orig: "TwoSampleMultiTestResult") -> ROCCurve:
        y_true = [0] * len(orig.stats) + [1] * len(self.stats)
        y_pred = orig.stats + self.stats
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        return list(fpr), list(tpr)

    @property
    def stat_avg(self) -> float:
        return sum(self.stats) / len(self.stats)

    @property
    def pvalue_avg(self) -> float:
        if self.pvalues is None:
            raise ValueError("pvalues is None, can't compute pvalue_avg")
        return sum(self.pvalues) / len(self.pvalues)

    def merge(self, other: "TwoSampleMultiTestResult") -> "TwoSampleMultiTestResult":
        """Merge two TwoSampleMultiTestResult objects, concatenating stats and pvalues."""
        if self.pvalues is None and other.pvalues is None:
            merged_pvalues = None
        elif self.pvalues is not None and other.pvalues is not None:
            merged_pvalues = self.pvalues + other.pvalues
        else:
            raise ValueError("Cannot merge: one has pvalues, the other doesn't")

        total_stats = len(self.stats) + len(other.stats)
        return TwoSampleMultiTestResult(
            stats=self.stats + other.stats,
            pvalues=merged_pvalues,
            input_token_avg=(
                self.input_token_avg * len(self.stats) + other.input_token_avg * len(other.stats)
            )
            / total_stats,
            output_token_avg=(
                self.output_token_avg * len(self.stats) + other.output_token_avg * len(other.stats)
            )
            / total_stats,
        )

    @staticmethod
    def multivariant_roc(
        orig: "TwoSampleMultiTestResult", variants: list["TwoSampleMultiTestResult"]
    ) -> tuple[np.ndarray, np.ndarray]:
        y_true = [0] * len(orig.stats) + sum([[1] * len(variant.stats) for variant in variants], [])
        y_pred = orig.stats + [stat for variant in variants for stat in variant.stats]
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        return fpr, tpr

    @staticmethod
    def multivariant_roc_auc(
        orig: "TwoSampleMultiTestResult", variants: list["TwoSampleMultiTestResult"]
    ) -> float:
        y_true = [0] * len(orig.stats) + sum([[1] * len(variant.stats) for variant in variants], [])
        y_pred = orig.stats + [stat for variant in variants for stat in variant.stats]
        return roc_auc_score(y_true, y_pred)


class SingleVariantResult(BaseModel):
    """Result for a single variant and method combination."""

    model_name: str
    variant: str  # The variant JSON string (e.g., '{"type": "finetune", "lora": true, ...}')
    method: DataSource
    n_tests: int
    pvalue_b: int
    h0_stats: TwoSampleMultiTestResult  # original/original + variant/variant merged
    h1_stats: TwoSampleMultiTestResult  # original vs variant
    params: dict[str, int | float] | None = None  # For sweeps


def get_method_dir_name(source: DataSource) -> str:
    """Returns directory name for a method (bi, met, met_t0, mmlu, mmlu_t0, lt)."""
    return {
        DataSource.BI: "bi",
        DataSource.GAO2025: "met",
        DataSource.GAO2025_T0: "met_t0",
        DataSource.MMLU: "mmlu",
        DataSource.MMLU_T0: "mmlu_t0",
        DataSource.LT: "lt",
    }[source]


def get_variant_filename(variant: str, params: dict[str, int | float] | None = None) -> str:
    """Generate a filename for a variant (and optional params) analysis result.

    Args:
        variant: The variant JSON string
        params: Optional sweep parameters to include in filename

    Returns:
        Filename like 'finetune,lora=true,n_samples=16.json' or
        'finetune,lora=true,n_samples=16,n_prompts=5,detection_samples=10.json'
    """
    variant_slug = _variant_preslug(variant)
    if params:
        params_str = ",".join(f"{k}={v}" for k, v in sorted(params.items()))
        return f"{variant_slug},{params_str}.json"
    return f"{variant_slug}.json"


def get_analysis_dir(sampling_dir: Path, source: DataSource) -> Path:
    """Get the analysis directory for a specific method."""
    return sampling_dir / "analysis" / get_method_dir_name(source)


def _variant_preslug(variant: str | None) -> str:
    if variant is None:
        return "all"
    variant_metadata = orjson.loads(variant)
    variant_preslug = (
        variant_metadata["type"]
        + ","
        + ",".join(f"{k}={v}" for k, v in variant_metadata.items() if k != "type")
    )
    return variant_preslug


def load_single_variant_results(
    sampling_dirs: list[Path], source: DataSource
) -> list[SingleVariantResult]:
    """Load all SingleVariantResult files from analysis directories across multiple sampling dirs."""
    results = []
    for sampling_dir in sampling_dirs:
        logger.info(f"[{source}] Loading results from {sampling_dir}...")
        analysis_dir = get_analysis_dir(sampling_dir, source)
        if not analysis_dir.exists():
            logger.warning(f"Analysis directory not found: {analysis_dir}")
            continue
        for json_path in analysis_dir.glob("*.json"):
            with open(json_path, "rb") as f:
                result = SingleVariantResult.model_validate(orjson.loads(f.read()))
                results.append(result)
    return results

import logging
import tomllib
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, Self

import orjson
import torch
from datasets import Dataset, load_dataset
from pydantic import BaseModel, Field, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from track_llm_apis import get_assets_dir
from track_llm_apis.util import dataset_info, load_lmsys_chat_1m

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("choreographer").setLevel(logging.WARNING)
logging.getLogger("kaleido").setLevel(logging.WARNING)
logger = logging.getLogger("track-llm-apis")


class AnalysisConfig(BaseSettings):
    device: str | None = "cuda" if torch.cuda.is_available() else None
    experiment: Literal[
        "baseline",
        "ablation_prompt",
        "bi_param_sweep",
        "met_param_sweep",
        "mmlu_param_sweep",
        "met_t0_param_sweep",
        "mmlu_t0_param_sweep",
        "lt_param_sweep",
    ] = "baseline"
    task: Literal["compute_stats", "plot"] = "compute_stats"
    # For task="compute_stats"
    sampling_dirname: str | None = None
    # For task="plot"
    sampling_dirnames: list[str] | None = None
    # Filter to analyze only a specific method (DataSource name, e.g., "BI", "GAO2025", "LT")
    method: str | None = None

    detector_alpha: float = 0.05
    results_alpha: float = 0.05
    n_tests: int = 1000
    n_bootstrap: int = 10000
    # Set to 0 to avoid computing the pvalue
    pvalue_b: int = 0

    # Parameter sweep defaults
    bi_n_prompts_values: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    bi_detection_samples_values: list[int] = Field(default_factory=lambda: [1, 3, 10, 20, 50])

    met_n_prompts_values: list[int] = Field(default_factory=lambda: [1, 5, 10, 25])
    met_t0_n_prompts_values: list[int] = Field(default_factory=lambda: [1, 5, 10, 25])
    met_output_tokens_values: list[int] = Field(default_factory=lambda: [1, 5, 10, 25, 50])
    met_t0_output_tokens_values: list[int] = Field(default_factory=lambda: [1, 5, 10, 25, 50])

    mmlu_n_prompts_values: list[int] = Field(default_factory=lambda: [1, 5, 10, 25, 50, 100])
    mmlu_t0_n_prompts_values: list[int] = Field(default_factory=lambda: [1, 5, 10, 25, 50, 100])

    lt_n_samples_values: list[int] = Field(default_factory=lambda: [1, 2, 5, 10, 20, 50])

    # Parameter chunking for SLURM parallelization
    # When set, only process param combinations in [chunk_index * chunk_size, (chunk_index + 1) * chunk_size)
    param_chunk_index: int | None = None
    param_chunk_size: int = 2

    _default_detection_samples: int = 10

    # Baseline params: which param combo to use for each method in baseline comparison
    baseline_bi_params: dict[str, int] = Field(
        default_factory=lambda: {"n_prompts": 5, "detection_samples": 3}
    )
    baseline_met_params: dict[str, int] = Field(
        default_factory=lambda: {"n_prompts": 25, "output_tokens": 5}
    )
    baseline_mmlu_params: dict[str, int] = Field(default_factory=lambda: {"n_prompts": 100})
    baseline_lt_params: dict[str, int] = Field(
        default_factory=lambda: {"detection_samples_per_prompt": 10}
    )
    # T0 baseline params (for completeness, detection samples is always 1 for T0)
    baseline_met_t0_params: dict[str, int] = Field(
        default_factory=lambda: {"n_prompts": 25, "output_tokens": 5}
    )
    baseline_mmlu_t0_params: dict[str, int] = Field(default_factory=lambda: {"n_prompts": 100})

    def get_detection_samples(self, source_name: str) -> int:
        """Get detection samples per prompt for a given data source from baseline params.

        Args:
            source_name: Name of the DataSource (e.g., "BI", "LT", "GAO2025", "MMLU", "GAO2025_T0", "MMLU_T0")

        Returns:
            Detection samples per prompt for that source:
            - T0 methods: 1 (deterministic output)
            - BI: from baseline_bi_params["detection_samples"], default 10
            - LT: from baseline_lt_params["detection_samples_per_prompt"], default 10
            - Others: 10
        """
        if source_name in ("GAO2025_T0", "MMLU_T0"):
            return 1
        if source_name == "BI":
            return self.baseline_bi_params["detection_samples"]
        if source_name == "LT":
            return self.baseline_lt_params["detection_samples_per_prompt"]
        # GAO2025, MMLU: baseline params don't specify detection_samples, use default
        return self._default_detection_samples


class PlottingConfig(BaseModel):
    template: str = "plotly_white"
    font_family: str = "Spectral"
    color_map: dict[str, str] = Field(
        default_factory=lambda: {
            "0": "#636EFA",  # Logprobs
            "1": "#EF553B",  # MMLU
            "2": "#00CC96",  # GAO2025
            "3": "#AB63FA",  # B3IT
            "4": "#FFA15A",  # GAO2025_T0
            "5": "#19D3F3",  # MMLU_T0
        }
    )
    source_name: dict[int, str] = Field(
        default_factory=lambda: {
            0: "LT",
            1: "MMLU-ALG",
            2: "MET",
            3: "B3IT (Ours)",
            4: "MET-T0",
            5: "MMLU-ALG-T0",
        }
    )


class DeviceConfig(BaseModel):
    device: str | None = None
    vllm_device: str | None = None
    original_model_device: str | None = None
    variants_device: str | None = None

    @model_validator(mode="after")
    def validate_device_config(self) -> Self:
        if (
            self.device is None
            and self.vllm_device is None
            and self.original_model_device is None
            and self.variants_device is None
        ):
            self.device = "cuda"
            self.vllm_device = self.device
            self.original_model_device = self.device
            self.variants_device = self.device
        elif self.device is not None:
            if (
                self.vllm_device is not None
                or self.original_model_device is not None
                or self.variants_device is not None
            ):
                raise ValueError(
                    "If 'device' is provided, 'vllm_device', 'original_model_device', and 'variants_device' must be None"
                )
            self.vllm_device = self.device
            self.original_model_device = self.device
            self.variants_device = self.device
        elif (
            self.vllm_device is None
            or self.original_model_device is None
            or self.variants_device is None
        ):
            raise ValueError(
                "Either provide 'device' alone, or all of 'vllm_device', 'original_model_device', and 'variants_device'"
            )

        return self


class Gao2025Config(BaseModel):
    n_wikipedia_prompts: int = 25
    reference_samples_per_prompt: int = 50
    wikipedia_seed: int = 0
    max_tokens: int = 50
    temperature: float = 1.0


class MMLUConfig(BaseModel):
    subset_name: str = "abstract_algebra"
    max_tokens: int = 10
    temperature: float = 0.1
    reference_samples_per_prompt: int = 50

    @property
    def answers(self) -> dict[str, int]:
        cache_path = config.data_dir / "mmlu_answers.json"
        try:
            with open(cache_path) as f:
                return orjson.loads(f.read())
        except FileNotFoundError:
            # Download the dataset and create the cache file
            answers = {}
            mmlu = load_dataset("cais/mmlu", self.subset_name, split="test")
            for row in mmlu:
                answers[row["question"]] = row["answer"]
            with open(cache_path, "wb") as f:
                f.write(orjson.dumps(answers, option=orjson.OPT_INDENT_2))
            return answers


class LogprobConfig(BaseModel):
    model_config = SettingsConfigDict(
        arbitrary_types_allowed=True,  # for Dataset
    )
    batch_size: int = 64
    topk: int = 20
    temperature: float = 0.0
    reference_samples_per_prompt: int = 50
    default_prompt: str = "x"
    _other_prompts_dataset: Dataset | None = None

    @property
    def other_prompts_dataset(self) -> Dataset:
        if self._other_prompts_dataset is None:
            self._other_prompts_dataset = load_lmsys_chat_1m(
                use_cache=True, datasets_dir=config.datasets_dir
            )
        return self._other_prompts_dataset

    @property
    def other_prompts(self) -> list[str]:
        return [item["conversation"][0]["content"] for item in self.other_prompts_dataset]

    @computed_field
    @property
    def other_prompts_dataset_info(self) -> dict[str, str | int]:
        return dataset_info(self.other_prompts_dataset)


class BISamplingConfig(BaseModel):
    temperature: float = 0.0
    per_batch_input_fraction: float = 0.95
    batch_size: int = 64
    phase_1_queries_per_token: int = 3
    phase_1_tokens_per_endpoint: int = 20000
    phase_1_target_border_inputs: int = 20000  # no early stopping
    phase_1_queries_per_bi: int = 1000
    phase_1b_min_bis: int = 10
    max_border_inputs: int = 120
    # Analysis
    n_prompts: int = 10
    reference_samples_per_prompt: int = 50
    detection_samples_per_prompt: int = 10
    # model_name -> max_tokens (how many tokens to generate; last token is used for BI)
    models: dict[str, int] = Field(
        default_factory=lambda: {
            "Qwen/Qwen2.5-0.5B-Instruct": 1,
            "Qwen/Qwen2.5-7B-Instruct": 1,
            "google/gemma-3-1b-it": 1,
            "google/gemma-2-9b-it": 1,
            "microsoft/Phi-4-mini-instruct": 1,
            "mistralai/Mistral-7B-Instruct-v0.3": 1,
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": 1,
            "meta-llama/Llama-3.1-8B-Instruct": 1,
            "allenai/OLMo-2-1124-7B-Instruct": 2,
        }
    )


class SamplingConfig(BaseModel):
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device_config: DeviceConfig = Field(default_factory=DeviceConfig)
    n_samples: int = 1_000
    vllm_enable_sleep_mode: bool = False
    vllm_use_tqdm: bool = True
    # Methods to run during sampling. If None, run all methods.
    # Valid values: GAO2025, GAO2025_T0, MMLU, MMLU_T0, LT, BI
    methods: list[str] | None = None

    @field_validator("methods", mode="before")
    @classmethod
    def parse_methods(cls, v: Any) -> list[str] | None:
        if v is None or isinstance(v, list):
            return v
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("[") and v.endswith("]"):
                v = v[1:-1]
            return [m.strip() for m in v.split(",") if m.strip()]
        return v

    gao2025: Gao2025Config = Field(default_factory=Gao2025Config)
    mmlu: MMLUConfig = Field(default_factory=MMLUConfig)
    logprob: LogprobConfig = Field(default_factory=LogprobConfig)
    bi: BISamplingConfig = Field(default_factory=BISamplingConfig)


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        arbitrary_types_allowed=False,
        validate_assignment=True,
        env_file=".env",
        extra="ignore",
        env_prefix="TRACKLLM__",
        # e.g. specify the model name: TRACKLLM__SAMPLING__MODEL_NAME=...
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    # Paths
    data_dir: Path = Field(default_factory=lambda: Path("/data"))

    # Prompts to send to the smaller list of endpoints
    prompts: list[str] = Field(
        default_factory=lambda: [
            "x ",
            "x " * 5,
            "x " * 20,
            "Let's generate random words! Only output the words, no other text. Continue the list: Underpay\nPolicy\nRisotto\nIdealist",
            "Let's generate random words! Only output the words, no other text. Continue the list: Sinuous\nCornbread\nStipulate\nOverreact",
            "reply in one token. 1+1=",
            # Random prompts
            # 2 characters
            "]\n",
            "HB",
            "e|",
            # 4 characters
            "xég",
            "\x04B\x02z",
            "\x1e·T",
            # 8 characters
            "\x06P\x1dz\x13ZTq",
            "ZZ\x17˚p|[",
            "\x14\x1ap88V?_",
        ]
    )
    # Prompts to send to both the smaller and the extended lists of endpoints
    prompts_extended: list[str] = Field(default_factory=lambda: ["x"])

    max_completion_tokens: int = 1
    seed: int = 0

    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    plotting: PlottingConfig = Field(default_factory=PlottingConfig)

    @property
    def assets_dir(self) -> Path:
        return get_assets_dir()

    @property
    def plots_dir(self) -> Path:
        return self.data_dir / "plots"

    @property
    def sampling_data_dir(self) -> Path:
        return self.data_dir / "sampling"

    @property
    def bi_phase_1_dir(self) -> Path:
        return self.data_dir / "bi_phase_1"

    @property
    def datasets_dir(self) -> Path:
        return self.data_dir / "datasets"

    @property
    def logger(self) -> logging.Logger:
        return logger

    @cached_property
    def chat_templates(self) -> dict[str, Any]:
        with open(get_assets_dir() / "chat_templates.toml", "rb") as f:
            return tomllib.load(f)

    @computed_field
    @property
    def date(self) -> str:
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


config = Config()

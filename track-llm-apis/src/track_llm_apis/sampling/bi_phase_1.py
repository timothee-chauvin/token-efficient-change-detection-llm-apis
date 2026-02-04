"""BI Phase 1: Identify and validate border inputs.

Phase 1a: Identify border inputs by querying models with single-token inputs.
Phase 1b: Validate border inputs by querying each one multiple times.
"""

import random
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import fire
import orjson
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from track_llm_apis.config import config
from track_llm_apis.sampling.vllm_sampling import cleanup_vllm, init_vllm_bi, vllm_inference_bi
from track_llm_apis.util import slugify

logger = config.logger


class BIPhase1ModelResult(BaseModel):
    """Results for a single model in BI phase 1."""

    model_name: str
    bi_config: dict[str, Any]
    phase: str
    # Map from input token string to Counter of output token strings
    # Using dict[str, int] as Pydantic-serializable Counter representation
    results: dict[str, dict[str, int]] = Field(default_factory=dict)
    prompt_lengths: dict[str, int] = Field(default_factory=dict)
    generation_time: float = 0.0

    def add_result(self, input_token: str, output_token: str) -> None:
        if input_token not in self.results:
            self.results[input_token] = {}
        self.results[input_token][output_token] = self.results[input_token].get(output_token, 0) + 1

    def get_border_inputs(self) -> list[str]:
        """Return inputs that produced at least 2 different outputs."""
        return [inp for inp, outputs in self.results.items() if len(outputs) >= 2]

    def get_border_input_ratio(self) -> float:
        if not self.results:
            return 0.0
        return len(self.get_border_inputs()) / len(self.results)

    @staticmethod
    def _get_phase_dir(phase: str, temperature: float) -> Path:
        return config.bi_phase_1_dir / f"phase_{phase}_T={temperature}"

    def _get_path(self) -> Path:
        T = self.bi_config["temperature"]
        phase_dir = self._get_phase_dir(self.phase, T)
        return phase_dir / f"{slugify(self.model_name)}.json"

    def save(self) -> None:
        path = self._get_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(orjson.dumps(self.model_dump(mode="json")))
        logger.info(f"Saved BI phase {self.phase} results for {self.model_name} to {path}")

    @classmethod
    def load(cls, model_name: str, phase: str, temperature: float) -> "BIPhase1ModelResult":
        phase_dir = cls._get_phase_dir(phase, temperature)
        path = phase_dir / f"{slugify(model_name)}.json"
        with open(path, "rb") as f:
            return cls.model_validate(orjson.loads(f.read()))


class BIPhase1Output(BaseModel):
    """Base output for BI phase 1."""

    bi_config: dict[str, Any]
    models: dict[str, BIPhase1ModelResult] = Field(default_factory=dict)

    @classmethod
    def _phase_suffix(cls) -> str:
        raise NotImplementedError

    def save_model(self, model_name: str) -> None:
        """Save a single model's results to its own file."""
        self.models[model_name].save()

    @classmethod
    def load(cls, temperature: float | None = None) -> "BIPhase1Output":
        """Load all model results for this phase and temperature."""
        T = temperature if temperature is not None else config.sampling.bi.temperature
        phase_dir = BIPhase1ModelResult._get_phase_dir(cls._phase_suffix(), T)

        if not phase_dir.exists():
            raise FileNotFoundError(f"Phase directory not found: {phase_dir}")

        models: dict[str, BIPhase1ModelResult] = {}
        bi_config: dict[str, Any] | None = None

        for path in phase_dir.glob("*.json"):
            with open(path, "rb") as f:
                result = BIPhase1ModelResult.model_validate(orjson.loads(f.read()))
                models[result.model_name] = result
                if bi_config is None:
                    bi_config = result.bi_config

        if bi_config is None:
            raise FileNotFoundError(f"No model files found in {phase_dir}")

        return cls(bi_config=bi_config, models=models)

    @classmethod
    def load_model(cls, model_name: str, temperature: float | None = None) -> BIPhase1ModelResult:
        """Load a single model's results."""
        T = temperature if temperature is not None else config.sampling.bi.temperature
        return BIPhase1ModelResult.load(model_name, cls._phase_suffix(), T)


class BIPhase1aOutput(BIPhase1Output):
    @classmethod
    def _phase_suffix(cls) -> str:
        return "1a"


class BIPhase1bOutput(BIPhase1Output):
    @classmethod
    def _phase_suffix(cls) -> str:
        return "1b"


def process_tokenizer(tokenizer: PreTrainedTokenizerBase) -> list[str] | None:
    """Extract unique token strings from a tokenizer.

    Returns a sorted list of unique strings found in the vocabulary.
    """
    token_map = tokenizer.get_vocab()
    # Get special tokens to exclude
    special_tokens: set[str] = set()
    if hasattr(tokenizer, "all_special_tokens"):
        special_tokens.update(tokenizer.all_special_tokens)
    if hasattr(tokenizer, "additional_special_tokens"):
        special_tokens.update(tokenizer.additional_special_tokens or [])

    special_ids: set[int] = set()
    if hasattr(tokenizer, "all_special_ids"):
        special_ids.update(tokenizer.all_special_ids)

    vocab_set: set[str] = set()
    for token_str, token_id in token_map.items():
        if token_str in special_tokens or token_id in special_ids:
            continue
        # Skip tokens that look like special tokens (e.g., <|endoftext|>, <pad>)
        if token_str.startswith("<") and token_str.endswith(">"):
            continue

        # make sure we don't have too much weird stuff in there
        assert token_str.encode("utf-8").decode("utf-8") == token_str

        if token_str.startswith("Ġ") or token_str.startswith("▁"):
            token_str = " " + token_str[1:]

        # make sure the strings get encoded as a single token
        tokenized = tokenizer.encode(token_str)
        if (
            len(tokenized) == 0
            or len(tokenized) > 2
            or (len(tokenized) == 2 and tokenized[0] not in special_ids)
        ):
            continue

        vocab_set.add(token_str)

    return sorted(list(vocab_set))


def get_single_token_inputs(tokenizer, n_tokens: int, seed: int = 0) -> list[str]:
    """Get up to n_tokens unique single-token strings from the tokenizer vocabulary in a fixed random order."""
    vocab = process_tokenizer(tokenizer)
    random.Random(seed).shuffle(vocab)
    return vocab[:n_tokens]


def _load_traffic_prompts() -> list[str]:
    from track_llm_apis.util import load_lmsys_chat_1m

    traffic_dataset = load_lmsys_chat_1m(use_cache=True, datasets_dir=config.datasets_dir)
    traffic_prompts = [item["conversation"][0]["content"] for item in traffic_dataset]
    logger.info(f"Loaded {len(traffic_prompts)} traffic prompts")
    return traffic_prompts


def _run_phase(
    output: BIPhase1Output,
    model_inputs: dict[str, list[str]],
    model_max_tokens: dict[str, int],
    n_samples: int,
    traffic_prompts: list[str],
    device: str,
) -> None:
    """Run inference for models and save results incrementally."""
    bi_config = config.sampling.bi

    for model_name, inputs in model_inputs.items():
        max_tokens = model_max_tokens[model_name]
        logger.info(
            f"Processing {model_name}: {len(inputs)} inputs, {n_samples} samples each, "
            f"max_tokens={max_tokens}"
        )
        start_time = time.time()

        llm = init_vllm_bi(model_name, device)
        results = vllm_inference_bi(
            llm=llm,
            inputs=inputs,
            traffic_prompts=traffic_prompts,
            n_samples=n_samples,
            temperature=bi_config.temperature,
            input_fraction=bi_config.per_batch_input_fraction,
            batch_size=bi_config.batch_size,
            max_tokens=max_tokens,
        )
        cleanup_vllm(llm)

        result = BIPhase1ModelResult(
            model_name=model_name,
            bi_config=output.bi_config,
            phase=output._phase_suffix(),
            results=results.counts,
            prompt_lengths=results.prompt_lengths,
            generation_time=time.time() - start_time,
        )
        output.models[model_name] = result
        output.save_model(model_name)

        logger.info(
            f"  Completed: {len(result.get_border_inputs())} BIs "
            f"({result.get_border_input_ratio():.1%}) in {timedelta(seconds=result.generation_time)}"
        )


def phase_1a(device: str = "cuda:0", model: str | None = None) -> None:
    """Run BI phase 1a: identify border inputs by querying models with single-token inputs."""
    bi_config = config.sampling.bi
    n_tokens = bi_config.phase_1_tokens_per_endpoint

    models_to_process = bi_config.models.keys()
    if model is not None:
        if model not in bi_config.models:
            raise ValueError(f"Model {model} not in config.sampling.bi.models")
        models_to_process = [model]

    model_inputs = {}
    for model_name in models_to_process:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_inputs[model_name] = get_single_token_inputs(tokenizer, n_tokens)

    output = BIPhase1aOutput(bi_config=bi_config.model_dump(mode="json"))
    _run_phase(
        output=output,
        model_inputs=model_inputs,
        model_max_tokens=bi_config.models,
        n_samples=bi_config.phase_1_queries_per_token,
        traffic_prompts=_load_traffic_prompts(),
        device=device,
    )


def phase_1b(device: str = "cuda:0", model: str | None = None) -> None:
    """Run BI phase 1b: query border inputs from phase 1a more times."""
    bi_config = config.sampling.bi

    phase_1a_output = BIPhase1aOutput.load(temperature=bi_config.temperature)

    models_to_process = phase_1a_output.models.items()
    if model is not None:
        if model not in bi_config.models:
            raise ValueError(f"Model {model} not in config.sampling.bi.models")
        if model not in phase_1a_output.models:
            raise ValueError(f"Model {model} not found in phase 1a output")
        models_to_process = [(model, phase_1a_output.models[model])]

    model_inputs = {}
    for model_name, result in models_to_process:
        if model_name not in bi_config.models:
            logger.info(f"Skipping {model_name}: not in current config")
            continue
        border_inputs = result.get_border_inputs()
        if len(border_inputs) < bi_config.phase_1b_min_bis:
            logger.info(
                f"Skipping {model_name}: only {len(border_inputs)} BIs "
                f"(min {bi_config.phase_1b_min_bis})"
            )
            continue
        model_inputs[model_name] = border_inputs

    output = BIPhase1bOutput(bi_config=bi_config.model_dump(mode="json"))
    _run_phase(
        output=output,
        model_inputs=model_inputs,
        model_max_tokens=bi_config.models,
        n_samples=bi_config.phase_1_queries_per_bi,
        traffic_prompts=_load_traffic_prompts(),
        device=device,
    )


def main(device: str = "cuda:0", model: str | None = None) -> None:
    """Run both BI phase 1a and 1b."""
    phase_1a(device=device, model=model)
    phase_1b(device=device, model=model)


if __name__ == "__main__":
    fire.Fire({"main": main, "phase_1a": phase_1a, "phase_1b": phase_1b})

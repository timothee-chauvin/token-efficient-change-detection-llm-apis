import contextlib
import copy
import gc
import hashlib
import logging
import os
import re
from functools import cache
from pathlib import Path
from typing import Any

import torch
import xxhash
from datasets import Dataset, load_dataset, load_from_disk
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("track-llm-apis")

MMLU_PREFIX = "Answer the following multiple choice question. The entire content of your response should be of the following format: ‘ANSWER: $LETTER’ (without quotes) where LETTER is one of A,B,C,D.\n\n"
MMLU_PREFIX_REGEX = (
    MMLU_PREFIX.replace("$", r"\$").replace(".", r"\.").replace("(", r"\(").replace(")", r"\)")
)


def trim_to_length(s: str, length: int) -> str:
    return s[:length] + "..." if len(s) > length else s


def load_lmsys_chat_1m(
    gpt4_filter: bool = True,
    redacted_filter: bool = True,
    flagged_filter: bool = True,
    first_turn_only: bool = True,
    use_cache: bool = True,
    datasets_dir: Path | None = None,
) -> Dataset:
    """
    Load the LMSYS Chat 1M dataset, returning only the "conversation" column.

    Args:
        gpt4_filter: Filter out non-GPT-4 conversations
        redacted_filter: Filter out redacted conversations
        flagged_filter: Filter out conversations with at least one message flagged per the "openai_moderation" column
        first_turn_only: Only keep the first turn of each conversation
        use_cache: Use the processed dataset if it exists on disk, otherwise create it
    """
    ds_name = "lmsys/lmsys-chat-1m"
    logger.info(f"Loading the {ds_name} dataset...")
    if use_cache:
        cache_path = datasets_dir / f"{slugify(ds_name, hash_length=0)}"
        if cache_path.exists():
            logger.info(f"Already processed dataset found at {cache_path}, loading...")
            dataset = load_from_disk(str(cache_path))
            assert isinstance(dataset, Dataset)
            return dataset
        logger.info(f"No processed dataset found at {cache_path}, creating...")

    def filter_fn(model, redacted):
        if gpt4_filter and not redacted_filter:
            return model == "gpt-4"
        elif not gpt4_filter and redacted_filter:
            return ~redacted
        elif gpt4_filter and redacted_filter:
            return (model == "gpt-4") & (~redacted)
        else:
            return True

    def flagged_filter_fn(moderation):
        return all(not m["flagged"] for m in moderation)

    dataset = load_dataset("lmsys/lmsys-chat-1m", token=os.getenv("HF_TOKEN"), split="train")
    assert isinstance(dataset, Dataset)

    logger.info("Filtering dataset...")
    dataset = (
        dataset.with_format("np")
        .filter(
            filter_fn,
            input_columns=["model", "redacted"],
            batched=True,
        )
        .with_format(None)
    )
    if first_turn_only:
        dataset = dataset.map(lambda x: {"conversation": x["conversation"][:2]}, batched=False)
    if flagged_filter:
        dataset = dataset.filter(
            flagged_filter_fn, input_columns=["openai_moderation"], batched=False
        )
    assert all(s["conversation"][0]["role"] == "user" for s in dataset)  # pyright: ignore[reportArgumentType,reportCallIssue]
    assert all(s["conversation"][1]["role"] == "assistant" for s in dataset)  # pyright: ignore[reportArgumentType,reportCallIssue]
    dataset = dataset.remove_columns([col for col in dataset.column_names if col != "conversation"])
    if use_cache:
        dataset.save_to_disk(str(cache_path))
    return dataset


def slugify(s: str, max_length: int = 50, hash_length: int = 8) -> str:
    """
    Convert a string to a slugified version suitable for Linux and MacOS filenames.

    Special characters are hex-encoded to preserve information while keeping
    the filename safe. For example, "|" becomes "7c".

    Args:
        s: The input string to slugify
        max_length: Maximum length of the output without the hash (default: 50)
        hash_length: Length of the hash to append to the output (default: 8)

    Returns:
        A slugified string safe for use as a Linux or MacOS filename
    """
    slug = ""

    for char in s:
        if char.isalnum() or char in "._-+=@~,":
            slug += char
        elif char == " ":
            slug += "-"
        else:
            slug += f"{ord(char):02x}"

    slug = slug[:max_length]

    if hash_length > 0:
        string_hash = hashlib.md5(s.encode("utf-8")).hexdigest()[:hash_length]
        slug += "_" + string_hash

    return slug


def available_gpu_memory_fraction():
    """
    Calculate the fraction of GPU memory that is currently available.
    """
    free, total = torch.cuda.mem_get_info()
    return free / total


def used_gpu_memory(cleanup: bool = False, as_str: bool = False) -> float | str:
    if cleanup:
        gc.collect()
        torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    if as_str:
        return f"Used GPU memory: {(total - free) / 1024**3:.2f} GB / {total / 1024**3:.2f} GB"
    else:
        return total - free


def format_mmlu_prompt(mmlu_item: dict) -> str:
    a, b, c, d = mmlu_item["choices"]
    choices_str = f"A. {a}\nB. {b}\nC. {c}\nD. {d}"
    return f"{MMLU_PREFIX}{mmlu_item['question']}\n\n{choices_str}"


@cache
def mmlu_prompt_to_question(prompt: str) -> str:
    """Opposite of format_mmlu_prompt"""
    pattern = rf"^{MMLU_PREFIX_REGEX}(.*)\n\nA\. .*\nB\. .*\nC\. .*\nD\. .*$"
    match = re.match(pattern, prompt, re.DOTALL)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Could not extract question from MMLU prompt: {prompt}")


@cache
def mmlu_answer_to_choice(answer: str) -> int:
    pattern = r"^.*ANSWER: ([A-D]).*$"
    match = re.match(pattern, answer, re.DOTALL | re.IGNORECASE)
    if match:
        return ord(match.group(1).upper()) - ord("A")
    else:
        # Not formatting the answer correctly is an error
        return -1


def format_wikipedia_prompt(wikipedia_item: dict) -> str:
    """Copied from https://github.com/i-gao/model-equality-testing/blob/fd2ee24d75c9fef87debff8caefa0c04d4a5d374/experiments/prompts.py"""
    out = "Continue the paragraph. Do not output anything except the continuation to the paragraph. Start the continuation immediately.\n"
    out += '"' + wikipedia_item["text"][:100] + '..."'
    return out


def get_model_hash(model):
    """
    Compute a hash of the model's parameters.

    Args:
        model: PyTorch model

    Returns:
        str: Hexadecimal hash string representing the model state
    """
    hasher = xxhash.xxh64()

    # Parameters
    for _, param in sorted(model.named_parameters()):
        # Convert to float32 before converting to bytes to ensure consistent hashing
        param_data = param.detach().cpu().to(torch.float32).numpy().tobytes()
        hasher.update(param_data)

    # Buffers
    for _, buffer in sorted(model.named_buffers()):
        if buffer is not None:
            buffer_data = buffer.detach().cpu().to(torch.float32).numpy().tobytes()
            hasher.update(buffer_data)

    return hasher.hexdigest()


def get_dataset_hash(dataset: Dataset) -> str:
    """
    Compute a hash of the dataset.
    """
    hasher = xxhash.xxh64()
    for item in dataset:
        hasher.update(str(item).encode("utf-8"))
    return hasher.hexdigest()


def fast_hash(s: str) -> str:
    return xxhash.xxh64(s).hexdigest()


@contextlib.contextmanager
def temporary_env(variable_name: str, value: str):
    """Context manager for temporarily setting an environment variable."""
    original_value = os.getenv(variable_name)
    os.environ[variable_name] = value
    try:
        yield
    finally:
        if original_value is None:
            os.environ.pop(variable_name, None)
        else:
            os.environ[variable_name] = original_value


def copy_model_to(model, device: str, dtype: torch.dtype | None = torch.bfloat16):
    """Copy a model to a new device."""
    logger.info(f"Copying model to {device} with dtype {dtype}...")
    return copy.deepcopy(model).to(device, dtype=dtype)


def patch_chat_template(tokenizer, chat_templates: dict[str, Any]):
    chat_template = tokenizer.chat_template
    if "{% generation %}" in chat_template:
        return
    else:
        h = fast_hash(chat_template)
        if h in chat_templates:
            tokenizer.chat_template = chat_templates[h]["template"]
        else:
            raise ValueError(
                f"Chat template hash {h} not found in config.chat_templates. You may need to update the chat_templates.toml file."
            )
    return


def dataset_info(dataset: Dataset | None) -> dict[str, str | int]:
    if dataset is None:
        return {
            "length": 0,
            "hash": "",
            "first": "",
            "last": "",
        }
    return {
        "length": len(dataset),
        "hash": get_dataset_hash(dataset),
        "first": trim_to_length(dataset[0]["conversation"][0]["content"], 100),
        "last": trim_to_length(dataset[-1]["conversation"][0]["content"], 100),
    }


def ci(values: list[float], alpha: float) -> tuple[float, float]:
    values = sorted(values)
    return values[int(alpha / 2 * len(values))], values[int((1 - alpha / 2) * len(values))]


def model_distances(original, variant):
    """Compute distance metrics between two models, including relative metrics
    that allow for fair comparison across different architectures/scales.
    """
    diff_l1_sum = 0.0  # sum(|var - orig|)
    diff_l2_sq_sum = 0.0  # sum((var - orig)^2)
    orig_l1_sum = 0.0  # sum(|orig|)
    orig_l2_sq_sum = 0.0  # sum(orig^2)
    dot_product = 0.0
    var_l2_sq_sum = 0.0  # sum(var^2)
    total_params = 0

    for (name_var, param_var), (name_orig, param_orig) in zip(
        sorted(variant.named_parameters()), sorted(original.named_parameters())
    ):
        assert name_var == name_orig, "Model parameter names do not match"

        param_var = param_var.to(param_orig.device)
        diff = param_var - param_orig

        diff_l1_sum += torch.sum(torch.abs(diff)).item()
        diff_l2_sq_sum += torch.sum(diff**2).item()

        orig_l1_sum += torch.sum(torch.abs(param_orig)).item()
        orig_l2_sq_sum += torch.sum(param_orig**2).item()
        var_l2_sq_sum += torch.sum(param_var**2).item()

        dot_product += torch.sum(param_var * param_orig).item()

        total_params += param_var.numel()

    assert total_params > 0, "Models have no parameters"

    l2_dist = diff_l2_sq_sum**0.5
    orig_l2_norm = orig_l2_sq_sum**0.5
    var_l2_norm = var_l2_sq_sum**0.5

    cosine_sim = dot_product / (orig_l2_norm * var_l2_norm)

    return {
        # Absolute Metrics (Scale-Dependent)
        "L1": diff_l1_sum,
        "L2": l2_dist,
        "MAE": diff_l1_sum / total_params,
        "MSE": diff_l2_sq_sum / total_params,
        # Relative Metrics (Scale-Invariant)
        "Relative L1": diff_l1_sum / orig_l1_sum,
        "Relative L2": l2_dist / orig_l2_norm,
        # Similarity Metrics
        "Cosine Similarity": cosine_sim,
        "Cosine Distance": 1 - cosine_sim,
    }


def compute_yearly_cost(
    input_tokens_per_sample: float,
    output_tokens_per_sample: float,
    n_samples: int,
    input_cost_per_token: float = 3.0 / 1e6,
    output_cost_per_token: float = 12.0 / 1e6,
) -> float:
    """
    Compute the yearly cost of monitoring based on hourly sampling.

    The cost of sampling for the reference distribution is ignored, since it is only done once.

    Args:
        input_tokens_per_sample: Average number of input tokens per sample
        output_tokens_per_sample: Average number of output tokens per sample
        n_samples: Number of samples collected per hour
        input_cost_per_token: Cost per input token (default: $3/1M for GPT-4.1)
        output_cost_per_token: Cost per output token (default: $12/1M for GPT-4.1)

    Returns:
        Estimated yearly cost in dollars.
    """
    hours_per_year = 24 * 365
    return (
        hours_per_year
        * n_samples
        * (
            input_cost_per_token * input_tokens_per_sample
            + output_cost_per_token * output_tokens_per_sample
        )
    )

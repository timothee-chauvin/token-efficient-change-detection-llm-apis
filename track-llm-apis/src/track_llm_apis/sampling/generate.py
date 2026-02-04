import glob
import json
import os
import random
import time
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import cast

import fire
import numpy as np
import orjson
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from track_llm_apis.config import DeviceConfig, config
from track_llm_apis.sampling.common import CompressedOutput, DataSource, get_methods_suffix
from track_llm_apis.sampling.vllm_sampling import (
    cleanup_vllm,
    init_vllm,
    load_model_to_vllm,
    vllm_inference,
    vllm_inference_bi,
    vllm_inference_random_traffic,
)
from track_llm_apis.tinychange import TinyChange, TinyChangeConfig
from track_llm_apis.util import (
    fast_hash,
    format_mmlu_prompt,
    format_wikipedia_prompt,
    get_model_hash,
    model_distances,
    patch_chat_template,
    slugify,
    used_gpu_memory,
)
from track_llm_apis.wikipedia import get_wikipedia_samples

logger = config.logger


@contextmanager
def timed(name: str):
    start = time.time()
    yield
    logger.info(f"{name} took {timedelta(seconds=time.time() - start)}")


random.seed(config.seed)
np.random.seed(config.seed)


def main():
    start_time = time.time()
    DEBUG = False
    if DEBUG:
        config.sampling.n_samples = 5

    config.sampling.device_config = DeviceConfig(
        vllm_device="cuda:0",
        original_model_device="cuda:0",
        variants_device="cuda:1",
    )
    tc_config = TinyChangeConfig(variants_device=config.sampling.device_config.variants_device)
    if DEBUG:
        # tc_config.enable_finetuning = False
        # tc_config.finetuning_samples = [1, 16]
        # tc_config.enable_lora_finetuning = False
        tc_config.enable_weight_pruning = False
        tc_config.finetuning_samples = [1]
        tc_config.weight_pruning_random_scale = []
        tc_config.weight_pruning_magnitude_scale = [0.1]
        tc_config.enable_quantization = False
        tc_config.enable_random_noise = False

    model_name = config.sampling.model_name
    model_slug = slugify(model_name, max_length=100, hash_length=0)
    prompts = config.prompts + config.prompts_extended

    # Parse methods to run
    methods_config = config.sampling.methods
    if methods_config is None:
        methods_to_run = set(DataSource.all_names())
    else:
        methods_to_run = set(methods_config)
        invalid = methods_to_run - set(DataSource.all_names())
        if invalid:
            raise ValueError(f"Invalid methods: {invalid}. Valid: {DataSource.all_names()}")
    logger.info(f"Methods to run: {methods_to_run}")

    methods_suffix = get_methods_suffix(config.sampling.methods)
    output_dir = config.sampling_data_dir / f"{config.date}_{model_slug}{methods_suffix}"
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(
        config.sampling.device_config.original_model_device
    )
    # if model.dtype.itemsize > 2:
    #     logger.info(f"Converting model from {model.dtype} to bfloat16")
    #     model.to(torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    patch_chat_template(tokenizer, config.chat_templates)
    assert isinstance(tc_config.finetuning_dataset, Dataset)
    tiny_change = TinyChange(model, tokenizer, tc_config)
    n_variants = tiny_change.n_variants
    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    compressed_output = CompressedOutput(model_name=model_name, gpus=gpus)

    gao2025_config = config.sampling.gao2025
    mmlu_config = config.sampling.mmlu
    logprob_config = config.sampling.logprob
    bi_config = config.sampling.bi

    # Load the MMLU prompts
    mmlu = load_dataset("cais/mmlu", mmlu_config.subset_name, split="test")

    # Load border inputs from BI phase 1
    from track_llm_apis.sampling.bi_phase_1 import BIPhase1bOutput

    bi_model_result = BIPhase1bOutput.load_model(
        model_name=model_name, temperature=bi_config.temperature
    )
    border_inputs = bi_model_result.get_border_inputs()[: bi_config.max_border_inputs]
    if not border_inputs:
        raise ValueError(f"No border inputs found for {model_name}")
    if model_name not in bi_config.models:
        raise ValueError(f"Model {model_name} not in bi_config.models (required for max_tokens)")
    bi_max_tokens = bi_config.models[model_name]

    wikipedia = get_wikipedia_samples(
        n=gao2025_config.n_wikipedia_prompts, seed=gao2025_config.wikipedia_seed
    )

    metadata = {
        "config": config.model_dump(
            mode="json",
            exclude={"api", "analysis"},
        ),
        "tinychange_config": tc_config.model_dump(mode="json"),
        "dtype": str(model.dtype),
        "model_hash": get_model_hash(model),
        "chat_template_hash": fast_hash(tokenizer.chat_template),
        "n_processed_variants": 0,
        "n_total_variants": n_variants,
        "processed_variants": [],
    }
    logger.info(f"Initial metadata:\n{json.dumps(metadata, indent=2)}")
    # Initialize vLLM instance
    llm = init_vllm(model, tokenizer, config.sampling.device_config.vllm_device)

    i = 0
    total_gen_time = 0.0
    total_inference_time = 0.0
    try:
        while True:
            gen_start = time.time()
            variant = tiny_change.__next__()
            gen_time = time.time() - gen_start
            gen_time_str = str(timedelta(seconds=gen_time))
            total_gen_time += gen_time
            total_gen_time_str = str(timedelta(seconds=total_gen_time))
            i += 1
            n_samples = config.sampling.n_samples
            logger.info(f"Generated variant {i}/{n_variants}: ({variant.model_hash})")
            logger.info(json.dumps(variant.description))
            logger.info(f"Generation time: {gen_time_str}")
            logger.info(used_gpu_memory(cleanup=True, as_str=True))
            variant_name = variant.name()

            inference_start = time.time()
            if llm is not None and config.sampling.vllm_enable_sleep_mode:
                llm.wake_up()

            with timed("Loading model to vLLM"):
                load_model_to_vllm(llm, variant.model)

            # Model Equality Testing: Which Model Is This API Serving?
            if "GAO2025" in methods_to_run:
                with timed("GAO2025"):
                    vllm_inference(
                        llm=llm,
                        prompts=[format_wikipedia_prompt(item) for item in wikipedia],
                        n_samples=n_samples,
                        max_tokens=gao2025_config.max_tokens,
                        temperature=gao2025_config.temperature,
                        variant=variant_name,
                        source=DataSource.GAO2025,
                        compressed_output=compressed_output,
                    )

            # MET at T=0
            if "GAO2025_T0" in methods_to_run:
                with timed("GAO2025_T0"):
                    vllm_inference(
                        llm=llm,
                        prompts=[format_wikipedia_prompt(item) for item in wikipedia],
                        n_samples=1,
                        max_tokens=gao2025_config.max_tokens,
                        temperature=0.0,
                        variant=variant_name,
                        source=DataSource.GAO2025_T0,
                        compressed_output=compressed_output,
                    )

            # MMLU
            if "MMLU" in methods_to_run:
                with timed("MMLU"):
                    vllm_inference(
                        llm=llm,
                        prompts=[format_mmlu_prompt(cast(dict, item)) for item in mmlu],
                        n_samples=n_samples,
                        max_tokens=mmlu_config.max_tokens,
                        temperature=mmlu_config.temperature,
                        variant=variant_name,
                        source=DataSource.MMLU,
                        compressed_output=compressed_output,
                    )

            # MMLU at T=0
            if "MMLU_T0" in methods_to_run:
                with timed("MMLU_T0"):
                    vllm_inference(
                        llm=llm,
                        prompts=[format_mmlu_prompt(cast(dict, item)) for item in mmlu],
                        n_samples=1,
                        max_tokens=mmlu_config.max_tokens,
                        temperature=0.0,
                        variant=variant_name,
                        source=DataSource.MMLU_T0,
                        compressed_output=compressed_output,
                    )

            # LT
            if "LT" in methods_to_run:
                with timed("LT"):
                    vllm_inference_random_traffic(
                        llm=llm,
                        prompts=prompts,
                        other_prompts=logprob_config.other_prompts,
                        batch_size=logprob_config.batch_size,
                        n_samples=n_samples,
                        max_tokens=config.max_completion_tokens,
                        temperature=logprob_config.temperature,
                        logprobs_topk=logprob_config.topk,
                        variant=variant_name,
                        source=DataSource.LT,
                        compressed_output=compressed_output,
                    )

            # BI
            if "BI" in methods_to_run:
                with timed("B3IT"):
                    bi_results = vllm_inference_bi(
                        llm=llm,
                        inputs=border_inputs,
                        traffic_prompts=logprob_config.other_prompts,
                        n_samples=n_samples,
                        temperature=bi_config.temperature,
                        input_fraction=bi_config.per_batch_input_fraction,
                        batch_size=logprob_config.batch_size,
                        max_tokens=bi_max_tokens,
                    )
                compressed_output.add_bi_results(
                    bi_results.counts,
                    bi_results.prompt_lengths,
                    variant=variant_name,
                    output_tokens=bi_results.output_tokens,
                )

            inference_time = time.time() - inference_start
            inference_time_str = str(timedelta(seconds=inference_time))
            total_inference_time += inference_time
            total_inference_time_str = str(timedelta(seconds=total_inference_time))
            logger.info(f"Inference time: {inference_time_str}")

            # Free up model weights and KV cache from vLLM memory
            if config.sampling.vllm_enable_sleep_mode:
                llm.sleep(level=2)

            total_time = time.time() - start_time
            total_time_str = str(timedelta(seconds=total_time))

            metadata["processed_variants"].append(
                {
                    variant_name: {
                        "description": variant.description,
                        "model_hash": variant.model_hash,
                        "distances": model_distances(original=model, variant=variant.model),
                        "generation_time": gen_time,
                        "generation_time_str": gen_time_str,
                        "inference_time": inference_time,
                        "inference_time_str": inference_time_str,
                    }
                }
            )
            metadata["n_processed_variants"] = len(metadata["processed_variants"])
            metadata["total_gen_time"] = total_gen_time
            metadata["total_gen_time_str"] = total_gen_time_str
            metadata["total_inference_time"] = total_inference_time
            metadata["total_inference_time_str"] = total_inference_time_str
            metadata["total_time"] = total_time
            metadata["total_time_str"] = total_time_str

            with open(output_dir / "metadata.json", "wb") as f:
                f.write(orjson.dumps(metadata, option=orjson.OPT_INDENT_2))

            compressed_output.dump_json(output_dir)
            del variant.model
            del variant

    except StopIteration:
        logger.info("All variants processed")

    if llm is not None:
        cleanup_vllm(llm)


def merge_directories(*input_dirs: str, output_dir: str):
    """Merge CompressedOutput from multiple directories into a new directory.

    Each input directory should contain one .json.gz file and one metadata.json,
    as produced by generate.py when running different method groups in parallel.

    Usage:
        python -m track_llm_apis.sampling.generate merge dir1 dir2 dir3 --output_dir=out
        python -m track_llm_apis.sampling.generate merge "path/to/dirs/*" --output_dir=out

    Args:
        input_dirs: Directories to merge (space-separated, or a single glob pattern)
        output_dir: Directory to write merged output (must not exist)
    """
    out_path = Path(output_dir)
    if out_path.exists():
        raise ValueError(f"Output directory already exists: {out_path}. Refusing to overwrite.")

    expanded_dirs: list[str] = []
    for d in input_dirs:
        if any(c in d for c in "*?["):
            matches = sorted(glob.glob(d))
            if not matches:
                raise ValueError(f"Glob pattern matched no directories: {d}")
            expanded_dirs.extend(matches)
        else:
            expanded_dirs.append(d)

    logger.info(f"Input directories: {expanded_dirs}")

    input_paths = [Path(d) for d in expanded_dirs]
    for p in input_paths:
        if not p.exists():
            raise ValueError(f"Input directory does not exist: {p}")
        if not p.is_dir():
            raise ValueError(f"Not a directory: {p}")

    merged_output: CompressedOutput | None = None
    merged_metadata: dict = {}
    all_methods: set[str] = set()

    for i, input_path in enumerate(input_paths):
        json_paths = list(input_path.glob("*.json.gz"))
        if len(json_paths) != 1:
            raise ValueError(f"Expected 1 .json.gz file in {input_path}, got {len(json_paths)}")

        metadata_path = input_path / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"metadata.json not found in {input_path}")

        logger.info(f"Loading {json_paths[0]}...")
        current = CompressedOutput.from_json_dir(input_path)

        with open(metadata_path) as f:
            metadata = json.load(f)

        methods = metadata.get("config", {}).get("sampling", {}).get("methods", [])
        if methods:
            all_methods.update(methods)

        if merged_output is None:
            merged_output = current
            merged_metadata = metadata
            merged_metadata["merged_from"] = [{"path": str(input_path), "methods": methods}]
        else:
            if current.model_name != merged_output.model_name:
                raise ValueError(
                    f"Model mismatch: {merged_output.model_name} vs {current.model_name}"
                )
            merged_output.merge(current)
            merged_metadata["merged_from"].append({"path": str(input_path), "methods": methods})

        logger.info(f"Merged {input_path}. Total rows: {len(merged_output.rows)}")

    if "config" in merged_metadata and "sampling" in merged_metadata["config"]:
        merged_metadata["config"]["sampling"]["methods"] = sorted(all_methods)

    merged_output.dump_json(out_path)

    with open(out_path / "metadata.json", "wb") as f:
        f.write(orjson.dumps(merged_metadata, option=orjson.OPT_INDENT_2))

    logger.info(f"Merged output saved to {out_path}")
    logger.info(f"Methods: {sorted(all_methods)}")
    logger.info(f"Total rows: {len(merged_output.rows)}")


if __name__ == "__main__":
    fire.Fire({"main": main, "merge": merge_directories})

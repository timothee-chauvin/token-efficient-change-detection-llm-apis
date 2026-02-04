import gc
import os
import random
import tempfile

import torch
import torch.distributed as dist
from torch.multiprocessing.reductions import reduce_tensor
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

from track_llm_apis.config import config, logger
from track_llm_apis.sampling.common import CompressedOutput, DataSource
from track_llm_apis.util import available_gpu_memory_fraction, temporary_env

# In order to be able to pass functions as args in LLM.collective_rpc()
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"


class WorkerExtension:
    """
    Class for vLLM's worker to inherit from.
    """

    def debug(self):
        return (
            repr(self.model_runner.model),  # pyright: ignore[reportAttributeAccessIssue]
            repr(dir(self.model_runner.model)),  # pyright: ignore[reportAttributeAccessIssue]
        )

    def update_weights_from_ipc_handles(self, ipc_handles):
        """Update model weights from IPC handles."""
        weights = []
        device_id = self.device.index  # pyright: ignore[reportAttributeAccessIssue]

        for name, handle in ipc_handles.items():
            func, args = handle
            list_args = list(args)
            # Update device ID to current device
            list_args[6] = device_id
            tensor = func(*list_args)
            weights.append((name, tensor))

        # Load the weights into the model
        self.model_runner.model.load_weights(weights=weights)  # pyright: ignore[reportAttributeAccessIssue]
        torch.cuda.synchronize()
        return f"Updated {len(weights)} weight tensors"


def create_ipc_handles(model: torch.nn.Module):
    """Create IPC handles for all model parameters."""
    ipc_handles = {}
    for name, param in model.named_parameters():
        # Ensure tensor is contiguous and on GPU
        if not param.is_contiguous():
            param = param.contiguous()
        ipc_handles[name] = reduce_tensor(param.detach())
    return ipc_handles


def vllm_inference_random_traffic(
    llm: LLM,
    prompts: list[str],
    other_prompts: list[str],
    batch_size: int,
    n_samples: int,
    max_tokens: int,
    temperature: float | int,
    logprobs_topk: int,
    variant: str,
    source: DataSource,
    compressed_output: CompressedOutput,
):
    """
    Add `n_samples` completions for the first `max_tokens` inference tokens of a list of prompts mixed with random traffic to the compressed output.

    Args:
        llm: initialized vLLM model
        prompts: List of prompts to track
        other_prompts: List of other prompts to mix with the prompts
        batch_size: Number of prompts to generate in each batch
        n_samples: Number of times to run the inference
        max_tokens: Number of output tokens to generate
        temperature: Sampling temperature
        logprobs_topk: Number of logprobs to return per token position
        compressed_output: Compressed output to add the completions to
    """
    if max_tokens > 1:
        raise NotImplementedError("max_tokens > 1 not implemented wrt saving (see OutputRow)")
    sampling_params = SamplingParams(
        n=1,
        max_tokens=max_tokens,
        temperature=temperature,
        logprobs=logprobs_topk,
    )
    # Generate a new random batch for each sample.
    for _ in range(n_samples):
        # choices: with replacement (traffic prompts can be repeated).
        # sample: without replacement (target positions must be unique).
        traffic_prompts = random.choices(other_prompts, k=batch_size - len(prompts))
        # Position in the batch of the first prompt, second prompt, etc.
        prompt_positions = random.sample(range(batch_size), k=len(prompts))
        other_positions = [i for i in range(batch_size) if i not in prompt_positions]
        batch_prompts = [""] * batch_size
        for i, prompt in enumerate(prompts):
            batch_prompts[prompt_positions[i]] = prompt
        for i in range(batch_size - len(prompts)):
            batch_prompts[other_positions[i]] = traffic_prompts[i]
        outputs = llm.generate(
            batch_prompts, sampling_params, use_tqdm=config.sampling.vllm_use_tqdm
        )
        for i, prompt in enumerate(prompts):
            prompt_position = prompt_positions[i]
            compressed_output.add_batch_from_request_output(
                outputs[prompt_position], variant=variant, source=source
            )


def vllm_inference(
    llm: LLM,
    prompts: list[str],
    n_samples: int,
    max_tokens: int,
    temperature: float | int,
    variant: str,
    source: DataSource,
    compressed_output: CompressedOutput,
):
    sampling_params = SamplingParams(
        n=n_samples,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    outputs = llm.generate(prompts, sampling_params, use_tqdm=config.sampling.vllm_use_tqdm)
    for output in outputs:
        compressed_output.add_batch_from_request_output(output, variant=variant, source=source)


def cleanup_vllm(llm):
    """Clean up vLLM instance and free GPU memory."""
    destroy_model_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()
    # https://github.com/vllm-project/vllm/issues/1908#issuecomment-2975218097
    llm.llm_engine.engine_core.shutdown()
    del llm
    gc.collect()
    torch.cuda.empty_cache()


def init_vllm(model, tokenizer, vllm_device: str) -> LLM:
    assert vllm_device == "cuda" or (vllm_device.startswith("cuda:") and vllm_device[5:].isdigit())
    if vllm_device == "cuda":
        visible_devices = "0"
    else:
        visible_devices = vllm_device[5:]

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save model and tokenizer to temporary directory
        model.save_pretrained(temp_dir)
        tokenizer.save_pretrained(temp_dir)

        with temporary_env("CUDA_VISIBLE_DEVICES", visible_devices):
            # Load into vLLM
            available_memory_fraction = available_gpu_memory_fraction()
            vllm_memory = 0.5 * available_memory_fraction
            while True:
                try:
                    llm = LLM(
                        model=temp_dir,
                        enforce_eager=True,
                        gpu_memory_utilization=vllm_memory,
                        worker_extension_cls="track_llm_apis.sampling.vllm_sampling.WorkerExtension",
                    )
                    return llm
                except RuntimeError:
                    vllm_memory += 0.1 * available_memory_fraction
                    if vllm_memory > available_memory_fraction:
                        raise RuntimeError("Failed to load model into vLLM")


def load_model_to_vllm(llm: LLM, model) -> None:
    """Load a model into a running instance of vLLM in-place, using IPC handles."""
    logger.info("Creating IPC handles from model weights...")
    ipc_handles = create_ipc_handles(model)

    logger.info("Updating vLLM weights via IPC...")
    result = llm.collective_rpc("update_weights_from_ipc_handles", args=(ipc_handles,))
    logger.info(f"IPC update result: {result}")


class BIInferenceResult:
    def __init__(self, output_tokens: int):
        self.counts: dict[str, dict[str, int]] = {}
        self.prompt_lengths: dict[str, int] = {}
        self.output_tokens: int = output_tokens


def vllm_inference_bi(
    llm: LLM,
    inputs: list[str],
    traffic_prompts: list[str],
    n_samples: int,
    temperature: float,
    input_fraction: float,
    batch_size: int,
    max_tokens: int,
) -> BIInferenceResult:
    """Run BI inference: mix inputs with random traffic, repeat n_samples times.

    Args:
        max_tokens: Number of tokens to generate; only the last token is used for BI.

    Returns:
        BIInferenceResult with counts (input -> {output -> count}), prompt_lengths, and output_tokens
    """
    result = BIInferenceResult(output_tokens=max_tokens)
    sampling_params = SamplingParams(n=1, max_tokens=max_tokens, temperature=temperature)
    tokenizer = llm.get_tokenizer()

    inputs_per_batch = int(batch_size * input_fraction)
    traffic_per_batch = batch_size - inputs_per_batch

    for _ in range(n_samples):
        input_queue = inputs.copy()
        random.shuffle(input_queue)

        while input_queue:
            batch_inputs = input_queue[:inputs_per_batch]
            input_queue = input_queue[inputs_per_batch:]

            # If batch isn't full, pad with more inputs (results discarded)
            filler_inputs = []
            if len(batch_inputs) < inputs_per_batch:
                n_filler = inputs_per_batch - len(batch_inputs)
                filler_inputs = random.choices(inputs, k=n_filler)

            traffic_selection = random.choices(traffic_prompts, k=traffic_per_batch)
            batch = batch_inputs + filler_inputs + traffic_selection

            positions = list(range(len(batch)))
            random.shuffle(positions)
            shuffled_batch = [batch[p] for p in positions]
            inverse_positions = [0] * len(positions)
            for new_pos, old_pos in enumerate(positions):
                inverse_positions[old_pos] = new_pos

            outputs = llm.generate(shuffled_batch, sampling_params, use_tqdm=False)

            for i, inp in enumerate(batch_inputs):
                out_pos = inverse_positions[i]
                output = outputs[out_pos]
                last_token_id = output.outputs[0].token_ids[-1]
                output_token = tokenizer.decode([last_token_id])
                if inp not in result.counts:
                    result.counts[inp] = {}
                result.counts[inp][output_token] = result.counts[inp].get(output_token, 0) + 1
                if inp not in result.prompt_lengths:
                    assert output.prompt_token_ids is not None
                    result.prompt_lengths[inp] = len(output.prompt_token_ids)

    for inp in inputs:
        assert sum(result.counts[inp].values()) == n_samples
        assert inp in result.prompt_lengths
    return result


def init_vllm_bi(model_name: str, device: str) -> LLM:
    """Initialize vLLM for BI phase 1."""
    assert device == "cuda" or (device.startswith("cuda:") and device[5:].isdigit())

    available_memory_fraction = available_gpu_memory_fraction()
    vllm_memory = 0.9 * available_memory_fraction
    llm = LLM(
        model=model_name,
        enforce_eager=True,
        gpu_memory_utilization=vllm_memory,
    )
    return llm

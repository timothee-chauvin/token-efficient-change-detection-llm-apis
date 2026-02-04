"""BI prevalence study: Query endpoints at multiple temperatures to measure border input prevalence."""

import asyncio
from pathlib import Path

from aiolimiter import AsyncLimiter

from b3it_monitoring.bi.common import (
    EndpointState,
    get_input_tokens,
    load_tokenizers,
    run_queries,
)
from b3it_monitoring.config import config, logger


async def run_bi_prevalence(
    temperatures: list[float], base_dir: Path | None = None
) -> None:
    """Run BI prevalence study across multiple temperatures.

    This is essentially the same as phase_1a, but:
    - Runs across multiple temperatures (shared rate limiters per endpoint)
    - Uses config.endpoints_bi_prevalence instead of config.endpoints_bi_phase_1
    - No early stopping (we want to measure prevalence for all tokens)
    """
    if base_dir is None:
        base_dir = config.bi.data_dir / "bi_prevalence"

    logger.info(f"Running BI prevalence with temperatures={temperatures}")
    tokenizer_index, fallback_tokens = load_tokenizers()

    endpoints = config.endpoints_bi_prevalence
    logger.info(
        f"Running for {len(endpoints)} endpoints across temperatures: {temperatures}"
    )

    requests_per_second = config.bi.phase_1.requests_per_second_per_endpoint
    max_concurrent_requests = config.bi.phase_1.max_concurrent_requests_per_endpoint
    max_concurrent_tokens = config.bi.phase_1.max_concurrent_tokens_per_endpoint

    states = [
        EndpointState(
            endpoint=ep,
            input_tokens=get_input_tokens(
                ep,
                tokenizer_index,
                fallback_tokens,
                config.bi.prevalence.tokens_per_endpoint,
            ),
            temperatures=temperatures,
            base_dir=base_dir,
            rate_limiter=AsyncLimiter(requests_per_second, 1),
            concurrency_semaphore=asyncio.Semaphore(max_concurrent_requests),
            pending_before_new_semaphore=asyncio.Semaphore(max_concurrent_tokens),
            queries_per_token=config.bi.prevalence.queries_per_token,
        )
        for ep in endpoints
    ]

    pending_lists = [s.get_unfinished_prompts() for s in states]

    await run_queries(
        states,
        pending_lists,
        config.bi.phase_1.request_delay_seconds,
        stop_early=None,
    )

    logger.info("BI prevalence study complete")


if __name__ == "__main__":
    TEMPERATURES = [0.0, 1e-10, 1e-5, 1e-3, 1e-2, 1e-1, 0.5, 1.5]
    asyncio.run(run_bi_prevalence(TEMPERATURES))

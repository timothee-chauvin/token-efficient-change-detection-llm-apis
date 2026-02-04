"""Tests for BI prevalence study with mocked API."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from b3it_monitoring.api import OpenRouterClient
from b3it_monitoring.bi.common import (
    EndpointState,
    TemperatureResults,
    get_input_tokens,
    get_output_path,
    load_existing_results,
    query_all_for_token,
    query_single,
    run_queries,
    save_results,
)
from b3it_monitoring.config import Endpoint
from b3it_monitoring.storage import Response, ResponseError


@pytest.fixture
def sample_endpoint():
    return Endpoint(
        api="openrouter",
        model="test-model",
        provider="test-provider",
        cost=(1.0, 2.0),
    )


@pytest.fixture
def sample_endpoints():
    return [
        Endpoint(
            api="openrouter",
            model=f"test-model-{i}",
            provider="test-provider",
            cost=(1.0, 2.0),
        )
        for i in range(3)
    ]


@pytest.fixture
def mock_response(sample_endpoint):
    def _make_response(content="output", error=None, cost=0.001, input_tokens=5):
        return Response(
            date=datetime.now(tz=timezone.utc),
            endpoint=sample_endpoint,
            prompt="test",
            content=content,
            logprobs=None,
            cost=cost,
            input_tokens=input_tokens,
            output_tokens=1,
            generation_id="gen-123",
            error=error,
        )

    return _make_response


class TestGetOutputPath:
    def test_different_temperatures(self, sample_endpoint, tmp_path):
        path_0 = get_output_path(sample_endpoint, 0.0, tmp_path)
        path_1 = get_output_path(sample_endpoint, 1.0, tmp_path)
        path_small = get_output_path(sample_endpoint, 1e-10, tmp_path)

        assert path_0.parent.name == "T=0"
        assert path_1.parent.name == "T=1"
        assert path_small.parent.name == "T=1e-10"


class TestLoadAndSaveResults:
    def test_load_nonexistent_file(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        results = load_existing_results(path)
        assert results == {}

    @pytest.mark.asyncio
    async def test_save_and_load_results(self, tmp_path):
        path = tmp_path / "results.json"
        results = {
            5: {"token1": ["a", "b"], "token2": ["c"]},
            10: {"token3": ["d", "e", "f"]},
        }
        await save_results(path, results)
        loaded = load_existing_results(path)
        assert loaded == results


class TestTemperatureResults:
    def test_initialization_creates_empty_results(self, tmp_path):
        path = tmp_path / "test.json"
        tr = TemperatureResults(temperature=0.0, output_path=path)
        assert tr.results == {}
        assert tr._prompt_query_counts == {}
        assert tr._prompt_unique_outputs == {}

    def test_initialization_loads_existing_results(self, tmp_path):
        path = tmp_path / "test.json"
        # Pre-create a results file
        import orjson

        results = {"5": {"token1": ["a", "b"], "token2": ["c"]}}
        path.write_bytes(orjson.dumps(results))

        tr = TemperatureResults(temperature=0.0, output_path=path)
        assert tr.results == {5: {"token1": ["a", "b"], "token2": ["c"]}}
        assert tr._prompt_query_counts["token1"] == 2
        assert tr._prompt_query_counts["token2"] == 1
        assert tr._prompt_unique_outputs["token1"] == {"a", "b"}

    @pytest.mark.asyncio
    async def test_record_result(self, tmp_path):
        path = tmp_path / "test.json"
        tr = TemperatureResults(temperature=0.0, output_path=path)

        await tr.record_result("token1", "output_a", num_input_tokens=5)
        await tr.record_result("token1", "output_b", num_input_tokens=5)

        assert tr.results[5]["token1"] == ["output_a", "output_b"]
        assert tr._prompt_query_counts["token1"] == 2
        assert tr._prompt_unique_outputs["token1"] == {"output_a", "output_b"}

    @pytest.mark.asyncio
    async def test_record_result_ignores_none(self, tmp_path):
        path = tmp_path / "test.json"
        tr = TemperatureResults(temperature=0.0, output_path=path)

        await tr.record_result("token1", None, num_input_tokens=5)

        assert tr.results == {}

    @pytest.mark.asyncio
    async def test_flush_saves_results(self, tmp_path):
        path = tmp_path / "test.json"
        tr = TemperatureResults(temperature=0.0, output_path=path)

        await tr.record_result("token1", "output_a", num_input_tokens=5)
        await tr.flush()

        loaded = load_existing_results(path)
        assert loaded == {5: {"token1": ["output_a"]}}


class TestEndpointState:
    @pytest.fixture
    def endpoint_state(self, sample_endpoint, tmp_path):
        from aiolimiter import AsyncLimiter

        return EndpointState(
            endpoint=sample_endpoint,
            input_tokens=["token1", "token2", "token3"],
            temperatures=[0.0, 1.0],
            base_dir=tmp_path,
            rate_limiter=AsyncLimiter(100, 1),
            concurrency_semaphore=asyncio.Semaphore(10),
            pending_before_new_semaphore=asyncio.Semaphore(100),
            queries_per_token=3,
        )

    def test_initialization_creates_temp_results(self, endpoint_state, tmp_path):
        assert 0.0 in endpoint_state._temp_results
        assert 1.0 in endpoint_state._temp_results
        assert endpoint_state._temp_results[0.0].temperature == 0.0
        assert endpoint_state._temp_results[1.0].temperature == 1.0

    def test_get_temp_results(self, endpoint_state):
        tr = endpoint_state.get_temp_results(0.0)
        assert tr.temperature == 0.0

    def test_get_completed_tokens_empty(self, endpoint_state):
        assert endpoint_state.get_completed_tokens() == 0

    @pytest.mark.asyncio
    async def test_get_completed_tokens_partial(self, endpoint_state):
        # Record 3 results for token1 at temp 0.0 (queries_per_token=3)
        for i in range(3):
            await endpoint_state.record_result(0.0, "token1", f"output{i}", 5)
        # Also need to record for temp 1.0
        for i in range(3):
            await endpoint_state.record_result(1.0, "token1", f"output{i}", 5)

        # token1 should now be complete at both temperatures
        assert endpoint_state.get_completed_tokens() == 2  # 1 token * 2 temps

    def test_get_pending_queries_per_temp(self, endpoint_state):
        # All tokens need 3 queries each at each temperature
        pending = endpoint_state.get_pending_queries_per_temp("token1")
        assert pending == {0.0: 3, 1.0: 3}
        pending = endpoint_state.get_pending_queries_per_temp("token2")
        assert pending == {0.0: 3, 1.0: 3}

    @pytest.mark.asyncio
    async def test_get_pending_queries_per_temp_after_recording(self, endpoint_state):
        await endpoint_state.record_result(0.0, "token1", "output1", 5)
        await endpoint_state.record_result(0.0, "token1", "output2", 5)
        # Now only 1 more needed for token1 at temp 0.0, but still 3 at temp 1.0
        pending = endpoint_state.get_pending_queries_per_temp("token1")
        assert pending == {0.0: 1, 1.0: 3}

    def test_get_unfinished_prompts(self, endpoint_state):
        unfinished = endpoint_state.get_unfinished_prompts()
        assert len(unfinished) == 3
        # Each token needs 3 queries at each of 2 temperatures
        for _, temp_pending in unfinished:
            assert temp_pending == {0.0: 3, 1.0: 3}

    @pytest.mark.asyncio
    async def test_record_result(self, endpoint_state):
        await endpoint_state.record_result(0.0, "token1", "output_a", 5)

        tr = endpoint_state.get_temp_results(0.0)
        assert tr.results[5]["token1"] == ["output_a"]

    @pytest.mark.asyncio
    async def test_flush_all_temperatures(self, endpoint_state, tmp_path):
        await endpoint_state.record_result(0.0, "token1", "output_a", 5)
        await endpoint_state.record_result(1.0, "token1", "output_b", 5)
        await endpoint_state.flush()

        path_0 = get_output_path(endpoint_state.endpoint, 0.0, tmp_path)
        path_1 = get_output_path(endpoint_state.endpoint, 1.0, tmp_path)
        assert path_0.exists()
        assert path_1.exists()

    def test_get_border_tokens_empty(self, endpoint_state):
        assert endpoint_state.get_border_tokens() == []
        assert endpoint_state.get_border_tokens_count() == 0

    @pytest.mark.asyncio
    async def test_get_border_tokens_with_variation(self, endpoint_state):
        # Token with single output - not a border input
        await endpoint_state.record_result(0.0, "token1", "same", 5)
        await endpoint_state.record_result(0.0, "token1", "same", 5)

        # Token with multiple outputs - border input
        await endpoint_state.record_result(0.0, "token2", "output_a", 5)
        await endpoint_state.record_result(0.0, "token2", "output_b", 5)

        border = endpoint_state.get_border_tokens()
        assert "token2" in border
        assert "token1" not in border
        assert endpoint_state.get_border_tokens_count() == 1


class TestGetInputTokens:
    def test_with_known_tokenizer(self, sample_endpoint):
        tokenizer_index = {"test-model": "gpt2"}
        fallback = ["fallback1", "fallback2"]

        with patch("b3it_monitoring.bi.common.load_tokenizer_vocab") as mock_load_vocab:
            mock_load_vocab.return_value = ["vocab1", "vocab2", "vocab3"]
            tokens = get_input_tokens(sample_endpoint, tokenizer_index, fallback, 2)
            assert tokens == ["vocab1", "vocab2"]
            mock_load_vocab.assert_called_once_with("gpt2", shuffle=True)

    def test_with_unknown_tokenizer(self, sample_endpoint):
        tokenizer_index = {}  # No known tokenizer for this model
        fallback = ["fallback1", "fallback2", "fallback3"]

        tokens = get_input_tokens(sample_endpoint, tokenizer_index, fallback, 2)
        assert tokens == ["fallback1", "fallback2"]


class TestQuerySingle:
    @pytest.fixture
    def mock_client(self):
        # Use spec to satisfy beartype's type checking
        client = AsyncMock(spec=OpenRouterClient)
        return client

    @pytest.fixture
    def endpoint_state_for_query(self, sample_endpoint, tmp_path):
        from aiolimiter import AsyncLimiter

        return EndpointState(
            endpoint=sample_endpoint,
            input_tokens=["token1"],
            temperatures=[0.0],
            base_dir=tmp_path,
            rate_limiter=AsyncLimiter(1000, 1),
            concurrency_semaphore=asyncio.Semaphore(10),
            pending_before_new_semaphore=asyncio.Semaphore(100),
            queries_per_token=3,
        )

    @pytest.mark.asyncio
    async def test_successful_query(
        self, mock_client, endpoint_state_for_query, mock_response
    ):
        mock_client.query.return_value = mock_response(content="result")

        result = await query_single(
            mock_client, endpoint_state_for_query, "token1", 0.0
        )

        assert result is True
        assert endpoint_state_for_query.completed_queries == 1
        tr = endpoint_state_for_query.get_temp_results(0.0)
        assert tr.results[5]["token1"] == ["result"]

    @pytest.mark.asyncio
    async def test_404_error_abandons_endpoint(
        self, mock_client, endpoint_state_for_query, mock_response
    ):
        error_response = mock_response(
            content=None, error=ResponseError(http_code=404, message="Not found")
        )
        mock_client.query.return_value = error_response

        result = await query_single(
            mock_client, endpoint_state_for_query, "token1", 0.0
        )

        assert result is False
        assert endpoint_state_for_query.got_404 is True

    @pytest.mark.asyncio
    async def test_already_404_skips_query(
        self, mock_client, endpoint_state_for_query, mock_response
    ):
        endpoint_state_for_query.got_404 = True

        result = await query_single(
            mock_client, endpoint_state_for_query, "token1", 0.0
        )

        assert result is False
        mock_client.query.assert_not_called()

    @pytest.mark.asyncio
    async def test_other_error_continues(
        self, mock_client, endpoint_state_for_query, mock_response
    ):
        error_response = mock_response(
            content=None, error=ResponseError(http_code=500, message="Server error")
        )
        mock_client.query.return_value = error_response

        result = await query_single(
            mock_client, endpoint_state_for_query, "token1", 0.0
        )

        assert result is True
        assert endpoint_state_for_query.got_404 is False

    @pytest.mark.asyncio
    async def test_rate_limit_callback_called(
        self, mock_client, endpoint_state_for_query, mock_response
    ):
        mock_client.query.return_value = mock_response(content="result")

        await query_single(mock_client, endpoint_state_for_query, "token1", 0.0)

        # Verify on_retry callback was passed
        call_kwargs = mock_client.query.call_args.kwargs
        assert "on_retry" in call_kwargs
        on_retry = call_kwargs["on_retry"]

        # Simulate a 429 callback
        on_retry(429)
        assert len(endpoint_state_for_query.rate_limit_timestamps) == 1


class TestQueryAllForToken:
    @pytest.fixture
    def mock_client(self):
        return AsyncMock(spec=OpenRouterClient)

    @pytest.fixture
    def endpoint_state(self, sample_endpoint, tmp_path):
        from aiolimiter import AsyncLimiter

        return EndpointState(
            endpoint=sample_endpoint,
            input_tokens=["token1"],
            temperatures=[0.0, 1.0],
            base_dir=tmp_path,
            rate_limiter=AsyncLimiter(1000, 1),
            concurrency_semaphore=asyncio.Semaphore(10),
            pending_before_new_semaphore=asyncio.Semaphore(100),
            queries_per_token=2,
        )

    @pytest.fixture
    def mock_pbar(self):
        from tqdm import tqdm

        return MagicMock(spec=tqdm)

    @pytest.mark.asyncio
    async def test_queries_all_temperatures(
        self, mock_client, endpoint_state, mock_response, mock_pbar
    ):
        mock_client.query.return_value = mock_response(content="result")

        # 2 pending at each of 2 temperatures
        temp_pending = {0.0: 2, 1.0: 2}
        await query_all_for_token(
            mock_client,
            endpoint_state,
            "token1",
            temp_pending=temp_pending,
            pbar=mock_pbar,
            request_delay_seconds=0.0,
        )

        # 2 pending * 2 temperatures = 4 queries
        assert mock_client.query.call_count == 4
        assert mock_pbar.update.call_count == 4

    @pytest.mark.asyncio
    async def test_stop_early_function(
        self, mock_client, endpoint_state, mock_response, mock_pbar
    ):
        mock_client.query.return_value = mock_response(content="result")

        call_count = 0

        def stop_early(state: EndpointState) -> bool:
            nonlocal call_count
            call_count += 1
            return call_count > 2  # Stop after 2 checks

        temp_pending = {0.0: 5, 1.0: 5}
        await query_all_for_token(
            mock_client,
            endpoint_state,
            "token1",
            temp_pending=temp_pending,
            pbar=mock_pbar,
            request_delay_seconds=0.0,
            stop_early=stop_early,
        )

        # Should stop before completing all queries
        assert mock_client.query.call_count < 10

    @pytest.mark.asyncio
    async def test_stops_on_404(
        self, mock_client, endpoint_state, mock_response, mock_pbar
    ):
        # First query succeeds, second returns 404
        mock_client.query.side_effect = [
            mock_response(content="result"),
            mock_response(
                content=None, error=ResponseError(http_code=404, message="Not found")
            ),
        ]

        temp_pending = {0.0: 5, 1.0: 5}
        await query_all_for_token(
            mock_client,
            endpoint_state,
            "token1",
            temp_pending=temp_pending,
            pbar=mock_pbar,
            request_delay_seconds=0.0,
        )

        assert endpoint_state.got_404 is True
        # Should have stopped after the 404
        assert mock_client.query.call_count == 2


class TestRunQueries:
    @pytest.fixture
    def mock_client_class(self, mock_response):
        with patch("b3it_monitoring.bi.common.OpenRouterClient") as MockClient:
            instance = AsyncMock(spec=OpenRouterClient)
            instance.query.return_value = mock_response(content="result")
            MockClient.return_value = instance
            yield MockClient

    @pytest.mark.asyncio
    async def test_run_queries_basic(
        self, sample_endpoint, tmp_path, mock_client_class
    ):
        from aiolimiter import AsyncLimiter

        state = EndpointState(
            endpoint=sample_endpoint,
            input_tokens=["token1", "token2"],
            temperatures=[0.0],
            base_dir=tmp_path,
            rate_limiter=AsyncLimiter(1000, 1),
            concurrency_semaphore=asyncio.Semaphore(10),
            pending_before_new_semaphore=asyncio.Semaphore(100),
            queries_per_token=2,
        )
        pending_list = state.get_unfinished_prompts()

        await run_queries(
            [state], [pending_list], request_delay_seconds=0.0, stop_early=None
        )

        # 2 tokens * 2 queries * 1 temp = 4 total queries
        client_instance = mock_client_class.return_value
        assert client_instance.query.call_count == 4
        assert client_instance.close.called

    @pytest.mark.asyncio
    async def test_run_queries_flushes_on_completion(
        self, sample_endpoint, tmp_path, mock_client_class
    ):
        from aiolimiter import AsyncLimiter

        state = EndpointState(
            endpoint=sample_endpoint,
            input_tokens=["token1"],
            temperatures=[0.0],
            base_dir=tmp_path,
            rate_limiter=AsyncLimiter(1000, 1),
            concurrency_semaphore=asyncio.Semaphore(10),
            pending_before_new_semaphore=asyncio.Semaphore(100),
            queries_per_token=1,
        )
        pending_list = state.get_unfinished_prompts()

        await run_queries(
            [state], [pending_list], request_delay_seconds=0.0, stop_early=None
        )

        # Results should be saved to disk
        output_path = get_output_path(sample_endpoint, 0.0, tmp_path)
        assert output_path.exists()


class TestBIPrevalenceIntegration:
    """Integration tests for the full bi_prevalence flow."""

    @pytest.fixture
    def mock_config(self, tmp_path):
        """Mock config with test settings."""
        mock_cfg = MagicMock()
        mock_cfg.bi.data_dir = tmp_path
        mock_cfg.bi.prevalence.tokens_per_endpoint = 5
        mock_cfg.bi.prevalence.queries_per_token = 2
        mock_cfg.bi.phase_1.requests_per_second_per_endpoint = 100.0
        mock_cfg.bi.phase_1.max_concurrent_requests_per_endpoint = 10
        mock_cfg.bi.phase_1.max_concurrent_tokens_per_endpoint = 100
        mock_cfg.bi.phase_1.request_delay_seconds = 0.0
        mock_cfg.endpoints_bi_prevalence = [
            Endpoint(
                api="openrouter",
                model="test-model-1",
                provider="test-provider",
                cost=(1.0, 2.0),
            ),
        ]
        return mock_cfg

    @pytest.mark.asyncio
    async def test_full_prevalence_run(self, mock_config, mock_response):
        """Test full bi_prevalence run with mocked dependencies."""
        with (
            patch("b3it_monitoring.bi.bi_prevalence.config", mock_config),
            patch("b3it_monitoring.bi.bi_prevalence.load_tokenizers") as mock_load_tok,
            patch("b3it_monitoring.bi.common.OpenRouterClient") as MockClient,
        ):
            # Setup tokenizer mock
            mock_load_tok.return_value = ({}, ["t1", "t2", "t3", "t4", "t5"])

            # Setup client mock with spec to satisfy beartype
            client_instance = AsyncMock(spec=OpenRouterClient)
            client_instance.query.return_value = mock_response(content="result")
            MockClient.return_value = client_instance

            from b3it_monitoring.bi.bi_prevalence import run_bi_prevalence

            await run_bi_prevalence(temperatures=[0.0, 1.0])

            # Verify queries were made
            assert client_instance.query.call_count > 0
            assert client_instance.close.called


class TestConcurrencyAndRateLimiting:
    """Tests for concurrency controls and rate limiting behavior."""

    @pytest.fixture
    def endpoint_state(self, sample_endpoint, tmp_path):
        from aiolimiter import AsyncLimiter

        return EndpointState(
            endpoint=sample_endpoint,
            input_tokens=["t1", "t2", "t3"],
            temperatures=[0.0],
            base_dir=tmp_path,
            rate_limiter=AsyncLimiter(1000, 1),
            concurrency_semaphore=asyncio.Semaphore(2),  # Max 2 concurrent
            pending_before_new_semaphore=asyncio.Semaphore(100),
            queries_per_token=1,
        )

    def test_requests_per_second_calculation(self, endpoint_state):
        import time

        # No requests yet
        assert endpoint_state.get_requests_per_second() == 0.0

        # Add some request timestamps
        now = time.monotonic()
        for i in range(5):
            endpoint_state.request_timestamps.append(now - i * 0.1)

        rps = endpoint_state.get_requests_per_second()
        assert rps > 0

    def test_recent_rate_limits_calculation(self, endpoint_state):
        import time

        # No rate limits
        assert endpoint_state.get_recent_rate_limits() == 0

        # Add some rate limit timestamps
        now = time.monotonic()
        for i in range(3):
            endpoint_state.rate_limit_timestamps.append(now - i)

        assert endpoint_state.get_recent_rate_limits() == 3

        # Old rate limits should not be counted
        endpoint_state.rate_limit_timestamps.append(now - 10)  # 10 seconds ago
        assert endpoint_state.get_recent_rate_limits() == 3  # Still 3


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_temperatures(self, sample_endpoint, tmp_path):
        from aiolimiter import AsyncLimiter

        state = EndpointState(
            endpoint=sample_endpoint,
            input_tokens=["t1"],
            temperatures=[],
            base_dir=tmp_path,
            rate_limiter=AsyncLimiter(100, 1),
            concurrency_semaphore=asyncio.Semaphore(10),
            pending_before_new_semaphore=asyncio.Semaphore(100),
            queries_per_token=1,
        )

        assert state.get_completed_tokens() == 0
        assert state.get_pending_queries_per_temp("t1") == {}
        assert state.get_border_tokens() == []

    def test_empty_input_tokens(self, sample_endpoint, tmp_path):
        from aiolimiter import AsyncLimiter

        state = EndpointState(
            endpoint=sample_endpoint,
            input_tokens=[],
            temperatures=[0.0],
            base_dir=tmp_path,
            rate_limiter=AsyncLimiter(100, 1),
            concurrency_semaphore=asyncio.Semaphore(10),
            pending_before_new_semaphore=asyncio.Semaphore(100),
            queries_per_token=1,
        )

        assert state.get_completed_tokens() == 0
        assert state.get_unfinished_prompts() == []

    @pytest.mark.asyncio
    async def test_atomic_file_write(self, tmp_path):
        """Verify file writes are atomic (via temp file + replace)."""
        path = tmp_path / "test.json"
        results = {1: {"a": ["x"]}}

        # Write initial data
        await save_results(path, results)

        # Write new data
        new_results = {2: {"b": ["y", "z"]}}
        await save_results(path, new_results)

        # Verify new content is correct
        loaded = load_existing_results(path)
        assert loaded == new_results

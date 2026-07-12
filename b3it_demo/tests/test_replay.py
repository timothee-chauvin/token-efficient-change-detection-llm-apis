"""The bundled real data must reproduce the documented production outcomes."""

import pytest

from b3it_demo.config import config
from b3it_demo.data import ENDPOINT_ALIASES, load_endpoint, reference_batch
from b3it_demo.detection import (
    adaptive_transitions,
    epoch_tv_series,
    is_unstable,
    select_top_bis,
)

EXPECTED_EVENTS = {
    "mistral": ["2026-01-30"],
    "deepseek": ["2026-01-24"],
    "gpt-4o-mini-azure": ["2026-02-04"],
    "gpt-4o-mini-openai": [],
    "qwen": [],
}


def run_pipeline(alias: str) -> tuple[list[str], bool]:
    endpoint = load_endpoint(alias)
    _, reference = reference_batch(endpoint.phase_2)
    kept = select_top_bis(reference, config.bi.reinit.top_k_bis)
    series = epoch_tv_series({p: reference[p] for p in kept}, endpoint.phase_2)
    events = adaptive_transitions(series)
    return [e[:10] for e in events], is_unstable(series)


@pytest.mark.parametrize("alias", ENDPOINT_ALIASES)
def test_expected_events(alias: str) -> None:
    events, _ = run_pipeline(alias)
    assert events == EXPECTED_EVENTS[alias]


def test_qwen_unstable() -> None:
    _, unstable = run_pipeline("qwen")
    assert unstable


def test_stable_control_not_unstable() -> None:
    _, unstable = run_pipeline("gpt-4o-mini-openai")
    assert not unstable

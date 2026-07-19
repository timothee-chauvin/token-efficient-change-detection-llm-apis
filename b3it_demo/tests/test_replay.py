"""The bundled real data must reproduce the documented production outcome."""

from b3it_demo.config import config
from b3it_demo.data import load_endpoint, reference_batch
from b3it_demo.detection import adaptive_transitions, epoch_tv_series, select_top_bis


def test_change_detected_on_2026_01_30() -> None:
    endpoint = load_endpoint()
    _, reference = reference_batch(endpoint.phase_2)
    kept = select_top_bis(reference, config.bi.reinit.top_k_bis)
    assert len(kept) == config.bi.reinit.top_k_bis
    series = epoch_tv_series({p: reference[p] for p in kept}, endpoint.phase_2)
    events = adaptive_transitions(series)
    assert [e[:10] for e in events] == ["2026-01-30"]

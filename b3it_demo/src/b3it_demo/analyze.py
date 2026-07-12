"""Vendored unchanged from the production monitor (bi/analyze.py, bi/phase_2.py)."""

from collections import Counter
from typing import NewType

from beartype.typing import Sequence

Prompt = NewType("Prompt", str)
Timestamp = NewType("Timestamp", str)
ResponseToken = NewType("ResponseToken", str)


def compute_tv_distance(
    dist_p: Counter[ResponseToken], dist_q: Counter[ResponseToken]
) -> float | None:
    """Compute total variation distance between two distributions."""
    all_tokens = set(dist_p.keys()) | set(dist_q.keys())
    total_p = sum(dist_p.values())
    total_q = sum(dist_q.values())
    if total_p == 0 or total_q == 0:
        return None
    tv = 0.0
    for token in all_tokens:
        p_prob = dist_p[token] / total_p
        q_prob = dist_q[token] / total_q
        tv += abs(p_prob - q_prob)
    return tv / 2


def get_distribution(
    responses: Sequence[Sequence[str]],
) -> Counter[str]:
    """Get token distribution from responses (each item is (timestamp, token))."""
    return Counter(token for _, token in responses)

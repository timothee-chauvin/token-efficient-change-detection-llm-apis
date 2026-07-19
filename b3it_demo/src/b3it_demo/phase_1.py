"""Border-input candidate extraction from raw phase-1a data.

Same criterion as the production monitor's parse_phase_1_results (bi/reinit.py):
a prompt is a Border Input candidate when it has >=2 distinct non-empty outputs.
Adapted to read a single bundled file instead of globbing a results directory.
"""

META_KEY = "_meta"


def iter_probes(data: dict) -> list[tuple[str, list[str]]]:
    """Flatten {token_count: {prompt: [outputs]}} into (prompt, outputs) pairs."""
    probes: list[tuple[str, list[str]]] = []
    for token_count, prompts_dict in data.items():
        if token_count == META_KEY:
            continue
        probes.extend(prompts_dict.items())
    return probes


def is_border_input(outputs: list[str]) -> bool:
    return len({o for o in outputs if o}) >= 2

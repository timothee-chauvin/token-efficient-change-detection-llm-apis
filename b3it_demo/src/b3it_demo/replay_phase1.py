"""Replay of phase 1: Border Input discovery and reference estimation."""

import time
from collections import Counter

from rich.live import Live
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text

from b3it_demo.analyze import get_distribution
from b3it_demo.config import config
from b3it_demo.data import EndpointData, reference_batch
from b3it_demo.detection import balance_score, select_top_bis
from b3it_demo.phase_1 import is_border_input, iter_probes
from b3it_demo.render import console, format_dist, money, pause, phase_header

PHASE_1A_SECONDS = 8.0
PHASE_1B_ROW_SECONDS = 0.12
PROBES_PER_TICK = 25


def input_token_counts(endpoint: EndpointData) -> dict[str, int]:
    """Per-prompt input token count (including chat template), from phase-1 grouping."""
    return {
        prompt: int(token_count)
        for token_count, prompts_dict in endpoint.phase_1.items()
        if token_count != "_meta"
        for prompt in prompts_dict
    }


def estimate_cost(endpoint: EndpointData, queries_by_prompt: dict[str, int]) -> float:
    """List-price estimate for 1-output-token queries.

    Input token counts are the ones measured by the API (recorded in the
    phase-1 data), so multi-token prompts and chat template overhead are
    accounted for; per-request fees are included.
    """
    counts = input_token_counts(endpoint)
    avg_input = sum(counts.values()) / len(counts)
    input_cost, output_cost = endpoint.meta.cost
    return sum(
        n * (counts.get(p, avg_input) * input_cost + 1 * output_cost) / 1e6
        + n * endpoint.meta.cost_per_request
        for p, n in queries_by_prompt.items()
    )


def replay_phase_1a(endpoint: EndpointData, fast: bool) -> list[str]:
    """Stream through the recorded single-token probes, surfacing flips as found."""
    c = config.bi.phase_1
    probes = iter_probes(endpoint.phase_1)
    phase_header(
        "Phase 1a — Border Input discovery",
        f"Probe the endpoint with up to {c.tokens_per_endpoint} single-token prompts, "
        f"{c.queries_per_token} queries each at T=0, one output token per query.\n\n"
        f"You are about to watch the {len(probes)} recorded probes stream by; "
        "each prompt with an unstable output becomes a [bold]Border Input[/] (BI) "
        "candidate.",
    )
    pause(fast, "press Enter to start probing")

    candidates: list[str] = []
    delay = 0.0 if fast else PHASE_1A_SECONDS / max(len(probes) / PROBES_PER_TICK, 1)
    # keep the whole row under 80 columns, or the probe stream gets cropped
    with Progress(
        TextColumn("probing"),
        BarColumn(bar_width=14),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("[dim]{task.description}[/]"),
        console=console,
    ) as progress:
        task = progress.add_task("", total=len(probes))
        for i, (prompt, outputs) in enumerate(probes):
            probe_line = f"{prompt!r} → {' '.join(repr(o) for o in outputs)}"
            progress.update(task, advance=1, description=probe_line[:40].ljust(40))
            if is_border_input(outputs):
                candidates.append(prompt)
                dist = Counter(outputs)
                # single print: partial lines get clobbered by live bar repaints
                line = Text(f"  candidate #{len(candidates):<2} {prompt!r:<18} → ")
                line.append_text(format_dist(dist, top=len(dist)))
                progress.console.print(line)
            if delay and i % PROBES_PER_TICK == 0:
                time.sleep(delay)
        progress.update(task, description="")

    cost = estimate_cost(endpoint, {p: c.queries_per_token for p, _ in probes})
    n_queries = len(probes) * c.queries_per_token
    console.print(
        f"\n  [bold]{len(candidates)} candidates[/] out of {len(probes)} prompts "
        f"({len(candidates) / len(probes):.1%} prevalence) — "
        f"{n_queries} queries, [bold]≈{money(cost)}[/]"
    )
    return candidates


def replay_phase_1b(
    endpoint: EndpointData, candidates: list[str], fast: bool
) -> dict[str, list[list[str]]]:
    """Replay reference estimation: 100 samples per candidate, keep top-k by balance."""
    r = config.bi.reinit
    phase_header(
        "Phase 1b — reference estimation",
        f"Sample each candidate {r.reference_samples}× to estimate its reference "
        "output distribution,\nthen keep the top "
        f"{r.top_k_bis} by balance score (second-most-common ÷ most-common count).\n"
        "Balanced BIs sit closest to a decision border, where a model change is "
        "most visible.\n\n"
        "The recorded reference distributions will fill in, ranked by balance.",
    )
    pause(fast, "press Enter to estimate the references")

    _, reference = reference_batch(endpoint.phase_2)
    kept = select_top_bis(reference, r.top_k_bis)
    ranked = select_top_bis(reference, len(reference))

    table = Table(show_edge=False, pad_edge=False)
    table.add_column("rank", justify="right", style="dim")
    table.add_column("Border Input")
    table.add_column("reference distribution")
    table.add_column("balance", justify="right")
    table.add_column("")

    with Live(table, console=console):
        for rank, prompt in enumerate(ranked, 1):
            dist = get_distribution(reference[prompt])
            in_top = prompt in kept
            table.add_row(
                str(rank),
                repr(prompt),
                format_dist(dist),
                f"{balance_score(dist):.2f}",
                "[green]kept[/]" if in_top else "[red dim]dropped[/]",
                style=None if in_top else "dim",
            )
            if not fast:
                time.sleep(PHASE_1B_ROW_SECONDS)

    cost = estimate_cost(endpoint, {p: r.reference_samples for p in reference})
    console.print(
        f"\n  [bold]{len(kept)} Border Inputs selected[/] "
        f"({len(reference)} candidates × {r.reference_samples} queries, ≈{money(cost)})"
    )
    return {p: reference[p] for p in kept}

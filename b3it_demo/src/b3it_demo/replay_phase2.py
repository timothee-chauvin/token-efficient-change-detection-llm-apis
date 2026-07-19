"""Replay of phase 2: daily monitoring of the selected Border Inputs."""

import statistics
import time

from rich.panel import Panel
from rich.text import Text

from b3it_demo.config import config
from b3it_demo.data import EndpointData
from b3it_demo.detection import adaptive_transitions, epoch_tv_series, is_unstable
from b3it_demo.render import console, money, pause, phase_header, tv_bar
from b3it_demo.replay_phase1 import estimate_cost

DAY_SECONDS = 0.35
SAMPLING_FRAMES = 4


def deviates(vals: list[float], i: int) -> bool:
    """Whether day i deviates from its trailing baseline.

    Display-only mirror of the baseline check in detection.adaptive_transitions;
    detection itself is done by calling adaptive_transitions on the growing series.
    """
    d = config.bi.detection
    baseline = vals[max(0, i - d.exclusion - d.window) : i - d.exclusion]
    if len(baseline) < d.min_baseline:
        return False
    dev = abs(vals[i] - statistics.mean(baseline))
    return dev > d.abs_delta and dev > d.sigma_k * statistics.stdev(baseline)


def sampling_flicker(day_samples: list[tuple[str, str]], fast: bool) -> None:
    """Briefly show the day's (Border Input → output token) queries streaming by."""
    if fast or not day_samples:
        return
    with console.status("") as status:
        for frame in range(SAMPLING_FRAMES):
            start = frame * 3 % len(day_samples)
            shown = " ".join(f"{p!r}→{t!r}" for p, t in day_samples[start : start + 3])
            status.update(f"[dim]sampling: {shown}[/]")
            time.sleep(DAY_SECONDS / SAMPLING_FRAMES)


def replay_phase_2(
    endpoint: EndpointData, reference: dict[str, list[list[str]]], fast: bool
) -> None:
    """Replay the daily monitoring loop, running detection as each day arrives."""
    c = config.bi.phase_2
    queries_per_day = len(reference) * c.queries_per_token
    day_cost = estimate_cost(endpoint, {p: c.queries_per_token for p in reference})
    d = config.bi.detection
    phase_header(
        "Phase 2 — monitoring",
        f"Every day, sample each of the {len(reference)} Border Inputs "
        f"{c.queries_per_token}× ({queries_per_day} one-token queries "
        f"≈ {money(day_cost)}/day).\n"
        "Compare each day's output distributions to the reference (mean total "
        "variation distance)\nand feed the series to the adaptive detector: "
        f"a day [bold]deviates[/] when its TV departs from the mean\nof a trailing "
        f"{d.window}-day baseline (excluding the {d.exclusion} most recent days) "
        f"by more than {d.abs_delta:g}\nand {d.sigma_k:g}σ of that baseline; "
        f"{d.persistence} consecutive deviating days flag a [bold]change[/], "
        "dated at the first.\n\n"
        "The recorded days will now replay one by one — watch the TV column.",
    )
    pause(fast, "press Enter to start monitoring")

    series = epoch_tv_series(reference, endpoint.phase_2)
    vals = [v for _, v in series]
    change_onset: str | None = None

    console.print("  [dim]date         mean TV[/]")
    for i, (ts, tv) in enumerate(series):
        day_samples = [
            (p, token)
            for p in reference
            if ts in endpoint.phase_2.get(p, {})
            for _, token in endpoint.phase_2[p][ts][:1]
        ]
        sampling_flicker(day_samples, fast)

        day = ts[:10]
        style = "red" if deviates(vals, i) else "cyan"
        line = Text(f"  {day}   {tv:.3f}  ")
        line.append(tv_bar(tv), style=style)
        if style == "red":
            line.append("  ▲ deviates", style="red")
        console.print(line)

        events = adaptive_transitions(series[: i + 1])
        if events:
            change_onset = events[0]
            console.print()
            console.print(
                Panel(
                    f"[bold]CHANGE DETECTED[/] on {day} — sustained deviation since "
                    f"[bold]{change_onset[:10]}[/] "
                    f"(persistence = {config.bi.detection.persistence} days)\n"
                    f"The model behind this endpoint is no longer the one observed "
                    f"at reference time.",
                    border_style="red",
                    padding=(0, 2),
                )
            )
            console.print(
                f"  [dim]epoch closed — in production, a detected change "
                f"re-initializes monitoring with\n  a fresh reference "
                f"({len(series) - i - 1} later days not shown)[/]"
            )
            break

    console.print()
    if change_onset is not None:
        monitored_days = len([ts for ts, _ in series if ts <= change_onset])
        console.print(
            f"  [bold red]Change detected[/] with onset {change_onset[:10]}, "
            f"{monitored_days} days into monitoring "
            f"(≈{money(day_cost * monitored_days)} spent)."
        )
    elif is_unstable(series):
        console.print(
            f"  [yellow]Endpoint flagged unstable[/] over {len(series)} days: "
            "TV noise too high for reliable detection."
        )
    else:
        console.print(
            f"  [bold green]No change detected[/] over {len(series)} days of "
            f"monitoring (≈{money(day_cost * len(series))} total)."
        )
    if endpoint.meta.postscript:
        console.print(f"\n  [italic]{endpoint.meta.postscript}[/]")

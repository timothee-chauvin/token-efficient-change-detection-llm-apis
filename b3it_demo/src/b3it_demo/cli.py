"""Command-line entry point for the B3IT replay demo."""

import sys

import fire
from rich.panel import Panel

from b3it_demo.config import config
from b3it_demo.data import ENDPOINT_ALIASES, EndpointData, load_endpoint
from b3it_demo.render import console
from b3it_demo.replay_phase1 import replay_phase_1a, replay_phase_1b
from b3it_demo.replay_phase2 import replay_phase_2

INTRO = """\
[bold]B3IT — Black-Box Border Input Tracking[/]
Detecting silent model changes behind LLM APIs from output tokens alone.

This demo replays, phase by phase, real data recorded by our production \
monitor against live API endpoints (via OpenRouter):

  [bold]phase 1a[/]  probe with single-token prompts to find [bold]border inputs[/] \
(prompts whose top output token flips)
  [bold]phase 1b[/]  estimate each border input's reference output distribution
  [bold]phase 2[/]   re-sample the border inputs daily and flag sustained \
deviations from the reference

Costs shown are list-price estimates from the recorded token counts, not billed \
amounts.\
"""


def print_endpoints() -> None:
    console.print("Bundled endpoints (real data recorded by our production monitor):")
    for alias in ENDPOINT_ALIASES:
        meta = load_endpoint(alias).meta
        console.print(f"  [bold]{alias:<19}[/] {meta.model} ({meta.provider}) — {meta.note}")


def replay(endpoint: EndpointData, fast: bool) -> None:
    console.print(
        Panel(
            f"Endpoint: [bold]{endpoint.meta.model}[/] served by "
            f"[bold]{endpoint.meta.provider}[/] — {endpoint.meta.note}.",
            border_style="blue",
            padding=(0, 2),
        )
    )
    candidates = replay_phase_1a(endpoint, fast)
    if not candidates:
        console.print("[red]No border input candidates found, cannot monitor.[/]")
        return
    reference = replay_phase_1b(endpoint, candidates, fast)
    replay_phase_2(endpoint, reference, fast)


DEFAULT_ENDPOINT = "mistral"


def menu_loop(fast: bool) -> None:
    console.print(Panel(INTRO, border_style="blue", padding=(0, 2)))
    default = load_endpoint(DEFAULT_ENDPOINT)
    try:
        console.input(
            f"\n[bold]press Enter to replay [/][bold cyan]{default.meta.model}[/]"
            f"[bold] served by [/][bold cyan]{default.meta.provider}[/][bold]:[/] "
        )
    except EOFError:
        return
    console.print()
    replay(default, fast)
    aliases = list(ENDPOINT_ALIASES)
    while True:
        console.print("\nChoose an endpoint to replay:")
        for i, alias in enumerate(aliases, 1):
            meta = load_endpoint(alias).meta
            console.print(f"  [bold]{i}[/]  {meta.model} ({meta.provider}) — {meta.note}")
        try:
            choice = console.input(
                f"\n[bold]endpoint [1-{len(aliases)}, Enter = 1, q = quit]:[/] "
            ).strip()
        except EOFError:
            return
        if choice.lower() in ("q", "quit", "exit"):
            return
        if not choice:
            choice = "1"
        if not (choice.isdigit() and 1 <= int(choice) <= len(aliases)):
            console.print(f"[red]invalid choice: {choice!r}[/]")
            continue
        console.print()
        replay(load_endpoint(aliases[int(choice) - 1]), fast)


def demo(
    endpoint: str | None = None,
    fast: bool = False,
    list: bool = False,
    sigma_k: float | None = None,
    abs_delta: float | None = None,
    persistence: int | None = None,
) -> None:
    """Replay B3IT on real data recorded against live LLM API endpoints.

    Args:
        endpoint: replay this endpoint and exit (see --list); default is an
            interactive menu.
        fast: skip the animation delays and pauses.
        list: list the bundled endpoints and exit.
        sigma_k: override the detector's sigma threshold.
        abs_delta: override the detector's absolute TV threshold.
        persistence: override the number of consecutive deviating days required.
    """
    if list:
        print_endpoints()
        return

    d = config.bi.detection
    if sigma_k is not None:
        d.sigma_k = sigma_k
    if abs_delta is not None:
        d.abs_delta = abs_delta
    if persistence is not None:
        d.persistence = persistence

    if endpoint is not None:
        replay(load_endpoint(endpoint), fast)
    elif sys.stdin.isatty():
        menu_loop(fast)
    else:
        replay(load_endpoint(DEFAULT_ENDPOINT), fast)


def main() -> None:
    fire.Fire(demo)


if __name__ == "__main__":
    main()

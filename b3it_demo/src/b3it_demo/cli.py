"""Command-line entry point for the B3IT replay demo."""

import fire
from rich.panel import Panel

from b3it_demo.config import config
from b3it_demo.data import load_endpoint
from b3it_demo.render import console
from b3it_demo.replay_phase1 import replay_phase_1a, replay_phase_1b
from b3it_demo.replay_phase2 import replay_phase_2

INTRO = """\
[bold]B3IT — Black-Box Border Input Tracking[/] (paper: \
https://arxiv.org/abs/2602.11083 [bold]Token-Efficient Change Detection in LLM APIs[/])
Detecting silent model changes behind LLM APIs from output tokens alone.

This demo replays, phase by phase, real data recorded by our production \
monitor against a live API endpoint (via OpenRouter):

  [bold]phase 1a[/]  probe with {tokens_per_endpoint} single-token prompts \
({queries_per_token} queries each at T=0) to find [bold]Border Inputs[/] \
(prompts where the first token of output at T=0 varies between queries)
  [bold]phase 1b[/]  estimate each Border Input's reference output distribution \
({reference_samples} samples each, keep the {top_k_bis} most balanced)
  [bold]phase 2[/]   re-sample the Border Inputs daily ({phase_2_samples}× each) \
and flag sustained deviations from the reference

Costs shown are list-price estimates from the recorded token counts, not billed \
amounts.\
"""


def demo(fast: bool = False) -> None:
    """Replay B3IT on real data recorded against a live LLM API endpoint.

    Args:
        fast: skip the animation delays and pauses.
    """
    c = config.bi
    console.print(
        Panel(
            INTRO.format(
                tokens_per_endpoint=c.phase_1.tokens_per_endpoint,
                queries_per_token=c.phase_1.queries_per_token,
                reference_samples=c.reinit.reference_samples,
                top_k_bis=c.reinit.top_k_bis,
                phase_2_samples=c.phase_2.queries_per_token,
            ),
            border_style="blue",
            padding=(0, 2),
        )
    )
    endpoint = load_endpoint()
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
        console.print("[red]No Border Input candidates found, cannot monitor.[/]")
        return
    reference = replay_phase_1b(endpoint, candidates, fast)
    replay_phase_2(endpoint, reference, fast)


def main() -> None:
    fire.Fire(demo)


if __name__ == "__main__":
    main()

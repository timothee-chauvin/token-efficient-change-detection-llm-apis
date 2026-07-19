"""Shared terminal rendering helpers."""

import os
import sys
import time
from collections import Counter

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console(highlight=False)


def pause(fast: bool, message: str = "press Enter when ready") -> None:
    """Wait for the user before things start moving."""
    if fast:
        return
    # used when recording the README demo (scripts/record_demo.sh): show the
    # pause but don't wait for input
    auto_seconds = os.environ.get("B3IT_DEMO_AUTO_PAUSE_SECONDS")
    if auto_seconds is not None:
        console.print(f"  [dim italic]{message}[/]")
        time.sleep(float(auto_seconds))
        console.print()
        return
    if not sys.stdin.isatty():
        return
    try:
        console.input(f"  [dim italic]{message}[/] ")
    except EOFError:
        return
    console.print()


_EIGHTHS = " ▏▎▍▌▋▊▉█"


def tv_bar(tv: float, width: int = 12) -> str:
    """Unicode bar for a TV distance in [0, 1]."""
    filled = min(tv, 1.0) * width
    full, frac = int(filled), filled - int(filled)
    return "█" * full + _EIGHTHS[round(frac * 8)] + " " * (width - full - 1)


def format_dist(dist: Counter[str], top: int = 2) -> Text:
    """Render the top tokens of an output distribution, e.g. ' Po' ×62 │ ' P' ×38."""
    text = Text()
    for i, (token, count) in enumerate(dist.most_common(top)):
        if i:
            text.append(" │ ", style="dim")
        text.append(repr(token), style="bold")
        text.append(f" ×{count}", style="dim")
    rest = len(dist) - top
    if rest > 0:
        text.append(f" │ +{rest} more", style="dim")
    return text


def money(dollars: float) -> str:
    return f"${dollars:.4f}" if dollars < 0.01 else f"${dollars:.2f}"


def phase_header(title: str, body: str) -> None:
    console.print()
    console.print(Panel(body, title=f"[bold]{title}[/]", title_align="left"))
    console.print()

"""Loading of the bundled real recorded data."""


import orjson
from pydantic import BaseModel

from b3it_demo.config import root

DATA_DIR = root / "data"

# short alias -> data directory name (slugified model#provider, as in production)
ENDPOINT_ALIASES = {
    "mistral": "mistralai2fmistral-7b-instruct-v0.323together",
    "deepseek": "deepseek2fdeepseek-chat-v3-032423hyperbolic2ffp8",
    "gpt-4o-mini-azure": "openai2fgpt-4o-mini23azure",
    "gpt-4o-mini-openai": "openai2fgpt-4o-mini23openai",
    "qwen": "qwen2fqwen3-235b-a22b-250723wandb2fbf16",
}

Phase1Data = dict[str, dict[str, list[str]]]
Phase2Data = dict[str, dict[str, list[list[str]]]]


class EndpointMeta(BaseModel):
    model: str
    provider: str
    cost: tuple[float, float]  # $/Mtok (input, output), OpenRouter list price
    cost_per_request: float
    note: str
    postscript: str | None


class EndpointData(BaseModel):
    alias: str
    meta: EndpointMeta
    phase_1: Phase1Data
    phase_2: Phase2Data

    def __str__(self) -> str:
        return f"{self.meta.model} ({self.meta.provider})"


def load_endpoint(alias: str) -> EndpointData:
    if alias not in ENDPOINT_ALIASES:
        raise ValueError(f"Unknown endpoint {alias!r}, available: {', '.join(ENDPOINT_ALIASES)}")
    endpoint_dir = DATA_DIR / ENDPOINT_ALIASES[alias]

    def read(name: str) -> dict:
        return orjson.loads((endpoint_dir / name).read_bytes())

    return EndpointData(
        alias=alias,
        meta=EndpointMeta(**read("meta.json")),
        phase_1=read("phase_1.json"),
        phase_2=read("phase_2.json"),
    )


def reference_batch(phase_2: Phase2Data) -> tuple[str, dict[str, list[list[str]]]]:
    """The earliest batch is the epoch reference (as in the production monitor)."""
    ref_ts = min(ts for batches in phase_2.values() for ts in batches)
    return ref_ts, {p: batches[ref_ts] for p, batches in phase_2.items() if ref_ts in batches}

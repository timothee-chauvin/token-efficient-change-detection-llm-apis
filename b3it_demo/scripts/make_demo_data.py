"""One-off: extract the bundled demo data from the production monitor's data.

Follows tests/make_fixtures.py from the monitor codebase: merge monthly phase-2
files, keep prompts present in the first (reference) batch, subsample later
batches to the current monitoring cadence (10 samples/BI/day, seed 0).
Phase-1 discovery files are copied verbatim.

Run from b3it_demo/: uv run python scripts/make_demo_data.py <monitor_repo_path>
"""

import random
import shutil
import sys
from pathlib import Path

import orjson

DEMO_DATA_DIR = Path(__file__).parent.parent / "data"
MAX_SAMPLES = 10
LAST_DAY = "2026-02-28"

ENDPOINTS = {
    "mistralai2fmistral-7b-instruct-v0.323together": {
        "model": "mistralai/mistral-7b-instruct-v0.3",
        "provider": "together",
        "cost": [0.2, 0.2],
        "cost_per_request": 0.0,
        "note": "silent model change caught on 2026-01-30",
        "postscript": (
            "Together's changelog corroborates the detection: on 2026-01-29, requests "
            "for mistral-7b-instruct-v0.3 started being redirected to "
            'Ministral-3-14B-Instruct-2512 as a "same-lineage upgrade with compatible '
            'behavior" (https://docs.together.ai/docs/changelog). B3IT flagged the '
            "swap at the next daily batch."
        ),
    },
    "deepseek2fdeepseek-chat-v3-032423hyperbolic2ffp8": {
        "model": "deepseek/deepseek-chat-v3-0324",
        "provider": "hyperbolic/fp8",
        "cost": [1.25, 1.25],
        "cost_per_request": 0.0,
        "note": "silent model change caught on 2026-01-24",
        "postscript": None,
    },
    "openai2fgpt-4o-mini23azure": {
        "model": "openai/gpt-4o-mini",
        "provider": "azure",
        "cost": [0.15, 0.6],
        "cost_per_request": 1.8e-06,
        "note": "silent change caught on 2026-02-04, while the same model at OpenAI stayed stable",
        "postscript": (
            "The change lasted from 2026-02-04 to 2026-02-18, then mostly reverted: "
            "the most balanced border inputs collapsed to near-deterministic outputs "
            "for two weeks — consistent with a serving-stack change on this Azure "
            "deployment, later rolled back. The same model served by OpenAI stayed "
            "stable throughout."
        ),
    },
    "openai2fgpt-4o-mini23openai": {
        "model": "openai/gpt-4o-mini",
        "provider": "openai",
        "cost": [0.15, 0.6],
        "cost_per_request": 1.8e-06,
        "note": "stable over the whole period (control)",
        "postscript": None,
    },
    "qwen2fqwen3-235b-a22b-250723wandb2fbf16": {
        "model": "qwen/qwen3-235b-a22b-2507",
        "provider": "wandb/bf16",
        "cost": [0.1, 0.1],
        "cost_per_request": 0.0,
        "note": "flagged unstable: output distributions too noisy for reliable detection",
        "postscript": None,
    },
}


def extract_phase_2(phase_2_dir: Path) -> dict:
    merged: dict[str, dict[str, list]] = {}
    for month_file in sorted(phase_2_dir.glob("*.json")):
        for prompt, batches in orjson.loads(month_file.read_bytes()).items():
            merged.setdefault(prompt, {}).update(batches)

    ref_ts = min(ts for batches in merged.values() for ts in batches)
    rng = random.Random(0)
    out: dict[str, dict[str, list]] = {}
    for prompt, batches in merged.items():
        if ref_ts not in batches:
            continue  # prompt from a later epoch
        out[prompt] = {}
        for ts, samples in sorted(batches.items()):
            if ts[:10] > LAST_DAY:
                continue
            if ts != ref_ts and len(samples) > MAX_SAMPLES:
                samples = rng.sample(samples, MAX_SAMPLES)
            out[prompt][ts] = samples
    return out


def main(monitor_repo: str) -> None:
    data_dir = Path(monitor_repo) / "website/data/b3it"
    for slug, meta in ENDPOINTS.items():
        dest = DEMO_DATA_DIR / slug
        dest.mkdir(parents=True, exist_ok=True)

        shutil.copy(data_dir / "phase_1/T=0" / f"{slug}.json", dest / "phase_1.json")

        phase_2 = extract_phase_2(data_dir / "phase_2" / slug)
        (dest / "phase_2.json").write_bytes(orjson.dumps(phase_2))
        (dest / "meta.json").write_bytes(orjson.dumps(meta))

        n_days = len({ts for b in phase_2.values() for ts in b})
        print(f"{slug}: {len(phase_2)} prompts, {n_days} days")


if __name__ == "__main__":
    main(sys.argv[1])

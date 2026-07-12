# Token-Efficient Change Detection in LLM APIs

Code for the paper [**Token-Efficient Change Detection in LLM APIs**](https://arxiv.org/abs/2602.11083) (arXiv:2602.11083).

LLM providers silently change what runs behind their APIs: model swaps, quantization, serving-stack updates. Detecting this from the outside is expensive — unless you know where to look. **B3IT (Black-Box Border Input Tracking)** detects such changes from output tokens alone, at ~30× lower cost than existing methods, with no access to weights or logprobs.

## How it works

Some prompts sit right on a decision border of the model: their top output token is not always the same across queries, even at temperature 0. We call them **border inputs**. Example, recorded against `mistral-7b-instruct-v0.3` on Together: the single-token prompt `'hike'` answered `' H'` 2 times out of 3, and `' Title'` once. Because these inputs sit where the model's output distribution is most sensitive, a small change to the model moves their output distribution measurably — that's where change detection per token is cheapest.

B3IT runs in two phases, using 1-token prompts and 1-token completions:

1. **Phase 1 — find border inputs.** Probe the endpoint with a few thousand single-token prompts (3 queries each); keep the prompts whose output flips, sample each ~100× to estimate its reference output distribution, and keep the most balanced ones.
2. **Phase 2 — monitor.** Re-sample the selected border inputs daily (a few hundred 1-token queries, a fraction of a cent) and compare each day's output distributions to the reference (mean total variation distance). A sustained deviation from the trailing baseline flags a change.

## Demo: replay real detections

The demo replays the full pipeline on **real data recorded by our production monitor** against live API endpoints — including a change caught in the wild that Together's changelog later corroborated (`mistral-7b-instruct-v0.3` silently redirected to `Ministral-3-14B-Instruct-2512` on 2026-01-29).

Requires only [uv](https://docs.astral.sh/uv/):

```bash
cd b3it_demo
uv run b3it-demo
```

```
  candidate #1  'hike'             → ' H' ×2 │ ' Title' ×1
  candidate #2  'piration'         → ' It' ×2 │ ' Insp' ×1
  ...
  2026-01-28   0.068  ▉
  2026-01-29   0.040  ▌
  2026-01-30   1.000  ████████████   ▲ deviates
  2026-01-31   1.000  ████████████   ▲ deviates
  2026-02-01   1.000  ████████████   ▲ deviates

╭──────────────────────────────────────────────────────────────────────────╮
│  CHANGE DETECTED on 2026-02-01 — sustained deviation since 2026-01-30    │
╰──────────────────────────────────────────────────────────────────────────╯
```

Five endpoints are bundled, covering the interesting outcomes: a model swap confirmed by the provider's changelog, a two-week serving-stack change on Azure that later reverted (while the same model at OpenAI stayed stable), a stable control, and an endpoint too noisy to monitor. Pick them from the in-demo menu, or play with the detector directly:

```bash
uv run b3it-demo --list                            # bundled endpoints
uv run b3it-demo --endpoint gpt-4o-mini-azure
uv run b3it-demo --fast --sigma-k 2 --persistence 1  # re-run detection with your thresholds
```

The detection code in the demo (`detection.py`, `analyze.py`) is vendored unchanged from the production monitor, with its production configuration; the replayed data is the monitor's raw recorded samples.

## Repository layout

The paper's code comes from two codebases, included here as separate packages, plus the demo:

| Directory | Contents |
|---|---|
| [`b3it_monitoring/`](b3it_monitoring/) | The B3IT monitoring pipeline as deployed against live endpoints (phase 1 discovery, phase 2 monitoring, analysis) — the paper's in-vivo experiments |
| [`track-llm-apis/`](track-llm-apis/) | In-vitro experiments: vLLM sampling of deliberately modified models (TinyChange), baselines ([Model Equality Testing](https://arxiv.org/abs/2410.20247), MMD, logprob-based methods) |
| [`b3it_demo/`](b3it_demo/) | The demo above: terminal replay of the pipeline on real recorded data |

## Citation

```bibtex
@misc{chauvin2026tokenefficient,
      title={Token-Efficient Change Detection in LLM APIs},
      author={Timoth\'ee Chauvin and Cl\'ement Lalanne and Erwan Le Merrer and Jean-Michel Loubes and Fran\c{c}ois Ta\"iani and Gilles Tredan},
      year={2026},
      eprint={2602.11083},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

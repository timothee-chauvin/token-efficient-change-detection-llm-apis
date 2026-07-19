"""Config mirroring the production monitor's config classes (subset used by the replay)."""

import tomllib
from pathlib import Path

from pydantic import BaseModel

root = Path(__file__).parent.parent.parent


class Phase1Config(BaseModel):
    queries_per_token: int
    tokens_per_endpoint: int
    target_border_inputs: int


class Phase2Config(BaseModel):
    queries_per_token: int


class DetectionConfig(BaseModel):
    window: int
    exclusion: int
    min_baseline: int
    sigma_k: float
    abs_delta: float
    persistence: int
    cooldown: int
    instability_window: int
    instability_threshold: float


class ReinitConfig(BaseModel):
    reference_samples: int
    top_k_bis: int
    min_bis: int


class BIConfig(BaseModel):
    samples_per_day: int
    phase_1: Phase1Config
    phase_2: Phase2Config
    detection: DetectionConfig
    reinit: ReinitConfig


class Config(BaseModel):
    bi: BIConfig


def _load() -> Config:
    with open(root / "config.toml", "rb") as f:
        return Config(**tomllib.load(f))


config = _load()

"""Training loop with early stopping on VALIDATION ROLLOUT error."""

from __future__ import annotations

from pathlib import Path


def set_seed(seed: int) -> None:
    """Seed numpy, torch and python RNGs."""
    raise NotImplementedError


def train(cfg_path: Path, seed: int, run_dir: Path) -> Path:
    """Train one model from config; returns path to the selected checkpoint."""
    raise NotImplementedError

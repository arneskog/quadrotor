"""Select ONE torque-label low-pass cutoff, shared by every network model, on VALIDATION flights.

    uv run python scripts/select_alpha.py

Rule (docs/final_protocol.md, declared before any selection result was looked at): for each model
m, cutoff c and seed s, take the checkpoint score `best_val_score` (mean position RMSE along 0.5 s
validation windows) of results/sel_<m>_a<c>_s<s> relative to the unfiltered run of the same model
and seed, r = score(m, c, s) / score(m, none, s). Median over seeds, then MEAN over the models
(costa, ddm, ddm_kin), so no model's scale or preference dominates. The cutoff with the lowest mean
ratio wins, but replaces "none" only if it is at least `min_rel_improvement` (1 %) better.
Writes results/alpha_selection.md; put the result in configs/train.yaml (labels.alpha_lowpass_hz).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("costa", "ddm", "ddm_kin")
CUTOFFS = ("none", "40", "20", "10")
SEEDS = (0, 1, 2)


def score(model: str, cutoff: str, seed: int) -> float:
    run = ROOT / "results" / f"sel_{model}_a{cutoff}_s{seed}"
    if not (run / "metrics_dev.json").exists():
        raise SystemExit(f"missing or unfinished selection run: {run.name}")
    return float(json.loads((run / "run.json").read_text())["best_val_score"])


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "train.yaml").read_text())
    rel = float(cfg["early_stopping"]["min_rel_improvement"])
    ratio = {(m, c): float(np.median([score(m, c, s) / score(m, "none", s) for s in SEEDS]))
             for m in MODELS for c in CUTOFFS}
    mean_ratio = {c: float(np.mean([ratio[m, c] for m in MODELS])) for c in CUTOFFS}
    best = min(mean_ratio, key=mean_ratio.get)
    if mean_ratio[best] > 1.0 - rel:
        best = "none"

    head = " | ".join("none" if c == "none" else f"{c} Hz" for c in CUTOFFS)
    lines = ["# Shared torque-label low-pass selection (validation flights only)\n",
             "Score ratio to the unfiltered run of the same model and seed (median over seeds "
             f"{list(SEEDS)}); < 1 is better. Rule: lowest mean over models, at least {rel:.0%} better "
             "than none.\n",
             f"| model | {head} |", "|---|" + "---|" * len(CUTOFFS)]
    lines += [f"| {m} | " + " | ".join(f"{ratio[m, c]:.3f}" for c in CUTOFFS) + " |" for m in MODELS]
    lines += ["| **mean** | " + " | ".join(f"**{mean_ratio[c]:.3f}**" for c in CUTOFFS) + " |",
              f"\nSelected for all models: **{best}**"
              f" -> configs/train.yaml labels.alpha_lowpass_hz: {'null' if best == 'none' else float(best)}"]
    text = "\n".join(lines)
    (ROOT / "results" / "alpha_selection.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

"""Tables of the final runs (results/final_<model>_s<seed>), median over seeds [min, max].

    uv run python scripts/summarize_final.py dev    # stage 1: train + val flights
    uv run python scripts/summarize_final.py test   # stage 2: test flights (after the single test pass)

Primary result (docs/final_protocol.md): position error at 0.5 s, median over windows, per role.
Writes results/final_summary_<tag>.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("pbm", "linear_drag", "costa", "ddm", "ddm_kin")
ROLES = {"dev": ("train", "val"), "test": ("test_interp", "test_near", "test_extrap")}
HORIZONS = ("0.1", "0.25", "0.5", "1.0")


def cell(vals: list[float], scale: float = 1.0, fmt: str = "{:.3g}") -> str:
    vals = [scale * v for v in vals]
    if not vals:
        return "–"
    if len(vals) == 1:
        return fmt.format(vals[0])
    return f"{fmt.format(np.median(vals))} [{fmt.format(min(vals))}, {fmt.format(max(vals))}]"


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else "dev"
    metrics = {m: [json.loads(p.read_text()) for p in sorted(ROOT.glob(f"results/final_{m}_s*/metrics_{tag}.json"))]
               for m in MODELS}
    models = [m for m in MODELS if metrics[m]]
    head = "| " + " | ".join(models) + " |"
    sep = "|---" * len(models) + "|"
    out = [f"# Final runs, {tag} ({', '.join(f'{m}: {len(metrics[m])} run(s)' for m in models)})\n",
           "Window-median errors; networks: median over seeds [min, max].\n"]
    roles = [r for r in ROLES[tag] if all(r in run for m in models for run in metrics[m])]

    out += ["\n## Primary: position error at 0.5 s [m]\n", "| role " + head, "|---" + sep]
    out += [f"| {r} | " + " | ".join(cell([run[r]["horizon"]["0.5"]["all"]["pos_m"]["median"] for run in metrics[m]])
                                     for m in models) + " |" for r in roles]

    for q, unit in (("pos_m", "m"), ("vel_mps", "m/s"), ("att_rad", "rad"), ("rate_radps", "rad/s")):
        out += [f"\n## {q} [{unit}] by horizon\n", "| role | horizon " + head, "|---|---" + sep]
        for r in roles:
            for h in HORIZONS:
                out.append(f"| {r} | {h} s | " + " | ".join(
                    cell([run[r]["horizon"][h]["all"][q]["median"] for run in metrics[m]]) for m in models) + " |")

    bins = [k for k in metrics[models[0]][0][roles[-1]]["horizon"]["0.5"] if k.startswith("speed_")]
    out += ["\n## Position error at 0.5 s [m] by window start speed\n", "| role | speed bin " + head, "|---|---" + sep]
    for r in roles:
        for b in bins:
            n = metrics[models[0]][0][r]["horizon"]["0.5"][b]["n_windows"]
            if n:
                out.append(f"| {r} | {b} (n={n}) | " + " | ".join(
                    cell([run[r]["horizon"]["0.5"][b]["pos_m"]["median"] for run in metrics[m]]) for m in models) + " |")

    out += ["\n## Force / torque error (RMSE vs unfiltered measurements)\n",
            "| role | component " + head, "|---|---" + sep]
    for r in roles:
        for c in ("F_x", "F_y", "F_z", "T_x", "T_y", "T_z"):
            scale, unit = (1e3, "mN m") if c.startswith("T") else (1.0, "N")
            out.append(f"| {r} | {c} [{unit}] | " + " | ".join(
                cell([run[r]["wrench_error"][c] for run in metrics[m]], scale) for m in models) + " |")

    out += ["\n## One-step error (RMSE)\n", "| role | quantity " + head, "|---|---" + sep]
    for r in roles:
        for q in ("vel_mps", "rate_radps"):
            out.append(f"| {r} | {q} | " + " | ".join(
                cell([run[r]["one_step"][q] for run in metrics[m]]) for m in models) + " |")

    box = metrics[models[0]][0]
    out += ["\n## Fraction of states outside the training box\n", "| role | any | speed/rates/motors max | tilt (g_b) max |",
            "|---|---|---|---|"]
    for r in roles:
        fb = box[r]["frac_outside_train_box"]
        tilt = max(fb[k] for k in ("gb_x", "gb_y", "gb_z"))
        other = max(v for k, v in fb.items() if k not in ("any", "gb_x", "gb_y", "gb_z"))
        out.append(f"| {r} | {fb['any']:.1%} | {other:.1%} | {tilt:.1%} |")

    text = "\n".join(out)
    (ROOT / "results" / f"final_summary_{tag}.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

"""Data efficiency on DEVELOPMENT flights: does CoSTA need less training data than the DDM?

    uv run python scripts/data_efficiency.py [--config configs/data_efficiency.yaml]

Fits every model on nested subsets of 1, 2, 4, 7 slow flights (PBM parameters refitted on the
same subset), selects network checkpoints on the pseudo-val flights, and scores the held-out
interp / near / extrap dev flights: rollout errors AND the correction (wrench) error, where a
physics prior should show first. Uses run.py's fitting and batched evaluation; test flights are
refused.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as R  # noqa: E402

import quadrotor  # noqa: E402
from quadrotor import dataset as ds  # noqa: E402
from quadrotor.labels import features  # noqa: E402
from quadrotor.physics import load_params  # noqa: E402
from pseudo_extrapolation import check_dev_only  # noqa: E402

ROOT = R.ROOT
NETWORK_MODELS = ("costa", "costa_force", "ddm", "ddm_kin")
HORIZON = "0.5"


def median_over_seeds(results: dict, kind: str, n: int, getter) -> tuple[float, float, float]:
    vals = [getter(r) for key, r in results.items() if key.startswith(f"n{n}/{kind}/")]
    return float(np.median(vals)), float(min(vals)), float(max(vals))


def summary(results: dict, models: list[str], sizes: list[int], roles: list[str]) -> str:
    lines = [f"\n### Rollout position error at {HORIZON} s [m], window median; networks: median [min, max] over seeds\n"]
    for role in roles:
        lines += [f"\n**{role}**\n", "| fit flights | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
        for n in sizes:
            cells = []
            for kind in models:
                m, lo, hi = median_over_seeds(results, kind, n, lambda r: r[role]["horizon"][HORIZON]["all"]["pos_m"]["median"])
                cells.append(f"{m:.3g}" if lo == hi else f"{m:.3g} [{lo:.3g}, {hi:.3g}]")
            lines.append(f"| {n} | " + " | ".join(cells) + " |")
    lines.append("\n### Correction (wrench) RMSE on pseudo_interp states, F_z [N] / T_x [mN m]: median over seeds\n")
    lines += ["| fit flights | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    for n in sizes:
        cells = []
        for kind in models:
            fz = median_over_seeds(results, kind, n, lambda r: r["pseudo_interp"]["wrench_error"]["F_z"])[0]
            tx = median_over_seeds(results, kind, n, lambda r: r["pseudo_interp"]["wrench_error"]["T_x"])[0]
            cells.append(f"{fz:.3f} / {1e3 * tx:.1f}")
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot(results: dict, models: list[str], sizes: list[int], roles: list[str], path: Path) -> None:
    fig, axes = plt.subplots(1, len(roles) + 1, figsize=(4.2 * (len(roles) + 1), 3.8))
    for j, role in enumerate(roles):
        for kind in models:
            stats = [median_over_seeds(results, kind, n, lambda r: r[role]["horizon"][HORIZON]["all"]["pos_m"]["median"]) for n in sizes]
            med, lo, hi = map(np.array, zip(*stats))
            axes[j].plot(sizes, med, marker="o", label=kind)
            axes[j].fill_between(sizes, lo, hi, alpha=0.2)
        axes[j].set(xscale="log", yscale="log", xlabel="fit flights", title=f"{role}: position error at {HORIZON} s [m]")
        axes[j].set_xticks(sizes, [str(s) for s in sizes])
    for kind in models:
        fz = [median_over_seeds(results, kind, n, lambda r: r["pseudo_interp"]["wrench_error"]["F_z"])[0] for n in sizes]
        axes[-1].plot(sizes, fz, marker="o", label=kind)
    axes[-1].set(xscale="log", yscale="log", xlabel="fit flights", title="correction error F_z on interp [N]")
    axes[-1].set_xticks(sizes, [str(s) for s in sizes])
    axes[0].legend(fontsize=8)
    fig.suptitle("Data efficiency on development flights (band: seed range)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "data_efficiency.yaml")
    args = ap.parse_args()
    t0 = time.time()
    log = lambda msg: print(f"[{time.time() - t0:6.0f} s] {msg}", flush=True)  # noqa: E731

    exp = yaml.safe_load(args.config.read_text())
    cfg = yaml.safe_load((ROOT / exp["base_config"]).read_text())
    roles_cfg = yaml.safe_load((ROOT / exp["roles_from"]).read_text())["roles"]
    splits = yaml.safe_load((ROOT / cfg["splits"]).read_text())
    raw_dir = ROOT / splits["raw_dir"]
    real = R.assign_roles(raw_dir, splits, cfg.get("split_variant"))
    used = {"fit": exp["fit_order"], **{r: roles_cfg[r] for r in ["pseudo_val", *exp["eval_roles"]]}}
    check_dev_only(used, set(real["train"]) | set(real["val"]))

    out = ROOT / "results" / f"data_efficiency_{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True)
    for src in (args.config, ROOT / exp["base_config"], ROOT / exp["roles_from"]):
        (out / src.name).write_text(src.read_text())
    log(f"quadrotor package: {Path(quadrotor.__file__).parent}")

    load = lambda flights: [R.filtered(pc, None) for pc in ds.role_pieces(raw_dir, flights, splits["min_height"], 2)]  # noqa: E731
    eval_pieces = {r: load(roles_cfg[r]) for r in exp["eval_roles"]}
    n_max = int(round(max(cfg["eval"]["horizons_s"]) / ds.DT)) + 1
    val_pieces = [pc for pc in load(roles_cfg["pseudo_val"]) if len(pc.x) > n_max]
    every = cfg["data"]["train_subsample"]
    rows_full = len(R.stack_rows(load(exp["fit_order"]), every).x)

    results: dict[str, dict] = {}
    for n in exp["fit_sizes"]:
        fit_rows = R.stack_rows(load(exp["fit_order"][:n]), every)
        fit_feats = R.box_features(fit_rows.x, fit_rows.u)
        p, rec = R.fit_pbm_params(load_params(ROOT / cfg["params"]), fit_rows, cfg)
        cfg_n = copy.deepcopy(cfg)
        if exp.get("scale_epochs_to_steps", False):
            k = rows_full / len(fit_rows.x)
            es, opt = cfg_n["early_stopping"], cfg_n["optim"]
            es["val_every_epochs"] = max(1, int(round(es["val_every_epochs"] * k)))
            opt["max_epochs"] = int(round(opt["max_epochs"] * k))
        log(f"n = {n}: {len(fit_rows.x)} rows, kf {p.kf:.4e}, torque gain {np.round(p.torque_gain, 3).tolist()}, "
            f"val every {cfg_n['early_stopping']['val_every_epochs']} epochs")
        for kind in exp["models"]:
            for seed in exp["seeds"] if kind in NETWORK_MODELS else [0]:
                model, info = R.fit_model(kind, p, fit_rows, val_pieces, cfg_n, seed)
                key = f"n{n}/{kind}/s{seed}"
                results[key] = {"best_epoch": info.get("best_epoch"), "pbm_fit": rec,
                                **{role: R.evaluate_role(model, p, eval_pieces[role], cfg_n, exp["eval_stride"], fit_feats)
                                   for role in exp["eval_roles"]}}
                r = results[key]
                log(f"  {kind:11s} seed {seed}: {HORIZON} s pos " + "  ".join(
                    f"{role.removeprefix('pseudo_')} {r[role]['horizon'][HORIZON]['all']['pos_m']['median']:.3f} m"
                    for role in exp["eval_roles"]) + f"  | F_z err {r['pseudo_interp']['wrench_error']['F_z']:.3f} N")
                (out / "metrics.json").write_text(json.dumps(results, indent=1, default=float))

    text = "# Data efficiency (dev flights only)\n" + summary(results, exp["models"], exp["fit_sizes"], exp["eval_roles"])
    (out / "summary.md").write_text(text + "\n")
    plot(results, exp["models"], exp["fit_sizes"], exp["eval_roles"], out / "data_efficiency.png")
    print(text)
    log(f"saved to {out}")


if __name__ == "__main__":
    main()

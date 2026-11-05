"""Pseudo-extrapolation on DEVELOPMENT flights: does CoSTA degrade less than the DDM?

    uv run python scripts/pseudo_extrapolation.py [--config configs/pseudo_extrap.yaml]

Fits every model on slow dev flights, selects network checkpoints on slow validation
flights, and scores held-out slow (interp), faster (near) and fastest (extrap) dev flights.
Only train/val flights of the real split are allowed (checked), so the test set stays
untouched and this can be run and acted on as often as needed. Uses exactly run.py's
fitting and (batched) evaluation code.
"""

from __future__ import annotations

import argparse
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
import run as R  # noqa: E402  (shared fitting / evaluation helpers)

import quadrotor  # noqa: E402
from quadrotor import dataset as ds  # noqa: E402
from quadrotor.labels import features  # noqa: E402
from quadrotor.physics import load_params  # noqa: E402

ROOT = R.ROOT
EVAL_ROLES = ("pseudo_val", "pseudo_interp", "pseudo_near", "pseudo_extrap")
SUMMARY_HORIZONS = ("0.25", "0.5", "1.0")
NETWORK_MODELS = ("costa", "costa_force", "ddm", "ddm_kin")


def check_dev_only(exp_roles: dict[str, list[str]], dev: set[str]) -> None:
    listed = [f for flights in exp_roles.values() for f in flights]
    dupes = {f for f in listed if listed.count(f) > 1}
    if dupes:
        raise SystemExit(f"flights in two pseudo roles: {sorted(dupes)}")
    outside = set(listed) - dev
    if outside:
        raise SystemExit(f"not development (train/val) flights, refusing to use them: {sorted(outside)}")


def summary_table(results: dict, models: list[str]) -> str:
    """Markdown: median over seeds of the window-median error, [min, max] over seeds."""
    lines = []
    for metric, unit in (("pos_m", "m"), ("att_rad", "rad")):
        lines.append(f"\n### {metric} [{unit}], window median; networks: median [min, max] over seeds\n")
        lines.append("| role | horizon | " + " | ".join(models) + " |")
        lines.append("|---|---|" + "---|" * len(models))
        for role in EVAL_ROLES[1:]:
            for h in SUMMARY_HORIZONS:
                cells = []
                for kind in models:
                    vals = [r[role]["horizon"][h]["all"][metric]["median"] for key, r in results.items()
                            if key.split("/")[0] == kind]
                    cells.append(f"{np.median(vals):.3g}" if len(vals) == 1
                                 else f"{np.median(vals):.3g} [{min(vals):.3g}, {max(vals):.3g}]")
                lines.append(f"| {role} | {h} s | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot(results: dict, models: list[str], path: Path) -> None:
    roles = EVAL_ROLES[1:]
    fig, axes = plt.subplots(2, len(roles), figsize=(4.2 * len(roles), 6.5), sharey="row")
    for j, role in enumerate(roles):
        for kind in models:
            runs = [r for key, r in results.items() if key.split("/")[0] == kind]
            hs = sorted(runs[0][role]["horizon"], key=float)
            for i, metric in enumerate(("pos_m", "att_rad")):
                v = np.array([[r[role]["horizon"][h]["all"][metric]["median"] for h in hs] for r in runs])
                x = [float(h) for h in hs]
                axes[i, j].plot(x, np.median(v, 0), marker="o", label=kind)
                if len(runs) > 1:
                    axes[i, j].fill_between(x, v.min(0), v.max(0), alpha=0.2)
                axes[i, j].set(yscale="log", xlabel="horizon [s]")
        frac = results["pbm/s0"][role]["frac_outside_train_box"]["any"]
        axes[0, j].set_title(f"{role} ({100 * frac:.0f}% of states outside fit box)")
    axes[0, 0].set_ylabel("position error [m]")
    axes[1, 0].set_ylabel("attitude error [rad]")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Pseudo-extrapolation on development flights: window-median error (band: seed range)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "pseudo_extrap.yaml")
    args = ap.parse_args()
    t0 = time.time()
    log = lambda msg: print(f"[{time.time() - t0:6.0f} s] {msg}", flush=True)  # noqa: E731

    exp = yaml.safe_load(args.config.read_text())
    cfg = yaml.safe_load((ROOT / exp["base_config"]).read_text())
    splits = yaml.safe_load((ROOT / cfg["splits"]).read_text())
    raw_dir = ROOT / splits["raw_dir"]
    real = R.assign_roles(raw_dir, splits, cfg.get("split_variant"))
    check_dev_only(exp["roles"], set(real["train"]) | set(real["val"]))

    out = ROOT / "results" / f"pseudo_extrap_{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True)
    for src in (args.config, ROOT / exp["base_config"]):
        (out / src.name).write_text(src.read_text())
    log(f"quadrotor package: {Path(quadrotor.__file__).parent}")

    pieces = {role: [R.filtered(pc, None) for pc in ds.role_pieces(raw_dir, fl, splits["min_height"], 2)]
              for role, fl in exp["roles"].items()}
    fit_rows = R.stack_rows(pieces["fit"], cfg["data"]["train_subsample"])
    fit_feats = R.box_features(fit_rows.x, fit_rows.u)
    p, fit_record = R.fit_pbm_params(load_params(ROOT / cfg["params"]), fit_rows, cfg)
    n_max = int(round(max(cfg["eval"]["horizons_s"]) / ds.DT)) + 1
    val_pieces = [pc for pc in pieces["pseudo_val"] if len(pc.x) > n_max]
    log(f"{len(fit_rows.x)} fit rows from {len(exp['roles']['fit'])} flights; kf {fit_record['kf_prior']:.4e} -> {p.kf:.4e}; torque gain {np.round(p.torque_gain, 3).tolist()}")

    results: dict[str, dict] = {}
    for kind in exp["models"]:
        for seed in exp["seeds"] if kind in NETWORK_MODELS else [0]:
            model, info = R.fit_model(kind, p, fit_rows, val_pieces, cfg, seed)
            results[f"{kind}/s{seed}"] = {"best_epoch": info.get("best_epoch"),
                                          **{role: R.evaluate_role(model, p, pieces[role], cfg, exp["eval_stride"], fit_feats)
                                             for role in EVAL_ROLES}}
            r = results[f"{kind}/s{seed}"]
            log(f"{kind:11s} seed {seed}: 0.5 s pos median " + "  ".join(
                f"{role.removeprefix('pseudo_')} {r[role]['horizon']['0.5']['all']['pos_m']['median']:.3f} m"
                for role in EVAL_ROLES) + (f"  (best epoch {info['best_epoch']})" if info else ""))
            (out / "metrics.json").write_text(json.dumps(results, indent=1, default=float))

    coverage = {role: results["pbm/s0"][role]["frac_outside_train_box"]["any"] for role in EVAL_ROLES}
    table = summary_table(results, list(exp["models"]))
    header = (f"# Pseudo-extrapolation (dev flights only)\n\nkf fitted on fit flights: {p.kf:.4e}. "
              f"Fraction of states outside the fit box: "
              + ", ".join(f"{k.removeprefix('pseudo_')} {100 * v:.0f}%" for k, v in coverage.items()) + ".\n")
    (out / "summary.md").write_text(header + table + "\n")
    plot(results, list(exp["models"]), out / "pseudo_extrapolation.png")
    print(header + table)
    log(f"saved to {out}")


if __name__ == "__main__":
    main()

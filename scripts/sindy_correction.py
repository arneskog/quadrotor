"""SINDy-style discovery on the learned CoSTA correction (L04 "closing the loop"). DEVELOPMENT DATA ONLY.

    uv run python scripts/sindy_correction.py

1. Sample f_theta over visited states: CoSTA correction on the saved TRAIN rows of the final runs,
   c = costa.wrench - pbm.wrench (identical train-fitted PBM parameters), 5 seeds; target = seed mean.
2. Structure check: how much of the correction a dense fit of the whole library explains (validation).
3. STLSQ per component over a threshold grid, fitted on TRAIN rows. The threshold is a hyperparameter,
   chosen on VALIDATION flights: the sparsest model within 1 % of the best validation RMSE against the
   network (the same 1 % rule as checkpoint selection). Coefficients = LS refit on the kept terms (train).
   The same procedure directly on the noisy measured residual is the "raw data" comparison.
4. Ensemble: refit at the chosen thresholds on bootstrap resamples of whole TRAIN flights, per seed
   -> term inclusion probabilities and coefficient ranges.
5. Validation rollouts of PBM + discovered terms through the same windows as the final runs.
Test flights are never read (they were evaluated once, scripts/run_final.sh).
Writes results/sindy/ (summary.md, sindy.json, metrics_val_*.json, curves_val.npz, figures).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import plot_final as pf  # noqa: E402  (course figure style, saved-data helpers)
import run as runmod  # noqa: E402  (the exact evaluation of the final runs)
from matplotlib import pyplot as plt  # noqa: E402

from quadrotor import dataset as ds  # noqa: E402
from quadrotor.discovery import (  # noqa: E402
    COMPONENTS, TERM_NAMES, TERMS, SparseCorrection, equation, fit_all, library, rmse,
)

OUT = ROOT / "results" / "sindy"
SEEDS = (0, 1, 2, 3, 4)
THRESHOLDS = np.concatenate([[0.0], np.logspace(-3, 0, 31)])
REL_TOL = 0.01          # sparsest model within 1 % of the best validation RMSE
N_BOOT = 20             # flight bootstraps per seed
UNIT = ("N",) * 3 + ("N m",) * 3


def rows(run: str, role: str) -> dict[str, np.ndarray]:
    z = np.load(ROOT / "results" / run / "errors_dev.npz")
    return {k: z[f"{role}__{k}"] for k in ("row_x", "row_u", "row_flight", "row_wrench_pred", "row_wrench_meas")}


def load_data(p) -> dict:
    d = {}
    for role in ("train", "val"):
        pbm = rows("final_pbm_s0", role)
        x, u = pbm["row_x"].astype(float), pbm["row_u"].astype(float)
        costa = []
        for s in SEEDS:
            r = rows(f"final_costa_s{s}", role)
            assert np.array_equal(r["row_x"], pbm["row_x"]), "rows differ between runs"
            costa.append(r["row_wrench_pred"].astype(float) - pbm["row_wrench_pred"].astype(float))
        d[role] = {
            "x": x, "u": u, "flight": pbm["row_flight"],
            "theta": library(x, u, p),
            "c_seeds": np.stack(costa),                                   # (S, N, 6) learned corrections
            "c": np.mean(costa, axis=0),                                  # seed mean
            "res": pbm["row_wrench_meas"].astype(float) - pbm["row_wrench_pred"].astype(float),  # measured residual
            "lin": rows("final_linear_drag_s0", role)["row_wrench_pred"].astype(float)
                   - pbm["row_wrench_pred"].astype(float),
        }
    return d


def sweep(theta_tr, y_tr, theta_va, y_va_fit, y_va_meas) -> dict:
    """Per component: validation RMSE (vs the fit target and vs measurement) and #terms over THRESHOLDS."""
    out = {"rmse_fit": [], "rmse_meas": [], "n_terms": []}
    for t in THRESHOLDS:
        xi = fit_all(theta_tr, y_tr, np.full(6, t))
        pred = theta_va @ xi
        out["rmse_fit"].append(rmse(pred, y_va_fit))
        out["rmse_meas"].append(rmse(pred, y_va_meas))
        out["n_terms"].append((xi != 0).sum(0))
    return {k: np.array(v) for k, v in out.items()}                     # (T, 6)


def choose(sw: dict) -> np.ndarray:
    """Per component: the largest threshold whose validation RMSE is within REL_TOL of the best."""
    best = sw["rmse_fit"].min(0)
    ok = sw["rmse_fit"] <= (1 + REL_TOL) * best
    return np.array([THRESHOLDS[np.flatnonzero(ok[:, c]).max()] for c in range(6)])


def ensemble(d: dict, thr: np.ndarray) -> np.ndarray:
    """Xi for every (seed, flight bootstrap): (S*B, K, 6)."""
    rng = np.random.default_rng(0)
    tr = d["train"]
    flights = np.unique(tr["flight"])
    idx_of = {f: np.flatnonzero(tr["flight"] == f) for f in flights}
    out = []
    for s in range(len(SEEDS)):
        for _ in range(N_BOOT):
            idx = np.concatenate([idx_of[f] for f in rng.choice(flights, len(flights), replace=True)])
            out.append(fit_all(tr["theta"][idx], tr["c_seeds"][s][idx], thr))
    return np.stack(out)


def val_rollouts(models: dict) -> tuple[dict, dict]:
    """Evaluate models on the validation flights exactly like the final runs (same cfg, windows, stride)."""
    run_dir = ROOT / "results" / "final_costa_s0"
    run = json.loads((run_dir / "run.json").read_text())
    cfg = yaml.safe_load((run_dir / "config" / "train.yaml").read_text())
    splits = yaml.safe_load((run_dir / "config" / "splits.yaml").read_text())
    p = runmod.params_from_json(run["fitted_params"])
    pieces = ds.role_pieces(ROOT / splits["raw_dir"], run["flights"]["val"], splits["min_height"], min_rows=2)
    tr = rows("final_pbm_s0", "train")
    train_feats = runmod.box_features(tr["row_x"].astype(float), tr["row_u"].astype(float))
    stride = int(cfg["eval"]["window_stride"])
    metrics, curves = {}, {}
    for name, model in models.items():
        t0 = time.time()
        raw: dict = {}
        metrics[name] = runmod.evaluate_role(model, p, pieces, cfg, stride, train_feats, raw)
        curves[name] = raw["curve_q"]
        curves["curve_t"] = raw["curve_t"]
        print(f"  val rollouts {name}: 0.5 s pos {metrics[name]['horizon']['0.5']['all']['pos_m']['median']:.4f} m"
              f"  ({time.time()-t0:.0f} s)")
    return metrics, curves


# ------------------------------------------------------------------ figures
STYLE = {
    "pbm": pf.MODEL_STYLE["pbm"], "linear_drag": pf.MODEL_STYLE["linear_drag"], "costa": pf.MODEL_STYLE["costa"],
    "sindy": ("yellow", "PBM + SINDy (from CoSTA)", "-", "P"),
    "sindy_direct": ("violet", "PBM + SINDy (from labels)", ":", "X"),
}


def fig_structure(d: dict) -> plt.Figure:
    tr = d["train"]
    rng = np.random.default_rng(0)
    sel = rng.choice(len(tr["x"]), 8000, replace=False)
    v = tr["theta"][:, 1:4]
    w = tr["theta"][:, 16:19]
    xvar = [(v[:, 0], "$v_{b,x}$ [m/s]"), (v[:, 1], "$v_{b,y}$ [m/s]"), (v[:, 2], "$v_{b,z}$ [m/s]"),
            (v[:, 1], "$v_{b,y}$ [m/s]"), (v[:, 0], "$v_{b,x}$ [m/s]"), (w[:, 2], r"$\omega_z$ [rad/s]")]
    s1 = (tr["u"] / 1000).sum(1)
    fig, ax = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    for c, a in enumerate(ax.ravel()):
        xv, lab = xvar[c]
        sc = a.scatter(xv[sel], tr["c"][sel, c] * (1e3 if c >= 3 else 1), c=s1[sel], s=2, lw=0, alpha=0.5,
                       cmap="viridis")
        a.set(xlabel=lab, ylabel=f"CoSTA correction ${COMPONENTS[c]}$ [{'mN m' if c >= 3 else 'N'}]")
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label(r"$\sum\Omega_i$ [krad/s]")
    fig.suptitle("Structure check: learned correction (seed mean) on training states vs one physical variable",
                 fontsize=11)
    return fig


def fig_pareto(sw: dict, swd: dict, thr: np.ndarray, d: dict) -> plt.Figure:
    va = d["val"]
    fig, ax = plt.subplots(2, 3, figsize=(12, 7))
    for c, a in enumerate(ax.ravel()):
        k = 1e3 if c >= 3 else 1.0
        n = sw["n_terms"][:, c]
        a.plot(n, sw["rmse_fit"][:, c] * k, "o-", ms=3, color=pf.col("yellow"), label="vs CoSTA correction")
        a.plot(n, sw["rmse_meas"][:, c] * k, "s--", ms=3, color=pf.col("blue"), label="vs measured residual")
        a.plot(swd["n_terms"][:, c], swd["rmse_meas"][:, c] * k, "x:", ms=4, color=pf.col("violet"),
               label="fit to labels, vs measured")
        j = int(np.flatnonzero(THRESHOLDS == thr[c])[0])
        a.plot(n[j], sw["rmse_fit"][j, c] * k, "o", mfc="none", mec=pf.col("text"), ms=11, mew=1.5)
        a.axhline(rmse(va["c"], np.zeros_like(va["c"]))[c] * k, color=pf.col("muted"), lw=0.8, ls=":")
        a.axhline(rmse(va["c"], va["res"])[c] * k, color=pf.col("green"), lw=0.8, ls="-.")
        a.set(xlabel="number of terms", ylabel=f"validation RMSE [{'mN m' if c >= 3 else 'N'}]",
              title=f"${COMPONENTS[c]}$: {n[j]} terms chosen", xscale="symlog")
    ax[0, 0].plot([], [], ":", color=pf.col("muted"), label="zero correction (PBM), vs CoSTA")
    ax[0, 0].plot([], [], "-.", color=pf.col("green"), label="CoSTA itself, vs measured")
    ax[0, 0].legend(fontsize=7)
    fig.suptitle("Threshold choice on validation flights (circle: sparsest within 1 % of the best)", fontsize=11)
    fig.tight_layout()
    return fig


def fig_inclusion(prob: np.ndarray, med: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(prob, cmap="Greens" if pf.MODE == "light" else "viridis", vmin=0, vmax=1, aspect="auto")
    for k in range(prob.shape[0]):
        for c in range(6):
            if prob[k, c] >= 0.05:
                val = med[k, c] * (1e3 if c >= 3 else 1)
                ax.text(c, k, f"{val:+.3g}", ha="center", va="center", fontsize=7,
                        color="white" if prob[k, c] > 0.6 and pf.MODE == "light" else pf.col("text"))
    ax.set_xticks(range(6), [f"${c}$" for c in COMPONENTS])
    ax.set_yticks(range(len(TERM_NAMES)), [f"{n}  ({why})" for n, why in TERMS], fontsize=8)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.5)
    cb.set_label("inclusion probability (5 seeds x 20 flight bootstraps)")
    ax.set_title("Discovered terms: probability selected; text = median coefficient\n"
                 "(force N per term unit, torque mN m; $o=\\Omega/1000$, S1 = $\\sum o$)", fontsize=10)
    fig.tight_layout()
    return fig


def fig_parity(d: dict, xi: np.ndarray) -> plt.Figure:
    va = d["val"]
    pred = va["theta"] @ xi
    fig, ax = plt.subplots(2, 3, figsize=(12, 7.5))
    for c, a in enumerate(ax.ravel()):
        k = 1e3 if c >= 3 else 1.0
        x, y = va["c"][:, c] * k, pred[:, c] * k
        a.scatter(x, y, s=2, lw=0, alpha=0.35, color=pf.col("yellow"))
        lim = np.percentile(np.abs(np.concatenate([x, y])), 99.7)
        a.plot([-lim, lim], [-lim, lim], "--", color=pf.col("text"), lw=1)
        a.set_xlim(-lim, lim), a.set_ylim(-lim, lim), a.set_aspect("equal")
        r2 = 1 - np.mean((x - y) ** 2) / np.var(x)
        u = "mN m" if c >= 3 else "N"
        a.set(xlabel=f"CoSTA ${COMPONENTS[c]}$ [{u}]", ylabel=f"SINDy ${COMPONENTS[c]}$ [{u}]",
              title=f"${COMPONENTS[c]}$: $R^2$ = {r2:.2f}, {(xi[:, c] != 0).sum()} terms")
    fig.suptitle("Validation flights: discovered equation vs the network it was distilled from", fontsize=11)
    fig.tight_layout()
    return fig


def fig_rollouts(curves: dict) -> plt.Figure:
    t = curves["curve_t"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, (j, lab) in enumerate(((0, "position error [m]"), (2, "attitude error [rad]"))):
        for name in ("pbm", "linear_drag", "costa", "sindy", "sindy_direct"):
            hue, label, ls, _ = STYLE[name]
            cq = curves[name]
            ax[k].semilogy(t[1:], cq[0, 2, 1:, j], ls=ls, color=pf.col(hue), lw=1.6, label=label)
        ax[k].axvline(0.5, color=pf.col("muted"), lw=0.8, ls=":")
        ax[k].set(xlabel="time since window start [s]", ylabel=lab)
    ax[0].legend(fontsize=8)
    fig.suptitle("Validation rollouts (median over windows; CoSTA: median over seeds)", fontsize=11)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------- main
def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    p = runmod.params_from_json(json.loads((ROOT / "results" / "final_pbm_s0" / "run.json").read_text())["fitted_params"])
    d = load_data(p)
    tr, va = d["train"], d["val"]
    print(f"[{time.time()-t0:4.0f} s] train rows {len(tr['x'])}, val rows {len(va['x'])}, library {len(TERM_NAMES)} terms")

    # 2. structure / compressibility: dense library fit
    dense = fit_all(tr["theta"], tr["c"], np.zeros(6))
    r2_dense = 1 - rmse(va["theta"] @ dense, va["c"]) ** 2 / va["c"].var(0)

    # 3. threshold sweep, chosen on validation
    sw = sweep(tr["theta"], tr["c"], va["theta"], va["c"], va["res"])
    thr = choose(sw)
    xi = fit_all(tr["theta"], tr["c"], thr)
    swd = sweep(tr["theta"], tr["res"], va["theta"], va["res"], va["res"])   # directly on noisy labels
    thr_d = choose(swd)
    xi_d = fit_all(tr["theta"], tr["res"], thr_d)

    # 4. ensemble over seeds and flight bootstraps
    ens = ensemble(d, thr)
    prob = (ens != 0).mean(0)
    with np.errstate(all="ignore"):
        masked = np.where(ens != 0, ens, np.nan)
        med = np.nan_to_num(np.nanmedian(masked, 0))
        lo, hi = np.nan_to_num(np.nanpercentile(masked, 5, 0)), np.nan_to_num(np.nanpercentile(masked, 95, 0))

    # correction errors on validation rows
    pred, pred_d = va["theta"] @ xi, va["theta"] @ xi_d
    corr = {
        "pbm (zero correction)": rmse(0 * va["res"], va["res"]),
        "linear_drag": rmse(va["lin"], va["res"]),
        "costa (seed median)": np.median([rmse(cs, va["res"]) for cs in va["c_seeds"]], 0),
        "costa seed mean": rmse(va["c"], va["res"]),
        "sindy from costa": rmse(pred, va["res"]),
        "sindy from labels": rmse(pred_d, va["res"]),
    }
    fidelity = {"sindy from costa": rmse(pred, va["c"]), "sindy from labels": rmse(pred_d, va["c"]),
                "costa seeds spread (rmse of each seed vs seed mean, median)":
                    np.median([rmse(cs, va["c"]) for cs in va["c_seeds"]], 0)}
    print(f"[{time.time()-t0:4.0f} s] sparse fits done; validation rollouts ...")

    # 5. validation rollouts
    metrics, curves = val_rollouts({"sindy": SparseCorrection(p, xi), "sindy_direct": SparseCorrection(p, xi_d)})
    for name in ("pbm", "linear_drag"):
        z = np.load(ROOT / "results" / f"final_{name}_s0" / "errors_dev.npz")
        curves[name] = z["val__curve_q"]
    curves["costa"] = np.median([np.load(ROOT / "results" / f"final_costa_s{s}" / "errors_dev.npz")["val__curve_q"]
                                 for s in SEEDS], 0)
    saved = {m: [json.loads((ROOT / "results" / r / "metrics_dev.json").read_text())["val"] for r in pf.run_ids(m)]
             for m in ("pbm", "linear_drag", "costa")}
    for name, m in metrics.items():
        (OUT / f"metrics_val_{name}.json").write_text(json.dumps(m, indent=1, default=float))
    np.savez_compressed(OUT / "curves_val.npz", **curves)

    # ---- save numbers
    result = {
        "terms": [{"name": n, "meaning": w} for n, w in TERMS], "components": COMPONENTS,
        "thresholds_grid": THRESHOLDS, "rel_tol": REL_TOL, "n_boot": N_BOOT, "seeds": SEEDS,
        "chosen_threshold": thr, "chosen_threshold_direct": thr_d,
        "xi_from_costa": xi, "xi_from_labels": xi_d, "xi_dense": dense,
        "r2_dense_val_vs_costa": r2_dense,
        "inclusion_prob": prob, "coef_median": med, "coef_p05": lo, "coef_p95": hi,
        "val_correction_rmse_vs_measured": corr, "val_fidelity_rmse_vs_costa": fidelity,
        "sweep_from_costa": sw, "sweep_from_labels": swd,
    }
    to_list = lambda o: o.tolist() if isinstance(o, np.ndarray) else o  # noqa: E731
    (OUT / "sindy.json").write_text(json.dumps(result, indent=1, default=to_list))

    # ---- summary
    H = ("0.1", "0.25", "0.5", "1.0")
    L = [f"# SINDy on the CoSTA correction (development data only)\n",
         f"Train rows {len(tr['x'])} (27 flights), val rows {len(va['x'])} (4 flights); library {len(TERM_NAMES)} terms; "
         f"threshold = sparsest within {REL_TOL:.0%} of best validation RMSE vs the network.\n",
         "\n## Discovered equations (fit to the seed-mean CoSTA correction, coefficients refit on train)\n",
         "o = Omega/1000 [krad/s], S1 = sum o, S2 = sum o^2; v body frame [m/s], w [rad/s]. Force N, torque N m.\n"]
    for c in range(6):
        L.append(f"- **{COMPONENTS[c]}** (threshold {thr[c]:.3g}, {(xi[:, c] != 0).sum()} terms): "
                 f"`{equation_str(xi[:, c])}`")
    L += ["\n## Same procedure directly on the measured residual (raw labels)\n"]
    for c in range(6):
        L.append(f"- **{COMPONENTS[c]}** ({(xi_d[:, c] != 0).sum()} terms): `{equation_str(xi_d[:, c])}`")
    L += ["\n## Ensemble: terms selected in >= 80 % of fits (5 seeds x 20 flight bootstraps)\n",
          "| component | term | prob | median coef | 5-95 % |", "|---|---|---|---|---|"]
    for c in range(6):
        for k in np.argsort(-prob[:, c]):
            if prob[k, c] >= 0.8:
                L.append(f"| {COMPONENTS[c]} | {TERM_NAMES[k]} | {prob[k, c]:.2f} | {med[k, c]:+.4g} | "
                         f"[{lo[k, c]:+.4g}, {hi[k, c]:+.4g}] |")
    L += ["\n## Validation rows: correction RMSE vs measured residual\n",
          "| model | " + " | ".join(f"{c} [{u}]" for c, u in zip(COMPONENTS, UNIT)) + " |", "|---" * 7 + "|"]
    L += [f"| {k} | " + " | ".join(f"{v:.4g}" for v in vals) + " |" for k, vals in corr.items()]
    L += ["\n## Validation rows: distance to the network (fidelity) and structure check\n",
          "| quantity | " + " | ".join(COMPONENTS) + " |", "|---" * 7 + "|"]
    L += [f"| RMSE {k} | " + " | ".join(f"{v:.4g}" for v in vals) + " |" for k, vals in fidelity.items()]
    L.append("| R^2 of dense library fit vs CoSTA (all terms) | " + " | ".join(f"{v:.3f}" for v in r2_dense) + " |")
    L.append("| R^2 of chosen sparse fit vs CoSTA | " + " | ".join(
        f"{v:.3f}" for v in 1 - rmse(pred, va['c']) ** 2 / va['c'].var(0)) + " |")
    L += ["\n## Validation rollouts: position error [m] (median over windows)\n",
          "| model | " + " | ".join(f"{h} s" for h in H) + " |", "|---" * 5 + "|"]
    for m in ("pbm", "linear_drag", "costa"):
        L.append(f"| {m} | " + " | ".join(
            f"{np.median([r['horizon'][h]['all']['pos_m']['median'] for r in saved[m]]):.4g}" for h in H) + " |")
    for name, m in metrics.items():
        L.append(f"| {STYLE[name][1]} | " + " | ".join(f"{m['horizon'][h]['all']['pos_m']['median']:.4g}" for h in H) + " |")
    L += ["\n## Validation rollouts: attitude error [rad] (median over windows)\n",
          "| model | " + " | ".join(f"{h} s" for h in H) + " |", "|---" * 5 + "|"]
    for m in ("pbm", "linear_drag", "costa"):
        L.append(f"| {m} | " + " | ".join(
            f"{np.median([r['horizon'][h]['all']['att_rad']['median'] for r in saved[m]]):.4g}" for h in H) + " |")
    for name, m in metrics.items():
        L.append(f"| {STYLE[name][1]} | " + " | ".join(f"{m['horizon'][h]['all']['att_rad']['median']:.4g}" for h in H) + " |")
    text = "\n".join(L) + "\n"
    (OUT / "summary.md").write_text(text)
    print(text)

    # ---- figures (course style)
    pf.OUT = OUT
    pf.save_dual(lambda: fig_structure(d), "sindy_structure")
    pf.save_dual(lambda: fig_pareto(sw, swd, thr, d), "sindy_pareto")
    pf.save_dual(lambda: fig_inclusion(prob, med), "sindy_inclusion")
    pf.save_dual(lambda: fig_parity(d, xi), "sindy_parity")
    pf.save_dual(lambda: fig_rollouts(curves), "sindy_rollouts")
    print(f"done in {time.time()-t0:.0f} s -> {OUT}")


def equation_str(xi: np.ndarray) -> str:
    return equation(xi, fmt="{:+.4g}")


if __name__ == "__main__":
    main()

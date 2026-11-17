"""Figures of the final runs, made only from saved data (never re-evaluates the test set).

    uv run python scripts/plot_final.py

Reads results/final_<model>_s<seed>/{metrics,errors}_{dev,test}.(json|npz) and history.json.
Writes results/figures_final/<name>.png (white paper) and <name>_dark.png (transparent, light ink),
in the course figure style (TTK29-2026/python/pyutils/plotstyle.py, reproduced here so the course
repo is only read). Trajectory, parity and training figures show seed 0; the others use all seeds
(median, band or error bar = min-max over seeds 0-4).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from cycler import cycler  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures_final"
SEEDS = (0, 1, 2, 3, 4)
NETS = ("costa", "ddm", "ddm_kin")
MODELS = ("pbm", "linear_drag", *NETS)
TEST_ROLES = ("test_interp", "test_near", "test_extrap")
DEV_ROLES = ("train", "val")
QTY = ("pos", "vel", "att", "rate")  # column order of every saved error array

# ---------------------------------------------------------------- course style
COLORS = {
    "light": {"blue": "#2A5DB0", "orange": "#C25A1E", "green": "#1B7A4E", "red": "#B03A3A",
              "violet": "#6B4FC8", "yellow": "#A87B00", "text": "#000000", "muted": "#555555"},
    "dark": {"blue": "#7BA0FF", "orange": "#FFA870", "green": "#34D399", "red": "#F87171",
             "violet": "#B89AFF", "yellow": "#FBBF24", "text": "#EEEEF5", "muted": "#8888A0"},
}
MODE = "light"  # set by save_dual while a figure is drawn

# Model identity: fixed hue + line style + marker, so no pair relies on colour alone.
MODEL_STYLE = {
    "pbm": ("orange", "PBM", "--", "s"),
    "linear_drag": ("blue", "PBM + linear drag", "-.", "D"),
    "costa": ("green", "CoSTA", "-", "o"),
    "ddm": ("red", "DDM", "-", "^"),
    "ddm_kin": ("violet", r"DDM$_\mathrm{kin}$", ":", "v"),
}
ROLE_STYLE = {
    "train": ("blue", "training"),
    "val": ("violet", "validation"),
    "test_interp": ("green", "test (interpolation)"),
    "test_near": ("yellow", "test (near, 10-13 m/s)"),
    "test_extrap": ("red", r"test (extrapolation, $\geq$13 m/s)"),
}


def col(name: str) -> str:
    return COLORS[MODE][name]


def mcol(m: str) -> str:
    return col(MODEL_STYLE[m][0])


def mlab(m: str) -> str:
    return MODEL_STYLE[m][1]


def rcol(r: str) -> str:
    return col(ROLE_STYLE[r][0])


def rc_params(mode: str) -> dict:
    c = COLORS[mode]
    rc = {"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 11,
          "axes.grid": True, "grid.alpha": 0.3, "savefig.dpi": 130, "savefig.bbox": "tight",
          "axes.prop_cycle": cycler(color=[c[k] for k in ("blue", "orange", "green", "red", "violet")])}
    if mode == "dark":
        fg = c["text"]
        rc.update({"text.color": fg, "axes.labelcolor": fg, "axes.titlecolor": fg, "axes.edgecolor": fg,
                   "xtick.color": fg, "ytick.color": fg, "grid.color": c["muted"], "legend.labelcolor": fg,
                   "legend.facecolor": "none", "legend.edgecolor": c["muted"], "figure.facecolor": "none",
                   "axes.facecolor": "none", "savefig.transparent": True})
    else:
        rc.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.transparent": False})
    return rc


def save_dual(plot: Callable[[], plt.Figure], name: str) -> None:
    """Draw the figure twice (rcParams bind at artist creation) and save both variants."""
    global MODE
    for mode, suffix in (("light", ""), ("dark", "_dark")):
        MODE = mode
        with plt.rc_context(rc_params(mode)):
            fig = plot()
            fig.savefig(OUT / f"{name}{suffix}.png")
            plt.close(fig)
    MODE = "light"
    print(f"wrote {name}.png")


# ---------------------------------------------------------------- data
def run_ids(m: str) -> list[str]:
    return [f"final_{m}_s0"] if m in ("pbm", "linear_drag") else [f"final_{m}_s{s}" for s in SEEDS]


def metrics(m: str, tag: str) -> list[dict]:
    return [json.loads((ROOT / "results" / r / f"metrics_{tag}.json").read_text()) for r in run_ids(m)]


_NPZ: dict[tuple[str, str], np.lib.npyio.NpzFile] = {}


def npz(run: str, tag: str) -> np.lib.npyio.NpzFile:
    if (run, tag) not in _NPZ:
        _NPZ[run, tag] = np.load(ROOT / "results" / run / f"errors_{tag}.npz")
    return _NPZ[run, tag]


def arr(m: str, role: str, key: str, seed: int = 0) -> np.ndarray:
    run = f"final_{m}_s0" if m in ("pbm", "linear_drag") else f"final_{m}_s{seed}"
    return npz(run, "dev" if role in DEV_ROLES else "test")[f"{role}__{key}"]


def per_seed(m: str, role: str, key: str) -> list[np.ndarray]:
    seeds = (0,) if m in ("pbm", "linear_drag") else SEEDS
    return [arr(m, role, key, s) for s in seeds]


def quat_angle(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    """Rotation angle [rad] between unit quaternions (w,x,y,z) along the last axis; q and -q equal."""
    d = np.abs(np.sum(qa * qb, axis=-1))
    return 2.0 * np.arccos(np.clip(d, 0.0, 1.0))


# ---------------------------------------------------------------- figures
def fig_headtohead() -> plt.Figure:
    """Primary metric for every role: 0.5 s position error, bars = seed median, whiskers = min-max."""
    roles = [*DEV_ROLES, *TEST_ROLES]
    vals = {m: {} for m in MODELS}
    for m in MODELS:
        for tag, rs in (("dev", DEV_ROLES), ("test", TEST_ROLES)):
            runs = metrics(m, tag)
            for r in rs:
                vals[m][r] = [run[r]["horizon"]["0.5"]["all"]["pos_m"]["median"] for run in runs]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    xb, w = np.arange(len(roles)), 0.16
    for j, m in enumerate(MODELS):
        v = np.array([np.median(vals[m][r]) for r in roles])
        lo = v - np.array([min(vals[m][r]) for r in roles])
        hi = np.array([max(vals[m][r]) for r in roles]) - v
        yerr = None if m in ("pbm", "linear_drag") else np.vstack([lo, hi])
        ax.bar(xb + (j - 2) * w, v, w * 0.92, color=mcol(m), label=mlab(m), yerr=yerr, capsize=2,
               hatch="//" if m in ("pbm", "linear_drag") else None, edgecolor=col("text") if MODE == "light" else "none",
               linewidth=0, error_kw=dict(ecolor=col("text"), lw=0.8))
    ax.set_yscale("log")
    ax.axvline(1.5, color=col("muted"), lw=0.8)
    ax.set_xticks(xb, [ROLE_STYLE[r][1].replace(" (", "\n(") for r in roles], fontsize=9)
    for t, r in zip(ax.get_xticklabels(), roles):
        t.set_color(rcol(r))
    ax.set_ylabel("position error at 0.5 s [m]\n(median over windows)")
    ax.set_title("Development flights (left) and the single test evaluation (right)", fontsize=11)
    ax.legend(fontsize=9, ncol=5, loc="upper left")
    ax.set_ylim(top=ax.get_ylim()[1] * 2.5)
    fig.tight_layout()
    return fig


def fig_horizon() -> plt.Figure:
    """Error growth along the windows: median over windows (line) per test role; CoSTA with its IQR."""
    fig, ax = plt.subplots(2, 3, figsize=(12, 6.4), sharex=True, sharey="row")
    rows = ((0, "position error [m]"), (2, "attitude error [rad]"))
    for i, (k, ylab) in enumerate(rows):
        for j, r in enumerate(TEST_ROLES):
            a = ax[i, j]
            for m in MODELS:
                t = arr(m, r, "curve_t")
                cq = np.median(np.stack(per_seed(m, r, "curve_q")), axis=0)[0]  # all windows, (Q, T, 4)
                a.semilogy(t[1:], cq[2, 1:, k], ls=MODEL_STYLE[m][2], color=mcol(m), lw=1.6, label=mlab(m))
                if m == "costa":
                    a.fill_between(t[1:], cq[1, 1:, k], cq[3, 1:, k], color=mcol(m), alpha=0.18, lw=0,
                                   label="CoSTA, 25-75 % of windows")
            a.axvline(0.5, color=col("muted"), lw=0.8, ls=":")
            if i == 0:
                n = int(arr("pbm", r, "curve_n")[0])
                a.set_title(f"{ROLE_STYLE[r][1]}\n{len(arr('pbm', r, 'flights'))} flights, {n} windows",
                            color=rcol(r), fontsize=11)
            if i == 1:
                a.set_xlabel("time since window start [s]")
            if j == 0:
                a.set_ylabel(ylab)
    ax[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Error vs. horizon, rollouts from measured states driven by recorded motor speeds "
                 "(networks: median over seeds)", fontsize=11)
    fig.tight_layout()
    return fig


def fig_speed() -> plt.Figure:
    """0.5 s position error vs window start speed, all test windows pooled, 1 m/s bins."""
    edges = np.arange(0.0, 21.0, 1.0)
    speed = np.concatenate([arr("pbm", r, "win_speed") for r in TEST_ROLES])
    n = np.histogram(speed, edges)[0]
    keep = n >= 30
    mid = 0.5 * (edges[1:] + edges[:-1])
    v_train = float(np.linalg.norm(arr("pbm", "train", "row_x")[:, 3:6], axis=1).max())

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
    for m in MODELS:
        curves = []
        for s_err in (np.concatenate(e) for e in zip(*[per_seed(m, r, "win_err_h0.5") for r in TEST_ROLES])):
            idx = np.digitize(speed, edges) - 1
            curves.append([np.median(s_err[idx == b, 0]) if keep[b] else np.nan for b in range(len(mid))])
        c = np.array(curves)
        med = np.median(c, axis=0)
        ax[0].semilogy(mid, med, ls=MODEL_STYLE[m][2], marker=MODEL_STYLE[m][3], ms=4, color=mcol(m), label=mlab(m))
        if len(c) > 1:
            ax[0].fill_between(mid, c.min(0), c.max(0), color=mcol(m), alpha=0.2, lw=0)
    for a in ax:
        a.axvspan(v_train, edges[-1], color=col("muted"), alpha=0.12, lw=0)
        a.axvline(v_train, color=col("muted"), lw=0.8)
    ax[0].text(v_train + 0.2, 0.04, "faster than any\ntraining state", transform=ax[0].get_xaxis_transform(),
               va="bottom", fontsize=9, color=col("text"))
    ax[0].set_ylabel("position error at 0.5 s [m]\n(median over windows)")
    ax[0].legend(fontsize=9, loc="upper left")
    ax[0].set_title("All test windows pooled; band = min-max over 5 seeds", fontsize=11)
    ax[1].bar(mid, n, 0.9, color=col("muted"))
    ax[1].set_yscale("log")
    ax[1].set(xlabel="window start speed [m/s]", ylabel="windows")
    ax[1].set_xlim(0, mid[keep].max() + 0.5)
    fig.tight_layout()
    return fig


def pick_window(role: str = "test_extrap", vmin: float = 13.0) -> int:
    """A typical fast window: start speed >= vmin, CoSTA (seed 0) 0.5 s error closest to the median."""
    meas = arr("pbm", role, "traj_meas")
    pred = arr("costa", role, "traj_pred")
    t = arr("pbm", role, "curve_t")
    k = int(np.argmin(np.abs(t - 0.5)))
    speed = np.linalg.norm(meas[:, 0, 3:6], axis=1)
    cand = np.flatnonzero(speed >= vmin)
    e = np.linalg.norm(pred[cand, k, :3] - meas[cand, k, :3], axis=1)
    return int(cand[np.argmin(np.abs(e - np.median(e)))])


def fig_window() -> plt.Figure:
    role = "test_extrap"
    i = pick_window(role)
    t = arr("pbm", role, "curve_t")
    meas = arr("pbm", role, "traj_meas")[i]
    flight = arr("pbm", role, "flights")[arr("pbm", role, "traj_flight")[i]]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].plot(meas[:, 0], meas[:, 1], "--", color=col("text"), lw=2.2, label="measured", zorder=5)
    ax[0].plot(meas[0, 0], meas[0, 1], "o", color=col("text"), ms=5, zorder=6)
    for m in MODELS:
        p = arr(m, role, "traj_pred")[i]
        ax[0].plot(p[:, 0], p[:, 1], ls=MODEL_STYLE[m][2], color=mcol(m), lw=1.4, label=mlab(m))
        ax[1].semilogy(t[1:], np.linalg.norm(p[1:, :3] - meas[1:, :3], axis=1), ls=MODEL_STYLE[m][2],
                       color=mcol(m), label=mlab(m))
        ax[2].semilogy(t[1:], np.degrees(quat_angle(p[1:, 6:10], meas[1:, 6:10])), ls=MODEL_STYLE[m][2],
                       color=mcol(m), label=mlab(m))
    ax[0].set_aspect("equal", adjustable="datalim")
    ax[0].set(xlabel="$x$ [m]", ylabel="$y$ [m]", title="top view (start at the dot)")
    ax[1].set(xlabel="$t$ [s]", ylabel="position error [m]", title="position")
    ax[2].set(xlabel="$t$ [s]", ylabel="attitude error [deg]", title="attitude")
    for a in ax[1:]:
        a.axvline(0.5, color=col("muted"), lw=0.8, ls=":")
    ax[0].legend(fontsize=8)
    v0 = np.linalg.norm(meas[0, 3:6])
    fig.suptitle(f"One extrapolation test window: flight {flight}, start speed {v0:.1f} m/s "
                 "(typical: CoSTA error at 0.5 s = median of windows $\\geq$13 m/s; seed 0)", fontsize=11)
    fig.tight_layout()
    return fig


def fig_training() -> plt.Figure:
    hist = {m: [json.loads((ROOT / "results" / f"final_{m}_s{s}" / "history.json").read_text()) for s in SEEDS]
            for m in NETS}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for m in NETS:
        h = hist[m][0]["history"]
        ax[0].semilogy(h["epoch"], h["train_mse"], color=mcol(m), label=f"{mlab(m)}, training")
        ax[0].semilogy(h["epoch"], h["val_mse"], color=mcol(m), ls="--", label=f"{mlab(m)}, validation")
        for s, H in enumerate(hist[m]):
            hh = H["history"]
            kw = dict(color=mcol(m), lw=1.4 if s == 0 else 0.6, alpha=1.0 if s == 0 else 0.35)
            ax[1].plot(hh["epoch"], hh["val_score"], marker=MODEL_STYLE[m][3] if s == 0 else None, ms=3,
                       label=mlab(m) if s == 0 else None, **kw)
            b = hh["epoch"].index(H["best_epoch"])
            ax[1].plot(H["best_epoch"], hh["val_score"][b], "o", mfc="none", mec=mcol(m), ms=10 if s == 0 else 6,
                       mew=1.5 if s == 0 else 0.8, alpha=kw["alpha"])
    ax[0].set(xlabel="epoch", ylabel="standardized MSE",
              title="Regression loss (measured-state inputs, seed 0)\neach model's own target, not comparable across models")
    ax[1].set_yscale("log")
    ax[1].set(xlabel="epoch", ylabel="mean position RMSE [m]",
              title="Validation rollouts, 0.5 s windows (circles: selected)\nseed 0 bold, seeds 1-4 thin")
    ax[0].legend(fontsize=8, ncol=1)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_parity() -> plt.Figure:
    """Body force correction: measured residual (meas - PBM) vs the model's (pred - PBM), seed 0."""
    rng = np.random.default_rng(0)
    comps = ("F_x", "F_y", "F_z")
    nets = ("costa", "ddm", "ddm_kin")
    fig, ax = plt.subplots(len(nets), 3, figsize=(12, 11))
    for r in ("test_interp", "test_near", "test_extrap"):
        meas = arr("pbm", r, "row_wrench_meas")
        pbm = arr("pbm", r, "row_wrench_pred")
        sel = rng.choice(len(meas), min(4000, len(meas)), replace=False)
        for i, m in enumerate(nets):
            assert np.array_equal(arr(m, r, "row_x"), arr("pbm", r, "row_x")), "rows differ between runs"
            pred = arr(m, r, "row_wrench_pred")
            for j in range(3):
                x, y = meas[sel, j] - pbm[sel, j], pred[sel, j] - pbm[sel, j]
                ax[i, j].scatter(x, y, s=3, lw=0, alpha=0.35, color=rcol(r),
                                 label=ROLE_STYLE[r][1] if (i, j) == (0, 0) else None)
    for i, m in enumerate(nets):
        for j, c in enumerate(comps):
            lim = np.percentile(np.abs(np.concatenate([arr("pbm", r, "row_wrench_meas")[:, j]
                                                       - arr("pbm", r, "row_wrench_pred")[:, j]
                                                       for r in TEST_ROLES])), 99.5)
            a = ax[i, j]
            a.plot([-lim, lim], [-lim, lim], "--", color=col("text"), lw=1)
            a.set_xlim(-lim, lim)
            a.set_ylim(-lim, lim)
            a.set_aspect("equal")
            rm = np.mean([run["test_extrap"]["wrench_error"][c] for run in metrics(m, "test")[:1]])
            a.set_title(f"{mlab(m)}: ${c}$, extrap. RMSE {rm:.2f} N", fontsize=10)
            if i == len(nets) - 1:
                a.set_xlabel(f"measured ${c}$ $-$ PBM [N]")
            if j == 0:
                a.set_ylabel("model $-$ PBM [N]")
    leg = ax[0, 0].legend(fontsize=8, markerscale=4, loc="upper left")
    for h in leg.legend_handles:
        h.set_alpha(1)
    fig.suptitle("Force correction on test rows (body frame, seed 0, 4000 rows per role; "
                 "for DDM the model's force minus PBM)", fontsize=11)
    fig.tight_layout()
    return fig


def fig_split() -> plt.Figure:
    """Where the roles fly: one flight per role (top view) and visited body velocities vs the training box."""
    roles = [*DEV_ROLES, *TEST_ROLES]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw=dict(width_ratios=[1, 1.3]))
    for r in reversed(roles):  # training drawn last, on top
        x, fl = arr("pbm", r, "row_x"), arr("pbm", r, "row_flight")
        speed = np.linalg.norm(x[:, 3:6], axis=1)
        f = fl[np.argmax(speed)]  # the role's fastest flight
        xf = x[fl == f]
        ax[0].plot(xf[:, 0], xf[:, 1], ".", ms=1, color=rcol(r), alpha=0.6)
        ax[0].plot([], [], "-", color=rcol(r), label=ROLE_STYLE[r][1])
        # body-frame velocity, the network's input
        q = x[:, 6:10]
        w, qx, qy, qz = q.T
        R = np.stack([np.stack([1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - w * qz), 2 * (qx * qz + w * qy)], -1),
                      np.stack([2 * (qx * qy + w * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - w * qx)], -1),
                      np.stack([2 * (qx * qz - w * qy), 2 * (qy * qz + w * qx), 1 - 2 * (qx**2 + qy**2)], -1)], -2)
        vb = np.einsum("nji,nj->ni", R, x[:, 3:6])
        sel = rng.choice(len(vb), min(6000, len(vb)), replace=False)
        ax[1].plot(vb[sel, 0], vb[sel, 1], ".", ms=1.5, color=rcol(r), alpha=0.35)
    lo, hi = arr("pbm", "test_extrap", "train_box_min")[:2], arr("pbm", "test_extrap", "train_box_max")[:2]
    ax[1].plot([lo[0], hi[0], hi[0], lo[0], lo[0]], [lo[1], lo[1], hi[1], hi[1], lo[1]], "--", color=col("text"), lw=1.2)
    ax[1].text(lo[0], hi[1] + 0.3, "training box", fontsize=9)
    ax[0].set(xlabel="$x$ [m]", ylabel="$y$ [m]", title="Fastest flight of each role (top view)")
    ax[0].set_aspect("equal", adjustable="datalim")
    h, lab = ax[0].get_legend_handles_labels()
    ax[0].legend(h[::-1], lab[::-1], fontsize=8, loc="upper right")
    ax[1].set(xlabel="body velocity $v_{b,x}$ [m/s]", ylabel="body velocity $v_{b,y}$ [m/s]",
              title="Visited body-frame velocities (network input)")
    ax[1].set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    return fig


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, f in (("headtohead", fig_headtohead), ("error_vs_horizon", fig_horizon),
                    ("error_vs_speed", fig_speed), ("example_window", fig_window),
                    ("training", fig_training), ("force_parity", fig_parity), ("split", fig_split)):
        save_dual(f, name)


if __name__ == "__main__":
    main()

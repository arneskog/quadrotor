"""Run one model end to end: fit on TRAIN, select on VAL, evaluate, save to results/<run_id>/.

    uv run python scripts/run.py --model pbm
    uv run python scripts/run.py --model costa --seed 0 --alpha-cutoff 20
    uv run python scripts/run.py --from-run results/<run_id> --roles test_interp test_near test_extrap

Test roles are only evaluated from a frozen, saved run, and only once per run (L04:
"test frozen, once"). Every model goes through the same flights, pieces, windows,
recorded motor speeds, RK4 step and metrics; only the WrenchModel differs.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from quadrotor import dataset as ds  # noqa: E402
from quadrotor.evaluate import (  # noqa: E402
    STATE_ERRORS,
    _window_errors,
    correction_error,
    one_step_error,
    one_step_predictions,
    state_errors,
    validation_score,
)
from quadrotor.integrators import rk4_step  # noqa: E402
from quadrotor.labels import features, lowpass, measured_wrench, targets  # noqa: E402
from quadrotor.models import (  # noqa: E402
    PBM,
    LinearDrag,
    WrenchModel,
    load_model,
    make_dynamics,
    save_model,
)
from quadrotor.physics import QuadParams, body_velocity, load_params, quat_to_rotmat  # noqa: E402
try:  # kinematics-only DDM (ddm_kin); a package without it still runs the other models
    from quadrotor.labels import network_features  # noqa: E402
except ImportError:  # pragma: no cover
    def network_features(kind: str, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return features(x, u)
from quadrotor.train import TrainConfig, fit_kf, fit_linear_drag, fit_network, fit_torque_gain  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("pbm", "linear_drag", "costa", "costa_force", "ddm", "ddm_kin")
KINEMATIC_MODELS = ("ddm_kin",)
BOX_COLUMNS = [*ds.FEATURES, "gb_x", "gb_y", "gb_z"]  # coverage also in tilt (gravity direction in body)
DEV_ROLES = ("train", "val")
TEST_ROLES = ("test_interp", "test_near", "test_extrap")


# ---------------------------------------------------------------- data ----


@dataclasses.dataclass(frozen=True)
class Rows:
    """Stacked rows of many pieces (alpha already low-passed per piece)."""

    x: np.ndarray
    u: np.ndarray
    acc: np.ndarray
    alpha: np.ndarray


def filtered(piece: ds.Piece, cutoff_hz: float | None) -> ds.Piece:
    """Piece with alpha low-passed inside the piece (never across a gap)."""
    return dataclasses.replace(piece, alpha=lowpass(piece.alpha, piece.dt, cutoff_hz))


def stack_rows(pieces: list[ds.Piece], every: int = 1) -> Rows:
    return Rows(*(np.vstack([getattr(pc, k)[::every] for pc in pieces]) for k in ("x", "u", "acc", "alpha")))


def box_features(x: np.ndarray, u: np.ndarray) -> pd.DataFrame:
    """Coverage columns for the training-box fraction: network features + gravity direction g_b."""
    g_b = quat_to_rotmat(x[:, 6:10])[:, 2, :]
    return pd.DataFrame(np.hstack([features(x, u), g_b]), columns=BOX_COLUMNS)


def assign_roles(raw_dir: Path, splits: dict, variant: str | None) -> dict[str, list[str]]:
    return ds.split_flights(ds.flight_ids(raw_dir), splits, variant)


# ------------------------------------------------------------- fitting ----


def fit_pbm_params(p: QuadParams, train: Rows, cfg: dict) -> tuple[QuadParams, dict]:
    """Refit the PBM parameters listed in cfg['pbm_fit'] on TRAIN rows; mass and the rest stay priors."""
    record = {"kf_prior": p.kf, "torque_gain_prior": p.torque_gain.tolist()}
    if cfg.get("pbm_fit", {}).get("kf", False):
        p = dataclasses.replace(p, kf=fit_kf(train.u, train.acc[:, 2], p))
    if cfg.get("pbm_fit", {}).get("torque_gain", False):  # after kf: the gain scales the refitted thrusts
        p = dataclasses.replace(p, torque_gain=fit_torque_gain(train.u, train.acc, train.alpha, train.x[:, 10:13], p))
    record["kf"] = p.kf
    record["torque_gain"] = p.torque_gain.tolist()
    return p, record


def val_scorer(p: QuadParams, val_pieces: list[ds.Piece], es: dict):
    """Validation-rollout score of a candidate model (lower is better), for checkpoint selection."""
    windows = [(pc.x, pc.u, pc.dt) for pc in val_pieces]
    return lambda model: validation_score(make_dynamics(model, p), windows, es["horizon_s"], es["window_stride_s"])


def fit_model(kind: str, p: QuadParams, train: Rows, val_pieces: list[ds.Piece], cfg: dict, seed: int,
              cutoff: float | None = None):
    """Returns (model, info). PBM: nothing to fit. Linear drag: least squares. Networks: fit_network.

    For networks, info["history"]["val_mse"] is the standardized regression loss on validation rows
    (labels low-passed with the same cutoff as training), logged at every validation. It is only
    recorded; checkpoints are still selected on the validation ROLLOUT score.
    """
    if kind == "pbm":
        return PBM(p), {}
    if kind == "linear_drag":
        y = targets(kind, train.x, train.u, train.acc, train.alpha, p)
        return LinearDrag(p, fit_linear_drag(body_velocity(train.x), y[:, :3])), {}
    scales = cfg["normalization"]["scales"]
    fallback = [scales["vb"]] * 3 + [scales["w"]] * 3 + [scales["mot"]] * 4
    if kind in KINEMATIC_MODELS:
        fallback += [scales.get("gb", 1.0)] * 3
    fallback = np.array(fallback, dtype=float)
    X = network_features(kind, train.x, train.u)
    Y = targets(kind, train.x, train.u, train.acc, train.alpha, p)
    score = val_scorer(p, val_pieces, cfg["early_stopping"])
    val = stack_rows([filtered(pc, cutoff) for pc in val_pieces], cfg["data"]["train_subsample"])
    Xv = network_features(kind, val.x, val.u)
    Yv = targets(kind, val.x, val.u, val.acc, val.alpha, p)
    val_mse: list[float] = []

    def score_and_log(model) -> float:
        z = model.net(model.norm.scale_in(Xv))
        val_mse.append(float(np.mean((z - model.norm.scale_out(Yv)) ** 2)))
        return score(model)

    model, info = fit_network(
        kind, p, X, Y, fallback, score_and_log, TrainConfig.from_dict(cfg), seed, cfg["normalization"]["min_std"]
    )
    info["history"]["val_mse"] = val_mse
    return model, info


# ---------------------------------------------------------- evaluation ----


def model_wrench_error(model: WrenchModel, p: QuadParams, rows: Rows) -> dict[str, float]:
    """RMSE of the model's wrench vs. the measured wrench on measured states.

    For CoSTA / linear drag this IS the correction error (motor parts cancel); for the PBM it is
    the size of the residual; for the DDM the error of its full-wrench regression.
    """
    force, torque = model.wrench(rows.x, rows.u)
    f_meas, t_meas = measured_wrench(rows.acc, rows.alpha, rows.x[:, 10:13], p)
    return correction_error(np.hstack([force, torque]), np.hstack([f_meas, t_meas]))


def summarize(err: np.ndarray, speed: np.ndarray, bins: list[float]) -> dict:
    """Median and quartiles per error channel, overall and per window-start speed bin."""
    def stats(e: np.ndarray) -> dict:
        q = np.quantile(e, [0.25, 0.5, 0.75], axis=0) if len(e) else np.full((3, len(STATE_ERRORS)), np.nan)
        return {name: {"q25": float(q[0, j]), "median": float(q[1, j]), "q75": float(q[2, j])}
                for j, name in enumerate(STATE_ERRORS)} | {"n_windows": int(len(e)),
                                                         "frac_non_finite": float(np.mean(~np.isfinite(e[:, 0]))) if len(e) else 0.0}
    out = {"all": stats(err)}
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (speed >= lo) & (speed < hi)
        out[f"speed_{lo:g}_{hi:g}"] = stats(err[sel])
    return out


def evaluate_role(model: WrenchModel, p: QuadParams, pieces: list[ds.Piece], cfg: dict,
                  stride: int, train_feats: pd.DataFrame, raw: dict | None = None) -> dict:
    """Metrics of one role. pieces must carry UNFILTERED alpha: the reported force/torque error is
    always against the same (raw) measured wrench, whatever label filter a model was trained with.

    If raw is a dict it is filled with arrays for plotting (saved by main as errors_<tag>.npz), so no
    figure ever needs a second evaluation of the test set. Errors are [pos m, vel m/s, att rad, rate rad/s].
      flights                              flight names (index -> id)
      win_err_h<h>        (W, 4)           error of every window at horizon h [s]
      win_speed, win_flight, win_piece, win_start   window start speed [m/s], flight index, piece index
                                           (order of the role's pieces), start row in that piece
      curve_t             (T,)             time grid [s] (eval.curve_dt_s) along the windows
      curve_q             (B, Q, T, 4)     error quantiles curve_quantiles over windows, per time; bin 0 = all
                                           windows, bins 1.. = window start speed bins (eval.speed_bins)
      curve_n             (B,)             windows per bin
      traj_pred, traj_meas (K, T, 13)      model vs measured state on curve_t for windows every
                                           eval.traj_every_s [s] of every piece; traj_flight, traj_piece, traj_start
      row_x, row_u, row_acc, row_alpha     measured state, motor speeds, specific force, UNFILTERED ang acc
                                           on rows ::stride (with row_flight, row_piece); recompute anything from these
      row_wrench_pred, row_wrench_meas (R, 6)  model vs measured [F N, T N m] on those rows
      os_err (S, 4), os_flight             one-step error per row (rows k = 0, stride, ...)
      box_columns, train_box_min, train_box_max   per-feature training box (coverage plots)
    """
    ev = cfg["eval"]
    f = make_dynamics(model, p)
    horizons = [float(h) for h in ev["horizons_s"]]
    bins = [float(b) for b in ev["speed_bins"]]
    t_grid = np.arange(0.0, max(horizons) + 1e-9, float(ev.get("curve_dt_s", 0.01)))
    qs = [float(q) for q in ev.get("curve_quantiles", [0.1, 0.25, 0.5, 0.75, 0.9])]
    per_h: dict[float, list] = {h: [] for h in horizons}
    curves, speeds, one_pred, one_meas, os_flight, win_flight, win_piece, win_start = [], [], [], [], [], [], [], []
    traj_pred, traj_meas, traj_flight, traj_piece, traj_start = [], [], [], [], []
    flights = sorted({pc.flight for pc in pieces})
    for pi, pc in enumerate(pieces):
        n_max = int(round(max(horizons) / pc.dt))
        if len(pc.x) <= n_max:
            continue
        fi = flights.index(pc.flight)
        err = _window_errors(pc.x, pc.u, f, pc.dt, n_max, stride)       # (W, n_max+1, 4), one batch
        for h in horizons:                                               # = horizon_rollout_error
            per_h[h].append(err[:, int(round(h / pc.dt))])
        idx = np.minimum(np.round(t_grid / pc.dt).astype(int), n_max)   # nearest sample (164 Hz flight)
        curves.append(err[:, idx].astype(np.float32))
        starts = ds.window_starts(len(pc.x), n_max, stride)
        speeds.append(np.linalg.norm(pc.x[starts, 3:6], axis=1))
        win_flight.append(np.full(len(starts), fi))
        win_piece.append(np.full(len(starts), pi))
        win_start.append(starts)
        if raw is not None:                                              # example trajectories
            sel = starts[:: max(1, int(round(float(ev.get("traj_every_s", 2.0)) / (pc.dt * stride))))]
            xs, keep = pc.x[sel].copy(), [pc.x[sel].copy()]
            with np.errstate(all="ignore"):
                for k in range(n_max):
                    xs = rk4_step(f, xs, pc.u[sel + k], pc.dt)
                    keep.append(xs)
            traj_pred.append(np.stack(keep, axis=1)[:, idx].astype(np.float32))
            traj_meas.append(np.stack([pc.x[i + idx] for i in sel]).astype(np.float32))
            traj_flight.append(np.full(len(sel), fi))
            traj_piece.append(np.full(len(sel), pi))
            traj_start.append(sel)
        pred = one_step_predictions(pc.x, pc.u, f, pc.dt, stride)
        one_pred.append(pred)
        one_meas.append(pc.x[1:][::stride])
        os_flight.append(np.full(len(pred), fi))
    speed = np.concatenate(speeds) if speeds else np.zeros(0)
    rows = stack_rows(pieces, stride)
    role_feats = box_features(rows.x, rows.u)
    if raw is not None and curves:
        curve = np.concatenate(curves)                                   # (W, T, 4)
        masks = [np.ones(len(speed), bool)] + [(speed >= lo) & (speed < hi) for lo, hi in zip(bins[:-1], bins[1:])]
        cq = np.full((len(masks), len(qs), len(t_grid), len(STATE_ERRORS)), np.nan, np.float32)
        for b, m in enumerate(masks):
            if m.any():
                cq[b] = np.quantile(curve[m], qs, axis=0)
        f_pred, t_pred = model.wrench(rows.x, rows.u)
        f_meas, t_meas = measured_wrench(rows.acc, rows.alpha, rows.x[:, 10:13], p)
        raw.update({
            "flights": np.array(flights),
            **{f"win_err_h{h:g}": np.vstack(per_h[h]).astype(np.float32) for h in horizons},
            "win_speed": speed.astype(np.float32),
            "win_flight": np.concatenate(win_flight).astype(np.int16),
            "win_piece": np.concatenate(win_piece).astype(np.int32),
            "win_start": np.concatenate(win_start).astype(np.int32),
            "curve_t": t_grid, "curve_quantiles": np.array(qs), "curve_q": cq,
            "curve_n": np.array([int(m.sum()) for m in masks]), "speed_bins": np.array(bins),
            "traj_pred": np.concatenate(traj_pred), "traj_meas": np.concatenate(traj_meas),
            "traj_flight": np.concatenate(traj_flight).astype(np.int16),
            "traj_piece": np.concatenate(traj_piece).astype(np.int32),
            "traj_start": np.concatenate(traj_start).astype(np.int32),
            "row_x": rows.x.astype(np.float32), "row_u": rows.u.astype(np.float32),
            "row_acc": rows.acc.astype(np.float32), "row_alpha": rows.alpha.astype(np.float32),
            "row_flight": np.concatenate([np.full(len(pc.x[::stride]), flights.index(pc.flight))
                                          for pc in pieces]).astype(np.int16),
            "row_piece": np.concatenate([np.full(len(pc.x[::stride]), pi)
                                         for pi, pc in enumerate(pieces)]).astype(np.int32),
            "row_wrench_pred": np.hstack([f_pred, t_pred]).astype(np.float32),
            "row_wrench_meas": np.hstack([f_meas, t_meas]).astype(np.float32),
            "os_err": state_errors(np.vstack(one_pred), np.vstack(one_meas)).astype(np.float32),
            "os_flight": np.concatenate(os_flight).astype(np.int16),
            "box_columns": np.array(BOX_COLUMNS),
            "train_box_min": train_feats.min().to_numpy(), "train_box_max": train_feats.max().to_numpy(),
        })
    return {
        "n_pieces": len(pieces),
        "frac_outside_train_box": ds.fraction_outside_box(train_feats, role_feats).to_dict(),
        "wrench_error": model_wrench_error(model, p, rows),
        "one_step": one_step_error(np.vstack(one_pred), np.vstack(one_meas)) if one_pred else {},
        "horizon": {str(h): summarize(np.vstack(per_h[h]), speed, bins) for h in horizons if per_h[h]},
    }


def plot_horizon(metrics: dict, path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, len(STATE_ERRORS), figsize=(4 * len(STATE_ERRORS), 3.4))
    for role, m in metrics.items():
        hs = sorted(m["horizon"], key=float)
        for ax, name in zip(axes, STATE_ERRORS):
            med = [m["horizon"][h]["all"][name]["median"] for h in hs]
            lo = [m["horizon"][h]["all"][name]["q25"] for h in hs]
            hi = [m["horizon"][h]["all"][name]["q75"] for h in hs]
            x = [float(h) for h in hs]
            ax.plot(x, med, marker="o", label=role)
            ax.fill_between(x, lo, hi, alpha=0.2)
    for ax, name in zip(axes, STATE_ERRORS):
        ax.set(xlabel="horizon [s]", ylabel=name, yscale="log")
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main ----


def params_to_json(p: QuadParams) -> dict:
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in dataclasses.asdict(p).items()}


def params_from_json(d: dict) -> QuadParams:
    return QuadParams(**{k: (np.asarray(v, dtype=float) if isinstance(v, list) else v) for k, v in d.items()})


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "train.yaml")
    ap.add_argument("--model", choices=MODELS, help="default: config 'model'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha-cutoff", type=float, default=None,
                    help="[Hz] low-pass of the ang acc training labels; default: config labels.alpha_lowpass_hz")
    ap.add_argument("--roles", nargs="+", default=list(DEV_ROLES), choices=list(DEV_ROLES + TEST_ROLES))
    ap.add_argument("--from-run", type=Path, help="evaluate a saved, frozen run instead of fitting")
    ap.add_argument("--eval-stride", type=int, help="rows between window starts (default: config eval.window_stride)")
    ap.add_argument("--run-id", help="default: <time>_<model>_s<seed>")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    t_start = time.time()
    if set(args.roles) & set(TEST_ROLES) and args.from_run is None:
        raise SystemExit("test roles are evaluated only from a frozen run: pass --from-run results/<run_id>")

    if args.from_run is not None:  # ---- frozen evaluation of a saved run
        run_dir = args.from_run
        run = json.loads((run_dir / "run.json").read_text())
        cfg = yaml.safe_load((run_dir / "config" / "train.yaml").read_text())
        splits = yaml.safe_load((run_dir / "config" / "splits.yaml").read_text())
        p = params_from_json(run["fitted_params"])
        model = load_model(run_dir / "model.npz", p)
        cutoff, kind = run["alpha_cutoff_hz"], run["model"]
        tag = "test" if set(args.roles) & set(TEST_ROLES) else "dev_rerun"
        if tag == "test" and (run_dir / "metrics_test.json").exists():
            raise SystemExit(f"{run_dir}/metrics_test.json exists: the test set is evaluated once per run")
    else:  # ---- fit a new run
        cfg = yaml.safe_load(args.config.read_text())
        splits = yaml.safe_load((ROOT / cfg["splits"]).read_text())
        kind = args.model or cfg["model"]
        # ONE cutoff shared by all network models (selected on validation, scripts/select_alpha.py)
        cutoff = args.alpha_cutoff if args.alpha_cutoff is not None else cfg["labels"].get("alpha_lowpass_hz")
        run_id = args.run_id or f"{datetime.now():%Y%m%d-%H%M%S}_{kind}_s{args.seed}"
        run_dir = ROOT / "results" / run_id
        (run_dir / "config").mkdir(parents=True)
        for src in (args.config, ROOT / cfg["params"], ROOT / cfg["splits"]):
            shutil.copy(src, run_dir / "config" / src.name)
        tag = "dev"

    raw_dir = ROOT / splits["raw_dir"]
    roles = assign_roles(raw_dir, splits, cfg.get("split_variant"))
    n_max = int(round(max(cfg["eval"]["horizons_s"]) / ds.DT)) + 1
    # Pieces keep the UNFILTERED alpha. The alpha low-pass (a validation choice) is applied only to
    # the network's training labels, so the PBM fit, every evaluation and the reported force/torque
    # error are identical for all models and cutoffs.
    load = lambda role: ds.role_pieces(raw_dir, roles[role], splits["min_height"], min_rows=2)  # noqa: E731
    print(f"[{time.time()-t_start:5.0f} s] loading train flights ({len(roles['train'])})")
    train_pieces = load("train")
    every = cfg["data"]["train_subsample"]
    train_rows = stack_rows(train_pieces, every)                                  # raw labels
    label_rows = stack_rows([filtered(pc, cutoff) for pc in train_pieces], every)  # network labels
    train_feats = box_features(train_rows.x, train_rows.u)

    if args.from_run is None:
        p0 = load_params(ROOT / cfg["params"])
        p, fit_record = fit_pbm_params(p0, train_rows, cfg)
        print(f"[{time.time()-t_start:5.0f} s] kf prior {fit_record['kf_prior']:.4e} -> train fit {p.kf:.4e}")
        val_pieces = [pc for pc in load("val") if len(pc.x) > n_max]
        print(f"[{time.time()-t_start:5.0f} s] fitting {kind} (seed {args.seed})")
        model, info = fit_model(kind, p, label_rows, val_pieces, cfg, args.seed, cutoff)
        save_model(model, run_dir / "model.npz")
        if info:
            (run_dir / "history.json").write_text(json.dumps(info, indent=1, default=float))
        run = {"model": kind, "seed": args.seed, "alpha_cutoff_hz": cutoff, "split_variant": cfg.get("split_variant"),
               "flights": roles, "fitted_params": params_to_json(p), "pbm_fit": fit_record, "git_commit": git_commit(),
               "best_epoch": info.get("best_epoch"), "best_val_score": info.get("best_val_score")}
        (run_dir / "run.json").write_text(json.dumps(run, indent=1, default=float))

    stride = args.eval_stride or int(cfg["eval"]["window_stride"])
    metrics, raw = {}, {}
    for role in args.roles:
        pieces = train_pieces if role == "train" else load(role)
        print(f"[{time.time()-t_start:5.0f} s] evaluating {kind} on {role} ({len(roles[role])} flights, stride {stride})")
        role_raw: dict = {}
        metrics[role] = evaluate_role(model, p, pieces, cfg, stride, train_feats, role_raw)
        raw.update({f"{role}__{k}": v for k, v in role_raw.items()})
    np.savez_compressed(run_dir / f"errors_{tag}.npz", **raw)
    metrics["_meta"] = {"eval_stride_rows": stride, "roles": args.roles, "seconds": time.time() - t_start,
                        "wrench_error_labels": "unfiltered measured alpha (same for every model and cutoff)"}
    (run_dir / f"metrics_{tag}.json").write_text(json.dumps(metrics, indent=1, default=float))
    plot_horizon({r: m for r, m in metrics.items() if r != "_meta"}, run_dir / f"horizon_error_{tag}.png",
                 f"{kind}, seed {run['seed']}: median and IQR over windows")

    for role in args.roles:
        h = metrics[role]["horizon"]
        line = "  ".join(f"{k}s pos {v['all']['pos_m']['median']:.3f} m att {v['all']['att_rad']['median']:.3f} rad"
                         for k, v in h.items())
        print(f"{role:12s} {line}")
    print(f"saved to {run_dir}  ({time.time()-t_start:.0f} s)")


if __name__ == "__main__":
    main()

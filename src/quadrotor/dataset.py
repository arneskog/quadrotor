"""NeuroBEM data loading, frame conversions and flight-level splits.

Splits are always by WHOLE FLIGHT, never by row.

Raw NeuroBEM conventions (verified against finite differences of pos / vel):
  pos [m] world (z-up); vel [m/s] BODY frame (front-left-up);
  quat (x, y, z, w) body -> world;
  acc [m/s^2] body-frame specific force: R(q) acc - g e_z = d(v_world)/dt;
  ang vel [rad/s], ang acc [rad/s^2] body frame; mot [rad/s]; dmot = d(mot)/dt.

Standardized columns produced by load_segment:
  t, p_{x,y,z}, v_{x,y,z} (world), vb_{x,y,z} (body), q_{w,x,y,z}, w_{x,y,z},
  acc_{x,y,z} (body specific force), alpha_{x,y,z}, mot_{1..4}, dmot_{1..4}, vbat.

Rows are dropped (min_height) and segments end, so a segment is NOT one
continuous time series in general: use continuous_pieces() before anything that
runs along time (rollouts, finite differences, filters).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

DT: float = 0.0025  # [s] NeuroBEM sample period (400 Hz)
ROLES: tuple[str, ...] = ("train", "val", "test_interp", "test_near", "test_extrap")
FEATURES: list[str] = (
    ["vb_x", "vb_y", "vb_z", "w_x", "w_y", "w_z"] + [f"mot_{i}" for i in range(1, 5)]
)

_SEG_RE = re.compile(r"merged_(.+)_seg_(\d+)\.csv$")
_RENAME = {
    "pos x": "p_x", "pos y": "p_y", "pos z": "p_z",
    "vel x": "vb_x", "vel y": "vb_y", "vel z": "vb_z",
    "quat w": "q_w", "quat x": "q_x", "quat y": "q_y", "quat z": "q_z",
    "ang vel x": "w_x", "ang vel y": "w_y", "ang vel z": "w_z",
    "acc x": "acc_x", "acc y": "acc_y", "acc z": "acc_z",
    "ang acc x": "alpha_x", "ang acc y": "alpha_y", "ang acc z": "alpha_z",
    **{f"mot {i}": f"mot_{i}" for i in range(1, 5)},
    **{f"dmot {i}": f"dmot_{i}" for i in range(1, 5)},
}  # fmt: skip


def load_segment(path: Path) -> pd.DataFrame:
    """Load one NeuroBEM segment CSV into SI units with standardized column names."""
    df = pd.read_csv(path).rename(columns=_RENAME)
    q_xyzw = df[["q_x", "q_y", "q_z", "q_w"]].to_numpy()
    v_world = Rotation.from_quat(q_xyzw).apply(df[["vb_x", "vb_y", "vb_z"]].to_numpy())
    df[["v_x", "v_y", "v_z"]] = v_world
    return df


def flight_ids(raw_dir: Path) -> list[str]:
    """Sorted ids of all flights in raw_dir (e.g. '2021-02-03-13-43-38')."""
    return sorted({_SEG_RE.search(p.name).group(1) for p in raw_dir.glob("merged_*_seg_*.csv")})


def load_flight(raw_dir: Path, flight: str) -> list[pd.DataFrame]:
    """All segments of one flight, in segment order."""
    paths = sorted(
        raw_dir.glob(f"merged_{flight}_seg_*.csv"),
        key=lambda p: int(_SEG_RE.search(p.name).group(2)),
    )
    if not paths:
        raise FileNotFoundError(f"no segments for flight {flight} in {raw_dir}")
    return [load_segment(p) for p in paths]


def load_flights(raw_dir: Path, flights: list[str] | None = None) -> dict[str, list[pd.DataFrame]]:
    """Segments of the given flights (default: all in raw_dir), keyed by flight id."""
    return {f: load_flight(raw_dir, f) for f in (flights or flight_ids(raw_dir))}


def flight_max_speed(segments: list[pd.DataFrame]) -> float:
    """Max speed ||v|| [m/s] over all segments of a flight (frame-invariant)."""
    return max(
        float(np.linalg.norm(s[["vb_x", "vb_y", "vb_z"]].to_numpy(), axis=1).max())
        for s in segments
    )


def sample_period(segment: pd.DataFrame) -> float:
    """Median sample period [s] of a segment. Not always DT: one flight
    (2021-02-18-16-43-54) is logged at ~164 Hz."""
    return float(np.median(np.diff(segment["t"].to_numpy())))


def continuous_pieces(
    segment: pd.DataFrame, min_height: float | None = None, dt: float | None = None
) -> list[pd.DataFrame]:
    """Split a segment into gap-free pieces after dropping rows with p_z < min_height [m].

    A new piece starts wherever consecutive kept rows are more than 1.5*dt apart;
    dt defaults to the segment's own sample_period (measured BEFORE dropping rows).
    Pieces with fewer than 2 rows are discarded. Index is reset per piece.
    """
    if dt is None:
        dt = sample_period(segment)
    df = segment if min_height is None else segment[segment["p_z"] >= min_height]
    if len(df) == 0:
        return []
    breaks = np.flatnonzero(np.diff(df["t"].to_numpy()) > 1.5 * dt) + 1
    bounds = np.concatenate([[0], breaks, [len(df)]])
    return [
        df.iloc[a:b].reset_index(drop=True) for a, b in zip(bounds[:-1], bounds[1:]) if b - a >= 2
    ]


def flight_pieces(
    segments: list[pd.DataFrame], min_height: float | None = None, dt: float | None = None
) -> list[pd.DataFrame]:
    """Continuous pieces of all segments of one flight (never spanning segments)."""
    return [p for s in segments for p in continuous_pieces(s, min_height, dt)]


def window_starts(n_rows: int, horizon_steps: int, stride: int = 1) -> np.ndarray:
    """Start indices of rollout windows inside ONE continuous piece of n_rows rows.

    A window starting at i uses rows i .. i + horizon_steps, so every returned i
    satisfies i + horizon_steps <= n_rows - 1. Empty if the piece is too short.
    """
    if horizon_steps < 1 or stride < 1:
        raise ValueError("horizon_steps and stride must be >= 1")
    return np.arange(0, max(n_rows - horizon_steps, 0), stride)


def rollout_windows(
    pieces: list[pd.DataFrame], horizon_steps: int, stride: int = 1
) -> list[tuple[int, int]]:
    """(piece index, start row) of every window that fits entirely inside one piece."""
    return [
        (k, int(i))
        for k, p in enumerate(pieces)
        for i in window_starts(len(p), horizon_steps, stride)
    ]


def split_flights(
    all_flights: list[str], split_cfg: dict, variant: str | None = None
) -> dict[str, list[str]]:
    """Assign flight ids to the roles in ROLES from the explicit lists in split_cfg.

    variant 'b' moves split_cfg['variants']['b']['add_to_train'] from test_near
    to train. Raises if roles overlap, a listed flight is unknown, or a flight
    in all_flights is unassigned.
    """
    roles = {r: list(split_cfg["roles"][r]) for r in ROLES}
    if variant is not None:
        moved = list(split_cfg["variants"][variant]["add_to_train"])
        not_near = set(moved) - set(roles["test_near"])
        if not_near:
            raise ValueError(f"variant {variant}: flights not in test_near: {sorted(not_near)}")
        roles["train"] += moved
        roles["test_near"] = [f for f in roles["test_near"] if f not in set(moved)]

    assigned = [f for r in ROLES for f in roles[r]]
    dupes = {f for f in assigned if assigned.count(f) > 1}
    if dupes:
        raise ValueError(f"flights in more than one role: {sorted(dupes)}")
    unknown = set(assigned) - set(all_flights)
    if unknown:
        raise ValueError(f"unknown flights in split config: {sorted(unknown)}")
    missing = set(all_flights) - set(assigned)
    if missing:
        raise ValueError(f"flights not assigned to any role: {sorted(missing)}")
    return roles


def fraction_outside_box(train_feats: pd.DataFrame, test_feats: pd.DataFrame) -> pd.Series:
    """Fraction of test rows outside the TRAINING per-feature [min, max] box.

    Returns one entry per feature column plus 'any' (row outside in at least one
    feature). Values in [0, 1]. Per-feature boxes miss unseen combinations of
    in-range values, so 'any' is a lower bound on novelty.
    """
    cols = list(train_feats.columns)
    lo, hi = train_feats.min(), train_feats.max()
    out = (test_feats[cols] < lo) | (test_feats[cols] > hi)
    return pd.concat([out.mean(), pd.Series({"any": out.any(axis=1).mean()})])

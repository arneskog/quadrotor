"""Survey all NeuroBEM flights: per-flight speed / height / tilt statistics.

Writes data/flight_summary.csv (sorted by max speed) and prints a short summary.

Frames (verified against d(pos)/dt): `vel` is body-frame (FLU), quaternion is
(x, y, z, w) body->world, `pos` is world-frame (z-up). v_world = R(q) v_body.

Usage: uv run python scripts/survey_flights.py [raw_dir] [out_csv]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

GROUND_Z = 0.5  # [m] below this height the flight is counted as including takeoff/landing
SEG_RE = re.compile(r"merged_(.+)_seg_(\d+)\.csv$")


def segment_stats(df: pd.DataFrame) -> dict[str, float]:
    """Stats for one segment. Speeds [m/s], heights [m], tilt [deg], duration [s]."""
    q = df[["quat x", "quat y", "quat z", "quat w"]].to_numpy()
    v_body = df[["vel x", "vel y", "vel z"]].to_numpy()
    v_world = Rotation.from_quat(q).apply(v_body)
    z = df["pos z"].to_numpy()
    # Tilt = angle between body z and world z: cos(tilt) = 1 - 2(qx^2 + qy^2).
    cos_tilt = np.clip(1.0 - 2.0 * (q[:, 0] ** 2 + q[:, 1] ** 2), -1.0, 1.0)
    t = df["t"].to_numpy()
    return {
        "duration_s": t[-1] - t[0] + np.median(np.diff(t)),
        "max_speed": np.linalg.norm(v_body, axis=1).max(),
        "max_hspeed": np.linalg.norm(v_world[:, :2], axis=1).max(),
        "min_z": z.min(),
        "max_z": z.max(),
        "max_tilt_deg": np.degrees(np.arccos(cos_tilt).max()),
    }


def survey(raw_dir: Path) -> pd.DataFrame:
    """One row per flight, aggregated over its segments."""
    rows = []
    for path in sorted(raw_dir.glob("merged_*_seg_*.csv")):
        m = SEG_RE.search(path.name)
        stats = segment_stats(pd.read_csv(path))
        rows.append({"flight": m.group(1), "seg": int(m.group(2)), **stats})
    seg = pd.DataFrame(rows)
    out = seg.groupby("flight").agg(
        n_segments=("seg", "count"),
        duration_s=("duration_s", "sum"),
        max_speed=("max_speed", "max"),
        max_hspeed=("max_hspeed", "max"),
        min_z=("min_z", "min"),
        max_z=("max_z", "max"),
        max_tilt_deg=("max_tilt_deg", "max"),
    )
    out["takeoff_landing"] = out["min_z"] < GROUND_Z
    return out.sort_values("max_speed").reset_index()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    raw_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "data/raw/processed_data"
    out_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "data/flight_summary.csv"
    df = survey(raw_dir)
    df.to_csv(out_csv, index=False, float_format="%.3f")

    s = df["max_speed"]
    print(f"{len(df)} flights, {df.n_segments.sum()} segments, "
          f"{df.duration_s.sum() / 60:.1f} min total -> {out_csv}")
    print("max speed [m/s] quantiles:")
    print(s.quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]).round(2).to_string())
    for thr in (5, 10, 15):
        sel = s > thr
        print(f"  > {thr:2d} m/s: {sel.sum():3d} flights, {df.duration_s[sel].sum() / 60:5.1f} min")
    print(f"takeoff/landing (z < {GROUND_Z} m): {df.takeoff_landing.sum()} flights")


if __name__ == "__main__":
    main()

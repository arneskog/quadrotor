from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from quadrotor.dataset import (
    DT,
    ROLES,
    continuous_pieces,
    flight_ids,
    fraction_outside_box,
    rollout_windows,
    split_flights,
    window_starts,
)

ROOT = Path(__file__).resolve().parents[1]
SPLITS = yaml.safe_load((ROOT / "configs/splits.yaml").read_text())
RAW = ROOT / SPLITS["raw_dir"]


def _segment(z: np.ndarray, t0: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame({"t": t0 + DT * np.arange(len(z)), "p_z": z})


def _cfg() -> dict:
    return {
        "roles": {
            "train": ["a", "b"],
            "val": ["c"],
            "test_interp": ["d"],
            "test_near": ["e", "f"],
            "test_extrap": ["g"],
        },
        "variants": {"b": {"add_to_train": ["e"]}},
    }


# --- continuous pieces / rollout windows -------------------------------------


def test_edge_rows_below_min_height_trimmed_without_split() -> None:
    z = np.r_[np.full(5, 0.2), np.full(20, 2.0), np.full(5, 0.2)]
    pieces = continuous_pieces(_segment(z), min_height=0.5)
    assert len(pieces) == 1
    assert len(pieces[0]) == 20


def test_interior_dropped_rows_split_the_segment() -> None:
    z = np.r_[np.full(10, 2.0), np.full(3, 0.2), np.full(15, 2.0)]
    pieces = continuous_pieces(_segment(z), min_height=0.5)
    assert [len(p) for p in pieces] == [10, 15]
    for p in pieces:
        assert np.allclose(np.diff(p["t"]), DT)


def test_time_gap_without_dropped_rows_splits() -> None:
    seg = pd.concat([_segment(np.full(10, 2.0)), _segment(np.full(10, 2.0), t0=1.0)])
    assert [len(p) for p in continuous_pieces(seg)] == [10, 10]


def test_window_starts_fit_inside_piece() -> None:
    n, h = 50, 7
    starts = window_starts(n, h, stride=1)
    assert starts[0] == 0
    assert starts[-1] + h == n - 1
    assert len(window_starts(h, h)) == 0  # too short: h rows need h + 1


def test_no_rollout_window_crosses_a_gap() -> None:
    z = np.r_[np.full(40, 2.0), np.full(3, 0.1), np.full(25, 2.0), np.full(2, 0.1), np.full(9, 2.0)]
    pieces = continuous_pieces(_segment(z), min_height=0.5)
    h = 10
    windows = rollout_windows(pieces, horizon_steps=h, stride=1)
    assert windows  # the two long pieces give windows, the 9-row one gives none
    assert {k for k, _ in windows} == {0, 1}
    for k, i in windows:
        t = pieces[k]["t"].to_numpy()[i : i + h + 1]
        assert len(t) == h + 1
        assert np.allclose(np.diff(t), DT)


# --- split_flights -----------------------------------------------------------


def test_split_roles_disjoint_and_complete() -> None:
    roles = split_flights(list("abcdefg"), _cfg())
    assigned = [f for r in ROLES for f in roles[r]]
    assert sorted(assigned) == list("abcdefg")


def test_variant_b_moves_flights_out_of_test_near() -> None:
    roles = split_flights(list("abcdefg"), _cfg(), variant="b")
    assert "e" in roles["train"]
    assert "e" not in roles["test_near"]


def test_split_rejects_overlap_unknown_and_missing() -> None:
    cfg = _cfg()
    cfg["roles"]["val"].append("a")
    with pytest.raises(ValueError, match="more than one role"):
        split_flights(list("abcdefg"), cfg)
    with pytest.raises(ValueError, match="unknown"):
        split_flights(list("abcdef"), _cfg())
    with pytest.raises(ValueError, match="not assigned"):
        split_flights(list("abcdefgh"), _cfg())


# --- fraction_outside_box ----------------------------------------------------


def test_fraction_outside_box_per_feature_and_any() -> None:
    train = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    test = pd.DataFrame({"x": [0.5, 2.0, 0.5, 2.0], "y": [0.5, 0.5, -1.0, 2.0]})
    out = fraction_outside_box(train, test)
    assert out["x"] == 0.5
    assert out["y"] == 0.5
    assert out["any"] == 0.75
    assert fraction_outside_box(train, train)["any"] == 0.0


# --- the real split config ---------------------------------------------------


needs_data = pytest.mark.skipif(not RAW.exists(), reason="NeuroBEM raw data not present")


@needs_data
@pytest.mark.parametrize("variant", [None, "b"])
def test_config_split_is_valid_for_all_flights(variant: str | None) -> None:
    roles = split_flights(flight_ids(RAW), SPLITS, variant=variant)
    assert "2021-02-03-13-43-38" in roles["train"]  # inspected flight never in test


@needs_data
def test_every_flight_has_a_maneuver_label() -> None:
    assert set(SPLITS["maneuver"]) == set(flight_ids(RAW))


def test_pieces_use_the_segment_sample_rate() -> None:
    # 2021-02-18-16-43-54 is logged at ~164 Hz; a fixed 400 Hz dt would split every row.
    dt = 1 / 164
    seg = pd.DataFrame({"t": dt * np.arange(30), "p_z": np.full(30, 2.0)})
    assert [len(p) for p in continuous_pieces(seg, min_height=0.5)] == [30]

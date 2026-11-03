"""Evaluation: correction error, one-step error, and multi-horizon rollout error
over many short windows (0.1-1 s) started from measured states."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def correction_error(pred: Array, label: Array) -> dict[str, float]:
    """RMSE of predicted vs. label force/torque correction."""
    raise NotImplementedError


def one_step_error(x_pred: Array, x_meas: Array) -> dict[str, float]:
    """One-step state prediction error (position, velocity, attitude, rates)."""
    raise NotImplementedError


def horizon_rollout_error(
    x_meas: Array, u_meas: Array, f, dt: float, horizons_s: list[float], stride: int
) -> dict[float, Array]:
    """Error vs. horizon from windows started at measured states."""
    raise NotImplementedError

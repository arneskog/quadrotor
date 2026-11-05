"""Evaluation: correction error, one-step error, and multi-horizon rollout error
over many short windows (0.1-1 s) started from measured states.

All functions take a Dynamics f(x, u) -> xdot (see models.make_dynamics), so every
model is scored by the same code, integrator, windows and recorded motor speeds.
State errors are [position m, velocity m/s, attitude angle rad, body rates rad/s].
Rollout windows of one piece are integrated together as one batch.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from quadrotor.integrators import rk4_step 
from quadrotor.dataset import window_starts

Array = NDArray[np.float64]
Dynamics = Callable[[Array, Array], Array]

STATE_ERRORS: tuple[str, ...] = ("pos_m", "vel_mps", "att_rad", "rate_radps")
WRENCH_COMPONENTS: tuple[str, ...] = ("F_x", "F_y", "F_z", "T_x", "T_y", "T_z")


def attitude_error(q_pred, q_meas):
    w1, v1 = q_pred[..., :1], q_pred[..., 1:]
    w2, v2 = q_meas[..., :1], q_meas[..., 1:]
    scalar = np.sum(q_pred * q_meas, axis=-1)          # w1 w2 + v1·v2
    vector = w1 * v2 - w2 * v1 - np.cross(v1, v2)      # vector part of conj(q1) ⊗ q2
    return 2.0 * np.arctan2(np.linalg.norm(vector, axis=-1), np.abs(scalar))


def state_errors(x_pred: Array, x_meas: Array) -> Array:
    """Per-row errors (..., 4): |dp| [m], |dv| [m/s], attitude angle [rad], |dw| [rad/s]."""
    return np.stack([
        np.linalg.norm(x_pred[..., 0:3] - x_meas[..., 0:3], axis=-1),     # position [m]
        np.linalg.norm(x_pred[..., 3:6] - x_meas[..., 3:6], axis=-1),     # velocity [m/s]
        attitude_error(x_pred[..., 6:10], x_meas[..., 6:10]),             # attitude [rad]
        np.linalg.norm(x_pred[..., 10:13] - x_meas[..., 10:13], axis=-1), # body rates [rad/s]
        ], axis=-1)


def correction_error(pred: Array, label: Array) -> dict[str, float]:
    """RMSE of predicted vs. label force [N] / torque [N m] correction, per component (N, 6)."""
    rmse = np.sqrt(np.mean((pred-label)**2, axis=0))
    return dict(zip(WRENCH_COMPONENTS, map(float, rmse)))


def one_step_predictions(
    x_meas: Array, u_meas: Array, f: Dynamics, dt: float, stride: int = 1
) -> Array:
    """One RK4 step from measured states k = 0, stride, ... < N-1: predictions of x_meas[1:][::stride]."""
    ks = np.arange(0, len(x_meas) - 1, stride) # What times start a prediction from
    with np.errstate(all="ignore"):
        return rk4_step(f, x_meas[ks], u_meas[ks], dt) # every start state at once, as a batch


def one_step_error(x_pred: Array, x_meas: Array) -> dict[str, float]:
    """RMSE over rows of the one-step state errors (position, velocity, attitude, rates)."""
    rmse = np.sqrt(np.mean(state_errors(x_pred, x_meas)**2, axis=0))
    return dict(zip(STATE_ERRORS, map(float, rmse)))


def _window_errors(x_meas: Array, u_meas: Array, f: Dynamics, dt: float, n_steps: int, stride: int) -> Array:
    """State errors along every window of n_steps in ONE continuous piece: (n_windows, n_steps+1, 4).

    Windows start at measured states (dataset.window_starts) and are driven by the recorded
    motor speeds; all windows advance together as one batch. A non-finite rollout scores +inf.
    """
    starts = window_starts(len(x_meas), n_steps, stride)
    out = np.zeros((len(starts), n_steps + 1, len(STATE_ERRORS)))
    x = x_meas[starts].copy()

    with np.errstate(all="ignore"):
        for k in range(n_steps):
            x = rk4_step(f, x, u_meas[starts + k], dt)

            err = state_errors(x, x_meas[starts + k + 1])
            out[:, k + 1] = np.where(np.isfinite(err), err, np.inf)
    return out


def horizon_rollout_error(
    x_meas: Array, u_meas: Array, f: Dynamics, dt: float, horizons_s: list[float], stride: int
) -> dict[float, Array]:
    """Error vs. horizon from windows started at measured states of ONE continuous piece.

    Every horizon uses the same window starts (those that fit the longest horizon).
    Returns {horizon [s]: state errors at that horizon, (n_windows, 4)}.
    """
    steps = {h: int(round(h / dt)) for h in horizons_s}
    err = _window_errors(x_meas, u_meas, f, dt, max(steps.values()), stride)
    return {h: err[:, k] for h, k in steps.items()}



def validation_score(
    f: Dynamics, pieces: Sequence[tuple[Array, Array, float]], horizon_s: float, stride_s: float
) -> float:
    """Checkpoint-selection metric: mean over all windows of the position RMSE [m] along the window.

    pieces: (x_meas, u_meas, dt) per continuous VALIDATION piece; window starts every stride_s [s]
    (converted per piece). +inf if any rollout is non-finite.
    """
    per_window = [np.sqrt(np.mean(_window_errors(x, u , f, dt, int(round(horizon_s / dt)), max(1, int(round(stride_s / dt))))[:, 1:, 0]**2, axis=1)) 
                  for x, u, dt in pieces]
    scores = np.concatenate(per_window)
    if scores.size == 0:
        raise ValueError("No validation window fits the horizon")
    return float(np.mean(scores)) if np.all(np.isfinite(scores)) else float("inf")

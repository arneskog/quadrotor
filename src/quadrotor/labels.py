"""Residual force/torque labels for CoSTA (body frame).

F_res = m * a_meas - F_pbm(x, u)
T_res = J * alpha_meas - T_pbm(x, u)
Computed directly from the residual, never via an operator applied to the state error.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, filtfilt

from quadrotor.physics import QuadParams

Array = NDArray[np.float64]


def residual_labels(
    x: Array, u: Array, a_meas: Array, alpha_meas: Array, p: QuadParams
) -> tuple[Array, Array]:
    """Body-frame residual force [N] (N, 3) and torque [N m] (N, 3)."""
    raise NotImplementedError


def lowpass(signal: Array, dt: float, cutoff_hz: float | None) -> Array:
    """Zero-phase 2nd-order Butterworth low-pass along axis 0. None -> unchanged copy.

    signal must be ONE continuous piece (see dataset.continuous_pieces): filtering
    across a time gap would smear the jump into the labels.
    """
    if cutoff_hz is None:
        return np.array(signal, dtype=np.float64, copy=True)
    b, a = butter(2, cutoff_hz, fs=1.0 / dt)
    return filtfilt(b, a, signal, axis=0)


def features(x: Array, u: Array) -> Array:
    """Network input (N, 10): body velocity, body rates, motor speeds. No position, no time."""
    raise NotImplementedError

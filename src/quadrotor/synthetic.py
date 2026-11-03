"""Synthetic benchmark: "true" model = PBM + known drag, a tracking controller,
and a reference-trajectory generator. Used to validate the CoSTA pipeline where
the true correction is known."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def true_drag(v_body: Array, w: Array, omega: Array) -> tuple[Array, Array]:
    """Known body-frame drag force [N] and torque [N m] added to the PBM."""
    raise NotImplementedError


def controller(x: Array, x_ref: Array) -> Array:
    """Geometric/cascaded controller returning motor speeds [rad/s] (4,)."""
    raise NotImplementedError


def generate_flight(seed: int, duration: float, dt: float, max_speed: float) -> dict[str, Array]:
    """Simulate one closed-loop flight. Returns dict with t, x, u, a, alpha, f_corr, t_corr."""
    raise NotImplementedError

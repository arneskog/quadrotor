"""Fixed-step integrators for the quadrotor state."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
Dynamics = Callable[[Array, Array], Array]  # (x, u) -> xdot


def rk4_step(f: Dynamics, x: Array, u: Array, dt: float) -> Array:
    """One RK4 step with zero-order-hold input; renormalizes the quaternion x[6:10]."""
    raise NotImplementedError


def rollout(x0: Array, u_seq: Array, f: Dynamics, dt: float) -> Array:
    """Integrate from x0 under inputs u_seq (N, 4). Returns states (N + 1, 13)."""
    raise NotImplementedError

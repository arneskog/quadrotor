"""Fixed-step integrators for the quadrotor state."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
Dynamics = Callable[[Array, Array], Array]  # (x, u) -> xdot


def rk4_step(f: Dynamics, x: Array, u: Array, dt: float) -> Array:
    """One RK4 step with zero-order-hold input; renormalizes the quaternion x[6:10]."""
    k1 = f(x, u)
    k2 = f(x + 0.5*dt*k1, u)
    k3 = f(x + 0.5*dt*k2, u)
    k4 = f(x + dt *k3, u)
    x_next = x + dt/6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    #Batch change
    x_next[..., 6:10] /= np.linalg.norm(x_next[..., 6:10], axis=-1, keepdims=True)
    return x_next


def rollout(x0: Array, u_seq: Array, f: Dynamics, dt: float) -> Array:
    """Integrate from x0 under inputs u_seq (N, 4). Returns states (N + 1, 13)."""
    xs = np.empty((len(u_seq) + 1, x0.size))
    xs[0] = x0
    for k, u in enumerate(u_seq):
        xs[k+1] = rk4_step(f, xs[k], u, dt) 
    return xs

"""Physics-based quadrotor model (PBM).

State x = [p(3), v(3), q(4), w(3)] (13,), world frame z-up, body frame front-left-up.
  p [m] world position, v [m/s] world velocity, q unit quaternion [w, x, y, z]
  (body -> world), w [rad/s] body rates.
Input u = motor speeds Omega (4,) [rad/s].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

G: float = 9.81  # [m/s^2]


@dataclass(frozen=True)
class QuadParams:
    """Quadrotor parameters (SI). Sources documented in configs/params.yaml."""

    mass: float  # [kg]
    inertia: Array  # (3, 3) [kg m^2]
    kf: float  # thrust coefficient [N / (rad/s)^2]
    km: float  # drag-torque coefficient [N m / (rad/s)^2]
    rotor_pos: Array  # (4, 3) rotor positions in body frame [m]
    rotor_dir: Array  # (4,) sign of rotor reaction torque about body +z, +1 / -1


def quat_to_rotmat(q: Array) -> Array:
    """Rotation matrix R (body -> world) from unit quaternion [w, x, y, z]."""
    raise NotImplementedError


def quat_mul(q1: Array, q2: Array) -> Array:
    """Hamilton product q1 ⊗ q2."""
    raise NotImplementedError


def motor_wrench(omega: Array, p: QuadParams) -> tuple[Array, Array]:
    """Body-frame thrust force [N] (3,) and torque [N m] (3,) from motor speeds [rad/s]."""
    raise NotImplementedError


def f_pbm(
    x: Array,
    u: Array,
    p: QuadParams,
    f_corr: Array | None = None,
    t_corr: Array | None = None,
) -> Array:
    """State derivative xdot (13,) of the PBM, with optional body-frame corrections.

    f_corr [N] and t_corr [N m] are body-frame; f_corr is rotated to world.
    f_corr = t_corr = 0 must reproduce the pure PBM exactly.
    """
    raise NotImplementedError

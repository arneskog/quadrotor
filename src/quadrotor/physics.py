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
from pathlib import Path
import yaml

Array = NDArray[np.float64]

def load_params(path: str | Path) -> QuadParams:
    cfg = yaml.safe_load(Path(path).read_text())
    return QuadParams(
        mass=cfg["mass"],
        inertia=np.diag(cfg["inertia_diag"]),
        kf=cfg["kf"], km=cfg["km"],
        rotor_pos=np.asarray(cfg["rotor_pos"], dtype=float),
        rotor_dir=np.asarray(cfg["rotor_dir"], dtype=float),
        g=cfg["g"],
    )

@dataclass(frozen=True)
class QuadParams:
    """Quadrotor parameters (SI). Sources documented in configs/params.yaml."""

    mass: float  # [kg]
    inertia: Array  # (3, 3) [kg m^2]
    kf: float  # thrust coefficient [N / (rad/s)^2]
    km: float  # drag-torque coefficient [N m / (rad/s)^2]
    rotor_pos: Array  # (4, 3) rotor positions in body frame [m]
    rotor_dir: Array  # (4,) sign of rotor reaction torque about body +z, +1 / -1
    g: float = 9.81


def quat_to_rotmat(q: Array) -> Array:
    """Rotation matrix R (body -> world) from unit quaternion [w, x, y, z]."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def quat_mul(q1: Array, q2: Array) -> Array:
    """Hamilton product q1 ⊗ q2."""
    s1, v1 = q1[0], q1[1:]
    s2, v2 = q2[0], q2[1:]
    return np.concatenate(([s1*s2 - v1 @ v2], s1*v2 + s2*v1 + np.cross(v1, v2)))


def motor_wrench(omega: Array, p: QuadParams) -> tuple[Array, Array]:
    """Body-frame thrust force [N] (3,) and torque [N m] (3,) from motor speeds [rad/s]."""
    thrust = p.kf*omega**2
    zeros = np.zeros_like(thrust[..., 0])
    force = np.stack([zeros, zeros, thrust.sum(axis=-1)], axis=-1)
    torque = np.stack([
        thrust @ p.rotor_pos[:, 1],                           #  Σ y_i T_i
        -(thrust @ p.rotor_pos[:, 0]),                        # −Σ x_i T_i
        p.km * (omega**2 @ p.rotor_dir),                      #  km Σ dir_i Ω_i²
    ], axis=-1)
    return force, torque

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

    Costa corrects the wrence, not state derivatives -> Correction goes in before Newton-Euler
    """
    v = x[3:6]
    q = x[6:10]
    w = x[10:13]

    force_b, torgue_b = motor_wrench(u, q)
    if f_corr is not None:
        force_b = force_b + f_corr
    if t_corr is not None:
        torgue_b = torgue_b + t_corr

    pos_dot = v
    v_dot = quat_to_rotmat(q) @ force_b / p.mass - np.array([0.0, 0.0, p.g])
    q_dot = 0.5*quat_mul(q, np.concatenate(([0.0], w)))
    w_dot = np.linalg.solve(p.inertia, torgue_b - np.cross(w, p.inertia @ w))

    return np.concatenate((pos_dot, v_dot, q_dot, w_dot))


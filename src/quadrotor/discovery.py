"""Equation discovery on the learned CoSTA correction (L04 "closing the loop", SINDy module 21).

    data -> black box f_theta -> correction samples -> sparse regression -> equation

The correction is a STATIC map of the network inputs, c = f_theta(v_b, w, Omega), so this is
SINDy-style sparse regression c ≈ Theta(v_b, w, Omega) xi (no time derivatives are taken):
  * library(x, u): physically motivated candidate terms (rotor drag, inflow, body drag, mixer, damping),
    built from the network's own inputs only (no position, no time, no gravity direction);
  * stlsq: sequentially thresholded least squares on RMS-scaled columns; the threshold selects terms,
    the final least-squares refit on the kept terms removes the spurious weight (L05: select, then refit);
  * SparseCorrection: PBM + Theta xi as a WrenchModel, so it rolls out through the same RK4 / rigid body.
Units: v_b [m/s], w [rad/s], Omega in krad/s inside the library (o = Omega / 1000) so coefficients read
in N or N m per term unit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from quadrotor.physics import QuadParams, body_velocity, motor_wrench

Array = NDArray[np.float64]
COMPONENTS = ("F_x", "F_y", "F_z", "T_x", "T_y", "T_z")

#: candidate terms, in library column order (o = Omega/1000 [krad/s], S1 = sum o, S2 = sum o^2)
TERMS: tuple[tuple[str, str], ...] = (
    ("1", "offset"),
    ("vx", "linear drag"), ("vy", "linear drag"), ("vz", "linear drag"),
    ("S1*vx", "rotor drag (~ sum Omega * v)"), ("S1*vy", "rotor drag"), ("S1*vz", "rotor drag / axial inflow"),
    ("S2", "thrust scale (kf error)"),
    ("vx^2+vy^2", "translational lift"), ("vz|vz|", "axial inflow, quadratic"),
    ("|v|vx", "body (parasitic) drag"), ("|v|vy", "body drag"), ("|v|vz", "body drag"),
    ("Mx", "roll mixer sum y_i o_i^2"), ("My", "pitch mixer -sum x_i o_i^2"), ("Mz", "yaw mixer sum dir_i o_i^2"),
    ("wx", "rate damping"), ("wy", "rate damping"), ("wz", "rate damping"),
    ("S1*wx", "rotor rate damping"), ("S1*wy", "rotor rate damping"), ("S1*wz", "rotor rate damping"),
    ("wy*wz", "inertia mismatch"), ("wx*wz", "inertia mismatch"), ("wx*wy", "inertia mismatch"),
)
TERM_NAMES: tuple[str, ...] = tuple(t for t, _ in TERMS)


def library(x: Array, u: Array, p: QuadParams) -> Array:
    """Candidate-term matrix Theta (..., K) from state(s) x (..., 13) and motor speeds u (..., 4) [rad/s]."""
    v = body_velocity(x)
    w = x[..., 10:13]
    o = u / 1000.0
    o2 = o**2
    s1, s2 = o.sum(-1), o2.sum(-1)
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    wx, wy, wz = w[..., 0], w[..., 1], w[..., 2]
    speed = np.linalg.norm(v, axis=-1)
    cols = [
        np.ones_like(vx),
        vx, vy, vz,
        s1 * vx, s1 * vy, s1 * vz,
        s2,
        vx**2 + vy**2, vz * np.abs(vz),
        speed * vx, speed * vy, speed * vz,
        o2 @ p.rotor_pos[:, 1], -(o2 @ p.rotor_pos[:, 0]), o2 @ p.rotor_dir,
        wx, wy, wz,
        s1 * wx, s1 * wy, s1 * wz,
        wy * wz, wx * wz, wx * wy,
    ]
    return np.stack(cols, axis=-1)


def stlsq(theta: Array, y: Array, threshold: float, max_iter: int = 25) -> Array:
    """Sequentially thresholded least squares for ONE output y (N,) on library theta (N, K).

    Columns and y are scaled by their RMS, so the threshold is a dimensionless coefficient size
    ("contribution per unit RMS of the term, in units of the RMS of y"). Repeats: least squares on
    the active terms, drop |xi_scaled| < threshold, until the active set stops changing. The returned
    coefficients (K,) are in physical units and are the plain least-squares refit on the kept terms.
    """
    col_rms = np.sqrt(np.mean(theta**2, axis=0))
    col_rms = np.where(col_rms > 0, col_rms, 1.0)
    y_rms = float(np.sqrt(np.mean(y**2))) or 1.0
    A, b = theta / col_rms, y / y_rms
    active = np.ones(theta.shape[1], bool)
    xi = np.zeros(theta.shape[1])
    for _ in range(max_iter):
        xi = np.zeros(theta.shape[1])
        if active.any():
            xi[active] = np.linalg.lstsq(A[:, active], b, rcond=None)[0]
        keep = np.abs(xi) >= threshold
        if np.array_equal(keep, active):
            break
        active = keep
    xi[~active] = 0.0
    return xi * y_rms / col_rms


def fit_all(theta: Array, Y: Array, thresholds: Array) -> Array:
    """STLSQ per output column of Y (N, C) with its own threshold; returns Xi (K, C)."""
    return np.stack([stlsq(theta, Y[:, c], float(thresholds[c])) for c in range(Y.shape[1])], axis=1)


def rmse(a: Array, b: Array) -> Array:
    """RMSE per column."""
    return np.sqrt(np.mean((a - b) ** 2, axis=0))


def equation(xi: Array, names: tuple[str, ...] = TERM_NAMES, fmt: str = "{:+.3g}") -> str:
    """Readable sparse expression, e.g. '-0.31 S1*vx +0.02 vx^2+vy^2'."""
    parts = [f"{fmt.format(c)} {n}" for c, n in zip(xi, names) if c != 0.0]
    return " ".join(parts) if parts else "0"


@dataclass(frozen=True)
class SparseCorrection:
    """PBM plus a discovered correction Theta(x, u) Xi, Xi (K, 6) [force N, torque N m]."""

    p: QuadParams
    xi: Array
    name: str = "pbm_sindy"

    def correction(self, x: Array, u: Array) -> Array:
        return library(x, u, self.p) @ self.xi

    def wrench(self, x: Array, u: Array) -> tuple[Array, Array]:
        force_b, torque_b = motor_wrench(u, self.p)
        c = self.correction(x, u)
        return force_b + c[..., :3], torque_b + c[..., 3:]

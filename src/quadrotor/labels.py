"""Measured wrench, training targets and network features (body frame).

F_meas = m * acc                      (acc = measured body-frame specific force)
T_meas = J * alpha + w x (J w)        (Euler's equation solved for the applied torque)

CoSTA / linear drag learn the residual  F_res = F_meas - F_motor,  T_res = T_meas - T_motor;
the DDM learns the full measured wrench. The residual is always computed directly,
never via an operator applied to the state error.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, filtfilt

from .physics import QuadParams, motor_wrench, body_velocity, quat_to_rotmat
Array = NDArray[np.float64]

RESIDUAL_MODELS: tuple[str, ...] = ("costa", "linear_drag")
FORCE_RESIDUAL_MODELS: tuple[str, ...] = ("costa_force",)  # residual force, full measured torque
FULL_WRENCH_MODELS: tuple[str, ...] = ("ddm",)
KINEMATIC_MODELS: tuple[str, ...] = ("ddm_kin",)



def measured_wrench(acc: Array, alpha: Array, w: Array, p: QuadParams) -> tuple[Array, Array]:
    """Body-frame wrench that actually acted on the body: force [N] (N, 3), torque [N m] (N, 3).

    acc [m/s^2] body specific force, alpha [rad/s^2] body angular acceleration, w [rad/s] body rates.
    """
    force = acc * p.mass
    torque = alpha @ p.inertia.T + np.cross(w, w @ p.inertia.T)
    return force, torque


def residual_labels(
    x: Array, u: Array, a_meas: Array, alpha_meas: Array, p: QuadParams
) -> tuple[Array, Array]:
    """Body-frame residual force [N] (N, 3) and torque [N m] (N, 3): measured minus motor wrench."""
    force_meas, torque_meas = measured_wrench(a_meas, alpha_meas, x[..., 10:13], p)
    force_motor, torque_motor = motor_wrench(u, p)
    return force_meas-force_motor , torque_meas - torque_motor 


def targets(
    model_name: str,
    x: Array,
    u: Array,
    a_meas: Array,
    alpha_meas: Array,
    p: QuadParams,
) -> Array:
    """Training target (N, 6) = [force, torque] for a model: residual or full measured wrench.

    costa / linear_drag: residual force and torque. costa_force: residual force, full measured
    torque. ddm: full measured force and torque. Raises ValueError otherwise (e.g. 'pbm').
    """
    if model_name in RESIDUAL_MODELS:
        force, torque = residual_labels(x, u , a_meas, alpha_meas, p)
    elif model_name in FORCE_RESIDUAL_MODELS:
        force, _ = residual_labels(x, u , a_meas, alpha_meas, p)
        _ , torque = measured_wrench(a_meas, alpha_meas, x[... , 10:13], p)
    elif model_name in FULL_WRENCH_MODELS:
        force, torque = measured_wrench(a_meas, alpha_meas, x[... , 10:13], p)
    elif model_name in KINEMATIC_MODELS:
        return np.concatenate([a_meas - p.g * gravity_body(x), alpha_meas], axis=-1)
    else:
        raise ValueError(f"no training target model for {model_name!r}")
    return np.concatenate([force, torque], axis = -1)



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
    """Network input (..., 10): body velocity [m/s], body rates [rad/s], motor speeds [rad/s].

    No position, no time. In a rollout x is the MODEL's own state, never a measurement.
    """
    return np.concatenate([body_velocity(x), x[..., 10:13], u], axis=-1)


def gravity_body(x: Array) -> Array:
    """'Up' direction e_z in the body frame, R(q)^T e_z (..., 3): tilt only, unique for q and -q."""
    return quat_to_rotmat(x[..., 6:10])[..., 2, :]

def features_kin(x: Array, u: Array) -> Array:
    """Kinematics-only DDM input (..., 13): features plus the gravity direction."""
    return np.concatenate([features(x, u), gravity_body(x)], axis=-1)

def network_features(kind: str, x: Array, u: Array) -> Array:
    """The network input of model `kind` (run.py builds training inputs with this)."""
    return features_kin(x, u) if kind in KINEMATIC_MODELS else features(x, u)
from pathlib import Path

import numpy as np
import pytest

from quadrotor.labels import features, lowpass, measured_wrench, residual_labels, targets
from quadrotor.physics import f_pbm, load_params, motor_wrench, quat_to_rotmat

DT = 0.0025
P = load_params(Path(__file__).parents[1] / "configs" / "params.yaml")


def test_lowpass_none_is_identity() -> None:
    x = np.random.default_rng(0).normal(size=(100, 3))
    assert np.array_equal(lowpass(x, DT, None), x)


def test_lowpass_keeps_slow_and_removes_fast_component() -> None:
    t = DT * np.arange(4000)
    slow, fast = np.sin(2 * np.pi * 1.0 * t), np.sin(2 * np.pi * 100.0 * t)
    y = lowpass((slow + fast)[:, None], DT, 10.0)[:, 0]
    inner = slice(400, -400)  # ignore filtfilt edge transients
    assert np.max(np.abs(y[inner] - slow[inner])) < 0.02


def pbm_generated_rows(n: int = 50, seed: int = 0) -> tuple[np.ndarray, ...]:
    """States, motor speeds and the acc / alpha a PBM-only world would measure."""
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    x = np.column_stack([rng.normal(size=(n, 6)), q / np.linalg.norm(q, axis=1, keepdims=True),
                         3 * rng.normal(size=(n, 3))])
    u = rng.uniform(700, 1500, (n, 4))
    xdot = np.array([f_pbm(x[k], u[k], P) for k in range(n)])
    R = quat_to_rotmat(x[:, 6:10])
    acc = np.einsum("nji,nj->ni", R, xdot[:, 3:6] + np.array([0.0, 0.0, P.g]))  # body specific force
    return x, u, acc, xdot[:, 10:13]


def test_residual_is_zero_when_data_comes_from_the_pbm() -> None:
    # Checks frames and signs end to end: acc convention, w x Jw sign, motor wrench.
    x, u, acc, alpha = pbm_generated_rows()
    f_res, t_res = residual_labels(x, u, acc, alpha, P)
    assert np.allclose(f_res, 0.0, atol=1e-10)
    assert np.allclose(t_res, 0.0, atol=1e-12)


def test_measured_wrench_equals_motor_wrench_for_pbm_data() -> None:
    x, u, acc, alpha = pbm_generated_rows()
    force, torque = measured_wrench(acc, alpha, x[:, 10:13], P)
    force_m, torque_m = motor_wrench(u, P)
    assert np.allclose(force, force_m) and np.allclose(torque, torque_m)


def test_targets_residual_vs_full_wrench() -> None:
    x, u, acc, alpha = pbm_generated_rows()
    acc = acc + np.array([0.5, -0.2, 0.1])  # a known extra force per unit mass
    res, full = targets("costa", x, u, acc, alpha, P), targets("ddm", x, u, acc, alpha, P)
    assert np.allclose(res[:, :3], P.mass * np.array([0.5, -0.2, 0.1]))
    assert np.allclose(full - res, np.concatenate(motor_wrench(u, P), axis=1))
    with pytest.raises(ValueError):
        targets("pbm", x, u, acc, alpha, P)


def test_features_layout_and_no_position() -> None:
    x, u, *_ = pbm_generated_rows(n=5)
    feats = features(x, u)
    assert feats.shape == (5, 10)
    assert np.array_equal(feats[:, 3:6], x[:, 10:13]) and np.array_equal(feats[:, 6:], u)
    shifted = x.copy()
    shifted[:, 0:3] += 100.0  # position must not change the features
    assert np.array_equal(features(shifted, u), feats)


def test_costa_force_target_is_residual_force_and_full_torque() -> None:
    x, u, acc, alpha = pbm_generated_rows()
    y = targets("costa_force", x, u, acc, alpha, P)
    full = targets("ddm", x, u, acc, alpha, P)
    res = targets("costa", x, u, acc, alpha, P)
    assert np.allclose(y[:, :3], res[:, :3]) and np.allclose(y[:, 3:], full[:, 3:])

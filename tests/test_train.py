from pathlib import Path

import numpy as np
import yaml

from quadrotor.models import CoSTA, DDM
from quadrotor.physics import load_params
from quadrotor.train import TrainConfig, fit_kf, fit_linear_drag, fit_network, fit_torque_gain

ROOT = Path(__file__).parents[1]
P = load_params(ROOT / "configs" / "params.yaml")
FALLBACK = np.array([1.0] * 6 + [100.0] * 4)


def test_fit_kf_recovers_thrust_coefficient() -> None:
    rng = np.random.default_rng(0)
    u = rng.uniform(800, 1400, (500, 4))
    acc_z = 1.4e-6 * np.sum(u**2, axis=1) / P.mass
    assert np.isclose(fit_kf(u, acc_z, P), 1.4e-6, rtol=1e-12)


def test_fit_linear_drag_recovers_known_matrix() -> None:
    rng = np.random.default_rng(0)
    D = np.array([[-0.3, 0.02, 0.0], [0.0, -0.35, 0.0], [0.01, 0.0, -0.1]])
    v = rng.normal(0, 3, (2000, 3))
    assert np.allclose(fit_linear_drag(v, v @ D.T), D)


def test_train_config_reads_project_config() -> None:
    cfg = TrainConfig.from_dict(yaml.safe_load((ROOT / "configs" / "train.yaml").read_text()))
    assert cfg.hidden == [64, 64] and cfg.activation == "silu" and cfg.patience == 20


def test_fit_network_selects_best_validation_checkpoint() -> None:
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.normal(size=(512, 6)), rng.uniform(800, 1400, (512, 4))])
    Y = np.column_stack([-0.3 * X[:, :3], np.zeros((512, 3))])  # residual = linear drag on features
    scores = iter([5.0, 3.0, 1.0, 2.0, 2.5, 4.0, 4.0, 4.0])  # fake validation rollouts
    cfg = TrainConfig(hidden=[16], batch_size=128, max_epochs=20, patience=3)
    model, info = fit_network("costa", P, X, Y, FALLBACK, lambda m: next(scores), cfg, seed=0)
    assert isinstance(model, CoSTA)
    assert info["best_epoch"] == 2 and info["best_val_score"] == 1.0
    assert info["stopped_epoch"] == 5  # three validations without improvement after the best


def test_fit_network_is_reproducible_and_learns() -> None:
    rng = np.random.default_rng(1)
    X = np.column_stack([rng.normal(size=(1024, 6)), rng.uniform(800, 1400, (1024, 4))])
    Y = np.column_stack([-0.3 * X[:, :3], 0.01 * X[:, 3:6]])
    cfg = TrainConfig(hidden=[32, 32], batch_size=128, max_epochs=30, patience=100)
    runs = [fit_network("ddm", P, X, Y, FALLBACK, lambda m: 1.0, cfg, seed=3) for _ in range(2)]
    assert isinstance(runs[0][0], DDM)
    assert runs[0][1]["history"]["train_mse"] == runs[1][1]["history"]["train_mse"]
    hist = runs[0][1]["history"]["train_mse"]
    assert hist[-1] < 0.05 * hist[0]


def test_fit_torque_gain_recovers_per_axis_scaling() -> None:
    import dataclasses

    from quadrotor.physics import motor_wrench

    rng = np.random.default_rng(2)
    n = 400
    u = rng.uniform(800, 1400, (n, 4))
    w = rng.normal(0, 1, (n, 3))
    gain = np.array([0.3, 0.1, 0.8])
    tau_true = motor_wrench(u, dataclasses.replace(P, torque_gain=np.ones(3)))[1] * gain
    # alpha such that J alpha + w x Jw = tau_true; acc irrelevant for the torque
    alpha = np.linalg.solve(P.inertia, (tau_true - np.cross(w, w @ P.inertia.T)).T).T
    assert np.allclose(fit_torque_gain(u, np.zeros((n, 3)), alpha, w, P), gain)

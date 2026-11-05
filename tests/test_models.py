from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from quadrotor.integrators import rollout
from quadrotor.labels import features
from quadrotor.models import (
    DDM,
    MLP,
    PBM,
    CoSTA,
    FrozenMLP,
    LinearDrag,
    Normalizer,
    load_model,
    make_dynamics,
    save_model,
)
from quadrotor.physics import body_velocity, f_pbm, load_params, motor_wrench

P = load_params(Path(__file__).parents[1] / "configs" / "params.yaml")
RNG = np.random.default_rng(0)


def random_state(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=4)
    return np.concatenate([rng.normal(size=6), q / np.linalg.norm(q), rng.normal(size=3)])


def zero_costa() -> CoSTA:
    """CoSTA whose correction is exactly zero (zero last layer, zero output mean)."""
    net = MLP([10, 8, 6])
    with torch.no_grad():
        net.net[-1].weight.zero_()
    frozen = FrozenMLP.from_torch(net)
    norm = Normalizer(np.zeros(10), np.ones(10), np.zeros(6), np.ones(6))
    return CoSTA(p=P, net=frozen, norm=norm)


def test_pbm_dynamics_is_f_pbm_bitwise() -> None:
    f = make_dynamics(PBM(P), P)
    for _ in range(50):
        x, u = random_state(RNG), RNG.uniform(700, 1500, 4)
        assert np.array_equal(f(x, u), f_pbm(x, u, P))


def test_zero_correction_costa_reproduces_pbm_rollout_bitwise() -> None:
    x0 = random_state(RNG)
    u_seq = RNG.uniform(1000, 1200, (200, 4))
    xs_pbm = rollout(x0, u_seq, make_dynamics(PBM(P), P), 0.0025)
    xs_costa = rollout(x0, u_seq, make_dynamics(zero_costa(), P), 0.0025)
    assert np.array_equal(xs_costa, xs_pbm)


def test_linear_drag_with_zero_matrix_is_pbm() -> None:
    f_ld, f_pbm_ = make_dynamics(LinearDrag(P, np.zeros((3, 3))), P), make_dynamics(PBM(P), P)
    x, u = random_state(RNG), RNG.uniform(700, 1500, 4)
    assert np.array_equal(f_ld(x, u), f_pbm_(x, u))


def test_linear_drag_adds_D_times_body_velocity() -> None:
    D = np.diag([-0.3, -0.4, -0.1])
    x, u = random_state(RNG), RNG.uniform(700, 1500, 4)
    force, _ = LinearDrag(P, D).wrench(x, u)
    assert np.allclose(force - motor_wrench(u, P)[0], D @ body_velocity(x))


def test_ddm_returns_its_network_output_as_wrench() -> None:
    net = MLP([10, 8, 6]).double()
    norm = Normalizer(np.zeros(10), np.full(10, 100.0), np.arange(6.0), np.full(6, 2.0))
    ddm = DDM(net=FrozenMLP.from_torch(net), norm=norm)
    x, u = random_state(RNG), RNG.uniform(700, 1500, 4)
    with torch.no_grad():
        z = net(torch.as_tensor(features(x, u) / 100.0)).numpy()
    force, torque = ddm.wrench(x, u)
    assert np.allclose(np.concatenate([force, torque]), z * 2.0 + np.arange(6.0))


def test_frozen_mlp_matches_torch() -> None:
    for act in ("silu", "tanh"):
        net = MLP([10, 16, 16, 6], act).double()
        z = RNG.normal(size=(32, 10))
        with torch.no_grad():
            ref = net(torch.as_tensor(z)).numpy()
        assert np.allclose(FrozenMLP.from_torch(net)(z), ref, atol=1e-12)


def test_normalizer_uses_declared_scale_for_constant_channel() -> None:
    X = np.column_stack([RNG.normal(size=100), np.full(100, 5.0)])
    norm = Normalizer.fit(X, RNG.normal(size=(100, 1)), x_fallback=np.array([1.0, 100.0]))
    assert norm.x_scale[1] == 100.0
    assert np.isclose(norm.x_scale[0], X[:, 0].std())
    assert np.all(np.abs(norm.scale_in(X)) < 10)


def test_save_load_roundtrip_gives_identical_wrench(tmp_path: Path) -> None:
    net = MLP([10, 8, 6]).double()
    norm = Normalizer(np.zeros(10), np.full(10, 100.0), np.arange(6.0), np.full(6, 2.0))
    models = [PBM(P), LinearDrag(P, np.diag([-0.3, -0.3, -0.1])),
              CoSTA(p=P, net=FrozenMLP.from_torch(net), norm=norm), DDM(net=FrozenMLP.from_torch(net), norm=norm)]
    x, u = np.array([random_state(RNG) for _ in range(5)]), RNG.uniform(700, 1500, (5, 4))
    for model in models:
        save_model(model, tmp_path / f"{model.name}.npz")
        loaded = load_model(tmp_path / f"{model.name}.npz", P)
        assert loaded.name == model.name
        for a, b in zip(loaded.wrench(x, u), model.wrench(x, u)):
            assert np.array_equal(a, b)


def test_batched_wrench_matches_row_by_row() -> None:
    net = MLP([10, 8, 6]).double()
    norm = Normalizer(np.zeros(10), np.full(10, 100.0), np.zeros(6), np.ones(6))
    x, u = np.array([random_state(RNG) for _ in range(4)]), RNG.uniform(700, 1500, (4, 4))
    for model in (LinearDrag(P, np.diag([-0.3, -0.3, -0.1])), CoSTA(p=P, net=FrozenMLP.from_torch(net), norm=norm)):
        force, torque = model.wrench(x, u)
        for k in range(4):
            f_k, t_k = model.wrench(x[k], u[k])
            assert np.allclose(force[k], f_k) and np.allclose(torque[k], t_k)


def test_body_velocity_matches_scipy() -> None:
    x = np.array([random_state(RNG) for _ in range(20)])
    q_xyzw = x[:, [7, 8, 9, 6]]
    ref = Rotation.from_quat(q_xyzw).inv().apply(x[:, 3:6])
    assert np.allclose(body_velocity(x), ref)
    assert np.allclose(body_velocity(x[0]), ref[0])


def test_torque_gain_scales_motor_torque_only() -> None:
    import dataclasses

    u = RNG.uniform(700, 1500, (6, 4))
    gain = np.array([0.5, 0.2, 2.0])
    f1, t1 = motor_wrench(u, P)
    f2, t2 = motor_wrench(u, dataclasses.replace(P, torque_gain=gain))
    assert np.array_equal(f1, f2) and np.allclose(t2, t1 * gain)


def test_force_only_costa_ignores_motor_torque() -> None:
    net = MLP([10, 8, 6]).double()
    norm = Normalizer(np.zeros(10), np.full(10, 100.0), np.zeros(6), np.ones(6))
    frozen = FrozenMLP.from_torch(net)
    full = CoSTA(p=P, net=frozen, norm=norm)
    force_only = CoSTA(p=P, net=frozen, norm=norm, name="costa_force", torque_prior=False)
    x, u = np.array([random_state(RNG) for _ in range(5)]), RNG.uniform(700, 1500, (5, 4))
    (f_a, t_a), (f_b, t_b) = full.wrench(x, u), force_only.wrench(x, u)
    assert np.array_equal(f_a, f_b)
    assert np.allclose(t_a - t_b, motor_wrench(u, P)[1])

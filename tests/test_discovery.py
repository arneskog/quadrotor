from pathlib import Path

import numpy as np

from quadrotor.discovery import TERM_NAMES, SparseCorrection, fit_all, library, stlsq
from quadrotor.integrators import rollout
from quadrotor.models import PBM, make_dynamics
from quadrotor.physics import body_velocity, load_params

P = load_params(Path(__file__).parents[1] / "configs" / "params.yaml")


def random_states(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    q = rng.normal(size=(n, 4))
    x = np.hstack([rng.normal(size=(n, 3)), 5 * rng.normal(size=(n, 3)), q / np.linalg.norm(q, axis=1, keepdims=True),
                   2 * rng.normal(size=(n, 3))])
    u = rng.uniform(600.0, 2200.0, size=(n, 4))
    return x, u


def test_library_shape_and_terms():
    rng = np.random.default_rng(0)
    x, u = random_states(50, rng)
    theta = library(x, u, P)
    assert theta.shape == (50, len(TERM_NAMES))
    v = body_velocity(x)
    s1 = (u / 1000).sum(1)
    np.testing.assert_allclose(theta[:, TERM_NAMES.index("1")], 1.0)
    np.testing.assert_allclose(theta[:, TERM_NAMES.index("S1*vx")], s1 * v[:, 0])
    # batched == row by row
    np.testing.assert_allclose(library(x[3], u[3], P), theta[3])


def test_stlsq_recovers_sparse_law_with_correlated_distractor():
    """Known sparse law + a distractor correlated with a true term: the threshold removes it,
    the refit returns the true coefficients (not shrunk)."""
    rng = np.random.default_rng(1)
    x, u = random_states(20000, rng)
    theta = library(x, u, P)
    true = np.zeros(len(TERM_NAMES))
    true[TERM_NAMES.index("S1*vx")] = -0.05
    true[TERM_NAMES.index("vz|vz|")] = 0.02
    true[TERM_NAMES.index("1")] = 0.3
    y = theta @ true + 0.01 * rng.normal(size=len(x))
    xi = stlsq(theta, y, threshold=0.05)
    assert set(np.flatnonzero(xi)) == set(np.flatnonzero(true))
    np.testing.assert_allclose(xi, true, atol=2e-3)


def test_stlsq_zero_threshold_is_least_squares_and_huge_threshold_is_empty():
    rng = np.random.default_rng(2)
    theta = rng.normal(size=(500, 6))
    y = theta @ rng.normal(size=6) + 0.1 * rng.normal(size=500)
    np.testing.assert_allclose(stlsq(theta, y, 0.0), np.linalg.lstsq(theta, y, rcond=None)[0], rtol=1e-10)
    assert not stlsq(theta, y, 1e6).any()


def test_fit_all_per_output_threshold():
    rng = np.random.default_rng(3)
    theta = rng.normal(size=(800, 5))
    Y = np.stack([theta[:, 0] * 2.0, theta[:, 1] * 0.1 + theta[:, 2] * 3.0], axis=1)
    Xi = fit_all(theta, Y, np.array([0.5, 0.0]))
    assert Xi.shape == (5, 2)
    np.testing.assert_allclose(Xi[:, 0], [2, 0, 0, 0, 0], atol=1e-10)
    np.testing.assert_allclose(Xi[:, 1], [0, 0.1, 3, 0, 0], atol=1e-10)


def test_zero_sparse_correction_is_pbm_bit_for_bit():
    rng = np.random.default_rng(4)
    x0, u = random_states(1, rng)
    u_seq = np.repeat(u, 30, axis=0)
    zero = SparseCorrection(P, np.zeros((len(TERM_NAMES), 6)))
    a = rollout(x0[0], u_seq, make_dynamics(PBM(P), P), 0.0025)
    b = rollout(x0[0], u_seq, make_dynamics(zero, P), 0.0025)
    assert np.array_equal(a, b)

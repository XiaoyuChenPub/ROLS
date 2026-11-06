"""Tests of the baseline regressions, mainly their Euclidean closed forms."""

import numpy as np
import pytest

from ROLS import Hyperbolic, SO3, SPD, Sphere, Torus, baselines, frechet_rmse
from ROLS.manifolds.base import Manifold


class Euclidean(Manifold):
    """R^d with the usual metric, where every baseline has a closed form."""

    name = "R^d"

    def __init__(self, d):
        self.d = d

    def logmap(self, base, points):
        return np.atleast_2d(points) - np.asarray(base)

    def expmap(self, base, tangents):
        return np.asarray(base) + np.atleast_2d(tangents)

    def dist(self, p, q):
        return np.linalg.norm(np.atleast_2d(p) - np.atleast_2d(q), axis=1)

    def transport(self, src, dst, tangents):
        return np.atleast_2d(tangents)

    def frame(self, base):
        return np.eye(self.d)

    def frechet_mean(self, points, weights=None, tol=1e-12, max_iter=500):
        return np.average(np.asarray(points), axis=0, weights=weights)


def _linear_fit(y, z, z_new):
    """Returns ordinary least squares predictions with an intercept."""
    design = np.column_stack([np.ones(len(z)), z])
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    return np.column_stack([np.ones(len(z_new)), z_new]) @ coef


def _sphere_data(rng, n=120):
    """Returns responses near the north pole with one covariate in effect."""
    m = Sphere(3)
    pole = np.array([0.0, 0.0, 1.0])
    z = rng.standard_normal((n, 2))
    tangent = np.column_stack([0.3 * z[:, 0], 0.05 * rng.standard_normal(n),
                               np.zeros(n)])
    return m, m.expmap(pole, tangent), z


def test_signed_mean_is_exact_in_euclidean_space():
    rng = np.random.default_rng(0)
    m = Euclidean(3)
    Y = rng.standard_normal((30, 3))
    w = rng.standard_normal(30)
    w = w - w.mean() + 1.0 / 30
    got = baselines.weighted_frechet_mean(m, Y, w, Y[0])
    assert got == pytest.approx(w @ Y, abs=1e-10)


@pytest.mark.parametrize("m, points", [
    (Sphere(3), lambda g: Sphere(3).expmap(
        np.array([0.0, 0.0, 1.0]),
        np.column_stack([0.3 * g.standard_normal((40, 2)), np.zeros(40)]))),
    (Torus(2), lambda g: 0.4 * g.standard_normal((40, 2))),
    (SO3(), lambda g: SO3().expmap(
        np.eye(3), 0.3 * g.standard_normal((40, 3)))),
    (SPD(3), lambda g: SPD(3).expmap(np.eye(3), SPD(3).from_coords(
        np.eye(3), 0.2 * g.standard_normal((40, 6))))),
    (Hyperbolic(3), lambda g: Hyperbolic(3).expmap(
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.column_stack([np.zeros(40), 0.3 * g.standard_normal((40, 3))]))),
])
def test_signed_mean_satisfies_first_order_condition(m, points):
    rng = np.random.default_rng(1)
    Y = points(rng)
    w = 1.0 / 40 + 0.01 * rng.standard_normal(40)
    w = w / w.sum()
    mu = baselines.weighted_frechet_mean(m, Y, w, m.frechet_mean(Y))
    assert np.linalg.norm(w @ m.coords(mu, m.logmap(mu, Y))) < 1e-6


def test_weighted_means_find_the_global_minimum_on_a_circle():
    rng = np.random.default_rng(13)
    m = Torus(1)
    Y = np.concatenate([2.8 + 0.1 * rng.standard_normal(30),
                        -0.5 + 0.1 * rng.standard_normal(30)])[:, None]
    Y_bar, W = baselines._tangent_response(m, Y)
    S = np.concatenate([np.full(30, 0.9 / 30), np.full(30, 0.1 / 30)])[None]
    got = baselines._weighted_means(m, Y, Y_bar, W, S)[0]
    grid = np.linspace(-np.pi, np.pi, 4001)[:, None]
    objective = [S[0] @ (m.logmap(y, Y)[:, 0] ** 2) for y in grid]
    assert S[0] @ (m.logmap(got, Y)[:, 0] ** 2) <= min(objective) + 1e-6


def test_frechet_regression_is_least_squares_in_euclidean_space():
    rng = np.random.default_rng(2)
    z, z_new = rng.standard_normal((80, 3)), rng.standard_normal((10, 3))
    Y = z @ rng.standard_normal((3, 2)) + rng.standard_normal((80, 2))
    got = baselines.frechet_regression(Euclidean(2), Y, z, z_new)
    assert got == pytest.approx(_linear_fit(Y, z, z_new), abs=1e-8)


def test_frechet_regression_recovers_signal_on_sphere():
    m, Y, z = _sphere_data(np.random.default_rng(3))
    pred = baselines.frechet_regression(m, Y, z, z)
    assert frechet_rmse(m, Y, pred) < 0.07
    assert frechet_rmse(m, Y, baselines.frechet_mean(m, Y, len(Y))) > 0.25


def test_local_linear_weights_sum_to_one_and_reproduce_lines():
    rng = np.random.default_rng(4)
    t, t0 = rng.standard_normal(100), np.linspace(-2.0, 2.0, 9)
    S = baselines._local_linear_weights(t, t0, 0.5)
    assert S.sum(1) == pytest.approx(np.ones(9))
    assert S @ (2.0 + 3.0 * t) == pytest.approx(2.0 + 3.0 * t0)


def test_local_linear_weights_far_from_the_data():
    t = np.array([0.0, 1.0, 2.0])
    S = baselines._local_linear_weights(t, np.array([500.0]), 0.01)
    assert np.isfinite(S).all() and S.sum() == pytest.approx(1.0)


def test_single_index_recovers_the_index():
    rng = np.random.default_rng(5)
    theta = np.array([0.6, 0.8, 0.0])
    z = rng.standard_normal((200, 3))
    z_new = rng.standard_normal((100, 3))

    def link(v):
        return np.column_stack([np.sin(v), v ** 2])

    Y = link(z @ theta) + 0.05 * rng.standard_normal((200, 2))
    pred = baselines.single_index_frechet(Euclidean(2), Y, z, z_new)
    assert frechet_rmse(Euclidean(2), link(z_new @ theta), pred) < 0.25


def test_single_index_is_reproducible():
    m, Y, z = _sphere_data(np.random.default_rng(6), n=60)
    a = baselines.single_index_frechet(m, Y, z, z[:5], n_directions=20, seed=3)
    b = baselines.single_index_frechet(m, Y, z, z[:5], n_directions=20, seed=3)
    assert a == pytest.approx(b)


def test_nonlinear_weights_reduce_to_frechet_regression_for_a_linear_kernel():
    """Theorem 5 of Bhattacharjee, Li and Xue (2025)."""
    rng = np.random.default_rng(7)
    z, z_new = rng.standard_normal((60, 3)), rng.standard_normal((5, 3))
    mu = z.mean(0)
    Zc, Zc_new = z - mu, z_new - mu
    n = len(z)
    A, _ = next(baselines._nonlinear_weight_maps(1.0 + Zc @ Zc.T, [1e-9]))
    got = 1.0 / n + (1.0 + Zc_new @ Zc.T) @ A.T
    want = (1.0 + Zc_new @ np.linalg.solve(Zc.T @ Zc / n, Zc.T)) / n
    assert got == pytest.approx(want, abs=1e-6)


def test_nonlinear_frechet_learns_a_nonlinear_map():
    rng = np.random.default_rng(8)
    m = Euclidean(1)
    z, z_new = rng.uniform(-2, 2, (150, 1)), rng.uniform(-1.5, 1.5, (50, 1))
    Y = np.sin(2 * z) + 0.05 * rng.standard_normal((150, 1))
    pred = baselines.nonlinear_frechet(m, Y, [], z, [], z_new)
    linear = baselines.frechet_regression(m, Y, z, z_new)
    truth = np.sin(2 * z_new)
    assert frechet_rmse(m, truth, pred) < 0.35 * frechet_rmse(m, truth, linear)


def test_nonlinear_frechet_uses_manifold_predictors():
    rng = np.random.default_rng(9)
    m = Sphere(3)
    pole = np.array([0.0, 0.0, 1.0])
    tangent = np.column_stack([0.4 * rng.standard_normal((180, 2)),
                               np.zeros(180)])
    X = m.expmap(pole, tangent)
    noise = np.column_stack([0.02 * rng.standard_normal((180, 2)),
                             np.zeros(180)])
    Y = m.expmap(pole, 0.5 * tangent + noise)
    pred = baselines.nonlinear_frechet(m, Y[:120], [X[:120]], None, [X[120:]],
                                       None)
    mean_only = baselines.frechet_mean(m, Y[:120], 60)
    baseline_error = frechet_rmse(m, Y[120:], mean_only)
    assert frechet_rmse(m, Y[120:], pred) < 0.5 * baseline_error


def test_nonlinear_frechet_on_noncompact_manifolds():
    rng = np.random.default_rng(14)
    for m in (SPD(3), Hyperbolic(3)):
        base = np.eye(3) if isinstance(m, SPD) else np.array([1.0, 0, 0, 0])
        coords = 0.3 * rng.standard_normal((80, m.d))
        tangent = m.from_coords(base, coords)
        X = m.expmap(base, tangent)
        Y = m.expmap(base, 0.5 * tangent)
        pred = baselines.nonlinear_frechet(m, Y[:60], [X[:60]], None, [X[60:]],
                                           None)
        mean_only = baselines.frechet_mean(m, Y[:60], 20)
        assert frechet_rmse(m, Y[60:], pred) < frechet_rmse(m, Y[60:],
                                                            mean_only)


def test_tangent_linear_ridge_fits_a_matrix_coefficient():
    rng = np.random.default_rng(10)
    m = Euclidean(2)
    X = rng.standard_normal((200, 2))
    A = np.array([[0.5, -0.4], [0.2, 0.7]])
    Y = X @ A.T + 0.01 * rng.standard_normal((200, 2))
    pred = baselines.tangent_linear_ridge(m, Y[:150], [X[:150]], None,
                                          [X[150:]], None)
    assert frechet_rmse(m, Y[150:], pred) < 0.02


def test_geodesic_regression_is_least_squares_in_euclidean_space():
    rng = np.random.default_rng(11)
    z, z_new = rng.standard_normal((70, 2)), rng.standard_normal((8, 2))
    Y = z @ rng.standard_normal((2, 2)) + rng.standard_normal((70, 2))
    got = baselines.geodesic_regression(Euclidean(2), Y, z, z_new)
    assert got == pytest.approx(_linear_fit(Y, z, z_new), abs=1e-6)


@pytest.mark.parametrize("method", [baselines.geodesic_regression,
                                    baselines.frechet_regression,
                                    baselines.single_index_frechet])
def test_no_covariates_predicts_the_mean(method):
    m, Y, _ = _sphere_data(np.random.default_rng(12), n=30)
    got = method(m, Y, np.zeros((30, 0)), np.zeros((4, 0)))
    assert got == pytest.approx(np.tile(m.frechet_mean(Y), (4, 1)))

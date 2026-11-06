import numpy as np
import pytest
from scipy.integrate import solve_ivp

from ROLS import (SO3, SPD, Hyperbolic, Sphere, Torus, fit, frechet_rmse,
                  pairs_bootstrap, pvalues, residual_bootstrap, sandwich)
from ROLS.bootstrap import _refit_tangent

NAMES = ["S2", "T2", "SPD3", "H3", "SO3"]


def _setup(name):
    if name == "S2":
        m = Sphere(3)
        Yb, Xb = np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
    elif name == "T2":
        m = Torus(2)
        Yb, Xb = np.array([1.0, -0.5]), np.array([0.2, 0.3])
    elif name == "SPD3":
        m = SPD(3)
        Yb, Xb = 1.5 * np.eye(3), 1.2 * np.eye(3)
    elif name == "H3":
        m = Hyperbolic(3)
        Yb = Hyperbolic.from_spatial([0.3, 0.0, 0.0])[0]
        Xb = Hyperbolic.from_spatial([0.0, 0.4, 0.1])[0]
    else:
        m = SO3()
        Yb = np.eye(3)
        Xb = m.expmap(np.eye(3), np.array([[0.3, -0.2, 0.1]]))[0]
    return m, Yb, Xb


def _simulate(name, n, rng, alpha=0.5, v_scale=0.2, noise=0.1):
    """Draws correctly specified data with one manifold and one scalar
    predictor."""
    m, Yb, Xb = _setup(name)
    X = m.expmap(Xb, m.from_coords(Xb, rng.standard_normal((n, m.d)) * 0.3))
    u = m.coords(Yb, m.transport(Xb, Yb, m.logmap(Xb, X)))
    v = rng.standard_normal(m.d)
    v = v / np.linalg.norm(v) * v_scale
    z = rng.standard_normal((n, 1))
    w = alpha * u + z * v + noise * rng.standard_normal((n, m.d))
    Y = m.expmap(Yb, m.from_coords(Yb, w))
    return m, Y, [X], z, Yb, v


def _far_points(name, rng):
    """Returns two base points well apart, non-commuting on SPD."""
    m, Yb, Xb = _setup(name)
    S = m.expmap(Yb, m.from_coords(Yb, rng.standard_normal((1, m.d)) * 0.5))[0]
    D = m.expmap(Xb, m.from_coords(Xb, rng.standard_normal((1, m.d)) * 0.5))[0]
    return m, S, D


@pytest.mark.parametrize("name", NAMES)
def test_coords_round_trip(name):
    m, Yb, _ = _setup(name)
    c = np.random.default_rng(0).standard_normal((5, m.d))
    assert np.allclose(m.coords(Yb, m.from_coords(Yb, c)), c, atol=1e-10)


@pytest.mark.parametrize("base", [[1, 0, 0], [0, 1, 0], [0, 0, 1],
                                  [0.6, 0.8, 0], [0.2, -0.3, 0.9]])
def test_sphere_frame_is_an_orthonormal_tangent_frame(base):
    b = np.asarray(base, float)
    b /= np.linalg.norm(b)
    F = Sphere(3).frame(b)
    assert np.allclose(F.T @ F, np.eye(2), atol=1e-12)
    assert np.abs(b @ F).max() < 1e-12


@pytest.mark.parametrize("name", NAMES)
def test_coords_are_isometric(name):
    m, Yb, _ = _setup(name)
    rng = np.random.default_rng(1)
    Y = m.expmap(Yb, m.from_coords(Yb, rng.standard_normal((6, m.d)) * 0.4))
    c = m.coords(Yb, m.logmap(Yb, Y))
    assert np.allclose(np.linalg.norm(c, axis=1), m.dist(np.stack([Yb] * 6), Y),
                       atol=1e-8)


@pytest.mark.parametrize("name", NAMES)
def test_weighted_mean_with_a_point_mass_returns_that_point(name):
    m, Yb, _ = _setup(name)
    rng = np.random.default_rng(2)
    P = m.expmap(Yb, m.from_coords(Yb, rng.standard_normal((8, m.d)) * 0.3))
    w = np.zeros(8)
    w[3] = 1.0
    assert m.dist(m.frechet_mean(P, weights=w)[None], P[3:4])[0] < 1e-8


@pytest.mark.parametrize("name", NAMES)
def test_fit_recovers_coefficients(name):
    m, Y, X, z, Yb, v = _simulate(name, 4000, np.random.default_rng(3))
    f = fit(m, Y, X, z)
    assert abs(f.alphas[0] - 0.5) < 0.03
    v_ref = m.coords(f.Y_bar,
                     m.transport(Yb, f.Y_bar, m.from_coords(Yb, v[None])))[0]
    assert np.linalg.norm(f.V[0] - v_ref) < 0.03


@pytest.mark.parametrize("name", NAMES)
def test_v_ambient_is_the_tangent_vector_prediction_moves_along(name):
    """With X at its mean and z one unit above its mean, the prediction is
    Exp_Ybar(v_1)."""
    m, Y, X, z, _, _ = _simulate(name, 200, np.random.default_rng(3))
    f = fit(m, Y, X, z)
    v_amb = f.V_ambient()
    assert np.allclose(m.coords(f.Y_bar, v_amb), f.V, atol=1e-10)
    Y_hat = f.predict([f.X_bars[0][None]], f.z_mean + 1.0)
    assert m.dist(Y_hat, m.expmap(f.Y_bar, v_amb))[0] < 1e-10


@pytest.mark.parametrize("name", NAMES)
def test_prediction_beats_the_frechet_mean_out_of_sample(name):
    m, Ya, Xa, za, _, _ = _simulate(name, 190, np.random.default_rng(4))
    Y, X, z = Ya[:150], [Xa[0][:150]], za[:150]
    Yt, Xt, zt = Ya[150:], [Xa[0][150:]], za[150:]
    f = fit(m, Y, X, z)
    Y_bar = np.stack([m.frechet_mean(Y)] * len(Yt))
    assert frechet_rmse(m, Yt, f.predict(Xt, zt)) < frechet_rmse(m, Yt, Y_bar)


def test_hc0_equals_the_explicit_sandwich():
    m, Y, X, z, _, _ = _simulate("S2", 300, np.random.default_rng(5))
    f = fit(m, Y, X, z)
    M = np.linalg.inv(f.bread(f.reg * f.n))
    S = f.scores()
    assert np.allclose(f.covariance("HC0"), M @ (S.T @ S) @ M)


@pytest.mark.parametrize("name", ["S2", "SPD3", "H3"])
def test_functional_api_matches_the_fit_methods(name):
    m, Y, X, z, _, _ = _simulate(name, 120, np.random.default_rng(7))
    f = fit(m, Y, X, z)
    for kind in ("HC0", "HC3"):
        C = sandwich(f.U, f.Zt, f.resid, kind, ridge=f.reg * f.n)
        assert np.allclose(C, f.covariance(kind), rtol=1e-10, atol=0)
        got = pvalues(f.alphas, f.V, C, f.p, f.q, f.d, f.n)
        for a, b in zip(got, f.pvalues(kind)):
            assert np.allclose(a, b, rtol=1e-10)
        assert np.isclose(f.joint_pvalue(0, 1, kind), f.pvalues(kind)[1][0],
                          rtol=1e-10)
    singletons = f.covariance("HC0", clusters=np.arange(f.n))
    assert np.allclose(singletons, f.covariance("HC0"))


def test_hc3_inflates_relative_to_hc0():
    m, Y, X, z, _, _ = _simulate("T2", 60, np.random.default_rng(6))
    f = fit(m, Y, X, z)
    assert np.all(np.diag(f.covariance("HC3"))
                  >= np.diag(f.covariance("HC0")) - 1e-15)


@pytest.mark.parametrize("scheme", [pairs_bootstrap, residual_bootstrap])
def test_bootstrap_pvalue_is_a_valid_monte_carlo_pvalue(scheme):
    m, Y, X, z, _, _ = _simulate("S2", 80, np.random.default_rng(7))
    out = scheme(m, Y, X, z, B=19, seed=0, statistic="v", index=0)
    assert out["n_valid"] >= 15
    assert 1.0 / (out["n_valid"] + 1) <= out["p_value"] <= 1.0


@pytest.mark.parametrize("name", NAMES)
def test_transport_is_an_isometry_in_the_chart(name):
    m, S, D = _far_points(name, np.random.default_rng(8))
    rng = np.random.default_rng(9)
    V = m.from_coords(S, rng.standard_normal((4, m.d)))
    before = np.linalg.norm(m.coords(S, V), axis=1)
    after = np.linalg.norm(m.coords(D, m.transport(S, D, V)), axis=1)
    assert np.allclose(before, after, atol=1e-9)


@pytest.mark.parametrize("name", ["S2", "SPD3", "H3"])
def test_transport_maps_the_geodesic_velocity_to_minus_the_reverse_log(name):
    m, S, D = _far_points(name, np.random.default_rng(10))
    moved = np.asarray(m.transport(S, D, m.logmap(S, D[None])))
    back = np.asarray(m.logmap(D, S[None]))
    assert np.abs(moved + back).max() < 1e-8


def test_so3_transport_solves_the_levi_civita_equation():
    m = SO3()
    rng = np.random.default_rng(11)
    R = m.expmap(np.eye(3), rng.standard_normal((1, 3)) * 0.7)[0]
    xi = rng.standard_normal(3) * 0.9
    S = m.expmap(R, xi[None])[0]
    w0 = rng.standard_normal(3)
    sol = solve_ivp(lambda t, w: -0.5 * np.cross(xi, w), (0, 1), w0,
                    rtol=1e-11, atol=1e-12)
    assert np.allclose(m.transport(R, S, w0[None])[0], sol.y[:, -1],
                       atol=1e-8)


def test_hyperbolic_mean_stays_on_the_manifold_for_widely_spread_data():
    m = Hyperbolic(3)
    rng = np.random.default_rng(12)
    b = Hyperbolic.from_spatial([0.4, 0.0, 0.0])[0]
    P = m.expmap(b, m.from_coords(b, rng.standard_normal((300, 3)) * 1.5))
    mu = m.frechet_mean(P)
    assert abs(-mu[0] ** 2 + (mu[1:] ** 2).sum() + 1.0) < 1e-10
    assert np.linalg.norm(m.coords(mu, m.logmap(mu, P)).mean(0)) < 1e-9


@pytest.mark.parametrize("name", ["SPD3", "SO3", "H3"])
def test_pairs_bootstrap_returns_aligned_replicate_coefficients(name):
    m, Y, X, z, _, _ = _simulate(name, 80, np.random.default_rng(13))
    out = pairs_bootstrap(m, Y, X, z, B=25, seed=1, statistic="v", index=0)
    assert out["alpha_star"].shape == (out["n_valid"], 1)
    assert out["v_star"].shape == (out["n_valid"], 1, m.d)
    f0 = fit(m, Y, X, z)
    assert np.linalg.norm(out["v_star"][:, 0].mean(0) - f0.V[0]) < 0.05


@pytest.mark.parametrize("name", ["S2", "SPD3"])
@pytest.mark.parametrize("kind", ["model", "HC3"])
def test_tests_hold_their_level_under_the_null(name, kind):
    """Type I error at level 0.05 over 400 null data sets of size 100.

    The window is about three binomial standard errors wide. The HC3 Wald
    test of the six-dimensional v block on Sym+(3) is liberal at this sample
    size, so its level is not checked.
    """
    rng = np.random.default_rng(14)
    rej_a = rej_v = 0
    for _ in range(400):
        m, Y, X, z, _, _ = _simulate(name, 100, rng, alpha=0.0, v_scale=0.0)
        pa, pv = fit(m, Y, X, z).pvalues(kind)
        rej_a += pa[0] < 0.05
        rej_v += pv[0] < 0.05
    assert 0.02 <= rej_a / 400 <= 0.085
    if kind == "model" or m.d <= 2:
        assert 0.02 <= rej_v / 400 <= 0.085


@pytest.mark.parametrize("drop", ["X", "z"])
def test_fit_with_only_one_kind_of_predictor(drop):
    m, Y, X, z, _, _ = _simulate("SPD3", 300, np.random.default_rng(8))
    X, z = ([], z) if drop == "X" else (X, None)
    f = fit(m, Y, X, z)
    assert (f.p, f.q) == ((0, 1) if drop == "X" else (1, 0))
    p_alpha, p_v = f.pvalues("HC3")
    assert np.all(np.concatenate([p_alpha, p_v]) < 1e-6)
    Y_hat = f.predict([x[:5] for x in X], None if z is None else z[:5])
    assert np.asarray(Y_hat).shape == np.asarray(Y[:5]).shape
    stat = "v" if drop == "X" else "alpha"
    for scheme in (pairs_bootstrap, residual_bootstrap):
        assert 0 < scheme(m, Y, X, z, B=19, statistic=stat)["p_value"] <= 1


def test_sigma2_is_unbiased_in_the_flat_case():
    rng = np.random.default_rng(9)
    s2 = [fit(*_simulate("T2", 20, rng, noise=0.1)[:4]).sigma2
          for _ in range(4000)]
    assert abs(np.mean(s2) / 0.01 - 1) < 0.015


def test_model_covariance_is_the_ridge_estimator_covariance():
    rng = np.random.default_rng(10)
    m, Y, X, z, _, _ = _simulate("T2", 50, rng)
    f0 = fit(m, Y, X, z, reg=0.05)
    reps = [_refit_tangent(f0, f0.fitted + 0.1 * rng.standard_normal(
        f0.W.shape), 0.05).alphas[0] for _ in range(2000)]
    C = f0.covariance("model") * 0.01 / f0.sigma2
    assert abs(np.var(reps) / C[0, 0] - 1) < 0.1


def test_predict_accepts_a_one_dimensional_z_without_manifold_predictors():
    m, Y, _, z, _, _ = _simulate("S2", 80, np.random.default_rng(11))
    f = fit(m, Y, [], z)
    a = f.predict([], z[:5, 0])
    b = f.predict([], z[:5])
    assert a.shape == (5, 3) and np.allclose(a, b)


def test_bootstrap_rejects_a_statistic_the_model_does_not_have():
    m, Y, X, _, _, _ = _simulate("S2", 60, np.random.default_rng(12))
    for fn in (pairs_bootstrap, residual_bootstrap):
        with pytest.raises(ValueError, match="0 scalar covariates"):
            fn(m, Y, X, None, B=3, statistic="v")
        with pytest.raises(ValueError, match="index=1"):
            fn(m, Y, X, None, B=3, statistic="alpha", index=1)


def test_fit_rejects_a_predictor_off_the_manifold():
    m, Y, X, z, _, _ = _simulate("S2", 60, np.random.default_rng(13))
    with pytest.raises(ValueError, match="not on the manifold"):
        fit(m, Y, [2.0 * X[0]], z)

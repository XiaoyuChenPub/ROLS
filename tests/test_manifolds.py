import numpy as np
import pytest

from ROLS.manifolds import SO3, SPD, Hyperbolic, Sphere, Torus
from ROLS.manifolds.spd import _powm
from ROLS.manifolds.torus import wrap

ALL = ["S2", "T2", "SPD3", "H3", "SO3"]


def _sample(name, n=40, seed=0):
    rng = np.random.default_rng(seed)
    if name == "S2":
        m = Sphere(3)
        P = m.expmap(np.array([0.0, 0.0, 1.0]),
                     rng.standard_normal((n, 3)) * 0.3)
    elif name == "T2":
        m = Torus(2)
        P = rng.uniform(-2.0, 2.0, (n, 2))
    elif name == "SPD3":
        m = SPD(3)
        A = rng.standard_normal((n, 3, 3))
        P = np.einsum("nij,nkj->nik", A, A) + 3 * np.eye(3)
    elif name == "H3":
        m = Hyperbolic(3)
        b = m.from_spatial(np.zeros((1, 3)))[0]
        P = m.expmap(b, m.from_coords(b, rng.standard_normal((n, 3)) * 0.3))
    else:
        m = SO3()
        P = m.expmap(np.eye(3), rng.standard_normal((n, 3)) * 0.3)
    return m, P


@pytest.mark.parametrize("name", ALL)
def test_log_exp_roundtrip(name):
    m, P = _sample(name)
    b = m.frechet_mean(P)
    back = np.asarray(m.expmap(b, m.logmap(b, P)))
    assert np.abs(back - np.asarray(P)).max() < 1e-10


@pytest.mark.parametrize("name", ALL)
def test_frechet_mean_is_a_stationary_point(name):
    m, P = _sample(name)
    b = m.frechet_mean(P)
    g = m.logmap(b, P).mean(0)
    assert m.tangent_norm(b, g) < 1e-9


@pytest.mark.parametrize("name", ALL)
def test_coords_have_intrinsic_dimension(name):
    m, P = _sample(name)
    b = m.frechet_mean(P)
    C = m.coords(b, m.logmap(b, P))
    assert C.shape == (len(P), m.d)


@pytest.mark.parametrize("name", ALL)
def test_tangent_norm_is_the_geodesic_distance(name):
    m, P = _sample(name)
    b = m.frechet_mean(P)
    riem = np.array([m.tangent_norm(b, v[None])
                     for v in np.asarray(m.logmap(b, P))])
    assert np.abs(riem - m.dist(np.stack([b] * len(P)), P)).max() < 1e-9


@pytest.mark.parametrize("name", ALL)
def test_dist_accepts_a_single_point(name):
    m, P = _sample(name)
    P = np.asarray(P)
    full = m.dist(np.stack([P[0]] * len(P)), P)
    assert np.allclose(m.dist(P[0], P), full, atol=1e-12)
    assert np.allclose(m.dist(P[0], P[1]), full[1:2], atol=1e-12)


def test_spd_coords_are_affine_invariant():
    rng = np.random.default_rng(3)
    m = SPD(3)
    A = rng.standard_normal((3, 3))
    B = A @ A.T + 2 * np.eye(3)
    S = rng.standard_normal((2, 3, 3))
    S = 0.5 * (S + np.swapaxes(S, 1, 2))
    C = m.coords(B, S)
    Bi = np.linalg.inv(B)
    want = np.trace(Bi @ S[0] @ Bi @ S[1])
    assert abs(float(C[0] @ C[1]) - want) < 1e-9


def test_spd_expmap_of_indefinite_tangent_matrices():
    S = np.diag([-1.0, 0.0, 2.0])
    got = _powm(S, np.exp, positive=False)
    assert np.abs(np.diag(got) - np.exp([-1.0, 0.0, 2.0])).max() < 1e-12
    moved = np.asarray(SPD(3).expmap(np.eye(3), S[None]))[0]
    assert np.abs(moved - np.diag(np.exp([-1.0, 0.0, 2.0]))).max() < 1e-12


@pytest.mark.parametrize("name", ALL)
def test_transport_preserves_length(name):
    m, P = _sample(name)
    b = m.frechet_mean(P)
    V = m.logmap(b, P)[:5]
    tgt = np.asarray(P)[0]
    moved = m.transport(b, tgt, V)
    a = np.array([m.tangent_norm(b, v[None]) for v in np.asarray(V)])
    c = np.array([m.tangent_norm(tgt, v[None]) for v in np.asarray(moved)])
    assert np.abs(a - c).max() < 1e-9


def test_hyperbolic_ball_to_lorentz_lands_on_the_hyperboloid():
    m = Hyperbolic(3)
    rng = np.random.default_rng(0)
    P = m.from_ball(rng.standard_normal((6, 3)) * 0.2)
    assert np.abs(-P[:, 0] ** 2 + (P[:, 1:] ** 2).sum(1) + 1.0).max() < 1e-10


def test_torus_mean_is_a_stationary_point_beyond_the_circular_mean():
    rng = np.random.default_rng(5)
    m = Torus(2)
    P = np.concatenate([rng.normal(2.8, 0.3, (60, 2)),
                        rng.normal(-2.8, 0.3, (40, 2))])
    mu = m.frechet_mean(wrap(P))
    assert np.linalg.norm(wrap(P - mu).mean(0)) < 1e-10


@pytest.mark.parametrize("name", ALL)
def test_transport_is_continuous_at_the_source(name):
    """Transport to the source is the identity, and to a point eps away it
    moves a tangent vector by O(eps)."""
    m, P = _sample(name)
    P = np.asarray(P)
    b = m.expmap(P[0], m.from_coords(
        P[0], 2.0 * np.ones((1, m.d)) / np.sqrt(m.d)))[0]
    V = np.asarray(m.logmap(b, P[1:5]))
    for eps in (0.0, 1e-12, 1e-9):
        t = m.expmap(b, m.from_coords(b, eps * np.ones((1, m.d))))[0]
        moved = np.asarray(m.transport(b, t, V))
        assert np.abs(moved - V).max() < 1e-7 * np.abs(V).max()


@pytest.mark.parametrize("theta",
                         [np.pi, np.pi - 1e-9, np.pi - 1e-7, 1.0, 1e-9, 0.0])
def test_so3_log_is_accurate_up_to_the_cut_locus(theta):
    m = SO3()
    R = m.expmap(np.eye(3), (theta * np.array([1.0, 2.0, 2.0]) / 3)[None])
    w = m.logmap(np.eye(3), R)
    assert abs(np.linalg.norm(w) - theta) < 1e-9
    assert np.abs(m.expmap(np.eye(3), w) - R).max() < 1e-12
    assert abs(m.dist(np.eye(3), R[0])[0] - theta) < 1e-9


@pytest.mark.parametrize("eps", [1e-9, 1e-7, 1e-5, 1.0])
def test_sphere_transport_lands_in_the_target_tangent_space(eps):
    m = Sphere(3)
    s = np.array([0.0, 0.6, 0.8])
    V = m.from_coords(s, np.array([[1.0, 0.5]]))
    t = m.expmap(s, m.from_coords(s, np.array([[eps, 0.0]])))[0]
    assert abs(float(m.transport(s, t, V)[0] @ t)) < 1e-14


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("eps", [1e-4, 1e-6, 1e-8])
def test_log_and_dist_are_accurate_at_small_separations(name, eps):
    m, P = _sample(name)
    b = np.asarray(P)[0]
    p = m.expmap(b, m.from_coords(b, eps * np.ones((1, m.d)) / np.sqrt(m.d)))
    tol = 1e-9 * eps + 4e-15
    assert abs(np.linalg.norm(m.coords(b, m.logmap(b, p))) - eps) < tol
    assert abs(m.dist(b, p)[0] - eps) < tol


@pytest.mark.parametrize("r", [2.0, 5.0])
def test_hyperbolic_dist_far_from_the_origin(r):
    m = Hyperbolic(3)
    b = Hyperbolic.from_spatial(np.sinh(r) * np.array([0.6, 0.8, 0.0]))[0]
    direction = np.array([[1.0, 0.3, -0.2]]) / np.sqrt(1.13)
    for eps in (1e-2, 1e-4, 1e-6):
        p = m.expmap(b, m.from_coords(b, eps * direction))
        assert abs(m.dist(b, p)[0] - eps) < 1e-6 * eps


@pytest.mark.parametrize("name", ["S2", "SPD3", "H3", "SO3"])
def test_check_rejects_points_off_the_manifold(name):
    m, P = _sample(name)
    P = np.asarray(P)
    m.check(P)
    bad = P.copy()
    if name == "S2":
        bad[3] *= 2.0
    elif name == "SPD3":
        bad[3, 0, 1] += 0.5
    elif name == "H3":
        bad[3, 0] += 0.3
    else:
        bad[3] *= 1.1
    with pytest.raises(ValueError, match="not on the manifold"):
        m.check(bad)


def test_frechet_mean_warns_when_the_data_have_no_unique_mean():
    m = SO3()
    P = m.project(np.random.default_rng(1).standard_normal((200, 3, 3)))
    with pytest.warns(RuntimeWarning, match="no unique mean"):
        m.frechet_mean(P)


def test_frechet_mean_rejects_invalid_weights():
    m, P = _sample("S2")
    with pytest.raises(ValueError, match="nonnegative"):
        m.frechet_mean(P, weights=-np.ones(len(P)))

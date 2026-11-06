"""Baseline regressions for a manifold-valued response.

Each function fits on training data and returns predicted responses at new
covariates, so every method can be scored with ``frechet_rmse``.

Frechet regression (Petersen and Mueller, 2019), single-index Frechet
regression (Bhattacharjee and Mueller, 2023) and nonlinear Frechet regression
(Bhattacharjee, Li and Xue, 2025) predict with a weighted Frechet mean whose
weights sum to one and may be negative. ``weighted_frechet_mean`` computes it.
"""

import numpy as np
from scipy.spatial.distance import cdist

from .estimator import fit

_BANDWIDTHS = np.array([0.1, 0.15, 0.2, 0.3, 0.45, 0.7, 1.0, 1.5, 2.5, 5.0])
_TIKHONOV_GRID = 10.0 ** np.arange(2, -7, -1)
_N_STARTS = 16


def _tangent_response(m, Y):
    """Returns the Frechet mean of ``Y`` and the coordinates of its log map."""
    Y_bar = m.frechet_mean(Y)
    return Y_bar, m.coords(Y_bar, m.logmap(Y_bar, Y))


def _center(z_tr, z_te):
    """Centers training and test covariates at the training mean."""
    z_tr = np.asarray(z_tr, float)
    z_te = np.asarray(z_te, float)
    mu = z_tr.mean(0)
    return z_tr - mu, z_te - mu


def _exp_coords(m, base, C):
    """Maps intrinsic coordinates at ``base`` to points on the manifold."""
    return m.expmap(base, m.from_coords(base, C))


def _has_covariates(z):
    """Returns whether ``z`` holds at least one scalar covariate."""
    return z is not None and np.asarray(z).shape[1] > 0


def weighted_frechet_mean(m, Y, weights, init, tol=1e-8, max_iter=100):
    """Minimizes sum_i w_i d^2(Y_i, y) over y by gradient descent.

    The objective is not convex when weights are negative or the manifold has
    positive curvature, so the result is a local minimizer near ``init``. A
    unit step solves the Euclidean problem exactly. Armijo backtracking keeps
    the objective decreasing otherwise.

    Args:
        m: Manifold.
        Y: Batch of points.
        weights: (n,) weights that sum to one, possibly negative.
        init: Starting point.
        tol: The iteration stops once the gradient norm is below ``tol``.
        max_iter: Maximum number of gradient steps.

    Returns:
        The minimizer as a single point.
    """
    w = np.asarray(weights, float)
    mu = init
    C = m.coords(mu, m.logmap(mu, Y))
    obj = float(w @ (C ** 2).sum(1))
    g = w @ C
    g2 = float(g @ g)
    t = 1.0
    for _ in range(max_iter):
        if g2 < tol ** 2:
            break
        while True:
            cand = _exp_coords(m, mu, t * g[None])[0]
            C = m.coords(cand, m.logmap(cand, Y))
            obj_c = float(w @ (C ** 2).sum(1))
            if obj_c <= obj - t * g2 or t < 1e-6:
                break
            t *= 0.5
        if obj_c >= obj:
            break
        mu, obj = cand, obj_c
        g = w @ C
        g2 = float(g @ g)
        t = min(1.0, 2.0 * t)
    return mu


def _weighted_means(m, Y, Y_bar, W, S):
    """Solves one weighted mean per row of ``S``.

    The objective can have several local minima when the responses are widely
    spread, as on a torus. Each solve therefore starts from the best, by
    objective value, of the weighted tangent average Exp_Ybar(S_i W) and a fixed
    sample of responses.
    """
    n = len(Y)
    rng = np.random.default_rng(0)
    sample = rng.choice(n, min(_N_STARTS, n), replace=False)
    d2 = np.stack([(m.coords(Y[j], m.logmap(Y[j], Y)) ** 2).sum(1)
                   for j in sample])
    means = []
    for s, y0, obj in zip(S, _exp_coords(m, Y_bar, S @ W), S @ d2.T):
        C = m.coords(y0, m.logmap(y0, Y))
        if obj.min() < s @ (C ** 2).sum(1):
            y0 = Y[sample[obj.argmin()]]
        means.append(weighted_frechet_mean(m, Y, s, y0))
    return np.stack(means)


def frechet_mean(m, Y_tr, n_test):
    """Predicts the Frechet mean of the training responses."""
    Y_bar = m.frechet_mean(Y_tr)
    return _exp_coords(m, Y_bar, np.zeros((n_test, m.d)))


def geodesic_regression(m, Y_tr, z_tr, z_te, reg=1e-6):
    """Fits a tangent-linear model at the Frechet mean of the response.

    This is ROLS with no manifold predictor, and it extends the geodesic
    regression of Fletcher (2013) to several scalar covariates.

    Args:
        m: Manifold.
        Y_tr: Training responses.
        z_tr: (n, q) training covariates. With q = 0 the method predicts the
            Frechet mean.
        z_te: (n_test, q) test covariates.
        reg: Ridge constant.

    Returns:
        Predicted responses at ``z_te``.
    """
    if not _has_covariates(z_tr):
        return frechet_mean(m, Y_tr, len(z_te))
    Y_bar, W = _tangent_response(m, Y_tr)
    Zc, Zc_te = _center(z_tr, z_te)
    V = np.linalg.solve(Zc.T @ Zc + reg * np.eye(Zc.shape[1]), Zc.T @ W)
    return _exp_coords(m, Y_bar, Zc_te @ V)


def frechet_regression(m, Y_tr, z_tr, z_te):
    """Global Frechet regression on scalar covariates.

    Predicts argmin_y sum_i s_i(z) d^2(Y_i, y) with the weights
    s_i(z) = (1 + (z_i - zbar)^T S^-1 (z - zbar)) / n of Petersen and Mueller
    (2019, eq. 2.9-2.10), where S is the sample covariance of the covariates.

    Args:
        m: Manifold.
        Y_tr: Training responses.
        z_tr: (n, q) training covariates. With q = 0 the method predicts the
            Frechet mean.
        z_te: (n_test, q) test covariates.

    Returns:
        Predicted responses at ``z_te``.
    """
    if not _has_covariates(z_tr):
        return frechet_mean(m, Y_tr, len(z_te))
    Y_bar, W = _tangent_response(m, Y_tr)
    Zc, Zc_te = _center(z_tr, z_te)
    n = len(Zc)
    S = (1.0 + Zc_te @ np.linalg.solve(Zc.T @ Zc / n, Zc.T)) / n
    return _weighted_means(m, Y_tr, Y_bar, W, S)


def _local_linear_weights(t, t0, b):
    """Returns local linear weights on an index with a Gaussian kernel.

    Args:
        t: (n,) index values of the observations.
        t0: (k,) index values to predict at.
        b: Bandwidth.

    Returns:
        (k, n) weights whose rows sum to one. A row reverts to kernel weights
        when fewer than two observations carry weight.
    """
    u = t[None, :] - t0[:, None]
    e = (u / b) ** 2
    K = np.exp(-0.5 * (e - e.min(1, keepdims=True)))
    s0 = K.sum(1, keepdims=True)
    s1 = (K * u).sum(1, keepdims=True)
    s2 = (K * u * u).sum(1, keepdims=True)
    det = s0 * s2 - s1 ** 2
    usable = det > 1e-12 * s0 * s2
    return np.where(usable, K * (s2 - s1 * u) / np.where(usable, det, 1.0),
                    K / s0)


def _cv_bandwidth(W, t, folds):
    """Returns the bandwidth with the smallest cross-validated tangent RSS."""
    best_err, best_b = np.inf, None
    for b in _BANDWIDTHS * t.std():
        err = 0.0
        for te, tr in folds:
            fitted = _local_linear_weights(t[tr], t[te], b) @ W[tr]
            err += ((W[te] - fitted) ** 2).sum()
        if err < best_err:
            best_err, best_b = err, b
    return best_b


def single_index_frechet(m, Y_tr, z_tr, z_te, n_directions=500, seed=0):
    """Single-index Frechet regression on scalar covariates.

    Follows Bhattacharjee and Mueller (2023, Section 4). For each unit
    direction in a random grid, the bandwidth is chosen by 5-fold
    cross-validation, and the direction with the smallest in-sample loss V_n
    is kept. V_n uses all observations, which their Section 3 allows in place
    of the binned loss, and the kernel is Gaussian. Cross-validation and V_n
    use tangent residuals at the Frechet mean of the response, which are exact
    in Euclidean space. The prediction is the local linear weighted Frechet
    mean on the selected index.

    Args:
        m: Manifold.
        Y_tr: Training responses.
        z_tr: (n, q) training covariates. With q = 0 the method predicts the
            Frechet mean.
        z_te: (n_test, q) test covariates.
        n_directions: Size of the grid of directions.
        seed: Seed of the grid and the folds.

    Returns:
        Predicted responses at ``z_te``.
    """
    if not _has_covariates(z_tr):
        return frechet_mean(m, Y_tr, len(z_te))
    Y_bar, W = _tangent_response(m, Y_tr)
    Zc, Zc_te = _center(z_tr, z_te)
    n, q = Zc.shape
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = [(te, np.setdiff1d(perm, te)) for te in np.array_split(perm, 5)]
    if q == 1:
        directions = np.ones((1, 1))
    else:
        directions = rng.standard_normal((n_directions, q))
        directions[:, 0] = np.abs(directions[:, 0])
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    best_loss, best_theta, best_b = np.inf, None, None
    for theta in directions:
        t = Zc @ theta
        b = _cv_bandwidth(W, t, folds)
        loss = ((W - _local_linear_weights(t, t, b) @ W) ** 2).sum(1).mean()
        if loss < best_loss:
            best_loss, best_theta, best_b = loss, theta, b
    S = _local_linear_weights(Zc @ best_theta, Zc_te @ best_theta, best_b)
    return _weighted_means(m, Y_tr, Y_bar, W, S)


def _features(m, X_tr, X_te, z_tr, z_te):
    """Stacks predictor log-map coordinates at their own means with ``z``."""
    F_tr, F_te = [], []
    for Xtr, Xte in zip(X_tr, X_te):
        X_bar = m.frechet_mean(Xtr)
        F_tr.append(m.coords(X_bar, m.logmap(X_bar, Xtr)))
        F_te.append(m.coords(X_bar, m.logmap(X_bar, Xte)))
    if _has_covariates(z_tr):
        F_tr.append(np.asarray(z_tr, float))
        F_te.append(np.asarray(z_te, float))
    return np.concatenate(F_tr, 1), np.concatenate(F_te, 1)


def _nonlinear_weight_maps(K, grid):
    """Yields the weight maps of nonlinear Frechet regression over a grid.

    The weights at x are w(x) = 1 / n + A k(x) with k(x)_i = kappa(X_i, x) and
    A = G (G + eps I)^-1 Q (K + eps I)^-1 Q, where G = Q K Q and
    Q = I - 1 1^T / n (Bhattacharjee, Li and Xue, 2025, Section 4.2).

    Args:
        K: (n, n) kernel matrix.
        grid: Tikhonov constants.

    Yields:
        A pair (A, df) for each constant, where
        df = tr[Q G (G + eps I)^-1 + 1 1^T / n] is the effective degrees of
        freedom in their generalized cross-validation criterion.
    """
    n = len(K)
    Q = np.eye(n) - 1.0 / n
    lam_g, U_g = np.linalg.eigh(Q @ K @ Q)
    lam_k, U_k = np.linalg.eigh(K)
    for eps in grid:
        shrink = lam_g / (lam_g + eps)
        A = (U_g * shrink) @ U_g.T @ Q @ ((U_k / (lam_k + eps)) @ U_k.T) @ Q
        yield A, float(shrink.sum()) + 1.0


def nonlinear_frechet(m, Y_tr, X_tr, z_tr, X_te, z_te):
    """Nonlinear Frechet regression on manifold and scalar predictors.

    Follows Bhattacharjee, Li and Xue (2025, Sections 4 and 6). The kernel is
    the Gaussian radial basis function exp(-gamma ||f - f'||^2) with
    gamma = 1 / (2 sigma^2), where sigma^2 is the mean pairwise squared
    distance. It acts on the log-map coordinates f of the manifold predictors
    and the scalars, because exp(-gamma d^2) with a geodesic distance d need not
    be positive definite on a curved space (their Assumption 1). The Tikhonov
    constant minimizes their generalized cross-validation criterion on
    {1e-6, ..., 1e2}. This extends their grid, which stops at 1e-1, because
    the criterion often has its minimum above that value. The search runs from
    the largest constant down and stops after two consecutive increases,
    because the weights grow without bound as the constant shrinks and the
    weighted mean eventually overflows on a noncompact manifold.

    Args:
        m: Manifold.
        Y_tr: Training responses.
        X_tr: List of p batches of training manifold predictors.
        z_tr: (n, q) training covariates or None.
        X_te: List of p batches of test manifold predictors.
        z_te: (n_test, q) test covariates or None.

    Returns:
        Predicted responses at the test predictors.
    """
    Y_bar, W = _tangent_response(m, Y_tr)
    F_tr, F_te = _features(m, X_tr, X_te, z_tr, z_te)
    n = len(F_tr)
    D = cdist(F_tr, F_tr, "sqeuclidean")
    gamma = 1.0 / (2.0 * D[np.triu_indices(n, 1)].mean())
    K = np.exp(-gamma * D)
    best_gcv, best_A, increases = np.inf, None, 0
    for A, df in _nonlinear_weight_maps(K, _TIKHONOV_GRID):
        fitted = _weighted_means(m, Y_tr, Y_bar, W, 1.0 / n + K @ A.T)
        rss = (m.dist(Y_tr, fitted) ** 2).mean()
        gcv = rss / (1.0 - df / n) ** 2
        if best_A is None or gcv < best_gcv:
            best_gcv, best_A, increases = gcv, A, 0
        else:
            increases += 1
            if increases == 2:
                break
    S = 1.0 / n + np.exp(-gamma * cdist(F_te, F_tr, "sqeuclidean")) @ best_A.T
    return _weighted_means(m, Y_tr, Y_bar, W, S)


def rols_manifold_only(m, Y_tr, X_tr, X_te):
    """Fits ROLS with the scalar block dropped, ROLS (m) in the paper."""
    return fit(m, Y_tr, X_tr, None).predict(X_te, None)


def tangent_linear_ridge(m, Y_tr, X_tr, z_tr, X_te, z_te, reg=1e-3):
    """Replaces each scalar alpha_j of ROLS with a full d x d matrix A_j.

    Fits w_i = sum_j A_j u_ij + V^T (z_i - zbar) by ridge regression on the
    ROLS design.

    Args:
        m: Manifold.
        Y_tr: Training responses.
        X_tr: List of p batches of training manifold predictors.
        z_tr: (n, q) training covariates or None.
        X_te: List of p batches of test manifold predictors.
        z_te: (n_test, q) test covariates or None.
        reg: Ridge constant.

    Returns:
        Predicted responses at the test predictors.
    """
    f = fit(m, Y_tr, X_tr, z_tr)
    W, U, Zt = f.W, f.U, f.Zt
    n, d = W.shape
    p, q = len(U), Zt.shape[1]
    cols = [np.einsum('nc,bB->nbBc', Uj, np.eye(d)).reshape(n, d, d * d)
            for Uj in U]
    cols += [Zt[:, k, None, None] * np.eye(d)[None] for k in range(q)]
    X = np.concatenate(cols, axis=2)
    XtX = np.einsum('nak,nal->kl', X, X) / n
    Xty = np.einsum('nak,na->k', X, W) / n
    theta = np.linalg.solve(XtX + reg * np.eye(XtX.shape[0]), Xty)
    A = [theta[j * d * d:(j + 1) * d * d].reshape(d, d) for j in range(p)]
    V = theta[p * d * d:].reshape(q, d)
    tangent = np.zeros((len(X_te[0]), d))
    for j in range(p):
        U_te = m.coords(f.Y_bar, m.transport(f.X_bars[j], f.Y_bar,
                                             m.logmap(f.X_bars[j], X_te[j])))
        tangent += U_te @ A[j].T
    if q:
        tangent += (np.asarray(z_te, float) - f.z_mean) @ V
    return _exp_coords(m, f.Y_bar, tangent)

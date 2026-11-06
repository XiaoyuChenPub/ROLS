"""Bootstrap tests for ROLS coefficients.

Two schemes are provided. The pairs bootstrap resamples observations and
refits everything, including the Frechet means, so it targets the full
sampling variance of the coefficients. The residual bootstrap keeps the
fitted base points and resamples tangent residuals, so it targets the
variance of the tangent-space model only.

Both compute Monte Carlo p-values (1 + #{b : T*_b >= T}) / (B + 1) for a
studentized statistic, where the observed statistic and every replicate are
studentized by their own model-based covariance.
"""

import numpy as np

from .estimator import ROLSFit, fit, solve_normal_equations


def _move(manifold, src, dst, coords):
    """Parallel transports intrinsic coordinates at ``src`` to ``dst``."""
    m = manifold
    V = m.from_coords(src, np.atleast_2d(coords))
    return m.coords(dst, m.transport(src, dst, V))


def _check_statistic(f0, statistic, index):
    if statistic not in ("v", "alpha"):
        raise ValueError(
            f"statistic must be 'v' or 'alpha', got {statistic!r}")
    count = f0.q if statistic == "v" else f0.p
    if not 0 <= index < count:
        what = ("scalar covariates" if statistic == "v"
                else "manifold predictors")
        raise ValueError(f"statistic={statistic!r} index={index} but the "
                         f"model has {count} {what}")


def _observed_statistic(f0, statistic, index):
    C0 = f0.covariance("model")
    if statistic == "v":
        sl = slice(f0.p + index * f0.d, f0.p + (index + 1) * f0.d)
        v0 = np.asarray(f0.V[index], float)
        return float(v0 @ np.linalg.solve(C0[sl, sl], v0))
    return float(f0.alphas[index] ** 2 / C0[index, index])


def pairs_bootstrap(manifold, Y, X_list=(), z=None, B=199, reg=1e-6, seed=0,
                    statistic="v", index=0):
    """Tests one coefficient with the studentized pairs bootstrap.

    Each replicate refits the model on a resample of the observations. The
    replicate statistic is formed in the resample's tangent space after
    parallel transporting the full-data coefficient there.

    Args:
        manifold: The manifold of the response and the manifold predictors.
        Y: Batch of n response points.
        X_list: List of p batches of manifold predictors.
        z: Array of shape (n, q) of scalar covariates.
        B: Number of bootstrap replicates.
        reg: Ridge parameter passed to ``fit``.
        seed: Seed of the resampling.
        statistic: "v" for the chi^2_d Wald statistic of v_index = 0, or
            "alpha" for the squared t statistic of alpha_index = 0.
        index: Which coefficient to test.

    Returns:
        A dict with the observed statistic ``T_obs``, the replicate
        statistics ``replicates``, the Monte Carlo ``p_value``, the number
        of successful replicates ``n_valid``, the ``scheme`` name, and the
        replicate coefficients ``alpha_star`` of shape (n_valid, p) and
        ``v_star`` of shape (n_valid, q, d), the latter transported to the
        tangent space of the full-data fit.

    Raises:
        ValueError: If the model does not have the requested coefficient.
    """
    rng = np.random.default_rng(seed)
    Y = np.asarray(Y)
    X_list = [np.asarray(X) for X in X_list]
    n = len(Y)
    f0 = fit(manifold, Y, X_list, z, reg=reg)
    _check_statistic(f0, statistic, index)
    T_obs = _observed_statistic(f0, statistic, index)
    p, d = f0.p, f0.d
    sl = slice(p + index * d, p + (index + 1) * d)

    reps, a_star, v_star = [], [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        try:
            fb = fit(manifold, Y[idx], [X[idx] for X in X_list],
                     None if z is None else np.asarray(z)[idx], reg=reg)
            Cb = fb.covariance("model")
            if statistic == "v":
                v0_b = _move(manifold, f0.Y_bar, fb.Y_bar,
                             f0.V[index:index + 1])[0]
                diff = np.asarray(fb.V[index], float) - v0_b
                Tb = float(diff @ np.linalg.solve(Cb[sl, sl], diff))
            else:
                Tb = float((fb.alphas[index] - f0.alphas[index]) ** 2
                           / Cb[index, index])
        except np.linalg.LinAlgError:
            continue
        if np.isfinite(Tb):
            reps.append(Tb)
            a_star.append(np.asarray(fb.alphas, float))
            v_star.append(_move(manifold, fb.Y_bar, f0.Y_bar, fb.V)
                          if f0.q else np.zeros((0, d)))
    reps = np.asarray(reps)
    p_value = (1.0 + int((reps >= T_obs).sum())) / (len(reps) + 1.0)
    return {"T_obs": T_obs, "replicates": reps, "p_value": float(p_value),
            "n_valid": int(len(reps)), "scheme": "pairs",
            "alpha_star": np.asarray(a_star).reshape(len(reps), p),
            "v_star": np.asarray(v_star).reshape(len(reps), f0.q, d)}


def residual_bootstrap(manifold, Y, X_list=(), z=None, B=199, reg=1e-6,
                       seed=0, statistic="v", index=0):
    """Tests one coefficient with the residual bootstrap at fixed base points.

    Each replicate adds recentered resampled tangent residuals to the fitted
    tangent responses and re-solves the normal equations.

    Args:
        See ``pairs_bootstrap``.

    Returns:
        A dict with ``T_obs``, ``replicates``, ``p_value``, ``n_valid`` and
        ``scheme``, as in ``pairs_bootstrap``.

    Raises:
        ValueError: If the model does not have the requested coefficient.
    """
    rng = np.random.default_rng(seed)
    f0 = fit(manifold, Y, X_list, z, reg=reg)
    _check_statistic(f0, statistic, index)
    T_obs = _observed_statistic(f0, statistic, index)
    p, d = f0.p, f0.d
    sl = slice(p + index * d, p + (index + 1) * d)

    reps = []
    for _ in range(B):
        r = f0.resid[rng.integers(0, f0.n, f0.n)]
        fb = _refit_tangent(f0, f0.fitted + r - r.mean(0), reg)
        Cb = fb.covariance("model")
        if statistic == "v":
            vb = np.asarray(fb.V[index], float) - np.asarray(f0.V[index], float)
            Tb = float(vb @ np.linalg.solve(Cb[sl, sl], vb))
        else:
            Tb = float((fb.alphas[index] - f0.alphas[index]) ** 2
                       / Cb[index, index])
        if np.isfinite(Tb):
            reps.append(Tb)
    reps = np.asarray(reps)
    p_value = (1.0 + int((reps >= T_obs).sum())) / (len(reps) + 1.0)
    return {"T_obs": T_obs, "replicates": reps, "p_value": float(p_value),
            "n_valid": int(len(reps)), "scheme": "residual"}


def _refit_tangent(f0, W_new, reg):
    """Re-solves the normal equations for new tangent responses."""
    p, q, d = f0.p, f0.q, f0.d
    theta = solve_normal_equations(f0.U, f0.Zt, W_new, d, reg * f0.n)
    return ROLSFit(f0.manifold, f0.Y_bar, f0.X_bars, theta[:p],
                   theta[p:].reshape(q, d), W_new, f0.U, f0.Zt, f0.z_mean, reg)

"""Design matrices, covariance estimators and tests for the tangent model.

The tangent model is w_i = Z_i beta + eps_i with the d x k block row

    Z_i = [u_i1, ..., u_ip, z~_i1 I_d, ..., z~_iq I_d],

beta = (alpha_1, ..., alpha_p, v_1, ..., v_q), the Gram matrix
A = sum_i Z_i^T Z_i and the scores s_i = Z_i^T r_i. The sandwich covariance
is M B M with M = (A + ridge I)^-1 and

    HC0    B = sum_i s_i s_i^T,
    HC3    B = sum_i s~_i s~_i^T, with s~_i built from (I_d - H_ii)^-1 r_i,

where H_ii = Z_i M Z_i^T is the diagonal block of the hat matrix
(MacKinnon and White, 1985).
"""

import numpy as np
from scipy import stats


def gram(U_list, Zt, d):
    """Returns the k x k Gram matrix A = sum_i Z_i^T Z_i.

    Args:
        U_list: List of p arrays of shape (n, d), the transported predictor
            tangents in intrinsic coordinates.
        Zt: Array of shape (n, q), the centered scalar covariates.
        d: Intrinsic dimension of the response manifold.
    """
    p, q = len(U_list), Zt.shape[1]
    k = p + q * d
    A = np.zeros((k, k))
    for a in range(p):
        for b in range(p):
            A[a, b] = float((U_list[a] * U_list[b]).sum())
        for b in range(q):
            blk = (U_list[a] * Zt[:, b:b + 1]).sum(0)
            A[a, p + b * d:p + (b + 1) * d] = blk
            A[p + b * d:p + (b + 1) * d, a] = blk
    for a in range(q):
        for b in range(q):
            A[p + a * d:p + (a + 1) * d, p + b * d:p + (b + 1) * d] = (
                float((Zt[:, a] * Zt[:, b]).sum()) * np.eye(d))
    return A


def design_rows(U_list, Zt, d):
    """Returns the stacked design rows Z_i as an (n, d, k) array."""
    n, q = Zt.shape
    p = len(U_list)
    k = p + q * d
    Zi = np.zeros((n, d, k))
    for a in range(p):
        Zi[:, :, a] = U_list[a]
    eye = np.eye(d)
    for b in range(q):
        Zi[:, :, p + b * d:p + (b + 1) * d] = Zt[:, b, None, None] * eye
    return Zi


def sandwich(U_list, Zt, resid, kind="HC0", clusters=None, ridge=0.0):
    """Returns the heteroskedasticity-consistent covariance of beta.

    Args:
        U_list: List of p arrays of shape (n, d), the transported predictor
            tangents in intrinsic coordinates.
        Zt: Array of shape (n, q), the centered scalar covariates.
        resid: Array of shape (n, d), the tangent residuals.
        kind: "HC0" for the White sandwich or "HC3" for the leverage
            corrected version.
        clusters: Optional labels of length n. Scores are summed within a
            cluster before forming B, which allows dependence within
            clusters.
        ridge: Ridge added to the Gram matrix, matching the one used in the
            fit. A solver that uses (1/n) A + reg I corresponds to
            ridge = n reg here.

    Returns:
        The k x k covariance matrix.

    Raises:
        ValueError: If ``kind`` is not "HC0" or "HC3".
    """
    if kind not in ("HC0", "HC3"):
        raise ValueError(f"kind must be 'HC0' or 'HC3', got {kind!r}")
    n, d = resid.shape
    k = len(U_list) + Zt.shape[1] * d
    M = np.linalg.inv(gram(U_list, Zt, d) + ridge * np.eye(k))
    Zi = design_rows(U_list, Zt, d)
    r = resid
    if kind == "HC3":
        H = np.einsum('nak,kl,nbl->nab', Zi, M, Zi)
        r = np.linalg.solve(np.eye(d)[None] - H, resid[:, :, None])[:, :, 0]
    S = np.einsum('nak,na->nk', Zi, r)
    if clusters is None:
        B = S.T @ S
    else:
        lab = np.asarray(clusters)
        B = np.zeros((k, k))
        for g in np.unique(lab):
            sg = S[lab == g].sum(0)
            B += np.outer(sg, sg)
    return M @ B @ M


def wald_statistic(v, cov):
    """Returns v^T cov^-1 v, using a pseudo-inverse if ``cov`` is singular."""
    v = np.asarray(v, float).ravel()
    try:
        return float(v @ np.linalg.solve(cov, v))
    except np.linalg.LinAlgError:
        return float(v @ np.linalg.pinv(cov) @ v)


def pvalues(alphas, V, cov, p, q, d, n):
    """Returns the t-test p-values for alpha_j and the chi^2_d Wald p-values
    for v_k.

    Args:
        alphas: The p scalar coefficients.
        V: Array of shape (q, d), the tangent coefficients in intrinsic
            coordinates.
        cov: The k x k coefficient covariance.
        p: Number of manifold predictors.
        q: Number of scalar covariates.
        d: Intrinsic dimension of the response manifold.
        n: Number of observations.

    Returns:
        A pair of arrays of length p and q.
    """
    df = max(n * d - (p + q * d) - d, 1)
    p_alpha = np.array([
        2 * stats.t.sf(abs(float(alphas[j]) / np.sqrt(cov[j, j])), df=df)
        for j in range(p)])
    V = np.asarray(V, float).reshape(q, d)
    p_v = np.array([
        stats.chi2.sf(wald_statistic(V[k], cov[p + k * d:p + (k + 1) * d,
                                                p + k * d:p + (k + 1) * d]),
                      df=d)
        for k in range(q)])
    return p_alpha, p_v


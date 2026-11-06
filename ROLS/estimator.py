"""Riemannian ordinary least squares.

With every variable lifted to the tangent space at the Frechet mean of the
response, the model is

    Log_Ybar(Y_i) = sum_j alpha_j PT_{Xbar_j -> Ybar}(Log_Xbar_j(X_ij))
                    + sum_k (z_ik - zbar_k) v_k + eps_i.

Each manifold predictor X_j has one scalar coefficient alpha_j and each
scalar covariate z_k has one tangent vector coefficient v_k. The
coefficients solve the normal equations in the manifold's d intrinsic
coordinates.
"""

import numpy as np
from scipy import stats

from .robust import design_rows, gram, pvalues, sandwich, wald_statistic


class ROLSFit:
    """Fitted ROLS model.

    Attributes:
        manifold: The manifold of the response and the manifold predictors.
        Y_bar: Frechet mean of the response, the base point of the model.
        X_bars: Frechet means of the manifold predictors.
        alphas: The p scalar coefficients.
        V: Array of shape (q, d), the tangent coefficients in intrinsic
            coordinates at ``Y_bar``.
        W: Array of shape (n, d), the tangent responses.
        U: List of p arrays of shape (n, d), the transported predictor
            tangents.
        Zt: Array of shape (n, q), the centered scalar covariates.
        z_mean: Means of the scalar covariates.
        reg: Ridge parameter of the fit.
        n: Number of observations.
        d: Intrinsic dimension of the manifold.
        p: Number of manifold predictors.
        q: Number of scalar covariates.
        k: Number of coefficients, p + q d.
        fitted: Array of shape (n, d), the fitted tangent responses.
        resid: Array of shape (n, d), the tangent residuals.
        df_resid: Residual degrees of freedom, n d - k - d.
        sigma2: Residual variance estimate.
    """

    def __init__(self, manifold, Y_bar, X_bars, alphas, V, W, U, Zt, z_mean,
                 reg):
        self.manifold, self.Y_bar, self.X_bars = manifold, Y_bar, X_bars
        self.alphas, self.V = alphas, V
        self.W, self.U, self.Zt, self.z_mean, self.reg = W, U, Zt, z_mean, reg
        self.n, self.d = W.shape
        self.p, self.q = len(U), Zt.shape[1]
        self.k = self.p + self.q * self.d
        self.fitted = self._tangent_prediction(U, Zt)
        self.resid = W - self.fitted
        self.df_resid = max(self.n * self.d - self.k - self.d, 1)
        self.sigma2 = float((self.resid ** 2).sum() / self.df_resid)

    def _tangent_prediction(self, U, Zt):
        out = np.zeros((Zt.shape[0], self.d))
        for j, Uj in enumerate(U):
            out += self.alphas[j] * Uj
        return out + Zt @ self.V

    def bread(self, ridge=0.0):
        """Returns the Gram matrix A = sum_i Z_i^T Z_i plus ``ridge`` I."""
        return gram(self.U, self.Zt, self.d) + ridge * np.eye(self.k)

    def scores(self, resid=None):
        """Returns the (n, k) scores s_i = Z_i^T r_i."""
        r = self.resid if resid is None else resid
        return np.einsum('nak,na->nk', design_rows(self.U, self.Zt, self.d), r)

    def covariance(self, kind="model", clusters=None, ridge=None):
        """Returns the k x k covariance of the coefficients.

        Args:
            kind: "model" for the homoskedastic covariance sigma2 M A M with
                M = (A + ridge I)^-1, which is sigma2 A^-1 without a ridge;
                "HC0" for the White sandwich; "HC3" for the leverage
                corrected sandwich.
            clusters: Optional labels of length n for a cluster-robust
                sandwich.
            ridge: Ridge added to the Gram matrix. Defaults to the ridge of
                the fit, reg n.
        """
        ridge = self.reg * self.n if ridge is None else ridge
        if kind == "model":
            M = np.linalg.inv(self.bread(ridge))
            return self.sigma2 * M @ self.bread() @ M
        return sandwich(self.U, self.Zt, self.resid, kind, clusters, ridge)

    def pvalues(self, kind="model", clusters=None):
        """Returns the t-test p-values for alpha_j and the chi^2_d Wald
        p-values for v_k.

        Args:
            kind: Covariance estimator, see ``covariance``.
            clusters: Optional cluster labels, see ``covariance``.

        Returns:
            A pair of arrays of length p and q.
        """
        return pvalues(self.alphas, self.V, self.covariance(kind, clusters),
                       self.p, self.q, self.d, self.n)

    def joint_pvalue(self, k_start, k_end, kind="model", clusters=None):
        """Returns the p-value of the joint Wald test that
        v_k_start, ..., v_(k_end - 1) all vanish."""
        C = self.covariance(kind, clusters)
        sl = slice(self.p + k_start * self.d, self.p + k_end * self.d)
        vv = np.asarray(self.V, float)[k_start:k_end].ravel()
        w = wald_statistic(vv, C[sl, sl])
        return float(stats.chi2.sf(max(w, 0.0), df=(k_end - k_start) * self.d))

    def V_ambient(self):
        """Returns the tangent coefficients as ambient tangent vectors at
        ``Y_bar``, one per scalar covariate.

        ``V`` is expressed in the intrinsic frame at the fitted base point,
        which is not canonical, so coefficients from two fits are compared
        through their ambient vectors after parallel transport to a common
        base point.
        """
        return self.manifold.from_coords(self.Y_bar, np.asarray(self.V, float))

    def predict(self, X_list=(), z=None):
        """Returns the predicted responses for new predictors.

        Args:
            X_list: List of p batches of manifold predictors.
            z: Array of shape (n_new, q) of scalar covariates.

        Raises:
            ValueError: If a predictor is not on the manifold.
        """
        m = self.manifold
        for pts in X_list:
            m.check(pts)
        U = [m.coords(self.Y_bar, m.transport(self.X_bars[j], self.Y_bar,
                                              m.logmap(self.X_bars[j],
                                                       X_list[j])))
             for j in range(self.p)]
        if self.q:
            Zt = np.reshape(np.asarray(z, float), (-1, self.q)) - self.z_mean
        else:
            Zt = np.zeros((len(X_list[0]) if self.p else 0, 0))
        tang = self._tangent_prediction(U, Zt)
        return m.expmap(self.Y_bar, m.from_coords(self.Y_bar, tang))


def solve_normal_equations(U, Zt, W, d, ridge):
    """Returns the coefficient vector of the tangent model.

    Args:
        U: List of p arrays of shape (n, d), the transported predictor
            tangents.
        Zt: Array of shape (n, q), the centered scalar covariates.
        W: Array of shape (n, d), the tangent responses.
        d: Intrinsic dimension of the manifold.
        ridge: Ridge added to the Gram matrix.
    """
    p, q = len(U), Zt.shape[1]
    k = p + q * d
    b = np.zeros(k)
    for a in range(p):
        b[a] = float((U[a] * W).sum())
    for a in range(q):
        b[p + a * d:p + (a + 1) * d] = (Zt[:, a:a + 1] * W).sum(0)
    return np.linalg.solve(gram(U, Zt, d) + ridge * np.eye(k), b)


def fit(manifold, Y, X_list=(), z=None, reg=1e-6):
    """Fits ROLS.

    Args:
        manifold: The manifold of the response and the manifold predictors.
        Y: Batch of n response points.
        X_list: List of p batches of manifold predictors.
        z: Array of shape (n, q) of scalar covariates, centered internally.
        reg: Ridge parameter; reg n is added to the diagonal of the Gram
            matrix.

    Returns:
        A ROLSFit.

    Raises:
        ValueError: If a response or predictor is not on the manifold.
    """
    m = manifold
    Y = np.asarray(Y)
    X_list = [np.asarray(X) for X in X_list]
    for pts in [Y, *X_list]:
        m.check(pts)
    Y_bar = m.frechet_mean(Y)
    W = m.coords(Y_bar, m.logmap(Y_bar, Y))
    X_bars, U = [], []
    for X in X_list:
        Xb = m.frechet_mean(X)
        X_bars.append(Xb)
        U.append(m.coords(Y_bar, m.transport(Xb, Y_bar, m.logmap(Xb, X))))
    n, d = W.shape
    Z = (np.zeros((n, 0)) if z is None
         else np.reshape(np.asarray(z, float), (n, -1)))
    z_mean = Z.mean(0)
    Zt = Z - z_mean
    p, q = len(U), Zt.shape[1]
    theta = solve_normal_equations(U, Zt, W, d, reg * n)
    return ROLSFit(m, Y_bar, X_bars, theta[:p], theta[p:].reshape(q, d), W, U,
                   Zt, z_mean, reg)

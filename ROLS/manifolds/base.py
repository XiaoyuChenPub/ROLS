"""Manifold interface shared by all response and predictor spaces.

A manifold supplies the exponential and logarithmic maps, the geodesic
distance, parallel transport, an intrinsic Frechet mean, and a chart that
maps ambient tangent vectors to ``d`` intrinsic coordinates. The chart is
isometric: the Euclidean inner product of two coordinate vectors equals the
Riemannian inner product of the tangent vectors. All regression and inference
in this package runs in these coordinates, because a covariance built in a
larger ambient space is rank deficient.
"""

import warnings

import numpy as np


class Manifold:
    """Riemannian manifold with the operations ROLS needs.

    Points and tangent vectors are numpy arrays with a leading batch axis. A
    subclass implements ``logmap``, ``expmap``, ``dist`` and ``transport``,
    and either ``frame`` or a matching ``coords`` / ``from_coords`` pair.

    Attributes:
        d: Intrinsic dimension.
        name: Short name of the space, used in messages.
    """

    d: int
    name: str = "manifold"

    def logmap(self, base, points):
        """Returns the tangent vectors at ``base`` that reach ``points``."""
        raise NotImplementedError

    def expmap(self, base, tangents):
        """Returns the endpoints of the geodesics from ``base`` along
        ``tangents``."""
        raise NotImplementedError

    def dist(self, p, q):
        """Returns the geodesic distance between corresponding rows."""
        raise NotImplementedError

    def transport(self, src, dst, tangents):
        """Parallel transports ``tangents`` from ``src`` to ``dst`` along the
        geodesic."""
        raise NotImplementedError

    def check(self, points, tol=1e-6):
        """Checks that every point lies on the manifold.

        Args:
            points: Batch of points.
            tol: Largest accepted relative deviation from the manifold.

        Raises:
            ValueError: If a point deviates from the manifold by more than
                ``tol``.
        """
        dev = self._deviation(np.asarray(points, float))
        if dev is not None and np.nanmax(dev) > tol:
            raise ValueError(
                f"{self.name}: {int((dev > tol).sum())} of {len(dev)} points "
                f"are not on the manifold (largest deviation "
                f"{np.nanmax(dev):.2e})")

    def _deviation(self, points):
        """Returns each point's deviation from the manifold, or None."""
        return None

    def frechet_mean(self, points, weights=None, tol=1e-12, max_iter=500):
        """Computes the weighted Frechet mean by a damped Karcher iteration.

        Each step moves along the gradient g = sum_i w_i Log_mu(y_i). The step
        size is halved until the gradient norm decreases and grows back
        afterwards, which keeps the iteration stable in negative curvature
        where the unit step overshoots for widely spread data.

        Args:
            points: Batch of points.
            weights: Optional nonnegative weights, normalized to sum to one.
            tol: The iteration stops once the gradient norm is below ``tol``.
            max_iter: Maximum number of gradient steps.

        Returns:
            The mean as a single point. A RuntimeWarning is issued when the
            gradient norm does not reach ``tol``, which happens when the data
            have no unique mean.
        """
        pts = np.asarray(points)
        w = _normalize_weights(weights, len(pts))
        mu = self._mean_init(pts, w)
        g = _weighted_sum(self.logmap(mu, pts), w)
        gn = self._grad_norm(mu, g)
        t = 1.0
        for _ in range(max_iter):
            if gn < tol:
                return mu
            while True:
                cand = self.expmap(mu, (t * g)[None])[0]
                g_c = _weighted_sum(self.logmap(cand, pts), w)
                gn_c = self._grad_norm(cand, g_c)
                if gn_c < gn or t < 1e-8:
                    break
                t *= 0.5
            if gn_c >= gn:
                break
            mu, g, gn = cand, g_c, gn_c
            t = min(1.0, 2.0 * t)
        if gn >= tol:
            warnings.warn(
                f"{self.name}: Frechet mean stopped at |grad| = {gn:.1e} > "
                f"tol = {tol:.0e}; the data may have no unique mean",
                RuntimeWarning)
        return mu

    def _grad_norm(self, base, g):
        """Returns the Riemannian norm of one tangent vector at ``base``."""
        return float(np.linalg.norm(self.coords(base, np.asarray(g)[None])))

    def _mean_init(self, points, weights=None):
        """Returns the starting point of the Frechet mean iteration."""
        pts = np.asarray(points)
        return pts[0] if weights is None else pts[int(np.argmax(weights))]

    def tangent_norm(self, base, tangent):
        """Returns the largest Riemannian norm over a batch of tangent
        vectors."""
        return float(np.linalg.norm(self.coords(base, tangent), axis=1).max())

    def frame(self, base):
        """Returns a (D, d) orthonormal basis of the tangent space at
        ``base``."""
        raise NotImplementedError

    def coords(self, base, tangents):
        """Maps ambient tangent vectors to (n, d) intrinsic coordinates."""
        return np.atleast_2d(np.asarray(tangents, float)) @ self.frame(base)

    def from_coords(self, base, coords):
        """Maps (n, d) intrinsic coordinates to ambient tangent vectors."""
        return np.atleast_2d(np.asarray(coords, float)) @ self.frame(base).T

    def project(self, points):
        """Returns the nearest points on the manifold."""
        return np.asarray(points, float)


def _normalize_weights(weights, n):
    """Returns weights that sum to one, or None for uniform weights.

    Raises:
        ValueError: If a weight is negative or the weights sum to zero.
    """
    if weights is None:
        return None
    w = np.asarray(weights, float).ravel()
    if len(w) != n or (w < 0).any() or w.sum() <= 0:
        raise ValueError(
            f"weights must be {n} nonnegative numbers with a positive sum")
    return w / w.sum()


def _weighted_sum(tangents, w):
    """Averages tangent rows of any trailing shape, weighted by ``w``."""
    T = np.asarray(tangents, float)
    return T.mean(0) if w is None else np.tensordot(w, T, axes=1)


def frechet_rmse(manifold, Y_true, Y_pred):
    """Returns the root mean squared geodesic distance between two batches."""
    return float(np.sqrt((manifold.dist(Y_true, Y_pred) ** 2).mean()))

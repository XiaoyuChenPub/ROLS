"""Unit sphere S^{D-1} in R^D with the round metric."""

import numpy as np

from .base import Manifold


class Sphere(Manifold):
    """Unit sphere. Points are unit vectors in R^D.

    Attributes:
        D: Ambient dimension.
        d: Intrinsic dimension, D - 1.
    """

    def __init__(self, ambient=3):
        self.D = ambient
        self.d = ambient - 1
        self.name = f"S^{self.d}"

    def _normalize(self, P):
        P = np.atleast_2d(np.asarray(P, float))
        return P / np.linalg.norm(P, axis=1, keepdims=True)

    def logmap(self, base, points):
        b = np.asarray(base, float).ravel()
        P = np.atleast_2d(np.asarray(points, float))
        c = P @ b
        V = P - c[:, None] * b[None, :]
        nv = np.linalg.norm(V, axis=1)
        theta = np.arctan2(nv, c)
        scale = np.where(nv > 0, theta / np.where(nv > 0, nv, 1.0), 1.0)
        return V * scale[:, None]

    def expmap(self, base, tangents):
        b = np.asarray(base, float).ravel()
        V = np.atleast_2d(np.asarray(tangents, float))
        nv = np.linalg.norm(V, axis=1, keepdims=True)
        out = np.cos(nv) * b[None, :] + np.sinc(nv / np.pi) * V
        return self._normalize(out)

    def dist(self, p, q):
        P = np.atleast_2d(np.asarray(p, float))
        Q = np.atleast_2d(np.asarray(q, float))
        c = (P * Q).sum(1)
        return np.arctan2(np.linalg.norm(P - c[:, None] * Q, axis=1), c)

    def transport(self, src, dst, tangents):
        """Parallel transports along the minimizing geodesic.

        Uses V - <t, V> / (1 + <s, t>) (s + t). At the antipode of ``src`` the
        geodesic is not unique and ``tangents`` are returned unchanged.
        """
        s = np.asarray(src, float).ravel()
        t = np.asarray(dst, float).ravel()
        V = np.atleast_2d(np.asarray(tangents, float))
        den = 1.0 + float(s @ t)
        if den < 1e-12:
            return V
        return V - np.outer(V @ t / den, s + t)

    def _deviation(self, points):
        return np.abs(np.linalg.norm(np.atleast_2d(points), axis=1) - 1.0)

    def _mean_init(self, points, weights=None):
        P = np.asarray(points, float)
        m = P.mean(0) if weights is None else weights @ P
        nm = np.linalg.norm(m)
        return P[0] if nm < 1e-12 else m / nm

    def frame(self, base):
        """Returns an orthonormal basis of the tangent space b^perp.

        The QR factorization of [b, I] places b in the first column, so the
        remaining D - 1 columns span b^perp at every base point.
        """
        b = np.asarray(base, float).ravel()
        b = b / np.linalg.norm(b)
        Q, _ = np.linalg.qr(np.column_stack([b, np.eye(self.D)]))
        return Q[:, 1:self.D]

    def project(self, points):
        return self._normalize(points)

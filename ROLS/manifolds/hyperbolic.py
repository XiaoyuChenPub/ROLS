"""Hyperbolic space H^d in the Lorentz (hyperboloid) model.

Points x in R^{d+1} satisfy <x, x>_L = -1 and x_0 > 0 for the Minkowski
product <x, y>_L = -x_0 y_0 + x_1 y_1 + ... + x_d y_d. Log, exp and parallel
transport are closed form in this model.
"""

import numpy as np

from .base import Manifold


def _mink(x, y):
    x, y = np.atleast_2d(x), np.atleast_2d(y)
    return -x[:, 0] * y[:, 0] + (x[:, 1:] * y[:, 1:]).sum(1)


class Hyperbolic(Manifold):
    """Hyperbolic space of curvature -1 in the Lorentz model.

    Attributes:
        d: Intrinsic dimension.
        D: Ambient dimension, d + 1.
    """

    def __init__(self, d=3):
        self.d = d
        self.D = d + 1
        self.name = f"H^{d}"

    @staticmethod
    def from_spatial(v):
        """Lifts spatial coordinates (x_1, ..., x_d) onto the hyperboloid."""
        v = np.atleast_2d(np.asarray(v, float))
        x0 = np.sqrt(1.0 + (v ** 2).sum(1, keepdims=True))
        return np.concatenate([x0, v], axis=1)

    @staticmethod
    def from_ball(b):
        """Maps Poincare ball points to the Lorentz model,
        x = (1 + |b|^2, 2b) / (1 - |b|^2)."""
        b = np.atleast_2d(np.asarray(b, float))
        n2 = (b ** 2).sum(1, keepdims=True)
        return np.concatenate([1.0 + n2, 2.0 * b], axis=1) / (1.0 - n2)

    @staticmethod
    def _dist_and_cosh(P, Q):
        """Returns d(p, q) and cosh d(p, q) from the difference vector.

        cosh d = -<p, q>_L = 1 + |p - q|_L^2 / 2, so d = 2 asinh(|p - q|_L / 2)
        stays accurate at small separations.
        """
        s2 = np.clip(_mink(P - Q, P - Q), 0.0, None)
        return 2.0 * np.arcsinh(np.sqrt(s2) / 2.0), 1.0 + s2 / 2.0

    def logmap(self, base, points):
        b = np.asarray(base, float).ravel()
        P = np.atleast_2d(np.asarray(points, float))
        dist, beta = self._dist_and_cosh(P, np.tile(b, (len(P), 1)))
        safe = np.where(dist > 0, dist, 1.0)
        scale = np.where(dist > 0, dist / np.sinh(safe), 1.0)
        return scale[:, None] * (P - beta[:, None] * b[None, :])

    def expmap(self, base, tangents):
        b = np.asarray(base, float).ravel()
        V = np.atleast_2d(np.asarray(tangents, float))
        nrm = np.sqrt(np.clip(_mink(V, V), 0.0, None))
        safe = np.where(nrm > 0, nrm, 1.0)
        sinhc = np.where(nrm > 0, np.sinh(safe) / safe, 1.0)
        out = np.cosh(nrm)[:, None] * b[None, :] + sinhc[:, None] * V
        return self.from_spatial(out[:, 1:])

    def dist(self, p, q):
        P = np.atleast_2d(np.asarray(p, float))
        Q = np.atleast_2d(np.asarray(q, float))
        Q = np.broadcast_to(Q, P.shape) if len(Q) == 1 else Q
        return self._dist_and_cosh(P, Q)[0]

    def transport(self, src, dst, tangents):
        """Parallel transports along the geodesic,
        V + <t, V>_L / (1 - <s, t>_L) (s + t)."""
        s = np.asarray(src, float).ravel()
        t = np.asarray(dst, float).ravel()
        V = np.atleast_2d(np.asarray(tangents, float))
        coef = _mink(V, np.tile(t, (len(V), 1))) / (1.0 - float(_mink(s, t)[0]))
        return V + coef[:, None] * (s + t)[None, :]

    def _deviation(self, points):
        P = np.atleast_2d(points)
        scale = np.maximum(P[:, 0] ** 2, 1.0)
        return np.abs(_mink(P, P) + 1.0) / scale + np.clip(-P[:, 0], 0.0, None)

    def _mean_init(self, points, weights=None):
        P = np.asarray(points, float)
        m = P.mean(0) if weights is None else weights @ P
        return self.from_spatial(m[1:])[0]

    def coords(self, base, tangents):
        """Returns Minkowski-orthonormal coordinates c_j = <V, e_j>_L."""
        V = np.atleast_2d(np.asarray(tangents, float))
        G = np.ones(self.D)
        G[0] = -1.0
        return (V * G) @ self.frame(base)

    def frame(self, base):
        """Returns a Minkowski-orthonormal basis of the tangent space by
        Gram-Schmidt on the projected spatial axes."""
        b = np.asarray(base, float).ravel()
        vecs = []
        for c in np.eye(self.D)[1:]:
            v = c + _mink(c[None], b[None])[0] * b
            for u in vecs:
                v = v - _mink(v[None], u[None])[0] * u
            vecs.append(v / np.sqrt(_mink(v[None], v[None])[0]))
        return np.stack(vecs, axis=1)

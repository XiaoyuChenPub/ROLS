"""Flat torus T^k = (S^1)^k in angle coordinates."""

import numpy as np

from .base import Manifold, _normalize_weights


def wrap(x):
    """Wraps angles to [-pi, pi)."""
    return (np.asarray(x, float) + np.pi) % (2 * np.pi) - np.pi


class Torus(Manifold):
    """Flat torus. Points are k-vectors of angles; parallel transport is the
    identity.

    Attributes:
        d: Intrinsic dimension, k.
    """

    def __init__(self, k=2):
        self.d = self.D = k
        self.name = f"T^{k}"

    def logmap(self, base, points):
        P = np.atleast_2d(np.asarray(points, float))
        return wrap(P - np.asarray(base, float).ravel())

    def expmap(self, base, tangents):
        V = np.atleast_2d(np.asarray(tangents, float))
        return wrap(np.asarray(base, float).ravel() + V)

    def dist(self, p, q):
        return np.linalg.norm(wrap(np.atleast_2d(p) - np.atleast_2d(q)), axis=1)

    def transport(self, src, dst, tangents):
        return np.atleast_2d(np.asarray(tangents, float))

    def frechet_mean(self, points, weights=None, tol=1e-13, max_iter=200):
        """Computes the Frechet mean, starting from the circular mean.

        The circular mean is only a starting point; the returned mean is a
        stationary point of the weighted sum of squared angular distances.
        """
        P = np.atleast_2d(np.asarray(points, float))
        w = _normalize_weights(weights, len(P))
        w = np.full(len(P), 1.0 / len(P)) if w is None else w
        mu = np.arctan2(w @ np.sin(P), w @ np.cos(P))
        for _ in range(max_iter):
            g = w @ wrap(P - mu)
            if np.linalg.norm(g) < tol:
                break
            mu = wrap(mu + g)
        return mu

    def frame(self, base):
        return np.eye(self.d)

    def project(self, points):
        return wrap(points)

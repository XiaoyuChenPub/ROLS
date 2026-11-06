"""Symmetric positive definite matrices with the affine-invariant metric.

Intrinsic coordinates whiten a tangent matrix by chol(base)^-1, so the
Euclidean inner product of coordinates equals the affine-invariant inner
product at the base point. Plain Frobenius coordinates are not isometric
unless the base is a multiple of the identity.
"""

import numpy as np

from .base import Manifold, _weighted_sum

_SQRT2 = np.sqrt(2.0)


def _sym(A):
    return 0.5 * (A + np.swapaxes(A, -1, -2))


def _powm(A, f, positive=True):
    """Applies ``f`` to the eigenvalues of a symmetric matrix.

    Args:
        A: Symmetric matrix or batch of matrices.
        f: Function applied to the eigenvalues.
        positive: Whether to clip the eigenvalues away from zero first, as
            required for logarithms and fractional powers. Must be False for
            the exponential, whose argument is an arbitrary symmetric matrix.
    """
    w, V = np.linalg.eigh(_sym(A))
    if positive:
        w = np.clip(w, 1e-300, None)
    return (V * f(w)[..., None, :]) @ np.swapaxes(V, -1, -2)


class SPD(Manifold):
    """Sym+(k), the k x k symmetric positive definite matrices.

    Points and tangent vectors are (n, k, k) arrays.

    Attributes:
        k: Matrix size.
        d: Intrinsic dimension, k (k + 1) / 2.
    """

    def __init__(self, k=3):
        self.k = k
        self.d = k * (k + 1) // 2
        self.name = f"Sym+({k})"
        self._iu = np.triu_indices(k)
        self._off = self._iu[0] != self._iu[1]

    def _vech(self, S):
        """Half-vectorizes with sqrt(2) on off-diagonal entries."""
        V = S[..., self._iu[0], self._iu[1]].copy()
        V[..., self._off] *= _SQRT2
        return V

    def _unvech(self, v):
        v = np.atleast_2d(np.asarray(v, float)).copy()
        v[..., self._off] /= _SQRT2
        S = np.zeros(v.shape[:-1] + (self.k, self.k))
        S[..., self._iu[0], self._iu[1]] = v
        S[..., self._iu[1], self._iu[0]] = v
        return S

    def logmap(self, base, points):
        B = np.asarray(base, float)
        Bi = _powm(B, lambda w: w ** -0.5)
        Bh = _powm(B, lambda w: w ** 0.5)
        P = np.asarray(points, float)
        P = P[None] if P.ndim == 2 else P
        return Bh @ _powm(Bi @ P @ Bi, np.log) @ Bh

    def expmap(self, base, tangents):
        B = np.asarray(base, float)
        Bi = _powm(B, lambda w: w ** -0.5)
        Bh = _powm(B, lambda w: w ** 0.5)
        V = np.asarray(tangents, float)
        V = V[None] if V.ndim == 2 else V
        return Bh @ _powm(Bi @ V @ Bi, np.exp, positive=False) @ Bh

    def dist(self, p, q):
        P, Q = np.asarray(p, float), np.asarray(q, float)
        P = P[None] if P.ndim == 2 else P
        Q = Q[None] if Q.ndim == 2 else Q
        Pi = _powm(P, lambda w: w ** -0.5)
        w = np.linalg.eigvalsh(_sym(Pi @ Q @ Pi))
        return np.sqrt((np.log(np.clip(w, 1e-300, None)) ** 2).sum(-1))

    def transport(self, src, dst, tangents):
        """Parallel transports along the geodesic, V -> E V E^T.

        E = (D S^-1)^{1/2} is computed as S^{1/2} (S^{-1/2} D S^{-1/2})^{1/2}
        S^{-1/2}, the principal square root of the nonsymmetric matrix D S^-1.
        """
        S, D = np.asarray(src, float), np.asarray(dst, float)
        Sh, Si = _powm(S, np.sqrt), _powm(S, lambda w: w ** -0.5)
        E = Sh @ _powm(Si @ D @ Si, np.sqrt) @ Si
        V = np.asarray(tangents, float)
        V = V[None] if V.ndim == 2 else V
        return _sym(E @ V @ np.swapaxes(E, -1, -2))

    def _deviation(self, points):
        P = points[None] if points.ndim == 2 else points
        scale = np.maximum(np.abs(P).max((1, 2)), 1e-300)
        asym = np.abs(P - np.swapaxes(P, -1, -2)).max((1, 2)) / scale
        not_pd = np.linalg.eigvalsh(_sym(P)).min(1) <= 0
        return asym + not_pd

    def _mean_init(self, points, weights=None):
        return _sym(_weighted_sum(np.asarray(points, float), weights))

    def coords(self, base, tangents):
        """Returns vech(L^-1 V L^-T) with L = chol(base) and sqrt(2) on
        off-diagonal entries."""
        L = np.linalg.cholesky(np.asarray(base, float))
        Li = np.linalg.inv(L)
        V = np.asarray(tangents, float)
        V = V[None] if V.ndim == 2 else V
        return self._vech(_sym(Li @ V @ Li.T))

    def from_coords(self, base, coords):
        L = np.linalg.cholesky(np.asarray(base, float))
        return _sym(L @ self._unvech(coords) @ L.T)

    def frame(self, base):
        raise NotImplementedError(
            "SPD implements coords and from_coords directly")

    def project(self, points):
        P = np.asarray(points, float)
        P = P.reshape(-1, self.k, self.k) if P.ndim != 3 else P
        return _powm(_sym(P), lambda w: np.clip(w, 1e-10, None))

"""Rotation group SO(3) with the bi-invariant metric."""

import numpy as np

from .base import Manifold, _weighted_sum


def _hat(w):
    """Returns the skew-symmetric matrices of a batch of 3-vectors."""
    w = np.atleast_2d(np.asarray(w, float))
    K = np.zeros((len(w), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -w[:, 2], w[:, 1]
    K[:, 1, 0], K[:, 1, 2] = w[:, 2], -w[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -w[:, 1], w[:, 0]
    return K


def _quat(R):
    """Returns unit quaternions (w, x, y, z), w >= 0, by Shepperd's method.

    The pivot is the largest of tr R and the diagonal entries, so the square
    root never has a small argument and the result is accurate at every
    rotation angle.
    """
    tr = np.trace(R, axis1=1, axis2=2)
    diag = np.diagonal(R, axis1=1, axis2=2)
    case = np.argmax(np.column_stack([tr, diag]), axis=1)
    q = np.empty((len(R), 4))
    for k in range(4):
        i = case == k
        if not i.any():
            continue
        M = R[i]
        if k == 0:
            s = 2.0 * np.sqrt(1.0 + tr[i])
            q[i, 0] = s / 4
            q[i, 1] = (M[:, 2, 1] - M[:, 1, 2]) / s
            q[i, 2] = (M[:, 0, 2] - M[:, 2, 0]) / s
            q[i, 3] = (M[:, 1, 0] - M[:, 0, 1]) / s
        else:
            a, b, c = k - 1, k % 3, (k + 1) % 3
            s = 2.0 * np.sqrt(np.clip(
                1.0 + M[:, a, a] - M[:, b, b] - M[:, c, c], 0.0, None))
            q[i, 0] = (M[:, c, b] - M[:, b, c]) / s
            q[i, 1 + a] = s / 4
            q[i, 1 + b] = (M[:, a, b] + M[:, b, a]) / s
            q[i, 1 + c] = (M[:, a, c] + M[:, c, a]) / s
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q * np.where(q[:, :1] < 0, -1.0, 1.0)


class SO3(Manifold):
    """Rotation group. Points are (n, 3, 3) rotation matrices.

    Tangent vectors are body-frame coordinates, Log_R(S) = vee(log(R^T S)),
    whose Euclidean norm is the rotation angle.

    Attributes:
        d: Intrinsic dimension, 3.
    """

    d = 3
    D = 3
    name = "SO(3)"

    def _log_identity(self, R):
        """Returns vee(log R) with rotation angle in [0, pi]."""
        R = np.asarray(R, float)
        R = R[None] if R.ndim == 2 else R
        q = _quat(R)
        nv = np.linalg.norm(q[:, 1:], axis=1)
        theta = 2.0 * np.arctan2(nv, q[:, 0])
        scale = np.where(nv > 0, theta / np.where(nv > 0, nv, 1.0), 2.0)
        return q[:, 1:] * scale[:, None]

    def logmap(self, base, points):
        B = np.asarray(base, float)
        P = np.asarray(points, float)
        P = P[None] if P.ndim == 2 else P
        return self._log_identity(np.einsum('ij,njk->nik', B.T, P))

    def expmap(self, base, tangents):
        B = np.asarray(base, float)
        w = np.atleast_2d(np.asarray(tangents, float))
        theta = np.linalg.norm(w, axis=1)[:, None, None]
        K = _hat(w)
        a = np.sinc(theta / np.pi)
        b = 0.5 * np.sinc(theta / (2 * np.pi)) ** 2
        R = np.eye(3)[None] + a * K + b * (K @ K)
        return np.einsum('ij,njk->nik', B, R)

    def dist(self, p, q):
        P, Q = np.asarray(p, float), np.asarray(q, float)
        P = P[None] if P.ndim == 2 else P
        Q = Q[None] if Q.ndim == 2 else Q
        return np.linalg.norm(self._log_identity(np.swapaxes(P, 1, 2) @ Q),
                              axis=1)

    def transport(self, src, dst, tangents):
        """Parallel transports body-frame coordinates along the geodesic.

        With the bi-invariant metric the transport from R_src to R_dst is the
        rotation w -> exp(-xi / 2) w, xi = Log(R_src^T R_dst).
        """
        S, D = np.asarray(src, float), np.asarray(dst, float)
        xi = self._log_identity(S.T @ D)[0]
        Q = self.expmap(np.eye(3), -0.5 * xi[None])[0]
        return np.atleast_2d(np.asarray(tangents, float)) @ Q.T

    def frechet_mean(self, points, weights=None, tol=1e-11, max_iter=500):
        """Computes the Frechet mean; see ``Manifold.frechet_mean``.

        The default tolerance is above the rounding floor of the log map,
        which is about 1e-12 for a batch of rotation matrices.
        """
        return super().frechet_mean(points, weights, tol=tol,
                                    max_iter=max_iter)

    def _deviation(self, points):
        R = points[None] if points.ndim == 2 else points
        orth = np.abs(np.swapaxes(R, 1, 2) @ R - np.eye(3)).max((1, 2))
        return orth + np.clip(-np.linalg.det(R), 0.0, None)

    def _mean_init(self, points, weights=None):
        """Projects the weighted arithmetic mean onto SO(3)."""
        mean = _weighted_sum(np.asarray(points, float), weights)
        return self.project(mean)[0]

    def frame(self, base):
        return np.eye(3)

    def project(self, points):
        P = np.asarray(points, float).reshape(-1, 3, 3)
        U, _, Vt = np.linalg.svd(P)
        R = U @ Vt
        bad = np.linalg.det(R) < 0
        if bad.any():
            U[bad, :, -1] *= -1
            R = U @ Vt
        return R

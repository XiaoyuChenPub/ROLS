"""Poincare-ball operations for the H^3 data-generating processes.

Data are generated in the Poincare ball of curvature -1 and mapped to the
Lorentz model, where ROLS fits, with ``to_lorentz`` and ``push_tangent``. The
formulas follow the Mobius gyrovector calculus (Ganea et al., 2018).
"""

import numpy as np

from ROLS.manifolds import Hyperbolic


def _dot(a, b):
    """Returns the inner product over the last axis, keeping the axis."""
    return (a * b).sum(-1, keepdims=True)


def lam(x):
    """Returns the conformal factor 2 / (1 - |x|^2)."""
    return 2.0 / np.clip(1.0 - _dot(x, x), 1e-15, None)


def mobius_add(x, y):
    """Returns the Mobius sum of ``x`` and ``y``."""
    xy, x2, y2 = _dot(x, y), _dot(x, x), _dot(y, y)
    num = (1 + 2 * xy + y2) * x + (1 - x2) * y
    return num / np.clip(1 + 2 * xy + x2 * y2, 1e-15, None)


def expmap(x, u):
    """Returns the exponential map at ``x`` of the ball tangents ``u``."""
    x, u = np.broadcast_arrays(np.atleast_2d(x), np.atleast_2d(u))
    nu = np.sqrt(np.clip(_dot(u, u), 1e-30, None))
    return mobius_add(x, np.tanh(lam(x) * nu / 2.0) * u / nu)


def logmap(x, y):
    """Returns the logarithmic map at ``x`` of the ball points ``y``."""
    x, y = np.broadcast_arrays(np.atleast_2d(x), np.atleast_2d(y))
    s = mobius_add(-x, y)
    ns = np.sqrt(np.clip(_dot(s, s), 1e-30, None))
    return 2.0 / lam(x) * np.arctanh(np.clip(ns, 0.0, 1 - 1e-15)) * s / ns


def _gyration(u, v, w):
    """Returns the gyration gyr[u, v] w."""
    u2, v2 = _dot(u, u), _dot(v, v)
    uv, uw, vw = _dot(u, v), _dot(u, w), _dot(v, w)
    a = -uw * v2 + vw + 2 * uv * vw
    b = -vw * u2 - uw
    d = 1 + 2 * uv + u2 * v2
    return w + 2 * (a * u + b * v) / np.clip(d, 1e-15, None)


def transport(x, y, v):
    """Returns the parallel transport of ``v`` from T_x to T_y."""
    x, y, v = np.broadcast_arrays(np.atleast_2d(x), np.atleast_2d(y),
                                  np.atleast_2d(v))
    return _gyration(y, -x, v) * lam(x) / lam(y)


def to_lorentz(b):
    """Maps ball points to the hyperboloid, (1 + |b|^2, 2b) / (1 - |b|^2)."""
    return Hyperbolic.from_ball(b)


def push_tangent(b, v):
    """Applies the differential of ``to_lorentz`` at ``b`` to tangents ``v``."""
    b, v = np.broadcast_arrays(np.atleast_2d(b), np.atleast_2d(v))
    s = 1.0 - _dot(b, b)
    bv = _dot(b, v)
    return np.concatenate([4 * bv / s ** 2, 2 * v / s + 4 * b * bv / s ** 2],
                          axis=1)

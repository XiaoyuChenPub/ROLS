"""Benchmarks prediction across five manifolds.

Produces Table 5 and Figures 4 and 2(d) of the paper, together with the
tangent-linear-ridge ablation reported in its Section 5.

Settings:

    manifolds     S^2, T^2, SO(3), Sym+(3), H^3
    default cell  (n, p, q, tau) = (200, 2, 3, 0.15)
    sweeps        tau in {0.05, 0.15, 0.30}, q in {0, 3, 6},
                  n in {100, 200, 500}, p in {1, 2, 4}, one at a time, so 9
                  distinct cells per manifold
    replications  100 per cell, each with a fresh training set of size n and a
                  held-out test set of 100 observations
    coefficients  alpha = (0.5, -0.3, 0.2, -0.15)[:p], ||v_k|| = 0.2,
                  z ~ N(0, I_q)
    metrics       test Frechet RMSE for every method; for ROLS also
                  |alpha_hat - alpha|, ||v_hat - v|| and model-based rejection
                  rates at level 0.05

Trial t uses the same seeds in every cell. The Sym+(3) predictor centers and the
true v_k have fixed seeds, so ``--seed`` changes only the Monte Carlo streams.

Methods, implemented in ``ROLS.baselines``:

    Fr. mean   Frechet mean of the training responses.
    Geodesic   Tangent-linear regression on z at the Frechet mean of the
               response, which is ROLS with no manifold predictor and extends
               the geodesic regression of Fletcher (2013) to several scalars.
    Fr. reg.   Global Frechet regression on z (Petersen and Mueller, 2019), a
               weighted Frechet mean with weights that may be negative.
    SI-Fr.     Single-index Frechet regression on z (Bhattacharjee and Mueller,
               2023). The direction minimizes the in-sample tangent loss over a
               random grid of 500, with a 5-fold cross-validated bandwidth, and
               the prediction is a local linear weighted Frechet mean on the
               index.
    NL-Fr.     Nonlinear Frechet regression on X and z (Bhattacharjee, Li and
               Xue, 2025), a weighted Frechet mean with weights from a Gaussian
               kernel on the predictors' log-map coordinates and z. The Tikhonov
               constant minimizes generalized cross-validation.
    ROLS (m)   ROLS with the scalar block dropped.
    ROLS       The proposed estimator.
    Tan. lin.  Ablation that replaces each scalar alpha_j of ROLS with a full
               d x d map A_j, fit by ridge regression with constant 1e-3.

Usage, from the repository root:

    python -m simulations.prediction [--quick] [--trials N] [--workers K]
                                     [--seed S]
"""

import numpy as np

from ROLS import SO3, SPD, Hyperbolic, Sphere, Torus, fit, frechet_rmse
from ROLS import baselines as B
from simulations import poincare
from simulations._common import (Timer, figure_path, load_results, parse_args,
                                 pmap, pyplot, rng_for, save_results,
                                 save_text)

DEFAULT = {"n": 200, "p": 2, "q": 3, "tau": 0.15}
GRID = {"tau": [0.05, 0.15, 0.30], "q": [0, 3, 6], "n": [100, 200, 500],
        "p": [1, 2, 4]}
N_TEST = 100
TRIALS = 100
LEVEL = 0.05
ALPHA_POOL = np.array([0.5, -0.3, 0.2, -0.15])
V_NORM = 0.2
Q_MAX = max(GRID["q"])

# These seeds fix the random Sym+(3) predictor centers and the directions of
# the true v_k, which are part of the design, not of the Monte Carlo error.
_CENTER_SEED = 49
_V_SEED = 55

MANIFOLDS = [("S2", r"$\mathbb{S}^2$"), ("T2", r"$\mathbb{T}^2$"),
             ("SO3", r"$\mathrm{SO}(3)$"), ("SPD3", r"$\mathrm{Sym}^+(3)$"),
             ("H3", r"$\mathbb{H}^3$")]
# (key, label, color, marker), in the order of table rows and figure legend.
METHODS = [("frechet_mean", "Fr. mean", "#BBBBBB", "o"),
           ("geodesic", "Geodesic", "#1f77b4", "s"),
           ("frechet_reg", "Fr. reg.", "#ff7f0e", "D"),
           ("sifr", "SI-Fr.", "#2ca02c", "^"),
           ("nlfr", "NL-Fr.", "#8c564b", "P"),
           ("rols_m", "ROLS (m)", "#17becf", "X"),
           ("rols", "ROLS", "#d62728", "*")]
ABLATION = "tangent_linear"


def cells():
    """Returns the 9 distinct (key, config) cells, the default one first."""
    out = [("default", dict(DEFAULT))]
    for name, values in GRID.items():
        for val in values:
            if val != DEFAULT[name]:
                cfg = dict(DEFAULT)
                cfg[name] = val
                out.append((cell_key(name, val), cfg))
    return out


def cell_key(name, value):
    """Returns the key of the cell that sets ``name`` to ``value``."""
    return "default" if value == DEFAULT[name] else f"{name}={value:g}"


def _vec_to_sym(v):
    """Maps upper-triangular 6-vectors to symmetric matrices.

    Diagonal entries come from the diagonal of v and each off-diagonal entry
    appears once in v, so ||v|| = 0.2 is not the Frobenius norm of the matrix.

    Args:
        v: Array of shape (..., 6).

    Returns:
        Array of shape (..., 3, 3).
    """
    v = np.asarray(v, float)
    iu = np.triu_indices(3)
    S = np.zeros(v.shape[:-1] + (3, 3))
    S[..., iu[0], iu[1]] = v
    S = S + np.swapaxes(S, -1, -2)
    S[..., np.arange(3), np.arange(3)] /= 2.0
    return S


def _true_v(dim, base=None):
    """Draws Q_MAX tangent coefficients of norm 0.2 in random directions.

    Args:
        dim: Storage dimension of a coefficient, 3 for S^2 and the Poincare
            ball and 6 for the upper triangle of Sym+(3).
        base: On S^2, the response mean. Each draw is projected onto its
            tangent plane and normalized again.

    Returns:
        Array of shape (Q_MAX, dim).
    """
    rng = np.random.default_rng(_V_SEED)
    out = []
    for _ in range(Q_MAX):
        v = rng.standard_normal(dim)
        v = v / (np.linalg.norm(v) + 1e-12) * V_NORM
        if base is not None:
            v = v - (v @ base) * base
            v = v / (np.linalg.norm(v) + 1e-12) * V_NORM
        out.append(v)
    return np.array(out)


class _DGP:
    """Model (1) at a fixed response mean with fixed predictor centers.

    Y_i = Exp_Ybar(sum_j alpha_j PT_{Xbar_j -> Ybar} Log_{Xbar_j}(X_ij)
                   + sum_k z_ik v_k + eps_i).

    Subclasses set ``m``, ``Y_bar``, ``centers`` (used cyclically) and
    ``v_star`` (true v_k as ambient tangent vectors at ``Y_bar``), and supply
    ``draw`` and ``noise``.
    """
    name = ""

    def draw(self, center, n, rng):
        """Draws n predictor points around ``center``."""
        raise NotImplementedError

    def noise(self, n, tau, rng):
        """Draws n ambient tangent noise vectors at ``Y_bar``."""
        raise NotImplementedError

    def sample(self, n, p, q, tau, rng):
        """Draws one data set.

        Args:
            n: Sample size.
            p: Number of manifold predictors.
            q: Number of scalar covariates.
            tau: Noise scale.
            rng: ``numpy`` generator.

        Returns:
            A tuple (Y, X, z), with X a list of p batches.
        """
        m = self.m
        z = rng.standard_normal((n, q))
        X = [self.draw(self.centers[j % len(self.centers)], n, rng)
             for j in range(p)]
        xi = (np.tensordot(z, self.v_star[:q], axes=1)
              + self.noise(n, tau, rng))
        for j in range(p):
            c = self.centers[j % len(self.centers)]
            xi = xi + ALPHA_POOL[j] * m.transport(c, self.Y_bar,
                                                  m.logmap(c, X[j]))
        return m.expmap(self.Y_bar, xi), X, z


class SphereDGP(_DGP):
    """S^2, Ybar = e3, equatorial centers, spread 0.6 |N(0,1)|."""
    name = "S2"

    def __init__(self):
        self.m = Sphere(3)
        self.Y_bar = np.array([0.0, 0.0, 1.0])
        c = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.7071, 0.7071, 0.0],
                      [-0.7071, 0.7071, 0.0]])
        self.centers = c / np.linalg.norm(c, axis=1, keepdims=True)
        self.v_star = _true_v(3, self.Y_bar)

    def draw(self, center, n, rng):
        """Draws predictor points in a uniform direction around ``center``."""
        v = rng.standard_normal((n, 3))
        v = v - np.outer(v @ center, center)
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
        return self.m.expmap(
            center, v * (0.6 * np.abs(rng.standard_normal(n)))[:, None])

    def noise(self, n, tau, rng):
        """Draws isotropic noise in the tangent plane at ``Y_bar``."""
        return self.m.from_coords(self.Y_bar, tau * rng.standard_normal((n, 2)))


class TorusDGP(_DGP):
    """Flat T^2, Ybar = (5.0, 1.2), Gaussian predictor spread 0.45 per angle."""
    name = "T2"

    def __init__(self):
        self.m = Torus(2)
        self.Y_bar = np.array([5.0, 1.2])
        self.centers = np.array([[4.8, 1.0], [5.4, 1.6], [4.4, 1.8],
                                 [5.8, 0.8]])
        self.v_star = _true_v(2)

    def draw(self, center, n, rng):
        """Draws Gaussian predictor points around ``center``."""
        return self.m.expmap(center, 0.45 * rng.standard_normal((n, 2)))

    def noise(self, n, tau, rng):
        """Draws isotropic Gaussian noise."""
        return tau * rng.standard_normal((n, 2))


class RotationDGP(_DGP):
    """SO(3), Ybar = I, centers exp(omega_j), Gaussian body-frame spread 0.5."""
    name = "SO3"

    def __init__(self):
        self.m = SO3()
        self.Y_bar = np.eye(3)
        om = np.array([[0.5, 0.3, 0.2], [-0.4, 0.6, -0.1], [0.2, -0.5, 0.4],
                       [-0.3, -0.2, 0.7]])
        self.centers = self.m.expmap(np.eye(3), om)
        self.v_star = _true_v(3)

    def draw(self, center, n, rng):
        """Draws Gaussian body-frame predictor points around ``center``."""
        return self.m.expmap(center, 0.5 * rng.standard_normal((n, 3)))

    def noise(self, n, tau, rng):
        """Draws isotropic Gaussian noise in body-frame coordinates."""
        return tau * rng.standard_normal((n, 3))


class SPDDGP(_DGP):
    """Sym+(3), Ybar = 1.5 I, with Gaussian tangent draws.

    Centers are Exp_{c_j I}(sym(0.1 A_j)) and the spread is sym(0.3 A). Tangent
    draws are symmetrized Gaussian matrices, (A + A^T) / 2, so diagonal
    entries have variance s^2 and off-diagonal entries s^2 / 2.
    """
    name = "SPD3"

    def __init__(self):
        self.m = SPD(3)
        self.Y_bar = 1.5 * np.eye(3)
        rng = np.random.default_rng(_CENTER_SEED)
        cs = []
        for scale in (1.2, 0.8, 1.0, 1.4):
            A = rng.standard_normal((3, 3)) * 0.1
            cs.append(self.m.expmap(scale * np.eye(3), 0.5 * (A + A.T))[0])
        self.centers = np.array(cs)
        self.v_star = _vec_to_sym(_true_v(6))

    @staticmethod
    def _sym_gauss(n, s, rng):
        """Draws n symmetrized Gaussian matrices of scale ``s``."""
        A = rng.standard_normal((n, 3, 3)) * s
        return 0.5 * (A + np.swapaxes(A, 1, 2))

    def draw(self, center, n, rng):
        """Draws predictor points around ``center``."""
        return self.m.expmap(center, self._sym_gauss(n, 0.3, rng))

    def noise(self, n, tau, rng):
        """Draws symmetrized Gaussian noise."""
        return self._sym_gauss(n, tau, rng)


class HyperbolicDGP(_DGP):
    """H^3, generated in the Poincare ball and returned in the Lorentz model.

    Ball fixtures are Ybar = (0.2, 0, 0) and centers 0.3 d_j, of norm about
    0.095. The predictor spread is 0.15 |N(0,1)| in a uniform direction, and
    the noise is N(0, tau^2 I_3) in ball coordinates, whose Riemannian size is
    scaled by the conformal factor lambda(Ybar), about 2.08.
    """
    name = "H3"

    def __init__(self):
        self.m = Hyperbolic(3)
        self.ball_Y_bar = np.array([[0.2, 0.0, 0.0]])
        d = np.array([[0.3, 0.1, 0.0], [-0.1, 0.3, 0.0], [0.0, -0.1, 0.3],
                      [0.2, 0.2, -0.1]])
        self.ball_centers = 0.3 * d
        self.ball_v = _true_v(3)
        self.Y_bar = poincare.to_lorentz(self.ball_Y_bar)[0]
        self.v_star = poincare.push_tangent(self.ball_Y_bar, self.ball_v)

    def sample(self, n, p, q, tau, rng):
        """Draws one data set in the ball and maps it to the hyperboloid."""
        z = rng.standard_normal((n, q))
        Xb = []
        for j in range(p):
            c = self.ball_centers[j % len(self.ball_centers)][None]
            v = rng.standard_normal((n, 3))
            v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
            v = v * (0.15 * np.abs(rng.standard_normal(n)))[:, None]
            Xb.append(poincare.expmap(c, v))
        xi = z @ self.ball_v[:q] + tau * rng.standard_normal((n, 3))
        for j in range(p):
            c = self.ball_centers[j % len(self.ball_centers)][None]
            xi = xi + ALPHA_POOL[j] * poincare.transport(
                c, self.ball_Y_bar, poincare.logmap(c, Xb[j]))
        Yb = poincare.expmap(self.ball_Y_bar, xi)
        return (poincare.to_lorentz(Yb), [poincare.to_lorentz(x) for x in Xb],
                z)


DGPS = {"S2": SphereDGP, "T2": TorusDGP, "SO3": RotationDGP, "SPD3": SPDDGP,
        "H3": HyperbolicDGP}


def evaluate(dgp, n, p, q, tau, rng_train, rng_test, n_test=N_TEST):
    """Fits every method on one training set and scores it on one test set.

    Args:
        dgp: Data-generating process.
        n: Training sample size.
        p: Number of manifold predictors.
        q: Number of scalar covariates.
        tau: Noise scale.
        rng_train: Generator of the training set.
        rng_test: Generator of the test set.
        n_test: Test sample size.

    Returns:
        A dict with the test RMSEs of all methods and, for ROLS, the coefficient
        errors and the model-based rejections at level 0.05.
    """
    m = dgp.m
    Y, X, z = dgp.sample(n, p, q, tau, rng_train)
    Yt, Xt, zt = dgp.sample(n_test, p, q, tau, rng_test)
    f = fit(m, Y, X, z)
    pred = {"frechet_mean": B.frechet_mean(m, Y, n_test),
            "geodesic": B.geodesic_regression(m, Y, z, zt),
            "frechet_reg": B.frechet_regression(m, Y, z, zt),
            "sifr": B.single_index_frechet(m, Y, z, zt),
            "nlfr": B.nonlinear_frechet(m, Y, X, z, Xt, zt),
            "rols_m": B.rols_manifold_only(m, Y, X, Xt),
            "rols": f.predict(Xt, zt),
            ABLATION: B.tangent_linear_ridge(m, Y, X, z, Xt, zt)}
    rmse = {k: frechet_rmse(m, Yt, v) for k, v in pred.items()}
    p_alpha, p_v = f.pvalues("model")
    if q:
        # The true v_k in the fitted chart, after transport to the fitted mean.
        v_true = m.coords(f.Y_bar, m.transport(dgp.Y_bar, f.Y_bar,
                                               dgp.v_star[:q]))
        v_err = np.linalg.norm(np.asarray(f.V) - v_true, axis=1)
    else:
        v_err = np.zeros(0)
    return {"rmse": rmse,
            "alpha_err": np.abs(f.alphas - ALPHA_POOL[:p]),
            "v_err": v_err,
            "alpha_reject": (p_alpha < LEVEL).astype(float),
            "v_reject": (p_v < LEVEL).astype(float)}


def _task(task):
    """Runs one replication of one cell."""
    name, _, cfg, seed, trial = task
    dgp = DGPS[name]()
    return evaluate(dgp, cfg["n"], cfg["p"], cfg["q"], cfg["tau"],
                    rng_for(seed, trial, 0), rng_for(seed, trial, 1))


def summarize(trials):
    """Returns the mean and population std of each metric over replications."""
    keys = list(trials[0]["rmse"])
    R = {k: np.array([t["rmse"][k] for t in trials]) for k in keys}
    out = {"trials": len(trials),
           "rmse": {k: {"mean": float(v.mean()), "std": float(v.std())}
                    for k, v in R.items()}}
    for field in ("alpha_err", "v_err"):
        A = np.array([t[field] for t in trials])
        out[field] = {"mean": A.mean(0).tolist(), "std": A.std(0).tolist()}
    for field in ("alpha_reject", "v_reject"):
        out[field] = np.array([t[field] for t in trials]).mean(0).tolist()
    return out


def run(args):
    """Runs every cell on every manifold and returns the summaries."""
    grid = cells()
    tasks = [(name, key, cfg, args.seed, t)
             for name, _ in MANIFOLDS for key, cfg in grid
             for t in range(args.trials)]
    with Timer() as tm:
        res = pmap(_task, tasks, args.workers, chunksize=4)
    by_cell = {}
    for (name, key, _, _, _), r in zip(tasks, res):
        by_cell.setdefault(name, {}).setdefault(key, []).append(r)
    out = {"design": {"default": DEFAULT, "grid": GRID, "n_test": N_TEST,
                      "trials": args.trials, "alpha_pool": ALPHA_POOL,
                      "v_norm": V_NORM},
           "seconds": tm.seconds, "cells": {}}
    for name, _ in MANIFOLDS:
        out["cells"][name] = {}
        for key, cfg in grid:
            s = summarize(by_cell[name][key])
            s["config"] = cfg
            out["cells"][name][key] = s
    return out


def default_cell_rmse(results=None, quick=False):
    """Returns the default-cell RMSE, for Figure 2(d).

    Args:
        results: Dict returned by ``run``. By default it is read from
            ``results/prediction.json``.
        quick: Whether to read the smoke-test results.

    Returns:
        A dict {manifold: {method: (mean, std)}}, including the tangent-linear
        ablation under the key ``'tangent_linear'``.
    """
    if results is None:
        results = load_results("prediction", quick)
    cells_ = results["cells"]
    return {name: {k: (v["mean"], v["std"])
                   for k, v in cells_[name]["default"]["rmse"].items()}
            for name, _ in MANIFOLDS}


def plot_default_cell(ax, rmse, title=True, spacing=0.12, legend_width=0.45,
                      legend_fontsize=None, legend_columns=2):
    """Draws the default-cell RMSE as a dot plot, for Figures 4(a) and 2(d).

    The x-axis is widened so that the legend in the lower right corner, which
    spans about the two lowest groups and ``legend_width`` of the axis, covers
    no point.

    Args:
        ax: Matplotlib axis.
        rmse: Output of ``default_cell_rmse``.
        title: Whether to draw the default title, or a string to use instead.
        spacing: Vertical distance between the methods of a group.
        legend_width: Share of the axis width taken by the legend.
        legend_fontsize: Font size of the legend.
        legend_columns: Number of columns of the legend.
    """
    from matplotlib.lines import Line2D
    names = [k for k, _ in MANIFOLDS]
    nm = len(METHODS)
    group = nm * spacing + 0.10
    y0 = np.arange(len(names)) * group
    right = np.array([[rmse[nmf][k][0] + rmse[nmf][k][1] for k, *_ in METHODS]
                      for nmf in names])
    xmax = max(1.05 * right.max(), right[:2].max() / (1.0 - legend_width))
    for mi, (key, label, col, mk) in enumerate(METHODS):
        mean = np.array([rmse[nmf][key][0] for nmf in names])
        std = np.array([rmse[nmf][key][1] for nmf in names])
        y = y0 + (mi - nm / 2 + 0.5) * spacing
        main = key == "rols"
        for j in range(len(names)):
            ax.plot([0, mean[j]], [y[j], y[j]], color=col,
                    lw=1.0 if main else 0.5, alpha=0.3, zorder=1)
        ax.errorbar(mean, y, xerr=std, fmt="none", ecolor=col,
                    elinewidth=1.2 if main else 0.6, capsize=2,
                    capthick=1.2 if main else 0.6, zorder=10 if main else 2)
        ax.scatter(mean, y, s=(10 if main else 5.5) ** 2, color=col, marker=mk,
                   zorder=11 if main else 3,
                   edgecolors="white" if main else "none", linewidths=0.5)
    for j in range(0, len(names), 2):
        ax.axhspan(y0[j] - group / 2 + 0.02, y0[j] + group / 2 - 0.02,
                   color="#f0f0f0", zorder=0)
    ax.set_yticks(y0)
    ax.set_yticklabels([lab for _, lab in MANIFOLDS], fontsize=12)
    ax.set_xlabel("Fréchet RMSE")
    ax.set_xlim(0, xmax)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title if isinstance(title, str)
                     else "(a) Default setting (n=200, p=2, q=3, τ=0.15)",
                     loc="left", fontweight="bold")
    handles = [Line2D([0], [0], marker=mk, color=col, linestyle="",
                      ms=6 if key == "rols" else 4.5,
                      markeredgecolor="white" if key == "rols" else "none",
                      markeredgewidth=0.5) for key, _, col, mk in METHODS]
    ax.legend(handles, [lab for _, lab, _, _ in METHODS], loc="lower right",
              ncol=legend_columns, frameon=True, fancybox=False,
              edgecolor="0.7", handlelength=1.0, borderpad=0.4,
              labelspacing=0.3, columnspacing=1.0, fontsize=legend_fontsize)


def make_figure(results, path):
    """Draws Figure 4 and returns ``path``.

    Panel (a) is the default cell on every manifold, and panels (b)-(e) are the
    S^2 sweeps over tau, q, n and p on a shared y-axis.

    Args:
        results: Dict returned by ``run``.
        path: Output path.

    Returns:
        ``path``.
    """
    plt = pyplot()
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11,
                         "axes.labelsize": 11, "legend.fontsize": 10})
    fig = plt.figure(figsize=(12, 4.2))
    plot_default_cell(fig.add_axes([0.05, 0.14, 0.42, 0.81]),
                      default_cell_rmse(results))
    s2 = results["cells"]["S2"]
    sweeps = [("tau", "τ"), ("q", "q"), ("n", "n"), ("p", "p")]
    vals = [s2[cell_key(par, v)]["rmse"][k]["mean"]
            + sgn * s2[cell_key(par, v)]["rmse"][k]["std"]
            for par, _ in sweeps for v in GRID[par]
            for k, *_ in METHODS for sgn in (-1, 1)]
    lo, hi = max(0.0, min(vals) - 0.02), max(vals) + 0.02
    boxes = [[0.55, 0.65, 0.21, 0.30], [0.78, 0.65, 0.21, 0.30],
             [0.55, 0.14, 0.21, 0.30], [0.78, 0.14, 0.21, 0.30]]
    xs = np.arange(3)
    for i, (par, sym) in enumerate(sweeps):
        ax = fig.add_axes(boxes[i])
        for key, _, col, mk in METHODS:
            main = key == "rols"
            st = [s2[cell_key(par, v)]["rmse"][key] for v in GRID[par]]
            ax.errorbar(xs, [c["mean"] for c in st],
                        yerr=[c["std"] for c in st], color=col, marker=mk,
                        lw=2.0 if main else 1.0, ms=6 if main else 3.5,
                        capsize=2, capthick=0.6, elinewidth=0.6,
                        zorder=10 if main else 2)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{v:.2f}" if par == "tau" else str(v)
                            for v in GRID[par]])
        ax.set_xlabel(sym)
        ax.set_ylim(lo, hi)
        if i % 2 == 0:
            ax.set_ylabel("Fréchet RMSE")
        else:
            ax.set_yticklabels([])
        ax.set_title(f"({'bcde'[i]}) Varying {sym}", loc="left",
                     fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _num(x):
    """Formats a number with two decimals and no leading zero."""
    s = f"{x:.2f}"
    return s[1:] if s.startswith("0.") else s


def make_table(results):
    """Returns the LaTeX source of Table 5.

    Column blocks are S^2, SO(3), Sym+(3), H^3, T^2 and row groups are tau, q,
    n, p. The source needs the macros \\V{mean}{std} and \\B{mean}{std}, and
    bold marks the smallest mean in each column. The tangent-linear ablation
    is not part of the table.

    Args:
        results: Dict returned by ``run``.

    Returns:
        The LaTeX source.
    """
    order = ["S2", "SO3", "SPD3", "H3", "T2"]
    label = dict(MANIFOLDS)
    rows = [("frechet_mean", r"Fr.\ mean  "), ("geodesic", "Geodesic    "),
            ("frechet_reg", r"Fr.\ reg    "), ("sifr", "SI-Fr.      "),
            ("nlfr", "NL-Fr.      "), ("rols_m", r"ROLS\,(m)   "),
            ("rols", "ROLS        ")]
    heads = {"tau": (r"$\tau$", _num), "q": ("$q$", str), "n": ("$n$", str),
             "p": ("$p$", str)}
    L = [r"\begin{tabular}{@{}l *{15}{c}@{}}", r"\toprule",
         "& " + " & ".join(r"\multicolumn{3}{c}{%s}" % label[k] for k in order)
         + r" \\",
         "".join(r"\cmidrule(lr){%d-%d}" % (2 + 3 * i, 4 + 3 * i)
                 for i in range(5))]
    for gi, par in enumerate(["tau", "q", "n", "p"]):
        name, fmt = heads[par]
        if gi:
            L.append(r"\midrule")
        L.append(name + " & "
                 + " & ".join(fmt(v) for _ in order for v in GRID[par])
                 + r" \\")
        L.append(r"\midrule")
        cols = [results["cells"][k][cell_key(par, v)]["rmse"]
                for k in order for v in GRID[par]]
        best = [min((k for k, _ in rows), key=lambda k: c[k]["mean"])
                for c in cols]
        for key, lab in rows:
            cells_ = [(r"\B" if b == key else r"\V")
                      + "{%s}{%s}" % (_num(c[key]["mean"]), _num(c[key]["std"]))
                      for c, b in zip(cols, best)]
            L.append(lab + "& " + " & ".join(cells_) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(L)


def report(results):
    """Prints the RMSE of every method and the ROLS diagnostics."""
    C = results["cells"]
    names = [k for k, _ in MANIFOLDS]
    keys = [k for k, *_ in METHODS] + [ABLATION]
    labels = [lab for _, lab, *_ in METHODS] + ["Tan. lin."]
    header = f"{'':10s}" + "".join(f"{n:>8s}" for n in names)
    print(f"Prediction benchmark, {results['design']['trials']} trials per "
          f"cell, {results['seconds']:.0f} s")
    print("\nDefault cell RMSE")
    print(header)
    for k, lab in zip(keys, labels):
        print(f"{lab:10s}" + "".join(
            f"{C[n]['default']['rmse'][k]['mean']:8.3f}" for n in names))
    print("\nROLS RMSE by cell")
    ks = [key for key, _ in cells()]
    print(header)
    for key in ks:
        print(f"{key:10s}" + "".join(f"{C[n][key]['rmse']['rols']['mean']:8.3f}"
                                     for n in names))
    print("\nROLS at the default cell")
    for n in names:
        d = C[n]["default"]
        print(f"  {n:5s} |a err| {np.mean(d['alpha_err']['mean']):.4f}  "
              f"||v err|| {np.mean(d['v_err']['mean']):.4f}  min rejection "
              f"alpha {min(d['alpha_reject']):.2f} "
              f"v {min(d['v_reject']):.2f}")
    gap = [C[n][k]["rmse"][ABLATION]["mean"] / C[n][k]["rmse"]["rols"]["mean"]
           - 1
           for n in names for k in ks]
    print(f"\nTangent-linear ridge vs ROLS, relative RMSE difference over all "
          f"cells: median {np.median(gap):+.1%}, "
          f"range [{min(gap):+.1%}, {max(gap):+.1%}]")


if __name__ == "__main__":
    args = parse_args("Prediction benchmark, paper Table 5 and Figure 4",
                      TRIALS)
    results = run(args)
    print("wrote", save_results("prediction", results, args))
    sfx = "_quick" if args.quick else ""
    print("wrote", save_text(f"table_prediction{sfx}.tex",
                             make_table(results)))
    print("wrote", make_figure(results,
                               figure_path(f"simulation_summary{sfx}.pdf")))
    report(results)

"""Measures the power and Type I error of the asymptotic ROLS tests.

Produces Figures 7, 8 and 2(c) of the paper.

One manifold predictor X (coefficient alpha) and one scalar covariate z
(tangent coefficient v) on five manifolds. The tests are the model-based t test
(df = n d - p - q d) of H0: alpha = 0 and chi^2_d Wald test of H0: v = 0, at
level 0.05.

Figure 7 (figures/power_curves.pdf). n = 200, tau = 0.15, 500 trials per point,
    effect sizes {0, .03, .06, .10, .15, .25, .50}. Panel (a) varies |alpha|
    with ||v|| = 0.3 and tests alpha = 0. Panel (b) varies ||v|| with
    alpha = 0.3 and tests v = 0.
Figure 8 (figures/typeI_calibration.pdf). tau = 0.15, n in {50, 100, 200, 500},
    500 trials per cell, H0: alpha = 0 with ||v|| = 0.3 and H0: v = 0 with
    alpha = 0.3.
Figure 2(c). The alpha power curves with the n = 200 Type I range.

In every case X_i is drawn around X_bar, u_i = PT_{X_bar -> Y_bar}
Log_{X_bar}(X_i), z_i ~ N(0, 1) and
Y_i = Exp_{Y_bar}(alpha u_i + z_i v + eps_i).

  S^2      Y_bar = e3, X_bar = e1. Log_{X_bar} X_i has a uniform direction and
           length 0.6 |N(0,1)|. v = ||v|| (0.8 e1 + 0.6 e2),
           eps = tau (N e1 + N e2).
  T^2      Y_bar = (0.5, 0.7), X_bar = (0.3, -0.2),
           X_i = X_bar + 0.5 N(0, I_2), v = ||v|| (0.8, 0.6),
           eps = tau N(0, I_2).
  SO(3)    Y_bar = I, X_bar = Exp_I(0.5, 0.3, 0.2),
           X_i = X_bar Exp(0.5 N(0, I_3)),
           v = ||v|| (0.6, -0.7, 0.37)/norm, eps = tau N(0, I_3), in body-frame
           coordinates.
  Sym+(3)  Y_bar = 1.5 I, X_bar = 1.2 I, X_i = Exp_{X_bar}(sym(0.3 G)),
           V = ||v|| V_dir with V_dir the symmetric matrix ``_SPD_VDIR`` of
           unit Frobenius norm, eps = sym(tau G), with G a standard Gaussian
           3 x 3 matrix and sym(A) = (A + A^T)/2, all as ambient matrices at
           the base point.
  H^3      Generated in the Poincare ball, Y_bar = (0.2, 0, 0),
           X_bar = 0.3 (3, 1, 0)/|.|, Log_{X_bar} X_i uniform in direction with
           Euclidean length 0.15 |N(0,1)|, v = ||v|| (0, 0.8, 0.6),
           eps = tau N(0, I_3), all in ball coordinates, then mapped
           isometrically to the Lorentz model for fitting.

Usage, from the repository root:

    python -m simulations.inference [--quick] [--trials N] [--workers K]
                                    [--seed S]
"""

import numpy as np
from scipy import stats

from ROLS import SO3, SPD, Hyperbolic, Sphere, Torus, fit
from simulations import poincare as pb
from simulations._common import (Timer, figure_path, parse_args, pmap, pyplot,
                                 rng_for, save_results)

TAU = 0.15
N_POWER = 200
EFFECTS = [0.0, 0.03, 0.06, 0.10, 0.15, 0.25, 0.50]
SAMPLE_SIZES = [50, 100, 200, 500]
FIXED = 0.3  # The coefficient that is not under test.
LEVEL = 0.05
TRIALS = 500
ORDER = ["S2", "T2", "SO3", "SPD", "H3"]
MANIFOLD_ID = {"S2": 1, "T2": 2, "SO3": 3, "SPD": 4, "H3": 5}
TESTS = ("alpha", "v")

_S2 = Sphere(3)
_T2 = Torus(2)
_SO3 = SO3()
_SPD = SPD(3)
_H3 = Hyperbolic(3)

_E = np.eye(3)
_SO3_XBAR = _SO3.expmap(np.eye(3), np.array([[0.5, 0.3, 0.2]]))[0]
_SO3_VDIR = np.array([0.6, -0.7, 0.37]) / np.linalg.norm([0.6, -0.7, 0.37])
_SPD_VDIR = np.array([[0.15, 0.08, 0.00],
                      [0.08, 0.10, 0.05],
                      [0.00, 0.05, 0.12]])
_SPD_VDIR = _SPD_VDIR / np.linalg.norm(_SPD_VDIR)
_H3_YBAR = np.array([0.2, 0.0, 0.0])
_H3_XBAR = 0.3 * np.array([0.3, 0.1, 0.0]) / np.linalg.norm([0.3, 0.1, 0.0])
_H3_VDIR = np.array([0.0, 0.8, 0.6])


def _sym(A):
    """Returns the symmetric part of ``A``."""
    return 0.5 * (A + np.swapaxes(A, -1, -2))


def _uniform_direction(rng, n, dim, radius_scale, project_out=None):
    """Draws isotropic tangent vectors of length radius_scale * |N(0,1)|."""
    v = rng.standard_normal((n, dim))
    if project_out is not None:
        v = v - np.outer(v @ project_out, project_out)
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    return v * (radius_scale * np.abs(rng.standard_normal(n)))[:, None]


def gen_s2(n, alpha, v_norm, tau, rng):
    """Draws (manifold, Y, X, z) on S^2."""
    M = _S2
    Y_bar, X_bar = _E[2], _E[0]
    t1, t2 = _E[0], _E[1]
    v = v_norm * (0.8 * t1 + 0.6 * t2)
    X = M.expmap(X_bar, _uniform_direction(rng, n, 3, 0.6, project_out=X_bar))
    z = rng.standard_normal(n)
    u = M.transport(X_bar, Y_bar, M.logmap(X_bar, X))
    g = rng.standard_normal((n, 2)) * tau
    eps = g[:, :1] * t1 + g[:, 1:] * t2
    Y = M.expmap(Y_bar, alpha * u + z[:, None] * v + eps)
    return M, Y, X, z


def gen_t2(n, alpha, v_norm, tau, rng):
    """Draws (manifold, Y, X, z) on T^2, where transport is the identity."""
    M = _T2
    Y_bar, X_bar = np.array([0.5, 0.7]), np.array([0.3, -0.2])
    v = v_norm * np.array([0.8, 0.6])
    X = M.expmap(X_bar, 0.5 * rng.standard_normal((n, 2)))
    z = rng.standard_normal(n)
    u = M.logmap(X_bar, X)
    eps = tau * rng.standard_normal((n, 2))
    Y = M.expmap(Y_bar, alpha * u + z[:, None] * v + eps)
    return M, Y, X, z


def gen_so3(n, alpha, v_norm, tau, rng):
    """Draws (manifold, Y, X, z) on SO(3)."""
    M = _SO3
    Y_bar, X_bar = np.eye(3), _SO3_XBAR
    v = v_norm * _SO3_VDIR
    X = M.expmap(X_bar, 0.5 * rng.standard_normal((n, 3)))
    z = rng.standard_normal(n)
    u = M.transport(X_bar, Y_bar, M.logmap(X_bar, X))
    eps = tau * rng.standard_normal((n, 3))
    Y = M.expmap(Y_bar, alpha * u + z[:, None] * v + eps)
    return M, Y, X, z


def gen_spd(n, alpha, v_norm, tau, rng):
    """Draws (manifold, Y, X, z) on Sym+(3)."""
    M = _SPD
    Y_bar, X_bar = 1.5 * np.eye(3), 1.2 * np.eye(3)
    V = v_norm * _SPD_VDIR
    X = M.expmap(X_bar, _sym(0.3 * rng.standard_normal((n, 3, 3))))
    z = rng.standard_normal(n)
    u = M.transport(X_bar, Y_bar, M.logmap(X_bar, X))
    eps = _sym(tau * rng.standard_normal((n, 3, 3)))
    Y = M.expmap(Y_bar, alpha * u + z[:, None, None] * V + eps)
    return M, Y, X, z


def gen_h3(n, alpha, v_norm, tau, rng):
    """Draws (manifold, Y, X, z) in the Poincare ball, on the hyperboloid."""
    v = v_norm * _H3_VDIR
    Xb = pb.expmap(_H3_XBAR, _uniform_direction(rng, n, 3, 0.15))
    z = rng.standard_normal(n)
    u = pb.transport(_H3_XBAR, _H3_YBAR, pb.logmap(_H3_XBAR, Xb))
    eps = tau * rng.standard_normal((n, 3))
    Yb = pb.expmap(_H3_YBAR, alpha * u + z[:, None] * v + eps)
    return _H3, pb.to_lorentz(Yb), pb.to_lorentz(Xb), z


GENERATORS = {"S2": gen_s2, "T2": gen_t2, "SO3": gen_so3, "SPD": gen_spd,
              "H3": gen_h3}


def generate(name, n, alpha, v_norm, tau, rng):
    """Draws (manifold, Y, X, z) from one of the five processes.

    Args:
        name: One of 'S2', 'T2', 'SO3', 'SPD', 'H3'.
        n: Sample size.
        alpha: Coefficient of the manifold predictor.
        v_norm: Norm of the tangent coefficient of the scalar covariate.
        tau: Noise scale.
        rng: ``numpy`` generator.

    Returns:
        The manifold object, the responses, the predictors and the covariates.
    """
    return GENERATORS[name](n, alpha, v_norm, tau, rng)


def _block(task):
    """Counts the rejections among trials [t0, t1) of one cell.

    The seed depends on the cell, not on the figure, so the effect-0 point of a
    power curve and the n = 200 Type I cell are the same experiment.

    Args:
        task: Tuple (name, test, n, effect, t0, t1, seed).

    Returns:
        The numbers of rejections, valid tests and failed fits.
    """
    name, test, n, effect, t0, t1, seed = task
    alpha, v_norm = (effect, FIXED) if test == "alpha" else (FIXED, effect)
    rej = valid = failed = 0
    for t in range(t0, t1):
        rng = rng_for(seed, MANIFOLD_ID[name], TESTS.index(test), n,
                      int(round(1000 * effect)), t)
        try:
            M, Y, X, z = generate(name, n, alpha, v_norm, TAU, rng)
            p_alpha, p_v = fit(M, Y, [X], z).pvalues("model")
            p = float(p_alpha[0] if test == "alpha" else p_v[0])
        except np.linalg.LinAlgError:
            failed += 1
            continue
        if np.isfinite(p):
            valid += 1
            rej += int(p < LEVEL)
        else:
            failed += 1
    return rej, valid, failed


def run(args):
    """Runs every cell of Figures 7 and 8 and returns the rejection rates."""
    cells = set()
    for name in ORDER:
        for test in TESTS:
            cells.update((name, test, N_POWER, e) for e in EFFECTS)
            cells.update((name, test, n, 0.0) for n in SAMPLE_SIZES)
    cells = sorted(cells)
    block = 50
    tasks = [(name, test, n, e, t0, min(t0 + block, args.trials), args.seed)
             for (name, test, n, e) in cells
             for t0 in range(0, args.trials, block)]
    with Timer() as tm:
        out = pmap(_block, tasks, workers=args.workers)
    tally = {}
    for (name, test, n, e, *_), (rej, valid, failed) in zip(tasks, out):
        r = tally.setdefault((name, test, n, e), [0, 0, 0])
        r[0] += rej
        r[1] += valid
        r[2] += failed
    rate = {c: r[0] / max(r[1], 1) for c, r in tally.items()}

    power = {name: {"effect_sizes": EFFECTS,
                    "power_alpha": [rate[(name, "alpha", N_POWER, e)]
                                    for e in EFFECTS],
                    "power_v": [rate[(name, "v", N_POWER, e)]
                                for e in EFFECTS]}
             for name in ORDER}
    typeI = {name: {"sample_sizes": SAMPLE_SIZES,
                    "typeI_alpha": [rate[(name, "alpha", n, 0.0)]
                                    for n in SAMPLE_SIZES],
                    "typeI_v": [rate[(name, "v", n, 0.0)]
                                for n in SAMPLE_SIZES]}
             for name in ORDER}
    return {"power": power, "typeI": typeI, "trials": args.trials,
            "n_valid_min": min(r[1] for r in tally.values()),
            "n_failed": sum(r[2] for r in tally.values()),
            "seconds": tm.seconds,
            "settings": {"tau": TAU, "n_power": N_POWER,
                         "fixed_coefficient": FIXED, "level": LEVEL,
                         "test": "model-based t (alpha), chi^2_d Wald (v)"}}


LABELS = {"S2": r"$S^2$", "T2": r"$\mathbb{T}^2$", "SO3": r"$\mathrm{SO}(3)$",
          "SPD": r"$\mathrm{Sym}^+(3)$", "H3": r"$\mathbb{H}^3$"}
MARKERS = {"S2": "o", "H3": "s", "SO3": "^", "SPD": "D", "T2": "P"}
# Figure 2 uses its own palette and markers, shared by its four panels.
HEADLINE_COLORS = {"S2": "#1f77b4", "T2": "#8c564b", "SO3": "#ff7f0e",
                   "SPD": "#2ca02c", "H3": "#d62728"}
HEADLINE_MARKERS = {"S2": "o", "T2": "P", "SO3": "s", "SPD": "D", "H3": "^"}
STYLE = {"font.family": "serif", "mathtext.fontset": "dejavuserif",
         "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
         "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11}


def plot_power(results, path):
    """Draws Figure 7 and returns ``path``."""
    plt = pyplot()
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for name in ORDER:
            r = results["power"][name]
            for ax, key in zip(axes, ("power_alpha", "power_v")):
                ax.plot(r["effect_sizes"], r[key], marker=MARKERS[name],
                        label=LABELS[name], markersize=7, linewidth=1.8)
        for ax in axes:
            ax.axhline(LEVEL, color="gray", linestyle="--", linewidth=1.0,
                       label=r"$\alpha=0.05$")
            ax.set_ylim(-0.02, 1.05)
            ax.set_ylabel("Rejection rate")
            ax.legend()
            ax.grid(True, alpha=0.3)
        axes[0].set_xlabel(r"Effect size $|\alpha|$")
        axes[0].set_title(r"(a) Power: $H_0\!: \alpha=0$")
        axes[1].set_xlabel(r"Effect size $\|v\|$")
        axes[1].set_title(r"(b) Power: $H_0\!: v=0$")
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    return path


def plot_typeI(results, path):
    """Draws Figure 8 and returns ``path``."""
    plt = pyplot()
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for name in ORDER:
            r = results["typeI"][name]
            for ax, key in zip(axes, ("typeI_alpha", "typeI_v")):
                ax.plot(r["sample_sizes"], r[key], marker=MARKERS[name],
                        label=LABELS[name], markersize=7, linewidth=1.8)
        for ax in axes:
            ax.axhline(LEVEL, color="red", linestyle="--", linewidth=1.2,
                       label="Nominal 0.05")
            ax.set_xscale("log")
            ax.set_xticks(SAMPLE_SIZES)
            ax.set_xticklabels([str(n) for n in SAMPLE_SIZES])
            ax.set_ylim(-0.005, 0.12)
            ax.set_ylabel("Type I error rate")
            ax.set_xlabel("Sample size $n$")
            ax.legend()
            ax.grid(True, alpha=0.3)
        axes[0].set_title(r"(a) $H_0\!: \alpha=0$ (manifold predictor)")
        axes[1].set_title(r"(b) $H_0\!: v=0$ (scalar predictor)")
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    return path


def panel_power(ax, results):
    """Draws Figure 2(c) on ``ax``.

    Args:
        ax: Matplotlib axis.
        results: Dict returned by ``run`` or ``load_results('inference')``.

    Returns:
        ``ax``.
    """
    for name in ORDER:
        r = results["power"][name]
        ax.plot(r["effect_sizes"], r["power_alpha"],
                marker=HEADLINE_MARKERS[name], color=HEADLINE_COLORS[name],
                lw=1.3, ms=4.0, label=LABELS[name])
    ax.axhline(LEVEL, color="gray", ls="--", lw=0.9)
    ax.set_ylim(-0.02, 1.18)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.set_xlabel(r"$|\alpha|$", labelpad=2)
    ax.set_ylabel("Rejection rate", labelpad=2)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="lower right", ncol=1, columnspacing=0.4,
              handlelength=1.2, frameon=True, edgecolor="0.75", borderpad=0.3,
              labelspacing=0.2, handletextpad=0.4)
    lo, hi = typeI_range_n200(results)
    ax.text(0.03, 0.97, rf"Type I $\in[{lo:.3f},{hi:.3f}]$",
            transform=ax.transAxes, ha="left", va="top", fontsize=11,
            bbox=dict(facecolor="white", edgecolor="0.75",
                      boxstyle="round,pad=0.25", alpha=0.5))
    return ax


def typeI_range_n200(results):
    """Returns the range of the alpha Type I rate at n = 200 over manifolds."""
    vals = []
    for m in ORDER:
        r = results["typeI"][m]
        vals.append(r["typeI_alpha"][r["sample_sizes"].index(200)])
    return min(vals), max(vals)


def _envelope(trials, p=LEVEL):
    """Returns the two-sided 95% binomial range of a rejection rate."""
    return (stats.binom.ppf(0.025, trials, p) / trials,
            stats.binom.ppf(0.975, trials, p) / trials)


def report(results):
    """Prints the power curves and the Type I rates."""
    T = results["trials"]
    print(f"Coefficient inference, tau = {TAU}, {T} trials per cell.")
    for test, title in (
            ("alpha", "Figure 7(a): t test of alpha = 0, n = 200, ||v|| = 0.3"),
            ("v", "Figure 7(b): Wald test of v = 0, n = 200, alpha = 0.3")):
        print("\n" + title)
        print("        " + "".join(f"{e:>7.2f}" for e in EFFECTS))
        for name in ORDER:
            rates = results["power"][name]["power_" + test]
            print(f"  {name:<5} " + "".join(f"  {a:.3f}" for a in rates))
    lo_env, hi_env = _envelope(T)
    allrates = []
    for test, title in (("alpha", "Figure 8(a): Type I of alpha = 0 "
                                  "(||v|| = 0.3)"),
                        ("v", "Figure 8(b): Type I of v = 0 (alpha = 0.3)")):
        print("\n" + title)
        print("        " + "".join(f"{'n=' + str(n):>7}"
                                   for n in SAMPLE_SIZES))
        for name in ORDER:
            rates = results["typeI"][name]["typeI_" + test]
            allrates += rates
            print(f"  {name:<5} " + "".join(f"  {a:.3f}" for a in rates))
    outside = sum(not lo_env <= r <= hi_env for r in allrates)
    print(f"\nAll {len(allrates)} Type I rates in "
          f"[{min(allrates):.3f}, {max(allrates):.3f}]; {outside} outside the "
          f"binomial 95% range [{lo_env:.3f}, {hi_env:.3f}] for {T} trials.")
    lo, hi = typeI_range_n200(results)
    print(f"Figure 2(c) inset, alpha Type I at n = 200: "
          f"[{lo:.3f}, {hi:.3f}].")
    print(f"Monte Carlo s.e. of a rate near 0.05: "
          f"{np.sqrt(0.05 * 0.95 / T):.3f}; failed fits: "
          f"{results['n_failed']}; wall time {results['seconds']:.0f} s.")


if __name__ == "__main__":
    args = parse_args(
        "Power and Type I calibration of the ROLS tests (Figures 7, 8, 2(c)).",
        trials=TRIALS)
    results = run(args)
    tag = "_quick" if args.quick else ""
    print("wrote", save_results("inference", results, args))
    print("wrote", plot_power(results, figure_path(f"power_curves{tag}.pdf")))
    print("wrote", plot_typeI(results,
                              figure_path(f"typeI_calibration{tag}.pdf")))
    report(results)

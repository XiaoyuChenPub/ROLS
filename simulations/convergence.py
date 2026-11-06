"""Measures the convergence rate of the ROLS coefficients.

Produces Table 6 and Figures 6 and 2(b) of the paper.

One manifold predictor and one scalar covariate on each manifold,

    Log_Ybar(Y_i) = alpha * PT_{Xbar -> Ybar}(Log_Xbar(X_i)) + z_i v + eps_i,

for n in {50, 100, 200, 500, 1000, 2000}, 100 trials at each n and a test set
of 200 points per trial. Table 6 reports the least-squares slope and R^2 of
log mean error on log n for |alpha_hat - alpha*| and ||v_hat - v*||.

    S^2      Ybar = e3, Xbar = e1, alpha 0.5, v = (0.4, 0.3, 0), noise sd 0.05,
             Log_Xbar(X) uniform in direction with length 0.6 |N(0, 1)|
    T^2      Ybar = (5.0, 1.2), Xbar = (4.8, 1.0), alpha 0.5, v = (0.3, 0.2),
             noise 0.05, Log_Xbar(X) ~ N(0, 0.45^2 I)
    SO(3)    Ybar = I, Xbar = Exp(0.5, 0.3, 0.2), alpha 0.4,
             v = (0.3, -0.2, 0.1), noise 0.05, Log_Xbar(X) ~ N(0, 0.5^2 I)
    Sym+(3)  Ybar = 1.5 I, Xbar = 1.2 I, alpha 0.5, v the symmetric ``SPD_V``,
             Log_Xbar(X) = sym(A), A_ij ~ N(0, 0.3^2); noise sym(A),
             A_ij ~ N(0, 0.05^2)
    H^3      generated in the Poincare ball, Ybar = (0.2, 0, 0),
             Xbar = (0.09, 0.03, 0), alpha 0.5, v = (0, 0.3, 0.2), noise 0.02,
             ball tangent directions uniform with Euclidean length
             0.15 |N(0, 1)|, then mapped to the Lorentz model

Usage, from the repository root:

    python -m simulations.convergence [--quick] [--trials N] [--workers K]
                                      [--seed S]
"""

import numpy as np

from ROLS import SO3, SPD, Hyperbolic, Sphere, Torus, fit, frechet_rmse
from simulations import poincare as pb
from simulations._common import (Timer, figure_path, loglog_slope, parse_args,
                                 pmap, pyplot, rng_for, save_results,
                                 save_text)

NS = (50, 100, 200, 500, 1000, 2000)
N_TEST = 200
TRIALS = 100
ORDER = ("S2", "T2", "SO3", "SPD", "H3")

LABELS = {"S2": r"$\mathbb{S}^2$", "T2": r"$\mathbb{T}^2$",
          "SO3": r"$\mathrm{SO}(3)$", "SPD": r"$\mathrm{Sym}^+(3)$",
          "H3": r"$\mathbb{H}^3$"}
MARKERS = {"S2": "o", "T2": "P", "SO3": "s", "SPD": "D", "H3": "^"}
COLORS = {"S2": "#1f77b4", "T2": "#8c564b", "SO3": "#ff7f0e",
          "SPD": "#2ca02c", "H3": "#d62728"}


def _unit_directions(rng, n, dim, normal=None):
    """Draws uniform directions, orthogonal to ``normal`` when it is given."""
    g = rng.standard_normal((n, dim))
    if normal is not None:
        g = g - np.outer(g @ normal, normal)
    return g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-12)


def _sym(A):
    """Returns the symmetric part of ``A``."""
    return 0.5 * (A + np.swapaxes(A, -1, -2))


S2_Y_BAR = np.array([0.0, 0.0, 1.0])
S2_X_BAR = np.array([1.0, 0.0, 0.0])
S2_V = np.array([0.4, 0.3, 0.0])

T2_Y_BAR = Torus(2).project(np.array([5.0, 1.2]))
T2_X_BAR = Torus(2).project(np.array([4.8, 1.0]))
T2_V = np.array([0.3, 0.2])

SO3_Y_BAR = np.eye(3)
SO3_X_BAR = SO3().expmap(np.eye(3), np.array([0.5, 0.3, 0.2]))[0]
SO3_V = np.array([0.3, -0.2, 0.1])

SPD_Y_BAR = 1.5 * np.eye(3)
SPD_X_BAR = 1.2 * np.eye(3)
SPD_V = np.array([[0.20, 0.10, 0.05],
                  [0.10, 0.15, 0.08],
                  [0.05, 0.08, 0.12]])

H3_Y_BALL = np.array([0.2, 0.0, 0.0])
H3_X_BALL = np.array([0.3, 0.1, 0.0]) * 0.3
H3_V_BALL = np.array([0.0, 0.3, 0.2])


def gen_s2(m, n, rng):
    """Draws (X, z, Y) on S^2."""
    yb, xb = S2_Y_BAR, S2_X_BAR
    X = m.expmap(xb, _unit_directions(rng, n, 3, xb)
                 * (0.6 * np.abs(rng.standard_normal(n)))[:, None])
    z = rng.standard_normal(n)
    u = m.transport(xb, yb, m.logmap(xb, X))
    eps = 0.05 * rng.standard_normal((n, 3))
    eps -= np.outer(eps @ yb, yb)
    return X, z, m.expmap(yb, 0.5 * u + z[:, None] * S2_V + eps)


def gen_t2(m, n, rng):
    """Draws (X, z, Y) on T^2."""
    X = m.expmap(T2_X_BAR, 0.45 * rng.standard_normal((n, 2)))
    z = rng.standard_normal(n)
    xi = (0.5 * m.logmap(T2_X_BAR, X) + z[:, None] * T2_V
          + 0.05 * rng.standard_normal((n, 2)))
    return X, z, m.expmap(T2_Y_BAR, xi)


def gen_so3(m, n, rng):
    """Draws (X, z, Y) on SO(3)."""
    X = m.expmap(SO3_X_BAR, 0.5 * rng.standard_normal((n, 3)))
    z = rng.standard_normal(n)
    u = m.transport(SO3_X_BAR, SO3_Y_BAR, m.logmap(SO3_X_BAR, X))
    xi = 0.4 * u + z[:, None] * SO3_V + 0.05 * rng.standard_normal((n, 3))
    return X, z, m.expmap(SO3_Y_BAR, xi)


def gen_spd(m, n, rng):
    """Draws (X, z, Y) on Sym+(3)."""
    z = rng.standard_normal(n)
    X = m.expmap(SPD_X_BAR, _sym(0.3 * rng.standard_normal((n, 3, 3))))
    u = m.transport(SPD_X_BAR, SPD_Y_BAR, m.logmap(SPD_X_BAR, X))
    eps = _sym(0.05 * rng.standard_normal((n, 3, 3)))
    return X, z, m.expmap(SPD_Y_BAR,
                          0.5 * u + z[:, None, None] * SPD_V + eps)


def gen_h3(m, n, rng):
    """Draws (X, z, Y) in the Poincare ball and returns hyperboloid points.

    Tangent vectors are Euclidean ball vectors, so the Riemannian spread at
    Xbar is lambda(Xbar) = 2 / (1 - |Xbar|^2) times the stated 0.15.
    """
    xb, yb = H3_X_BALL[None], H3_Y_BALL[None]
    Xb = pb.expmap(xb, _unit_directions(rng, n, 3)
                   * (0.15 * np.abs(rng.standard_normal(n)))[:, None])
    z = rng.standard_normal(n)
    u = pb.transport(xb, yb, pb.logmap(xb, Xb))
    xi = 0.5 * u + z[:, None] * H3_V_BALL + 0.02 * rng.standard_normal((n, 3))
    Yb = pb.expmap(yb, xi)
    return pb.to_lorentz(Xb), z, pb.to_lorentz(Yb)


# name -> (manifold, generator, alpha*, true Ybar, v* as an ambient tangent)
SETTINGS = {
    "S2": (Sphere(3), gen_s2, 0.5, S2_Y_BAR, S2_V),
    "T2": (Torus(2), gen_t2, 0.5, T2_Y_BAR, T2_V),
    "SO3": (SO3(), gen_so3, 0.4, SO3_Y_BAR, SO3_V),
    "SPD": (SPD(3), gen_spd, 0.5, SPD_Y_BAR, SPD_V),
    "H3": (Hyperbolic(3), gen_h3, 0.5, pb.to_lorentz(H3_Y_BALL)[0],
           pb.push_tangent(H3_Y_BALL, H3_V_BALL)[0]),
}


def errors(m, f, alpha, Y_bar, v):
    """Returns |alpha_hat - alpha*| and ||v_hat - v*||.

    ``v_hat`` lives in the intrinsic frame at the fitted base point, so ``v*``
    is parallel transported there before the comparison.

    Args:
        m: Manifold.
        f: Fit returned by ``ROLS.fit``.
        alpha: True alpha.
        Y_bar: True base point of the response.
        v: True v as an ambient tangent vector at ``Y_bar``.

    Returns:
        The pair of errors.
    """
    v_fit = m.coords(f.Y_bar,
                     m.transport(Y_bar, f.Y_bar, np.asarray(v)[None]))[0]
    return (abs(float(f.alphas[0]) - alpha),
            float(np.linalg.norm(f.V[0] - v_fit)))


def _trial(task):
    """Returns the coefficient errors and test RMSE of one replication."""
    name, n, trial, seed = task
    m, gen, alpha, Y_bar, v = SETTINGS[name]
    key = ORDER.index(name)
    X, z, Y = gen(m, n, rng_for(seed, key, n, trial, 0))
    Xt, zt, Yt = gen(m, N_TEST, rng_for(seed, key, n, trial, 1))
    f = fit(m, Y, [X], z)
    a_err, v_err = errors(m, f, alpha, Y_bar, v)
    return a_err, v_err, frechet_rmse(m, Yt, f.predict([Xt], zt))


def slopes(errs, ns):
    """Returns (slope, R^2) of log mean error on log n per manifold."""
    out = {}
    for name, per_n in errs.items():
        out[name] = {}
        for coef, key in (("alpha", "alpha_err"), ("v", "v_err")):
            means = [np.mean(per_n[str(n)][key]) for n in ns]
            out[name][coef] = loglog_slope(ns, means)
    return out


def run(args):
    """Runs the experiment and returns the results as a nested dict."""
    ns = list(NS)
    tasks = [(name, n, t, args.seed)
             for name in ORDER for n in ns for t in range(args.trials)]
    with Timer() as clock:
        out = pmap(_trial, tasks, args.workers)
    errs = {name: {str(n): {"alpha_err": [], "v_err": [], "rmse": []}
                   for n in ns}
            for name in ORDER}
    for (name, n, _, _), (a, v, r) in zip(tasks, out):
        cell = errs[name][str(n)]
        cell["alpha_err"].append(a)
        cell["v_err"].append(v)
        cell["rmse"].append(r)
    return {"ns": ns, "n_test": N_TEST, "trials": args.trials, "errors": errs,
            "slopes": slopes(errs, ns), "seconds": clock.seconds}


def table_tex(results):
    """Returns the LaTeX source of Table 6."""
    s = results["slopes"]
    cols = " & ".join(LABELS[m] for m in ORDER)
    rows = []
    for coef, sym in (("alpha", r"$\hat\alpha$"), ("v", r"$\hat v$")):
        rows.append(f"{sym} slope & "
                    + " & ".join(f"${s[m][coef][0]:.2f}$" for m in ORDER)
                    + r" \\")
        rows.append(f"{sym} $R^2$  & "
                    + " & ".join(f"${s[m][coef][1]:.2f}$" for m in ORDER)
                    + r" \\")
    return ("\\begin{tabular}{lccccc}\n\\toprule\n& " + cols
            + " \\\\\n\\midrule\n" + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n")


def _curve(results, name, key):
    """Returns n, the mean error and its standard error for one manifold."""
    ns = results["ns"]
    vals = [np.asarray(results["errors"][name][str(n)][key], float)
            for n in ns]
    mean = np.array([v.mean() for v in vals])
    sem = np.array([v.std() / np.sqrt(len(v)) for v in vals])
    return np.asarray(ns), mean, sem


def _reference(results, key):
    """Returns the O(n^-1/2) line through the mean errors at the middle n.

    The line passes through the geometric mean of the manifold means.
    """
    ns = np.asarray(results["ns"], float)
    mid = results["ns"][len(ns) // 2]
    anchor = np.exp(np.mean(
        [np.log(np.mean(results["errors"][m][str(mid)][key])) for m in ORDER]))
    return ns, anchor * (ns / mid) ** -0.5


def headline_data(results):
    """Returns the data of Figure 2(b), the alpha errors and reference line."""
    ns, ref = _reference(results, "alpha_err")
    curves = {}
    for name in ORDER:
        _, mean, sem = _curve(results, name, "alpha_err")
        curves[name] = {"mean": mean, "sem": sem}
    return {"ns": ns, "curves": curves, "reference": ref}


def headline_panel(ax, results):
    """Draws Figure 2(b) on ``ax``."""
    d = headline_data(results)
    for name in ORDER:
        c = d["curves"][name]
        ax.errorbar(d["ns"], c["mean"], yerr=c["sem"], label=LABELS[name],
                    marker=MARKERS[name], color=COLORS[name], lw=1.2, ms=4.0,
                    capsize=1.6, capthick=0.6, elinewidth=0.6)
    ax.plot(d["ns"], d["reference"], "k--", alpha=0.7, lw=1.2,
            label=r"$O(n^{-1/2})$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$|\hat{\alpha} - \alpha^*|$")
    ax.set_ylim(bottom=2.5e-4)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="lower left", ncol=2, columnspacing=0.5, handlelength=1.2,
              borderpad=0.3, labelspacing=0.2, handletextpad=0.4)


def make_figure(results, path):
    """Draws Figure 6, alpha and v errors against n, and returns ``path``."""
    from matplotlib.ticker import LogLocator

    plt = pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    panels = (("alpha_err", r"$|\hat\alpha - \alpha^*|$",
               r"(a) $\alpha$ estimation error"),
              ("v_err", r"$\|\hat v - v^*\|$", r"(b) $v$ estimation error"))
    for ax, (key, ylabel, title) in zip(axes, panels):
        for name in ORDER:
            ns, mean, sem = _curve(results, name, key)
            ax.errorbar(ns, mean, yerr=sem, label=LABELS[name],
                        marker=MARKERS[name], color=COLORS[name], capsize=3,
                        linewidth=1.5, markersize=6)
        ns, ref = _reference(results, key)
        ax.plot(ns, ref, "k--", alpha=0.6, linewidth=1.5,
                label=r"$O(n^{-1/2})$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$n$ (training sample size)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=11, loc="lower left", ncol=2, columnspacing=0.6)
        ax.tick_params(axis="both", which="major", labelsize=11)
        ax.grid(True, alpha=0.3, which="both")
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _fmt(slope_r2):
    """Formats a (slope, R^2) pair."""
    return f"{slope_r2[0]:+.2f} (R2 {slope_r2[1]:.2f})"


def report(results):
    """Prints the mean errors and the log-log slopes."""
    ns = results["ns"]
    print(f"Convergence rate, {results['trials']} trials per n, "
          f"n_test = {results['n_test']}")
    for name in ORDER:
        e = results["errors"][name]
        a = " ".join(f"{np.mean(e[str(n)]['alpha_err']):.5f}" for n in ns)
        v = " ".join(f"{np.mean(e[str(n)]['v_err']):.5f}" for n in ns)
        print(f"  {name:4s} alpha_err {a}\n       v_err     {v}")
    print(f"\n  log-log slopes (theory -0.50), n = {ns}")
    print(f"  {'':4s} {'alpha':>15s} {'v':>15s}")
    for name in ORDER:
        s = results["slopes"][name]
        print(f"  {name:4s} {_fmt(s['alpha']):>15s} {_fmt(s['v']):>15s}")
    print(f"\n  wall time {results['seconds']:.1f} s")


if __name__ == "__main__":
    args = parse_args(
        "Convergence rate of the ROLS coefficients (Table 6, Figure 6).",
        TRIALS)
    res = run(args)
    tag = "_quick" if args.quick else ""
    print("wrote", save_results("convergence", res, args))
    print("wrote", save_text(f"table_convergence{tag}.tex", table_tex(res)))
    print("wrote", make_figure(res, figure_path(f"convergence_rate{tag}.pdf")))
    report(res)

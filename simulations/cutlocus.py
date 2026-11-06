"""Sweeps the predictor spread toward the cut locus on S^2.

Produces Figures 5 and 2(a) of the paper.

ROLS test RMSE as the spread r0 of the manifold predictor grows, with one
manifold predictor and one scalar covariate,

    Log_Ybar(Y_i) = alpha * PT_{Xbar -> Ybar}(Log_Xbar(X_i)) + z_i v + eps_i,

where Ybar = e3, Xbar = Exp_e3((0.3, 0.4, 0)) at distance 0.5, alpha 0.5,
v = (0.2, -0.3, 0), eps ~ N(0, 0.08^2 I_3) projected onto T_Ybar and
z ~ N(0, 1). Predictor points are X = Exp_Xbar(r u) with u uniform on the unit
circle of T_Xbar and r = min(|N(0, r0^2)|, pi - 0.01). The response tangent xi
is clipped to length pi - 0.05 before Exp. Each trial draws n = 300 training
and 100 test points, with 50 trials at each of the 16 spreads r0 in
{0.1, ..., 3.0}.

Panel (a) of Figure 5 is RMSE against r0. Panel (b) compares the empirical
ratio RMSE(r_min)^2 / RMSE(r0)^2 with the Hessian comparison factor
eta(1, r0) = r0 / tan(r0).

Usage, from the repository root:

    python -m simulations.cutlocus [--quick] [--trials N] [--workers K]
                                   [--seed S]
"""

import warnings

import numpy as np

from ROLS import Sphere, fit, frechet_rmse
from simulations._common import (Timer, figure_path, parse_args, pmap, pyplot,
                                 rng_for, save_results)

SPREADS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.3,
           2.5, 2.8, 3.0)
N, N_TEST, TRIALS = 300, 100, 50

ALPHA = 0.5
NOISE = 0.08
Y_BAR = np.array([0.0, 0.0, 1.0])
X_BAR = Sphere(3).expmap(Y_BAR, np.array([0.3, 0.4, 0.0]))[0]
V = np.array([0.2, -0.3, 0.0])


def generate(m, r0, n, rng):
    """Draws (X, z, Y) at spread ``r0``."""
    g = rng.standard_normal((n, 3))
    g -= np.outer(g @ X_BAR, X_BAR)
    u = g / np.linalg.norm(g, axis=1, keepdims=True)
    r = np.minimum(np.abs(rng.normal(0.0, r0, n)), np.pi - 0.01)
    X = m.expmap(X_BAR, u * r[:, None])
    z = rng.standard_normal(n)
    eps = NOISE * rng.standard_normal((n, 3))
    eps -= np.outer(eps @ Y_BAR, Y_BAR)
    xi = (ALPHA * m.transport(X_BAR, Y_BAR, m.logmap(X_BAR, X))
          + z[:, None] * V + eps)
    xi -= np.outer(xi @ Y_BAR, Y_BAR)
    nx = np.linalg.norm(xi, axis=1, keepdims=True)
    xi = np.where(nx > np.pi - 0.05,
                  xi / np.maximum(nx, 1e-12) * (np.pi - 0.05), xi)
    return X, z, m.expmap(Y_BAR, xi)


def _trial(task):
    """Returns the test RMSE and largest d(X, Xbar) of one replication.

    Beyond the convexity radius the data may have no unique Frechet mean, so
    the warning that ``ROLS.fit`` issues in that case is silenced.
    """
    i, trial, seed = task
    m, r0 = Sphere(3), SPREADS[i]
    X, z, Y = generate(m, r0, N, rng_for(seed, i, trial, 0))
    Xt, zt, Yt = generate(m, r0, N_TEST, rng_for(seed, i, trial, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f = fit(m, Y, [X], z)
    rmse = frechet_rmse(m, Yt, f.predict([Xt], zt))
    return rmse, float(m.dist(f.X_bars[0][None], X).max())


def run(args):
    """Runs the sweep and returns the results as a nested dict."""
    tasks = [(i, t, args.seed)
             for i in range(len(SPREADS)) for t in range(args.trials)]
    with Timer() as clock:
        out = pmap(_trial, tasks, args.workers)
    rmse = {str(r0): [] for r0 in SPREADS}
    spread = {str(r0): [] for r0 in SPREADS}
    for (i, _, _), (r, s) in zip(tasks, out):
        rmse[str(SPREADS[i])].append(r)
        spread[str(SPREADS[i])].append(s)
    return {"spreads": list(SPREADS), "n": N, "n_test": N_TEST,
            "trials": args.trials, "rmse": rmse, "max_spread": spread,
            "seconds": clock.seconds}


PLOT_FLOOR = 0.01


def eta(r0, kappa=1.0):
    """Returns the Hessian comparison factor eta(kappa, r0).

    eta(kappa, r0) = sqrt(kappa) r0 / tan(sqrt(kappa) r0) is 1 at r0 = 0,
    reaches 0 at the convexity radius pi / (2 sqrt(kappa)) and is negative
    beyond. The figures clip it at ``PLOT_FLOOR``.

    Args:
        r0: Spread, a scalar or an array.
        kappa: Upper bound on the sectional curvature.

    Returns:
        An array of the shape of ``np.atleast_1d(r0)``.
    """
    r = np.sqrt(kappa) * np.atleast_1d(np.asarray(r0, float))
    safe = np.where(r > 0, r, 1.0)
    return np.where(r > 0, safe / np.tan(safe), 1.0)


def summary(results):
    """Returns the spreads and the mean and sd of the RMSE at each spread."""
    spreads = np.asarray(results["spreads"], float)
    vals = [np.asarray(results["rmse"][str(r0)], float)
            for r0 in results["spreads"]]
    return (spreads, np.array([v.mean() for v in vals]),
            np.array([v.std() for v in vals]))


def headline_data(results):
    """Returns the data of Figure 2(a), the RMSE curve and eta(1, r0)."""
    spreads, mean, sd = summary(results)
    grid = np.linspace(0.05, np.pi - 1e-3, 200)
    return {"spreads": spreads, "mean": mean, "sd": sd, "eta_r": grid,
            "eta": np.maximum(eta(grid), PLOT_FLOOR)}


def headline_panel(ax, results):
    """Draws Figure 2(a) on ``ax`` and returns the twin axis of eta."""
    from matplotlib.lines import Line2D

    d = headline_data(results)
    emp, th = "#1f4f8a", "#b2182b"
    ax.plot(d["spreads"], d["mean"], "o-", color=emp, lw=1.5, ms=4.0,
            mfc="white", mew=1.1, zorder=5)
    ax.fill_between(d["spreads"], d["mean"] - d["sd"], d["mean"] + d["sd"],
                    alpha=0.18, color=emp, zorder=2)
    ax.axvline(np.pi / 2, color=th, ls="--", lw=1.4, alpha=0.9, zorder=3)
    ax.axvline(np.pi, color="#4a148c", ls="-.", lw=1.4, alpha=0.9, zorder=3)
    ax.text(np.pi / 2 - 0.04, 0.96, r"$\pi/2$", color=th, ha="right", va="top",
            transform=ax.get_xaxis_transform())
    ax.text(np.pi - 0.04, 0.80, r"$\pi$", color="#4a148c", ha="right",
            va="top", transform=ax.get_xaxis_transform())
    ax2 = ax.twinx()
    ax2.plot(d["eta_r"], d["eta"], "--", color=th, lw=1.5, alpha=0.95,
             zorder=4)
    ax2.set_ylim(-0.05, 1.10)
    ax2.set_ylabel(r"$\eta(\kappa, r_0)$", color=th, labelpad=2)
    ax2.tick_params(axis="y", colors=th)
    ax2.spines["right"].set_color(th)
    ax.set_xlabel(r"$r_0$ on $\mathbb{S}^2$", labelpad=2)
    ax.set_ylabel("Fréchet RMSE", color=emp, labelpad=2)
    ax.tick_params(axis="y", colors=emp)
    ax.spines["left"].set_color(emp)
    ax.set_xlim(0, 3.2)
    ax.set_ylim(0, max(1.05, float(d["mean"].max() + 0.05)))
    ax.grid(True, alpha=0.3, zorder=0)
    ax.legend(handles=[Line2D([0], [0], color=emp, marker="o", lw=1.5, ms=4.0,
                              mfc="white", mew=1.1, label="RMSE"),
                       Line2D([0], [0], color=th, ls="--", lw=1.5,
                              label=r"$\eta$")],
              loc="center right", bbox_to_anchor=(0.96, 0.30), frameon=True,
              edgecolor="0.75", handlelength=1.2, borderpad=0.3,
              labelspacing=0.25)
    return ax2


def make_figure(results, path):
    """Draws Figure 5, RMSE and eta against r0, and returns ``path``."""
    plt = pyplot()
    spreads, mean, sd = summary(results)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    xlim = (0, 3.25)
    blue, red = "#2166ac", "#b2182b"

    ax1.plot(spreads, mean, "o-", color=blue, linewidth=2, markersize=5,
             markerfacecolor="white", markeredgewidth=1.5, zorder=5,
             label="ROLS test RMSE")
    ax1.fill_between(spreads, mean - sd, mean + sd, alpha=0.15, color=blue)
    ax1.axvline(np.pi / 2, color=red, linestyle="--", linewidth=1.6,
                alpha=0.85, label=r"convexity radius $\pi/2$")
    ax1.axvline(np.pi, color="gray", linestyle=":", linewidth=1.3, alpha=0.7,
                label=r"$\mathrm{inj}(S^2) = \pi$")
    ax1.set_xlabel(r"Data spread $r_0$", fontsize=12)
    ax1.set_ylabel("Fréchet RMSE", fontsize=12)
    ax1.set_title(r"(a) Cut-Locus Challenge on $S^2$", fontsize=12)
    ax1.legend(fontsize=11, loc="upper left")
    ax1.set_xlim(*xlim)
    ax1.grid(True, alpha=0.3)

    ratio = np.clip((mean[0] / mean) ** 2, 0, 1.2)
    ax2.plot(spreads, np.maximum(eta(spreads), PLOT_FLOOR), "s--", color=red,
             linewidth=2, markersize=5, label=r"Theory: $\eta(\kappa, r_0)$",
             zorder=4)
    ax2.plot(spreads, ratio, "o-", color=blue, linewidth=2, markersize=5,
             markerfacecolor="white", markeredgewidth=1.5,
             label=(r"Empirical: $\mathrm{RMSE}(r_{\min})^2\, /\, "
                    r"\mathrm{RMSE}(r_0)^2$"),
             zorder=5)
    ax2.axvline(np.pi / 2, color=red, linestyle="--", linewidth=1.6,
                alpha=0.85, label=r"convexity radius $\pi/2$")
    ax2.axvline(np.pi, color="gray", linestyle=":", linewidth=1.3, alpha=0.7,
                label=r"$\mathrm{inj}(S^2) = \pi$")
    ax2.axhline(0, color="gray", linewidth=0.5)
    ax2.set_xlabel(r"Data spread $r_0$", fontsize=12)
    ax2.set_ylabel(r"$\eta(\kappa, r_0)$", fontsize=12)
    ax2.set_title("(b) Curvature Comparison Factor", fontsize=12)
    ax2.set_ylim(-0.05, 1.15)
    ax2.set_xlim(*xlim)
    ax2.legend(fontsize=11, loc="upper right")
    ax2.grid(True, alpha=0.3)
    for ax in (ax1, ax2):
        ax.tick_params(axis="both", which="major", labelsize=11)

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def report(results):
    """Prints the RMSE, the largest spread and eta at each r0."""
    spreads, mean, sd = summary(results)
    print(f"Cut-locus sweep on S^2, n = {results['n']}, "
          f"n_test = {results['n_test']}, {results['trials']} trials per "
          "spread")
    print(f"  {'r0':>5s} {'RMSE':>16s} {'max d(X,Xbar)':>14s} {'eta':>6s}")
    for r0, mu, s in zip(spreads, mean, sd):
        md = np.mean(results["max_spread"][str(float(r0))])
        print(f"  {r0:5.2f} {mu:8.4f} +- {s:.4f} {md:14.2f} "
              f"{eta(r0)[0]:6.2f}")
    print(f"\n  wall time {results['seconds']:.1f} s")


if __name__ == "__main__":
    args = parse_args("Cut-locus sweep on S^2 (Figure 5).", TRIALS)
    res = run(args)
    print("wrote", save_results("cutlocus", res, args))
    print("wrote", make_figure(res, figure_path(
        f"cutlocus_sphere{'_quick' if args.quick else ''}.pdf")))
    report(res)

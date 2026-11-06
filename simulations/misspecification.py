"""Compares prediction under tangent-space misspecification on S^2.

Produces Figure 9 of the paper.

With Xbar = e1 and Ybar = e3,

    Log_Ybar(Y_i) = alpha u_i + z_i v + gamma ||u_i||^2 nhat + eps_i,
    u_i = PT_{Xbar -> Ybar} Log_Xbar(X_i),

where alpha = 0.5, v = 0.4 e1 + 0.3 e2, nhat = 0.6 e1 + 0.8 e2 and
eps_i ~ N(0, 0.05^2 I_2) in T_Ybar. The predictor is
X_i = Exp_Xbar(0.6 |g_i| d_i), with d_i a uniform unit direction in T_Xbar and
g_i ~ N(0, 1), and z_i ~ N(0, 1). gamma runs over {0, 0.1, ..., 0.6}, with
n = 300 training and 100 test points and 30 trials. Each trial reuses the same
draws at every gamma.

The methods are ROLS, nonlinear Frechet regression
(``ROLS.baselines.nonlinear_frechet``), geodesic regression on z alone and the
Frechet mean, which is saved but not plotted. All are scored by held-out
Frechet RMSE.

Usage, from the repository root:

    python -m simulations.misspecification [--quick] [--trials N] [--workers K]
                                           [--seed S]
"""

import numpy as np

from ROLS import Sphere, baselines, fit, frechet_rmse
from simulations._common import (Timer, figure_path, parse_args, pmap, pyplot,
                                 rng_for, save_results)

S2 = Sphere(3)
Y_BAR = np.array([0.0, 0.0, 1.0])
X_BAR = np.array([1.0, 0.0, 0.0])
T1 = np.array([1.0, 0.0, 0.0])
T2 = np.array([0.0, 1.0, 0.0])
ALPHA = 0.5
V_TRUE = 0.4 * T1 + 0.3 * T2
NHAT = (0.6 * T1 + 0.8 * T2) / np.linalg.norm(0.6 * T1 + 0.8 * T2)
NOISE = 0.05
SPREAD = 0.6
GAMMAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
N_TRAIN, N_TEST, TRIALS = 300, 100, 30
METHODS = ("rols", "nl_frechet", "geodesic", "frechet_mean")


def simulate(n, gamma, rng):
    """Draws (X, z, Y). The random draws do not depend on ``gamma``."""
    dirs = rng.standard_normal((n, 3))
    dirs -= (dirs @ X_BAR)[:, None] * X_BAR[None, :]
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    X = S2.expmap(X_BAR, dirs * SPREAD * np.abs(rng.standard_normal((n, 1))))
    z = rng.standard_normal(n)
    eps = NOISE * rng.standard_normal((n, 2)) @ np.stack([T1, T2])
    u = S2.transport(X_BAR, Y_BAR, S2.logmap(X_BAR, X))
    xi = (ALPHA * u + z[:, None] * V_TRUE[None, :]
          + gamma * (u ** 2).sum(1, keepdims=True) * NHAT[None, :] + eps)
    return X, z[:, None], S2.expmap(Y_BAR, xi)


def _trial(task):
    """Returns the held-out RMSE of every method on one replication."""
    seed, gamma, trial, n, n_test = task
    X, z, Y = simulate(n, gamma, rng_for(seed, 0, trial))
    X_te, z_te, Y_te = simulate(n_test, gamma, rng_for(seed, 1, trial))
    f = fit(S2, Y, [X], z)
    preds = {
        "rols": f.predict([X_te], z_te),
        "nl_frechet": baselines.nonlinear_frechet(S2, Y, [X], z, [X_te], z_te),
        "geodesic": baselines.geodesic_regression(S2, Y, z, z_te),
        "frechet_mean": baselines.frechet_mean(S2, Y, n_test),
    }
    return {k: frechet_rmse(S2, Y_te, v) for k, v in preds.items()}


def run(args):
    """Runs the experiment and returns the results as a nested dict."""
    tasks = [(args.seed, g, t, N_TRAIN, N_TEST)
             for g in GAMMAS for t in range(args.trials)]
    out = pmap(_trial, tasks, args.workers)
    T = args.trials
    rmse = {k: [[o[k] for o in out[i * T:(i + 1) * T]]
                for i in range(len(GAMMAS))]
            for k in METHODS}
    return {"gammas": GAMMAS, "n": N_TRAIN, "n_test": N_TEST,
            "trials": args.trials, "rmse": rmse,
            "mean": {k: [float(np.mean(r)) for r in v]
                     for k, v in rmse.items()},
            "se": {k: [float(np.std(r) / np.sqrt(len(r))) for r in v]
                   for k, v in rmse.items()}}


def plot(results, path):
    """Draws Figure 9 and returns ``path``."""
    plt = pyplot()
    g, mu, se = results["gammas"], results["mean"], results["se"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(g, mu["rols"], yerr=se["rols"], marker="*", color="#d62728",
                linewidth=2, markersize=10, capsize=3, label="ROLS", zorder=10)
    ax.errorbar(g, mu["nl_frechet"], yerr=se["nl_frechet"], marker="P",
                color="#8c564b", linewidth=1.5, markersize=7, capsize=3,
                label="NL-Fréchet")
    ax.errorbar(g, mu["geodesic"], yerr=se["geodesic"], marker="s",
                color="#1f77b4", linewidth=1.5, markersize=6, capsize=3,
                label="Geodesic")
    for key, i, dx, dy in (("rols", 0, 0.06, -0.02),
                           ("nl_frechet", 0, 0.06, 0.015),
                           ("rols", -1, -0.15, -0.025)):
        col = "#d62728" if key == "rols" else "#8c564b"
        ax.annotate(f"{mu[key][i]:.3f}", xy=(g[i], mu[key][i]),
                    xytext=(g[i] + dx, mu[key][i] + dy), fontsize=7, color=col,
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.5))
    ax.set_xlabel(r"Misspecification strength $\gamma$", fontsize=12)
    ax.set_ylabel("Fréchet RMSE", fontsize=12)
    ax.set_title(r"Robustness under model misspecification ($\mathbb{S}^2$)",
                 fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def report(results):
    """Prints the mean RMSE of every method at each gamma."""
    mu, se = results["mean"], results["se"]
    print(f"Figure 9, S^2, n={results['n']}, {results['trials']} trials. "
          "Mean Frechet RMSE (SE).")
    print(f"{'gamma':>5} | {'ROLS':>15} | {'NL-Frechet':>15} | "
          f"{'Geodesic':>15} | {'Frechet mean':>12}")
    for i, g in enumerate(results["gammas"]):
        cells = [f"{mu[k][i]:.4f} ({se[k][i]:.4f})"
                 for k in ("rols", "nl_frechet", "geodesic")]
        print(f"{g:5.1f} | " + " | ".join(cells)
              + f" | {mu['frechet_mean'][i]:12.4f}")
    r0, n0 = mu["rols"][0], mu["nl_frechet"][0]
    r6, n6 = mu["rols"][-1], mu["nl_frechet"][-1]
    print(f"ROLS vs NL-Frechet reduction: gamma=0 {1 - r0 / n0:.0%}, "
          f"gamma={results['gammas'][-1]} {1 - r6 / n6:.0%}")


if __name__ == "__main__":
    args = parse_args(
        "Figure 9: ROLS, NL-Frechet and geodesic RMSE under misspecification",
        TRIALS)
    with Timer() as tm:
        res = run(args)
    res["seconds"] = tm.seconds
    print("wrote", save_results("misspecification", res, args))
    print("wrote", plot(res, figure_path(
        "misspecification_quick.pdf" if args.quick
        else "misspecification.pdf")))
    report(res)

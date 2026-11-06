"""Measures Wald calibration and the error rate in higher tangent dimension.

Produces Table 12 of the paper and the rate quoted in its text.

Sym+(k) with the affine-invariant metric, one manifold predictor and one scalar
covariate,

    X_i = Exp_{1.2 I}(0.35 S_i),    z_i ~ N(0, 1),
    Y_i = Exp_{1.5 I}(alpha* PT_{1.2 I -> 1.5 I} Log_{1.2 I}(X_i)
                      + z_i V* + tau E_i),

where S_i, E_i and D are symmetrized standard Gaussian matrices, (G + G^T) / 2,
and V* = v_scale D / |D|_F with D drawn afresh for every data set.

Calibration (the table). k = 3, 6, 10, so d = k(k+1)/2 = 6, 21, 55; tau = 0.5;
joint null alpha* = 0, V* = 0; n = max(round(m d), d + 5) for m = 1.5, 3, 6, 12,
which gives the grids in ``N_GRID``; 500 trials per cell. The model-based Wald
statistic W for v is referred to chi^2_d and, through W / d, to F(d, df_resid)
with df_resid = n d - (1 + d), the pooled residual degrees of freedom of
``ROLSFit``.

Rate. d = 21, alpha* = 0.5, |V*|_F = 0.5, tau = 0.5,
n in {60, 120, 240, 480, 960}, 83 trials per n. The error is
|beta_hat - beta*| = (|alpha_hat - alpha*|^2 + |v_hat - v*|^2)^{1/2}, with v*
expressed in the chart at the true center 1.5 I and v_hat in the chart at the
fitted Frechet mean.

Usage, from the repository root:

    python -m simulations.high_dimension [--quick] [--trials N] [--workers K]
                                         [--seed S]

``--seed S`` is added to the seed of every trial. Seeds that are multiples of
10**9 give independent runs.
"""

import numpy as np
from scipy import stats

from ROLS import SPD, fit
from simulations._common import (Timer, loglog_slope, parse_args, pmap,
                                 save_results, save_text)

Y_CENTER, X_CENTER, SPREAD = 1.5, 1.2, 0.35
KS = (3, 6, 10)
N_GRID = {3: (11, 18, 36, 72), 6: (32, 63, 126, 252), 10: (82, 165, 330, 660)}
NOISE = 0.5
RATE_K, RATE_NS, RATE_ALPHA, RATE_V = 6, (60, 120, 240, 480, 960), 0.5, 0.5
LEVEL = 0.05


def _sym_gauss(rng, shape):
    """Draws symmetrized standard Gaussian matrices."""
    G = rng.standard_normal(shape)
    return 0.5 * (G + np.swapaxes(G, -1, -2))


def generate(m, n, alpha, v_scale, noise, rng, V_true=None):
    """Draws one data set from the Sym+(k) process.

    Args:
        m: ``SPD(k)`` manifold.
        n: Sample size.
        alpha: Coefficient of the manifold predictor.
        v_scale: Frobenius norm of V*.
        noise: Noise scale tau.
        rng: ``numpy`` generator.
        V_true: Ambient tangent coefficient V*. By default it is drawn in a
            random direction with norm ``v_scale``.

    Returns:
        A tuple (X, z, Y, V*), with V* the ambient tangent coefficient.
    """
    k = m.k
    Y_bar, X_bar = Y_CENTER * np.eye(k), X_CENTER * np.eye(k)
    if V_true is None:
        D = _sym_gauss(rng, (k, k))
        V_true = v_scale * D / np.linalg.norm(D)
    X = m.expmap(X_bar, SPREAD * _sym_gauss(rng, (n, k, k)))
    z = rng.standard_normal(n)
    u = m.transport(X_bar, Y_bar, m.logmap(X_bar, X))
    eps = noise * _sym_gauss(rng, (n, k, k))
    xi = alpha * u + z[:, None, None] * V_true + eps
    return X, z, m.expmap(Y_bar, xi), V_true


def v_wald(f):
    """Returns the model-based Wald statistic of the scalar covariate."""
    C = f.covariance("model")
    sl = slice(f.p, f.p + f.d)
    v = np.asarray(f.V[0], float)
    return float(v @ np.linalg.solve(C[sl, sl], v))


def null_trial(k, n, rng):
    """Returns (W, p_chi2, p_F) for one data set under the joint null."""
    m = SPD(k)
    X, z, Y, _ = generate(m, n, 0.0, 0.0, NOISE, rng)
    f = fit(m, Y, [X], z)
    w = v_wald(f)
    return w, float(stats.chi2.sf(w, f.d)), float(stats.f.sf(w / f.d, f.d,
                                                             f.df_resid))


def rate_trial(k, n, rng):
    """Returns (|alpha error|, |v error|) at alpha* = 0.5, |V*|_F = 0.5."""
    m = SPD(k)
    X, z, Y, V_true = generate(m, n, RATE_ALPHA, RATE_V, NOISE, rng)
    f = fit(m, Y, [X], z)
    v_true = m.coords(Y_CENTER * np.eye(k), V_true)[0]
    return (abs(float(f.alphas[0]) - RATE_ALPHA),
            float(np.linalg.norm(f.V[0] - v_true)))


def _null_task(task):
    """Runs one calibration replication."""
    k, n, seed, t = task
    return null_trial(k, n, np.random.default_rng(42 + seed + 7919 * k + t))


def _rate_task(task):
    """Runs one rate replication."""
    k, n, seed, t = task
    return rate_trial(k, n, np.random.default_rng(42 + seed + 104729 * k + t))


def run(args):
    """Runs the calibration grid and the rate, and returns the results."""
    T = args.trials
    tasks = [(k, n, args.seed, t)
             for k in KS for n in N_GRID[k] for t in range(T)]
    out = iter(pmap(_null_task, tasks, args.workers, chunksize=8))
    calib = {}
    for k in KS:
        cell = {"d": k * (k + 1) // 2, "n": list(N_GRID[k]), "typeI_chi2": [],
                "typeI_F": [], "mean_wald": [], "trials": T}
        for _ in N_GRID[k]:
            r = np.array([next(out) for _ in range(T)])
            cell["typeI_chi2"].append(float((r[:, 1] < LEVEL).mean()))
            cell["typeI_F"].append(float((r[:, 2] < LEVEL).mean()))
            cell["mean_wald"].append(float(r[:, 0].mean()))
        calib[str(k)] = cell

    T_rate = T if args.quick else max(60, T // 6)
    tasks = [(RATE_K, n, args.seed, t)
             for n in RATE_NS for t in range(T_rate)]
    out = iter(pmap(_rate_task, tasks, args.workers, chunksize=4))
    rate = {"k": RATE_K, "d": RATE_K * (RATE_K + 1) // 2, "n": list(RATE_NS),
            "beta_err_mean": [], "beta_err_se": [], "alpha_err_mean": [],
            "v_err_mean": [], "trials": T_rate}
    for _ in RATE_NS:
        r = np.array([next(out) for _ in range(T_rate)])
        beta = np.sqrt((r ** 2).sum(1))
        rate["beta_err_mean"].append(float(beta.mean()))
        rate["beta_err_se"].append(float(beta.std() / np.sqrt(len(beta))))
        rate["alpha_err_mean"].append(float(r[:, 0].mean()))
        rate["v_err_mean"].append(float(r[:, 1].mean()))
    rate["loglog_slope"], rate["r2"] = loglog_slope(RATE_NS,
                                                    rate["beta_err_mean"])
    return {"calibration": calib, "rate": rate, "noise": NOISE, "level": LEVEL}


def _rate3(x):
    """Formats a rate for the LaTeX table."""
    return f"{x:.3f}".lstrip("0") if x < 1 else f"{x:.3f}"


def latex_table(res):
    """Returns the LaTeX source of Table 12."""
    calib, rate = res["calibration"], res["rate"]
    T = calib[str(KS[0])]["trials"]
    lines = [r"\begin{table}[h]", r"\centering",
             r"\caption{Type~I error of the $v$-Wald test on "
             r"$\mathrm{Sym}^+(k)$ at nominal "
             rf"level ${LEVEL}$ (${T}$ trials), with the $\chi^2_d$ and $F$ "
             r"references.}",
             r"\label{tab:highdim}", r"\footnotesize",
             r"\begin{tabular}{lcccc}", r"\toprule",
             r"$d$ ($\chi^2$ / $F$) & $n\approx1.5d$ & $n\approx3d$ & "
             r"$n\approx6d$ & $n\approx12d$ \\", r"\midrule"]
    for k in KS:
        c = calib[str(k)]
        cells = " & ".join(f"${_rate3(a)}$ / ${_rate3(b)}$"
                           for a, b in zip(c["typeI_chi2"], c["typeI_F"]))
        lines.append(f"${c['d']}$ & {cells} \\\\")
    lines += [r"\midrule",
              rf"\multicolumn{{5}}{{l}}{{$d={rate['d']}$: log--log slope of "
              r"$\|\hat\beta-\beta^*\|$ over $n\in\{"
              + ",".join(map(str, rate["n"]))
              + rf"\}}$ is ${rate['loglog_slope']:.3f}$ "
              rf"(${rate['trials']}$ trials).}} \\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def report(res):
    """Prints the Type I rates and the parameter errors."""
    calib, rate = res["calibration"], res["rate"]
    T = calib[str(KS[0])]["trials"]
    se = np.sqrt(LEVEL * (1 - LEVEL) / T)
    print(f"Table 12, Type I of the v-Wald test at {LEVEL} ({T} trials per "
          f"cell, MC s.e. {se:.3f})")
    print(f"{'d':>3} {'n':>4}  {'chi2':>5}  {'F':>5}  {'mean W':>7}")
    for k in KS:
        c = calib[str(k)]
        for i, n in enumerate(c["n"]):
            print(f"{c['d']:>3} {n:>4}  {c['typeI_chi2'][i]:5.3f}  "
                  f"{c['typeI_F'][i]:5.3f}  {c['mean_wald'][i]:7.2f}")
    print(f"\nParameter error at d = {rate['d']} ({rate['trials']} trials per "
          "n)")
    print(f"{'n':>4}  {'|b-b*|':>7} {'s.e.':>6}")
    for i, n in enumerate(rate["n"]):
        print(f"{n:>4}  {rate['beta_err_mean'][i]:7.4f} "
              f"{rate['beta_err_se'][i]:6.4f}")
    print(f"log-log slope {rate['loglog_slope']:.3f} (R^2 {rate['r2']:.3f})")


if __name__ == "__main__":
    args = parse_args(__doc__.splitlines()[0], trials=500)
    with Timer() as timer:
        res = run(args)
    res["seconds"] = timer.seconds
    tag = "_quick" if args.quick else ""
    print("wrote", save_results("high_dimension", res, args))
    print("wrote", save_text(f"table_high_dimension{tag}.tex",
                             latex_table(res)))
    report(res)

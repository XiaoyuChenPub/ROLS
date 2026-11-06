"""Measures inference under tangent-space misspecification.

Produces Tables 8, 9 and 10 of the paper.

Xbar = e1, Ybar = e3, X_i = Exp_Xbar(0.6 |g_i| d_i) with d_i a uniform unit
direction in T_Xbar, two scalar covariates z1, z2 ~ N(0, 1), a real effect
v1 = 0.4 e1 + 0.3 e2 and the true nulls alpha = 0 and v2 = 0. u_i is the
parallel transport of Log_Xbar(X_i) to Ybar, eps_i is isotropic Gaussian noise
in T_Ybar and nhat = 0.6 e1 + 0.8 e2. Every fit is ROLS with the manifold
predictor X and the scalars (z1, z2).

Table 8. n = 300, 500 trials, noise sd 0.10, gamma in {0, 0.1, ..., 0.6},
    Log_Ybar(Y_i) = z1_i v1 + gamma ||u_i||^2 nhat + eps_i.
    Rows are the Type I error of the model-based t test of alpha = 0, of the
    same test with the HC0 sandwich standard error, and the rejection rate of
    the lack-of-fit F test (``lack_of_fit``).

Table 9. n = 200, 1500 trials, noise sd 0.15, gamma in {0, 0.1, 0.2, 0.3},
    Log_Ybar(Y_i) = z1_i v1 + gamma h_i nhat + eps_i,
    with h centered and standardized to unit variance. "in u" uses
    h = (||u||^2 - m) / s, with m and s the mean and sd of ||u||^2, and "in z2"
    uses h = (z2^2 - 1) / sqrt 2. Rows compare model-based and HC3 tests of
    alpha = 0 and v2 = 0.

Table 10. Lack-of-fit power on the Table 9 data, against h in the test basis
    (quadratic in u) and outside it, h = z1^3 / sqrt 15 and
    h = (u . e1) z1 / s1, with s1 the sd of u . e1.

The moments m, s and s1 are estimated once from 400000 draws of the predictor,
which gives about 0.361, 0.510 and 0.425. All designs at a given (gamma, trial)
share the same draws, so the gamma = 0 columns coincide across designs.

Usage, from the repository root:

    python -m simulations.robust_inference [--quick] [--trials N]
                                            [--trials8 N] [--workers K]
                                            [--seed S]

``--seed S`` is added to the seed of every trial and does not change the
moments. Seeds that are multiples of 10**9 give independent runs.
"""

import numpy as np
from scipy import stats

from ROLS import Sphere, fit
from simulations._common import Timer, parse_args, pmap, save_results, save_text

S2 = Sphere(3)
Y_BAR = np.array([0.0, 0.0, 1.0])
X_BAR = np.array([1.0, 0.0, 0.0])
T1 = np.array([1.0, 0.0, 0.0])
T2 = np.array([0.0, 1.0, 0.0])
V1_TRUE = 0.4 * T1 + 0.3 * T2
NHAT = (0.6 * T1 + 0.8 * T2) / np.linalg.norm(0.6 * T1 + 0.8 * T2)
SPREAD = 0.6
LEVEL = 0.05

T8 = dict(n=300, trials=500, noise=0.10,
          gammas=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
T9 = dict(n=200, trials=1500, noise=0.15, gammas=[0.0, 0.1, 0.2, 0.3])
DESIGNS = ("in_u", "in_z2", "cubic", "interaction")
TABLE9_ROWS = (("in_u", "alpha_model"), ("in_u", "alpha_HC3"),
               ("in_u", "v2_model"), ("in_z2", "v2_model"),
               ("in_z2", "v2_HC3"), ("in_z2", "alpha_model"))
TABLE10_DESIGNS = ("in_u", "cubic", "interaction")


def lack_of_fit(manifold, Y, X_list, z, null_fit=None):
    """Runs the targeted lack-of-fit F test of the tangent-linear mean.

    Appends one quadratic term per predictor as an extra scalar covariate,
    ||u_ij||^2 for each manifold predictor and ztilde_ik^2 for each scalar
    covariate, refits, and compares residual sums of squares over the stacked
    n*d tangent coordinates with the nested-model F test

        F = ((RSS_0 - RSS_1) / (m d)) / (RSS_1 / (n d - k_1)),

    where m = p + q is the number of added covariates and
    k_1 = p + (q + m) d. The extra columns are standardized. The F reference
    assumes homoskedastic isotropic tangent noise under the null.

    Args:
        manifold: Manifold.
        Y: Responses.
        X_list: List of p batches of manifold predictors.
        z: (n, q) scalar covariates.
        null_fit: Fit of the null model, computed when it is not given.

    Returns:
        A dict with ``F``, its ``pvalue``, the degrees of freedom ``df1`` and
        ``df2``, and ``frac`` = (RSS_0 - RSS_1) / RSS_0, the share of residual
        variation that the quadratic terms absorb.
    """
    f0 = fit(manifold, Y, X_list, z) if null_fit is None else null_fit
    G = np.column_stack([(Uj ** 2).sum(1) for Uj in f0.U] + [f0.Zt ** 2])
    G = (G - G.mean(0)) / (G.std(0) + 1e-12)
    f1 = fit(manifold, Y, X_list, np.column_stack([f0.Zt, G]), reg=f0.reg)
    rss0, rss1 = float((f0.resid ** 2).sum()), float((f1.resid ** 2).sum())
    df1, df2 = G.shape[1] * f0.d, f1.df_resid
    F = max((rss0 - rss1) / df1 / (rss1 / df2), 0.0)
    return {"F": F, "pvalue": float(stats.f.sf(F, df1, df2)), "df1": df1,
            "df2": df2, "frac": (rss0 - rss1) / rss0}


def draw(n, noise, rng):
    """Draws the predictor, the scalar covariates and the tangent noise.

    Args:
        n: Sample size.
        noise: Standard deviation of the tangent noise.
        rng: ``numpy`` generator.

    Returns:
        A tuple (X, z1, z2, u, eps), where u holds the transported predictor
        tangents at Ybar.
    """
    dirs = rng.standard_normal((n, 3))
    dirs -= (dirs @ X_BAR)[:, None] * X_BAR[None, :]
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    X = S2.expmap(X_BAR, dirs * SPREAD * np.abs(rng.standard_normal((n, 1))))
    z1, z2 = rng.standard_normal(n), rng.standard_normal(n)
    eps = noise * rng.standard_normal((n, 2)) @ np.stack([T1, T2])
    u = S2.transport(X_BAR, Y_BAR, S2.logmap(X_BAR, X))
    return X, z1, z2, u, eps


def predictor_moments():
    """Returns the mean and sd of ||u||^2 and the sd of u . e1."""
    _, _, _, u, _ = draw(400000, 0.15, np.random.default_rng(12345))
    norm2 = (u ** 2).sum(1)
    return float(norm2.mean()), float(norm2.std()), float((u @ T1).std())


def omitted(design, u, z1, z2, moments=None):
    """Returns the omitted scalar h of a design.

    Args:
        design: 'raw' for Table 8's ||u||^2, or one of ``DESIGNS``, which are
            standardized.
        u: Transported predictor tangents at Ybar.
        z1: First scalar covariate.
        z2: Second scalar covariate.
        moments: Output of ``predictor_moments``, used by 'in_u' and
            'interaction'.

    Returns:
        The (n,) values of h.
    """
    if design == "raw":
        return (u ** 2).sum(1)
    if design == "in_u":
        return ((u ** 2).sum(1) - moments[0]) / moments[1]
    if design == "in_z2":
        return (z2 ** 2 - 1.0) / np.sqrt(2.0)
    if design == "cubic":
        return z1 ** 3 / np.sqrt(15.0)
    if design == "interaction":
        return (u @ T1) * z1 / moments[2]
    raise ValueError(design)


def tests(X, z1, z2, Y):
    """Fits the null model and returns every p-value the tables use."""
    z = np.column_stack([z1, z2])
    f = fit(S2, Y, [X], z)
    out = {}
    for kind in ("model", "HC0", "HC3"):
        pa, pv = f.pvalues(kind)
        out[f"alpha_{kind}"], out[f"v2_{kind}"] = float(pa[0]), float(pv[1])
    out["lof"] = lack_of_fit(S2, Y, [X], z, null_fit=f)["pvalue"]
    return out


def _trial8(task):
    """Returns the p-values of one replication of Table 8."""
    seed, gi, trial = task
    gamma = T8["gammas"][gi]
    rng = np.random.default_rng(42 + seed + 20261 * trial + int(gamma * 1000))
    X, z1, z2, u, eps = draw(T8["n"], T8["noise"], rng)
    h = omitted("raw", u, z1, z2)
    xi = z1[:, None] * V1_TRUE + gamma * h[:, None] * NHAT + eps
    return tests(X, z1, z2, S2.expmap(Y_BAR, xi))


def _trial9(task):
    """Returns the p-values of one replication of Tables 9 and 10."""
    seed, gi, trial, moments = task
    gamma = T9["gammas"][gi]
    rng = np.random.default_rng(555 + seed + 104729 * trial + int(gamma * 1000))
    X, z1, z2, u, eps = draw(T9["n"], T9["noise"], rng)
    base = z1[:, None] * V1_TRUE + eps
    out = {}
    for des in DESIGNS:
        xi = base + gamma * omitted(des, u, z1, z2, moments)[:, None] * NHAT
        out[des] = tests(X, z1, z2, S2.expmap(Y_BAR, xi))
    return out


def _rates(pvals):
    """Returns the rejection rate at LEVEL for every key of the dicts."""
    return {k: float(np.mean([p[k] < LEVEL for p in pvals])) for k in pvals[0]}


def run(args):
    """Runs Tables 8, 9 and 10 and returns the rejection rates."""
    n8, g8, g9 = args.trials8, T8["gammas"], T9["gammas"]
    tasks = [(args.seed, gi, t) for gi in range(len(g8)) for t in range(n8)]
    out = pmap(_trial8, tasks, args.workers, chunksize=16)
    table8 = [_rates(out[gi * n8:(gi + 1) * n8]) for gi in range(len(g8))]

    T, moments = args.trials, predictor_moments()
    tasks = [(args.seed, gi, t, moments)
             for gi in range(len(g9)) for t in range(T)]
    out = pmap(_trial9, tasks, args.workers, chunksize=16)
    table9 = {des: [_rates([o[des] for o in out[gi * T:(gi + 1) * T]])
                    for gi in range(len(g9))]
              for des in DESIGNS}
    return {"table8": {"n": T8["n"], "trials": n8, "noise_sd": T8["noise"],
                       "gammas": g8, "rates": table8},
            "table9": {"n": T9["n"], "trials": T, "noise_sd": T9["noise"],
                       "gammas": g9, "rates": table9},
            "level": LEVEL}


def _f(x):
    """Formats a rate for a LaTeX table."""
    return "$" + f"{x:.3f}".lstrip("0") + "$"


def write_tables(results, quick=False):
    """Writes the LaTeX rows of Tables 8, 9 and 10 and returns the paths."""
    sfx = "_quick" if quick else ""
    r8, r9 = results["table8"], results["table9"]
    head = "% generated by python -m simulations.robust_inference, {} trials\n"
    rows8 = [("Model Wald", "alpha_model"), (r"Robust $\alpha$", "alpha_HC0"),
             ("Lack-of-fit", "lof")]
    body8 = head.format(r8["trials"]) + "".join(
        f"{lab} & " + " & ".join(_f(r[key]) for r in r8["rates"]) + r" \\"
        + "\n"
        for lab, key in rows8)

    rates = r9["rates"]
    lab = {"in_u": "in $u$", "in_z2": "in $z_2$", "alpha": r"$\alpha$",
           "v2": "$v_2$"}
    lines = []
    for i, (des, key) in enumerate(TABLE9_ROWS):
        if i == 3:
            lines.append(r"\addlinespace")
        blk, kind = key.split("_")
        lines.append(f"{lab[des]} & {lab[blk]} & {kind} & "
                     + " & ".join(_f(r[key]) for r in rates[des]) + r" \\")
    body9 = head.format(r9["trials"]) + "\n".join(lines) + "\n"

    rows10 = [("in_u", r"Quadratic in $u$ (Figure~\ref{fig:misspec}) & yes"),
              ("cubic", r"Cubic in $z_1$ & no"),
              ("interaction", r"$u\times z_1$ interaction & no")]
    body10 = head.format(r9["trials"]) + "".join(
        f"{txt} & " + " & ".join(_f(r["lof"]) for r in rates[des]) + r" \\"
        + "\n"
        for des, txt in rows10)
    return [save_text(f"table_misspec_calibration{sfx}.tex", body8),
            save_text(f"table_hc3_calibration{sfx}.tex", body9),
            save_text(f"table_lof_power{sfx}.tex", body10)]


def report(results):
    """Prints the rejection rates of the three tables."""
    r8, r9 = results["table8"], results["table9"]

    def line(name, rates):
        return f"  {name:26s} " + " ".join(f"{x:.3f}" for x in rates)

    print(f"Table 8, n={r8['n']}, {r8['trials']} trials, gamma={r8['gammas']}")
    for key, name in (("alpha_model", "model Wald alpha"),
                      ("alpha_HC0", "robust alpha (HC0)"),
                      ("lof", "lack-of-fit"),
                      ("v2_model", "model Wald v2"),
                      ("alpha_HC3", "HC3 alpha")):
        print(line(name, [r[key] for r in r8["rates"]]))

    rates = r9["rates"]
    print(f"Table 9, n={T9['n']}, {r9['trials']} trials, gamma={r9['gammas']}")
    for des, key in TABLE9_ROWS:
        print(line(f"{des} {key}", [r[key] for r in rates[des]]))
    print(f"Table 10, lack-of-fit power, n={T9['n']}")
    for des in TABLE10_DESIGNS:
        print(line(des, [r["lof"] for r in rates[des]]))
    se8 = np.sqrt(LEVEL * (1 - LEVEL) / r8["trials"])
    se9 = np.sqrt(LEVEL * (1 - LEVEL) / r9["trials"])
    print(f"Monte Carlo SE of a rate near {LEVEL}: {se8:.4f} (Table 8), "
          f"{se9:.4f} (Tables 9-10)")


def _extra(ap):
    """Adds the arguments specific to this module."""
    ap.add_argument("--trials8", type=int, default=None,
                    help="Table 8 replications (default: min(500, --trials))")


if __name__ == "__main__":
    args = parse_args("Tables 8-10: Type I error and lack-of-fit power under "
                      "misspecification", T9["trials"], _extra)
    if args.trials8 is None:
        args.trials8 = min(T8["trials"], args.trials)
    with Timer() as tm:
        res = run(args)
    res["seconds"] = tm.seconds
    print("wrote", save_results("robust_inference", res, args))
    for path in write_tables(res, args.quick):
        print("wrote", path)
    report(res)

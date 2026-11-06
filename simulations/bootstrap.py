"""Compares the bootstrap calibration of the v Wald test.

Produces Table 7 of the paper.

Type I error at level 0.05 of four tests of H0: v = 0, on the five processes of
``simulations.inference`` under the joint null alpha = 0, v = 0, with tau = 0.5,
n in {50, 100, 200, 500}, 300 trials per cell and B = 199 resamples.

  Asym.      Model-based chi^2_d Wald test.
  Pairs-t    Pairs bootstrap. Every resample refits everything, Frechet means
             included. The observed statistic v^T C^{-1} v and each replicate
             (v*_b - v_b)^T C*_b^{-1} (v*_b - v_b) are studentized by their own
             model-based covariance, and v_b is v_hat parallel transported into
             the resample's tangent space T_{Ybar*_b}.
  Pairs-emp  The same replicates, with the observed statistic studentized by
             the empirical covariance of the replicates v*_b, transported back
             to T_{Ybar}.
  Resid.     Fixed-base residual bootstrap (``ROLS.residual_bootstrap``), which
             conditions on the fitted base points.

Every p-value is (1 + #{T*_b >= T}) / (B_valid + 1). The alpha tests (squared t,
same four constructions) come from the same refits and are saved in the JSON
file, while Table 7 reports only v.

Usage, from the repository root:

    python -m simulations.bootstrap [--quick] [--trials N] [--workers K]
                                    [--seed S] [--B B]
"""

import numpy as np

from ROLS import fit, residual_bootstrap
from ROLS.robust import wald_statistic
from simulations._common import (Timer, parse_args, pmap, rng_for,
                                 save_results, save_text)
from simulations.inference import MANIFOLD_ID, ORDER, generate

TAU = 0.5
SAMPLE_SIZES = [50, 100, 200, 500]
RESAMPLES = 199
TRIALS = 300
LEVEL = 0.05
COLUMNS = ["asym", "pairs_t", "pairs_emp", "resid"]
HEAD = {"asym": "Asym.", "pairs_t": "Pairs-$t$", "pairs_emp": "Pairs-emp",
        "resid": "Resid."}
TABLE_ORDER = ["S2", "T2", "H3", "SO3", "SPD"]
TEX = {"S2": r"$S^2$", "T2": r"$\mathbb{T}^2$", "H3": r"$\mathbb{H}^3$",
       "SO3": r"$\mathrm{SO}(3)$", "SPD": r"$\mathrm{Sym}^+(3)$"}


def _to_base(M, src, dst, coords):
    """Parallel transports intrinsic coordinates from ``src`` to ``dst``."""
    return M.coords(dst, M.transport(src, dst, M.from_coords(src, coords)))


def pairs_calibration(M, Y, X, z, f0, B, rng, min_valid=20):
    """Computes the Pairs-t and Pairs-emp p-values from one set of B refits.

    Args:
        M: Manifold.
        Y: Responses.
        X: Manifold predictor.
        z: Scalar covariate.
        f0: Fit on the full sample.
        B: Number of resamples.
        rng: ``numpy`` generator of the resample indices.
        min_valid: The p-values are NaN below this number of valid resamples.

    Returns:
        A dict with ``n_valid`` and the p-values ``v_pairs_t``, ``v_pairs_emp``,
        ``alpha_pairs_t`` and ``alpha_pairs_emp``.
    """
    n, d, p = f0.n, f0.d, f0.p
    sv = slice(p, p + d)
    C0 = f0.covariance("model")
    a0, v0 = float(f0.alphas[0]), np.asarray(f0.V[0], float)
    t_star = np.full(B, np.nan)  # Squared t replicates of alpha.
    w_star = np.full(B, np.nan)  # Wald replicates of v.
    a_star = np.full(B, np.nan)
    v_back = np.full((B, d), np.nan)  # v*_b transported to T_{Ybar}.
    for b in range(B):
        idx = rng.integers(0, n, n)
        try:
            fb = fit(M, Y[idx], [X[idx]], z[idx])
            Cb = fb.covariance("model")
            v0_in_b = _to_base(M, f0.Y_bar, fb.Y_bar, v0[None])[0]
            w_star[b] = max(wald_statistic(np.asarray(fb.V[0]) - v0_in_b,
                                           Cb[sv, sv]), 0.0)
            t_star[b] = (float(fb.alphas[0]) - a0) ** 2 / Cb[0, 0]
            a_star[b] = float(fb.alphas[0])
            v_back[b] = _to_base(M, fb.Y_bar, f0.Y_bar, fb.V[:1])[0]
        except np.linalg.LinAlgError:
            continue
    ok = (np.isfinite(w_star) & np.isfinite(t_star) & np.isfinite(a_star)
          & np.isfinite(v_back).all(1))
    nv = int(ok.sum())
    out = {"n_valid": nv}
    if nv < min_valid:
        return dict(out, alpha_pairs_t=np.nan, alpha_pairs_emp=np.nan,
                    v_pairs_t=np.nan, v_pairs_emp=np.nan)
    def mc(reps, T):
        return (1.0 + int((reps >= T).sum())) / (nv + 1.0)

    w, t = w_star[ok], t_star[ok]
    out["v_pairs_t"] = mc(w, wald_statistic(v0, C0[sv, sv]))
    out["v_pairs_emp"] = mc(w, wald_statistic(v0, np.cov(v_back[ok],
                                                         rowvar=False)))
    out["alpha_pairs_t"] = mc(t, a0 ** 2 / C0[0, 0])
    out["alpha_pairs_emp"] = mc(t, a0 ** 2 / np.var(a_star[ok], ddof=1))
    return out


def _trial(task):
    """Returns the eight p-values of one simulated data set."""
    name, n, trial, B, seed = task
    key = (seed, MANIFOLD_ID[name], n, trial)
    M, Y, X, z = generate(name, n, 0.0, 0.0, TAU, rng_for(*key, 0))
    f0 = fit(M, Y, [X], z)
    pa, pv = f0.pvalues("model")
    res = {"alpha_asym": float(pa[0]), "v_asym": float(pv[0])}
    res.update(pairs_calibration(M, Y, X, z, f0, B, rng_for(*key, 1)))
    rseed = int(rng_for(*key, 2).integers(2 ** 32))
    for stat in ("alpha", "v"):
        res[f"{stat}_resid"] = residual_bootstrap(
            M, Y, [X], z, B=B, seed=rseed, statistic=stat)["p_value"]
    return res


def run(args):
    """Runs every cell and returns the rejection rates."""
    B = args.B if args.B is not None else (39 if args.quick else RESAMPLES)
    tasks = [(name, n, t, B, args.seed) for name in ORDER for n in SAMPLE_SIZES
             for t in range(args.trials)]
    with Timer() as tm:
        out = pmap(_trial, tasks, workers=args.workers, chunksize=4)
    rates = {}
    for name in ORDER:
        rates[name] = {"sample_sizes": SAMPLE_SIZES}
        for stat in ("v", "alpha"):
            for col in COLUMNS:
                row = []
                for n in SAMPLE_SIZES:
                    ps = np.array([r[f"{stat}_{col}"]
                                   for (m, nn, *_), r in zip(tasks, out)
                                   if m == name and nn == n], float)
                    ok = np.isfinite(ps)
                    row.append(float((ps[ok] < LEVEL).mean()) if ok.any()
                               else float("nan"))
                rates[name][f"{stat}_{col}"] = row
    return {"rates": rates, "trials": args.trials, "B": B, "tau": TAU,
            "min_valid_replicates": min(r["n_valid"] for r in out),
            "seconds": tm.seconds}


def table_tex(results):
    """Returns the LaTeX source of Table 7, the v Wald test only."""
    def fmt(x):
        return "--" if not np.isfinite(x) else f"${x:.3f}$".replace("$0.", "$.")

    ns = SAMPLE_SIZES
    lines = [r"\begin{tabular}{l" + "c" * (4 * len(ns)) + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{$n={n}$}}"
                                for n in ns) + r" \\",
             " ".join(rf"\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}"
                      for i in range(len(ns))),
             "Manifold & " + " & ".join(HEAD[c] for _ in ns for c in COLUMNS)
             + r" \\",
             r"\midrule"]
    for name in TABLE_ORDER:
        r = results["rates"][name]
        cells = [fmt(r[f"v_{c}"][i]) for i in range(len(ns)) for c in COLUMNS]
        lines.append(TEX[name] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def report(results):
    """Prints the rejection rates and their means over the cells."""
    T, B = results["trials"], results["B"]
    plain = {c: HEAD[c].replace("$", "") for c in COLUMNS}
    se = np.sqrt(LEVEL * (1 - LEVEL) / T)
    print(f"Table 7: Type I of the v Wald test at 0.05, tau = {TAU}, {T} "
          f"trials, B = {B}. Monte Carlo s.e. {se:.3f}.")
    print("            " + "".join(f"{plain[c]:>11}" for c in COLUMNS))
    for name in TABLE_ORDER:
        r = results["rates"][name]
        for i, n in enumerate(SAMPLE_SIZES):
            cells = "".join(f"{r['v_' + c][i]:>11.3f}" for c in COLUMNS)
            print(f"  {name:<4} {n:>4} {cells}")
    for c in COLUMNS:
        rates = np.array([results["rates"][m]["v_" + c]
                          for m in TABLE_ORDER]).ravel()
        print(f"  {plain[c]:<10} mean over 20 cells {np.nanmean(rates):.3f}, "
              f"range [{np.nanmin(rates):.3f}, {np.nanmax(rates):.3f}]")
    a = {c: np.nanmean([results["rates"][m]["alpha_" + c]
                        for m in TABLE_ORDER])
         for c in COLUMNS}
    print("  alpha tests, mean over 20 cells: "
          + ", ".join(f"{plain[c]} {a[c]:.3f}" for c in COLUMNS))
    print(f"  fewest valid pairs replicates in a trial: "
          f"{results['min_valid_replicates']}; wall time "
          f"{results['seconds']:.0f} s.")


if __name__ == "__main__":
    args = parse_args(
        "Bootstrap calibration of the v Wald test (Table 7).", trials=TRIALS,
        extra=lambda ap: ap.add_argument(
            "--B", type=int, default=None,
            help=f"bootstrap resamples (default: {RESAMPLES}; --quick: 39)"))
    results = run(args)
    tag = "_quick" if args.quick else ""
    print("wrote", save_results("bootstrap", results, args))
    print("wrote", save_text(f"table_bootstrap{tag}.tex", table_tex(results)))
    report(results)

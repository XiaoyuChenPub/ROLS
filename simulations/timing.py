"""Measures the pairs-bootstrap to fixed-base SE ratio and the refit cost.

Produces the numbers quoted in Appendices H.4 and H.8 of the paper.

(a) SE ratio. For S^2, T^2 and Sym+(3) with the processes of
    ``simulations.inference`` at alpha = 0.5, ||v|| = 0.5, tau = 0.3 and
    n = 200, the ratio of the pairs-bootstrap standard deviation of alpha_hat
    (B = 200 full refits, Frechet means included) to the model-based standard
    error sqrt(sigma^2 [A^{-1}]_{alpha alpha}), which holds the base points
    fixed. Median over 60 trials.
(b) Refit time. Median over 7 single-threaded runs, after one warm-up, of one
    full ROLS fit (both Frechet means, all logs and transports, the normal
    equations) on Sym+(3) with p = q = 1, data from the Sym+(3) process at
    alpha = ||v|| = 0.5, tau = 0.3, for n in {100, 500, 1000, 5000, 10000}.

Usage, from the repository root:

    python -m simulations.timing [--quick] [--trials N] [--workers K]
                                 [--seed S]
"""

import os

# Timings are single threaded. This must run before numpy loads its BLAS, and
# worker processes inherit it.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import platform
import time

import numpy as np

from ROLS import fit
from simulations._common import (Timer, parse_args, pmap, rng_for,
                                 save_results)
from simulations.inference import MANIFOLD_ID, generate

SE_MANIFOLDS = ["S2", "T2", "SPD"]
SE_N, SE_B, SE_TRIALS = 200, 200, 60
ALPHA, V_NORM, TAU = 0.5, 0.5, 0.3
TIME_NS = [100, 500, 1000, 5000, 10000]
TIME_REPS = 7


def _se_ratio_trial(task):
    """Returns the pairs-bootstrap SD of alpha_hat over its model-based SE."""
    name, trial, n, B, seed = task
    M, Y, X, z = generate(name, n, ALPHA, V_NORM, TAU,
                          rng_for(seed, MANIFOLD_ID[name], trial, 0))
    f0 = fit(M, Y, [X], z)
    se_model = float(np.sqrt(f0.covariance("model")[0, 0]))
    rng = rng_for(seed, MANIFOLD_ID[name], trial, 1)
    boot = np.full(B, np.nan)
    for b in range(B):
        idx = rng.integers(0, n, n)
        try:
            boot[b] = fit(M, Y[idx], [X[idx]], z[idx]).alphas[0]
        except np.linalg.LinAlgError:
            continue
    return float(np.nanstd(boot)) / se_model


def refit_times(ns=TIME_NS, reps=TIME_REPS, seed=0):
    """Times one full Sym+(3) refit.

    Args:
        ns: Sample sizes.
        reps: Number of timed runs at each sample size.
        seed: Seed of the data.

    Returns:
        A dict with the sample sizes and the median and interquartile range of
        the wall time in seconds.
    """
    out = {"n": [], "median_s": [], "iqr_s": []}
    for n in ns:
        M, Y, X, z = generate("SPD", n, ALPHA, V_NORM, TAU,
                              rng_for(seed, 99, n))
        fit(M, Y, [X], z)  # Warm-up, not timed.
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fit(M, Y, [X], z)
            ts.append(time.perf_counter() - t0)
        q1, med, q3 = np.percentile(ts, [25, 50, 75])
        out["n"].append(n)
        out["median_s"].append(float(med))
        out["iqr_s"].append(float(q3 - q1))
    return out


def run(args):
    """Runs both measurements and returns the results as a nested dict."""
    B = 20 if args.quick else SE_B
    tasks = [(m, t, SE_N, B, args.seed)
             for m in SE_MANIFOLDS for t in range(args.trials)]
    with Timer() as tm:
        ratios = pmap(_se_ratio_trial, tasks, workers=args.workers)
    by = {m: [r for (mm, *_), r in zip(tasks, ratios) if mm == m]
          for m in SE_MANIFOLDS}
    se = {m: {"median": float(np.median(v)),
              "q25": float(np.percentile(v, 25)),
              "q75": float(np.percentile(v, 75))}
          for m, v in by.items()}
    timing = refit_times(TIME_NS[:3] if args.quick else TIME_NS,
                         3 if args.quick else TIME_REPS, args.seed)
    timing["platform"] = (f"{platform.machine()}, {platform.system()}, Python "
                          f"{platform.python_version()}, numpy "
                          f"{np.__version__}, single thread")
    return {"se_ratio": se,
            "se_settings": {"n": SE_N, "B": B, "trials": args.trials,
                            "alpha": ALPHA, "v_norm": V_NORM, "tau": TAU},
            "timing": timing, "seconds_se_ratio": tm.seconds}


def report(results):
    """Prints the SE ratios and the refit times."""
    s = results["se_settings"]
    print(f"(a) Pairs-bootstrap SD / model-based SE of alpha_hat, median [IQR] "
          f"over {s['trials']} trials, n = {s['n']}, B = {s['B']}:")
    for m in SE_MANIFOLDS:
        r = results["se_ratio"][m]
        print(f"  {m:<4} {r['median']:.3f} [{r['q25']:.3f}, {r['q75']:.3f}]")
    t = results["timing"]
    print(f"(b) One full Sym+(3) refit, p = q = 1, median [IQR] seconds "
          f"({t['platform']}):")
    for n, med, iqr in zip(t["n"], t["median_s"], t["iqr_s"]):
        print(f"  n = {n:>6}: {med:.4f} [{iqr:.4f}]")


if __name__ == "__main__":
    args = parse_args(
        "Pairs/fixed-base SE ratio and refit timing (Appendix H.4, H.8).",
        trials=SE_TRIALS)
    results = run(args)
    print("wrote", save_results("timing", results, args))
    report(results)

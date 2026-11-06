"""Measures the sensitivity of ROLS to the base point of the linearization.

Produces Table 11 of the paper.

Sym+(3) with the affine-invariant metric, the process of Table 12
(``simulations.high_dimension.generate``) with alpha* = 0.5, |V*|_F = 0.5,
noise tau = 0.3, n = 200 training and 100 held-out observations, 200 trials.
Response and predictor are refit together at

  frechet     the Frechet means, which reproduces ``ROLS.fit``
  delta=c     each Frechet mean displaced by geodesic distance c * spread in a
              random direction, spread being the root mean squared geodesic
              distance of that variable's sample to its Frechet mean, for c in
              0.1, 0.25, 0.5, 1, 2 (the table prints c <= 1)
  ambient     the arithmetic mean of the matrices, projected onto Sym+(3)
  random      one training observation of Y and an independently chosen one
              of X

The table reports |alpha_hat - alpha*| and the held-out affine-invariant Frechet
RMSE, both averaged over trials. The held-out sample is an independent draw of
the process with the V* of the training sample.

Usage, from the repository root:

    python -m simulations.base_point [--quick] [--trials N] [--workers K]
                                     [--seed S]

``--seed S`` is added to the seed of every trial. Seeds that are multiples of
10**9 give independent runs.
"""

import numpy as np

from ROLS import SPD, ROLSFit, frechet_rmse
from simulations._common import (Timer, parse_args, pmap, save_results,
                                 save_text)
from simulations.high_dimension import generate

K, N, N_TEST, NOISE, ALPHA, V_SCALE = 3, 200, 100, 0.3, 0.5, 0.5
DELTAS = (0.1, 0.25, 0.5, 1.0, 2.0)
LABELS = (("frechet",) + tuple(f"delta={c}" for c in DELTAS)
          + ("ambient", "random"))
TABLE = ("frechet", "delta=0.1", "delta=0.25", "delta=0.5", "delta=1.0",
         "ambient", "random")
HEAD = (r"Fr\'echet", r"$\delta=0.1$", r"$\delta=0.25$", r"$\delta=0.5$",
        r"$\delta=1.0$", "Ambient", "Random")


def fit_at(m, Y_bar, X_bar, Y, X, z, reg=1e-6):
    """Fits ROLS linearized at supplied base points.

    Builds the design of ``ROLS.fit`` at (Y_bar, X_bar) and solves the same
    normal equations, (sum_i Z_i^T Z_i + reg n I) theta = sum_i Z_i^T w_i,
    through ``ROLSFit.bread`` and ``ROLSFit.scores``.

    Args:
        m: Manifold.
        Y_bar: Base point of the response.
        X_bar: Base point of the predictor.
        Y: Responses.
        X: Manifold predictor.
        z: Scalar covariates.
        reg: Ridge constant.

    Returns:
        A ``ROLSFit`` whose ``predict`` maps back through the base points.
    """
    W = m.coords(Y_bar, m.logmap(Y_bar, Y))
    U = m.coords(Y_bar, m.transport(X_bar, Y_bar, m.logmap(X_bar, X)))
    n, d = W.shape
    Z = np.asarray(z, float).reshape(n, -1)
    z_mean = Z.mean(0)
    Zt = Z - z_mean
    q = Zt.shape[1]
    design = ROLSFit(m, Y_bar, [X_bar], np.zeros(1), np.zeros((q, d)), W, [U],
                     Zt, z_mean, reg)
    theta = np.linalg.solve(design.bread(reg * n), design.scores(W).sum(0))
    return ROLSFit(m, Y_bar, [X_bar], theta[:1], theta[1:].reshape(q, d), W,
                   [U], Zt, z_mean, reg)


def rms_dist(m, base, points):
    """Returns the root mean squared geodesic distance to ``base``."""
    return float(np.sqrt((m.dist(base, points) ** 2).mean()))


def displace(m, base, eps, rng):
    """Moves ``base`` by geodesic distance ``eps`` along a random direction.

    The direction is a symmetrized Gaussian matrix normalized in the
    affine-invariant norm at ``base``, so Exp_base(eps * direction) lies exactly
    ``eps`` away.

    Args:
        m: ``SPD`` manifold.
        base: Point to move.
        eps: Geodesic distance.
        rng: ``numpy`` generator.

    Returns:
        The displaced point.
    """
    A = rng.standard_normal((m.k, m.k))
    A = 0.5 * (A + A.T)
    A = A / np.linalg.norm(m.coords(base, A))
    return m.expmap(base, eps * A)[0]


def trial(rng, rng_test):
    """Runs one replication.

    Args:
        rng: Generator of the training data and the base points.
        rng_test: Generator of the held-out data.

    Returns:
        A dict mapping each base-point label to the pair
        (|alpha_hat - alpha*|, held-out RMSE).
    """
    m = SPD(K)
    X, z, Y, V_true = generate(m, N, ALPHA, V_SCALE, NOISE, rng)
    Xte, zte, Yte, _ = generate(m, N_TEST, ALPHA, V_SCALE, NOISE, rng_test,
                                V_true)
    Yf, Xf = m.frechet_mean(Y), m.frechet_mean(X)
    sY, sX = rms_dist(m, Yf, Y), rms_dist(m, Xf, X)
    bases = {"frechet": (Yf, Xf)}
    for c in DELTAS:
        bases[f"delta={c}"] = (displace(m, Yf, c * sY, rng),
                               displace(m, Xf, c * sX, rng))
    bases["ambient"] = (m.project(Y.mean(0))[0], m.project(X.mean(0))[0])
    bases["random"] = (Y[rng.integers(0, N)], X[rng.integers(0, N)])
    out = {}
    for lab, (Yb, Xb) in bases.items():
        f = fit_at(m, Yb, Xb, Y, X, z)
        out[lab] = (abs(float(f.alphas[0]) - ALPHA),
                    frechet_rmse(m, Yte, f.predict([Xte], zte)))
    return out


def _task(task):
    """Runs replication ``t``."""
    seed, t = task
    return trial(np.random.default_rng(42 + seed + t),
                 np.random.default_rng(42 + seed + 9000 + t))


def run(args):
    """Runs the experiment and returns the results as a nested dict."""
    out = pmap(_task, [(args.seed, t) for t in range(args.trials)],
               args.workers, chunksize=4)
    res = {"trials": args.trials, "n": N, "n_test": N_TEST, "noise": NOISE,
           "alpha_err": {}, "alpha_err_se": {}, "rmse": {}, "rmse_se": {}}
    for lab in LABELS:
        r = np.array([o[lab] for o in out])
        se = (r.std(0, ddof=1) / np.sqrt(len(r)) if len(r) > 1
              else np.full(2, np.nan))
        res["alpha_err"][lab] = float(r[:, 0].mean())
        res["rmse"][lab] = float(r[:, 1].mean())
        res["alpha_err_se"][lab] = float(se[0])
        res["rmse_se"][lab] = float(se[1])
    return res


def latex_table(res):
    """Returns the LaTeX source of Table 11."""
    ae = " & ".join(f"${res['alpha_err'][lab]:.4f}$" for lab in TABLE)
    rm = " & ".join(f"${res['rmse'][lab]:.3f}$" for lab in TABLE)
    return "\n".join([
        r"\begin{table}[h]", r"\centering",
        r"\caption{Coefficient error and held-out Fr\'echet RMSE on "
        r"$\mathrm{Sym}^+(3)$ under "
        rf"different base points (${res['trials']}$ trials).}}",
        r"\label{tab:base-point}", r"\footnotesize",
        r"\begin{tabular}{lccccccc}", r"\toprule",
        "Base & " + " & ".join(HEAD) + r" \\", r"\midrule",
        rf"Coeff.\ error & {ae} \\", rf"RMSE & {rm} \\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])


def report(res):
    """Prints the coefficient error and the RMSE at each base point."""
    print(f"Table 11, Sym+(3), n = {res['n']}, {res['trials']} trials")
    print(f"{'base':<11} {'|a-a*|':>7} {'s.e.':>6}   {'RMSE':>6} {'s.e.':>6}")
    for lab in LABELS:
        print(f"{lab:<11} {res['alpha_err'][lab]:7.4f} "
              f"{res['alpha_err_se'][lab]:6.4f}   {res['rmse'][lab]:6.3f} "
              f"{res['rmse_se'][lab]:6.3f}")


if __name__ == "__main__":
    args = parse_args(__doc__.splitlines()[0], trials=200)
    with Timer() as timer:
        res = run(args)
    res["seconds"] = timer.seconds
    tag = "_quick" if args.quick else ""
    print("wrote", save_results("base_point", res, args))
    print("wrote", save_text(f"table_base_point{tag}.tex", latex_table(res)))
    report(res)

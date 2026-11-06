"""Shared utilities of the simulation modules.

Every experiment module is launched from the repository root as

    python -m simulations.<module> [--quick] [--trials N] [--workers K]
                                   [--seed S]

``--quick`` shrinks the Monte Carlo budget to a smoke test. Results and LaTeX
tables are written to ``results/``, figures to ``figures/``.
"""

import argparse
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(REPO, "results")
FIGURES = os.path.join(REPO, "figures")


def parse_args(description, trials, extra=None):
    """Parses the command line shared by the experiment modules.

    Args:
        description: Text shown by ``--help``.
        trials: The paper's number of Monte Carlo replications.
        extra: Optional callable that adds arguments to the parser.

    Returns:
        The parsed arguments. ``args.trials`` defaults to ``trials``, or to 3
        under ``--quick``.
    """
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--quick", action="store_true",
                    help="smoke test with a tiny Monte Carlo budget")
    ap.add_argument("--trials", type=int, default=None,
                    help=f"Monte Carlo replications (default: {trials})")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 2) - 1),
                    help="number of worker processes")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed of the Monte Carlo streams (default: 0)")
    if extra is not None:
        extra(ap)
    args = ap.parse_args()
    if args.trials is None:
        args.trials = 3 if args.quick else trials
    return args


def _single_thread():
    """Limits the numerical libraries to one thread per process."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"


def pmap(fn, tasks, workers=1, chunksize=1):
    """Maps ``fn`` over ``tasks``, in parallel when ``workers > 1``.

    Each task must be picklable and carry its own seed, so that results do not
    depend on the number of workers.

    Args:
        fn: Function of one task. It must be defined at module level.
        tasks: Iterable of tasks.
        workers: Number of worker processes.
        chunksize: Number of tasks sent to a worker at a time.

    Returns:
        The list of results, in the order of ``tasks``.
    """
    tasks = list(tasks)
    if workers <= 1 or len(tasks) <= 1:
        return [fn(t) for t in tasks]
    _single_thread()
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context("spawn"),
                             initializer=_single_thread) as ex:
        return list(ex.map(fn, tasks, chunksize=chunksize))


def rng_for(*keys):
    """Returns an independent, reproducible generator for integer keys."""
    return np.random.default_rng([int(k) for k in keys])


def _jsonable(o):
    """Converts nested results to types that ``json`` can write."""
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def _results_path(name, quick):
    """Returns the path of the JSON file of experiment ``name``."""
    return os.path.join(RESULTS, f"{name}{'_quick' if quick else ''}.json")


def save_results(name, obj, args=None):
    """Writes ``obj`` and the settings to ``results/<name>.json``.

    Args:
        name: Name of the experiment.
        obj: Nested results.
        args: Parsed command line. Under ``--quick`` the file name ends in
            ``_quick``.

    Returns:
        The path of the file.
    """
    os.makedirs(RESULTS, exist_ok=True)
    payload = {"experiment": name,
               "created": time.strftime("%Y-%m-%d %H:%M:%S"),
               "settings": vars(args) if args is not None else {},
               "results": obj}
    path = _results_path(name, args is not None and args.quick)
    with open(path, "w") as fh:
        json.dump(_jsonable(payload), fh, indent=1)
    return path


def load_results(name, quick=False):
    """Returns the results saved by ``save_results``."""
    with open(_results_path(name, quick)) as fh:
        return json.load(fh)["results"]


def save_text(filename, text):
    """Writes ``text`` to ``results/<filename>`` and returns the path."""
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, filename)
    with open(path, "w") as fh:
        fh.write(text)
    return path


def figure_path(filename):
    """Returns the path of ``filename`` in ``figures/``, creating the folder."""
    os.makedirs(FIGURES, exist_ok=True)
    return os.path.join(FIGURES, filename)


def pyplot():
    """Returns ``matplotlib.pyplot`` on the non-interactive Agg backend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def loglog_slope(ns, errs):
    """Returns the least-squares slope and R^2 of log(err) on log(n)."""
    x, y = np.log(np.asarray(ns, float)), np.log(np.asarray(errs, float))
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    r2 = 1.0 - (resid @ resid) / max(((y - y.mean()) ** 2).sum(), 1e-300)
    return float(coef[0]), float(r2)


class Timer:
    """Context manager that stores the elapsed wall time in ``seconds``."""

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.seconds = time.time() - self.t0

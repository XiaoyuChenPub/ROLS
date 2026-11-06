"""Runs every simulation of the paper, then assembles Figure 2.

Each experiment runs in its own process with the paper's Monte Carlo budget, or
the smoke-test budget under ``--quick``. ``timing`` runs first, alone, because
it measures wall-clock refit times.

Usage, from the repository root:

    python -m simulations.run_all [--quick] [--workers K] [--only a,b,...]
"""

import argparse
import os
import subprocess
import sys
import time

from simulations._common import REPO

ORDER = ["timing", "prediction", "convergence", "cutlocus", "inference",
         "bootstrap", "misspecification", "robust_inference", "base_point",
         "high_dimension"]


def main():
    """Parses the command line and runs the selected experiments."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true",
                    help="smoke test with a tiny Monte Carlo budget")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 2) - 1),
                    help="number of worker processes")
    ap.add_argument("--only", type=str, default=None,
                    help="comma-separated subset of " + ", ".join(ORDER)
                         + ", headline")
    args = ap.parse_args()
    todo = (ORDER if args.only is None
            else [m for m in ORDER if m in args.only.split(",")])
    for mod in todo:
        cmd = [sys.executable, "-m", f"simulations.{mod}",
               "--workers", str(1 if mod == "timing" else args.workers)]
        if args.quick:
            cmd.append("--quick")
        t0 = time.time()
        print(f"== {mod}: {' '.join(cmd[1:])}", flush=True)
        subprocess.run(cmd, cwd=REPO, check=True)
        print(f"== {mod} done in {time.time() - t0:.0f} s\n", flush=True)
    if args.only is None or "headline" in args.only.split(","):
        cmd = [sys.executable, "-m", "simulations.headline"]
        if args.quick:
            cmd.append("--quick")
        subprocess.run(cmd, cwd=REPO, check=True)


if __name__ == "__main__":
    main()

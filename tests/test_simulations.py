"""Smoke test of the simulation modules.

``prediction``, ``misspecification`` and ``headline`` are left out because the
nonlinear and single-index Frechet baselines take minutes under ``--quick``.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MODULES = ["timing", "convergence", "cutlocus", "inference", "bootstrap",
           "robust_inference", "base_point", "high_dimension"]


def test_run_all_quick(tmp_path):
    """Runs ``run_all --quick`` in a temporary copy of the repository."""
    pytest.importorskip("matplotlib")
    for folder in ("ROLS", "simulations"):
        shutil.copytree(REPO / folder, tmp_path / folder,
                        ignore=shutil.ignore_patterns("__pycache__"))
    subprocess.run([sys.executable, "-m", "simulations.run_all", "--quick",
                    "--workers", "2", "--only", ",".join(MODULES)],
                   cwd=tmp_path, check=True)
    for module in MODULES:
        assert (tmp_path / "results" / f"{module}_quick.json").exists()

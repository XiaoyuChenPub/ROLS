"""Assembles Figure 2 of the paper, the simulation summary, from saved results.

(a) cut-locus sweep on S^2          simulations.cutlocus
(b) convergence of |alpha_hat|      simulations.convergence
(c) t-test power, H0: alpha = 0     simulations.inference
(d) default-cell Frechet RMSE       simulations.prediction

Usage, from the repository root, after running the four experiments:

    python -m simulations.headline [--quick]

``--quick`` draws the figure from the smoke-test results. The text is set by
LaTeX, so this module needs a LaTeX installation.
"""

import argparse

from simulations import convergence, cutlocus, inference, prediction
from simulations._common import figure_path, load_results, pyplot

LATEX_PREAMBLE = (r"\usepackage{amsfonts}\usepackage{amsmath}"
                  r"\usepackage{mathptmx}")
TITLES = (r"\textbf{(a) Cut-locus on} $\mathbb{S}^2$",
          r"\textbf{(b) Convergence at} $O(n^{-1/2})$",
          r"\textbf{(c)} $t$\textbf{-test power}",
          r"\textbf{(d) Predictive performance}")


def make_figure(quick=False, path=None):
    """Draws Figure 2.

    Args:
        quick: Whether to read the smoke-test results.
        path: Output path, by default ``figures/simulation_headline.pdf``.

    Returns:
        The path of the figure.
    """
    plt = pyplot()
    plt.rcParams.update({"text.usetex": True,
                         "text.latex.preamble": LATEX_PREAMBLE,
                         "font.family": "serif", "font.size": 13,
                         "axes.titlesize": 14, "axes.labelsize": 14,
                         "legend.fontsize": 10, "xtick.labelsize": 12,
                         "ytick.labelsize": 12})
    fig, axes = plt.subplots(
        1, 4, figsize=(14.0, 3.3),
        gridspec_kw={"width_ratios": [1.0, 0.95, 0.9, 1.15], "wspace": 0.68})
    cutlocus.headline_panel(axes[0], load_results("cutlocus", quick))
    convergence.headline_panel(axes[1], load_results("convergence", quick))
    inference.panel_power(axes[2], load_results("inference", quick))
    prediction.plot_default_cell(axes[3],
                                 prediction.default_cell_rmse(quick=quick),
                                 title=False, legend_width=0.55,
                                 legend_fontsize=9, legend_columns=1)
    for ax, title in zip(axes, TITLES):
        ax.set_title(title, loc="left")
    path = path or figure_path(
        f"simulation_headline{'_quick' if quick else ''}.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Figure 2, simulation summary.")
    ap.add_argument("--quick", action="store_true",
                    help="use the smoke-test results")
    print("wrote", make_figure(ap.parse_args().quick))

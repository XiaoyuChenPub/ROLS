# ROLS: Riemannian Ordinary Least Squares

Regression for a manifold-valued response with mixed manifold-valued and scalar predictors.
The estimator is closed form, and each coefficient has a hypothesis test.
This repository contains the `ROLS` Python package and the simulations for

> Xiaoyu Chen*, Yujing (Zipan) Huang, Yingyan Zeng. *Riemannian Ordinary Least Squares.*
> Advances in Neural Information Processing Systems (NeurIPS), 2026.

## Installation

```bash
git clone https://github.com/XiaoyuChenPub/ROLS.git
cd ROLS
pip install .            # needs numpy and scipy
pip install ".[test]"    # adds pytest and matplotlib
pip install ".[simulations]"    # adds matplotlib, for the paper's simulations
```

Python 3.9 or later.

## Quick start

```python
import numpy as np
from ROLS import fit, Sphere

m = Sphere(3)                        # the unit sphere S^2 in R^3
f = fit(m, Y, [X1, X2], z)           # Y, X1, X2: (n, 3) points on S^2; z: (n, q)

f.alphas, f.V                        # coefficients (V in intrinsic coordinates at f.Y_bar)
p_alpha, p_v = f.pvalues()           # per-coefficient p-values
Y_hat = f.predict([X1_new, X2_new], z_new)
```

`examples/toy_example.py` simulates data from the model with known coefficients on the sphere S^2 and on 3 x 3 SPD matrices, then fits, tests, bootstraps, and predicts.

```bash
python examples/toy_example.py
```

```
Sphere S^2, n = 200
  alpha    estimate  0.587   truth  0.600
  v_1      estimate [-0.243 -0.142  0.001]   truth [-0.25  -0.15   0.001]
  v_2      estimate [-0.005  0.005  0.   ]   truth [0. 0. 0.]
  p-values (model)  alpha 1.8e-130   v_1 0   v_2 0.52
  p-values (HC3  )  alpha 3.9e-130   v_1 0   v_2 0.60
  pairs bootstrap p-value for v_2 = 0: 0.56
  test Frechet RMSE   ROLS 0.142   Frechet mean 0.418   noise 0.141

SPD(3), n = 200
  alpha    estimate  0.592   truth  0.600
  p-values (model)  alpha 0   v_1 4.8e-302   v_2 0.10
  test Frechet RMSE   ROLS 0.247   Frechet mean 0.568   noise 0.245
```

The true `v_2` is zero, and the out-of-sample Fréchet root mean squared error (RMSE) of ROLS is at the noise level.

## Manifolds

| class | space | metric |
|---|---|---|
| `Sphere(D)` | S^(D-1) in R^D | round metric |
| `Torus(k)` | flat torus (S^1)^k, angles | flat metric |
| `SPD(k)` | k x k symmetric positive definite matrices | affine-invariant metric |
| `Hyperbolic(d)` | H^d, Lorentz (hyperboloid) model | curvature -1 |
| `SO3()` | rotations, body-frame tangent coordinates | bi-invariant metric |

To add a manifold that is not in the table, subclass `ROLS.Manifold` and implement four maps.

- `logmap` and `expmap` convert a point on the manifold to a tangent vector, and convert a tangent vector back to a point.
- `dist` returns the geodesic distance between two points.
- `transport` moves a tangent vector from one base point to another by parallel transport.
- `coords` and `from_coords` write a tangent vector as `d` numbers and convert those numbers back to a tangent vector.
  The ordinary dot product of the numbers equals the Riemannian inner product of the vectors.

The package computes the Fréchet mean by a damped Karcher iteration and stops at machine precision.
If the iteration stops short of that precision, it warns that the data may have no unique mean.

`fit` and `predict` first check that every point lies on the manifold.
A point that fails the check raises `ValueError`.
Examples are a vector of norm 2 passed to `Sphere`, an asymmetric matrix passed to `SPD`, and a vector that does not lie on the hyperboloid.
The function does not move the point onto the manifold.

Conventions:

- Sym+(k) uses the isometric chart `vech(L^-1 V L^-T)` with `L = chol(base)` and sqrt(2) on the off-diagonal entries.
- H^d points are vectors in R^(d+1) on the hyperboloid.
  `Hyperbolic.from_ball` maps Poincaré ball coordinates to the Lorentz model.
- SO(3) tangent vectors are body-frame coordinates, `Log_R(S) = vee(log(R^T S))`.
  Parallel transport is the bi-invariant Levi-Civita transport `w -> exp(-xi/2) w`, with `xi = Log_R(S)`.
- `f.V` is expressed in the intrinsic frame at the fitted mean `f.Y_bar`.
  To compare `v_k` across fits, or with a known vector, use `f.V_ambient()` and parallel-transport the vectors to a common base point, as in the toy example.

## Inference

| call | covariance |
|---|---|
| `f.pvalues("model")` | homoskedastic, `sigma2 * A^-1` |
| `f.pvalues("HC0")` | White sandwich |
| `f.pvalues("HC3")` | leverage-corrected sandwich (MacKinnon and White, 1985) |
| `f.pvalues("HC0", clusters=g)` | cluster-robust, with scores summed within each cluster |
| `f.joint_pvalue(k0, k1)` | joint Wald test of `v_k0, ..., v_(k1-1)`, for example a (sin, cos) pair |
| `pairs_bootstrap(...)` | refits the Fréchet means and the coefficients, and estimates the full influence-function variance |
| `residual_bootstrap(...)` | conditions on the base points and estimates the tangent-space variance |

All inference is computed in intrinsic coordinates.
A covariance built from ambient coordinates, for example R^3 for S^2, is rank deficient and gives Wald statistics that are too large.
`ROLS.sandwich` and `ROLS.pvalues` apply the same formulas to a design assembled by hand.

With few observations on a higher-dimensional manifold, the sandwich Wald test of a d-dimensional `v_k` rejects too often.
On Sym+(3) (d = 6) with n = 100, the HC3 test rejects a true null about 15% of the time at level 0.05, while the model-based test holds its level.

## Baselines

`ROLS.baselines` contains the comparison methods used in the paper.
Each function fits on the training data and returns predicted responses at new covariates.

| function | method |
|---|---|
| `frechet_mean` | Fréchet mean of the training responses |
| `geodesic_regression` | tangent-linear regression on scalars (Fletcher, 2013), ROLS with no manifold predictor |
| `frechet_regression` | global Fréchet regression on scalars (Petersen and Müller, 2019) |
| `single_index_frechet` | single-index Fréchet regression on scalars (Bhattacharjee and Müller, 2023) |
| `nonlinear_frechet` | nonlinear Fréchet regression on manifold and scalar predictors (Bhattacharjee, Li and Xue, 2025) |
| `rols_manifold_only` | ROLS with the scalar block removed |
| `tangent_linear_ridge` | ROLS with a full `d x d` matrix in place of each scalar `alpha_j` |

The docstrings record the choices made where the cited paper leaves the procedure unspecified.

## Simulations

Each simulation table and figure in the paper is produced by one module in `simulations/`, run from the repository root.
Tables are written to `results/` and figures to `figures/`.

```bash
python -m simulations.run_all --workers 16      # everything
python -m simulations.base_point --workers 16   # a single module
python -m simulations.run_all --quick           # small Monte Carlo budget
```

| module | paper |
|---|---|
| `prediction` | Table 5, Figure 4 |
| `cutlocus` | Figure 5 |
| `convergence` | Table 6, Figure 6 |
| `inference` | Figures 7 and 8 |
| `bootstrap` | Table 7 |
| `misspecification` | Figure 9 |
| `robust_inference` | Tables 8, 9 and 10 |
| `base_point` | Table 11 |
| `high_dimension` | Table 12 |
| `timing` | standard-error ratios and refit times in the text |
| `headline` | Figure 2, assembled from the saved results |

`prediction` takes about 14 core-hours, or about 46 minutes with 18 workers, because of the nonlinear and single-index Fréchet baselines.
The other modules take minutes.
`headline` requires a LaTeX installation, because the text in Figure 2 is set in LaTeX.

## Tests

```bash
pytest
```

The unit tests check the geometry, including isometric charts, isometric transport, the Levi-Civita equation on SO(3), and the Poincaré-to-Lorentz isometry.
They also check coefficient recovery and prediction on all five manifolds, the sandwich covariances, the level of the tests under the null, the bootstrap p-values, the closed forms of the baselines in Euclidean space, and a short run of the simulation modules.
The suite finishes in under a minute.

## Citation

```bibtex
@inproceedings{chen2026rols,
  title     = {Riemannian Ordinary Least Squares},
  author    = {Chen, Xiaoyu and Huang, Yujing (Zipan) and Zeng, Yingyan},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## Acknowledgement

This work was supported in part by the National Science Foundation under CAREER Award CMMI-2543434 and EAGER Award CMMI-2430998, and by the American Heart Association Collaborative Science Award 23CSA1052735.

## License

Creative Commons Attribution-NonCommercial 4.0 International ([CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)).
You may share and adapt the code for non-commercial purposes with attribution.
See `LICENSE`.

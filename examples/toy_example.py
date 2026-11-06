"""ROLS on the sphere S^2 and on 3 x 3 SPD matrices.

After ``pip install .``, run from the repository root with

    python examples/toy_example.py

The data follow the ROLS model with known coefficients. In the tangent space
at the response mean Ybar,

    Log_Ybar(Y) = alpha PT_{Xbar -> Ybar}(Log_Xbar(X)) + z_1 v_1 + z_2 v_2
                  + eps,

with one manifold predictor X and two scalar covariates, of which z_2 has no
effect.
"""

import numpy as np

from ROLS import SPD, Sphere, fit, frechet_rmse, pairs_bootstrap

rng = np.random.default_rng(0)


def simulate(m, Ybar, Xbar, alpha, V, n, noise=0.1):
    """Draws n observations; V holds v_k as rows in coordinates at Ybar."""
    X = m.expmap(Xbar, m.from_coords(Xbar, 0.3 * rng.standard_normal((n, m.d))))
    u = m.coords(Ybar, m.transport(Xbar, Ybar, m.logmap(Xbar, X)))
    z = rng.standard_normal((n, len(V)))
    w = alpha * u + z @ V + noise * rng.standard_normal((n, m.d))
    return m.expmap(Ybar, m.from_coords(Ybar, w)), X, z


# Sphere S^2. Points are unit vectors in R^3.
m = Sphere(3)
Ybar = np.array([0.0, 0.0, 1.0])
Xbar = np.array([1.0, 0.0, 0.0])
alpha = 0.6
V = np.array([[0.25, -0.15],
              [0.00, 0.00]])

Y, X, z = simulate(m, Ybar, Xbar, alpha, V, n=200)
f = fit(m, Y, [X], z)

print("Sphere S^2, n = 200")
print(f"  alpha    estimate {f.alphas[0]: .3f}   truth {alpha: .3f}")

# f.V is expressed in the intrinsic frame at the fitted mean f.Y_bar.
# f.V_ambient() gives the same vectors in R^3, comparable with the truth
# after parallel transport to f.Y_bar.
V_true = m.transport(Ybar, f.Y_bar, m.from_coords(Ybar, V))
for k in range(2):
    print(f"  v_{k + 1}      estimate {np.round(f.V_ambient()[k], 3)}   "
          f"truth {np.round(V_true[k], 3)}")

for kind in ("model", "HC3"):
    p_alpha, p_v = f.pvalues(kind)
    print(f"  p-values ({kind:5s})  alpha {p_alpha[0]:.2g}   "
          f"v_1 {p_v[0]:.2g}   v_2 {p_v[1]:.2f}")

bs = pairs_bootstrap(m, Y, [X], z, B=199, statistic="v", index=1, seed=1)
print(f"  pairs bootstrap p-value for v_2 = 0: {bs['p_value']:.2f}")

Y_new, X_new, z_new = simulate(m, Ybar, Xbar, alpha, V, n=1000)
Y_hat = f.predict([X_new], z_new)
Y_mean = np.tile(f.Y_bar, (1000, 1))
print(f"  test Frechet RMSE   ROLS {frechet_rmse(m, Y_new, Y_hat):.3f}   "
      f"Frechet mean {frechet_rmse(m, Y_new, Y_mean):.3f}   "
      f"noise {0.1 * np.sqrt(m.d):.3f}")

# SPD(3) with the affine-invariant metric. The same calls take (n, 3, 3)
# arrays; the tangent space has d = 6 intrinsic coordinates.
m = SPD(3)
Ybar = 1.5 * np.eye(3)
Xbar = np.diag([1.0, 2.0, 0.5])
V = np.zeros((2, m.d))
V[0, :3] = [0.2, -0.1, 0.15]

Y, X, z = simulate(m, Ybar, Xbar, alpha, V, n=200)
f = fit(m, Y, [X], z)
p_alpha, p_v = f.pvalues("model")

print("\nSPD(3), n = 200")
print(f"  alpha    estimate {f.alphas[0]: .3f}   truth {alpha: .3f}")
print(f"  p-values (model)  alpha {p_alpha[0]:.2g}   v_1 {p_v[0]:.2g}   "
      f"v_2 {p_v[1]:.2f}")

Y_new, X_new, z_new = simulate(m, Ybar, Xbar, alpha, V, n=1000)
Y_hat = f.predict([X_new], z_new)
Y_mean = np.stack([f.Y_bar] * 1000)
print(f"  test Frechet RMSE   ROLS {frechet_rmse(m, Y_new, Y_hat):.3f}   "
      f"Frechet mean {frechet_rmse(m, Y_new, Y_mean):.3f}   "
      f"noise {0.1 * np.sqrt(m.d):.3f}")

"""ROLS, Riemannian ordinary least squares.

Regression for a manifold-valued response with mixed manifold-valued and
scalar predictors, with a hypothesis test for every coefficient.

Typical usage example:

    from ROLS import fit, Sphere

    f = fit(Sphere(3), Y, [X1, X2], z)
    p_alpha, p_v = f.pvalues("HC3")
    Y_hat = f.predict([X1_new, X2_new], z_new)
"""

from .bootstrap import pairs_bootstrap, residual_bootstrap
from .estimator import ROLSFit, fit
from .manifolds import SO3, SPD, Hyperbolic, Manifold, Sphere, Torus
from .manifolds import frechet_rmse
from .robust import pvalues, sandwich

__version__ = "1.0.0"
__all__ = ["fit", "ROLSFit", "pairs_bootstrap", "residual_bootstrap",
           "sandwich", "pvalues", "Manifold", "frechet_rmse", "Sphere",
           "Torus", "SPD", "Hyperbolic", "SO3", "__version__"]

"""
mlx_hyperbolic: Hyperbolic Geometry Operations for MLX

Fast hyperbolic neural network primitives on Apple Silicon.
Supports both Poincaré ball and Lorentz (hyperboloid) models.

Models:
    - Poincaré ball: Classic model, intuitive but numerically unstable near boundary
    - Lorentz (hyperboloid): More stable, preferred for modern hyperbolic ML

Example (Poincaré):
    >>> import mlx.core as mx
    >>> from mlx_hyperbolic import mobius_add, poincare_distance
    >>>
    >>> x = mx.array([0.1, 0.2, 0.3])
    >>> y = mx.array([0.2, 0.1, 0.2])
    >>> z = mobius_add(x, y)  # Möbius addition
    >>> d = poincare_distance(x, y)  # Geodesic distance

Example (Lorentz):
    >>> from mlx_hyperbolic import lorentz_distance, exp_map_lorentz
    >>> from mlx_hyperbolic import poincare_to_lorentz
    >>>
    >>> # Convert Poincaré points to Lorentz
    >>> x_L = poincare_to_lorentz(x)
    >>> y_L = poincare_to_lorentz(y)
    >>> d = lorentz_distance(x_L, y_L)  # Same distance, more stable
"""

__version__ = "0.2.0"
__author__ = "Nitin Borwankar"

# Poincaré ball model operations
from .ops import (
    mobius_add,
    poincare_distance,
    exp_map,
    log_map,
)

# Lorentz (hyperboloid) model operations
from .lorentz import (
    # Core operations
    minkowski_inner,
    minkowski_norm,
    lorentz_distance,
    lorentz_distance_squared,
    # Mappings
    exp_map_lorentz,
    log_map_lorentz,
    parallel_transport_lorentz,
    # Utilities
    project_to_hyperboloid,
    lorentz_centroid,
    check_on_hyperboloid,
    # Model conversions
    poincare_to_lorentz,
    lorentz_to_poincare,
)

__all__ = [
    # Version
    "__version__",
    # === Poincaré Ball Model ===
    "mobius_add",
    "poincare_distance",
    "exp_map",
    "log_map",
    # === Lorentz (Hyperboloid) Model ===
    "minkowski_inner",
    "minkowski_norm",
    "lorentz_distance",
    "lorentz_distance_squared",
    "exp_map_lorentz",
    "log_map_lorentz",
    "parallel_transport_lorentz",
    "project_to_hyperboloid",
    "lorentz_centroid",
    "check_on_hyperboloid",
    # === Model Conversions ===
    "poincare_to_lorentz",
    "lorentz_to_poincare",
]

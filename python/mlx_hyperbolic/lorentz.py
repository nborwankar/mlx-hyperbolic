"""
Lorentz (Hyperboloid) model operations for hyperbolic geometry.

The Lorentz model represents hyperbolic space as the upper sheet of a hyperboloid
in Minkowski space. It is numerically more stable than the Poincaré ball model,
especially for points far from the origin.

Key advantages over Poincaré:
1. No boundary singularities (Poincaré has issues as ||x|| → 1)
2. Simpler distance computation (arccosh of inner product)
3. Better gradient flow (no conformal factor distortion)

Coordinates:
- Points have n+1 dimensions: x = (x_0, x_1, ..., x_n)
- Time component x_0 > 0 (positive sheet)
- Constraint: -x_0² + x_1² + ... + x_n² = -1/c (for curvature c)

References:
- Nickel & Kiela, "Learning Continuous Hierarchies in the Lorentz Model" (2018)
- Law et al., "Lorentzian Distance Learning for Hyperbolic Representations" (2019)
- Chami et al., "Hyperbolic Graph Convolutional Neural Networks" (HGCN, 2019)
"""

import mlx.core as mx
from typing import Optional

# Small epsilon for numerical stability
EPS = 1e-6


def minkowski_inner(x: mx.array, y: mx.array, keepdims: bool = False) -> mx.array:
    """
    Compute the Minkowski (Lorentzian) inner product.

    ⟨x, y⟩_L = -x₀y₀ + x₁y₁ + x₂y₂ + ... + xₙyₙ

    The first component has negative sign (time-like), rest are positive (space-like).

    Args:
        x: Points on hyperboloid, shape (..., n+1)
        y: Points on hyperboloid, shape (..., n+1)
        keepdims: Keep the last dimension

    Returns:
        Minkowski inner product, shape (...,) or (..., 1)
    """
    # Split into time and space components
    x_time = x[..., 0:1]  # First component (time-like)
    x_space = x[..., 1:]  # Rest (space-like)
    y_time = y[..., 0:1]
    y_space = y[..., 1:]

    # Minkowski metric: -dt² + dx²
    inner = -x_time * y_time + mx.sum(x_space * y_space, axis=-1, keepdims=True)

    if keepdims:
        return inner
    return mx.squeeze(inner, axis=-1)


def minkowski_norm(x: mx.array, keepdims: bool = False) -> mx.array:
    """
    Compute the Minkowski norm: √|⟨x, x⟩_L|

    For points on the hyperboloid, ⟨x, x⟩_L = -1/c (negative).
    For tangent vectors, ⟨v, v⟩_L ≥ 0 (non-negative).

    Args:
        x: Vector in Minkowski space
        keepdims: Keep the last dimension

    Returns:
        Minkowski norm
    """
    inner = minkowski_inner(x, x, keepdims=keepdims)
    return mx.sqrt(mx.abs(inner) + EPS)


def project_to_hyperboloid(x: mx.array, c: float = 1.0) -> mx.array:
    """
    Project a point onto the hyperboloid.

    Given space components (x₁, ..., xₙ), compute x₀ such that
    -x₀² + ||x_space||² = -1/c, with x₀ > 0.

    Args:
        x: Points with space components, shape (..., n)
           OR points already in n+1 dims (will recompute x₀)
        c: Curvature parameter (default 1.0)

    Returns:
        Points on hyperboloid, shape (..., n+1)
    """
    if x.shape[-1] == 1:
        # Edge case: 1D space, just add time component
        x_space = x
    else:
        # Use all components as space (will prepend time)
        x_space = x

    # Compute ||x_space||²
    space_norm_sq = mx.sum(x_space * x_space, axis=-1, keepdims=True)

    # x₀ = √(1/c + ||x_space||²)
    x_time = mx.sqrt(1.0 / c + space_norm_sq)

    return mx.concatenate([x_time, x_space], axis=-1)


def lorentz_distance(x: mx.array, y: mx.array, c: float = 1.0) -> mx.array:
    """
    Compute geodesic distance on the hyperboloid.

    d(x, y) = (1/√c) * arccosh(-c * ⟨x, y⟩_L)

    For c = 1: d(x, y) = arccosh(-⟨x, y⟩_L)

    This is much simpler than the Poincaré distance formula!

    Args:
        x: Points on hyperboloid, shape (..., n+1)
        y: Points on hyperboloid, shape (..., n+1)
        c: Curvature parameter (default 1.0)

    Returns:
        Geodesic distances, shape (...)
    """
    inner = minkowski_inner(x, y)

    # Clamp for numerical stability (inner should be ≤ -1/c for points on hyperboloid)
    # arccosh is only defined for values ≥ 1
    clamped = mx.maximum(-c * inner, 1.0 + EPS)

    return mx.arccosh(clamped) / mx.sqrt(c)


def lorentz_distance_squared(x: mx.array, y: mx.array, c: float = 1.0) -> mx.array:
    """
    Compute squared geodesic distance (avoids sqrt, useful for optimization).

    Args:
        x: Points on hyperboloid, shape (..., n+1)
        y: Points on hyperboloid, shape (..., n+1)
        c: Curvature parameter (default 1.0)

    Returns:
        Squared geodesic distances, shape (...)
    """
    d = lorentz_distance(x, y, c)
    return d * d


def exp_map_lorentz(v: mx.array, x: mx.array, c: float = 1.0) -> mx.array:
    """
    Exponential map: project tangent vector to the hyperboloid.

    exp_x(v) = cosh(√c * ||v||_L) * x + sinh(√c * ||v||_L) * v / (√c * ||v||_L)

    where ||v||_L = √⟨v, v⟩_L is the Minkowski norm of the tangent vector.

    Args:
        v: Tangent vector at x, shape (..., n+1)
           Must satisfy ⟨v, x⟩_L = 0 (orthogonal in Minkowski sense)
        x: Base point on hyperboloid, shape (..., n+1)
        c: Curvature parameter (default 1.0)

    Returns:
        Point on hyperboloid, shape (..., n+1)
    """
    sqrt_c = mx.sqrt(c)

    # Compute Minkowski norm of tangent vector
    v_norm = minkowski_norm(v, keepdims=True)

    # Handle small norms (return base point)
    v_norm_safe = mx.maximum(v_norm, EPS)

    # exp_x(v) = cosh(√c ||v||) x + sinh(√c ||v||) v / (√c ||v||)
    scaled_norm = sqrt_c * v_norm_safe

    cosh_term = mx.cosh(scaled_norm)
    sinh_term = mx.sinh(scaled_norm)

    result = cosh_term * x + sinh_term * v / (sqrt_c * v_norm_safe)

    # For very small v, return x
    result = mx.where(v_norm < EPS, x, result)

    return result


def log_map_lorentz(y: mx.array, x: mx.array, c: float = 1.0) -> mx.array:
    """
    Logarithmic map: project point to tangent space at x.

    log_x(y) = d(x, y) * (y + c * ⟨x, y⟩_L * x) / ||y + c * ⟨x, y⟩_L * x||_L

    where d(x, y) is the geodesic distance.

    Args:
        y: Point on hyperboloid, shape (..., n+1)
        x: Base point on hyperboloid, shape (..., n+1)
        c: Curvature parameter (default 1.0)

    Returns:
        Tangent vector at x, shape (..., n+1)
    """
    # Compute inner product and distance
    inner = minkowski_inner(x, y, keepdims=True)
    dist = lorentz_distance(x, y, c)

    # Direction in tangent space: y + c * ⟨x, y⟩_L * x
    # This is the component of y orthogonal to x (in Minkowski sense)
    direction = y + c * inner * x

    # Normalize by Minkowski norm
    dir_norm = minkowski_norm(direction, keepdims=True)
    dir_norm_safe = mx.maximum(dir_norm, EPS)

    # Scale by distance
    result = mx.expand_dims(dist, axis=-1) * direction / dir_norm_safe

    # For x ≈ y, return zero vector
    result = mx.where(dist[..., None] < EPS, mx.zeros_like(result), result)

    return result


def parallel_transport_lorentz(v: mx.array, x: mx.array, y: mx.array, c: float = 1.0) -> mx.array:
    """
    Parallel transport a tangent vector from x to y along the geodesic.

    This is essential for Riemannian optimization (e.g., Riemannian SGD).

    P_{x→y}(v) = v - (⟨y, v⟩_L / (1 - ⟨x, y⟩_L)) * (x + y)

    Args:
        v: Tangent vector at x, shape (..., n+1)
        x: Source point on hyperboloid, shape (..., n+1)
        y: Target point on hyperboloid, shape (..., n+1)
        c: Curvature parameter (default 1.0)

    Returns:
        Transported tangent vector at y, shape (..., n+1)
    """
    inner_xy = minkowski_inner(x, y, keepdims=True)
    inner_yv = minkowski_inner(y, v, keepdims=True)

    # Coefficient for the correction term
    denom = 1.0 - c * inner_xy
    denom_safe = mx.where(mx.abs(denom) < EPS, EPS * mx.sign(denom), denom)

    coef = c * inner_yv / denom_safe

    result = v - coef * (x + y)

    return result


def lorentz_centroid(
    points: mx.array, weights: Optional[mx.array] = None, c: float = 1.0
) -> mx.array:
    """
    Compute the Einstein midpoint (Lorentzian centroid) of points.

    This is a weighted average in the ambient space, projected back to hyperboloid.

    centroid = Σ wᵢ xᵢ / ||Σ wᵢ xᵢ||_L

    then project to satisfy hyperboloid constraint.

    Args:
        points: Points on hyperboloid, shape (n_points, dim+1)
        weights: Optional weights, shape (n_points,). Default: uniform.
        c: Curvature parameter (default 1.0)

    Returns:
        Centroid on hyperboloid, shape (dim+1,)
    """
    if weights is None:
        weights = mx.ones(points.shape[0]) / points.shape[0]
    else:
        weights = weights / mx.sum(weights)  # Normalize

    # Weighted sum in ambient space
    weighted_sum = mx.sum(weights[:, None] * points, axis=0)

    # Project back to hyperboloid
    # For Lorentz, we need to normalize by sqrt(-⟨x, x⟩_L)
    inner = minkowski_inner(weighted_sum, weighted_sum)
    norm = mx.sqrt(mx.abs(inner) + EPS)

    # Normalize and ensure on hyperboloid
    centroid = weighted_sum / norm

    # Ensure x_0 > 0 (upper sheet)
    centroid = mx.where(centroid[0] < 0, -centroid, centroid)

    return centroid


def poincare_to_lorentz(x: mx.array, c: float = 1.0) -> mx.array:
    """
    Convert from Poincaré ball to Lorentz (hyperboloid) model.

    Given x in Poincaré ball (||x|| < 1/√c):
    y₀ = (1 + c||x||²) / (1 - c||x||²)
    yᵢ = 2√c xᵢ / (1 - c||x||²)    for i = 1, ..., n

    Args:
        x: Points in Poincaré ball, shape (..., n)
        c: Curvature parameter (default 1.0)

    Returns:
        Points on hyperboloid, shape (..., n+1)
    """
    x_norm_sq = mx.sum(x * x, axis=-1, keepdims=True)

    denom = 1.0 - c * x_norm_sq
    denom_safe = mx.maximum(denom, EPS)

    # Time component
    y_time = (1.0 + c * x_norm_sq) / denom_safe

    # Space components
    y_space = 2.0 * mx.sqrt(c) * x / denom_safe

    return mx.concatenate([y_time, y_space], axis=-1)


def lorentz_to_poincare(y: mx.array, c: float = 1.0) -> mx.array:
    """
    Convert from Lorentz (hyperboloid) to Poincaré ball model.

    Given y on hyperboloid:
    xᵢ = yᵢ / (√c * (1 + y₀))    for i = 1, ..., n

    Args:
        y: Points on hyperboloid, shape (..., n+1)
        c: Curvature parameter (default 1.0)

    Returns:
        Points in Poincaré ball, shape (..., n)
    """
    y_time = y[..., 0:1]
    y_space = y[..., 1:]

    denom = mx.sqrt(c) * (1.0 + y_time)
    denom_safe = mx.maximum(denom, EPS)

    return y_space / denom_safe


def check_on_hyperboloid(x: mx.array, c: float = 1.0, tol: float = 1e-4) -> mx.array:
    """
    Check if points satisfy the hyperboloid constraint.

    -x₀² + x₁² + ... + xₙ² should equal -1/c

    Args:
        x: Points to check, shape (..., n+1)
        c: Curvature parameter (default 1.0)
        tol: Tolerance for constraint check

    Returns:
        Boolean array indicating if constraint is satisfied
    """
    inner = minkowski_inner(x, x)
    expected = -1.0 / c
    return mx.abs(inner - expected) < tol

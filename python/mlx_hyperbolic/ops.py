"""
Poincaré ball model operations for hyperbolic geometry.

This module provides hyperbolic geometry operations for the Poincaré ball model,
a popular representation of hyperbolic space where points lie within the unit ball.

Key operations:
- mobius_add: Möbius addition (hyperbolic "addition")
- poincare_distance: Geodesic distance in the Poincaré ball
- exp_map: Project tangent vector to manifold
- log_map: Project point back to tangent space

For the numerically stable Lorentz (hyperboloid) model, see lorentz.py.
"""

import mlx.core as mx


# ==============================================================================
# Hyperbolic Geometry Operations (Poincaré Ball Model)
# ==============================================================================


def mobius_add(x: mx.array, y: mx.array, c: float = 1.0) -> mx.array:
    """
    Möbius addition in the Poincaré ball.

    Computes x ⊕_c y using the formula:
    (x ⊕_c y) = ((1 + 2c<x,y> + c||y||²)x + (1 - c||x||²)y) / (1 + 2c<x,y> + c²||x||²||y||²)

    Args:
        x: First vector in the Poincaré ball (||x|| < 1/√c)
        y: Second vector in the Poincaré ball (||y|| < 1/√c)
        c: Curvature parameter (default 1.0)

    Returns:
        Result of Möbius addition x ⊕_c y

    Example:
        >>> x = mx.array([0.3, 0.4]) * 0.5
        >>> y = mx.array([0.5, 0.0]) * 0.5
        >>> mobius_add(x, y)
        array([...], dtype=float32)
    """
    # Compute norms and dot product
    x_norm_sq = mx.sum(x * x, keepdims=True)
    y_norm_sq = mx.sum(y * y, keepdims=True)
    xy_dot = mx.sum(x * y, keepdims=True)

    # Möbius addition formula
    num_x_coef = 1.0 + 2.0 * c * xy_dot + c * y_norm_sq
    num_y_coef = 1.0 - c * x_norm_sq
    denom = 1.0 + 2.0 * c * xy_dot + c * c * x_norm_sq * y_norm_sq

    result = (num_x_coef * x + num_y_coef * y) / denom
    return result


def poincare_distance(x: mx.array, y: mx.array, c: float = 1.0) -> mx.array:
    """
    Compute geodesic distance in the Poincaré ball.

    d_c(x, y) = (2/√c) * arctanh(√c * ||(-x) ⊕_c y||)

    Args:
        x: First point in the Poincaré ball
        y: Second point in the Poincaré ball
        c: Curvature parameter (default 1.0)

    Returns:
        Geodesic distance between x and y
    """
    # Compute -x ⊕_c y
    neg_x = -x
    diff = mobius_add(neg_x, y, c)

    # Compute norm of difference
    diff_norm = mx.sqrt(mx.sum(diff * diff, keepdims=True))

    # Distance formula
    sqrt_c = mx.sqrt(mx.array(c))
    distance = (2.0 / sqrt_c) * mx.arctanh(sqrt_c * diff_norm)

    return mx.squeeze(distance)


def exp_map(v: mx.array, x: mx.array, c: float = 1.0) -> mx.array:
    """
    Exponential map: project tangent vector to the Poincaré ball.

    exp_x^c(v) = x ⊕_c (tanh(√c * λ_x^c * ||v|| / 2) * v / (√c * ||v||))

    where λ_x^c = 2 / (1 - c||x||²) is the conformal factor.

    Args:
        v: Tangent vector at point x
        x: Base point in the Poincaré ball
        c: Curvature parameter (default 1.0)

    Returns:
        Point in the Poincaré ball
    """
    sqrt_c = mx.sqrt(mx.array(c))
    v_norm = mx.sqrt(mx.sum(v * v, keepdims=True))
    x_norm_sq = mx.sum(x * x, keepdims=True)

    # Conformal factor
    lambda_x = 2.0 / (1.0 - c * x_norm_sq)

    # Scaled tangent vector
    tanh_arg = sqrt_c * lambda_x * v_norm / 2.0
    tanh_val = mx.tanh(tanh_arg)

    # Avoid division by zero
    v_normalized = v / mx.maximum(v_norm, 1e-8)
    scaled_v = tanh_val * v_normalized / sqrt_c

    return mobius_add(x, scaled_v, c)


def log_map(y: mx.array, x: mx.array, c: float = 1.0) -> mx.array:
    """
    Logarithmic map: project point from Poincaré ball to tangent space.

    log_x^c(y) = (2 / (√c * λ_x^c)) * arctanh(√c * ||(-x) ⊕_c y||) * ((-x) ⊕_c y) / ||(-x) ⊕_c y||

    Args:
        y: Point in the Poincaré ball
        x: Base point in the Poincaré ball
        c: Curvature parameter (default 1.0)

    Returns:
        Tangent vector at point x
    """
    sqrt_c = mx.sqrt(mx.array(c))
    x_norm_sq = mx.sum(x * x, keepdims=True)

    # Conformal factor
    lambda_x = 2.0 / (1.0 - c * x_norm_sq)

    # Compute -x ⊕_c y
    diff = mobius_add(-x, y, c)
    diff_norm = mx.sqrt(mx.sum(diff * diff, keepdims=True))

    # Log map formula
    arctanh_val = mx.arctanh(sqrt_c * diff_norm)
    scale = (2.0 / (sqrt_c * lambda_x)) * arctanh_val

    # Avoid division by zero
    diff_normalized = diff / mx.maximum(diff_norm, 1e-8)

    return scale * diff_normalized

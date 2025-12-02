"""
mlx_hyperbolic: Hardware-Accelerated Hyperbolic Embeddings

Fast hyperbolic neural network primitives on Apple Silicon using GPU
Texture Mapping Units (TMUs) for transcendental function computation.

Example:
    >>> import mlx.core as mx
    >>> from mlx_hyperbolic import fast_exp, mobius_add
    >>>
    >>> x = mx.array([0.5, 1.0, 2.0], dtype=mx.float16)
    >>> result = fast_exp(x)  # TMU-accelerated exponential
    >>>
    >>> u = mx.random.uniform(shape=(16,), dtype=mx.float16) * 0.5
    >>> v = mx.random.uniform(shape=(16,), dtype=mx.float16) * 0.5
    >>> w = mobius_add(u, v)  # Hyperbolic addition
"""

__version__ = "0.1.0"
__author__ = "Nitin Borwankar"

# Import from ops module
from .ops import (
    # LUT generation
    generate_exp_lut,
    generate_log_lut,
    generate_tanh_lut,
    # Fast operations
    fast_exp,
    fast_log,
    fast_tanh,
    # Hyperbolic operations
    mobius_add,
    poincare_distance,
    exp_map,
    log_map,
    # LUT cache management
    LUTCache,
    clear_lut_cache,
)

__all__ = [
    # Version
    "__version__",
    # LUT generation
    "generate_exp_lut",
    "generate_log_lut",
    "generate_tanh_lut",
    # Fast operations
    "fast_exp",
    "fast_log",
    "fast_tanh",
    # Hyperbolic operations
    "mobius_add",
    "poincare_distance",
    "exp_map",
    "log_map",
    # Cache management
    "LUTCache",
    "clear_lut_cache",
]

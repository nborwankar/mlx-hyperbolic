"""
Correctness tests for Poincaré ball model operations.

Tests verify mathematical properties (identities, symmetry, round-trips)
rather than comparing to a reference implementation.

Requires: mlx, numpy
Run with: python -m pytest tests/test_poincare.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import mlx.core as mx
import numpy as np
import pytest

from mlx_hyperbolic import mobius_add, poincare_distance, exp_map, log_map


def to_np(x):
    return np.array(x, dtype=np.float32)


def allclose(a, b, atol=1e-5):
    return np.allclose(to_np(a), to_np(b), atol=atol)


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------

def random_in_ball(shape, max_norm=0.8):
    """Generate random points strictly inside the Poincaré ball."""
    x = mx.random.normal(shape=shape)
    x = x / mx.sqrt(mx.sum(x * x, keepdims=True))  # unit norm
    r = mx.random.uniform(shape=(1,)) * max_norm
    return x * r


# ===========================================================================
# mobius_add
# ===========================================================================

class TestMobiusAdd:

    def test_right_identity(self):
        """x ⊕ 0 = x"""
        x = mx.array([0.3, 0.4, -0.2])
        zero = mx.zeros(3)
        result = mobius_add(x, zero)
        assert allclose(result, x)

    def test_left_identity(self):
        """0 ⊕ y = y"""
        y = mx.array([0.1, -0.3, 0.2])
        zero = mx.zeros(3)
        result = mobius_add(zero, y)
        assert allclose(result, y)

    def test_inverse(self):
        """x ⊕ (-x) ≈ 0 for c=1 (the Möbius inverse of x is -x)."""
        x = mx.array([0.3, -0.2, 0.1])
        result = mobius_add(x, -x)
        assert allclose(result, mx.zeros(3), atol=1e-5)

    def test_result_stays_in_ball(self):
        """||x ⊕ y|| < 1 for all x, y in the unit ball."""
        for _ in range(20):
            x = random_in_ball((4,))
            y = random_in_ball((4,))
            result = mobius_add(x, y)
            norm = float(mx.sqrt(mx.sum(result * result)))
            assert norm < 1.0, f"Result norm {norm} >= 1"

    def test_origin_is_identity_element(self):
        """Adding origin from both sides should return the other point."""
        pts = [
            mx.array([0.5, 0.0]),
            mx.array([0.0, -0.5]),
            mx.array([0.1, 0.2]),
        ]
        zero = mx.zeros(2)
        for p in pts:
            assert allclose(mobius_add(p, zero), p)
            assert allclose(mobius_add(zero, p), p)

    def test_curvature_zero_reduces_to_euclidean(self):
        """At c→0 Möbius addition should approach Euclidean addition."""
        x = mx.array([0.3, 0.4])
        y = mx.array([0.1, -0.2])
        result = mobius_add(x, y, c=1e-8)
        assert allclose(result, x + y, atol=1e-4)


# ===========================================================================
# poincare_distance
# ===========================================================================

class TestPoincareDistance:

    def test_self_distance_is_zero(self):
        """d(x, x) = 0"""
        x = mx.array([0.3, 0.4, -0.1])
        d = poincare_distance(x, x)
        assert float(d) < 1e-5

    def test_symmetry(self):
        """d(x, y) = d(y, x)"""
        x = mx.array([0.3, 0.0, 0.0])
        y = mx.array([0.0, 0.4, 0.0])
        d_xy = float(poincare_distance(x, y))
        d_yx = float(poincare_distance(y, x))
        assert abs(d_xy - d_yx) < 1e-5

    def test_distance_is_nonnegative(self):
        """d(x, y) >= 0"""
        for _ in range(20):
            x = random_in_ball((4,))
            y = random_in_ball((4,))
            d = float(poincare_distance(x, y))
            assert d >= -1e-6, f"Negative distance: {d}"

    def test_triangle_inequality(self):
        """d(x, z) <= d(x, y) + d(y, z)"""
        x = mx.array([0.3, 0.0, 0.0])
        y = mx.array([0.0, 0.3, 0.0])
        z = mx.array([0.0, 0.0, 0.3])
        d_xz = float(poincare_distance(x, z))
        d_xy = float(poincare_distance(x, y))
        d_yz = float(poincare_distance(y, z))
        assert d_xz <= d_xy + d_yz + 1e-5

    def test_distance_from_origin_closed_form(self):
        """d(0, x) = 2 * arctanh(||x||) for c=1."""
        x = mx.array([0.3] + [0.0] * 3)
        origin = mx.zeros(4)
        d = float(poincare_distance(origin, x))
        expected = 2.0 * float(mx.arctanh(mx.array(0.3)))
        assert abs(d - expected) < 1e-5, f"d={d}, expected={expected}"

    def test_farther_points_have_larger_distance(self):
        """Points farther from origin should have larger distance from origin."""
        origin = mx.zeros(3)
        x_near = mx.array([0.1, 0.0, 0.0])
        x_far = mx.array([0.5, 0.0, 0.0])
        d_near = float(poincare_distance(origin, x_near))
        d_far = float(poincare_distance(origin, x_far))
        assert d_near < d_far


# ===========================================================================
# exp_map / log_map round-trips
# ===========================================================================

class TestExpLogMap:

    def test_exp_then_log_round_trip(self):
        """log_x(exp_x(v)) ≈ v"""
        x = mx.array([0.1, 0.2, -0.1])
        v = mx.array([0.05, -0.03, 0.02])
        y = exp_map(v, x)
        v_recovered = log_map(y, x)
        assert allclose(v_recovered, v, atol=1e-4), (
            f"v={to_np(v)}, recovered={to_np(v_recovered)}"
        )

    def test_log_then_exp_round_trip(self):
        """exp_x(log_x(y)) ≈ y"""
        x = mx.array([0.1, 0.2, -0.1])
        y = mx.array([0.3, -0.1, 0.2])
        v = log_map(y, x)
        y_recovered = exp_map(v, x)
        assert allclose(y_recovered, y, atol=1e-4), (
            f"y={to_np(y)}, recovered={to_np(y_recovered)}"
        )

    def test_exp_map_zero_tangent(self):
        """exp_x(0) = x"""
        x = mx.array([0.2, -0.3, 0.1])
        zero_v = mx.zeros(3)
        result = exp_map(zero_v, x)
        assert allclose(result, x, atol=1e-5)

    def test_exp_map_from_origin(self):
        """exp_0(v) should land inside the ball."""
        origin = mx.zeros(3)
        v = mx.array([0.5, -0.3, 0.2])
        result = exp_map(v, origin)
        norm = float(mx.sqrt(mx.sum(result * result)))
        assert norm < 1.0, f"exp_map result norm {norm} >= 1"

    def test_exp_map_result_in_ball(self):
        """exp_map should always return points inside the ball."""
        for _ in range(20):
            x = random_in_ball((4,), max_norm=0.5)
            v = mx.random.normal(shape=(4,)) * 0.1  # small tangent vector
            result = exp_map(v, x)
            norm = float(mx.sqrt(mx.sum(result * result)))
            assert norm < 1.0, f"exp_map result norm {norm} >= 1"

    def test_log_map_self_is_zero(self):
        """log_x(x) ≈ 0"""
        x = mx.array([0.2, -0.3, 0.1])
        v = log_map(x, x)
        assert allclose(v, mx.zeros(3), atol=1e-5)

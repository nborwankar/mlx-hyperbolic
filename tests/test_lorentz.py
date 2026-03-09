"""
Correctness tests for Lorentz (hyperboloid) model operations.

Tests verify mathematical properties: hyperboloid constraint, distance axioms,
round-trips, parallel transport, model conversions, and centroid behavior.

Requires: mlx, numpy
Run with: python -m pytest tests/test_lorentz.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

import mlx.core as mx
import numpy as np
import pytest

from mlx_hyperbolic import (
    minkowski_inner,
    minkowski_norm,
    lorentz_distance,
    lorentz_distance_squared,
    exp_map_lorentz,
    log_map_lorentz,
    parallel_transport_lorentz,
    project_to_hyperboloid,
    lorentz_centroid,
    check_on_hyperboloid,
    poincare_to_lorentz,
    lorentz_to_poincare,
)


def to_np(x):
    return np.array(x, dtype=np.float32)


def allclose(a, b, atol=1e-5):
    return np.allclose(to_np(a), to_np(b), atol=atol)


def make_hyperboloid_point(space_coords):
    """Create a point on the hyperboloid from space coordinates."""
    return project_to_hyperboloid(mx.array(space_coords))


# ===========================================================================
# minkowski_inner / minkowski_norm
# ===========================================================================

class TestMinkowskiInner:

    def test_hyperboloid_self_inner_is_minus_one(self):
        """⟨x, x⟩_L = -1 for points on the hyperboloid (c=1)."""
        p = make_hyperboloid_point([0.3, -0.2, 0.1])
        inner = float(minkowski_inner(p, p))
        assert abs(inner - (-1.0)) < 1e-4, f"inner={inner}, expected=-1"

    def test_symmetry(self):
        """⟨x, y⟩_L = ⟨y, x⟩_L"""
        x = make_hyperboloid_point([0.3, 0.0])
        y = make_hyperboloid_point([0.0, 0.5])
        assert abs(float(minkowski_inner(x, y)) - float(minkowski_inner(y, x))) < 1e-5

    def test_origin_is_unit_time(self):
        """The hyperboloid origin is (1, 0, 0, ..., 0) and ⟨o, o⟩ = -1."""
        origin = make_hyperboloid_point([0.0, 0.0, 0.0])
        assert allclose(origin, mx.array([1.0, 0.0, 0.0, 0.0]))
        assert abs(float(minkowski_inner(origin, origin)) + 1.0) < 1e-5


# ===========================================================================
# project_to_hyperboloid / check_on_hyperboloid
# ===========================================================================

class TestProjection:

    def test_projected_point_satisfies_constraint(self):
        """project_to_hyperboloid output should pass check_on_hyperboloid."""
        space = mx.array([0.5, -0.3, 0.7])
        p = project_to_hyperboloid(space)
        assert bool(check_on_hyperboloid(p)), f"Point {to_np(p)} not on hyperboloid"

    def test_time_component_positive(self):
        """Projected points should have x_0 > 0 (upper sheet)."""
        for _ in range(10):
            space = mx.random.normal(shape=(4,))
            p = project_to_hyperboloid(space)
            assert float(p[0]) > 0

    def test_zero_space_gives_origin(self):
        """project_to_hyperboloid([0,0,...]) = (1, 0, 0, ...)"""
        p = project_to_hyperboloid(mx.zeros(3))
        expected = mx.array([1.0, 0.0, 0.0, 0.0])
        assert allclose(p, expected)


# ===========================================================================
# lorentz_distance
# ===========================================================================

class TestLorentzDistance:

    def test_self_distance_is_zero(self):
        """d(x, x) = 0"""
        x = make_hyperboloid_point([0.3, -0.2])
        d = float(lorentz_distance(x, x))
        assert d < 1e-3, f"Self-distance={d}"

    def test_symmetry(self):
        """d(x, y) = d(y, x)"""
        x = make_hyperboloid_point([0.3, 0.0])
        y = make_hyperboloid_point([0.0, 0.5])
        d_xy = float(lorentz_distance(x, y))
        d_yx = float(lorentz_distance(y, x))
        assert abs(d_xy - d_yx) < 1e-5

    def test_distance_nonnegative(self):
        """d(x, y) >= 0"""
        for _ in range(20):
            x = make_hyperboloid_point(to_np(mx.random.normal(shape=(3,)) * 0.5))
            y = make_hyperboloid_point(to_np(mx.random.normal(shape=(3,)) * 0.5))
            d = float(lorentz_distance(x, y))
            assert d >= -1e-5, f"Negative distance: {d}"

    def test_triangle_inequality(self):
        """d(x, z) <= d(x, y) + d(y, z)"""
        x = make_hyperboloid_point([0.3, 0.0, 0.0])
        y = make_hyperboloid_point([0.0, 0.4, 0.0])
        z = make_hyperboloid_point([0.0, 0.0, 0.5])
        d_xz = float(lorentz_distance(x, z))
        d_xy = float(lorentz_distance(x, y))
        d_yz = float(lorentz_distance(y, z))
        assert d_xz <= d_xy + d_yz + 1e-5

    def test_squared_distance_consistent(self):
        """d²(x, y) = d(x, y)²"""
        x = make_hyperboloid_point([0.3, 0.1])
        y = make_hyperboloid_point([-0.2, 0.4])
        d = float(lorentz_distance(x, y))
        d_sq = float(lorentz_distance_squared(x, y))
        assert abs(d_sq - d * d) < 1e-4

    def test_farther_points_larger_distance(self):
        """More spread-out space coords → larger distance from origin."""
        origin = make_hyperboloid_point([0.0, 0.0])
        near = make_hyperboloid_point([0.1, 0.0])
        far = make_hyperboloid_point([1.0, 0.0])
        d_near = float(lorentz_distance(origin, near))
        d_far = float(lorentz_distance(origin, far))
        assert d_near < d_far


# ===========================================================================
# exp_map_lorentz / log_map_lorentz
# ===========================================================================

class TestLorentzExpLogMap:

    def _make_tangent(self, x, space_v):
        """Create a tangent vector at x orthogonal in Minkowski sense.

        For a tangent vector v at x: ⟨v, x⟩_L = 0.
        Given space components, compute v_0 = (v_space · x_space) / x_0.
        """
        x_np = to_np(x)
        v_space = np.array(space_v, dtype=np.float32)
        # ⟨v, x⟩_L = -v0*x0 + v_space·x_space = 0 → v0 = v_space·x_space / x0
        v0 = np.dot(v_space, x_np[1:]) / x_np[0]
        return mx.array(np.concatenate([[v0], v_space]))

    def test_exp_result_on_hyperboloid(self):
        """exp_x(v) should land on the hyperboloid."""
        x = make_hyperboloid_point([0.3, -0.2])
        v = self._make_tangent(x, [0.1, -0.05])
        y = exp_map_lorentz(v, x)
        assert bool(check_on_hyperboloid(y, tol=1e-3)), (
            f"exp result not on hyperboloid: inner={float(minkowski_inner(y, y))}"
        )

    def test_exp_zero_tangent(self):
        """exp_x(0) = x"""
        x = make_hyperboloid_point([0.3, -0.2, 0.1])
        zero_v = mx.zeros(4)
        result = exp_map_lorentz(zero_v, x)
        assert allclose(result, x, atol=1e-4)

    def test_exp_then_log_round_trip(self):
        """log_x(exp_x(v)) ≈ v"""
        x = make_hyperboloid_point([0.2, -0.1])
        v = self._make_tangent(x, [0.1, -0.05])
        y = exp_map_lorentz(v, x)
        v_rec = log_map_lorentz(y, x)
        assert allclose(v_rec, v, atol=1e-3), (
            f"v={to_np(v)}, recovered={to_np(v_rec)}"
        )

    def test_log_then_exp_round_trip(self):
        """exp_x(log_x(y)) ≈ y"""
        x = make_hyperboloid_point([0.1, 0.2])
        y = make_hyperboloid_point([0.3, -0.2])
        v = log_map_lorentz(y, x)
        y_rec = exp_map_lorentz(v, x)
        assert allclose(y_rec, y, atol=1e-3), (
            f"y={to_np(y)}, recovered={to_np(y_rec)}"
        )

    def test_log_self_is_zero(self):
        """log_x(x) ≈ 0"""
        x = make_hyperboloid_point([0.3, -0.2])
        v = log_map_lorentz(x, x)
        assert allclose(v, mx.zeros(3), atol=1e-4)


# ===========================================================================
# parallel_transport_lorentz
# ===========================================================================

class TestParallelTransport:

    def _make_tangent(self, x, space_v):
        x_np = to_np(x)
        v_space = np.array(space_v, dtype=np.float32)
        v0 = np.dot(v_space, x_np[1:]) / x_np[0]
        return mx.array(np.concatenate([[v0], v_space]))

    def test_transported_vector_tangent_at_target(self):
        """P_{x→y}(v) should be tangent at y: ⟨P(v), y⟩_L ≈ 0."""
        x = make_hyperboloid_point([0.3, -0.2])
        y = make_hyperboloid_point([0.0, 0.4])
        v = self._make_tangent(x, [0.1, -0.05])
        transported = parallel_transport_lorentz(v, x, y)
        inner = float(minkowski_inner(transported, y))
        assert abs(inner) < 1e-3, f"⟨P(v), y⟩_L = {inner}, expected ≈ 0"

    def test_transport_preserves_norm(self):
        """||P_{x→y}(v)||_L ≈ ||v||_L (parallel transport is an isometry)."""
        x = make_hyperboloid_point([0.2, 0.1])
        y = make_hyperboloid_point([-0.1, 0.3])
        v = self._make_tangent(x, [0.15, -0.1])
        transported = parallel_transport_lorentz(v, x, y)
        norm_v = float(minkowski_norm(v))
        norm_t = float(minkowski_norm(transported))
        assert abs(norm_v - norm_t) < 1e-3, f"||v||={norm_v}, ||P(v)||={norm_t}"

    def test_transport_to_self_is_identity(self):
        """P_{x→x}(v) = v"""
        x = make_hyperboloid_point([0.3, -0.2])
        v = self._make_tangent(x, [0.1, 0.05])
        transported = parallel_transport_lorentz(v, x, x)
        assert allclose(transported, v, atol=1e-3)


# ===========================================================================
# Model conversions: Poincaré ↔ Lorentz
# ===========================================================================

class TestModelConversions:

    def test_poincare_to_lorentz_origin(self):
        """Poincaré origin (0,...,0) maps to Lorentz origin (1, 0,...,0)."""
        p = mx.zeros(3)
        l = poincare_to_lorentz(p)
        expected = mx.array([1.0, 0.0, 0.0, 0.0])
        assert allclose(l, expected, atol=1e-5)

    def test_lorentz_to_poincare_origin(self):
        """Lorentz origin (1, 0,...,0) maps to Poincaré origin (0,...,0)."""
        l = mx.array([1.0, 0.0, 0.0, 0.0])
        p = lorentz_to_poincare(l)
        assert allclose(p, mx.zeros(3), atol=1e-5)

    def test_round_trip_poincare_to_lorentz_to_poincare(self):
        """lorentz_to_poincare(poincare_to_lorentz(x)) ≈ x"""
        x = mx.array([0.3, -0.2, 0.1])
        recovered = lorentz_to_poincare(poincare_to_lorentz(x))
        assert allclose(recovered, x, atol=1e-4)

    def test_round_trip_lorentz_to_poincare_to_lorentz(self):
        """poincare_to_lorentz(lorentz_to_poincare(y)) ≈ y"""
        y = make_hyperboloid_point([0.3, -0.2, 0.1])
        recovered = poincare_to_lorentz(lorentz_to_poincare(y))
        assert allclose(recovered, y, atol=1e-4)

    def test_converted_point_on_hyperboloid(self):
        """poincare_to_lorentz should produce valid hyperboloid points."""
        x = mx.array([0.3, -0.4, 0.1])
        l = poincare_to_lorentz(x)
        assert bool(check_on_hyperboloid(l, tol=1e-3))

    def test_distances_match_across_models(self):
        """Poincaré distance should equal Lorentz distance for same points."""
        from mlx_hyperbolic import poincare_distance

        x_p = mx.array([0.3, -0.2])
        y_p = mx.array([0.1, 0.4])

        d_poincare = float(poincare_distance(x_p, y_p))

        x_l = poincare_to_lorentz(x_p)
        y_l = poincare_to_lorentz(y_p)
        d_lorentz = float(lorentz_distance(x_l, y_l))

        assert abs(d_poincare - d_lorentz) < 1e-3, (
            f"Poincaré d={d_poincare}, Lorentz d={d_lorentz}"
        )


# ===========================================================================
# lorentz_centroid
# ===========================================================================

class TestLorentzCentroid:

    def test_centroid_on_hyperboloid(self):
        """Centroid should be on the hyperboloid."""
        points = mx.stack([
            make_hyperboloid_point([0.3, 0.0]),
            make_hyperboloid_point([0.0, 0.3]),
            make_hyperboloid_point([-0.2, 0.1]),
        ])
        c = lorentz_centroid(points)
        assert bool(check_on_hyperboloid(c, tol=1e-2))

    def test_centroid_of_single_point(self):
        """Centroid of one point is that point."""
        p = make_hyperboloid_point([0.3, -0.2])
        c = lorentz_centroid(mx.expand_dims(p, axis=0))
        assert allclose(c, p, atol=1e-3)

    def test_centroid_of_symmetric_points(self):
        """Centroid of points symmetric about origin should be near origin."""
        points = mx.stack([
            make_hyperboloid_point([0.5, 0.0]),
            make_hyperboloid_point([-0.5, 0.0]),
        ])
        c = lorentz_centroid(points)
        # Space components should be near zero
        assert allclose(c[1:], mx.zeros(2), atol=1e-3)

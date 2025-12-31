# TODO: MLX Hyperbolic Embeddings (mlx_hyp)

**Last Updated**: 2025-12-30

## Status: PROJECT COMPLETE ✅

The core hyperbolic geometry operations are working and performant.
Benchmarks showed TMU optimization is not beneficial - native MLX is faster.

---

## ✅ COMPLETED

### Phase 1: Infrastructure
- [x] Directory structure (src/, python/, tests/)
- [x] CMakeLists.txt (for reference, not needed)
- [x] Python package with pure MLX implementation
- [x] README.md documentation

### Phase 4: Python Integration (Core Value)
- [x] `mobius_add()` - Möbius addition in Poincaré ball
- [x] `poincare_distance()` - Geodesic distance
- [x] `exp_map()` - Project tangent vector to manifold
- [x] `log_map()` - Project point to tangent space

### Phase 5: Benchmarking
- [x] LUT vs Native MLX comparison → Native wins (1.6-2.4x faster)
- [x] Hyperbolic operations throughput → 2-17M ops/sec
- [x] Decision: Use pure MLX, abandon TMU optimization

---

## ❌ ABANDONED (Not Beneficial)

### Phase 2: Metal Kernel Integration
- ~~Test Metal shader compilation~~
- ~~TMU-based LUT sampling~~

**Reason**: Native MLX transcendentals are already faster than LUT approach.

### Phase 3: C++ Glue Code
- ~~MLX buffer extraction~~
- ~~Zero-copy texture binding~~

**Reason**: Complexity not justified; no performance benefit.

---

## ⏳ OPTIONAL CLEANUP

These are nice-to-haves, not required for library to be useful:

### Simplify Package
- [ ] Remove LUT-based transcendentals (fast_exp, fast_log, fast_tanh)
- [ ] Remove C++/Metal code (src/ directory)
- [ ] Update imports in __init__.py
- [ ] Simplify README to focus on hyperbolic operations

### Quality
- [ ] Add unit tests for hyperbolic operations
- [ ] Fix fast_log precision issue (or just remove it)
- [ ] Add type hints validation (mypy)

### Distribution
- [ ] Publish to PyPI as `mlx-hyperbolic`
- [ ] Add GitHub Actions CI

---

## Usage (Current State)

```python
import mlx.core as mx
from mlx_hyperbolic import mobius_add, poincare_distance, exp_map, log_map

# Möbius addition in Poincaré ball
x = mx.array([0.1, 0.2, 0.3])
y = mx.array([0.2, 0.1, 0.2])
result = mobius_add(x, y)

# Geodesic distance
dist = poincare_distance(x, y)

# Exponential/logarithmic maps
tangent = mx.array([0.1, 0.1, 0.1])
origin = mx.zeros(3)
point = exp_map(tangent, origin)
recovered = log_map(point, origin)
```

---

## Performance Summary

| Operation | Dim=16, Batch=10K | Dim=768, Batch=10K |
|-----------|------------------|-------------------|
| mobius_add | 16.1M ops/sec | 3.6M ops/sec |
| poincare_distance | 17.0M ops/sec | 2.6M ops/sec |
| exp_map | 13.7M ops/sec | 2.1M ops/sec |
| log_map | 14.3M ops/sec | 2.0M ops/sec |

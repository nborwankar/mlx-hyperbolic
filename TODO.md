# TODO: MLX Hyperbolic Embeddings (mlx_hyp)

**Last Updated**: 2025-12-30

## Status: PROJECT COMPLETE ✅

Both Poincaré ball and Lorentz (hyperboloid) models implemented and benchmarked.

---

## ✅ COMPLETED

### Phase 1: Core Implementation
- [x] Poincaré ball model (ops.py)
  - `mobius_add()`, `poincare_distance()`, `exp_map()`, `log_map()`
- [x] Lorentz hyperboloid model (lorentz.py)
  - `lorentz_distance()`, `exp_map_lorentz()`, `log_map_lorentz()`
  - `parallel_transport_lorentz()`, `lorentz_centroid()`
  - Model conversions: `poincare_to_lorentz()`, `lorentz_to_poincare()`
- [x] Comprehensive documentation (README.md)

### Phase 2: Benchmarking
- [x] vs PyManopt: 122-183x faster (see PYMANOPT_vs_MLX.md)
- [x] vs Geoopt (PyTorch MPS): 2.2x faster (see GEOOPT_vs_MLX.md)
- [x] Roofline analysis: memory-bound workload (honest performance analysis)

### Phase 3: Package Cleanup
- [x] Removed deprecated TMU/LUT code (moved to _deprecated/)
- [x] Clean __init__.py with only hyperbolic operations
- [x] Updated pyproject.toml for v0.2.0
- [x] Added LICENSE (MIT) and .gitignore

---

## ⏳ NEXT: PyPI Distribution

### GitHub Setup
- [ ] Create repo: github.com/nborwankar/mlx-hyperbolic
- [ ] Push code
- [ ] Add GitHub Actions CI (optional)

### PyPI Publishing
- [ ] Create PyPI account (if needed)
- [ ] Build distribution: `python -m build`
- [ ] Upload to TestPyPI first: `twine upload --repository testpypi dist/*`
- [ ] Test install: `pip install -i https://test.pypi.org/simple/ mlx-hyperbolic`
- [ ] Upload to PyPI: `twine upload dist/*`

---

## Usage

```python
import mlx.core as mx
from mlx_hyperbolic import (
    # Poincaré ball
    mobius_add, poincare_distance, exp_map, log_map,
    # Lorentz hyperboloid
    lorentz_distance, exp_map_lorentz, log_map_lorentz,
    # Conversions
    poincare_to_lorentz, lorentz_to_poincare,
)

# Poincaré ball
x = mx.array([0.1, 0.2, 0.3])
y = mx.array([0.2, 0.1, 0.2])
d = poincare_distance(x, y)

# Lorentz (more stable)
x_L = poincare_to_lorentz(x)
y_L = poincare_to_lorentz(y)
d_L = lorentz_distance(x_L, y_L)  # Same value, better numerics
```

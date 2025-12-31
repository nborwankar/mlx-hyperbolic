# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

mlx-hyperbolic is a hyperbolic geometry library for MLX (Apple's machine learning framework). It provides GPU-accelerated operations for hyperbolic neural networks on Apple Silicon, supporting both **Poincaré ball** and **Lorentz (hyperboloid)** models.

**Status**: Production-ready (v0.2.1). Both models implemented and benchmarked.

## Build & Development Commands

```bash
# Install in development mode
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests/benchmarks
python tests/benchmark_speed.py
python tests/benchmark_speed.py --precision-only

# Linting and formatting
ruff check .
ruff format .
black .

# Type checking
mypy python/mlx_hyperbolic/

# Build distribution
python -m build

# Verify installation
python -c "from mlx_hyperbolic import lorentz_distance; print('✓ Installed')"
```

## Architecture

### Package Structure

```
python/mlx_hyperbolic/
├── __init__.py    # Public API exports (all hyperbolic ops)
├── ops.py         # Poincaré ball model (~150 lines)
└── lorentz.py     # Lorentz hyperboloid model (~370 lines)
```

### Two Hyperbolic Models

| Model | File | Dimensions | Best For |
|-------|------|------------|----------|
| **Poincaré Ball** | `ops.py` | n (unit ball) | Visualization, intuitive |
| **Lorentz Hyperboloid** | `lorentz.py` | n+1 (Minkowski space) | Training, numerical stability |

Both represent the same geometry. Conversion functions (`poincare_to_lorentz`, `lorentz_to_poincare`) allow seamless switching.

### Key Design Decisions

1. **Pure MLX**: Originally planned TMU/Metal kernel optimization, but benchmarks showed native MLX transcendentals are faster. Deprecated Metal code is in `_deprecated/`.

2. **Lorentz Preferred**: Poincaré has numerical instability near boundary (||x|| → 1). Lorentz has no such issues and is 10-33% faster.

3. **Curvature Parameter**: All operations accept `c` (curvature, default 1.0). Higher curvature = more hyperbolic, lower = closer to Euclidean.

### Core Operations

**Poincaré** (`ops.py`):
- `mobius_add(x, y, c)` - Möbius addition x ⊕ y
- `poincare_distance(x, y, c)` - Geodesic distance
- `exp_map(v, x, c)` - Tangent vector → manifold
- `log_map(y, x, c)` - Manifold → tangent space

**Lorentz** (`lorentz.py`):
- `lorentz_distance(x, y, c)` - Simple: arccosh(-⟨x,y⟩_L)
- `exp_map_lorentz(v, x, c)` - Tangent → hyperboloid
- `log_map_lorentz(y, x, c)` - Hyperboloid → tangent space
- `parallel_transport_lorentz(v, x, y, c)` - For Riemannian SGD
- `minkowski_inner(x, y)` - Lorentzian inner product

**Utilities**:
- `project_to_hyperboloid(x)` - ℝⁿ → ℍⁿ
- `lorentz_centroid(points, weights)` - Einstein midpoint
- `check_on_hyperboloid(x)` - Verify constraint

## Performance Characteristics

- **122-183x faster** than PyManopt (CPU)
- **2.2x faster** than geoopt + PyTorch MPS
- Memory-bound workload (~0.06% GPU compute utilization, expected)

See `PYMANOPT_vs_MLX.md` and `GEOOPT_vs_MLX.md` for detailed benchmarks.

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- macOS 13.0+ (Ventura)
- Python 3.11+
- MLX 0.0.10+

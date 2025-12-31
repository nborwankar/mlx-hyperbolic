# mlx-hyperbolic: Hyperbolic Geometry for MLX

Fast hyperbolic neural network primitives on Apple Silicon. Supports both **Poincaré ball** and **Lorentz (hyperboloid)** models with GPU acceleration via MLX.

## Why Hyperbolic?

Hyperbolic space naturally represents hierarchical data (trees, taxonomies, knowledge graphs) with exponentially more space as you move from the center. This makes it ideal for:

- **Knowledge graph embeddings** (WordNet, Freebase)
- **Hierarchical clustering**
- **Recommender systems** with implicit hierarchies
- **Natural language** (words have hierarchical relationships)

## Models Supported

| Model | Representation | Best For |
|-------|---------------|----------|
| **Poincaré Ball** | Unit ball in ℝⁿ | Visualization, intuitive |
| **Lorentz (Hyperboloid)** | Upper sheet of hyperboloid in ℝⁿ⁺¹ | Training, numerical stability |

Both represent the same geometric space — convert freely between them.

## Installation

```bash
# Install MLX (Apple Silicon required)
pip install mlx

# Clone and install
git clone https://github.com/nborwankar/mlx_hyp.git
cd mlx_hyp
pip install -e .
```

Verify installation:
```bash
python -c "from mlx_hyperbolic import lorentz_distance; print('✓ Installed')"
```

## Quick Start

### Poincaré Ball Model

```python
import mlx.core as mx
from mlx_hyperbolic import mobius_add, poincare_distance, exp_map, log_map

# Points in the Poincaré ball (must have ||x|| < 1)
x = mx.array([0.1, 0.2, 0.3])
y = mx.array([0.2, 0.1, 0.2])

# Möbius addition (hyperbolic "addition")
z = mobius_add(x, y)
print(f"x ⊕ y = {z}")

# Geodesic distance
d = poincare_distance(x, y)
print(f"Distance: {d}")

# Exponential map: tangent vector → manifold
tangent = mx.array([0.05, 0.05, 0.05])
origin = mx.zeros(3)
point = exp_map(tangent, origin)

# Logarithmic map: manifold → tangent space
recovered = log_map(point, origin)
```

### Lorentz (Hyperboloid) Model — Recommended for Training

```python
import mlx.core as mx
from mlx_hyperbolic import (
    lorentz_distance,
    exp_map_lorentz,
    log_map_lorentz,
    project_to_hyperboloid,
    poincare_to_lorentz,
    lorentz_to_poincare,
)

# Create points on hyperboloid (n+1 dimensions)
# Method 1: Project from space coordinates
x = project_to_hyperboloid(mx.array([0.3, 0.4, 0.5]))  # 3D → 4D
y = project_to_hyperboloid(mx.array([0.2, 0.3, 0.1]))

# Method 2: Convert from Poincaré
x_poincare = mx.array([0.1, 0.2, 0.3])
x = poincare_to_lorentz(x_poincare)

# Distance (simple formula: arccosh of Minkowski inner product)
d = lorentz_distance(x, y)
print(f"Distance: {d}")

# Exponential/logarithmic maps
tangent = log_map_lorentz(y, x)  # Direction from x to y
recovered = exp_map_lorentz(tangent, x)  # Should equal y

# Convert back to Poincaré for visualization
x_viz = lorentz_to_poincare(x)
```

### Model Conversion

```python
from mlx_hyperbolic import poincare_to_lorentz, lorentz_to_poincare

# Poincaré (3D) → Lorentz (4D)
p = mx.array([0.2, 0.3, 0.4])
L = poincare_to_lorentz(p)

# Lorentz (4D) → Poincaré (3D)
p_back = lorentz_to_poincare(L)
# p_back ≈ p (zero error round-trip)
```

## API Reference

### Poincaré Ball Operations

| Function | Description |
|----------|-------------|
| `mobius_add(x, y, c=1.0)` | Möbius addition x ⊕ y |
| `poincare_distance(x, y, c=1.0)` | Geodesic distance |
| `exp_map(v, x, c=1.0)` | Project tangent vector to manifold |
| `log_map(y, x, c=1.0)` | Project point to tangent space |

### Lorentz (Hyperboloid) Operations

| Function | Description |
|----------|-------------|
| `lorentz_distance(x, y, c=1.0)` | Geodesic distance: arccosh(-⟨x,y⟩_L) |
| `lorentz_distance_squared(x, y, c=1.0)` | Squared distance (avoids sqrt) |
| `exp_map_lorentz(v, x, c=1.0)` | Project tangent to hyperboloid |
| `log_map_lorentz(y, x, c=1.0)` | Project point to tangent space |
| `parallel_transport_lorentz(v, x, y, c=1.0)` | Transport vector along geodesic |
| `lorentz_centroid(points, weights, c=1.0)` | Einstein midpoint |
| `minkowski_inner(x, y)` | Minkowski inner product ⟨x,y⟩_L |
| `minkowski_norm(x)` | Minkowski norm √\|⟨x,x⟩_L\| |

### Utilities

| Function | Description |
|----------|-------------|
| `project_to_hyperboloid(x, c=1.0)` | Project ℝⁿ to hyperboloid ℍⁿ |
| `poincare_to_lorentz(x, c=1.0)` | Convert Poincaré → Lorentz |
| `lorentz_to_poincare(y, c=1.0)` | Convert Lorentz → Poincaré |
| `check_on_hyperboloid(x, c=1.0)` | Verify constraint satisfied |

## Why Lorentz Over Poincaré?

### Numerical Stability

| Norm ‖x‖ | Poincaré Error | Lorentz Error |
|----------|---------------|---------------|
| 0.99 | 0.0005% | 0% |
| 0.999 | 0.018% | 0% |
| **0.9999** | **4.8%** | **0%** |

Poincaré has a conformal factor λ = 2/(1-‖x‖²) that explodes near the boundary:

| ‖x‖ | Conformal Factor λ |
|-----|-------------------|
| 0.9 | 10.5 |
| 0.99 | 100.5 |
| 0.999 | 1,000.5 |
| 0.9999 | 10,000.5 ← Gradient explosion! |

**Lorentz has no conformal factor** — gradients are stable everywhere.

### Performance (M2 Pro, MLX GPU)

| Operation | Poincaré | Lorentz | Speedup |
|-----------|----------|---------|---------|
| Distance (batch=10K, dim=64) | 0.52ms | 0.39ms | **1.33x** |
| Distance (batch=10K, dim=256) | 0.88ms | 0.76ms | **1.15x** |

### Recommendation

| Use Case | Model |
|----------|-------|
| **Training embeddings** | Lorentz (stability) |
| **Visualization** | Poincaré (intuitive unit ball) |
| **Inference** | Either (convert as needed) |

## Hyperbolic Operations Throughput

All operations run on Apple Silicon GPU via MLX:

| Operation | Dim=16, Batch=10K | Dim=768, Batch=10K |
|-----------|------------------|-------------------|
| `mobius_add` | 16.1M ops/sec | 3.6M ops/sec |
| `poincare_distance` | 17.0M ops/sec | 2.6M ops/sec |
| `lorentz_distance` | 22.6M ops/sec | 3.0M ops/sec |
| `exp_map` | 13.7M ops/sec | 2.1M ops/sec |
| `log_map` | 14.3M ops/sec | 2.0M ops/sec |

### vs PyManopt

| Batch Size | PyManopt (CPU) | MLX (GPU) | Speedup |
|------------|---------------|-----------|---------|
| 1,000 | 140K/s | 3.1M/s | **22x** |
| 10,000 | 140K/s | 25.7M/s | **183x** |

> **Note**: Both implementations are memory-bound (~8 GFLOPS achieved vs 13,600 GFLOPS M2 Max peak). The speedup comes from eliminating Python loop overhead and using GPU memory bandwidth, not from saturating compute. See [PYMANOPT_vs_MLX.md](PYMANOPT_vs_MLX.md) for detailed roofline analysis.

### vs Geoopt (PyTorch MPS)

| Operation | Geoopt (MPS) | MLX (Metal) | Speedup |
|-----------|--------------|-------------|---------|
| Poincaré Distance | 1.31 ms | 0.52 ms | **2.5x** |
| ExpMap | 1.77 ms | 0.79 ms | **2.2x** |
| LogMap | 1.69 ms | 0.74 ms | **2.3x** |

Both run on the same Apple Silicon GPU. MLX's native Metal backend outperforms PyTorch's MPS translation layer.

**Important**: Geoopt's Lorentz model requires `float64`, which MPS doesn't support. MLX Hyperbolic has no such limitation.

See [GEOOPT_vs_MLX.md](GEOOPT_vs_MLX.md) for detailed comparison.

## Project Structure

```
mlx_hyp/
├── python/mlx_hyperbolic/
│   ├── __init__.py      # Package exports
│   ├── ops.py           # Poincaré ball operations
│   └── lorentz.py       # Lorentz hyperboloid operations
├── tests/
│   └── benchmark_speed.py
├── README.md
├── BENCHMARKS.md        # LUT vs native MLX benchmarks
├── PYMANOPT_vs_MLX.md   # Comparison with PyManopt + roofline analysis
├── GEOOPT_vs_MLX.md     # Comparison with geoopt (PyTorch MPS)
├── DONE.md              # Development log
└── TODO.md              # Status tracking
```

## Requirements

- **Hardware**: Apple Silicon Mac (M1/M2/M3/M4)
- **OS**: macOS 13.0+ (Ventura or later)
- **Python**: 3.10+
- **MLX**: 0.20+ (`pip install mlx`)

## Mathematical Background

### Poincaré Ball Model

The Poincaré ball is the unit ball 𝔹ⁿ = {x ∈ ℝⁿ : ‖x‖ < 1} with the Riemannian metric:

```
g_x = (2 / (1 - ‖x‖²))² · I
```

Distance formula:
```
d(x, y) = arccosh(1 + 2‖x-y‖² / ((1-‖x‖²)(1-‖y‖²)))
```

### Lorentz (Hyperboloid) Model

The Lorentz model uses the upper sheet of a hyperboloid in Minkowski space:

```
ℍⁿ = {x ∈ ℝⁿ⁺¹ : ⟨x,x⟩_L = -1, x₀ > 0}
```

where ⟨x,y⟩_L = -x₀y₀ + x₁y₁ + ... + xₙyₙ is the Minkowski inner product.

Distance formula (much simpler!):
```
d(x, y) = arccosh(-⟨x,y⟩_L)
```

## References

- [Poincaré Embeddings for Learning Hierarchical Representations](https://arxiv.org/abs/1705.08039) (Nickel & Kiela, 2017)
- [Learning Continuous Hierarchies in the Lorentz Model](https://arxiv.org/abs/1806.03417) (Nickel & Kiela, 2018)
- [Hyperbolic Neural Networks](https://arxiv.org/abs/1805.09112) (Ganea et al., 2018)
- [Hyperbolic Graph Convolutional Neural Networks](https://arxiv.org/abs/1910.12933) (Chami et al., 2019)
- [MLX: Machine Learning on Apple Silicon](https://ml-explore.github.io/mlx/)

## License

MIT License — see [LICENSE](LICENSE) for details.

## Author

Nitin Borwankar ([@nborwankar](https://github.com/nborwankar))

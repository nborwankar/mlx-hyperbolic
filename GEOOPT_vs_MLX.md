# Geoopt (PyTorch MPS) vs MLX Hyperbolic: Performance Comparison

**Date**: 2025-12-30
**Hardware**: M2 Max (36 GPU cores, 400 GB/s memory bandwidth)

## Executive Summary

| Implementation | Device | Avg Speedup |
|---------------|--------|-------------|
| Geoopt (PyTorch) | MPS (Apple GPU) | 1x (baseline) |
| **MLX Hyperbolic** | **Metal (Apple GPU)** | **2.2x faster** |

Both use the same Apple Silicon GPU, but MLX's Metal backend is significantly faster than PyTorch's MPS backend for hyperbolic operations.

---

## Detailed Benchmarks

### Poincaré Ball Operations (dim=64)

| Operation | Batch | Geoopt (MPS) | MLX (Metal) | Speedup |
|-----------|-------|--------------|-------------|---------|
| **Distance** | 1,000 | 0.221 ms | 0.116 ms | **1.90x** |
| **Distance** | 10,000 | 1.314 ms | 0.524 ms | **2.51x** |
| **Distance** | 100,000 | 12.87 ms | 4.42 ms | **2.91x** |
| **ExpMap** | 1,000 | 0.263 ms | 0.150 ms | **1.75x** |
| **ExpMap** | 10,000 | 1.771 ms | 0.787 ms | **2.25x** |
| **ExpMap** | 100,000 | 17.12 ms | 7.63 ms | **2.24x** |
| **LogMap** | 1,000 | 0.250 ms | 0.150 ms | **1.67x** |
| **LogMap** | 10,000 | 1.694 ms | 0.737 ms | **2.30x** |
| **LogMap** | 100,000 | 16.38 ms | 7.15 ms | **2.29x** |

**Average speedup: 2.20x** (MLX faster across all operations)

---

## Why MLX is Faster Than PyTorch MPS

Both frameworks use the same Apple Silicon GPU, yet MLX consistently outperforms PyTorch MPS:

### 1. Native Metal vs MPS Translation Layer
- **MLX**: Compiled directly to Metal shaders, optimized for Apple Silicon
- **PyTorch MPS**: Translation layer from PyTorch ops → Metal (overhead)

### 2. Lazy Evaluation
- **MLX**: Builds computation graph, fuses operations, executes optimally
- **PyTorch**: Eager execution by default (less optimization opportunity)

### 3. Unified Memory Optimization
- **MLX**: Designed from ground up for Apple's unified memory architecture
- **PyTorch**: Originally designed for discrete GPU memory model

### 4. Framework Maturity on Apple Silicon
- **MLX**: Apple-developed, specifically for Apple Silicon
- **PyTorch MPS**: Community-contributed backend, still maturing

---

## Lorentz Model: MPS Limitation

**Important**: Geoopt's Lorentz (hyperboloid) implementation requires `float64`, which MPS does not support:

```python
# Geoopt Lorentz fails on MPS:
RuntimeError: Cannot convert a MPS Tensor to float64 dtype
as the MPS framework doesn't support float64
```

**MLX Hyperbolic has no such limitation** — our Lorentz implementation works with `float32` and runs on GPU without issues.

| Model | Geoopt + MPS | MLX Hyperbolic |
|-------|--------------|----------------|
| Poincaré Ball | ✅ Works | ✅ Works |
| Lorentz (Hyperboloid) | ❌ Fails (float64) | ✅ Works |

---

## Practical Recommendations

| Use Case | Best Choice | Reasoning |
|----------|-------------|-----------|
| **Apple Silicon + Hyperbolic ML** | **MLX Hyperbolic** | 2.2x faster, no dtype limitations |
| **Cross-platform (CUDA + MPS)** | Geoopt | Better ecosystem, CUDA optimized |
| **Research prototyping** | Geoopt | More manifolds, established library |
| **Production on Apple Silicon** | **MLX Hyperbolic** | Performance + stability |
| **Lorentz model on Mac** | **MLX Hyperbolic** | Only option that works |

---

## Code: Running the Benchmark

```python
import torch
import time
import mlx.core as mx
from geoopt import PoincareBall

# Setup
dim, batch = 64, 10000
device = torch.device('mps')

# Geoopt setup
manifold = PoincareBall()
x_torch = torch.randn(batch, dim, device=device) * 0.3
y_torch = torch.randn(batch, dim, device=device) * 0.3
# Project to ball
x_torch = x_torch / (torch.norm(x_torch, dim=-1, keepdim=True) + 1) * 0.8
y_torch = y_torch / (torch.norm(y_torch, dim=-1, keepdim=True) + 1) * 0.8

# MLX setup
x_mlx = mx.array(x_torch.cpu().numpy())
y_mlx = mx.array(y_torch.cpu().numpy())

# Benchmark Geoopt
torch.mps.synchronize()
start = time.perf_counter()
for _ in range(100):
    d_geo = manifold.dist(x_torch, y_torch)
torch.mps.synchronize()
geoopt_time = (time.perf_counter() - start) / 100

# Benchmark MLX
from mlx_hyperbolic import poincare_distance
mx.eval(x_mlx)  # Ensure data is ready
start = time.perf_counter()
for _ in range(100):
    d_mlx = poincare_distance(x_mlx, y_mlx)
    mx.eval(d_mlx)
mlx_time = (time.perf_counter() - start) / 100

print(f"Geoopt (MPS): {geoopt_time*1000:.3f} ms")
print(f"MLX:          {mlx_time*1000:.3f} ms")
print(f"Speedup:      {geoopt_time/mlx_time:.2f}x")
```

---

## Comparison Summary: All Frameworks

| Framework | Backend | Speed (10K batch) | Notes |
|-----------|---------|-------------------|-------|
| PyManopt | NumPy/CPU | 140K ops/sec | Python loop overhead |
| Geoopt | PyTorch MPS | 7.6M ops/sec | MPS translation layer |
| **MLX Hyperbolic** | **Metal** | **17M ops/sec** | **Native Apple Silicon** |

**Relative to PyManopt**:
- Geoopt (MPS): ~54x faster
- MLX Hyperbolic: ~122x faster

---

## Conclusion

For hyperbolic geometry on Apple Silicon:

1. **MLX Hyperbolic is 2.2x faster** than geoopt with PyTorch MPS
2. **MLX supports Lorentz model** on GPU; geoopt doesn't (float64 limitation)
3. **Both are memory-bound** — the speedup comes from better Metal integration, not compute saturation
4. **For production on Apple Silicon**, MLX Hyperbolic is the clear choice

For cross-platform needs (CUDA servers), geoopt remains a good choice due to its mature CUDA backend.

---

## References

- [Geoopt: Riemannian Optimization in PyTorch](https://geoopt.readthedocs.io/)
- [MLX: Machine Learning on Apple Silicon](https://ml-explore.github.io/mlx/)
- [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html)

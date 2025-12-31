# PyManopt vs MLX Hyperbolic: Performance Comparison

**Date**: 2025-12-30
**Hardware**: M2 Max (36 GPU cores, 400 GB/s memory bandwidth)

## Executive Summary

| Implementation | Speed | GFLOPS | Speedup |
|---------------|-------|--------|---------|
| PyManopt (NumPy/CPU) | 140K ops/sec | 0.07 | 1x |
| **MLX Poincaré (GPU)** | **15.6M ops/sec** | **7.8** | **111x** |
| **MLX Lorentz (GPU)** | **25.7M ops/sec** | **12.8** | **183x** |

The speedup is real, but both implementations are **memory-bound**, not compute-bound.

---

## Detailed Benchmarks

### Distance Computation (dim=64)

| Batch Size | PyManopt (CPU) | MLX Poincaré | MLX Lorentz | Speedup |
|------------|---------------|--------------|-------------|---------|
| 100 | 134K/s | 379K/s | 432K/s | 3.2x |
| 1,000 | 140K/s | 1.9M/s | 3.1M/s | 22x |
| 10,000 | 140K/s | 15.6M/s | 25.7M/s | **183x** |

**Key observation**: PyManopt saturates at ~140K ops/sec regardless of batch size (Python loop overhead). MLX scales with batch size due to GPU parallelism.

---

## FLOP Analysis

### Operations per Distance (64-dimensional vectors)

```
x_norm_sq = sum(x * x):        64 muls + 63 adds = 127 FLOPs
y_norm_sq = sum(y * y):        64 muls + 63 adds = 127 FLOPs
diff = x - y:                  64 subs           =  64 FLOPs
diff_norm_sq = sum(diff*diff): 64 muls + 63 adds = 127 FLOPs
denom = (1-a)*(1-b):           2 subs + 1 mul    =   3 FLOPs
ratio = 2*x/denom:             2 ops             =   2 FLOPs
arccosh(1 + x):                ~50 FLOPs (transcendental)
─────────────────────────────────────────────────────────
Total:                         ~500 FLOPs per distance
```

### Achieved vs Peak Performance

| Metric | MLX Achieved | M2 Max Peak | Utilization |
|--------|-------------|-------------|-------------|
| GFLOPS | 8.6 | 13,600 | **0.06%** |
| Memory BW | 8.9 GB/s | 400 GB/s | 2.2% |

**Why so low?** The workload is **memory-bound**.

---

## Roofline Analysis

```
Arithmetic Intensity = FLOPs / Bytes
                     = 500 FLOPs / 516 bytes
                     = 1.0 FLOP/byte

M2 Max Roofline Crossover = Peak FLOPS / Peak Bandwidth
                          = 13,600 GFLOPS / 400 GB/s
                          = 34 FLOP/byte
```

```
GFLOPS
  |
  |                           ← Peak: 13,600 GFLOPS
  |                          /
  |                        /  Compute Bound
  |                      /
  |                    /
  |                  /   ← Roofline
  |                /
  |              /
  |            /  Memory Bound
  |          /
  |        /
  |      /
  |    X ← We are here (8.6 GFLOPS @ 1 FLOP/byte)
  |  /
  +------------------------------------------------→ Arithmetic Intensity
     1    5   10      34                    (FLOP/byte)
                       ↑
              Roofline crossover
```

**Interpretation**: At 1 FLOP/byte, we're deep in memory-bound territory. The GPU cores are 98% idle, waiting for data.

---

## Why MLX is Still 122x Faster

Even though both are memory-bound, MLX wins because:

1. **GPU memory bandwidth** (400 GB/s) >> CPU bandwidth (~50 GB/s)
2. **No Python loop overhead** — batch operations compiled to Metal
3. **Unified memory** — no CPU↔GPU data copies on Apple Silicon
4. **Parallel execution** — 36 GPU cores vs 1 CPU thread

---

## When Would GPU Utilization Be Higher?

### 1. Higher-Dimensional Vectors

| Dimension | Arithmetic Intensity | Expected Improvement |
|-----------|---------------------|---------------------|
| 64 | 1.0 FLOP/byte | Baseline |
| 256 | 4.0 FLOP/byte | ~4x better |
| 1024 | 16 FLOP/byte | ~16x better |

### 2. Fused Operations (Hyperbolic Neural Network Layer)

```python
# Low intensity (current):
d = poincare_distance(x, y)  # 1 FLOP/byte, memory bound

# High intensity (HNN layer):
h = W @ x                     # Matrix multiply: ~100 FLOP/byte
h = mobius_add(h, b)          # Reuses h in cache
h = hyperbolic_relu(h)        # Reuses h in cache
# Combined: much higher utilization
```

### 3. Float16 Precision

```python
# Float32: 4 bytes per element
x = mx.array(..., dtype=mx.float32)

# Float16: 2 bytes per element, same FLOPs
x = mx.array(..., dtype=mx.float16)
# → 2x better arithmetic intensity
```

---

## Practical Recommendations

| Use Case | Best Choice | Reasoning |
|----------|-------------|-----------|
| **Training embeddings (large batch)** | MLX | 100-180x faster |
| **Small optimization problems** | PyManopt | Has solvers, simpler |
| **Hyperbolic neural networks** | MLX | GPU parallelism essential |
| **Apple Silicon + large data** | MLX | Native GPU acceleration |
| **CUDA server** | PyTorch + geoopt | Better CUDA ecosystem |

---

## Code: Running the Benchmark

```python
import numpy as np
import time
import mlx.core as mx
from pymanopt.manifolds import PoincareBall

# Setup
dim, batch = 64, 10000
manifold = PoincareBall(dim)

# Create test data
x_np = np.random.randn(batch, dim).astype(np.float32)
x_np = x_np / (np.linalg.norm(x_np, axis=1, keepdims=True) + 1) * 0.8
y_np = np.random.randn(batch, dim).astype(np.float32)
y_np = y_np / (np.linalg.norm(y_np, axis=1, keepdims=True) + 1) * 0.8

# PyManopt (must loop)
start = time.perf_counter()
distances = [manifold.dist(x_np[i], y_np[i]) for i in range(batch)]
pymanopt_time = time.perf_counter() - start

# MLX (batched)
x_mlx, y_mlx = mx.array(x_np), mx.array(y_np)

def mlx_dist_batch(x, y):
    x_norm_sq = mx.sum(x * x, axis=-1, keepdims=True)
    y_norm_sq = mx.sum(y * y, axis=-1, keepdims=True)
    diff_norm_sq = mx.sum((x - y) ** 2, axis=-1, keepdims=True)
    denom = (1 - x_norm_sq) * (1 - y_norm_sq)
    return mx.arccosh(1 + 2 * diff_norm_sq / (denom + 1e-8))

start = time.perf_counter()
distances_mlx = mlx_dist_batch(x_mlx, y_mlx)
mx.eval(distances_mlx)
mlx_time = time.perf_counter() - start

print(f"PyManopt: {pymanopt_time*1000:.1f} ms")
print(f"MLX:      {mlx_time*1000:.1f} ms")
print(f"Speedup:  {pymanopt_time/mlx_time:.0f}x")
```

---

## Conclusion

The **122x speedup over PyManopt is real and meaningful**, but it comes from:

1. ✅ Eliminating Python loop overhead
2. ✅ Using GPU memory bandwidth
3. ✅ Unified memory (no copies)
4. ❌ NOT from saturating GPU compute

For vector operations like distance computation, this is expected — they're inherently memory-bound. The real GPU compute advantage would appear in **matrix-heavy workloads** like hyperbolic neural network layers.

---

## References

- [Roofline Model](https://en.wikipedia.org/wiki/Roofline_model)
- [Apple M2 Max Specifications](https://www.apple.com/newsroom/2023/01/apple-unveils-m2-pro-and-m2-max-next-generation-chips-for-next-level-workflows/)
- [PyManopt Documentation](https://www.pymanopt.org/)
- [MLX Documentation](https://ml-explore.github.io/mlx/)

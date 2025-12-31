# Benchmark Results

**Date**: 2025-12-30
**Hardware**: M2 MacBook Pro (96GB RAM, 12 CPU, 36 GPU cores)
**MLX Version**: 0.30.1.dev

## Executive Summary

1. **LUT-based transcendentals are SLOWER than native MLX** (1.6-2.4x)
2. **Hyperbolic operations are performant** (2-17M ops/sec depending on dimension)
3. **Recommendation**: Use native MLX ops; ship hyperbolic operations as the library's value

---

## 1. LUT vs Native MLX Transcendentals

Testing whether pre-computed LUT + interpolation beats native GPU transcendentals.

### Results (100 iterations, time in ms per call)

| Operation | Size | LUT (ms) | Native (ms) | Ratio | Winner |
|-----------|------|----------|-------------|-------|--------|
| exp | 10K | 0.395 | 0.194 | 2.03x | Native |
| exp | 100K | 0.342 | 0.197 | 1.74x | Native |
| exp | 1M | 0.520 | 0.216 | 2.41x | Native |
| tanh | 10K | 0.298 | 0.182 | 1.64x | Native |
| tanh | 100K | 0.323 | 0.196 | 1.65x | Native |
| tanh | 1M | 0.504 | 0.254 | 1.98x | Native |
| log | 10K | 0.288 | 0.180 | 1.59x | Native |
| log | 100K | 0.317 | 0.195 | 1.62x | Native |
| log | 1M | 0.513 | 0.211 | 2.43x | Native |

### Conclusion

**Native MLX wins across all operations and sizes.** The LUT approach:
- Requires normalize → index → fetch → interpolate (multiple ops)
- M-series GPUs have efficient transcendental units
- MLX already compiles to optimized Metal shaders

**Decision**: Abandon LUT optimization; use native MLX transcendentals.

---

## 2. Hyperbolic Operations Performance

These are the unique value of the library - operations not available in native MLX.

### Möbius Addition (mobius_add)

| Dim | Batch=1 | Batch=100 | Batch=1K | Batch=10K |
|-----|---------|-----------|----------|-----------|
| 16 | 2.6K/s | 237K/s | 2.2M/s | **16.1M/s** |
| 64 | 2.3K/s | 221K/s | 1.9M/s | 16.1M/s |
| 256 | 2.6K/s | 219K/s | 1.9M/s | 7.8M/s |
| 768 | 2.6K/s | 199K/s | 1.6M/s | 3.6M/s |

### Poincaré Distance

| Dim | Batch=1 | Batch=100 | Batch=1K | Batch=10K |
|-----|---------|-----------|----------|-----------|
| 16 | 1.9K/s | 142K/s | 1.6M/s | **17.0M/s** |
| 64 | 1.8K/s | 155K/s | 1.7M/s | 12.1M/s |
| 256 | 1.5K/s | 167K/s | 1.5M/s | 7.4M/s |
| 768 | 1.8K/s | 144K/s | 958K/s | 2.6M/s |

### Exponential Map (exp_map)

| Dim | Batch=1 | Batch=100 | Batch=1K | Batch=10K |
|-----|---------|-----------|----------|-----------|
| 16 | 1.6K/s | 171K/s | 1.1M/s | **13.7M/s** |
| 64 | 1.8K/s | 129K/s | 1.3M/s | 9.7M/s |
| 256 | 1.4K/s | 160K/s | 1.1M/s | 6.4M/s |
| 768 | 1.8K/s | 105K/s | 1.1M/s | 2.1M/s |

### Logarithmic Map (log_map)

| Dim | Batch=1 | Batch=100 | Batch=1K | Batch=10K |
|-----|---------|-----------|----------|-----------|
| 16 | 1.3K/s | 161K/s | 1.6M/s | **14.3M/s** |
| 64 | 1.7K/s | 135K/s | 1.5M/s | 8.3M/s |
| 256 | 1.7K/s | 156K/s | 1.3M/s | 6.4M/s |
| 768 | 1.3K/s | 148K/s | 1.1M/s | 2.0M/s |

### Conclusion

Hyperbolic operations achieve **2-17 million operations per second** depending on:
- **Dimension**: Higher dimensions = more compute per operation
- **Batch size**: Larger batches = better GPU utilization

These are practical throughput numbers for hyperbolic neural networks and embedding systems.

---

## 3. Recommendations

### Ship As-Is
The library provides genuine value with the hyperbolic geometry operations.

### Remove LUT Code
The fast_exp/fast_log/fast_tanh functions should be:
- Removed (use `mx.exp`, `mx.log`, `mx.tanh` directly)
- Or deprecated with warnings

### No TMU Integration Needed
The C++/Metal TMU integration is not worth pursuing because:
- Native MLX is already faster
- Adds significant complexity
- Marginal benefit at best

---

## Appendix: Test Environment

```
Platform: macOS Darwin 24.6.0
Device: Apple M2 Pro GPU (36 cores)
MLX: 0.30.1.dev (compiled with Metal backend)
Python: 3.11
```

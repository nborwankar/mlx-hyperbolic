# DONE: Hardware-Accelerated Hyperbolic Embeddings

## 2025-12-01

### Phase 1: Infrastructure & Build Pipeline

- [x] **Created directory structure**
  ```
  mlx_hyp/
  ├── src/
  │   └── kernels/
  ├── python/
  │   └── mlx_hyperbolic/
  └── tests/
  ```

- [x] **Created README.md**
  - Project overview and motivation
  - Installation instructions (prerequisites, build from source)
  - Quick start usage examples
  - API reference table
  - Technical explanation (TMU optimization, zero-copy, Float16)
  - Benchmark section (placeholder for results)
  - Development instructions
  - References to papers and documentation

- [x] **Created CMakeLists.txt**
  - MLX library discovery
  - Python/pybind11 bindings configuration
  - ARM64/Apple Silicon compiler flags
  - Metal shader compilation (.metal → .air → .metallib)
  - Installation targets for Python package

- [x] **Created Metal kernel** (`src/kernels/hyperbolic.metal`)
  - `lut_lookup_kernel`: TMU-based LUT sampling with hardware interpolation
  - `lut_lookup_batch_kernel`: Batch processing variant
  - `mobius_add_precomputed_kernel`: Möbius addition with pre-computed norms
  - Configured sampler with `filter::linear` for FREE hardware interpolation
  - Float16 precision throughout for maximum throughput

- [x] **Created C++ layer**
  - `src/lut_ops.h`: Header with LUTOps class interface
  - `src/lut_ops.cpp`: Metal dispatch logic, zero-copy texture creation
  - `src/main.cpp`: Python bindings via pybind11
  - Implements buffer→texture view (zero-copy via `buffer->newTexture()`)

- [x] **Created Python package** (`python/mlx_hyperbolic/`)
  - `__init__.py`: Package exports
  - `ops.py`: Full implementation with pure MLX fallback
    - LUT generation: `generate_exp_lut`, `generate_log_lut`, `generate_tanh_lut`
    - Fast operations: `fast_exp`, `fast_log`, `fast_tanh`
    - Hyperbolic ops: `mobius_add`, `poincare_distance`, `exp_map`, `log_map`
    - LUTCache with lazy initialization

- [x] **Created benchmark suite** (`tests/benchmark_speed.py`)
  - Precision tests comparing fast_* vs standard MLX
  - Latency benchmarks (1M operations)
  - Möbius addition throughput tests
  - Poincaré distance verification
  - LUT generation timing

- [x] **Created pyproject.toml**
  - Build system configuration (setuptools + cmake)
  - Dependencies (mlx, numpy)
  - Development tools (pytest, black, ruff, mypy)
  - Package metadata and classifiers

### Phase 1 Status: INFRASTRUCTURE COMPLETE

All source files created. Pure MLX fallback implementation working.

---

## 2025-12-30

### Phase 5: Benchmarking & Optimization Decision

- [x] **Fixed MLX API compatibility issues**
  - Changed `mx.astype(x, dtype)` to `x.astype(dtype)` throughout ops.py
  - All operations now work with MLX 0.30.1+

- [x] **Ran comprehensive benchmarks** (see BENCHMARKS.md)
  - Tested LUT-based ops vs native MLX transcendentals
  - Tested hyperbolic operations throughput at various dimensions/batch sizes

- [x] **Key Finding: LUT optimization NOT beneficial**
  | Operation | LUT vs Native | Ratio |
  |-----------|---------------|-------|
  | exp | Native wins | 1.7-2.4x faster |
  | tanh | Native wins | 1.6-2.0x faster |
  | log | Native wins | 1.6-2.4x faster |

- [x] **Key Finding: Hyperbolic operations are performant**
  - Möbius addition: up to 16M ops/sec
  - Poincaré distance: up to 17M ops/sec
  - exp_map/log_map: up to 14M ops/sec

- [x] **Decision: Abandon TMU/Metal integration**
  - Native MLX is already faster than LUT approach
  - M-series GPUs have efficient transcendental units
  - TMU optimization adds complexity for no gain

### Phase 2-3 Status: ABANDONED (Not Beneficial)

Benchmarks showed that:
1. Native MLX transcendentals outperform LUT-based approach
2. TMU hardware interpolation advantage doesn't apply to modern M-series chips
3. C++/Metal integration complexity not justified

### Project Status: COMPLETE (Pivot to Pure MLX)

The library's value is in the **hyperbolic geometry operations**:
- `mobius_add()` - Möbius addition in Poincaré ball
- `poincare_distance()` - Geodesic distance
- `exp_map()` / `log_map()` - Tangent space operations

These are production-ready and performant using pure MLX.

### Remaining Cleanup (Optional)
- [ ] Remove/deprecate LUT-based transcendentals (fast_exp, fast_log, fast_tanh)
- [ ] Simplify package to focus on hyperbolic operations
- [x] Update README to reflect pure MLX approach ✓
- [ ] Publish to PyPI

---

### Lorentz (Hyperboloid) Model Implementation

- [x] **Created lorentz.py module** (~370 lines)
  - Core operations: `minkowski_inner`, `minkowski_norm`, `lorentz_distance`, `lorentz_distance_squared`
  - Mappings: `exp_map_lorentz`, `log_map_lorentz`, `parallel_transport_lorentz`
  - Utilities: `project_to_hyperboloid`, `lorentz_centroid`, `check_on_hyperboloid`
  - Model conversions: `poincare_to_lorentz`, `lorentz_to_poincare`

- [x] **Validated Lorentz model benefits**
  - More numerically stable than Poincaré near boundary
  - 10-33% faster in benchmark comparisons
  - No conformal factor explosion (λ → ∞ as ||x|| → 1)

- [x] **Updated package exports**
  - Version bumped to 0.2.0
  - Both Poincaré and Lorentz operations exported from `__init__.py`

### Comprehensive Benchmarking

- [x] **PyManopt comparison** (see PYMANOPT_vs_MLX.md)
  - MLX 122-183x faster than PyManopt (CPU)
  - Honest FLOP analysis: ~500 FLOPs per distance computation
  - Roofline model: memory-bound (1 FLOP/byte vs 34 FLOP/byte crossover)
  - 0.06% GPU compute utilization (expected for memory-bound workloads)

- [x] **Geoopt comparison** (see GEOOPT_vs_MLX.md)
  - MLX 2.2x faster than geoopt + PyTorch MPS on same GPU
  - Geoopt Lorentz doesn't work on MPS (requires float64)
  - MLX has no dtype limitations

### Documentation

- [x] **README.md** - Complete rewrite with both models
- [x] **BENCHMARKS.md** - LUT vs native MLX results
- [x] **PYMANOPT_vs_MLX.md** - Comparison with roofline analysis
- [x] **GEOOPT_vs_MLX.md** - PyTorch MPS comparison

### Project Status: COMPLETE ✓

The mlx-hyperbolic library provides:
1. **Poincaré ball model** - Classic hyperbolic operations
2. **Lorentz (hyperboloid) model** - Numerically stable alternative
3. **Model conversions** - Seamless switching between representations
4. **122x speedup** over PyManopt, **2.2x** over geoopt+MPS

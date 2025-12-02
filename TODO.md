# TODO: Hardware-Accelerated Hyperbolic Embeddings (mlx_hyp)

## Phase 1: Infrastructure & Build Pipeline

- [ ] Create directory structure:
  - [ ] `src/` with `main.cpp`, `lut_ops.cpp`
  - [ ] `src/kernels/` with `hyperbolic.metal`
  - [ ] `python/mlx_hyperbolic/` with `__init__.py`, `ops.py`
  - [ ] `tests/` with `benchmark_speed.py`
- [ ] Create `CMakeLists.txt` with:
  - [ ] MLX library discovery
  - [ ] Python bindings configuration
  - [ ] ARM64/Apple Silicon compiler flags
  - [ ] Metal shader compilation
- [ ] Verify build pipeline with minimal "hello world" extension
- [ ] Document build instructions in README

## Phase 2: Metal Kernel (TMU Utilization)

- [ ] Implement `hyperbolic.metal` kernel with:
  - [ ] `constexpr sampler` with `filter::linear` for hardware interpolation
  - [ ] `texture1d<half>` for pre-computed LUT
  - [ ] Input normalization to [0.0, 1.0] coordinate space
  - [ ] Half (Float16) precision throughout
- [ ] Test kernel compilation
- [ ] Add boundary handling with `clamp_to_edge` address mode

## Phase 3: C++ Glue Code (Zero-Copy Texture Binding)

- [ ] Implement `lut_ops.cpp` with:
  - [ ] MLX array to MTL::Buffer extraction
  - [ ] Zero-copy MTL::Texture creation via `buffer->newTexture()`
  - [ ] Texture descriptor setup (`MTLPixelFormatR16Float`)
  - [ ] Metal encoder setup (buffers + texture binding)
  - [ ] Grid dispatch logic
- [ ] Implement Python bindings in `main.cpp`
- [ ] Handle `MTLResourceStorageModeShared` for unified memory

## Phase 4: Python Integration

- [ ] Implement LUT generation in `ops.py`:
  - [ ] `generate_exp_lut(size=4096, range=(0, 10))` - returns float16 MLX array
  - [ ] `generate_log_lut(size, range)` - for log operations
  - [ ] `generate_tanh_lut(size, range)` - for tanh operations
- [ ] Implement primitive wrappers:
  - [ ] `fast_exp(x)` with lazy LUT initialization
  - [ ] `fast_log(x)` with lazy LUT initialization
  - [ ] `fast_tanh(x)` with lazy LUT initialization
- [ ] Implement Mobius addition using fast primitives:
  - [ ] Standard formula: `(x + y) / (1 + c * <x,y>)` (Poincare ball)
  - [ ] Verify numerical stability

## Phase 5: Verification & Benchmarking

- [ ] Implement precision tests:
  - [ ] Compare `fast_exp(x)` vs `np.exp(x)` - target error < 1e-3
  - [ ] Compare `fast_log(x)` vs `np.log(x)`
  - [ ] Test edge cases (boundaries, large values)
- [ ] Implement latency benchmarks:
  - [ ] 1M operations: standard `mx.exp` vs `fast_exp`
  - [ ] Measure memory bandwidth utilization
- [ ] Implement throughput benchmarks:
  - [ ] Full Mobius addition layer (16D vectors)
  - [ ] MLX Standard (ALU) vs MLX Custom (Texture)
  - [ ] Report ops/second and speedup factor
- [ ] Document results in BENCHMARKS.md

## Future Extensions (Post-MVP)

- [ ] Extend to higher-dimensional Mobius operations
- [ ] Implement full HNN layer (forward + backward pass)
- [ ] Add gradient computation through LUT operations
- [ ] Explore Float32 fallback for precision-critical applications
- [ ] Package for PyPI distribution

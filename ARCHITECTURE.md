# Architecture: Hardware-Accelerated Hyperbolic Embeddings

## Overview

This project implements hyperbolic neural network primitives on Apple Silicon by exploiting the GPU's Texture Mapping Units (TMUs) for fast transcendental function computation. The key insight is that texture sampling with linear interpolation is essentially "free" on modern GPUs, allowing us to replace expensive ALU operations (exp, log, tanh) with texture lookups from pre-computed tables.

```
┌─────────────────────────────────────────────────────────────────┐
│                     Python API Layer                             │
│  fast_exp(x)  |  fast_log(x)  |  fast_tanh(x)  |  mobius_add()  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                   C++ Extension Layer                            │
│  • MLX Array → MTL::Buffer extraction                           │
│  • Zero-copy MTL::Texture creation                              │
│  • Metal encoder setup & dispatch                               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                   Metal Kernel Layer                             │
│  • TMU-based LUT sampling with hardware interpolation           │
│  • Float16 precision for maximum throughput                     │
│  • Normalized coordinate mapping                                │
└─────────────────────────────────────────────────────────────────┘
```

## Core Innovation: TMU-Based Math

### Traditional Approach (ALU)
```
Input → Compute exp/log/tanh in ALU → Output
        (10-20+ cycles per operation)
```

### Our Approach (TMU + LUT)
```
Input → Normalize to [0,1] → Sample Texture (1 cycle) → Output
                                    ↑
                         Pre-computed LUT in GPU texture memory
                         (Hardware linear interpolation)
```

The GPU's texture units include dedicated hardware for:
1. **Address calculation** - mapping normalized coordinates to texels
2. **Bilinear interpolation** - blending between adjacent values
3. **Filtering** - all done in fixed-function hardware, not ALU

## Memory Architecture

### M2 Max Unified Memory Model

```
┌─────────────────────────────────────────────────────────────────┐
│                    Unified Memory (96GB)                         │
│  ┌─────────────────┐    ┌─────────────────┐                     │
│  │   MTL::Buffer   │◄──►│  MTL::Texture   │  Zero-Copy View     │
│  │   (MLX Array)   │    │   (LUT Data)    │                     │
│  └─────────────────┘    └─────────────────┘                     │
│           ▲                     ▲                               │
│           │                     │                               │
│     CPU Access            GPU TMU Access                        │
│  (Python/NumPy)         (Hardware Sampling)                     │
└─────────────────────────────────────────────────────────────────┘
```

Key properties:
- **MTLResourceStorageModeShared**: Both CPU and GPU can access
- **Zero-copy texture creation**: `buffer->newTexture()` creates a texture view without copying data
- **Cache-coherent**: M2's unified memory ensures consistency

## Component Details

### 1. Metal Kernel (`hyperbolic.metal`)

```
┌────────────────────────────────────────────────────┐
│  Kernel: lut_lookup_kernel                         │
├────────────────────────────────────────────────────┤
│  Inputs:                                           │
│    • buffer[0]: input_values (half*)               │
│    • buffer[1]: output_values (half*)              │
│    • buffer[2]: input_scale (float)                │
│    • texture[0]: lut_texture (texture1d<half>)     │
├────────────────────────────────────────────────────┤
│  Processing:                                       │
│    1. Read input x                                 │
│    2. coord = x * input_scale  (normalize to 0-1) │
│    3. result = lut_texture.sample(sampler, coord) │
│    4. Write result                                 │
├────────────────────────────────────────────────────┤
│  Sampler Config:                                   │
│    • coord: normalized                             │
│    • address: clamp_to_edge                        │
│    • filter: linear (enables HW interpolation)    │
└────────────────────────────────────────────────────┘
```

### 2. C++ Glue Layer (`lut_ops.cpp`)

```cpp
// Pseudo-code structure
class HyperbolicOps {
    // Extract MTL::Buffer from MLX array
    MTL::Buffer* getBuffer(mlx::array& arr);

    // Create texture view (zero-copy)
    MTL::Texture* createTextureView(
        MTL::Buffer* buffer,
        MTLPixelFormat format,  // R16Float for half
        size_t width            // LUT size
    );

    // Dispatch kernel
    void dispatch(
        mlx::array& input,
        mlx::array& output,
        mlx::array& lut,
        float scale
    );
};
```

### 3. Python API (`ops.py`)

```python
# API Design
class LUTCache:
    """Lazy-initialized lookup table cache"""
    _exp_lut: mx.array | None = None
    _log_lut: mx.array | None = None
    _tanh_lut: mx.array | None = None

def fast_exp(x: mx.array) -> mx.array:
    """TMU-accelerated exponential"""
    if LUTCache._exp_lut is None:
        LUTCache._exp_lut = generate_exp_lut(4096)
    return _ext.lut_lookup(x, LUTCache._exp_lut, scale=...)

def mobius_add(x: mx.array, y: mx.array, c: float = 1.0) -> mx.array:
    """Möbius addition in Poincaré ball using fast primitives"""
    # Uses fast_exp, fast_log internally
    ...
```

## Data Flow: Möbius Addition Example

```
           x (16D vector)           y (16D vector)
                 │                        │
                 ▼                        ▼
         ┌───────────────────────────────────┐
         │         Möbius Addition           │
         │  (x ⊕ y) = (x + y) / (1 + c<x,y>) │
         │                                   │
         │  Internal calls:                  │
         │    • fast_exp() for norm calcs    │
         │    • fast_tanh() for projections  │
         └───────────────┬───────────────────┘
                         │
                         ▼
              Result (16D vector in Poincaré ball)
```

## Precision Considerations

| Operation | LUT Size | Range | Expected Error |
|-----------|----------|-------|----------------|
| exp(x)    | 4096     | [0, 10] | < 1e-3 |
| log(x)    | 4096     | [1e-6, 10] | < 1e-3 |
| tanh(x)   | 4096     | [-5, 5] | < 1e-4 |

Float16 is mandatory for TMU optimization:
- Texture units optimized for half-precision
- Float32 halves cache efficiency
- Float32 reduces throughput advantage

## Build System

```
CMakeLists.txt
├── Find MLX package
├── Find Python (for bindings)
├── Compile Metal shaders (.metal → .metallib)
├── Build C++ extension
│   ├── lut_ops.cpp
│   └── main.cpp (pybind11 bindings)
└── Install to python/mlx_hyperbolic/
```

## Directory Structure

```
mlx_hyperbolic/
├── CMakeLists.txt           # Build configuration
├── ARCHITECTURE.md          # This file
├── TODO.md                  # Task tracking
├── PLAN.txt                 # Original requirements
│
├── src/
│   ├── main.cpp             # Python bindings (pybind11)
│   ├── lut_ops.cpp          # Metal dispatch logic
│   └── kernels/
│       └── hyperbolic.metal # GPU kernel
│
├── python/
│   └── mlx_hyperbolic/
│       ├── __init__.py      # Package init
│       └── ops.py           # Python API
│
└── tests/
    └── benchmark_speed.py   # Verification & benchmarks
```

## Performance Targets

| Metric | Standard MLX | Target (TMU) | Speedup |
|--------|--------------|--------------|---------|
| exp latency (1M ops) | ~X ms | ~X/5 ms | 5x |
| Möbius add (16D, batch 10K) | ~Y ms | ~Y/3 ms | 3x |

Note: Actual numbers to be determined through benchmarking.

## Dependencies

- **MLX** (latest): Apple's ML framework for Apple Silicon
- **Python 3.11+**: For bindings and API
- **Clang/LLVM**: For C++ compilation with Metal support
- **metal-cpp**: Metal C++ headers (bundled with MLX)
- **pybind11**: Python bindings for C++ extension

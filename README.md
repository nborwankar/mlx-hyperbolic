# mlx_hyp: Hardware-Accelerated Hyperbolic Embeddings

Fast hyperbolic neural network primitives on Apple Silicon using GPU Texture Mapping Units (TMUs) for transcendental function computation.

## Overview

This project implements high-performance hyperbolic operations (exp, log, tanh, Möbius addition) by exploiting a key GPU optimization: **texture sampling with hardware linear interpolation is essentially "free"** on modern GPUs. Instead of computing expensive transcendental functions in the ALU (10-20+ cycles), we sample pre-computed lookup tables through the TMU (1 cycle with interpolation).

```
Traditional: Input → ALU exp/log/tanh (slow) → Output
This project: Input → Normalize → TMU Sample (fast) → Output
                                      ↑
                          Pre-computed LUT in GPU texture
```

## Requirements

- **Hardware**: Apple Silicon Mac (M1/M2/M3 series)
- **OS**: macOS 13.0+ (Ventura or later)
- **Python**: 3.11+
- **MLX**: Latest version (`pip install mlx`)
- **Clang/LLVM**: Included with Xcode Command Line Tools

## Installation

### Prerequisites

```bash
# Install Xcode Command Line Tools (includes Metal compiler)
xcode-select --install

# Install MLX
pip install mlx

# Verify MLX installation
python -c "import mlx.core as mx; print(mx.default_device())"
```

### Build from Source

```bash
# Clone the repository
git clone https://github.com/nborwankar/mlx_hyp.git
cd mlx_hyp

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build
make -j8

# Install the Python extension
cd ..
pip install -e .
```

### Verify Installation

```bash
python -c "import mlx_hyperbolic; print('mlx_hyperbolic installed successfully')"
```

## Quick Start

```python
import mlx.core as mx
from mlx_hyperbolic import fast_exp, fast_log, fast_tanh, mobius_add

# Fast transcendental operations via TMU
x = mx.array([0.5, 1.0, 2.0], dtype=mx.float16)
result = fast_exp(x)  # TMU-accelerated exponential

# Möbius addition in Poincaré ball
u = mx.random.uniform(shape=(16,), dtype=mx.float16) * 0.5  # Keep in ball
v = mx.random.uniform(shape=(16,), dtype=mx.float16) * 0.5
w = mobius_add(u, v, c=1.0)  # Hyperbolic addition
```

## Project Structure

```
mlx_hyp/
├── CMakeLists.txt           # Build configuration
├── README.md                # This file
├── ARCHITECTURE.md          # Technical design documentation
├── TODO.md                  # Development task tracking
├── DONE.md                  # Completed work log
├── PLAN.txt                 # Original project requirements
│
├── src/
│   ├── main.cpp             # Python bindings (pybind11)
│   ├── lut_ops.cpp          # Metal dispatch logic
│   └── kernels/
│       └── hyperbolic.metal # GPU kernel (TMU sampling)
│
├── python/
│   └── mlx_hyperbolic/
│       ├── __init__.py      # Package initialization
│       └── ops.py           # Python API (fast_exp, mobius_add, etc.)
│
└── tests/
    └── benchmark_speed.py   # Precision verification & benchmarks
```

## API Reference

### Transcendental Functions

| Function | Description | Input Range | Precision |
|----------|-------------|-------------|-----------|
| `fast_exp(x)` | TMU-accelerated exp | [0, 10] | < 1e-3 |
| `fast_log(x)` | TMU-accelerated log | [1e-6, 10] | < 1e-3 |
| `fast_tanh(x)` | TMU-accelerated tanh | [-5, 5] | < 1e-4 |

### Hyperbolic Operations

| Function | Description |
|----------|-------------|
| `mobius_add(x, y, c=1.0)` | Möbius addition in Poincaré ball |

## How It Works

### The TMU Optimization

Modern GPUs have dedicated Texture Mapping Units (TMUs) with hardware support for:
1. **Address calculation** - mapping normalized coordinates to texels
2. **Bilinear interpolation** - blending between adjacent values (FREE!)
3. **Filtering** - all in fixed-function hardware, not ALU

We exploit this by:
1. Pre-computing exp/log/tanh values into a lookup table (LUT)
2. Storing the LUT as a 1D texture in GPU memory
3. Sampling with `filter::linear` to get hardware-interpolated results

### Zero-Copy Memory

On Apple Silicon's unified memory architecture:
- MLX arrays use `MTLResourceStorageModeShared` buffers
- We create texture **views** of these buffers (`buffer->newTexture()`)
- No memory copy - the texture references the same physical memory
- Both CPU (Python/NumPy) and GPU (TMU) access the same data

### Float16 Requirement

Half-precision (Float16) is mandatory because:
- M-series TMUs are optimized for half-precision textures
- Float32 halves cache efficiency and throughput
- For hyperbolic embeddings, 1e-3 precision is sufficient

## Benchmarks

Run the benchmark suite:

```bash
python tests/benchmark_speed.py
```

### Expected Results (M2 Max)

| Operation | Standard MLX | TMU-Accelerated | Speedup |
|-----------|--------------|-----------------|---------|
| exp (1M ops) | TBD | TBD | ~5x |
| Möbius add (16D, 10K batch) | TBD | TBD | ~3x |

*Note: Actual numbers to be determined through benchmarking.*

## Development

### Building for Development

```bash
# Debug build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make

# Run tests
python -m pytest tests/
```

### Code Style

- C++: Follow MLX conventions
- Python: Black formatter, type hints
- Metal: Apple MSL style guide

## Technical Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Detailed technical design, memory architecture, kernel implementation
- **[TODO.md](TODO.md)** - Development task tracking
- **[DONE.md](DONE.md)** - Completed work log

## References

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [Hyperbolic Neural Networks (Ganea et al., 2018)](https://arxiv.org/abs/1805.09112)
- [Poincaré Embeddings (Nickel & Kiela, 2017)](https://arxiv.org/abs/1705.08039)

## License

MIT License - see LICENSE file for details.

## Author

Nitin Borwankar ([@nborwankar](https://github.com/nborwankar))

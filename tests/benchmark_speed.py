#!/usr/bin/env python3
"""
Benchmark suite for mlx_hyperbolic.

Compares TMU-accelerated operations against standard MLX implementations
and verifies numerical precision.

Usage:
    python benchmark_speed.py [--size SIZE] [--iterations ITER]
"""

import argparse
import time
from typing import Callable, Tuple
import mlx.core as mx
import numpy as np

# Add parent directory to path for development testing
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from mlx_hyperbolic import (
    fast_exp, fast_log, fast_tanh,
    mobius_add, poincare_distance,
    generate_exp_lut, generate_log_lut, generate_tanh_lut,
    clear_lut_cache,
)


def benchmark_fn(
    fn: Callable,
    *args,
    warmup: int = 5,
    iterations: int = 100,
    **kwargs
) -> Tuple[float, float]:
    """
    Benchmark a function with warmup and multiple iterations.

    Returns:
        Tuple of (mean_time_ms, std_time_ms)
    """
    # Warmup
    for _ in range(warmup):
        result = fn(*args, **kwargs)
        mx.eval(result)

    # Timed runs
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        mx.eval(result)  # Force evaluation
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return np.mean(times), np.std(times)


def precision_test(
    fast_fn: Callable,
    reference_fn: Callable,
    x: mx.array,
    name: str
) -> Tuple[float, float, float]:
    """
    Compare fast function against reference implementation.

    Returns:
        Tuple of (max_abs_error, mean_abs_error, mean_rel_error)
    """
    fast_result = fast_fn(x)
    ref_result = reference_fn(x)

    # Convert to float32 for comparison
    fast_np = np.array(mx.astype(fast_result, mx.float32))
    ref_np = np.array(mx.astype(ref_result, mx.float32))

    abs_error = np.abs(fast_np - ref_np)
    rel_error = abs_error / (np.abs(ref_np) + 1e-10)

    max_abs = np.max(abs_error)
    mean_abs = np.mean(abs_error)
    mean_rel = np.mean(rel_error)

    return max_abs, mean_abs, mean_rel


def run_precision_tests():
    """Run precision tests for all fast operations."""
    print("\n" + "=" * 60)
    print("PRECISION TESTS")
    print("=" * 60)

    # Test data
    exp_x = mx.linspace(0.0, 9.9, 10000)
    log_x = mx.linspace(1e-5, 9.9, 10000)
    tanh_x = mx.linspace(-4.9, 4.9, 10000)

    tests = [
        ("fast_exp", fast_exp, mx.exp, exp_x, 1e-3),
        ("fast_log", fast_log, mx.log, log_x, 1e-3),
        ("fast_tanh", fast_tanh, mx.tanh, tanh_x, 1e-4),
    ]

    all_passed = True
    for name, fast_fn, ref_fn, x, threshold in tests:
        max_err, mean_err, rel_err = precision_test(fast_fn, ref_fn, x, name)
        passed = max_err < threshold
        status = "PASS" if passed else "FAIL"
        all_passed = all_passed and passed

        print(f"\n{name}:")
        print(f"  Max absolute error:  {max_err:.6e} (threshold: {threshold:.0e}) [{status}]")
        print(f"  Mean absolute error: {mean_err:.6e}")
        print(f"  Mean relative error: {rel_err:.6e}")

    return all_passed


def run_latency_benchmarks(size: int = 1_000_000, iterations: int = 100):
    """Run latency benchmarks comparing fast vs standard operations."""
    print("\n" + "=" * 60)
    print(f"LATENCY BENCHMARKS (n={size:,}, {iterations} iterations)")
    print("=" * 60)

    # Test data
    exp_x = mx.random.uniform(0.0, 9.0, shape=(size,))
    log_x = mx.random.uniform(1e-5, 9.0, shape=(size,))
    tanh_x = mx.random.uniform(-4.0, 4.0, shape=(size,))

    benchmarks = [
        ("exp", exp_x, fast_exp, mx.exp),
        ("log", log_x, fast_log, mx.log),
        ("tanh", tanh_x, fast_tanh, mx.tanh),
    ]

    print(f"\n{'Operation':<15} {'Fast (ms)':<15} {'Standard (ms)':<15} {'Speedup':<10}")
    print("-" * 55)

    for name, x, fast_fn, std_fn in benchmarks:
        # Clear cache to ensure fair comparison
        clear_lut_cache()

        # Benchmark fast version (includes LUT generation on first call)
        fast_mean, fast_std = benchmark_fn(fast_fn, x, iterations=iterations)

        # Benchmark standard MLX
        std_mean, std_std = benchmark_fn(std_fn, x, iterations=iterations)

        speedup = std_mean / fast_mean if fast_mean > 0 else 0
        print(f"{name:<15} {fast_mean:>6.2f} ± {fast_std:>4.2f}   {std_mean:>6.2f} ± {std_std:>4.2f}   {speedup:>6.2f}x")


def run_mobius_benchmarks(dim: int = 16, batch_size: int = 10_000, iterations: int = 100):
    """Run benchmarks for Möbius addition."""
    print("\n" + "=" * 60)
    print(f"MÖBIUS ADDITION BENCHMARK (dim={dim}, batch={batch_size:,})")
    print("=" * 60)

    # Generate random vectors in the Poincaré ball (norm < 1)
    x = mx.random.uniform(-0.4, 0.4, shape=(batch_size, dim))
    y = mx.random.uniform(-0.4, 0.4, shape=(batch_size, dim))

    # Single vector pair
    x_single = x[0]
    y_single = y[0]

    print(f"\nSingle vector pair (dim={dim}):")
    mean_time, std_time = benchmark_fn(mobius_add, x_single, y_single, iterations=iterations)
    print(f"  Time: {mean_time:.4f} ± {std_time:.4f} ms")

    # Batch operation (using vmap or loop)
    print(f"\nBatch ({batch_size:,} pairs, dim={dim}):")

    def batch_mobius(x_batch, y_batch):
        # Process each pair (no native batch support yet)
        results = []
        for i in range(x_batch.shape[0]):
            results.append(mobius_add(x_batch[i], y_batch[i]))
        return mx.stack(results)

    mean_time, std_time = benchmark_fn(batch_mobius, x, y, iterations=min(10, iterations))
    throughput = batch_size / (mean_time / 1000)  # ops per second
    print(f"  Time: {mean_time:.2f} ± {std_time:.2f} ms")
    print(f"  Throughput: {throughput:,.0f} ops/sec")


def run_poincare_distance_test():
    """Test Poincaré distance computation."""
    print("\n" + "=" * 60)
    print("POINCARÉ DISTANCE TEST")
    print("=" * 60)

    # Test points
    origin = mx.zeros(16)
    x = mx.array([0.3] + [0.0] * 15)
    y = mx.array([0.5] + [0.0] * 15)

    # Distance from origin
    d_origin_x = poincare_distance(origin, x)
    d_origin_y = poincare_distance(origin, y)
    d_x_y = poincare_distance(x, y)

    print(f"\nTest vectors (16D):")
    print(f"  x = [0.3, 0, 0, ...]")
    print(f"  y = [0.5, 0, 0, ...]")
    print(f"\nDistances:")
    print(f"  d(origin, x) = {float(d_origin_x):.4f}")
    print(f"  d(origin, y) = {float(d_origin_y):.4f}")
    print(f"  d(x, y)      = {float(d_x_y):.4f}")

    # Verify triangle inequality
    triangle_holds = float(d_x_y) <= float(d_origin_x) + float(d_origin_y) + 1e-5
    print(f"\nTriangle inequality: {'PASS' if triangle_holds else 'FAIL'}")


def run_lut_generation_benchmark():
    """Benchmark LUT generation time."""
    print("\n" + "=" * 60)
    print("LUT GENERATION BENCHMARK")
    print("=" * 60)

    sizes = [1024, 4096, 16384, 65536]

    print(f"\n{'Size':<10} {'exp (ms)':<12} {'log (ms)':<12} {'tanh (ms)':<12}")
    print("-" * 46)

    for size in sizes:
        exp_mean, _ = benchmark_fn(generate_exp_lut, size, iterations=20)
        log_mean, _ = benchmark_fn(generate_log_lut, size, iterations=20)
        tanh_mean, _ = benchmark_fn(generate_tanh_lut, size, iterations=20)
        print(f"{size:<10} {exp_mean:<12.3f} {log_mean:<12.3f} {tanh_mean:<12.3f}")


def main():
    parser = argparse.ArgumentParser(description="mlx_hyperbolic benchmarks")
    parser.add_argument("--size", type=int, default=1_000_000,
                        help="Number of elements for latency tests")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of iterations per benchmark")
    parser.add_argument("--precision-only", action="store_true",
                        help="Run only precision tests")
    args = parser.parse_args()

    print("=" * 60)
    print("mlx_hyperbolic Benchmark Suite")
    print("=" * 60)
    print(f"Device: {mx.default_device()}")
    print(f"MLX version: {mx.__version__ if hasattr(mx, '__version__') else 'unknown'}")

    # Run tests
    precision_passed = run_precision_tests()

    if not args.precision_only:
        run_latency_benchmarks(args.size, args.iterations)
        run_mobius_benchmarks(iterations=args.iterations)
        run_poincare_distance_test()
        run_lut_generation_benchmark()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Precision tests: {'ALL PASSED' if precision_passed else 'SOME FAILED'}")
    print()


if __name__ == "__main__":
    main()

#include <metal_stdlib>
using namespace metal;

// ==============================================================================
// TMU-Based LUT Lookup Kernel
// ==============================================================================
// This kernel exploits the GPU's Texture Mapping Unit (TMU) for fast
// transcendental function approximation. Instead of computing exp/log/tanh
// in the ALU (10-20+ cycles), we sample pre-computed lookup tables through
// the TMU (1 cycle with hardware linear interpolation).
//
// Key optimizations:
// - filter::linear enables FREE hardware bilinear interpolation
// - Half (Float16) precision for maximum TMU throughput on M-series
// - Normalized coordinates [0,1] map to full LUT range
// - clamp_to_edge prevents out-of-bounds access
// ==============================================================================

// Sampler configuration: force TMU usage with hardware interpolation
constexpr sampler lut_sampler(
    coord::normalized,      // Input coords in [0.0, 1.0]
    address::clamp_to_edge, // Clamp out-of-range to boundary values
    filter::linear          // Hardware bilinear interpolation (FREE!)
);

// ==============================================================================
// Generic LUT Lookup Kernel
// ==============================================================================
// Maps input values to normalized texture coordinates and samples the LUT.
// The scale factor converts from input domain to [0,1] texture coordinates.
//
// Example for exp(x) with range [0, 10]:
//   input_scale = 1.0 / 10.0 = 0.1
//   coord = x * 0.1 maps x=0->0.0, x=10->1.0
// ==============================================================================
kernel void lut_lookup_kernel(
    device const half* input_values [[buffer(0)]],
    device half* output_values [[buffer(1)]],
    constant float& input_scale [[buffer(2)]],
    constant float& input_offset [[buffer(3)]],
    texture1d<half, access::sample> lut_texture [[texture(0)]],
    uint index [[thread_position_in_grid]])
{
    // 1. Fetch input value
    half x = input_values[index];

    // 2. Normalize to texture coordinate space [0, 1]
    //    coord = (x - offset) * scale
    float coord = (float(x) - input_offset) * input_scale;

    // 3. Sample LUT via TMU with hardware interpolation
    //    This is the "free" math - TMU handles interpolation in fixed-function HW
    half4 result = lut_texture.sample(lut_sampler, coord);

    // 4. Write output (using .r channel for 1D texture)
    output_values[index] = result.r;
}

// ==============================================================================
// Batch LUT Lookup (Multiple Operations)
// ==============================================================================
// Processes multiple independent lookups in a single dispatch.
// Useful for computing exp, log, tanh on different data in one kernel call.
// ==============================================================================
kernel void lut_lookup_batch_kernel(
    device const half* input_values [[buffer(0)]],
    device half* output_values [[buffer(1)]],
    constant float& input_scale [[buffer(2)]],
    constant float& input_offset [[buffer(3)]],
    constant uint& batch_stride [[buffer(4)]],
    texture1d<half, access::sample> lut_texture [[texture(0)]],
    uint2 index [[thread_position_in_grid]])  // (element, batch)
{
    uint global_idx = index.y * batch_stride + index.x;

    half x = input_values[global_idx];
    float coord = (float(x) - input_offset) * input_scale;
    half4 result = lut_texture.sample(lut_sampler, coord);
    output_values[global_idx] = result.r;
}

// ==============================================================================
// Fused Möbius Addition Kernel (Poincaré Ball)
// ==============================================================================
// Computes Möbius addition: x ⊕_c y = ((1 + 2c<x,y> + c||y||²)x + (1 - c||x||²)y)
//                                      / (1 + 2c<x,y> + c²||x||²||y||²)
//
// This kernel fuses the entire operation for a single vector pair.
// For batch operations, see mobius_add_batch_kernel.
//
// Note: This version uses standard ALU math. A fully TMU-optimized version
// would require LUT lookups for any exp/log/tanh used in distance computations.
// ==============================================================================
kernel void mobius_add_kernel(
    device const half* x [[buffer(0)]],       // First vector
    device const half* y [[buffer(1)]],       // Second vector
    device half* result [[buffer(2)]],        // Output vector
    constant uint& dim [[buffer(3)]],         // Vector dimension
    constant float& curvature [[buffer(4)]],  // Curvature c (typically 1.0)
    uint index [[thread_position_in_grid]])
{
    // Compute for one dimension at a time (parallelized across dimensions)
    if (index >= dim) return;

    // We need to compute dot products and norms, which require reduction
    // For now, this is a placeholder - actual implementation needs
    // either threadgroup reduction or pre-computed norms passed in

    // Placeholder: simple element-wise addition (to be replaced)
    float xi = float(x[index]);
    float yi = float(y[index]);
    result[index] = half(xi + yi);
}

// ==============================================================================
// Optimized Möbius Addition with Pre-computed Norms
// ==============================================================================
// This version expects pre-computed ||x||², ||y||², and <x,y> to avoid
// reduction operations in the kernel. The Python layer computes these
// using MLX's efficient reduction primitives.
// ==============================================================================
kernel void mobius_add_precomputed_kernel(
    device const half* x [[buffer(0)]],
    device const half* y [[buffer(1)]],
    device half* result [[buffer(2)]],
    constant float& x_norm_sq [[buffer(3)]],   // ||x||²
    constant float& y_norm_sq [[buffer(4)]],   // ||y||²
    constant float& xy_dot [[buffer(5)]],      // <x,y>
    constant float& curvature [[buffer(6)]],   // c
    constant uint& dim [[buffer(7)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= dim) return;

    float c = curvature;
    float xi = float(x[index]);
    float yi = float(y[index]);

    // Möbius addition formula components
    float numerator_x_coef = 1.0f + 2.0f * c * xy_dot + c * y_norm_sq;
    float numerator_y_coef = 1.0f - c * x_norm_sq;
    float denominator = 1.0f + 2.0f * c * xy_dot + c * c * x_norm_sq * y_norm_sq;

    // Compute result for this dimension
    float res = (numerator_x_coef * xi + numerator_y_coef * yi) / denominator;
    result[index] = half(res);
}

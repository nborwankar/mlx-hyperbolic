#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <mlx/mlx.h>
#include "lut_ops.h"

namespace py = pybind11;
namespace mx = mlx::core;

namespace mlx_hyperbolic {

// ==============================================================================
// Helper: Extract MTL::Buffer from MLX Array
// ==============================================================================
// MLX arrays on GPU are backed by MTL::Buffer in unified memory.
// This function extracts the underlying buffer for direct Metal access.
// ==============================================================================
MTL::Buffer* getBuffer(const mx::array& arr) {
    // Ensure array is on GPU and evaluated
    mx::eval(arr);

    // Get the underlying buffer
    // Note: This requires MLX internals access - implementation depends on MLX version
    // For now, we'll use the data pointer approach

    // TODO: Implement proper buffer extraction from MLX array
    // This is a placeholder that needs MLX internal API access
    throw std::runtime_error("Buffer extraction not yet implemented - needs MLX internal API");
}

// ==============================================================================
// Python-Exposed Functions
// ==============================================================================

/// Perform TMU-accelerated LUT lookup
/// @param input Input array (float16)
/// @param lut   Lookup table array (float16)
/// @param scale Scale factor for coordinate normalization
/// @param offset Offset for coordinate normalization
/// @return Output array with looked-up values
mx::array lut_lookup(
    const mx::array& input,
    const mx::array& lut,
    float scale,
    float offset)
{
    // Validate inputs
    if (input.dtype() != mx::float16) {
        throw std::invalid_argument("Input must be float16");
    }
    if (lut.dtype() != mx::float16) {
        throw std::invalid_argument("LUT must be float16");
    }

    // Create output array
    mx::array output = mx::zeros_like(input);

    // For now, fall back to MLX native operations
    // TODO: Implement Metal kernel dispatch when buffer extraction is available

    // Fallback implementation using MLX
    // This doesn't use TMU but provides correct results
    auto normalized = (input - offset) * scale;
    auto indices = mx::clip(normalized * static_cast<float>(lut.size() - 1), 0.0f,
                           static_cast<float>(lut.size() - 1));

    // Linear interpolation between LUT entries
    auto idx_low = mx::floor(indices);
    auto idx_high = mx::minimum(idx_low + 1, static_cast<float>(lut.size() - 1));
    auto frac = indices - idx_low;

    auto val_low = mx::take(lut, mx::astype(idx_low, mx::int32));
    auto val_high = mx::take(lut, mx::astype(idx_high, mx::int32));

    output = val_low + frac * (val_high - val_low);
    return output;
}

/// Perform Möbius addition in Poincaré ball
/// @param x First vector
/// @param y Second vector
/// @param c Curvature (default 1.0)
/// @return Result of x ⊕_c y
mx::array mobius_add(
    const mx::array& x,
    const mx::array& y,
    float c = 1.0f)
{
    // Compute norms and dot product using MLX
    auto x_norm_sq = mx::sum(x * x);
    auto y_norm_sq = mx::sum(y * y);
    auto xy_dot = mx::sum(x * y);

    // Möbius addition formula
    // (x ⊕_c y) = ((1 + 2c<x,y> + c||y||²)x + (1 - c||x||²)y) / (1 + 2c<x,y> + c²||x||²||y||²)

    auto num_x_coef = 1.0f + 2.0f * c * xy_dot + c * y_norm_sq;
    auto num_y_coef = 1.0f - c * x_norm_sq;
    auto denom = 1.0f + 2.0f * c * xy_dot + c * c * x_norm_sq * y_norm_sq;

    auto result = (num_x_coef * x + num_y_coef * y) / denom;
    return result;
}

/// Generate exponential LUT
/// @param size Number of entries
/// @param min_val Minimum input value
/// @param max_val Maximum input value
/// @return LUT array (float16)
mx::array generate_exp_lut(size_t size, float min_val, float max_val) {
    auto t = mx::linspace(min_val, max_val, static_cast<int>(size));
    auto lut = mx::exp(t);
    return mx::astype(lut, mx::float16);
}

/// Generate logarithm LUT
/// @param size Number of entries
/// @param min_val Minimum input value (must be > 0)
/// @param max_val Maximum input value
/// @return LUT array (float16)
mx::array generate_log_lut(size_t size, float min_val, float max_val) {
    if (min_val <= 0) {
        throw std::invalid_argument("min_val must be > 0 for log LUT");
    }
    auto t = mx::linspace(min_val, max_val, static_cast<int>(size));
    auto lut = mx::log(t);
    return mx::astype(lut, mx::float16);
}

/// Generate tanh LUT
/// @param size Number of entries
/// @param min_val Minimum input value
/// @param max_val Maximum input value
/// @return LUT array (float16)
mx::array generate_tanh_lut(size_t size, float min_val, float max_val) {
    auto t = mx::linspace(min_val, max_val, static_cast<int>(size));
    auto lut = mx::tanh(t);
    return mx::astype(lut, mx::float16);
}

}  // namespace mlx_hyperbolic

// ==============================================================================
// Python Module Definition
// ==============================================================================
PYBIND11_MODULE(_mlx_hyperbolic_ext, m) {
    m.doc() = "Hardware-accelerated hyperbolic operations using TMU-based LUT lookups";

    // LUT generation functions
    m.def("generate_exp_lut", &mlx_hyperbolic::generate_exp_lut,
          py::arg("size") = 4096,
          py::arg("min_val") = 0.0f,
          py::arg("max_val") = 10.0f,
          "Generate exponential lookup table");

    m.def("generate_log_lut", &mlx_hyperbolic::generate_log_lut,
          py::arg("size") = 4096,
          py::arg("min_val") = 1e-6f,
          py::arg("max_val") = 10.0f,
          "Generate logarithm lookup table");

    m.def("generate_tanh_lut", &mlx_hyperbolic::generate_tanh_lut,
          py::arg("size") = 4096,
          py::arg("min_val") = -5.0f,
          py::arg("max_val") = 5.0f,
          "Generate tanh lookup table");

    // Core operations
    m.def("lut_lookup", &mlx_hyperbolic::lut_lookup,
          py::arg("input"),
          py::arg("lut"),
          py::arg("scale"),
          py::arg("offset"),
          "Perform TMU-accelerated LUT lookup");

    m.def("mobius_add", &mlx_hyperbolic::mobius_add,
          py::arg("x"),
          py::arg("y"),
          py::arg("c") = 1.0f,
          "Möbius addition in Poincaré ball");

    // Version info
    m.attr("__version__") = "0.1.0";
}

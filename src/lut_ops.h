#pragma once

#include <Metal/Metal.hpp>
#include <Foundation/Foundation.hpp>
#include <cstddef>

namespace mlx_hyperbolic {

// ==============================================================================
// LUTOps: Metal-based LUT Operations Manager
// ==============================================================================
// Manages Metal resources for TMU-based lookup table operations.
// Key features:
// - Zero-copy texture creation from MLX buffers
// - Pre-compiled compute pipeline states
// - Thread-safe single-instance design
// ==============================================================================
class LUTOps {
public:
    LUTOps();
    ~LUTOps();

    // Disable copy
    LUTOps(const LUTOps&) = delete;
    LUTOps& operator=(const LUTOps&) = delete;

    // ==============================================================================
    // Core Operations
    // ==============================================================================

    /// Perform LUT lookup using TMU hardware interpolation
    /// @param input_buffer  Input values (half precision)
    /// @param output_buffer Output values (half precision)
    /// @param lut_buffer    Pre-computed lookup table (half precision)
    /// @param input_size    Number of input elements
    /// @param lut_size      Number of LUT entries
    /// @param scale         Scale factor: 1.0 / (max_input - min_input)
    /// @param offset        Offset: min_input value
    void lutLookup(
        MTL::Buffer* input_buffer,
        MTL::Buffer* output_buffer,
        MTL::Buffer* lut_buffer,
        size_t input_size,
        size_t lut_size,
        float scale,
        float offset
    );

    /// Perform Möbius addition in Poincaré ball
    /// @param x_buffer      First vector (half precision)
    /// @param y_buffer      Second vector (half precision)
    /// @param result_buffer Output vector (half precision)
    /// @param dim           Vector dimension
    /// @param x_norm_sq     Pre-computed ||x||²
    /// @param y_norm_sq     Pre-computed ||y||²
    /// @param xy_dot        Pre-computed <x,y>
    /// @param curvature     Ball curvature (typically 1.0)
    void mobiusAdd(
        MTL::Buffer* x_buffer,
        MTL::Buffer* y_buffer,
        MTL::Buffer* result_buffer,
        size_t dim,
        float x_norm_sq,
        float y_norm_sq,
        float xy_dot,
        float curvature
    );

    /// Get the Metal device
    MTL::Device* device() const { return device_; }

    /// Check if initialized
    bool isInitialized() const { return initialized_; }

private:
    void initialize();
    void cleanup();
    void createPipelineState(const char* kernel_name, MTL::ComputePipelineState*& pso);

    /// Create a texture view from a buffer (zero-copy)
    MTL::Texture* createTextureView(MTL::Buffer* buffer, size_t lut_size);

    MTL::Device* device_;
    MTL::Library* library_;
    MTL::CommandQueue* command_queue_;

    // Compute pipeline states
    MTL::ComputePipelineState* lut_lookup_pso_;
    MTL::ComputePipelineState* lut_lookup_batch_pso_;
    MTL::ComputePipelineState* mobius_add_pso_;

    bool initialized_;
};

/// Get the global LUTOps instance
LUTOps& getLUTOps();

}  // namespace mlx_hyperbolic

#include "lut_ops.h"
#include <stdexcept>

namespace mlx_hyperbolic {

// ==============================================================================
// LUTOps Implementation
// ==============================================================================

LUTOps::LUTOps() : device_(nullptr), library_(nullptr), initialized_(false) {
    initialize();
}

LUTOps::~LUTOps() {
    cleanup();
}

void LUTOps::initialize() {
    if (initialized_) return;

    // Get the default Metal device
    device_ = MTL::CreateSystemDefaultDevice();
    if (!device_) {
        throw std::runtime_error("Failed to create Metal device");
    }

    // Load the compiled Metal library
    NS::Error* error = nullptr;
    NS::String* path = NS::String::string(METALLIB_PATH, NS::UTF8StringEncoding);
    library_ = device_->newLibrary(path, &error);

    if (!library_) {
        std::string err_msg = "Failed to load Metal library";
        if (error) {
            err_msg += ": ";
            err_msg += error->localizedDescription()->utf8String();
        }
        throw std::runtime_error(err_msg);
    }

    // Create compute pipeline states for each kernel
    createPipelineState("lut_lookup_kernel", lut_lookup_pso_);
    createPipelineState("lut_lookup_batch_kernel", lut_lookup_batch_pso_);
    createPipelineState("mobius_add_precomputed_kernel", mobius_add_pso_);

    // Create command queue
    command_queue_ = device_->newCommandQueue();
    if (!command_queue_) {
        throw std::runtime_error("Failed to create command queue");
    }

    initialized_ = true;
}

void LUTOps::createPipelineState(const char* kernel_name,
                                  MTL::ComputePipelineState*& pso) {
    NS::Error* error = nullptr;
    NS::String* name = NS::String::string(kernel_name, NS::UTF8StringEncoding);
    MTL::Function* function = library_->newFunction(name);

    if (!function) {
        throw std::runtime_error(std::string("Failed to find kernel: ") + kernel_name);
    }

    pso = device_->newComputePipelineState(function, &error);
    function->release();

    if (!pso) {
        std::string err_msg = std::string("Failed to create pipeline for ") + kernel_name;
        if (error) {
            err_msg += ": ";
            err_msg += error->localizedDescription()->utf8String();
        }
        throw std::runtime_error(err_msg);
    }
}

void LUTOps::cleanup() {
    if (lut_lookup_pso_) { lut_lookup_pso_->release(); lut_lookup_pso_ = nullptr; }
    if (lut_lookup_batch_pso_) { lut_lookup_batch_pso_->release(); lut_lookup_batch_pso_ = nullptr; }
    if (mobius_add_pso_) { mobius_add_pso_->release(); mobius_add_pso_ = nullptr; }
    if (command_queue_) { command_queue_->release(); command_queue_ = nullptr; }
    if (library_) { library_->release(); library_ = nullptr; }
    if (device_) { device_->release(); device_ = nullptr; }
    initialized_ = false;
}

// ==============================================================================
// Zero-Copy Texture Creation
// ==============================================================================
// Creates a texture view over an existing buffer WITHOUT copying data.
// This is the key optimization - unified memory allows both buffer and
// texture access to the same physical memory.
// ==============================================================================
MTL::Texture* LUTOps::createTextureView(MTL::Buffer* buffer, size_t lut_size) {
    // Create texture descriptor for 1D R16Float texture
    MTL::TextureDescriptor* desc = MTL::TextureDescriptor::alloc()->init();
    desc->setTextureType(MTL::TextureType1D);
    desc->setPixelFormat(MTL::PixelFormatR16Float);  // Half precision
    desc->setWidth(lut_size);
    desc->setStorageMode(buffer->storageMode());
    desc->setUsage(MTL::TextureUsageShaderRead);

    // Create texture view from buffer (ZERO COPY!)
    // bytesPerRow = 0 for 1D textures
    MTL::Texture* texture = buffer->newTexture(desc, 0, lut_size * sizeof(uint16_t));
    desc->release();

    if (!texture) {
        throw std::runtime_error("Failed to create texture view from buffer");
    }

    return texture;
}

// ==============================================================================
// LUT Lookup Dispatch
// ==============================================================================
void LUTOps::lutLookup(
    MTL::Buffer* input_buffer,
    MTL::Buffer* output_buffer,
    MTL::Buffer* lut_buffer,
    size_t input_size,
    size_t lut_size,
    float scale,
    float offset)
{
    if (!initialized_) {
        throw std::runtime_error("LUTOps not initialized");
    }

    // Create texture view of LUT buffer (zero-copy)
    MTL::Texture* lut_texture = createTextureView(lut_buffer, lut_size);

    // Create command buffer and encoder
    MTL::CommandBuffer* cmd_buffer = command_queue_->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = cmd_buffer->computeCommandEncoder();

    // Set pipeline and buffers
    encoder->setComputePipelineState(lut_lookup_pso_);
    encoder->setBuffer(input_buffer, 0, 0);   // input_values
    encoder->setBuffer(output_buffer, 0, 1);  // output_values
    encoder->setBytes(&scale, sizeof(float), 2);   // input_scale
    encoder->setBytes(&offset, sizeof(float), 3);  // input_offset
    encoder->setTexture(lut_texture, 0);      // lut_texture

    // Calculate grid size
    NS::UInteger thread_group_size = lut_lookup_pso_->maxTotalThreadsPerThreadgroup();
    if (thread_group_size > input_size) {
        thread_group_size = input_size;
    }

    MTL::Size grid_size = MTL::Size(input_size, 1, 1);
    MTL::Size group_size = MTL::Size(thread_group_size, 1, 1);

    // Dispatch
    encoder->dispatchThreads(grid_size, group_size);
    encoder->endEncoding();

    // Execute and wait
    cmd_buffer->commit();
    cmd_buffer->waitUntilCompleted();

    // Cleanup
    lut_texture->release();
}

// ==============================================================================
// Möbius Addition Dispatch
// ==============================================================================
void LUTOps::mobiusAdd(
    MTL::Buffer* x_buffer,
    MTL::Buffer* y_buffer,
    MTL::Buffer* result_buffer,
    size_t dim,
    float x_norm_sq,
    float y_norm_sq,
    float xy_dot,
    float curvature)
{
    if (!initialized_) {
        throw std::runtime_error("LUTOps not initialized");
    }

    MTL::CommandBuffer* cmd_buffer = command_queue_->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = cmd_buffer->computeCommandEncoder();

    encoder->setComputePipelineState(mobius_add_pso_);
    encoder->setBuffer(x_buffer, 0, 0);
    encoder->setBuffer(y_buffer, 0, 1);
    encoder->setBuffer(result_buffer, 0, 2);
    encoder->setBytes(&x_norm_sq, sizeof(float), 3);
    encoder->setBytes(&y_norm_sq, sizeof(float), 4);
    encoder->setBytes(&xy_dot, sizeof(float), 5);
    encoder->setBytes(&curvature, sizeof(float), 6);

    uint32_t dim_u32 = static_cast<uint32_t>(dim);
    encoder->setBytes(&dim_u32, sizeof(uint32_t), 7);

    NS::UInteger thread_group_size = mobius_add_pso_->maxTotalThreadsPerThreadgroup();
    if (thread_group_size > dim) {
        thread_group_size = dim;
    }

    MTL::Size grid_size = MTL::Size(dim, 1, 1);
    MTL::Size group_size = MTL::Size(thread_group_size, 1, 1);

    encoder->dispatchThreads(grid_size, group_size);
    encoder->endEncoding();

    cmd_buffer->commit();
    cmd_buffer->waitUntilCompleted();
}

// Global instance for Python bindings
static LUTOps* g_lut_ops = nullptr;

LUTOps& getLUTOps() {
    if (!g_lut_ops) {
        g_lut_ops = new LUTOps();
    }
    return *g_lut_ops;
}

}  // namespace mlx_hyperbolic

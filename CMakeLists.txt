cmake_minimum_required(VERSION 3.24)
project(mlx_hyperbolic LANGUAGES CXX)

# ==============================================================================
# Build Configuration
# ==============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Apple Silicon optimization flags
if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=armv8.5-a")
endif()

# ==============================================================================
# Find Dependencies
# ==============================================================================

# Find MLX
find_package(MLX CONFIG REQUIRED)
message(STATUS "Found MLX: ${MLX_INCLUDE_DIRS}")

# Find Python and pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)
message(STATUS "Found Python: ${Python3_EXECUTABLE}")
message(STATUS "Found pybind11: ${pybind11_INCLUDE_DIRS}")

# ==============================================================================
# Metal Shader Compilation
# ==============================================================================
if(APPLE)
    # Find Metal compiler
    find_program(METAL_COMPILER xcrun REQUIRED)

    set(METAL_SOURCE ${CMAKE_SOURCE_DIR}/src/kernels/hyperbolic.metal)
    set(METAL_AIR ${CMAKE_BINARY_DIR}/hyperbolic.air)
    set(METAL_LIB ${CMAKE_BINARY_DIR}/hyperbolic.metallib)

    # Compile .metal -> .air
    add_custom_command(
        OUTPUT ${METAL_AIR}
        COMMAND ${METAL_COMPILER} -sdk macosx metal
                -c ${METAL_SOURCE}
                -o ${METAL_AIR}
                -std=metal3.0
        DEPENDS ${METAL_SOURCE}
        COMMENT "Compiling Metal shader to AIR"
    )

    # Link .air -> .metallib
    add_custom_command(
        OUTPUT ${METAL_LIB}
        COMMAND ${METAL_COMPILER} -sdk macosx metallib
                ${METAL_AIR}
                -o ${METAL_LIB}
        DEPENDS ${METAL_AIR}
        COMMENT "Linking Metal library"
    )

    # Custom target for Metal compilation
    add_custom_target(metal_shaders ALL DEPENDS ${METAL_LIB})
endif()

# ==============================================================================
# C++ Extension Library
# ==============================================================================
pybind11_add_module(_mlx_hyperbolic_ext
    src/main.cpp
    src/lut_ops.cpp
)

target_include_directories(_mlx_hyperbolic_ext PRIVATE
    ${MLX_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(_mlx_hyperbolic_ext PRIVATE
    mlx
    "-framework Metal"
    "-framework Foundation"
)

# Ensure Metal shaders are built before the extension
if(APPLE)
    add_dependencies(_mlx_hyperbolic_ext metal_shaders)
endif()

# Define path to metallib for runtime loading
target_compile_definitions(_mlx_hyperbolic_ext PRIVATE
    METALLIB_PATH="${METAL_LIB}"
)

# ==============================================================================
# Installation
# ==============================================================================
# Install the extension module to the Python package directory
install(TARGETS _mlx_hyperbolic_ext
    LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/python/mlx_hyperbolic
)

# Install the Metal library alongside the extension
if(APPLE)
    install(FILES ${METAL_LIB}
        DESTINATION ${CMAKE_SOURCE_DIR}/python/mlx_hyperbolic
    )
endif()

# ==============================================================================
# Summary
# ==============================================================================
message(STATUS "")
message(STATUS "========================================")
message(STATUS "mlx_hyperbolic Configuration Summary")
message(STATUS "========================================")
message(STATUS "  C++ Standard:    ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type:      ${CMAKE_BUILD_TYPE}")
message(STATUS "  MLX Include:     ${MLX_INCLUDE_DIRS}")
message(STATUS "  Python:          ${Python3_EXECUTABLE}")
message(STATUS "  Metal Shader:    ${METAL_SOURCE}")
message(STATUS "  Metal Library:   ${METAL_LIB}")
message(STATUS "========================================")

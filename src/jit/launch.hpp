#pragma once

// Kernel launch through the driver API.

#include <cstdint>
#include <vector>

#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "../runtime/exception.hpp"
#include "compiler.hpp"

namespace nvfp4_gemm::jit {

struct LaunchConfig
{
    int  grid_dim    = 1;
    int  num_threads = 128;
    int  smem_size   = 0;
    bool enable_pdl  = true;
};

/// Launch a JIT-compiled kernel on the current stream.
///
/// Arguments are forwarded by address, so they must outlive the call; every
/// caller here passes lvalues from an `Args` struct that does.
template<typename... Ts>
void launch(const KernelPtr& kernel, const LaunchConfig& config, Ts&... args)
{
    // Anything above 48 KB of shared memory needs an explicit opt-in per
    // function, and this kernel is far above it.
    if (config.smem_size > 0)
        NVFP4_CU_CHECK(cuFuncSetAttribute(kernel->function,
                                          CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                          config.smem_size));

    void* arg_ptrs[] = {static_cast<void*>(&args)...};

    CUlaunchConfig launch_config{};
    launch_config.gridDimX       = static_cast<unsigned>(config.grid_dim);
    launch_config.gridDimY       = 1;
    launch_config.gridDimZ       = 1;
    launch_config.blockDimX      = static_cast<unsigned>(config.num_threads);
    launch_config.blockDimY      = 1;
    launch_config.blockDimZ      = 1;
    launch_config.sharedMemBytes = static_cast<unsigned>(config.smem_size);
    launch_config.hStream        = static_cast<CUstream>(at::cuda::getCurrentCUDAStream().stream());

    // Programmatic dependent launch lets the next kernel start its prologue
    // while this one drains; the kernel's `cudaGridDependencySynchronize()` is
    // what makes that safe.
    CUlaunchAttribute attributes[1];
    launch_config.numAttrs = 0;
    if (config.enable_pdl)
    {
        attributes[0].id                                           = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
        attributes[0].value.programmaticStreamSerializationAllowed = 1;
        launch_config.attrs                                        = attributes;
        launch_config.numAttrs                                     = 1;
    }

    NVFP4_CU_CHECK(cuLaunchKernelEx(&launch_config, kernel->function, arg_ptrs, nullptr));
}

}   // namespace nvfp4_gemm::jit

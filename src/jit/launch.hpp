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
    /// 1 = no clustering. 2 pairs CTAs so a TMA multicast can deliver one A
    /// tile into both CTAs' shared memory with a single L2 read.
    int cluster_dim = 1;
    /// Pin this window (the weights) as L2-persisting for the launch.
    void*  l2_window_ptr   = nullptr;
    size_t l2_window_bytes = 0;
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

    const auto stream = static_cast<CUstream>(at::cuda::getCurrentCUDAStream().stream());
    if (config.l2_window_ptr != nullptr)
    {
        // Persisting hints need a carve-out first, and the window rides on the
        // stream. (Not captured by CUDA graphs -- callers measuring this must
        // launch eagerly.)
        cuCtxSetLimit(CU_LIMIT_PERSISTING_L2_CACHE_SIZE, config.l2_window_bytes);
        CUstreamAttrValue val{};
        val.accessPolicyWindow.base_ptr  = config.l2_window_ptr;
        val.accessPolicyWindow.num_bytes = config.l2_window_bytes;
        val.accessPolicyWindow.hitRatio  = 1.0f;
        val.accessPolicyWindow.hitProp   = CU_ACCESS_PROPERTY_PERSISTING;
        val.accessPolicyWindow.missProp  = CU_ACCESS_PROPERTY_STREAMING;
        cuStreamSetAttribute(stream, CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW, &val);
    }

    CUlaunchConfig launch_config{};
    launch_config.gridDimX       = static_cast<unsigned>(config.grid_dim);
    launch_config.gridDimY       = 1;
    launch_config.gridDimZ       = 1;
    launch_config.blockDimX      = static_cast<unsigned>(config.num_threads);
    launch_config.blockDimY      = 1;
    launch_config.blockDimZ      = 1;
    launch_config.sharedMemBytes = static_cast<unsigned>(config.smem_size);
    launch_config.hStream        = stream;

    // Programmatic dependent launch lets the next kernel start its prologue
    // while this one drains; the kernel's `cudaGridDependencySynchronize()` is
    // what makes that safe.
    CUlaunchAttribute attributes[2];
    launch_config.attrs    = attributes;
    launch_config.numAttrs = 0;
    if (config.enable_pdl)
    {
        attributes[launch_config.numAttrs].id                                           = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
        attributes[launch_config.numAttrs].value.programmaticStreamSerializationAllowed = 1;
        ++launch_config.numAttrs;
    }
    if (config.cluster_dim > 1)
    {
        NVFP4_HOST_ASSERT_MSG(config.grid_dim % config.cluster_dim == 0,
                              "grid must be a multiple of the cluster size");
        attributes[launch_config.numAttrs].id                       = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
        attributes[launch_config.numAttrs].value.clusterDim.x       = static_cast<unsigned>(config.cluster_dim);
        attributes[launch_config.numAttrs].value.clusterDim.y       = 1;
        attributes[launch_config.numAttrs].value.clusterDim.z       = 1;
        ++launch_config.numAttrs;
    }

    NVFP4_CU_CHECK(cuLaunchKernelEx(&launch_config, kernel->function, arg_ptrs, nullptr));
}

}   // namespace nvfp4_gemm::jit

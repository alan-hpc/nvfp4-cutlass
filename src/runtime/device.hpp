#pragma once

// Device properties the launcher needs, queried once and cached.

#include <string>
#include <utility>

#include <cuda_runtime.h>

#include "exception.hpp"

namespace nvfp4_gemm {

/// Shared memory available to one CTA on SM100/SM103, in bytes.
///
/// 227 KB of the SM's 228 KB; the last KB is reserved by the driver. This is a
/// constant rather than a query because `cudaDevAttrMaxSharedMemoryPerBlockOptin`
/// reports it only after the opt-in, and the pipeline depth has to be chosen
/// before the kernel exists.
constexpr int kSmemCapacitySm100 = 232448;

class DeviceRuntime
{
    cudaDeviceProp prop{};
    bool           loaded  = false;
    int            num_sms = 0;

    void ensure_loaded()
    {
        if (loaded)
            return;
        int device = 0;
        NVFP4_CUDA_CHECK(cudaGetDevice(&device));
        NVFP4_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        loaded = true;
    }

public:
    int get_num_sms()
    {
        ensure_loaded();
        return num_sms != 0 ? num_sms : prop.multiProcessorCount;
    }

    /// Override the persistent grid size.  Below the SM count this leaves room
    /// for a communication kernel; above it (up to 2 per SM) it oversubscribes
    /// so that a slim-tile configuration can co-schedule two CTAs per SM and
    /// overlap their pipeline chains.
    void set_num_sms(int value)
    {
        ensure_loaded();
        NVFP4_HOST_ASSERT(0 <= value and value <= 2 * prop.multiProcessorCount);
        num_sms = value;
    }

    std::pair<int, int> get_arch_pair()
    {
        ensure_loaded();
        return {prop.major, prop.minor};
    }
};

inline DeviceRuntime* device_runtime = new DeviceRuntime();

}   // namespace nvfp4_gemm

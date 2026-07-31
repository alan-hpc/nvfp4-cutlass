#pragma once

// Device properties the launcher needs, queried once and cached.

#include <string>

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

    /// Override the SM count, e.g. to leave room for a communication kernel.
    void set_num_sms(int value)
    {
        ensure_loaded();
        NVFP4_HOST_ASSERT(0 <= value and value <= prop.multiProcessorCount);
        num_sms = value;
    }

    std::pair<int, int> get_arch_pair()
    {
        ensure_loaded();
        return {prop.major, prop.minor};
    }

    /// The `-arch` target to compile for.
    ///
    /// B200 (10.0) and B300 (10.3) share the SM100 *family* target, so one cubin
    /// runs on both. `sm_100a` would be arch-specific to 10.0 and would fail to
    /// load on a B300, which is why family targets -- and thus CUDA >= 12.9 --
    /// are required rather than merely preferred.
    std::string get_arch(bool support_arch_family)
    {
        ensure_loaded();
        if (prop.major == 10)
            return support_arch_family ? "100f" : "100a";
        return std::to_string(prop.major * 10 + prop.minor) + "a";
    }
};

inline DeviceRuntime* device_runtime = new DeviceRuntime();

}   // namespace nvfp4_gemm

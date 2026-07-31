#pragma once

// Warp-level primitives.
//
// Derived from DeepGEMM (MIT, Copyright (c) 2025 DeepSeek),
// `deep_gemm/ptx/utils.cuh`, reduced to what this kernel uses.

#include <cstdint>

#include <nvfp4_gemm/common/macros.cuh>

#ifdef NVFP4_IN_CUDA_COMPILATION

namespace nvfp4_gemm::ptx {

CUTLASS_DEVICE uint32_t get_lane_idx()
{
    uint32_t lane_id;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane_id));
    return lane_id;
}

/// Broadcast a value from one lane to the whole warp.
///
/// The MMA warp keeps the per-stage UMMA descriptor of stage `s` in lane `s`, so
/// selecting a stage costs one shuffle instead of recomputing an address.
template<typename dtype_t>
CUTLASS_DEVICE dtype_t exchange(dtype_t value, const uint32_t& src_lane_idx)
{
    NVFP4_STATIC_ASSERT(sizeof(dtype_t) % sizeof(uint32_t) == 0, "Must be a whole number of words");
    const auto send_int_values = reinterpret_cast<uint32_t*>(&value);
    dtype_t    recv_dtype;
    auto       recv_int_values = reinterpret_cast<uint32_t*>(&recv_dtype);
#    pragma unroll
    for (uint32_t i = 0; i < sizeof(dtype_t) / sizeof(uint32_t); ++i)
        recv_int_values[i] = __shfl_sync(0xffffffff, send_int_values[i], static_cast<int>(src_lane_idx));
    return recv_dtype;
}

}   // namespace nvfp4_gemm::ptx

#endif

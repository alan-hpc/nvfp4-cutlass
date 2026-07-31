#pragma once

// Small math helpers used by the kernel and its host-side config.
//
// Derived from DeepGEMM (MIT, Copyright (c) 2025 DeepSeek),
// `deep_gemm/common/math.cuh`, reduced to the functions this kernel calls.

#include <cstdint>

#include <nvfp4_gemm/common/macros.cuh>

namespace nvfp4_gemm::math {

template<typename T>
CUTLASS_HOST_DEVICE T ceil_div(T a, T b)
{
    return (a + b - 1) / b;
}

template<typename T>
CUTLASS_HOST_DEVICE constexpr T constexpr_ceil_div(T a, T b)
{
    return (a + b - 1) / b;
}

template<typename T>
CUTLASS_HOST_DEVICE constexpr T constexpr_align(T a, T b)
{
    return constexpr_ceil_div(a, b) * b;
}

template<typename T>
CUTLASS_HOST_DEVICE constexpr T constexpr_min(T a, T b)
{
    return a < b ? a : b;
}

#ifdef NVFP4_IN_CUDA_COMPILATION

/// Fast approximate reciprocal, ~1 ulp of 2^-23.
///
/// The transform uses this instead of a true divide: the block scale it divides
/// by is itself an E4M3 round-trip, so the extra ulp is far below the
/// quantization error already present.
CUTLASS_HOST_DEVICE float fast_rcp(const float& x)
{
    float ret;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(ret) : "f"(x));
    return ret;
}

/// Convert a pair of FP32 values to BF16 and pack them into one 32-bit word.
template<typename old_t>
CUTLASS_DEVICE int cast_into_bf16_and_pack(old_t& x, old_t& y)
{
    auto bf16x2 = __float22bfloat162_rn({*reinterpret_cast<float*>(&x), *reinterpret_cast<float*>(&y)});
    return *reinterpret_cast<int*>(&bf16x2);
}

#endif

}   // namespace nvfp4_gemm::math

#pragma once

// Compile-time helpers for shared-memory addressing and tensor-memory sizing.
//
// Derived from DeepGEMM (MIT, Copyright (c) 2025 DeepSeek),
// `deep_gemm/common/utils.cuh`, reduced to what this kernel uses.

#include <utility>

#include <cuda/std/cstdint>

#include <nvfp4_gemm/common/macros.cuh>

namespace nvfp4_gemm::utils {

/// Indexable view over a pointer-producing lambda.
///
/// Lets the kernel write `smem_a[stage]` for shared-memory regions whose stride
/// is a compile-time expression, without materializing an array of pointers.
template<typename FuncT>
struct PatternVisitor
{
    FuncT func;

    CUTLASS_HOST_DEVICE explicit PatternVisitor(FuncT&& func)
        : func(std::forward<FuncT>(func))
    {}

    CUTLASS_HOST_DEVICE auto operator[](const uint32_t& i) const
    {
        return func(i);
    }
};

/// Round a tensor-memory column count up to an allocatable size.
///
/// The TMEM allocator only grants 32/64/128/256/512 columns.
template<uint32_t kNumCols>
CUTLASS_DEVICE constexpr uint32_t get_num_aligned_tmem_cols()
{
    NVFP4_STATIC_ASSERT(kNumCols <= 512, "Too many tensor memory columns");
    if constexpr (kNumCols <= 32)
        return 32;
    if constexpr (kNumCols <= 64)
        return 64;
    if constexpr (kNumCols <= 128)
        return 128;
    if constexpr (kNumCols <= 256)
        return 256;
    return 512;
}

}   // namespace nvfp4_gemm::utils

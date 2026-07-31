#pragma once

// TMA bulk-tensor copies into shared memory.
//
// Derived from DeepGEMM (MIT, Copyright (c) 2025 DeepSeek),
// `deep_gemm/common/tma_copy.cuh`, reduced to the 2D single-CTA path -- this
// kernel does not use 2-SM multicast or 3D (batched) descriptors.

#include <cute/arch/copy_sm90_tma.hpp>
#include <cutlass/arch/barrier.h>

#include <nvfp4_gemm/common/macros.cuh>

namespace nvfp4_gemm::tma {

/// Elements of the inner (contiguous) dimension covered by one swizzle atom.
///
/// A swizzle mode is expressed in bytes, so an atom holds
/// `kSwizzleMode / sizeof(T)` elements; mode 0 means no swizzling and the whole
/// block is one atom.
template<uint32_t BLOCK_INNER, uint32_t kSwizzleMode, typename dtype_t>
constexpr uint32_t get_inner_block_atom_size()
{
    return kSwizzleMode == 0 ? BLOCK_INNER : kSwizzleMode / sizeof(dtype_t);
}

/// Issue one TMA load per swizzle atom along the inner dimension.
///
/// `inner_idx` / `outer_idx` are tensor coordinates in *elements* of the tensor
/// map's data type. For packed FP4 under `16U4_ALIGN8B` that means FP4 elements,
/// not bytes, even though shared memory receives two per byte.
template<uint32_t BLOCK_INNER, uint32_t BLOCK_OUTER, uint32_t kSwizzleMode, typename dtype_t>
CUTLASS_DEVICE void copy(void const*                               desc_ptr,
                         cutlass::arch::ClusterTransactionBarrier* barrier_ptr,
                         dtype_t*                                  smem_ptr,
                         const uint32_t&                           inner_idx,
                         const uint32_t&                           outer_idx)
{
    constexpr uint32_t BLOCK_INNER_ATOM = get_inner_block_atom_size<BLOCK_INNER, kSwizzleMode, dtype_t>();

#pragma unroll
    for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM; ++i)
    {
        cute::SM90_TMA_LOAD_2D::copy(desc_ptr,
                                     reinterpret_cast<uint64_t*>(barrier_ptr),
                                     static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL),
                                     smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM,
                                     inner_idx + i * BLOCK_INNER_ATOM,
                                     outer_idx);
    }
}

}   // namespace nvfp4_gemm::tma

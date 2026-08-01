#pragma once

// Shared-memory load/store intrinsics.
//
// Derived from DeepGEMM (MIT, Copyright (c) 2025 DeepSeek),
// `deep_gemm/ptx/ld_st.cuh`, reduced to the widths this kernel issues.
//
// These exist rather than plain dereferences because the transform's access
// pattern is a hand-computed swizzle: going through `ld.shared` / `st.shared`
// with an explicit generic-to-shared cast keeps the compiler from re-deriving
// addresses it cannot see through.

#include <cstdint>

#include <nvfp4_gemm/common/macros.cuh>

#ifdef NVFP4_IN_CUDA_COMPILATION

namespace nvfp4_gemm::ptx {

CUTLASS_DEVICE uint32_t ld_shared(const uint32_t* ptr)
{
    uint32_t ret;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

CUTLASS_DEVICE uint4 ld_shared(const uint4* ptr)
{
    uint4 ret;
    asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
                 : "l"(ptr));
    return ret;
}

CUTLASS_DEVICE void st_shared(const uint32_t* ptr, uint32_t val)
{
    asm volatile("st.shared.u32 [%0], %1;" ::"l"(ptr), "r"(val));
}

/// Half-word store, for writers that own only part of a 4-byte SF atom word.
CUTLASS_DEVICE void st_shared(const uint16_t* ptr, uint16_t val)
{
    asm volatile("st.shared.u16 [%0], %1;" ::"l"(ptr), "h"(val));
}

/// Transposed 8x8 BF16 matrix store, for the swap-AB epilogue: lanes hold
/// accumulator columns (tokens) and `stmatrix.trans` lands them as rows.
struct SM90_U32x4_STSM_T
{
    CUTLASS_DEVICE static void copy(uint32_t src_0, uint32_t src_1, uint32_t src_2, uint32_t src_3, void* smem_dst)
    {
        asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16.trans [%0], {%1, %2, %3, %4};\n" ::"l"(__cvta_generic_to_shared(smem_dst)),
                     "r"(src_0),
                     "r"(src_1),
                     "r"(src_2),
                     "r"(src_3));
    }
};

CUTLASS_DEVICE void st_shared(const void* ptr, uint32_t x, uint32_t y, uint32_t z, uint32_t w)
{
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(ptr), "r"(x), "r"(y), "r"(z), "r"(w));
}

}   // namespace nvfp4_gemm::ptx

#endif

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

#include <cuda/std/cstdint>

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

CUTLASS_DEVICE void st_shared(const void* ptr, uint32_t x, uint32_t y, uint32_t z, uint32_t w)
{
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(ptr), "r"(x), "r"(y), "r"(z), "r"(w));
}

}   // namespace nvfp4_gemm::ptx

#endif

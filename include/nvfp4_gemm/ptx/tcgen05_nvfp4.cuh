#pragma once

#include <cuda/std/cstdint>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <deep_gemm/common/exception.cuh>

namespace nvfp4_gemm::ptx {

/// NVFP4 block-scaled UMMA (SM100/SM103).
///
/// DeepGEMM ships `SM100_MMA_MXF4_SS`, which is *MXFP4*: block size 32 with
/// UE8M0 scales.  NVFP4 is a different operand kind -- block size 16 with E4M3
/// scales -- so it needs its own instruction, `kind::mxf4nvf4` with `.block16`.
///
/// One `.block16` instruction covers UMMA_K = 64 elements, i.e. four scale
/// sub-blocks per row.  That is exactly one 128x4 scale-factor atom, which is
/// why `kTmemStartColOfSFA` advances by 4 columns per UMMA-K step.
struct SM100_MMA_MXF4NVF4_SS {
    CUTLASS_DEVICE static void
    fma(uint64_t const& desc_a,
        uint64_t const& desc_b,
        uint32_t const& tmem_c,
        uint32_t const& scale_c,
        uint64_t const& desc,
        uint32_t const& tmem_sfa,
        uint32_t const& tmem_sfb) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
#if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9)
            "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#else
            "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#endif
            "}\n"
            :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(static_cast<uint32_t>(desc >> 32)), "r"(scale_c),
               "r"(tmem_sfa), "r"(tmem_sfb));
    }
};

struct SM100_MMA_MXF4NVF4_2x1SM_SS {
    CUTLASS_DEVICE static void
    fma(uint64_t const& desc_a,
        uint64_t const& desc_b,
        uint32_t const& tmem_c,
        uint32_t const& scale_c,
        uint64_t const& desc,
        uint32_t const& tmem_sfa,
        uint32_t const& tmem_sfb) {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
#if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9)
            "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#else
            "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#endif
            "}\n"
            :: "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(static_cast<uint32_t>(desc >> 32)), "r"(scale_c),
               "r"(tmem_sfa), "r"(tmem_sfb));
    }
};

/// Hardware FP32 -> E2M1 conversion, two elements per instruction.
///
/// `cvt.rn.satfinite.e2m1x2.f32 d, a, b` packs `a` into the high nibble and `b`
/// into the low nibble of the destination byte.  The argument names below make
/// that packing order explicit at every call site.
CUTLASS_DEVICE uint32_t cvt_e2m1x2(const float& hi_nibble, const float& lo_nibble) {
    uint16_t packed;
    asm volatile("cvt.rn.satfinite.e2m1x2.f32 %0, %1, %2;\n"
                 : "=h"(packed) : "f"(hi_nibble), "f"(lo_nibble));
    return static_cast<uint32_t>(packed) & 0xffu;
}

/// Hardware FP32 -> E4M3 conversion for a single value.
///
/// There is no scalar `cvt.rn.satfinite.e4m3.f32`, so we convert a pair and drop
/// the unused half.
CUTLASS_DEVICE uint32_t cvt_e4m3(const float& x) {
    uint16_t packed;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n"
                 : "=h"(packed) : "f"(0.0f), "f"(x));
    return static_cast<uint32_t>(packed) & 0xffu;
}

/// Decode a packed E2M1 byte back into two FP16 values.
///
/// The residual pass needs `dec(q0)` right after quantizing it.  Going through
/// this hardware unpack is what lets pass 1 skip an integer piecewise dequant --
/// the doc measures that as 25.76us -> 19.71us on the transform kernel.
CUTLASS_DEVICE uint32_t cvt_e2m1x2_to_f16x2(const uint32_t& packed) {
    uint32_t result;
    asm volatile("cvt.rn.f16x2.e2m1x2 %0, %1;\n"
                 : "=r"(result) : "h"(static_cast<uint16_t>(packed & 0xffu)));
    return result;
}

/// Decode an E4M3 byte back to FP32.
///
/// `s0` has to come back to FP32 to build the reciprocal and to derive `s1`, and
/// the value must be the *rounded* one -- carrying the pre-rounding float
/// forward would make the kernel disagree with the reference.
CUTLASS_DEVICE float cvt_e4m3_to_f32(const uint32_t& code) {
    __nv_fp8_storage_t storage = static_cast<__nv_fp8_storage_t>(code & 0xffu);
    return __half2float(__nv_cvt_fp8_to_halfraw(storage, __NV_E4M3));
}

} // namespace nvfp4_gemm::ptx

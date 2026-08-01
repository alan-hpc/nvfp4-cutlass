#pragma once

// tcgen05 (SM100/SM103 tensor core) instructions and the FP4/E4M3 conversions
// the transform needs.
//
// The fences are derived from DeepGEMM (MIT, Copyright (c) 2025 DeepSeek),
// `deep_gemm/ptx/tcgen05.cuh`. The NVFP4 MMA and the conversion intrinsics are
// new: DeepGEMM only ships MXFP4 (`kind::mxf4`, block 32, UE8M0 scales), and
// NVFP4 is a different operand kind -- block 16 with E4M3 scales.

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <nvfp4_gemm/common/macros.cuh>

namespace nvfp4_gemm::ptx {

/// NVFP4 block-scaled UMMA, single CTA.
///
/// One `.block16` instruction covers UMMA_K = 64 elements, i.e. four scale
/// sub-blocks per row. That is exactly one 128x4 scale-factor atom, which is why
/// the SF TMEM column advances by 4 per UMMA-K step.
struct SM100_MMA_MXF4NVF4_SS
{
    CUTLASS_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t const& tmem_c,
                                   uint32_t const& scale_c,
                                   uint64_t const& desc,
                                   uint32_t const& tmem_sfa,
                                   uint32_t const& tmem_sfb)
    {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
#if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9)
            "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#else
            "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#endif
            "}\n" ::"r"(tmem_c),
            "l"(desc_a),
            "l"(desc_b),
            "r"(static_cast<uint32_t>(desc >> 32)),
            "r"(scale_c),
            "r"(tmem_sfa),
            "r"(tmem_sfb));
    }
};

/// 2-SM variant: the cluster's leader CTA issues one UMMA_M = 256 instruction
/// whose A rows 128..255 (and their scales) come from the peer CTA's memories.
/// Only the leader may call this.
struct SM100_MMA_MXF4NVF4_2x1SM_SS
{
    CUTLASS_DEVICE static void fma(uint64_t const& desc_a,
                                   uint64_t const& desc_b,
                                   uint32_t const& tmem_c,
                                   uint32_t const& scale_c,
                                   uint64_t const& desc,
                                   uint32_t const& tmem_sfa,
                                   uint32_t const& tmem_sfb)
    {
        asm volatile(
            "{\n\t"
            ".reg .pred p;\n\t"
            "setp.ne.b32 p, %4, 0;\n\t"
#if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9)
            "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#else
            "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#endif
            "}\n" ::"r"(tmem_c),
            "l"(desc_a),
            "l"(desc_b),
            "r"(static_cast<uint32_t>(desc >> 32)),
            "r"(scale_c),
            "r"(tmem_sfa),
            "r"(tmem_sfb));
    }
};

/// Tensor-memory ordering fences.
///
/// tcgen05 accesses tensor memory asynchronously with respect to the issuing
/// thread, so any handoff through a barrier needs an explicit fence on both
/// sides.
CUTLASS_DEVICE void tcgen05_before_thread_sync()
{
    asm volatile("tcgen05.fence::before_thread_sync;");
}

CUTLASS_DEVICE void tcgen05_after_thread_sync()
{
    asm volatile("tcgen05.fence::after_thread_sync;");
}

/// FP32 -> E2M1, two elements per instruction.
///
/// These go through the CUDA headers' intrinsics rather than hand-written PTX.
/// `cvt.rn.satfinite.e2m1x2.f32` takes a `.b8` destination and
/// `cvt.rn.f16x2.e2m1x2` a `.b8` source, and inline asm has no 8-bit constraint
/// class -- writing them by hand needs a `.reg .b8` temporary plus explicit
/// widening moves, which is exactly what these intrinsics already do.
///
/// Using them also settles the packing order by definition instead of by
/// assumption: `float2{x, y}` puts `x` in the low nibble and `y` in the high
/// nibble, matching how UMMA reads the operand back.
CUTLASS_DEVICE uint32_t cvt_e2m1x2(const float& hi_nibble, const float& lo_nibble)
{
    const __nv_fp4x2_storage_t packed =
        __nv_cvt_float2_to_fp4x2(make_float2(lo_nibble, hi_nibble), __NV_E2M1, cudaRoundNearest);
    return static_cast<uint32_t>(packed) & 0xffu;
}

/// FP32 -> E4M3, one value.
CUTLASS_DEVICE uint32_t cvt_e4m3(const float& x)
{
    return static_cast<uint32_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3)) & 0xffu;
}

/// Decode a packed E2M1 byte back into two FP16 values.
///
/// The residual pass needs `dec(q0)` right after quantizing it. Going through
/// this hardware unpack is what lets pass 1 skip an integer piecewise dequant --
/// the algorithm doc measures that as 25.76us -> 19.71us on the transform.
CUTLASS_DEVICE uint32_t cvt_e2m1x2_to_f16x2(const uint32_t& packed)
{
    const __half2_raw halves =
        __nv_cvt_fp4x2_to_halfraw2(static_cast<__nv_fp4x2_storage_t>(packed & 0xffu), __NV_E2M1);
    return *reinterpret_cast<const uint32_t*>(&halves);
}

/// Packed BF16 pair -> E2M1 pair, one instruction, no FP32 round trip.
///
/// SM100's converter accepts bf16x2 directly (probed: ptxas assembles
/// `cvt.rn.satfinite.e2m1x2.bf16x2` under sm_100f). The low bf16 half lands
/// in the low nibble, mirroring the f32 variant's packing order.
CUTLASS_DEVICE uint32_t cvt_e2m1x2_bf16x2(const uint32_t& pair)
{
    uint32_t r;
    asm volatile(
        "{\n\t"
        ".reg .b8 t;\n\t"
        "cvt.rn.satfinite.e2m1x2.bf16x2 t, %1;\n\t"
        "cvt.u32.u8 %0, t;\n\t"
        "}\n" : "=r"(r) : "r"(pair));
    return r;
}

/// Four packed BF16 pairs -> one word of four packed E2M1 pairs.
///
/// Same vector-mov trick as the f32 form: the four `.b8` results assemble in
/// the destination register without widening movs or shift/or chains.
CUTLASS_DEVICE uint32_t cvt_e2m1x8_bf16x2(const uint32_t& p0, const uint32_t& p1,
                                          const uint32_t& p2, const uint32_t& p3)
{
    uint32_t r;
    asm volatile(
        "{\n\t"
        ".reg .b8 t0, t1, t2, t3;\n\t"
        "cvt.rn.satfinite.e2m1x2.bf16x2 t0, %1;\n\t"
        "cvt.rn.satfinite.e2m1x2.bf16x2 t1, %2;\n\t"
        "cvt.rn.satfinite.e2m1x2.bf16x2 t2, %3;\n\t"
        "cvt.rn.satfinite.e2m1x2.bf16x2 t3, %4;\n\t"
        "mov.b32 %0, {t0, t1, t2, t3};\n\t"
        "}\n"
        : "=r"(r)
        : "r"(p0), "r"(p1), "r"(p2), "r"(p3));
    return r;
}

/// Eight FP32 values -> one word of four packed E2M1 pairs.
///
/// The four `.b8` results pack straight into the destination through a vector
/// mov, so ptxas emits a couple of PRMT merges instead of four widening movs
/// plus a shift/or chain.  Element 2i lands in the low nibble of byte i,
/// mirroring `cvt_e2m1x2`'s float2 convention (PTX `cvt.e2m1x2.f32 d, a, b`
/// puts b in the low nibble).
CUTLASS_DEVICE uint32_t cvt_e2m1x8_f32(const float& f0, const float& f1,
                                       const float& f2, const float& f3,
                                       const float& f4, const float& f5,
                                       const float& f6, const float& f7)
{
    uint32_t r;
    asm volatile(
        "{\n\t"
        ".reg .b8 t0, t1, t2, t3;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 t0, %2, %1;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 t1, %4, %3;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 t2, %6, %5;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 t3, %8, %7;\n\t"
        "mov.b32 %0, {t0, t1, t2, t3};\n\t"
        "}\n"
        : "=r"(r)
        : "f"(f0), "f"(f1), "f"(f2), "f"(f3), "f"(f4), "f"(f5), "f"(f6), "f"(f7));
    return r;
}

/// Decode a packed E2M1 byte straight into a BF16 pair.
///
/// The packed residual path subtracts `dec(q0)` from `u` in bf16x2; both are
/// exact on the E2M1 grid's scale, and by Sterbenz's lemma the subtraction of
/// two bf16 numbers within 2x of each other is exact, so nothing here loses
/// bits the E2M1 quantization of the residual would have kept.
CUTLASS_DEVICE uint32_t cvt_bf16x2_e2m1x2(const uint32_t& packed)
{
    uint32_t r;
    asm volatile(
        "{\n\t"
        ".reg .b8 t;\n\t"
        "cvt.u8.u32 t, %1;\n\t"
        "cvt.rn.bf16x2.e2m1x2 %0, t;\n\t"
        "}\n" : "=r"(r) : "r"(packed));
    return r;
}

/// Decode an E4M3 byte to FP32.
///
/// `s0` has to come back to FP32 to build the reciprocal and to derive `s1`, and
/// it must be the *rounded* value -- carrying the pre-rounding float forward
/// would make the kernel disagree with the reference on every block whose
/// `amax / 6` is not exactly representable in E4M3.
CUTLASS_DEVICE float cvt_e4m3_to_f32(const uint32_t& code)
{
    return __half2float(__nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(code & 0xffu), __NV_E4M3));
}

}   // namespace nvfp4_gemm::ptx

#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cute/swizzle.hpp>

#include <nvfp4_gemm/common/macros.cuh>
#include <nvfp4_gemm/common/math.cuh>
#include <nvfp4_gemm/ptx/ld_st.cuh>

#include <nvfp4_gemm/ptx/tcgen05.cuh>

namespace nvfp4_gemm::transform {

/// NVFP4 block granularity along K.
constexpr uint32_t kBlockSize = 16;
/// One UMMA-K step of `kind::mxf4nvf4.block16` consumes 64 K elements, which is
/// four scale sub-blocks -- exactly one 128x4 scale-factor atom.
constexpr uint32_t kKPerSFAtom = 64;
constexpr uint32_t kSFPerAtom  = kKPerSFAtom / kBlockSize;   // 4

/// Reciprocal of the E2M1 max finite magnitude.
constexpr float kInvE2M1Max = 1.0f / 6.0f;

/// `derived_div8` floors `s0` at 8 * 2^-9 == 2^-6 so that `s1 = s0 / 8` cannot
/// underflow past the smallest positive E4M3 subnormal.  A zero block therefore
/// yields q0 = q1 = 0 with finite scales instead of inf/NaN.
constexpr float kS0FloorDerivedDiv8 = 0.015625f;   // 2^-6

enum class ScalePolicy
{
    /// s1 = Q_e4m3(s0 * 2^-3), no second reduction.  Kernel default.
    DerivedDiv8 = 0,
    /// s1 = Q_e4m3(amax(residual) / 6).  More accurate, one extra reduction.
    ResidualAmax = 1,
    /// s1 = Q_e4m3(s0 * 2^-2): same cost as div8, but the residual never
    /// clips.  The E2M1 grid's largest half-gap is 1 (between 4 and 6), so a
    /// x8 residual reaches 8 > 6 and saturates on ~6% of blocks; x4 tops out
    /// at 4 <= 6 at the price of half the resolution on small residuals.
    /// Which effect wins is an empirical question -- hence a policy, not a
    /// replacement.
    DerivedDiv4 = 2,
    /// s0 = Q_e4m3(amax / 4) with the default x8 residual (R40).  q0 then
    /// spends only the |u| <= 4 half of the E2M1 grid, whose largest gap is 1
    /// (not the 2 between 4 and 6): max|e0| halves to 1/2, and the x8
    /// residual r = 8*e0 <= 4 never clips, so max|e1| = 1/16 -- worst case
    /// M/64 vs the default's M/24.  The price is one wasted q0 code point
    /// (6) and 1.5x less headroom before the e4m3 s0 encode saturates.
    Div4Radix8 = 3,
};

/// Residual radix per policy: how far the normalized residual is scaled up
/// before its own E2M1 quantization, with s1 derived as s0 / radix.
template<ScalePolicy kScalePolicy>
constexpr float kResidualRadix = kScalePolicy == ScalePolicy::DerivedDiv4 ? 4.0f : 8.0f;

/// Reciprocal of the s0 divisor: amax/6 fills the whole E2M1 range (default);
/// Div4Radix8 trades range for the tighter grid below 4.
template<ScalePolicy kScalePolicy>
constexpr float kInvS0Div = kScalePolicy == ScalePolicy::Div4Radix8 ? 0.25f : kInvE2M1Max;

/// Byte offset of element (m, k) inside a K-major, `kSwizzleMode`-swizzled tile.
///
/// The transform warps read A and write A0/A1 with plain LDS/STS, so they have to
/// land on the same addresses TMA and UMMA use. Rather than re-derive the bit
/// pattern, apply CuTe's own swizzle functor -- `Swizzle<3,4,3>`, `<2,4,3>` and
/// `<1,4,3>` are the definitions of SW128/SW64/SW32, and are the same objects
/// `UMMA::Layout_K_SW*_Atom` and the TMA descriptors are built from. Whatever
/// CUTLASS means by a swizzle mode, this means too.
///
/// (Hand-deriving it is how this went wrong once: the XOR source was written as
/// `row % chunks`, which happens to be right for 128 B and wrong for 64 B, and
/// the GEMM came back with the right magnitude and no correlation.)
///
/// Everything is byte-granular, because the swizzle is: one helper then serves
/// the BF16 A tile and the packed-FP4 A0/A1 tiles alike.
template<uint32_t BLOCK_MN, uint32_t kSwizzleMode>
CUTLASS_DEVICE uint32_t swizzled_byte_offset(const uint32_t& m, const uint32_t& k_byte)
{
    NVFP4_STATIC_ASSERT(kSwizzleMode == 32 or kSwizzleMode == 64 or kSwizzleMode == 128, "Unsupported swizzle mode");
    using swizzle_t = cute::conditional_t<kSwizzleMode == 128,
                                          cute::Swizzle<3, 4, 3>,
                                          cute::conditional_t<kSwizzleMode == 64, cute::Swizzle<2, 4, 3>, cute::Swizzle<1, 4, 3>>>;

    // TMA splits a row into `row_bytes / kSwizzleMode` atoms, each stored as its
    // own [BLOCK_MN, kSwizzleMode] sub-tile; the swizzle acts within one atom.
    const uint32_t atom_idx     = k_byte / kSwizzleMode;
    const uint32_t byte_in_atom = k_byte % kSwizzleMode;
    const uint32_t linear       = m * kSwizzleMode + byte_in_atom;

    return atom_idx * (BLOCK_MN * kSwizzleMode) + static_cast<uint32_t>(swizzle_t{}(linear));
}

/// uint32 index of row `m` inside a 128x4 scale-factor atom, in the layout UTCCP
/// expects.
///
/// DeepGEMM's UTCCP transposer produces `new[j] = old[(j % 4) * 32 + j / 4]`,
/// where `old` is indexed by row.  We generate the scales ourselves, so we write
/// them pre-transposed and skip that warp shuffle entirely for SFA.  Inverting
/// the permutation gives `j = (m % 32) * 4 + m / 32`.
CUTLASS_DEVICE uint32_t sf_atom_word_idx(const uint32_t& m)
{
    return (m % 32) * 4 + (m / 32);
}

/// Result of decomposing one 16-element block.
struct BlockCodes
{
    uint32_t q0_packed;   // 8 bytes worth of nibbles, 2 elements per byte
    uint32_t q1_packed;
    uint32_t s0_code;   // E4M3
    uint32_t s1_code;   // E4M3
};

/// Algorithm 1: decompose one 16-element BF16 block into two NVFP4 passes.
///
/// `x` holds the 16 values already widened to FP32.  `q0_bytes` / `q1_bytes`
/// receive 8 packed bytes each.  Everything after the amax stays in the
/// *normalized* domain: the residual is `(x * rcp(s0) - dec(q0)) * 8`, never
/// `(x - dec(q0) * s0) / s1`, which saves a dequant multiply per element.
/// `kEnableResidualPass = false` degenerates this to plain single-pass NVFP4.
/// That mode exists so the benchmark can measure the residual pass against an
/// otherwise byte-identical kernel: same pipeline, same tiles, same epilogue.
template<ScalePolicy kScalePolicy, bool kEnableResidualPass = true>
CUTLASS_DEVICE void decompose_block(const float (&x)[kBlockSize],
                                    uint8_t (&q0_bytes)[kBlockSize / 2],
                                    uint8_t (&q1_bytes)[kBlockSize / 2],
                                    uint32_t& s0_code,
                                    uint32_t& s1_code)
{
    // 1: block amax.
    float amax = 0.0f;
#pragma unroll
    for (uint32_t i = 0; i < kBlockSize; ++i)
        amax = fmaxf(amax, fabsf(x[i]));

    // 2: s0 = Q_e4m3(max(amax / 6, floor)).  The floor keeps s1 representable.
    constexpr float kFloor = kScalePolicy == ScalePolicy::ResidualAmax
                                 ? 1.0f / 512.0f   // 2^-9, smallest positive E4M3
                                 : kS0FloorDerivedDiv8;
    s0_code                = ptx::cvt_e4m3(fmaxf(amax * kInvS0Div<kScalePolicy>, kFloor));

    // The reciprocal has to be built from the *rounded* s0, otherwise the kernel
    // and the reference disagree on every block whose amax/6 is not exactly
    // representable in E4M3.
    const float s0f    = ptx::cvt_e4m3_to_f32(s0_code);
    const float inv_s0 = math::fast_rcp(s0f);

    // 3: q0 = Q_e2m1(x * rcp(s0)), two elements per hardware instruction.
    float u[kBlockSize];
#pragma unroll
    for (uint32_t i = 0; i < kBlockSize; ++i)
        u[i] = x[i] * inv_s0;

    float q0_dec[kBlockSize];
#pragma unroll
    for (uint32_t i = 0; i < kBlockSize / 2; ++i)
    {
        // Element 2i goes in the low nibble, 2i+1 in the high nibble, matching
        // the E2M1 packing order UMMA reads back.
        const uint32_t packed = ptx::cvt_e2m1x2(u[2 * i + 1], u[2 * i]);
        q0_bytes[i]           = static_cast<uint8_t>(packed);

        if constexpr (kEnableResidualPass)
        {
            const uint32_t halves  = ptx::cvt_e2m1x2_to_f16x2(packed);
            const float2   decoded = __half22float2(*reinterpret_cast<const __half2*>(&halves));
            q0_dec[2 * i]          = decoded.x;
            q0_dec[2 * i + 1]      = decoded.y;
        }
    }

    if constexpr (not kEnableResidualPass)
    {
        // Single-pass mode: emit a zero residual so the second MMA, if it is
        // still issued, contributes nothing. The scale must stay a valid finite
        // E4M3 rather than zero.
        s1_code = ptx::cvt_e4m3(s0f / kResidualRadix<kScalePolicy>);
#pragma unroll
        for (uint32_t i = 0; i < kBlockSize / 2; ++i)
            q1_bytes[i] = 0;
        return;
    }

    // 5/6: residual pass.
    if constexpr (kScalePolicy != ScalePolicy::ResidualAmax)
    {
        constexpr float kRadix = kResidualRadix<kScalePolicy>;
        s1_code                = ptx::cvt_e4m3(s0f / kRadix);
#pragma unroll
        for (uint32_t i = 0; i < kBlockSize / 2; ++i)
        {
            const float r_lo = (u[2 * i] - q0_dec[2 * i]) * kRadix;
            const float r_hi = (u[2 * i + 1] - q0_dec[2 * i + 1]) * kRadix;
            q1_bytes[i]      = static_cast<uint8_t>(ptx::cvt_e2m1x2(r_hi, r_lo));
        }
    }
    else
    {
        // Residual amax costs a second 16-way reduction but picks the tightest
        // possible s1, which matters when a block's residual is far below s0/8.
        float r[kBlockSize], r_amax = 0.0f;
#pragma unroll
        for (uint32_t i = 0; i < kBlockSize; ++i)
        {
            r[i]   = (u[i] - q0_dec[i]) * s0f;
            r_amax = fmaxf(r_amax, fabsf(r[i]));
        }
        s1_code            = ptx::cvt_e4m3(fmaxf(r_amax * kInvE2M1Max, 1.0f / 512.0f));
        const float inv_s1 = math::fast_rcp(ptx::cvt_e4m3_to_f32(s1_code));
#pragma unroll
        for (uint32_t i = 0; i < kBlockSize / 2; ++i)
            q1_bytes[i] = static_cast<uint8_t>(
                ptx::cvt_e2m1x2(r[2 * i + 1] * inv_s1, r[2 * i] * inv_s1));
    }
}

/// Packed Algorithm 1, op-count tuned against SASS reality.
///
/// Measured pipeline verdict (B300, warp clocks): the loop is issue-bound,
/// not convert-pipe-bound -- an all-f32 variant with 3 convert ops per pair
/// but more total ops measured SLOWER (0.960 vs 0.885 us/kb).  So this form
/// minimizes TOTAL ops in bf16 domain:
///   * u = x * rcp(s0) stays __hmul2 (ptxas emulates it as 2xFMUL + one
///     F2FP repack -- 3 ops; the f32 unpack/mul alternative is 4+),
///   * q0/q1 encode four pairs per asm block, vector-mov'ing the `.b8`
///     results into one word (kills 4 widening movs + the shift/or chain),
///   * the residual's *8 is an INTEGER exponent bump: r and 8r are both
///     bf16-exact, so adding 3 (log2 radix) to both exponent fields via one
///     IADD replaces the emulated multiply's 2xFMUL + repack.  Zero halves
///     (u == dec) become +-2^-125*1.5, which e2m1's round-to-nearest returns
///     to +-0 -- bitwise the same code the multiply produced,
///   * scalars: hardware s0 ENCODE (rounding must match the reference), but
///     integer decode -- the derived floor (2^-6) keeps s0 e4m3-NORMAL, whose
///     f32 image is exactly (code << 20) + 0x3C000000.  ResidualAmax's 2^-9
///     floor can reach subnormal codes and keeps the convert-chain decode.
/// The residual subtraction u - dec is exact in bf16 by Sterbenz's lemma
/// (both within 2x on the e2m1 grid, and trivially exact at dec = 0).
template<ScalePolicy kScalePolicy, bool kEnableResidualPass = true>
CUTLASS_DEVICE void decompose_block_packed(const uint32_t (&xw)[kBlockSize / 2],
                                           uint32_t (&q0w)[kBlockSize / 8],
                                           uint32_t (&q1w)[kBlockSize / 8],
                                           uint32_t& s0_code,
                                           uint32_t& s1_code)
{
    // 1: block amax on packed pairs, as a TREE -- max is exact under any
    // association, and the linear chain was 7 serial VHMNMX deep on the
    // per-kb latency path (the transform's exposed cost is chain depth, not
    // throughput: tw8 == tw16 measured).  Depth drops to 3.
    __nv_bfloat162 t[kBlockSize / 4];
#pragma unroll
    for (uint32_t i = 0; i < kBlockSize / 4; ++i)
        t[i] = __hmax2(__habs2(*reinterpret_cast<const __nv_bfloat162*>(&xw[2 * i])),
                       __habs2(*reinterpret_cast<const __nv_bfloat162*>(&xw[2 * i + 1])));
#pragma unroll
    for (uint32_t stride = kBlockSize / 8; stride > 0; stride /= 2)
#pragma unroll
        for (uint32_t i = 0; i < stride; ++i)
            t[i] = __hmax2(t[i], t[i + stride]);
    const __nv_bfloat162 m     = t[0];
    const uint32_t       mswap = __byte_perm(*reinterpret_cast<const uint32_t*>(&m), 0, 0x1032);
    const __nv_bfloat162 mboth = __hmax2(m, *reinterpret_cast<const __nv_bfloat162*>(&mswap));
    const float          amax  = __bfloat162float(mboth.x);

    // 2: s0 -- hardware encode, integer decode where the floor allows.
    constexpr float kFloor = kScalePolicy == ScalePolicy::ResidualAmax
                                 ? 1.0f / 512.0f
                                 : kS0FloorDerivedDiv8;
    s0_code            = ptx::cvt_e4m3(fmaxf(amax * kInvS0Div<kScalePolicy>, kFloor));
    const float s0f    = kScalePolicy == ScalePolicy::ResidualAmax
                             ? ptx::cvt_e4m3_to_f32(s0_code)
                             : __int_as_float((s0_code << 20) + 0x3C000000u);
    const float inv_s0 = math::fast_rcp(s0f);

    // 3: u = x * rcp(s0), two elements per (emulated) instruction.
    const __nv_bfloat162 inv2 = __float2bfloat162_rn(inv_s0);
    uint32_t             u2[kBlockSize / 2];
#pragma unroll
    for (uint32_t i = 0; i < kBlockSize / 2; ++i)
    {
        const __nv_bfloat162 u =
            __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&xw[i]), inv2);
        u2[i] = *reinterpret_cast<const uint32_t*>(&u);
    }
#pragma unroll
    for (uint32_t w = 0; w < kBlockSize / 8; ++w)
        q0w[w] = ptx::cvt_e2m1x8_bf16x2(u2[4 * w + 0], u2[4 * w + 1], u2[4 * w + 2], u2[4 * w + 3]);

    if constexpr (not kEnableResidualPass)
    {
        s1_code = ptx::cvt_e4m3(s0f / kResidualRadix<kScalePolicy>);
#pragma unroll
        for (uint32_t w = 0; w < kBlockSize / 8; ++w)
            q1w[w] = 0;
        return;
    }

    // 5/6: residual pass.
    if constexpr (kScalePolicy != ScalePolicy::ResidualAmax)
    {
        constexpr float    kRadix   = kResidualRadix<kScalePolicy>;
        constexpr uint32_t kExpBump = (kRadix == 8.0f ? 3u : 2u) << 7;
        s1_code                     = ptx::cvt_e4m3(s0f / kRadix);
#pragma unroll
        for (uint32_t w = 0; w < kBlockSize / 8; ++w)
        {
            uint32_t r8[4];
#pragma unroll
            for (uint32_t j = 0; j < 4; ++j)
            {
                const uint32_t       dec_bits = ptx::cvt_bf16x2_e2m1x2(q0w[w] >> (j * 8));
                const __nv_bfloat162 d        =
                    __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&u2[4 * w + j]),
                            *reinterpret_cast<const __nv_bfloat162*>(&dec_bits));
                r8[j] = *reinterpret_cast<const uint32_t*>(&d) + (kExpBump | kExpBump << 16);
            }
            q1w[w] = ptx::cvt_e2m1x8_bf16x2(r8[0], r8[1], r8[2], r8[3]);
        }
    }
    else
    {
        // Absolute-domain residual (r = (u - dec) * s0) in f32: it needs the
        // whole block's r before its own amax picks s1, and its extra scale
        // multiplies are not power-of-2.  Non-default policy; correctness
        // over cycles.
        float r[kBlockSize];
        float ramax = 0.0f;
#pragma unroll
        for (uint32_t i = 0; i < kBlockSize / 2; ++i)
        {
            const uint32_t dec = ptx::cvt_bf16x2_e2m1x2(q0w[i / 4] >> (i % 4 * 8));
            const float    ulo = __int_as_float(u2[i] << 16);
            const float    uhi = __int_as_float(u2[i] & 0xffff0000u);
            r[2 * i]           = (ulo - __int_as_float(dec << 16)) * s0f;
            r[2 * i + 1]       = (uhi - __int_as_float(dec & 0xffff0000u)) * s0f;
            ramax              = fmaxf(ramax, fmaxf(fabsf(r[2 * i]), fabsf(r[2 * i + 1])));
        }
        s1_code            = ptx::cvt_e4m3(fmaxf(ramax * kInvE2M1Max, 1.0f / 512.0f));
        const float inv_s1 = math::fast_rcp(ptx::cvt_e4m3_to_f32(s1_code));
#pragma unroll
        for (uint32_t w = 0; w < kBlockSize / 8; ++w)
            q1w[w] = ptx::cvt_e2m1x8_f32(r[8 * w + 0] * inv_s1, r[8 * w + 1] * inv_s1,
                                         r[8 * w + 2] * inv_s1, r[8 * w + 3] * inv_s1,
                                         r[8 * w + 4] * inv_s1, r[8 * w + 5] * inv_s1,
                                         r[8 * w + 6] * inv_s1, r[8 * w + 7] * inv_s1);
    }
}

/// Two independent blocks decomposed with their chains explicitly interleaved.
///
/// The transform's exposed cost is per-kb dependency-chain latency (R31), and
/// a slot's blocks are independent: phase-ordering both amax trees and both
/// scalar chains (E4M3 round trip + MUFU rcp, the longest-latency links)
/// ahead of the element math lets block B's scalar latency hide under block
/// A's quantization instead of serializing after it.  Same operations in the
/// same rounding order as the single-block form -- bitwise identical output.
template<ScalePolicy kScalePolicy, bool kEnableResidualPass = true>
CUTLASS_DEVICE void decompose_block_packed_x2(const uint32_t (&xa)[kBlockSize / 2],
                                              const uint32_t (&xb)[kBlockSize / 2],
                                              uint32_t (&q0a)[kBlockSize / 8],
                                              uint32_t (&q0b)[kBlockSize / 8],
                                              uint32_t (&q1a)[kBlockSize / 8],
                                              uint32_t (&q1b)[kBlockSize / 8],
                                              uint32_t& s0ca,
                                              uint32_t& s0cb,
                                              uint32_t& s1ca,
                                              uint32_t& s1cb)
{
    if constexpr (kScalePolicy == ScalePolicy::ResidualAmax)
    {
        // The absolute-domain policy has data-dependent scalar work mid-chain;
        // keep it on the plain path (non-default, correctness over cycles).
        decompose_block_packed<kScalePolicy, kEnableResidualPass>(xa, q0a, q1a, s0ca, s1ca);
        decompose_block_packed<kScalePolicy, kEnableResidualPass>(xb, q0b, q1b, s0cb, s1cb);
        return;
    }

    // ---- Phase 1: both amax trees, interleaved -----------------------------
    __nv_bfloat162 ta[kBlockSize / 4], tb[kBlockSize / 4];
#pragma unroll
    for (uint32_t i = 0; i < kBlockSize / 4; ++i)
    {
        ta[i] = __hmax2(__habs2(*reinterpret_cast<const __nv_bfloat162*>(&xa[2 * i])),
                        __habs2(*reinterpret_cast<const __nv_bfloat162*>(&xa[2 * i + 1])));
        tb[i] = __hmax2(__habs2(*reinterpret_cast<const __nv_bfloat162*>(&xb[2 * i])),
                        __habs2(*reinterpret_cast<const __nv_bfloat162*>(&xb[2 * i + 1])));
    }
#pragma unroll
    for (uint32_t stride = kBlockSize / 8; stride > 0; stride /= 2)
#pragma unroll
        for (uint32_t i = 0; i < stride; ++i)
        {
            ta[i] = __hmax2(ta[i], ta[i + stride]);
            tb[i] = __hmax2(tb[i], tb[i + stride]);
        }
    const uint32_t swa = __byte_perm(*reinterpret_cast<const uint32_t*>(&ta[0]), 0, 0x1032);
    const uint32_t swb = __byte_perm(*reinterpret_cast<const uint32_t*>(&tb[0]), 0, 0x1032);
    const __nv_bfloat162 ma = __hmax2(ta[0], *reinterpret_cast<const __nv_bfloat162*>(&swa));
    const __nv_bfloat162 mb = __hmax2(tb[0], *reinterpret_cast<const __nv_bfloat162*>(&swb));
    const float amax_a = __bfloat162float(ma.x);
    const float amax_b = __bfloat162float(mb.x);

    // ---- Phase 2: both scalar chains (the two MUFU rcps pipeline) ----------
    constexpr float kRadix = kResidualRadix<kScalePolicy>;
    s0ca = ptx::cvt_e4m3(fmaxf(amax_a * kInvS0Div<kScalePolicy>, kS0FloorDerivedDiv8));
    s0cb = ptx::cvt_e4m3(fmaxf(amax_b * kInvS0Div<kScalePolicy>, kS0FloorDerivedDiv8));
    const float s0fa = __int_as_float((s0ca << 20) + 0x3C000000u);
    const float s0fb = __int_as_float((s0cb << 20) + 0x3C000000u);
    const float inva = math::fast_rcp(s0fa);
    const float invb = math::fast_rcp(s0fb);
    s1ca = ptx::cvt_e4m3(s0fa / kRadix);
    s1cb = ptx::cvt_e4m3(s0fb / kRadix);
    const __nv_bfloat162 inv2a = __float2bfloat162_rn(inva);
    const __nv_bfloat162 inv2b = __float2bfloat162_rn(invb);

    // ---- Phase 3: element math, word-interleaved A/B -----------------------
    constexpr uint32_t kExpBump = ((kRadix == 8.0f ? 3u : 2u) << 7);
    constexpr uint32_t kBump2   = kExpBump | kExpBump << 16;
#pragma unroll
    for (uint32_t w = 0; w < kBlockSize / 8; ++w)
    {
        uint32_t ua[4], ub[4];
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j)
        {
            const __nv_bfloat162 va =
                __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&xa[4 * w + j]), inv2a);
            const __nv_bfloat162 vb =
                __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&xb[4 * w + j]), inv2b);
            ua[j] = *reinterpret_cast<const uint32_t*>(&va);
            ub[j] = *reinterpret_cast<const uint32_t*>(&vb);
        }
        q0a[w] = ptx::cvt_e2m1x8_bf16x2(ua[0], ua[1], ua[2], ua[3]);
        q0b[w] = ptx::cvt_e2m1x8_bf16x2(ub[0], ub[1], ub[2], ub[3]);
        if constexpr (kEnableResidualPass)
        {
            uint32_t ra[4], rb[4];
#pragma unroll
            for (uint32_t j = 0; j < 4; ++j)
            {
                const uint32_t       da = ptx::cvt_bf16x2_e2m1x2(q0a[w] >> (j * 8));
                const uint32_t       db = ptx::cvt_bf16x2_e2m1x2(q0b[w] >> (j * 8));
                const __nv_bfloat162 sa =
                    __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&ua[j]),
                            *reinterpret_cast<const __nv_bfloat162*>(&da));
                const __nv_bfloat162 sb =
                    __hsub2(*reinterpret_cast<const __nv_bfloat162*>(&ub[j]),
                            *reinterpret_cast<const __nv_bfloat162*>(&db));
                ra[j] = *reinterpret_cast<const uint32_t*>(&sa) + kBump2;
                rb[j] = *reinterpret_cast<const uint32_t*>(&sb) + kBump2;
            }
            q1a[w] = ptx::cvt_e2m1x8_bf16x2(ra[0], ra[1], ra[2], ra[3]);
            q1b[w] = ptx::cvt_e2m1x8_bf16x2(rb[0], rb[1], rb[2], rb[3]);
        }
        else
        {
            q1a[w] = 0;
            q1b[w] = 0;
        }
    }
}

/// Transform one A tile: BF16 [BLOCK_M, BLOCK_K] -> (A0, SFA0) + (A1, SFA1).
///
/// Thread mapping: every thread owns one whole -- or, at 16 warps, exactly half
/// of one -- 128x4 scale atom's worth of K (`kKPerSFAtom = 64` contiguous
/// elements of one row).  The slot never straddles atoms, and that matters:
///
///   * the block scales a thread produces land in a single aligned SFA store
///     (4 bytes for a whole atom, 2 bytes for a half),
///   * no lane holds a partial 16-element block, so the amax needs no `shfl`
///     at all, and
///   * no scale is ever computed twice.
///
/// The doc's kernel instead gives each lane half a block and merges the amax
/// with a butterfly shuffle, which makes both lanes of a pair compute the same
/// `s0`, `rcp(s0)` and `s1`, and store the same two scale bytes to the same
/// address.  Owning whole blocks removes that duplicated scalar work outright
/// rather than trying to predicate it away.
///
/// The half-atom split exists because the transform sits on the mainloop's
/// critical path (measured: +1 MMA pass costs its full issue time, and a small
/// extra reduction in the transform shows up 1:1 in kernel latency).  With one
/// atom per thread the warp count is capped at 8 by task count; halving the
/// slot doubles the usable warps to 16.
template<uint32_t BLOCK_M, uint32_t BLOCK_K, uint32_t kSwizzleAMode, uint32_t kSwizzleA0Mode, uint32_t kNumTransformThreads, ScalePolicy kScalePolicy, bool kEnableResidualPass = true, bool kPairDecompose = true, bool kHalfWorkProbe = false, bool kQuarterChainProbe = false>
CUTLASS_DEVICE void transform_a_tile(const __nv_bfloat16* smem_a,
                                     uint8_t*             smem_a0,
                                     uint8_t*             smem_a1,
                                     uint8_t*             smem_sfa0,
                                     uint8_t*             smem_sfa1,
                                     const uint32_t&      transform_tid)
{
    constexpr uint32_t kAtomsPerRow = BLOCK_K / kKPerSFAtom;
    constexpr uint32_t kTotalBlocks = BLOCK_M * BLOCK_K / kBlockSize;
    NVFP4_STATIC_ASSERT(BLOCK_K % kKPerSFAtom == 0, "Block K must cover whole SF atoms");
    NVFP4_STATIC_ASSERT(kTotalBlocks % kNumTransformThreads == 0,
                        "Transform threads must evenly divide the tile");

    // A slot is one thread's slice of one iteration: at most a whole scale atom
    // (the SFA word write assumes it), at least half of one (so the packed FP4
    // output is still one aligned 16-byte chunk).
    constexpr uint32_t kBlocksPerSlot =
        kTotalBlocks / kNumTransformThreads < kSFPerAtom ? kTotalBlocks / kNumTransformThreads
                                                         : kSFPerAtom;
    NVFP4_STATIC_ASSERT(kBlocksPerSlot >= 1 and kSFPerAtom % kBlocksPerSlot == 0,
                        "A transform slot must be a whole, half or quarter scale atom");
    constexpr uint32_t kThreadsPerAtom = kSFPerAtom / kBlocksPerSlot;
    constexpr uint32_t kElemsPerSlot   = kBlocksPerSlot * kBlockSize;
    constexpr uint32_t kNumSlots       = kTotalBlocks / kBlocksPerSlot;
    constexpr uint32_t kSlotsPerThread = kNumSlots / kNumTransformThreads;

    // Byte size of one 128-row scale atom: 128 rows x 4 E4M3 bytes.  A tile
    // shorter than 128 rows (the swap-AB token tile) fills only its rows of
    // the atom; the tail is never read, because UMMA with N = BLOCK_M reads
    // exactly BLOCK_M scale rows.
    constexpr uint32_t kSFAtomBytes = 128 * kSFPerAtom;
    NVFP4_STATIC_ASSERT(BLOCK_M <= 128 and 128 % BLOCK_M == 0,
                        "SF atom addressing assumes at most one 128-row atom");

#pragma unroll
    for (uint32_t t = 0; t < kSlotsPerThread; ++t)
    {
        const uint32_t slot     = t * kNumTransformThreads + transform_tid;
        const uint32_t atom_seq = slot / kThreadsPerAtom;
        const uint32_t sub      = slot % kThreadsPerAtom;   // half index inside the atom
        const uint32_t m        = atom_seq / kAtomsPerRow;
        const uint32_t atom_idx = atom_seq % kAtomsPerRow;
        const uint32_t k_base   = atom_idx * kKPerSFAtom + sub * kElemsPerSlot;

        // ---- Load the slot's BF16 as swizzled 16-byte groups, kept packed ----
        uint32_t xw[kElemsPerSlot / 2];
#pragma unroll
        for (uint32_t g = 0; g < kElemsPerSlot * sizeof(__nv_bfloat16) / 16; ++g)
        {
            const uint32_t k_byte = (k_base + g * 8) * sizeof(__nv_bfloat16);
            const uint32_t offset = swizzled_byte_offset<BLOCK_M, kSwizzleAMode>(m, k_byte);
            const uint4    raw    = ptx::ld_shared(
                reinterpret_cast<const uint4*>(reinterpret_cast<const uint8_t*>(smem_a) + offset));
            xw[g * 4 + 0] = raw.x;
            xw[g * 4 + 1] = raw.y;
            xw[g * 4 + 2] = raw.z;
            xw[g * 4 + 3] = raw.w;
        }

        // ---- Algorithm 1 on each of the slot's 16-element blocks -------------
        // Packed E2M1 words, four pairs each; the stores group them by four.
        uint32_t q0[kElemsPerSlot / 8];
        uint32_t q1[kElemsPerSlot / 8];
        uint32_t sf0_word = 0, sf1_word = 0;
        // Half-atom slots (2 blocks) take the interleaved two-block path: it
        // shortens the per-slot chain, which is the pacing recurrence in the
        // deep-stage regime (bk64/s6: 50.6 -> 49.2 us at 8k).  Whole-atom
        // slots (4 blocks) stay on the per-block loop -- the pair variant's
        // register footprint cost 2 us at swap decode128 (bk256/tw8).
        if constexpr (kBlocksPerSlot == 2 and kPairDecompose)
        {
            uint32_t ba[kBlockSize / 2], bb[kBlockSize / 2];
#pragma unroll
            for (uint32_t i = 0; i < kBlockSize / 2; ++i)
            {
                ba[i] = xw[i];
                bb[i] = xw[kBlockSize / 2 + i];
            }
            uint32_t s0ca, s0cb, s1ca, s1cb;
            uint32_t q0a[kBlockSize / 8], q0b[kBlockSize / 8];
            uint32_t q1a[kBlockSize / 8], q1b[kBlockSize / 8];
            if constexpr (kQuarterChainProbe)
            {
                // Timing probe: one 8-element micro chain per thread (a single
                // LDS group's amax, s0 and packed q0 word), every output slot
                // filled by duplication (WRONG RESULTS).  Targets the ~190ns
                // chain the R37 response curve says closes the idle window.
                const __nv_bfloat162 h0 =
                    __hmax2(__habs2(*reinterpret_cast<const __nv_bfloat162*>(&ba[0])),
                            __habs2(*reinterpret_cast<const __nv_bfloat162*>(&ba[1])));
                const __nv_bfloat162 h1 =
                    __hmax2(__habs2(*reinterpret_cast<const __nv_bfloat162*>(&ba[2])),
                            __habs2(*reinterpret_cast<const __nv_bfloat162*>(&ba[3])));
                const __nv_bfloat162 hm = __hmax2(h0, h1);
                const uint32_t       hs = __byte_perm(*reinterpret_cast<const uint32_t*>(&hm), 0, 0x1032);
                const float          amax =
                    __bfloat162float(__hmax2(hm, *reinterpret_cast<const __nv_bfloat162*>(&hs)).x);
                s0ca            = ptx::cvt_e4m3(fmaxf(amax * kInvS0Div<kScalePolicy>, kS0FloorDerivedDiv8));
                const float s0f = __int_as_float((s0ca << 20) + 0x3C000000u);
                const __nv_bfloat162 inv2 = __float2bfloat162_rn(math::fast_rcp(s0f));
                uint32_t             u[4];
#pragma unroll
                for (uint32_t j = 0; j < 4; ++j)
                {
                    const __nv_bfloat162 v =
                        __hmul2(*reinterpret_cast<const __nv_bfloat162*>(&ba[j]), inv2);
                    u[j] = *reinterpret_cast<const uint32_t*>(&v);
                }
                const uint32_t w0 = ptx::cvt_e2m1x8_bf16x2(u[0], u[1], u[2], u[3]);
#pragma unroll
                for (uint32_t w = 0; w < kBlockSize / 8; ++w)
                {
                    q0a[w] = w0;
                    q0b[w] = w0;
                    q1a[w] = 0;
                    q1b[w] = 0;
                }
                s0cb = s0ca;
                s1ca = ptx::cvt_e4m3(s0f / kResidualRadix<kScalePolicy>);
                s1cb = s1ca;
            }
            else if constexpr (kHalfWorkProbe)
            {
                // Timing probe: run the full dual chain on block A only and
                // duplicate its outputs for block B (WRONG RESULTS).  Halves
                // the element math while stores, UTCCP and both UMMA passes
                // stay identical -- the ~400ns-chain data point.
                decompose_block_packed<kScalePolicy, kEnableResidualPass>(
                    ba, q0a, q1a, s0ca, s1ca);
#pragma unroll
                for (uint32_t w = 0; w < kBlockSize / 8; ++w)
                {
                    q0b[w] = q0a[w];
                    q1b[w] = q1a[w];
                }
                s0cb = s0ca;
                s1cb = s1ca;
            }
            else
                decompose_block_packed_x2<kScalePolicy, kEnableResidualPass>(
                    ba, bb, q0a, q0b, q1a, q1b, s0ca, s0cb, s1ca, s1cb);
#pragma unroll
            for (uint32_t w = 0; w < kBlockSize / 8; ++w)
            {
                q0[w]                    = q0a[w];
                q0[kBlockSize / 8 + w]   = q0b[w];
                q1[w]                    = q1a[w];
                q1[kBlockSize / 8 + w]   = q1b[w];
            }
            sf0_word = s0ca | s0cb << 8;
            sf1_word = s1ca | s1cb << 8;
        }
        else
        {
#pragma unroll
            for (uint32_t b = 0; b < kBlocksPerSlot; ++b)
            {
                uint32_t block[kBlockSize / 2];
#pragma unroll
                for (uint32_t i = 0; i < kBlockSize / 2; ++i)
                    block[i] = xw[b * (kBlockSize / 2) + i];

                uint32_t q0wb[kBlockSize / 8], q1wb[kBlockSize / 8];
                uint32_t s0_code, s1_code;
                decompose_block_packed<kScalePolicy, kEnableResidualPass>(
                    block,
                    q0wb,
                    q1wb,
                    s0_code,
                    s1_code);

#pragma unroll
                for (uint32_t w = 0; w < kBlockSize / 8; ++w)
                {
                    q0[b * (kBlockSize / 8) + w] = q0wb[w];
                    q1[b * (kBlockSize / 8) + w] = q1wb[w];
                }
                // The scales of an atom pack into one word, in sub-block
                // order; a half-atom slot fills its two bytes of that word.
                sf0_word |= s0_code << (b * 8);
                sf1_word |= s1_code << (b * 8);
            }
        }

// ---- Store A0 / A1 as swizzled 16-byte groups ------------------------
        // A quarter-atom slot owns only 8 packed bytes; two such slots share
        // one 16-byte swizzle unit with disjoint 8-byte halves.
        if constexpr (kElemsPerSlot / 2 >= 16)
        {
#pragma unroll
            for (uint32_t g = 0; g < kElemsPerSlot / 2 / 16; ++g)
            {
                const uint32_t k_byte = k_base / 2 + g * 16;
                const uint32_t offset = swizzled_byte_offset<BLOCK_M, kSwizzleA0Mode>(m, k_byte);
                ptx::st_shared(smem_a0 + offset, q0[g * 4 + 0], q0[g * 4 + 1], q0[g * 4 + 2], q0[g * 4 + 3]);
                ptx::st_shared(smem_a1 + offset, q1[g * 4 + 0], q1[g * 4 + 1], q1[g * 4 + 2], q1[g * 4 + 3]);
            }
        }
        else
        {
            const uint32_t offset = swizzled_byte_offset<BLOCK_M, kSwizzleA0Mode>(m, k_base / 2);
            ptx::st_shared_v2(smem_a0 + offset, q0[0], q0[1]);
            ptx::st_shared_v2(smem_a1 + offset, q1[0], q1[1]);
        }

        // ---- Store the scale word, already in UTCCP order --------------------
        // Byte j of the atom word is block j; a half-atom slot owns bytes
        // [sub * kBlocksPerSlot, sub * kBlocksPerSlot + 2) and stores them as
        // one u16, so the two halves of a word never race.
        const uint32_t sf_offset =
            atom_idx * kSFAtomBytes + sf_atom_word_idx(m) * 4 + sub * kBlocksPerSlot;
        if constexpr (kThreadsPerAtom == 1)
        {
            ptx::st_shared(reinterpret_cast<const uint32_t*>(smem_sfa0 + sf_offset), sf0_word);
            ptx::st_shared(reinterpret_cast<const uint32_t*>(smem_sfa1 + sf_offset), sf1_word);
        }
        else if constexpr (kThreadsPerAtom == 2)
        {
            ptx::st_shared(reinterpret_cast<const uint16_t*>(smem_sfa0 + sf_offset),
                           static_cast<uint16_t>(sf0_word));
            ptx::st_shared(reinterpret_cast<const uint16_t*>(smem_sfa1 + sf_offset),
                           static_cast<uint16_t>(sf1_word));
        }
        else
        {
            ptx::st_shared(smem_sfa0 + sf_offset, static_cast<uint8_t>(sf0_word));
            ptx::st_shared(smem_sfa1 + sf_offset, static_cast<uint8_t>(sf1_word));
        }
    }
}

/// Split-phase variant: signal A0/SFA0 readiness before the residual pass.
///
/// The doc's pipeline lets MMA start `A0 x W` as soon as A0 is packed, while
/// the residual quantization still runs; this variant implements that split.
/// Phase 0 stores A0/SFA0 and calls `arrive0` (which must fence + arrive the
/// A0 barrier); phase 1 rebuilds `u` from a second read of the BF16 tile --
/// the packed q0 bytes and scale words stay in registers, so nothing else is
/// re-derived -- and stores A1/SFA1.
template<uint32_t BLOCK_M, uint32_t BLOCK_K, uint32_t kSwizzleAMode, uint32_t kSwizzleA0Mode, uint32_t kNumTransformThreads, ScalePolicy kScalePolicy, typename Arrive0>
CUTLASS_DEVICE void transform_a_tile_split(const __nv_bfloat16* smem_a,
                                           uint8_t*             smem_a0,
                                           uint8_t*             smem_a1,
                                           uint8_t*             smem_sfa0,
                                           uint8_t*             smem_sfa1,
                                           const uint32_t&      transform_tid,
                                           Arrive0&&            arrive0)
{
    constexpr uint32_t kAtomsPerRow = BLOCK_K / kKPerSFAtom;
    constexpr uint32_t kTotalBlocks = BLOCK_M * BLOCK_K / kBlockSize;
    NVFP4_STATIC_ASSERT(kTotalBlocks % kNumTransformThreads == 0,
                        "Transform threads must evenly divide the tile");
    constexpr uint32_t kBlocksPerSlot =
        kTotalBlocks / kNumTransformThreads < kSFPerAtom ? kTotalBlocks / kNumTransformThreads
                                                         : kSFPerAtom;
    NVFP4_STATIC_ASSERT(kBlocksPerSlot == kSFPerAtom or kBlocksPerSlot * 2 == kSFPerAtom,
                        "A transform slot must be a whole or half scale atom");
    constexpr uint32_t kThreadsPerAtom = kSFPerAtom / kBlocksPerSlot;
    constexpr uint32_t kElemsPerSlot   = kBlocksPerSlot * kBlockSize;
    constexpr uint32_t kNumSlots       = kTotalBlocks / kBlocksPerSlot;
    NVFP4_STATIC_ASSERT(kNumSlots == kNumTransformThreads,
                        "Split transform assumes exactly one slot per thread");

    constexpr uint32_t kSFAtomBytes = 128 * kSFPerAtom;
    NVFP4_STATIC_ASSERT(BLOCK_M <= 128 and 128 % BLOCK_M == 0,
                        "SF atom addressing assumes at most one 128-row atom");

    const uint32_t slot     = transform_tid;
    const uint32_t atom_seq = slot / kThreadsPerAtom;
    const uint32_t sub      = slot % kThreadsPerAtom;
    const uint32_t m        = atom_seq / kAtomsPerRow;
    const uint32_t atom_idx = atom_seq % kAtomsPerRow;
    const uint32_t k_base   = atom_idx * kKPerSFAtom + sub * kElemsPerSlot;

    auto load_x = [&](float(&x)[kElemsPerSlot]) {
#pragma unroll
        for (uint32_t g = 0; g < kElemsPerSlot * sizeof(__nv_bfloat16) / 16; ++g)
        {
            const uint32_t k_byte = (k_base + g * 8) * sizeof(__nv_bfloat16);
            const uint32_t offset = swizzled_byte_offset<BLOCK_M, kSwizzleAMode>(m, k_byte);
            const uint4    raw    = ptx::ld_shared(
                reinterpret_cast<const uint4*>(reinterpret_cast<const uint8_t*>(smem_a) + offset));
            const uint32_t words[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
            for (uint32_t w = 0; w < 4; ++w)
            {
                const __nv_bfloat162 pair = *reinterpret_cast<const __nv_bfloat162*>(&words[w]);
                const float2         vals = __bfloat1622float2(pair);
                x[g * 8 + w * 2]          = vals.x;
                x[g * 8 + w * 2 + 1]      = vals.y;
            }
        }
    };

    // ---- Phase 0: first pass only -----------------------------------------
    __align__(16) uint8_t q0[kElemsPerSlot / 2];
    uint32_t              sf0_word = 0;
    {
        float x[kElemsPerSlot];
        load_x(x);
#pragma unroll
        for (uint32_t b = 0; b < kBlocksPerSlot; ++b)
        {
            float amax = 0.0f;
#pragma unroll
            for (uint32_t i = 0; i < kBlockSize; ++i)
                amax = fmaxf(amax, fabsf(x[b * kBlockSize + i]));
            constexpr float kFloor = kScalePolicy == ScalePolicy::ResidualAmax
                                         ? 1.0f / 512.0f
                                         : kS0FloorDerivedDiv8;
            const uint32_t s0_code = ptx::cvt_e4m3(fmaxf(amax * kInvS0Div<kScalePolicy>, kFloor));
            const float    inv_s0  = math::fast_rcp(ptx::cvt_e4m3_to_f32(s0_code));
#pragma unroll
            for (uint32_t i = 0; i < kBlockSize / 2; ++i)
            {
                const float u_lo               = x[b * kBlockSize + 2 * i] * inv_s0;
                const float u_hi               = x[b * kBlockSize + 2 * i + 1] * inv_s0;
                q0[b * (kBlockSize / 2) + i]   = static_cast<uint8_t>(ptx::cvt_e2m1x2(u_hi, u_lo));
            }
            sf0_word |= s0_code << (b * 8);
        }
#pragma unroll
        for (uint32_t g = 0; g < kElemsPerSlot / 2 / 16; ++g)
        {
            const uint32_t k_byte = k_base / 2 + g * 16;
            const uint32_t offset = swizzled_byte_offset<BLOCK_M, kSwizzleA0Mode>(m, k_byte);
            ptx::st_shared(smem_a0 + offset,
                           *reinterpret_cast<const uint32_t*>(q0 + g * 16 + 0),
                           *reinterpret_cast<const uint32_t*>(q0 + g * 16 + 4),
                           *reinterpret_cast<const uint32_t*>(q0 + g * 16 + 8),
                           *reinterpret_cast<const uint32_t*>(q0 + g * 16 + 12));
        }
        const uint32_t sf_offset =
            atom_idx * kSFAtomBytes + sf_atom_word_idx(m) * 4 + sub * kBlocksPerSlot;
        if constexpr (kThreadsPerAtom == 1)
            ptx::st_shared(reinterpret_cast<const uint32_t*>(smem_sfa0 + sf_offset), sf0_word);
        else
            ptx::st_shared(reinterpret_cast<const uint16_t*>(smem_sfa0 + sf_offset),
                           static_cast<uint16_t>(sf0_word));
    }

    // A0/SFA0 are complete for this thread: fence + arrive happens in the
    // caller-provided hook so MMA's first pass can start.
    arrive0();

    // ---- Phase 1: residual pass -------------------------------------------
    {
        float x[kElemsPerSlot];
        load_x(x);
        __align__(16) uint8_t q1[kElemsPerSlot / 2];
        uint32_t              sf1_word = 0;
#pragma unroll
        for (uint32_t b = 0; b < kBlocksPerSlot; ++b)
        {
            const uint32_t s0_code = (sf0_word >> (b * 8)) & 0xffu;
            const float    s0f     = ptx::cvt_e4m3_to_f32(s0_code);
            const float    inv_s0  = math::fast_rcp(s0f);

            uint32_t s1_code;
            if constexpr (kScalePolicy != ScalePolicy::ResidualAmax)
            {
                constexpr float kRadix = kResidualRadix<kScalePolicy>;
                s1_code                = ptx::cvt_e4m3(s0f / kRadix);
#pragma unroll
                for (uint32_t i = 0; i < kBlockSize / 2; ++i)
                {
                    const uint32_t packed  = q0[b * (kBlockSize / 2) + i];
                    const uint32_t halves  = ptx::cvt_e2m1x2_to_f16x2(packed);
                    const float2   decoded = __half22float2(*reinterpret_cast<const __half2*>(&halves));
                    const float    r_lo = (x[b * kBlockSize + 2 * i] * inv_s0 - decoded.x) * kRadix;
                    const float    r_hi = (x[b * kBlockSize + 2 * i + 1] * inv_s0 - decoded.y) * kRadix;
                    q1[b * (kBlockSize / 2) + i] = static_cast<uint8_t>(ptx::cvt_e2m1x2(r_hi, r_lo));
                }
            }
            else
            {
                // The residual scale is absolute, so the residual leaves the
                // normalized domain here (r = (u - dec) * s0), exactly as the
                // merged path does.
                float r[kBlockSize], ramax = 0.0f;
#pragma unroll
                for (uint32_t i = 0; i < kBlockSize / 2; ++i)
                {
                    const uint32_t packed  = q0[b * (kBlockSize / 2) + i];
                    const uint32_t halves  = ptx::cvt_e2m1x2_to_f16x2(packed);
                    const float2   decoded = __half22float2(*reinterpret_cast<const __half2*>(&halves));
                    r[2 * i]     = (x[b * kBlockSize + 2 * i] * inv_s0 - decoded.x) * s0f;
                    r[2 * i + 1] = (x[b * kBlockSize + 2 * i + 1] * inv_s0 - decoded.y) * s0f;
                    ramax        = fmaxf(ramax, fmaxf(fabsf(r[2 * i]), fabsf(r[2 * i + 1])));
                }
                s1_code = ptx::cvt_e4m3(fmaxf(ramax * kInvE2M1Max, 1.0f / 512.0f));
                const float inv_s1 = math::fast_rcp(ptx::cvt_e4m3_to_f32(s1_code));
#pragma unroll
                for (uint32_t i = 0; i < kBlockSize / 2; ++i)
                    q1[b * (kBlockSize / 2) + i] = static_cast<uint8_t>(
                        ptx::cvt_e2m1x2(r[2 * i + 1] * inv_s1, r[2 * i] * inv_s1));
            }
            sf1_word |= s1_code << (b * 8);
        }
#pragma unroll
        for (uint32_t g = 0; g < kElemsPerSlot / 2 / 16; ++g)
        {
            const uint32_t k_byte = k_base / 2 + g * 16;
            const uint32_t offset = swizzled_byte_offset<BLOCK_M, kSwizzleA0Mode>(m, k_byte);
            ptx::st_shared(smem_a1 + offset,
                           *reinterpret_cast<const uint32_t*>(q1 + g * 16 + 0),
                           *reinterpret_cast<const uint32_t*>(q1 + g * 16 + 4),
                           *reinterpret_cast<const uint32_t*>(q1 + g * 16 + 8),
                           *reinterpret_cast<const uint32_t*>(q1 + g * 16 + 12));
        }
        const uint32_t sf_offset =
            atom_idx * kSFAtomBytes + sf_atom_word_idx(m) * 4 + sub * kBlocksPerSlot;
        if constexpr (kThreadsPerAtom == 1)
            ptx::st_shared(reinterpret_cast<const uint32_t*>(smem_sfa1 + sf_offset), sf1_word);
        else
            ptx::st_shared(reinterpret_cast<const uint16_t*>(smem_sfa1 + sf_offset),
                           static_cast<uint16_t>(sf1_word));
    }
}

/// Global-direct variant: read the BF16 tile straight from global memory.
///
/// The normal path TMA-loads BF16 A into shared memory only for the transform
/// to read it back once -- 32 KB per k-block per SM through the TMA ingress
/// port that profiling shows to be the prefill floor.  Reading via LDG instead
/// keeps the bytes on the L2->L1 load path (13% utilized), frees the A stage
/// buffer, and decouples the transform from the TMA arrival entirely.
///
/// `gmem_a` points at the tile's (row 0, k 0); rows stride by `lda` elements.
/// Same slot decomposition and output layouts as `transform_a_tile`.
template<uint32_t BLOCK_M, uint32_t BLOCK_K, uint32_t kSwizzleA0Mode, uint32_t kNumTransformThreads, ScalePolicy kScalePolicy, bool kEnableResidualPass = true>
CUTLASS_DEVICE void transform_a_tile_gmem(const __nv_bfloat16* __restrict__ gmem_a,
                                          const uint32_t& lda,
                                          uint8_t*        smem_a0,
                                          uint8_t*        smem_a1,
                                          uint8_t*        smem_sfa0,
                                          uint8_t*        smem_sfa1,
                                          const uint32_t& transform_tid)
{
    constexpr uint32_t kAtomsPerRow = BLOCK_K / kKPerSFAtom;
    constexpr uint32_t kTotalBlocks = BLOCK_M * BLOCK_K / kBlockSize;
    NVFP4_STATIC_ASSERT(kTotalBlocks % kNumTransformThreads == 0,
                        "Transform threads must evenly divide the tile");
    constexpr uint32_t kBlocksPerSlot =
        kTotalBlocks / kNumTransformThreads < kSFPerAtom ? kTotalBlocks / kNumTransformThreads
                                                         : kSFPerAtom;
    NVFP4_STATIC_ASSERT(kBlocksPerSlot == kSFPerAtom or kBlocksPerSlot * 2 == kSFPerAtom,
                        "A transform slot must be a whole or half scale atom");
    constexpr uint32_t kThreadsPerAtom = kSFPerAtom / kBlocksPerSlot;
    constexpr uint32_t kElemsPerSlot   = kBlocksPerSlot * kBlockSize;
    constexpr uint32_t kNumSlots       = kTotalBlocks / kBlocksPerSlot;
    constexpr uint32_t kSlotsPerThread = kNumSlots / kNumTransformThreads;

    constexpr uint32_t kSFAtomBytes = 128 * kSFPerAtom;
    NVFP4_STATIC_ASSERT(BLOCK_M <= 128 and 128 % BLOCK_M == 0,
                        "SF atom addressing assumes at most one 128-row atom");

#pragma unroll
    for (uint32_t t = 0; t < kSlotsPerThread; ++t)
    {
        const uint32_t slot     = t * kNumTransformThreads + transform_tid;
        const uint32_t atom_seq = slot / kThreadsPerAtom;
        const uint32_t sub      = slot % kThreadsPerAtom;
        const uint32_t m        = atom_seq / kAtomsPerRow;
        const uint32_t atom_idx = atom_seq % kAtomsPerRow;
        const uint32_t k_base   = atom_idx * kKPerSFAtom + sub * kElemsPerSlot;

        // ---- Load the slot's BF16 straight from global memory ----------------
        float x[kElemsPerSlot];
        const __nv_bfloat16* row_ptr = gmem_a + static_cast<uint64_t>(m) * lda + k_base;
#pragma unroll
        for (uint32_t g = 0; g < kElemsPerSlot * sizeof(__nv_bfloat16) / 16; ++g)
        {
            const uint4    raw      = __ldg(reinterpret_cast<const uint4*>(row_ptr + g * 8));
            const uint32_t words[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
            for (uint32_t w = 0; w < 4; ++w)
            {
                const __nv_bfloat162 pair = *reinterpret_cast<const __nv_bfloat162*>(&words[w]);
                const float2         vals = __bfloat1622float2(pair);
                x[g * 8 + w * 2]          = vals.x;
                x[g * 8 + w * 2 + 1]      = vals.y;
            }
        }

        // ---- Algorithm 1 on each of the slot's 16-element blocks -------------
        __align__(16) uint8_t q0[kElemsPerSlot / 2];
        __align__(16) uint8_t q1[kElemsPerSlot / 2];
        uint32_t              sf0_word = 0, sf1_word = 0;
#pragma unroll
        for (uint32_t b = 0; b < kBlocksPerSlot; ++b)
        {
            float block[kBlockSize];
#pragma unroll
            for (uint32_t i = 0; i < kBlockSize; ++i)
                block[i] = x[b * kBlockSize + i];

            uint8_t  q0_bytes[kBlockSize / 2], q1_bytes[kBlockSize / 2];
            uint32_t s0_code, s1_code;
            decompose_block<kScalePolicy, kEnableResidualPass>(
                block,
                q0_bytes,
                q1_bytes,
                s0_code,
                s1_code);

#pragma unroll
            for (uint32_t i = 0; i < kBlockSize / 2; ++i)
            {
                q0[b * (kBlockSize / 2) + i] = q0_bytes[i];
                q1[b * (kBlockSize / 2) + i] = q1_bytes[i];
            }
            sf0_word |= s0_code << (b * 8);
            sf1_word |= s1_code << (b * 8);
        }

// ---- Store A0 / A1 as swizzled 16-byte groups ------------------------
#pragma unroll
        for (uint32_t g = 0; g < kElemsPerSlot / 2 / 16; ++g)
        {
            const uint32_t k_byte = (atom_idx * kKPerSFAtom + sub * kElemsPerSlot) / 2 + g * 16;
            const uint32_t offset = swizzled_byte_offset<BLOCK_M, kSwizzleA0Mode>(m, k_byte);
            ptx::st_shared(smem_a0 + offset,
                           *reinterpret_cast<const uint32_t*>(q0 + g * 16 + 0),
                           *reinterpret_cast<const uint32_t*>(q0 + g * 16 + 4),
                           *reinterpret_cast<const uint32_t*>(q0 + g * 16 + 8),
                           *reinterpret_cast<const uint32_t*>(q0 + g * 16 + 12));
            ptx::st_shared(smem_a1 + offset,
                           *reinterpret_cast<const uint32_t*>(q1 + g * 16 + 0),
                           *reinterpret_cast<const uint32_t*>(q1 + g * 16 + 4),
                           *reinterpret_cast<const uint32_t*>(q1 + g * 16 + 8),
                           *reinterpret_cast<const uint32_t*>(q1 + g * 16 + 12));
        }

        // ---- Store the scale word, already in UTCCP order --------------------
        const uint32_t sf_offset =
            atom_idx * kSFAtomBytes + sf_atom_word_idx(m) * 4 + sub * kBlocksPerSlot;
        if constexpr (kThreadsPerAtom == 1)
        {
            ptx::st_shared(reinterpret_cast<const uint32_t*>(smem_sfa0 + sf_offset), sf0_word);
            ptx::st_shared(reinterpret_cast<const uint32_t*>(smem_sfa1 + sf_offset), sf1_word);
        }
        else
        {
            ptx::st_shared(reinterpret_cast<const uint16_t*>(smem_sfa0 + sf_offset),
                           static_cast<uint16_t>(sf0_word));
            ptx::st_shared(reinterpret_cast<const uint16_t*>(smem_sfa1 + sf_offset),
                           static_cast<uint16_t>(sf1_word));
        }
    }
}

}   // namespace nvfp4_gemm::transform

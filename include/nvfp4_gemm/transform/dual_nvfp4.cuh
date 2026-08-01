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
};

/// Residual radix per policy: how far the normalized residual is scaled up
/// before its own E2M1 quantization, with s1 derived as s0 / radix.
template<ScalePolicy kScalePolicy>
constexpr float kResidualRadix = kScalePolicy == ScalePolicy::DerivedDiv4 ? 4.0f : 8.0f;

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
    s0_code                = ptx::cvt_e4m3(fmaxf(amax * kInvE2M1Max, kFloor));

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
template<uint32_t BLOCK_M, uint32_t BLOCK_K, uint32_t kSwizzleAMode, uint32_t kSwizzleA0Mode, uint32_t kNumTransformThreads, ScalePolicy kScalePolicy, bool kEnableResidualPass = true>
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
    NVFP4_STATIC_ASSERT(kBlocksPerSlot == kSFPerAtom or kBlocksPerSlot * 2 == kSFPerAtom,
                        "A transform slot must be a whole or half scale atom");
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

        // ---- Load the slot's BF16 as swizzled 16-byte groups -----------------
        float x[kElemsPerSlot];
#pragma unroll
        for (uint32_t g = 0; g < kElemsPerSlot * sizeof(__nv_bfloat16) / 16; ++g)
        {
            const uint32_t k_byte = (k_base + g * 8) * sizeof(__nv_bfloat16);
            const uint32_t offset = swizzled_byte_offset<BLOCK_M, kSwizzleAMode>(m, k_byte);
            const uint4    raw    = ptx::ld_shared(
                reinterpret_cast<const uint4*>(reinterpret_cast<const uint8_t*>(smem_a) + offset));

            // Each uint32 holds two BF16; widen both.
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
        // 16-byte aligned: the stores below reinterpret these as uint4 groups.
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
            // The scales of an atom pack into one word, in sub-block order; a
            // half-atom slot fills its two bytes of that word.
            sf0_word |= s0_code << (b * 8);
            sf1_word |= s1_code << (b * 8);
        }

// ---- Store A0 / A1 as swizzled 16-byte groups ------------------------
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
            ptx::st_shared(smem_a1 + offset,
                           *reinterpret_cast<const uint32_t*>(q1 + g * 16 + 0),
                           *reinterpret_cast<const uint32_t*>(q1 + g * 16 + 4),
                           *reinterpret_cast<const uint32_t*>(q1 + g * 16 + 8),
                           *reinterpret_cast<const uint32_t*>(q1 + g * 16 + 12));
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
        else
        {
            ptx::st_shared(reinterpret_cast<const uint16_t*>(smem_sfa0 + sf_offset),
                           static_cast<uint16_t>(sf0_word));
            ptx::st_shared(reinterpret_cast<const uint16_t*>(smem_sfa1 + sf_offset),
                           static_cast<uint16_t>(sf1_word));
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
            const uint32_t s0_code = ptx::cvt_e4m3(fmaxf(amax * kInvE2M1Max, kFloor));
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

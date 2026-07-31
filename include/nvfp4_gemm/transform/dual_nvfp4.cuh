#pragma once

#include <cuda_bf16.h>
#include <cuda/std/cstdint>

#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/ptx/ld_st.cuh>

#include "../ptx/tcgen05_nvfp4.cuh"

namespace nvfp4_gemm::transform {

/// NVFP4 block granularity along K.
constexpr uint32_t kBlockSize = 16;
/// One UMMA-K step of `kind::mxf4nvf4.block16` consumes 64 K elements, which is
/// four scale sub-blocks -- exactly one 128x4 scale-factor atom.
constexpr uint32_t kKPerSFAtom = 64;
constexpr uint32_t kSFPerAtom = kKPerSFAtom / kBlockSize;  // 4

/// Reciprocal of the E2M1 max finite magnitude.
constexpr float kInvE2M1Max = 1.0f / 6.0f;

/// `derived_div8` floors `s0` at 8 * 2^-9 == 2^-6 so that `s1 = s0 / 8` cannot
/// underflow past the smallest positive E4M3 subnormal.  A zero block therefore
/// yields q0 = q1 = 0 with finite scales instead of inf/NaN.
constexpr float kS0FloorDerivedDiv8 = 0.015625f;  // 2^-6

enum class ScalePolicy {
    /// s1 = Q_e4m3(s0 * 2^-3), no second reduction.  Kernel default.
    DerivedDiv8 = 0,
    /// s1 = Q_e4m3(amax(residual) / 6).  More accurate, one extra reduction.
    ResidualAmax = 1,
};

/// Byte offset of element (m, k) inside a K-major, `kSwizzleMode`-swizzled tile.
///
/// TMA splits a row into `BLOCK_K * sizeof(T) / kSwizzleMode` independent
/// swizzle atoms, each stored as its own [BLOCK_MN, atom] sub-tile.  Inside an
/// atom the 16-byte bank group index is XORed with `m % (kSwizzleMode / 16)`.
/// The transform warps read A with plain LDS, so they have to reproduce exactly
/// this mapping -- there is no TMA to do it for them.
template <uint32_t BLOCK_MN, uint32_t kSwizzleMode>
CUTLASS_DEVICE uint32_t swizzled_byte_offset(const uint32_t& m, const uint32_t& k_byte) {
    DG_STATIC_ASSERT(kSwizzleMode == 32 or kSwizzleMode == 64 or kSwizzleMode == 128,
                     "Unsupported swizzle mode");
    constexpr uint32_t kNumBankGroups = kSwizzleMode / 16;
    // Bytes of one row inside a single swizzle atom.
    constexpr uint32_t kAtomRowBytes = kSwizzleMode;

    const uint32_t atom_idx = k_byte / kAtomRowBytes;
    const uint32_t byte_in_atom = k_byte % kAtomRowBytes;
    const uint32_t bank_group = byte_in_atom / 16;
    const uint32_t byte_in_group = byte_in_atom % 16;

    const uint32_t swizzled_group = bank_group ^ (m % kNumBankGroups);
    return atom_idx * (BLOCK_MN * kAtomRowBytes) +     // atom sub-tile
           m * kAtomRowBytes +                          // row inside the atom
           swizzled_group * 16 + byte_in_group;         // swizzled position
}

/// uint32 index of row `m` inside a 128x4 scale-factor atom, in the layout UTCCP
/// expects.
///
/// DeepGEMM's UTCCP transposer produces `new[j] = old[(j % 4) * 32 + j / 4]`,
/// where `old` is indexed by row.  We generate the scales ourselves, so we write
/// them pre-transposed and skip that warp shuffle entirely for SFA.  Inverting
/// the permutation gives `j = (m % 32) * 4 + m / 32`.
CUTLASS_DEVICE uint32_t sf_atom_word_idx(const uint32_t& m) {
    return (m % 32) * 4 + (m / 32);
}

/// Result of decomposing one 16-element block.
struct BlockCodes {
    uint32_t q0_packed;  // 8 bytes worth of nibbles, 2 elements per byte
    uint32_t q1_packed;
    uint32_t s0_code;    // E4M3
    uint32_t s1_code;    // E4M3
};

/// Algorithm 1: decompose one 16-element BF16 block into two NVFP4 passes.
///
/// `x` holds the 16 values already widened to FP32.  `q0_bytes` / `q1_bytes`
/// receive 8 packed bytes each.  Everything after the amax stays in the
/// *normalized* domain: the residual is `(x * rcp(s0) - dec(q0)) * 8`, never
/// `(x - dec(q0) * s0) / s1`, which saves a dequant multiply per element.
template <ScalePolicy kScalePolicy>
CUTLASS_DEVICE void decompose_block(const float (&x)[kBlockSize],
                                    uint8_t (&q0_bytes)[kBlockSize / 2],
                                    uint8_t (&q1_bytes)[kBlockSize / 2],
                                    uint32_t& s0_code, uint32_t& s1_code) {
    // 1: block amax.
    float amax = 0.0f;
    #pragma unroll
    for (uint32_t i = 0; i < kBlockSize; ++ i)
        amax = fmaxf(amax, fabsf(x[i]));

    // 2: s0 = Q_e4m3(max(amax / 6, floor)).  The floor keeps s1 representable.
    constexpr float kFloor = kScalePolicy == ScalePolicy::DerivedDiv8
                           ? kS0FloorDerivedDiv8
                           : 1.0f / 512.0f;  // 2^-9, smallest positive E4M3
    s0_code = ptx::cvt_e4m3(fmaxf(amax * kInvE2M1Max, kFloor));

    // The reciprocal has to be built from the *rounded* s0, otherwise the kernel
    // and the reference disagree on every block whose amax/6 is not exactly
    // representable in E4M3.
    const float s0f = ptx::cvt_e4m3_to_f32(s0_code);
    const float inv_s0 = deep_gemm::math::fast_rcp(s0f);

    // 3: q0 = Q_e2m1(x * rcp(s0)), two elements per hardware instruction.
    float u[kBlockSize];
    #pragma unroll
    for (uint32_t i = 0; i < kBlockSize; ++ i)
        u[i] = x[i] * inv_s0;

    float q0_dec[kBlockSize];
    #pragma unroll
    for (uint32_t i = 0; i < kBlockSize / 2; ++ i) {
        // Element 2i goes in the low nibble, 2i+1 in the high nibble, matching
        // the E2M1 packing order UMMA reads back.
        const uint32_t packed = ptx::cvt_e2m1x2(u[2 * i + 1], u[2 * i]);
        q0_bytes[i] = static_cast<uint8_t>(packed);

        const uint32_t halves = ptx::cvt_e2m1x2_to_f16x2(packed);
        const float2 decoded = __half22float2(*reinterpret_cast<const __half2*>(&halves));
        q0_dec[2 * i] = decoded.x;
        q0_dec[2 * i + 1] = decoded.y;
    }

    // 5/6: residual pass.
    if constexpr (kScalePolicy == ScalePolicy::DerivedDiv8) {
        s1_code = ptx::cvt_e4m3(s0f * 0.125f);
        #pragma unroll
        for (uint32_t i = 0; i < kBlockSize / 2; ++ i) {
            const float r_lo = (u[2 * i] - q0_dec[2 * i]) * 8.0f;
            const float r_hi = (u[2 * i + 1] - q0_dec[2 * i + 1]) * 8.0f;
            q1_bytes[i] = static_cast<uint8_t>(ptx::cvt_e2m1x2(r_hi, r_lo));
        }
    } else {
        // Residual amax costs a second 16-way reduction but picks the tightest
        // possible s1, which matters when a block's residual is far below s0/8.
        float r[kBlockSize], r_amax = 0.0f;
        #pragma unroll
        for (uint32_t i = 0; i < kBlockSize; ++ i) {
            r[i] = (u[i] - q0_dec[i]) * s0f;
            r_amax = fmaxf(r_amax, fabsf(r[i]));
        }
        s1_code = ptx::cvt_e4m3(fmaxf(r_amax * kInvE2M1Max, 1.0f / 512.0f));
        const float inv_s1 = deep_gemm::math::fast_rcp(ptx::cvt_e4m3_to_f32(s1_code));
        #pragma unroll
        for (uint32_t i = 0; i < kBlockSize / 2; ++ i)
            q1_bytes[i] = static_cast<uint8_t>(
                ptx::cvt_e2m1x2(r[2 * i + 1] * inv_s1, r[2 * i] * inv_s1));
    }
}

/// Transform one A tile: BF16 [BLOCK_M, BLOCK_K] -> (A0, SFA0) + (A1, SFA1).
///
/// Thread mapping: every thread owns one full 128x4 scale atom's worth of K,
/// i.e. `kKPerSFAtom = 64` contiguous elements of one row.  That choice matters:
///
///   * the four block scales of an atom become a *single* 4-byte SFA store
///     instead of four 1-byte stores,
///   * no lane holds a partial block, so the amax needs no `shfl` at all, and
///   * no scale is ever computed twice.
///
/// The doc's kernel instead gives each lane half a block and merges the amax
/// with a butterfly shuffle, which makes both lanes of a pair compute the same
/// `s0`, `rcp(s0)` and `s1`, and store the same two scale bytes to the same
/// address.  Owning whole atoms removes that duplicated scalar work outright
/// rather than trying to predicate it away.
template <uint32_t BLOCK_M, uint32_t BLOCK_K,
          uint32_t kSwizzleAMode, uint32_t kSwizzleA0Mode,
          uint32_t kNumTransformThreads,
          ScalePolicy kScalePolicy>
CUTLASS_DEVICE void transform_a_tile(const __nv_bfloat16* smem_a,
                                     uint8_t* smem_a0, uint8_t* smem_a1,
                                     uint8_t* smem_sfa0, uint8_t* smem_sfa1,
                                     const uint32_t& transform_tid) {
    constexpr uint32_t kAtomsPerRow = BLOCK_K / kKPerSFAtom;
    constexpr uint32_t kNumTasks = BLOCK_M * kAtomsPerRow;
    DG_STATIC_ASSERT(BLOCK_K % kKPerSFAtom == 0, "Block K must cover whole SF atoms");
    DG_STATIC_ASSERT(kNumTasks % kNumTransformThreads == 0,
                     "Transform threads must evenly divide the tile");
    constexpr uint32_t kTasksPerThread = kNumTasks / kNumTransformThreads;

    // Byte size of one 128-row scale atom: 128 rows x 4 E4M3 bytes.
    constexpr uint32_t kSFAtomBytes = 128 * kSFPerAtom;
    DG_STATIC_ASSERT(BLOCK_M == 128, "SF atom addressing assumes exactly 128 rows");

    #pragma unroll
    for (uint32_t t = 0; t < kTasksPerThread; ++ t) {
        const uint32_t task = t * kNumTransformThreads + transform_tid;
        const uint32_t m = task / kAtomsPerRow;
        const uint32_t atom_idx = task % kAtomsPerRow;
        const uint32_t k_base = atom_idx * kKPerSFAtom;

        // ---- Load 64 BF16 (128 bytes) as eight swizzled 16-byte groups -------
        float x[kKPerSFAtom];
        #pragma unroll
        for (uint32_t g = 0; g < kKPerSFAtom * sizeof(__nv_bfloat16) / 16; ++ g) {
            const uint32_t k_byte = (k_base + g * 8) * sizeof(__nv_bfloat16);
            const uint32_t offset = swizzled_byte_offset<BLOCK_M, kSwizzleAMode>(m, k_byte);
            const uint4 raw = deep_gemm::ptx::ld_shared(
                reinterpret_cast<const uint4*>(reinterpret_cast<const uint8_t*>(smem_a) + offset));

            // Each uint32 holds two BF16; widen both.
            const uint32_t words[4] = {raw.x, raw.y, raw.z, raw.w};
            #pragma unroll
            for (uint32_t w = 0; w < 4; ++ w) {
                const __nv_bfloat162 pair = *reinterpret_cast<const __nv_bfloat162*>(&words[w]);
                const float2 vals = __bfloat1622float2(pair);
                x[g * 8 + w * 2] = vals.x;
                x[g * 8 + w * 2 + 1] = vals.y;
            }
        }

        // ---- Algorithm 1 on each of the atom's four 16-element blocks --------
        // 16-byte aligned: the stores below reinterpret these as uint4 groups.
        __align__(16) uint8_t q0[kKPerSFAtom / 2];
        __align__(16) uint8_t q1[kKPerSFAtom / 2];
        uint32_t sf0_word = 0, sf1_word = 0;
        #pragma unroll
        for (uint32_t b = 0; b < kSFPerAtom; ++ b) {
            float block[kBlockSize];
            #pragma unroll
            for (uint32_t i = 0; i < kBlockSize; ++ i)
                block[i] = x[b * kBlockSize + i];

            uint8_t q0_bytes[kBlockSize / 2], q1_bytes[kBlockSize / 2];
            uint32_t s0_code, s1_code;
            decompose_block<kScalePolicy>(block, q0_bytes, q1_bytes, s0_code, s1_code);

            #pragma unroll
            for (uint32_t i = 0; i < kBlockSize / 2; ++ i) {
                q0[b * (kBlockSize / 2) + i] = q0_bytes[i];
                q1[b * (kBlockSize / 2) + i] = q1_bytes[i];
            }
            // The four scales of an atom pack into one word, in sub-block order.
            sf0_word |= s0_code << (b * 8);
            sf1_word |= s1_code << (b * 8);
        }

        // ---- Store A0 / A1 as swizzled 16-byte groups ------------------------
        #pragma unroll
        for (uint32_t g = 0; g < kKPerSFAtom / 2 / 16; ++ g) {
            const uint32_t k_byte = k_base / 2 + g * 16;
            const uint32_t offset = swizzled_byte_offset<BLOCK_M, kSwizzleA0Mode>(m, k_byte);
            deep_gemm::ptx::st_shared(smem_a0 + offset,
                                      *reinterpret_cast<const uint32_t*>(q0 + g * 16 + 0),
                                      *reinterpret_cast<const uint32_t*>(q0 + g * 16 + 4),
                                      *reinterpret_cast<const uint32_t*>(q0 + g * 16 + 8),
                                      *reinterpret_cast<const uint32_t*>(q0 + g * 16 + 12));
            deep_gemm::ptx::st_shared(smem_a1 + offset,
                                      *reinterpret_cast<const uint32_t*>(q1 + g * 16 + 0),
                                      *reinterpret_cast<const uint32_t*>(q1 + g * 16 + 4),
                                      *reinterpret_cast<const uint32_t*>(q1 + g * 16 + 8),
                                      *reinterpret_cast<const uint32_t*>(q1 + g * 16 + 12));
        }

        // ---- Store the scale word, already in UTCCP order --------------------
        const uint32_t sf_offset = atom_idx * kSFAtomBytes + sf_atom_word_idx(m) * 4;
        deep_gemm::ptx::st_shared(reinterpret_cast<const uint32_t*>(smem_sfa0 + sf_offset), sf0_word);
        deep_gemm::ptx::st_shared(reinterpret_cast<const uint32_t*>(smem_sfa1 + sf_offset), sf1_word);
    }
}

} // namespace nvfp4_gemm::transform

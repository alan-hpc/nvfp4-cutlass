#pragma once

#include <cute/atom/copy_traits_sm100.hpp>

#include <nvfp4_gemm/common/math.cuh>
#include <nvfp4_gemm/common/types.cuh>
#include <nvfp4_gemm/common/utils.cuh>
#include <nvfp4_gemm/ptx/ld_st.cuh>
#include <nvfp4_gemm/ptx/tcgen05.cuh>

namespace nvfp4_gemm::epilogue {

using utils::PatternVisitor;

/// Swap-AB epilogue with the per-expert FP32 weight scale.
///
/// Derived from DeepGEMM (MIT, Copyright (c) 2025 DeepSeek),
/// `deep_gemm/epilogue/sm100_store_cd_swap_ab.cuh`, restricted to BF16 output
/// and extended with the `G_W` multiply the dual-NVFP4 kernel applies before
/// the cast (see `sm100_store_cd_gscale`).
///
/// In swapped coordinates the accumulator holds (128 weight rows) x
/// (BLOCK_N token columns): TMEM rows are the *model* dimension and columns
/// are tokens.  The store must therefore transpose -- `tcgen05.ld .16x256b`
/// hands each lane pair a 2x8 fragment and `stmatrix.trans` lands it
/// row-major in shared memory, so the TMA store writes D as (tokens, model_n)
/// directly.
///
/// `STORE_BLOCK_M` counts *token columns* per store stage; `STORE_BLOCK_N` is
/// pinned at 128 because a full warpgroup is needed to read all TMEM rows.
template<uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t STORE_BLOCK_M, uint32_t STORE_BLOCK_N, uint32_t kSwizzleCDMode, uint32_t kNumTMAStoreStages, uint32_t kNumUMMAStoreThreads, typename cd_dtype_t, typename pattern_cd_t>
CUTLASS_DEVICE void
sm100_store_cd_swap_ab_gscale(const PatternVisitor<pattern_cd_t>& smem_cd, uint32_t& tma_stage_idx, const uint32_t& tmem_base_addr, const uint32_t& base_m_idx, const uint32_t& base_n_idx, const float& global_scale, const uint32_t& epilogue_warp_idx, const uint32_t& subpartition_idx, const uint32_t& lane_idx, const cutlass::arch::ClusterTransactionBarrier* tmem_empty_barrier, const cute::TmaDescriptor& tensor_map_cd)
{
    NVFP4_STATIC_ASSERT(STORE_BLOCK_N == 128, "STORE_BLOCK_N must be 128 to match TMEM rows");
    NVFP4_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>,
                        "Swap-AB epilogue only stores BF16");

    constexpr uint32_t STORE_BLOCK_N_ATOM  = kSwizzleCDMode / sizeof(cd_dtype_t);
    constexpr uint32_t kNumBankGroupBytes  = 16;
    constexpr uint32_t kNumSwizzleAtomRows = 8;
    NVFP4_STATIC_ASSERT(kSwizzleCDMode == 128, "TMA D must be 128 B swizzled");
    NVFP4_STATIC_ASSERT(BLOCK_N % STORE_BLOCK_M == 0, "Invalid store tiling");
    NVFP4_STATIC_ASSERT(STORE_BLOCK_M % kNumSwizzleAtomRows == 0, "Invalid swizzling");
    NVFP4_STATIC_ASSERT(STORE_BLOCK_N % STORE_BLOCK_N_ATOM == 0, "Invalid swizzling");

    auto advance_store_pipeline = [&]() {
        tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages;
    };

    // Iterate over token-column blocks of the accumulator.
    constexpr uint32_t kNumStores = BLOCK_N / STORE_BLOCK_M;
#pragma unroll
    for (uint32_t s = 0; s < kNumStores; ++s, advance_store_pipeline())
    {
        if (epilogue_warp_idx == 0)
            cute::tma_store_wait<kNumTMAStoreStages - 1>();
        cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

#pragma unroll
        for (uint32_t i = 0; i < STORE_BLOCK_M / kNumSwizzleAtomRows; ++i)
        {
            const uint32_t tmem_addr = tmem_base_addr + s * STORE_BLOCK_M + i * kNumSwizzleAtomRows;
            uint32_t       values[kNumSwizzleAtomRows];

            NVFP4_STATIC_ASSERT(STORE_BLOCK_N_ATOM % 32 == 0, "Invalid block sizes");
            constexpr uint32_t kNumWarpsPerAtom = STORE_BLOCK_N_ATOM / 32;
            // `tcgen05.ld` reads the 32 TMEM rows of the warp's *hardware*
            // sub-partition (absolute warp index mod 4), so the destination
            // model-column block must follow that, not the warp's rank among
            // the epilogue warps (see the identical fix in the normal
            // epilogue).
            const uint32_t outer_atom_offset =
                (subpartition_idx / kNumWarpsPerAtom) * STORE_BLOCK_M * kSwizzleCDMode;
            const uint32_t inner_atom_offset = i * kNumSwizzleAtomRows * kSwizzleCDMode;
            auto           smem_base_ptr     = reinterpret_cast<uint8_t*>(smem_cd[tma_stage_idx]) +
                                 outer_atom_offset + inner_atom_offset;

            // `.16x256b` twice (lanes 0-15 / 16-31) to satisfy the STSM layout.
            cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr,
                                                   values[0],
                                                   values[1],
                                                   values[2],
                                                   values[3]);
            cute::SM100_TMEM_LOAD_16dp256b1x::copy(tmem_addr | 0x00100000,
                                                   values[4],
                                                   values[5],
                                                   values[6],
                                                   values[7]);
            cutlass::arch::fence_view_async_tmem_load();

            // Apply G_W in FP32, then pack pairs into BF16 and store transposed.
            float scaled[kNumSwizzleAtomRows];
#pragma unroll
            for (uint32_t j = 0; j < kNumSwizzleAtomRows; ++j)
                scaled[j] = *reinterpret_cast<float*>(&values[j]) * global_scale;

            const uint32_t row      = lane_idx % 8;
            const uint32_t col      = (subpartition_idx % 2) * 4 + lane_idx / 8;
            auto           smem_ptr = smem_base_ptr + row * (kNumBankGroupBytes * 8) +
                             (col ^ row) * kNumBankGroupBytes;

            ptx::SM90_U32x4_STSM_T::copy(math::cast_into_bf16_and_pack(scaled[0], scaled[1]),
                                         math::cast_into_bf16_and_pack(scaled[2], scaled[3]),
                                         math::cast_into_bf16_and_pack(scaled[4], scaled[5]),
                                         math::cast_into_bf16_and_pack(scaled[6], scaled[7]),
                                         smem_ptr);
        }

        // Release the accumulator as soon as the last stage's loads are done.
        if (s == kNumStores - 1)
        {
            ptx::tcgen05_before_thread_sync();
            tmem_empty_barrier->arrive();
        }

        cute::tma_store_fence();
        cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);
        if (epilogue_warp_idx == 0 and cute::elect_one_sync())
        {
#pragma unroll
            for (uint32_t i = 0; i < STORE_BLOCK_N / STORE_BLOCK_N_ATOM; ++i)
            {
                auto smem_ptr = reinterpret_cast<uint8_t*>(smem_cd[tma_stage_idx]) +
                                i * STORE_BLOCK_M * kSwizzleCDMode;
                // D is (tokens, model_n): the token index is the outer (m)
                // coordinate, the model dimension the inner (n) one.
                const uint32_t m_idx = base_n_idx + s * STORE_BLOCK_M;
                const uint32_t n_idx = base_m_idx + i * STORE_BLOCK_N_ATOM;
                cute::SM90_TMA_STORE_2D::copy(&tensor_map_cd, smem_ptr, n_idx, m_idx);
            }
            cute::tma_store_arrive();
        }
        __syncwarp();
    }
}

}   // namespace nvfp4_gemm::epilogue

#pragma once

#include <cute/atom/copy_traits_sm100.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>

namespace nvfp4_gemm::epilogue {

using deep_gemm::GemmType;
using deep_gemm::utils::PatternVisitor;

/// Epilogue store with a per-expert FP32 global weight scale.
///
/// This is DeepGEMM's `sm100_store_cd` with one addition: the accumulator is
/// multiplied by `G_W` before the BF16 cast.  NVFP4 weights carry a per-tensor
/// FP32 scale on top of their E4M3 block scales, and the mainloop deliberately
/// does not fold it in -- it is one multiply per output element here versus one
/// per MMA operand in the mainloop.
///
/// The activation side needs no such multiply: the recipe this kernel targets
/// pins the activation `constant_amax` to 2688 = 448 x 6, so `G_A == 1` and the
/// term vanishes at compile time.
template<uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t STORE_BLOCK_M, uint32_t STORE_BLOCK_N, uint32_t kSwizzleCDMode, uint32_t kNumTMAStoreStages, uint32_t kNumUMMAStoreThreads, GemmType kGemmType, typename cd_dtype_t, typename pattern_cd_t>
CUTLASS_DEVICE void
sm100_store_cd_gscale(const PatternVisitor<pattern_cd_t>& smem_cd, uint32_t& tma_stage_idx, const uint32_t& tmem_base_addr, const uint32_t& base_m_idx, const uint32_t& base_n_idx, const uint32_t& batch_idx, const float& global_scale, const uint32_t& epilogue_warp_idx, const uint32_t& lane_idx, const cutlass::arch::ClusterTransactionBarrier* tmem_empty_barrier, const cute::TmaDescriptor& tensor_map_cd)
{
    constexpr uint32_t kNumBankGroupBytes    = 16;
    constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(cd_dtype_t);
    DG_STATIC_ASSERT(kSwizzleCDMode > 0, "TMA D must be swizzled");
    DG_STATIC_ASSERT(STORE_BLOCK_N % kNumElemsPerBankGroup == 0, "Invalid swizzling");
    DG_STATIC_ASSERT(BLOCK_M % STORE_BLOCK_M == 0, "Invalid block sizes");
    DG_STATIC_ASSERT(BLOCK_N % STORE_BLOCK_N == 0, "Invalid block sizes");
    DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>,
                     "Dual-NVFP4 GEMM only stores BF16");

    auto advance_store_pipeline = [&]() {
        tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages;
    };

    constexpr auto kNumMWaves = BLOCK_M / STORE_BLOCK_M;
#pragma unroll
    for (uint32_t w = 0; w < kNumMWaves; ++w)
    {
        constexpr uint32_t kNumStores = BLOCK_N / STORE_BLOCK_N;
#pragma unroll
        for (uint32_t s = 0; s < kNumStores; ++s, advance_store_pipeline())
        {
            auto smem_base_ptr = reinterpret_cast<uint8_t*>(smem_cd[tma_stage_idx]);

            if (epilogue_warp_idx == 0)
                cute::tma_store_wait<kNumTMAStoreStages - 1>();
            cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

            const auto m_idx = base_m_idx + w * STORE_BLOCK_M;
            const auto n_idx = base_n_idx + s * STORE_BLOCK_N;

#pragma unroll
            for (uint32_t i = 0; i < STORE_BLOCK_N / kNumElemsPerBankGroup; ++i)
            {
                auto           bank_group_index = i + lane_idx * (kSwizzleCDMode / kNumBankGroupBytes);
                constexpr bool kHasShortcut     = (kSwizzleCDMode / kNumBankGroupBytes) == 8;
                auto           row              = kHasShortcut ? (i / 8 + lane_idx) : (bank_group_index / 8);
                auto           col              = kHasShortcut ? (i) : (bank_group_index % 8);
                col ^= row % (kSwizzleCDMode / 16);

                uint32_t tmem_addr = tmem_base_addr + w * BLOCK_N +
                                     s * STORE_BLOCK_N + i * kNumElemsPerBankGroup;
                auto     smem_ptr  = smem_base_ptr +
                                     epilogue_warp_idx * 32 * kSwizzleCDMode +
                                     row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;

                uint32_t values[kNumElemsPerBankGroup];
                DG_STATIC_ASSERT(kNumElemsPerBankGroup == 8, "Invalid type");
                cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_addr,
                                                      values[0],
                                                      values[1],
                                                      values[2],
                                                      values[3],
                                                      values[4],
                                                      values[5],
                                                      values[6],
                                                      values[7]);
                cutlass::arch::fence_view_async_tmem_load();

                // Apply G_W in FP32, then pack pairs into BF16.
                float scaled[kNumElemsPerBankGroup];
#pragma unroll
                for (uint32_t j = 0; j < kNumElemsPerBankGroup; ++j)
                    scaled[j] = *reinterpret_cast<float*>(&values[j]) * global_scale;

                deep_gemm::ptx::st_shared(
                    smem_ptr,
                    deep_gemm::math::cast_into_bf16_and_pack(scaled[0], scaled[1]),
                    deep_gemm::math::cast_into_bf16_and_pack(scaled[2], scaled[3]),
                    deep_gemm::math::cast_into_bf16_and_pack(scaled[4], scaled[5]),
                    deep_gemm::math::cast_into_bf16_and_pack(scaled[6], scaled[7]));
            }

            // Release the accumulator as early as possible so the MMA warp can
            // start the next block while this store drains.
            if (w == kNumMWaves - 1 and s == kNumStores - 1)
            {
                deep_gemm::ptx::tcgen05_before_thread_sync();
                tmem_empty_barrier->arrive(0u);
            }

            cute::tma_store_fence();
            cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);
            if (epilogue_warp_idx == 0 and cute::elect_one_sync())
            {
                if constexpr (kGemmType == GemmType::MGroupedMasked)
                {
                    cute::SM90_TMA_STORE_3D::copy(&tensor_map_cd, smem_base_ptr, n_idx, m_idx, batch_idx);
                }
                else
                {
                    cute::SM90_TMA_STORE_2D::copy(&tensor_map_cd, smem_base_ptr, n_idx, m_idx);
                }
                cute::tma_store_arrive();
            }
            __syncwarp();
        }
    }
}

}   // namespace nvfp4_gemm::epilogue

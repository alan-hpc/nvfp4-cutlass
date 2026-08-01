#pragma once

#include <cute/atom/copy_traits_sm100.hpp>

#include <nvfp4_gemm/common/math.cuh>
#include <nvfp4_gemm/common/types.cuh>
#include <nvfp4_gemm/common/utils.cuh>
#include <nvfp4_gemm/ptx/ld_st.cuh>
#include <nvfp4_gemm/ptx/tcgen05.cuh>

namespace nvfp4_gemm::epilogue {

using utils::PatternVisitor;

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
sm100_store_cd_gscale(const PatternVisitor<pattern_cd_t>& smem_cd, uint32_t& tma_stage_idx, const uint32_t& tmem_base_addr, const uint32_t& base_m_idx, const uint32_t& base_n_idx, const uint32_t& batch_idx, const float& global_scale, const uint32_t& epilogue_warp_idx, const uint32_t& subpartition_idx, const uint32_t& lane_idx, const cutlass::arch::ClusterTransactionBarrier* tmem_empty_barrier, const cute::TmaDescriptor& tensor_map_cd)
{
    constexpr uint32_t kNumBankGroupBytes    = 16;
    constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(cd_dtype_t);
    NVFP4_STATIC_ASSERT(kSwizzleCDMode > 0, "TMA D must be swizzled");
    NVFP4_STATIC_ASSERT(STORE_BLOCK_N % kNumElemsPerBankGroup == 0, "Invalid swizzling");
    NVFP4_STATIC_ASSERT(BLOCK_M % STORE_BLOCK_M == 0, "Invalid block sizes");
    NVFP4_STATIC_ASSERT(BLOCK_N % STORE_BLOCK_N == 0, "Invalid block sizes");
    NVFP4_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>,
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

            // Issue every bank group's TMEM load up front and wait once.
            // `tcgen05.wait::ld` waits for *all* outstanding loads of the warp,
            // so a per-group fence would serialize kNumBankGroups full TMEM
            // round trips where a single one covers them all -- ncu shows that
            // serialization as the kernel's dominant stall (long_scoreboard,
            // ~45% of issue latency).  The register cost (kNumBankGroups x 8
            // u32) is what the launch bounds leave room for.
            constexpr uint32_t kNumBankGroups = STORE_BLOCK_N / kNumElemsPerBankGroup;
            NVFP4_STATIC_ASSERT(kNumElemsPerBankGroup == 8, "Invalid type");
            uint32_t values[kNumBankGroups][kNumElemsPerBankGroup];
#pragma unroll
            for (uint32_t i = 0; i < kNumBankGroups; ++i)
            {
                const uint32_t tmem_addr = tmem_base_addr + w * BLOCK_N +
                                           s * STORE_BLOCK_N + i * kNumElemsPerBankGroup;
                cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_addr,
                                                      values[i][0],
                                                      values[i][1],
                                                      values[i][2],
                                                      values[i][3],
                                                      values[i][4],
                                                      values[i][5],
                                                      values[i][6],
                                                      values[i][7]);
            }
            cutlass::arch::fence_view_async_tmem_load();

            // The accumulator now lives in registers: release it before the
            // pack/store half, not after, so the MMA warp can start the next
            // tile while this store still drains.
            if (w == kNumMWaves - 1 and s == kNumStores - 1)
            {
                ptx::tcgen05_before_thread_sync();
                tmem_empty_barrier->arrive();
            }

#pragma unroll
            for (uint32_t i = 0; i < kNumBankGroups; ++i)
            {
                auto           bank_group_index = i + lane_idx * (kSwizzleCDMode / kNumBankGroupBytes);
                constexpr bool kHasShortcut     = (kSwizzleCDMode / kNumBankGroupBytes) == 8;
                auto           row              = kHasShortcut ? (i / 8 + lane_idx) : (bank_group_index / 8);
                auto           col              = kHasShortcut ? (i) : (bank_group_index % 8);
                col ^= row % (kSwizzleCDMode / 16);

                // `tcgen05.ld` takes its 32-lane TMEM sub-partition from the
                // warp's *absolute* index mod 4, not from its rank among the
                // epilogue warps.  Those two agree only when the epilogue
                // happens to start on a multiple-of-4 warp, so the destination
                // row block has to follow the sub-partition the hardware picked
                // -- otherwise the accumulator comes out rotated by whole
                // 32-row blocks.
                auto smem_ptr = smem_base_ptr +
                                subpartition_idx * 32 * kSwizzleCDMode +
                                row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;

                // Apply G_W in FP32, then pack pairs into BF16.
                float scaled[kNumElemsPerBankGroup];
#pragma unroll
                for (uint32_t j = 0; j < kNumElemsPerBankGroup; ++j)
                    scaled[j] = *reinterpret_cast<float*>(&values[i][j]) * global_scale;

                ptx::st_shared(
                    smem_ptr,
                    math::cast_into_bf16_and_pack(scaled[0], scaled[1]),
                    math::cast_into_bf16_and_pack(scaled[2], scaled[3]),
                    math::cast_into_bf16_and_pack(scaled[4], scaled[5]),
                    math::cast_into_bf16_and_pack(scaled[6], scaled[7]));
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

/// Tail-split epilogue: k-pieces of one output tile deposit raw FP32 partials
/// in a workspace; the statically-last piece (kTailSplit - 1) spins for its
/// siblings, sums every piece from global memory, and runs the normal
/// scale/pack/TMA-store half.
///
/// Invariants the correctness argument rests on:
///   * tmem_empty arrives exactly once per piece, on the last subtile's TMEM
///     drain, BEFORE any workspace traffic -- past that point this function
///     touches only registers and gmem, because the accumulator stage is
///     already being refilled by the next scheduler item's UMMA.
///   * Workspace access is plain ld/st.global (the pointer must never be
///     __restrict__ or const-qualified): ld.global.nc sits outside the
///     ordering the fence/ticket protocol establishes.
///   * Each thread orders its own partial stores before its ticket increment
///     (__threadfence is cumulative at .gpu scope), so once the merger has
///     seen kTailSplit * kNumUMMAStoreThreads tickets every partial is
///     visible.  The spin cannot deadlock: all pieces of a tile are resident
///     simultaneously in the persistent grid.
///   * The slot layout is register-order, not [m][n]: writer and merger share
///     the exact thread map, so each warp's traffic is contiguous both ways.
// __noinline__ is load-bearing, not a style choice: ptxas allocates registers
// kernel-wide, and inlining this cold path inflated the hot code from 56 to
// 80+ registers with stack spills -- a measured +13..25 us on every ACTIVE
// shape. Behind an ABI call the tail pays its own register bill.
template<uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t STORE_BLOCK_M, uint32_t STORE_BLOCK_N, uint32_t kSwizzleCDMode, uint32_t kNumTMAStoreStages, uint32_t kNumUMMAStoreThreads, GemmType kGemmType, uint32_t kTailSplit, uint32_t kTimingProbe, typename cd_dtype_t, typename pattern_cd_t>
__device__ __noinline__ void
sm100_store_cd_tail_split(const PatternVisitor<pattern_cd_t>& smem_cd, uint32_t& tma_stage_idx, const uint32_t& tmem_base_addr, const uint32_t& base_m_idx, const uint32_t& base_n_idx, const uint32_t& batch_idx, const float& global_scale, const uint32_t& epilogue_warp_idx, const uint32_t& subpartition_idx, const uint32_t& lane_idx, const cutlass::arch::ClusterTransactionBarrier* tmem_empty_barrier, const cute::TmaDescriptor& tensor_map_cd, float* workspace, const uint32_t& tail_slot, unsigned long long* probe_buf)
{
    // kTimingProbe == 3: per-phase cycle accounting into slots 13/14/15
    // (phase-1 TMEM drain+STG / ticket+spin / merge+store).
    const bool probe_on = kTimingProbe == 3 and probe_buf != nullptr and
                          epilogue_warp_idx == 0 and lane_idx == 0;
    const auto pt0 = kTimingProbe == 3 ? clock64() : 0;
    constexpr uint32_t kNumBankGroupBytes    = 16;
    constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(cd_dtype_t);
    constexpr uint32_t kNumMWaves            = BLOCK_M / STORE_BLOCK_M;
    constexpr uint32_t kNumStores            = BLOCK_N / STORE_BLOCK_N;
    constexpr uint32_t kNumBankGroups        = STORE_BLOCK_N / kNumElemsPerBankGroup;
    constexpr uint32_t kSlotFloats           = BLOCK_M * BLOCK_N;
    NVFP4_STATIC_ASSERT(kNumElemsPerBankGroup == 8, "Invalid type");

    int*           counters = reinterpret_cast<int*>(workspace);
    float*         partials = workspace + 1024;
    const uint32_t tile     = tail_slot / kTailSplit;
    float*         own_slot = partials + static_cast<size_t>(tail_slot) * kSlotFloats;

    // ---- Phase 1: drain TMEM into this piece's slot ----------------------
#pragma unroll
    for (uint32_t w = 0; w < kNumMWaves; ++w)
    {
#pragma unroll
        for (uint32_t s = 0; s < kNumStores; ++s)
        {
            uint32_t values[kNumBankGroups][kNumElemsPerBankGroup];
#pragma unroll
            for (uint32_t i = 0; i < kNumBankGroups; ++i)
            {
                const uint32_t tmem_addr = tmem_base_addr + w * BLOCK_N +
                                           s * STORE_BLOCK_N + i * kNumElemsPerBankGroup;
                cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_addr,
                                                      values[i][0],
                                                      values[i][1],
                                                      values[i][2],
                                                      values[i][3],
                                                      values[i][4],
                                                      values[i][5],
                                                      values[i][6],
                                                      values[i][7]);
            }
            cutlass::arch::fence_view_async_tmem_load();
            if (w == kNumMWaves - 1 and s == kNumStores - 1)
            {
                ptx::tcgen05_before_thread_sync();
                tmem_empty_barrier->arrive();
            }
#pragma unroll
            for (uint32_t i = 0; i < kNumBankGroups; ++i)
            {
                const uint32_t chunk = (w * kNumStores + s) * kNumBankGroups + i;
                uint32_t*      dst   = reinterpret_cast<uint32_t*>(own_slot) +
                                (chunk * STORE_BLOCK_M + subpartition_idx * 32 + lane_idx) *
                                    kNumElemsPerBankGroup;
                *reinterpret_cast<uint4*>(dst) =
                    make_uint4(values[i][0], values[i][1], values[i][2], values[i][3]);
                *reinterpret_cast<uint4*>(dst + 4) =
                    make_uint4(values[i][4], values[i][5], values[i][6], values[i][7]);
            }
        }
    }

    const auto pt1 = kTimingProbe == 3 ? clock64() : 0;
    if (probe_on)
        atomicAdd(probe_buf + 13, static_cast<unsigned long long>(pt1 - pt0));

    // ---- Phase 2: publish ------------------------------------------------
    // One ticket per PIECE, not per thread: 2S atomics on the counter line
    // instead of S*128 RMWs plus 128 spinning readers -- the first cut of
    // this protocol measured ~15 us of pure line contention per piece.
    cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);
    if (epilogue_warp_idx == 0 and cute::elect_one_sync())
    {
        __threadfence();
        atomicAdd(counters + tile, 1);
    }

    // ---- Phase 3: the last piece merges and stores -----------------------
    if (tail_slot % kTailSplit != kTailSplit - 1)
        return;
    if (epilogue_warp_idx == 0 and cute::elect_one_sync())
        while (*reinterpret_cast<volatile int*>(counters + tile) < static_cast<int>(kTailSplit))
            __nanosleep(128);
    cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);
    __threadfence();
    const auto pt2 = kTimingProbe == 3 ? clock64() : 0;
    if (probe_on)
        atomicAdd(probe_buf + 14, static_cast<unsigned long long>(pt2 - pt1));

    auto advance_store_pipeline = [&]() {
        tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages;
    };
    const float* tile_base = partials + static_cast<size_t>(tile) * kTailSplit * kSlotFloats;

#pragma unroll
    for (uint32_t w = 0; w < kNumMWaves; ++w)
    {
#pragma unroll
        for (uint32_t s = 0; s < kNumStores; ++s, advance_store_pipeline())
        {
            auto smem_base_ptr = reinterpret_cast<uint8_t*>(smem_cd[tma_stage_idx]);

            // Issue the whole subtile's gather BEFORE the store-pipeline sync:
            // the 2*kTailSplit*kNumBankGroups loads are independent, so their
            // L2 round trip overlaps tma_store_wait + the named barrier
            // instead of serializing after them (measured 13 us -> the loads
            // were one dependent round trip per subtile).
            float4 part_lo[kNumBankGroups][kTailSplit];
            float4 part_hi[kNumBankGroups][kTailSplit];
#pragma unroll
            for (uint32_t i = 0; i < kNumBankGroups; ++i)
            {
                const uint32_t chunk = (w * kNumStores + s) * kNumBankGroups + i;
                const uint32_t base  = (chunk * STORE_BLOCK_M + subpartition_idx * 32 + lane_idx) *
                                      kNumElemsPerBankGroup;
#pragma unroll
                for (uint32_t piece = 0; piece < kTailSplit; ++piece)
                {
                    const float* src   = tile_base + piece * kSlotFloats + base;
                    part_lo[i][piece]  = *reinterpret_cast<const float4*>(src);
                    part_hi[i][piece]  = *reinterpret_cast<const float4*>(src + 4);
                }
            }

            if (epilogue_warp_idx == 0)
                cute::tma_store_wait<kNumTMAStoreStages - 1>();
            cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

            const auto m_idx = base_m_idx + w * STORE_BLOCK_M;
            const auto n_idx = base_n_idx + s * STORE_BLOCK_N;
#pragma unroll
            for (uint32_t i = 0; i < kNumBankGroups; ++i)
            {
                float sum[kNumElemsPerBankGroup] = {};
#pragma unroll
                for (uint32_t piece = 0; piece < kTailSplit; ++piece)
                {
                    sum[0] += part_lo[i][piece].x, sum[1] += part_lo[i][piece].y;
                    sum[2] += part_lo[i][piece].z, sum[3] += part_lo[i][piece].w;
                    sum[4] += part_hi[i][piece].x, sum[5] += part_hi[i][piece].y;
                    sum[6] += part_hi[i][piece].z, sum[7] += part_hi[i][piece].w;
                }

                auto           bank_group_index = i + lane_idx * (kSwizzleCDMode / kNumBankGroupBytes);
                constexpr bool kHasShortcut     = (kSwizzleCDMode / kNumBankGroupBytes) == 8;
                auto           row              = kHasShortcut ? (i / 8 + lane_idx) : (bank_group_index / 8);
                auto           col              = kHasShortcut ? (i) : (bank_group_index % 8);
                col ^= row % (kSwizzleCDMode / 16);
                auto smem_ptr = smem_base_ptr + subpartition_idx * 32 * kSwizzleCDMode +
                                row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;
                float scaled[kNumElemsPerBankGroup];
#pragma unroll
                for (uint32_t j = 0; j < kNumElemsPerBankGroup; ++j)
                    scaled[j] = sum[j] * global_scale;
                ptx::st_shared(smem_ptr,
                               math::cast_into_bf16_and_pack(scaled[0], scaled[1]),
                               math::cast_into_bf16_and_pack(scaled[2], scaled[3]),
                               math::cast_into_bf16_and_pack(scaled[4], scaled[5]),
                               math::cast_into_bf16_and_pack(scaled[6], scaled[7]));
            }

            cute::tma_store_fence();
            cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);
            if (epilogue_warp_idx == 0 and cute::elect_one_sync())
            {
                if constexpr (kGemmType == GemmType::MGroupedMasked)
                    cute::SM90_TMA_STORE_3D::copy(&tensor_map_cd, smem_base_ptr, n_idx, m_idx, batch_idx);
                else
                    cute::SM90_TMA_STORE_2D::copy(&tensor_map_cd, smem_base_ptr, n_idx, m_idx);
                cute::tma_store_arrive();
            }
            __syncwarp();
        }
    }

    if (probe_on)
        atomicAdd(probe_buf + 15, static_cast<unsigned long long>(clock64() - pt2));

    // ---- Phase 4: self-clean for the launch after next -------------------
    // (Parity double-buffering on the host isolates the immediately-following
    // launch, so resetting here is race-free.)
    cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);
    if (epilogue_warp_idx == 0 and lane_idx == 0)
        counters[tile] = 0;
}

}   // namespace nvfp4_gemm::epilogue

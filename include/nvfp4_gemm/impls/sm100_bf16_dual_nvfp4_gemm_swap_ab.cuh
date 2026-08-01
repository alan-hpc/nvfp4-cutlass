#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>

#include <nvfp4_gemm/common/math.cuh>
#include <nvfp4_gemm/common/tma_copy.cuh>
#include <nvfp4_gemm/common/types.cuh>
#include <nvfp4_gemm/common/utils.cuh>
#include <nvfp4_gemm/mma/sm100.cuh>
#include <nvfp4_gemm/ptx/ld_st.cuh>
#include <nvfp4_gemm/ptx/tcgen05.cuh>
#include <nvfp4_gemm/ptx/utils.cuh>
#include <nvfp4_gemm/scheduler/grouped_gemm.cuh>

#include <nvfp4_gemm/epilogue/sm100_store_cd_swap_ab.cuh>
#include <nvfp4_gemm/transform/dual_nvfp4.cuh>

namespace nvfp4_gemm {

using transform::ScalePolicy;

/// Swap-AB variant of the fused BF16 x NVFP4 grouped GEMM, for small-M
/// (decode / small-prefill) batches.
///
/// The normal orientation puts activations on the 128-row M side, so a
/// decode expert with 3 tokens still buys a whole 128-row tile of BF16
/// delivery and transform work.  Here the operands swap:
///
///   * the M side (UMMA_M = 128) carries *weight rows* -- the model dimension,
///     which is always a multiple of 128 -- loaded by TMA as packed FP4 with
///     no transform at all;
///   * the N side (UMMA_N = BLOCK_T, 32..128) carries *tokens*, so the
///     per-expert granularity drops from 128 rows to BLOCK_T columns; the
///     BF16 activation tile shrinks by the same factor, and with it the
///     per-k-block L2->SMEM delivery that bounds the sub-wave latency floor
///     (~50 KB -> ~17 KB at BLOCK_T = 32).
///
/// Warp specialization mirrors the normal kernel; only the payloads swap:
///
///   warp 0                     TMA: W (packed E2M1), SFW, act (BF16)
///   warp 1                     MMA issue + UTCCP of SFW/SFACT0/SFACT1
///   warp 2                     SFW UTCCP transpose (weight scales from gmem)
///   warps 3 .. 2+T             transform producers over the token tile
///   warps 3+T ..               epilogue: TMEM -> reg, x G_W, transposed store
///
/// The accumulator is (128 weight rows) x (BLOCK_T tokens); the epilogue
/// stores it transposed so D keeps the (tokens, model_n) layout the normal
/// kernel produces.  Scale-factor atoms are 128-row granular on both sides;
/// the token side fills only BLOCK_T rows of each atom and UMMA with
/// N = BLOCK_T never reads the tail, so no zero-fill is needed.
template<uint32_t SHAPE_N, uint32_t SHAPE_K, uint32_t BLOCK_T, uint32_t BLOCK_K, uint32_t kNumGroups, uint32_t kSwizzleAMode, uint32_t kSwizzleABMode, uint32_t kSwizzleCDMode, uint32_t kNumStages, uint32_t kNumTransformWarps, uint32_t kNumEpilogueThreads, uint32_t kNumSMs, ScalePolicy kScalePolicy, bool kEnableResidualPass>
CUTLASS_GLOBAL void __launch_bounds__((3 + kNumTransformWarps) * 32 + kNumEpilogueThreads, 1)
    sm100_bf16_dual_nvfp4_gemm_swap_ab_impl(int* grouped_layout,
                                            const float* __restrict__ weight_global_scales,
                                            uint32_t                shape_m,
                                            uint32_t                shape_n,
                                            uint32_t                shape_k,
                                            const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                                            const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                                            const __grid_constant__ cute::TmaDescriptor tensor_map_sfb,
                                            const __grid_constant__ cute::TmaDescriptor tensor_map_cd)
{
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)
    using Barrier     = cutlass::arch::ClusterTransactionBarrier;
    using Allocator   = cute::TMEM::Allocator1Sm;
    using cd_dtype_t  = cutlass::bfloat16_t;
    using fp4_dtype_t = cutlass::float_e2m1_t;

    // ---- MMA shape (swapped) --------------------------------------------
    constexpr uint32_t BLOCK_WR = 128;   // weight rows per tile == UMMA_M
    constexpr uint32_t UMMA_M   = BLOCK_WR;
    constexpr uint32_t UMMA_N   = BLOCK_T;
    constexpr uint32_t UMMA_K   = transform::kKPerSFAtom;
    NVFP4_STATIC_ASSERT(BLOCK_K % UMMA_K == 0, "Block K must be divisible by UMMA K");
    NVFP4_STATIC_ASSERT(UMMA_N % 32 == 0 and 32 <= UMMA_N and UMMA_N <= 128, "Invalid token tile");
    NVFP4_STATIC_ASSERT(kSwizzleABMode * 2 == BLOCK_K, "Packed-FP4 swizzle must cover one K block");
    NVFP4_STATIC_ASSERT(kSwizzleAMode == 128, "BF16 activation tile assumes 128 B swizzle");

    // ---- Scale-factor geometry ------------------------------------------
    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    constexpr uint32_t SF_BLOCK_T            = math::constexpr_align(BLOCK_T, kNumUTCCPAlignedElems);
    constexpr uint32_t kNumKAtoms            = BLOCK_K / UMMA_K;
    constexpr uint32_t kNumSFWSubAtoms       = 1;   // 128 weight rows = one atom
    constexpr uint32_t kNumSFTSubAtoms       = SF_BLOCK_T / kNumUTCCPAlignedElems;
    NVFP4_STATIC_ASSERT(kNumSFTSubAtoms == 1, "Token tile exceeds one SF atom");
    constexpr uint32_t kNumSFWTmemCols = kNumKAtoms * 4;
    constexpr uint32_t kNumSFTTmemCols = kNumKAtoms * 4;

    // ---- Epilogue --------------------------------------------------------
    constexpr uint32_t kNumSFTmemCols     = kNumSFWTmemCols + 2 * kNumSFTTmemCols;
    constexpr uint32_t kNumEpilogueStages = 2;   // accumulator is tiny in swap space
    constexpr uint32_t kNumTMAStoreStages = 2;
    constexpr uint32_t STORE_BLOCK_T      = BLOCK_T;   // token cols per TMA store
    constexpr uint32_t STORE_BLOCK_WR     = 128;       // model cols per store row
    constexpr uint32_t kNumUMMAStoreThreads = 128;     // full warpgroup for TMEM rows
    NVFP4_STATIC_ASSERT(kNumUMMAStoreThreads <= kNumEpilogueThreads, "Not enough epilogue threads");

    // ---- Tensor memory budget -------------------------------------------
    constexpr uint32_t kNumAccumTmemCols = UMMA_N * kNumEpilogueStages;
    constexpr uint32_t kNumTmemCols =
        utils::get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFTmemCols>();
    constexpr uint32_t kTmemStartColOfSFW  = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFT0 = kTmemStartColOfSFW + kNumSFWTmemCols;
    constexpr uint32_t kTmemStartColOfSFT1 = kTmemStartColOfSFT0 + kNumSFTTmemCols;
    NVFP4_STATIC_ASSERT(kNumAccumTmemCols + kNumSFTmemCols <= 512, "Tensor memory overflow");

    // ---- Shared memory budget -------------------------------------------
    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE   = STORE_BLOCK_T * STORE_BLOCK_WR * sizeof(cd_dtype_t);
    constexpr uint32_t SMEM_CD_SIZE             = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_ACT_SIZE_PER_STAGE  = BLOCK_T * BLOCK_K * sizeof(__nv_bfloat16);
    constexpr uint32_t SMEM_ACT0_SIZE_PER_STAGE = BLOCK_T * BLOCK_K / 2;
    constexpr uint32_t SMEM_W_SIZE_PER_STAGE    = BLOCK_WR * BLOCK_K / 2;
    constexpr uint32_t SMEM_SFW_SIZE_PER_STAGE  = kNumKAtoms * BLOCK_WR * 4;
    constexpr uint32_t SMEM_SFT_SIZE_PER_STAGE  = kNumKAtoms * SF_BLOCK_T * 4;

    // ---- Thread roles ----------------------------------------------------
    constexpr uint32_t kNumTransformThreads   = kNumTransformWarps * 32;
    constexpr uint32_t kNumNonEpilogueWarps   = 3 + kNumTransformWarps;
    constexpr uint32_t kNumNonEpilogueThreads = kNumNonEpilogueWarps * 32;
    constexpr uint32_t kFirstTransformWarp    = 3;
    NVFP4_STATIC_ASSERT(BLOCK_T * BLOCK_K / 16 % kNumTransformThreads == 0,
                        "Transform threads must evenly divide the token tile");

    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = ptx::get_lane_idx();

    if (warp_idx == 0)
    {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_sfb);
        cute::prefetch_tma_descriptor(&tensor_map_cd);
    }

    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;
    const auto shape_sfw_k = math::ceil_div(shape_k, UMMA_K);

    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // Layout: [CD][act][act0][act1][W][SFW][SFACT0][SFACT1][barriers]
    auto               smem_cd          = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cd_dtype_t*>(smem_buffer + i * SMEM_CD_SIZE_PER_STAGE);
    });
    constexpr uint32_t kOffsetAct       = SMEM_CD_SIZE;
    constexpr uint32_t kOffsetAct0      = kOffsetAct + kNumStages * SMEM_ACT_SIZE_PER_STAGE;
    constexpr uint32_t kOffsetAct1      = kOffsetAct0 + kNumStages * SMEM_ACT0_SIZE_PER_STAGE;
    constexpr uint32_t kOffsetW         = kOffsetAct1 + kNumStages * SMEM_ACT0_SIZE_PER_STAGE;
    constexpr uint32_t kOffsetSFW       = kOffsetW + kNumStages * SMEM_W_SIZE_PER_STAGE;
    constexpr uint32_t kOffsetSFT0      = kOffsetSFW + kNumStages * SMEM_SFW_SIZE_PER_STAGE;
    constexpr uint32_t kOffsetSFT1      = kOffsetSFT0 + kNumStages * SMEM_SFT_SIZE_PER_STAGE;
    constexpr uint32_t kOffsetBarriers  = kOffsetSFT1 + kNumStages * SMEM_SFT_SIZE_PER_STAGE;

    auto smem_act  = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_bfloat16*>(smem_buffer + kOffsetAct + i * SMEM_ACT_SIZE_PER_STAGE);
    });
    auto smem_act0 = utils::PatternVisitor([&](const uint32_t& i) {
        return smem_buffer + kOffsetAct0 + i * SMEM_ACT0_SIZE_PER_STAGE;
    });
    auto smem_act1 = utils::PatternVisitor([&](const uint32_t& i) {
        return smem_buffer + kOffsetAct1 + i * SMEM_ACT0_SIZE_PER_STAGE;
    });
    auto smem_w    = utils::PatternVisitor([&](const uint32_t& i) {
        return smem_buffer + kOffsetW + i * SMEM_W_SIZE_PER_STAGE;
    });
    auto smem_sfw  = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(smem_buffer + kOffsetSFW + i * SMEM_SFW_SIZE_PER_STAGE);
    });
    auto smem_sft0 = utils::PatternVisitor([&](const uint32_t& i) {
        return smem_buffer + kOffsetSFT0 + i * SMEM_SFT_SIZE_PER_STAGE;
    });
    auto smem_sft1 = utils::PatternVisitor([&](const uint32_t& i) {
        return smem_buffer + kOffsetSFT1 + i * SMEM_SFT_SIZE_PER_STAGE;
    });

    // ---- Barriers --------------------------------------------------------
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer + kOffsetBarriers);
    auto full_barriers     = utils::PatternVisitor(
        [=](const uint32_t& i) { return barrier_start_ptr + i; });
    auto empty_barriers = utils::PatternVisitor(
        [=](const uint32_t& i) { return barrier_start_ptr + kNumStages + i; });
    auto transform_full_barriers = utils::PatternVisitor(
        [=](const uint32_t& i) { return barrier_start_ptr + kNumStages * 2 + i; });
    auto tmem_full_barriers = utils::PatternVisitor(
        [=](const uint32_t& i) { return barrier_start_ptr + kNumStages * 3 + i; });
    auto tmem_empty_barriers = utils::PatternVisitor(
        [=](const uint32_t& i) { return barrier_start_ptr + kNumStages * 3 + kNumEpilogueStages + i; });
    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(
        barrier_start_ptr + kNumStages * 3 + kNumEpilogueStages * 2);

    if (warp_idx == 1 and cute::elect_one_sync())
    {
#    pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++i)
        {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(1);
            // Every transform thread arrives, plus every lane of the SFW warp.
            transform_full_barriers[i]->init(kNumTransformThreads + 32);
        }
#    pragma unroll
        for (uint32_t i = 0; i < kNumEpilogueStages; ++i)
        {
            tmem_full_barriers[i]->init(1);
            tmem_empty_barriers[i]->init(kNumUMMAStoreThreads);
        }
        cutlass::arch::fence_barrier_init();
    }
    else if (warp_idx == 2)
    {
        Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
    }
    __syncthreads();

    cudaGridDependencySynchronize();

    // Scheduler in swapped coordinates: its "M" axis is the token axis (the
    // grouped one, tiled by BLOCK_T so the expert lookup lands on token
    // blocks), its "N" axis is the model dimension tiled by 128.
    uint32_t tok_block_idx, wr_block_idx;
    auto     scheduler = sched::Scheduler<GemmType::MGroupedContiguous, BLOCK_T, BLOCK_WR, kNumGroups, kNumSMs>(
        shape_m,
        shape_n,
        grouped_layout);

    uint32_t stage_idx = 0, phase = 0;
    auto     advance_pipeline = [&](uint32_t& k_block_idx) {
        ++k_block_idx;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // ==================================================================
    // Warp 0: TMA producer
    // ==================================================================
    if (warp_idx == 0 and cute::elect_one_sync())
    {
        while (scheduler.get_next_block(tok_block_idx, wr_block_idx))
        {
            const auto num_total_k_blocks = math::ceil_div(shape_k, BLOCK_K);

            // Tile-invariant coordinates (the expert-offset load hoisted out of
            // the K loop, as in the normal kernel).
            const uint32_t tok_idx = tok_block_idx * BLOCK_T;
            const uint32_t wr_idx =
                scheduler.template get_global_idx<true>(shape_n, BLOCK_WR, wr_block_idx, tok_block_idx);
            const uint32_t sfw_k_base =
                scheduler.template get_global_idx<true>(shape_sfw_k, 1, 0, tok_block_idx);

            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx))
            {
                empty_barriers[stage_idx]->wait(phase ^ 1);

                const uint32_t k_idx = k_block_idx * BLOCK_K;

                // Activations: BF16, K-major, BLOCK_T rows.
                tma::copy<BLOCK_K, BLOCK_T, kSwizzleAMode, __nv_bfloat16>(
                    &tensor_map_a,
                    full_barriers[stage_idx],
                    smem_act[stage_idx],
                    k_idx,
                    tok_idx);
                // Weights: packed E2M1, 128 rows of the model dimension.
                cute::SM90_TMA_LOAD_2D::copy(
                    &tensor_map_b,
                    reinterpret_cast<uint64_t*>(full_barriers[stage_idx]),
                    static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
                    smem_w[stage_idx],
                    k_idx,
                    wr_idx);

                uint32_t num_arrival_bytes = SMEM_ACT_SIZE_PER_STAGE + SMEM_W_SIZE_PER_STAGE;

                // Weight scales: one int32 atom word per (sf_k, weight row).
                const uint32_t sfw_n_idx = wr_block_idx * BLOCK_WR;
                const uint32_t sfw_k_idx = sfw_k_base + k_idx / UMMA_K;
                tma::copy<BLOCK_WR, kNumKAtoms, 0>(
                    &tensor_map_sfb,
                    full_barriers[stage_idx],
                    smem_sfw[stage_idx],
                    sfw_n_idx,
                    sfw_k_idx);
                num_arrival_bytes += BLOCK_WR * kNumKAtoms * sizeof(uint32_t);

                full_barriers[stage_idx]->arrive_and_expect_tx(num_arrival_bytes);
            }
        }
        // ==================================================================
        // Warp 1: MMA issue -- two block-scaled passes per K step
        // ==================================================================
    }
    else if (warp_idx == 1)
    {
        auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<
            fp4_dtype_t,
            fp4_dtype_t,
            float,
            cutlass::float_ue4m3_t,
            UMMA_M,
            UMMA_N,
            cute::UMMA::Major::K,
            cute::UMMA::Major::K>();
        auto sf_desc = mma::sm100::make_sf_desc(nullptr);

        NVFP4_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
        auto w_desc  = mma::sm100::make_packed_fp4_desc<BLOCK_WR, BLOCK_K, kSwizzleABMode>(smem_w[0]);
        auto t0_desc = mma::sm100::make_packed_fp4_desc<BLOCK_T, BLOCK_K, kSwizzleABMode>(smem_act0[0]);
        auto t1_desc = t0_desc;
        mma::sm100::replace_smem_desc_addr(t1_desc, smem_act1[0]);

        uint32_t w_desc_lo  = lane_idx < kNumStages ? w_desc.lo + lane_idx * SMEM_W_SIZE_PER_STAGE / 16 : 0u;
        uint32_t t0_desc_lo = lane_idx < kNumStages ? t0_desc.lo + lane_idx * SMEM_ACT0_SIZE_PER_STAGE / 16 : 0u;
        uint32_t t1_desc_lo = lane_idx < kNumStages ? t1_desc.lo + lane_idx * SMEM_ACT0_SIZE_PER_STAGE / 16 : 0u;

        while (scheduler.get_next_block(tok_block_idx, wr_block_idx))
        {
            const auto accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
            const auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;
            tmem_empty_barriers[accum_stage_idx]->wait(accum_phase_idx ^ 1);
            ptx::tcgen05_after_thread_sync();

            const auto num_total_k_blocks = math::ceil_div(shape_k, BLOCK_K);
#    pragma unroll 2
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx))
            {
                transform_full_barriers[stage_idx]->wait(phase);
                ptx::tcgen05_after_thread_sync();

                const auto w_base_lo  = ptx::exchange(w_desc_lo, stage_idx);
                const auto t0_base_lo = ptx::exchange(t0_desc_lo, stage_idx);
                const auto t1_base_lo = ptx::exchange(t1_desc_lo, stage_idx);

                if (cute::elect_one_sync())
                {
                    auto utccp_sf = [&](uint8_t* smem_ptr, const uint32_t& tmem_col) {
                        mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                        cute::SM100_UTCCP_4x32dp128bit_1cta::copy(sf_desc, tmem_col);
                    };
#    pragma unroll
                    for (uint32_t a = 0; a < kNumKAtoms; ++a)
                    {
                        utccp_sf(reinterpret_cast<uint8_t*>(smem_sfw[stage_idx]) +
                                     a * kNumUTCCPAlignedElems * 4,
                                 kTmemStartColOfSFW + a * 4);
                        utccp_sf(smem_sft0[stage_idx] + a * kNumUTCCPAlignedElems * 4,
                                 kTmemStartColOfSFT0 + a * 4);
                        if constexpr (kEnableResidualPass)
                            utccp_sf(smem_sft1[stage_idx] + a * kNumUTCCPAlignedElems * 4,
                                     kTmemStartColOfSFT1 + a * 4);
                    }

                    // Pass p over K atom `a`: W stays the A operand, the token
                    // tile supplies the B operand and its scales.
                    auto issue_pass = [&](const uint32_t& a, const uint32_t& t_base_lo, const uint32_t& sft_tmem_base, const bool& accumulate) {
                        auto desc_w = w_desc;
                        desc_w.lo   = mma::sm100::advance_packed_fp4_desc_lo(w_base_lo, a * UMMA_K);
                        auto desc_t = t0_desc;
                        desc_t.lo   = mma::sm100::advance_packed_fp4_desc_lo(t_base_lo, a * UMMA_K);
                        const auto runtime_desc =
                            mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, 0, 0);
                        ptx::SM100_MMA_MXF4NVF4_SS::fma(
                            desc_w,
                            desc_t,
                            accum_stage_idx * UMMA_N,
                            accumulate,
                            runtime_desc,
                            kTmemStartColOfSFW + a * 4,
                            sft_tmem_base + a * 4);
                    };

#    pragma unroll
                    for (uint32_t a = 0; a < kNumKAtoms; ++a)
                    {
                        issue_pass(a, t0_base_lo, kTmemStartColOfSFT0, a > 0 or k_block_idx > 0);
                        if constexpr (kEnableResidualPass)
                            issue_pass(a, t1_base_lo, kTmemStartColOfSFT1, true);
                    }
                }
                __syncwarp();

                cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]));
                if (k_block_idx == num_total_k_blocks - 1)
                    cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barriers[accum_stage_idx]));
                __syncwarp();
            }
        }
        // ==================================================================
        // Warp 2: SFW transpose into the layout UTCCP expects
        // ==================================================================
    }
    else if (warp_idx == 2)
    {
        auto transpose_atom = [&](uint32_t* smem_ptr) {
            uint32_t values[4];
#    pragma unroll
            for (uint32_t i = 0; i < 4; ++i)
                values[i] = ptx::ld_shared(smem_ptr + i * 32 + lane_idx);
            __syncwarp();
            ptx::st_shared(smem_ptr + lane_idx * 4,
                           values[0],
                           values[1],
                           values[2],
                           values[3]);
        };

        while (scheduler.get_next_block(tok_block_idx, wr_block_idx))
        {
            const auto num_total_k_blocks = math::ceil_div(shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx))
            {
                full_barriers[stage_idx]->wait(phase);

#    pragma unroll
                for (uint32_t i = 0; i < kNumKAtoms * kNumSFWSubAtoms; ++i)
                    transpose_atom(smem_sfw[stage_idx] + i * kNumUTCCPAlignedElems);
                cutlass::arch::fence_view_async_shared();

                transform_full_barriers[stage_idx]->arrive();
            }
        }
        // ==================================================================
        // Warps 3..2+T: transform producers over the token tile
        // ==================================================================
    }
    else if (warp_idx >= kFirstTransformWarp and warp_idx < kNumNonEpilogueWarps)
    {
        const uint32_t transform_tid = threadIdx.x - kFirstTransformWarp * 32;

        while (scheduler.get_next_block(tok_block_idx, wr_block_idx))
        {
            const auto num_total_k_blocks = math::ceil_div(shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx))
            {
                full_barriers[stage_idx]->wait(phase);

                transform::transform_a_tile<
                    BLOCK_T,
                    BLOCK_K,
                    kSwizzleAMode,
                    kSwizzleABMode,
                    kNumTransformThreads,
                    kScalePolicy,
                    kEnableResidualPass>(
                    smem_act[stage_idx],
                    smem_act0[stage_idx],
                    smem_act1[stage_idx],
                    smem_sft0[stage_idx],
                    smem_sft1[stage_idx],
                    transform_tid);

                cutlass::arch::fence_view_async_shared();
                transform_full_barriers[stage_idx]->arrive();
            }
        }
        // ==================================================================
        // Epilogue warps
        // ==================================================================
    }
    else if (warp_idx >= kNumNonEpilogueWarps and
             warp_idx < (kNumNonEpilogueThreads + kNumUMMAStoreThreads) / 32)
    {
        const auto epilogue_warp_idx = warp_idx - kNumNonEpilogueWarps;
        NVFP4_TRAP_ONLY_DEVICE_ASSERT(ptx::ld_shared(tmem_ptr_in_smem) == 0);

        uint32_t tma_stage_idx = 0;
        while (scheduler.get_next_block(tok_block_idx, wr_block_idx))
        {
            const auto accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
            const auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;

            tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx);
            ptx::tcgen05_after_thread_sync();

            const auto base_tok_idx = tok_block_idx * BLOCK_T;
            const auto base_wr_idx  = wr_block_idx * BLOCK_WR;

            const uint32_t expert_idx   = scheduler.get_expert_idx(tok_block_idx);
            const float    global_scale =
                weight_global_scales == nullptr ? 1.0f : __ldg(weight_global_scales + expert_idx);

            epilogue::sm100_store_cd_swap_ab_gscale<
                BLOCK_WR,
                BLOCK_T,
                STORE_BLOCK_T,
                STORE_BLOCK_WR,
                kSwizzleCDMode,
                kNumTMAStoreStages,
                kNumUMMAStoreThreads,
                cd_dtype_t>(smem_cd, tma_stage_idx, accum_stage_idx * UMMA_N, base_wr_idx, base_tok_idx, global_scale, epilogue_warp_idx, static_cast<uint32_t>(warp_idx) % 4u, lane_idx, tmem_empty_barriers[accum_stage_idx], tensor_map_cd);
        }
    }

    __syncthreads();
    if (warp_idx == 0)
        Allocator().free(0, kNumTmemCols);

#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        NVFP4_DEVICE_ASSERT(false and "This kernel only supports sm_100f / sm_103f");
#endif
}

}   // namespace nvfp4_gemm

#pragma clang diagnostic pop

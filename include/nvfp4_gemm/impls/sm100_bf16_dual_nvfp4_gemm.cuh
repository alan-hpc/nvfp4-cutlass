#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/scheduler/gemm.cuh>

#include "../epilogue/sm100_store_cd_gscale.cuh"
#include "../ptx/tcgen05_nvfp4.cuh"
#include "../transform/dual_nvfp4.cuh"

namespace nvfp4_gemm {

using deep_gemm::GemmType;
using transform::ScalePolicy;

/// Advance a packed-FP4 UMMA descriptor along K.
///
/// DeepGEMM's `advance_umma_desc_lo` multiplies by `sizeof(dtype_t)` without
/// dividing by the pack factor, so it over-advances by 2x for E2M1 operands.
/// Two FP4 elements share a byte, so `k_elems` elements are `k_elems / 2` bytes.
CUTLASS_DEVICE uint32_t advance_packed_fp4_desc_lo(const uint32_t& base, const uint32_t& k_elems)
{
    return base + ((k_elems / 2) >> 4u);
}

/// Fused BF16 x NVFP4 grouped GEMM via in-mainloop dual-NVFP4 decomposition.
///
/// `C = A0 W^T + A1 W^T`, where `(A0, SFA0)` and `(A1, SFA1)` are produced from
/// the BF16 activation tile inside the mainloop and never touch global memory.
///
/// Warp specialization (single CTA, no 2-SM multicast):
///
///   warp 0                     TMA producer: A (BF16), W (packed E2M1), SFB
///   warp 1                     MMA issue + UTCCP of SFA0/SFA1/SFB into TMEM
///   warp 2                     SFB UTCCP transpose (SFA needs none, see below)
///   warps 3 .. 2+T             transform producers, running Algorithm 1
///   warps 3+T ..               epilogue: TMEM -> reg, x G_W, BF16 store
///
/// Unlike the reference design in the doc, the transform workers are a dedicated
/// warp group rather than the epilogue warps doing double duty.  Reusing the
/// epilogue warps saves 128 threads, but it serializes the epilogue of block i
/// against the mainloop of block i+1; with a dedicated group the persistent
/// scheduler keeps that overlap, which is what `kNumEpilogueStages == 2` buys.
/// Whether the thread saving or the overlap wins is shape-dependent and has to
/// be settled by measurement on hardware.
template<uint32_t SHAPE_N, uint32_t SHAPE_K, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t kNumGroups, uint32_t kSwizzleAMode, uint32_t kSwizzleABMode, uint32_t kSwizzleCDMode, uint32_t kNumStages, uint32_t kNumTransformWarps, uint32_t kNumEpilogueThreads, uint32_t kNumSMs, GemmType kGemmType, ScalePolicy kScalePolicy, bool kEnableResidualPass>
CUTLASS_GLOBAL void __launch_bounds__((3 + kNumTransformWarps) * 32 + kNumEpilogueThreads, 1)
    sm100_bf16_dual_nvfp4_gemm_impl(int* grouped_layout,
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

    // ---- MMA shape ------------------------------------------------------
    constexpr uint32_t LAYOUT_AD_M = 128;
    constexpr uint32_t UMMA_M      = LAYOUT_AD_M;
    constexpr uint32_t UMMA_N      = BLOCK_N;
    // `kind::mxf4nvf4.block16` steps 64 elements of K per instruction.
    constexpr uint32_t UMMA_K = transform::kKPerSFAtom;
    DG_STATIC_ASSERT(BLOCK_M == LAYOUT_AD_M, "Dual-NVFP4 GEMM requires BLOCK_M == 128");
    DG_STATIC_ASSERT(BLOCK_K % UMMA_K == 0, "Block K must be divisible by UMMA K");
    DG_STATIC_ASSERT(UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256, "Invalid UMMA N");
    // A partially-filled SF atom would leave the tail rows of SFB uninitialized,
    // since TMA only writes BLOCK_N of the SF_BLOCK_N rows UTCCP later reads.
    DG_STATIC_ASSERT(BLOCK_N % 128 == 0, "Block N must be a multiple of the 128-row SF atom");
    // Packed FP4 with BLOCK_K elements per row must be exactly one swizzle atom.
    DG_STATIC_ASSERT(kSwizzleABMode * 2 == BLOCK_K, "Packed-FP4 swizzle must cover one K block");
    DG_STATIC_ASSERT(kSwizzleAMode == 128, "BF16 A tile assumes 128 B swizzle");

    // ---- Scale-factor geometry -------------------------------------------
    // One 128x4 SF atom = 128 rows x 4 E4M3 bytes = 512 B of SMEM -> 4 TMEM cols.
    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    constexpr uint32_t SF_BLOCK_M            = deep_gemm::math::constexpr_align(BLOCK_M, kNumUTCCPAlignedElems);
    constexpr uint32_t SF_BLOCK_N            = deep_gemm::math::constexpr_align(BLOCK_N, kNumUTCCPAlignedElems);
    constexpr uint32_t kNumKAtoms            = BLOCK_K / UMMA_K;
    constexpr uint32_t kNumSFASubAtoms       = SF_BLOCK_M / kNumUTCCPAlignedElems;
    constexpr uint32_t kNumSFBSubAtoms       = SF_BLOCK_N / kNumUTCCPAlignedElems;
    constexpr uint32_t kNumSFATmemCols       = kNumKAtoms * kNumSFASubAtoms * 4;
    constexpr uint32_t kNumSFBTmemCols       = kNumKAtoms * kNumSFBSubAtoms * 4;

    // ---- Epilogue --------------------------------------------------------
    // Dual-A spends two extra SF column groups (SFA0 and SFA1) that a plain
    // NVFP4 GEMM does not, so the accumulator stage count has to be derived
    // rather than fixed at 2.  BLOCK_N = 256 leaves room for exactly one.
    constexpr uint32_t kNumSFTmemCols       = kNumSFATmemCols * 2 + kNumSFBTmemCols;
    constexpr uint32_t kNumEpilogueStages   = (UMMA_N * 2 + kNumSFTmemCols <= 512) ? 2 : 1;
    constexpr uint32_t kNumTMAStoreStages   = 2;
    constexpr uint32_t STORE_BLOCK_M        = cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t STORE_BLOCK_N        = kSwizzleCDMode / sizeof(cd_dtype_t);
    constexpr uint32_t kNumUMMAStoreThreads = STORE_BLOCK_M;
    DG_STATIC_ASSERT(kNumUMMAStoreThreads <= kNumEpilogueThreads, "Not enough epilogue threads");

    // ---- Tensor memory budget -------------------------------------------
    constexpr uint32_t kNumAccumTmemCols = UMMA_N * kNumEpilogueStages;
    constexpr uint32_t kNumTmemCols =
        deep_gemm::utils::get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFTmemCols>();
    constexpr uint32_t kTmemStartColOfSFA0 = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFA1 = kTmemStartColOfSFA0 + kNumSFATmemCols;
    constexpr uint32_t kTmemStartColOfSFB  = kTmemStartColOfSFA1 + kNumSFATmemCols;
    DG_STATIC_ASSERT(kNumAccumTmemCols + kNumSFTmemCols <= 512, "Tensor memory overflow");

    // ---- Shared memory budget -------------------------------------------
    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE  = STORE_BLOCK_M * STORE_BLOCK_N * sizeof(cd_dtype_t);
    constexpr uint32_t SMEM_CD_SIZE            = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE   = BLOCK_M * BLOCK_K * sizeof(__nv_bfloat16);
    constexpr uint32_t SMEM_A0_SIZE_PER_STAGE  = BLOCK_M * BLOCK_K / 2;
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE   = BLOCK_N * BLOCK_K / 2;
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = kNumKAtoms * SF_BLOCK_M * 4;
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = kNumKAtoms * SF_BLOCK_N * 4;
    DG_STATIC_ASSERT(SMEM_A_SIZE_PER_STAGE % 1024 == 0 and SMEM_A0_SIZE_PER_STAGE % 1024 == 0 and
                         SMEM_B_SIZE_PER_STAGE % 1024 == 0,
                     "A/B shared memory must be 1024 B aligned");

    // ---- Thread roles ----------------------------------------------------
    constexpr uint32_t kNumTransformThreads   = kNumTransformWarps * 32;
    constexpr uint32_t kNumNonEpilogueWarps   = 3 + kNumTransformWarps;
    constexpr uint32_t kNumNonEpilogueThreads = kNumNonEpilogueWarps * 32;
    constexpr uint32_t kFirstTransformWarp    = 3;
    DG_STATIC_ASSERT(BLOCK_M * kNumKAtoms % kNumTransformThreads == 0,
                     "Transform threads must evenly divide the A tile");

    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = deep_gemm::ptx::get_lane_idx();

    if (warp_idx == 0)
    {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_sfb);
        cute::prefetch_tma_descriptor(&tensor_map_cd);
    }

    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;
    // SFB rows cover 64 K elements each (4 NVFP4 blocks), matching one SF atom.
    const auto shape_sfb_k = deep_gemm::math::ceil_div(shape_k, UMMA_K);

    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // Layout: [CD stages][A stages][A0 stages][A1 stages][B stages][SFA0][SFA1][SFB][barriers]
    auto               smem_cd         = deep_gemm::utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cd_dtype_t*>(smem_buffer + i * SMEM_CD_SIZE_PER_STAGE);
    });
    constexpr uint32_t kOffsetA        = SMEM_CD_SIZE;
    constexpr uint32_t kOffsetA0       = kOffsetA + kNumStages * SMEM_A_SIZE_PER_STAGE;
    constexpr uint32_t kOffsetA1       = kOffsetA0 + kNumStages * SMEM_A0_SIZE_PER_STAGE;
    constexpr uint32_t kOffsetB        = kOffsetA1 + kNumStages * SMEM_A0_SIZE_PER_STAGE;
    constexpr uint32_t kOffsetSFA0     = kOffsetB + kNumStages * SMEM_B_SIZE_PER_STAGE;
    constexpr uint32_t kOffsetSFA1     = kOffsetSFA0 + kNumStages * SMEM_SFA_SIZE_PER_STAGE;
    constexpr uint32_t kOffsetSFB      = kOffsetSFA1 + kNumStages * SMEM_SFA_SIZE_PER_STAGE;
    constexpr uint32_t kOffsetBarriers = kOffsetSFB + kNumStages * SMEM_SFB_SIZE_PER_STAGE;

    auto smem_a    = deep_gemm::utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_bfloat16*>(smem_buffer + kOffsetA + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_a0   = deep_gemm::utils::PatternVisitor([&](const uint32_t& i) {
        return smem_buffer + kOffsetA0 + i * SMEM_A0_SIZE_PER_STAGE;
    });
    auto smem_a1   = deep_gemm::utils::PatternVisitor([&](const uint32_t& i) {
        return smem_buffer + kOffsetA1 + i * SMEM_A0_SIZE_PER_STAGE;
    });
    auto smem_b    = deep_gemm::utils::PatternVisitor([&](const uint32_t& i) {
        return smem_buffer + kOffsetB + i * SMEM_B_SIZE_PER_STAGE;
    });
    auto smem_sfa0 = deep_gemm::utils::PatternVisitor([&](const uint32_t& i) {
        return smem_buffer + kOffsetSFA0 + i * SMEM_SFA_SIZE_PER_STAGE;
    });
    auto smem_sfa1 = deep_gemm::utils::PatternVisitor([&](const uint32_t& i) {
        return smem_buffer + kOffsetSFA1 + i * SMEM_SFA_SIZE_PER_STAGE;
    });
    auto smem_sfb  = deep_gemm::utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(smem_buffer + kOffsetSFB + i * SMEM_SFB_SIZE_PER_STAGE);
    });

    // ---- Barriers --------------------------------------------------------
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer + kOffsetBarriers);
    auto full_barriers     = deep_gemm::utils::PatternVisitor(
        [=](const uint32_t& i) { return barrier_start_ptr + i; });
    auto empty_barriers = deep_gemm::utils::PatternVisitor(
        [=](const uint32_t& i) { return barrier_start_ptr + kNumStages + i; });
    // Signalled once the stage's A0/A1/SFA0/SFA1 are complete *and* SFB has been
    // transposed -- the MMA warp's single wait covers both producers.
    auto transform_full_barriers = deep_gemm::utils::PatternVisitor(
        [=](const uint32_t& i) { return barrier_start_ptr + kNumStages * 2 + i; });
    auto tmem_full_barriers = deep_gemm::utils::PatternVisitor(
        [=](const uint32_t& i) { return barrier_start_ptr + kNumStages * 3 + i; });
    auto tmem_empty_barriers = deep_gemm::utils::PatternVisitor(
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
            // Every transform thread arrives, plus every lane of the SFB warp.
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

    uint32_t m_block_idx, n_block_idx;
    auto     scheduler = deep_gemm::sched::Scheduler<
        kGemmType,
        BLOCK_M,
        BLOCK_N,
        kNumGroups,
        1,
        false,
        kNumSMs>(
        shape_m,
        shape_n,
        shape_k,
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
        while (scheduler.get_next_block(m_block_idx, n_block_idx))
        {
            const auto num_total_k_blocks = deep_gemm::math::ceil_div(scheduler.current_shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx))
            {
                empty_barriers[stage_idx]->wait(phase ^ 1);

                const uint32_t m_idx = scheduler.template get_global_idx<
                    (kGemmType == GemmType::MGroupedMasked),
                    deep_gemm::sched::IndexType::MN>(
                    shape_m,
                    BLOCK_M,
                    m_block_idx);
                const uint32_t n_idx = scheduler.template get_global_idx<
                    true,
                    deep_gemm::sched::IndexType::MN>(shape_n, BLOCK_N, n_block_idx, m_block_idx);
                const uint32_t k_idx = k_block_idx * BLOCK_K;

                // A is BF16 and K-major: 2 bytes per element, and no SFA to load
                // -- the whole point of the fused path is that A's scales are
                // produced on chip.
                deep_gemm::tma::copy<BLOCK_K, BLOCK_M, kSwizzleAMode, __nv_bfloat16>(
                    &tensor_map_a,
                    full_barriers[stage_idx],
                    smem_a[stage_idx],
                    k_idx,
                    m_idx);
                // W is packed E2M1 under `CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B`,
                // whose global dims and coordinates are counted in *FP4 elements*
                // even though shared memory receives two of them per byte.  With
                // BLOCK_K == 128 the row is exactly one 64 B swizzle atom, so a
                // single 2D copy covers it and no atom loop is needed.
                cute::SM90_TMA_LOAD_2D::copy(
                    &tensor_map_b,
                    reinterpret_cast<uint64_t*>(full_barriers[stage_idx]),
                    static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
                    smem_b[stage_idx],
                    k_idx,
                    n_idx);

                uint32_t num_arrival_bytes = SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE;

                const uint32_t sfb_n_idx = n_block_idx * BLOCK_N;
                const uint32_t sfb_k_idx = scheduler.template get_global_idx<
                    true,
                    deep_gemm::sched::IndexType::SF_K>(
                    shape_sfb_k,
                    1,
                    k_idx / UMMA_K,
                    m_block_idx);
                deep_gemm::tma::copy<BLOCK_N, kNumKAtoms, 0>(
                    &tensor_map_sfb,
                    full_barriers[stage_idx],
                    smem_sfb[stage_idx],
                    sfb_n_idx,
                    sfb_k_idx);
                num_arrival_bytes += BLOCK_N * kNumKAtoms * sizeof(uint32_t);

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
        auto sf_desc = deep_gemm::mma::sm100::make_sf_desc(nullptr);

        DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
        auto a0_desc = deep_gemm::mma::sm100::make_umma_desc<
            cute::UMMA::Major::K,
            BLOCK_M,
            BLOCK_K,
            kSwizzleABMode>(
            reinterpret_cast<fp4_dtype_t*>(smem_a0[0]),
            0,
            0);
        auto a1_desc = a0_desc;
        auto b_desc  = deep_gemm::mma::sm100::make_umma_desc<
            cute::UMMA::Major::K,
            BLOCK_N,
            BLOCK_K,
            kSwizzleABMode>(
            reinterpret_cast<fp4_dtype_t*>(smem_b[0]),
            0,
            0);
        deep_gemm::mma::sm100::replace_smem_desc_addr(a1_desc, smem_a1[0]);

        // Stage selection by lane broadcast, as in DeepGEMM: lane `s` holds the
        // descriptor for stage `s`, and `exchange` picks it in one shuffle.
        uint32_t a0_desc_lo = lane_idx < kNumStages ? a0_desc.lo + lane_idx * SMEM_A0_SIZE_PER_STAGE / 16 : 0u;
        uint32_t a1_desc_lo = lane_idx < kNumStages ? a1_desc.lo + lane_idx * SMEM_A0_SIZE_PER_STAGE / 16 : 0u;
        uint32_t b_desc_lo  = lane_idx < kNumStages ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;

        while (scheduler.get_next_block(m_block_idx, n_block_idx))
        {
            const auto accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
            const auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;
            tmem_empty_barriers[accum_stage_idx]->wait(accum_phase_idx ^ 1);
            deep_gemm::ptx::tcgen05_after_thread_sync();

            const auto num_total_k_blocks = deep_gemm::math::ceil_div(scheduler.current_shape_k, BLOCK_K);
#    pragma unroll 2
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx))
            {
                transform_full_barriers[stage_idx]->wait(phase);
                deep_gemm::ptx::tcgen05_after_thread_sync();

                const auto a0_base_lo = deep_gemm::ptx::exchange(a0_desc_lo, stage_idx);
                const auto a1_base_lo = deep_gemm::ptx::exchange(a1_desc_lo, stage_idx);
                const auto b_base_lo  = deep_gemm::ptx::exchange(b_desc_lo, stage_idx);

                if (cute::elect_one_sync())
                {
                    // Copy all three scale operands into tensor memory.  SFA0 and
                    // SFA1 were written pre-transposed by the transform warps;
                    // only SFB needed warp 2's shuffle.
                    auto utccp_sf = [&](uint8_t* smem_ptr, const uint32_t& tmem_col) {
                        deep_gemm::mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                        cute::SM100_UTCCP_4x32dp128bit_1cta::copy(sf_desc, tmem_col);
                    };
#    pragma unroll
                    for (uint32_t a = 0; a < kNumKAtoms; ++a)
                    {
#    pragma unroll
                        for (uint32_t i = 0; i < kNumSFASubAtoms; ++i)
                        {
                            const uint32_t byte_off = (a * kNumSFASubAtoms + i) * kNumUTCCPAlignedElems * 4;
                            const uint32_t col      = (a * kNumSFASubAtoms + i) * 4;
                            utccp_sf(smem_sfa0[stage_idx] + byte_off, kTmemStartColOfSFA0 + col);
                            if constexpr (kEnableResidualPass)
                                utccp_sf(smem_sfa1[stage_idx] + byte_off, kTmemStartColOfSFA1 + col);
                        }
#    pragma unroll
                        for (uint32_t i = 0; i < kNumSFBSubAtoms; ++i)
                        {
                            const uint32_t byte_off = (a * kNumSFBSubAtoms + i) * kNumUTCCPAlignedElems * 4;
                            const uint32_t col      = (a * kNumSFBSubAtoms + i) * 4;
                            utccp_sf(reinterpret_cast<uint8_t*>(smem_sfb[stage_idx]) + byte_off,
                                     kTmemStartColOfSFB + col);
                        }
                    }

                    // Pass p over K atom `a`.  Both passes share the same W and
                    // SFB fragments and land in the same accumulator; only the A
                    // operand and its scales differ.
                    auto issue_pass = [&](const uint32_t& a, const uint32_t& a_base_lo, const uint32_t& sfa_tmem_base, const bool& accumulate) {
                        auto desc_a = a0_desc;
                        desc_a.lo   = advance_packed_fp4_desc_lo(a_base_lo, a * UMMA_K);
                        auto desc_b = b_desc;
                        desc_b.lo   = advance_packed_fp4_desc_lo(b_base_lo, a * UMMA_K);
                        const auto runtime_desc =
                            deep_gemm::mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, 0, 0);
                        ptx::SM100_MMA_MXF4NVF4_SS::fma(
                            desc_a,
                            desc_b,
                            accum_stage_idx * UMMA_N,
                            accumulate,
                            runtime_desc,
                            sfa_tmem_base + a * kNumSFASubAtoms * 4,
                            kTmemStartColOfSFB + a * kNumSFBSubAtoms * 4);
                    };

#    pragma unroll
                    for (uint32_t a = 0; a < kNumKAtoms; ++a)
                    {
                        // A0 x W: only the very first instruction of the very
                        // first K block starts a fresh accumulator.
                        issue_pass(a, a0_base_lo, kTmemStartColOfSFA0, a > 0 or k_block_idx > 0);
                        // A1 x W always accumulates on top.  Single-pass mode
                        // drops the instruction rather than multiplying by a
                        // zeroed A1, so the benchmark measures the real cost of
                        // the second pass and not just its numerical effect.
                        if constexpr (kEnableResidualPass)
                            issue_pass(a, a1_base_lo, kTmemStartColOfSFA1, true);
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
        // Warp 2: SFB transpose into the layout UTCCP expects
        // ==================================================================
    }
    else if (warp_idx == 2)
    {
        auto transpose_atom = [&](uint32_t* smem_ptr) {
            uint32_t values[4];
#    pragma unroll
            for (uint32_t i = 0; i < 4; ++i)
                values[i] = deep_gemm::ptx::ld_shared(smem_ptr + i * 32 + lane_idx);
            __syncwarp();
            deep_gemm::ptx::st_shared(smem_ptr + lane_idx * 4,
                                      values[0],
                                      values[1],
                                      values[2],
                                      values[3]);
        };

        while (scheduler.get_next_block(m_block_idx, n_block_idx))
        {
            const auto num_total_k_blocks = deep_gemm::math::ceil_div(scheduler.current_shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx))
            {
                full_barriers[stage_idx]->wait(phase);

#    pragma unroll
                for (uint32_t i = 0; i < kNumKAtoms * kNumSFBSubAtoms; ++i)
                    transpose_atom(smem_sfb[stage_idx] + i * kNumUTCCPAlignedElems);
                cutlass::arch::fence_view_async_shared();

                transform_full_barriers[stage_idx]->arrive(0u);
            }
        }
        // ==================================================================
        // Warps 3..2+T: transform producers running Algorithm 1
        // ==================================================================
    }
    else if (warp_idx >= kFirstTransformWarp and warp_idx < kNumNonEpilogueWarps)
    {
        const uint32_t transform_tid = threadIdx.x - kFirstTransformWarp * 32;

        while (scheduler.get_next_block(m_block_idx, n_block_idx))
        {
            const auto num_total_k_blocks = deep_gemm::math::ceil_div(scheduler.current_shape_k, BLOCK_K);
            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx))
            {
                full_barriers[stage_idx]->wait(phase);

                transform::transform_a_tile<
                    BLOCK_M,
                    BLOCK_K,
                    kSwizzleAMode,
                    kSwizzleABMode,
                    kNumTransformThreads,
                    kScalePolicy,
                    kEnableResidualPass>(
                    smem_a[stage_idx],
                    smem_a0[stage_idx],
                    smem_a1[stage_idx],
                    smem_sfa0[stage_idx],
                    smem_sfa1[stage_idx],
                    transform_tid);

                // Make the FP4 tiles and scales visible to UTCCP/UMMA, which read
                // shared memory through the async proxy.
                cutlass::arch::fence_view_async_shared();
                transform_full_barriers[stage_idx]->arrive(0u);
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
        DG_TRAP_ONLY_DEVICE_ASSERT(deep_gemm::ptx::ld_shared(tmem_ptr_in_smem) == 0);

        uint32_t tma_stage_idx = 0;
        while (scheduler.get_next_block(m_block_idx, n_block_idx))
        {
            const auto accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
            const auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;

            tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx);
            deep_gemm::ptx::tcgen05_after_thread_sync();

            const auto base_m_idx = scheduler.template get_global_idx<
                (not deep_gemm::is_m_grouped_contiguous(kGemmType)),
                deep_gemm::sched::IndexType::MN>(
                shape_m,
                BLOCK_M,
                m_block_idx);
            const auto base_n_idx = n_block_idx * BLOCK_N;

            // Per-expert FP32 weight scale, deliberately kept out of the mainloop:
            // one multiply per output element beats one per MMA operand.
            uint32_t expert_idx = 0;
            if constexpr (kGemmType == GemmType::MGroupedContiguous)
            {
                expert_idx = static_cast<uint32_t>(max(0, grouped_layout[m_block_idx * BLOCK_M]));
            }
            else if constexpr (kGemmType == GemmType::MGroupedMasked)
            {
                expert_idx = scheduler.current_group_idx;
            }
            const float global_scale =
                weight_global_scales == nullptr ? 1.0f : __ldg(weight_global_scales + expert_idx);

            epilogue::sm100_store_cd_gscale<
                BLOCK_M,
                BLOCK_N,
                STORE_BLOCK_M,
                STORE_BLOCK_N,
                kSwizzleCDMode,
                kNumTMAStoreStages,
                kNumUMMAStoreThreads,
                kGemmType,
                cd_dtype_t>(smem_cd, tma_stage_idx, accum_stage_idx * UMMA_N, base_m_idx, base_n_idx, scheduler.current_group_idx, global_scale, epilogue_warp_idx, lane_idx, tmem_empty_barriers[accum_stage_idx], tensor_map_cd);
        }
    }

    __syncthreads();
    if (warp_idx == 0)
        Allocator().free(0, kNumTmemCols);

#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only supports sm_100f / sm_103f");
#endif
}

}   // namespace nvfp4_gemm

#pragma clang diagnostic pop

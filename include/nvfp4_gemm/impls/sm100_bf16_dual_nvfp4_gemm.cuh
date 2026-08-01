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

#include <nvfp4_gemm/epilogue/sm100_store_cd.cuh>
#include <nvfp4_gemm/ptx/tcgen05.cuh>
#include <nvfp4_gemm/transform/dual_nvfp4.cuh>

namespace nvfp4_gemm {

using transform::ScalePolicy;

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
template<uint32_t SHAPE_N, uint32_t SHAPE_K, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t kNumGroups, uint32_t kSwizzleAMode, uint32_t kSwizzleABMode, uint32_t kSwizzleCDMode, uint32_t kNumStages, uint32_t kNumTransformWarps, uint32_t kNumEpilogueThreads, uint32_t kNumSMs, uint32_t kNumForcedEpilogueStages, GemmType kGemmType, ScalePolicy kScalePolicy, bool kEnableResidualPass, bool kDirectGlobalA = false, uint32_t kNumBlocksPerGroup = 0, bool kSplitTransform = false, uint32_t kTimingProbe = 0, uint32_t kTailSplit = 0, uint32_t kNumShallowStages = 0, uint32_t kNumProducedStages = 0>
CUTLASS_GLOBAL void __launch_bounds__((3 + kNumTransformWarps) * 32 + kNumEpilogueThreads, 1)
    sm100_bf16_dual_nvfp4_gemm_impl(int* grouped_layout,
                                    const float* __restrict__ weight_global_scales,
                                    uint32_t                shape_m,
                                    uint32_t                shape_n,
                                    uint32_t                shape_k,
                                    const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                                    const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                                    const __grid_constant__ cute::TmaDescriptor tensor_map_sfb,
                                    const __grid_constant__ cute::TmaDescriptor tensor_map_cd,
                                    const __nv_bfloat16* __restrict__ a_gmem,
                                    uint32_t                          lda,
                                    unsigned long long*               probe_buf,
                                    float*                            tail_ws)
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
    NVFP4_STATIC_ASSERT(BLOCK_M == LAYOUT_AD_M, "Dual-NVFP4 GEMM requires BLOCK_M == 128");
    NVFP4_STATIC_ASSERT(BLOCK_K % UMMA_K == 0, "Block K must be divisible by UMMA K");
    NVFP4_STATIC_ASSERT(UMMA_N % 16 == 0 and 16 <= UMMA_N and UMMA_N <= 256, "Invalid UMMA N");
    // A partially-filled SF atom would leave the tail rows of SFB uninitialized,
    // since TMA only writes BLOCK_N of the SF_BLOCK_N rows UTCCP later reads.
    NVFP4_STATIC_ASSERT(BLOCK_N % 128 == 0, "Block N must be a multiple of the 128-row SF atom");
    // Packed FP4 with BLOCK_K elements per row must be exactly one swizzle atom.
    NVFP4_STATIC_ASSERT(kSwizzleABMode * 2 == BLOCK_K, "Packed-FP4 swizzle must cover one K block");
    NVFP4_STATIC_ASSERT(kSwizzleAMode == 128, "BF16 A tile assumes 128 B swizzle");

    // ---- Scale-factor geometry -------------------------------------------
    // One 128x4 SF atom = 128 rows x 4 E4M3 bytes = 512 B of SMEM -> 4 TMEM cols.
    constexpr uint32_t kNumUTCCPAlignedElems = 128;
    constexpr uint32_t SF_BLOCK_M            = math::constexpr_align(BLOCK_M, kNumUTCCPAlignedElems);
    constexpr uint32_t SF_BLOCK_N            = math::constexpr_align(BLOCK_N, kNumUTCCPAlignedElems);
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
    // A forced value below the derived one shrinks the TMEM footprint; with the
    // slim tile that is what lets two CTAs share one SM's 512 columns.
    constexpr uint32_t kNumDerivedEpilogueStages = (UMMA_N * 2 + kNumSFTmemCols <= 512) ? 2 : 1;
    constexpr uint32_t kNumEpilogueStages        = kNumForcedEpilogueStages > 0
                                                       ? kNumForcedEpilogueStages
                                                       : kNumDerivedEpilogueStages;
    NVFP4_STATIC_ASSERT(kNumEpilogueStages <= kNumDerivedEpilogueStages,
                        "Forced epilogue stages exceed the tensor memory budget");
    // Per-buffer pipeline depth (R35): buffers whose tenants release early
    // (A via transform_full, A0/A1/SFA via the UMMA-read commit) may run
    // SHALLOWER than the logical stage count, buying logical depth from the
    // SMEM budget.  0 means uniform (today's layout, byte-identical).  B and
    // SFB must stay at full depth: both release via the same
    // umma_arrive(empty) at read completion, so a shallow SFB would rebuild
    // the exact empty->tx->full loop the depth exists to divide.
    constexpr uint32_t kShallowStages = kNumShallowStages == 0 ? kNumStages : kNumShallowStages;
    // The transform's products live even shorter than A (write ~mid-transform
    // to UMMA-read completion), so A0/A1/SFA0/SFA1 get their own, smaller
    // depth -- which is what buys back the double-buffered cd64 epilogue.
    constexpr uint32_t kProducedStages = kNumProducedStages == 0 ? kShallowStages : kNumProducedStages;
    NVFP4_STATIC_ASSERT(kShallowStages >= 1 and kShallowStages <= kNumStages,
                        "Shallow depth must not exceed the logical stage count");
    NVFP4_STATIC_ASSERT(kProducedStages >= 1 and kProducedStages <= kShallowStages,
                        "Produced depth must not exceed the shallow depth");
    NVFP4_STATIC_ASSERT(kShallowStages == kNumStages or
                            (not kSplitTransform and kTailSplit == 0 and not kDirectGlobalA),
                        "Per-buffer depths exclude split/tail-split/direct-A in v1");
    constexpr uint32_t kNumTMAStoreStages   = 2;
    constexpr uint32_t STORE_BLOCK_M        = cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
    constexpr uint32_t STORE_BLOCK_N        = kSwizzleCDMode / sizeof(cd_dtype_t);
    constexpr uint32_t kNumUMMAStoreThreads = STORE_BLOCK_M;
    NVFP4_STATIC_ASSERT(kNumUMMAStoreThreads <= kNumEpilogueThreads, "Not enough epilogue threads");

    // ---- Tensor memory budget -------------------------------------------
    constexpr uint32_t kNumAccumTmemCols = UMMA_N * kNumEpilogueStages;
    constexpr uint32_t kNumTmemCols =
        utils::get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFTmemCols>();
    constexpr uint32_t kTmemStartColOfSFA0 = kNumAccumTmemCols;
    constexpr uint32_t kTmemStartColOfSFA1 = kTmemStartColOfSFA0 + kNumSFATmemCols;
    constexpr uint32_t kTmemStartColOfSFB  = kTmemStartColOfSFA1 + kNumSFATmemCols;
    NVFP4_STATIC_ASSERT(kNumAccumTmemCols + kNumSFTmemCols <= 512, "Tensor memory overflow");

    // ---- Shared memory budget -------------------------------------------
    constexpr uint32_t SMEM_CD_SIZE_PER_STAGE  = STORE_BLOCK_M * STORE_BLOCK_N * sizeof(cd_dtype_t);
    constexpr uint32_t SMEM_CD_SIZE            = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
    constexpr uint32_t SMEM_A_SIZE_PER_STAGE   = BLOCK_M * BLOCK_K * sizeof(__nv_bfloat16);
    constexpr uint32_t SMEM_A0_SIZE_PER_STAGE  = BLOCK_M * BLOCK_K / 2;
    constexpr uint32_t SMEM_B_SIZE_PER_STAGE   = BLOCK_N * BLOCK_K / 2;
    constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = kNumKAtoms * SF_BLOCK_M * 4;
    constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = kNumKAtoms * SF_BLOCK_N * 4;
    NVFP4_STATIC_ASSERT(SMEM_A_SIZE_PER_STAGE % 1024 == 0 and SMEM_A0_SIZE_PER_STAGE % 1024 == 0 and
                            SMEM_B_SIZE_PER_STAGE % 1024 == 0,
                        "A/B shared memory must be 1024 B aligned");

    // ---- Thread roles ----------------------------------------------------
    constexpr uint32_t kNumTransformThreads   = kNumTransformWarps * 32;
    constexpr uint32_t kNumNonEpilogueWarps   = 3 + kNumTransformWarps;
    constexpr uint32_t kNumNonEpilogueThreads = kNumNonEpilogueWarps * 32;
    constexpr uint32_t kFirstTransformWarp    = 3;
    // Work is dealt in 16-element blocks; a thread may own as little as half a
    // scale atom (2 blocks), which is what admits 16 transform warps.
    NVFP4_STATIC_ASSERT(BLOCK_M * BLOCK_K / 16 % kNumTransformThreads == 0,
                        "Transform threads must evenly divide the A tile");
    NVFP4_STATIC_ASSERT(kTailSplit == 0 or (SHAPE_K / BLOCK_K) % kTailSplit == 0,
                        "Tail split must divide the k-block count evenly");

    const auto warp_idx = cutlass::canonical_warp_idx_sync();
    const auto lane_idx = ptx::get_lane_idx();

    // kTimingProbe == 3: per-role clock accounting.  Lane 0 of one warp per
    // role accumulates wait/work cycles locally and flushes once per tile, so
    // the instrumentation itself stays off the fast path.
    unsigned long long probe_wait = 0, probe_work = 0, probe_iters = 0;
    auto probe_flush = [&](uint32_t role) {
        if constexpr (kTimingProbe == 3)
        {
            if (probe_buf != nullptr and lane_idx == 0)
            {
                atomicAdd(probe_buf + role * 2 + 0, probe_wait);
                atomicAdd(probe_buf + role * 2 + 1, probe_work);
                // Slot 8 keeps the all-role sum; 9+role is per role, so each
                // role's wait+work normalizes by its OWN iteration count (the
                // epilogue counts tiles, the others count k-blocks).
                atomicAdd(probe_buf + 8, probe_iters);
                atomicAdd(probe_buf + 9 + role, probe_iters);
                probe_wait = probe_work = probe_iters = 0;
            }
        }
    };

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
    const auto shape_sfb_k = math::ceil_div(shape_k, UMMA_K);

    extern __shared__ __align__(1024) uint8_t smem_buffer[];

    // Layout: [CD stages][A stages][A0 stages][A1 stages][B stages][SFA0][SFA1][SFB][barriers]
    auto               smem_cd         = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cd_dtype_t*>(smem_buffer + i * SMEM_CD_SIZE_PER_STAGE);
    });
    constexpr uint32_t kOffsetA        = SMEM_CD_SIZE;
    constexpr uint32_t kOffsetA0       = kOffsetA + kShallowStages * SMEM_A_SIZE_PER_STAGE;
    constexpr uint32_t kOffsetA1       = kOffsetA0 + kProducedStages * SMEM_A0_SIZE_PER_STAGE;
    constexpr uint32_t kOffsetB        = kOffsetA1 + kProducedStages * SMEM_A0_SIZE_PER_STAGE;
    constexpr uint32_t kOffsetSFA0     = kOffsetB + kNumStages * SMEM_B_SIZE_PER_STAGE;
    constexpr uint32_t kOffsetSFA1     = kOffsetSFA0 + kProducedStages * SMEM_SFA_SIZE_PER_STAGE;
    constexpr uint32_t kOffsetSFB      = kOffsetSFA1 + kProducedStages * SMEM_SFA_SIZE_PER_STAGE;
    constexpr uint32_t kOffsetBarriers = kOffsetSFB + kNumStages * SMEM_SFB_SIZE_PER_STAGE;

    auto smem_a    = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_bfloat16*>(smem_buffer + kOffsetA + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_a0   = utils::PatternVisitor([&](const uint32_t& i) {
        return smem_buffer + kOffsetA0 + i * SMEM_A0_SIZE_PER_STAGE;
    });
    auto smem_a1   = utils::PatternVisitor([&](const uint32_t& i) {
        return smem_buffer + kOffsetA1 + i * SMEM_A0_SIZE_PER_STAGE;
    });
    auto smem_b    = utils::PatternVisitor([&](const uint32_t& i) {
        return smem_buffer + kOffsetB + i * SMEM_B_SIZE_PER_STAGE;
    });
    auto smem_sfa0 = utils::PatternVisitor([&](const uint32_t& i) {
        return smem_buffer + kOffsetSFA0 + i * SMEM_SFA_SIZE_PER_STAGE;
    });
    auto smem_sfa1 = utils::PatternVisitor([&](const uint32_t& i) {
        return smem_buffer + kOffsetSFA1 + i * SMEM_SFA_SIZE_PER_STAGE;
    });
    auto smem_sfb  = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<uint32_t*>(smem_buffer + kOffsetSFB + i * SMEM_SFB_SIZE_PER_STAGE);
    });

    // ---- Barriers --------------------------------------------------------
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_buffer + kOffsetBarriers);
    auto full_barriers     = utils::PatternVisitor(
        [=](const uint32_t& i) { return barrier_start_ptr + i; });
    // A-only arrival: the transform needs just A, and the TMA engine lands
    // A (issued first) ahead of B/SFB -- waiting for the combined barrier
    // put their tx tail on the transform's critical chain (R36).
    auto full_a_barriers = utils::PatternVisitor(
        [=](const uint32_t& i) { return barrier_start_ptr + kNumStages + i; });
    auto empty_barriers = utils::PatternVisitor(
        [=](const uint32_t& i) { return barrier_start_ptr + kNumStages * 2 + i; });
    // Signalled once the stage's A0/A1/SFA0/SFA1 are complete *and* SFB has been
    // transposed -- the MMA warp's single wait covers both producers.
    auto transform_full_barriers = utils::PatternVisitor(
        [=](const uint32_t& i) { return barrier_start_ptr + kNumStages * 3 + i; });
    // A0/SFA0 readiness, for the split-transform pipeline: MMA's first pass
    // waits this instead of the full transform barrier.
    auto a0_full_barriers = utils::PatternVisitor(
        [=](const uint32_t& i) { return barrier_start_ptr + kNumStages * 4 + i; });
    auto tmem_full_barriers = utils::PatternVisitor(
        [=](const uint32_t& i) { return barrier_start_ptr + kNumStages * 5 + i; });
    auto tmem_empty_barriers = utils::PatternVisitor(
        [=](const uint32_t& i) { return barrier_start_ptr + kNumStages * 5 + kNumEpilogueStages + i; });
    auto tmem_ptr_in_smem = reinterpret_cast<uint32_t*>(
        barrier_start_ptr + kNumStages * 5 + kNumEpilogueStages * 2);

    if (warp_idx == 1 and cute::elect_one_sync())
    {
#    pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++i)
        {
            full_barriers[i]->init(1);
            full_a_barriers[i]->init(1);
            empty_barriers[i]->init(1);
            // Split mode: pass 0 needs A0/SFA0 (transform) plus SFB (warp 2),
            // so those arrivals move to the A0 barrier and the full barrier
            // counts only the residual pass. Non-split keeps the single merge.
            transform_full_barriers[i]->init(
                kSplitTransform ? kNumTransformThreads : kNumTransformThreads + 32);
            a0_full_barriers[i]->init(kSplitTransform ? kNumTransformThreads + 32 : 1);
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
    // The L2 grouping width defaults to the usage heuristic; a non-zero
    // override pins it for measurement.
    constexpr uint32_t kBPG = kNumBlocksPerGroup != 0
                                  ? kNumBlocksPerGroup
                                  : sched::get_num_1d_blocks_per_group<BLOCK_M, BLOCK_N, kNumSMs>();
    auto scheduler = sched::Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumSMs, kBPG, kTailSplit>(
        shape_m,
        shape_n,
        grouped_layout,
        math::ceil_div(shape_k, BLOCK_K));

    uint32_t stage_idx = 0, phase = 0;
    // Shallow-slot cursor (logical t mod kShallowStages) plus a shadow of the
    // (stage, parity) cursor running kShallowStages behind, for the recycle
    // waits.  The shadow's init is logical t - kShallowStages, so its waits
    // fall through for the first kShallowStages iterations by the same
    // init-parity trick the empty wait's `phase ^ 1` prologue uses.  NOTE:
    // slot_sh is t mod kShallowStages, NOT stage_idx mod kShallowStages --
    // the latter repeats slot 0 across the wrap and corrupts a live tenant.
    uint32_t slot_sh  = 0;
    uint32_t stage_sh = kNumStages - kShallowStages, phase_sh = 1;
    uint32_t slot_pr  = 0;
    uint32_t stage_pr = kNumStages - kProducedStages, phase_pr = 1;
    auto     advance_pipeline = [&](uint32_t& k_block_idx) {
        ++k_block_idx;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
        slot_sh  = slot_sh == kShallowStages - 1 ? 0 : slot_sh + 1;
        stage_sh = stage_sh == kNumStages - 1 ? 0 : stage_sh + 1;
        phase_sh ^= stage_sh == 0;
        slot_pr  = slot_pr == kProducedStages - 1 ? 0 : slot_pr + 1;
        stage_pr = stage_pr == kNumStages - 1 ? 0 : stage_pr + 1;
        phase_pr ^= stage_pr == 0;
    };

    // ==================================================================
    // Warp 0: TMA producer
    // ==================================================================
    if (warp_idx == 0 and cute::elect_one_sync())
    {
        while (scheduler.get_next_block(m_block_idx, n_block_idx))
        {
            // Tile-invariant coordinates, hoisted out of the K loop.  The
            // expert offset inside `get_global_idx` is a dependent load of
            // `m_indices` from global memory, and the asm barriers in the loop
            // body otherwise force nvcc to reissue it on every K step of the
            // producer's critical path -- twice.
            const uint32_t m_idx = scheduler.template get_global_idx<(kGemmType == GemmType::MGroupedMasked)>(
                shape_m,
                BLOCK_M,
                m_block_idx);
            const uint32_t n_idx = scheduler.template get_global_idx<true>(shape_n, BLOCK_N, n_block_idx, m_block_idx);
            const uint32_t sfb_n_idx  = n_block_idx * BLOCK_N;
            const uint32_t sfb_k_base = scheduler.template get_global_idx<true>(shape_sfb_k, 1, 0, m_block_idx);

            // Piece bounds hoisted into locals for the same reason as the
            // coordinate hoist above (O5): the asm barriers in the loop body
            // force a reload of anything read through the scheduler object on
            // every K step of the critical path.
            const uint32_t k_lo = scheduler.k_block_lo, k_hi = scheduler.k_block_hi;
            for (uint32_t k_block_idx = k_lo; k_block_idx < k_hi; advance_pipeline(k_block_idx))
            {
                const auto probe_t0 = kTimingProbe == 3 ? clock64() : 0;
                empty_barriers[stage_idx]->wait(phase ^ 1);
                if constexpr (kShallowStages != kNumStages)
                    // The A slot (t mod shallow) recycles from logical
                    // t - shallow; its consumer releases via the FINAL
                    // transform_full.  (Never a0_full: split mode re-reads
                    // smem_a after arrive0 -- excluded in v1, documented so
                    // it stays excluded.)  Shadow parity, no ^1.
                    transform_full_barriers[stage_sh]->wait(phase_sh);
                const auto probe_t1 = kTimingProbe == 3 ? clock64() : 0;

                const uint32_t k_idx = k_block_idx * BLOCK_K;

                // A is BF16 and K-major: 2 bytes per element, and no SFA to load
                // -- the whole point of the fused path is that A's scales are
                // produced on chip.  In direct-global mode the transform warps
                // read A via LDG instead, so its 32 KB never crosses the TMA
                // ingress port and the copy is skipped outright.
                if constexpr (not kDirectGlobalA)
                {
                    tma::copy<BLOCK_K, BLOCK_M, kSwizzleAMode, __nv_bfloat16>(
                        &tensor_map_a,
                        full_a_barriers[stage_idx],
                        smem_a[slot_sh],
                        k_idx,
                        m_idx);
                    full_a_barriers[stage_idx]->arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE);
                }
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

                uint32_t num_arrival_bytes = SMEM_B_SIZE_PER_STAGE;

                const uint32_t sfb_k_idx = sfb_k_base + k_idx / UMMA_K;
                tma::copy<BLOCK_N, kNumKAtoms, 0>(
                    &tensor_map_sfb,
                    full_barriers[stage_idx],
                    smem_sfb[stage_idx],
                    sfb_n_idx,
                    sfb_k_idx);
                num_arrival_bytes += BLOCK_N * kNumKAtoms * sizeof(uint32_t);

                full_barriers[stage_idx]->arrive_and_expect_tx(num_arrival_bytes);
                if constexpr (kTimingProbe == 3)
                {
                    probe_wait += probe_t1 - probe_t0;
                    probe_work += clock64() - probe_t1;
                    ++probe_iters;
                }
            }
            probe_flush(0);
        }
        // This CTA has issued its last global read (B/SFB always; A too unless
        // direct-global mode keeps the transform warps LDG'ing it).  Fire the
        // PDL trigger so the dependent grid launches once the SLOWEST CTA
        // passes this point: under wave-ceil imbalance the early-finishing
        // CTAs' SMs then host the successor's prologue instead of idling.
        // Contract note (same as DeepGEMM): the epilogue still reads
        // `global_scales`, TMA-stores D, and (tail-split mode) exchanges FP32
        // partials through `tail_ws` after this -- the launcher must not let a
        // dependent overwrite those; the host parity-double-buffers tail_ws
        // for exactly that reason.
        if constexpr (not kDirectGlobalA)
            cudaTriggerProgrammaticLaunchCompletion();
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
        auto a0_desc = mma::sm100::make_packed_fp4_desc<BLOCK_M, BLOCK_K, kSwizzleABMode>(smem_a0[0]);
        auto a1_desc = a0_desc;
        auto b_desc  = mma::sm100::make_packed_fp4_desc<BLOCK_N, BLOCK_K, kSwizzleABMode>(smem_b[0]);
        mma::sm100::replace_smem_desc_addr(a1_desc, smem_a1[0]);

        // Stage selection by lane broadcast, as in DeepGEMM: lane `s` holds the
        // descriptor for stage `s`, and `exchange` picks it in one shuffle.
        // A0/A1 tables span the SHALLOW slot count and are picked by the
        // slot cursor below; a stage-indexed lane past the shallow count
        // would point straight into A1's / B's regions.
        uint32_t a0_desc_lo = lane_idx < kProducedStages ? a0_desc.lo + lane_idx * SMEM_A0_SIZE_PER_STAGE / 16 : 0u;
        uint32_t a1_desc_lo = lane_idx < kProducedStages ? a1_desc.lo + lane_idx * SMEM_A0_SIZE_PER_STAGE / 16 : 0u;
        uint32_t b_desc_lo  = lane_idx < kNumStages ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16 : 0u;

        while (scheduler.get_next_block(m_block_idx, n_block_idx))
        {
            const auto accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
            const auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;
            tmem_empty_barriers[accum_stage_idx]->wait(accum_phase_idx ^ 1);
            ptx::tcgen05_after_thread_sync();

            const uint32_t k_lo = scheduler.k_block_lo, k_hi = scheduler.k_block_hi;
#    pragma unroll 2
            for (uint32_t k_block_idx = k_lo; k_block_idx < k_hi; advance_pipeline(k_block_idx))
            {
                // Split pipeline: pass 0 launches on A0/SFA0/SFB readiness while
                // the transform warps still quantize the residual.
                const auto probe_t0 = kTimingProbe == 3 ? clock64() : 0;
                if constexpr (kSplitTransform and kEnableResidualPass)
                    a0_full_barriers[stage_idx]->wait(phase);
                else
                    transform_full_barriers[stage_idx]->wait(phase);
                // B/SFB landing gate: the UMMA reads B from SMEM right after
                // issue, and the transform no longer vouches for it.  Lands
                // well before the transform finishes -- fast-path.
                full_barriers[stage_idx]->wait(phase);
                const auto probe_t1 = kTimingProbe == 3 ? clock64() : 0;
                ptx::tcgen05_after_thread_sync();

                const auto a0_base_lo = ptx::exchange(a0_desc_lo, slot_pr);
                const auto a1_base_lo = ptx::exchange(a1_desc_lo, slot_pr);
                const auto b_base_lo  = ptx::exchange(b_desc_lo, stage_idx);

                if (cute::elect_one_sync())
                {
                    // Copy all three scale operands into tensor memory.  SFA0 and
                    // SFA1 were written pre-transposed by the transform warps;
                    // only SFB needed warp 2's shuffle.
                    //
                    // (Interleaving these copies with the UMMAs below -- copy
                    // atom 0's scales, issue atom 0's passes, copy atom 1's --
                    // was measured at exactly zero effect on every shape, so
                    // the hardware already overlaps UTCCP drain with UMMA
                    // execution; keep the simpler front-loaded order.)
                    auto utccp_sf = [&](uint8_t* smem_ptr, const uint32_t& tmem_col) {
                        mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
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
                            utccp_sf(smem_sfa0[slot_pr] + byte_off, kTmemStartColOfSFA0 + col);
                            // kTimingProbe == 1 drops the SFA1 copies (WRONG
                            // RESULTS -- timing decomposition only).
                            if constexpr (kEnableResidualPass and not kSplitTransform and kTimingProbe != 1)
                                utccp_sf(smem_sfa1[slot_pr] + byte_off, kTmemStartColOfSFA1 + col);
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
                        desc_a.lo   = mma::sm100::advance_packed_fp4_desc_lo(a_base_lo, a * UMMA_K);
                        auto desc_b = b_desc;
                        desc_b.lo   = mma::sm100::advance_packed_fp4_desc_lo(b_base_lo, a * UMMA_K);
                        const auto runtime_desc =
                            mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, 0, 0);
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
                        // A0 x W: only the very first instruction of the piece's
                        // first K block starts a fresh accumulator.  Keyed to
                        // k_block_lo, not zero -- a tail k-piece starting at
                        // k_lo > 0 must NOT accumulate onto the stale contents
                        // of its recycled accumulator stage.
                        issue_pass(a, a0_base_lo, kTmemStartColOfSFA0, a > 0 or k_block_idx > k_lo);
                        // A1 x W always accumulates on top.  Single-pass mode
                        // drops the instruction rather than multiplying by a
                        // zeroed A1, so the benchmark measures the real cost of
                        // the second pass and not just its numerical effect.
                        // Under the split pipeline the residual pass issues in
                        // the second phase below instead.  kTimingProbe == 2
                        // drops the second UMMA (WRONG RESULTS -- timing only).
                        if constexpr (kEnableResidualPass and not kSplitTransform and kTimingProbe != 2)
                            issue_pass(a, a1_base_lo, kTmemStartColOfSFA1, true);
                    }
                }
                __syncwarp();

                // Split pipeline, phase 2: wait for the residual quantization
                // and issue the A1 passes on top of the same accumulator.
                if constexpr (kSplitTransform and kEnableResidualPass)
                {
                    transform_full_barriers[stage_idx]->wait(phase);
                    ptx::tcgen05_after_thread_sync();
                    if (cute::elect_one_sync())
                    {
                        auto utccp_sf = [&](uint8_t* smem_ptr, const uint32_t& tmem_col) {
                            mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
                            cute::SM100_UTCCP_4x32dp128bit_1cta::copy(sf_desc, tmem_col);
                        };
                        auto issue_pass = [&](const uint32_t& a, const uint32_t& a_base_lo, const uint32_t& sfa_tmem_base) {
                            auto desc_a = a0_desc;
                            desc_a.lo   = mma::sm100::advance_packed_fp4_desc_lo(a_base_lo, a * UMMA_K);
                            auto desc_b = b_desc;
                            desc_b.lo   = mma::sm100::advance_packed_fp4_desc_lo(b_base_lo, a * UMMA_K);
                            const auto runtime_desc =
                                mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, 0, 0);
                            ptx::SM100_MMA_MXF4NVF4_SS::fma(
                                desc_a,
                                desc_b,
                                accum_stage_idx * UMMA_N,
                                true,
                                runtime_desc,
                                sfa_tmem_base + a * kNumSFASubAtoms * 4,
                                kTmemStartColOfSFB + a * kNumSFBSubAtoms * 4);
                        };
#    pragma unroll
                        for (uint32_t a = 0; a < kNumKAtoms; ++a)
                        {
#    pragma unroll
                            for (uint32_t i = 0; i < kNumSFASubAtoms; ++i)
                                utccp_sf(smem_sfa1[stage_idx] +
                                             (a * kNumSFASubAtoms + i) * kNumUTCCPAlignedElems * 4,
                                         kTmemStartColOfSFA1 + (a * kNumSFASubAtoms + i) * 4);
                            issue_pass(a, a1_base_lo, kTmemStartColOfSFA1);
                        }
                    }
                    __syncwarp();
                }

                cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(empty_barriers[stage_idx]));
                if (k_block_idx == k_hi - 1)
                    cutlass::arch::umma_arrive(reinterpret_cast<uint64_t*>(tmem_full_barriers[accum_stage_idx]));
                __syncwarp();
                if constexpr (kTimingProbe == 3)
                {
                    probe_wait += probe_t1 - probe_t0;
                    probe_work += clock64() - probe_t1;
                    ++probe_iters;
                }
            }
            probe_flush(2);
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
                values[i] = ptx::ld_shared(smem_ptr + i * 32 + lane_idx);
            __syncwarp();
            ptx::st_shared(smem_ptr + lane_idx * 4,
                           values[0],
                           values[1],
                           values[2],
                           values[3]);
        };

        while (scheduler.get_next_block(m_block_idx, n_block_idx))
        {
            const uint32_t k_lo = scheduler.k_block_lo, k_hi = scheduler.k_block_hi;
            for (uint32_t k_block_idx = k_lo; k_block_idx < k_hi; advance_pipeline(k_block_idx))
            {
                full_barriers[stage_idx]->wait(phase);

#    pragma unroll
                for (uint32_t i = 0; i < kNumKAtoms * kNumSFBSubAtoms; ++i)
                    transpose_atom(smem_sfb[stage_idx] + i * kNumUTCCPAlignedElems);
                cutlass::arch::fence_view_async_shared();

                if constexpr (kSplitTransform)
                    a0_full_barriers[stage_idx]->arrive();
                else
                    transform_full_barriers[stage_idx]->arrive();
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
            // Direct mode reads global A per tile; contiguous activations are
            // one flat tensor, so the row is just the tile's m index.
            const uint32_t tile_m_idx = m_block_idx * BLOCK_M;

            const uint32_t k_lo = scheduler.k_block_lo, k_hi = scheduler.k_block_hi;
            for (uint32_t k_block_idx = k_lo; k_block_idx < k_hi; advance_pipeline(k_block_idx))
            {
                const auto probe_t0 = kTimingProbe == 3 ? clock64() : 0;
                if constexpr (kDirectGlobalA)
                    full_barriers[stage_idx]->wait(phase);
                else
                    full_a_barriers[stage_idx]->wait(phase);
                if constexpr (kProducedStages != kNumStages)
                    // The A0/A1/SFA slot recycles from logical t - produced:
                    // wait its UMMA reads (the empty commit covers UTCCP and
                    // both MMA passes).  Fast-path in steady state.
                    empty_barriers[stage_pr]->wait(phase_pr);
                const auto probe_t1 = kTimingProbe == 3 ? clock64() : 0;

                if constexpr (kSplitTransform and kEnableResidualPass and not kDirectGlobalA)
                {
                    transform::transform_a_tile_split<
                        BLOCK_M,
                        BLOCK_K,
                        kSwizzleAMode,
                        kSwizzleABMode,
                        kNumTransformThreads,
                        kScalePolicy>(
                        smem_a[stage_idx],
                        smem_a0[stage_idx],
                        smem_a1[stage_idx],
                        smem_sfa0[stage_idx],
                        smem_sfa1[stage_idx],
                        transform_tid,
                        [&] {
                            cutlass::arch::fence_view_async_shared();
                            a0_full_barriers[stage_idx]->arrive();
                        });
                }
                else if constexpr (kDirectGlobalA)
                    transform::transform_a_tile_gmem<
                        BLOCK_M,
                        BLOCK_K,
                        kSwizzleABMode,
                        kNumTransformThreads,
                        kScalePolicy,
                        kEnableResidualPass>(
                        a_gmem + static_cast<uint64_t>(tile_m_idx) * lda + k_block_idx * BLOCK_K,
                        lda,
                        smem_a0[stage_idx],
                        smem_a1[stage_idx],
                        smem_sfa0[stage_idx],
                        smem_sfa1[stage_idx],
                        transform_tid);
                // kTimingProbe == 4 drops the transform compute entirely
                // (WRONG RESULTS -- measures the pipeline's transform-free
                // ceiling, i.e. the upper bound any transform speedup can
                // reach).  Barrier traffic stays intact.
                else if constexpr (kTimingProbe != 4)
                    transform::transform_a_tile<
                        BLOCK_M,
                        BLOCK_K,
                        kSwizzleAMode,
                        kSwizzleABMode,
                        kNumTransformThreads,
                        kScalePolicy,
                        kEnableResidualPass>(
                        smem_a[slot_sh],
                        smem_a0[slot_pr],
                        smem_a1[slot_pr],
                        smem_sfa0[slot_pr],
                        smem_sfa1[slot_pr],
                        transform_tid);

                // Make the FP4 tiles and scales visible to UTCCP/UMMA, which read
                // shared memory through the async proxy.
                cutlass::arch::fence_view_async_shared();
                transform_full_barriers[stage_idx]->arrive();
                if constexpr (kTimingProbe == 3)
                {
                    if (warp_idx == kFirstTransformWarp)
                    {
                        probe_wait += probe_t1 - probe_t0;
                        probe_work += clock64() - probe_t1;
                        ++probe_iters;
                    }
                }
            }
            if (warp_idx == kFirstTransformWarp)
                probe_flush(1);
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
        while (scheduler.get_next_block(m_block_idx, n_block_idx))
        {
            const auto accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
            const auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;

            const auto probe_t0 = kTimingProbe == 3 ? clock64() : 0;
            if constexpr (kTimingProbe == 5)
            {
                // Fake-helper probe (results CORRECT): burn a transform-shaped
                // op stream (emulated bf16x2 mul + E2M1 cvt round trip, the
                // same FMA/CVT pipe mix as the hot loop) while polling.  If
                // this alone slows the kernel, the SM has no spare issue
                // capacity during transform bursts and work-stealing is dead.
                uint32_t sink = threadIdx.x | 0x3c003c00u;
                while (not tmem_full_barriers[accum_stage_idx]->try_wait(accum_phase_idx))
                {
#    pragma unroll
                    for (int burn = 0; burn < 32; ++burn)
                    {
                        __nv_bfloat162 x = *reinterpret_cast<__nv_bfloat162*>(&sink);
                        x                = __hmul2(x, x);
                        x                = __hsub2(x, __float2bfloat162_rn(0.25f));
                        sink ^= ptx::cvt_e2m1x2_bf16x2(*reinterpret_cast<uint32_t*>(&x));
                        sink += ptx::cvt_bf16x2_e2m1x2(sink & 0xffu);
                        sink |= 0x3c003c00u;
                    }
                }
                if (sink == 0xdeadbeefu)
                    __trap();
            }
            else
                tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx);
            const auto probe_t1 = kTimingProbe == 3 ? clock64() : 0;
            ptx::tcgen05_after_thread_sync();

            // Activations in the contiguous layout are one flat tensor, so the
            // expert offset applies only to the masked layout.
            const auto base_m_idx =
                scheduler.template get_global_idx<(kGemmType != GemmType::MGroupedContiguous)>(
                    shape_m,
                    BLOCK_M,
                    m_block_idx);
            const auto base_n_idx = n_block_idx * BLOCK_N;

            // Per-expert FP32 weight scale, deliberately kept out of the mainloop:
            // one multiply per output element beats one per MMA operand.
            const uint32_t expert_idx = scheduler.get_expert_idx(m_block_idx);
            const float    global_scale =
                weight_global_scales == nullptr ? 1.0f : __ldg(weight_global_scales + expert_idx);

            auto store_normal = [&] {
                epilogue::sm100_store_cd_gscale<
                    BLOCK_M,
                    BLOCK_N,
                    STORE_BLOCK_M,
                    STORE_BLOCK_N,
                    kSwizzleCDMode,
                    kNumTMAStoreStages,
                    kNumUMMAStoreThreads,
                    kGemmType,
                    cd_dtype_t>(smem_cd, tma_stage_idx, accum_stage_idx * UMMA_N, base_m_idx, base_n_idx, scheduler.current_group_idx, global_scale, epilogue_warp_idx, static_cast<uint32_t>(warp_idx) % 4u, lane_idx, tmem_empty_barriers[accum_stage_idx], tensor_map_cd);
            };
            if constexpr (kTailSplit > 1)
            {
                if (scheduler.tail_slot != sched::Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumSMs, kBPG, kTailSplit>::kNoTailSlot)
                    epilogue::sm100_store_cd_tail_split<
                        BLOCK_M,
                        BLOCK_N,
                        STORE_BLOCK_M,
                        STORE_BLOCK_N,
                        kSwizzleCDMode,
                        kNumTMAStoreStages,
                        kNumUMMAStoreThreads,
                        kGemmType,
                        kTailSplit,
                        kTimingProbe,
                        cd_dtype_t>(smem_cd, tma_stage_idx, accum_stage_idx * UMMA_N, base_m_idx, base_n_idx, scheduler.current_group_idx, global_scale, epilogue_warp_idx, static_cast<uint32_t>(warp_idx) % 4u, lane_idx, tmem_empty_barriers[accum_stage_idx], tensor_map_cd, tail_ws, scheduler.tail_slot, probe_buf);
                else
                    store_normal();
            }
            else
                store_normal();
            if constexpr (kTimingProbe == 3)
            {
                if (epilogue_warp_idx == 0)
                {
                    probe_wait += probe_t1 - probe_t0;
                    probe_work += clock64() - probe_t1;
                    ++probe_iters;
                }
            }
        }
        if (epilogue_warp_idx == 0)
            probe_flush(3);
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

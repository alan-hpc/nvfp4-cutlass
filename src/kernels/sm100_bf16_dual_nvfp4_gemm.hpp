#pragma once

// Host-side configuration, TMA descriptors and launch for the fused
// BF16 x NVFP4 dual-A grouped GEMM.

#include <algorithm>
#include <sstream>
#include <string>

#include <torch/python.h>

#include <nvfp4_gemm/common/types.cuh>

#include "../jit/compiler.hpp"
#include "../jit/launch.hpp"
#include "../runtime/device.hpp"
#include "../runtime/exception.hpp"
#include "../runtime/tma_desc.hpp"

namespace nvfp4_gemm {

/// Packed FP4 has no torch dtype yet, so it travels as int8 with two elements
/// per byte.
constexpr auto kPackedFP4 = torch::kInt8;

/// Static configuration of the fused kernel.
///
/// The tile shape is not swept by a heuristic: `128 x 256 x 128` is the shape
/// the algorithm doc measures, and it is also the largest N that leaves tensor
/// memory for the two extra SFA column groups dual-A needs (256 accumulator
/// columns + SFA0 8 + SFA1 8 + SFB 16 = 288 of 512).
struct DualNVFP4Config
{
    int block_m = 128;
    int block_n = 256;
    int block_k = 128;

    int swizzle_a_mode  = 128;   // BF16 A: 128 B per row per swizzle atom
    int swizzle_ab_mode = 64;    // packed FP4 A0/A1/W: 64 B per row
    int swizzle_cd_mode = 128;   // BF16 D

    int num_transform_warps  = 8;
    int num_epilogue_threads = 128;
    int num_stages           = 0;   // filled in by `compute_stages`
    int num_shallow_stages   = 0;   // 0 = uniform; else A depth (R35)
    int num_produced_stages  = 0;   // 0 = follows shallow; else A0/A1/SFA depth
    /// 0 = derive from the TMEM budget. Forcing 1 on a slim tile halves the
    /// TMEM footprint, which is what admits two CTAs per SM.
    int forced_epilogue_stages = 0;

    int num_threads() const
    {
        return (3 + num_transform_warps) * 32 + num_epilogue_threads;
    }

    int sf_atoms_per_k() const
    {
        return block_k / 64;
    }

    /// Bytes of shared memory consumed by one mainloop stage.
    int shallow_stages() const { return num_shallow_stages > 0 ? num_shallow_stages : num_stages; }
    int produced_stages() const { return num_produced_stages > 0 ? num_produced_stages : shallow_stages(); }

    /// A + A0/A1 + SFA0/SFA1: released early (transform_full / UMMA-read
    /// commit), so these may run shallower than the logical stage count.
    int smem_shallow_stage() const { return block_m * block_k * 2; }   // A, BF16

    int smem_produced_stage() const
    {
        const int sf_block_m = ((block_m + 127) / 128) * 128;
        return 2 * (block_m * block_k / 2) +               // A0 + A1, packed FP4
               2 * (sf_atoms_per_k() * sf_block_m * 4);    // SFA0 + SFA1
    }

    /// W + SFB: released by the same umma_arrive at read completion -- these
    /// MUST stay at the full logical depth (a shallow SFB rebuilds the very
    /// empty->tx->full loop the depth divides).
    int smem_deep_stage() const
    {
        const int sf_block_n = ((block_n + 127) / 128) * 128;
        return block_n * block_k / 2 +                     // W, packed FP4
               sf_atoms_per_k() * sf_block_n * 4;          // SFB
    }

    int smem_per_stage() const
    {
        return smem_shallow_stage() + smem_produced_stage() + smem_deep_stage();
    }

    int smem_epilogue() const
    {
        const int store_block_m = std::min(block_m, 128);
        const int store_block_n = swizzle_cd_mode / 2;   // BF16
        return store_block_m * store_block_n * 2 * /* TMA store stages */ 2;
    }

    int num_epilogue_stages() const
    {
        if (forced_epilogue_stages > 0)
            return forced_epilogue_stages;
        const int sf_cols = 2 * sf_atoms_per_k() * (((block_m + 127) / 128) * 4) +
                            sf_atoms_per_k() * (((block_n + 127) / 128) * 4);
        return (block_n * 2 + sf_cols <= 512) ? 2 : 1;
    }

    int smem_barriers() const
    {
        // Five per-stage rows: full, full-A, empty, transform-full, A0-full.
        return (5 * num_stages + 2 * num_epilogue_stages()) * 8 + 8;
    }

    int smem_size() const
    {
        return smem_epilogue() + num_stages * smem_deep_stage() +
               shallow_stages() * smem_shallow_stage() +
               produced_stages() * smem_produced_stage() + smem_barriers();
    }

    /// Pick the deepest pipeline that fits.
    ///
    /// Dual-A is unusually shared-memory hungry: a stage carries the BF16 A tile
    /// *and* both FP4 copies of it, so a 128x256x128 stage costs ~68 KB against
    /// ~40 KB for a plain NVFP4 GEMM. That caps the pipeline at 2 stages.
    ///
    /// `forced` (non-zero) pins the depth instead, for autotuning: a shallower
    /// pipeline than the maximum can win by freeing shared memory, so the search
    /// has to be able to ask for one.
    void compute_stages(int forced = 0)
    {
        if (forced > 0)
        {
            num_stages = forced;
            NVFP4_HOST_ASSERT_MSG(smem_size() <= kSmemCapacitySm100,
                                  "forced num_stages does not fit into shared memory");
            return;
        }
        for (int candidate = 8; candidate >= 1; --candidate)
        {
            num_stages = candidate;
            if (smem_size() <= kSmemCapacitySm100)
                return;
        }
        NVFP4_HOST_ASSERT_MSG(false, "cannot fit even a single stage into shared memory");
    }

    /// Reject configurations the kernel's static asserts would reject anyway.
    ///
    /// The search drives these from Python, so a bad point has to raise rather
    /// than build a kernel whose `NVFP4_STATIC_ASSERT`s fail at nvcc time (or,
    /// worse, one that runs and is silently wrong).
    void validate() const
    {
        NVFP4_HOST_ASSERT_MSG(block_m == 128, "BLOCK_M must be 128");
        NVFP4_HOST_ASSERT_MSG(block_n % 128 == 0 and block_n >= 128 and block_n <= 256,
                              "BLOCK_N must be 128 or 256");
        NVFP4_HOST_ASSERT_MSG(block_k % 64 == 0, "BLOCK_K must cover whole 64-element SF atoms");
        NVFP4_HOST_ASSERT_MSG(swizzle_ab_mode * 2 == block_k,
                              "packed-FP4 swizzle must cover exactly one K block");
        // The transform deals work in 16-element blocks; a thread owns at least
        // half a scale atom (2 blocks), so threads are capped at blocks / 2.
        const int num_transform_threads = num_transform_warps * 32;
        const int num_transform_blocks  = block_m * block_k / 16;
        NVFP4_HOST_ASSERT_MSG(num_transform_warps > 0 and
                                  num_transform_blocks % num_transform_threads == 0 and
                                  num_transform_threads <= num_transform_blocks,
                              "transform threads must evenly divide the A tile");
        // `kNumUMMAStoreThreads == STORE_BLOCK_M == 128` reads the accumulator.
        NVFP4_HOST_ASSERT_MSG(num_epilogue_threads >= std::min(block_m, 128) and
                                  num_epilogue_threads % 32 == 0,
                              "not enough epilogue threads");
        // The epilogue stores 16 B bank groups: STORE_BLOCK_N = swizzle / 2
        // BF16 elements must hold whole groups and divide the tile width.
        NVFP4_HOST_ASSERT_MSG((swizzle_cd_mode == 32 or swizzle_cd_mode == 64 or swizzle_cd_mode == 128) and
                                  block_n % (swizzle_cd_mode / 2) == 0,
                              "swizzle_cd_mode must be 64 or 128 and divide the N tile");
        const int sf_cols = 2 * sf_atoms_per_k() * (((block_m + 127) / 128) * 4) +
                            sf_atoms_per_k() * (((block_n + 127) / 128) * 4);
        NVFP4_HOST_ASSERT_MSG(block_n * num_epilogue_stages() + sf_cols <= 512,
                              "tensor memory overflow");
        NVFP4_HOST_ASSERT_MSG(smem_size() <= kSmemCapacitySm100, "shared memory overflow");
    }
};

inline const char* to_string(const GemmType& type)
{
    switch (type)
    {
    case GemmType::Normal:
        return "GemmType::Normal";
    case GemmType::MGroupedContiguous:
        return "GemmType::MGroupedContiguous";
    case GemmType::MGroupedMasked:
        return "GemmType::MGroupedMasked";
    }
    return "GemmType::Normal";
}

/// Warp-clock probe target: set by the API layer before a probed launch,
/// consumed by the very next launch. Not thread-safe -- probing is a
/// measurement mode, not a production path.
inline void* probe_buffer_for_launch = nullptr;

/// Tail-split workspace: per device, parity double-buffered (under PDL, launch
/// N+1 may overlap launch N's merge, so consecutive launches must not share
/// slots), intentionally leaked (a static tensor would outlive CUDA teardown).
/// Each half: 4 KB of per-tile ticket counters -- zeroed once at allocation,
/// reset by each tile's merger after use -- then one 128x256 FP32 slot per
/// possible piece. Single-stream use per device, like the probe buffer.
/// First tail-split call per device allocates: run it once outside any CUDA
/// graph capture (the JIT compile on first use imposes the same warmup).
inline void* tail_ws_cache[64] = {};
inline int   tail_ws_parity    = 0;
constexpr size_t kTailWsHalfBytes = 4096 + static_cast<size_t>(296) * 128 * 256 * 4;

/// Fused BF16 x NVFP4 m-grouped contiguous GEMM (MoE expert GEMM).
///
/// Args:
///   a:          BF16 activations, (M, K), K-major.
///   b:          packed E2M1 weights viewed as int8, (E * N, K / 2), K-major.
///   sfb:        E4M3 weight block scales in the 128x4 NVFP4 atom layout,
///               viewed as int32 of shape (E * K / 64, N).
///   gw:         FP32 per-expert weight global scale, (E,). Applied in the epilogue.
///   d:          BF16 output, (M, N).
///   m_indices:  int32 (M,), expert index per row.
inline void sm100_m_grouped_bf16_dual_nvfp4_gemm_contiguous(const torch::Tensor& a,
                                                            const torch::Tensor& b,
                                                            const torch::Tensor& sfb,
                                                            const torch::Tensor& gw,
                                                            const torch::Tensor& d,
                                                            const torch::Tensor& m_indices,
                                                            int                  num_groups,
                                                            int                  m,
                                                            int                  n,
                                                            int                  k,
                                                            const std::string&   scale_policy,
                                                            bool                 enable_residual_pass,
                                                            int                  tune_block_n,
                                                            int                  tune_transform_warps,
                                                            int                  tune_epilogue_threads,
                                                            int                  tune_num_stages,
                                                            int                  tune_swizzle_cd,
                                                            int                  tune_block_k,
                                                            int                  tune_epilogue_stages,
                                                            int                  tune_cluster,
                                                            int                  tune_direct_a,
                                                            int                  tune_l2_pin_weights,
                                                            int                  tune_no_pdl,
                                                            int                  tune_sched_group,
                                                            int                  tune_split_transform,
                                                            int                  tune_timing_probe,
                                                            int                  tune_tail_split,
                                                            int                  tune_shallow_stages,
                                                            int                  tune_produced_stages)
{
    DualNVFP4Config config;
    if (tune_block_n > 0)
        config.block_n = tune_block_n;
    // The packed-FP4 swizzle must cover exactly one K block, so it is tied.
    if (tune_block_k > 0)
    {
        config.block_k         = tune_block_k;
        config.swizzle_ab_mode = tune_block_k / 2;
    }
    if (tune_transform_warps > 0)
        config.num_transform_warps = tune_transform_warps;
    if (tune_epilogue_threads > 0)
        config.num_epilogue_threads = tune_epilogue_threads;
    // A 64 B CD swizzle halves the epilogue staging buffer, which is what lets
    // a third mainloop stage fit under the 227 KB shared-memory budget.
    if (tune_swizzle_cd > 0)
        config.swizzle_cd_mode = tune_swizzle_cd;
    if (tune_epilogue_stages > 0)
        config.forced_epilogue_stages = tune_epilogue_stages;

    // Steady state is the stage-lifecycle loop divided by the pipeline depth
    // (R33 in the opt log), so once the grid runs two-plus waves a half-size
    // k-block that fits SIX stages beats the full-size one at three: measured
    // on every multi-wave prefill point (largest: 16k down 47.0 -> 41.2 us),
    // while sub-wave shapes lose to the deeper prologue and keep the old
    // default. Same all-knobs-untouched gate as the rule below.
    const int  heuristic_tiles = ((m + config.block_m - 1) / config.block_m) *
                                ((n + config.block_n - 1) / config.block_n);
    const bool multiwave = heuristic_tiles >= 2 * device_runtime->get_num_sms();
    if (tune_swizzle_cd == 0 and tune_transform_warps == 0 and tune_block_n == 0 and
        tune_block_k == 0 and tune_num_stages == 0 and tune_shallow_stages == 0 and
        tune_produced_stages == 0 and multiwave)
    {
        if (k >= 1024)
        {
            // Deep-K multi-wave: per-buffer depths (R35).  A lives to the
            // transform's end (3 slots), the transform's products only to the
            // UMMA-read commit (2 slots), and the freed SMEM buys a FOURTH
            // logical stage at full-size k-blocks: the loop/depth law then
            // meets the lower bk128 transform recurrence (8k gate_up
            // 49.0 -> 47.5, 16k 62-65 -> 61.5 with the 16k jitter band gone).
            config.swizzle_cd_mode     = 64;
            config.num_transform_warps = 16;
            config.num_shallow_stages  = 3;
            config.num_produced_stages = 2;
            // compute_stages' deepest-fit picks 4 for this geometry.
        }
        else
        {
            // Short-K multi-wave: the half-size k-block at depth 6 wins (the
            // shallow config loses here -- only 4 k-blocks per tile).
            config.block_k             = 64;
            config.swizzle_ab_mode     = 32;
            // Sixteen warps at one block per thread (quarter-atom slots).
            config.num_transform_warps = 16;
            config.swizzle_cd_mode     = 64;
            // compute_stages' deepest-fit picks 6 for this geometry.
        }
    }
    // Measured default for deep-K shapes (Qwen3.5-35B-A3B gate_up, N=1024
    // K=2048, B300): the 64 B CD swizzle admits a third mainloop stage and 16
    // transform warps deepen the overlap -- +11..15% beyond one wave, neutral
    // below it. Short-K shapes (down, K=512) lose from the doubled TMA stores,
    // so the switch is keyed on K. Any explicit tune argument wins.
    else if (tune_swizzle_cd == 0 and tune_transform_warps == 0 and tune_block_n == 0 and
             tune_block_k == 0 and k >= 1024)
    {
        config.swizzle_cd_mode     = 64;
        config.num_transform_warps = 16;
    }

    if (tune_shallow_stages > 0)
    {
        NVFP4_HOST_ASSERT_MSG(tune_split_transform == 0 and tune_direct_a == 0,
                              "shallow stages exclude split-transform and direct-A");
        config.num_shallow_stages = tune_shallow_stages;
    }
    if (tune_produced_stages > 0)
        config.num_produced_stages = tune_produced_stages;
    config.compute_stages(tune_num_stages);
    config.validate();

    const int num_sms = device_runtime->get_num_sms();

    // Tail-wave split (3D scheduling): cut the last round's tiles into
    // k-pieces so the remainder SMs join the tail instead of idling.  Guards:
    // a full round must exist, the pieces must fit the SM count, and the
    // k-block count must divide evenly -- an empty piece would deadlock its
    // epilogue.  Orthogonal to the cd64/w16 gate above by design: this knob
    // changes scheduling, not tile geometry, and must never join that
    // conjunction.
    int tail_split = 0;
    if (tune_tail_split > 1 and config.num_shallow_stages == 0)
    {
        const int tiles  = ((m + config.block_m - 1) / config.block_m) *
                          ((n + config.block_n - 1) / config.block_n);
        const int rem    = tiles % num_sms;
        const int num_kb = k / config.block_k;
        if (tiles / num_sms >= 1 and rem > 0 and rem * tune_tail_split <= num_sms and
            tune_tail_split <= num_kb and num_kb % tune_tail_split == 0)
            tail_split = tune_tail_split;
    }

    // A: BF16, K-major. A 128 B swizzle atom holds 64 BF16 elements.
    const auto tensor_map_a = tma::make_2d(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                           a.data_ptr(),
                                           /*gmem_inner=*/k,
                                           /*gmem_outer=*/m,
                                           /*box_inner=*/config.swizzle_a_mode / 2,
                                           /*box_outer=*/config.block_m,
                                           /*outer_stride=*/static_cast<uint64_t>(a.stride(0)) * 2,
                                           config.swizzle_a_mode);

    // W: packed E2M1. `16U4_ALIGN8B` keeps shared memory truly packed, which is
    // what the transform warps write for A0/A1 -- both UMMA operands must agree
    // on the packing. Dimensions are in FP4 elements even though the int8 view's
    // stride is in bytes.
    const auto tensor_map_b = tma::make_2d(CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
                                           b.data_ptr(),
                                           /*gmem_inner=*/k,
                                           /*gmem_outer=*/static_cast<uint64_t>(n) * num_groups,
                                           /*box_inner=*/config.block_k,
                                           /*box_outer=*/config.block_n,
                                           /*outer_stride=*/static_cast<uint64_t>(b.stride(0)),
                                           config.swizzle_ab_mode);

    // SFB: one int32 atom word per (sf_k, n); each row covers 64 K elements.
    const auto tensor_map_sfb = tma::make_2d(CU_TENSOR_MAP_DATA_TYPE_INT32,
                                             sfb.data_ptr(),
                                             /*gmem_inner=*/n,
                                             /*gmem_outer=*/static_cast<uint64_t>(k / 64) * num_groups,
                                             /*box_inner=*/config.block_n,
                                             /*box_outer=*/config.sf_atoms_per_k(),
                                             /*outer_stride=*/static_cast<uint64_t>(n) * 4,
                                             /*swizzle=*/0);

    const auto tensor_map_cd = tma::make_2d(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                            d.data_ptr(),
                                            /*gmem_inner=*/n,
                                            /*gmem_outer=*/m,
                                            /*box_inner=*/config.swizzle_cd_mode / 2,
                                            /*box_outer=*/std::min(config.block_m, 128),
                                            /*outer_stride=*/static_cast<uint64_t>(d.stride(0)) * 2,
                                            config.swizzle_cd_mode);

    std::ostringstream code;
    code << "#include <nvfp4_gemm/impls/sm100_bf16_dual_nvfp4_gemm.cuh>\n\n"
         << "using namespace nvfp4_gemm;\n\n"
         << "static void __instantiate_kernel()\n{\n"
         << "    auto ptr = reinterpret_cast<void*>(&sm100_bf16_dual_nvfp4_gemm_impl<\n"
         << "        " << n << ", " << k << ",\n"
         << "        " << config.block_m << ", " << config.block_n << ", " << config.block_k << ",\n"
         << "        " << num_groups << ",\n"
         << "        " << config.swizzle_a_mode << ", " << config.swizzle_ab_mode << ", " << config.swizzle_cd_mode << ",\n"
         << "        " << config.num_stages << ",\n"
         << "        " << config.num_transform_warps << ",\n"
         << "        " << config.num_epilogue_threads << ",\n"
         << "        " << num_sms << ",\n"
         << "        " << config.forced_epilogue_stages << ",\n"
         << "        " << to_string(GemmType::MGroupedContiguous) << ",\n"
         << "        " << scale_policy << ",\n"
         << "        " << (enable_residual_pass ? "true" : "false") << ",\n"
         << "        " << (tune_direct_a ? "true" : "false") << ",\n"
         << "        " << tune_sched_group << ",\n"
         << "        " << (tune_split_transform ? "true" : "false") << ",\n"
         << "        " << tune_timing_probe << ",\n"
         << "        " << tail_split << ",\n"
         << "        " << config.num_shallow_stages << ",\n"
         << "        " << config.num_produced_stages << ">);\n"
         << "    (void) ptr;\n}\n";

    // Distinct cache names per variant keep the two pass counts from colliding
    // in a reader's cache listing; the hash would separate them anyway.
    const auto kernel = jit::compiler->build(enable_residual_pass ? "bf16_dual_nvfp4_gemm" : "bf16_single_nvfp4_gemm",
                                             code.str());

    jit::LaunchConfig launch_config{num_sms, config.num_threads(), config.smem_size(),
                                    tune_no_pdl == 0,
                                    tune_cluster > 1 ? tune_cluster : 1};
    if (tune_l2_pin_weights)
    {
        launch_config.l2_window_ptr   = b.data_ptr();
        launch_config.l2_window_bytes = static_cast<size_t>(b.numel());
    }

    auto* grouped_layout = m_indices.data_ptr();
    auto* global_scales  = gw.data_ptr();
    auto  shape_m        = static_cast<uint32_t>(m);
    auto  shape_n        = static_cast<uint32_t>(n);
    auto  shape_k        = static_cast<uint32_t>(k);
    auto  map_a          = tensor_map_a;
    auto  map_b          = tensor_map_b;
    auto  map_sfb        = tensor_map_sfb;
    auto  map_cd         = tensor_map_cd;
    // Direct-global-A reads the activations via LDG inside the transform; the
    // pointer travels as a plain kernel argument next to the TMA descriptors.
    auto* a_ptr = a.data_ptr();
    auto  lda   = static_cast<uint32_t>(a.stride(0));
    // Warp-clock probe buffer: m_indices' storage is borrowed as an anchor
    // only when probing is off; when tune_timing_probe==3 the caller appends
    // a 16-element int64 tensor via the `gw` trick... instead we thread an
    // explicit optional pointer through the environment-free path below.
    void* probe_ptr = probe_buffer_for_launch;

    float* tail_ws = nullptr;
    if (tail_split > 1)
    {
        int dev = 0;
        NVFP4_CUDA_CHECK(cudaGetDevice(&dev));
        if (tail_ws_cache[dev] == nullptr)
        {
            void* buf = nullptr;
            NVFP4_CUDA_CHECK(cudaMalloc(&buf, 2 * kTailWsHalfBytes));
            const auto stream = at::cuda::getCurrentCUDAStream().stream();
            NVFP4_CUDA_CHECK(cudaMemsetAsync(buf, 0, 4096, stream));
            NVFP4_CUDA_CHECK(cudaMemsetAsync(static_cast<char*>(buf) + kTailWsHalfBytes, 0, 4096, stream));
            tail_ws_cache[dev] = buf;
        }
        tail_ws_parity ^= 1;
        tail_ws = reinterpret_cast<float*>(static_cast<char*>(tail_ws_cache[dev]) +
                                           (tail_ws_parity ? kTailWsHalfBytes : 0));
    }

    jit::launch(kernel, launch_config, grouped_layout, global_scales, shape_m, shape_n, shape_k, map_a, map_b, map_sfb, map_cd, a_ptr, lda, probe_ptr, tail_ws);
}

/// Swap-AB variant for small-M batches (decode / small prefill).
///
/// Weights take the 128-row M side of the UMMA and tokens the N side, so the
/// per-expert granularity is `block_t` (32) token columns instead of 128 rows,
/// and the per-k-block delivery drops ~3x. `m_indices` groups must therefore be
/// `block_t`-aligned, not 128-aligned. Same tensor arguments and layouts as the
/// normal entry.
inline void sm100_m_grouped_bf16_dual_nvfp4_gemm_contiguous_swap_ab(const torch::Tensor& a,
                                                                    const torch::Tensor& b,
                                                                    const torch::Tensor& sfb,
                                                                    const torch::Tensor& gw,
                                                                    const torch::Tensor& d,
                                                                    const torch::Tensor& m_indices,
                                                                    int                  num_groups,
                                                                    int                  m,
                                                                    int                  n,
                                                                    int                  k,
                                                                    const std::string&   scale_policy,
                                                                    bool                 enable_residual_pass,
                                                                    int                  tune_block_t,
                                                                    int                  tune_transform_warps,
                                                                    int                  tune_num_stages,
                                                                    int                  tune_block_k)
{
    const int block_t         = tune_block_t > 0 ? tune_block_t : 32;
    // A deeper K block halves the number of pipeline steps -- and with them
    // the per-k-block synchronization chain the decode floor is made of.
    // Measured: 256 matches 128 at tiny batches and is +10% at decode-128
    // (18.5 vs 20.5 us on gate_up), so it is the default whenever K allows.
    const int block_k         = tune_block_k > 0 ? tune_block_k : (k % 256 == 0 ? 256 : 128);
    const int swizzle_a_mode  = 128;
    const int swizzle_ab_mode = block_k / 2;
    const int swizzle_cd_mode = 128;
    // The deeper K block doubles the transform work per pipeline step and puts
    // it back on the critical path; 8 warps halve it again (+11..17% measured
    // at decode). bk128's smaller tile only admits 4 (2-block slot floor).
    const int transform_warps = tune_transform_warps > 0 ? tune_transform_warps
                                                         : (block_k == 256 ? 8 : 4);
    const int epilogue_threads = 128;
    const int katoms           = block_k / 64;

    NVFP4_HOST_ASSERT_MSG(block_t == 32 or block_t == 64 or block_t == 128,
                          "block_t must be 32, 64 or 128");
    NVFP4_HOST_ASSERT_MSG(m % block_t == 0, "M must be a multiple of block_t");
    NVFP4_HOST_ASSERT_MSG(n % 128 == 0 and k % 128 == 0, "N and K must be multiples of 128");
    const int transform_blocks = block_t * block_k / 16;
    NVFP4_HOST_ASSERT_MSG(transform_blocks % (transform_warps * 32) == 0 and
                              transform_warps * 32 * 2 <= transform_blocks,
                          "transform threads must evenly divide the token tile");

    // Shared memory: the swapped stage is tiny (~23 KB at block_t = 32), so the
    // pipeline can go deep; past ~6 stages the returns vanish.
    const int smem_per_stage = block_t * block_k * 2 +          // act BF16
                               2 * (block_t * block_k / 2) +    // act0 + act1
                               128 * block_k / 2 +              // W
                               katoms * 128 * 4 +               // SFW
                               2 * (katoms * 128 * 4);          // SFACT0/1
    const int smem_cd       = 2 * block_t * 128 * 2;
    int       num_stages    = tune_num_stages > 0 ? tune_num_stages : (block_k == 256 ? 4 : 8);
    auto      smem_size     = [&](int stages) {
        return smem_cd + stages * smem_per_stage + (3 * stages + 2 * 2) * 8 + 8;
    };
    while (num_stages > 1 and smem_size(num_stages) > kSmemCapacitySm100)
        --num_stages;

    const int num_sms = device_runtime->get_num_sms();

    // Activations: BF16, K-major, block_t rows per box.
    const auto tensor_map_a = tma::make_2d(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                           a.data_ptr(),
                                           /*gmem_inner=*/k,
                                           /*gmem_outer=*/m,
                                           /*box_inner=*/swizzle_a_mode / 2,
                                           /*box_outer=*/block_t,
                                           /*outer_stride=*/static_cast<uint64_t>(a.stride(0)) * 2,
                                           swizzle_a_mode);

    // Weights: packed E2M1, 128 model rows per box.
    const auto tensor_map_b = tma::make_2d(CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
                                           b.data_ptr(),
                                           /*gmem_inner=*/k,
                                           /*gmem_outer=*/static_cast<uint64_t>(n) * num_groups,
                                           /*box_inner=*/block_k,
                                           /*box_outer=*/128,
                                           /*outer_stride=*/static_cast<uint64_t>(b.stride(0)),
                                           swizzle_ab_mode);

    // Weight scales: one int32 atom word per (sf_k, weight row).
    const auto tensor_map_sfb = tma::make_2d(CU_TENSOR_MAP_DATA_TYPE_INT32,
                                             sfb.data_ptr(),
                                             /*gmem_inner=*/n,
                                             /*gmem_outer=*/static_cast<uint64_t>(k / 64) * num_groups,
                                             /*box_inner=*/128,
                                             /*box_outer=*/katoms,
                                             /*outer_stride=*/static_cast<uint64_t>(n) * 4,
                                             /*swizzle=*/0);

    // D: (tokens, model_n) BF16, stored in (block_t x 128) transposed tiles.
    const auto tensor_map_cd = tma::make_2d(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                            d.data_ptr(),
                                            /*gmem_inner=*/n,
                                            /*gmem_outer=*/m,
                                            /*box_inner=*/swizzle_cd_mode / 2,
                                            /*box_outer=*/block_t,
                                            /*outer_stride=*/static_cast<uint64_t>(d.stride(0)) * 2,
                                            swizzle_cd_mode);

    NVFP4_HOST_ASSERT_MSG(scale_policy == "derived_div8" or scale_policy == "residual_amax" or
                              scale_policy == "derived_div4",
                          "scale_policy must be 'derived_div8', 'derived_div4' or 'residual_amax'");
    const std::string policy = scale_policy == "residual_amax" ? "ScalePolicy::ResidualAmax"
                               : scale_policy == "derived_div4" ? "ScalePolicy::DerivedDiv4"
                                                                : "ScalePolicy::DerivedDiv8";

    std::ostringstream code;
    code << "#include <nvfp4_gemm/impls/sm100_bf16_dual_nvfp4_gemm_swap_ab.cuh>\n\n"
         << "using namespace nvfp4_gemm;\n\n"
         << "static void __instantiate_kernel()\n{\n"
         << "    auto ptr = reinterpret_cast<void*>(&sm100_bf16_dual_nvfp4_gemm_swap_ab_impl<\n"
         << "        " << n << ", " << k << ",\n"
         << "        " << block_t << ", " << block_k << ",\n"
         << "        " << num_groups << ",\n"
         << "        " << swizzle_a_mode << ", " << swizzle_ab_mode << ", " << swizzle_cd_mode << ",\n"
         << "        " << num_stages << ",\n"
         << "        " << transform_warps << ",\n"
         << "        " << epilogue_threads << ",\n"
         << "        " << num_sms << ",\n"
         << "        " << policy << ",\n"
         << "        " << (enable_residual_pass ? "true" : "false") << ">);\n"
         << "    (void) ptr;\n}\n";

    const auto kernel = jit::compiler->build("bf16_dual_nvfp4_gemm_swap_ab",
                                             code.str(),
                                             "sm100_bf16_dual_nvfp4_gemm_swap_ab_impl");

    const int num_threads = (3 + transform_warps) * 32 + epilogue_threads;
    jit::LaunchConfig launch_config{num_sms, num_threads, smem_size(num_stages), true};

    auto* grouped_layout = m_indices.data_ptr();
    auto* global_scales  = gw.data_ptr();
    auto  shape_m        = static_cast<uint32_t>(m);
    auto  shape_n        = static_cast<uint32_t>(n);
    auto  shape_k        = static_cast<uint32_t>(k);
    auto  map_a          = tensor_map_a;
    auto  map_b          = tensor_map_b;
    auto  map_sfb        = tensor_map_sfb;
    auto  map_cd         = tensor_map_cd;

    jit::launch(kernel, launch_config, grouped_layout, global_scales, shape_m, shape_n, shape_k, map_a, map_b, map_sfb, map_cd);
}

}   // namespace nvfp4_gemm

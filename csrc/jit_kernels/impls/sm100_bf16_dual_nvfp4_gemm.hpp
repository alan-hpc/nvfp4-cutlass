#pragma once

#include <torch/python.h>

#include "../../../3rdparty/DeepGEMM/csrc/jit/compiler.hpp"
#include "../../../3rdparty/DeepGEMM/csrc/jit/device_runtime.hpp"
#include "../../../3rdparty/DeepGEMM/csrc/jit/kernel_runtime.hpp"
#include "../../../3rdparty/DeepGEMM/csrc/jit_kernels/heuristics/sm100.hpp"
#include "../../../3rdparty/DeepGEMM/csrc/jit_kernels/impls/runtime_utils.hpp"
#include "../../../3rdparty/DeepGEMM/csrc/utils/exception.hpp"
#include "../../../3rdparty/DeepGEMM/csrc/utils/format.hpp"
#include "../../../3rdparty/DeepGEMM/csrc/utils/math.hpp"

namespace nvfp4_gemm {

using deep_gemm::GemmType;

/// Static configuration of the fused dual-NVFP4 kernel.
///
/// The tile shape is not swept by a heuristic yet: `128 x 256 x 128` is the one
/// the algorithm doc measures, and it is also the largest N that leaves tensor
/// memory for the two extra SFA column groups dual-A needs (256 accumulator
/// columns + SFA0 8 + SFA1 8 + SFB 16 = 288 of 512).
struct DualNVFP4Config
{
    int block_m = 128;
    int block_n = 256;
    int block_k = 128;

    int swizzle_a_mode  = 128;   // BF16 A: 128 B rows per swizzle atom
    int swizzle_ab_mode = 64;    // packed FP4 A0/A1/W: 64 B per row
    int swizzle_cd_mode = 128;   // BF16 D

    int num_transform_warps  = 8;
    int num_epilogue_threads = 128;
    int num_stages           = 0;   // filled in by `compute_stages`

    int num_threads() const
    {
        return (3 + num_transform_warps) * 32 + num_epilogue_threads;
    }

    /// Bytes of shared memory consumed by one mainloop stage.
    int smem_per_stage() const
    {
        const int sf_atoms_per_k = block_k / 64;
        const int sf_block_m     = deep_gemm::ceil_div(block_m, 128) * 128;
        const int sf_block_n     = deep_gemm::ceil_div(block_n, 128) * 128;
        return block_m * block_k * 2 +                   // A, BF16
               2 * (block_m * block_k / 2) +             // A0 + A1, packed FP4
               block_n * block_k / 2 +                   // W, packed FP4
               2 * (sf_atoms_per_k * sf_block_m * 4) +   // SFA0 + SFA1
               sf_atoms_per_k * sf_block_n * 4;          // SFB
    }

    int smem_epilogue() const
    {
        const int store_block_m = std::min(block_m, 128);
        const int store_block_n = swizzle_cd_mode / 2;   // BF16
        return store_block_m * store_block_n * 2 * /* TMA store stages */ 2;
    }

    int num_epilogue_stages() const
    {
        const int sf_cols = 2 * (block_k / 64) * (deep_gemm::ceil_div(block_m, 128) * 4) +
                            (block_k / 64) * (deep_gemm::ceil_div(block_n, 128) * 4);
        return (block_n * 2 + sf_cols <= 512) ? 2 : 1;
    }

    int smem_barriers() const
    {
        return (3 * num_stages + 2 * num_epilogue_stages()) * 8 + 8;
    }

    int smem_size() const
    {
        return smem_epilogue() + num_stages * smem_per_stage() + smem_barriers();
    }

    /// Pick the deepest pipeline that fits, then verify it.
    ///
    /// Dual-A is unusually shared-memory hungry: a stage carries the BF16 A tile
    /// *and* both FP4 copies of it, so a 128x256x128 stage costs ~68 KB against
    /// ~40 KB for a plain NVFP4 GEMM.  That caps the pipeline at 2 stages on a
    /// 227 KB SM.
    void compute_stages()
    {
        for (int candidate = 4; candidate >= 1; --candidate)
        {
            num_stages = candidate;
            if (smem_size() <= deep_gemm::SM100ArchSpec::smem_capacity)
                return;
        }
        DG_HOST_ASSERT(false and "Cannot fit even a single stage into shared memory");
    }
};

class SM100BF16DualNVFP4GemmRuntime final : public deep_gemm::LaunchRuntime<SM100BF16DualNVFP4GemmRuntime>
{
public:
    struct Args
    {
        int                   shape_m, shape_n, shape_k, num_groups;
        int                   compiled_n, compiled_k;
        GemmType              gemm_type;
        std::string           scale_policy;
        bool                  enable_residual_pass;
        DualNVFP4Config       config;
        deep_gemm::LaunchArgs launch_args;

        void*       grouped_layout;
        void*       weight_global_scales;
        CUtensorMap tensor_map_a;
        CUtensorMap tensor_map_b;
        CUtensorMap tensor_map_sfb;
        CUtensorMap tensor_map_cd;
    };

    static std::string generate_impl(const Args& args)
    {
        return fmt::format(R"(
#include <nvfp4_gemm/impls/sm100_bf16_dual_nvfp4_gemm.cuh>

using namespace nvfp4_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_bf16_dual_nvfp4_gemm_impl<
        {}, {},
        {}, {}, {},
        {},
        {}, {}, {},
        {},
        {},
        {},
        {},
        {},
        {},
        {}
    >);
}};
)",
                           args.compiled_n,
                           args.compiled_k,
                           args.config.block_m,
                           args.config.block_n,
                           args.config.block_k,
                           args.num_groups,
                           args.config.swizzle_a_mode,
                           args.config.swizzle_ab_mode,
                           args.config.swizzle_cd_mode,
                           args.config.num_stages,
                           args.config.num_transform_warps,
                           args.config.num_epilogue_threads,
                           args.launch_args.grid_dim.first,
                           to_string(args.gemm_type),
                           args.scale_policy,
                           args.enable_residual_pass ? "true" : "false");
    }

    static void launch_impl(const deep_gemm::KernelHandle&       kernel,
                            const deep_gemm::LaunchConfigHandle& config,
                            Args                                 args)
    {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config, args.grouped_layout, args.weight_global_scales, args.shape_m, args.shape_n, args.shape_k, args.tensor_map_a, args.tensor_map_b, args.tensor_map_sfb, args.tensor_map_cd));
    }
};

/// Fused BF16 x NVFP4 m-grouped contiguous GEMM (MoE expert GEMM).
///
/// Args:
///   a:                BF16 activations, (M, K), K-major.
///   b:                packed E2M1 weights viewed as int8, (E * N, K / 2), K-major.
///   sfb:              E4M3 weight block scales in the 128x4 NVFP4 atom layout,
///                     viewed as int32 of shape (E * K / 64, N).
///   weight_global_scales: FP32, (E,).  Applied once in the epilogue.
///   d:                BF16 output, (M, N).
///   grouped_layout:   int32 (M,), expert index per row (contiguous layout).
static void sm100_m_grouped_bf16_dual_nvfp4_gemm_contiguous(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const torch::Tensor& sfb,
    const torch::Tensor& weight_global_scales,
    const torch::Tensor& d,
    const torch::Tensor& grouped_layout,
    const int&           num_groups,
    const int&           m,
    const int&           n,
    const int&           k,
    const std::string&   scale_policy         = "ScalePolicy::DerivedDiv8",
    const bool&          enable_residual_pass = true)
{
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(b.scalar_type() == deep_gemm::kPackedFP4);
    DG_HOST_ASSERT(weight_global_scales.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(weight_global_scales.numel() == num_groups);
    // The transform's SF-atom addressing assumes whole 128-row / 64-K atoms.
    DG_HOST_ASSERT(k % 128 == 0 and n % 128 == 0);

    DualNVFP4Config config;
    config.compute_stages();

    const int num_sms = deep_gemm::device_runtime->get_num_sms();

    // A: BF16, K-major, one swizzle atom per 64 elements of K.
    const auto tensor_map_a = deep_gemm::make_tma_a_desc(
        cute::UMMA::Major::K,
        a,
        m,
        k,
        config.block_m,
        config.block_k,
        static_cast<int>(a.stride(0)),
        1,
        config.swizzle_a_mode);

    // W: packed E2M1.  `fp4_unpacked_smem = false` selects
    // `CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B`, which keeps shared memory truly
    // packed (two elements per byte) -- that is what the transform warps write
    // for A0/A1, and both UMMA operands must agree on the packing.  Note the
    // global dims are in FP4 *elements*, while `b.stride(0)` is in bytes because
    // the tensor is an int8 view.
    const auto tensor_map_b = deep_gemm::make_tma_2d_desc(
        b,
        k,
        n * num_groups,
        config.block_k,
        config.block_n,
        static_cast<int>(b.stride(0)),
        config.swizzle_ab_mode,
        0,
        false,
        false);

    const auto tensor_map_cd = deep_gemm::make_tma_cd_desc(
        d,
        m,
        n,
        std::min(config.block_m, 128),
        config.swizzle_cd_mode / 2,
        static_cast<int>(d.stride(0)),
        1,
        config.swizzle_cd_mode);

    // SFB rows cover 64 K elements each, matching one 128x4 atom.
    const auto tensor_map_sfb = deep_gemm::make_tma_2d_desc(
        sfb,
        n,
        (k / 64) * num_groups,
        config.block_n,
        config.block_k / 64,
        n,
        0);

    const SM100BF16DualNVFP4GemmRuntime::Args args = {
        .shape_m    = m,
        .shape_n    = n,
        .shape_k    = k,
        .num_groups = num_groups,
        // N and K are compile-time constants; M varies with the MoE routing.
        .compiled_n           = n,
        .compiled_k           = k,
        .gemm_type            = GemmType::MGroupedContiguous,
        .scale_policy         = scale_policy,
        .enable_residual_pass = enable_residual_pass,
        .config               = config,
        .launch_args          = deep_gemm::LaunchArgs(num_sms, config.num_threads(), config.smem_size()),
        .grouped_layout       = grouped_layout.data_ptr(),
        .weight_global_scales = weight_global_scales.data_ptr(),
        .tensor_map_a         = tensor_map_a,
        .tensor_map_b         = tensor_map_b,
        .tensor_map_sfb       = tensor_map_sfb,
        .tensor_map_cd        = tensor_map_cd};
    const auto code    = SM100BF16DualNVFP4GemmRuntime::generate(args);
    const auto runtime = deep_gemm::compiler->build(
        enable_residual_pass ? "sm100_m_grouped_bf16_dual_nvfp4_gemm_contiguous"
                             : "sm100_m_grouped_bf16_single_nvfp4_gemm_contiguous",
        code);
    SM100BF16DualNVFP4GemmRuntime::launch(runtime, args);
}

}   // namespace nvfp4_gemm

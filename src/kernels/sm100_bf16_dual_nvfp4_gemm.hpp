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

    int num_threads() const
    {
        return (3 + num_transform_warps) * 32 + num_epilogue_threads;
    }

    int sf_atoms_per_k() const
    {
        return block_k / 64;
    }

    /// Bytes of shared memory consumed by one mainloop stage.
    int smem_per_stage() const
    {
        const int sf_block_m = ((block_m + 127) / 128) * 128;
        const int sf_block_n = ((block_n + 127) / 128) * 128;
        return block_m * block_k * 2 +                     // A, BF16
               2 * (block_m * block_k / 2) +               // A0 + A1, packed FP4
               block_n * block_k / 2 +                     // W, packed FP4
               2 * (sf_atoms_per_k() * sf_block_m * 4) +   // SFA0 + SFA1
               sf_atoms_per_k() * sf_block_n * 4;          // SFB
    }

    int smem_epilogue() const
    {
        const int store_block_m = std::min(block_m, 128);
        const int store_block_n = swizzle_cd_mode / 2;   // BF16
        return store_block_m * store_block_n * 2 * /* TMA store stages */ 2;
    }

    int num_epilogue_stages() const
    {
        const int sf_cols = 2 * sf_atoms_per_k() * (((block_m + 127) / 128) * 4) +
                            sf_atoms_per_k() * (((block_n + 127) / 128) * 4);
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

    /// Pick the deepest pipeline that fits.
    ///
    /// Dual-A is unusually shared-memory hungry: a stage carries the BF16 A tile
    /// *and* both FP4 copies of it, so a 128x256x128 stage costs ~68 KB against
    /// ~40 KB for a plain NVFP4 GEMM. That caps the pipeline at 2 stages.
    void compute_stages()
    {
        for (int candidate = 4; candidate >= 1; --candidate)
        {
            num_stages = candidate;
            if (smem_size() <= kSmemCapacitySm100)
                return;
        }
        NVFP4_HOST_ASSERT_MSG(false, "cannot fit even a single stage into shared memory");
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
                                                            bool                 enable_residual_pass)
{
    DualNVFP4Config config;
    config.compute_stages();

    const int num_sms = device_runtime->get_num_sms();

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
         << "        " << to_string(GemmType::MGroupedContiguous) << ",\n"
         << "        " << scale_policy << ",\n"
         << "        " << (enable_residual_pass ? "true" : "false") << ">);\n"
         << "    (void) ptr;\n}\n";

    // Distinct cache names per variant keep the two pass counts from colliding
    // in a reader's cache listing; the hash would separate them anyway.
    const auto kernel = jit::compiler->build(enable_residual_pass ? "bf16_dual_nvfp4_gemm" : "bf16_single_nvfp4_gemm",
                                             code.str());

    jit::LaunchConfig launch_config{num_sms, config.num_threads(), config.smem_size(), true};

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

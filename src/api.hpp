#pragma once

// Python-facing entry points: argument validation, then dispatch.

#include <string>

#include <pybind11/pybind11.h>
#include <torch/python.h>

#include "jit/compiler.hpp"
#include "kernels/sm100_bf16_dual_nvfp4_gemm.hpp"
#include "runtime/device.hpp"
#include "runtime/exception.hpp"

namespace nvfp4_gemm::api {

/// Fused BF16 x NVFP4 m-grouped contiguous GEMM.
///
/// Shapes, all on the current CUDA device:
///   a          BF16   (M, K)              activations, K-major
///   b          int8   (E * N, K / 2)      packed E2M1 weights, K-major
///   sfb        int32  (E * K / 64, N)     E4M3 block scales, four per word
///   gw         fp32   (E,)                per-expert weight global scale
///   d          BF16   (M, N)              output
///   m_indices  int32  (M,)                expert index per row
inline void m_grouped_bf16_dual_nvfp4_gemm_contiguous(const torch::Tensor& a,
                                                      const torch::Tensor& b,
                                                      const torch::Tensor& sfb,
                                                      const torch::Tensor& gw,
                                                      const torch::Tensor& d,
                                                      const torch::Tensor& m_indices,
                                                      const std::string&   scale_policy,
                                                      bool                 enable_residual_pass,
                                                      int                  tune_block_n,
                                                      int                  tune_transform_warps,
                                                      int                  tune_epilogue_threads,
                                                      int                  tune_num_stages,
                                                      int                  tune_swizzle_cd,
                                                      int                  tune_block_k,
                                                      int                  tune_epilogue_stages,
                                                      int                  tune_swap_ab,
                                                      int                  tune_cluster,
                                                      int                  tune_direct_a,
                                                      int                  tune_l2_pin_weights,
                                                      int                  tune_no_pdl,
                                                      int                  tune_sched_group,
                                                      int                  tune_split_transform,
                                                      int                  tune_timing_probe)
{
    const int m          = static_cast<int>(a.size(0));
    const int k          = static_cast<int>(a.size(1));
    const int n          = static_cast<int>(d.size(1));
    const int num_groups = static_cast<int>(gw.numel());

    NVFP4_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    NVFP4_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    NVFP4_HOST_ASSERT(b.scalar_type() == kPackedFP4);
    NVFP4_HOST_ASSERT(sfb.scalar_type() == torch::kInt);
    NVFP4_HOST_ASSERT(gw.scalar_type() == torch::kFloat);
    NVFP4_HOST_ASSERT(m_indices.scalar_type() == torch::kInt);

    NVFP4_HOST_ASSERT(a.dim() == 2 and d.dim() == 2);
    NVFP4_HOST_ASSERT(d.size(0) == m);
    NVFP4_HOST_ASSERT(m_indices.dim() == 1 and m_indices.size(0) == m);
    NVFP4_HOST_ASSERT(b.dim() == 2 and b.size(0) == static_cast<int64_t>(num_groups) * n);
    NVFP4_HOST_ASSERT(b.size(1) == k / 2);
    NVFP4_HOST_ASSERT(sfb.dim() == 2 and sfb.size(0) == static_cast<int64_t>(num_groups) * (k / 64));
    NVFP4_HOST_ASSERT(sfb.size(1) == n);

    // The scale-atom addressing assumes whole 128-row / 64-element atoms, and
    // TMA fills only BLOCK_N of the padded SF rows UTCCP later reads.
    NVFP4_HOST_ASSERT_MSG(k % 128 == 0 and n % 128 == 0, "N and K must be multiples of 128");
    // Normal orientation: the transform covers a full 128-row tile.  Swap-AB
    // puts tokens on the N side, so groups only need block_t (32) alignment.
    NVFP4_HOST_ASSERT_MSG(m % (tune_swap_ab ? 32 : 128) == 0,
                          "M must be a multiple of the token-tile alignment");

    if (m == 0 or n == 0)
        return;

    if (tune_swap_ab)
    {
        sm100_m_grouped_bf16_dual_nvfp4_gemm_contiguous_swap_ab(
            a,
            b,
            sfb,
            gw,
            d,
            m_indices,
            num_groups,
            m,
            n,
            k,
            scale_policy,
            enable_residual_pass,
            /*tune_block_t=*/tune_block_n,
            tune_transform_warps,
            tune_num_stages,
            tune_block_k);
        return;
    }

    NVFP4_HOST_ASSERT_MSG(scale_policy == "derived_div8" or scale_policy == "residual_amax" or
                              scale_policy == "derived_div4",
                          "scale_policy must be 'derived_div8', 'derived_div4' or 'residual_amax'");
    const std::string policy = scale_policy == "residual_amax" ? "ScalePolicy::ResidualAmax"
                               : scale_policy == "derived_div4" ? "ScalePolicy::DerivedDiv4"
                                                                : "ScalePolicy::DerivedDiv8";

    sm100_m_grouped_bf16_dual_nvfp4_gemm_contiguous(
        a,
        b,
        sfb,
        gw,
        d,
        m_indices,
        num_groups,
        m,
        n,
        k,
        policy,
        enable_residual_pass,
        tune_block_n,
        tune_transform_warps,
        tune_epilogue_threads,
        tune_num_stages,
        tune_swizzle_cd,
        tune_block_k,
        tune_epilogue_stages,
        tune_cluster,
        tune_direct_a,
        tune_l2_pin_weights,
        tune_no_pdl,
        tune_sched_group,
        tune_split_transform,
        tune_timing_probe);
}

inline void register_apis(pybind11::module_& m)
{
    m.def("init", [](const std::string& include_root, const std::string& cuda_home) { jit::compiler->init(include_root, cuda_home); }, pybind11::arg("include_root"), pybind11::arg("cuda_home"));

    m.def("get_num_sms", []() { return device_runtime->get_num_sms(); });
    // Warp-clock probe: hand a 16-element int64 CUDA tensor to the next
    // launch (tune_timing_probe=3); pass None to detach.
    m.def("set_probe_buffer", [](std::optional<torch::Tensor> t) {
        probe_buffer_for_launch = t.has_value() ? t->data_ptr() : nullptr;
    });
    m.def("set_num_sms", [](int value) { device_runtime->set_num_sms(value); }, pybind11::arg("num_sms"));

    m.def("m_grouped_bf16_dual_nvfp4_gemm_contiguous",
          &m_grouped_bf16_dual_nvfp4_gemm_contiguous,
          pybind11::arg("a"),
          pybind11::arg("b"),
          pybind11::arg("sfb"),
          pybind11::arg("gw"),
          pybind11::arg("d"),
          pybind11::arg("m_indices"),
          pybind11::arg("scale_policy") = "derived_div8",
          // Single-pass mode drops the residual MMA entirely; it exists so the
          // benchmark can isolate what the second pass costs against a kernel
          // that is otherwise identical.
          pybind11::arg("enable_residual_pass") = true,
          // Tuning overrides. Zero means "leave the shipped default alone", so
          // an autotuner can vary one axis at a time without restating the rest.
          pybind11::arg("tune_block_n")           = 0,
          pybind11::arg("tune_transform_warps")   = 0,
          pybind11::arg("tune_epilogue_threads")  = 0,
          pybind11::arg("tune_num_stages")        = 0,
          pybind11::arg("tune_swizzle_cd")        = 0,
          pybind11::arg("tune_block_k")           = 0,
          pybind11::arg("tune_epilogue_stages")   = 0,
          // Swap-AB: weights on the 128-row M side, tokens on the N side.
          // For decode-sized batches; m_indices groups need only 32-row
          // alignment. tune_block_n doubles as the token tile (32/64/128).
          pybind11::arg("tune_swap_ab")           = 0,
          // Cluster co-scheduling (no multicast yet): pairs of consecutive
          // tiles share a W tile, so 2 lands them in one GPC for L2 locality.
          pybind11::arg("tune_cluster")           = 0,
          // Direct-global-A: the transform reads BF16 activations via LDG
          // instead of a TMA->SMEM round trip -- removes 32 KB/k-block of TMA
          // ingress, the measured prefill floor.
          pybind11::arg("tune_direct_a")          = 0,
          pybind11::arg("tune_l2_pin_weights")    = 0,
          pybind11::arg("tune_no_pdl")            = 0,
          pybind11::arg("tune_sched_group")       = 0,
          // Split transform: signal A0/SFA0 readiness before the residual
          // quantization, so MMA pass 0 overlaps the residual pass (the
          // algorithm doc's fine-grained pipeline).
          pybind11::arg("tune_split_transform")   = 0,
          // TIMING DECOMPOSITION ONLY -- 1 drops SFA1 UTCCPs, 2 drops the
          // second UMMA; both produce wrong numbers by design.
          pybind11::arg("tune_timing_probe")      = 0);
}

}   // namespace nvfp4_gemm::api

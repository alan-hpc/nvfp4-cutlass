#pragma once

#include <torch/python.h>

#include "../jit_kernels/impls/sm100_bf16_dual_nvfp4_gemm.hpp"

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
static void m_grouped_bf16_dual_nvfp4_gemm_contiguous(
        const torch::Tensor& a,
        const torch::Tensor& b, const torch::Tensor& sfb,
        const torch::Tensor& gw,
        const torch::Tensor& d,
        const torch::Tensor& m_indices,
        const std::string& scale_policy,
        const bool& enable_residual_pass) {
    const int m = static_cast<int>(a.size(0));
    const int k = static_cast<int>(a.size(1));
    const int n = static_cast<int>(d.size(1));
    const int num_groups = static_cast<int>(gw.numel());

    DG_HOST_ASSERT(a.dim() == 2 and d.dim() == 2);
    DG_HOST_ASSERT(d.size(0) == m);
    DG_HOST_ASSERT(m_indices.dim() == 1 and m_indices.size(0) == m);
    DG_HOST_ASSERT(m_indices.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(b.dim() == 2 and b.size(0) == static_cast<int64_t>(num_groups) * n);
    DG_HOST_ASSERT(b.size(1) == k / 2);
    DG_HOST_ASSERT(sfb.dim() == 2 and sfb.size(0) == static_cast<int64_t>(num_groups) * (k / 64));
    DG_HOST_ASSERT(sfb.size(1) == n);
    // The kernel's SF-atom addressing assumes whole 128-row / 64-element atoms,
    // and TMA fills only BLOCK_N of SF_BLOCK_N rows.
    DG_HOST_ASSERT(k % 128 == 0 and n % 128 == 0);
    // The transform covers a full 128-row tile; a ragged tail would leave part
    // of A0/A1 undefined for rows UMMA still reads.
    DG_HOST_ASSERT(m % 128 == 0);

    if (m == 0 or n == 0)
        return;

    const std::string policy = scale_policy == "residual_amax"
                             ? "ScalePolicy::ResidualAmax"
                             : "ScalePolicy::DerivedDiv8";
    DG_HOST_ASSERT(scale_policy == "derived_div8" or scale_policy == "residual_amax");

    sm100_m_grouped_bf16_dual_nvfp4_gemm_contiguous(
        a, b, sfb, gw, d, m_indices, num_groups, m, n, k, policy, enable_residual_pass);
}

static void register_apis(pybind11::module_& m) {
    m.def("m_grouped_bf16_dual_nvfp4_gemm_contiguous",
          &m_grouped_bf16_dual_nvfp4_gemm_contiguous,
          pybind11::arg("a"), pybind11::arg("b"), pybind11::arg("sfb"),
          pybind11::arg("gw"), pybind11::arg("d"), pybind11::arg("m_indices"),
          pybind11::arg("scale_policy") = "derived_div8",
          // Single-pass mode drops the residual MMA entirely; it exists so the
          // benchmark can isolate what the second pass costs against a kernel
          // that is otherwise identical.
          pybind11::arg("enable_residual_pass") = true);
}

} // namespace nvfp4_gemm::api

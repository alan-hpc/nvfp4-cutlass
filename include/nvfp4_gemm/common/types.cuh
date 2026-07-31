#pragma once

// Problem-shape enumerations shared between the kernel and its launcher.
//
// Derived from DeepGEMM (MIT, Copyright (c) 2025 DeepSeek),
// `deep_gemm/common/types.cuh`, reduced to the layouts this kernel supports.
// The k-grouped and psum layouts are deliberately absent: dual-A needs the whole
// K range of a row to produce that row's scales, so a split-K layout would need
// a different transform, not just a different scheduler.

#include <nvfp4_gemm/common/macros.cuh>

namespace nvfp4_gemm {

enum class GemmType
{
    /// Plain GEMM, one weight matrix.
    Normal = 0,
    /// MoE expert GEMM: rows are sorted by expert, `grouped_layout[m]` names the
    /// expert owning row `m`.
    MGroupedContiguous = 1,
    /// MoE expert GEMM with a fixed per-expert row capacity and a runtime count,
    /// `grouped_layout[e]` giving the valid row count of expert `e`.
    MGroupedMasked = 2,
};

constexpr CUTLASS_HOST_DEVICE bool is_m_grouped(const GemmType& gemm_type)
{
    return gemm_type == GemmType::MGroupedContiguous or gemm_type == GemmType::MGroupedMasked;
}

}   // namespace nvfp4_gemm

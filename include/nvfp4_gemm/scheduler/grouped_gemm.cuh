#pragma once

// Persistent output-tile scheduler.
//
// Derived from DeepGEMM (MIT, Copyright (c) 2025 DeepSeek),
// `deep_gemm/scheduler/gemm.cuh`. Reduced to single-CTA (no multicast) and to
// the three m-layouts this kernel supports; the k-grouped and psum variants are
// gone, along with the SM90-only multicast bookkeeping.
//
// Every warp role runs an identical copy of this loop, so all roles walk the
// same sequence of tiles and their pipeline stage counters stay in lockstep.

#include <nvfp4_gemm/common/macros.cuh>
#include <nvfp4_gemm/common/math.cuh>
#include <nvfp4_gemm/common/types.cuh>

namespace nvfp4_gemm::sched {

/// Number of consecutive tiles grouped along the secondary axis for L2 reuse.
///
/// Walking tiles in raster order would stream a fresh weight tile per step;
/// grouping keeps a working set of `kNum1DBlocksPerGroup` weight tiles resident
/// while the other axis advances.
template<uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kNumSMs>
static constexpr uint32_t get_num_1d_blocks_per_group()
{
    uint32_t num_best_blocks = 0, min_usage = 0xffffffffu;
    for (const auto candidate : {8u, 16u})
    {
        const auto usage = candidate * BLOCK_M + math::constexpr_ceil_div(kNumSMs, candidate) * BLOCK_N;
        if (usage < min_usage)
            min_usage = usage, num_best_blocks = candidate;
    }
    return num_best_blocks;
}

template<GemmType kGemmType,
         uint32_t BLOCK_M,
         uint32_t BLOCK_N,
         uint32_t kNumGroups,
         uint32_t kNumSMs,
         uint32_t kNum1DBlocksPerGroup = get_num_1d_blocks_per_group<BLOCK_M, BLOCK_N, kNumSMs>()>
struct Scheduler
{
    int current_iter = -1;

    uint32_t num_blocks;
    uint32_t num_m_blocks;
    uint32_t num_n_blocks;
    uint32_t num_blocks_in_group;

    int*     grouped_layout;
    uint32_t current_group_idx = 0;
    /// Only used by the masked layout: rows already consumed by earlier experts.
    uint32_t current_m_cumsum = 0;

    CUTLASS_DEVICE explicit Scheduler(const uint32_t& shape_m,
                                      const uint32_t& shape_n,
                                      int*            grouped_layout = nullptr)
    {
        num_m_blocks         = math::ceil_div(shape_m, BLOCK_M);
        num_n_blocks         = math::ceil_div(shape_n, BLOCK_N);
        num_blocks           = num_m_blocks * num_n_blocks;
        this->grouped_layout = grouped_layout;
    }

    /// Map a linear block index to (m, n), grouped along M for L2 reuse.
    CUTLASS_DEVICE void get_swizzled_block_idx(const uint32_t& block_idx, uint32_t& m_block_idx, uint32_t& n_block_idx)
    {
        const auto num_blocks_per_group = num_n_blocks * kNum1DBlocksPerGroup;
        const auto group_idx            = block_idx / num_blocks_per_group;
        const auto first_block_idx      = group_idx * kNum1DBlocksPerGroup;
        const auto in_group_idx         = block_idx % num_blocks_per_group;
        num_blocks_in_group             = min(kNum1DBlocksPerGroup, num_m_blocks - first_block_idx);

        m_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
        n_block_idx = in_group_idx / num_blocks_in_group;
    }

    /// Global element offset along a dimension for the current tile.
    ///
    /// `kWithGroupOffset` selects whether the expert offset applies. Activations
    /// in the contiguous layout are one flat tensor, so they take no offset;
    /// weights and their scales are stacked per expert, so they do.
    template<bool kWithGroupOffset>
    CUTLASS_DEVICE uint32_t get_global_idx(const uint32_t  shape_dim,
                                           const uint32_t  block_size,
                                           const uint32_t& block_idx,
                                           const uint32_t& m_block_idx = 0) const
    {
        if constexpr (kGemmType == GemmType::Normal)
            return block_idx * block_size;
        else if constexpr (kGemmType == GemmType::MGroupedContiguous)
        {
            const auto offset = kWithGroupOffset ? static_cast<uint32_t>(max(0, grouped_layout[m_block_idx * BLOCK_M])) : 0u;
            return offset * shape_dim + block_idx * block_size;
        }
        else
        {
            const auto offset = kWithGroupOffset ? current_group_idx : 0u;
            return offset * shape_dim + block_idx * block_size;
        }
    }

    /// Expert owning the current tile, for the epilogue's per-expert weight scale.
    CUTLASS_DEVICE uint32_t get_expert_idx(const uint32_t& m_block_idx) const
    {
        if constexpr (kGemmType == GemmType::MGroupedContiguous)
            return static_cast<uint32_t>(max(0, grouped_layout[m_block_idx * BLOCK_M]));
        else if constexpr (kGemmType == GemmType::MGroupedMasked)
            return current_group_idx;
        else
            return 0u;
    }

    CUTLASS_DEVICE bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx)
    {
        const auto next_block_idx = (++current_iter) * kNumSMs + blockIdx.x;

        if constexpr (kGemmType == GemmType::MGroupedMasked)
        {
            // Expert row counts are only known at run time, so walk experts
            // until the linear index falls inside one.
            while (true)
            {
                if (current_group_idx == kNumGroups)
                    return false;

                num_m_blocks                      = math::ceil_div(static_cast<uint32_t>(grouped_layout[current_group_idx]), BLOCK_M);
                const auto current_m_block_cumsum = current_m_cumsum + num_m_blocks;
                if (next_block_idx < current_m_block_cumsum * num_n_blocks)
                    break;

                current_group_idx++, current_m_cumsum = current_m_block_cumsum;
            }
            get_swizzled_block_idx(next_block_idx - current_m_cumsum * num_n_blocks, m_block_idx, n_block_idx);
        }
        else
        {
            if (next_block_idx >= num_blocks)
                return false;
            get_swizzled_block_idx(next_block_idx, m_block_idx, n_block_idx);
        }
        return true;
    }
};

}   // namespace nvfp4_gemm::sched

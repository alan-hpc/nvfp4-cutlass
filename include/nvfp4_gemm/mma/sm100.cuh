#pragma once

// UMMA shared-memory descriptors for SM100/SM103.
//
// Derived from DeepGEMM (MIT, Copyright (c) 2025 DeepSeek),
// `deep_gemm/mma/sm100.cuh`, reduced to the K-major packed-FP4 operands this
// kernel uses.
//
// One deliberate divergence: `advance_packed_fp4_desc_lo` replaces DeepGEMM's
// `advance_umma_desc_lo`, which multiplies by `sizeof(dtype_t)` without dividing
// by the pack factor and therefore over-advances by 2x on E2M1 operands.

#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/atom/mma_traits_sm100.hpp>

#include <nvfp4_gemm/common/macros.cuh>
#include <nvfp4_gemm/common/tma_copy.cuh>

namespace nvfp4_gemm::mma::sm100 {

CUTLASS_DEVICE cute::UMMA::SmemDescriptor make_smem_desc(cute::UMMA::LayoutType layout,
                                                         void*                  smem_ptr,
                                                         const uint32_t&        stride_byte_offset,
                                                         const uint32_t&        leading_byte_offset)
{
    cute::UMMA::SmemDescriptor desc;
    desc.version_     = 1;   // SM100
    desc.lbo_mode_    = 0;   // legacy
    desc.layout_type_ = static_cast<uint8_t>(layout);

    const auto uint_ptr = cute::cast_smem_ptr_to_uint(smem_ptr);
    desc.start_address_ = static_cast<uint16_t>(uint_ptr >> 4);
    desc.base_offset_   = 0;

    desc.stride_byte_offset_  = stride_byte_offset >> 4;
    desc.leading_byte_offset_ = leading_byte_offset >> 4;
    return desc;
}

/// Descriptor for a scale-factor operand consumed by UTCCP.
///
/// The UTCCP layout is K-major with an 8 x 128-bit atom, and the copies this
/// kernel issues are 128 bits wide (one atom on K), so LBO can be zero.
CUTLASS_DEVICE cute::UMMA::SmemDescriptor make_sf_desc(void* smem_ptr)
{
    return make_smem_desc(cute::UMMA::LayoutType::SWIZZLE_NONE, smem_ptr, 8 * 16, 0);
}

CUTLASS_DEVICE void replace_smem_desc_addr(cute::UMMA::SmemDescriptor& desc, const void* smem_ptr)
{
    const auto uint_ptr = cute::cast_smem_ptr_to_uint(smem_ptr);
    desc.start_address_ = static_cast<uint16_t>(uint_ptr >> 4);
}

CUTLASS_DEVICE static uint32_t get_atom_base(const cute::UMMA::LayoutType& layout_type)
{
    return layout_type == cute::UMMA::LayoutType::SWIZZLE_128B_BASE32B ? 32 : 16;
}

template<uint32_t kSwizzleMode>
constexpr static cute::UMMA::LayoutType to_umma_layout_type()
{
    NVFP4_STATIC_ASSERT(kSwizzleMode == 0 or kSwizzleMode == 16 or kSwizzleMode == 32 or kSwizzleMode == 64 or
                            kSwizzleMode == 128,
                        "Invalid swizzling mode");
    if constexpr (kSwizzleMode == 0 or kSwizzleMode == 16)
        return cute::UMMA::LayoutType::SWIZZLE_NONE;
    if constexpr (kSwizzleMode == 32)
        return cute::UMMA::LayoutType::SWIZZLE_32B;
    if constexpr (kSwizzleMode == 64)
        return cute::UMMA::LayoutType::SWIZZLE_64B;
    if constexpr (kSwizzleMode == 128)
        return cute::UMMA::LayoutType::SWIZZLE_128B;
}

/// Descriptor for a K-major packed-FP4 operand.
///
/// Two E2M1 elements share a byte, so a row of `BLOCK_K` elements occupies
/// `BLOCK_K / 2` bytes and the swizzle mode must cover exactly that -- one
/// atom on the K axis, hence LBO = 0.
template<uint32_t BLOCK_MN, uint32_t BLOCK_K, uint32_t kSwizzleMode>
CUTLASS_DEVICE cute::UMMA::SmemDescriptor make_packed_fp4_desc(void* base_smem_ptr)
{
    NVFP4_STATIC_ASSERT(kSwizzleMode * 2 == BLOCK_K, "Packed-FP4 swizzle must cover one K block");

    constexpr auto layout_type        = to_umma_layout_type<kSwizzleMode>();
    const uint32_t num_non_contiguous = 128 / get_atom_base(layout_type);
    // Byte stride between consecutive 8-row atoms on MN.
    const uint32_t stride_byte_offset = num_non_contiguous * BLOCK_K / 2;
    return make_smem_desc(layout_type, base_smem_ptr, stride_byte_offset, 0);
}

/// Advance a packed-FP4 descriptor by `k_elems` along K.
///
/// The descriptor's start address is in 16-byte units, and `k_elems` FP4
/// elements are `k_elems / 2` bytes.
CUTLASS_DEVICE uint32_t advance_packed_fp4_desc_lo(const uint32_t& base, const uint32_t& k_elems)
{
    return base + ((k_elems / 2) >> 4u);
}

CUTLASS_DEVICE uint64_t make_runtime_instr_desc_with_sf_id(cute::UMMA::InstrDescriptorBlockScaled desc,
                                                           const uint32_t&                        sfa_id,
                                                           const uint32_t&                        sfb_id)
{
    desc.a_sf_id_ = sfa_id, desc.b_sf_id_ = sfb_id;
    return static_cast<uint64_t>(static_cast<uint32_t>(desc)) << 32;
}

}   // namespace nvfp4_gemm::mma::sm100

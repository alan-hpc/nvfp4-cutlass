#pragma once

// TMA tensor-map construction via the driver API.

#include <cstdint>

#include <cuda.h>

#include "exception.hpp"

namespace nvfp4_gemm::tma {

inline CUtensorMapSwizzle to_swizzle(int mode)
{
    switch (mode)
    {
    case 0:
    case 16:
        return CU_TENSOR_MAP_SWIZZLE_NONE;
    case 32:
        return CU_TENSOR_MAP_SWIZZLE_32B;
    case 64:
        return CU_TENSOR_MAP_SWIZZLE_64B;
    case 128:
        return CU_TENSOR_MAP_SWIZZLE_128B;
    default:
        NVFP4_HOST_ASSERT_MSG(false, "unsupported swizzle mode");
        return CU_TENSOR_MAP_SWIZZLE_NONE;
    }
}

/// Encode a 2D tensor map.
///
/// `gmem_inner` / `gmem_outer` and the box dims are in *elements* of `dtype`,
/// innermost dimension first; `gmem_outer_stride_bytes` is the byte stride
/// between consecutive rows.
///
/// The FP4 case is the one that bites: `CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B`
/// counts dimensions and coordinates in FP4 elements even though memory holds
/// two per byte, so a K of 128 is `gmem_inner = 128`, not 64.
inline CUtensorMap make_2d(CUtensorMapDataType dtype,
                           void*               ptr,
                           uint64_t            gmem_inner,
                           uint64_t            gmem_outer,
                           uint32_t            box_inner,
                           uint32_t            box_outer,
                           uint64_t            gmem_outer_stride_bytes,
                           int                 swizzle_mode)
{
    CUtensorMap      map{};
    const cuuint64_t dims[2]         = {gmem_inner, gmem_outer};
    const cuuint64_t strides[1]      = {gmem_outer_stride_bytes};
    const cuuint32_t box[2]          = {box_inner, box_outer};
    const cuuint32_t elem_strides[2] = {1, 1};

    NVFP4_CU_CHECK(cuTensorMapEncodeTiled(&map,
                                          dtype,
                                          /*rank=*/2,
                                          ptr,
                                          dims,
                                          strides,
                                          box,
                                          elem_strides,
                                          CU_TENSOR_MAP_INTERLEAVE_NONE,
                                          to_swizzle(swizzle_mode),
                                          CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return map;
}

}   // namespace nvfp4_gemm::tma

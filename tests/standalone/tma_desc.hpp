#pragma once
//
// Minimal TMA descriptor construction via the driver API.
//
// DeepGEMM builds these through `make_tma_2d_desc`, which needs a `torch::Tensor`.
// The standalone tests deliberately avoid PyTorch, so the same encoding is done
// here directly.  Parameter meanings follow DeepGEMM's helper exactly:
//
//   * `gmem_inner` / `gmem_outer` are in *elements* of the tensor map data type,
//     innermost dimension first.
//   * `gmem_outer_stride_bytes` is the byte stride between consecutive rows.
//   * `box_inner` / `box_outer` are the shared-memory box, also in elements.
//
// The FP4 case is the one that bites: `CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B`
// counts dimensions and coordinates in FP4 *elements* even though memory holds
// two per byte, so a K of 128 is `gmem_inner = 128`, not 64.

#include <cstdio>
#include <cstdlib>

#include <cuda.h>
#include <cuda_runtime.h>

#define CU_CHECK(expr)                                                              \
    do {                                                                            \
        CUresult _r = (expr);                                                       \
        if (_r != CUDA_SUCCESS) {                                                   \
            const char* _s = nullptr; cuGetErrorString(_r, &_s);                    \
            std::fprintf(stderr, "%s:%d: driver error %d (%s) in: %s\n",            \
                         __FILE__, __LINE__, static_cast<int>(_r),                  \
                         _s ? _s : "?", #expr);                                     \
            std::exit(1);                                                           \
        }                                                                           \
    } while (0)

#define CUDA_CHECK(expr)                                                            \
    do {                                                                            \
        cudaError_t _r = (expr);                                                    \
        if (_r != cudaSuccess) {                                                    \
            std::fprintf(stderr, "%s:%d: cuda error %d (%s) in: %s\n",              \
                         __FILE__, __LINE__, static_cast<int>(_r),                  \
                         cudaGetErrorString(_r), #expr);                            \
            std::exit(1);                                                           \
        }                                                                           \
    } while (0)

namespace tma_test {

inline CUtensorMapSwizzle swizzle_of(int mode) {
    switch (mode) {
        case 0:
        case 16:  return CU_TENSOR_MAP_SWIZZLE_NONE;
        case 32:  return CU_TENSOR_MAP_SWIZZLE_32B;
        case 64:  return CU_TENSOR_MAP_SWIZZLE_64B;
        case 128: return CU_TENSOR_MAP_SWIZZLE_128B;
        default:
            std::fprintf(stderr, "unsupported swizzle mode %d\n", mode);
            std::exit(1);
    }
}

inline CUtensorMap make_2d(CUtensorMapDataType dtype, void* ptr,
                           uint64_t gmem_inner, uint64_t gmem_outer,
                           uint32_t box_inner, uint32_t box_outer,
                           uint64_t gmem_outer_stride_bytes,
                           int swizzle_mode) {
    CUtensorMap map{};
    const cuuint64_t dims[2] = {gmem_inner, gmem_outer};
    const cuuint64_t strides[1] = {gmem_outer_stride_bytes};
    const cuuint32_t box[2] = {box_inner, box_outer};
    const cuuint32_t elem_strides[2] = {1, 1};

    CU_CHECK(cuTensorMapEncodeTiled(
        &map, dtype, /*rank=*/2, ptr, dims, strides, box, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle_of(swizzle_mode),
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return map;
}

} // namespace tma_test

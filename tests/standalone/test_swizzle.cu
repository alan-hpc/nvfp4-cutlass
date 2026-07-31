// Stage 0: check `swizzled_byte_offset` against TMA itself.
//
// The transform warps read A and write A0/A1 with plain LDS/STS, so they
// reproduce TMA's shared-memory swizzle by hand. Nothing in the transform test
// can validate the *write* side: it reads back through the same helper that
// wrote, so a systematic error cancels. Its real consumer is UMMA, and a
// mismatch there is silent -- the GEMM returns numbers of the right magnitude
// that correlate with nothing.
//
// That is not hypothetical. `swizzled_byte_offset` originally XORed the chunk
// index with `row % chunks`, which is right only for the 128 B mode; the 64 B
// mode used for A0/A1 needs `(row >> 1) % 4`. The transform test passed and the
// end-to-end GEMM came back at cosine 0.0003.
//
// So use TMA as the oracle: it applies the hardware swizzle by definition. Load
// a known pattern with a swizzled descriptor, read it back through the helper,
// and the result must be the pattern again.

#include <cstdio>
#include <vector>

#include <cutlass/arch/barrier.h>

#include <nvfp4_gemm/common/tma_copy.cuh>
#include <nvfp4_gemm/transform/dual_nvfp4.cuh>

#include "tma_desc.hpp"

using namespace nvfp4_gemm;

constexpr int kRows = 128;

template<int kRowBytes, int kSwizzleMode>
__global__ __launch_bounds__(256, 1) void swizzle_roundtrip_kernel(const __grid_constant__ cute::TmaDescriptor tma,
                                                                   uint8_t* __restrict__ out)
{
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000))
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    extern __shared__ __align__(1024) uint8_t smem[];
    auto barrier = reinterpret_cast<Barrier*>(smem + kRows * kRowBytes);

    if (threadIdx.x == 0)
    {
        barrier->init(1);
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        tma::copy<kRowBytes, kRows, kSwizzleMode, uint8_t>(&tma, barrier, smem, 0, 0);
        barrier->arrive_and_expect_tx(kRows * kRowBytes);
    }
    barrier->wait(0);
    __syncthreads();

    // Every thread walks part of the logical tile and un-swizzles it.
    for (uint32_t i = threadIdx.x; i < kRows * kRowBytes; i += blockDim.x)
    {
        const uint32_t m      = i / kRowBytes;
        const uint32_t k_byte = i % kRowBytes;
        out[i]                = smem[transform::swizzled_byte_offset<kRows, kSwizzleMode>(m, k_byte)];
    }
#endif
}

/// Returns true when the helper's mapping matches what TMA wrote.
template<int kRowBytes, int kSwizzleMode>
static bool check_case(const char* label)
{
    std::vector<uint8_t> host(kRows * kRowBytes);
    for (int m = 0; m < kRows; ++m)
        for (int k = 0; k < kRowBytes; ++k)
            // Distinct enough that any transposition or chunk swap shows up.
            host[m * kRowBytes + k] = static_cast<uint8_t>((m * 131 + k * 17 + (k >> 4) * 7) & 0xff);

    uint8_t *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, host.size()));
    CUDA_CHECK(cudaMalloc(&d_out, host.size()));
    CUDA_CHECK(cudaMemcpy(d_in, host.data(), host.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, host.size()));

    // A swizzled box is exactly one atom wide; TMA issues one copy per atom.
    const auto tma = tma_test::make_2d(CU_TENSOR_MAP_DATA_TYPE_UINT8, d_in,
                                       /*gmem_inner=*/kRowBytes, /*gmem_outer=*/kRows,
                                       /*box_inner=*/kSwizzleMode, /*box_outer=*/kRows,
                                       /*outer_stride=*/kRowBytes, kSwizzleMode);

    const int smem_size = kRows * kRowBytes + 1024;
    CUDA_CHECK(cudaFuncSetAttribute(swizzle_roundtrip_kernel<kRowBytes, kSwizzleMode>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size));
    swizzle_roundtrip_kernel<kRowBytes, kSwizzleMode><<<1, 256, smem_size>>>(tma, d_out);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint8_t> back(host.size());
    CUDA_CHECK(cudaMemcpy(back.data(), d_out, back.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    int mismatches = 0, first_m = -1, first_k = -1;
    for (int m = 0; m < kRows; ++m)
        for (int k = 0; k < kRowBytes; ++k)
            if (back[m * kRowBytes + k] != host[m * kRowBytes + k])
            {
                if (mismatches++ == 0)
                    first_m = m, first_k = k;
            }

    const bool ok = mismatches == 0;
    std::printf("  %-34s %s", label, ok ? "ok\n" : "");
    if (not ok)
        std::printf("FAIL  %d / %d bytes wrong, first at row %d byte %d\n",
                    mismatches, kRows * kRowBytes, first_m, first_k);
    return ok;
}

int main()
{
    CU_CHECK(cuInit(0));
    std::printf("swizzled_byte_offset() vs TMA:\n");

    bool ok = true;
    // The A tile: BF16 with BLOCK_K = 128 is 256 B per row, two 128 B atoms.
    ok &= check_case<256, 128>("128 B swizzle, 2 atoms (A tile)");
    ok &= check_case<128, 128>("128 B swizzle, 1 atom");
    // The A0/A1 tiles: packed FP4 with BLOCK_K = 128 is 64 B per row, one atom.
    ok &= check_case<64, 64>("64 B swizzle, 1 atom (A0/A1)");
    ok &= check_case<128, 64>("64 B swizzle, 2 atoms");

    std::printf("\n%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

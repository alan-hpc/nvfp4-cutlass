// Stage 1: isolate the A-tile swizzle replication and Algorithm 1.
//
// No MMA, no UTCCP, no epilogue.  One CTA TMA-loads a single 128x128 BF16 tile,
// runs `transform_a_tile` on it, and dumps A0/A1/SFA0/SFA1 back out in a plain
// row-major layout for the host to check against `reference.hpp`.
//
// What this actually proves, and what it does not:
//
//   * The A *read* swizzle is genuinely validated.  The host knows A[m][k] at
//     every position; if `swizzled_byte_offset` mis-addresses the BF16 tile, a
//     thread quantizes the wrong 64 values and its scales and codes will not
//     match the reference for its (m, k).  This is the highest-risk item in the
//     whole implementation, because a mistake here is silent.
//   * Algorithm 1's device numerics are validated against the host reference.
//   * The SFA atom word packing is validated.
//   * The A0/A1 *write* swizzle is only checked for self-consistency: the dump
//     reads back through the same helper that wrote it, so a systematic error
//     would cancel.  Its real consumer is UMMA -- only `test_gemm` can confirm
//     that layout.

#include <cstdio>
#include <vector>

#include <cutlass/arch/barrier.h>

#include <nvfp4_gemm/common/tma_copy.cuh>
#include <nvfp4_gemm/ptx/tcgen05.cuh>
#include <nvfp4_gemm/transform/dual_nvfp4.cuh>

#include "reference.hpp"
#include "tma_desc.hpp"

using namespace nvfp4_gemm;

constexpr int BLOCK_M           = 128;
constexpr int BLOCK_K           = 128;
constexpr int kSwizzleA         = 128;
constexpr int kSwizzleA0        = 64;
constexpr int kTransformWarps   = 8;
constexpr int kTransformThreads = kTransformWarps * 32;
constexpr int kNumThreads       = kTransformThreads + 32;   // + TMA warp

constexpr int kSFPerRow    = BLOCK_K / ref::kBlockSize;    // 8
constexpr int kAtomsPerRow = BLOCK_K / ref::kKPerSFAtom;   // 2

constexpr int kSmemA   = BLOCK_M * BLOCK_K * 2;
constexpr int kSmemA0  = BLOCK_M * BLOCK_K / 2;
constexpr int kSmemSFA = kAtomsPerRow * 128 * 4;

__global__ __launch_bounds__(kNumThreads, 1) void transform_test_kernel(const __grid_constant__ cute::TmaDescriptor tma_a,
                                                                        uint8_t* __restrict__ out_a0,
                                                                        uint8_t* __restrict__ out_a1,
                                                                        uint8_t* __restrict__ out_sfa0,
                                                                        uint8_t* __restrict__ out_sfa1)
{
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000))
    using Barrier = cutlass::arch::ClusterTransactionBarrier;

    extern __shared__ __align__(1024) uint8_t smem[];
    auto                                      smem_a    = reinterpret_cast<__nv_bfloat16*>(smem);
    auto                                      smem_a0   = smem + kSmemA;
    auto                                      smem_a1   = smem_a0 + kSmemA0;
    auto                                      smem_sfa0 = smem_a1 + kSmemA0;
    auto                                      smem_sfa1 = smem_sfa0 + kSmemSFA;
    auto                                      barrier   = reinterpret_cast<Barrier*>(smem_sfa1 + kSmemSFA);

    const auto warp_idx = cutlass::canonical_warp_idx_sync();

    if (warp_idx == 0 and cute::elect_one_sync())
    {
        barrier->init(1);
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    if (warp_idx == 0 and cute::elect_one_sync())
    {
        tma::copy<BLOCK_K, BLOCK_M, kSwizzleA, __nv_bfloat16>(
            &tma_a,
            barrier,
            smem_a,
            0,
            0);
        barrier->arrive_and_expect_tx(kSmemA);
    }
    barrier->wait(0);
    __syncthreads();

    if (threadIdx.x < kTransformThreads)
    {
        transform::transform_a_tile<
            BLOCK_M,
            BLOCK_K,
            kSwizzleA,
            kSwizzleA0,
            kTransformThreads,
            transform::ScalePolicy::DerivedDiv8>(
            smem_a,
            smem_a0,
            smem_a1,
            smem_sfa0,
            smem_sfa1,
            threadIdx.x);
    }
    __syncthreads();

    // Dump into a plain row-major layout the host can read without knowing any
    // swizzle.  Each thread handles the atom it just produced.
    if (threadIdx.x < kTransformThreads)
    {
        const uint32_t m      = threadIdx.x / kAtomsPerRow;
        const uint32_t atom   = threadIdx.x % kAtomsPerRow;
        const uint32_t k_base = atom * ref::kKPerSFAtom;

        for (uint32_t g = 0; g < ref::kKPerSFAtom / 2 / 16; ++g)
        {
            const uint32_t k_byte                     = k_base / 2 + g * 16;
            const uint32_t off                        = transform::swizzled_byte_offset<BLOCK_M, kSwizzleA0>(m, k_byte);
            const uint32_t plain                      = m * (BLOCK_K / 2) + k_byte;
            *reinterpret_cast<uint4*>(out_a0 + plain) = *reinterpret_cast<const uint4*>(smem_a0 + off);
            *reinterpret_cast<uint4*>(out_a1 + plain) = *reinterpret_cast<const uint4*>(smem_a1 + off);
        }

        const uint32_t sf_off   = atom * (128 * 4) + transform::sf_atom_word_idx(m) * 4;
        const uint32_t plain_sf = m * kSFPerRow + atom * transform::kSFPerAtom;
        *reinterpret_cast<uint32_t*>(out_sfa0 + plain_sf) =
            *reinterpret_cast<const uint32_t*>(smem_sfa0 + sf_off);
        *reinterpret_cast<uint32_t*>(out_sfa1 + plain_sf) =
            *reinterpret_cast<const uint32_t*>(smem_sfa1 + sf_off);
    }
#endif
}

/// Report how the hardware converter breaks ties on the E2M1 grid.
///
/// Every midpoint between two E2M1 levels is a tie, and the rule is not
/// something to assume: the reference here rounds half-to-even, and a run on
/// B300 disagreed at 2.5 (hardware -> 3.0, reference -> 2.0). The dual-pass
/// structure absorbs it -- q1 moves to compensate -- but a single-pass kernel
/// would not, and the reference should describe the hardware rather than a
/// guess.
__global__ void e2m1_tie_probe_kernel(uint8_t* __restrict__ out)
{
    const float midpoints[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
    if (threadIdx.x == 0)
    {
        for (int i = 0; i < 7; ++i)
        {
            out[i]     = static_cast<uint8_t>(nvfp4_gemm::ptx::cvt_e2m1x2(0.0f, midpoints[i]));
            out[7 + i] = static_cast<uint8_t>(nvfp4_gemm::ptx::cvt_e2m1x2(0.0f, -midpoints[i]));
        }
    }
}

static void report_tie_rule()
{
    uint8_t* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, 14));
    e2m1_tie_probe_kernel<<<1, 32>>>(d_out);
    CUDA_CHECK(cudaGetLastError());

    uint8_t codes[14];
    CUDA_CHECK(cudaMemcpy(codes, d_out, 14, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out));

    const float midpoints[7]  = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
    int         disagreements = 0;
    std::printf("E2M1 tie rule (midpoint -> hardware / this reference):\n  ");
    for (int i = 0; i < 7; ++i)
    {
        // The converter packs the first argument into the low nibble.
        const float hw   = ref::decode_e2m1(codes[i] & 0xf);
        const float host = ref::decode_e2m1(ref::quantize_e2m1_code(midpoints[i]));
        std::printf("%.2f -> %.1f/%.1f%s  ", midpoints[i], hw, host, hw == host ? "" : " (!)");
        if (hw != host)
            ++disagreements;
    }
    std::printf("\n");
    if (disagreements)
        std::printf("  %d midpoint(s) differ. Harmless here -- the residual pass compensates --\n"
                    "  but tests/python/reference/dual_nvfp4.py should adopt the hardware rule.\n",
                    disagreements);
    std::printf("\n");
}

int main()
{
    // ---------------------------------------------------------------- input --
    ref::Rng           rng(1234);
    std::vector<float> a_host(BLOCK_M * BLOCK_K);
    for (auto& v : a_host)
        v = rng.normal();

    // A few adversarial rows: an all-zero row exercises the `derived_div8` scale
    // floor, and a row of huge values exercises E2M1 saturation.
    for (int k = 0; k < BLOCK_K; ++k)
    {
        a_host[0 * BLOCK_K + k] = 0.0f;
        a_host[1 * BLOCK_K + k] = (k % 2 ? 1e4f : -1e4f);
        a_host[2 * BLOCK_K + k] = 1e-8f;
    }

    // Round-trip through BF16 so the host reference sees exactly what the TMA
    // will deliver to the device.
    std::vector<__nv_bfloat16> a_bf16(BLOCK_M * BLOCK_K);
    for (size_t i = 0; i < a_host.size(); ++i)
        a_bf16[i] = __float2bfloat16(a_host[i]);
    for (size_t i = 0; i < a_host.size(); ++i)
        a_host[i] = __bfloat162float(a_bf16[i]);

    __nv_bfloat16* d_a  = nullptr;
    uint8_t *      d_a0 = nullptr, *d_a1 = nullptr, *d_sfa0 = nullptr, *d_sfa1 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, a_bf16.size() * 2));
    CUDA_CHECK(cudaMalloc(&d_a0, BLOCK_M * BLOCK_K / 2));
    CUDA_CHECK(cudaMalloc(&d_a1, BLOCK_M * BLOCK_K / 2));
    CUDA_CHECK(cudaMalloc(&d_sfa0, BLOCK_M * kSFPerRow));
    CUDA_CHECK(cudaMalloc(&d_sfa1, BLOCK_M * kSFPerRow));
    CUDA_CHECK(cudaMemcpy(d_a, a_bf16.data(), a_bf16.size() * 2, cudaMemcpyHostToDevice));

    CU_CHECK(cuInit(0));
    // BF16 K-major: 128 B swizzle means a 64-element inner box.
    auto tma_a = tma_test::make_2d(CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, d_a, BLOCK_K, BLOCK_M, kSwizzleA / 2, BLOCK_M, BLOCK_K * 2, kSwizzleA);

    const int smem_size = kSmemA + 2 * kSmemA0 + 2 * kSmemSFA + 1024;
    CUDA_CHECK(cudaFuncSetAttribute(transform_test_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size));
    std::printf("shared memory: %d B, threads: %d\n", smem_size, kNumThreads);

    transform_test_kernel<<<1, kNumThreads, smem_size>>>(tma_a, d_a0, d_a1, d_sfa0, d_sfa1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    report_tie_rule();

    std::vector<uint8_t> a0(BLOCK_M * BLOCK_K / 2), a1(BLOCK_M * BLOCK_K / 2);
    std::vector<uint8_t> sfa0(BLOCK_M * kSFPerRow), sfa1(BLOCK_M * kSFPerRow);
    CUDA_CHECK(cudaMemcpy(a0.data(), d_a0, a0.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(a1.data(), d_a1, a1.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sfa0.data(), d_sfa0, sfa0.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sfa1.data(), d_sfa1, sfa1.size(), cudaMemcpyDeviceToHost));

    // ------------------------------------------------------------- compare ---
    // The question this test answers is "does the device reproduce the
    // reference", so the primary metric compares the two *reconstructions*, not
    // the raw codes and not the original input.
    //
    // Codes are allowed to differ where the difference is numerically inert, and
    // two such cases really do occur:
    //
    //   * signed zero. The residual is `(x * rcp(s0) - dec(q0)) * 8`; when it
    //     lands on zero, `rcp.approx` versus an exact divide decides whether it
    //     arrives from below, so the device may emit -0.0 (code 8) where the
    //     reference emits +0.0 (code 0).
    //   * ties on the E2M1 grid. If the hardware and the reference break a
    //     midpoint the other way, q0 moves one step and q1 moves to compensate:
    //     -3*s0 + 4*(s0/8) and -2*s0 + -4*(s0/8) are both exactly -2.5*s0. The
    //     dual-pass structure absorbs it.
    //
    // Measuring against the *input* instead would mostly measure NVFP4's own
    // range limit: with G_A = 1 the largest representable magnitude is
    // 448 * 6 = 2688, so the deliberately out-of-range row below saturates by
    // design, in the reference exactly as much as on the device.
    int    scale_mismatch = 0, real_mismatch = 0, benign_code_diff = 0, first_bad_row = -1;
    double max_dev_ref_err = 0.0, sum_sq_dev_ref = 0.0, sum_sq_ref_recon = 0.0;
    double sum_sq_quant_err = 0.0, sum_sq_input = 0.0, max_quant_err = 0.0;

    for (int m = 0; m < BLOCK_M; ++m)
    {
        for (int b = 0; b < kSFPerRow; ++b)
        {
            uint8_t q0[ref::kBlockSize], q1[ref::kBlockSize], s0, s1;
            ref::decompose_block(&a_host[static_cast<size_t>(m) * BLOCK_K + b * ref::kBlockSize], q0, q1, s0, s1);

            const uint8_t got_s0 = sfa0[m * kSFPerRow + b];
            const uint8_t got_s1 = sfa1[m * kSFPerRow + b];
            if (got_s0 != s0 || got_s1 != s1)
            {
                if (++scale_mismatch <= 5)
                    std::printf("  scale mismatch m=%d block=%d: s0 %u vs %u, s1 %u vs %u\n", m, b, got_s0, s0, got_s1, s1);
                if (first_bad_row < 0)
                    first_bad_row = m;
            }

            uint8_t got_q0[ref::kBlockSize], got_q1[ref::kBlockSize];
            for (int i = 0; i < ref::kBlockSize; ++i)
            {
                const int     col   = b * ref::kBlockSize + i;
                const uint8_t byte0 = a0[static_cast<size_t>(m) * (BLOCK_K / 2) + col / 2];
                const uint8_t byte1 = a1[static_cast<size_t>(m) * (BLOCK_K / 2) + col / 2];
                // Element 2i occupies the low nibble, 2i+1 the high nibble.
                got_q0[i] = (col % 2) ? (byte0 >> 4) : (byte0 & 0xf);
                got_q1[i] = (col % 2) ? (byte1 >> 4) : (byte1 & 0xf);
            }

            float dev_recon[ref::kBlockSize], ref_recon[ref::kBlockSize];
            ref::reconstruct_block(got_q0, got_q1, got_s0, got_s1, dev_recon);
            ref::reconstruct_block(q0, q1, s0, s1, ref_recon);

            // Half of the finest quantum this block can express. A code
            // difference that stays inside it cannot change any downstream MMA.
            const double inert = 0.5 * ref::decode_e4m3(s1);

            for (int i = 0; i < ref::kBlockSize; ++i)
            {
                const int    col  = b * ref::kBlockSize + i;
                const double diff = static_cast<double>(dev_recon[i]) - ref_recon[i];

                if (got_q0[i] != q0[i] || got_q1[i] != q1[i])
                {
                    if (std::fabs(diff) <= inert)
                    {
                        ++benign_code_diff;
                    }
                    else
                    {
                        if (++real_mismatch <= 5)
                            std::printf("  code mismatch m=%d k=%d: q0 %u vs %u, q1 %u vs %u  (recon %.6g vs %.6g)\n",
                                        m,
                                        col,
                                        got_q0[i],
                                        q0[i],
                                        got_q1[i],
                                        q1[i],
                                        dev_recon[i],
                                        ref_recon[i]);
                        if (first_bad_row < 0)
                            first_bad_row = m;
                    }
                }

                max_dev_ref_err = std::fmax(max_dev_ref_err, std::fabs(diff));
                sum_sq_dev_ref += diff * diff;
                sum_sq_ref_recon += static_cast<double>(ref_recon[i]) * ref_recon[i];

                const double want      = a_host[static_cast<size_t>(m) * BLOCK_K + col];
                const double quant_err = static_cast<double>(dev_recon[i]) - want;
                sum_sq_quant_err += quant_err * quant_err;
                sum_sq_input += want * want;
                max_quant_err = std::fmax(max_quant_err, std::fabs(quant_err));
            }
        }
    }

    const double rel_dev_ref = std::sqrt(sum_sq_dev_ref / (sum_sq_ref_recon + 1e-30));
    const double rel_quant   = std::sqrt(sum_sq_quant_err / (sum_sq_input + 1e-30));

    std::printf("\nscale mismatches:        %d / %d\n", scale_mismatch, BLOCK_M * kSFPerRow);
    std::printf("code mismatches (real):  %d / %d\n", real_mismatch, BLOCK_M * BLOCK_K);
    std::printf("code diffs (inert):      %d   (signed zero / grid ties, same value)\n", benign_code_diff);
    std::printf("device vs reference:     rel L2 %.3e, max abs %.3e\n", rel_dev_ref, max_dev_ref_err);
    std::printf("device vs input:         rel L2 %.6f, max abs %.6g\n", rel_quant, max_quant_err);
    std::printf("  (the second line is NVFP4's own error, and includes the row of\n"
                "   +-1e4 that saturates on purpose: with G_A = 1 nothing above\n"
                "   448 * 6 = 2688 is representable.)\n");

    bool ok = true;
    if (scale_mismatch)
    {
        std::printf("\nFAIL: block scales differ from the reference (first bad row %d)\n", first_bad_row);
        std::printf("      Scales come from the amax of the 64 values a thread read, so this\n"
                    "      almost always means swizzled_byte_offset() disagrees with what TMA\n"
                    "      actually wrote into shared memory.\n");
        ok = false;
    }
    if (real_mismatch)
    {
        std::printf("\nFAIL: %d codes differ by more than the block's finest quantum\n", real_mismatch);
        std::printf("      Differences confined to adjacent element pairs point at the E2M1\n"
                    "      nibble packing order instead.\n");
        ok = false;
    }
    // rcp.approx costs about 2^-23 relative, and a code that flips at a rounding
    // boundary is absorbed by the residual pass, so anything above this is a
    // genuine disagreement rather than arithmetic noise.
    if (rel_dev_ref > 1e-5)
    {
        std::printf("\nFAIL: device reconstruction differs from the reference by %.3e\n", rel_dev_ref);
        ok = false;
    }

    std::printf("\n%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

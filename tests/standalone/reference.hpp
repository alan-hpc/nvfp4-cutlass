#pragma once
//
// Host-side reference for the dual-NVFP4 algorithm.
//
// This is the C++ twin of `tests/reference/dual_nvfp4.py` and follows the same
// operation order as the kernel: E4M3 round trip on the scale, `s1` re-rounded
// rather than exactly `s0 / 8`, and the residual formed in the normalized
// domain.  The E4M3 conversions go through the CUDA headers' host-callable
// intrinsics, so they are bit-exact with the device.
//
// The one deliberate difference: the kernel uses `rcp.approx.ftz.f32` where this
// uses true division.  That is worth ~1 ulp and is why the tests compare with a
// tolerance rather than bitwise.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace ref {

constexpr int   kBlockSize          = 16;   // NVFP4 scale block along K
constexpr int   kKPerSFAtom         = 64;   // one 128x4 scale atom
constexpr float kE2M1Max            = 6.0f;
constexpr float kS0FloorDerivedDiv8 = 0.015625f;   // 2^-6 == 8 * 2^-9

inline float e2m1_levels(int idx)
{
    static const float levels[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    return levels[idx];
}

/// Round to nearest even on the E2M1 grid, saturating to +-6.  Returns the code
/// in the low nibble (sign in bit 3).
inline uint8_t quantize_e2m1_code(float x)
{
    const float   mag  = std::fabs(x);
    const uint8_t sign = (std::signbit(x) ? 0x8u : 0x0u);

    static const float mid[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
    static const int   tie[7] = {0, 2, 2, 4, 4, 6, 6};   // round-half-to-even targets

    int idx = 0;
    for (int i = 0; i < 7; ++i)
        if (mag > mid[i])
            idx = i + 1;
    for (int i = 0; i < 7; ++i)
        if (mag == mid[i])
            idx = tie[i];
    if (std::isnan(mag) || std::isinf(mag))
        idx = 7;

    return static_cast<uint8_t>(sign | idx);
}

inline float decode_e2m1(uint8_t code)
{
    const float mag = e2m1_levels(code & 0x7u);
    return (code & 0x8u) ? -mag : mag;
}

inline uint8_t quantize_e4m3_code(float x)
{
    return static_cast<uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
}

inline float decode_e4m3(uint8_t code)
{
    return __half2float(__nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(code), __NV_E4M3));
}

/// Algorithm 1 on one 16-element block.
///
/// `q0_codes` / `q1_codes` receive 16 four-bit codes (one per byte, unpacked --
/// packing is the caller's business since it depends on the target layout).
inline void decompose_block(const float* x, uint8_t* q0_codes, uint8_t* q1_codes, uint8_t& s0_code, uint8_t& s1_code)
{
    float amax = 0.0f;
    for (int i = 0; i < kBlockSize; ++i)
        amax = std::fmax(amax, std::fabs(x[i]));

    s0_code        = quantize_e4m3_code(std::fmax(amax * (1.0f / 6.0f), kS0FloorDerivedDiv8));
    const float s0 = decode_e4m3(s0_code);
    s1_code        = quantize_e4m3_code(s0 * 0.125f);

    for (int i = 0; i < kBlockSize; ++i)
    {
        const float u = x[i] / s0;
        q0_codes[i]   = quantize_e2m1_code(u);
        q1_codes[i]   = quantize_e2m1_code((u - decode_e2m1(q0_codes[i])) * 8.0f);
    }
}

/// x_hat = dec(q0) * s0 + dec(q1) * s1
inline void reconstruct_block(const uint8_t* q0_codes, const uint8_t* q1_codes, uint8_t s0_code, uint8_t s1_code, float* out)
{
    const float s0 = decode_e4m3(s0_code), s1 = decode_e4m3(s1_code);
    for (int i = 0; i < kBlockSize; ++i)
        out[i] = decode_e2m1(q0_codes[i]) * s0 + decode_e2m1(q1_codes[i]) * s1;
}

/// Offline NVFP4 weight quantization: w ~= dec(q) * S * G, with G per expert.
///
/// Outputs: `q_codes` is (E, N, K), `sf_codes` is (E, N, K/16), `g` is (E,).
inline void quantize_weight(const std::vector<float>& w, int num_experts, int n, int k, std::vector<uint8_t>& q_codes, std::vector<uint8_t>& sf_codes, std::vector<float>& g)
{
    q_codes.assign(static_cast<size_t>(num_experts) * n * k, 0);
    sf_codes.assign(static_cast<size_t>(num_experts) * n * (k / kBlockSize), 0);
    g.assign(num_experts, 1.0f);

    for (int e = 0; e < num_experts; ++e)
    {
        float amax = 0.0f;
        for (int i = 0; i < n * k; ++i)
            amax = std::fmax(amax, std::fabs(w[static_cast<size_t>(e) * n * k + i]));
        g[e] = std::fmax(amax / (448.0f * 6.0f), 1e-30f);

        for (int row = 0; row < n; ++row)
        {
            for (int b = 0; b < k / kBlockSize; ++b)
            {
                const size_t base       = (static_cast<size_t>(e) * n + row) * k + b * kBlockSize;
                float        block_amax = 0.0f;
                for (int i = 0; i < kBlockSize; ++i)
                    block_amax = std::fmax(block_amax, std::fabs(w[base + i]));

                const uint8_t s = quantize_e4m3_code(std::fmax((block_amax / 6.0f) / g[e], 1.0f / 512.0f));

                sf_codes[(static_cast<size_t>(e) * n + row) * (k / kBlockSize) + b] = s;

                const float denom = std::fmax(decode_e4m3(s) * g[e], 1e-30f);
                for (int i = 0; i < kBlockSize; ++i)
                    q_codes[base + i] = quantize_e2m1_code(w[base + i] / denom);
            }
        }
    }
}

/// Dequantized weight for expert `e`, row-major (N, K).
inline std::vector<float> dequantize_weight(const std::vector<uint8_t>& q_codes,
                                            const std::vector<uint8_t>& sf_codes,
                                            const std::vector<float>&   g,
                                            int                         e,
                                            int                         n,
                                            int                         k)
{
    std::vector<float> out(static_cast<size_t>(n) * k);
    for (int row = 0; row < n; ++row)
    {
        for (int col = 0; col < k; ++col)
        {
            const size_t qi = (static_cast<size_t>(e) * n + row) * k + col;
            const size_t si = (static_cast<size_t>(e) * n + row) * (k / kBlockSize) + col / kBlockSize;
            out[static_cast<size_t>(row) * k + col] =
                decode_e2m1(q_codes[qi]) * decode_e4m3(sf_codes[si]) * g[e];
        }
    }
    return out;
}

/// Ground truth, exactly as the algorithm doc defines it:
/// C_ref = A_bf16 * (dec(W) * S_W * G_W)^T, accumulated in FP32.
///
/// The BF16 activation must go in unquantized -- putting any FP8/FP4
/// intermediate on A here would understate the error the kernel actually makes.
inline std::vector<float> reference_gemm(const std::vector<float>&   a,
                                         const std::vector<uint8_t>& q_codes,
                                         const std::vector<uint8_t>& sf_codes,
                                         const std::vector<float>&   g,
                                         const std::vector<int>&     expert_of_row,
                                         int                         m,
                                         int                         n,
                                         int                         k,
                                         int                         num_experts)
{
    std::vector<float> out(static_cast<size_t>(m) * n, 0.0f);
    for (int e = 0; e < num_experts; ++e)
    {
        const auto w = dequantize_weight(q_codes, sf_codes, g, e, n, k);
        for (int row = 0; row < m; ++row)
        {
            if (expert_of_row[row] != e)
                continue;
            for (int col = 0; col < n; ++col)
            {
                float acc = 0.0f;
                for (int kk = 0; kk < k; ++kk)
                    acc += a[static_cast<size_t>(row) * k + kk] * w[static_cast<size_t>(col) * k + kk];
                out[static_cast<size_t>(row) * n + col] = acc;
            }
        }
    }
    return out;
}

/// Same GEMM, but with A pushed through the dual-NVFP4 decomposition first.
/// This is what the kernel should reproduce.
inline std::vector<float> dual_gemm(const std::vector<float>&   a,
                                    const std::vector<uint8_t>& q_codes,
                                    const std::vector<uint8_t>& sf_codes,
                                    const std::vector<float>&   g,
                                    const std::vector<int>&     expert_of_row,
                                    int                         m,
                                    int                         n,
                                    int                         k,
                                    int                         num_experts)
{
    std::vector<float> a_hat(static_cast<size_t>(m) * k);
    for (int row = 0; row < m; ++row)
    {
        for (int b = 0; b < k / kBlockSize; ++b)
        {
            const size_t base = static_cast<size_t>(row) * k + b * kBlockSize;
            uint8_t      q0[kBlockSize], q1[kBlockSize], s0, s1;
            decompose_block(&a[base], q0, q1, s0, s1);
            reconstruct_block(q0, q1, s0, s1, &a_hat[base]);
        }
    }
    return reference_gemm(a_hat, q_codes, sf_codes, g, expert_of_row, m, n, k, num_experts);
}

inline double cosine(const std::vector<float>& x, const std::vector<float>& y)
{
    double dot = 0, nx = 0, ny = 0;
    for (size_t i = 0; i < x.size(); ++i)
    {
        dot += static_cast<double>(x[i]) * y[i];
        nx += static_cast<double>(x[i]) * x[i];
        ny += static_cast<double>(y[i]) * y[i];
    }
    return dot / (std::sqrt(nx) * std::sqrt(ny) + 1e-30);
}

/// Deterministic pseudo-random values, so a failure is reproducible without
/// pulling in <random>'s implementation-defined engines.
struct Rng
{
    uint32_t state;
    explicit Rng(uint32_t seed)
        : state(seed ? seed : 1u)
    {}
    uint32_t next()
    {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    }
    float normal()
    {
        // Sum of 4 uniforms, centred and scaled -- close enough to Gaussian for
        // an accuracy smoke test, and cheap.
        float s = 0.0f;
        for (int i = 0; i < 4; ++i)
            s += static_cast<float>(next() >> 8) / 16777216.0f;
        return (s - 2.0f) * 1.732f;
    }
};

}   // namespace ref

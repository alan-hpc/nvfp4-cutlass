"""Numerics tests for the dual-NVFP4 reference (CPU only, no GPU required).

Run with `python tests/test_dual_nvfp4_reference.py`, or under pytest.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reference.dual_nvfp4 import (  # noqa: E402
    BLOCK_SIZE,
    E2M1_LEVELS,
    E2M1_MAX,
    S0_FLOOR_DERIVED_DIV8,
    cosine_similarity,
    decompose_dual_nvfp4,
    dual_nvfp4_grouped_gemm,
    quantize_e2m1,
    quantize_weight_nvfp4,
    reference_grouped_gemm,
)


def test_e2m1_grid_and_ties() -> None:
    """Every grid point is exact, saturation clips at 6, ties round to even."""
    grid = torch.tensor([-v for v in reversed(E2M1_LEVELS)] + list(E2M1_LEVELS))
    assert torch.equal(quantize_e2m1(grid), grid), "grid points must be fixed points"

    saturating = torch.tensor([6.5, 1e4, -7.0, float("inf"), float("-inf")])
    expected = torch.tensor([E2M1_MAX, E2M1_MAX, -E2M1_MAX, E2M1_MAX, -E2M1_MAX])
    assert torch.equal(quantize_e2m1(saturating), expected), "must saturate to +-6"

    ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    expected_ties = torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0])
    assert torch.equal(quantize_e2m1(ties), expected_ties), "ties must round to even"
    print("ok  e2m1 grid / saturation / ties")


def test_zero_block_is_exact_and_finite() -> None:
    """A block of zeros must decode back to zero without inf/NaN.

    This is the `M = 0` case the doc calls out: the scale is clamped to the
    smallest positive E4M3, so `s1 = s0 / 8` stays representable.
    """
    x = torch.zeros(1, BLOCK_SIZE, dtype=torch.bfloat16)
    dual = decompose_dual_nvfp4(x)

    assert torch.all(dual.s0 >= S0_FLOOR_DERIVED_DIV8), "s0 must be floored"
    assert torch.all(dual.s1 > 0), "s1 must not underflow to zero"
    assert torch.all(torch.isfinite(dual.s1)), "s1 must stay finite"
    assert torch.equal(dual.reconstruct(), torch.zeros(1, BLOCK_SIZE)), "zeros decode to zeros"
    print("ok  zero block stays finite and exact")


def test_single_pass_grid_points_are_exact() -> None:
    """If a block is already on the E2M1 grid scaled by a power of two, pass 0 is exact."""
    block = torch.tensor([list(E2M1_LEVELS) + [-v for v in E2M1_LEVELS]], dtype=torch.float32)
    dual = decompose_dual_nvfp4(block)
    err = (dual.reconstruct() - block).abs().max()
    assert err == 0.0, f"expected exact reconstruction, got max err {err}"
    print("ok  on-grid block reconstructs exactly")


def test_residual_pass_beats_single_pass() -> None:
    """The whole point of dual-A: pass 1 must cut the reconstruction error hard."""
    torch.manual_seed(0)
    x = torch.randn(512, 256, dtype=torch.bfloat16)
    dual = decompose_dual_nvfp4(x)

    x32 = x.to(torch.float32)
    blocks = x32.reshape(512, -1, BLOCK_SIZE)
    single = (dual.q0.reshape(blocks.shape) * dual.s0.unsqueeze(-1)).reshape(x32.shape)

    err_single = (single - x32).norm() / x32.norm()
    err_dual = (dual.reconstruct() - x32).norm() / x32.norm()
    ratio = float(err_single / err_dual)

    assert err_dual < err_single, "residual pass must reduce error"
    assert ratio > 4.0, f"expected >4x error reduction, got {ratio:.2f}x"
    print(f"ok  dual vs single pass: {err_single:.5f} -> {err_dual:.5f} ({ratio:.1f}x better)")


def test_residual_amax_is_at_least_as_accurate() -> None:
    """`residual_amax` costs an extra reduction and must buy accuracy for it."""
    torch.manual_seed(1)
    x = torch.randn(256, 128, dtype=torch.bfloat16)
    x32 = x.to(torch.float32)

    err_div8 = (decompose_dual_nvfp4(x, "derived_div8").reconstruct() - x32).norm()
    err_amax = (decompose_dual_nvfp4(x, "residual_amax").reconstruct() - x32).norm()

    assert err_amax <= err_div8, "residual_amax should not be worse than derived_div8"
    print(f"ok  derived_div8 {err_div8:.4f} vs residual_amax {err_amax:.4f}")


def test_grouped_gemm_matches_ground_truth() -> None:
    """End-to-end Algorithm 2 against the doc's ground truth definition."""
    torch.manual_seed(2)
    num_experts, m_per_expert, n, k = 2, 128, 64, 256

    a = torch.randn(num_experts * m_per_expert, k, dtype=torch.bfloat16)
    w = torch.randn(num_experts, n, k, dtype=torch.bfloat16) * 0.5
    w_q, w_sf, w_g = quantize_weight_nvfp4(w)
    m_indptr = torch.tensor([0, m_per_expert, num_experts * m_per_expert])

    out = dual_nvfp4_grouped_gemm(a, w_q, w_sf, w_g, m_indptr)
    ref = reference_grouped_gemm(a, w_q, w_sf, w_g, m_indptr)

    cos = cosine_similarity(out, ref)
    rel = float((out.to(torch.float32) - ref).norm() / ref.norm())

    assert cos > 0.999, f"cosine {cos:.6f} too low"
    print(f"ok  grouped GEMM vs ground truth: cosine {cos:.6f}, rel L2 {rel:.6f}")


def test_reported_accuracy_on_doc_shape() -> None:
    """Report accuracy on the doc's main shape family (scaled down for CPU).

    This is a reporting test, not a tight assertion: it prints the numbers the
    CUDA kernel must reproduce on hardware.
    """
    torch.manual_seed(3)
    num_experts, m_per_expert, n, k = 2, 256, 128, 512

    a = torch.randn(num_experts * m_per_expert, k, dtype=torch.bfloat16)
    w = torch.randn(num_experts, n, k, dtype=torch.bfloat16) * 0.5
    w_q, w_sf, w_g = quantize_weight_nvfp4(w)
    m_indptr = torch.tensor([0, m_per_expert, num_experts * m_per_expert])

    ref = reference_grouped_gemm(a, w_q, w_sf, w_g, m_indptr)
    for policy in ("derived_div8", "residual_amax"):
        out = dual_nvfp4_grouped_gemm(a, w_q, w_sf, w_g, m_indptr, scale_policy=policy)
        print(f"    {policy:>14}: cosine {cosine_similarity(out, ref):.6f}")

    # Single-pass NVFP4 activations, for context on what the second pass buys.
    dual = decompose_dual_nvfp4(a)
    blocks = a.to(torch.float32).reshape(a.shape[0], -1, BLOCK_SIZE)
    single = (dual.q0.reshape(blocks.shape) * dual.s0.unsqueeze(-1)).reshape(a.shape)
    single_out = torch.empty_like(ref)
    from reference.dual_nvfp4 import dequantize_nvfp4  # noqa: PLC0415

    for e in range(num_experts):
        lo, hi = int(m_indptr[e]), int(m_indptr[e + 1])
        w_deq = dequantize_nvfp4(w_q[e], w_sf[e]) * float(w_g[e])
        single_out[lo:hi] = single[lo:hi] @ w_deq.T
    print(f"    {'single-pass':>14}: cosine {cosine_similarity(single_out, ref):.6f}")
    print("ok  accuracy report")


def main() -> int:
    tests = [
        test_e2m1_grid_and_ties,
        test_zero_block_is_exact_and_finite,
        test_single_pass_grid_points_are_exact,
        test_residual_pass_beats_single_pass,
        test_residual_amax_is_at_least_as_accurate,
        test_grouped_gemm_matches_ground_truth,
        test_reported_accuracy_on_doc_shape,
    ]
    for test in tests:
        test()
    print(f"\n{len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

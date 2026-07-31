"""BF16 -> two-pass NVFP4 (MSD + residual) reference, bit-faithful to the kernel.

This module is the executable specification for `Algorithm 1` (per 16-element
block decomposition) and `Algorithm 2` (two-pass block-scaled GEMM) described in
`docs/bf16-dual-nvfp4-algorithm.html`.  It is pure PyTorch and runs on CPU, so it
can be used to validate the CUDA kernel's numerics without a Blackwell GPU.

Everything here mirrors the operation order of the device code, not the
idealized math:

  * the block scale goes through a real E4M3 round trip,
  * `s1` is derived as `Q_e4m3(float(s0) * 2**-3)` -- it is re-rounded, it is
    not exactly `s0 / 8`,
  * the residual is formed in the *normalized* domain, i.e.
    `(x * rcp(s0) - q0) * 8`, never as `(x - q0 * s0) / s1`,
  * E2M1 conversion is round-to-nearest-even with saturation to +-6.

The only intentional deviation is `rcp_approx`: the kernel uses
`rcp.approx.ftz.f32` (~1 ulp of 2**-23 relative error), which has no portable
CPU equivalent.  `reciprocal_mode="exact"` (the default) uses true division;
`reciprocal_mode="rn"` rounds the reciprocal to FP32 first, which brackets the
hardware behaviour closely enough for accuracy studies.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# E2M1 (FP4) representable magnitudes.  Max finite value is 6.0.
E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_MAX = 6.0

# Midpoints between consecutive E2M1 levels.  A value exactly on a midpoint is a
# tie and must round to the level with an even mantissa bit.
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
# Tie target for each midpoint, picked by round-half-to-even on the 1-bit
# mantissa: 0.25->0, 0.75->1.0, 1.25->1.0, 1.75->2.0, 2.5->2.0, 3.5->4.0, 5.0->4.0.
_E2M1_TIE_LEVEL_IDX = (0, 2, 2, 4, 4, 6, 6)

# NVFP4 block size along K.
BLOCK_SIZE = 16

# Smallest positive E4M3 subnormal is 2**-9.  `derived_div8` clamps s0 to
# 8 * 2**-9 == 2**-6 so that s1 = s0 / 8 cannot underflow to zero.
E4M3_MIN_SUBNORMAL = 2.0**-9
S0_FLOOR_DERIVED_DIV8 = 8.0 * E4M3_MIN_SUBNORMAL  # == 2**-6


def quantize_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round to nearest even on the E2M1 grid, saturating to [-6, 6].

    Mirrors `cvt.rn.satfinite.e2m1x2.f32`.  Returns the *decoded* FP32 values,
    since the reference never needs the 4-bit codes themselves.
    """
    x = x.to(torch.float32)
    sign = torch.sign(x)
    mag = x.abs()

    # Start at the lowest level and step up past every midpoint we exceed.
    out = torch.zeros_like(mag)
    for level_idx in range(1, len(E2M1_LEVELS)):
        mid = _E2M1_MIDPOINTS[level_idx - 1]
        out = torch.where(mag > mid, torch.full_like(out, E2M1_LEVELS[level_idx]), out)

    # Fix up exact ties, which the strict `>` above pushed to the lower level.
    for mid, tie_idx in zip(_E2M1_MIDPOINTS, _E2M1_TIE_LEVEL_IDX):
        out = torch.where(mag == mid, torch.full_like(out, E2M1_LEVELS[tie_idx]), out)

    # NaN/Inf saturate to the max finite magnitude, matching `satfinite`.
    out = torch.where(torch.isnan(mag), torch.full_like(out, E2M1_MAX), out)
    out = torch.where(torch.isinf(mag), torch.full_like(out, E2M1_MAX), out)
    return sign * out


def quantize_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Round to nearest even on the E4M3 grid, returning decoded FP32 values."""
    return x.to(torch.float32).to(torch.float8_e4m3fn).to(torch.float32)


@dataclass
class DualNVFP4:
    """Result of decomposing a BF16 tensor into two NVFP4 passes."""

    q0: torch.Tensor  # decoded E2M1 values, same shape as the input
    s0: torch.Tensor  # decoded E4M3 block scales, shape (..., K // 16)
    q1: torch.Tensor
    s1: torch.Tensor

    def reconstruct(self) -> torch.Tensor:
        """x_hat = q0 * s0 + q1 * s1, in FP32."""
        blocks = self.q0.shape[:-1] + (self.q0.shape[-1] // BLOCK_SIZE, BLOCK_SIZE)
        q0 = self.q0.reshape(blocks)
        q1 = self.q1.reshape(blocks)
        s0 = self.s0.unsqueeze(-1)
        s1 = self.s1.unsqueeze(-1)
        return (q0 * s0 + q1 * s1).reshape(self.q0.shape)


def decompose_dual_nvfp4(
    x: torch.Tensor,
    scale_policy: str = "derived_div8",
    reciprocal_mode: str = "exact",
) -> DualNVFP4:
    """Algorithm 1, applied to every 16-element block along the last axis.

    Args:
        x: BF16 (or any float) tensor whose last dim is a multiple of 16.
        scale_policy: ``"derived_div8"`` sets ``s1 = Q_e4m3(s0 * 2**-3)`` with no
            extra reduction (the kernel default).  ``"residual_amax"`` spends a
            second amax pass on the residual, which is more accurate but slower.
        reciprocal_mode: ``"exact"`` divides by ``s0``; ``"rn"`` multiplies by an
            FP32-rounded reciprocal, closer to the device's ``rcp.approx.ftz``.
    """
    if x.shape[-1] % BLOCK_SIZE != 0:
        raise ValueError(f"last dim {x.shape[-1]} is not a multiple of {BLOCK_SIZE}")
    if scale_policy not in ("derived_div8", "residual_amax"):
        raise ValueError(f"unknown scale_policy {scale_policy!r}")
    if reciprocal_mode not in ("exact", "rn"):
        raise ValueError(f"unknown reciprocal_mode {reciprocal_mode!r}")

    orig_shape = x.shape
    xb = x.to(torch.float32).reshape(*orig_shape[:-1], -1, BLOCK_SIZE)

    # 1: block amax.  2: s0 = Q_e4m3(amax / 6), floored so s1 cannot underflow.
    amax = xb.abs().amax(dim=-1)
    s0_pre = amax * (1.0 / 6.0)
    if scale_policy == "derived_div8":
        s0_pre = torch.clamp(s0_pre, min=S0_FLOOR_DERIVED_DIV8)
    else:
        s0_pre = torch.clamp(s0_pre, min=E4M3_MIN_SUBNORMAL)
    s0 = quantize_e4m3(s0_pre)

    # 3: q0 = Q_e2m1(x / s0), computed in the normalized domain.
    if reciprocal_mode == "exact":
        x_scaled = xb / s0.unsqueeze(-1)
    else:
        x_scaled = xb * torch.reciprocal(s0).to(torch.float32).unsqueeze(-1)
    q0 = quantize_e2m1(x_scaled)

    # 5/6: residual pass.  `derived_div8` stays in the normalized domain, so the
    # residual is `(x_scaled - q0) * 8` and never pays a dequant.
    if scale_policy == "derived_div8":
        s1 = quantize_e4m3(s0 * 0.125)
        q1 = quantize_e2m1((x_scaled - q0) * 8.0)
    else:
        residual = xb - q0 * s0.unsqueeze(-1)
        r_amax = residual.abs().amax(dim=-1)
        s1 = quantize_e4m3(torch.clamp(r_amax * (1.0 / 6.0), min=E4M3_MIN_SUBNORMAL))
        q1 = quantize_e2m1(residual / s1.unsqueeze(-1))

    return DualNVFP4(
        q0=q0.reshape(orig_shape),
        s0=s0,
        q1=q1.reshape(orig_shape),
        s1=s1,
    )


def dequantize_nvfp4(
    q: torch.Tensor, sf: torch.Tensor, global_scale: torch.Tensor | float = 1.0
) -> torch.Tensor:
    """Decode an NVFP4 tensor: ``x = q * sf * global_scale``."""
    blocks = q.shape[:-1] + (q.shape[-1] // BLOCK_SIZE, BLOCK_SIZE)
    out = q.to(torch.float32).reshape(blocks) * sf.to(torch.float32).unsqueeze(-1)
    out = out.reshape(q.shape)
    if isinstance(global_scale, torch.Tensor):
        # One FP32 scale per expert, broadcast over (N, K).
        out = out * global_scale.reshape(-1, *([1] * (out.dim() - 1)))
    else:
        out = out * global_scale
    return out


def quantize_weight_nvfp4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Offline NVFP4 weight quantization: returns (q, block scales, global scale).

    Follows the two-level NVFP4 model ``w = q * S * G`` with
    ``G = amax / (448 * 6)`` per expert and ``S = Q_e4m3((amax_block / 6) / G)``.
    """
    if w.dim() != 3:
        raise ValueError("expected weights shaped (E, N, K)")
    w32 = w.to(torch.float32)
    num_experts = w32.shape[0]

    amax = w32.abs().reshape(num_experts, -1).amax(dim=-1)
    g = torch.clamp(amax / (448.0 * 6.0), min=torch.finfo(torch.float32).tiny)

    wb = w32.reshape(num_experts, w32.shape[1], -1, BLOCK_SIZE)
    block_amax = wb.abs().amax(dim=-1)
    s_pre = (block_amax / 6.0) / g.reshape(-1, 1, 1)
    s = quantize_e4m3(torch.clamp(s_pre, min=E4M3_MIN_SUBNORMAL))

    denom = (s.unsqueeze(-1) * g.reshape(-1, 1, 1, 1)).clamp(min=torch.finfo(torch.float32).tiny)
    q = quantize_e2m1(wb / denom).reshape(w32.shape)
    return q, s, g


def dual_nvfp4_grouped_gemm(
    a_bf16: torch.Tensor,
    w_q: torch.Tensor,
    w_sf: torch.Tensor,
    w_g: torch.Tensor,
    m_indptr: torch.Tensor,
    scale_policy: str = "derived_div8",
    reciprocal_mode: str = "exact",
) -> torch.Tensor:
    """Algorithm 2: ``C = A0 W^T + A1 W^T``, accumulated in FP32, cast to BF16.

    Args:
        a_bf16: activations, shape (M, K).
        w_q/w_sf/w_g: NVFP4 weights from :func:`quantize_weight_nvfp4`,
            shapes (E, N, K), (E, N, K // 16), (E,).
        m_indptr: expert row offsets, shape (E + 1,), so expert ``e`` owns rows
            ``[m_indptr[e], m_indptr[e + 1])``.
    """
    num_experts = w_q.shape[0]
    if m_indptr.numel() != num_experts + 1:
        raise ValueError("m_indptr must have E + 1 entries")

    dual = decompose_dual_nvfp4(a_bf16, scale_policy, reciprocal_mode)
    a_hat = dual.reconstruct()

    out = torch.empty(a_bf16.shape[0], w_q.shape[1], dtype=torch.float32)
    for e in range(num_experts):
        lo, hi = int(m_indptr[e]), int(m_indptr[e + 1])
        if lo == hi:
            continue
        w_deq = dequantize_nvfp4(w_q[e], w_sf[e]) * float(w_g[e])
        # Both passes land in the same FP32 accumulator, exactly as the kernel
        # accumulates A0xW and A1xW into one TMEM tile.
        out[lo:hi] = a_hat[lo:hi] @ w_deq.T
    return out.to(torch.bfloat16)


def reference_grouped_gemm(
    a_bf16: torch.Tensor,
    w_q: torch.Tensor,
    w_sf: torch.Tensor,
    w_g: torch.Tensor,
    m_indptr: torch.Tensor,
) -> torch.Tensor:
    """Ground truth: BF16 activations times dequantized weights, FP32 accumulation.

    Per the doc, the ground truth must use the *unquantized* BF16 activation --
    introducing any FP8/FP4 intermediate on A would understate the real error.
    """
    num_experts = w_q.shape[0]
    a32 = a_bf16.to(torch.float32)
    out = torch.empty(a_bf16.shape[0], w_q.shape[1], dtype=torch.float32)
    for e in range(num_experts):
        lo, hi = int(m_indptr[e]), int(m_indptr[e + 1])
        if lo == hi:
            continue
        w_deq = dequantize_nvfp4(w_q[e], w_sf[e]) * float(w_g[e])
        out[lo:hi] = a32[lo:hi] @ w_deq.T
    return out


def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> float:
    """Flattened cosine similarity, the accuracy metric used throughout the doc."""
    x32, y32 = x.to(torch.float32).flatten(), y.to(torch.float32).flatten()
    return float(torch.dot(x32, y32) / (x32.norm() * y32.norm()))

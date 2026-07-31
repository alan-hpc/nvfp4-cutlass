"""NVFP4 weight quantization and the physical layouts the kernel expects.

Two layout facts drive everything here, both of them consequences of the
`128 x 4` scale-factor atom that `tcgen05.mma ... .block16` consumes:

  * one atom covers 64 K elements (four 16-element blocks) of 128 rows, and
  * the four scales of an atom are packed into a single 32-bit word, sub-block
    ``j`` in byte ``j``.

The kernel's transform warps write SFA in exactly this packing, so SFB has to
match or the two operands would disagree about which scale belongs to which
sub-block.
"""

from __future__ import annotations

import torch

# NVFP4 block size along K.
BLOCK_SIZE = 16
# One 128x4 scale atom spans this many K elements.
K_PER_SF_ATOM = 64

E2M1_MAX = 6.0
E4M3_MAX = 448.0
E4M3_MIN_SUBNORMAL = 2.0**-9

# E2M1 magnitudes, indexed by the low three bits of the code.
_E2M1_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_E2M1_MIDPOINTS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
_E2M1_TIE_CODE = torch.tensor([0, 2, 2, 4, 4, 6, 6], dtype=torch.uint8)


def quantize_e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    """Round to nearest even on the E2M1 grid; returns 4-bit codes as uint8."""
    x = x.to(torch.float32)
    mag = x.abs()
    mids = _E2M1_MIDPOINTS.to(x.device)

    code = torch.zeros_like(mag, dtype=torch.uint8)
    for i in range(mids.numel()):
        code = torch.where(mag > mids[i], torch.full_like(code, i + 1), code)
    for i in range(mids.numel()):
        code = torch.where(mag == mids[i], _E2M1_TIE_CODE.to(x.device)[i], code)
    code = torch.where(~torch.isfinite(mag), torch.full_like(code, 7), code)

    sign = torch.where(torch.signbit(x), torch.full_like(code, 8), torch.zeros_like(code))
    return code | sign


def pack_e2m1_pairs(codes: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit codes two per byte along the last axis.

    Element ``2i`` goes to the low nibble and ``2i + 1`` to the high nibble,
    matching both `cvt.rn.satfinite.e2m1x2.f32` and how UMMA reads the operand.
    """
    if codes.shape[-1] % 2 != 0:
        raise ValueError("last dim must be even to pack FP4 pairs")
    lo = codes[..., 0::2]
    hi = codes[..., 1::2]
    return (lo | (hi << 4)).to(torch.int8)


def _e4m3_codes(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float32).to(torch.float8_e4m3fn).view(torch.uint8)


def quantize_weight_nvfp4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Offline NVFP4 weight quantization.

    Args:
        w: BF16/FP32 weights, shape (E, N, K).

    Returns:
        ``(b, sfb, gw)`` in the exact physical layouts the kernel's TMA
        descriptors describe:

        * ``b``   int8,  (E * N, K // 2)     -- packed E2M1, K-major
        * ``sfb`` int32, (E * K // 64, N)    -- one atom word per (sf_k, n)
        * ``gw``  fp32,  (E,)
    """
    if w.dim() != 3:
        raise ValueError("expected weights shaped (E, N, K)")
    num_experts, n, k = w.shape
    if k % K_PER_SF_ATOM != 0:
        raise ValueError(f"K={k} must be a multiple of {K_PER_SF_ATOM}")

    w32 = w.to(torch.float32)

    # Two-level NVFP4 scaling: G is per expert, S is per 16-element block.
    amax = w32.reshape(num_experts, -1).abs().amax(dim=-1)
    gw = torch.clamp(amax / (E4M3_MAX * E2M1_MAX), min=torch.finfo(torch.float32).tiny)

    blocks = w32.reshape(num_experts, n, k // BLOCK_SIZE, BLOCK_SIZE)
    block_amax = blocks.abs().amax(dim=-1)
    s_pre = (block_amax / E2M1_MAX) / gw.reshape(-1, 1, 1)
    s_codes = _e4m3_codes(torch.clamp(s_pre, min=E4M3_MIN_SUBNORMAL))

    s_dec = s_codes.view(torch.float8_e4m3fn).to(torch.float32)
    denom = torch.clamp(s_dec.unsqueeze(-1) * gw.reshape(-1, 1, 1, 1),
                        min=torch.finfo(torch.float32).tiny)
    codes = quantize_e2m1_codes(blocks / denom).reshape(num_experts, n, k)

    b = pack_e2m1_pairs(codes).reshape(num_experts * n, k // 2).contiguous()
    return b, pack_sf_atoms(s_codes), gw.contiguous()


def pack_sf_atoms(s_codes: torch.Tensor) -> torch.Tensor:
    """(E, N, K // 16) E4M3 codes -> (E * K // 64, N) int32 atom words.

    Sub-block ``j`` of an atom lands in byte ``j`` of the word, which is the same
    packing `transform_a_tile` uses for SFA.  The transposed (sf_k, n) ordering
    is what the TMA box wants: the kernel loads ``BLOCK_K // 64`` atom rows of
    ``BLOCK_N`` words each.
    """
    num_experts, n, num_blocks = s_codes.shape
    if num_blocks % 4 != 0:
        raise ValueError("K // 16 must be a multiple of 4 (one atom = 4 sub-blocks)")
    num_atoms = num_blocks // 4

    words = s_codes.reshape(num_experts, n, num_atoms, 4).to(torch.int32)
    packed = (words[..., 0]
              | (words[..., 1] << 8)
              | (words[..., 2] << 16)
              | (words[..., 3] << 24))
    # (E, N, atoms) -> (E, atoms, N) -> (E * atoms, N)
    return packed.permute(0, 2, 1).reshape(num_experts * num_atoms, n).contiguous()


def dequantize_weight_nvfp4(b: torch.Tensor, sfb: torch.Tensor, gw: torch.Tensor,
                            num_experts: int, n: int, k: int) -> torch.Tensor:
    """Inverse of :func:`quantize_weight_nvfp4`, for building references.

    Returns FP32 weights of shape (E, N, K).
    """
    packed = b.reshape(num_experts, n, k // 2).to(torch.uint8)
    lo = (packed & 0x0F).to(torch.int64)
    hi = (packed >> 4).to(torch.int64)
    codes = torch.stack([lo, hi], dim=-1).reshape(num_experts, n, k)

    levels = _E2M1_LEVELS.to(b.device)
    mag = levels[codes & 0x7]
    values = torch.where((codes & 0x8) != 0, -mag, mag)

    num_atoms = k // K_PER_SF_ATOM
    words = sfb.reshape(num_experts, num_atoms, n).permute(0, 2, 1).to(torch.int64)
    sub = torch.stack([(words >> (8 * j)) & 0xFF for j in range(4)], dim=-1)
    s_codes = sub.reshape(num_experts, n, k // BLOCK_SIZE).to(torch.uint8)
    s_dec = s_codes.view(torch.float8_e4m3fn).to(torch.float32)

    scales = s_dec.repeat_interleave(BLOCK_SIZE, dim=-1)
    return values * scales * gw.reshape(-1, 1, 1).to(torch.float32)


def make_m_indices(group_sizes: list[int] | torch.Tensor, device=None) -> torch.Tensor:
    """Expert index per row, for the m-grouped contiguous layout.

    The kernel reads ``m_indices[m_block_idx * 128]``, so each expert's row range
    must start on a 128-row boundary.
    """
    sizes = torch.as_tensor(group_sizes, dtype=torch.int64).tolist()
    for i, size in enumerate(sizes):
        if size % 128 != 0:
            raise ValueError(f"group {i} has {size} rows; each must be a multiple of 128")
    out = torch.cat([torch.full((size,), e, dtype=torch.int32) for e, size in enumerate(sizes)])
    return out.to(device) if device is not None else out

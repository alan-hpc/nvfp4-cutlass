"""Offline weight quantization for the benchmark baselines.

Only the MXFP4 path lives here: the NVFP4 weight layout is `nvfp4_gemm.layout`'s
job, since it has to match the kernel's own SF atom packing.

MXFP4 differs from NVFP4 in both block size and scale type -- block 32 with UE8M0
(power-of-two) scales rather than block 16 with E4M3. That is why the same GEMM
cannot serve both, and it is the reason DeepGEMM's FP4 kernel (which asserts
`gran_k in {32, 128}`) is an MXFP4 baseline rather than an NVFP4 one.
"""

from __future__ import annotations

import torch

MXFP4_BLOCK = 32


def quantize_weight_mxfp4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(E, N, K) BF16 -> DeepGEMM's MXFP4 weight layout.

    Returns `(packed, scales)` shaped for
    `deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous`, which takes B as
    `(E, N, K // 2)` packed int8 plus packed UE8M0 scales.
    """
    try:
        from deep_gemm.utils.math import per_token_cast_to_fp4
    except ImportError as exc:  # pragma: no cover - depends on the submodule build
        raise RuntimeError(
            f'deep_gemm.utils.math.per_token_cast_to_fp4 unavailable ({exc}); '
            'build DeepGEMM first (3rdparty/DeepGEMM/develop.sh)') from exc

    if w.dim() != 3:
        raise ValueError('expected weights shaped (E, N, K)')
    num_experts, n, k = w.shape

    packed, scales = [], []
    for e in range(num_experts):
        # UE8M0 scales at block 32 is exactly the MX format; `use_packed_ue8m0`
        # gives the int32-packed scale layout the SM100 kernel consumes.
        p, s = per_token_cast_to_fp4(w[e], use_ue8m0=True, gran_k=MXFP4_BLOCK,
                                     use_packed_ue8m0=True)
        packed.append(p)
        scales.append(s)

    # DeepGEMM's grouped-SF check wants each expert's scales MN-major
    # (`stride(-2) == 1`) with expert blocks laid out back to back
    # (`stride(-3) == stride(-1) * size(-1)`). A plain stack is row-major and
    # fails both, so materialize as (E, K-words, N) and hand back the
    # transposed view -- same bytes, MN-major strides.
    sfb = torch.stack([s.transpose(0, 1).contiguous() for s in scales])
    return torch.stack(packed).contiguous(), sfb.transpose(-2, -1)


def activation_global_scale(a: torch.Tensor) -> torch.Tensor:
    """Per-tensor NVFP4 activation scale, `amax / (448 * 6)`.

    Computed offline on purpose. The recipe this kernel targets pins the
    activation `constant_amax` to 2688 = 448 x 6 so that `G_A == 1`; a runtime
    reduction here would be measuring a cost production does not pay.
    """
    return ((448.0 * 6.0) / a.abs().max().float()).reshape(1).contiguous()

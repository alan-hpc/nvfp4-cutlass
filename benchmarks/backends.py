"""Backend adapters for the MoE expert-GEMM forward comparison.

Every backend measures the *complete online chain* a real forward pays, not a
bare GEMM:

    dual_nvfp4     BF16 -> [fused: decompose + 2x block-scaled MMA] -> BF16
    single_nvfp4   BF16 -> [fused: decompose + 1x block-scaled MMA] -> BF16
    mxfp8_mxfp4    BF16 -> [mxfp8_quantize] -> [MXFP8 x MXFP4 grouped GEMM] -> BF16
    fi_nvfp4_moe   BF16 -> [fp4_quantize] -> [FlashInfer NVFP4 fused MoE] -> BF16

Weights are quantized offline and excluded from timing, matching inference.
Activation quantization is *inside* the timed region for every baseline, because
in production the activation arrives as BF16 and someone has to pay to convert
it. Comparing bare GEMMs would hide exactly the cost the fused kernel exists to
remove.

Two things worth knowing before reading the numbers:

  * **DeepGEMM has no NVFP4 path.** Its FP4 kernel asserts `gran_k in {32, 128}`,
    i.e. MXFP4 (block 32, UE8M0 scales) and MXFP8 -- not NVFP4's block 16 with
    E4M3 scales. So DeepGEMM can serve as the MXFP8xMXFP4 baseline but cannot
    serve as an NVFP4 one. That gap is why this repo has its own NVFP4 UMMA.

  * **`single_nvfp4` is the controlled A/B.** It is the same kernel with
    `enable_residual_pass=False`: same tiles, same pipeline, same epilogue, one
    MMA instead of two. It answers "what does the second pass cost, and what
    does it buy" without any cross-vendor confound. Cross-library comparisons
    always mix in scheduler, epilogue and tile-choice differences.

FlashInfer entry points mirror what vLLM actually calls (`vllm/utils/flashinfer.py`):
`flashinfer.fp4_quantize`, `flashinfer.mxfp8_quantize`,
`flashinfer.fused_moe.cutlass_fused_moe`, `flashinfer.trtllm_fp4_block_scale_moe`.
Backends probe for their dependencies and say precisely what is missing rather
than failing obscurely -- FlashInfer's surface moves between releases, and a
version mismatch should be reported, not silently benchmarked around.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import torch


@dataclasses.dataclass(frozen=True)
class Problem:
    """One expert-GEMM shape: (M, K) activations against (E, N, K) weights."""

    num_experts: int
    m_per_expert: int
    n: int
    k: int

    @property
    def m(self) -> int:
        return self.num_experts * self.m_per_expert

    def label(self) -> str:
        return f'E={self.num_experts} M={self.m} N={self.n} K={self.k}'

    def flops(self) -> float:
        """Logical FLOPs of the expert GEMM, counted once regardless of passes."""
        return 2.0 * self.m * self.n * self.k


class Unavailable(Exception):
    """Raised by `setup` when a backend's dependencies are missing."""


class Backend:
    """A measurable forward chain.

    `setup` does all offline work (weight quantization, workspace allocation) and
    returns a zero-argument callable performing exactly the online chain. Only
    that callable is captured into the CUDA graph and timed.
    """

    name = 'base'
    description = ''
    #: True when the chain performs the same work as the reference expert GEMM.
    #: False marks a backend whose scope differs (e.g. a full fused MoE), so the
    #: driver can flag it instead of printing a misleading speedup.
    comparable = True

    def setup(self, problem: Problem, a: torch.Tensor, w: torch.Tensor,
              m_indices: torch.Tensor) -> Callable[[], torch.Tensor]:
        raise NotImplementedError

    def stages(self, problem: Problem, a: torch.Tensor, w: torch.Tensor,
               m_indices: torch.Tensor) -> dict[str, Callable[[], object]]:
        """Optional per-stage breakdown: {stage name -> callable}.

        Stages are measured independently, so they do not sum to the chain total:
        the GPU overlaps work across kernels and each measurement pays its own
        launch. They answer "at most how much would removing this save", which is
        the question worth asking.
        """
        return {}


# --------------------------------------------------------------------------
# Ours
# --------------------------------------------------------------------------

class _FusedBackend(Backend):
    """Shared setup for the fused kernel; subclasses pick the pass count."""

    residual_pass = True

    def setup(self, problem, a, w, m_indices):
        try:
            import nvfp4_gemm
        except ImportError as exc:
            raise Unavailable(f'nvfp4_gemm not importable ({exc}); run ./develop.sh') from exc

        b, sfb, gw = nvfp4_gemm.quantize_weight_nvfp4(w)
        d = torch.empty(problem.m, problem.n, dtype=torch.bfloat16, device=a.device)
        residual = self.residual_pass

        def run():
            # One kernel. The BF16 -> NVFP4 transform happens in the mainloop, so
            # there is no separate quantization launch and A0/A1 never reach
            # global memory.
            nvfp4_gemm.m_grouped_bf16_dual_nvfp4_gemm_contiguous(
                a, b, sfb, gw, d, m_indices, enable_residual_pass=residual)
            return d

        return run


class DualNVFP4Backend(_FusedBackend):
    name = 'dual_nvfp4'
    description = 'ours: BF16 in, dual-NVFP4 decomposition fused into the mainloop'
    residual_pass = True


class SingleNVFP4Backend(_FusedBackend):
    name = 'single_nvfp4'
    description = 'ours, residual pass disabled: single-pass NVFP4 activations'
    residual_pass = False


# --------------------------------------------------------------------------
# MXFP8 activations x MXFP4 weights
# --------------------------------------------------------------------------

class MXFP8MXFP4Backend(Backend):
    """BF16 -> MXFP8 quantize (+ scale layout) -> MXFP8 x MXFP4 grouped GEMM.

    The algorithm doc measures this as the stronger end-to-end baseline, and the
    reason is structural rather than incidental: a single 8-bit activation costs
    one GEMM instead of two, and MXFP8's activation cosine (~0.9996) is high
    enough that the accuracy case for dual-NVFP4 rests on the weight side.

    The scale layout reorder is inside the timed chain. It is small measured in
    isolation, but it is real work the forward pays and omitting it would flatter
    this path.
    """

    name = 'mxfp8_mxfp4'
    description = 'FlashInfer mxfp8_quantize + DeepGEMM MXFP8 x MXFP4 grouped GEMM'

    def __init__(self, swizzled_scales: bool = True):
        # vLLM passes swizzled scales for the CUTLASS path and non-swizzled for
        # TRTLLM; the reorder cost differs, so it stays a flag.
        self.swizzled_scales = swizzled_scales

    def _quantize_fn(self, a):
        try:
            from flashinfer import mxfp8_quantize
        except ImportError as exc:
            raise Unavailable(
                f'flashinfer.mxfp8_quantize unavailable ({exc}); pip install flashinfer-python'
            ) from exc
        swizzled = self.swizzled_scales

        # flashinfer >= 0.6 returns the swizzled SF as a flat uint8 tensor;
        # DeepGEMM asserts int32. The 128x4 swizzle is byte-identical to
        # DeepGEMM's packed UE8M0 MN-major layout -- word w of column-block c
        # holds rows' 4 consecutive k-scales, columns strided by the TMA-aligned
        # row count -- so the adaptation is a zero-copy view, not a reorder.
        # (A wrong guess here would show up immediately as cosine ~0 in the
        # accuracy column; the measured 0.9996 confirms the layout.)
        m, k = a.shape
        m_pad = (m + 127) // 128 * 128

        def quantize():
            aq, sfa = mxfp8_quantize(a, is_sf_swizzled_layout=swizzled)
            if sfa.dtype == torch.uint8 and swizzled:
                sfa = sfa.view(torch.int32).as_strided((m, k // 128), (1, m_pad))
            return aq, sfa

        return quantize

    def setup(self, problem, a, w, m_indices):
        try:
            import deep_gemm
        except ImportError as exc:
            raise Unavailable(
                f'deep_gemm not importable ({exc}); build it in 3rdparty/DeepGEMM') from exc

        from quantize import quantize_weight_mxfp4  # local helper

        quantize = self._quantize_fn(a)
        b, sfb = quantize_weight_mxfp4(w)
        d = torch.empty(problem.m, problem.n, dtype=torch.bfloat16, device=w.device)

        def run():
            aq, sfa = quantize()
            # Both operands carry MX-format 1x32 scale blocks; packed-int SF
            # without an explicit recipe would be read as the FP8 1x128 layout.
            deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
                (aq, sfa), (b, sfb), d, m_indices,
                recipe_a=(1, 32), recipe_b=(1, 32))
            return d

        return run

    def stages(self, problem, a, w, m_indices):
        try:
            quantize = self._quantize_fn(a)
        except Unavailable:
            return {}
        return {'quantize': quantize}


# --------------------------------------------------------------------------
# FlashInfer's own NVFP4 MoE
# --------------------------------------------------------------------------

class FlashInferNVFP4MoEBackend(Backend):
    """BF16 -> fp4_quantize -> FlashInfer NVFP4 fused MoE.

    Scope warning, and it is not a small one: `cutlass_fused_moe` is a *full* MoE
    -- routing, two GEMMs, SwiGLU and the weighted reduce -- while every other
    backend here is one expert GEMM. The wall-clock numbers are therefore not
    directly comparable, and the driver marks this backend accordingly.

    It is still worth measuring: it is the production alternative someone would
    actually deploy, and it bounds what the fused expert GEMM has to beat once
    the surrounding MoE work is added to our side too.
    """

    name = 'fi_nvfp4_moe'
    description = 'FlashInfer fp4_quantize + cutlass_fused_moe (full MoE, different scope)'
    comparable = False

    def setup(self, problem, a, w, m_indices):
        try:
            from flashinfer import fp4_quantize  # noqa: F401
            from flashinfer.fused_moe import cutlass_fused_moe  # noqa: F401
        except ImportError as exc:
            raise Unavailable(
                f'FlashInfer NVFP4 fused MoE unavailable ({exc}); '
                'needs flashinfer with fused_moe.cutlass_fused_moe') from exc

        raise Unavailable(
            'not wired up: cutlass_fused_moe needs a full MoE problem (router logits, '
            'top-k, w13/w2 weight pair, SwiGLU) rather than the single expert GEMM this '
            'harness builds. Wiring it means benchmarking a different computation; do '
            'that in a separate full-MoE harness where both sides run the same work.')


ALL_BACKENDS: dict[str, type[Backend]] = {
    DualNVFP4Backend.name: DualNVFP4Backend,
    SingleNVFP4Backend.name: SingleNVFP4Backend,
    MXFP8MXFP4Backend.name: MXFP8MXFP4Backend,
    FlashInferNVFP4MoEBackend.name: FlashInferNVFP4MoEBackend,
}

DEFAULT_BACKENDS = [DualNVFP4Backend.name, SingleNVFP4Backend.name, MXFP8MXFP4Backend.name]

"""Production end-to-end test: the real API, the real JIT, on a real B200/B300.

This is the acceptance test. It goes through the same path production would:
quantize weights offline into the kernel's physical layouts, hand BF16
activations to the op, and check the output against the ground truth the
algorithm doc defines.

    python tests/test_gemm.py            # correctness
    python tests/test_gemm.py --bench    # correctness, then timing

Accuracy gate: cosine > 0.999 against `A_bf16 @ dequant(W).T` in FP32.

That threshold is deliberately tighter than the 0.9954 the algorithm doc quotes.
A single-pass NVFP4 activation already scores ~0.9954 against this reference, so
gating at 0.995 would accept a kernel whose residual pass contributes nothing.
See docs/implementation-notes.md section 8.1.
"""

from __future__ import annotations

import argparse
import sys

import torch

import nvfp4_gemm
from nvfp4_gemm.layout import dequantize_weight_nvfp4, quantize_weight_nvfp4, make_m_indices

# (num_experts, rows per expert, N, K). M must be a multiple of 128 per expert,
# N and K multiples of 128.
SHAPES = [
    (1, 128, 128, 256),
    (2, 128, 256, 512),
    (4, 256, 1024, 2048),   # the doc's main production shape
    (4, 512, 2048, 512),    # down-projection shape
]


def reference_gemm(a: torch.Tensor, w_deq: torch.Tensor, m_indices: torch.Tensor) -> torch.Tensor:
    """C_ref = A_bf16 @ (dec(W) * S_W * G_W).T, FP32 accumulation.

    A goes in unquantized: per the doc, introducing any FP8/FP4 intermediate on
    the activation here would understate the error the kernel actually makes.
    """
    out = torch.empty(a.shape[0], w_deq.shape[1], dtype=torch.float32, device=a.device)
    for e in range(w_deq.shape[0]):
        rows = m_indices == e
        if rows.any():
            out[rows] = a[rows].to(torch.float32) @ w_deq[e].T
    return out


def cosine(x: torch.Tensor, y: torch.Tensor) -> float:
    x32, y32 = x.to(torch.float32).flatten(), y.to(torch.float32).flatten()
    return float(torch.dot(x32, y32) / (x32.norm() * y32.norm()))


def build_case(num_experts: int, m_per_expert: int, n: int, k: int, device: str, seed: int):
    torch.manual_seed(seed)
    m = num_experts * m_per_expert

    a = torch.randn(m, k, dtype=torch.bfloat16, device=device)
    w = (torch.randn(num_experts, n, k, dtype=torch.bfloat16, device=device) * 0.5)

    b, sfb, gw = quantize_weight_nvfp4(w)
    m_indices = make_m_indices([m_per_expert] * num_experts, device=device)
    d = torch.empty(m, n, dtype=torch.bfloat16, device=device)
    return a, w, b, sfb, gw, d, m_indices


def run_shape(num_experts: int, m_per_expert: int, n: int, k: int,
              device: str, policy: str, seed: int = 0) -> tuple[bool, float]:
    a, w, b, sfb, gw, d, m_indices = build_case(num_experts, m_per_expert, n, k, device, seed)
    m = a.shape[0]

    nvfp4_gemm.m_grouped_bf16_dual_nvfp4_gemm_contiguous(
        a, b, sfb, gw, d, m_indices, scale_policy=policy)
    torch.cuda.synchronize()

    w_deq = dequantize_weight_nvfp4(b, sfb, gw, num_experts, n, k)
    ref = reference_gemm(a, w_deq, m_indices)

    cos = cosine(d, ref)
    rel = float((d.to(torch.float32) - ref).norm() / ref.norm())
    ok = cos > 0.999 and torch.isfinite(d.to(torch.float32)).all().item()

    status = 'ok  ' if ok else 'FAIL'
    print(f'  {status} E={num_experts} M={m} N={n} K={k} [{policy}]: '
          f'cosine {cos:.6f}, rel L2 {rel:.6f}')
    if not ok:
        # A cosine near 0.9954 is the specific signature of a dead residual pass:
        # that is what single-pass NVFP4 activations score.
        if 0.99 < cos < 0.998:
            print('       cosine ~0.995 means the A1 pass is not reaching the accumulator.')
            print('       Check that A1 x W issues with accumulate=true and that SFA1 lands')
            print('       in its own TMEM columns.')
        elif cos < 0.9:
            print('       A large error points at layout, not arithmetic: the A-tile swizzle,')
            print('       the packed-FP4 SMEM assumption, or the SF atom ordering.')
            print('       Run the standalone transform test first (scripts/run.sh transform).')
    return ok, cos


def benchmark(device: str, num_iters: int = 100) -> None:
    print('\nbenchmark')
    for num_experts, m_per_expert, n, k in SHAPES:
        a, w, b, sfb, gw, d, m_indices = build_case(num_experts, m_per_expert, n, k, device, 0)

        for _ in range(10):
            nvfp4_gemm.m_grouped_bf16_dual_nvfp4_gemm_contiguous(a, b, sfb, gw, d, m_indices)
        torch.cuda.synchronize()

        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(num_iters):
            nvfp4_gemm.m_grouped_bf16_dual_nvfp4_gemm_contiguous(a, b, sfb, gw, d, m_indices)
        end.record()
        torch.cuda.synchronize()

        us = start.elapsed_time(end) * 1000 / num_iters
        m = num_experts * m_per_expert
        # Two passes over the same weights, so the useful FLOPs are counted once
        # but the tensor cores do twice this.
        tflops = 2.0 * m * n * k / (us * 1e-6) / 1e12
        print(f'  E={num_experts} M={m} N={n} K={k}: {us:8.2f} us, {tflops:7.1f} TFLOPS (logical)')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bench', action='store_true', help='also run timing')
    parser.add_argument('--policy', default='both',
                        choices=['derived_div8', 'residual_amax', 'both'])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print('no CUDA device available', file=sys.stderr)
        return 1

    device = 'cuda'
    major, minor = torch.cuda.get_device_capability()
    print(f'device: {torch.cuda.get_device_name()} (sm_{major}{minor}), '
          f'{nvfp4_gemm.get_num_sms()} SMs')
    if major != 10:
        print(f'error: this kernel needs SM100/SM103 (Blackwell), found sm_{major}{minor}',
              file=sys.stderr)
        return 1

    policies = ['derived_div8', 'residual_amax'] if args.policy == 'both' else [args.policy]

    print('\ncorrectness')
    all_ok = True
    for policy in policies:
        for shape in SHAPES:
            ok, _ = run_shape(*shape, device=device, policy=policy)
            all_ok &= ok

    if args.bench and all_ok:
        benchmark(device)

    print('\nPASS' if all_ok else '\nFAIL')
    return 0 if all_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())

"""Routing-aware benchmark: sampled top-8 dispatch, decode and prefill.

    python benchmarks/bench_decode.py
    python benchmarks/bench_decode.py --backends dual_nvfp4,mxfp8_mxfp4

`tokens` are input tokens. Each one fans out to `topk = 8` of the 256 experts,
and at EP=8 this GPU executes the assignments that land on its 32 local
experts -- in expectation exactly `tokens` rows. The grouped GEMM runs them in
ONE kernel; nothing is padded by hand.

What cannot be avoided is tile granularity: a 128-row M tile serves exactly one
expert, so any grouped GEMM -- contiguous, masked, or a CUTLASS group GEMM at
the same tile -- executes ceil(rows_e / 128) tiles per active expert. The
`grid rows` column is that sum; `gran x` is grid rows over real assignments.
A masked layout (device-side counts, no host-side buffer alignment; this
kernel's device code already carries `GemmType::MGroupedMasked`) changes the
bookkeeping, not the tile count.

Prefill uses the same sampled routing instead of assuming perfectly balanced
experts: with counts ~ tokens/32 +- noise, the per-expert ceil costs a real
5-25% extra tiles over the idealized aligned sweep in bench_moe_forward.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))

from backends import ALL_BACKENDS, Unavailable  # noqa: E402
from bench_moe_forward import cosine, graph_time_us, reference_output  # noqa: E402

from nvfp4_gemm.layout import make_m_indices  # noqa: E402

# Qwen3.5-35B-A3B: 256 experts, top-8; EP=8 leaves 32 on this GPU.
TOTAL_EXPERTS = 256
TOPK = 8
LOCAL_EXPERTS = 32
DECODE_TOKENS = [16, 32, 128]
PREFILL_TOKENS = [2048, 8192, 16384]
SHAPE_FAMILIES = {
    'gate_up': (1024, 2048),
    'down': (2048, 512),
}
TILE_M = 128


class RoutedProblem:
    """Duck-typed stand-in for `backends.Problem` with ragged expert counts."""

    def __init__(self, counts: list[int], n: int, k: int):
        self.counts = counts
        self.num_experts = len(counts)
        self.n = n
        self.k = k
        self.m = sum(counts)
        self.m_per_expert = 0   # not meaningful under sampled routing

    def label(self) -> str:
        return f'E={self.num_experts} M={self.m} N={self.n} K={self.k}'


def route_tokens(tokens: int, seed: int = 0, align: int = TILE_M) -> tuple[int, list[int]]:
    """Sample top-8 routing; return (local assignments, per-expert grid rows).

    Grid rows per expert are ceil(count / TILE_M) * TILE_M -- the rows the
    kernel's 128-row tiles actually execute for that expert.
    """
    g = torch.Generator().manual_seed(seed)
    scores = torch.rand(tokens, TOTAL_EXPERTS, generator=g)
    picks = scores.topk(TOPK, dim=1).indices
    local = picks[picks < LOCAL_EXPERTS]
    counts = torch.bincount(local, minlength=LOCAL_EXPERTS)
    grid = [(int(c) + align - 1) // align * align for c in counts]
    return int(local.numel()), grid


def run_phase(phase: str, token_list: list[int], names: list[str], iters: int, seed: int) -> None:
    device = 'cuda'
    for family, (n, k) in SHAPE_FAMILIES.items():
        print('=' * 92)
        print(f'{phase} · {family}  (N={n}, K={k})')
        print('=' * 92)
        header = f'{"tokens":>7} | {"assign":>6} | {"grid rows":>9} | {"gran x":>6} |'
        for name in names:
            header += f' {name:>14} |'
        print(header + f' {"dual_swap bt32":>14} |')

        rows_acc = []
        for tokens in token_list:
            assignments, grid = route_tokens(tokens, seed=seed)
            problem = RoutedProblem(grid, n, k)

            torch.manual_seed(seed)
            a = torch.randn(problem.m, k, dtype=torch.bfloat16, device=device)
            w = torch.randn(LOCAL_EXPERTS, n, k, dtype=torch.bfloat16, device=device) * 0.5
            m_indices = make_m_indices(grid, device=device)
            ref = reference_output(problem, a, w, m_indices)

            line = (f'{tokens:>7} | {assignments:>6} | {problem.m:>9} | '
                    f'{problem.m / max(1, assignments):>5.2f}x |')
            accs = {}

            # Swap-AB runs on its own 32-aligned grid (its token tiles are 32
            # columns, not 128 rows), so its inputs are rebuilt at that
            # alignment rather than shared with the 128-grid backends.
            _, grid32 = route_tokens(tokens, seed=seed, align=32)
            sw_problem = RoutedProblem(grid32, n, k)
            torch.manual_seed(seed)
            sw_a = torch.randn(sw_problem.m, k, dtype=torch.bfloat16, device=device)
            sw_mi = make_m_indices(grid32, device=device, alignment=32)

            for name in names:
                backend = ALL_BACKENDS[name]()
                try:
                    run = backend.setup(problem, a, w, m_indices)
                except Unavailable as exc:
                    line += f' {"n/a":>14} |'
                    print(f'    [{name} unavailable: {exc}]')
                    continue
                out = run()
                torch.cuda.synchronize()
                accs[name] = cosine(out, ref)
                us = graph_time_us(run, num_iters=iters)
                line += f' {us:>11.2f}us |'

            # dual_nvfp4 + swap-AB (bt=32) on the 32-aligned grid.
            import nvfp4_gemm as _ng
            _b, _sfb, _gw = _ng.quantize_weight_nvfp4(w)
            sw_d = torch.empty(sw_problem.m, n, dtype=torch.bfloat16, device=device)
            def sw_run():
                _ng.m_grouped_bf16_dual_nvfp4_gemm_contiguous(
                    sw_a, _b, _sfb, _gw, sw_d, sw_mi, tune_swap_ab=1, tune_block_n=32)
                return sw_d
            sw_out = sw_run()
            torch.cuda.synchronize()
            sw_ref = reference_output(sw_problem, sw_a, w, sw_mi)
            accs['dual_swap'] = cosine(sw_out, sw_ref)
            sw_us = graph_time_us(sw_run, num_iters=iters)
            line += f' {sw_us:>11.2f}us*|'
            print(line)
            rows_acc.append((tokens, accs))

        print('\naccuracy (cosine vs FP32 ground truth over the executed grid)')
        for tokens, accs in rows_acc:
            cells = ' | '.join(f'{name} {c:>9.6f}' for name, c in accs.items())
            print(f'{tokens:>7} | {cells}')
        print()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--backends', default='dual_nvfp4,single_nvfp4,mxfp8_mxfp4')
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--phase', default='decode,prefill')
    args = parser.parse_args()
    names = args.backends.split(',')

    print(f'device: {torch.cuda.get_device_name(0)}')
    print(f'model: Qwen3.5-35B-A3B, EP=8 ({LOCAL_EXPERTS} local experts), top-{TOPK}, sampled routing')
    print(f'assign = token assignments landing on this GPU (~= tokens); grid rows = '
          f'sum of ceil(rows_e/{TILE_M})*{TILE_M} -- the 128-row tiles any grouped GEMM executes\n')

    if 'decode' in args.phase:
        run_phase('decode', DECODE_TOKENS, names, args.iters, args.seed)
    if 'prefill' in args.phase:
        run_phase('prefill', PREFILL_TOKENS, names, args.iters, args.seed)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

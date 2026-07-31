"""Forward-chain benchmark: fused dual-NVFP4 against the production alternatives.

    python benchmarks/bench_moe_forward.py
    python benchmarks/bench_moe_forward.py --backends dual_nvfp4,mxfp8_mxfp4
    python benchmarks/bench_moe_forward.py --breakdown --shapes gate_up

What is measured, and why it is measured that way:

  * **The whole online chain, not the GEMM.** A real forward hands the expert
    GEMM a BF16 activation. Every quantized baseline must therefore convert it
    first, and that conversion is inside the timed region. Timing bare GEMMs
    would hide precisely the cost the fused kernel exists to remove.

  * **CUDA Graph replay, not eager launches.** The algorithm doc observes API
    event times going bimodal between ~0.060 and ~0.123 ms on the same kernel,
    with the difference being host dispatch gap. A multi-kernel baseline pays
    that gap more often than a single-kernel one, so eager timing would
    systematically flatter the fused path. Graph capture removes it from both.

  * **Accuracy alongside speed.** A cheaper activation format that loses
    precision is not a free win, so every backend reports cosine against the
    same FP32 ground truth. Read the two columns together.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backends import ALL_BACKENDS, DEFAULT_BACKENDS, Problem, Unavailable  # noqa: E402

# Shape families from the algorithm doc's sweep, plus the MoE projections they
# correspond to. M per expert is swept; N and K are fixed per family.
SHAPE_FAMILIES = {
    'gate_up': (1024, 2048),
    'down': (2048, 512),
}
M_PER_EXPERT = [128, 256, 512, 1024, 2048]
NUM_EXPERTS = 4


def build_problem_inputs(problem: Problem, device: str, seed: int = 0):
    torch.manual_seed(seed)
    a = torch.randn(problem.m, problem.k, dtype=torch.bfloat16, device=device)
    w = torch.randn(problem.num_experts, problem.n, problem.k,
                    dtype=torch.bfloat16, device=device) * 0.5

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from nvfp4_gemm.layout import make_m_indices
    m_indices = make_m_indices([problem.m_per_expert] * problem.num_experts, device=device)
    return a, w, m_indices


def reference_output(problem: Problem, a, w, m_indices) -> torch.Tensor:
    """FP32 ground truth against the *unquantized* BF16 activation and weights.

    Deliberately not the dequantized-weight reference used by the correctness
    test: here different backends quantize the weights differently (NVFP4 vs
    MXFP4), so the only common baseline is full precision on both sides.
    """
    out = torch.empty(problem.m, problem.n, dtype=torch.float32, device=a.device)
    for e in range(problem.num_experts):
        rows = m_indices == e
        out[rows] = a[rows].to(torch.float32) @ w[e].to(torch.float32).T
    return out


def cosine(x: torch.Tensor, y: torch.Tensor) -> float:
    x32, y32 = x.to(torch.float32).flatten(), y.to(torch.float32).flatten()
    return float(torch.dot(x32, y32) / (x32.norm() * y32.norm()))


def graph_time_us(fn, num_iters: int = 100, warmup: int = 10) -> float:
    """Capture `fn` into a CUDA graph and time steady-state replay.

    Falls back to eager timing if capture fails (some backends allocate or
    synchronize internally), and says so, because eager numbers include host
    dispatch gaps that graph numbers do not -- mixing them silently would be
    worse than not measuring.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    try:
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(graph):
            fn()
        replay = graph.replay
    except Exception as exc:  # noqa: BLE001 - report and degrade, do not hide
        print(f'      (graph capture failed, timing eagerly: {exc})')
        replay = fn

    for _ in range(warmup):
        replay()
    torch.cuda.synchronize()

    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(num_iters):
        replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / num_iters


def run_family(family: str, backend_names: list[str], device: str,
               num_iters: int, breakdown: bool) -> None:
    n, k = SHAPE_FAMILIES[family]
    print(f'\n{"=" * 78}')
    print(f'{family}  (N={n}, K={k}, E={NUM_EXPERTS})')
    print('=' * 78)

    header = f'{"M":>7} | ' + ' | '.join(f'{name:>22}' for name in backend_names)
    print(header)
    print('-' * len(header))

    baseline_name = backend_names[0]
    accuracy: dict[tuple[int, str], float] = {}

    for m_per_expert in M_PER_EXPERT:
        problem = Problem(NUM_EXPERTS, m_per_expert, n, k)
        a, w, m_indices = build_problem_inputs(problem, device)
        ref = reference_output(problem, a, w, m_indices)

        cells, times = [], {}
        for name in backend_names:
            backend = ALL_BACKENDS[name]()
            try:
                run = backend.setup(problem, a, w, m_indices)
            except Unavailable as exc:
                cells.append(f'{"n/a":>22}')
                if m_per_expert == M_PER_EXPERT[0]:
                    print(f'  [{name}] unavailable: {exc}')
                continue

            out = run()
            torch.cuda.synchronize()
            accuracy[(m_per_expert, name)] = cosine(out, ref)

            us = graph_time_us(run, num_iters)
            times[name] = us

            tflops = problem.flops() / (us * 1e-6) / 1e12
            cell = f'{us:8.2f}us {tflops:6.1f}T'
            if name != baseline_name and baseline_name in times:
                cell += f' {times[baseline_name] / us:4.2f}x'
            cells.append(f'{cell:>22}')

        print(f'{problem.m:>7} | ' + ' | '.join(cells))

    print(f'\naccuracy (cosine vs FP32 A_bf16 @ W_bf16^T)')
    print(f'{"M":>7} | ' + ' | '.join(f'{name:>22}' for name in backend_names))
    for m_per_expert in M_PER_EXPERT:
        cells = []
        for name in backend_names:
            cos = accuracy.get((m_per_expert, name))
            cells.append(f'{cos:>22.6f}' if cos is not None else f'{"n/a":>22}')
        print(f'{NUM_EXPERTS * m_per_expert:>7} | ' + ' | '.join(cells))

    if breakdown:
        print('\nper-stage (measured independently; stages overlap, so they do not sum)')
        problem = Problem(NUM_EXPERTS, M_PER_EXPERT[-1], n, k)
        a, w, m_indices = build_problem_inputs(problem, device)
        for name in backend_names:
            backend = ALL_BACKENDS[name]()
            stages = backend.stages(problem, a, w, m_indices)
            if not stages:
                continue
            print(f'  {name} at M={problem.m}:')
            for stage_name, stage_fn in stages.items():
                print(f'    {stage_name:>12}: {graph_time_us(stage_fn, num_iters):8.2f} us')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--backends', default=','.join(DEFAULT_BACKENDS),
                        help=f'comma-separated; available: {",".join(ALL_BACKENDS)}')
    parser.add_argument('--shapes', default='gate_up,down',
                        help=f'comma-separated; available: {",".join(SHAPE_FAMILIES)}')
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--breakdown', action='store_true',
                        help='also measure each backend\'s stages separately')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print('no CUDA device available', file=sys.stderr)
        return 1

    major, minor = torch.cuda.get_device_capability()
    print(f'device: {torch.cuda.get_device_name()} (sm_{major}{minor})')
    if major != 10:
        print(f'error: needs SM100/SM103 (Blackwell), found sm_{major}{minor}', file=sys.stderr)
        return 1

    backend_names = [b.strip() for b in args.backends.split(',') if b.strip()]
    for name in backend_names:
        if name not in ALL_BACKENDS:
            print(f'unknown backend {name!r}; available: {", ".join(ALL_BACKENDS)}',
                  file=sys.stderr)
            return 2

    print('\nbackends:')
    for name in backend_names:
        backend = ALL_BACKENDS[name]()
        note = '' if backend.comparable else '   [different scope -- see backends.py]'
        print(f'  {name:>14}  {backend.description}{note}')
    print('\nTFLOPS is the logical expert-GEMM rate (2*M*N*K), counted once regardless')
    print('of how many passes a backend makes over the weights.')

    for family in (s.strip() for s in args.shapes.split(',') if s.strip()):
        if family not in SHAPE_FAMILIES:
            print(f'unknown shape family {family!r}', file=sys.stderr)
            return 2
        run_family(family, backend_names, 'cuda', args.iters, args.breakdown)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

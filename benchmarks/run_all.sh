#!/bin/bash
# Full benchmark: every backend x phase x shape family, speed + cosine.
#
#   ./benchmarks/run_all.sh                 # GPU 2, 100 iters (defaults)
#   GPU=0 ITERS=300 ./benchmarks/run_all.sh
#
# Covers, in one run:
#   1. bench_decode.py    -- routed top-8 dispatch: decode 16/32/128 +
#      prefill 2k/8k/16k, gate_up + down, backends dual_nvfp4 /
#      single_nvfp4 / mxfp8_mxfp4 plus the dual_swap bt32 column,
#      per-backend cosine.  (The canonical table.)
#   2. bench_moe_forward.py -- idealized aligned sweep, same backends.
#
# All timings are CUDA-graph full-call medians; mxfp8_mxfp4 input is
# pre-quantized (its bf16->fp8 activation cast is NOT counted, ours is
# fused in-kernel and always counted).
set -e
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${GPU:-2}"
export PYTHONPATH="$PWD/python:$PWD/3rdparty/DeepGEMM${PYTHONPATH:+:$PYTHONPATH}"
ITERS="${ITERS:-100}"
LOG="bench_all_$(date +%m%d_%H%M).log"

{
  echo "== nvfp4-cutlass full bench | commit $(git rev-parse --short HEAD 2>/dev/null || echo n/a) | GPU=$CUDA_VISIBLE_DEVICES | iters=$ITERS =="
  nvidia-smi --query-gpu=name,clocks.sm --format=csv,noheader 2>/dev/null | sed -n "$((${CUDA_VISIBLE_DEVICES%%,*} + 1))p"
  echo
  echo "---------- 1/2 routed decode+prefill (bench_decode.py) ----------"
  python3 benchmarks/bench_decode.py --iters "$ITERS"
  echo
  echo "---------- 2/2 aligned sweep (bench_moe_forward.py) ----------"
  python3 benchmarks/bench_moe_forward.py --iters "$ITERS"
} 2>&1 | tee "$LOG"

echo
echo "results saved to: $LOG"

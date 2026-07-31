#!/usr/bin/env bash
#
# Run the standalone bring-up test.
#
#   ./scripts/run.sh
#
# This covers only the transform stage: the A-tile swizzle replication and
# Algorithm 1, with no MMA, no UTCCP and no epilogue. It is the first thing to
# run on new hardware, because everything downstream depends on it and a swizzle
# mistake is silent -- wrong numbers, no fault.
#
# End-to-end correctness is the production path's job:
#
#   ./develop.sh
#   PYTHONPATH=$PWD/python python tests/python/test_gemm.py [--bench]
#
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build"

say()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

if command -v nvidia-smi >/dev/null; then
    say "GPU: $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1)"
else
    warn "nvidia-smi not found; is this a GPU host?"
fi

[[ -x "$BUILD/test_transform" ]] || die "$BUILD/test_transform not built; run ./scripts/build.sh first"

echo
say "Stage 1: A-tile swizzle + Algorithm 1"
printf '%.0s-' {1..72}; echo

if "$BUILD/test_transform"; then
    echo
    say "transform stage OK"
    say "next: ./develop.sh && PYTHONPATH=\$PWD/python python tests/python/test_gemm.py"
    exit 0
else
    rc=$?
    echo
    warn "transform test failed (exit $rc)"
    warn "fix this before looking at the full GEMM -- the transform feeds it,"
    warn "so an end-to-end result would not tell you anything new"
    exit $rc
fi

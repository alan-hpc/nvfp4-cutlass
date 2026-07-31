#!/usr/bin/env bash
#
# Run the standalone bring-up test.
#
#   ./scripts/run.sh
#
# Two stages, each isolating one layer:
#
#   0  swizzled_byte_offset() against TMA itself. TMA applies the hardware
#      swizzle by definition, so a round trip through it is an oracle the
#      transform test cannot provide -- that one reads back through the same
#      helper it wrote with, so a systematic error cancels.
#   1  the transform: A-tile reads and Algorithm 1, no MMA, no UTCCP, no epilogue.
#
# Run these before anything end-to-end: a swizzle mistake is silent, and shows up
# downstream only as a GEMM whose output has the right magnitude and no
# correlation at all.
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

for name in test_swizzle test_transform; do
    [[ -x "$BUILD/$name" ]] || die "$BUILD/$name not built; run ./scripts/build.sh first"
done

echo
say "Stage 0: shared-memory swizzle vs TMA"
printf '%.0s-' {1..72}; echo
if ! "$BUILD/test_swizzle"; then
    rc=$?
    echo
    warn "swizzle test failed (exit $rc)"
    warn "everything downstream reads shared memory through this mapping, so"
    warn "nothing else will tell you anything until it is right"
    exit $rc
fi

echo
say "Stage 1: A-tile reads + Algorithm 1"
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

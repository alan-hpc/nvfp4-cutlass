#!/usr/bin/env bash
#
# Build the dual-NVFP4 kernel and its standalone hardware tests.
#
# The tests are deliberately free of PyTorch and of DeepGEMM's JIT layer: they
# build TMA descriptors with the driver API directly, so bring-up on a B300 only
# needs nvcc and a GPU.
#
#   ./scripts/build.sh                 # compile-check + build both tests
#   ./scripts/build.sh --check-only    # just instantiate the kernel, no binaries
#   ./scripts/build.sh --arch sm_103a  # override arch detection
#   ./scripts/build.sh --ptx --sass    # also dump PTX / SASS for inspection
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build"

ARCH=""
CHECK_ONLY=0
WANT_PTX=0
WANT_SASS=0
JOBS="$(nproc 2>/dev/null || echo 4)"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arch)       ARCH="$2"; shift 2 ;;
        --check-only) CHECK_ONLY=1; shift ;;
        --ptx)        WANT_PTX=1; shift ;;
        --sass)       WANT_SASS=1; shift ;;
        -j)           JOBS="$2"; shift 2 ;;
        -h|--help)    sed -n '2,14p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

say()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- toolchain --
command -v nvcc >/dev/null || die "nvcc not found; put CUDA's bin/ on PATH"

NVCC_VER="$(nvcc --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')"
NVCC_MAJOR="${NVCC_VER%%.*}"
NVCC_MINOR="${NVCC_VER##*.}"
say "nvcc $NVCC_VER"

# The NVFP4 UMMA spelling and the sm_100f family target both need 12.9.  Below
# that, nvcc can only emit sm_100a, which will not load on a B300 (10.3).
if (( NVCC_MAJOR < 12 || (NVCC_MAJOR == 12 && NVCC_MINOR < 9) )); then
    die "CUDA >= 12.9 required (found $NVCC_VER):
       - 'kind::mxf4nvf4.block_scale.block16' needs 12.9+
       - the sm_100f family target needs 12.9+; sm_100a will not run on B300"
fi

# ---------------------------------------------------------------- submodules --
CUTLASS_INC="$ROOT/3rdparty/cutlass/include"
DEEPGEMM_INC="$ROOT/3rdparty/DeepGEMM/deep_gemm/include"

if [[ ! -f "$CUTLASS_INC/cute/tensor.hpp" ]]; then
    die "CUTLASS headers missing at $CUTLASS_INC
       run: git submodule update --init --depth 1 3rdparty/cutlass
       (behind a flaky proxy, retry a few times; a half-finished clone leaves
        3rdparty/cutlass empty but .git/modules/3rdparty/cutlass populated)"
fi
[[ -f "$DEEPGEMM_INC/deep_gemm/common/math.cuh" ]] \
    || die "DeepGEMM headers missing at $DEEPGEMM_INC
       run: git submodule update --init --depth 1 3rdparty/DeepGEMM"

# --------------------------------------------------------------------- arch --
if [[ -z "$ARCH" ]]; then
    if command -v nvidia-smi >/dev/null; then
        CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')"
    else
        CC=""
    fi
    case "$CC" in
        100|103) ARCH="sm_100f" ;;   # B200 and B300 share the sm_100 family
        "")      ARCH="sm_100f"; warn "no GPU detected, defaulting to $ARCH" ;;
        *)       die "compute capability $CC is not SM100/SM103; this kernel needs Blackwell" ;;
    esac
    [[ -n "$CC" ]] && say "detected compute capability ${CC:0:2}.${CC:2}"
fi
say "target architecture: $ARCH"

# ------------------------------------------------------------------- compile --
mkdir -p "$BUILD"

NVCC_FLAGS=(
    -std=c++20
    --gpu-architecture="$ARCH"
    -O3
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -I"$CUTLASS_INC"
    -I"$DEEPGEMM_INC"
    -I"$ROOT/include"
    --diag-suppress=39,161,174,177,186,940
    -Xcompiler -Wno-deprecated-declarations,-Wno-abi
    -lcuda
)
(( WANT_SASS )) && NVCC_FLAGS+=(-lineinfo)

failed=0

compile_check() {
    say "compile-check: instantiating the kernel template"
    if nvcc "${NVCC_FLAGS[@]}" -c "$ROOT/tests/standalone/instantiate.cu" \
            -o "$BUILD/instantiate.o" 2>&1 | tee "$BUILD/instantiate.log"; then
        say "  kernel instantiates cleanly"
    else
        warn "  instantiation FAILED -- see $BUILD/instantiate.log"
        failed=1
    fi
}

build_test() {
    local name="$1"
    say "building test: $name"
    if nvcc "${NVCC_FLAGS[@]}" "$ROOT/tests/standalone/$name.cu" -o "$BUILD/$name" \
            2>&1 | tee "$BUILD/$name.log"; then
        say "  -> $BUILD/$name"
    else
        warn "  $name FAILED to build -- see $BUILD/$name.log"
        failed=1
    fi
}

compile_check

if (( ! CHECK_ONLY )); then
    # Only the transform test lives here.  End-to-end GEMM correctness goes
    # through the production path instead (./develop.sh && python tests/test_gemm.py),
    # so there is no second full GEMM harness to keep in sync.
    build_test test_transform
fi

# ------------------------------------------------------------------- dumps ---
if (( WANT_PTX )); then
    say "dumping PTX"
    nvcc "${NVCC_FLAGS[@]}" -ptx "$ROOT/tests/standalone/instantiate.cu" \
         -o "$BUILD/kernel.ptx" && say "  -> $BUILD/kernel.ptx"
    # The single most useful grep during bring-up: did the NVFP4 MMA survive?
    if grep -q "mxf4nvf4" "$BUILD/kernel.ptx" 2>/dev/null; then
        say "  found tcgen05.mma ... kind::mxf4nvf4 in PTX"
    else
        warn "  no 'mxf4nvf4' in PTX -- the NVFP4 MMA was not emitted"
    fi
fi

if (( WANT_SASS )); then
    say "dumping SASS"
    nvcc "${NVCC_FLAGS[@]}" -cubin "$ROOT/tests/standalone/instantiate.cu" \
         -o "$BUILD/kernel.cubin"
    cuobjdump -sass "$BUILD/kernel.cubin" > "$BUILD/kernel.sass" \
        && say "  -> $BUILD/kernel.sass"
fi

if (( failed )); then
    die "build finished with errors"
fi
say "build OK. Run: ./scripts/run.sh"

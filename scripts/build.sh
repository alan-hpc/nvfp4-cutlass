#!/usr/bin/env bash
#
# Build the dual-NVFP4 kernel and its standalone hardware tests.
#
# The standalone test is deliberately free of PyTorch and of the JIT: it builds
# TMA descriptors with the driver API directly, so bring-up on a B300 needs only
# nvcc and a GPU.
#
#   ./scripts/build.sh                 # compile-check + build both tests
#   ./scripts/build.sh --check-only    # just instantiate the kernel, no binaries
#   ./scripts/build.sh --arch "--gpu-architecture=sm_103a"   # skip the probe
#   ./scripts/build.sh --ptx --sass    # also dump PTX / SASS for inspection
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build"

ARCH_FLAGS=""
CHECK_ONLY=0
WANT_PTX=0
WANT_SASS=0
JOBS="$(nproc 2>/dev/null || echo 4)"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arch)       ARCH_FLAGS="$2"; shift 2 ;;
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
# CUTLASS is the only third-party dependency of the kernel itself.
CUTLASS_INC="$ROOT/3rdparty/cutlass/include"

if [[ ! -f "$CUTLASS_INC/cute/tensor.hpp" ]]; then
    die "CUTLASS headers missing at $CUTLASS_INC
       run: git submodule update --init --depth 1 3rdparty/cutlass
       (behind a flaky proxy, retry a few times; a half-finished clone leaves
        3rdparty/cutlass empty but .git/modules/3rdparty/cutlass populated)"
fi

# --------------------------------------------------------------------- arch --
# The tcgen05 NVFP4 MMA only exists on architecture-specific ("a") or
# architecture-family ("f") targets; a plain `sm_100` has none of it.
#
# `--gpu-architecture=sm_100f` is not enough on its own. In whole-compilation
# mode it expands to `code=[compute_100, sm_100f]`, and to embed that
# forward-compatible PTX nvcc emits `.target sm_100` -- ptxas then honours the
# directive inside the file and rejects every suffixed instruction. Naming the
# virtual architecture explicitly (`arch=compute_100f,code=sm_100f`) drops the
# PTX-embedding half and keeps the suffix on both stages.
#
# The probe therefore has to compile the same way the real build does: `-cubin`
# skips PTX embedding entirely and would pass a flag that then fails under `-c`.
probe_arch() {
    local probe_src="$BUILD/arch_probe.cu"
    mkdir -p "$BUILD"
    cat > "$probe_src" <<'PROBE'
__global__ void probe(unsigned long long a, unsigned long long b, unsigned c, unsigned d)
{
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %3, 0;\n\t"
        "tcgen05.fence::before_thread_sync;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%2], %0, %1, %3, [%2], [%2], p;\n\t"
        "}\n" ::"l"(a), "l"(b), "r"(c), "r"(d));
}
PROBE

    local candidate rc=1
    for candidate in "$@"; do
        if nvcc $candidate -c -o "$BUILD/arch_probe.o" "$probe_src" >/dev/null 2>&1; then
            printf '%s' "$candidate"
            rc=0
            break
        fi
    done
    rm -f "$BUILD/arch_probe.o"
    return $rc
}

if [[ -z "$ARCH_FLAGS" ]]; then
    say "probing which architecture flag assembles the NVFP4 MMA"
    # Explicit virtual+real pairs first: those keep the suffix on the emitted
    # PTX. Family before architecture-specific, so one cubin still covers B200
    # and B300 when the toolchain allows it; both 100f and 103f are tried because
    # which capability roots the Blackwell family has moved between releases.
    ARCH_FLAGS="$(probe_arch \
        "-gencode=arch=compute_100f,code=sm_100f" \
        "-gencode=arch=compute_103f,code=sm_103f" \
        "-gencode=arch=compute_103a,code=sm_103a" \
        "-gencode=arch=compute_100a,code=sm_100a" \
        "--gpu-architecture=sm_100f" \
        "--gpu-architecture=sm_103a" \
        "--gpu-architecture=sm_100a" || true)"
    if [[ -z "$ARCH_FLAGS" ]]; then
        die "no architecture flag assembled 'tcgen05.mma ... kind::mxf4nvf4.block16'.
       Tried compute_100f/103f/103a/100a as explicit -gencode pairs and as
       --gpu-architecture. Needs CUDA >= 12.9 and a Blackwell target.
       Reproduce by hand:  nvcc -gencode=arch=compute_103a,code=sm_103a -c -o /tmp/p.o $BUILD/arch_probe.cu
       or pass the working flags with --arch '<flags>'."
    fi
    say "using: $ARCH_FLAGS"
    case "$ARCH_FLAGS" in
        *100f*) ;;
        *) warn "not a family target -- the cubin will run only on this exact architecture" ;;
    esac
fi

# ------------------------------------------------------------------- compile --
mkdir -p "$BUILD"

NVCC_FLAGS=(
    -std=c++20
    $ARCH_FLAGS
    -O3
    --expt-relaxed-constexpr
    --expt-extended-lambda
    -I"$CUTLASS_INC"
    -I"$ROOT/include"
    --diag-suppress=39,161,174,177,186,550,940
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
    # End-to-end GEMM correctness goes through the production path instead
    # (./develop.sh, then tests/python/test_gemm.py), so there is no second GEMM
    # harness to maintain here -- only the two stages that isolate a layer.
    build_test test_swizzle
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

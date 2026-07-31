#!/usr/bin/env bash
#
# In-place development build.
#
# Builds the C++ extension and links it, plus the JIT include root, back into
# the source tree so `import nvfp4_gemm` works from the repo without installing.
#
set -euo pipefail

original_dir="$(pwd)"
script_dir="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
cd "$script_dir"

if [[ ! -d 3rdparty/cutlass/include/cutlass ]]; then
    echo "error: missing 3rdparty/cutlass/include/cutlass" >&2
    echo "run: git submodule update --init --depth 1 3rdparty/cutlass" >&2
    exit 1
fi

# Generated kernels are compiled with a single -I, so every header namespace they
# can reference has to live under one root. nvfp4_gemm's device headers depend on
# nothing but CUTLASS.
echo "==> linking JIT include root"
mkdir -p python/nvfp4_gemm/include
ln -sfn "$script_dir/include/nvfp4_gemm"               python/nvfp4_gemm/include/nvfp4_gemm
ln -sfn "$script_dir/3rdparty/cutlass/include/cutlass" python/nvfp4_gemm/include/cutlass
ln -sfn "$script_dir/3rdparty/cutlass/include/cute"    python/nvfp4_gemm/include/cute

echo "==> building the C++ extension"
rm -rf build dist ./*.egg-info
python setup.py build

so_file="$(find build -name '_C*.so' -type f | head -n 1)"
if [[ -z "$so_file" ]]; then
    echo "error: no extension .so produced under build/" >&2
    exit 1
fi
ln -sf "$script_dir/$so_file" python/nvfp4_gemm/
echo "==> linked $so_file -> python/nvfp4_gemm/"

echo
echo "done. Add python/ to PYTHONPATH, then:"
echo "  PYTHONPATH=\$PWD/python python tests/python/test_gemm.py"

cd "$original_dir"

#!/usr/bin/env bash
#
# In-place development build, mirroring DeepGEMM's develop.sh.
#
# Builds the C++ extension and symlinks it plus the JIT include root back into
# the source tree, so `import nvfp4_gemm` works from the repo without installing.
#
set -euo pipefail

original_dir="$(pwd)"
script_dir="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
cd "$script_dir"

for path in 3rdparty/cutlass/include/cutlass \
            3rdparty/DeepGEMM/deep_gemm/include/deep_gemm; do
    if [[ ! -d "$path" ]]; then
        echo "error: missing $path" >&2
        echo "run: git submodule update --init --recursive --depth 1" >&2
        exit 1
    fi
done

# The JIT compiles generated code with a single -I, so collect every header
# namespace it can reference under one root.
echo "==> linking JIT include root"
mkdir -p nvfp4_gemm/include
ln -sfn "$script_dir/include/nvfp4_gemm"                          nvfp4_gemm/include/nvfp4_gemm
ln -sfn "$script_dir/3rdparty/DeepGEMM/deep_gemm/include/deep_gemm" nvfp4_gemm/include/deep_gemm
ln -sfn "$script_dir/3rdparty/cutlass/include/cutlass"            nvfp4_gemm/include/cutlass
ln -sfn "$script_dir/3rdparty/cutlass/include/cute"               nvfp4_gemm/include/cute

echo "==> building the C++ extension"
rm -rf build dist ./*.egg-info
python setup.py build

so_file="$(find build -name '_C*.so' -type f | head -n 1)"
if [[ -z "$so_file" ]]; then
    echo "error: no extension .so produced under build/" >&2
    exit 1
fi
ln -sf "../$so_file" nvfp4_gemm/
echo "==> linked $so_file -> nvfp4_gemm/"

echo
echo "done. Try:"
echo "  python tests/test_gemm.py"

cd "$original_dir"

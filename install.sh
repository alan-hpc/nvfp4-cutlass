#!/usr/bin/env bash
#
# Build a wheel and install it, mirroring DeepGEMM's install.sh.
#
set -euo pipefail

original_dir="$(pwd)"
script_dir="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
cd "$script_dir"

rm -rf build dist ./*.egg-info
python setup.py bdist_wheel
pip install dist/*.whl --force-reinstall

cd "$original_dir"

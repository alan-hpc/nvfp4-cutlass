#!/usr/bin/env bash
#
# Apply .clang-format to the project's C++/CUDA sources.
#
#   ./scripts/format.sh          # format in place
#   ./scripts/format.sh --check  # fail if anything is unformatted (for CI)
#
# Only tracked sources outside 3rdparty/ are touched -- the submodules keep
# upstream's style, and reformatting them would make every future submodule
# update a conflict.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

say()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

command -v clang-format >/dev/null || die "clang-format not found; pip install clang-format"

mapfile -t FILES < <(git ls-files '*.cuh' '*.cu' '*.hpp' '*.cpp' | grep -v '^3rdparty/')
[[ ${#FILES[@]} -gt 0 ]] || die "no sources found"

say "clang-format $(clang-format --version | sed 's/.*version //') on ${#FILES[@]} files"

if [[ "${1:-}" == "--check" ]]; then
    if clang-format --style=file --dry-run -Werror "${FILES[@]}"; then
        say "all files are formatted"
    else
        die "formatting differences above; run ./scripts/format.sh"
    fi
else
    clang-format --style=file -i "${FILES[@]}"
    say "formatted in place"
fi

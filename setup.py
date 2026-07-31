import os
import shutil
from pathlib import Path

import setuptools
import torch
from setuptools import find_packages
from setuptools.command.build_py import build_py
from torch.utils.cpp_extension import CUDAExtension, CUDA_HOME, BuildExtension

current_dir = os.path.dirname(os.path.realpath(__file__))

# The JIT resolves generated `#include <nvfp4_gemm/...>`, `<deep_gemm/...>`,
# `<cutlass/...>` and `<cute/...>` against a *single* include root, so the
# package ships one directory containing all four.
jit_include_dirs = {
    'nvfp4_gemm': 'include/nvfp4_gemm',
    'deep_gemm': '3rdparty/DeepGEMM/deep_gemm/include/deep_gemm',
    'cutlass': '3rdparty/cutlass/include/cutlass',
    'cute': '3rdparty/cutlass/include/cute',
}

cxx_flags = [
    '-std=c++20', '-O3', '-fPIC', '-fconcepts',
    '-Wno-psabi', '-Wno-deprecated-declarations', '-Wno-abi',
    f'-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}',
]

build_include_dirs = [
    f'{CUDA_HOME}/include',
    f'{CUDA_HOME}/include/cccl',
    os.path.join(current_dir, 'include'),
    os.path.join(current_dir, '3rdparty/DeepGEMM/deep_gemm/include'),
    os.path.join(current_dir, '3rdparty/cutlass/include'),
    os.path.join(current_dir, '3rdparty/DeepGEMM/third-party/fmt/include'),
]


def _check_submodules() -> None:
    missing = [name for name, path in jit_include_dirs.items()
               if not os.path.isdir(os.path.join(current_dir, path))]
    if missing:
        raise SystemExit(
            f'missing headers for: {", ".join(missing)}\n'
            'run: git submodule update --init --recursive --depth 1'
        )


class CustomBuildPy(build_py):
    """Materialize the JIT include root inside the package."""

    def run(self):
        _check_submodules()
        self.prepare_includes()
        super().run()

    def prepare_includes(self):
        dst_root = Path(self.build_lib) / 'nvfp4_gemm' / 'include'
        if dst_root.exists():
            shutil.rmtree(dst_root)
        dst_root.mkdir(parents=True)
        for name, src in jit_include_dirs.items():
            shutil.copytree(os.path.join(current_dir, src), dst_root / name,
                            dirs_exist_ok=True)


if __name__ == '__main__':
    _check_submodules()
    setuptools.setup(
        name='nvfp4_gemm',
        version='0.1.0',
        packages=find_packages('.', include=['nvfp4_gemm', 'nvfp4_gemm.*']),
        package_data={'nvfp4_gemm': ['include/**/*']},
        ext_modules=[
            CUDAExtension(
                name='nvfp4_gemm._C',
                sources=['csrc/python_api.cpp'],
                include_dirs=build_include_dirs,
                libraries=['cudart', 'cuda', 'nvrtc'],
                library_dirs=[f'{CUDA_HOME}/lib64', f'{CUDA_HOME}/lib64/stubs'],
                extra_compile_args={'cxx': cxx_flags},
            )
        ],
        cmdclass={
            'build_py': CustomBuildPy,
            'build_ext': BuildExtension,
        },
        zip_safe=False,
    )

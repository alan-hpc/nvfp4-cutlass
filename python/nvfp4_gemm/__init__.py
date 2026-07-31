"""Fused BF16 x NVFP4 dual-A grouped GEMM for SM100 / SM103 (B200 / B300)."""

import os
import subprocess

import torch

__version__ = '0.1.0'

from . import _C
from ._C import (
    get_num_sms,
    m_grouped_bf16_dual_nvfp4_gemm_contiguous,
    set_num_sms,
)

from . import layout
from .layout import (
    dequantize_weight_nvfp4,
    make_m_indices,
    quantize_weight_nvfp4,
)


def _find_cuda_home() -> str:
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home:
        return cuda_home
    try:
        nvcc = subprocess.check_output(['which', 'nvcc']).decode().strip()
        return os.path.dirname(os.path.dirname(nvcc))
    except Exception:
        return '/usr/local/cuda'


# Generated kernels are compiled against a single include root, so the package
# ships one directory holding nvfp4_gemm/, cutlass/ and cute/ headers.
# develop.sh symlinks it; setup.py copies it into the wheel.
_C.init(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'include'), _find_cuda_home())

__all__ = [
    'm_grouped_bf16_dual_nvfp4_gemm_contiguous',
    'quantize_weight_nvfp4',
    'dequantize_weight_nvfp4',
    'make_m_indices',
    'layout',
    'get_num_sms',
    'set_num_sms',
]

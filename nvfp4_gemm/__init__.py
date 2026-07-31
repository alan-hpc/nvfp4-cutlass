"""Fused BF16 x NVFP4 dual-A grouped GEMM for SM100 / SM103 (B200 / B300)."""

import os
import subprocess

import torch

__version__ = '0.1.0'

from . import _C
from ._C import (
    m_grouped_bf16_dual_nvfp4_gemm_contiguous,
    set_num_sms,
    get_num_sms,
    set_tc_util,
    get_tc_util,
    set_pdl,
    get_pdl,
)

from . import layout
from .layout import (
    quantize_weight_nvfp4,
    dequantize_weight_nvfp4,
    make_m_indices,
)


def _find_cuda_home() -> str:
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home:
        return cuda_home
    # Same fallback DeepGEMM uses: locate nvcc and walk up two levels.
    try:
        nvcc = subprocess.check_output(['which', 'nvcc']).decode().strip()
        return os.path.dirname(os.path.dirname(nvcc))
    except Exception:
        return '/usr/local/cuda'


# The JIT compiles generated kernels against a single include root, so the
# package ships one directory holding nvfp4_gemm/, deep_gemm/, cutlass/ and
# cute/ headers (see develop.sh / setup.py).
_C.init(os.path.dirname(os.path.abspath(__file__)), _find_cuda_home())

__all__ = [
    'm_grouped_bf16_dual_nvfp4_gemm_contiguous',
    'quantize_weight_nvfp4',
    'dequantize_weight_nvfp4',
    'make_m_indices',
    'layout',
    'set_num_sms', 'get_num_sms',
    'set_tc_util', 'get_tc_util',
    'set_pdl', 'get_pdl',
]

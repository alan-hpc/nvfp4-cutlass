#include <pybind11/pybind11.h>
#include <torch/python.h>

// DeepGEMM's JIT runtime is reused wholesale: the same NVCC driver, cubin cache,
// include hashing and launch path.  Only the kernel and its launcher are ours,
// so there is no second JIT to keep in sync.
#include "../3rdparty/DeepGEMM/csrc/apis/runtime.hpp"

#include "apis/gemm.hpp"

#ifndef TORCH_EXTENSION_NAME
#    define TORCH_EXTENSION_NAME _C
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "Fused BF16 x NVFP4 dual-A GEMM for SM100/SM103";

    // `init`, `set_num_sms`, `set_tc_util`, `set_pdl`, ...
    deep_gemm::runtime::register_apis(m);

    nvfp4_gemm::api::register_apis(m);
}

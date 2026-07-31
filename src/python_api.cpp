#include <pybind11/pybind11.h>
#include <torch/python.h>

#include "api.hpp"

#ifndef TORCH_EXTENSION_NAME
#    define TORCH_EXTENSION_NAME _C
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "Fused BF16 x NVFP4 dual-A GEMM for SM100/SM103";
    nvfp4_gemm::api::register_apis(m);
}

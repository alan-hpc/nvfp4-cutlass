#pragma once

// Host-side error reporting.

#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

namespace nvfp4_gemm {

class NVFP4Exception final : public std::exception
{
    std::string message;

public:
    NVFP4Exception(const char* kind, const char* file, int line, const std::string& what)
    {
        std::ostringstream oss;
        oss << "nvfp4_gemm " << kind << " error (" << file << ":" << line << "): " << what;
        message = oss.str();
    }

    const char* what() const noexcept override
    {
        return message.c_str();
    }
};

}   // namespace nvfp4_gemm

#ifndef NVFP4_HOST_ASSERT
#    define NVFP4_HOST_ASSERT(cond)                                                       \
        do                                                                                \
        {                                                                                 \
            if (not(cond))                                                                \
                throw nvfp4_gemm::NVFP4Exception("assertion", __FILE__, __LINE__, #cond); \
        } while (0)
#endif

#ifndef NVFP4_HOST_ASSERT_MSG
#    define NVFP4_HOST_ASSERT_MSG(cond, msg)                                                                          \
        do                                                                                                            \
        {                                                                                                             \
            if (not(cond))                                                                                            \
                throw nvfp4_gemm::NVFP4Exception("assertion", __FILE__, __LINE__, std::string(#cond) + ": " + (msg)); \
        } while (0)
#endif

#ifndef NVFP4_CUDA_CHECK
#    define NVFP4_CUDA_CHECK(expr)                                                                    \
        do                                                                                            \
        {                                                                                             \
            const cudaError_t _e = (expr);                                                            \
            if (_e != cudaSuccess)                                                                    \
                throw nvfp4_gemm::NVFP4Exception("cuda", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        } while (0)
#endif

#ifndef NVFP4_CU_CHECK
#    define NVFP4_CU_CHECK(expr)                                                               \
        do                                                                                     \
        {                                                                                      \
            const CUresult _e = (expr);                                                        \
            if (_e != CUDA_SUCCESS)                                                            \
            {                                                                                  \
                const char* _s = nullptr;                                                      \
                cuGetErrorString(_e, &_s);                                                     \
                throw nvfp4_gemm::NVFP4Exception("driver", __FILE__, __LINE__, _s ? _s : "?"); \
            }                                                                                  \
        } while (0)
#endif

#pragma once

// Compilation-mode detection and device assertion macros.
//
// Derived from DeepGEMM (MIT, Copyright (c) 2025 DeepSeek),
// `deep_gemm/common/{compile,exception}.cuh`, reduced to what this kernel uses
// and renamed into the `NVFP4_` prefix so the two can coexist in one build.

#include <cuda/std/cstdint>
#include <cutlass/detail/helper_macros.hpp>

#if defined(__NVCC__) or (defined(__clang__) and defined(__CUDA__)) or defined(__CUDACC_RTC__) or defined(__CLION_IDE__)
#    define NVFP4_IN_CUDA_COMPILATION
#endif

#ifndef NVFP4_DEVICE_ASSERT
#    define NVFP4_DEVICE_ASSERT(cond)                                                          \
        do                                                                                     \
        {                                                                                      \
            if (not(cond))                                                                     \
            {                                                                                  \
                printf("Assertion failed: %s:%d, condition: %s\n", __FILE__, __LINE__, #cond); \
                asm("trap;");                                                                  \
            }                                                                                  \
        } while (0)
#endif

// Same check without the printf, for hot paths where the format string and its
// arguments would otherwise cost registers.
#ifndef NVFP4_TRAP_ONLY_DEVICE_ASSERT
#    define NVFP4_TRAP_ONLY_DEVICE_ASSERT(cond) \
        do                                      \
        {                                       \
            if (not(cond))                      \
                asm("trap;");                   \
        } while (0)
#endif

#ifndef NVFP4_STATIC_ASSERT
#    define NVFP4_STATIC_ASSERT(cond, ...) static_assert(cond, __VA_ARGS__)
#endif

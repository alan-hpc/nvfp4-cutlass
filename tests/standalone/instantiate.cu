// Compile-only instantiation of the dual-NVFP4 kernel.
//
// This is the cheapest possible check and the first thing `build.sh` runs: it
// forces the whole template through the compiler and the inline PTX through
// ptxas, without needing a GPU. It catches the two failure modes that are pure
// compile-time -- a wrong CUTLASS type name and a malformed `tcgen05` mnemonic.

#include <nvfp4_gemm/impls/sm100_bf16_dual_nvfp4_gemm.cuh>

using namespace nvfp4_gemm;

// Default production shape: E=4 experts, N=1024, K=2048, tile 128x256x128.
static void __instantiate_default() {
    auto ptr = reinterpret_cast<void*>(&sm100_bf16_dual_nvfp4_gemm_impl<
        /*SHAPE_N=*/1024, /*SHAPE_K=*/2048,
        /*BLOCK_M=*/128, /*BLOCK_N=*/256, /*BLOCK_K=*/128,
        /*kNumGroups=*/4,
        /*kSwizzleAMode=*/128, /*kSwizzleABMode=*/64, /*kSwizzleCDMode=*/128,
        /*kNumStages=*/2,
        /*kNumTransformWarps=*/8,
        /*kNumEpilogueThreads=*/128,
        /*kNumSMs=*/148,
        deep_gemm::GemmType::MGroupedContiguous,
        transform::ScalePolicy::DerivedDiv8>);
    (void) ptr;
}

// A second shape, to make sure the derived constants really are derived:
// BLOCK_N=128 flips `kNumEpilogueStages` from 1 to 2.
static void __instantiate_small_n() {
    auto ptr = reinterpret_cast<void*>(&sm100_bf16_dual_nvfp4_gemm_impl<
        0, 0,
        128, 128, 128,
        4,
        128, 64, 128,
        3,
        8,
        128,
        148,
        deep_gemm::GemmType::MGroupedContiguous,
        transform::ScalePolicy::ResidualAmax>);
    (void) ptr;
}

int main() {
    // Never called; referencing them keeps the instantiations alive.
    (void) &__instantiate_default;
    (void) &__instantiate_small_n;
    return 0;
}

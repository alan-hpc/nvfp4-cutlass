# nvfp4-cutlass

Fused **BF16 x NVFP4** grouped-MoE GEMM for SM100/SM103 (B200/B300), built on the
DeepGEMM kernel skeleton.

BF16 activations are decomposed *inside the GEMM mainloop* into two NVFP4 passes
-- a most-significant pass plus a residual pass -- and both are multiplied against
the same NVFP4 weight tile into a single accumulator. That lets a BF16 x FP4
mixed-input GEMM run on the FP4 tensor cores without ever materializing a
quantized activation in global memory.

The algorithm is specified in [`docs/bf16-dual-nvfp4-algorithm.html`](docs/bf16-dual-nvfp4-algorithm.html).

```
C = A0 W^T + A1 W^T
A0 = dec(q0) * s0        s0 = Q_e4m3(amax(x) / 6)
A1 = dec(q1) * s1        s1 = Q_e4m3(s0 * 2^-3)
                         q0 = Q_e2m1(x * rcp(s0))
                         q1 = Q_e2m1((x * rcp(s0) - dec(q0)) * 8)
```

## Repository layout

| Path | Contents |
| --- | --- |
| `include/nvfp4_gemm/impls/sm100_bf16_dual_nvfp4_gemm.cuh` | The fused kernel (Algorithm 2) |
| `include/nvfp4_gemm/transform/dual_nvfp4.cuh` | Algorithm 1 plus the SMEM swizzle addressing it needs |
| `include/nvfp4_gemm/ptx/tcgen05_nvfp4.cuh` | NVFP4 `block16` UMMA and the FP4/E4M3 conversion intrinsics |
| `include/nvfp4_gemm/epilogue/sm100_store_cd_gscale.cuh` | Epilogue store with the per-expert `G_W` multiply |
| `csrc/jit_kernels/impls/sm100_bf16_dual_nvfp4_gemm.hpp` | Host-side JIT launcher and config |
| `tests/reference/dual_nvfp4.py` | Executable numerics specification (CPU) |
| `tests/test_dual_nvfp4_reference.py` | Numerics tests |

## What DeepGEMM provides, and what had to be added

The kernel follows `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh`
closely: the persistent `sched::Scheduler`, the ring-buffer barrier discipline,
the lane-broadcast trick for per-stage UMMA descriptors, the UTCCP scale path,
and the swizzled TMEM->SMEM->TMA store epilogue are all DeepGEMM's.

Four things are not in DeepGEMM and are new here:

1. **NVFP4 UMMA.** DeepGEMM ships MXFP4 only (`kind::mxf4.block_scale.block32`,
   UE8M0 scales). NVFP4 is block 16 with E4M3 scales, a different operand kind:
   `kind::mxf4nvf4.block_scale.block16`.
2. **The in-mainloop transform.** A dedicated producer warp group runs
   Algorithm 1 on each staged A tile, writing A0/A1 and both scale sets into
   shared memory. DeepGEMM's producers only move bytes; they never compute.
3. **Two MMA passes per K step** sharing one W/SFB fragment and one accumulator,
   with `accumulate = false` on exactly the first instruction of the first K
   block.
4. **The `G_W` epilogue.** NVFP4 weights carry a per-tensor FP32 scale on top of
   their block scales; it is applied once per output element instead of once per
   MMA operand.

## Warp specialization

| Warp | Role |
| --- | --- |
| 0 | TMA producer: A (BF16), W (packed E2M1), SFB |
| 1 | MMA issue + UTCCP of SFA0/SFA1/SFB into TMEM |
| 2 | SFB transpose into UTCCP order |
| 3 .. 10 | Transform producers running Algorithm 1 |
| 11 .. 14 | Epilogue: TMEM -> reg, `x G_W`, BF16 store |

480 threads for the default `128 x 256 x 128` tile.

## Resource budget (default tile 128 x 256 x 128)

| Resource | Value |
| --- | --- |
| Tensor memory | accumulator 256 + SFA0 8 + SFA1 8 + SFB 16 = **288 / 512 columns** |
| Accumulator stages | 1 (two would need 544 columns) |
| Shared memory per stage | 68 KB (A 32 + A0 8 + A1 8 + W 16 + scales 4) |
| Pipeline depth | 2 stages, 168 KB of the 227 KB SM budget |
| Transform work | 1 scale-atom (64 K elements) per thread per K tile |

The 288/512 figure matches the column budget the algorithm doc reports for the
same tile, which is a useful independent check on the scale-factor layout.

## Deliberate deviations from the doc's kernel

**Transform threads own whole scale atoms.** The doc gives each lane 8 BF16 --
half of a 16-element block -- and merges the amax with a butterfly shuffle. Both
lanes of a pair then compute the same `s0`, `rcp(s0)` and `s1`, and store the same
two scale bytes to the same address. The doc notes this redundancy and concludes
it cannot be predicated away profitably under SIMT, which is correct. Here each
thread instead owns a full 64-element scale atom, so the redundancy never exists:
no shuffle, one scale computation per block, one 4-byte SFA store per atom instead
of eight 1-byte stores. Per 64 elements per thread this trades 60 `fmax` for the
doc's 56 `fmax` + 8 `shfl`, and halves the scalar scale work.

**SFA skips the UTCCP transpose.** DeepGEMM has to shuffle TMA-loaded scales into
the layout UTCCP wants. We generate SFA ourselves, so the transform warps write it
pre-transposed (`j = (m % 32) * 4 + m / 32`). Only SFB, which still arrives by TMA,
needs warp 2.

**Transform is a dedicated warp group, not the epilogue warps.** The doc reuses
warps 0-3 for both transform and epilogue, saving 128 threads and measuring
0.172 ms -> 0.1395 ms. That reuse also serializes the epilogue of block *i* against
the mainloop of block *i+1*, which the persistent scheduler would otherwise overlap.
Which effect wins is shape-dependent; this needs measurement, and the choice here
is a template parameter away from being flipped.

**No 2-CTA multicast.** `kNumMulticast == 1` only. 2-SM UMMA with producer warps
writing operand tiles that the peer CTA's MMA consumes is a real extension, not a
parameter flip, so it is left out rather than half-done.

## Accuracy

`tests/test_dual_nvfp4_reference.py` runs on CPU and passes:

```
dual vs single pass:  0.09519 -> 0.01205 relative L2  (7.9x better)
grouped GEMM vs ground truth: cosine 0.999926
```

On the doc's own ground-truth definition -- `C_ref = A_bf16 * (dec(W) * S_W * G_W)^T`
in FP32 -- the dual-pass path reaches **cosine 0.9999**, against **0.9954** for a
single NVFP4 activation pass.

> **Note on the doc's reported cosine.** The doc quotes 0.995396 (fused) and
> 0.995407 (hybrid) against that ground truth. Those numbers are reproducible only
> when comparing against *full-precision BF16 weights*, where weight quantization
> dominates the error: weight quantization alone gives cosine 0.995438, and the
> single-pass activation path gives 0.995417. Measured against the ground truth as
> the doc defines it, a working dual-A path should land near 0.9999, not 0.9954.
> Use 0.999+ as the kernel's acceptance threshold; 0.9954 would silently accept a
> kernel whose second pass does nothing.

## Status

The Python reference is tested and passing. **The CUDA kernel has not been
compiled or run**: this checkout has no CUTLASS (the `3rdparty/cutlass` submodule
is not initialized), no `nvcc`, and no Blackwell GPU.

Before trusting the kernel, validate these on hardware. Each is a point where the
implementation encodes an assumption that could not be checked offline:

1. **`cutlass::float_ue4m3_t`** is the right SF type for
   `make_instr_desc_block_scaled` on the NVFP4 path.
2. **NVFP4 SF addressing in TMEM.** The kernel advances the SF TMEM column by 4
   per UMMA-K step and passes `sf_id = 0`, on the model that one `.block16`
   instruction consumes exactly one 128x4 atom (4 sub-blocks of 16 over 64 K).
   Confirm against the PTX ISA description of `tcgen05.mma ... .block16`.
3. **Packed-FP4 SMEM.** The transform writes truly packed A0/A1 (two elements per
   byte) and the host builds W's TMA descriptor with `fp4_unpacked_smem = false`.
   DeepGEMM's own FP4 path uses the *unpacked* SMEM variant, so confirm UMMA reads
   the packed layout with a 64 B swizzle as assumed.
4. **`cvt.rn.satfinite.e2m1x2.f32` operand order** -- which argument lands in the
   high nibble. `ptx/tcgen05_nvfp4.cuh` assumes first argument -> high nibble.
5. **The swizzle replication** in `swizzled_byte_offset`. The transform warps read
   A with plain `LDS`, so they reproduce the TMA swizzle by hand; a mismatch is
   silent and produces wrong numbers, not a fault.

Item 5 is the cheapest to test in isolation: TMA a known pattern into the A stage,
run only the transform, copy A0/SFA0 back out, and compare against
`tests/reference/dual_nvfp4.py`.

## Next steps

- **Decouple the A and A0/A1 pipelines.** The BF16 A tile is 32 KB of the 68 KB
  stage and is consumed by the transform warps, while A0/A1 are consumed by MMA.
  Staging them separately would let the HBM-latency-bound A path run deeper than
  2 stages without paying for extra FP4 buffers.
- **Reuse A across N tiles.** The doc's measurements are unambiguous that this is
  the structural win (0.1142 ms fused vs 0.0683 ms hybrid, ~1.67x): the fused path
  recomputes the same transform for every N tile. The hybrid path trades a global
  workspace for doing it once.
- **Sweep `num_transform_warps` and the epilogue-reuse choice** on hardware.

## Running the reference tests

```bash
python3 tests/test_dual_nvfp4_reference.py
```

Requires only PyTorch (CPU is fine).

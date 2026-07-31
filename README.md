# nvfp4-cutlass

Fused **BF16 x NVFP4** grouped-MoE GEMM for SM100/SM103 (B200/B300), built on the
DeepGEMM kernel skeleton.

BF16 activations are decomposed *inside the GEMM mainloop* into two NVFP4 passes
-- a most-significant pass plus a residual pass -- and both are multiplied against
the same NVFP4 weight tile into a single accumulator. That lets a BF16 x FP4
mixed-input GEMM run on the FP4 tensor cores without ever materializing a
quantized activation in global memory.

The algorithm is specified in [`docs/bf16-dual-nvfp4-algorithm.html`](docs/bf16-dual-nvfp4-algorithm.html).
How this implementation was derived from it -- the layout derivations, the cross-checks,
and the list of assumptions still to validate -- is written up in
[`docs/implementation-notes.md`](docs/implementation-notes.md) (Chinese).

```
C = A0 W^T + A1 W^T
A0 = dec(q0) * s0        s0 = Q_e4m3(amax(x) / 6)
A1 = dec(q1) * s1        s1 = Q_e4m3(s0 * 2^-3)
                         q0 = Q_e2m1(x * rcp(s0))
                         q1 = Q_e2m1((x * rcp(s0) - dec(q0)) * 8)
```

## Quick start

Requires CUDA >= 12.9 (the NVFP4 UMMA spelling and the `sm_100f` family target
both need it; `sm_100a` will not load on a B300), a Blackwell GPU, and PyTorch.

```bash
git submodule update --init --recursive --depth 1
./develop.sh                       # build the extension, link the JIT include root
python tests/test_gemm.py          # end-to-end correctness
python tests/test_gemm.py --bench  # and timing
```

Production use:

```python
import torch, nvfp4_gemm

# Offline: quantize weights into the kernel's physical layouts.
b, sfb, gw = nvfp4_gemm.quantize_weight_nvfp4(w)          # w: (E, N, K) BF16

# Online: BF16 activations in, BF16 out. A is never quantized to global memory.
m_indices = nvfp4_gemm.make_m_indices([rows_per_expert] * num_experts, device='cuda')
d = torch.empty(m, n, dtype=torch.bfloat16, device='cuda')
nvfp4_gemm.m_grouped_bf16_dual_nvfp4_gemm_contiguous(a, b, sfb, gw, d, m_indices)
```

Kernels are JIT-compiled on first use through DeepGEMM's compiler and cached in
`~/.deep_gemm`, so there is no CUDA build at install time.

## Benchmarks

```bash
python benchmarks/bench_moe_forward.py              # all shapes, all backends
python benchmarks/bench_moe_forward.py --breakdown  # plus per-stage timing
```

The comparison is over the **complete online chain**, not a bare GEMM. A real
forward hands the expert GEMM a BF16 activation, so every quantized baseline must
convert it first and that conversion sits inside the timed region:

| backend | online chain |
| --- | --- |
| `dual_nvfp4` | BF16 → fused decompose + 2× block-scaled MMA → BF16 (one kernel) |
| `single_nvfp4` | same kernel, residual pass disabled |
| `mxfp8_mxfp4` | BF16 → `mxfp8_quantize` → MXFP8 × MXFP4 grouped GEMM |
| `fi_nvfp4_moe` | BF16 → `fp4_quantize` → FlashInfer fused MoE (**different scope**) |

Timing is CUDA Graph replay. The doc records API event times going bimodal
between ~0.060 and ~0.123 ms on one kernel, the difference being host dispatch
gap; a multi-kernel baseline pays that gap more often than a single-kernel one,
so eager timing would systematically flatter the fused path.

Three things about this comparison are worth stating plainly:

- **`single_nvfp4` is the controlled A/B.** Same kernel, same tiles, same
  pipeline, same epilogue, one MMA instead of two. It isolates what the second
  pass costs and what it buys, with no cross-library confound. Cross-vendor
  numbers always mix in scheduler, tile and epilogue differences.
- **DeepGEMM cannot be the NVFP4 baseline.** Its FP4 kernel asserts
  `gran_k ∈ {32, 128}` -- MXFP4 (block 32, UE8M0) and MXFP8, not NVFP4's block 16
  with E4M3 scales. That gap is exactly why this repo has its own NVFP4 UMMA.
- **`fi_nvfp4_moe` measures different work.** `cutlass_fused_moe` is a full MoE
  (routing, two GEMMs, SwiGLU, weighted reduce); the rest are one expert GEMM.
  The harness refuses to print a speedup for it rather than mislead.

Accuracy is reported next to speed for every backend, against the same FP32
ground truth. A cheaper activation format that loses precision is not a free win,
and the doc's own sweep found MXFP8 ahead by 1.61-2.48x on this workload -- read
both columns.

## Bring-up on new hardware

Before trusting an end-to-end number, run the transform in isolation. It has no
MMA, no UTCCP and no epilogue, so a failure points at exactly one thing:

```bash
./scripts/build.sh          # compile-check + build the standalone test
./scripts/run.sh            # A-tile swizzle + Algorithm 1, checked per element
```

`./scripts/build.sh --ptx` additionally dumps PTX and greps it for the NVFP4 MMA,
which is the fastest way to confirm `kind::mxf4nvf4` actually survived codegen.

## Repository layout

| Path | Contents |
| --- | --- |
| `include/nvfp4_gemm/impls/sm100_bf16_dual_nvfp4_gemm.cuh` | The fused kernel (Algorithm 2) |
| `include/nvfp4_gemm/transform/dual_nvfp4.cuh` | Algorithm 1 plus the SMEM swizzle addressing it needs |
| `include/nvfp4_gemm/ptx/tcgen05_nvfp4.cuh` | NVFP4 `block16` UMMA and the FP4/E4M3 conversion intrinsics |
| `include/nvfp4_gemm/epilogue/sm100_store_cd_gscale.cuh` | Epilogue store with the per-expert `G_W` multiply |
| `csrc/jit_kernels/impls/sm100_bf16_dual_nvfp4_gemm.hpp` | Host-side JIT launcher and config |
| `csrc/apis/gemm.hpp`, `csrc/python_api.cpp` | pybind entry points |
| `nvfp4_gemm/layout.py` | Weight quantization and the NVFP4 physical layouts |
| `tests/test_gemm.py` | Production end-to-end test (needs a GPU) |
| `tests/test_layout.py` | Layout tests (CPU) |
| `tests/reference/dual_nvfp4.py` | Executable numerics specification (CPU) |
| `tests/test_dual_nvfp4_reference.py` | Numerics tests (CPU) |
| `tests/standalone/` | Torch-free bring-up harness |

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

## Target architecture

Compile for **`sm_100f`**. B200 (10.0) and B300 (10.3) share the SM100 family
target, and `scripts/build.sh` selects it automatically. The kernel uses no
sm_103-exclusive feature; its arch guard is `__CUDA_ARCH__ >= 1000`, and
DeepGEMM's JIT independently resolves 10.3 to `sm_100f` as well.

`sm_100a` is arch-specific to 10.0 and **will not load on a B300** -- which is
why CUDA >= 12.9 is a hard requirement rather than a recommendation, since older
nvcc cannot emit family targets at all.

## Accuracy

`tests/test_dual_nvfp4_reference.py` and `tests/test_layout.py` run on CPU and pass:

```
dual vs single pass:  0.09519 -> 0.01205 relative L2  (7.9x better)
grouped GEMM vs ground truth: cosine 0.999926
weight quantization roundtrip: rel L2 0.09547, cosine 0.995415
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

The CPU-side tests -- numerics reference and physical layouts -- are passing.
**The CUDA kernel has not been compiled or run**: the machine it was written on
has no CUTLASS checkout, no `nvcc` and no Blackwell GPU, so `scripts/build.sh`
has never been executed. Expect to fix compile errors on the first run.

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

Item 1 fails at compile time, so `scripts/build.sh --check-only` settles it.
Items 4 and 5 both surface in `scripts/run.sh`, and their signatures differ: a
wrong `cvt` operand order swaps adjacent element pairs, while a wrong swizzle
corrupts whole rows. Items 2 and 3 need either the PTX ISA text or a run of
`tests/test_gemm.py`.

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

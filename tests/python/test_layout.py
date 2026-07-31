"""Tests for the NVFP4 weight layouts (CPU only, no GPU or extension needed).

These check the *physical* layouts the kernel's TMA descriptors describe, which
is exactly the class of bug that produces silently wrong GEMM output rather than
a crash. `layout.py` is loaded by path, so this runs without building the C++
extension or having a GPU.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

# Load layout.py by path rather than as `nvfp4_gemm.layout`: the package's
# __init__ imports the compiled extension, and these tests are pure Python.
_spec = importlib.util.spec_from_file_location(
    'nvfp4_layout', Path(__file__).resolve().parents[2] / 'python' / 'nvfp4_gemm' / 'layout.py')
layout = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(layout)

BLOCK_SIZE = layout.BLOCK_SIZE
K_PER_SF_ATOM = layout.K_PER_SF_ATOM
dequantize_weight_nvfp4 = layout.dequantize_weight_nvfp4
make_m_indices = layout.make_m_indices
pack_e2m1_pairs = layout.pack_e2m1_pairs
quantize_e2m1_codes = layout.quantize_e2m1_codes
quantize_weight_nvfp4 = layout.quantize_weight_nvfp4
pack_sf_atoms = layout.pack_sf_atoms


def test_e2m1_codes() -> None:
    """Codes must match the E2M1 encoding, including ties and the sign bit."""
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    assert torch.equal(quantize_e2m1_codes(grid), torch.arange(8, dtype=torch.uint8))
    assert torch.equal(quantize_e2m1_codes(-grid[1:]),
                       (torch.arange(1, 8, dtype=torch.uint8) | 8))

    ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    assert torch.equal(quantize_e2m1_codes(ties),
                       torch.tensor([0, 2, 2, 4, 4, 6, 6], dtype=torch.uint8))

    saturating = torch.tensor([6.5, 1e4, float('inf')])
    assert torch.equal(quantize_e2m1_codes(saturating), torch.full((3,), 7, dtype=torch.uint8))
    print('ok  e2m1 codes: grid, ties, sign, saturation')


def test_nibble_packing_order() -> None:
    """Element 2i must land in the low nibble, 2i+1 in the high nibble.

    Getting this backwards swaps every adjacent pair of activations against the
    weights -- a bug that still produces plausible-looking output.
    """
    codes = torch.tensor([[1, 2, 3, 4]], dtype=torch.uint8)
    packed = pack_e2m1_pairs(codes).to(torch.uint8)
    assert packed.tolist() == [[0x21, 0x43]], packed.tolist()
    print('ok  nibble packing: low = even element, high = odd element')


def test_sf_atom_word_packing() -> None:
    """Sub-block j of an atom must sit in byte j, and the tensor must be (sf_k, n)."""
    num_experts, n, k = 2, 3, 128
    num_blocks = k // BLOCK_SIZE           # 8
    num_atoms = k // K_PER_SF_ATOM         # 2

    # Distinct, recoverable codes: expert/row/block encoded into the value.
    codes = torch.arange(num_experts * n * num_blocks, dtype=torch.uint8) % 251 + 1
    codes = codes.reshape(num_experts, n, num_blocks)


    words = pack_sf_atoms(codes)
    assert words.shape == (num_experts * num_atoms, n), words.shape

    for e in range(num_experts):
        for row in range(n):
            for atom in range(num_atoms):
                w = int(words[e * num_atoms + atom, row]) & 0xFFFFFFFF
                for j in range(4):
                    got = (w >> (8 * j)) & 0xFF
                    want = int(codes[e, row, atom * 4 + j])
                    assert got == want, f'e={e} row={row} atom={atom} j={j}: {got} != {want}'
    print('ok  SF atom words: sub-block j in byte j, laid out as (E*K/64, N)')


def test_weight_quantization_roundtrip() -> None:
    """quantize -> dequantize must recover the weights to NVFP4 accuracy.

    This exercises the packing and the layout permutation together: if either the
    nibble order or the (sf_k, n) transpose were wrong, the recovered weights
    would be scrambled and the cosine would collapse.
    """
    torch.manual_seed(0)
    num_experts, n, k = 3, 256, 512
    w = torch.randn(num_experts, n, k, dtype=torch.bfloat16) * 0.5

    b, sfb, gw = quantize_weight_nvfp4(w)
    assert b.shape == (num_experts * n, k // 2) and b.dtype == torch.int8
    assert sfb.shape == (num_experts * k // K_PER_SF_ATOM, n) and sfb.dtype == torch.int32
    assert gw.shape == (num_experts,) and gw.dtype == torch.float32

    w_deq = dequantize_weight_nvfp4(b, sfb, gw, num_experts, n, k)
    w32 = w.to(torch.float32)

    rel = float((w_deq - w32).norm() / w32.norm())
    cos = float((w_deq.flatten() @ w32.flatten()) / (w_deq.norm() * w32.norm()))

    # Per-16-block E4M3-scaled E2M1 lands near 9.5% relative L2 on Gaussian data
    # -- that is simply what one NVFP4 pass costs, and it matches the single-pass
    # figure the activation reference measures. The bound only has to catch a
    # scrambled layout, which collapses the cosine outright.
    assert rel < 0.12, f'roundtrip rel L2 {rel:.4f} too high'
    assert cos > 0.995, f'roundtrip cosine {cos:.6f} too low'
    print(f'ok  weight roundtrip: rel L2 {rel:.5f}, cosine {cos:.6f}')


def test_dequantized_weight_is_per_expert() -> None:
    """Each expert's global scale must apply only to its own block.

    A single shared G_W would still give a high overall cosine, so this checks
    the per-expert separation explicitly by giving experts very different scales.
    """
    torch.manual_seed(1)
    num_experts, n, k = 2, 128, 128
    w = torch.randn(num_experts, n, k, dtype=torch.bfloat16)
    w[0] *= 1000.0
    w[1] *= 0.001

    b, sfb, gw = quantize_weight_nvfp4(w)
    assert gw[0] > gw[1] * 100, f'expected well-separated global scales, got {gw.tolist()}'

    w_deq = dequantize_weight_nvfp4(b, sfb, gw, num_experts, n, k)
    for e in range(num_experts):
        rel = float((w_deq[e] - w[e].to(torch.float32)).norm() / w[e].to(torch.float32).norm())
        assert rel < 0.12, f'expert {e} rel L2 {rel:.4f}'
    print(f'ok  per-expert global scales: gw = {[round(v, 8) for v in gw.tolist()]}')


def test_m_indices() -> None:
    idx = make_m_indices([128, 256])
    assert idx.shape == (384,) and idx.dtype == torch.int32
    assert int(idx[0]) == 0 and int(idx[127]) == 0 and int(idx[128]) == 1
    # The kernel reads m_indices[m_block_idx * 128], so groups must start on a
    # 128-row boundary; a non-multiple size must be rejected rather than silently
    # mis-routed.
    try:
        make_m_indices([100, 128])
    except ValueError:
        pass
    else:
        raise AssertionError('expected ValueError for a non-128-multiple group')
    print('ok  m_indices: per-row expert ids, 128-row alignment enforced')


def main() -> int:
    tests = [
        test_e2m1_codes,
        test_nibble_packing_order,
        test_sf_atom_word_packing,
        test_weight_quantization_roundtrip,
        test_dequantized_weight_is_per_expert,
        test_m_indices,
    ]
    for test in tests:
        test()
    print(f'\n{len(tests)} tests passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

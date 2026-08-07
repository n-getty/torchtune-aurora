import torch

pytest = __import__("pytest")
compressed_tensors = pytest.importorskip("compressed_tensors")

from compressed_tensors.compressors.quantized_compressors.fp4_quantized import (
    kE2M1ToFloat,
    pack_fp4_to_uint8,
)

from experiments.kimi_k3_serving.hf_kimi_logits_reference import dequantize_mxfp4


def _random_e2m1_grid_tensor(shape, seed):
    torch.manual_seed(seed)
    grid = torch.cat([kE2M1ToFloat, -kE2M1ToFloat])
    idx = torch.randint(0, 16, shape)
    return grid[idx]


def test_dequantize_mxfp4_recovers_exact_grid_values_at_unit_scale():
    """dequantize_mxfp4 reimplements MXFP4 decompress (this compressed_tensors
    install has no decompress_weight -- see hf_kimi_logits_reference.py for
    why). Values placed exactly on the E2M1 grid must round-trip exactly
    through pack -> dequantize at scale=2^0, isolating nibble unpack + sign
    handling from any scale-related error."""
    exact_vals = _random_e2m1_grid_tensor((8, 64), seed=0)
    packed = pack_fp4_to_uint8(exact_vals)
    scale_bytes = torch.full((8, 64 // 32), 127, dtype=torch.uint8)  # 2^(127-127)=1.0

    recovered = dequantize_mxfp4(packed, scale_bytes)
    torch.testing.assert_close(recovered.float(), exact_vals, atol=0, rtol=0)


@pytest.mark.parametrize("scale_byte,expected_pow2", [(130, 8.0), (124, 0.125)])
def test_dequantize_mxfp4_applies_e8m0_scale_exactly(scale_byte, expected_pow2):
    """E8M0 scale byte b decodes to 2^(b-127) (bias-127, matching
    torch.float8_e8m0fnu). Verify both an amplifying and an attenuating scale
    apply exactly -- this is the exponent-bias direction that was cross-checked
    against vLLM's own weight_scale.view(torch.float8_e8m0fnu) reinterpretation
    rather than hand-derived, so a sign or bias error here would silently
    corrupt every MXFP4 layer's magnitude."""
    exact_vals = _random_e2m1_grid_tensor((8, 64), seed=1)
    packed = pack_fp4_to_uint8(exact_vals)
    scale_bytes = torch.full((8, 64 // 32), scale_byte, dtype=torch.uint8)

    recovered = dequantize_mxfp4(packed, scale_bytes)
    torch.testing.assert_close(
        recovered.float(), exact_vals * expected_pow2, atol=0, rtol=0
    )

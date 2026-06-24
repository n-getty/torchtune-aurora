"""Correctness check: IPEX varlen path must produce ~same output as SDPA."""
import os
os.environ["TORCHTUNE_USE_IPEX_VARLEN"] = "1"
import torch

# Force the import to pick up the env flag
import sys
sys.path.insert(0, "/lus/flare/projects/ModCon/ngetty/torchtune")
from torchtune.modules.attention_utils import _sdpa_or_flex_attention, _USE_IPEX_VARLEN

print(f"TORCHTUNE_USE_IPEX_VARLEN={_USE_IPEX_VARLEN}")
assert _USE_IPEX_VARLEN, "IPEX varlen flag not picked up"

device = torch.device("xpu:0")
dtype = torch.bfloat16
torch.manual_seed(42)

B, Hq, Hkv, S, D = 4, 32, 8, 1536, 128
# Caller already does GQA expansion in attention.py, so q,k,v all have Hq heads.
q = torch.randn(B, Hq, S, D, dtype=dtype, device=device)
k = torch.randn(B, Hq, S, D, dtype=dtype, device=device)  # post-expand
v = torch.randn(B, Hq, S, D, dtype=dtype, device=device)

# Path 1: stock SDPA (env off)
os.environ["TORCHTUNE_USE_IPEX_VARLEN"] = "0"
import importlib
import torchtune.modules.attention_utils as attn_utils
importlib.reload(attn_utils)
attn_call_sdpa = attn_utils._sdpa_or_flex_attention()
out_sdpa = attn_call_sdpa(q, k, v, mask=None, dropout_p=0.0, is_causal=True)
print(f"SDPA out: shape={out_sdpa.shape} dtype={out_sdpa.dtype}")

# Path 2: IPEX varlen (env on)
os.environ["TORCHTUNE_USE_IPEX_VARLEN"] = "1"
importlib.reload(attn_utils)
print(f"USE_IPEX_VARLEN after reload: {attn_utils._USE_IPEX_VARLEN}")
attn_call_varlen = attn_utils._sdpa_or_flex_attention()
out_varlen = attn_call_varlen(q, k, v, mask=None, dropout_p=0.0, is_causal=True)
print(f"varlen out: shape={out_varlen.shape} dtype={out_varlen.dtype}")

# Compare
diff = (out_sdpa.float() - out_varlen.float()).abs()
max_diff = diff.max().item()
mean_diff = diff.mean().item()
sdpa_mean = out_sdpa.float().abs().mean().item()
print(f"\nMax abs diff: {max_diff:.6f}")
print(f"Mean abs diff: {mean_diff:.6f}")
print(f"SDPA mean abs: {sdpa_mean:.6f}")
print(f"Relative mean diff: {mean_diff / sdpa_mean:.6f}")

# bf16 tolerance: max ~0.01 absolute, mean ~0.001 typical for these shapes
if max_diff < 0.05 and mean_diff < 0.005:
    print("✓ PASS: varlen output matches SDPA within bf16 tolerance")
else:
    print("✗ FAIL: outputs differ beyond bf16 tolerance")

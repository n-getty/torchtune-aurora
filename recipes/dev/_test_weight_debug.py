#!/usr/bin/env python3
"""Debug weight loading for Gemma4 vLLM model.
Test if k→v copy works correctly for global layers.
"""
import sys
sys.path.insert(0, "/lus/flare/projects/ModCon/ngetty/torchtune/recipes/dev/vllm_gemma4_overlay")

import torch
import os
os.environ["ZE_AFFINITY_MASK"] = "0"

# Simulate the model config loading
from vllm_gemma4_config import Gemma4TextConfig
config = Gemma4TextConfig.from_pretrained("/tmp/torchtune/gemma-4-31B")
print(f"Config: layer_types[5]={config.layer_types[5]}, k_eq_v={config.attention_k_eq_v}")
print(f"  head_dim={config.head_dim}, global_head_dim={config.global_head_dim}")
print(f"  num_kv_heads={config.num_key_value_heads}, num_global_kv_heads={config.num_global_key_value_heads}")

# Check safetensors for layer 5 (first global layer)
import json
from safetensors import safe_open

idx = json.load(open("/tmp/torchtune/gemma-4-31B/model.safetensors.index.json"))
wmap = idx["weight_map"]

# Check what weights exist for layer 5
layer5_keys = [k for k in sorted(wmap.keys()) if "layers.5." in k and "self_attn" in k]
print(f"\nLayer 5 (global) attn weights in checkpoint:")
for k in layer5_keys:
    print(f"  {k}")

# Check layer 0 for comparison
layer0_keys = [k for k in sorted(wmap.keys()) if "layers.0." in k and "self_attn" in k]
print(f"\nLayer 0 (sliding) attn weights in checkpoint:")
for k in layer0_keys:
    print(f"  {k}")

# Load actual weights for layer 5
print("\nLayer 5 weight shapes:")
for k in layer5_keys:
    sf_file = wmap[k]
    fpath = os.path.join("/tmp/torchtune/gemma-4-31B", sf_file)
    with safe_open(fpath, framework="pt") as f:
        t = f.get_tensor(k)
        print(f"  {k}: {t.shape}")

# Check: does layer 5 have v_proj?
has_v = any("v_proj" in k for k in layer5_keys)
has_k = any("k_proj" in k for k in layer5_keys)
print(f"\nLayer 5 has v_proj: {has_v}")
print(f"Layer 5 has k_proj: {has_k}")

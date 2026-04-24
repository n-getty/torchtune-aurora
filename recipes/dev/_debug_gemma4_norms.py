#!/usr/bin/env python3
"""Debug Gemma4 vLLM model: compare hidden states and check weights.

Loads the model on a single tile, runs a forward pass, and prints
intermediate values to diagnose the 'terterter' degenerate output.
"""
import sys
sys.path.insert(0, "/lus/flare/projects/ModCon/ngetty/torchtune/recipes/dev/vllm_gemma4_overlay")

import os
os.environ["ZE_AFFINITY_MASK"] = "0"

import json
import torch
from safetensors import safe_open


MODEL_PATH = "/tmp/torchtune/gemma-4-31B"

# 1. Check layer_scalar values from the checkpoint
print("=" * 60)
print("1. LAYER SCALAR VALUES FROM CHECKPOINT")
print("=" * 60)
idx = json.load(open(f"{MODEL_PATH}/model.safetensors.index.json"))
wmap = idx["weight_map"]

layer_scalar_keys = sorted([k for k in wmap if "layer_scalar" in k])
print(f"Found {len(layer_scalar_keys)} layer_scalar entries")
for k in layer_scalar_keys:
    sf_file = wmap[k]
    fpath = os.path.join(MODEL_PATH, sf_file)
    with safe_open(fpath, framework="pt") as f:
        t = f.get_tensor(k)
        print(f"  {k}: shape={list(t.shape)} value={t.item():.6f}")

# 2. Check norm weight statistics
print()
print("=" * 60)
print("2. NORM WEIGHT STATS (first 3 layers)")
print("=" * 60)
norm_names = ["input_layernorm", "post_attention_layernorm",
              "pre_feedforward_layernorm", "post_feedforward_layernorm"]
for layer_idx in range(3):
    for norm_name in norm_names:
        key = f"model.language_model.layers.{layer_idx}.{norm_name}.weight"
        if key not in wmap:
            key = f"model.language_model.model.layers.{layer_idx}.{norm_name}.weight"
        if key not in wmap:
            print(f"  Layer {layer_idx} {norm_name}: NOT FOUND")
            continue
        sf_file = wmap[key]
        fpath = os.path.join(MODEL_PATH, sf_file)
        with safe_open(fpath, framework="pt") as f:
            t = f.get_tensor(key)
            print(f"  Layer {layer_idx} {norm_name}: shape={list(t.shape)} "
                  f"mean={t.float().mean():.6f} std={t.float().std():.6f} "
                  f"min={t.float().min():.6f} max={t.float().max():.6f}")

# 3. Check q_norm and k_norm weight stats
print()
print("=" * 60)
print("3. Q_NORM / K_NORM WEIGHT STATS (layers 0, 5)")
print("=" * 60)
for layer_idx in [0, 5]:
    for norm_name in ["q_norm", "k_norm"]:
        key = f"model.language_model.layers.{layer_idx}.self_attn.{norm_name}.weight"
        if key not in wmap:
            print(f"  Layer {layer_idx} self_attn.{norm_name}: NOT FOUND")
            continue
        sf_file = wmap[key]
        fpath = os.path.join(MODEL_PATH, sf_file)
        with safe_open(fpath, framework="pt") as f:
            t = f.get_tensor(key)
            print(f"  Layer {layer_idx} self_attn.{norm_name}: shape={list(t.shape)} "
                  f"mean={t.float().mean():.6f} std={t.float().std():.6f}")

# 4. Check embedding weight stats
print()
print("=" * 60)
print("4. EMBEDDING AND FINAL NORM STATS")
print("=" * 60)
for key_pattern in ["embed_tokens.weight", "norm.weight"]:
    matching = [k for k in wmap if key_pattern in k]
    for key in matching:
        sf_file = wmap[key]
        fpath = os.path.join(MODEL_PATH, sf_file)
        with safe_open(fpath, framework="pt") as f:
            t = f.get_tensor(key)
            print(f"  {key}: shape={list(t.shape)} "
                  f"mean={t.float().mean():.6f} std={t.float().std():.6f}")

# 5. Check HF config for relevant parameters
print()
print("=" * 60)
print("5. CONFIG PARAMETERS")
print("=" * 60)
config = json.load(open(f"{MODEL_PATH}/config.json"))
text_cfg = config.get("text_config", config)
for key in ["hidden_size", "num_attention_heads", "num_key_value_heads",
            "num_global_key_value_heads", "head_dim", "global_head_dim",
            "intermediate_size", "vocab_size", "num_hidden_layers",
            "rms_norm_eps", "hidden_activation", "final_logit_softcapping",
            "tie_word_embeddings", "attention_k_eq_v", "sliding_window",
            "query_pre_attn_scalar"]:
    val = text_cfg.get(key, "NOT_SET")
    print(f"  {key}: {val}")

print()
print("=" * 60)
print("6. ROPE PARAMETERS")
print("=" * 60)
rope_params = text_cfg.get("rope_parameters", {})
print(json.dumps(rope_params, indent=2))

# 7. Compare vLLM GemmaRMSNorm with checkpoint values
print()
print("=" * 60)
print("7. VERIFY GemmaRMSNorm (1+w) BEHAVIOR")
print("=" * 60)
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
norm = GemmaRMSNorm(8, eps=1e-6)
# Set weights to known values (Gemma init: zeros)
with torch.no_grad():
    norm.weight.fill_(0.0)
x = torch.randn(2, 8)
out = norm(x)
# Manual: x_normed * (1 + 0) = x_normed
x_float = x.float()
x_normed = x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + 1e-6)
manual_out = x_normed  # (1+0) = 1
print(f"  Norm with w=0: match={torch.allclose(out.float(), manual_out, atol=1e-5)}")
print(f"  Max diff: {(out.float() - manual_out).abs().max():.8f}")

with torch.no_grad():
    norm.weight.fill_(0.5)
out2 = norm(x)
manual_out2 = x_normed * 1.5  # (1+0.5)
print(f"  Norm with w=0.5: match={torch.allclose(out2.float(), manual_out2, atol=1e-5)}")
print(f"  Max diff: {(out2.float() - manual_out2).abs().max():.8f}")

print()
print("Done!")

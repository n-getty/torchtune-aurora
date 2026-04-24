#!/usr/bin/env python3
"""Debug Gemma4 vLLM model forward pass.

Loads the model on 2 tiles (TP=2), does a single forward pass with known tokens,
and prints hidden states at key points to diagnose 'terterter' output.
"""
import sys
sys.path.insert(0, "/lus/flare/projects/ModCon/ngetty/torchtune/recipes/dev/vllm_gemma4_overlay")

import os
os.environ.setdefault("ZE_AFFINITY_MASK", "4")

import torch
import json

MODEL_PATH = "/tmp/torchtune/gemma-4-31B"

# Load tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

prompt = "The capital of France is"
input_ids = tokenizer.encode(prompt, add_special_tokens=True)
print(f"Prompt: {prompt!r}")
print(f"Token IDs: {input_ids}")
print(f"Decoded tokens: {[tokenizer.decode([t]) for t in input_ids]}")

# Load the model manually (non-vLLM, just raw weights)
from vllm_gemma4_config import Gemma4TextConfig
config = Gemma4TextConfig.from_pretrained(MODEL_PATH)
print(f"\nConfig: hidden={config.hidden_size}, heads={config.num_attention_heads}, "
      f"kv_heads={config.num_key_value_heads}, head_dim={config.head_dim}")

# Load model weights directly to check forward pass numerics
from safetensors import safe_open
idx = json.load(open(f"{MODEL_PATH}/model.safetensors.index.json"))
wmap = idx["weight_map"]

# 1. Test embedding + normalizer
print("\n" + "="*60)
print("1. EMBEDDING TEST")
print("="*60)

# Load embed_tokens weight
embed_key = "model.language_model.embed_tokens.weight"
sf_file = wmap[embed_key]
with safe_open(os.path.join(MODEL_PATH, sf_file), framework="pt") as f:
    embed_weight = f.get_tensor(embed_key)
print(f"Embed weight shape: {embed_weight.shape}")

# Embed input tokens
input_tensor = torch.tensor(input_ids)
embedded = embed_weight[input_tensor]  # (seq_len, hidden_size)
normalizer = config.hidden_size ** 0.5
embedded = embedded * normalizer
print(f"Embedded shape: {embedded.shape}")
print(f"Normalizer: {normalizer:.4f}")
print(f"Embedded stats: mean={embedded.float().mean():.4f}, std={embedded.float().std():.4f}, "
      f"min={embedded.float().min():.4f}, max={embedded.float().max():.4f}")
print(f"Embedded[0, :5]: {embedded[0, :5].float()}")

# 2. Test layer 0 input_layernorm
print("\n" + "="*60)
print("2. LAYER 0 INPUT_LAYERNORM TEST")
print("="*60)
norm_key = "model.language_model.layers.0.input_layernorm.weight"
sf_file = wmap[norm_key]
with safe_open(os.path.join(MODEL_PATH, sf_file), framework="pt") as f:
    norm_weight = f.get_tensor(norm_key)
print(f"Norm weight shape: {norm_weight.shape}")
print(f"Norm weight stats: mean={norm_weight.float().mean():.4f}, "
      f"std={norm_weight.float().std():.4f}")

# Apply GemmaRMSNorm manually: x_normed * (1 + weight)
x = embedded.float()
variance = x.pow(2).mean(dim=-1, keepdim=True)
x_normed = x * torch.rsqrt(variance + config.rms_norm_eps)
normed_output = x_normed * (1.0 + norm_weight.float())
print(f"After input_layernorm: mean={normed_output.mean():.4f}, "
      f"std={normed_output.std():.4f}, "
      f"min={normed_output.min():.4f}, max={normed_output.max():.4f}")
print(f"After norm[0, :5]: {normed_output[0, :5]}")

# 3. Check layer_scalar values
print("\n" + "="*60)
print("3. LAYER_SCALAR CUMULATIVE EFFECT")
print("="*60)
cumulative = 1.0
for layer_idx in range(60):
    key = f"model.language_model.layers.{layer_idx}.layer_scalar"
    sf_file = wmap[key]
    with safe_open(os.path.join(MODEL_PATH, sf_file), framework="pt") as f:
        scalar = f.get_tensor(key).item()
    cumulative *= scalar
    if layer_idx < 5 or layer_idx == 59:
        print(f"  Layer {layer_idx}: scalar={scalar:.6f}, cumulative={cumulative:.10e}")
print(f"  Cumulative product (all 60 layers): {cumulative:.10e}")

# 4. Final norm
print("\n" + "="*60)
print("4. FINAL NORM WEIGHT")
print("="*60)
final_norm_key = "model.language_model.norm.weight"
sf_file = wmap[final_norm_key]
with safe_open(os.path.join(MODEL_PATH, sf_file), framework="pt") as f:
    final_norm_weight = f.get_tensor(final_norm_key)
print(f"Final norm weight: shape={final_norm_weight.shape}, "
      f"mean={final_norm_weight.float().mean():.4f}, "
      f"std={final_norm_weight.float().std():.4f}")

# 5. Check if lm_head is tied to embed_tokens
print("\n" + "="*60)
print("5. LM HEAD / TIE_WORD_EMBEDDINGS")
print("="*60)
lm_head_keys = [k for k in wmap if "lm_head" in k]
print(f"lm_head keys in checkpoint: {lm_head_keys}")
print(f"tie_word_embeddings config: {config.tie_word_embeddings}")

# 6. Verify final_logit_softcapping
print("\n" + "="*60)
print("6. LOGIT SOFTCAPPING")
print("="*60)
print(f"final_logit_softcapping: {config.final_logit_softcapping}")
# Simulate: if raw logits are ~100, tanh(100/30)*30 = 30*tanh(3.33) ≈ 30*0.997 ≈ 29.9
# This clamps logits to [-30, 30] range
import math
for raw in [0.1, 1.0, 10.0, 50.0, 100.0]:
    capped = config.final_logit_softcapping * math.tanh(raw / config.final_logit_softcapping)
    print(f"  raw={raw:.1f} → capped={capped:.4f}")

print("\nDone!")

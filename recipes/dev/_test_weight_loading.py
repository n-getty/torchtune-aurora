#!/usr/bin/env python3
"""Test weight name remapping for Gemma4 vLLM model."""
import json
import sys

sys.path.insert(0, "/lus/flare/projects/ModCon/ngetty/torchtune/recipes/dev/vllm_gemma4_overlay")

# Get HF weight names from safetensors index
idx = json.load(open("/tmp/torchtune/gemma-4-31B/model.safetensors.index.json"))
hf_names = sorted(idx["weight_map"].keys())

# Prefix to strip
HF_PREFIX = "model.language_model."
skip_prefixes = [
    "model.vision_tower.", "model.multi_modal_projector.",
    "model.embed_vision.", "vision_tower.",
    "multi_modal_projector.", "embed_vision.",
    "lm_head.",
]

# Stacked params mapping
stacked_params_mapping = [
    ("qkv_proj", "q_proj", "q"),
    ("qkv_proj", "k_proj", "k"),
    ("qkv_proj", "v_proj", "v"),
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
]

# Simulate remapping
remapped_names = []
for name in hf_names:
    orig = name
    if name.startswith(HF_PREFIX):
        name = name[len(HF_PREFIX):]

    if any(name.startswith(p) for p in skip_prefixes):
        continue

    # Apply stacked params mapping
    final_name = name
    for (param_name, shard_name, shard_id) in stacked_params_mapping:
        if shard_name in name:
            final_name = name.replace(shard_name, param_name)
            break

    remapped_names.append((orig, name, final_name))

# Show first 20 and last 5
print("=== Weight name remapping (first 20) ===")
for orig, stripped, final in remapped_names[:20]:
    print(f"  HF: {orig}")
    print(f"  → stripped: {stripped}")
    print(f"  → final: {final}")
    print()

print(f"\n=== Total: {len(remapped_names)} weights (from {len(hf_names)} HF weights) ===")

# Count by type
layer_scalars = [n for _, n, _ in remapped_names if "layer_scalar" in n]
print(f"layer_scalar weights: {len(layer_scalars)}")
if layer_scalars:
    print(f"  example: {layer_scalars[0]}")

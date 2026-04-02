#!/usr/bin/env python
"""Per-rank wrapper that sets ZE_AFFINITY_MASK before importing torch.

When using torchrun (not mpiexec), each process sees all XPU tiles.
This wrapper sets ZE_AFFINITY_MASK=$LOCAL_RANK before any GPU runtime
initialization, so each process sees only its assigned tile as xpu:0.

Usage:
    torchrun --standalone --nproc_per_node=2 \
        recipes/dev/xpu_rank_wrapper.py \
        recipes/dev/grpo_full_finetune_distributed_xpu.py \
        --config recipes/configs/dev/qwen3B_grpo_xpu_baseline.yaml
"""
import os
import sys

# Set ZE_AFFINITY_MASK BEFORE importing torch
local_rank = os.environ.get("LOCAL_RANK", "0")
os.environ["ZE_AFFINITY_MASK"] = local_rank

# Remove this wrapper script from sys.argv so the recipe sees correct args
recipe_script = sys.argv[1]
sys.argv = sys.argv[1:]

# Now run the recipe script
with open(recipe_script) as f:
    code = compile(f.read(), recipe_script, "exec")
    exec(code, {"__name__": "__main__", "__file__": recipe_script})

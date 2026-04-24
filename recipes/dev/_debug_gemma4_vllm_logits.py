#!/usr/bin/env python3
"""Debug: load Gemma4 vLLM model on 1 tile, run forward, check logits."""
import sys
sys.path.insert(0, "/lus/flare/projects/ModCon/ngetty/torchtune/recipes/dev/vllm_gemma4_overlay")
import os
os.environ.setdefault("ZE_AFFINITY_MASK", "4")
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import numpy as np

MODEL_PATH = "/tmp/torchtune/gemma-4-31B"

# Use vLLM's LLM interface for a single generation
from vllm import LLM, SamplingParams

print("Loading model via vLLM (TP=1)...")
llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=1,
    enforce_eager=True,
    dtype="bfloat16",
    gpu_memory_utilization=0.80,
    max_model_len=256,
)
print("Model loaded!")

# Quick generation test
params = SamplingParams(
    max_tokens=32,
    temperature=0.0,
    logprobs=5,
)

prompts = [
    "The capital of France is",
    "2 + 2 =",
    "Hello, my name is",
]

outputs = llm.generate(prompts, params)
for out in outputs:
    print(f"\nPrompt: {out.prompt!r}")
    generated = out.outputs[0].text
    print(f"Generated: {generated!r}")
    # Show top logprobs for first few tokens
    if out.outputs[0].logprobs:
        for i, lp in enumerate(out.outputs[0].logprobs[:5]):
            top = sorted(lp.items(), key=lambda x: x[1].logprob, reverse=True)[:3]
            tok_strs = [(llm.get_tokenizer().decode([tid]), f"{v.logprob:.3f}")
                        for tid, v in top]
            print(f"  Token {i}: {tok_strs}")

print("\nDone!")

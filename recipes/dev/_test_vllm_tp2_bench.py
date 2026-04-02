"""Benchmark vLLM TP=2 generation throughput on XPU."""
import sys
import os
import time

# Set env vars BEFORE any vllm imports
os.environ["ZE_FLAT_DEVICE_HIERARCHY"] = "FLAT"
os.environ["CCL_PROCESS_LAUNCHER"] = "None"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
# Do NOT set VLLM_ENABLE_V1_MULTIPROCESSING — let V1 use subprocess mode
os.environ.pop("VLLM_ENABLE_V1_MULTIPROCESSING", None)

tp = int(sys.argv[1]) if len(sys.argv) > 1 else 2
model = sys.argv[2] if len(sys.argv) > 2 else "/lus/flare/projects/ModCon/ngetty/models/Qwen2.5-3B"

from vllm import LLM, SamplingParams

print(f"=== vLLM TP={tp} generation benchmark ===")

t0 = time.time()
llm = LLM(
    model=model,
    tensor_parallel_size=tp,
    dtype="bfloat16",
    enforce_eager=True,
    gpu_memory_utilization=0.5,
    max_model_len=512,
    distributed_executor_backend="mp" if tp > 1 else None,
)
print(f"LLM init: {time.time()-t0:.1f}s")

# Simulate GRPO Config B: 4 prompts, 16 completions each = 64 sequences
prompts = [
    "Solve: What is 15 * 23? Show your work step by step.",
    "Calculate the sum of all prime numbers less than 50.",
    "If a train travels 120 miles in 2 hours, what is its average speed?",
    "What is the derivative of x^3 + 2x^2 - 5x + 3?",
] * 4  # 16 prompts to simulate batched generation

for max_tokens in [256]:
    print(f"\n--- {len(prompts)} prompts, max_tokens={max_tokens} ---")
    params = SamplingParams(max_tokens=max_tokens, temperature=0.7, top_p=0.9)

    t0 = time.time()
    outputs = llm.generate(prompts, params)
    elapsed = time.time() - t0

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"Total tokens: {total_tokens}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Throughput: {total_tokens/elapsed:.1f} tok/s")
    avg_len = total_tokens / len(prompts)
    print(f"Avg response: {avg_len:.0f} tokens")

print("\n=== Benchmark complete ===")

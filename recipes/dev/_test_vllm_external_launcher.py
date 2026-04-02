"""Test 1: vLLM with external_launcher backend on XPU.

Tests that vLLM can initialize with distributed_executor_backend="external_launcher"
inside a torchrun-managed process. external_launcher means vLLM piggybacks on the
existing torch.distributed process group instead of spawning its own workers.

This is the foundation for colocated vLLM + FSDP training: same process runs both
vLLM inference and FSDP training, time-multiplexed.

Usage (single tile):
    torchrun --standalone --nproc_per_node=1 \
        recipes/dev/_test_vllm_external_launcher.py [model_path]

Usage (multi-tile, Test 3):
    torchrun --standalone --nproc_per_node=4 \
        recipes/dev/_test_vllm_external_launcher.py [model_path] --tp 4
"""
import argparse
import importlib.util
import os
import pathlib
import sys
import time
import types

# Must set before any torch imports
os.environ["ZE_FLAT_DEVICE_HIERARCHY"] = "FLAT"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
# external_launcher requires deterministic V1 — disable multiprocessing
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

# --- Patch transformers version check (huggingface-hub version mismatch) ---
_utils_stub = types.ModuleType("transformers.utils")
_utils_stub.__path__ = []
sys.modules["transformers.utils"] = _utils_stub

for p in sys.path:
    _vp = pathlib.Path(p) / "transformers" / "utils" / "versions.py"
    if _vp.exists():
        _spec = importlib.util.spec_from_file_location(
            "transformers.utils.versions", str(_vp)
        )
        _vm = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_vm)
        _vm._compare_versions = lambda *a, **kw: None
        sys.modules["transformers.utils.versions"] = _vm
        break

del sys.modules["transformers.utils"]

import torch
import torch.distributed as dist


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default="/tmp/torchtune/Qwen2.5-3B")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--gpu-mem", type=float, default=0.5,
                        help="GPU memory utilization for vLLM")
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--test-sleep", action="store_true",
                        help="Test sleep/wake_up cycle (may fail on XPU)")
    return parser.parse_args()


def main():
    args = parse_args()

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # Init torch.distributed — external_launcher will reuse this PG
    # Use xccl backend for XPU
    # NOTE: Do NOT pass device_id here. If device_id is set, PyTorch binds the
    # PG to XPU, then vLLM's new_group(backend="gloo") fails with
    # "No backend type associated with device type xpu".
    torch.xpu.set_device(local_rank)
    dist.init_process_group(backend="xccl")

    print(f"[rank {rank}] torch.distributed initialized: "
          f"world_size={world_size}, backend=xccl, device=xpu:{local_rank}")
    print(f"[rank {rank}] XPU device: {torch.xpu.get_device_name(local_rank)}")

    # Verify PG works
    test_tensor = torch.ones(4, device=f"xpu:{local_rank}") * (rank + 1)
    dist.all_reduce(test_tensor)
    expected = sum(range(1, world_size + 1))
    print(f"[rank {rank}] PG allreduce test: {test_tensor[0].item()} (expected {expected})")

    # Now create vLLM with external_launcher
    from vllm import LLM, SamplingParams

    print(f"[rank {rank}] Creating vLLM LLM with external_launcher "
          f"(model={args.model}, tp={args.tp})...")

    t0 = time.time()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        distributed_executor_backend="external_launcher",
        enforce_eager=True,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
    )
    init_time = time.time() - t0
    print(f"[rank {rank}] vLLM LLM created in {init_time:.1f}s")

    # Memory stats after init
    mem_alloc = torch.xpu.memory_allocated(local_rank) / 1e9
    mem_reserved = torch.xpu.memory_reserved(local_rank) / 1e9
    print(f"[rank {rank}] Memory after vLLM init: "
          f"{mem_alloc:.1f} GiB allocated, {mem_reserved:.1f} GiB reserved")

    # Test generation
    print(f"[rank {rank}] Testing generation...")
    params = SamplingParams(max_tokens=32, temperature=0.7)

    t0 = time.time()
    outputs = llm.generate(
        ["What is 2+2? Answer briefly:"],
        sampling_params=params,
        use_tqdm=False,
    )
    gen_time = time.time() - t0

    text = outputs[0].outputs[0].text.strip()
    n_tokens = len(outputs[0].outputs[0].token_ids)
    print(f"[rank {rank}] Generation: {n_tokens} tokens in {gen_time:.2f}s")
    print(f"[rank {rank}] Output: {text[:100]}")

    # Test batch generation (multiple prompts)
    prompts = [
        "The capital of France is",
        "Write a haiku about programming:",
        "What is machine learning?",
        "Count from 1 to 5:",
    ]

    t0 = time.time()
    outputs = llm.generate(
        prompts,
        sampling_params=SamplingParams(max_tokens=64, temperature=0.7),
        use_tqdm=False,
    )
    batch_time = time.time() - t0
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"[rank {rank}] Batch generation: {total_tokens} tokens from "
          f"{len(prompts)} prompts in {batch_time:.2f}s "
          f"({total_tokens/batch_time:.1f} tok/s)")

    # Test token_ids input (what the GRPO recipe uses)
    t0 = time.time()
    outputs = llm.generate(
        [{"prompt_token_ids": [1, 2, 3, 4, 5, 6, 7, 8]}],
        sampling_params=SamplingParams(max_tokens=32, temperature=0.7),
        use_tqdm=False,
    )
    print(f"[rank {rank}] Token-ids generation: "
          f"{len(outputs[0].outputs[0].token_ids)} tokens in {time.time()-t0:.2f}s")

    # Optional: Test sleep/wake_up
    if args.test_sleep:
        print(f"\n[rank {rank}] --- Testing sleep/wake_up ---")
        try:
            mem_before = torch.xpu.memory_allocated(local_rank) / 1e9
            print(f"[rank {rank}] Memory before sleep: {mem_before:.1f} GiB")

            llm.sleep(level=2)
            mem_after_sleep = torch.xpu.memory_allocated(local_rank) / 1e9
            print(f"[rank {rank}] Memory after sleep(2): {mem_after_sleep:.1f} GiB "
                  f"(freed {mem_before - mem_after_sleep:.1f} GiB)")

            llm.wake_up(tags=["weights"])
            llm.wake_up(tags=["kv_cache"])
            mem_after_wake = torch.xpu.memory_allocated(local_rank) / 1e9
            print(f"[rank {rank}] Memory after wake_up: {mem_after_wake:.1f} GiB")

            # Verify generation still works
            outputs = llm.generate(
                ["After sleep test:"],
                sampling_params=SamplingParams(max_tokens=16, temperature=0.7),
                use_tqdm=False,
            )
            print(f"[rank {rank}] Post-wake generation: "
                  f"{len(outputs[0].outputs[0].token_ids)} tokens OK")
        except Exception as e:
            print(f"[rank {rank}] Sleep/wake_up failed (expected on XPU): {e}")
            print(f"[rank {rank}] Sleep mode requires CuMemAllocator (CUDA-only)")

    # Verify torch.distributed still works after vLLM usage
    print(f"\n[rank {rank}] --- Verifying PG still works after vLLM ---")
    test_tensor = torch.ones(4, device=f"xpu:{local_rank}") * (rank + 1)
    dist.all_reduce(test_tensor)
    print(f"[rank {rank}] Post-vLLM allreduce: {test_tensor[0].item()} (expected {expected})")

    # Final memory stats
    mem_alloc = torch.xpu.memory_allocated(local_rank) / 1e9
    mem_reserved = torch.xpu.memory_reserved(local_rank) / 1e9
    print(f"[rank {rank}] Final memory: {mem_alloc:.1f} GiB allocated, "
          f"{mem_reserved:.1f} GiB reserved")

    # Cleanup
    del llm
    dist.destroy_process_group()
    print(f"[rank {rank}] SUCCESS - external_launcher vLLM works on XPU!")


if __name__ == "__main__":
    main()

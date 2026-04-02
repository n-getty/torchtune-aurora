"""Test 4: TP=4 subgroups with DP=3 using external_launcher.

12 ranks total, split into 3 independent TP=4 groups:
  Group 0: ranks [0,1,2,3]
  Group 1: ranks [4,5,6,7]
  Group 2: ranks [8,9,10,11]

Each group creates its own vLLM LLM engine with TP=4.
Each group generates different prompts (data parallelism).

Usage:
    torchrun --standalone --nproc_per_node=12 \
        recipes/dev/_test_vllm_tp_subgroups.py [model_path]
"""
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
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

# --- Patch transformers version check ---
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


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/lus/flare/projects/ModCon/ngetty/models/Qwen2.5-3B"
    tp_size = 4

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    assert world_size % tp_size == 0, f"world_size={world_size} must be divisible by tp_size={tp_size}"
    dp_size = world_size // tp_size
    dp_rank = rank // tp_size        # Which DP group (0, 1, or 2)
    tp_rank = rank % tp_size         # Position within TP group (0-3)

    torch.xpu.set_device(local_rank)

    # Init the global PG (all 12 ranks)
    dist.init_process_group(backend="xccl")

    print(f"[rank {rank}] Global PG: world={world_size}, dp_rank={dp_rank}, "
          f"tp_rank={tp_rank}, device=xpu:{local_rank}")

    # Verify global PG
    test = torch.ones(1, device=f"xpu:{local_rank}") * (rank + 1)
    dist.all_reduce(test)
    expected = sum(range(1, world_size + 1))
    print(f"[rank {rank}] Global allreduce: {test.item()} (expected {expected})")

    # --- Create TP subgroup for this rank's DP shard ---
    # Each DP group has tp_size ranks
    tp_group_ranks = list(range(dp_rank * tp_size, (dp_rank + 1) * tp_size))
    print(f"[rank {rank}] TP group ranks: {tp_group_ranks}")

    # Now we need to trick vLLM into thinking this TP subgroup is the world.
    # external_launcher reads RANK, WORLD_SIZE, LOCAL_RANK from env.
    # We need to override these so vLLM sees a world of tp_size ranks.

    # Save original env
    saved_env = {}
    for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        saved_env[key] = os.environ.get(key)

    # Override for vLLM: each TP group is a "world" of tp_size
    os.environ["WORLD_SIZE"] = str(tp_size)
    os.environ["RANK"] = str(tp_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)  # Keep real local_rank for device

    # Destroy global PG before creating vLLM's PG
    # vLLM's external_launcher expects to reuse an existing PG, but it needs
    # to be the right size (tp_size, not world_size).
    dist.destroy_process_group()

    # Create a TP-sized PG for this subgroup
    # Use a unique MASTER_PORT per DP group to avoid collisions
    original_port = os.environ.get("MASTER_PORT", "29500")
    os.environ["MASTER_PORT"] = str(int(original_port) + dp_rank)

    dist.init_process_group(
        backend="xccl",
        world_size=tp_size,
        rank=tp_rank,
    )
    print(f"[rank {rank}] TP subgroup PG initialized: size={dist.get_world_size()}")

    # Create vLLM LLM engine for this TP subgroup
    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        distributed_executor_backend="external_launcher",
        enforce_eager=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.5,
        max_model_len=512,
        max_num_seqs=16,
    )
    init_time = time.time() - t0
    print(f"[rank {rank}] vLLM LLM created (dp_rank={dp_rank}) in {init_time:.1f}s")

    # Each DP group generates different prompts
    prompt_sets = [
        ["What is 2+2?", "Capital of France?"],
        ["What is 3+3?", "Capital of Germany?"],
        ["What is 4+4?", "Capital of Japan?"],
    ]
    prompts = prompt_sets[dp_rank] if dp_rank < len(prompt_sets) else prompt_sets[0]

    t0 = time.time()
    outputs = llm.generate(
        prompts,
        sampling_params=SamplingParams(max_tokens=32, temperature=0.7),
        use_tqdm=False,
    )
    gen_time = time.time() - t0
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

    # Only tp_rank=0 of each group prints full results
    if tp_rank == 0:
        print(f"[rank {rank}] DP group {dp_rank}: {total_tokens} tokens from "
              f"{len(prompts)} prompts in {gen_time:.2f}s ({total_tokens/gen_time:.1f} tok/s)")
        for i, out in enumerate(outputs):
            print(f"[rank {rank}]   [{i}] {out.outputs[0].text.strip()[:80]}")

    # Memory stats
    mem_alloc = torch.xpu.memory_allocated(local_rank) / 1e9
    print(f"[rank {rank}] Memory: {mem_alloc:.1f} GiB allocated")

    # Cleanup vLLM
    del llm

    # Destroy TP subgroup PG
    dist.destroy_process_group()

    # Restore env and recreate global PG for final verification
    for key, val in saved_env.items():
        if val is not None:
            os.environ[key] = val
    os.environ["MASTER_PORT"] = original_port

    dist.init_process_group(backend="xccl")
    torch.xpu.set_device(local_rank)

    # Verify global PG still works
    test = torch.ones(1, device=f"xpu:{local_rank}") * (rank + 1)
    dist.all_reduce(test)
    print(f"[rank {rank}] Post-vLLM global allreduce: {test.item()} (expected {expected})")

    dist.destroy_process_group()
    print(f"[rank {rank}] SUCCESS - TP subgroups work!")


if __name__ == "__main__":
    main()

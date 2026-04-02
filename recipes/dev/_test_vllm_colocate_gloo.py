"""Test colocated vLLM init using pre-initialized gloo PG inside torchrun.

This script is designed to be run with torchrun (2+ ranks).
Only rank 0 creates a vLLM LLM engine using the gloo PG trick:
1. Pre-init a gloo PG (world_size=1) with file:// store
2. Monkey-patch new_group to force gloo backend
3. Monkey-patch all_reduce to skip XPU tensor ops on gloo groups (warmup)
4. Create LLM(tp=1) — vLLM sees is_initialized()=True, skips its own init
5. Destroy the gloo PG, restore env, init the real XCCL PG

Usage:
    python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
        recipes/dev/_test_vllm_colocate_gloo.py [model_path]
"""
import os
import sys
import time
import tempfile

rank = int(os.environ.get("RANK", "0"))
local_rank = int(os.environ.get("LOCAL_RANK", "0"))

# XPU env
os.environ["ZE_FLAT_DEVICE_HIERARCHY"] = "FLAT"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import torch

model_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/torchtune/Qwen2.5-3B"

print(f"[rank {rank}] Starting colocated vLLM test")

if rank == 0:
    print(f"[rank 0] Initializing vLLM with gloo PG trick...")

    # Save torchrun env vars
    saved_env = {}
    for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "GROUP_RANK",
                "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                "TORCHELASTIC_RUN_ID"):
        saved_env[key] = os.environ.pop(key, None)

    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29599"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    # Pre-init gloo PG with file store
    store_file = tempfile.mktemp(prefix="vllm_gloo_store_")
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{store_file}",
        world_size=1,
        rank=0,
    )
    print(f"[rank 0] Gloo PG initialized")

    # Monkey-patch new_group to force gloo backend
    _orig_new_group = torch.distributed.new_group
    def _gloo_new_group(*args, **kwargs):
        kwargs["backend"] = "gloo"
        return _orig_new_group(*args, **kwargs)
    torch.distributed.new_group = _gloo_new_group

    # Monkey-patch all_reduce to handle XPU tensors on gloo groups.
    # vLLM's xpu_worker does all_reduce(torch.zeros(1).xpu()) as a warmup.
    # With gloo backend this fails. For world_size=1, allreduce is a no-op.
    _orig_all_reduce = torch.distributed.all_reduce
    def _safe_all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=None, async_op=False):
        # For world_size=1, allreduce is a no-op
        if group is not None and group.size() == 1:
            return None
        if tensor.is_xpu:
            # Can't do allreduce on XPU tensor with gloo backend
            return None
        return _orig_all_reduce(tensor, op=op, group=group, async_op=async_op)
    torch.distributed.all_reduce = _safe_all_reduce

    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.3,
        max_model_len=512,
        max_num_seqs=16,
        enforce_eager=True,
        dtype="bfloat16",
        disable_custom_all_reduce=True,
    )
    print(f"[rank 0] vLLM LLM created in {time.time()-t0:.1f}s")

    # Test generation
    params = SamplingParams(max_tokens=32, temperature=0.7)
    outputs = llm.generate(
        [{"prompt_token_ids": [1, 2, 3, 4, 5]}],
        sampling_params=params,
        use_tqdm=False,
    )
    print(f"[rank 0] Generation test: {len(outputs[0].outputs[0].token_ids)} tokens")

    # Restore all monkey-patches
    torch.distributed.new_group = _orig_new_group
    torch.distributed.all_reduce = _orig_all_reduce

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    # Clear vLLM's cached world
    import vllm.distributed.parallel_state as vllm_ps
    vllm_ps._WORLD = None

    try:
        os.unlink(store_file)
    except OSError:
        pass

    # Restore torchrun env
    for key, val in saved_env.items():
        if val is not None:
            os.environ[key] = val
        elif key in os.environ:
            del os.environ[key]

    print(f"[rank 0] vLLM init complete, PG destroyed, ready for training PG")

# Barrier using a simple file-based sync (no PG yet)
if rank == 0:
    with open("/tmp/torchtune/vllm_init_done", "w") as f:
        f.write("done")
else:
    print(f"[rank {rank}] Waiting for rank 0 to finish vLLM init...")
    while not os.path.exists("/tmp/torchtune/vllm_init_done"):
        time.sleep(0.5)
    print(f"[rank {rank}] Rank 0 vLLM init done, proceeding")

# Now init the real training XCCL PG
print(f"[rank {rank}] Initializing training XCCL PG...")
torch.distributed.init_process_group(backend="xccl")
print(f"[rank {rank}] Training PG initialized: world_size={torch.distributed.get_world_size()}")

# Test a simple allreduce
device = torch.device(f"xpu:{local_rank}")
torch.xpu.set_device(device)
tensor = torch.ones(4, device=device) * (rank + 1)
torch.distributed.all_reduce(tensor)
print(f"[rank {rank}] allreduce result: {tensor.tolist()} (expected [3.0, 3.0, 3.0, 3.0])")

# Broadcast test (simulating generation broadcast)
if rank == 0:
    data = torch.tensor([42, 99, 7, 13], device=device)
else:
    data = torch.zeros(4, dtype=torch.long, device=device)
torch.distributed.broadcast(data, src=0)
print(f"[rank {rank}] broadcast result: {data.tolist()} (expected [42, 99, 7, 13])")

# Cleanup
if rank == 0:
    try:
        os.unlink("/tmp/torchtune/vllm_init_done")
    except OSError:
        pass

torch.distributed.destroy_process_group()
print(f"[rank {rank}] SUCCESS - colocated vLLM + training PG both work!")

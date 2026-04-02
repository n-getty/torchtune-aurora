"""Test all-rank colocated vLLM init using gloo PG trick inside torchrun.

Every rank creates its own vLLM LLM engine on its own XPU tile, then all
ranks init the real XCCL training PG and do an allgather.

Usage:
    python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
        recipes/dev/_test_vllm_allrank_gloo.py [model_path]
"""
import os
import sys
import time
import tempfile

rank = int(os.environ.get("RANK", "0"))
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))

# XPU env
os.environ["ZE_FLAT_DEVICE_HIERARCHY"] = "FLAT"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import torch

model_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/torchtune/Qwen2.5-3B"

print(f"[rank {rank}] Starting all-rank vLLM test (world_size={world_size}, local_rank={local_rank})")

# --- Sequential vLLM init with file-based barriers ---
run_id = os.environ.get("TORCHELASTIC_RUN_ID", str(os.getpid()))
barrier_dir = f"/tmp/torchtune/vllm_init_barriers_{run_id}"
os.makedirs(barrier_dir, exist_ok=True)

# Wait for previous rank
if rank > 0:
    prev_barrier = os.path.join(barrier_dir, f"rank_{rank - 1}_done")
    print(f"[rank {rank}] Waiting for rank {rank - 1}...")
    while not os.path.exists(prev_barrier):
        time.sleep(0.5)
    print(f"[rank {rank}] Rank {rank - 1} done, proceeding")

print(f"[rank {rank}] Initializing vLLM with gloo PG trick on xpu:{local_rank}...")

# Save torchrun env vars
saved_env = {}
for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "GROUP_RANK",
            "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
            "TORCHELASTIC_RUN_ID"):
    saved_env[key] = os.environ.pop(key, None)

os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = str(local_rank)  # Use real tile index
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = str(29599 + rank)
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

# Pre-init gloo PG with file store
store_file = tempfile.mktemp(prefix=f"vllm_gloo_store_r{rank}_")
torch.distributed.init_process_group(
    backend="gloo",
    init_method=f"file://{store_file}",
    world_size=1,
    rank=0,
)

# Monkey-patches
_orig_new_group = torch.distributed.new_group
def _gloo_new_group(*args, **kwargs):
    kwargs["backend"] = "gloo"
    return _orig_new_group(*args, **kwargs)
torch.distributed.new_group = _gloo_new_group

_orig_all_reduce = torch.distributed.all_reduce
def _safe_all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=None, async_op=False):
    if group is not None and group.size() == 1:
        return None
    if tensor.is_xpu:
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
    max_num_seqs=4,
    enforce_eager=True,
    dtype="bfloat16",
    disable_custom_all_reduce=True,
)
print(f"[rank {rank}] vLLM LLM created on xpu:{local_rank} in {time.time()-t0:.1f}s")

# Skip pre-XCCL generation test — vLLM's prepare_inputs_event gets
# invalidated when XCCL init_process_group changes the device context.
# In the real recipe, first generation happens AFTER XCCL init, so it's fine.
print(f"[rank {rank}] vLLM engine created (skipping pre-XCCL gen test)")

# Restore patches
torch.distributed.new_group = _orig_new_group
torch.distributed.all_reduce = _orig_all_reduce

if torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()

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

# Re-set XPU device
torch.xpu.set_device(local_rank)

print(f"[rank {rank}] vLLM init complete on xpu:{local_rank}")

# Signal next rank
my_barrier = os.path.join(barrier_dir, f"rank_{rank}_done")
with open(my_barrier, "w") as f:
    f.write("done")

# Last rank cleans up
if rank == world_size - 1:
    time.sleep(0.5)
    for i in range(world_size):
        try:
            os.unlink(os.path.join(barrier_dir, f"rank_{i}_done"))
        except OSError:
            pass

# --- Now init the real training XCCL PG ---
print(f"[rank {rank}] Initializing training XCCL PG...")
torch.distributed.init_process_group(
    backend="xccl",
    device_id=torch.device(f"xpu:{local_rank}"),
)
# Re-set device after XCCL init (it may change the default)
torch.xpu.set_device(local_rank)
print(f"[rank {rank}] Training PG: world_size={torch.distributed.get_world_size()}")

# Test allgather (simulates distributed generation)
device = torch.device(f"xpu:{local_rank}")
local_data = torch.tensor([rank * 10 + 1, rank * 10 + 2, rank * 10 + 3, rank * 10 + 4], device=device)
gathered = [torch.zeros_like(local_data) for _ in range(world_size)]
torch.distributed.all_gather(gathered, local_data)
print(f"[rank {rank}] allgather result: {[t.tolist() for t in gathered]}")

# Verify each rank's vLLM engine is still functional
params2 = SamplingParams(max_tokens=16, temperature=0.7)
outputs2 = llm.generate(
    [{"prompt_token_ids": [10, 20, 30]}],
    sampling_params=params2,
    use_tqdm=False,
)
print(f"[rank {rank}] Post-PG generation test: {len(outputs2[0].outputs[0].token_ids)} tokens")

# Cleanup
torch.distributed.destroy_process_group()
print(f"[rank {rank}] SUCCESS - all-rank colocated vLLM + training PG both work!")

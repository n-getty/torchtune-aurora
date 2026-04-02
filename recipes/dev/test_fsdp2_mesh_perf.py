"""Quick test: compare FSDP2 forward pass time with 1D vs 2D mesh on same 10 ranks.
Run with: torchrun --standalone --nproc_per_node=10 recipes/dev/test_fsdp2_mesh_perf.py
"""
import os, sys, time, types
# Pre-register torchtune to bypass __init__.py
if "torchtune" not in sys.modules:
    sys.modules["torchtune"] = types.ModuleType("torchtune")
    sys.modules["torchtune"].__path__ = [os.path.join(os.path.dirname(__file__), "..", "..", "torchtune")]

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

# Init
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.xpu.set_device(local_rank)
device = torch.device(f"xpu:{local_rank}")
dist.init_process_group("xccl", timeout=__import__("datetime").timedelta(minutes=5))
rank = dist.get_rank()
world_size = dist.get_world_size()

# Load model on meta device
from torchtune.models.qwen3 import qwen3_32b
from torchtune.training._distributed import shard_model
from torchtune import training

if rank == 0:
    print(f"World size: {world_size}, testing 1D vs 2D mesh forward pass")

# Monkey-patch FSDP2 for XPU (ReduceOp.AVG)
try:
    import torch.distributed.fsdp._fully_shard._fsdp_collectives as _fsdp_coll
    def _patched_get_gradient_divide_factors(world_size):
        return (world_size, 1)
    _fsdp_coll._get_gradient_divide_factors = _patched_get_gradient_divide_factors
except Exception:
    pass

# Force math-only SDPA
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

def test_forward(mesh_label, dp_mesh):
    """Time a single forward pass with the given mesh."""
    with training.set_default_dtype(torch.bfloat16), torch.device("meta"):
        model = qwen3_32b()
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Enable activation checkpointing
    from torchtune.modules import TransformerSelfAttentionLayer
    training.set_activation_checkpointing(model, auto_wrap_policy={TransformerSelfAttentionLayer})

    # Shard model
    from functools import partial
    shard_conditions = [partial(training.get_shard_conditions, names_to_match=None)]
    shard_model(model=model, shard_conditions=shard_conditions, cpu_offload=False,
                reshard_after_forward=True, dp_mesh=dp_mesh)

    # RoPE init
    with training.set_default_dtype(torch.bfloat16), device:
        for m in model.modules():
            if hasattr(m, "rope_init"):
                m.rope_init()

    # Load checkpoint into sharded model
    if rank == 0:
        print(f"  [{mesh_label}] Model sharded, loading checkpoint...")
    t0 = time.perf_counter()
    ckpt_path = "/tmp/torchtune/Qwen3-32B"
    from torchtune.training import FullModelHFCheckpointer
    checkpointer = FullModelHFCheckpointer(
        checkpoint_dir=ckpt_path,
        checkpoint_files=[f"model-{i:05d}-of-00017.safetensors" for i in range(1, 18)],
        output_dir="/tmp/torchtune/test_mesh_perf",
        model_type="QWEN3",
    )
    ckpt = checkpointer.load_checkpoint()
    training.load_from_full_model_state_dict(model, ckpt["model"], device, strict=True, cpu_offload=False)
    training.validate_no_params_on_meta_device(model)
    dist.barrier()
    if rank == 0:
        print(f"  [{mesh_label}] Checkpoint loaded in {time.perf_counter() - t0:.1f}s")

    # Create dummy input (1 sequence, 512 tokens)
    input_ids = torch.randint(0, 151936, (1, 512), device=device)
    position_ids = torch.arange(512, device=device).unsqueeze(0)

    # Warmup
    with torch.no_grad():
        _ = model(input_ids, input_pos=position_ids)
    torch.xpu.synchronize()
    dist.barrier()

    # Timed forward
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(input_ids, input_pos=position_ids)
    torch.xpu.synchronize()
    fwd_time = time.perf_counter() - t0

    if rank == 0:
        print(f"  [{mesh_label}] Forward pass: {fwd_time:.2f}s")

    # Cleanup
    del model, ckpt
    import gc; gc.collect()
    torch.xpu.synchronize()
    dist.barrier()
    return fwd_time

# Test 1: 1D mesh (flat FSDP, mesh=None)
if rank == 0:
    print("\n=== Test 1: 1D mesh (flat FSDP, mesh=None) ===")
t1 = test_forward("1D", dp_mesh=None)

# Test 2: 2D mesh (HSDP with dp_replicate=1 — same ranks, 2D structure)
if rank == 0:
    print("\n=== Test 2: 2D mesh (1x10, simulated HSDP) ===")
mesh_2d = init_device_mesh("xpu", (1, world_size), mesh_dim_names=("dp_replicate", "dp_shard"))
t2 = test_forward("2D(1x10)", dp_mesh=mesh_2d)

if rank == 0:
    print(f"\n=== Results ===")
    print(f"  1D mesh: {t1:.2f}s")
    print(f"  2D mesh: {t2:.2f}s")
    print(f"  Ratio: {t2/t1:.1f}x")

dist.destroy_process_group()

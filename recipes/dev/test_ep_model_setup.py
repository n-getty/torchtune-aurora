"""
Test EP=4/DP=3 model setup: load Gemma4 26B-A4B on meta device, apply
ExpertParallel + FSDP2 sharding on 12 tiles, and run one dummy forward pass.

Run with:
    torchrun --nproc_per_node=12 recipes/dev/test_ep_model_setup.py

Order of operations (CRITICAL):
  1. Build model on meta device
  2. Apply EP hooks (parallelize_module) — stores _ep_device_mesh on GroupedExperts modules
  3. to_empty(device=xpu) + load_state_dict → loads full checkpoint on each rank
  4. apply_ep_weight_sharding → slices expert params to local EP shard (BEFORE FSDP2)
  5. shard_model() → FSDP2 wrapping (parameters are now local-shard sized)
  6. Run forward pass

NOTE: Each rank loads the full 52 GiB checkpoint independently onto its XPU tile.
12 ranks × 52 GiB = 624 GiB total XPU, fits within 12 × 64 GiB = 768 GiB available.
After apply_ep_weight_sharding, expert params shrink 4× before FSDP2 shards further.
"""
import os
import sys
import time
import types as _types
import importlib.util as _imp_util

# XPU/XCCL compatibility shim: pre-register torchtune to bypass __init__.py
# which imports torchao and corrupts XCCL's USM pointer table on Aurora.
# Must happen before any torch or torchtune import.
if "torchtune" not in sys.modules:
    _spec = _imp_util.find_spec("torchtune")
    if _spec is not None and _spec.submodule_search_locations:
        _torchtune_path = list(_spec.submodule_search_locations)[0]
    else:
        _torchtune_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "torchtune",
        )
    if os.path.isdir(_torchtune_path):
        _pkg = _types.ModuleType("torchtune")
        _pkg.__path__ = [_torchtune_path]
        _pkg.__file__ = os.path.join(_torchtune_path, "__init__.py")
        _pkg.__version__ = ""
        sys.modules["torchtune"] = _pkg

import torch
import torch.distributed as dist
from torch import nn


def setup_dist():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if torch.xpu.is_available():
        # DO NOT set ZE_AFFINITY_MASK for single-node multi-rank training:
        # CCL needs all device UUIDs visible for topology-aware routing.
        # Use device_id= in init_process_group instead.
        import intel_extension_for_pytorch  # noqa: F401
        torch.xpu.set_device(local_rank)
        device = torch.device(f"xpu:{local_rank}")
        dist.init_process_group(
            backend="xccl",
            device_id=torch.device(f"xpu:{local_rank}"),
        )
    else:
        device = torch.device("cpu")
        dist.init_process_group(backend="gloo")

    return device, rank, world_size


def log0(rank, msg):
    if rank == 0:
        print(f"[Rank 0] {msg}", flush=True)


def main():
    device, rank, world_size = setup_dist()
    assert world_size == 12, f"Requires 12 ranks, got {world_size}"

    # Topology: EP=4, DP=3
    dp_replicate = 3
    dp_shard = 4
    assert dp_replicate * dp_shard == world_size

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor.parallel import parallelize_module
    import torchtune.training as training
    from torchtune.models.gemma4 import gemma4_26b_a4b
    from torchtune.models.gemma4._parallelism import gemma4_ep_plan
    from torchtune.modules.moe._parallelism import apply_ep_weight_sharding
    from torchtune.training import shard_model, get_shard_conditions
    from functools import partial

    log0(rank, f"Building 2D mesh: dp_replicate={dp_replicate} × dp_shard={dp_shard}")
    dp_mesh = init_device_mesh(
        device.type,
        (dp_replicate, dp_shard),
        mesh_dim_names=("dp_replicate", "dp_shard"),
    )
    ep_mesh = dp_mesh["dp_shard"]

    # Force eager creation of all process groups before any collectives.
    # XCCL creates submesh groups lazily; deferring to first use in the forward
    # hook can cause a global barrier that deadlocks with EP-only collectives.
    # Calling get_group/get_all_groups() here triggers XCCL group initialization synchronously.
    _ = ep_mesh.get_group()            # ep_mesh is 1D (dp_shard submesh)
    _ = dp_mesh.get_all_groups()       # dp_mesh is 2D — use get_all_groups()
    dist.barrier()
    log0(rank, "All process groups initialized")

    log0(rank, "Instantiating Gemma4 26B-A4B on meta device...")
    t0 = time.perf_counter()
    with training.set_default_dtype(torch.bfloat16), torch.device("meta"):
        model = gemma4_26b_a4b()
    log0(rank, f"Model instantiated on meta in {time.perf_counter() - t0:.1f}s")

    # Apply EP hooks BEFORE loading checkpoint
    log0(rank, "Applying gemma4_ep_plan + ExpertParallel...")
    ep_plan = gemma4_ep_plan(model)
    if ep_plan:
        parallelize_module(model, ep_mesh, ep_plan)
        log0(rank, f"ExpertParallel applied to {len(ep_plan)} modules")

    # Stage checkpoint to /tmp (local NVMe) if not already present.
    # Rank 0 does the copy; all ranks barrier-wait, then load from /tmp.
    # This avoids 12-rank simultaneous Lustre I/O (357s) — /tmp gives ~62s.
    TMP_PATH = "/tmp/torchtune/gemma-4-26B-A4B"
    LUSTRE_PATH = "/lus/flare/projects/ModCon/ngetty/models/gemma-4-26B-A4B"
    CKPT_FILES = [
        "config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    if rank == 0:
        import shutil
        os.makedirs(TMP_PATH, exist_ok=True)
        for fname in CKPT_FILES:
            dst = os.path.join(TMP_PATH, fname)
            if not os.path.exists(dst):
                print(f"[Rank 0] Staging {fname} to /tmp...", flush=True)
                shutil.copy2(os.path.join(LUSTRE_PATH, fname), dst)
        print("[Rank 0] Staging complete", flush=True)
    dist.barrier()  # all ranks wait for rank 0 to finish staging

    MODEL_PATH = TMP_PATH
    log0(rank, f"Loading checkpoint from {MODEL_PATH} (pre-FSDP2, direct to XPU)...")
    t1 = time.perf_counter()

    from torchtune.training import FullModelHFCheckpointer
    checkpointer = FullModelHFCheckpointer(
        checkpoint_dir=MODEL_PATH,
        checkpoint_files=["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"],
        recipe_checkpoint=None,
        output_dir="/tmp/torchtune/ep4_test",
        model_type="GEMMA4",
    )
    # All ranks load from /tmp independently (each has its own XPU tile).
    # Parallel reads from local NVMe are fast and avoid broadcast overhead.
    sd = checkpointer.load_checkpoint()

    # Materialize model on XPU (from meta device)
    model.to_empty(device=device)

    # Initialize RoPE buffers (computed, not in checkpoint)
    with training.set_default_dtype(torch.bfloat16), device:
        for m in model.modules():
            if hasattr(m, "rope_init"):
                m.rope_init()

    # Load full state dict onto XPU
    missing, unexpected = model.load_state_dict(sd["model"], strict=False)
    if rank == 0:
        print(f"[Rank 0] load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected", flush=True)
        if missing:
            print(f"[Rank 0]   missing[:5]: {missing[:5]}", flush=True)

    del sd  # free CPU memory from loaded checkpoint

    log0(rank, f"Checkpoint loaded in {time.perf_counter() - t1:.1f}s")

    if torch.xpu.is_available():
        mem = torch.xpu.memory_allocated(device) / 1024**3
        log0(rank, f"Peak memory after load (pre-slice): {mem:.2f} GiB")

    # Slice expert weights to local EP shards BEFORE any FSDP2 wrapping
    # At this point params are plain tensors (not DTensors), so slice works directly
    log0(rank, "Applying EP weight sharding (pre-FSDP2)...")
    n_ep_sharded = apply_ep_weight_sharding(model)
    log0(rank, f"EP weight sharding applied to {n_ep_sharded} expert modules")

    if torch.xpu.is_available():
        mem = torch.xpu.memory_allocated(device) / 1024**3
        log0(rank, f"Peak memory after EP slice: {mem:.2f} GiB")

    # Apply FSDP2 with disable_prefetch=True to prevent AllGather/AllToAll overlap.
    # Uses the full 2D dp_mesh (dp_replicate=3, dp_shard=4) so FSDP2 shards
    # non-expert params across dp_shard while EP all-to-all runs within dp_shard.
    # disable_prefetch=True ensures AllGather prefetch doesn't race with EP all-to-all.
    log0(rank, "Applying FSDP2 (2D mesh, disable_prefetch=True)...")
    shard_model(
        model=model,
        shard_conditions=[partial(get_shard_conditions, names_to_match=None)],
        cpu_offload=False,
        reshard_after_forward=True,
        dp_mesh=dp_mesh,
        disable_prefetch=True,
    )
    dist.barrier()
    log0(rank, "FSDP2 sharding complete")

    # Run dummy forward pass
    log0(rank, "Running dummy forward pass...")
    bs, slen = 1, 512
    tokens = torch.zeros(bs, slen, dtype=torch.long, device=device)

    t2 = time.perf_counter()
    with torch.no_grad():
        out = model(tokens)
    dist.barrier()
    log0(rank, f"Forward pass OK in {time.perf_counter() - t2:.2f}s, out shape: {out.shape}")

    if torch.xpu.is_available():
        mem_peak = torch.xpu.memory_allocated(device) / 1024**3
        log0(rank, f"Peak memory after forward: {mem_peak:.2f} GiB")

    dist.destroy_process_group()
    log0(rank, "=== EP=4/DP=3 Model Setup Test: PASS ===")


if __name__ == "__main__":
    main()

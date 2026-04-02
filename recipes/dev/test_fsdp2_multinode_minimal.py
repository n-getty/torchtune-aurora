"""Minimal FSDP2 multi-node test to isolate CCL sub-communicator performance.

Tests whether FSDP2 AllGather on an intra-node shard sub-group runs at full
intra-node bandwidth when the world PG spans multiple nodes.

This script has NO dependencies on torchtune, vLLM, or any RL logic.
It creates a simple transformer-like model with configurable size, wraps it
with FSDP2, and measures forward+backward time with different mesh configs.

Run single-node (10 tiles):
  torchrun --standalone --nproc_per_node=10 recipes/dev/test_fsdp2_multinode_minimal.py

Run multi-node (2 nodes, 10 tiles each):
  mpiexec -n 20 -ppn 10 --hostfile $PBS_NODEFILE \\
      bash recipes/dev/test_fsdp2_multinode_wrapper.sh \\
      recipes/dev/test_fsdp2_multinode_minimal.py

Environment variables:
  MODEL_SIZE: "small" (16 linears, ~1.6 GiB), "medium" (~10 GiB), "3B" (Qwen 3B)
              Default: "small"
  NUM_LAYERS: Override number of transformer layers (default: model-size dependent)
  HIDDEN_DIM: Override hidden dimension (default: model-size dependent)
  SEQ_LEN:    Sequence length for input (default: 512)
  BATCH_SIZE:  Batch size for input (default: 1)
  NUM_ITERS:  Number of timed iterations (default: 5)
  WARMUP:     Number of warmup iterations (default: 2)
  TEST_BACKWARD: "1" to also test backward pass (default: "0")
  SKIP_1D:    "1" to skip 1D mesh test (default: "0")
"""
import os
import sys
import time
import datetime
import gc

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard


# ============================================================
# Configuration from env
# ============================================================
MODEL_SIZE = os.environ.get("MODEL_SIZE", "small")
SEQ_LEN = int(os.environ.get("SEQ_LEN", "512"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
NUM_ITERS = int(os.environ.get("NUM_ITERS", "5"))
WARMUP = int(os.environ.get("WARMUP", "2"))
TEST_BACKWARD = os.environ.get("TEST_BACKWARD", "0") == "1"
SKIP_1D = os.environ.get("SKIP_1D", "0") == "1"

# Model size presets
PRESETS = {
    "small": {"num_layers": 16, "hidden_dim": 4096, "ffn_mult": 4},    # ~1.6 GiB
    "medium": {"num_layers": 32, "hidden_dim": 4096, "ffn_mult": 4},   # ~3.2 GiB
    "large": {"num_layers": 64, "hidden_dim": 5120, "ffn_mult": 4},    # ~13 GiB (32B-like layer count)
}

preset = PRESETS.get(MODEL_SIZE, PRESETS["small"])
NUM_LAYERS = int(os.environ.get("NUM_LAYERS", str(preset["num_layers"])))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", str(preset["hidden_dim"])))
FFN_MULT = preset["ffn_mult"]


# ============================================================
# Simple transformer-like model (no attention, just FFN blocks)
# ============================================================
class FFNBlock(nn.Module):
    """Simplified transformer layer: LayerNorm -> Linear -> GELU -> Linear -> residual."""
    def __init__(self, dim, ffn_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.up = nn.Linear(dim, ffn_dim, bias=False)
        self.gate = nn.Linear(dim, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x):
        h = self.norm(x)
        return x + self.down(nn.functional.silu(self.gate(h)) * self.up(h))


class SimpleModel(nn.Module):
    """Stack of FFN blocks — mimics transformer parameter/communication patterns."""
    def __init__(self, num_layers, hidden_dim, ffn_mult):
        super().__init__()
        ffn_dim = hidden_dim * ffn_mult
        self.layers = nn.ModuleList([
            FFNBlock(hidden_dim, ffn_dim) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ============================================================
# FSDP2 monkey-patch for XPU (ReduceOp.AVG not supported)
# ============================================================
def patch_fsdp2_for_xpu():
    """Patch FSDP2 to avoid ReduceOp.AVG which XCCL doesn't support."""
    try:
        import torch.distributed.fsdp._fully_shard._fsdp_collectives as _fsdp_coll
        _orig = _fsdp_coll._get_gradient_divide_factors

        import inspect
        sig = inspect.signature(_orig)
        num_params = len(sig.parameters)

        if num_params == 1:
            # Old signature: (world_size) -> (world_size, 1)
            _fsdp_coll._get_gradient_divide_factors = lambda ws: (ws, 1)
        else:
            # New 6-arg signature: force_sum_reduction_for_comms is the 6th positional arg
            def _patched(rs_group, ar_group, reduce_dtype, device_type='',
                         factor=None, force_sum_reduction_for_comms=False):
                return _orig(rs_group, ar_group, reduce_dtype, device_type,
                             factor, True)
            _fsdp_coll._get_gradient_divide_factors = _patched
    except Exception as e:
        if dist.get_rank() == 0:
            print(f"WARNING: Could not patch FSDP2 for XPU: {e}")


# ============================================================
# Main
# ============================================================
def main():
    # If CCL_ATL_TRANSPORT=mpi, pre-init MPI so CCL can use it.
    # This follows the official Aurora DDP pattern from the user guides.
    if os.environ.get("CCL_ATL_TRANSPORT") == "mpi":
        try:
            from mpi4py import MPI
            MPI.COMM_WORLD.Barrier()
        except ImportError:
            print("WARNING: CCL_ATL_TRANSPORT=mpi but mpi4py not available")

    # Init distributed
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.xpu.set_device(local_rank)
    device = torch.device(f"xpu:{local_rank}")
    dist.init_process_group("xccl", timeout=datetime.timedelta(minutes=5))
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Determine topology
    # Use LOCAL_WORLD_SIZE to compute number of nodes
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    num_nodes = world_size // local_world_size

    if rank == 0:
        print("=" * 70)
        print("FSDP2 Multi-Node Minimal Benchmark")
        print("=" * 70)
        print(f"World size:       {world_size} ({num_nodes} nodes × {local_world_size} tiles)")
        print(f"Model:            {MODEL_SIZE} ({NUM_LAYERS} layers, dim={HIDDEN_DIM}, ffn={HIDDEN_DIM * FFN_MULT})")
        print(f"Input:            batch={BATCH_SIZE}, seq_len={SEQ_LEN}")
        print(f"Test backward:    {TEST_BACKWARD}")
        print(f"Iters:            {WARMUP} warmup + {NUM_ITERS} timed")
        print(f"ZE_AFFINITY_MASK: {os.environ.get('ZE_AFFINITY_MASK', '<unset>')}")
        print(f"CCL_ALLREDUCE:    {os.environ.get('CCL_ALLREDUCE', '<default>')}")
        print(f"CCL_REDUCE_SCATTER: {os.environ.get('CCL_REDUCE_SCATTER', '<default>')}")
        print(f"FI_PROVIDER:      {os.environ.get('FI_PROVIDER', '<default>')}")

        # Estimate model size
        params_per_layer = HIDDEN_DIM * HIDDEN_DIM * FFN_MULT * 3 + HIDDEN_DIM * 2  # 3 linears + norm
        total_params = params_per_layer * NUM_LAYERS + HIDDEN_DIM * 2  # + final norm
        print(f"Est. params:      {total_params / 1e6:.0f}M ({total_params * 2 / 1e9:.1f} GiB in BF16)")
        print("=" * 70)

    # Patch FSDP2
    patch_fsdp2_for_xpu()

    # Force math-only SDPA (prevent flash/mem_efficient kernel issues)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)

    # Define mesh configurations to test
    configs = []

    if not SKIP_1D:
        configs.append(("1D_flat", None))  # 1D: flat FSDP across all ranks

    if num_nodes > 1:
        # 2D HSDP: replicate across nodes, shard within node
        configs.append((f"2D_HSDP_{num_nodes}x{local_world_size}", (num_nodes, local_world_size)))
    elif world_size >= 4:
        # Single-node: test 2D with simulated replicate=2
        half = world_size // 2
        configs.append((f"2D_sim_{2}x{half}", (2, half)))

    for config_name, mesh_dims in configs:
        if rank == 0:
            print(f"\n{'='*50}")
            print(f"Config: {config_name}")
            print(f"{'='*50}")

        run_benchmark(config_name, mesh_dims, device, rank, world_size,
                      local_world_size, num_nodes)

        # Cleanup between configs
        gc.collect()
        torch.xpu.synchronize()
        dist.barrier()

    # Also run raw AllGather benchmark for comparison
    if rank == 0:
        print(f"\n{'='*50}")
        print(f"Raw AllGather Benchmark (no FSDP)")
        print(f"{'='*50}")
    run_raw_allgather(device, rank, world_size, local_world_size, num_nodes)

    # Raw ReduceScatter + AllReduce benchmark (backward collectives)
    if rank == 0:
        print(f"\n{'='*50}")
        print(f"Raw ReduceScatter + AllReduce Benchmark (backward path)")
        print(f"{'='*50}")
    run_raw_reduce_scatter_allreduce(device, rank, world_size, local_world_size, num_nodes)

    # FSDP2 backward: reshard_after_forward=True vs False
    if rank == 0:
        print(f"\n{'='*50}")
        print(f"FSDP2 Backward: FULL_SHARD vs SHARD_GRAD_OP")
        print(f"{'='*50}")
    run_fsdp2_backward_comparison(device, rank, world_size, local_world_size, num_nodes)

    dist.destroy_process_group()


def run_benchmark(config_name, mesh_dims, device, rank, world_size,
                  local_world_size, num_nodes):
    """Run forward (and optionally backward) benchmark with given mesh config."""

    # Create mesh
    if mesh_dims is not None:
        dp_mesh = init_device_mesh("xpu", mesh_dims,
                                   mesh_dim_names=("dp_replicate", "dp_shard"))
        shard_mesh = dp_mesh["dp_shard"]
        if rank == 0:
            shard_group = dp_mesh.get_group("dp_shard")
            rep_group = dp_mesh.get_group("dp_replicate")
            print(f"  Shard group size: {shard_group.size()}, "
                  f"Replicate group size: {rep_group.size()}")
    else:
        dp_mesh = None
        shard_mesh = None

    # Create model directly on device (small benchmark — no need for meta device)
    model = SimpleModel(NUM_LAYERS, HIDDEN_DIM, FFN_MULT).to(dtype=torch.bfloat16, device=device)

    if not TEST_BACKWARD:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

    # Apply FSDP2 per-layer (like torchtune does for real models)
    for layer in model.layers:
        fully_shard(layer, mesh=shard_mesh, reshard_after_forward=True)
    fully_shard(model, mesh=shard_mesh, reshard_after_forward=True)

    dist.barrier()
    if rank == 0:
        mem = torch.xpu.memory_allocated() / 1e9
        print(f"  Model sharded. Memory: {mem:.2f} GiB")

    # Create input
    x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, dtype=torch.bfloat16, device=device)

    # Warmup
    for i in range(WARMUP):
        if TEST_BACKWARD:
            out = model(x)
            loss = out.sum()
            loss.backward()
            model.zero_grad()
        else:
            with torch.no_grad():
                _ = model(x)
        torch.xpu.synchronize()
    dist.barrier()

    if rank == 0:
        print(f"  Warmup done ({WARMUP} iters)")

    # Timed iterations
    fwd_times = []
    bwd_times = []

    for i in range(NUM_ITERS):
        torch.xpu.synchronize()
        dist.barrier()

        t0 = time.perf_counter()
        if TEST_BACKWARD:
            out = model(x)
            torch.xpu.synchronize()
            t_fwd = time.perf_counter() - t0

            t1 = time.perf_counter()
            loss = out.sum()
            loss.backward()
            torch.xpu.synchronize()
            t_bwd = time.perf_counter() - t1

            model.zero_grad()
            fwd_times.append(t_fwd)
            bwd_times.append(t_bwd)
        else:
            with torch.no_grad():
                _ = model(x)
            torch.xpu.synchronize()
            fwd_times.append(time.perf_counter() - t0)

    if rank == 0:
        avg_fwd = sum(fwd_times) / len(fwd_times)
        min_fwd = min(fwd_times)
        max_fwd = max(fwd_times)
        print(f"  Forward:  avg={avg_fwd*1000:.1f}ms  "
              f"min={min_fwd*1000:.1f}ms  max={max_fwd*1000:.1f}ms  "
              f"times={[f'{t*1000:.1f}' for t in fwd_times]}")

        if TEST_BACKWARD:
            avg_bwd = sum(bwd_times) / len(bwd_times)
            min_bwd = min(bwd_times)
            print(f"  Backward: avg={avg_bwd*1000:.1f}ms  "
                  f"min={min_bwd*1000:.1f}ms  max={max(bwd_times)*1000:.1f}ms  "
                  f"times={[f'{t*1000:.1f}' for t in bwd_times]}")
            print(f"  Total:    avg={avg_fwd*1000 + avg_bwd*1000:.1f}ms")

        mem_active = torch.xpu.memory_allocated() / 1e9
        mem_reserved = torch.xpu.memory_reserved() / 1e9
        print(f"  Memory:   {mem_active:.2f} / {mem_reserved:.2f} GiB (active/reserved)")

    # Cleanup (do NOT call torch.xpu.empty_cache() — leaks UR handles with FSDP)
    del model, x
    gc.collect()


def run_raw_allgather(device, rank, world_size, local_world_size, num_nodes):
    """Raw AllGather benchmark on world group and sub-groups."""
    sizes_mb = [10, 100, 500]

    groups_to_test = [("world", dist.group.WORLD, world_size)]

    if num_nodes > 1:
        # Create HSDP-like sub-groups
        mesh = init_device_mesh("xpu", (num_nodes, local_world_size),
                                mesh_dim_names=("dp_replicate", "dp_shard"))
        shard_pg = mesh.get_group("dp_shard")
        rep_pg = mesh.get_group("dp_replicate")
        groups_to_test.append(("shard_sub", shard_pg, local_world_size))
        groups_to_test.append(("replicate_sub", rep_pg, num_nodes))

    for group_name, group, group_size in groups_to_test:
        if rank == 0:
            print(f"\n  --- {group_name} (size={group_size}) ---")

        for size_mb in sizes_mb:
            shard_elems = size_mb * 1024 * 1024 // 2  # BF16
            shard = torch.randn(shard_elems, dtype=torch.bfloat16, device=device)
            output = torch.empty(shard_elems * group_size, dtype=torch.bfloat16, device=device)

            # Warmup
            for _ in range(3):
                dist.all_gather_into_tensor(output, shard, group=group)
            torch.xpu.synchronize()
            dist.barrier()

            # Timed
            times = []
            for _ in range(5):
                torch.xpu.synchronize()
                dist.barrier()
                t0 = time.perf_counter()
                dist.all_gather_into_tensor(output, shard, group=group)
                torch.xpu.synchronize()
                t = time.perf_counter() - t0
                times.append(t)

            avg = sum(times) / len(times)
            total_mb = size_mb * group_size
            bw = total_mb / avg / 1024  # GiB/s
            if rank == 0:
                print(f"    AllGather {size_mb:>4}MiB × {group_size} = {total_mb:>5}MiB: "
                      f"avg={avg*1000:.1f}ms  bw={bw:.1f} GiB/s  "
                      f"times={[f'{t*1000:.1f}' for t in times]}")

            del shard, output
            torch.xpu.synchronize()
            dist.barrier()


def run_raw_reduce_scatter_allreduce(device, rank, world_size, local_world_size, num_nodes):
    """Raw ReduceScatter + AllReduce benchmark on HSDP sub-groups.

    This measures the backward-path collectives:
      1. ReduceScatter on dp_shard (intra-node) — gradient sharding
      2. AllReduce on dp_replicate (inter-node) — gradient averaging
    """
    if num_nodes <= 1:
        if rank == 0:
            print("  Skipping ReduceScatter+AllReduce benchmark (single node)")
        return

    mesh = init_device_mesh("xpu", (num_nodes, local_world_size),
                            mesh_dim_names=("dp_replicate", "dp_shard"))
    shard_pg = mesh.get_group("dp_shard")
    rep_pg = mesh.get_group("dp_replicate")

    # Test sizes representing per-layer gradient sizes for 32B model:
    #   Full layer gradient: ~1 GiB (64M params × 2 bytes BF16)
    #   Shard gradient (after RS): ~100 MiB (1 GiB / 10 shards)
    sizes_mb = [100, 500, 1000]

    for size_mb in sizes_mb:
        elems = size_mb * 1024 * 1024 // 2  # BF16
        shard_elems = elems // local_world_size

        # --- ReduceScatter on dp_shard (intra-node) ---
        input_tensor = torch.randn(elems, dtype=torch.bfloat16, device=device)
        output_tensor = torch.empty(shard_elems, dtype=torch.bfloat16, device=device)

        # Warmup
        for _ in range(3):
            dist.reduce_scatter_tensor(output_tensor, input_tensor, group=shard_pg)
        torch.xpu.synchronize()
        dist.barrier()

        rs_times = []
        for _ in range(5):
            torch.xpu.synchronize()
            dist.barrier()
            t0 = time.perf_counter()
            dist.reduce_scatter_tensor(output_tensor, input_tensor, group=shard_pg)
            torch.xpu.synchronize()
            rs_times.append(time.perf_counter() - t0)

        del input_tensor

        # --- AllReduce on dp_replicate (inter-node) ---
        # AllReduce operates on the sharded gradient (size_mb / local_world_size)
        ar_tensor = output_tensor.clone()

        for _ in range(3):
            dist.all_reduce(ar_tensor, group=rep_pg)
        torch.xpu.synchronize()
        dist.barrier()

        ar_times = []
        for _ in range(5):
            torch.xpu.synchronize()
            dist.barrier()
            t0 = time.perf_counter()
            dist.all_reduce(ar_tensor, group=rep_pg)
            torch.xpu.synchronize()
            ar_times.append(time.perf_counter() - t0)

        del output_tensor, ar_tensor
        torch.xpu.synchronize()

        if rank == 0:
            rs_avg = sum(rs_times) / len(rs_times)
            ar_avg = sum(ar_times) / len(ar_times)
            shard_mb = size_mb // local_world_size
            rs_bw = size_mb / rs_avg / 1024  # GiB/s (input size)
            ar_bw = shard_mb / ar_avg / 1024  # GiB/s
            print(f"    {size_mb:>4}MiB layer: "
                  f"RS(shard,{local_world_size})={rs_avg*1000:.1f}ms ({rs_bw:.1f} GiB/s)  "
                  f"AR(rep,{num_nodes})={ar_avg*1000:.1f}ms ({shard_mb}MiB, {ar_bw:.1f} GiB/s)  "
                  f"total={rs_avg*1000 + ar_avg*1000:.1f}ms")


def run_fsdp2_backward_comparison(device, rank, world_size, local_world_size, num_nodes):
    """Compare FSDP2 backward with reshard_after_forward=True vs False.

    True = FULL_SHARD: AllGather + compute + ReduceScatter + AllReduce per layer
    False = SHARD_GRAD_OP: compute + ReduceScatter + AllReduce per layer (no re-AllGather)
    """
    if num_nodes <= 1:
        if rank == 0:
            print("  Skipping backward comparison (single node)")
        return

    mesh = init_device_mesh("xpu", (num_nodes, local_world_size),
                            mesh_dim_names=("dp_replicate", "dp_shard"))

    for reshard_mode in [True, False]:
        mode_name = "FULL_SHARD" if reshard_mode else "SHARD_GRAD_OP"
        if rank == 0:
            print(f"\n  --- {mode_name} (reshard_after_forward={reshard_mode}) ---")

        model = SimpleModel(NUM_LAYERS, HIDDEN_DIM, FFN_MULT).to(
            dtype=torch.bfloat16, device=device)

        for layer in model.layers:
            fully_shard(layer, mesh=mesh, reshard_after_forward=reshard_mode)
        fully_shard(model, mesh=mesh, reshard_after_forward=False)  # root always False

        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        # Warmup
        for _ in range(2):
            out = model(x)
            loss = out.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        torch.xpu.synchronize()
        dist.barrier()

        # Timed
        fwd_times = []
        bwd_times = []
        opt_times = []
        for _ in range(5):
            torch.xpu.synchronize()
            dist.barrier()

            t0 = time.perf_counter()
            out = model(x)
            torch.xpu.synchronize()
            t_fwd = time.perf_counter() - t0

            t1 = time.perf_counter()
            loss = out.sum()
            loss.backward()
            torch.xpu.synchronize()
            t_bwd = time.perf_counter() - t1

            t2 = time.perf_counter()
            optimizer.step()
            optimizer.zero_grad()
            torch.xpu.synchronize()
            t_opt = time.perf_counter() - t2

            fwd_times.append(t_fwd)
            bwd_times.append(t_bwd)
            opt_times.append(t_opt)

        if rank == 0:
            avg_fwd = sum(fwd_times) / len(fwd_times) * 1000
            avg_bwd = sum(bwd_times) / len(bwd_times) * 1000
            avg_opt = sum(opt_times) / len(opt_times) * 1000
            mem = torch.xpu.memory_allocated() / 1e9
            mem_res = torch.xpu.memory_reserved() / 1e9
            print(f"    Forward:  {avg_fwd:.1f}ms")
            print(f"    Backward: {avg_bwd:.1f}ms")
            print(f"    Optimizer:{avg_opt:.1f}ms")
            print(f"    Total:    {avg_fwd + avg_bwd + avg_opt:.1f}ms")
            print(f"    Memory:   {mem:.2f} / {mem_res:.2f} GiB")

        del model, x, optimizer
        gc.collect()
        torch.xpu.synchronize()
        dist.barrier()


if __name__ == "__main__":
    main()

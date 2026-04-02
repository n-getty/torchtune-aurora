"""Compare FSDP2 forward: 1D (flat) vs 2D (HSDP) mesh on 2 nodes.

Uses Qwen3-3B for fast loading. Run with mpiexec across 2 nodes (10 ranks/node):
  mpiexec -n 20 -ppn 10 --hostfile $PBS_NODEFILE ...wrapper... this_script.py

Or torchrun for single-node baseline:
  torchrun --standalone --nproc_per_node=10 this_script.py --test 1d
"""
import os, sys, time, types, gc, datetime, argparse
import torch
import torch.nn as nn
import torch.distributed as dist

# Pre-register torchtune to bypass __init__.py
if "torchtune" not in sys.modules:
    sys.modules["torchtune"] = types.ModuleType("torchtune")
    sys.modules["torchtune"].__path__ = [os.path.join(os.path.dirname(__file__), "..", "..", "torchtune")]

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--test", choices=["1d", "2d", "both"], default="both",
                    help="Which mesh test to run (default: both)")
parser.add_argument("--model-size", choices=["simple", "3b"], default="simple",
                    help="Model to test: simple (linear stack) or 3b (Qwen3-3B)")
parser.add_argument("--hidden", type=int, default=8192, help="Hidden dim for simple model")
parser.add_argument("--layers", type=int, default=36, help="Num layers for simple model")
parser.add_argument("--warmup", type=int, default=3)
parser.add_argument("--trials", type=int, default=5)
args = parser.parse_args()

# Init
local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("PALS_LOCAL_RANKID", "0")))
torch.xpu.set_device(local_rank)
device = torch.device(f"xpu:{local_rank}")
dist.init_process_group("xccl", timeout=datetime.timedelta(minutes=5))
rank = dist.get_rank()
world_size = dist.get_world_size()

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

# Monkey-patch FSDP2 for XPU (ReduceOp.AVG)
try:
    import torch.distributed.fsdp._fully_shard._fsdp_collectives as _fsdp_coll
    def _patched_get_gradient_divide_factors(world_size):
        return (world_size, 1)
    _fsdp_coll._get_gradient_divide_factors = _patched_get_gradient_divide_factors
except Exception:
    pass

if rank == 0:
    print(f"World size: {world_size}, model: {args.model_size}, test: {args.test}")
    print(f"Device: {device}")


class BigLinearStack(nn.Module):
    """Approximates transformer memory/compute pattern."""
    def __init__(self, hidden=8192, layers=36):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def build_model():
    if args.model_size == "simple":
        model = BigLinearStack(hidden=args.hidden, layers=args.layers)
        model = model.to(dtype=torch.bfloat16, device=device)
        return model
    elif args.model_size == "3b":
        from torchtune.models.qwen2_5 import qwen2_5_3b
        from torchtune import training
        with training.set_default_dtype(torch.bfloat16), torch.device("meta"):
            model = qwen2_5_3b()
        return model
    else:
        raise ValueError(f"Unknown model size: {args.model_size}")


def shard_and_materialize(model, dp_mesh):
    if args.model_size == "3b":
        from torchtune.training._distributed import shard_model
        from torchtune import training
        from functools import partial
        shard_conditions = [partial(training.get_shard_conditions, names_to_match=None)]
        shard_model(model=model, shard_conditions=shard_conditions, cpu_offload=False,
                    reshard_after_forward=True, dp_mesh=dp_mesh)
        # Initialize on device
        from torchtune.training import FullModelHFCheckpointer
        ckpt_path = "/tmp/torchtune/Qwen2.5-3B"
        ckpt_files = sorted([f for f in os.listdir(ckpt_path) if f.endswith('.safetensors')])
        checkpointer = FullModelHFCheckpointer(
            checkpoint_dir=ckpt_path, checkpoint_files=ckpt_files,
            output_dir="/tmp/torchtune/test_mesh_2node", model_type="QWEN2",
        )
        ckpt = checkpointer.load_checkpoint()
        training.load_from_full_model_state_dict(model, ckpt["model"], device, strict=True, cpu_offload=False)
        # RoPE init (must come before validate — pos_embeddings.theta is a buffer on meta)
        with training.set_default_dtype(torch.bfloat16), device:
            for m in model.modules():
                if hasattr(m, "rope_init"):
                    m.rope_init()
        training.validate_no_params_on_meta_device(model)
        del ckpt
    else:
        for layer in model.layers:
            fully_shard(layer, mesh=dp_mesh, reshard_after_forward=True)
        fully_shard(model, mesh=dp_mesh, reshard_after_forward=False)


def run_forward(model, warmup=3, trials=5):
    if args.model_size == "3b":
        input_ids = torch.randint(0, 151936, (1, 512), device=device)
        position_ids = torch.arange(512, device=device).unsqueeze(0)
        fwd = lambda: model(input_ids, input_pos=position_ids)
    else:
        x = torch.randn(4, 512, args.hidden, dtype=torch.bfloat16, device=device)
        fwd = lambda: model(x)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = fwd()
        torch.xpu.synchronize()
    dist.barrier()

    # Timed trials
    times = []
    for i in range(trials):
        torch.xpu.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = fwd()
        torch.xpu.synchronize()
        t = time.perf_counter() - t0
        times.append(t)
        if rank == 0:
            print(f"    trial {i}: {t*1000:.1f}ms")

    return times


def test_mesh(label, dp_mesh):
    if rank == 0:
        print(f"\n=== {label} ===")
        t0 = time.perf_counter()
    model = build_model()
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    shard_and_materialize(model, dp_mesh)
    dist.barrier()
    if rank == 0:
        print(f"  Model loaded in {time.perf_counter()-t0:.1f}s")

    times = run_forward(model, warmup=args.warmup, trials=args.trials)
    avg = sum(times) / len(times)
    if rank == 0:
        print(f"  [{label}] avg: {avg*1000:.1f}ms  (min={min(times)*1000:.1f}, max={max(times)*1000:.1f})")

    del model
    gc.collect()
    torch.xpu.synchronize()
    dist.barrier()
    return avg


results = {}

if args.test in ("1d", "both"):
    results["1D"] = test_mesh("1D flat FSDP (mesh=None)", dp_mesh=None)

if args.test in ("2d", "both"):
    if world_size >= 4 and world_size % 2 == 0:
        # For 2 nodes with equal ranks per node
        local_size = world_size // 2
        if rank == 0:
            print(f"\nCreating 2D mesh: (2, {local_size})")
        mesh_2d = init_device_mesh("xpu", (2, local_size),
                                    mesh_dim_names=("dp_replicate", "dp_shard"))
        results["2D"] = test_mesh(f"2D HSDP (2x{local_size})", dp_mesh=mesh_2d)
    else:
        if rank == 0:
            print(f"Skipping 2D test (world_size={world_size}, need >=4 and even)")

if rank == 0:
    print(f"\n{'='*50}")
    print("RESULTS:")
    for label, t in results.items():
        print(f"  {label}: {t*1000:.1f}ms")
    if "1D" in results and "2D" in results:
        print(f"  Ratio (2D/1D): {results['2D']/results['1D']:.2f}x")
    print(f"{'='*50}")

dist.destroy_process_group()

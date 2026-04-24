"""
EP correctness test: verify that EP=2 and EP=1 (replicated) expert computation
produces identical outputs given the same weights and inputs.

Strategy:
- Rank 0 and Rank 1 both hold a full copy of GroupedExperts (EP=1 reference).
- Then we apply EP=2 via ExpertParallel to a COPY of the module.
- We run both and compare outputs — they must match exactly (no randomness in experts).

Run with:
    torchrun --nproc_per_node=2 recipes/dev/test_ep_correctness.py
"""
import os
import sys
import copy

import torch
import torch.distributed as dist
import torch.nn.functional as F
from functools import partial


def setup_dist():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if torch.xpu.is_available():
        os.environ["ZE_AFFINITY_MASK"] = str(local_rank)
        import intel_extension_for_pytorch  # noqa: F401
        torch.xpu.set_device(local_rank)
        device = torch.device(f"xpu:{local_rank}")
        dist.init_process_group(backend="xccl")
    else:
        device = torch.device("cpu")
        dist.init_process_group(backend="gloo")

    return device, rank, world_size


def test_ep_vs_replicated(device, rank, world_size):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor.parallel import parallelize_module
    from torchtune.modules.moe._parallelism import ExpertParallel
    from torchtune.modules.moe.experts import GroupedExperts

    ep_degree = world_size  # EP=2
    num_experts = ep_degree * 2  # 4 total experts
    dim = 32
    hidden_dim = 64
    num_tokens = 8  # total routed tokens

    torch.manual_seed(0)

    # Build reference experts (replicated, both ranks hold full copy)
    ref_experts = GroupedExperts(
        dim=dim, hidden_dim=hidden_dim, num_experts=num_experts,
        activation=partial(F.gelu, approximate="tanh"),
    )
    ref_experts.reset_parameters()
    ref_experts = ref_experts.to(device)

    # Build EP experts from same weights
    ep_experts = copy.deepcopy(ref_experts)

    # Apply ExpertParallel to ep_experts
    ep_mesh = init_device_mesh(device.type, (ep_degree,), mesh_dim_names=("ep",))
    parallelize_module(ep_experts, ep_mesh, {"": ExpertParallel()})

    # Fixed routing: distribute tokens evenly across experts
    # ntpe = [2, 2, 2, 2] — each expert gets 2 tokens
    ntpe = torch.tensor([2, 2, 2, 2], dtype=torch.long, device=device)

    # Same input on all ranks
    torch.manual_seed(42)
    routed_input = torch.randn(num_tokens, dim, device=device, dtype=torch.float32)

    # Reference forward (no EP, replicated)
    with torch.no_grad():
        ref_out = ref_experts(routed_input, ntpe)

    # EP forward
    with torch.no_grad():
        ep_out = ep_experts(routed_input, ntpe)

    max_err = (ref_out - ep_out).abs().max().item()
    mean_err = (ref_out - ep_out).abs().mean().item()

    print(f"Rank {rank}: max_err={max_err:.6e}, mean_err={mean_err:.6e}", flush=True)

    tol = 1e-4
    assert max_err < tol, (
        f"Rank {rank}: EP output differs from reference! max_err={max_err:.6e} >= {tol}"
    )
    print(f"Rank {rank}: EP output matches reference (tol={tol})", flush=True)
    return True


def main():
    device, rank, world_size = setup_dist()

    assert world_size == 2, f"Run with exactly 2 ranks (got {world_size})"

    try:
        dist.barrier()
        ok = test_ep_vs_replicated(device, rank, world_size)
        dist.barrier()

        if rank == 0:
            print("\n=== EP Correctness Test: PASS ===" if ok else "=== EP Correctness Test: FAIL ===")
            sys.exit(0 if ok else 1)

    except Exception as e:
        print(f"[FAIL] Rank {rank}: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

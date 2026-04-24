"""
EP smoke test: verify all_to_all_single_autograd, ExpertParallel dispatch/combine,
and forward pass correctness on a live distributed job.

Run with:
    mpiexec -n 2 python recipes/dev/test_ep_smoke.py          # EP=2 smoke
    mpiexec -n 4 python recipes/dev/test_ep_smoke.py --ep 4   # EP=4 smoke
"""
import argparse
import os
import sys

import torch
import torch.distributed as dist


def setup_dist():
    # torchrun sets RANK, LOCAL_RANK, WORLD_SIZE automatically
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if torch.xpu.is_available():
        os.environ["ZE_AFFINITY_MASK"] = str(local_rank)
        import intel_extension_for_pytorch  # noqa: F401
        torch.xpu.set_device(local_rank)
        device = torch.device(f"xpu:{local_rank}")
        backend = "xccl"
        dist.init_process_group(backend=backend)
    else:
        device = torch.device("cpu")
        backend = "gloo"
        dist.init_process_group(backend=backend)

    return device, rank, world_size


def test_imports():
    print("[1] Testing imports...", flush=True)
    from torch.distributed._functional_collectives import (
        all_to_all_single,
        all_to_all_single_autograd,
    )
    from torchtune.modules.moe._parallelism import ExpertParallel
    from torchtune.modules.moe.utils import _permute, _unpermute
    from torchtune.models.gemma4._parallelism import gemma4_ep_plan
    from torchtune.training._distributed import ParallelDims, shard_experts_for_ep
    print("[1] All imports OK", flush=True)
    return True


def test_permute_unpermute(device, rank):
    """Verify _permute/_unpermute round-trips correctly."""
    print(f"[2] Rank {rank}: testing _permute/_unpermute...", flush=True)
    from torchtune.modules.moe.utils import _permute, _unpermute

    ep_degree = 2
    num_local_experts = 4
    dim = 16

    # Simulate 3 tokens for exp0, 2 for exp1, 1 for exp2, 4 for exp3 (from rank0)
    # and 2 for exp0, 0 for exp1, 3 for exp2, 1 for exp3 (from rank1)
    ntpe_group = torch.tensor(
        [3, 2, 1, 4, 2, 0, 3, 1], dtype=torch.long, device=device
    )  # shape: (ep_degree * num_local_experts,) = (8,)
    total = ntpe_group.sum().item()

    x = torch.arange(total * dim, dtype=torch.float32, device=device).reshape(total, dim)

    permuted, local_ntpe, perm = _permute(x, ntpe_group, ep_degree, num_local_experts)

    # local_ntpe should sum tokens per expert across both source ranks
    expected_local_ntpe = torch.tensor([5, 2, 4, 5], dtype=torch.long, device=device)
    assert torch.equal(local_ntpe, expected_local_ntpe), (
        f"local_ntpe mismatch: {local_ntpe} vs {expected_local_ntpe}"
    )
    assert permuted.shape == (total, dim), f"permuted shape wrong: {permuted.shape}"

    # Round-trip: unpermute should recover original
    recovered = _unpermute(permuted, perm, total)
    assert torch.allclose(recovered, x), "unpermute did not recover original"

    print(f"[2] Rank {rank}: _permute/_unpermute OK (total={total} tokens)", flush=True)
    return True


def test_alltoall(device, rank, world_size, ep_degree):
    """Verify all_to_all_single_autograd works over the EP process group."""
    print(f"[3] Rank {rank}: testing all_to_all_single_autograd (ep={ep_degree})...", flush=True)
    from torch.distributed._functional_collectives import (
        all_to_all_single,
        all_to_all_single_autograd,
    )

    # Simple test: each rank sends rank+1 tokens to each other rank
    tokens_per_rank = rank + 1
    send_counts = [tokens_per_rank] * ep_degree
    recv_counts = [r + 1 for r in range(ep_degree)]

    dim = 8
    send_buf = torch.full((sum(send_counts), dim), float(rank), device=device,
                          requires_grad=True)
    recv_buf = all_to_all_single_autograd(
        send_buf, recv_counts, send_counts, dist.group.WORLD
    )

    # Check shapes
    assert recv_buf.shape[0] == sum(recv_counts), (
        f"Rank {rank}: recv shape {recv_buf.shape[0]} != {sum(recv_counts)}"
    )

    # Check grad flows
    loss = recv_buf.sum()
    loss.backward()
    assert send_buf.grad is not None, "Gradient did not flow through all_to_all"

    print(f"[3] Rank {rank}: all_to_all_single_autograd OK, grad flows", flush=True)
    return True


def test_ep_forward(device, rank, world_size, ep_degree):
    """Forward pass through ExpertParallel on a toy GroupedExperts module."""
    print(f"[4] Rank {rank}: testing ExpertParallel forward (ep={ep_degree})...", flush=True)

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor.parallel import parallelize_module
    from torchtune.modules.moe._parallelism import ExpertParallel
    from torchtune.modules.moe.experts import GroupedExperts
    import torch.nn.functional as F
    from functools import partial

    # Toy config: 4 total experts, 2 local per rank (for EP=2)
    num_experts = ep_degree * 2  # ensures even division
    dim = 32
    hidden_dim = 64
    top_k = 2

    # Build EP mesh
    ep_mesh = init_device_mesh(device.type, (ep_degree,), mesh_dim_names=("ep",))

    # Build toy GroupedExperts (initialize params to avoid NaN from torch.empty)
    experts = GroupedExperts(
        dim=dim,
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        activation=partial(F.gelu, approximate="tanh"),
    )
    experts.reset_parameters()
    experts = experts.to(device)

    # Apply ExpertParallel
    parallelize_module(experts, ep_mesh, {"": ExpertParallel()})

    # Simulate router output: bs*slen=4 tokens, top_k=2
    bs_slen = 4
    num_tokens = bs_slen * top_k  # 8 routed tokens

    torch.manual_seed(42 + rank)
    routed_input = torch.randn(num_tokens, dim, device=device, dtype=torch.float32)

    # Build num_tokens_per_expert (random distribution summing to num_tokens)
    # Use same seed on all ranks so routing is consistent
    g = torch.Generator(device=device)
    g.manual_seed(42)
    raw = torch.rand(num_experts, device=device, generator=g)
    ntpe = (raw / raw.sum() * num_tokens).long()
    # Ensure sum == num_tokens
    diff = num_tokens - ntpe.sum().item()
    ntpe[0] += diff

    # Forward through experts (EP dispatch happens transparently)
    out = experts(routed_input, ntpe)

    assert out.shape == (num_tokens, dim), (
        f"Rank {rank}: output shape {out.shape} != {(num_tokens, dim)}"
    )
    assert not torch.isnan(out).any(), f"Rank {rank}: NaN in output"

    print(f"[4] Rank {rank}: ExpertParallel forward OK, output shape {out.shape}", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, default=2)
    args = parser.parse_args()

    device, rank, world_size = setup_dist()

    assert world_size == args.ep, (
        f"world_size={world_size} must equal ep={args.ep} for this smoke test"
    )

    results = []
    try:
        results.append(("imports", test_imports()))
        dist.barrier()
        results.append(("permute_unpermute", test_permute_unpermute(device, rank)))
        dist.barrier()
        results.append(("alltoall", test_alltoall(device, rank, world_size, args.ep)))
        dist.barrier()
        results.append(("ep_forward", test_ep_forward(device, rank, world_size, args.ep)))
        dist.barrier()

        if rank == 0:
            print("\n=== EP Smoke Test Results ===")
            all_pass = True
            for name, ok in results:
                status = "PASS" if ok else "FAIL"
                print(f"  {status}: {name}")
                all_pass = all_pass and ok
            print(f"\n{'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
            sys.exit(0 if all_pass else 1)

    except Exception as e:
        print(f"[FAIL] Rank {rank}: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        dist.barrier()
        sys.exit(1)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

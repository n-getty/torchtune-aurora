"""Production-scale XCCL broadcast test: 10 training + 2 vLLM ranks.

Matches the 10+2 tile layout used in run_grpo_vllm_xpu.sh.
Tests sub-group operations at realistic sizes:
  - Training allreduce/reduce_scatter on 10-rank sub-group (FSDP pattern)
  - Broadcast from rank 0 to all 12 ranks (weight sync replacement)
  - Bandwidth at model scale: 6 GB (3B bf16), and scaling series up to available memory
  - Overlap test: non-blocking broadcast concurrent with training compute

Usage:
  python3 -m torch.distributed.run --standalone --nproc_per_node=12 \\
      recipes/dev/_test_xccl_broadcast_12rank.py
"""
import os
import sys
import time

import types as _types
import importlib.util as _imp_util
if "torchtune" not in sys.modules:
    _spec = _imp_util.find_spec("torchtune")
    if _spec is not None and _spec.submodule_search_locations:
        _torchtune_path = list(_spec.submodule_search_locations)[0]
        _pkg = _types.ModuleType("torchtune")
        _pkg.__path__ = [_torchtune_path]
        _pkg.__file__ = os.path.join(_torchtune_path, "__init__.py")
        _pkg.__version__ = ""
        sys.modules["torchtune"] = _pkg

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as _dc10d

TRAIN_RANKS = 10
VLLM_RANKS = 2

rank = int(os.environ.get("RANK", "0"))
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "12"))
device = torch.device(f"xpu:{local_rank}")
torch.xpu.set_device(local_rank)
is_training = rank < TRAIN_RANKS
role = "TRAIN" if is_training else "VLLM"

results = []


def log(msg):
    if rank == 0 or rank == TRAIN_RANKS:
        print(f"[rank {rank}/{role}] {msg}", flush=True)


def log_all(msg):
    print(f"[rank {rank}/{role}] {msg}", flush=True)


def record(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    results.append((name, passed))
    if rank == 0:
        print(f"  {name}: {status} {detail}", flush=True)


# ============================================================
# Step 0: Global PG init (all 12 ranks)
# ============================================================
log(f"Initializing global XCCL PG (world_size={world_size}, device={device})")
dist.init_process_group(backend="xccl", device_id=device)
log(f"Global PG ready")

t = torch.ones(10, device=device)
dist.all_reduce(t)
record("global_allreduce_12rank", abs(t[0].item() - 12.0) < 1e-3,
       f"sum={t[0].item()}")

# ============================================================
# Step 1: Create sub-groups
# ============================================================
_default_pg = _dc10d._get_default_group()
_orig_bound = _default_pg.bound_device_id
_default_pg.bound_device_id = None

try:
    train_ranks = list(range(TRAIN_RANKS))
    vllm_ranks = list(range(TRAIN_RANKS, world_size))

    training_pg = dist.new_group(train_ranks)
    vllm_pg = dist.new_group(vllm_ranks)
    record("new_group_10train_2vllm", True)
except Exception as e:
    record("new_group_10train_2vllm", False, str(e))
    dist.destroy_process_group()
    sys.exit(1)
finally:
    _default_pg.bound_device_id = _orig_bound

# ============================================================
# Test A: 10-rank training allreduce
# ============================================================
if is_training:
    t_a = torch.ones(1000, device=device) * (rank + 1)
    dist.all_reduce(t_a, group=training_pg)
    expected_a = sum(range(1, TRAIN_RANKS + 1))  # 55
    record("test_a_10rank_allreduce", abs(t_a[0].item() - expected_a) < 1e-1,
           f"sum={t_a[0].item()}, expected={expected_a}")
else:
    pass
dist.barrier()
if not is_training:
    record("test_a_10rank_allreduce", True, "skipped (vLLM)")

# ============================================================
# Test B: 10-rank reduce_scatter (FSDP pattern)
# ============================================================
if is_training:
    chunk_size = 10000
    t_b = torch.ones(chunk_size * TRAIN_RANKS, device=device, dtype=torch.bfloat16) * (rank + 1)
    output_b = torch.zeros(chunk_size, device=device, dtype=torch.bfloat16)
    dist.reduce_scatter_tensor(output_b, t_b, group=training_pg)
    expected_b = sum(range(1, TRAIN_RANKS + 1))  # 55
    record("test_b_reduce_scatter", abs(output_b[0].item() - expected_b) < 1.0,
           f"val={output_b[0].item()}, expected={expected_b}")
    del t_b, output_b
else:
    pass
dist.barrier()
if not is_training:
    record("test_b_reduce_scatter", True, "skipped (vLLM)")

# ============================================================
# Test C: Global broadcast 100MB (correctness)
# ============================================================
SIZE_100MB = 100 * 1024 * 1024 // 2  # bf16
if rank == 0:
    t_c = torch.arange(SIZE_100MB, device=device, dtype=torch.float32).to(torch.bfloat16)
else:
    t_c = torch.zeros(SIZE_100MB, device=device, dtype=torch.bfloat16)

dist.broadcast(t_c, src=0)
torch.xpu.synchronize(device)

check_idx = 54321
check_val = t_c[check_idx].item()
expected_val = float(check_idx)
ok = abs(check_val - expected_val) / max(abs(expected_val), 1) < 0.01  # bf16 tolerance
record("test_c_broadcast_100mb_correctness", ok,
       f"val@{check_idx}={check_val}, expected={expected_val}")
del t_c
dist.barrier()

# ============================================================
# Test D: Post-broadcast training ops still work
# ============================================================
if is_training:
    t_d = torch.ones(1000, device=device, dtype=torch.bfloat16) * rank
    dist.all_reduce(t_d, group=training_pg)
    expected_d = sum(range(TRAIN_RANKS))  # 45
    record("test_d_coexistence", abs(t_d[0].item() - expected_d) < 1.0,
           f"sum={t_d[0].item()}, expected={expected_d}")
else:
    record("test_d_coexistence", True, "skipped (vLLM)")
dist.barrier()

# ============================================================
# Test E: Bandwidth scaling series
# ============================================================
sizes_gb = [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]
if rank == 0:
    print(f"\n  Broadcast bandwidth (rank 0 → 12 ranks, bf16):", flush=True)
    print(f"  {'Size':>8s}  {'Time':>8s}  {'BW':>10s}", flush=True)
    print(f"  {'-'*8}  {'-'*8}  {'-'*10}", flush=True)

bw_results = []
for size_gb in sizes_gb:
    n_elements = int(size_gb * 1024**3 / 2)  # bf16 = 2 bytes
    try:
        if rank == 0:
            t_e = torch.randn(n_elements, device=device, dtype=torch.bfloat16)
        else:
            t_e = torch.zeros(n_elements, device=device, dtype=torch.bfloat16)

        torch.xpu.synchronize(device)
        dist.barrier()

        # Warmup
        dist.broadcast(t_e, src=0)
        torch.xpu.synchronize(device)
        dist.barrier()

        # Timed run (3 iterations, take median)
        times = []
        for _ in range(3):
            torch.xpu.synchronize(device)
            dist.barrier()
            t0 = time.perf_counter()
            dist.broadcast(t_e, src=0)
            torch.xpu.synchronize(device)
            dt = time.perf_counter() - t0
            times.append(dt)

        median_t = sorted(times)[1]
        bw = size_gb / median_t
        bw_results.append((size_gb, median_t, bw))

        if rank == 0:
            print(f"  {size_gb:>7.1f}G  {median_t:>7.3f}s  {bw:>8.2f} GB/s", flush=True)

        del t_e
        torch.xpu.synchronize(device)

    except Exception as ex:
        if rank == 0:
            print(f"  {size_gb:>7.1f}G  FAILED: {ex}", flush=True)
        bw_results.append((size_gb, -1, -1))
        break

dist.barrier()
if bw_results:
    best_bw = max(bw for _, _, bw in bw_results if bw > 0)
    record("test_e_bandwidth_scaling", best_bw > 0,
           f"best={best_bw:.2f} GB/s at {[s for s, _, bw in bw_results if bw == best_bw][0]:.1f}G")

# ============================================================
# Test F: Interleaved training + broadcast (10 iterations)
# ============================================================
SIZE_50MB = 50 * 1024 * 1024 // 2  # bf16
try:
    for i in range(10):
        if is_training:
            t_f1 = torch.randn(SIZE_50MB, device=device, dtype=torch.bfloat16)
            dist.all_reduce(t_f1, group=training_pg)
            del t_f1

        dist.barrier()

        if rank == 0:
            t_f2 = torch.randn(SIZE_50MB, device=device, dtype=torch.bfloat16)
        else:
            t_f2 = torch.zeros(SIZE_50MB, device=device, dtype=torch.bfloat16)
        dist.broadcast(t_f2, src=0)
        del t_f2

        dist.barrier()

    record("test_f_interleaved_10iter", True, "10 iterations OK")
except Exception as e:
    record("test_f_interleaved_10iter", False, str(e))

# ============================================================
# Test G: Non-blocking broadcast overlap with training compute
# ============================================================
SIZE_1GB = 1024 * 1024 * 1024 // 2  # bf16
try:
    if rank == 0:
        t_bcast = torch.randn(SIZE_1GB, device=device, dtype=torch.bfloat16)
    else:
        t_bcast = torch.zeros(SIZE_1GB, device=device, dtype=torch.bfloat16)

    torch.xpu.synchronize(device)
    dist.barrier()

    # Baseline: blocking broadcast
    t0 = time.perf_counter()
    dist.broadcast(t_bcast, src=0)
    torch.xpu.synchronize(device)
    blocking_time = time.perf_counter() - t0

    dist.barrier()

    # Overlap: non-blocking broadcast + training matmul
    if rank == 0:
        t_bcast.fill_(1.0)
    torch.xpu.synchronize(device)
    dist.barrier()

    t0 = time.perf_counter()
    work = dist.broadcast(t_bcast, src=0, async_op=True)

    # Simulate training compute on training ranks (large matmul)
    if is_training:
        a = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
        b = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
        for _ in range(5):
            c = torch.mm(a, b)
        torch.xpu.synchronize(device)
        del a, b, c

    work.wait()
    torch.xpu.synchronize(device)
    overlap_time = time.perf_counter() - t0

    # Check overlap effectiveness
    speedup = blocking_time / overlap_time if overlap_time > 0 else 0
    hidden = max(0, 1.0 - overlap_time / (blocking_time + 0.001)) * 100

    record("test_g_async_overlap",
           overlap_time < blocking_time * 1.5,  # allow some overhead
           f"blocking={blocking_time:.3f}s, overlapped={overlap_time:.3f}s, "
           f"speedup={speedup:.2f}×, ~{hidden:.0f}% hidden")

    del t_bcast

except Exception as e:
    record("test_g_async_overlap", False, str(e))

# ============================================================
# Summary
# ============================================================
dist.barrier()
if rank == 0:
    print("\n" + "=" * 70)
    print("XCCL BROADCAST 12-RANK TEST RESULTS")
    print("=" * 70)
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    print("=" * 70)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 70)

dist.destroy_process_group()

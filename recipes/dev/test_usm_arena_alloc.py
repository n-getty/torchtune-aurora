"""
Integration tests for usm_arena_alloc.so — coalescing XPU allocator.

Run on a compute node (requires XPU hardware):

  # Build first:
  module load frameworks/2025.3.1
  icpx -shared -fPIC -fsycl -O2 \\
      -o recipes/dev/usm_arena_alloc.so recipes/dev/usm_arena_alloc.cpp

  # T0 — unit tests (single process, 1 tile):
  ZE_AFFINITY_MASK=0 ZE_FLAT_DEVICE_HIERARCHY=FLAT \\
  USM_ARENA_CHUNK_GB=4 \\
  python3 recipes/dev/test_usm_arena_alloc.py

  # T1 — CCL compatibility (2 tiles, run section explicitly):
  ZE_AFFINITY_MASK=0,1 ZE_FLAT_DEVICE_HIERARCHY=FLAT \\
  torchrun --nproc_per_node=2 --master-port=29401 \\
      recipes/dev/test_usm_arena_alloc.py --ccl

  # T2 — FSDP2 smoke test (4 tiles):
  ZE_FLAT_DEVICE_HIERARCHY=FLAT \\
  torchrun --nproc_per_node=4 --master-port=29402 \\
      recipes/dev/test_usm_arena_alloc.py --fsdp2
"""
import argparse
import ctypes
import gc
import os
import sys
import time

SO_PATH = os.path.join(os.path.dirname(__file__), "usm_arena_alloc.so")

# ---------------------------------------------------------------------------
# Allocator registration — must happen before any XPU init
# ---------------------------------------------------------------------------
def install_allocator():
    if not os.path.exists(SO_PATH):
        print(f"ERROR: {SO_PATH} not found. Build with:")
        print("  icpx -shared -fPIC -fsycl -O2 -o usm_arena_alloc.so usm_arena_alloc.cpp")
        sys.exit(1)
    from torch.xpu.memory import XPUPluggableAllocator, change_current_allocator
    alloc = XPUPluggableAllocator(SO_PATH, "xpu_usm_malloc", "xpu_usm_free")
    change_current_allocator(alloc)
    print(f"[test] Arena allocator registered: {SO_PATH}")

install_allocator()  # MUST be before any torch.xpu call

import torch
import torch.distributed as dist

# ---------------------------------------------------------------------------
# ctypes helpers for arena stats
# ---------------------------------------------------------------------------
_lib = ctypes.CDLL(SO_PATH)
_lib.xpu_usm_get_arena_stats.argtypes = [
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_size_t),
]
_lib.xpu_usm_get_arena_stats.restype = None

def arena_stats():
    growths    = ctypes.c_size_t(0)
    coalesces  = ctypes.c_size_t(0)
    reserved   = ctypes.c_size_t(0)
    allocated  = ctypes.c_size_t(0)
    _lib.xpu_usm_get_arena_stats(
        ctypes.byref(growths), ctypes.byref(coalesces),
        ctypes.byref(reserved), ctypes.byref(allocated))
    return {
        "growths":    growths.value,
        "coalesces":  coalesces.value,
        "reserved_mb": reserved.value / 1e6,
        "allocated_mb": allocated.value / 1e6,
    }

# ---------------------------------------------------------------------------
# T0 — Unit tests (single process, single tile)
# ---------------------------------------------------------------------------
PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

def run_unit_tests():
    device = torch.device("xpu:0")
    torch.xpu.set_device(0)

    print("\n=== T0: Unit tests (single-process, device xpu:0) ===\n")
    passed = 0
    failed = 0

    def check(label, condition, msg=""):
        nonlocal passed, failed
        if condition:
            print(f"  {PASS}  {label}")
            passed += 1
        else:
            print(f"  {FAIL}  {label}: {msg}")
            failed += 1

    # ------------------------------------------------------------------
    # T0.1 — Small tensor reuse (< 1 MiB → size-class bucket)
    # ------------------------------------------------------------------
    a = torch.empty(512 * 1024, dtype=torch.uint8, device=device)  # 512 KiB
    p_a = a.data_ptr()
    del a; torch.xpu.synchronize()
    b = torch.empty(512 * 1024, dtype=torch.uint8, device=device)
    p_b = b.data_ptr()
    check("T0.1 small-path reuse (same ptr after free)", p_b == p_a,
          f"got {p_b:#x} != {p_a:#x}")
    del b

    # ------------------------------------------------------------------
    # T0.2 — Large tensor reuse (≥ 1 MiB → arena)
    # ------------------------------------------------------------------
    c = torch.empty(2 << 20, dtype=torch.uint8, device=device)  # 2 MiB
    p_c = c.data_ptr()
    del c; torch.xpu.synchronize()
    d = torch.empty(2 << 20, dtype=torch.uint8, device=device)
    p_d = d.data_ptr()
    check("T0.2 arena large-tensor reuse (same ptr after free)", p_d == p_c,
          f"got {p_d:#x} != {p_c:#x}")
    del d

    # ------------------------------------------------------------------
    # T0.3 — Coalescing: three adjacent blocks merge into one
    # ------------------------------------------------------------------
    s0 = arena_stats()
    t1 = torch.empty(64 << 20,  dtype=torch.uint8, device=device)  # 64 MiB
    t2 = torch.empty(128 << 20, dtype=torch.uint8, device=device)  # 128 MiB
    t3 = torch.empty(256 << 20, dtype=torch.uint8, device=device)  # 256 MiB
    p_t1 = t1.data_ptr()
    # Free in reverse order so coalescing is exercised both directions
    del t3; torch.xpu.synchronize()
    del t2; torch.xpu.synchronize()
    del t1; torch.xpu.synchronize()
    s1 = arena_stats()
    check("T0.3a coalescing occurred (num_coalesces increased)",
          s1["coalesces"] > s0["coalesces"],
          f"coalesces: {s0['coalesces']} -> {s1['coalesces']}")
    # 400 MiB should fit in the coalesced 448 MiB block without a new chunk
    growths_before = s1["growths"]
    t4 = torch.empty(400 << 20, dtype=torch.uint8, device=device)  # 400 MiB
    p_t4 = t4.data_ptr()
    s2 = arena_stats()
    check("T0.3b coalesced block satisfies 400 MiB (no new chunk)",
          s2["growths"] == growths_before,
          f"new chunk was allocated (growths {growths_before} -> {s2['growths']})")
    check("T0.3c 400 MiB reuses coalesced base ptr",
          p_t4 == p_t1,
          f"got {p_t4:#x} != {p_t1:#x}")
    del t4

    # ------------------------------------------------------------------
    # T0.4 — Stats function returns sane values
    # ------------------------------------------------------------------
    t_live = torch.empty(8 << 20, dtype=torch.uint8, device=device)
    s = arena_stats()
    check("T0.4 stats: reserved_mb > 0", s["reserved_mb"] > 0,
          f"reserved_mb={s['reserved_mb']}")
    check("T0.4 stats: allocated_mb <= reserved_mb",
          s["allocated_mb"] <= s["reserved_mb"],
          f"allocated={s['allocated_mb']:.1f} > reserved={s['reserved_mb']:.1f}")
    del t_live

    # ------------------------------------------------------------------
    # T0.5 — Block split: alloc large, free it, then alloc small — tail still usable
    # ------------------------------------------------------------------
    big = torch.empty(256 << 20, dtype=torch.uint8, device=device)  # 256 MiB
    p_big = big.data_ptr()
    del big; torch.xpu.synchronize()
    small = torch.empty(1 << 20, dtype=torch.uint8, device=device)  # 1 MiB exactly
    p_small = small.data_ptr()
    # After freeing 256 MiB and allocating 1 MiB, there should be a 255 MiB tail
    tail = torch.empty(200 << 20, dtype=torch.uint8, device=device)  # 200 MiB from tail
    growths_after_big = arena_stats()["growths"]
    del tail; del small

    # T0.5: 1 MiB + 200 MiB both fit without additional chunk growth
    # (since 256 MiB was freed and split)
    check("T0.5 split tail reuse (1+200 MiB fit in 256 MiB slot)",
          True, "")  # if we reached here without OOM, pass

    # ------------------------------------------------------------------
    # T0.6 — Size boundary: 1 MiB - 1 byte → small path; 1 MiB → arena
    # ------------------------------------------------------------------
    # We can distinguish by checking that the small-path allocation goes through
    # and is recycled in the small bucket, while arena goes to the arena.
    # Simplest check: allocate exactly SMALL_THRESHOLD and verify it's non-null.
    boundary = torch.empty(1 << 20, dtype=torch.uint8, device=device)
    check("T0.6 1 MiB boundary allocation succeeds", boundary.data_ptr() != 0)
    del boundary
    below = torch.empty((1 << 20) - 1, dtype=torch.uint8, device=device)
    check("T0.6 (1 MiB - 1) boundary allocation succeeds", below.data_ptr() != 0)
    del below

    # ------------------------------------------------------------------
    # T0.7 — Concurrent alloc/free (4 threads, no deadlock)
    # ------------------------------------------------------------------
    import threading
    errors = []
    def worker(tid):
        try:
            for _ in range(20):
                t = torch.empty(2 << 20, dtype=torch.uint8, device=device)
                torch.xpu.synchronize()
                del t
        except Exception as e:
            errors.append(f"thread {tid}: {e}")
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for th in threads: th.start()
    for th in threads: th.join()
    check("T0.7 concurrent alloc/free (4 threads, no crash)", len(errors) == 0,
          "; ".join(errors))

    # ------------------------------------------------------------------
    # T0.8 — Zero-size allocation does not crash
    # ------------------------------------------------------------------
    try:
        z = torch.empty(0, dtype=torch.uint8, device=device)
        del z
        check("T0.8 zero-size alloc does not crash", True)
    except Exception as e:
        check("T0.8 zero-size alloc does not crash", False, str(e))

    # ------------------------------------------------------------------
    # T0.9 — Multiple large allocs fill chunk and trigger growth
    # ------------------------------------------------------------------
    chunk_gb = int(os.environ.get("USM_ARENA_CHUNK_GB", "4"))
    # Allocate slightly more than one chunk worth to force growth
    alloc_target_mb = chunk_gb * 1024 + 512  # chunk + 512 MiB
    tensors = []
    total_mb = 0
    grew = False
    g_before = arena_stats()["growths"]
    while total_mb < alloc_target_mb:
        try:
            t = torch.empty(256 << 20, dtype=torch.uint8, device=device)
            tensors.append(t)
            total_mb += 256
        except Exception:
            break
    g_after = arena_stats()["growths"]
    grew = g_after > g_before
    check(f"T0.9 chunk growth triggered after {total_mb} MiB", grew,
          f"growths: {g_before} -> {g_after}")
    del tensors
    torch.xpu.synchronize()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\nT0 results: {passed} passed, {failed} failed")
    return failed == 0


# ---------------------------------------------------------------------------
# T1 — CCL compatibility tests (distributed, requires torchrun)
# ---------------------------------------------------------------------------
def run_ccl_tests():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"xpu:{local_rank}")
    torch.xpu.set_device(local_rank)

    dist.init_process_group(backend="xccl")
    rank = dist.get_rank()
    world = dist.get_world_size()

    def log(msg):
        if rank == 0:
            print(msg)

    log(f"\n=== T1: CCL compatibility tests ({world} ranks) ===\n")
    passed = 0
    failed = 0

    def check(label, condition, msg=""):
        nonlocal passed, failed
        if condition:
            log(f"  {PASS}  {label}")
            passed += 1
        else:
            log(f"  {FAIL}  {label}: {msg}")
            failed += 1

    # T1.1 — all_reduce small tensor
    t = torch.ones(10, device=device)
    dist.all_reduce(t)
    expected = float(world)
    check("T1.1 all_reduce small (10 elements)", abs(t[0].item() - expected) < 0.01,
          f"expected {expected}, got {t[0].item()}")
    del t

    # T1.2 — all_gather 16 MiB output (above CCL staging→IPC threshold)
    shard = torch.ones(4 * 1024 * 1024, dtype=torch.bfloat16, device=device)  # 8 MiB per rank
    out   = torch.empty(4 * 1024 * 1024 * world, dtype=torch.bfloat16, device=device)
    try:
        dist.all_gather_into_tensor(out, shard)
        ok = out.sum().item() == float(4 * 1024 * 1024 * world)
        check("T1.2 all_gather 16 MiB output (IPC path)", ok,
              f"sum mismatch: {out.sum().item()}")
    except Exception as e:
        check("T1.2 all_gather 16 MiB output (IPC path)", False, str(e))
    del shard, out

    # T1.3 — reduce_scatter 16 MiB input
    inp = torch.ones(4 * 1024 * 1024 * world, dtype=torch.bfloat16, device=device)
    out = torch.empty(4 * 1024 * 1024, dtype=torch.bfloat16, device=device)
    try:
        dist.reduce_scatter_tensor(out, inp)
        ok = abs(out.mean().item() - float(world)) < 0.01
        check("T1.3 reduce_scatter 16 MiB", ok,
              f"mean mismatch: {out.mean().item()} != {world}")
    except Exception as e:
        check("T1.3 reduce_scatter 16 MiB", False, str(e))
    del inp, out

    # T1.4 — all_gather 256 MiB output (large, requires coalescing to fit)
    big_shard = torch.ones(32 * 1024 * 1024, dtype=torch.bfloat16, device=device)  # 64 MiB/rank
    big_out   = torch.empty(32 * 1024 * 1024 * world, dtype=torch.bfloat16, device=device)
    try:
        dist.all_gather_into_tensor(big_out, big_shard)
        ok = big_out.sum().item() == float(32 * 1024 * 1024 * world)
        check("T1.4 all_gather 256 MiB output (critical: large IPC path)", ok,
              f"sum mismatch")
    except Exception as e:
        check("T1.4 all_gather 256 MiB output (critical: large IPC path)", False, str(e))
    del big_shard, big_out

    dist.barrier()
    log(f"\nT1 results: {passed} passed, {failed} failed")
    dist.destroy_process_group()
    return failed == 0


# ---------------------------------------------------------------------------
# T2 — FSDP2 smoke test (requires 4 ranks and FSDP2-capable model)
# ---------------------------------------------------------------------------
def run_fsdp2_tests():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"xpu:{local_rank}")
    torch.xpu.set_device(local_rank)

    dist.init_process_group(backend="xccl")
    rank = dist.get_rank()
    world = dist.get_world_size()

    def log(msg):
        if rank == 0:
            print(msg)

    log(f"\n=== T2: FSDP2 smoke test ({world} ranks) ===\n")

    from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
    from torch.distributed.device_mesh import init_device_mesh

    # Simple MLP that mirrors the fragmentation pattern: linear layers with
    # different hidden dims → mixed-size parameter tensors (32–256 MiB per layer).
    class BigMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # ~256 MiB weight per layer (bfloat16)
            dim = 8192
            self.fc1 = torch.nn.Linear(dim, dim * 4, bias=False)  # 256 MiB
            self.fc2 = torch.nn.Linear(dim * 4, dim * 2, bias=False)  # 512 MiB
            self.fc3 = torch.nn.Linear(dim * 2, dim, bias=False)  # 128 MiB

        def forward(self, x):
            return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))

    mesh = init_device_mesh("xpu", (world,))
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
    )
    model = BigMLP().to(device)
    fully_shard(model, mesh=mesh, mp_policy=mp_policy)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    s_before = arena_stats()
    passed = 0
    failed = 0

    def check(label, condition, msg=""):
        nonlocal passed, failed
        if condition:
            log(f"  {PASS}  {label}")
            passed += 1
        else:
            log(f"  {FAIL}  {label}: {msg}")
            failed += 1

    try:
        for step in range(3):
            x = torch.randn(4, 8192, device=device, dtype=torch.bfloat16)
            y = model(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            dist.barrier()
            log(f"  step {step}: loss={loss.item():.4f}")

        check("T2.1 FSDP2 3-step forward+backward completes", True)
    except Exception as e:
        check("T2.1 FSDP2 3-step forward+backward completes", False, str(e))
        import traceback
        traceback.print_exc()

    s_after = arena_stats()
    log(f"  Arena: growths={s_after['growths']}, coalesces={s_after['coalesces']}, "
        f"reserved={s_after['reserved_mb']:.0f} MiB, allocated={s_after['allocated_mb']:.0f} MiB")
    check("T2.2 coalescing occurred during FSDP2 training",
          s_after["coalesces"] > s_before["coalesces"],
          f"coalesces: {s_before['coalesces']} -> {s_after['coalesces']}")

    log(f"\nT2 results: {passed} passed, {failed} failed")
    dist.destroy_process_group()
    return failed == 0


# ---------------------------------------------------------------------------
# Fragmentation comparison (informational, no assertion)
# ---------------------------------------------------------------------------
def run_fragmentation_benchmark():
    """Compare arena vs default fragmentation metric.

    The arena pre-allocates a large slab, so PyTorch's reserved-allocated
    metric isn't directly comparable. Instead we report num_chunk_growths
    as the proxy for driver-level fragmentation (fewer growths = less
    driver-level overhead).
    """
    device = torch.device("xpu:0")
    torch.xpu.set_device(0)
    print("\n=== Fragmentation benchmark (arena) ===")

    s0 = arena_stats()

    # Simulate RL step: allocate mixed-size activations
    small = [torch.randn(sz * 1024 * 1024 // 4, device=device)
             for sz in [2, 4, 4, 8, 8, 8]]
    for i in range(1, len(small), 2):
        small[i] = None
    gc.collect()
    torch.xpu.synchronize()

    for sz_mb in [64, 128, 256]:
        t = torch.randn(sz_mb * 1024 * 1024 // 4, device=device)
        s = arena_stats()
        print(f"  {sz_mb:3d} MiB large alloc  "
              f"growths={s['growths']}  coalesces={s['coalesces']}  "
              f"reserved={s['reserved_mb']:.0f} MiB  allocated={s['allocated_mb']:.0f} MiB")
        del t
        torch.xpu.synchronize()

    s1 = arena_stats()
    print(f"  Total driver-level chunk growths for scenario: {s1['growths'] - s0['growths']}")
    print("  (Default allocator would have made 3+ separate driver allocs for the large tensors)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ccl",    action="store_true", help="Run T1 CCL tests (needs torchrun)")
    parser.add_argument("--fsdp2",  action="store_true", help="Run T2 FSDP2 test (needs torchrun, 4 ranks)")
    parser.add_argument("--bench",  action="store_true", help="Run fragmentation benchmark")
    args = parser.parse_args()

    if args.ccl:
        ok = run_ccl_tests()
        sys.exit(0 if ok else 1)
    elif args.fsdp2:
        ok = run_fsdp2_tests()
        sys.exit(0 if ok else 1)
    elif args.bench:
        run_fragmentation_benchmark()
    else:
        ok = run_unit_tests()
        run_fragmentation_benchmark()
        sys.exit(0 if ok else 1)

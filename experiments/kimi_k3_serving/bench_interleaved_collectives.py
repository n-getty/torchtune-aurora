"""Why is a K3 decode step 1249 ms when its parts only add to ~534 ms?

Measured so far, per token:
    device compute      138 ms   (from the trace)
    kernel dispatch     138 ms   (21,293 launches x 6.5 us, measured on-node;
                                  unaffected by ZE_ENABLE_API_TRACING)
    collectives         258 ms   (463 x 0.557 ms, measured ISOLATED)
    -----------------------------------------------------------------
    unexplained         715 ms   (57% of the step)

The isolated collective benchmark synchronized before every timing, so it
measured a collective on an idle queue with all 32 ranks already at the
barrier. A real decode step is nothing like that: each rank issues ~46 kernel
launches between consecutive collectives, and `CCL_OP_SYNC=1` (set in every
K3 run) makes each collective BLOCKING. So every one of the 463 collectives
is a 32-rank rendezvous that cannot complete until the slowest rank has
drained its queued work.

That turns per-rank skew into step time: cost is not 463 x (network latency)
but 463 x (max over ranks of pending work + network). This measures exactly
that difference.

Three regimes, same collective count:
  isolated    -- sync, then time the collective alone (reproduces the 0.557 ms)
  interleaved -- queue K tiny kernels, then collective, no sync between
                 (reproduces the real decode pattern)
  skewed      -- as interleaved, but rank r queues extra work proportional to
                 r, to measure how hard the barrier amplifies imbalance

If `interleaved` >> `isolated` + (K x 6.5 us), the gap is rendezvous/skew
cost, and the lever is REDUCING SYNC POINTS (fusing collectives, or
CCL_OP_SYNC=0 to let them overlap) rather than shrinking dispatch.

Launch: mpiexec -n 32 -ppn 11 --pmi=pmix python bench_interleaved_collectives.py
"""

import json
import os
import socket
import statistics
import sys
import time

import torch
import torch.distributed as dist

K3_HIDDEN = 3584
# 21,293 launches / 463 collectives ~= 46 kernels between consecutive
# collectives in a real K3 decode step.
KERNELS_BETWEEN = int(os.environ.get("KERNELS_BETWEEN", "46"))
ITERS = int(os.environ.get("BENCH_ITERS", "100"))
WARMUP = 10
COLLECTIVES_PER_STEP = 463


def main() -> int:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank, world = comm.Get_rank(), comm.Get_size()
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world))
    os.environ.setdefault("MASTER_ADDR", comm.bcast(socket.gethostname(), root=0))
    os.environ.setdefault("MASTER_PORT", "29762")
    local_rank = int(os.environ.get("PALS_LOCAL_RANKID", rank % 12))
    torch.xpu.set_device(local_rank)
    comm.Barrier()
    dist.init_process_group(backend="xccl", rank=rank, world_size=world)

    dev = f"xpu:{local_rank}"
    msg = torch.ones(K3_HIDDEN, dtype=torch.bfloat16, device=dev)
    work = torch.ones(112, dtype=torch.bfloat16, device=dev)

    def isolated():
        torch.xpu.synchronize()
        dist.barrier()
        t = time.perf_counter()
        dist.all_reduce(msg)
        torch.xpu.synchronize()
        return (time.perf_counter() - t) * 1e3

    def interleaved(extra=0):
        # No sync before timing: the collective must drain whatever this rank
        # has queued, exactly like the real decode loop.
        t = time.perf_counter()
        w = work
        for _ in range(KERNELS_BETWEEN + extra):
            w = w * 1.0001
        dist.all_reduce(msg)
        torch.xpu.synchronize()
        return (time.perf_counter() - t) * 1e3

    for _ in range(WARMUP):
        isolated()
        interleaved()

    modes = {
        "isolated": lambda: isolated(),
        "interleaved": lambda: interleaved(),
        # Rank-proportional extra work: 0..31 extra kernels. Mean extra work is
        # only ~16 kernels (~0.1 ms), so any large slowdown is the barrier
        # amplifying imbalance, not the work itself.
        "skewed": lambda: interleaved(extra=rank),
    }

    out = {}
    for name, fn in modes.items():
        dist.barrier()
        samples = sorted(fn() for _ in range(ITERS))
        stats = {
            "median_ms": samples[len(samples) // 2],
            "mean_ms": statistics.fmean(samples),
            "p90_ms": samples[min(len(samples) - 1, int(len(samples) * 0.9))],
        }
        gathered = comm.gather(stats, root=0)
        if rank == 0:
            worst = max(gathered, key=lambda s: s["median_ms"])
            out[name] = worst

    if rank == 0:
        kernels_ms = KERNELS_BETWEEN * 6.5 / 1000  # measured 6.5 us/launch
        print(f"world_size={world} kernels_between={KERNELS_BETWEEN} iters={ITERS}")
        print(f"(slowest rank; {KERNELS_BETWEEN} launches ~= {kernels_ms:.2f} ms of dispatch)")
        print()
        print(f"{'mode':<14}{'median_ms':>10}{'p90_ms':>9}{'x463_ms':>10}{'vs 1249ms':>11}")
        for name, s in out.items():
            per_step = s["median_ms"] * COLLECTIVES_PER_STEP
            print(f"{name:<14}{s['median_ms']:>10.3f}{s['p90_ms']:>9.3f}"
                  f"{per_step:>10.0f}{per_step / 1249:>10.0%}")

        iso, inter = out["isolated"]["median_ms"], out["interleaved"]["median_ms"]
        excess = inter - iso - kernels_ms
        print()
        print(f"interleaved - isolated - dispatch = {excess:.3f} ms per collective")
        print(f"  x463 = {excess * COLLECTIVES_PER_STEP:.0f} ms/token of pure rendezvous cost")
        print(f"  (the unexplained remainder of the 1249 ms step is ~715 ms)")
        if excess * COLLECTIVES_PER_STEP > 300:
            print("  => RENDEZVOUS-DOMINATED. The lever is FEWER SYNC POINTS")
            print("     (fuse collectives; try CCL_OP_SYNC=0 so they overlap),")
            print("     not fewer kernels and not fewer bytes.")
        elif excess * COLLECTIVES_PER_STEP < 100:
            print("  => rendezvous is cheap; the 715 ms is elsewhere. Do not")
            print("     pursue collective fusion on the strength of this.")
        else:
            print("  => partial explanation only; keep looking.")

        skew = out["skewed"]["median_ms"]
        print()
        print(f"skew amplification: skewed/interleaved = {skew / inter:.2f}x")
        print("  (mean extra work is only ~16 launches ~= 0.10 ms, so a large")
        print("   ratio means the barrier is charging every rank for the straggler)")

        dest = os.environ.get("BENCH_OUT")
        if dest:
            with open(dest, "w") as handle:
                json.dump({"world_size": world, "kernels_between": KERNELS_BETWEEN,
                           "modes": out}, handle, indent=2)
            print(f"\nwrote {dest}")

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())

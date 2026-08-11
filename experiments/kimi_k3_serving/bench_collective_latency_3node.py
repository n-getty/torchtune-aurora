"""Measure the ONE number the K3 single-user plan is built on.

The plan estimates the c=1 decode step as ~463 collectives x 1-2 ms =
460-930 ms of a 1249 ms step, and that estimate alone decides whether to cut
the collective count (Step 2) or pursue graph capture (Step 5) -- capture
replays the same round-trips, so if collectives dominate it cannot reach the
target. The "1-2 ms" was never measured on this topology.

The Kineto route to this number failed twice on job 8748640: the traces
contain zero CPU rows and zero collective kernels despite
`distributedInfo: backend=xccl, world_size=32, pg_count=263`, so 91.5% of
wall time is unattributed gap that could be host dispatch OR blocking inside
oneCCL. This measures the collective directly instead, with no profiler in
the path at all.

What it does: stand up the real 32-rank / 3-node XCCL group at the K3
topology and time `all_reduce` at K3's actual message size (hidden 3584 x
bf16 = 7 KiB for one token; 14 KiB is the plan's figure, so both are swept)
plus a size ladder to separate fixed per-collective latency from bandwidth.

Reading the result:
  * per-call latency x 463 vs the ~1249 ms step gives the collective share
    directly -- the number the trace could not produce.
  * if latency is flat across 7 KiB -> 1 MiB, the cost is per-collective
    overhead and REMOVING collectives (Step 2) is the only lever that helps;
    shrinking bytes does nothing. That would match the already-measured
    WS6/WS7/FP8-wire null results on this fabric.

Launch (inside a 3-node allocation, 12 ranks/node = 36 > 32, so pin 32):
  mpiexec -n 32 -ppn 11 --pmi=pmix python bench_collective_latency_3node.py
"""

import json
import os
import socket
import statistics
import sys
import time

import torch
import torch.distributed as dist

# K3 at TP=32: hidden_size 3584, bf16. One token's activation is the message
# every per-layer all_reduce carries at c=1.
K3_HIDDEN = 3584
SIZES_BYTES = [
    K3_HIDDEN * 2,        # 7 KiB  -- one bf16 token, the real c=1 message
    K3_HIDDEN * 2 * 2,    # 14 KiB -- the plan's stated figure
    64 * 1024,
    256 * 1024,
    1024 * 1024,
    4 * 1024 * 1024,
]
# ~463 collectives per decode step, counted from the model definition:
# 93 attention o_proj + 92 MoE-final + 92 routed_expert_up_proj + 92
# shared-expert down_proj all_reduce, + 92 routed_expert_down_proj
# all_gather, + embedding/logits.
COLLECTIVES_PER_STEP = 463
MEASURED_STEP_MS = 1249.0


def bench_one(tensor: torch.Tensor, iters: int, warmup: int) -> dict:
    for _ in range(warmup):
        dist.all_reduce(tensor)
    torch.xpu.synchronize()
    dist.barrier()

    samples = []
    for _ in range(iters):
        torch.xpu.synchronize()
        start = time.perf_counter()
        dist.all_reduce(tensor)
        torch.xpu.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)

    samples.sort()
    return {
        "mean_ms": statistics.fmean(samples),
        "median_ms": samples[len(samples) // 2],
        "p10_ms": samples[max(0, int(len(samples) * 0.10))],
        "p90_ms": samples[min(len(samples) - 1, int(len(samples) * 0.90))],
        "min_ms": samples[0],
        "max_ms": samples[-1],
    }


def main() -> int:
    iters = int(os.environ.get("BENCH_ITERS", "200"))
    warmup = int(os.environ.get("BENCH_WARMUP", "20"))

    # Match the K3 server's own init path (see xpu_worker.init_device).
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank, world = comm.Get_rank(), comm.Get_size()
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world))
    os.environ.setdefault("MASTER_ADDR", comm.bcast(socket.gethostname(), root=0))
    os.environ.setdefault("MASTER_PORT", "29761")
    local_rank = int(os.environ.get("PALS_LOCAL_RANKID", rank % 12))
    torch.xpu.set_device(local_rank)

    comm.Barrier()
    dist.init_process_group(backend="xccl", rank=rank, world_size=world)

    if rank == 0:
        print(f"world_size={world} backend=xccl device=xpu:{local_rank}")
        print(f"iters={iters} warmup={warmup}")
        print(f"{'bytes':>10} {'median_ms':>10} {'mean_ms':>9} {'p10':>8} "
              f"{'p90':>8} {'x463_ms':>9} {'share_of_1249ms':>16}")

    results = []
    for nbytes in SIZES_BYTES:
        tensor = torch.ones(nbytes // 2, dtype=torch.bfloat16, device=f"xpu:{local_rank}")
        stats = bench_one(tensor, iters, warmup)
        stats["bytes"] = nbytes
        # Report the SLOWEST rank: a collective is only as fast as its
        # straggler, and averaging across ranks hides exactly that.
        gathered = comm.gather(stats, root=0)
        if rank == 0:
            worst = max(gathered, key=lambda s: s["median_ms"])
            per_step = worst["median_ms"] * COLLECTIVES_PER_STEP
            worst["slowest_rank_median_ms"] = worst["median_ms"]
            worst["projected_per_step_ms"] = per_step
            worst["projected_share_of_step"] = per_step / MEASURED_STEP_MS
            results.append(worst)
            print(f"{nbytes:>10} {worst['median_ms']:>10.3f} "
                  f"{worst['mean_ms']:>9.3f} {worst['p10_ms']:>8.3f} "
                  f"{worst['p90_ms']:>8.3f} {per_step:>9.1f} "
                  f"{per_step / MEASURED_STEP_MS:>15.1%}")
        del tensor

    if rank == 0:
        small = results[1]["median_ms"]          # 14 KiB -- K3's actual size
        large = results[-1]["median_ms"]         # 4 MiB  -- far off K3's point
        print("")
        print(f"4MiB/14KiB latency ratio = {large / small:.2f}x  (context only)")

        # Judge flatness IN K3'S REGIME, not against 4 MiB. The first run
        # (job 8748815) reported "bandwidth matters" purely because the 4 MiB
        # point is 6.4x the 14 KiB point -- but K3 never sends 4 MiB at c=1,
        # and across 7 KiB -> 256 KiB (a 36x byte increase) latency moved only
        # 0.544 -> 0.680 ms. Comparing against a size the workload does not
        # use answers a question nobody asked.
        in_regime = [r for r in results if r["bytes"] <= 256 * 1024]
        lo = in_regime[0]
        hi = in_regime[-1]
        byte_factor = hi["bytes"] / lo["bytes"]
        time_factor = hi["median_ms"] / lo["median_ms"]
        print(
            f"in-regime ({lo['bytes']}B -> {hi['bytes']}B, {byte_factor:.0f}x "
            f"bytes): latency {lo['median_ms']:.3f} -> {hi['median_ms']:.3f} ms "
            f"= {time_factor:.2f}x"
        )
        if time_factor < 1.5:
            print("  -> LATENCY-BOUND at K3's message size: a large byte")
            print("     increase barely moves the time, so cost is")
            print("     per-collective overhead. Only REMOVING collectives")
            print("     helps; shrinking messages will not. (Consistent with")
            print("     the measured WS6/WS7/FP8-wire null results.)")
        else:
            print("  -> bandwidth already matters inside K3's own size range.")

        share = results[1]["projected_share_of_step"]
        print("")
        print(f"PROJECTED collective share of the 1249 ms c=1 step: {share:.1%}")
        print("  (= 463 collectives x the 14 KiB median, slowest rank)")
        if share > 0.40:
            print("  VERDICT-INPUT: consistent with CUT-COLLECTIVES (Step 2).")
        elif share < 0.15:
            print("  VERDICT-INPUT: collectives are NOT the dominant term --")
            print("  the plan's 460-930 ms estimate is refuted; look at")
            print("  dispatch/kernels instead.")
        else:
            print("  VERDICT-INPUT: intermediate; neither branch is clear-cut.")
        print("")
        print("CAVEAT: this is an isolated microbenchmark on an idle fabric. A")
        print("real decode step interleaves collectives with compute and other")
        print("ranks' traffic, so treat this as a LOWER bound on per-collective")
        print("cost, not a measurement of the step itself.")

        out = os.environ.get("BENCH_OUT")
        if out:
            with open(out, "w") as handle:
                json.dump({"world_size": world, "results": results}, handle, indent=2)
            print(f"wrote {out}")

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())

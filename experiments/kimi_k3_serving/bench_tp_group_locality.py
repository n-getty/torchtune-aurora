"""Does making the TP group NODE-LOCAL make its all_reduce cheaper?

WHY. The c=1 decode step spends ~207 ms (24%) in ~371 all_reduces across a
32-rank TP group that spans 3 nodes, so every one crosses Slingshot.
`PLAN_C1_ROOT_CAUSES.md` §3 considered only PP=2 x TP=16 and assumed halving
the group width halves the cost. But TP=16 on 12-tile nodes still straddles a
node boundary, so it would remain a fabric collective.

The configuration the plan never considered is **TP=12 x PP=3 = 36 ranks**,
which is exactly 3 nodes x 12 tiles and makes every TP group fit INSIDE one
node. Then all 371 all_reduces become intra-node (Xe Link), and only the 2
PP boundary sends per token cross the fabric.

That is a categorically different change from "fewer ranks", and its whole
value rests on one unmeasured number: **is a 12-rank intra-node all_reduce
actually much faster than a 32-rank cross-node one at K3's 7-14 KiB size?**

It might not be. `project_k3_collective_latency_measured_20260811` established
that this fabric is LATENCY-bound in K3's size regime -- 37x more bytes cost
only 1.25x more time. If the 0.557 ms is dominated by fixed per-collective
software overhead (oneCCL dispatch, not wire time), then going intra-node
will barely move it and TP=12xPP=3 is not worth a K3 hold. Measure first.

WHAT IT MEASURES. Inside one 3-node allocation, builds several XCCL subgroups
and times all_reduce on each at K3's real message sizes:

  tp32_3node   32 ranks over 3 nodes  -- today's production topology
  tp16_2node   16 ranks over 2 nodes  -- the PP=2 x TP=16 the plan proposed
  tp12_local   12 ranks inside 1 node -- the TP=12 x PP=3 candidate
  tp8_local     8 ranks inside 1 node -- width vs locality discriminator
  pp_pair       2 ranks on 2 nodes    -- the PP boundary send, the new cost

All groups are built by every rank in the same order (XCCL requires collective
participation in `new_group`), and each is timed only by its members.

READING IT. Let L = tp12_local median, X = tp32_3node median.

  * L / X <= ~0.4  -> locality is real. Projected collectives 207 ms ->
    371*L + 2*(pp_pair). Worth a K3 hold to bring up TP=12 x PP=3.
  * L / X >= ~0.8  -> the cost is fixed per-collective overhead, not the
    wire. Locality buys nothing; DROP the parallelism lever entirely and say
    so. This is the outcome the latency-bound finding predicts.

tp8_local vs tp12_local separates "fewer ranks helped" from "staying on-node
helped" -- if tp8 ~= tp12 the win is locality; if tp8 << tp12 it is width.

CAVEAT (inherited from the 3-node bench, and it applies here too): an
isolated microbenchmark on an idle fabric is a LOWER bound on per-collective
cost. It is a valid RATIO test between topologies, which is all this decides;
it is not a prediction of the step time.

Launch (inside a >=3-node allocation, 12 ranks/node):
  mpiexec -n 36 -ppn 12 --pmi=pmix python bench_tp_group_locality.py
"""

import json
import os
import socket
import statistics
import sys
import time

import torch
import torch.distributed as dist

# K3 at TP=32: hidden 3584 bf16 = 7 KiB is one token's activation. At TP=12
# the per-rank shard is larger but the all_reduce message is the FULL hidden
# vector either way -- an all_reduce sums same-shaped tensors -- so the
# message size is a property of the model, not of TP. Same sizes are correct
# for every group here, which is what makes the comparison apples-to-apples.
K3_HIDDEN = 3584
SIZES_BYTES = [K3_HIDDEN * 2, K3_HIDDEN * 2 * 2, 64 * 1024]

# 371 remaining all_reduces/step after the shared-expert AR fusion removed 92
# of the original 463 (project_k3_eager_ceiling_and_first_win_20260811).
COLLECTIVES_PER_STEP = 371
# Today's measured c=1 step under eager, broken fused-KDA kernel OFF.
MEASURED_STEP_MS = 864.0


def bench_one(tensor, group, iters, warmup):
    for _ in range(warmup):
        dist.all_reduce(tensor, group=group)
    torch.xpu.synchronize()
    dist.barrier(group=group)

    samples = []
    for _ in range(iters):
        torch.xpu.synchronize()
        start = time.perf_counter()
        dist.all_reduce(tensor, group=group)
        torch.xpu.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)

    samples.sort()
    return {
        "mean_ms": statistics.fmean(samples),
        "median_ms": samples[len(samples) // 2],
        "p90_ms": samples[min(len(samples) - 1, int(len(samples) * 0.90))],
        "min_ms": samples[0],
    }


def main() -> int:
    iters = int(os.environ.get("BENCH_ITERS", "200"))
    warmup = int(os.environ.get("BENCH_WARMUP", "20"))

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank, world = comm.Get_rank(), comm.Get_size()
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world))
    os.environ.setdefault("MASTER_ADDR", comm.bcast(socket.gethostname(), root=0))
    os.environ.setdefault("MASTER_PORT", "29763")
    local_rank = int(os.environ.get("PALS_LOCAL_RANKID", rank % 12))
    torch.xpu.set_device(local_rank)

    host = socket.gethostname()
    hosts = comm.allgather(host)
    # Rank -> node index, in rank order. With -ppn 12 this is rank//12, but
    # deriving it from the actual hostnames means a different -ppn cannot
    # silently mislabel a group as "local" when it is not.
    node_of_rank = []
    seen = {}
    for h in hosts:
        if h not in seen:
            seen[h] = len(seen)
        node_of_rank.append(seen[h])
    nnodes = len(seen)

    comm.Barrier()
    dist.init_process_group(backend="xccl", rank=rank, world_size=world)

    ranks_on_node0 = [r for r in range(world) if node_of_rank[r] == 0]
    ranks_on_first2 = [r for r in range(world) if node_of_rank[r] < 2]
    first_of_node = [
        min(r for r in range(world) if node_of_rank[r] == n) for n in range(nnodes)
    ]

    specs = []
    if world >= 32:
        specs.append(("tp32_3node", list(range(32))))
    if len(ranks_on_first2) >= 16:
        specs.append(("tp16_2node", ranks_on_first2[:16]))
    if len(ranks_on_node0) >= 12:
        specs.append(("tp12_local", ranks_on_node0[:12]))
    if len(ranks_on_node0) >= 8:
        specs.append(("tp8_local", ranks_on_node0[:8]))
    if nnodes >= 2:
        specs.append(("pp_pair", [first_of_node[0], first_of_node[1]]))

    if rank == 0:
        print(f"world={world} nodes={nnodes} hosts={sorted(seen)}")
        print(f"iters={iters} warmup={warmup}")
        for name, rs in specs:
            spanned = sorted({node_of_rank[r] for r in rs})
            print(f"  {name:<12} ranks={len(rs):<3} spans_nodes={spanned}")
        sys.stdout.flush()

    # EVERY rank must call new_group for EVERY group, in the same order.
    groups = {}
    for name, rs in specs:
        groups[name] = dist.new_group(ranks=rs)

    results = {}
    for name, rs in specs:
        group = groups[name]
        member = rank in rs
        for nbytes in SIZES_BYTES:
            stats = None
            if member:
                tensor = torch.ones(
                    nbytes // 2, dtype=torch.bfloat16, device=f"xpu:{local_rank}"
                )
                stats = bench_one(tensor, group, iters, warmup)
                stats["bytes"] = nbytes
                del tensor
            # Report the SLOWEST member: a collective is only as fast as its
            # straggler; averaging hides exactly that.
            gathered = [s for s in comm.gather(stats, root=0) or [] if s]
            if rank == 0 and gathered:
                worst = max(gathered, key=lambda s: s["median_ms"])
                results.setdefault(name, {})[nbytes] = worst
        comm.Barrier()

    if rank == 0:
        print()
        hdr = f"{'group':<12}" + "".join(f"{b // 1024 or 7:>10}KiB" for b in SIZES_BYTES)
        print(hdr)
        for name, _ in specs:
            row = f"{name:<12}"
            for nbytes in SIZES_BYTES:
                r = results.get(name, {}).get(nbytes)
                row += f"{r['median_ms']:>13.3f}" if r else f"{'-':>13}"
            print(row)

        ref = results.get("tp32_3node", {}).get(SIZES_BYTES[1])
        loc = results.get("tp12_local", {}).get(SIZES_BYTES[1])
        w8 = results.get("tp8_local", {}).get(SIZES_BYTES[1])
        pp = results.get("pp_pair", {}).get(SIZES_BYTES[1])

        print()
        if ref and loc:
            ratio = loc["median_ms"] / ref["median_ms"]
            print(f"14KiB: tp32_3node={ref['median_ms']:.3f} ms  "
                  f"tp12_local={loc['median_ms']:.3f} ms  ratio={ratio:.2f}x")

            today = ref["median_ms"] * COLLECTIVES_PER_STEP
            # PP=3 adds 2 boundary sends/token; all_reduce on a 2-rank group
            # is a generous stand-in for a point-to-point send.
            proj = loc["median_ms"] * COLLECTIVES_PER_STEP
            if pp:
                proj += 2 * pp["median_ms"]
            saved = today - proj
            print(f"  projected collectives/step: {today:.0f} ms -> {proj:.0f} ms "
                  f"(saves {saved:.0f} ms)")
            if MEASURED_STEP_MS - saved > 0:
                new_step = MEASURED_STEP_MS - saved
                print(f"  step {MEASURED_STEP_MS:.0f} -> {new_step:.0f} ms = "
                      f"{1000 / new_step:.3f} tok/s "
                      f"({MEASURED_STEP_MS / new_step:.2f}x)")

            print()
            if ratio <= 0.4:
                print("VERDICT-INPUT: LOCALITY IS REAL. An intra-node TP group is")
                print("  much cheaper. TP=12 x PP=3 is worth a K3 bring-up.")
            elif ratio >= 0.8:
                print("VERDICT-INPUT: LOCALITY BUYS NOTHING. The cost is fixed")
                print("  per-collective overhead, not wire time -- consistent with")
                print("  the latency-bound finding (37x bytes -> 1.25x time).")
                print("  DROP the TP/PP re-topology lever; do not spend a hold.")
            else:
                print("VERDICT-INPUT: partial. Weigh the saving against a")
                print("  multi-hour PP bring-up before committing hold time.")

        if w8 and loc:
            print()
            print(f"  tp8_local={w8['median_ms']:.3f} vs tp12_local="
                  f"{loc['median_ms']:.3f} ms -> "
                  f"{'width matters too' if w8['median_ms'] < 0.8 * loc['median_ms'] else 'width is NOT the driver; locality is'}")

        print()
        print("CAVEAT: isolated microbenchmark on an idle fabric = LOWER bound on")
        print("per-collective cost. Valid as a RATIO test between topologies,")
        print("which is all it is used for; not a prediction of step time.")

        out = os.environ.get("BENCH_OUT")
        if out:
            with open(out, "w") as handle:
                json.dump(
                    {"world_size": world, "nodes": nnodes,
                     "results": {k: {str(b): v for b, v in d.items()}
                                 for k, d in results.items()}},
                    handle, indent=2,
                )
            print(f"wrote {out}")

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())

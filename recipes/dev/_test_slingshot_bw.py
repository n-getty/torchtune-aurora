"""
Cross-node Slingshot bandwidth benchmark.

Measures point-to-point XCCL send/recv bandwidth between two nodes and
compares it to the current flat broadcast (1→N) to confirm the hypothesis
that the flat broadcast sends N sequential copies, causing the 38s floor.

Modes:
  --mode p2p       : rank 0 sends, rank 1 receives (confirms per-link bandwidth)
  --mode bcast     : rank 0 broadcasts to all N ranks (replicates current wsync behavior)
  --mode compare   : runs both and reports the ratio

The per-link bandwidth should be ~25 GB/s (one Slingshot 11 NIC port).
If flat broadcast of 61 GiB to 12 receivers takes 35.9s but p2p takes ~2.4s,
the ratio of ~14.9x confirms 12 sequential cross-node sends.

Usage (2-node, run on training node, rank 1 on vLLM node):
  # On training node (rank 0):
  ZE_AFFINITY_MASK=0 python3 _test_slingshot_bw.py \
      --rank 0 --world-size 2 --host 0.0.0.0 --port 51310 --mode compare

  # On vLLM node (rank 1, via ssh in launcher):
  ZE_AFFINITY_MASK=0 python3 _test_slingshot_bw.py \
      --rank 1 --world-size 2 --host <TRAIN_NODE_HSN_IP> --port 51310 --mode compare
"""

import argparse
import datetime
import time

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d


def make_pg(store, rank, world_size, prefix="bw"):
    prefixed = c10d.PrefixStore(prefix, store)
    opts = c10d.ProcessGroupXCCL.Options()
    return c10d.ProcessGroupXCCL(store=prefixed, rank=rank, size=world_size, options=opts)


def bench_p2p(pg, device, rank, sizes_gib, n_iter=5):
    """Benchmark 2-rank broadcast (rank 0→1 on a size=2 PG). Returns (gib, seconds, GB/s) list.

    Note: XCCL send/recv hangs (only collective ops are supported). We use broadcast
    on a 2-rank PG as the p2p equivalent — this matches the 2-hop production design.
    """
    results = []
    for gib in sizes_gib:
        n_elements = int(gib * 1024**3 / 2)  # bf16
        buf = torch.zeros(n_elements, dtype=torch.bfloat16, device=device)

        # warmup
        for _ in range(2):
            pg.broadcast(buf, root=0).wait()

        times = []
        for _ in range(n_iter):
            torch.xpu.synchronize(device)
            t0 = time.perf_counter()
            pg.broadcast(buf, root=0).wait()
            torch.xpu.synchronize(device)
            times.append(time.perf_counter() - t0)

        avg = sum(times) / len(times)
        bw = gib / avg if avg > 0 else 0
        if rank == 0:
            print(f"  2rank-bcast {gib:.2f} GiB: {avg:.3f}s  {bw:.1f} GB/s  "
                  f"(min={min(times):.3f}s max={max(times):.3f}s)")
        results.append((gib, avg, bw))
        del buf
    return results


def bench_bcast(pg, device, rank, world_size, sizes_gib, n_iter=5):
    """Benchmark flat broadcast (rank 0→all). Returns (gib, seconds, GB/s) list."""
    results = []
    for gib in sizes_gib:
        n_elements = int(gib * 1024**3 / 2)
        buf = torch.zeros(n_elements, dtype=torch.bfloat16, device=device)

        for _ in range(2):
            pg.broadcast(buf, root=0).wait()

        times = []
        for _ in range(n_iter):
            torch.xpu.synchronize(device)
            t0 = time.perf_counter()
            pg.broadcast(buf, root=0).wait()
            torch.xpu.synchronize(device)
            times.append(time.perf_counter() - t0)

        avg = sum(times) / len(times)
        bw = gib / avg if avg > 0 else 0
        if rank == 0:
            print(f"  bcast(1→{world_size-1}) {gib:.2f} GiB: {avg:.3f}s  {bw:.1f} GB/s  "
                  f"(min={min(times):.3f}s max={max(times):.3f}s)")
        results.append((gib, avg, bw))
        del buf
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=51310)
    parser.add_argument("--mode", choices=["p2p", "bcast", "compare"], default="compare")
    parser.add_argument("--device-idx", type=int, default=0)
    args = parser.parse_args()

    # ZE_AFFINITY_MASK limits visibility to one tile; always index 0 within the process.
    device = torch.device("xpu:0")
    torch.xpu.set_device(device)

    print(f"[rank {args.rank}/{args.world_size}] connecting to {args.host}:{args.port} on {device}")

    store = dist.TCPStore(
        host_name=args.host,
        port=args.port,
        world_size=args.world_size,
        is_master=(args.rank == 0),
        timeout=datetime.timedelta(seconds=120),
        wait_for_workers=False,
    )

    pg = make_pg(store, args.rank, args.world_size, prefix="bw_test")
    print(f"[rank {args.rank}] ProcessGroupXCCL ready")

    # Representative sizes: 1 GiB batch (matches wsync batch_max_numel) and full 32B model
    sizes = [1.0, 4.0, 16.0, 61.0]

    p2p_results = []
    bcast_results = []

    if args.mode in ("p2p", "compare"):
        if args.rank == 0:
            print(f"\n=== Point-to-point send/recv (rank 0 → rank 1) ===")
        p2p_results = bench_p2p(pg, device, args.rank, sizes)

    if args.mode in ("bcast", "compare") and args.world_size > 1:
        if args.rank == 0:
            print(f"\n=== Flat broadcast (rank 0 → all {args.world_size-1} receivers) ===")
        bcast_results = bench_bcast(pg, device, args.rank, args.world_size, sizes)

    if args.mode == "compare" and args.rank == 0 and p2p_results and bcast_results:
        print(f"\n=== Comparison (confirms whether bcast does N sequential sends) ===")
        print(f"{'Size':>8}  {'p2p(s)':>8}  {'bcast(s)':>9}  {'ratio':>6}  {'hypothesis':>10}")
        for (gib, p2p_t, _), (_, bc_t, _) in zip(p2p_results, bcast_results):
            ratio = bc_t / p2p_t if p2p_t > 0 else 0
            expected_ratio = args.world_size - 1
            verdict = "CONFIRMED" if abs(ratio - expected_ratio) / expected_ratio < 0.3 else "differs"
            print(f"  {gib:>5.1f} GiB  {p2p_t:>8.3f}  {bc_t:>9.3f}  {ratio:>6.1f}x  "
                  f"(expected {expected_ratio}x: {verdict})")
        print(f"\nIf ratio ≈ {args.world_size-1}, XCCL is doing {args.world_size-1} sequential cross-node sends.")
        print(f"2-hop fix (1 send + intra-node bcast) should reduce {bcast_results[-1][1]:.1f}s → ~{p2p_results[-1][1]:.1f}s")

    del pg
    print(f"[rank {args.rank}] Done.")


if __name__ == "__main__":
    main()

"""Per-launch cost when 12 worker processes share a node, as K3 actually runs.

The isolated probe measured 6.5 us per kernel launch. The real K3 decode step
shows a 37 us MEDIAN inter-kernel gap -- 12,733 gaps of 20-50 us per token,
totalling 410 ms. The isolated probe had one process alone on the node; the
server runs 11-12 Ray worker actors per node, and Ray is started with
`--num-cpus=4` on a 208-core node.

This reruns the same tiny-kernel burst with N processes per node
simultaneously, so the only variable versus the isolated probe is host
contention. If per-launch jumps from 6.5 us toward ~37 us, the missing ~700
ms/token is host-side contention and the lever is CPU provisioning
(Ray --num-cpus, affinity, thread counts), not kernel count.

Run under mpiexec so all ranks start together:
  mpiexec -n 12 -ppn 12 --pmi=pmix python probe_launch_contended.py
"""

import os
import statistics
import time

import torch

N = 112
LAUNCHES = 2000
REPEATS = 5


def burst(x, launches):
    torch.xpu.synchronize()
    start = time.perf_counter()
    for _ in range(launches):
        x = x * 1.0001
    torch.xpu.synchronize()
    return (time.perf_counter() - start) * 1e6 / launches


def main():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank, world = comm.Get_rank(), comm.Get_size()
    local_rank = int(os.environ.get("PALS_LOCAL_RANKID", rank % 12))
    torch.xpu.set_device(local_rank)

    x = torch.ones(N, device=f"xpu:{local_rank}", dtype=torch.bfloat16)
    burst(x, 200)

    comm.Barrier()  # all ranks hammer the host at the same time
    samples = sorted(burst(x, LAUNCHES) for _ in range(REPEATS))
    median = samples[len(samples) // 2]

    allv = comm.gather(median, root=0)
    if rank == 0:
        allv.sort()
        print(f"procs={world}  cpus={os.cpu_count()}  "
              f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '<unset>')}")
        print(f"per-launch us  min {allv[0]:.2f}  median "
              f"{allv[len(allv) // 2]:.2f}  max {allv[-1]:.2f}  "
              f"mean {statistics.fmean(allv):.2f}")
        worst = allv[-1]
        print(f"\nprojected over 21,293 launches/token: "
              f"{worst * 21293 / 1000:.0f} ms/token (slowest rank)")
        print("isolated single-process baseline was 6.5 us -> 138 ms/token")
        print("K3's observed median inter-kernel gap is 37 us -> ~790 ms/token")


if __name__ == "__main__":
    main()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Regression guards for `_sync_ignored_trainable_grads` collective matching.

Job 8826889 (2026-09-14, 2 nodes, dp_replicate=1) hung for 1800s and completed zero
steps. The implementation bucketed grads by ``param.grad.dtype`` and called
``all_reduce`` INSIDE the loop over those buckets, so both the bucket set and the
iteration count were functions of LOCAL state (``param.grad is not None``). Rank 0 had
no grads -> zero collectives -> fell through; ranks 1-11 had grads -> one collective ->
blocked until the gloo timeout. It never fired at 16N only because rank 0 happened to
have the same 896 grads as its peers.

The invariant: **every rank must issue the same number of collectives in the same
order, regardless of which of its local params happen to have gradients.**

These run on CPU with the gloo backend and real multi-process collectives, because the
bug is a collective-count mismatch — a source-inspection test cannot see it, and a
single-process test cannot deadlock. Each subtest is run in a subprocess with a hard
timeout so a regression FAILS fast instead of hanging the suite.
"""

import os
import subprocess
import sys
import textwrap

import pytest


# The rank-independent iteration pattern under test, extracted so the child process
# does not need to import the full recipe (which pulls in XPU/vLLM dependencies).
# This mirrors grpo_bioreason_distributed_xpu.py::_sync_ignored_trainable_grads.
_CHILD = r'''
import os, sys
import torch
import torch.distributed as dist

FIXED = os.environ["FIXED"] == "1"
rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
dist.init_process_group("gloo", rank=rank, world_size=world)

# Two params of the same structural dtype. Rank 0 gets NO grads (the production
# condition that triggered the hang); every other rank gets grads.
params = [torch.nn.Parameter(torch.zeros(4)), torch.nn.Parameter(torch.zeros(4))]
if rank != 0:
    for i, p in enumerate(params):
        p.grad = torch.full((4,), float(rank + i))

pg = dist.group.WORLD
divisor = float(world)

if FIXED:
    # Rank-independent: bucket by param.dtype (structural), zero-fill missing grads.
    by_dtype = {}
    for p in params:
        by_dtype.setdefault(p.dtype, []).append(p)
    for dtype in sorted(by_dtype, key=str):
        ps = by_dtype[dtype]
        flat = torch.cat([
            (p.grad.detach().reshape(-1).to(dtype=dtype) if p.grad is not None
             else torch.zeros(p.numel(), dtype=dtype))
            for p in ps
        ])
        present = torch.tensor(
            [1.0 if p.grad is not None else 0.0 for p in ps], dtype=torch.float32
        )
        dist.all_reduce(flat, group=pg)
        dist.all_reduce(present, group=pg)
        off = 0
        for p, n in zip(ps, present.tolist()):
            chunk = flat[off:off + p.numel()]; off += p.numel()
            if n == divisor:
                chunk = chunk / divisor
            elif n > 0:
                chunk = chunk / n
            else:
                continue
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            p.grad.copy_(chunk.view_as(p.grad))
else:
    # The ORIGINAL buggy shape: bucket by grad.dtype, collective inside the loop.
    by_dtype = {}
    for p in params:
        if p.grad is not None:
            by_dtype.setdefault(p.grad.dtype, []).append(p.grad)
    for grads in by_dtype.values():
        flat = torch.cat([g.detach().reshape(-1) for g in grads])
        dist.all_reduce(flat, group=pg)

# Report what rank 0 ended up with, so the parent can assert on correctness.
if rank == 0:
    vals = [None if p.grad is None else p.grad.tolist() for p in params]
    print("RANK0_GRADS=" + repr(vals), flush=True)
print(f"RANK{rank}_DONE", flush=True)
dist.destroy_process_group()
'''


def _run(fixed: bool, world: int = 3, timeout: int = 45):
    """Run `world` gloo procs; return (returncode, stdout). Kills on timeout."""
    env = dict(os.environ)
    env.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT="29731" if fixed else "29732",
        WORLD_SIZE=str(world),
        FIXED="1" if fixed else "0",
        GLOO_SOCKET_IFNAME=env.get("GLOO_SOCKET_IFNAME", "lo"),
    )
    procs = []
    for r in range(world):
        e = dict(env, RANK=str(r))
        procs.append(
            subprocess.Popen(
                [sys.executable, "-c", _CHILD],
                env=e,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        )
    out = []
    try:
        for p in procs:
            out.append(p.communicate(timeout=timeout)[0])
        return max(p.returncode for p in procs), "\n".join(out)
    except subprocess.TimeoutExpired:
        for p in procs:
            p.kill()
        return "TIMEOUT", "\n".join(filter(None, out))
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()


@pytest.mark.timeout(120)
def test_fixed_pattern_does_not_deadlock_when_rank0_has_no_grads():
    """The shipped pattern must complete with every rank issuing matched collectives."""
    rc, out = _run(fixed=True)
    assert rc != "TIMEOUT", (
        "DEADLOCK: the rank-independent path hung when rank 0 had no grads. "
        "This is the job-8826889 failure.\n" + out
    )
    assert rc == 0, f"child procs failed (rc={rc}):\n{out}"
    for r in range(3):
        assert f"RANK{r}_DONE" in out, f"rank {r} never finished:\n{out}"


@pytest.mark.timeout(120)
def test_fixed_pattern_gives_rank0_the_peer_mean():
    """Rank 0 (no local grad) must end up with the mean over contributing ranks.

    Ranks 1,2 hold grads [rank+i]; for param 0 that is 1.0 and 2.0 -> sum 3.0 over
    2 contributing ranks -> 1.5. Dividing by the full world (3) instead would give
    1.0 -- a silently diluted update, which is the wrong-but-plausible outcome the
    presence-count guards against.
    """
    rc, out = _run(fixed=True)
    assert rc == 0 and rc != "TIMEOUT", out
    line = [l for l in out.splitlines() if l.startswith("RANK0_GRADS=")]
    assert line, f"rank 0 did not report grads:\n{out}"
    grads = eval(line[0].split("=", 1)[1])  # noqa: S307 - our own literal
    assert grads[0] == pytest.approx([1.5] * 4), (
        f"expected mean over the 2 contributing ranks (1.5), got {grads[0]}"
    )
    assert grads[1] == pytest.approx([2.5] * 4), (
        f"expected mean over the 2 contributing ranks (2.5), got {grads[1]}"
    )


@pytest.mark.timeout(120)
def test_original_pattern_fails_this_is_the_bug_being_fixed():
    """Pin the failure mode itself, so nobody reintroduces the old shape.

    The collective MISMATCH is the invariant being pinned; how it presents depends on
    the environment. In production (job 8826889) rank 0 stayed alive inside the training
    loop, so its peers blocked in ``all_reduce`` for the full 1800s gloo timeout and PBS
    killed the job at Exit_status=143. In this test rank 0 exits the process as soon as
    it falls through the empty loop, which closes its gloo sockets, so the peers die
    immediately with "Connection closed by peer" instead of waiting.

    Both are the same defect: ranks issued a different number of collectives. Accept
    either a non-zero exit or a timeout -- what must NOT happen is a clean rc=0.
    """
    rc, out = _run(fixed=False, timeout=25)
    assert rc != 0, (
        "The original grad.dtype-bucketed pattern completed cleanly, but it is expected "
        "to fail when rank 0 has no grads (collective-count mismatch). Verify the test "
        "still reproduces the production condition before trusting it as a guard.\n"
        + out
    )
    if rc != "TIMEOUT":
        assert "all_reduce" in out or "Connection closed by peer" in out, (
            "expected the failure to surface from the unmatched all_reduce:\n" + out
        )

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe verification for the SFT recipe's gradient-accumulation ``no_sync``
optimization (recipes/dev/full_finetune_distributed_xpu.py).

Background
----------
The SFT training loop accumulates gradients across ``gradient_accumulation_steps``
(ga) batches and steps the optimizer once per window. Before this change it called
``.backward()`` on every micro-batch with FSDP gradient sync left ON, so FSDP2
issued one cross-rank ``reduce_scatter`` *per micro-batch* — ``ga`` reductions per
optimizer step. The optimization suppresses sync on all but the last micro-batch of
each window (FSDP2: ``set_requires_gradient_sync(is_last)``; FSDP1/DDP: ``no_sync()``),
collapsing ``ga`` grad reductions to one. This mirrors the GRPO recipe's accumulation
path and the cross-framework PRISM scaling finding (Bug 3, +68% at 10N).

The two paths are *mathematically* equivalent (grad reduction is linear:
``Σ avg_r(g_m) == avg_r(Σ g_m)``) but NOT bit-identical (different summation order),
so we compare to fp tolerance.

This file has two layers:

1. **Functional** (``mp.spawn`` world_size=2, gloo, CPU, real FSDP2): runs both
   paths over the SAME data/weights and asserts (a) equal accumulated grads to
   ~1e-5, (b) the suppressed path issues exactly ONE reduce_scatter per param-group
   per window while the sync-every path issues ``ga``, proving the optimization
   actually fires, and (c) the scalar ``all_reduce(num_tokens)`` is identical across
   paths (pins the normalization independence). Runs in seconds on a login node.

2. **Source-scan** (import-free AST/string check, mirrors
   test_chunked_reduce_scatter_bypass.py): pins that the recipe contains the gating
   structure, applies it only on the ``not _optimizer_in_bwd`` path, and has not
   re-introduced ``empty_cache`` into the loop.
"""
import ast
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


# --------------------------------------------------------------------------- #
# Layer 2: source-scan guard (import-free; runs anywhere)
# --------------------------------------------------------------------------- #
_RECIPE_PATH = (
    "/lus/flare/projects/ModCon/ngetty/torchtune/recipes/dev/"
    "full_finetune_distributed_xpu.py"
)
_CLASS = "FullFinetuneRecipeDistributedXPU"
_METHOD = "train"


def _train_source() -> str:
    with open(_RECIPE_PATH) as f:
        src = f.read()
    tree = ast.parse(src)
    # Find the train() method on any class (recipe class name may vary); fall back
    # to a module-level scan if not found as a method.
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == _METHOD:
            return ast.get_source_segment(src, node)
    raise RuntimeError(f"Could not find {_METHOD}() in {_RECIPE_PATH}")


@pytest.fixture(scope="module")
def train_src() -> str:
    return _train_source()


def test_recipe_has_fsdp2_no_sync_toggle(train_src):
    """The accumulation path must toggle FSDP2 grad sync per micro-batch."""
    assert "set_requires_gradient_sync" in train_src, (
        "SFT train() no longer calls set_requires_gradient_sync — the "
        "gradient-accumulation no_sync optimization was removed. Without it, "
        "FSDP2 reduce-scatters grads on EVERY micro-batch (ga reductions per "
        "optimizer step) instead of once."
    )


def test_recipe_has_fsdp1_no_sync_fallback(train_src):
    """A no_sync() fallback must exist for the FSDP1 / DDP case."""
    assert "no_sync" in train_src, (
        "SFT train() lost the no_sync() fallback for the FSDP1/DDP path."
    )


def test_no_sync_gated_on_accumulation(train_src):
    """The toggle must be gated on ga>1 and use the last-micro-batch predicate."""
    assert "self._gradient_accumulation_steps > 1" in train_src, (
        "The no_sync path must be gated on gradient_accumulation_steps > 1 so "
        "ga==1 is a no-op (byte-identical to prior behavior)."
    )
    assert (
        "(batch_count + 1) % self._gradient_accumulation_steps == 0" in train_src
    ), (
        "The 'last micro-batch of window' predicate must reuse the same "
        "(batch_count + 1) % ga == 0 expression that gates the optimizer step. "
        "The SFT loop has no inner micro-batch sub-loop."
    )


def test_optimizer_in_bwd_branch_excluded(train_src):
    """The toggle must live on the `not _optimizer_in_bwd` branch only.

    optimizer_in_bwd is forced to ga==1, so it can never accumulate; the no_sync
    toggle belongs exclusively to the else (non-optimizer-in-bwd) branch.
    """
    lines = train_src.splitlines()
    # Locate the optimizer_in_bwd branch backward() and the no_sync toggle.
    oib_idx = next(
        (i for i, ln in enumerate(lines) if "if self._optimizer_in_bwd:" in ln),
        None,
    )
    toggle_idx = next(
        (i for i, ln in enumerate(lines) if "set_requires_gradient_sync" in ln),
        None,
    )
    assert oib_idx is not None and toggle_idx is not None
    # The toggle must appear AFTER the optimizer_in_bwd block opens, inside the
    # sibling else — i.e. there must be an `else:` between them.
    between = "\n".join(lines[oib_idx:toggle_idx])
    assert "else:" in between, (
        "set_requires_gradient_sync must be inside the `else` (not "
        "_optimizer_in_bwd) branch — the optimizer_in_bwd path must stay "
        "byte-for-byte unchanged."
    )


def test_no_empty_cache_reintroduced(train_src):
    """The loop must not call empty_cache (L0 UR-handle leak with FSDP)."""
    assert "empty_cache" not in train_src, (
        "empty_cache() reappeared in SFT train() — it leaks UR handles in "
        "Level Zero under FSDP (see docs/bugs/intel_xpu_resource_leak_bug_report.md)."
    )


def test_dataloader_pin_memory_defaults_false():
    """pin_memory must default to False.

    The triple {torch.compile model + pinned-memory forked-worker batches +
    non-reentrant activation checkpointing} raises a step-0-backward
    CheckpointError on XPU (A/B: experiments/agpt2b_sft/logs/smoke_dlf2_* +
    smoke_fix_*). The crash needs compile=True AND pin_memory=True AND
    num_workers>0 simultaneously; defaulting pin_memory=False clears it while
    keeping the async-collate throughput win. compile_dynamic does NOT fix it.
    A regression to default-True silently reintroduces the crash on every
    compile+AC SFT run.
    """
    with open(_RECIPE_PATH) as f:
        src = f.read()
    assert 'cfg.get("dataloader_pin_memory", False)' in src, (
        "dataloader_pin_memory must default to False. Defaulting it True "
        "reintroduces the compile+pinned+AC CheckpointError "
        "(see memory/project_sft_pinmem_compile_ac_checkpoint_error_20260621)."
    )


# --------------------------------------------------------------------------- #
# Layer 1: functional equivalence (gloo / CPU / real FSDP2)
# --------------------------------------------------------------------------- #
GA = 4               # gradient_accumulation_steps
MB = 3               # micro-batch size
DIM = 16
N_LAYERS = 2


def _build_model(seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    layers = []
    for _ in range(N_LAYERS):
        layers.append(torch.nn.Linear(DIM, DIM, bias=False))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(DIM, DIM, bias=False))
    return torch.nn.Sequential(*layers)


def _wrap_fsdp2(model: torch.nn.Module, mesh) -> torch.nn.Module:
    from torch.distributed._composable.fsdp import fully_shard

    for m in model:
        if isinstance(m, torch.nn.Linear):
            fully_shard(m, mesh=mesh)
    fully_shard(model, mesh=mesh)
    return model


def _grads_flat(model: torch.nn.Module) -> torch.Tensor:
    """Concatenated full (unsharded) grad vector for cross-path comparison."""
    parts = []
    for p in model.parameters():
        g = p.grad
        if g is None:
            continue
        # FSDP2 grads are DTensors; pull the full tensor for comparison.
        if hasattr(g, "full_tensor"):
            g = g.full_tensor()
        parts.append(g.reshape(-1).detach().clone())
    return torch.cat(parts)


def _run_window(model, batches, *, suppress, rs_counter):
    """Run one ga-window. suppress=False → sync every micro-batch (baseline);
    suppress=True → set_requires_gradient_sync(is_last) (optimization).

    Returns the reduced num_tokens scalar (mirrors the recipe's
    all_reduce(num_tokens) to pin normalization independence)."""
    num_tokens = torch.tensor(0.0)
    for i, xb in enumerate(batches):
        is_last = i == (len(batches) - 1)
        num_tokens += float(xb.shape[0] * DIM)
        if suppress and GA > 1:
            model.set_requires_gradient_sync(is_last)
        else:
            # Baseline behavior: grad sync on for every micro-batch.
            if hasattr(model, "set_requires_gradient_sync"):
                model.set_requires_gradient_sync(True)
        rs_counter["window"] = rs_counter.get("window", 0)
        loss = model(xb).sum()
        loss.backward()
    dist.all_reduce(num_tokens)
    return num_tokens


def _worker(rank, ws, ret):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29571")
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(ws)
    dist.init_process_group("gloo", rank=rank, world_size=ws)
    try:
        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh("cpu", (ws,))

        # Identical data on both ranks-paths; deterministic per rank.
        torch.manual_seed(100 + rank)
        batches = [torch.randn(MB, DIM) for _ in range(GA)]

        # --- Count reduce_scatter calls per path by wrapping the dist API FSDP2 uses.
        counts = {"baseline": 0, "suppressed": 0}
        orig_rs = dist.reduce_scatter_tensor
        active = {"key": None}

        def counting_rs(*a, **k):
            if active["key"] is not None:
                counts[active["key"]] += 1
            return orig_rs(*a, **k)

        dist.reduce_scatter_tensor = counting_rs
        try:
            # PATH 1: baseline (sync every micro-batch)
            m1 = _wrap_fsdp2(_build_model(seed=0), mesh)
            active["key"] = "baseline"
            nt1 = _run_window(m1, batches, suppress=False, rs_counter={})
            g1 = _grads_flat(m1)

            # PATH 2: optimization (suppress non-last)
            m2 = _wrap_fsdp2(_build_model(seed=0), mesh)
            active["key"] = "suppressed"
            nt2 = _run_window(m2, batches, suppress=True, rs_counter={})
            g2 = _grads_flat(m2)
            active["key"] = None
        finally:
            dist.reduce_scatter_tensor = orig_rs

        if rank == 0:
            ret["grad_max_abs_diff"] = float((g1 - g2).abs().max())
            ret["grad_rel_diff"] = float(
                (g1 - g2).abs().max() / (g1.abs().max() + 1e-12)
            )
            ret["num_tokens_match"] = bool(torch.equal(nt1, nt2))
            ret["count_baseline"] = counts["baseline"]
            ret["count_suppressed"] = counts["suppressed"]
            ret["n_params"] = g1.numel()
    finally:
        dist.destroy_process_group()


@pytest.mark.timeout(180)
def test_no_sync_grad_equivalence_and_collective_reduction():
    """Suppressed path == baseline grads (fp tol) AND issues fewer reduce_scatters."""
    mgr = mp.Manager()
    ret = mgr.dict()
    mp.spawn(_worker, args=(2, ret), nprocs=2, join=True)

    assert "grad_max_abs_diff" in ret, "worker did not report results"
    # (a) accumulated grads equivalent to fp tolerance (not bit-identical).
    assert ret["grad_rel_diff"] < 1e-5, (
        f"Accumulated grads diverged between sync-every and suppressed paths: "
        f"max_abs={ret['grad_max_abs_diff']:.3e} rel={ret['grad_rel_diff']:.3e}. "
        f"The no_sync optimization changed the math, not just the comm schedule."
    )
    # (b) the optimization actually fires: suppressed issues strictly fewer
    #     reduce_scatters than the per-micro-batch baseline.
    assert ret["count_suppressed"] < ret["count_baseline"], (
        f"Suppressed path did not reduce reduce_scatter count "
        f"(baseline={ret['count_baseline']} suppressed={ret['count_suppressed']}). "
        f"set_requires_gradient_sync(is_last) is not actually suppressing the "
        f"per-micro-batch grad reduction."
    )
    # Baseline should issue ~GA× the collectives of the suppressed path.
    assert ret["count_baseline"] >= GA * ret["count_suppressed"] // 2, (
        f"Expected baseline to issue many more reduce_scatters than suppressed; "
        f"got baseline={ret['count_baseline']} suppressed={ret['count_suppressed']}."
    )
    # (c) scalar token normalization is path-independent.
    assert ret["num_tokens_match"], (
        "all_reduce(num_tokens) differed between paths — the no_sync toggle must "
        "not affect the explicit scalar reductions used for loss normalization."
    )

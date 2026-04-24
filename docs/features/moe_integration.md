# MoE Expert Parallelism Integration — Aurora/XPU

Status and implementation notes for Expert Parallelism (EP) support for Gemma4 26B-A4B
on Aurora HPC (Intel Max Series GPUs / XPU), targeting EP=4/DP=3 on 12 tiles per node.

## Goal

Reduce per-tile expert parameter memory by partitioning Gemma4's 128 experts across 4 EP
ranks, so each tile holds 32 experts instead of 128. Primary benefit is memory reduction
(~3 GiB/tile at bfloat16) which enables larger batch sizes or reduces optimizer state
pressure during GRPO training.

Topology: `dp_replicate=3 × dp_shard=4` on 12 tiles. The `dp_shard` submesh doubles as
the EP communicator group — the same 4 ranks handle both FSDP sharding of non-expert
parameters and Expert Parallel token dispatch.

---

## Model Architecture

Gemma4 26B-A4B has 62 transformer layers, 30 of which are MoE. Each MoE layer:

```
Gemma4TransformerLayer
  ├── attn          (Gemma4Attention)      — replicated
  ├── mlp           (dense FFN)            — replicated (non-MoE layers only)
  └── moe_block     (MoE)                  — MoE layers only
        ├── router  (Gemma4MoeRouter)      — replicated
        └── experts (GroupedExperts)       — EP-partitioned
```

Key parameters: 128 experts/layer, top-2 routing. Each expert is a small FFN:
`gate_proj` + `up_proj` + `down_proj` with shapes `(num_experts, dim, hidden_dim)`.
After EP=4 partitioning: `(32, dim, hidden_dim)` per tile.

---

## Files Modified

| File | Change |
|------|--------|
| `torchtune/modules/moe/utils.py` | Added `_permute`, `_unpermute` (pure torch, no Triton) |
| `torchtune/modules/moe/_parallelism.py` | Added `ExpertParallel`, `apply_ep_weight_sharding` |
| `torchtune/modules/moe/__init__.py` | Exported `ExpertParallel`, `apply_ep_weight_sharding` |
| `torchtune/modules/moe/experts.py` | Fixed `_forward_no_grouped_mm` to use local expert count |
| `torchtune/models/gemma4/_parallelism.py` | New: `gemma4_ep_plan()` returning EP plan for 30 MoE layers |
| `torchtune/models/gemma4/__init__.py` | Exported `gemma4_ep_plan` |

### Test / validation scripts

| File | Purpose |
|------|---------|
| `recipes/dev/test_ep_smoke.py` | 4-subtest smoke test (imports, permute, all-to-all, forward) |
| `recipes/dev/test_ep_correctness.py` | EP=2 vs replicated output comparison (bit-exact) |
| `recipes/dev/test_ep_model_setup.py` | 12-tile EP=4/DP=3 topology validation with real checkpoint |
| `recipes/dev/run_ep_smoke.sh` | Smoke test launcher |

---

## Implementation Design

### Why hook-based EP (not DTensor)

The natural approach is to mark expert parameters as `DTensor` with `Shard(0)` placement
on the EP mesh. This was tried first and fails:

When the outer `shard_model()` calls `fully_shard(layer, mesh=dp_mesh)`, FSDP2 internally
tries to compose mesh dimensions and produces `('dp_replicate', 'dp_shard', 'dp_shard')` —
an invalid 3D mesh from a 2D source.

**Solution**: register plain PyTorch forward hooks instead of using `distribute_module`.
Expert parameters stay as regular `nn.Parameter` tensors (sliced to local shard size).
All-to-All dispatch/combine runs in `register_forward_pre_hook` / `register_forward_hook`.
No DTensor, no mesh composition conflict.

### ExpertParallel (`torchtune/modules/moe/_parallelism.py`)

Registered via `parallelize_module(model, ep_mesh, ep_plan)`. For each `GroupedExperts`
module it:

1. Stores `module._ep_device_mesh = device_mesh` for later weight slicing
2. Registers a forward pre-hook (`_token_dispatch`) and forward hook (`_token_combine`)

**`_token_dispatch`** (pre-hook):
1. All-gather the per-rank `num_tokens_per_expert` histograms across the EP group to build
   the full `(ep_degree, num_experts)` dispatch matrix
2. Compute `input_splits` (tokens this rank sends to each EP peer) and `output_splits`
   (tokens this rank receives from each EP peer)
3. `all_to_all_single_autograd`: route tokens to expert-owning ranks
4. `_permute`: reorder received tokens from source-rank-major to local-expert-major order

**`_token_combine`** (post-hook):
1. `_unpermute`: scatter expert outputs back to source-rank-major order
2. `all_to_all_single_autograd` (swapped splits): return processed tokens to originating ranks

Gradient flow is automatic via `all_to_all_single_autograd`.

### apply_ep_weight_sharding (`torchtune/modules/moe/_parallelism.py`)

After loading the full checkpoint, call this to slice each EP module's parameters to the
local shard:

```python
n_sharded = apply_ep_weight_sharding(model)
# → 30 modules × 3 params each (gate_proj, up_proj, down_proj)
# → param.data[ep_rank * n_local : (ep_rank+1) * n_local].contiguous()
```

**Critical ordering constraint**: must be called AFTER `load_state_dict` but BEFORE
`shard_model()`. After FSDP2 `fully_shard()`, named parameters are absorbed into flat
buffers — `named_parameters(recurse=False)` returns nothing for FSDP2-sharded modules.

### _permute / _unpermute (`torchtune/modules/moe/utils.py`)

Pure torch implementation (no Triton). Builds a permutation index that reorders tokens
from the source-rank-major layout produced by All-to-All into the expert-major layout
required by `GroupedExperts.forward()`:

```
After All-to-All:   [ep0_exp0_toks, ep0_exp1_toks, ..., ep1_exp0_toks, ...]
After _permute:     [exp0_from_all_ranks, exp1_from_all_ranks, ...]
```

Compatible with XPU's loop-based expert path (no `grouped_mm` padding required).

### experts.py fix

`GroupedExperts._forward_no_grouped_mm` used `self.num_experts` (global count = 128) as
the loop bound `E`. After EP weight slicing, `gate_proj` has only `num_experts / ep_degree`
rows. Fix: use `E = num_tokens_per_expert.shape[0]` (local count = 32 for EP=4).

### gemma4_ep_plan (`torchtune/models/gemma4/_parallelism.py`)

Iterates `model.layers` and maps each MoE layer's `experts` to `ExpertParallel()`:

```python
ep_plan = gemma4_ep_plan(model)      # 30 entries: layers.{i}.moe_block.experts
parallelize_module(model, ep_mesh, ep_plan)
```

Note: `model.layers` is a `ModuleList`, not `ModuleDict` — iterate with `enumerate`,
not `.values()`.

---

## Setup Sequence (CRITICAL Ordering)

```python
# 1. Build 2D mesh
dp_mesh = init_device_mesh(device.type, (dp_replicate, dp_shard),
                            mesh_dim_names=("dp_replicate", "dp_shard"))
ep_mesh = dp_mesh["dp_shard"]

# 2. Eager process group initialization (prevents XCCL deadlock in forward hooks)
_ = ep_mesh.get_group()          # ep_mesh is 1D
_ = dp_mesh.get_all_groups()     # dp_mesh is 2D — must use get_all_groups(), NOT get_group()
dist.barrier()

# 3. Build model on meta device
with training.set_default_dtype(torch.bfloat16), torch.device("meta"):
    model = gemma4_26b_a4b()

# 4. Apply EP hooks (stores _ep_device_mesh, registers forward hooks)
ep_plan = gemma4_ep_plan(model)
parallelize_module(model, ep_mesh, ep_plan)

# 5. Stage checkpoint to /tmp (rank 0 copies, barrier, all ranks load)
if rank == 0:
    shutil.copy2(LUSTRE_CKPT, TMP_CKPT)   # ~62s vs 357s from Lustre with 12 concurrent readers
dist.barrier()

# 6. Materialize model on XPU and load checkpoint
model.to_empty(device=device)
sd = checkpointer.load_checkpoint()
model.load_state_dict(sd["model"], strict=False)

# 7. Slice expert weights to local EP shards (PRE-FSDP2)
apply_ep_weight_sharding(model)

# 8. shard_model() / FSDP2 — see "FSDP2 integration" below
```

---

## XCCL / Environment Requirements

For jobs that create multiple process groups (2D mesh → submesh groups):

```bash
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_KVS_MODE=pmi
export CCL_KVS_IFACE=hsn0          # required for submesh group creation
export FI_PROVIDER=cxi              # NOT shm — shm fails for nested groups
export CCL_WORKER_COUNT=1
export CCL_OP_SYNC=1
export CCL_CONFIGURATION=cpu_gpu_dpcpp
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
```

Also required: bypass `torchtune.__init__` to prevent XCCL USM pointer corruption
(torchtune imports torchao which corrupts XCCL's USM table on Aurora):

```python
# Must happen before any torch or torchtune import
import types, importlib.util, sys
if "torchtune" not in sys.modules:
    spec = importlib.util.find_spec("torchtune")
    pkg = types.ModuleType("torchtune")
    pkg.__path__ = list(spec.submodule_search_locations)
    pkg.__file__ = ...
    sys.modules["torchtune"] = pkg
```

---

## Validation Status (April 10, 2026)

| Test | Config | Status | Notes |
|------|--------|--------|-------|
| Import smoke test | EP=2, 2 tiles | **PASS** | `ExpertParallel`, `_permute`, `all_to_all_single_autograd` |
| Permute/unpermute | EP=2, 2 tiles | **PASS** | Round-trip, all token counts |
| All-to-All forward | EP=2, 2 tiles | **PASS** | Correct shapes and values |
| Import smoke test | EP=4, 4 tiles | **PASS** | |
| All-to-All forward | EP=4, 4 tiles | **PASS** | |
| EP=2 vs replicated | EP=2, 2 tiles | **PASS** | Bit-exact match (max_err=0.0) |
| 12-tile forward pass | EP=4/DP=3, 12 tiles | **PASS** | See results below |

### 12-tile forward pass results (test_ep_model_setup.py)

Node: `x4117c4s3b0n0`, job 8431784

```
[Rank 0] All process groups initialized
[Rank 0] ExpertParallel applied to 30 modules
[Rank 0] load_state_dict: 0 missing, 0 unexpected
[Rank 0] Checkpoint loaded in 357.6s          ← from Lustre; /tmp staging now added (~62s)
[Rank 0] Peak memory after load (pre-slice): 47.12 GiB
[Rank 0] EP weight sharding applied to 30 expert modules
[Rank 0] Peak memory after EP slice: 47.12 GiB  ← XPU allocator holds pool; actual RSS reduced
[Rank 0] Forward pass OK in 2.07s, out shape: torch.Size([1, 512, 262144])
[Rank 0] Peak memory after forward: 47.62 GiB
[Rank 0] === EP=4/DP=3 Model Setup Test: PASS ===
```

Memory note: peak stays at 47 GiB because the XPU standalone allocator retains freed
blocks in its pool rather than returning them to the OS. The actual expert parameter
resident set did shrink 4× (32 experts vs 128); the benefit will show in training where
optimizer states are the dominant factor.

---

## FSDP2 + EP Communicator Conflict

Running FSDP2 all-gather (12-rank `dp_mesh` group) and EP All-to-All (4-rank `ep_mesh`
group) concurrently during a single forward pass causes XCCL deadlock on Aurora.
Symptoms: all 12 ranks spin at 92–97% CPU (XCCL busy-wait), no system calls, no progress.

The conflict arises because FSDP2 prefetches the next layer's all-gather while the current
layer's EP all-to-all is in-flight, and XCCL can't interleave operations across different
communicator groups.

**Current workaround**: FSDP2 is skipped in the 12-tile forward pass test.

**Solutions for training integration** (in order of preference):

1. **Wrap non-expert params on `dp_replicate` mesh only (3 ranks)**
   FSDP2 all-gather runs within a 3-rank group; EP all-to-all runs within a 4-rank group.
   No overlap. Non-expert memory savings are 3× (vs 12× without EP). Expert params
   are already EP-partitioned (4×) and can be FSDP-wrapped separately on `ep_mesh`.

2. **Sequential FSDP scheduling** (`reshard_after_forward=False`)
   Disables FSDP prefetch, forcing all-gather to complete before the next layer starts.
   Eliminates overlap but reduces throughput.

3. **Skip FSDP2 for experts, use FSDP2 on `dp_replicate` for all other params**
   Expert params don't participate in FSDP2 at all — they're sliced by EP, gradient
   reduction happens via the EP all-to-all backward. Clean separation.

---

## Memory Model

Without any FSDP2 (current test state):

| Component | EP=1 | EP=4 (no FSDP2) |
|-----------|------|-----------------|
| Expert params | ~4.0 GiB | ~1.0 GiB |
| Non-expert params | ~16.5 GiB | ~16.5 GiB |
| **Total params** | ~20.5 GiB | ~17.5 GiB |

With FSDP2 on `dp_replicate` (3-rank) for non-expert params (training target):

| Component | EP=1 / FSDP2 12-rank | EP=4 / FSDP2 3-rank |
|-----------|----------------------|---------------------|
| Expert params | ~4.0 GiB | ~1.0 GiB |
| Non-expert FSDP shard | ~1.4 GiB (÷12) | ~5.5 GiB (÷3) |
| **Total params** | ~5.4 GiB | ~6.5 GiB |
| Optimizer states (Adam) | ~10.8 GiB | ~13.0 GiB |
| Activations (training) | ~4.5 GiB | ~4.5 GiB |
| **Estimated peak** | ~20.7 GiB | ~24.0 GiB |

EP=4 peak may be slightly higher than EP=1 in the param+optimizer breakdown because
non-expert FSDP sharding is coarser (3-rank vs 12-rank). The benefit is in
**load balancing** and enabling larger effective batch sizes without expert OOM.

---

## What's Next

1. **FSDP2 integration**: wire `shard_model()` to use `dp_replicate` mesh for non-expert
   params, and wrap expert modules separately on `ep_mesh`. Validate no deadlock.

2. **GRPO recipe wiring**: update `grpo_full_finetune_distributed_xpu.py` to accept
   `expert_parallel_degree` config, build the 2D mesh, apply EP hooks, call
   `apply_ep_weight_sharding`, then call `shard_model`.

3. **Full EP=4/DP=3 GRPO benchmark**: step time and peak memory vs EP=1 baseline
   (24.5 s/step, 20.66 GiB). Target: memory neutral-to-better, step time ≤ +10%.

---

## Backward Dispatch Saga (v141–v153, April 22, 2026)

After abandoning AllToAll (which deadlocked v18–v136 with XCCL SIGSEGV in backward),
EP dispatch was reformulated as **AllGather + ReduceScatter** (Mula paper, arXiv 2604.00785):

```
Forward:   AllGather tokens → index-select for local experts → expert GEMM
                            → scatter_add → ReduceScatter
Backward:  automatic via PyTorch autograd (AG ↔ RS swap)
```

This unblocked the SIGSEGV but produced a new deterministic deadlock: **op #259 RS-BWD**.

### Op structure (300 EP ops/step)

With AC wrapping `TransformerSelfAttentionLayer` (full layer including MoE) and
`use_reentrant=True`, each MoE layer contributes 4 BWD ops: AG-FWD recompute,
RS-FWD recompute, AG-BWD, RS-BWD. Total per step:

| Pass | Range | Notes |
|------|-------|-------|
| Policy logprob FWD | 0–59 | `no_grad`; 30 layers × 2 ops |
| Ref logprob FWD | 60–119 | `no_grad`; ref model also wired with EP |
| Training FWD | 120–179 | 30 layers × 2 ops |
| Training BWD | 180–299 | 30 layers × 4 ops (AC recompute + grad) |

Op #259 = RS-BWD for layer 10 (10th layer from front, 19th-from-back in BWD order).

### v141–v152: gloo CPU-bounce migration

To avoid XCCL deadlock in the AC-recompute interleaving, all EP collectives were
migrated to gloo CPU-bounce (move tensor to CPU, gloo all_reduce, slice, move back):

| Version | Change | Result |
|---------|--------|--------|
| v141 | Native XCCL `_c10d_reduce_scatter` (bypass gloo monkey-patch) | All 260+ ops complete; needs full re-run for confirmation |
| v143 | `dist.barrier(ep_pg)` after each XCCL collective (diagnostic) | All ops complete; barriers masked the bug + drained OFI CQ |
| v144 | Remove diagnostic barriers | Immediate failure — confirms diagnostic was load-bearing |
| v148–v151 | Pure gloo CPU-bounce for AG and RS | All FWD ops complete; **op #259 RS-BWD timeout** |
| v152 | Dedicated `_GLOO_EP_PG` (separate gloo group from FSDP2 grad sync) | Same op #259 RS-BWD deadlock; eliminated FSDP2 contention as cause |

**v152 forensics (job 8444929):** all 12 ranks complete ops 0–258. At op #259:
6 ranks print `ENTER RS-BWD`, 0 print `COLL-DONE`. The other ~6 ranks (one per
EP group, plus some receivers) never call op #259 at all. Between ops 258 and 259
there is **only local computation** — no inter-rank collective. So either ranks
crashed silently between ops, or the gloo TCP connection state itself was corrupted.

### v153: GLOO_SOCKET_IFNAME=lo + zero XCCL in EP path (job 8445116)

**Hypothesis**: gloo TCP defaults to the CXI HSN NIC on Aurora. XCCL also uses the
CXI NIC via OFI. After 256+ ops of heavy XCCL traffic, the CXI NIC's OFI completion
queue accumulates stale events that contaminate gloo TCP connection state. Some ranks
lose the connection silently and never call op #259 → ring deadlock.

**Changes:**
1. `GLOO_SOCKET_IFNAME=lo` — force gloo TCP onto kernel loopback (127.0.0.1),
   bypassing the CXI NIC entirely. All 12 training tiles are on one node, so
   loopback is correct.
2. Replace the last XCCL call in EP dispatch (NTPE histogram all-gather between
   AG-FWD and RS-FWD) with gloo CPU-bounce. **Zero XCCL in EP dispatch path.**
3. Per-rank logging (`[rank{r}]` prefix) on every EP op + new `PRE-RS-BWD` print
   in `_AllGatherRS.backward`.
4. EP gloo group timeout reduced to 120s (was 1800s) for faster diagnosis.

**Result:** different failure mode at the same op #259.

- **All 12 ranks now reach op #259** (loopback fix solved the "missing rank" problem).
- But rank 8 **crashes hard with exitcode 1** during the gloo all_reduce
  (other ranks were SIGTERM'd, exit -15).
- Loopback peers print `Connection closed by peer [127.0.0.1]:<port>` —
  consistent with rank 8's process dying mid-collective and dropping its TCP socket.
- Other ranks then time out at 120s waiting for send/recv that will never complete.

**The new clue (rank desync at op #258):** with per-rank logging on, a structural
problem became visible. At op #258, ranks split into two groups:

| Ranks | EP-OP #258 label |
|-------|------------------|
| 0, 2, 3, 4, 6, 7, 8, 10, 11 | `AG-BWD` |
| **1, 5, 9** | `AG-FWD` (forward recompute of NEXT layer) |

Ranks 1, 5, 9 are the **local-index-1 rank within each EP group**
([0,1,2,3], [4,5,6,7], [8,9,10,11]). They are exactly **one op behind** the others.

`_EP_OP_N` is a per-rank Python counter — each rank increments it on every collective
it calls. Gloo matches collectives by **call order**, not by op label. So a same-numbered
op on different ranks within the same EP group can refer to **different layers**.
Once one rank in a group is out of step, every subsequent collective in that group
mismatches: ranks send a tensor sized for layer N while their peer expects layer M.

This explains everything:
- Why op #259 is the failure point: 19 prior matched gloo ops worked because the
  routing distribution happened to produce compatible sizes; eventually a mismatch
  in tensor shape or content produces an unrecoverable state.
- Why one rank crashes hard (rank 8 exitcode 1): a size-mismatched gloo op asserts
  or segfaults on one process before the others.
- Why the v152 "missing ranks" pattern matched the same local-index-1 ranks
  (~ranks 3, 5, 9) — the desync in v152 manifested as never-arriving ranks; in v153
  with the timing-shifted loopback transport, it manifests as a hard crash mid-collective.

### Root cause — autograd execution order is not deterministic per rank

PyTorch's autograd engine schedules backward hooks in a topological order determined
by the operation order seen during forward. With AC + `use_reentrant=True`, the forward
recompute happens during backward. **Subtle differences in tensor allocation order or
hook registration can cause autograd to schedule layers in different orders on different
ranks** — even when the model and inputs are identical. The MoE forward hook
(`_token_dispatch`/`_token_combine`) registration order interacts with PyTorch's
autograd sort, and the result is rank-dependent.

When backward ops are not lock-step across ranks within a collective group, the global
op counter desyncs and gloo collectives mismatch.

### Why this is hard to fix

The fix cannot be "set GLOO_SOCKET_IFNAME=lo" or "use a different transport" — those
are timing band-aids that will eventually expose the same desync. The real fix requires
one of:

1. **Tag/identify each collective explicitly** instead of relying on call order.
   Gloo doesn't support tagging well. NCCL/XCCL with `barrier()` after each op would
   force lock-step but kills throughput.
2. **Force deterministic backward order** via `torch.autograd.backward(... retain_graph=True)`
   with explicit per-layer backward calls instead of a single `loss.backward()`.
   Major recipe surgery.
3. **Avoid AC for MoE layers** — `use_reentrant=False` AC, or no AC at all on layers
   with EP. Memory cost: ~30 layers × activations / layer; may not fit at G=4.
4. **Move EP collectives outside the AC region** — pre-dispatch all tokens before
   the layer's forward, run experts as a side effect, post-combine after. Decouples
   collective ordering from autograd scheduling.

### Recommendation (revised)

EP for Gemma4 26B-A4B is blocked on a deeper autograd-ordering issue, not a transport
issue. After 153 versions, the path forward is:

- **Short term**: keep EP as a research effort. Production training continues with
  EP=1 (replicated experts) using the proven 12-tile or 2-node HSDP recipes.
- **Next experiment**: option 4 above — restructure `_token_dispatch`/`_token_combine`
  to run as side-effecting operations outside autograd, with manual gradient handling.
  This is the cleanest decoupling and is reusable for any MoE+AC combination.
- **Fallback**: option 3 — disable AC for the 30 MoE layers and accept the memory hit;
  measure whether G=2 still fits.

---

## Critical Review (2026-04-22, post-v153, no new tests)

After 150+ versions the doc had absorbed claims as fixed truth that don't survive a
re-read of the current code, the v153 logs, and `MEMORY.md`. This section calls those
out and proposes experiments before the next compute spend.

### 1. The doc no longer matches the code

**"FSDP2 + EP Communicator Conflict — Current workaround: FSDP2 is skipped"** (lines 254–264) —
stale. The recipe (`grpo_full_finetune_distributed_xpu.py:1934–2120`) has, for many
versions, used a layered scheme:

1. Non-expert params: FSDP2 on `dp_replicate` (3 ranks), `reshard_after_forward=False`
   (ZeRO-2). Disjoint from `ep_mesh` ranks → no XCCL group conflict.
2. Expert params: trivial 1-rank "solo" FSDP2 with `reduce_grads=False`. Pure local
   wrapping; no comm. Used to silence `ze_handle_manager` crashes on 1-rank RS.
3. All FSDP2 `reduce_grads` are suppressed; gradient sync is done post-backward by
   `_ep_post_backward_grad_sync_xccl()` over an explicit `_XCCL_DP_REP_PG`.
4. EP collectives use **gloo CPU-bounce** (`_GLOO_EP_PG`), not XCCL.

This is materially different from anything described in §"Setup Sequence" or §"Memory
Model". The doc still implies `apply_ep_weight_sharding` is called post-load — the
recipe replaced that with **pre-FSDP2 meta-param shrinking + model_sd pre-slicing**
(v41/v42 lines 1865–1902, 2160–2189). Anyone reading just the doc to understand the
system is missing the core of the actual implementation.

### 2. The "300 ops/step" arithmetic is suspect

Doc says: 30 MoE layers × 2 ops × 5 passes = 300 ops; "180–299: training BWD (4 ops/layer)".
Verify: 30 × 4 = 120 BWD ops → 180 + 120 = 300 ✓. But the **NTPE all-gather** added in
v153 (`_parallelism.py:528–536`) is **not counted** by `_EP_OP_N`. So per-layer dispatch
is actually AG-FWD + NTPE-AG + RS-FWD with two collectives matched against one
counter. The diagnostic logs are misleading by exactly one collective per dispatch.

**Implication**: when the v153 forensics say "ranks 1, 5, 9 are one op behind", the
gloo `_EP_OP_N` they see may be off by an integer multiple of NTPE-AG calls (which all
ranks fire but don't print). The "AG-FWD vs AG-BWD at op 258" finding is real (the
labels are different, not just the counters), but the broader claim of "exactly one
op behind" needs to be re-derived counting NTPE.

### 3. "Per-rank autograd ordering non-determinism" is plausible but unproven

The doc's root-cause story is: AC + `use_reentrant=True` reorders backward execution
on different ranks, so `_EP_OP_N` (a per-rank Python counter) goes out of step within
an EP group, gloo matches collectives by call order, and we eventually crash.

What we actually know from the log:

- Within EP group {0,1,2,3}: at op #258 ranks 0,2,3 are at AG-BWD, rank 1 is at AG-FWD.
- Same pattern in EP groups {4,5,6,7} and {8,9,10,11} — local-index-1 is always the
  outlier. **This is a structural pattern across groups, not random per-rank scheduling.**
- Earlier (op #218 etc.) some ranks complete in 4 enters not 6 (parsed `uniq -c`),
  suggesting partial events were lost or interleaved differently across logs.

A fundamentally non-deterministic explanation should not produce the *exact same*
per-rank pattern across three independent EP groups. Candidate root causes and
audit results (2026-04-22):

1. **A local-rank-dependent code path** firing an extra Python-level collective on
   rank-1 only — **RULED OUT by code audit.** Searched `_parallelism.py`, `moe.py`,
   `experts.py`, `gemma4/_moe.py`, `gemma4/_parallelism.py`, and the EP wiring block
   in `grpo_full_finetune_distributed_xpu.py:1815–2240`. No `if ep_rank == X` or
   `if local_rank == X` branches alter the collective sequence.
2. **Routing-imbalance feedback loop** (still open, now the leading hypothesis):
   per-rank `routed_input.shape[0]` differs each step. Under reentrant AC, the
   *order* in which the recompute pass evaluates submodules of a `TransformerSelfAttentionLayer`
   could depend on tensor sizes registered with autograd, causing a different
   call-order of `_token_dispatch`/`_token_combine` on the rank with the smallest
   batch (which, given the v110 interleaved routing fix, would consistently be the
   *same* local index — explaining the local-index-1 pattern across all three groups
   without invoking randomness). This is the only candidate consistent with both the
   structural pattern and the determinism of `_EP_OP_N` mismatch.
3. **`_ag_gather_idx` instance-cache aliasing** under reentrant AC — **UNLIKELY**
   per code audit. `gemma4/_parallelism.py:49` constructs a fresh `ExpertParallel()`
   per layer (`gemma4_ep_plan`), so each MoE layer has its own
   `self._ag_gather_idx`/`self._ag_s_local`. Cross-layer aliasing is not possible.
   Within-layer recompute could in principle overwrite the attr between FWD-recompute
   of `_token_dispatch` and BWD of `_token_combine`, but `_AllGatherRS.backward`
   captures `gather_idx` from the saved tensors at FWD time, not from `self.*`.

(1) is closed; (3) is closed by construction. **The remaining open root-cause story
is (2) — routing-imbalance interacting with reentrant-AC scheduling.** This is a
narrower target than the doc's previous "autograd ordering is non-deterministic"
framing.

### 4. Things the doc treats as load-bearing that may not be

- **CPU-bounce for AG/RS**: 4× bandwidth tax, kept "to avoid OFI CQ deadlock". Memory
  `project_ccl_naive_bypass_failed.md` documents L0 IPC ALWAYS being used for
  intra-node XPU↔XPU; gloo CPU-bounce dodges L0 IPC entirely. Whether dense Gemma4
  (no EP) gets away with XCCL RS/AG is an interesting datapoint, but the EP path
  has *additional* IPC pressure (12 expert weight handles per layer × 30 layers) that
  may push it past whatever margin dense FSDP2 enjoys. **Note**: my earlier draft of
  this section suggested setting `XPU_USM_ALLOC_SO=usm_caching_alloc.so` to dodge the
  pluggable-allocator IPC bug — that is **wrong at this scale** per
  `feedback_alloc_conf_env_var.md`: the arena allocator has the same IPC
  sub-allocation crash as `expandable_segments` for 26B+ models, and the launcher
  correctly does `unset PYTORCH_ALLOC_CONF` and leaves `XPU_USM_ALLOC_SO` unset. The
  CPU-bounce architecture is therefore **necessary**, not optional, until the
  underlying L0/CCL IPC bug is fixed upstream. Don't try to remove it.
- **`use_reentrant=True`**: chosen v114 specifically because non-reentrant AC was
  triggering a SIGSEGV in the **AllToAll** path. Since AllToAll is gone (replaced by
  AG+RS), the v114 reason for reentrant AC no longer applies. Switching to
  `use_reentrant=False` is independently worth re-trying — it eliminates the *entire*
  category of "FWD recompute interleaved with BWD grad" issues that the doc fingers.
- **NTPE all-gather is needed at all**: AG-FWD already gives every rank the full
  `(EP*S, dim)` tensor. The NTPE collective only exists to compute the gather indices.
  These can be derived locally from the routed token positions (which are themselves
  expert-sorted on each source rank) if we standardize the per-rank S and a known
  cumsum scheme. Removing NTPE drops one gloo collective per layer and one source of
  desync.

### 5. Memory model is internally inconsistent

The two tables in §"Memory Model" disagree: the "without FSDP2" table says EP=4 saves
3 GiB on expert params; the "with FSDP2" table says EP=4 *peaks higher* than EP=1
because non-expert FSDP becomes coarser (3-rank vs 12-rank). Reconciled by the v17/v40
runtime data (project memory `project_ep_implementation.md`): EP=4 step 0 reaches
~12 GiB pre-step; the "memory savings" claim in §Goal (1 GiB/tile vs 4 GiB/tile)
is consumed by the +2.1 GiB of coarser FSDP and +4.2 GiB of expert optimizer state.
**Net memory at the achieved configuration is roughly neutral, not the 3 GiB/tile
saving promised in §Goal.** Doc should retract the headline savings claim until a
batch-size sweep shows EP enables a configuration EP=1 cannot.

### 6. The EP=1 vs EP=4 throughput finding is buried

`project_ep_implementation.md` (April 11): EP=4 is **2.6× slower end-to-end** at
batch=1, grpo_samples=2 — strictly worse on every phase, including 24× slower opt
because expert AdamW runs on CPU. The doc's §"What's Next" item 3 cites a target of
"≤+10% step time", but the only comparable measurement so far shows +160%. This
should be in §Validation Status, not memory only.

### 7. Path forward — concrete experiments before more compute

Re-prioritized after the 2026-04-22 audit (hypothesis 1 closed, 3 closed,
`XPU_USM_ALLOC_SO` route closed):

1. **Try `use_reentrant=False` again** — highest leverage, no compute prerequisite.
   The v114 reason to use reentrant AC was a SIGSEGV in the **AllToAll** path, which
   is gone. Non-reentrant AC eliminates the entire "interleaved FWD-recompute and
   BWD" category that the leading hypothesis (#2) depends on. If this works, the EP
   path stops being autograd-fragile.
2. **Drop the NTPE all-gather collective** by deriving gather indices locally
   (`routed_input.shape[0]` + local `num_tokens_per_expert` are sufficient given
   the v110 interleaved routing). One fewer gloo collective per layer; removes one
   source of desync regardless of (1).
3. **Add NTPE counting + COLL labels to `_EP_OP_N`** (~5 LOC). The current
   `_EP_OP_N` skips NTPE-AG, so per-rank logs are off-by-one-per-layer in a way that
   makes the v153 forensics ambiguous. Pure diagnostic — only worth doing if (1)+(2)
   don't fix the desync and we need clearer logs to decide between hypothesis (2)
   and a still-unidentified candidate.
4. **Verify the `topk` → `argsort(stable=True)` and `histc` → `bincount` fixes in
   `gemma4/_moe.py`** (per `project_ep_implementation.md`) are still in place and
   produce identical routing across ranks for identical inputs. If routing diverged
   after some refactor, hypothesis (2) becomes a different problem entirely.
5. **If 1–4 fail**: implement option 4 (move dispatch outside autograd) — but only
   after the cheap experiments are exhausted. This is the largest change and is only
   worth attempting once the cheap fixes are ruled out.

**Do not** retry XCCL for AG/RS with the arena/caching USM allocator — the IPC
sub-allocation bug bites at 26B+ scale (`feedback_alloc_conf_env_var.md`). The
4× CPU-bounce tax is the price of stability at this model size and is not the
bottleneck for an algorithm that doesn't run.

### 8. What the doc should say but doesn't

- Where the actual EP wiring lives (recipe lines 1873–2189, not the §"Setup Sequence"
  meta-pseudocode).
- That `apply_ep_weight_sharding` is no longer called post-load; pre-FSDP2 meta-param
  shrinking is the current mechanism.
- That `_token_dispatch`/`_token_combine` carry **per-instance state** (`_ag_gather_idx`,
  `_ag_s_local`), but each MoE layer has its own `ExpertParallel` instance
  (`gemma4/_parallelism.py:49`), so there is no cross-layer aliasing. The relevant
  saved tensors for backward are captured by `_AllGatherRS.forward`'s `ctx.save_for_backward`,
  not from `self.*`, so within-layer FWD-recompute does not corrupt BWD inputs either.
  This rules out the per-instance-state hypothesis.
- That every EP collective currently bounces through CPU; performance numbers must be
  read with that caveat.
- The 2.6× regression at batch=1 (from project memory) — currently absent from §Validation
  Status entirely.

### 9. Audit summary (2026-04-22)

After the code audit, the open question is narrower than the doc previously suggested:

- **Closed**: local-rank-conditional collectives (none exist in the EP path);
  per-instance-state aliasing across layers (each layer has its own
  `ExpertParallel`); switching the USM allocator to dodge the IPC bug (the
  arena/caching paths have the same bug at this scale).
- **Open**: why local-index-1 ranks are *consistently* one labeled-op behind their
  EP-group peers at op #258. The leading explanation is reentrant-AC scheduling
  reacting to per-rank routing imbalance, but it is not proven.
- **Cheapest next step**: `use_reentrant=False` retry. It directly tests the leading
  hypothesis, requires no new infrastructure, and can be tried inside one debug-queue
  job. If non-reentrant AC desyncs identically, hypothesis (2) is wrong and we need
  to look at routing determinism (item 4 in §7) before anything else.

---

## v154 Result (2026-04-23) — non-reentrant AC exposes router non-determinism

Test: `use_reentrant=True → False` at recipe `_no_reentrant_ac_wrapper`
(`recipes/dev/grpo_full_finetune_distributed_xpu.py:157-165`).

**Outcome (job 8445872, hold_ep_fsdp2_v154.out):**
- The op-#259 RS-BWD desync from v153 is **gone**. EP collective lockstep held
  through op #182 across all 12 ranks.
- New failure at `(_c_loss / grad_scale).backward()`:
  ```
  [rank8]:  RuntimeError: Function ScatterAddBackward0 returned an invalid gradient
           at index 1 - got [4579, 2816] but expected shape compatible with [4578, 2816]
  [rank10]: RuntimeError: Function ScatterAddBackward0 returned an invalid gradient
           at index 1 - got [7505, 2816] but expected shape compatible with [7506, 2816]
  ```
  Off-by-one in opposite directions → one token migrated between ranks 8 and 10
  during AC recompute relative to the original FWD.

**Mechanism.** `ExpertParallel._token_dispatch` saves `_ag_gather_idx` /
`_ag_s_local` as **instance attributes** on the `ExpertParallel` object
(`torchtune/modules/moe/_parallelism.py:511, 578`). Non-reentrant AC re-runs
the FWD during BWD; the recompute overwrites those attributes with
freshly-derived values. If the recomputed router (`Gemma4MoERouter` —
sigmoid + `argsort(stable=True)`) produces even bitwise-different `scores`,
top-k assignments at tie boundaries can flip → `bincount` differs by ±1
across ranks within an EP group → `_ag_gather_idx.numel()` shifts → the
saved `routed_output` tensor and the recomputed `idx_exp` no longer match in
shape during `partial_out.scatter_add(0, idx_exp, routed_output)`. The
gradient for `scatter_add` then has the recomputed gather count, not the
original. Reentrant AC happened to mask this because reentrant recompute
runs as a sub-graph that re-derives both sides consistently within one frame.

**Stale comment.** Recipe lines 144–149 describe a `_fwd_step_counter` /
`_is_reuse` / saved-`ntpe_group` cache that was supposed to make AC recompute
reuse the original AllToAll counts. **That cache does not exist anywhere in
the current source** (`grep '_fwd_step_counter\|_is_reuse\|saved_ntpe' torchtune/`
returns nothing). Either it was removed during the v141 AllToAll → AllGather/RS
rewrite, or it never landed. The recipe comment should be deleted or rewritten.

**Hypothesis status.** v154 confirms half of the v153 conclusion: the op-#259
desync was tied to reentrant AC scheduling, not to the gloo/CXI stack or the
NTPE collective. The "rank with smallest routed batch falls one op behind"
story for **collective-ordering** desync looks right. The router-determinism
problem is orthogonal and was previously hidden because reentrant AC papered
over it.

### Decision: pause before more compute

Three plausible fixes; none is obviously cheapest, and the right pick depends
on whether we want to keep AC at all:

1. **Cache `_ag_gather_idx` and `_ag_s_local` per AC region.** Add a
   per-instance save/restore so the recompute reuses the values from the
   original FWD instead of regenerating them. Smallest diff, but every
   `ExpertParallel` instance now holds a tensor-sized side-channel through
   the BWD pass — easy to forget on multi-step training, and it does not fix
   the underlying router-determinism issue (it just ensures both sides agree
   on the *first* answer). ~30 lines in `_parallelism.py`.
2. **Mark router outputs as saved tensors.** Force AC to keep
   `(scores, selected_experts, num_tokens_per_expert)` from the original FWD
   instead of recomputing them. Fixes the determinism issue at its source but
   adds memory (router outputs are O(T·num_experts) per layer; for 30 layers,
   not free). Requires a custom `checkpoint_fn` that snapshots through the
   router boundary.
3. **Move EP dispatch/combine outside autograd.** The plan from §7 option 4 of
   "What's Next" — restructure dispatch/combine as side-effecting ops with
   manual gradient handling. Cleanest decoupling and reusable for any
   MoE + AC combination, but the largest diff and the only one that needs a
   real plan before implementation.

The v155–v157 sequence (drop NTPE AG, add NTPE labels, routing-determinism
probes) was designed assuming reentrant AC; it doesn't address the
ScatterAddBackward0 mismatch and would burn allocation chasing the wrong
target. Hold-node 8445872 was returned without further launches.

**Status.** v154 hypothesis on collective-ordering desync confirmed. Router-
recompute determinism is the next blocker. No more compute until we pick one
of fixes 1–3 above.

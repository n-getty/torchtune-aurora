# IPEX BGMV LoRA kernels trigger GPU PDE under concurrent decode (Aurora XPU)

**Status**: ROOT CAUSE LOCALISED to IPEX `bgmv_*` kernels (2026-05-05). Workaround: merged-weight publish path (default in TorchTune LoRA-GRPO; `use_runtime_lora=false`). Upstream IPEX bug not yet filed.

## Summary

Passing `--enable-lora` to `vllm.entrypoints.openai.api_server` on Intel Data Center GPU Max 1550 (XPU) causes a deterministic `Segmentation fault from GPU` PDE fault under concurrent decode workloads (≥7 in-flight requests with `n>=24`, `max_tokens=512`). The fault signature is invariant:

```
Segmentation fault from GPU at 0xff00ffffffe00000,
  ctx_id: 1 (CCS) type: 0 (NotPresent), level: 1 (PDE),
  access: 0 (Read), banned: 1, aborting.
```

Three monkey-patch probes injected via `usercustomize.py` localise the trigger to the IPEX BGMV kernel chain inside `vllm.lora.punica_wrapper.punica_xpu.PunicaWrapperXPU`:

```
torch.ops.torch_ipex.bgmv_shrink         # called from add_shrink
torch.ops.torch_ipex.bgmv_expand_slice   # called from add_expand
torch.ops.torch_ipex.bgmv_expand         # called from add_lora_logits
```

The fault fires whether or not a LoRA adapter is loaded — `--enable-lora` alone routes every linear layer call through `add_lora_linear` even in the all-base-model case (the path runs the BGMV kernels with the metadata describing "no active adapter"), and that is enough to PDE.

**The workaround is**: do not pass `--enable-lora` to vLLM at all. Apply the LoRA delta on the trainer side (`W_eff = W + B@A·s`) and push the merged BF16 weights to vLLM via `/collective_rpc`. This skips BGMV entirely and is bit-exact to the runtime path.

## Environment

- **Hardware**: Intel Data Center GPU Max 1550 (Ponte Vecchio), 64 GiB HBM/tile
- **System**: ALCF Aurora HPC
- **i915 Driver**: I915_25.2.29
- **Level Zero**: 1.24.0
- **PyTorch**: 2.10.0a0+git449b176 (Aurora `frameworks/2025.3.1`)
- **IPEX**: shipped with frameworks/2025.3.1
- **vLLM**: 0.15.0 (XPU build shipped with frameworks/2025.3.1)
- **Model**: Qwen3-4B BF16, single tile via `ZE_AFFINITY_MASK=0`

## Minimal reproducer (vLLM application level — REPRODUCES)

`experiments/lora_grpo/lora_async_off_20260504_162145/probe_v9.sh` launches one vLLM tile and fires 7 concurrent curl requests against the base model:

```
vllm.entrypoints.openai.api_server
  --model Qwen3-4B
  --tensor-parallel-size 1
  --enforce-eager --dtype bfloat16
  --gpu-memory-utilization 0.85
  --max-model-len 1536 --max-num-seqs 64
  --enable-lora --max-lora-rank 16 --max-loras 2
  --distributed-executor-backend mp
  --no-async-scheduling
```

Workload: 7 prompts × `n=24` × `max_tokens=512`, all targeting the base model. KV cache usage stays under 5%; no memory pressure. Reliable repro: 6/7 requests succeed, 1 hits PDE; engine recovers and reports healthy. In production (12 tiles × DP=12) the per-step failure rate climbs and at least one tile dies fully, killing the recipe.

## Standalone PyTorch/IPEX reproducers — DO NOT REPRODUCE

A vLLM-free reproducer was attempted to give an upstream IPEX bug filer something they can run without spinning up vLLM. Two variants were tried on a fresh single tile (`x4709c5s6b0n0`, hold 8469834, `frameworks/2025.3.1`, `ZE_AFFINITY_MASK=0`). **Both PASS** — the standalone path cannot trigger the PDE that vLLM trips reliably.

| Repro | Shape / pattern | Calls | Result |
|-------|-----------------|-------|--------|
| `standalone_bgmv_repro.py` | fixed `n=168`, `bgmv_shrink + bgmv_expand` only, single thread | 1000 iters / 2000 calls | PASS, 0.06 s (16285 iter/s) |
| `standalone_bgmv_repro_v2.py` | variable `n ∈ {1..256}`, 36 fake layers × 200 steps, includes `bgmv_expand_slice` (QKV/gate_up fused-linear path that vLLM's `add_lora_linear` actually uses) | 200 steps / 57600 calls | PASS, 1.69 s (34148 calls/s) |

These results refine the H2 finding from the bisection. The trigger is **not** "call BGMV often enough" or "call BGMV with shape variation" or "exercise the slice path" — single-threaded sequential replay survives ~30× the call volume of the v9 vLLM crash. So the PDE requires *something else* that vLLM provides on top of the kernel calls themselves. Plausible candidates, in rough order of likelihood:

1. **Concurrent SYCL queue submissions from vLLM's V1 worker threads.** V1 dispatches per-request work on its own threads inside the EngineCore process. The standalone test runs in one Python thread and one default SYCL queue; vLLM may be hitting a kernel-internal race when bgmv kernels from different request contexts interleave on the same tile.
2. **Aliasing / lifetime interaction with PagedAttention KV-cache buffers.** The BGMV kernels run in the same SYCL queue context as the paged-attention reads/writes; an in-flight KV cache page table update concurrent with `bgmv_expand_slice` writing into a fused QKV output tensor could trip the PDE that the standalone (no KV cache) cannot.
3. **Specific pointer / stride patterns vLLM creates** that the standalone's freshly-allocated tensors do not (e.g. fused-linear output tensors that are slices of a larger contiguous arena, where the `add_inputs=True` accumulation reads the previous LoRA contribution).

What this means for the upstream IPEX bug filing: the v9 vLLM reproducer remains the only known minimal repro. It is small (one tile, one model, one server, one curl loop) and deterministic. Pair it with the v11 bisection patch (which makes v9 PASS) as the proof of where the fault lives.

The standalone scripts (kept for future debugging) live alongside the v9 probe under `experiments/lora_grpo/lora_async_off_20260504_162145/`.

### Notes on the standalone scripts (gotchas for upstream filers)

The IPEX `bgmv_*` API signatures and docstrings disagree with the actual assertions in `intel_extension_for_pytorch/transformers/models/xpu/fusions/activation_fusion.py`. Anyone writing a fresh reproducer will hit `AssertionError` until they line up with the assertions, not the docstrings:

- `lora_indices_tensor.dtype` must be `torch.int64` (docstring is silent; `int32` raises).
- `bgmv_shrink` `lora_a_weights` must be `[num_loras, rank, hidden_in]` — the IPEX docstring matches.
- `bgmv_expand` / `bgmv_expand_slice` `lora_b_weights` must be `[num_loras, hidden_out, rank]` — the IPEX docstring claims `[num_loras, rank, hidden_size]`, which is **wrong**; the assertion is `inputs.size(1) == lora_b_weights.size(-1)` so the rank axis must be last.
- `bgmv_expand_slice` takes per-slice `lora_b_weights` (i.e. `lb.size(-2) == slice_size`), not the fused output size.

## Bisection inside the LoRA path

Three probes patched a different layer of `vllm.lora.punica_wrapper`. All three reused the v9 reproducer above. Patches were injected via a `usercustomize.py` wrapped on `builtins.__import__`, with re-arm logic — vLLM's existing `_usercustomize_vllm/usercustomize.py` self-uninstalls its own import hook, which clobbers naively-chained hooks before `punica_xpu` is imported.

| Probe | Patch | n_choices | PDE | Verdict |
|-------|-------|-----------|-----|---------|
| v9 (baseline) | none | 6/7 | YES | reproduces |
| v11 | `PunicaWrapperXPU.add_lora_{linear,logits,embedding}` → no-op (returns `y` unchanged) | **7/7** | none | **PASS** |
| v12 | `convert_mapping` memoised by signature key (cap=256 LRU); reused for both `vllm.lora.punica_wrapper.utils` and `punica_base` (the latter does `from .utils import convert_mapping`) | 6/7 | YES | FAIL |
| v13 | persistent grown-on-demand FP32 buffer in `add_lora_linear` (replaces per-call `torch.zeros((n_slices, n_tok, r), dtype=fp32, device=xpu)` at `punica_xpu.py:207-215`); real `add_shrink`/`add_expand` calls preserved | 4/7 | YES | FAIL |

Probe sources: `experiments/lora_grpo/lora_async_off_20260504_162145/probe_v{11,12,13}.{sh,log}` and `probe_v{11,12,13}_uc/usercustomize.py`.

### What this rules in / out

- **Rules out per-step XPU tensor allocator churn from `convert_mapping`** (H1). v12 preserved every BGMV call but cached the metadata path so the per-step `torch.tensor(indices_list, dtype=long, device=xpu)` (utils.py line 113) and downstream `torch.where`/`torch.stack`/`torch.arange` allocations vanished. PDE still hit.
- **Rules out the per-call FP32 staging buffer alloc** at `punica_xpu.py:207-215` (sub-hypothesis of H1). v13 replaces that allocation with a persistent buffer reused across calls and grown only when needed (same persistent-buffer pattern that worked for IPEX `varlen_attention`). Real `add_shrink`/`add_expand` still ran. PDE hit; the rate was actually *worse* (4/7 vs baseline's 6/7), suggesting the stable-but-resliced view may even accelerate the fault.
- **Rules in the IPEX BGMV kernel chain** (H2). v11 is the only configuration that is clean, and the only thing v11 changes is to skip `add_shrink`/`add_expand`/`add_lora_logits` entirely. Metadata setup (`PunicaWrapperBase._update_base_metadata` → `convert_mapping`) and staging-buffer allocation are unchanged across v11 vs baseline. The differential isolates the fault to one or more of:

  - `torch.ops.torch_ipex.bgmv_shrink`
  - `torch.ops.torch_ipex.bgmv_expand_slice`
  - `torch.ops.torch_ipex.bgmv_expand`

The fault address class (`0xff00ffffffe00000`, level 1, NotPresent, banned:1) matches the WS8.5 `varlen_fwd` autograd-fallthrough corruption (memory `feedback_varlen_no_autograd_kernel.md`), but the trigger is unrelated — there is no autograd path through BGMV.

## Why probe instrumentation matters (`__import__`-hook re-arm)

The earlier "v11 hook installed but never patched" failure was caused by vLLM's pre-existing `_usercustomize_vllm/usercustomize.py` self-uninstalling its own `__import__` hook with `builtins.__import__ = _original_import` (lines 81-82) once its registry+xpu_worker patches fire. That clobbered the probe hook before `punica_xpu` was imported. Fix: every invocation of the probe hook re-checks `_builtins.__import__ is _vN_hook` and re-installs itself on top of whatever is there now. Future probes wrapping `__import__` from `usercustomize.py` on Aurora must do the same.

## Earlier sub-hypotheses that probes invalidated

- "SHM `cancelled` race in `step_with_batch_queue`" — disproven by `--no-async-scheduling` already being on across all v9-v13 probes.
- "Downstream node poisoning from a previously killed EngineCore" — disproven by reproducer running on a freshly held node with no prior vLLM activity.
- "LoRA forward kernels need an adapter to fault" — disproven by v9 firing against the base model with no adapter loaded; vLLM still routes through `add_lora_linear` because `--enable-lora` is set.
- "Memory pressure / KV cache exhaustion" — disproven by v9 vllm.log showing `GPU KV cache usage: 1.3%` at the moment of fault.

## Workarounds

### Validated (in tree)

**Merged-weight publish path** (`use_runtime_lora=false` — currently the TorchTune LoRA-GRPO default). Trainer rank 0 computes `W_eff = W + B@A·s` per LoRA-tagged linear, broadcasts the merged BF16 tensors over `_xccl_wsync_pg`, and vLLM workers swap the merged weights into their FusedLinear modules via `/collective_rpc`. vLLM is launched **without** `--enable-lora`. BGMV is never hit.

Bit-exact to the runtime path; cost is one extra `B@A·s` matmul on the trainer per step (negligible vs the wsync wire cost).

### Not workarounds (do not use)

- **Python-level skip-BGMV (v11-style)** is what the bisection uses to prove the kernel is at fault. It "works" only because it makes `--enable-lora` semantically equivalent to "load no adapter, ever," which is strictly worse than the merged-weight path (which actually applies the adapter).
- **Lower `--max-num-seqs`** — earlier hypothesis. Reduces fault frequency but does not eliminate the trigger; v9 hits PDE with KV cache at <5%. Not a fix.
- **`VLLM_USE_V1=0`** — does not apply (vLLM 0.15.0 only ships V1 on XPU).

## Recommended next steps

1. **File an IPEX bug** with the v9 reproducer + v11/v12/v13 bisection table + the three suspect kernel names. Minimal repro is one tile, one vLLM, 7 concurrent `n=24` requests against the base model.
2. **Stay on the merged-weight publish path** for production LoRA-GRPO. It is the default and has been validated end-to-end on Qwen3-4B GSM8K (`project_lora_grpo_4b_postfix_validated.md`) and is being exercised on Qwen3-8B gene_recall.
3. **Cheap upstream check**: install vLLM main + IPEX nightly on a held tile and re-run `probe_v9.sh`. If BGMV is fixed upstream, the workaround can retire and `--enable-lora` can come back. If still broken, the IPEX bug is the only path forward.
4. **Do not** layer additional Python-level workarounds onto `--enable-lora`; the bisection has shown the kernel chain is the trigger and a Python wrapper that skips the kernels delivers no LoRA inference.

## Cross-references

- Full session report: `docs/reports/enable_lora_issue.md` (includes the iter1/iter2 trainer-side PDE narrative on the merged-weight path itself, which is a separate issue).
- Memory `project_lora_grpo_vllm_crash_mechanism.md` — bisection result + workaround guidance.
- Memory `project_lora_grpo_merged_weight_path.md` — merged-weight publish path implementation.
- Memory `feedback_usercustomize_eager_vllm_import.md` — why the probe `__import__` hooks must re-arm after `_usercustomize_vllm` self-uninstalls.
- Reproducer scripts (kept on shared FS):
  - `experiments/lora_grpo/lora_async_off_20260504_162145/probe_v9.sh` (vLLM baseline, REPRODUCES)
  - `experiments/lora_grpo/lora_async_off_20260504_162145/probe_v11.sh` + `probe_v11_uc/usercustomize.py` (BGMV no-op, PASS)
  - `experiments/lora_grpo/lora_async_off_20260504_162145/probe_v12.sh` + `probe_v12_uc/usercustomize.py` (convert_mapping cache, FAIL)
  - `experiments/lora_grpo/lora_async_off_20260504_162145/probe_v13.sh` + `probe_v13_uc/usercustomize.py` (persistent FP32 buffer, FAIL)
  - `experiments/lora_grpo/lora_async_off_20260504_162145/standalone_bgmv_repro.py` + `run_standalone_bgmv.sh` (PASS, single-thread fixed shape)
  - `experiments/lora_grpo/lora_async_off_20260504_162145/standalone_bgmv_repro_v2.py` + `run_standalone_bgmv_v2.sh` (PASS, variable shape + slice path + 36-layer pattern)

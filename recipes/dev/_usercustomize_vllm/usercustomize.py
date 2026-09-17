"""
Patches for Aurora frameworks environment, applied to all Python processes.

1. Transformers version check: huggingface-hub 1.7.x vs <1.0 requirement.
2. vLLM registry subprocess: segfaults on XPU, run in-process instead.
   MUST be patched BEFORE vllm.config is imported (vLLM 0.15.0 changed the
   import chain so the registry is reached during config.__init__).
3. XPU memory: fix vLLM XPUWorker.determine_available_memory on Aurora.
"""
import importlib
import importlib.util
import os
import sys
import types

_pid = os.getpid()

# --- Patch 1: transformers version check ---
# Pre-register stub modules to prevent the version check from firing
# during transformers import. The check rejects huggingface-hub 1.7.x.
_m = types.ModuleType("transformers.utils.versions")
_m.require_version = lambda *a, **kw: None
_m.require_version_core = lambda *a, **kw: None
_m._compare_versions = lambda *a, **kw: None
sys.modules["transformers.utils.versions"] = _m

_dvc = types.ModuleType("transformers.dependency_versions_check")
_dvc.dep_version_check = lambda *a, **kw: None
sys.modules["transformers.dependency_versions_check"] = _dvc

# --- Patches 2 & 3: lazy via __import__ hook ---
# Patch 2 must happen BEFORE vllm.config is imported (vLLM 0.15 reaches the
# registry during config.__init__). DO NOT eager-import the registry here:
# that pulls in the full vllm package during interpreter startup, which
# kills the EngineCore spawn child silently (it dies before any logging,
# parent shows "Failed core proc(s): {}"). Use the same lazy hook pattern
# as Patch 3, triggered when `vllm.config` is being loaded.
import builtins
_original_import = builtins.__import__
_registry_patched = False
_xpu_worker_patched = False
_prompt_embeds_patched = False
_torch_threads_configured = False
_in_hook = False


def _patching_import(name, *args, **kwargs):
    global _registry_patched, _xpu_worker_patched, _prompt_embeds_patched
    global _torch_threads_configured
    global _in_hook

    mod = _original_import(name, *args, **kwargs)

    if _in_hook:
        return mod
    _in_hook = True

    try:
        if (
            not _torch_threads_configured
            and os.environ.get("TORCHTUNE_VLLM_CPU_THREADS")
        ):
            _torch = sys.modules.get("torch")
            if _torch is not None and hasattr(_torch, "set_num_threads"):
                _threads = int(os.environ["TORCHTUNE_VLLM_CPU_THREADS"])
                _torch.set_num_threads(_threads)
                _torch.set_num_interop_threads(_threads)
                _torch_threads_configured = True

        # Patch 2: registry subprocess. The registry module itself is
        # imported by vllm.config; trigger our patch as soon as it's in
        # sys.modules so subsequent _run_in_subprocess calls hit the safe
        # fallback.
        if not _registry_patched:
            _reg = sys.modules.get("vllm.model_executor.models.registry")
            if _reg is not None and hasattr(_reg, "_run_in_subprocess"):
                _orig_run = _reg._run_in_subprocess

                def _safe_run_in_subprocess(fn):
                    try:
                        return _orig_run(fn)
                    except Exception:
                        return fn()

                _reg._run_in_subprocess = _safe_run_in_subprocess
                _registry_patched = True

        # Patch 3: XPU memory. xpu_worker is imported much later during
        # engine init.
        if not _xpu_worker_patched:
            _xpu_mod = sys.modules.get("vllm.v1.worker.xpu_worker")
            if _xpu_mod is not None and hasattr(_xpu_mod, "XPUWorker"):
                _xpu_worker_patched = True
                _apply_xpu_memory_patch(_xpu_mod.XPUWorker)

        # Patch 5: stale is_token_ids.gpu on the async-scheduling pure-decode
        # path (backport of upstream vLLM PR #45673, merged 2026-06-16, which
        # postdates this 0.15.0 pin). See _apply_prompt_embeds_patch below.
        if not _prompt_embeds_patched:
            _gmr = sys.modules.get("vllm.v1.worker.gpu_model_runner")
            if _gmr is not None and hasattr(_gmr, "GPUModelRunner"):
                _prompt_embeds_patched = True
                _apply_prompt_embeds_patch(_gmr.GPUModelRunner)
                # Patch 6 (opt-in): bounds-check the embedding gather.
                if os.environ.get("TORCHTUNE_VLLM_EMBED_GUARD") == "1":
                    _apply_embed_guard_patch(_gmr.GPUModelRunner)

        if (
            _registry_patched
            and _xpu_worker_patched
            and _prompt_embeds_patched
            and (
                _torch_threads_configured
                or not os.environ.get("TORCHTUNE_VLLM_CPU_THREADS")
            )
        ):
            builtins.__import__ = _original_import
    finally:
        _in_hook = False

    return mod


def _apply_embed_guard_patch(GPUModelRunner):
    """Opt-in (`TORCHTUNE_VLLM_EMBED_GUARD=1`): bounds-check the embedding gather.

    WHY. banned:1 on this harness fires at a BYTE-IDENTICAL GPU address on every
    crash (`0xff00ffffffe00000`), across different nodes, tiles and jobs. That is
    the signature of a deterministic bad *index*, not a driver flake or memory
    pressure (crashes happen at 2-3% KV usage). The prime suspect is the
    prompt_embeds branch of `_preprocess`, which gathers
    `input_ids.gpu[token_ids_idx]` and feeds the result to `embed_input_ids` --
    an out-of-range token id there indexes off the end of the embedding table and
    produces exactly this class of fault.

    Five config-level hypotheses have already been refuted by A/B (KV
    over-allocation, prefix caching, NEO debug keys, CCL tuning, async
    scheduling). Rather than guess a sixth, catch the bad value in the act.

    WHAT IT DOES. Before the gather, copies the candidate ids to CPU and checks
    them against `[0, vocab_size)`. On violation it logs the offending values,
    their positions, the batch shape and the mask population count, then CLAMPS
    into range so the process survives to report instead of aborting -- a fatal
    abort loses the diagnostic, which is what has happened on every crash so far.

    COST. One D2H sync of a small int32 tensor per decode step. Real but modest;
    strictly a diagnostic mode, hence opt-in and default OFF.

    NOTE this deliberately wraps `_preprocess`, not `embed_input_ids`: we need the
    ids at the point they are *selected* by the mask, to tell an out-of-range id
    apart from a correct id selected at a wrong position.
    """
    orig = GPUModelRunner._preprocess

    def _preprocess(self, *args, **kwargs):
        try:
            if getattr(self, "enable_prompt_embeds", False):
                # Mirror vLLM's own selection exactly: it slices is_token_ids by
                # total_num_scheduled_tokens (the UNpadded count), which comes off
                # scheduler_output -- not by num_input_tokens (padded).
                so = kwargs.get("scheduler_output") or (args[0] if args else None)
                n = getattr(so, "total_num_scheduled_tokens", None)
                idx = self.is_token_ids.gpu[:n].nonzero(as_tuple=False).squeeze(1) \
                    if n else None
                if idx is not None and idx.numel() > 0:
                    ids = self.input_ids.gpu[idx]
                    vocab = int(self.model_config.get_vocab_size())
                    lo = int(ids.min().item())
                    hi = int(ids.max().item())
                    if lo < 0 or hi >= vocab:
                        bad = ((ids < 0) | (ids >= vocab)).nonzero().squeeze(1)
                        print(
                            f"[embed_guard] OUT-OF-RANGE token id(s) before "
                            f"embed_input_ids: min={lo} max={hi} vocab={vocab} "
                            f"n_bad={bad.numel()} of {ids.numel()} selected "
                            f"(mask popcount={idx.numel()}, num_scheduled={n}); "
                            f"bad_positions={bad[:16].tolist()} "
                            f"bad_values={ids[bad[:16]].tolist()} -- CLAMPING so "
                            f"the process survives to report this",
                            flush=True,
                        )
                        self.input_ids.gpu[idx] = ids.clamp_(0, vocab - 1)
                    else:
                        # In-range ids do NOT mean the mask was right: a stale mask can
                        # select valid-looking ids at wrong positions, which would not
                        # fault here but still corrupts the batch. Emit a compact
                        # heartbeat of the mask population vs what the scheduler says is
                        # a decode step, so a divergence is visible in the log even on
                        # steps that don't trip the bounds check. Rate-limited to keep
                        # this out of the way on healthy runs.
                        k = getattr(self, "_embed_guard_seen", 0) + 1
                        self._embed_guard_seen = k
                        nreq = len(getattr(so, "num_scheduled_tokens", ()) or ())
                        # Pure-decode step => one token per running request.
                        pure_decode = (nreq > 0 and n == nreq)
                        prev = getattr(self, "_embed_guard_prev_pop", None)
                        pop = int(idx.numel())
                        if pure_decode and prev is not None and pop != prev:
                            print(
                                f"[embed_guard] mask population CHANGED on a pure-decode "
                                f"step: {prev} -> {pop} (num_scheduled={n}, "
                                f"running_reqs={nreq}, ids in range) -- not fatal, but "
                                f"this is where a stale is_token_ids would show up",
                                flush=True,
                            )
                        self._embed_guard_prev_pop = pop
                        if k % 200 == 1:
                            print(
                                f"[embed_guard] ok: step~{k} num_scheduled={n} "
                                f"mask_pop={pop} running_reqs={nreq} "
                                f"id_range=[{lo},{hi}) vocab={vocab}",
                                flush=True,
                            )
        except Exception as e:  # never let the guard itself kill the engine
            print(f"[embed_guard] check skipped: {type(e).__name__}: {e}", flush=True)
        return orig(self, *args, **kwargs)

    GPUModelRunner._preprocess = _preprocess
    print(
        f"[usercustomize] PID={os.getpid()} embed_guard ACTIVE "
        "(TORCHTUNE_VLLM_EMBED_GUARD=1): bounds-checking input_ids before "
        "embed_input_ids on the prompt_embeds path; adds a per-step D2H sync",
        flush=True,
    )


def _apply_prompt_embeds_patch(GPUModelRunner):
    """Fix stale `is_token_ids.gpu` on the async-scheduling decode fast path.

    Backport of upstream vLLM PR #45673 ("[BugFix] Support async scheduling
    with prompt embeds"), merged 2026-06-16 — after the 0.15.0 build pinned in
    frameworks/2025.3.1, so this installation still has the bug.

    THE BUG. `_prepare_input_ids` uploads `is_token_ids` to the GPU only inside
    the `num_commmon_tokens < total_without_spec` branch. On a *pure decode*
    step where every request carried over from the previous iteration, that
    branch is skipped; if the batch was also reordered (`InputBatch.condense`
    after some request finished, so `indices_match` is False), control reaches
    the `input_ids.gpu.scatter_()` tail, which refreshes `input_ids.gpu` but
    leaves `is_token_ids.gpu` holding the PREVIOUS step's mask.

    `_preprocess` then does `is_token_ids.gpu.nonzero()` to pick which rows are
    real token ids, gathers `input_ids.gpu[those_rows]`, and feeds them to the
    embedding lookup. With prompt_embeds requests the prompt region of
    `token_ids_cpu` is never written at all (gpu_input_batch.py only sets
    `is_token_ids[...] = False` there), so a stale mask selects uninitialised
    rows and hands garbage int32 indices to an embedding gather → out-of-bounds
    device read → L0 reports "Segmentation fault from GPU ... NotPresent, PDE,
    banned: 1" and the driver aborts the process.

    WHY IT MATCHED OUR CRASHES. Every captured fault was a pure decode batch of
    8, at 2-3% KV usage (so not memory pressure), at a byte-identical address on
    every node — the hallmark of a deterministic bad index, not a race or a bad
    tile. Serialising to max_num_seqs=1 was clean because with one request the
    batch never reorders, `indices_match` holds, and that fast path *does* set
    `is_token_ids.gpu = True`.

    THE FIX. Hoist the upload so `is_token_ids.gpu` is refreshed on every path.
    Cheap: one H2D copy of a bool tensor of length total_num_scheduled_tokens.

    Only patches when the buggy pattern is actually present, and only for
    prompt_embeds runs; on a build that already contains #45673 this is a no-op.
    """
    orig = GPUModelRunner._prepare_input_ids

    def _prepare_input_ids(self, *args, **kwargs):
        # Accept the count positionally (how vLLM 0.15.0 calls it) or by keyword,
        # so a future signature tweak degrades to "no refresh" rather than a
        # TypeError that would take the engine down.
        n = kwargs.get("total_num_scheduled_tokens")
        if n is None and len(args) >= 2:
            n = args[1]
        # Nothing to do when prompt embeds are off (is_token_ids is unused) or
        # on the synchronous path (which already uploads unconditionally).
        if n is not None and getattr(self, "enable_prompt_embeds", False) and \
                getattr(self.input_batch, "prev_sampled_token_ids", None) is not None:
            try:
                # Safe here: is_token_ids.cpu was just filled by _prepare_inputs
                # (gpu_model_runner.py:1486) before this call (:1557).
                self.is_token_ids.copy_to_gpu(n)
            except Exception:
                pass
        return orig(self, *args, **kwargs)

    GPUModelRunner._prepare_input_ids = _prepare_input_ids
    print(
        f"[usercustomize] PID={os.getpid()} GPUModelRunner._prepare_input_ids "
        "patched (vLLM PR #45673 backport: refresh is_token_ids.gpu on the "
        "async-scheduling decode path; fixes prompt_embeds banned:1 GPU fault)",
        flush=True,
    )


def _apply_xpu_memory_patch(XPUWorker):
    """Patch XPUWorker.determine_available_memory for Aurora L0 context overhead."""
    import torch
    import logging
    _logger = logging.getLogger("usercustomize_xpu")

    @torch.inference_mode()
    def _patched_determine_available_memory(self) -> int:
        """KV cache budget based on actual free memory after profiling.

        On Aurora, L0 driver contexts consume ~52 GiB. The default impl
        computes budget as total(64G) * util - peak, which over-allocates.
        We use actual free memory from mem_get_info() after profiling.
        """
        torch.xpu.empty_cache()
        torch.xpu.reset_peak_memory_stats()

        free_before, total = torch.xpu.mem_get_info()
        _logger.info(
            "XPU before profile: total=%.1f GiB, free=%.1f GiB, "
            "pytorch=%.1f GiB",
            total / 1024**3, free_before / 1024**3,
            torch.xpu.memory_allocated() / 1024**3,
        )

        self.model_runner.profile_run()
        torch.xpu.empty_cache()

        free_after, _ = torch.xpu.mem_get_info()
        pytorch_current = torch.xpu.memory_allocated()

        available = int(free_after * self.cache_config.gpu_memory_utilization)

        _logger.info(
            "XPU memory (patched): free_after=%.1f GiB, "
            "pytorch=%.1f GiB, util=%.0f%%, KV budget=%.1f GiB",
            free_after / 1024**3,
            pytorch_current / 1024**3,
            self.cache_config.gpu_memory_utilization * 100,
            available / 1024**3,
        )
        return available

    XPUWorker.determine_available_memory = _patched_determine_available_memory
    print(
        f"[usercustomize] PID={_pid} "
        "XPUWorker.determine_available_memory patched",
        flush=True,
    )


builtins.__import__ = _patching_import

# --- Patch 4: optional XPU TP gloo all_reduce fallback ---
# Activated only when TORCHTUNE_VLLM_XPU_GLOO_TP=1. Used to bypass the vLLM
# TP>1 prefill XCCL all_reduce hang at plen>32 on torch211 stack. Module
# self-installs via builtins.__import__ hook on first vllm xpu_communicator
# import. No-op when env unset.
try:
    import _xpu_gloo_allreduce_patch  # noqa: F401  -- top-level on PYTHONPATH
except Exception as _e:
    print(f"[usercustomize] xpu gloo patch import failed: {_e}", flush=True)

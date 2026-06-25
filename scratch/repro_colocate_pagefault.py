#!/usr/bin/env python3
"""Standalone reproducer for the in-process colocate vLLM generation page fault (XPU).

DEPENDENCY-LIMITED — imports ONLY torch + vllm (+ optional intel_extension_for_pytorch).
NO torchtune imports. This is the artifact handed to Intel / vLLM if the fault turns out to
be below our RL framework: it constructs the SAME in-process TP=1 vLLM engine the LoRA-GRPO
colocate recipe uses (kwargs + monkeypatches copied verbatim from
torchtune/dev/rl/vllm_backend.py:_init_vllm_tp1) and reproduces the
`CCS NotPresent / PDE/PTE banned:1` GPU page fault that fires during generation when a torch
"trainer" workload is co-resident on the same Aurora PVC tile.

The script is a LADDER: each flag adds one ingredient of the real colocate loop so the first
rung that faults names the responsible layer:

  rung    resident-model  torch-compute  empty-cache  reset-prefix/load/fsdp   hypothesis
  R-A     none            off            off          off                      H2 pure vLLM
  R-B     qwen/tensor     off            off          off                      H3 static co-residence
  R-C     qwen            fwdbwd         off          off                      H3 live co-tenancy
  R-D     qwen            fwdbwd         each-gen     off                      H1 allocator free/remap
  R-E     qwen            fwdbwd         off          reset|load|fsdp          H1 which free/remap op

A GPU page fault ABORTS the process (SIGABRT / segfault) — Python cannot catch it. So the
script prints exactly ONE terminal line on CLEAN completion:

    REPRO_DONE node=<host> tile=<t> rung=<r> iters_done=<n> crashed=0 free_gib=<f>

The driver shell (run_repro_ladder.sh) infers a CRASH from: nonzero exit code AND/OR absence
of the REPRO_DONE line AND/OR a `banned:1`/`PDE`/`UR_RESULT` signature in stderr/dmesg.

Single tile per process. Run 12 in parallel (ZE_AFFINITY_MASK=0..11) for N=12 per trial.

Usage (single tile, faithful to 4B colocate):
  ZE_AFFINITY_MASK=0 python3 scratch/repro_colocate_pagefault.py \
      --model /lus/flare/projects/ModCon/ngetty/models/Qwen3-4B \
      --rung R-D --max-gen 768 --max-model-len 1536 --iters 60 --burst-every 4
"""

import argparse
import os
import socket
import sys
import time


# ---------------------------------------------------------------------------
# Env MUST be set before `import vllm`. Mirror vllm_backend.py:96-100, 297-313.
# ---------------------------------------------------------------------------
def _prime_env(local_rank: int) -> None:
    # vLLM V1: disable multiprocessing (EngineCore subprocess hangs on XPU colocate).
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    # vLLM sees a world of 1 (the recipe pops the torchrun vars and sets these).
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(29599 + local_rank)
    # CRITICAL: pin to one tile (mem_get_info ignores its device arg on XPU).
    # ZE_AFFINITY_MASK is normally exported by the driver shell; honor it if set,
    # else set from local_rank so a bare invocation still pins.
    os.environ.setdefault("ZE_AFFINITY_MASK", str(local_rank))
    # ALCF proxy bypass + offline (mirror _vllm_env_setup.sh) — harmless if already set.
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ftp_proxy"):
        os.environ.pop(k, None)
    os.environ.setdefault("no_proxy", "*")
    os.environ.setdefault("NO_PROXY", "*")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("ZE_FLAT_DEVICE_HIERARCHY", "FLAT")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.pop("PYTORCH_ALLOC_CONF", None)


def _log(msg: str) -> None:
    print(f"[repro {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _free_gib(torch, device) -> float:
    try:
        free, _ = torch.xpu.mem_get_info(device)
        return free / 1024**3
    except Exception:
        return -1.0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="HF model dir (faithful: Qwen3-4B)")
    p.add_argument(
        "--rung",
        default="R-A",
        choices=["R-A", "R-B", "R-C", "R-D", "R-E", "R-LW"],
        help="ladder rung (see module docstring); overrides individual ingredient flags",
    )
    # Individual ingredient flags (rung presets set these; explicit flags override).
    p.add_argument("--resident-model", default=None,
                   choices=["none", "tensor", "qwen"],
                   help="co-resident HBM occupant: none / a big bf16 tensor / a synthetic "
                        "transformer-ish stack sized to ~the 4B trainer footprint")
    p.add_argument("--torch-compute", default=None, choices=["off", "fwd", "fwdbwd"],
                   help="interleave trainer-like compute between generate calls")
    p.add_argument("--empty-cache", default=None, choices=["off", "each-gen", "stride"],
                   help="call torch.xpu.empty_cache() between generates (H1 trigger)")
    p.add_argument("--empty-cache-stride", type=int, default=5)
    p.add_argument("--reset-prefix", default=None, choices=["off", "each"],
                   help="call llm_engine.reset_prefix_cache() each iter (recipe line 1481)")
    p.add_argument("--load-weights", default=None, choices=["off", "once", "each"],
                   help="exercise model.load_weights() (the per-step adapter publish path)")
    p.add_argument("--fsdp", default=None, choices=["off", "on"],
                   help="wrap the resident model in FSDP to exercise storage.resize_()")
    # vLLM engine envelope (faithful defaults to the colocate YAML).
    p.add_argument("--max-gen", type=int, default=768)
    p.add_argument("--max-model-len", type=int, default=1536)
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--gpu-mem", type=float, default=0.35)
    p.add_argument("--num-gpu-blocks", type=int, default=0,
                   help="num_gpu_blocks_override; 0 = auto (batch*grpo*blocks_per_seq*1.1)")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--grpo", type=int, default=8, help="seqs per generate (batch*grpo)")
    p.add_argument("--enable-prefix-cache", action="store_true")
    p.add_argument("--temperature", type=float, default=1.0)
    # Loop control.
    p.add_argument("--iters", type=int, default=60, help="generate iterations")
    p.add_argument("--burst-every", type=int, default=4,
                   help="every Nth iter, force a full max-len ignore_eos burst (long-rollout "
                        "spike that raises P(crash) per the bug doc); 0 disables")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt-len", type=int, default=256,
                   help="base prompt length for normal (non-burst) iters")
    args = p.parse_args()

    # Resolve rung presets -> ingredients (explicit flags win).
    preset = {
        "R-A": dict(resident_model="none", torch_compute="off", empty_cache="off",
                    reset_prefix="off", load_weights="off", fsdp="off"),
        "R-B": dict(resident_model="qwen", torch_compute="off", empty_cache="off",
                    reset_prefix="off", load_weights="off", fsdp="off"),
        "R-C": dict(resident_model="qwen", torch_compute="fwdbwd", empty_cache="off",
                    reset_prefix="off", load_weights="off", fsdp="off"),
        "R-D": dict(resident_model="qwen", torch_compute="fwdbwd", empty_cache="each-gen",
                    reset_prefix="off", load_weights="off", fsdp="off"),
        "R-E": dict(resident_model="qwen", torch_compute="fwdbwd", empty_cache="off",
                    reset_prefix="each", load_weights="each", fsdp="off"),
        # R-LW: the CONFIRMED trigger in minimal form — vLLM + real-weight load_weights() into
        # the live engine each step, NO resident trainer, NO compute, NO XCCL. If this faults on
        # ONE tile alone, the reproducer collapses to a single-process script (simplest handoff).
        "R-LW": dict(resident_model="none", torch_compute="off", empty_cache="off",
                     reset_prefix="off", load_weights="each", fsdp="off"),
    }[args.rung]
    resident_model = args.resident_model or preset["resident_model"]
    torch_compute = args.torch_compute or preset["torch_compute"]
    empty_cache = args.empty_cache or preset["empty_cache"]
    reset_prefix = args.reset_prefix or preset["reset_prefix"]
    load_weights = args.load_weights or preset["load_weights"]
    fsdp = args.fsdp or preset["fsdp"]

    # ZE_AFFINITY_MASK may name the tile; local_rank=0 because the mask already
    # restricts visibility to a single device (so torch sees device index 0).
    aff = os.environ.get("ZE_AFFINITY_MASK", "0")
    tile = aff.split(",")[0]
    _prime_env(local_rank=0)

    host = socket.gethostname()
    _log(f"host={host} tile={tile} rung={args.rung} model={args.model}")
    _log(f"ingredients: resident={resident_model} compute={torch_compute} "
         f"empty_cache={empty_cache} reset_prefix={reset_prefix} "
         f"load_weights={load_weights} fsdp={fsdp}")

    import torch
    try:
        import intel_extension_for_pytorch as ipex  # noqa: F401
    except Exception:
        pass

    assert torch.xpu.is_available(), "XPU not available — run on a compute node"
    device = torch.device("xpu:0")
    torch.xpu.set_device(device)
    torch.manual_seed(args.seed)

    # -----------------------------------------------------------------------
    # Build the in-process vLLM engine — VERBATIM from vllm_backend.py.
    # -----------------------------------------------------------------------
    import tempfile
    _store_file = tempfile.mktemp(prefix="vllm_gloo_store_repro_")
    torch.distributed.init_process_group(
        backend="gloo", init_method=f"file://{_store_file}", world_size=1, rank=0,
    )
    _orig_new_group = torch.distributed.new_group

    def _gloo_new_group(*a, **k):
        k["backend"] = "gloo"
        return _orig_new_group(*a, **k)
    torch.distributed.new_group = _gloo_new_group

    _orig_all_reduce = torch.distributed.all_reduce

    def _safe_all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=None, async_op=False):
        if group is not None and group.size() == 1:
            return None
        if tensor.is_xpu:
            return None
        return _orig_all_reduce(tensor, op=op, group=group, async_op=async_op)
    torch.distributed.all_reduce = _safe_all_reduce

    from vllm import LLM, SamplingParams
    from vllm.v1.executor.uniproc_executor import UniProcExecutor
    _orig_distributed_args = UniProcExecutor._distributed_args

    def _patched_distributed_args(self_exec):
        method, _rank, _lr = _orig_distributed_args(self_exec)
        return method, _rank, 0
    UniProcExecutor._distributed_args = _patched_distributed_args

    num_seqs = max(args.batch * args.grpo, 1)
    if args.num_gpu_blocks > 0:
        block_override = args.num_gpu_blocks
    else:
        block_size = 16
        blocks_per_seq = (args.max_model_len + block_size - 1) // block_size
        block_override = int(num_seqs * blocks_per_seq * 1.1)

    llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=True,
        dtype="bfloat16",
        disable_custom_all_reduce=True,
        num_gpu_blocks_override=block_override,
        enable_prefix_caching=bool(args.enable_prefix_cache),
    )
    _log(f"LLM kwargs: num_gpu_blocks_override={block_override} "
         f"max_model_len={args.max_model_len} gpu_mem={args.gpu_mem} "
         f"enable_prefix_caching={bool(args.enable_prefix_cache)}")
    t0 = time.perf_counter()
    llm = LLM(**llm_kwargs)
    _log(f"vLLM engine up in {time.perf_counter()-t0:.1f}s; free={_free_gib(torch, device):.2f} GiB")

    # Restore monkeypatches (recipe does this post-init; training then proceeds).
    UniProcExecutor._distributed_args = _orig_distributed_args
    torch.distributed.new_group = _orig_new_group
    torch.distributed.all_reduce = _orig_all_reduce

    # Handle to the resident vLLM model (for load_weights / reset_prefix faithfulness).
    try:
        vllm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    except Exception as e:
        vllm_model = None
        _log(f"WARN could not reach vllm model handle: {e!r}")

    # -----------------------------------------------------------------------
    # Co-resident "trainer" — synthetic, torch-only (mimics HBM + alloc churn).
    # -----------------------------------------------------------------------
    trainer = None
    opt = None
    if resident_model == "tensor":
        # ~8 GiB bf16 occupant (mimic the 4B frozen base residency).
        n = int(8 * 1024**3 / 2)  # bf16 elements
        trainer = torch.empty(n, dtype=torch.bfloat16, device=device)
        _log(f"resident tensor allocated; free={_free_gib(torch, device):.2f} GiB")
    elif resident_model == "qwen":
        # Synthetic transformer-ish stack: large Linear layers exercise the same
        # alloc/free pattern as the trainer fwd/bwd without needing transformers.
        H, FF, L = 2560, 9728, 12  # Qwen3-4B-ish hidden/ffn, truncated depth for footprint
        layers = []
        for _ in range(L):
            layers += [torch.nn.Linear(H, FF, bias=False, dtype=torch.bfloat16),
                       torch.nn.Linear(FF, H, bias=False, dtype=torch.bfloat16)]
        trainer = torch.nn.Sequential(*layers).to(device)
        if fsdp == "on":
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            # world=1 gloo PG is still alive (we only destroy it at teardown); wrap to
            # exercise FSDP storage.resize_() on reshard between fwd/bwd.
            trainer = FSDP(trainer, use_orig_params=True)
            _log("resident model FSDP-wrapped (storage.resize_ path active)")
        opt = torch.optim.SGD(trainer.parameters(), lr=1e-6)
        _log(f"resident synthetic model built (H={H} FF={FF} L={L}); "
             f"free={_free_gib(torch, device):.2f} GiB")

    def _trainer_step():
        if trainer is None or torch_compute == "off":
            return
        x = torch.randn(args.batch * 4, 2560, dtype=torch.bfloat16, device=device)
        if isinstance(trainer, torch.Tensor):
            return
        if torch_compute == "fwd":
            with torch.no_grad():
                _ = trainer(x)
        else:  # fwdbwd
            opt.zero_grad(set_to_none=True)
            y = trainer(x)
            loss = y.float().pow(2).mean()
            loss.backward()
            opt.step()
        torch.xpu.synchronize(device)

    # Load the REAL model HF-named safetensors so vLLM's load_weights actually executes
    # (the engine's own fused param names are REJECTED — that was the v5 false-negative).
    # This is THE confirmed trigger (multi-tile v6): real weight-publish into the live engine.
    _real_named = None
    if load_weights != "off" and vllm_model is not None:
        try:
            import glob as _glob
            from safetensors import safe_open
            _shards = sorted(_glob.glob(os.path.join(args.model, "*.safetensors")))
            _real_named = []
            for _sh in _shards:
                with safe_open(_sh, framework="pt", device="cpu") as _f:
                    for _k in _f.keys():
                        _real_named.append((_k, _f.get_tensor(_k).to(device, torch.bfloat16)))
            _log(f"real-weights path ready ({len(_real_named)} HF params from {len(_shards)} shards)")
        except Exception as e:
            _log(f"WARN cannot load real safetensors: {e!r}")

    def _do_load_weights():
        if vllm_model is None or load_weights == "off" or _real_named is None:
            return
        try:
            vllm_model.load_weights(_real_named)
        except Exception as e:
            _log(f"WARN load_weights failed: {e!r}")

    def _do_reset_prefix():
        if reset_prefix == "off":
            return
        try:
            llm.llm_engine.reset_prefix_cache()
        except Exception as e:
            _log(f"WARN reset_prefix_cache failed: {e!r}")

    # -----------------------------------------------------------------------
    # The loop: generate, interleave trainer compute + free/remap ops.
    # -----------------------------------------------------------------------
    import random
    rng = random.Random(args.seed)
    vocab = 150000
    if load_weights == "once":
        _do_load_weights()

    iters_done = 0
    total_tok = 0
    for it in range(args.iters):
        burst = args.burst_every > 0 and (it % args.burst_every == 0) and it > 0
        if burst:
            plen = args.max_model_len - args.max_gen
            sp = SamplingParams(max_tokens=args.max_gen, temperature=args.temperature,
                                top_k=-1, detokenize=False, ignore_eos=True)
        else:
            plen = args.prompt_len
            sp = SamplingParams(max_tokens=args.max_gen, temperature=args.temperature,
                                top_k=-1, detokenize=False)
        prompts = [{"prompt_token_ids": [rng.randint(1, vocab - 1) for _ in range(plen)]}
                   for _ in range(num_seqs)]

        outs = llm.generate(prompts=prompts, sampling_params=sp, use_tqdm=False)
        for o in outs:
            total_tok += len(o.outputs[0].token_ids)
        # vLLM may shift the default device; restore (recipe does this).
        torch.xpu.set_device(device)
        torch.xpu.synchronize(device)

        if load_weights == "each":
            _do_load_weights()
        _do_reset_prefix()
        _trainer_step()

        if empty_cache == "each-gen":
            torch.xpu.empty_cache()
        elif empty_cache == "stride" and (it % args.empty_cache_stride == 0):
            torch.xpu.empty_cache()

        iters_done = it + 1
        if it % 2 == 0:
            _log(f"iter={it} burst={int(burst)} tok={total_tok} "
                 f"free={_free_gib(torch, device):.2f} GiB")

    # Clean completion — the ONE terminal line the driver greps for.
    print(
        f"REPRO_DONE node={host} tile={tile} rung={args.rung} "
        f"iters_done={iters_done} crashed=0 free_gib={_free_gib(torch, device):.2f} "
        f"total_tok={total_tok}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

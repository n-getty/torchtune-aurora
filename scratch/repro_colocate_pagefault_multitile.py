#!/usr/bin/env python3
"""MULTI-TILE standalone reproducer for the in-process colocate vLLM generation page fault (XPU).

DEPENDENCY-LIMITED: torch + vllm + oneCCL bindings only. NO torchtune imports. This is the
FAITHFUL Intel/vLLM handoff artifact — the single-tile reproducer
(`repro_colocate_pagefault.py`) proved the fault does NOT reproduce on one tile (R-A/R-B/R-D all
clean); the trigger is the **multi-rank in-process co-residence** that distinguishes the
crashing colocate recipe from clean server-mode: N ranks, each running an in-process vLLM TP=1
engine on its own tile WHILE participating in XCCL collectives (FSDP-style reduce-scatter +
an all-reduce) across all N tiles, in one OS process per tile.

Observed real-recipe signature this mirrors (Qwen3-4B, mg768, 12 ranks): steps 0-2 clean
(~23s/step), then at STEP 3 (deterministic, 5/5 runs) the cross-rank all-reduce explodes
1.5s → ~40s, immediately followed by `CCS NotPresent / PDE banned:1` GPU page fault with
~24 GiB HBM free. Points to L0 IPC-handle / event-pool accumulation under vLLM+trainer
co-tenancy (the in-process analogue of the Ray TP=8 UR40 co-tenancy bug).

Launch with mpiexec under PBS (pmix), one rank per tile:

  mpiexec --pmi=pmix -n 12 -ppn 12 --hostfile $PBS_NODEFILE \
      --cpu-bind depth --depth 8 \
      bash experiments/colocate/_repro_multitile_wrapper.sh \
          scratch/repro_colocate_pagefault_multitile.py \
          --model /tmp/models/Qwen3-4B --max-gen 768 --steps 8

The wrapper sets ZE_AFFINITY_MASK / RANK / LOCAL_RANK per rank (mirrors _rank_wrapper.sh) and
the pmix CCL env. Each rank: builds in-process vLLM (verbatim from
torchtune/dev/rl/vllm_backend.py:_init_vllm_tp1), then re-creates the XCCL world PG, wraps a
synthetic ~4B-shaped model in FSDP across all ranks, and loops generate→fwd→bwd→all_reduce.

Each rank prints `MTREPRO_STEP rank=<r> step=<s> ar_s=<t> gen_s=<t> free_gib=<f>` per step and
`MTREPRO_DONE rank=<r> steps=<n>` on clean completion (the GPU fault aborts the process).
"""

import argparse
import os
import socket
import sys
import time


def _log(rank, msg):
    print(f"[mtrepro r{rank} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--max-gen", type=int, default=768)
    p.add_argument("--max-model-len", type=int, default=1536)
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--gpu-mem", type=float, default=0.35)
    p.add_argument("--grpo", type=int, default=8)
    p.add_argument("--fbs", type=int, default=8, help="policy fwd+bwd micro-batch (activation pressure)")
    p.add_argument("--ref-fbs", type=int, default=16, help="ref no-grad fwd micro-batch")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--burst-every", type=int, default=2)
    p.add_argument("--load-weights", default="each", choices=["off", "once", "each"],
                   help="round-trip weights into the live vLLM engine each step (the "
                        "recipe's per-step colocate publish — the key co-residence mutation)")
    p.add_argument("--load-real-weights", action="store_true",
                   help="load the REAL Qwen3-4B HF-named safetensors into the engine (faithful; "
                        "the self-named round-trip is rejected by vLLM's fused param naming)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt-len", type=int, default=256)
    p.add_argument("--no-vllm", action="store_true",
                   help="control: skip in-process vLLM (XCCL-only) to test whether the cross-rank "
                        "collectives fault WITHOUT the co-resident engine")
    args = p.parse_args()

    rank = int(os.environ.get("RANK", os.environ.get("PMI_RANK", "0")))
    world = int(os.environ.get("WORLD_SIZE", os.environ.get("PMI_SIZE", "1")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("PMI_LOCAL_RANK", "0")))
    host = socket.gethostname()
    _log(rank, f"host={host} world={world} local_rank={local_rank} model={args.model} "
               f"no_vllm={args.no_vllm}")

    # --- env priming (mirror vllm_backend.py for the in-process engine) ---
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(k, None)
    os.environ.setdefault("no_proxy", "*"); os.environ.setdefault("NO_PROXY", "*")
    os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    import torch
    try:
        import intel_extension_for_pytorch as ipex  # noqa: F401
    except Exception:
        pass
    # torch 2.10 on the Aurora frameworks stack registers the 'xccl' backend natively
    # (dist.Backend.default_device_backend_map) — NO oneccl_bindings import needed (and that
    # module isn't present here). The recipe uses init_xpu_process_group -> backend="xccl".

    assert torch.xpu.is_available(), "XPU unavailable"
    # ZE_AFFINITY_MASK (set by wrapper) restricts visibility to one tile → device index 0.
    device = torch.device("xpu:0")
    torch.xpu.set_device(device)
    torch.manual_seed(args.seed)

    # =====================================================================
    # 1. Build in-process vLLM TP=1 (verbatim init from vllm_backend.py).
    #    vLLM needs a world=1 gloo PG during its own init; we save the real
    #    torchrun env, give vLLM a private world, then restore + build the
    #    real XCCL world PG afterward (exactly what the recipe does).
    # =====================================================================
    llm = None
    if not args.no_vllm:
        saved = {}
        for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "GROUP_RANK",
                    "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "ZE_AFFINITY_MASK"):
            saved[key] = os.environ.pop(key, None)
        os.environ["WORLD_SIZE"] = "1"; os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(29599 + rank)
        os.environ["ZE_AFFINITY_MASK"] = saved.get("ZE_AFFINITY_MASK") or str(local_rank)

        import tempfile
        _sf = tempfile.mktemp(prefix=f"vllm_gloo_store_r{rank}_")
        torch.distributed.init_process_group(backend="gloo",
                                             init_method=f"file://{_sf}", world_size=1, rank=0)
        _ong = torch.distributed.new_group
        def _gloo_ng(*a, **k): k["backend"] = "gloo"; return _ong(*a, **k)
        torch.distributed.new_group = _gloo_ng
        _oar = torch.distributed.all_reduce
        def _safe_ar(t, op=torch.distributed.ReduceOp.SUM, group=None, async_op=False):
            if group is not None and group.size() == 1: return None
            if t.is_xpu: return None
            return _oar(t, op=op, group=group, async_op=async_op)
        torch.distributed.all_reduce = _safe_ar
        from vllm import LLM, SamplingParams
        from vllm.v1.executor.uniproc_executor import UniProcExecutor
        _oda = UniProcExecutor._distributed_args
        def _pda(se): m, r, _ = _oda(se); return m, r, 0
        UniProcExecutor._distributed_args = _pda

        bps = (args.max_model_len + 15) // 16
        block_override = int(args.grpo * bps * 1.1)
        t0 = time.perf_counter()
        llm = LLM(model=args.model, tensor_parallel_size=1, gpu_memory_utilization=args.gpu_mem,
                  max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
                  enforce_eager=True, dtype="bfloat16", disable_custom_all_reduce=True,
                  num_gpu_blocks_override=block_override, enable_prefix_caching=False)
        _log(rank, f"vLLM up in {time.perf_counter()-t0:.1f}s")

        UniProcExecutor._distributed_args = _oda
        torch.distributed.new_group = _ong
        torch.distributed.all_reduce = _oar
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        import vllm.distributed.parallel_state as vps
        vps._WORLD = None
        try: os.unlink(_sf)
        except OSError: pass
        for k, v in saved.items():
            if v is not None: os.environ[k] = v
            elif k in os.environ: del os.environ[k]
        from vllm import SamplingParams  # noqa: F811
    else:
        from vllm import SamplingParams  # type: ignore

    # =====================================================================
    # 2. Build the REAL multi-rank XCCL world PG (this is what the recipe's
    #    FSDP + adapter all-reduce run on, co-resident with vLLM above).
    # =====================================================================
    os.environ.setdefault("MASTER_ADDR", os.environ.get("MASTER_ADDR", "127.0.0.1"))
    os.environ["MASTER_PORT"] = os.environ.get("TRAIN_MASTER_PORT", "29400")
    from mpi4py import MPI
    MPI.COMM_WORLD.Barrier()
    torch.distributed.init_process_group(backend="xccl", world_size=world, rank=rank)
    _log(rank, f"XCCL world PG up (world={world})")

    # Synthetic ~4B-shaped model, FSDP-sharded across all ranks (FULL_SHARD, like the recipe).
    # Uses REAL attention (SDPA) blocks, not MLP-only: the MLP-only model stayed clean, and the
    # trainer's SDPA kernels co-resident with vLLM's paged-attention on one tile is the suspected
    # missing co-residence ingredient (both contend for L0 attention scratch/resources).
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
    import torch.nn.functional as F
    H, FF, L, NH = 2560, 9728, 18, 20  # hidden, ffn, layers, heads (Qwen3-4B-ish)
    HD = H // NH

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = torch.nn.Linear(H, 3 * H, bias=False, dtype=torch.bfloat16)
            self.o = torch.nn.Linear(H, H, bias=False, dtype=torch.bfloat16)
            self.w1 = torch.nn.Linear(H, FF, bias=False, dtype=torch.bfloat16)
            self.w2 = torch.nn.Linear(FF, H, bias=False, dtype=torch.bfloat16)

        def forward(self, x):  # x: [B, S, H]
            B, S, _ = x.shape
            qkv = self.qkv(x).view(B, S, 3, NH, HD).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # [B, NH, S, HD]
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # SDPA kernel
            a = a.transpose(1, 2).reshape(B, S, H)
            x = x + self.o(a)
            x = x + self.w2(F.gelu(self.w1(x)))
            return x

    model = torch.nn.Sequential(*[Block() for _ in range(L)]).to(device)
    model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD, device_id=device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-6)
    # A replicated "adapter" tensor that we all_reduce each step (mirrors ADAPTER_AR — the XCCL
    # collective whose step-3 explosion precedes the fault in the real run).
    adapter = torch.randn(64 * 1024 * 1024 // 2, dtype=torch.bfloat16, device=device)  # ~64 MB
    _log(rank, "FSDP model + adapter ready")

    import random
    rng = random.Random(args.seed + rank)
    vocab = 150000

    # Handle to the live vLLM model for the per-step load_weights mutation (the recipe's
    # defining colocate action: rewrite vLLM's resident device weights every step while the
    # engine + KV state is live). Round-trip the engine's OWN current weights so values are
    # unchanged but the exact copy_-into-resident-device-tensor path runs — this is the one
    # structural element the synthetic was missing (co-residence/mem/attn/volume all clean).
    vllm_model = None
    _real_named = None
    if llm is not None and args.load_weights != "off":
        try:
            vllm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
            _self_named = [(n, p.detach().clone()) for n, p in vllm_model.named_parameters()]
            _log(rank, f"load_weights path ready ({len(_self_named)} self params)")
        except Exception as e:
            _log(rank, f"WARN cannot reach vllm model for load_weights: {e!r}")
        if args.load_real_weights:
            # Faithful path: load the REAL Qwen3-4B HF-named safetensors back into the live
            # engine each step (vLLM's load_weights expects HF checkpoint names like
            # ...self_attn.q_proj.weight, fused internally). This exactly mirrors the recipe's
            # per-step merged-weight publish — the one structural element the self-named
            # round-trip could not exercise (vLLM rejects the module's own fused param names).
            try:
                import glob as _glob  # NB: model dir is node-local /tmp (not DAOS) — glob safe
                from safetensors import safe_open
                shards = sorted(_glob.glob(os.path.join(args.model, "*.safetensors")))
                _real_named = []
                for sh in shards:
                    with safe_open(sh, framework="pt", device="cpu") as f:
                        for k in f.keys():
                            _real_named.append((k, f.get_tensor(k).to(device, torch.bfloat16)))
                _log(rank, f"real-weights path ready ({len(_real_named)} HF params from "
                           f"{len(shards)} shards)")
            except Exception as e:
                _log(rank, f"WARN cannot load real safetensors: {e!r}")

    for step in range(args.steps):
        # --- generate (in-process vLLM, co-resident) ---
        gen_s = 0.0
        if llm is not None:
            burst = args.burst_every > 0 and step % args.burst_every == 0 and step > 0
            plen = (args.max_model_len - args.max_gen) if burst else args.prompt_len
            sp = SamplingParams(max_tokens=args.max_gen, temperature=1.0, top_k=-1,
                                detokenize=False, ignore_eos=bool(burst))
            prompts = [{"prompt_token_ids": [rng.randint(1, vocab - 1) for _ in range(plen)]}
                       for _ in range(args.grpo)]
            g0 = time.perf_counter()
            llm.generate(prompts=prompts, sampling_params=sp, use_tqdm=False)
            torch.xpu.set_device(device); torch.xpu.synchronize(device)
            gen_s = time.perf_counter() - g0

        # --- load_weights into the LIVE vLLM engine (the recipe's per-step colocate publish) ---
        if vllm_model is not None and (args.load_weights == "each"
                                       or (args.load_weights == "once" and step == 0)):
            _names = _real_named if _real_named is not None else _self_named
            try:
                vllm_model.load_weights(_names)
                llm.llm_engine.reset_prefix_cache()
                torch.xpu.synchronize(device)
            except Exception as e:
                _log(rank, f"WARN load_weights step {step} failed: {e!r}")

        # --- trainer fwd+bwd over the GENERATED-SEQUENCE batch (the missing pressure) ---
        # The real recipe runs ref_fwd + policy fwd + grpo_step backward over
        # [num_seqs, prompt+completion, hidden] activations (GiBs), co-resident with vLLM KV.
        # The trivial [2,H] step did NOT reproduce; this sizes activations to the real rollout.
        seqlen = args.max_model_len if (args.burst_every > 0 and step % args.burst_every == 0
                                        and step > 0) else (args.prompt_len + args.max_gen)
        seqlen = min(seqlen, args.max_model_len)
        bsz = args.grpo
        # ref forward (no-grad, like disable_adapter ref_fwd) — chunked over the batch.
        with torch.no_grad():
            for c in range(0, bsz, max(1, args.ref_fbs)):
                xb = torch.randn(min(args.ref_fbs, bsz - c), seqlen, H,
                                 dtype=torch.bfloat16, device=device)
                _ = model(xb)
        torch.xpu.synchronize(device)
        # policy fwd+bwd (grad) — the activation-heavy training step.
        opt.zero_grad(set_to_none=True)
        for c in range(0, bsz, max(1, args.fbs)):
            xb = torch.randn(min(args.fbs, bsz - c), seqlen, H,
                             dtype=torch.bfloat16, device=device)
            loss = model(xb).float().pow(2).mean()
            loss.backward()
        opt.step()
        torch.xpu.synchronize(device)

        # --- adapter all-reduce (the ADAPTER_AR XCCL collective) ---
        a0 = time.perf_counter()
        torch.distributed.all_reduce(adapter, op=torch.distributed.ReduceOp.SUM)
        torch.xpu.synchronize(device)
        ar_s = time.perf_counter() - a0

        free = torch.xpu.mem_get_info(device)[0] / 1024**3
        _log(rank, f"MTREPRO_STEP rank={rank} step={step} ar_s={ar_s:.2f} gen_s={gen_s:.1f} "
                   f"free_gib={free:.2f}")

    MPI.COMM_WORLD.Barrier()
    print(f"MTREPRO_DONE rank={rank} steps={args.steps}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

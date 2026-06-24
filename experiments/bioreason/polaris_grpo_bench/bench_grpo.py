#!/usr/bin/env python
"""Apples-to-apples GRPO throughput benchmark: Aurora (XPU, torchtune) vs Polaris (CUDA, TRL).

Text-proxy GRPO step-throughput on Qwen3-4B using HF TRL GRPOTrainer. Companion to
sft_bench/bench_sft.py (same harness conventions: config fingerprint, deterministic
synthetic data, warmup/measured split, results.json).

WHY a proxy: the upstream bowang-lab/BioReason-Pro repo never released its RL/GRPO training
driver (SFT-only + a TRL plugin's reward/embed hooks). So a literally-faithful multimodal RL
baseline is impossible here. This measures the *external framework's* GRPO loop throughput
(generation + policy fwd/bwd + KL) at our prod envelope (G=8, max_gen=1024) so we have a CUDA
target to compare our torchtune-XPU step times against. It does NOT carry protein/GO embeds —
prompt *length* is matched, modality is not.

Design choices kept identical to bench_sft.py for parity:
  * Deterministic synthetic prompts (seeded) -> constant token volume per step, no download.
  * Fixed num_generations / max_completion_length / batch -> known work per optimizer step.
  * Warmup steps discarded; median + p10/p90 of per-step wall time reported.
  * Headline metric for gen-dominated GRPO: generated-tokens/sec (node + per-device).
"""
import argparse, json, os, platform, sys, time

import torch
import torch.distributed as dist


# ----------------------------------------------------------------------------- device
def device_info():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        peak = 312e12 if "A100" in name else None  # A100 bf16 dense peak
        return "cuda", name, peak
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        try:
            name = torch.xpu.get_device_name(0)
        except Exception:
            name = "Intel XPU"
        peak = 419.5e12  # PVC Max 1550 per-tile bf16
        return "xpu", name, peak
    return "cpu", platform.processor() or "cpu", None


DEV_TYPE, DEV_NAME, DEV_PEAK_FLOPS = device_info()


def synchronize():
    if DEV_TYPE == "cuda":
        torch.cuda.synchronize()
    elif DEV_TYPE == "xpu":
        torch.xpu.synchronize()


# ----------------------------------------------------------------------------- dist env
def dist_env():
    return dict(
        rank=int(os.environ.get("RANK", "0")),
        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
    )


def is_main():
    return dist_env()["rank"] == 0


def log0(*a):
    if is_main():
        print(*a, flush=True)


# ----------------------------------------------------------------------------- args
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--gen", choices=["hf", "vllm"], default="hf",
                   help="rollout backend: HF .generate() or vLLM colocate")
    p.add_argument("--attn", default="sdpa",
                   choices=["sdpa", "flash_attention_2", "eager"])
    p.add_argument("--prompt-len", type=int, default=2048,
                   help="approx prompt token length (proxy for the faithful 2048-protein "
                        "+ 200-GO + text context)")
    p.add_argument("--num-generations", type=int, default=8, help="G (group size)")
    p.add_argument("--max-completion-length", type=int, default=1024)
    p.add_argument("--micro-bsz", type=int, default=2,
                   help="per_device_train_batch_size (completions/device/step)")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--steps", type=int, default=20, help="measured optimizer steps")
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--n-examples", type=int, default=512)
    p.add_argument("--grad-checkpoint", type=int, default=1)
    p.add_argument("--vllm-gpu-mem", type=float, default=0.3,
                   help="vllm_gpu_memory_utilization for colocate mode")
    p.add_argument("--beta", type=float, default=0.04, help="KL coefficient")
    p.add_argument("--out", required=True)
    p.add_argument("--tag", default="")
    return p.parse_args()


# ----------------------------------------------------------------------------- dataset
def build_dataset(n_examples, prompt_len, model_path, seed=1234):
    """Deterministic synthetic dataset with a 'prompt' text column of ~prompt_len tokens.

    Built by sampling random token ids (seeded) and decoding to text. Re-tokenization is
    near-identical length (recorded in results as observed mean). Identical given the same
    (n, prompt_len, model, seed).
    """
    from datasets import Dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    vocab_size = tok.vocab_size if tok.vocab_size else 150000
    lo, hi = 10, min(vocab_size - 10, 150000)

    g = torch.Generator().manual_seed(seed)
    big = torch.randint(lo, hi, (n_examples, prompt_len), generator=g, dtype=torch.long)
    prompts = []
    for i in range(n_examples):
        text = tok.decode(big[i].tolist(), skip_special_tokens=True)
        prompts.append(text)
    return Dataset.from_dict({"prompt": prompts}), tok


# ----------------------------------------------------------------------------- reward
def make_reward_fn(tok, holder):
    """Reward = normalized completion token length. Gives non-zero group variance (so
    advantages/loss/backward are exercised, not degenerate) and lets us accumulate the
    generated-token volume needed for the throughput metric.
    """
    def reward_len(prompts, completions, **kw):
        rewards = []
        for c in completions:
            text = c if isinstance(c, str) else (c[-1]["content"] if c else "")
            n = len(tok(text, add_special_tokens=False)["input_ids"])
            holder["gen_tokens"] += n
            holder["gen_count"] += 1
            # reward in [0,1]; longer-but-not-maxed favoured, with mild variance
            rewards.append(min(n, 1024) / 1024.0)
        return rewards
    return reward_len


# ----------------------------------------------------------------------------- throughput cb
def make_callback(warmup, measured, holder):
    from transformers import TrainerCallback

    class _CB(TrainerCallback):
        def __init__(self):
            self.times = []
            self._t = None
            self.count = 0
            # snapshot gen-token counters at the warmup boundary so the throughput metric
            # covers exactly the measured window
            self.gen_tokens_at_measure_start = None

        def on_step_end(self, args, state, control, **kw):
            now = time.perf_counter()
            if self._t is not None:
                self.count += 1
                if self.count == warmup:
                    self.gen_tokens_at_measure_start = holder["gen_tokens"]
                if self.count > warmup:
                    self.times.append(now - self._t)
                if len(self.times) >= measured:
                    holder["gen_tokens_measured"] = (
                        holder["gen_tokens"] - (self.gen_tokens_at_measure_start or 0))
                    control.should_training_stop = True
            self._t = time.perf_counter()
            return control

    cb = _CB()
    holder["cb"] = cb
    return cb


# ----------------------------------------------------------------------------- main
def main():
    args = get_args()
    de = dist_env()
    ws = de["world_size"]

    if DEV_TYPE == "cuda":
        torch.cuda.set_device(de["local_rank"])
    elif DEV_TYPE == "xpu":
        torch.xpu.set_device(de["local_rank"])

    visible_devices = (torch.cuda.device_count() if DEV_TYPE == "cuda"
                       else torch.xpu.device_count() if DEV_TYPE == "xpu" else 0)

    from transformers import AutoModelForCausalLM, AutoConfig
    from trl import GRPOConfig, GRPOTrainer

    cfg = AutoConfig.from_pretrained(args.model_path)

    ds, tok = build_dataset(args.n_examples, args.prompt_len, args.model_path)
    # record an observed prompt length sample (proxy fidelity check)
    obs_prompt_len = len(tok(ds[0]["prompt"], add_special_tokens=False)["input_ids"])

    log0(f"[bench] device={DEV_TYPE} name={DEV_NAME} world_size={ws} "
         f"visible_per_rank={visible_devices} gen={args.gen} G={args.num_generations} "
         f"max_gen={args.max_completion_length} micro_bsz={args.micro_bsz} "
         f"prompt_len~{obs_prompt_len}")

    # ---- model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, attn_implementation=args.attn,
    )
    n_params = sum(p.numel() for p in model.parameters())

    # LoRA — the faithful BioReason-Pro RL surface (frozen backbone + r16/a32 adapters)
    import peft.import_utils as _piu
    try:
        _piu.is_torchao_available.cache_clear()
    except Exception:
        pass
    _piu.is_torchao_available = lambda: False
    try:
        import peft.tuners.lora.torchao as _plt
        _plt.is_torchao_available = lambda: False
    except Exception:
        pass
    from peft import LoraConfig
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    holder = {"gen_tokens": 0, "gen_count": 0, "gen_tokens_measured": 0}
    reward_fn = make_reward_fn(tok, holder)
    total_steps = args.warmup_steps + args.steps + 2

    grpo_kwargs = dict(
        output_dir=f"/tmp/grpo_bench_out_{de['rank']}",
        per_device_train_batch_size=args.micro_bsz,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        num_iterations=1,             # one optimizer pass per generation (on-policy)
        steps_per_generation=1,       # fresh rollouts every step -> step time includes gen
        beta=args.beta,               # KL penalty -> reference fwd exercised
        max_steps=total_steps,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        bf16=True,
        gradient_checkpointing=bool(args.grad_checkpoint),
        dataloader_num_workers=0,
        learning_rate=1e-6,
    )
    if args.gen == "vllm":
        grpo_kwargs.update(
            use_vllm=True,
            vllm_mode="colocate",
            vllm_gpu_memory_utilization=args.vllm_gpu_mem,
            vllm_max_model_length=args.prompt_len + args.max_completion_length + 128,
        )

    grpo_config = GRPOConfig(**grpo_kwargs)
    cb = make_callback(args.warmup_steps, args.steps, holder)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=ds,
        peft_config=peft_config,
        callbacks=[cb],
    )

    synchronize()
    t_start = time.perf_counter()
    trainer.train()
    synchronize()
    t_end = time.perf_counter()

    times = cb.times
    if not times:
        log0("[bench] ERROR: no measured steps recorded")
        sys.exit(2)

    import statistics as st
    times_sorted = sorted(times)

    def pct(p):
        k = max(0, min(len(times_sorted) - 1, int(round(p * (len(times_sorted) - 1)))))
        return times_sorted[k]

    med = st.median(times)
    p10, p90 = pct(0.10), pct(0.90)
    measured_wall = sum(times)

    # generated tokens during the measured window (summed across ALL ranks for node metric)
    gen_tok_measured_local = holder.get("gen_tokens_measured", 0)
    gen_tok_measured_node = gen_tok_measured_local
    if dist.is_initialized():
        t = torch.tensor([gen_tok_measured_local], dtype=torch.long, device=dev_for_reduce())
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        gen_tok_measured_node = int(t.item())

    gen_tok_per_sec_node = gen_tok_measured_node / measured_wall if measured_wall else None
    gen_tok_per_sec_dev = gen_tok_per_sec_node / ws if gen_tok_per_sec_node else None
    completions_per_step = args.micro_bsz * ws * args.grad_accum
    mean_completion_len = (gen_tok_measured_node / (completions_per_step * len(times))
                           if completions_per_step and times else None)

    result = dict(
        tag=args.tag,
        platform=DEV_TYPE,
        device_name=DEV_NAME,
        device_peak_bf16_flops=DEV_PEAK_FLOPS,
        framework="trl-grpo",
        gen_backend=args.gen,
        world_size=ws,
        attn=args.attn,
        prompt_len_requested=args.prompt_len,
        prompt_len_observed=obs_prompt_len,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        micro_bsz=args.micro_bsz,
        grad_accum=args.grad_accum,
        grad_checkpoint=bool(args.grad_checkpoint),
        beta=args.beta,
        n_params=n_params,
        completions_per_step=completions_per_step,
        measured_steps=len(times),
        warmup_steps=args.warmup_steps,
        step_time_median_s=med,
        step_time_p10_s=p10,
        step_time_p90_s=p90,
        step_time_cov=(p90 - p10) / med if med else None,
        mean_completion_len=mean_completion_len,
        gen_tokens_measured_node=gen_tok_measured_node,
        gen_tok_per_sec_node=gen_tok_per_sec_node,
        gen_tok_per_sec_device=gen_tok_per_sec_dev,
        fingerprint=dict(
            model=os.path.basename(args.model_path.rstrip("/")),
            n_params=n_params, prompt_len=args.prompt_len,
            num_generations=args.num_generations,
            max_completion_length=args.max_completion_length,
            micro_bsz=args.micro_bsz, grad_accum=args.grad_accum,
            gen_backend=args.gen, dataset="synthetic-seed1234",
        ),
        versions=dict(
            python=platform.python_version(),
            torch=torch.__version__,
            transformers=__import__("transformers").__version__,
            trl=__import__("trl").__version__,
            datasets=__import__("datasets").__version__,
            accelerate=__import__("accelerate").__version__,
            peft=__import__("peft").__version__,
            vllm=_safe_ver("vllm"),
        ),
        node=platform.node(),
        visible_devices_per_rank=visible_devices,
        peak_mem_gb=(torch.cuda.max_memory_allocated() / 1e9 if DEV_TYPE == "cuda"
                     else torch.xpu.max_memory_allocated() / 1e9 if DEV_TYPE == "xpu"
                     else None),
    )

    if is_main():
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        log0("[bench] RESULT " + json.dumps({
            k: result[k] for k in
            ("platform", "gen_backend", "world_size", "step_time_median_s",
             "gen_tok_per_sec_node", "gen_tok_per_sec_device", "mean_completion_len",
             "peak_mem_gb")}, indent=2))
        log0(f"[bench] wrote {args.out}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def dev_for_reduce():
    de = dist_env()
    if DEV_TYPE == "cuda":
        return torch.device(f"cuda:{de['local_rank']}")
    if DEV_TYPE == "xpu":
        return torch.device(f"xpu:{de['local_rank']}")
    return torch.device("cpu")


def _safe_ver(mod):
    try:
        return __import__(mod).__version__
    except Exception:
        return None


if __name__ == "__main__":
    main()

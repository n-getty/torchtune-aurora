"""GPU stress test: real SFT training on Qwen/Qwen3.6-27B (text-only path,
new hybrid Gated-DeltaNet/linear-attention + full-attention architecture,
`Qwen3_5ForConditionalGeneration`) restricted to GPUs 2 and 3 specifically.

Purpose: this is a fault-reproduction probe, not a quality-training run. The
node's intermittent "busy or unavailable" GPU fault has historically
recurred under real training workloads (not just synthetic matmuls), and
GPUs 2/3 are two of the historically-flagged GPUs (see
memory/feedback_h200_gpu_topology_pix_vs_sys.md). Using a genuinely new
model architecture (not our existing Qwen3-32B dense recipe) gives an
independent signal from the BioReason-specific SFT scripts already run on
this node many times.

Recipe: plain-Tensor DDP (no FSDP2 -- 27B dense in bf16 is ~54GB per full
replica, fits on a single H200's 143GB with room for activations) with LoRA
adapters (rank 16) on the text backbone's linear projections, following the
same frozen-base + trainable-adapter pattern already validated at scale in
this project (BioReason Stage-2 LoRA). Only 2 GPUs are used per the explicit
instruction to restrict to GPUs 2,3 -- pass CUDA_VISIBLE_DEVICES=2,3 to the
launcher, ranks are then 0,1 mapping to physical 2,3.

Dataset: GSM8K (already cached locally, openai/gsm8k), plain question/answer
text SFT -- no BioReason-specific data plumbing needed for a stress test.

Launch (GPUs 2,3 only):
  CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 \
      sft_qwen3_6_27b_gpu_stress.py --steps 100 --batch_size 1 --grad_accum 4
"""
import argparse
import json
import os
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from peft import LoraConfig, get_peft_model

MODEL_DIR = "/raid/ngetty/hf_cache/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
IGNORE_IDX = -100


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default=MODEL_DIR)
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=1, help="per-GPU micro batch size")
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--warmup_steps", type=int, default=3)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--out_json", default=None)
    return p.parse_args()


class GSM8KTextDataset(Dataset):
    """Minimal question->answer SFT formatting, loss only on the answer span."""

    def __init__(self, tokenizer, max_seq_len):
        from datasets import load_dataset
        self.ds = load_dataset(
            "openai/gsm8k", "main", split="train",
            cache_dir="/raid/ngetty/hf_cache/datasets",
        )
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        prompt = f"Question: {ex['question']}\nAnswer:"
        target = f" {ex['answer']}{self.tokenizer.eos_token}"

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"]

        input_ids = (prompt_ids + target_ids)[: self.max_seq_len]
        labels = ([IGNORE_IDX] * len(prompt_ids) + target_ids)[: self.max_seq_len]
        return {"input_ids": input_ids, "labels": labels}


def collate_fn(batch, pad_id):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), IGNORE_IDX, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    real_lens = torch.zeros(len(batch), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        input_ids[i, :n] = torch.tensor(b["input_ids"], dtype=torch.long)
        labels[i, :n] = torch.tensor(b["labels"], dtype=torch.long)
        attention_mask[i, :n] = 1
        real_lens[i] = n
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask, "real_lens": real_lens}


def main():
    args = parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    device = torch.device("cuda", local_rank)
    dtype = torch.bfloat16
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")

    def log(msg):
        if rank == 0:
            print(f"[qwen3.6_27b_stress] {msg}", flush=True)

    log(f"world_size={world_size} rank={rank} local_rank={local_rank} CUDA_VISIBLE_DEVICES={visible}")
    log(f"config: {vars(args)}")

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_dir, dtype=dtype, attn_implementation="sdpa",
    ).to(device)

    # Qwen3.6's text backbone mixes two block types (see config.text_config
    # .layer_types): full_attention blocks expose self_attn.{q,k,v,o}_proj;
    # linear_attention blocks are Qwen3_5GatedDeltaNet with a different
    # projection set (in_proj_qkv/in_proj_a/in_proj_b/in_proj_z/out_proj),
    # confirmed directly from a meta-device instantiation. Cover both plus
    # the shared mlp.{gate,up,down}_proj so LoRA actually touches every
    # layer, not just the 1-in-4 full_attention ones.
    lora_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z", "out_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    dist.barrier()
    log(f"Qwen3.6-27B (LoRA r={args.lora_rank}) constructed in {time.perf_counter() - t0:.1f}s")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    log(f"trainable params: {n_trainable / 1e6:.2f}M / {n_total / 1e9:.2f}B total")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    dataset = GSM8KTextDataset(tokenizer, args.max_seq_len)
    log(f"dataset size: {len(dataset)}")

    generator = torch.Generator().manual_seed(42 + rank)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        num_workers=0, drop_last=True, generator=generator,
    )

    def infinite(dl):
        while True:
            for b in dl:
                yield b

    data_iter = infinite(loader)

    step_times, token_counts, loss_history = [], [], []

    for step in range(args.steps):
        torch.cuda.synchronize()
        t_step0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)

        total_tokens_this_step = 0
        total_loss_val = 0.0

        for micro in range(args.grad_accum):
            batch = next(data_iter)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.autocast(device_type="cuda", dtype=dtype):
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss

            (loss / args.grad_accum).backward()
            total_loss_val += loss.item()
            total_tokens_this_step += int(batch["real_lens"].sum().item())

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        step_is_finite = bool(grad_norm == grad_norm)
        if step_is_finite:
            optimizer.step()
        else:
            log(f"step={step}: skipping optimizer step (non-finite grad norm)")

        torch.cuda.synchronize()
        step_time = time.perf_counter() - t_step0

        tok_tensor = torch.tensor([total_tokens_this_step], device=device, dtype=torch.long)
        dist.all_reduce(tok_tensor, op=dist.ReduceOp.SUM)
        global_tokens = int(tok_tensor.item())

        peak_mem = torch.cuda.max_memory_allocated(device) / (1024**3)

        if step >= args.warmup_steps:
            step_times.append(step_time)
            token_counts.append(global_tokens)
        loss_history.append(total_loss_val / args.grad_accum)

        log(
            f"step={step} loss={total_loss_val / args.grad_accum:.4f} "
            f"grad_norm={float(grad_norm):.4f} step_time={step_time:.3f}s "
            f"global_tokens={global_tokens} tok/s={global_tokens / step_time:.1f} "
            f"peak_mem_gib={peak_mem:.2f}"
        )

    dist.barrier()
    if rank == 0 and step_times:
        import statistics
        summary = {
            "steps_completed": args.steps,
            "steady_state_steps": len(step_times),
            "mean_step_time_s": statistics.mean(step_times),
            "stdev_step_time_s": statistics.stdev(step_times) if len(step_times) > 1 else 0.0,
            "mean_tokens_per_sec": statistics.mean(t / s for t, s in zip(token_counts, step_times)),
            "world_size": world_size,
            "batch_size_per_gpu": args.batch_size,
            "grad_accum": args.grad_accum,
            "final_loss": loss_history[-1],
            "initial_loss": loss_history[0],
        }
        log(f"SUMMARY: {json.dumps(summary, indent=2)}")
        if args.out_json:
            with open(args.out_json, "w") as f:
                json.dump(summary, f, indent=2)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

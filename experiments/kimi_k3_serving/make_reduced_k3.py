#!/usr/bin/env python3
"""Generate a tiny but structurally faithful Kimi-K3 for one-node decode debugging.

WHY
---
K3 serves a correct FIRST token (prefill) then degenerates from token 2
(decode), while a Kimi-Linear-48B surrogate on the same shared XPU decode path
stays clean. The 48B structurally CANNOT cover the K3-only suspects: it has no
routed_expert_hidden_size (no latent MoE), hidden_act silu (no SiTU), no
attn_res_block_size, and no MXFP4. Debugging on the real K3 costs a 3-node,
~1.5 TiB capacity load per hypothesis.

This writes a ~0.7 GiB K3 that keeps every K3-only structure but drops the
scale: 4 layers instead of 93, 32 experts instead of 896, hidden 1024 instead
of 7168. It serves on ONE node at TP=8 in the free debug queue.

WHAT IS PRESERVED (the whole point -- do not "simplify" these away)
------------------------------------------------------------------
  * latent MoE          routed_expert_hidden_size < hidden_size, + norm
  * SiTU activation     hidden_act=situ, with K3's real beta/linear_beta
  * attention residual  attn_res_block_size, exercised repeatedly (set to 2)
  * MXFP4 experts       real packed/scale dtypes and shapes
  * BOTH attention types  layers 0-2 KDA, layer 3 MLA
  * dense first layer   first_k_dense_replace=1
  * TopK=16 routing     with sigmoid scoring + e_score_correction_bias

TRAPS HONORED (each of these was verified against the real checkpoint)
----------------------------------------------------------------------
  * kda_layers/full_attn_layers are 1-INDEXED; the checkpoint is 0-indexed.
    is_kda_layer applies (layer_idx + 1). So kda_layers [1,2,3] +
    full_attn_layers [4] means checkpoint layers 0,1,2 = KDA and 3 = MLA.
  * A_log is stored (head_dim,) = (128,) tail-padded, NOT (num_heads,).
  * experts use w1/w2/w3 (not a fused w13), each with weight_packed +
    weight_scale.
  * MXFP4 is WEIGHT-ONLY (W4A16): packed is uint8 (any byte is a valid FP4
    pair) and scale is E8M0 uint8 where 127 == 2^0.
  * quantization_config.ignore is copied verbatim from the real config, so the
    attention/shared-expert/dense-MLP/lm_head paths stay unquantized.
  * the loader is strict (default_loader.py:427 fails on any uninitialized
    param), so every tensor the model builds must be emitted here.

USAGE
-----
    python3 make_reduced_k3.py --out /tmp/reduced-k3
    # then de-risk on a login node before spending an allocation:
    python3 make_reduced_k3.py --out /tmp/reduced-k3 --verify

Serving (single node, TP=8 skips the TP==32 guard in serve_k3.sh:196):
    vllm serve /tmp/reduced-k3 --tp 8 --trust-remote-code --enforce-eager
"""

import argparse
import json
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Reduced geometry. Divisibility is what constrains these, not taste:
# num_attention_heads == kda num_heads == 8 so TP=8 gives one head per rank,
# and moe_intermediate_size 256 stays divisible by the MXFP4 group size (32)
# after a /8 shard.
# ---------------------------------------------------------------------------
HIDDEN = 1024
NUM_LAYERS = 4
NUM_EXPERTS = 32
TOPK = 16
NUM_HEADS = 8
KDA_HEAD_DIM = 128          # keep real head_dim: A_log padding depends on it
MOE_INTERMEDIATE = 256
DENSE_INTERMEDIATE = 512
SHARED_INTERMEDIATE = 512
ROUTED_HIDDEN = 512         # latent MoE: must be < HIDDEN
# Must match the tokenizer we copy in (--tokenizer-src). The real K3 ships a
# custom tiktoken tokenizer (tokenization_kimi.py) whose class is unknown to
# AutoTokenizer without trust_remote_code, and whose 163840-token vocab would
# dominate this model's size. Borrowing a small standard tokenizer keeps the
# server path realistic -- it still tokenizes, schedules and detokenizes real
# text -- without pulling in K3's tokenizer machinery, which is not under test.
VOCAB = 151936  # Qwen3-0.6B
# KV_LORA_RANK / QK_ROPE_HEAD_DIM are CONSTRAINED, not free choices. vLLM must
# unify the KDA (MambaSpec) and MLA page sizes, and can only do so by scaling
# block_size -- so the MLA page must divide the mamba page exactly:
#
#   mamba_page = 3*(heads*head_dim*(conv_k-1))*2  +  heads*head_dim^2*4
#              = 542720 B at heads=8, head_dim=128, conv_k=4  (HW-verified)
#   mla_page   = block_size * (kv_lora_rank + qk_rope_head_dim) * 2
#
# and block_size is additionally rounded up to a multiple of the 64-token XPU
# kernel alignment. kv_lora=128/qk_rope=32 needs block_size 1696, which is not
# a multiple of 64; it got rounded to 1728, overshooting to 552960 and failing
# with "page size of the layer is not divisible by the maximum page size"
# (552960 % 542720 = 10240). kv_lora=64/qk_rope=16 gives exactly 3392 = 53*64.
#
# Changing heads/head_dim/conv_k or these two REQUIRES re-solving; only two
# geometries in a wide search satisfy every constraint at once.
#
# ⚠ SECOND CONSTRAINT, discovered after the first was solved: MLA only accepts
# head_dim (= kv_lora_rank + qk_rope_head_dim) in {320, 576}
# (mla_attention.py:1234, enforced at :1348). kv_lora=64/qk_rope=16 gives 80 and
# fails at the FIRST DECODE, not at startup: "Head dimension 80 is not supported
# by MLA." So kv_lora_rank cannot be shrunk freely -- the real K3's 512+64=576
# is one of only two legal values.
#
# Combining both constraints (mamba page divisible by 2*head_dim, quotient a
# multiple of 64, head_dim in {320,576}) leaves only TWO geometries in a wide
# search, and both are much larger than a useful reduction:
#   heads=32 head_dim=128 conv_k=4 mla_head_dim=320 -> block_size=3392
#   heads=16 head_dim=256 conv_k=2 mla_head_dim=320 -> block_size=6592
# Values below are the LAST TESTED state (startup OK, decode fails). Resolving
# this needs one of the two geometries above, or a KDA/MLA layer-count split
# that avoids hybrid page unification. See the plan for the decision.
KV_LORA_RANK = 64
Q_LORA_RANK = 192
QK_NOPE_HEAD_DIM = 64
QK_ROPE_HEAD_DIM = 16
V_HEAD_DIM = 64
CONV_KERNEL = 4
ATTN_RES_BLOCK_SIZE = 2     # smaller than real 12, so 4 layers still exercise it
MXFP4_GROUP = 32

# 1-INDEXED, mirroring the real config. Checkpoint layers 0,1,2 = KDA, 3 = MLA.
KDA_LAYERS = [1, 2, 3]
FULL_ATTN_LAYERS = [4]

QUANT_IGNORE = [
    "re:.*self_attn.*",
    "re:.*shared_experts.*",
    "re:.*mlp\\.(gate|up|gate_up|down)_proj.*",
    "re:.*lm_head.*",
    "re:.*vision_tower.*",
    "re:.*mm_projector.*",
]


def bf16(*shape, generator, scale=0.02):
    return (torch.randn(*shape, generator=generator, dtype=torch.float32) * scale).to(
        torch.bfloat16
    )


def fp32(*shape, generator, scale=0.02):
    return torch.randn(*shape, generator=generator, dtype=torch.float32) * scale


def mxfp4_packed(rows, cols, generator):
    """uint8 FP4 pairs -- every byte is a valid pair of nibbles."""
    return torch.randint(
        0, 256, (rows, cols // 2), dtype=torch.uint8, generator=generator
    )


def mxfp4_scale(rows, cols, generator):
    """E8M0 uint8 scales. 127 == 2^0; [120,135) keeps them near 1.0 so the
    reduced model produces sane magnitudes instead of exponent blowups."""
    return torch.randint(
        120, 135, (rows, cols // MXFP4_GROUP), dtype=torch.uint8, generator=generator
    )


def is_kda(layer_idx):
    """Mirror KimiLinearConfig.is_kda_layer: config lists are 1-indexed."""
    return (layer_idx + 1) in KDA_LAYERS


def build_config():
    text_config = {
        "architectures": ["KimiLinearForCausalLM"],
        "model_type": "kimi_linear",
        "hidden_size": HIDDEN,
        "intermediate_size": DENSE_INTERMEDIATE,
        "moe_intermediate_size": MOE_INTERMEDIATE,
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": NUM_HEADS,
        "num_key_value_heads": NUM_HEADS,
        "num_experts": NUM_EXPERTS,
        "num_experts_per_token": TOPK,
        "num_expert_group": 1,
        "topk_group": 1,
        "moe_renormalize": True,
        "moe_router_activation_func": "sigmoid",
        "routed_scaling_factor": 1.0,
        "use_grouped_topk": False,
        "first_k_dense_replace": 1,
        "n_shared_experts": 1,
        "shared_expert_intermediate_size": SHARED_INTERMEDIATE,
        # --- K3-only structures under test ---
        "routed_expert_hidden_size": ROUTED_HIDDEN,
        "latent_moe_use_norm": True,
        "attn_res_block_size": ATTN_RES_BLOCK_SIZE,
        "hidden_act": "situ",
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
        # --- MLA ---
        # mla_use_nope=True is REQUIRED, not cosmetic: KimiMLAAttention asserts
        # `self.use_nope is True` (kimi_linear.py:344) and the config default is
        # False, so omitting it aborts construction. The real K3 sets it True.
        # Note qk_rope_head_dim stays NONZERO under NoPE -- verified against the
        # checkpoint: kv_a_proj_with_mqa is [kv_lora_rank + qk_rope_head_dim,
        # hidden] = [512+64, 7168] = [576, 7168], matching the real file.
        "mla_use_nope": True,
        "kv_lora_rank": KV_LORA_RANK,
        "q_lora_rank": Q_LORA_RANK,
        "qk_nope_head_dim": QK_NOPE_HEAD_DIM,
        "qk_rope_head_dim": QK_ROPE_HEAD_DIM,
        "v_head_dim": V_HEAD_DIM,
        # --- KDA ---
        "linear_attn_config": {
            "kda_layers": KDA_LAYERS,
            "full_attn_layers": FULL_ATTN_LAYERS,
            "head_dim": KDA_HEAD_DIM,
            "num_heads": NUM_HEADS,
            "short_conv_kernel_size": CONV_KERNEL,
            "gate_lower_bound": -5.0,
            # Real K3 sets this True, which selects the single g_proj gate
            # (kda.py) over the low-rank g_a_proj/g_b_proj pair. The emitted
            # KDA tensors below assume the full-rank form, so these must agree.
            "use_full_rank_gate": True,
        },
        "vocab_size": VOCAB,
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 4096,
        "rope_theta": 50000.0,
        "torch_dtype": "bfloat16",
        "tie_word_embeddings": False,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 2,
    }
    return {
        "architectures": ["KimiK3ForConditionalGeneration"],
        "model_type": "kimi_k3",
        "dtype": "bfloat16",
        "torch_dtype": "bfloat16",
        "text_config": text_config,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 2,
        "quantization_config": {
            "config_groups": {
                "group_0": {
                    "format": "mxfp4-pack-quantized",
                    "input_activations": None,
                    "output_activations": None,
                    "targets": ["Linear"],
                    "weights": {
                        "actorder": None,
                        "block_structure": None,
                        "dynamic": False,
                        "group_size": MXFP4_GROUP,
                        "num_bits": 4,
                        "observer": "minmax",
                        "observer_kwargs": {},
                        "scale_dtype": "torch.uint8",
                        "strategy": "group",
                        "symmetric": True,
                        "type": "float",
                        "zp_dtype": None,
                    },
                }
            },
            "format": "mxfp4-pack-quantized",
            "global_compression_ratio": None,
            "ignore": QUANT_IGNORE,
            "kv_cache_scheme": None,
            "quant_method": "compressed-tensors",
            "quantization_status": "compressed",
        },
    }


def build_weights(generator):
    weights = {}
    prefix = "language_model.model"

    weights[f"{prefix}.embed_tokens.weight"] = bf16(VOCAB, HIDDEN, generator=generator)
    weights[f"{prefix}.norm.weight"] = bf16(HIDDEN, generator=generator, scale=1.0)
    weights[f"{prefix}.output_attn_res_norm.weight"] = bf16(
        HIDDEN, generator=generator, scale=1.0
    )
    weights[f"{prefix}.output_attn_res_proj.weight"] = bf16(1, HIDDEN, generator=generator)
    weights["language_model.lm_head.weight"] = bf16(VOCAB, HIDDEN, generator=generator)

    kda_inner = NUM_HEADS * KDA_HEAD_DIM

    for layer in range(NUM_LAYERS):
        base = f"{prefix}.layers.{layer}"
        for name in (
            "input_layernorm",
            "post_attention_layernorm",
            "self_attention_res_norm",
            "mlp_res_norm",
        ):
            weights[f"{base}.{name}.weight"] = bf16(
                HIDDEN, generator=generator, scale=1.0
            )
        for name in ("self_attention_res_proj", "mlp_res_proj"):
            weights[f"{base}.{name}.weight"] = bf16(1, HIDDEN, generator=generator)

        if is_kda(layer):
            attn = f"{base}.self_attn"
            # A_log is (head_dim,), tail-padded past num_heads -- mirror it.
            a_log = torch.zeros(KDA_HEAD_DIM, dtype=torch.float32)
            a_log[:NUM_HEADS] = torch.rand(
                NUM_HEADS, generator=generator, dtype=torch.float32
            )
            weights[f"{attn}.A_log"] = a_log
            weights[f"{attn}.dt_bias"] = fp32(kda_inner, generator=generator)
            for proj in ("q_proj", "k_proj", "v_proj", "g_proj"):
                weights[f"{attn}.{proj}.weight"] = bf16(
                    kda_inner, HIDDEN, generator=generator
                )
            weights[f"{attn}.b_proj.weight"] = bf16(NUM_HEADS, HIDDEN, generator=generator)
            weights[f"{attn}.f_a_proj.weight"] = bf16(
                KDA_HEAD_DIM, HIDDEN, generator=generator
            )
            weights[f"{attn}.f_b_proj.weight"] = bf16(
                kda_inner, KDA_HEAD_DIM, generator=generator
            )
            weights[f"{attn}.o_proj.weight"] = bf16(HIDDEN, kda_inner, generator=generator)
            weights[f"{attn}.o_norm.weight"] = fp32(
                KDA_HEAD_DIM, generator=generator, scale=1.0
            )
            for conv in ("q_conv1d", "k_conv1d", "v_conv1d"):
                weights[f"{attn}.{conv}.weight"] = fp32(
                    kda_inner, 1, CONV_KERNEL, generator=generator
                )
        else:
            attn = f"{base}.self_attn"
            qk_head = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
            weights[f"{attn}.q_a_proj.weight"] = bf16(
                Q_LORA_RANK, HIDDEN, generator=generator
            )
            weights[f"{attn}.q_a_layernorm.weight"] = bf16(
                Q_LORA_RANK, generator=generator, scale=1.0
            )
            weights[f"{attn}.q_b_proj.weight"] = bf16(
                NUM_HEADS * qk_head, Q_LORA_RANK, generator=generator
            )
            weights[f"{attn}.kv_a_proj_with_mqa.weight"] = bf16(
                KV_LORA_RANK + QK_ROPE_HEAD_DIM, HIDDEN, generator=generator
            )
            weights[f"{attn}.kv_a_layernorm.weight"] = bf16(
                KV_LORA_RANK, generator=generator, scale=1.0
            )
            weights[f"{attn}.kv_b_proj.weight"] = bf16(
                NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM),
                KV_LORA_RANK,
                generator=generator,
            )
            weights[f"{attn}.o_proj.weight"] = bf16(
                HIDDEN, NUM_HEADS * V_HEAD_DIM, generator=generator
            )

        if layer < 1:  # first_k_dense_replace=1 -> layer 0 is a dense MLP
            for proj, shape in (
                ("gate_proj", (DENSE_INTERMEDIATE, HIDDEN)),
                ("up_proj", (DENSE_INTERMEDIATE, HIDDEN)),
                ("down_proj", (HIDDEN, DENSE_INTERMEDIATE)),
            ):
                weights[f"{base}.mlp.{proj}.weight"] = bf16(*shape, generator=generator)
            continue

        moe = f"{base}.block_sparse_moe"
        weights[f"{moe}.gate.weight"] = bf16(NUM_EXPERTS, HIDDEN, generator=generator)
        weights[f"{moe}.gate.e_score_correction_bias"] = fp32(
            NUM_EXPERTS, generator=generator
        )
        # Latent MoE: hidden -> routed_hidden -> hidden, with a norm between.
        weights[f"{moe}.routed_expert_down_proj.weight"] = bf16(
            ROUTED_HIDDEN, HIDDEN, generator=generator
        )
        weights[f"{moe}.routed_expert_norm.weight"] = bf16(
            ROUTED_HIDDEN, generator=generator, scale=1.0
        )
        weights[f"{moe}.routed_expert_up_proj.weight"] = bf16(
            HIDDEN, ROUTED_HIDDEN, generator=generator
        )
        for proj, shape in (
            ("gate_proj", (SHARED_INTERMEDIATE, HIDDEN)),
            ("up_proj", (SHARED_INTERMEDIATE, HIDDEN)),
            ("down_proj", (HIDDEN, SHARED_INTERMEDIATE)),
        ):
            weights[f"{moe}.shared_experts.{proj}.weight"] = bf16(
                *shape, generator=generator
            )
        # Experts operate in the LATENT space (ROUTED_HIDDEN), not HIDDEN.
        for expert in range(NUM_EXPERTS):
            e = f"{moe}.experts.{expert}"
            for gate_up in ("w1", "w3"):
                weights[f"{e}.{gate_up}.weight_packed"] = mxfp4_packed(
                    MOE_INTERMEDIATE, ROUTED_HIDDEN, generator
                )
                weights[f"{e}.{gate_up}.weight_scale"] = mxfp4_scale(
                    MOE_INTERMEDIATE, ROUTED_HIDDEN, generator
                )
            weights[f"{e}.w2.weight_packed"] = mxfp4_packed(
                ROUTED_HIDDEN, MOE_INTERMEDIATE, generator
            )
            weights[f"{e}.w2.weight_scale"] = mxfp4_scale(
                ROUTED_HIDDEN, MOE_INTERMEDIATE, generator
            )

    return weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tokenizer-src",
        type=Path,
        default=Path("/flare/ModCon/ngetty/models/Qwen3-0.6B"),
        help="checkpoint to copy tokenizer files from; its vocab_size must "
        "equal VOCAB in this script",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="after writing, construct the model on meta device and run "
        "load_weights over the emitted tensors (login-node safe)",
    )
    args = parser.parse_args()

    try:
        from safetensors.torch import save_file
    except ImportError:
        print("FATAL: safetensors not installed", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(args.seed)

    # Copy the tokenizer BEFORE writing weights, and fail loudly on a vocab
    # mismatch: a tokenizer that can emit ids past vocab_size produces an
    # index error deep in the sampler, which is a confusing way to learn this.
    src_config = json.loads((args.tokenizer_src / "config.json").read_text())
    if src_config["vocab_size"] != VOCAB:
        print(
            f"FATAL: tokenizer vocab {src_config['vocab_size']} != VOCAB {VOCAB}. "
            f"Set VOCAB to match {args.tokenizer_src}.",
            file=sys.stderr,
        )
        return 2
    copied = []
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
    ):
        source = args.tokenizer_src / name
        if source.exists():
            (args.out / name).write_bytes(source.read_bytes())
            copied.append(name)

    config = build_config()
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    weights = build_weights(generator)
    total_bytes = sum(t.numel() * t.element_size() for t in weights.values())
    save_file(weights, str(args.out / "model.safetensors"))
    (args.out / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": total_bytes},
                "weight_map": {name: "model.safetensors" for name in weights},
            },
            indent=2,
        )
        + "\n"
    )

    kda_layers = [i for i in range(NUM_LAYERS) if is_kda(i)]
    mla_layers = [i for i in range(NUM_LAYERS) if not is_kda(i)]
    print(f"wrote {args.out}")
    print(f"  tensors      : {len(weights)}")
    print(f"  size         : {total_bytes / 2**30:.3f} GiB")
    print(f"  layers       : {NUM_LAYERS} (KDA {kda_layers}, MLA {mla_layers})")
    print(f"  experts      : {NUM_EXPERTS}, topk {TOPK}, MXFP4 latent {ROUTED_HIDDEN}")
    print(f"  tokenizer    : {args.tokenizer_src.name} ({', '.join(copied)})")
    print(f"\nserve: vllm serve {args.out} --tp 8 --trust-remote-code --enforce-eager")

    if args.verify:
        print("\n--verify not yet implemented; construct+load_weights harness pending")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())

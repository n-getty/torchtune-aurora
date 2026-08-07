#!/usr/bin/env python3
"""Slice REAL Kimi-K3 weights into a small servable model.

WHY THIS EXISTS (and why make_reduced_k3.py is not enough)
----------------------------------------------------------
make_reduced_k3.py builds a structurally faithful K3 from RANDOM weights. That
is sufficient to test names, shapes, and that the server comes up -- but it
cannot answer a numerics question. A random-weight model emits a near-uniform
distribution: measured on the reduced K3, every top-20 logprob sat in
[-9.04, -9.60] against a uniform value of -11.93 over a 151936-token vocab.
With thousands of near-tied tokens, top-k membership and argmax are decided by
tie-breaking, not by the model, so prefill-vs-decode comparison is meaningless
there (it produced a false DIVERGE that had to be retracted).

This script instead takes REAL tensors out of the 1.5 TiB checkpoint. Every
per-layer dimension is then K3's own -- hidden 7168, 96 heads, kv_lora 512,
latent 3584, real MXFP4 expert weights -- so the output distribution is a real
distribution and the probe can return a verdict.

WHAT SHRINKS, AND WHY ONLY THESE
--------------------------------
Only two things can shrink without touching the numerics under test:
  * LAYERS   93 -> 4  (keeps layers 0-3: dense layer 0, KDA 0-2, MLA 3)
  * EXPERTS  896 -> N (routing still runs, just over fewer experts)
Everything else is inherited from the real tensors. Cost at 4 layers:
  8 experts  ->  250 tensors, ~10.1 GiB
  32 experts ->  682 tensors, ~11.3 GiB
Only 5 of the 96 source shards are touched (~54 GiB read).

NOT A CORRECTNESS ORACLE. A 4-layer slice of a 93-layer model is not the model:
its outputs are not "what K3 should say". It is a numerics FIXTURE -- the point
is that prefill and decode must agree WITH EACH OTHER on the same weights,
whatever those weights compute. That self-consistency property is exactly what
the K3 bug appears to violate, and it survives truncation.

TRAPS HONORED (all verified against the real checkpoint)
--------------------------------------------------------
  * kda_layers/full_attn_layers are 1-INDEXED (is_kda_layer applies +1), so a
    4-layer slice needs kda_layers [1,2,3] + full_attn_layers [4] to mean
    checkpoint layers 0,1,2 = KDA and 3 = MLA. Real layer 3 IS the first MLA
    layer, so the slice keeps the real KDA/MLA mix.
  * num_experts must be rewritten in the config to match how many were kept,
    or the router indexes experts that were not emitted.
  * The gate weight and e_score_correction_bias must be SLICED to the kept
    experts, not copied whole.
  * quantization_config (incl. the `ignore` regex list) is copied verbatim.
  * mla_use_nope / use_full_rank_gate carry over from the source config.

USAGE
-----
    python3 slice_real_k3.py --out /tmp/k3-slice --layers 4 --experts 8
"""

import argparse
import json
import re
import sys
from pathlib import Path

SRC_DEFAULT = "/flare/ModCon/ngetty/models/Kimi-K3"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=Path(SRC_DEFAULT))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--experts", type=int, default=8)
    args = parser.parse_args()

    from safetensors import safe_open
    from safetensors.torch import save_file

    src, out = args.src, args.out
    out.mkdir(parents=True, exist_ok=True)

    index = json.loads((src / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]

    def keep(name):
        if name.startswith(("vision_tower.", "mm_projector.")):
            return False
        layer = re.search(r"\.layers\.(\d+)\.", name)
        if layer and int(layer.group(1)) >= args.layers:
            return False
        expert = re.search(r"\.experts\.(\d+)\.", name)
        if expert and int(expert.group(1)) >= args.experts:
            return False
        return True

    wanted = sorted(n for n in weight_map if keep(n))
    by_shard = {}
    for name in wanted:
        by_shard.setdefault(weight_map[name], []).append(name)
    print(f"selecting {len(wanted)} tensors from {len(by_shard)} shards")

    tensors = {}
    for shard, names in sorted(by_shard.items()):
        print(f"  reading {shard} ({len(names)} tensors)", flush=True)
        with safe_open(str(src / shard), framework="pt") as f:
            for name in names:
                tensors[name.removeprefix("language_model.")] = f.get_tensor(name)

    # The router gate is [num_experts, hidden]; slicing experts without slicing
    # the gate leaves the router scoring experts that no longer exist.
    for name in list(tensors):
        if name.endswith("block_sparse_moe.gate.weight"):
            tensors[name] = tensors[name][: args.experts].contiguous()
        elif name.endswith("block_sparse_moe.gate.e_score_correction_bias"):
            tensors[name] = tensors[name][: args.experts].contiguous()

    config = json.loads((src / "config.json").read_text())
    text = config["text_config"]
    text["num_hidden_layers"] = args.layers
    text["num_experts"] = args.experts
    if "num_experts_per_token" in text:
        # topk cannot exceed the number of experts that survive the slice.
        text["num_experts_per_token"] = min(text["num_experts_per_token"], args.experts)
    # 1-INDEXED lists: [1..layers-1] are KDA, [layers] is the MLA layer, which
    # matches the real checkpoint's layer 3 being its first MLA layer.
    text["linear_attn_config"] = dict(text["linear_attn_config"])
    text["linear_attn_config"]["kda_layers"] = list(range(1, args.layers))
    text["linear_attn_config"]["full_attn_layers"] = [args.layers]
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    for name in (
        "tokenizer_config.json",
        "tokenization_kimi.py",
        "tiktoken.model",
        "generation_config.json",
    ):
        source = src / name
        if source.exists():
            (out / name).write_bytes(source.read_bytes())

    total = sum(t.numel() * t.element_size() for t in tensors.values())
    save_file(tensors, str(out / "model.safetensors"))
    (out / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": total},
                "weight_map": {n: "model.safetensors" for n in tensors},
            },
            indent=2,
        )
        + "\n"
    )

    print(f"\nwrote {out}")
    print(f"  tensors : {len(tensors)}")
    print(f"  size    : {total / 2**30:.2f} GiB")
    print(f"  layers  : {args.layers} (KDA 0..{args.layers - 2}, MLA {args.layers - 1})")
    print(f"  experts : {args.experts}, topk {text['num_experts_per_token']}")
    print(
        "\nNOTE: a truncated slice is a numerics FIXTURE, not a correctness "
        "oracle. Its outputs are not 'what K3 should say' -- the testable "
        "property is that prefill and decode agree with EACH OTHER."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""BioReason-Pro CAFA5 F_max eval driver (Aurora/XPU).

Measures GO-term F_max of a BioReason checkpoint (SFT or RL) at FAITHFUL inputs
(200 GO embeddings, 2048-residue proteins) on the held-out CAFA5 split, to compare
against the paper's reported 73.6% F_max (BioReason-Pro/README.md — a whole-pipeline
number on a CAFA-framework temporal holdout, so treat it as the ceiling, not the
RL-stage delta).

WHY THIS EXISTS (not the repo's eval.py): the paper's eval.py is CUDA-hardcoded
(`torch.cuda.*`, `.to("cuda")`) and drives a separate `ProteinLLMModel` + in-process
vLLM. We instead reuse our XPU-working `torchtune.dev.bioreason.model.BioReasonModel`
(same checkpoint format: safetensors + go_encoder.pt + go_embedding.pt(200,2560) +
projections) for greedy generation, emit the SAME per-sample JSON schema eval.py
produces, and feed those to the paper's metric code (evals/cafa_evals.py →
cafaeval.cafa_eval) UNCHANGED. Only the inference half is re-hosted on XPU.

OUTPUT CONTRACT (consumed by BioReason-Pro/evals/cafa_evals.py in its OFFICIAL mode
`--reasoning_mode True --final_answer_only False`, per evals/run_cafa_eval.sh):
  <out>/<protein_id>_<ASPECT>_k00.json with fields:
    {protein_id, go_aspect, generated_response, success, protein_sequence,
     go_bp, go_mf, go_cc, ground_truth, input_prompt}
  - reasoning_mode reads ground truth from the go_bp/go_mf/go_cc LIST columns
    (NOT the `ground_truth` text field) and extracts predictions by regex over the
    ENTIRE generated_response. We emit both so either scorer mode works.
  - ASPECT in {BP, MF, CC} (GO_ASPECT_CODES); one JSON per (protein, present aspect).

Score with the paper's UNMODIFIED scorer (go-basic.obo ships in the repo):
  python BioReason-Pro/evals/cafa_evals.py \
      --input_dir <out> \
      --ontology BioReason-Pro/bioreason2/dataset/go-basic.obo \
      --ia_file BioReason-Pro/data/IA.txt \
      --reasoning_mode True --final_answer_only False --threads 0
  (IA.txt only needed for *weighted* F_max; unweighted F_max runs without it.)

STATUS: generation + JSON-emit + the paper-loader call in load_eval_samples are
final and import-clean. The only thing gated is the DATA itself (wanglab/cafa5);
once `snapshot_download` lands it under --cafa5_cache_dir this runs unchanged.
The synthetic scoring path is exercised end-to-end (no data, no XPU) by
tests/torchtune/dev/rl/test_cafa_fmax_eval_pipeline.py.

Run (single XPU tile; greedy; faithful inputs):
  ZE_AFFINITY_MASK=0 PYTHONNOUSERSITE=1 \
    PYTHONPATH=$BIOREASON_DEPS:$BIOREASON_SRC:$(aurora_pythonpath $TT) \
    python experiments/bioreason/eval_cafa_fmax.py \
      --ckpt_dir /lus/flare/.../models/bioreason-pro-sft \
      --esm3_cache_path /lus/flare/.../datasets/bioreason_rl/esm3_cache.pt \
      --out experiments/bioreason/eval_out/sft \
      --max_protein_len 2048 --num_go_tokens 200 --max_new_tokens 2048
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

# GO aspect long-form -> short code, verbatim from BioReason-Pro/eval.py:36.
GO_ASPECT_CODES = {
    "molecular_function": "MF",
    "cellular_component": "CC",
    "biological_process": "BP",
}
# aspect code -> reasoning-mode GT column (cafa_evals.extract_reasoning_ground_truth).
ASPECT_TO_COLUMN = {"MF": "go_mf", "BP": "go_bp", "CC": "go_cc"}
_GO_RE = re.compile(r"GO:\d{7}")

# Placeholder tokens — kept identical to torchtune.dev.bioreason.dataset so the eval
# prompt is byte-for-byte what training/rollout feed the model. Imported lazily in
# build_input_ids to avoid a hard torch import at module load (the synthetic test
# imports the pure helpers without torch/XPU).
_PROTEIN_PAD = "<|protein_pad|>"
_GO_PAD = "<|go_graph_pad|>"


def aspect_code(go_aspect: str) -> str:
    return GO_ASPECT_CODES.get(go_aspect, go_aspect)


def _as_list(v):
    """Normalize a GO-term column to list[str] (datasets may hand back list, str, None,
    NaN-float, or numpy array). The bioreason_pro_test parquet stores empty aspects as
    NaN (float) and go_ids as a string-repr — handle both without crashing."""
    if v is None:
        return []
    # NaN comes through as a float; also guards any stray scalar float/int.
    if isinstance(v, float):
        return []
    # numpy arrays: empty -> [], else element list (avoid ambiguous truth-value).
    try:
        import numpy as _np
        if isinstance(v, _np.ndarray):
            return [str(x) for x in v.tolist()]
    except Exception:
        pass
    if isinstance(v, str):
        import ast
        v = v.strip()
        if not v:
            return []
        try:
            parsed = ast.literal_eval(v)
            return list(parsed) if isinstance(parsed, (list, tuple)) else [v]
        except (ValueError, SyntaxError):
            return _GO_RE.findall(v)
    return list(v)


def build_prompt_string(sample, tokenizer, enable_thinking: bool = True) -> str:
    """Render the system+user chat turn into a generation prompt string.

    Mirrors BioReason-Pro/eval.py:process_single_sample: take the sample's `prompt`
    chat list, keep only system/user roles (stop at the first assistant turn), and
    apply_chat_template(add_generation_prompt=True). The CAFA5 loader's
    format_cafa5_for_protein_llm folds the system text into the user message and
    inserts {"type":"protein"} / {"type":"go_graph"} content blocks, so the rendered
    string contains exactly one <|protein_pad|> and one <|go_graph_pad|> placeholder
    (expanded to true counts by build_input_ids).
    """
    conversation = sample.get("prompt")
    user_conversation = []
    if isinstance(conversation, list):
        for message in conversation:
            role = message.get("role")
            if role in ("system", "user"):
                user_conversation.append(message)
            elif role == "assistant":
                break
    _hf_tok = getattr(tokenizer, "_tok", None) \
        or getattr(tokenizer, "tokenizer", tokenizer)
    try:
        return _hf_tok.apply_chat_template(
            user_conversation, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        # Older templates don't accept enable_thinking.
        return _hf_tok.apply_chat_template(
            user_conversation, tokenize=False, add_generation_prompt=True,
        )


def build_input_ids(prompt_string, protein_seq, tokenizer, num_go_tokens):
    """Expand placeholders + encode, identical formula to dataset.py.__getitem__.

    protein placeholders = len(truncated seq) + 2 (ESM3 BOS/EOS); GO placeholders =
    num_go_tokens. Returns a 1-D LongTensor. The expansion formula is pinned against
    dataset.py by tests/torchtune/dev/rl/test_cafa_fmax_eval_pipeline.py.
    """
    import torch
    protein_count = len(protein_seq) + 2
    s = prompt_string.replace(_PROTEIN_PAD, _PROTEIN_PAD * protein_count, 1)
    s = s.replace(_GO_PAD, _GO_PAD * num_go_tokens, 1)
    # Match dataset.py exactly: the chat template already emits special tokens as
    # text, so encode WITHOUT auto-adding BOS/special tokens (raw HF .encode would
    # otherwise prepend one and shift every position by 1 vs. training).
    try:
        encoded = tokenizer.encode(s, add_special_tokens=False)
    except TypeError:
        encoded = tokenizer.encode(s)  # wrapped torchtune tok already does =False
    tokens = encoded["input_ids"] if isinstance(encoded, dict) else encoded
    return torch.tensor(tokens, dtype=torch.long)


def _aspect_long(code: str) -> str:
    """Short aspect code (MF/BP/CC) -> long form used as go_aspect in records."""
    inv = {v: k for k, v in GO_ASPECT_CODES.items()}
    return inv.get(code, code)


def load_local_parquet_samples(args):
    """Yield eval samples from a LOCAL parquet using the paper's prompt formatter.

    This is the data-on-hand path: our RL parquet (datasets/bioreason_rl) carries the
    SAME row columns the paper's CAFA5 formatter consumes — protein_id, sequence,
    go_bp/go_mf/go_cc (real GO-term lists, the reasoning-mode ground truth),
    protein_function, organism, ppi_formatted, interpro_formatted. So we render the
    EXACT paper-faithful per-aspect eval prompt (generate_cafa5_examples_split_aspects
    -> format_cafa5_for_protein_llm) over local rows — no gated wanglab/cafa5 needed.

    CAVEAT: this parquet is the RL TRAIN set, so a published-ckpt F_max here is
    optimistic vs a held-out number — fine for the input-fidelity A/B and a model-health
    sanity baseline, NOT for an apples-to-apples claim against the paper's 73.6%
    temporal-holdout. For that, use load_eval_samples (gated wanglab/cafa5 temporal split).
    """
    import pandas as pd
    from bioreason2.dataset.cafa5.processor import generate_cafa5_examples_split_aspects
    from bioreason2.dataset.cafa5.format import format_cafa5_for_protein_llm

    paths = []
    for root, _d, files in os.walk(args.local_parquet):
        for fn in files:
            if fn.endswith(".parquet"):
                paths.append(os.path.join(root, fn))
    if not paths and args.local_parquet.endswith(".parquet"):
        paths = [args.local_parquet]
    paths.sort()
    if not paths:
        raise FileNotFoundError(f"no parquet under {args.local_parquet}")

    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)  # deterministic shuffle
    n = len(df) if args.max_samples <= 0 else min(args.max_samples, len(df))

    # Shard across processes/tiles: each shard takes a strided subset of proteins
    # (stride keeps long/short sequences spread evenly so per-shard runtime is
    # balanced). Each shard writes to the SAME --out dir with unique filenames
    # ({pid}_{ASPECT}_k00.json), so the scorer over --out sees the union.
    idxs = list(range(n))
    if args.num_shards > 1:
        idxs = idxs[args.shard_id::args.num_shards]

    # CRITICAL (2026-06-23 bug fix): the model was TRAINED to refine GO-GPT's predictions
    # (the `go_pred` column), injected into the user prompt as `go_speculations` via the
    # paper's _format_reasoning_prompt -> CAFA5_REASONING_TEMPLATE_WITH_CONTEXT*. The old
    # path here (generate_cafa5_examples_split_aspects) has NO go_pred parameter and built a
    # cold prompt missing GO-GPT entirely -> model scored 0.41, BELOW GO-GPT's own 0.54.
    # We now use the paper's _format_reasoning_prompt with go_gpt_predictions_column='go_pred'
    # to build a per-protein prompt, then split per-aspect for scoring. Set
    # --no-inject_go_pred to reproduce the old (broken) cold-prompt behavior for A/B.
    from bioreason2.dataset.cafa5.load import _format_reasoning_prompt

    out = []
    for i in idxs:
        row = df.iloc[i]
        seq = (row.get("sequence", "") or "")[: args.max_protein_len]
        # which aspects have GT for this protein (one eval example per present aspect)
        present = [a for a in ("MF", "BP", "CC") if _as_list(row.get(ASPECT_TO_COLUMN[a]))]
        if not present:
            continue
        # Build the paper-faithful prompt ONCE per protein (it injects go_pred +
        # interpro/ppi and selects the matching WITH_CONTEXT template). _format_reasoning_
        # prompt reads dict-like rows; pass the row as a plain dict.
        row_d = {k: row[k] for k in row.index}
        if getattr(args, "inject_go_pred", True):
            fr = _format_reasoning_prompt(
                row_d,
                go_gpt_predictions_column="go_pred",
                interpro_in_prompt=args.interpro_in_prompt,
                ppi_in_prompt=args.ppi_in_prompt,
            )
            prompt_dict = fr["prompt"]
            chat = format_cafa5_for_protein_llm({
                **row_d, "prompt": prompt_dict, "sequence": seq,
            })
            base_prompt = chat["prompt"]
        else:
            # legacy cold path (no go_pred) — for the A/B that proves the bug
            exs = generate_cafa5_examples_split_aspects(
                row, interpro_in_prompt=args.interpro_in_prompt,
                ppi_in_prompt=args.ppi_in_prompt, include_go_defs=args.include_go_defs)
            base_prompt = None
        for a in present:
            if base_prompt is not None:
                prompt = base_prompt
            else:
                _e = next((x for x in exs if x.get("go_aspect") == a), None)
                if _e is None:
                    continue
                prompt = format_cafa5_for_protein_llm(
                    {**row_d, "prompt": _e, "go_aspect": a, "sequence": seq})["prompt"]
            out.append({
                "protein_id": row.get("protein_id", f"unknown_{i}"),
                "sequence": seq,
                "go_aspect": _aspect_long(a),
                "prompt": prompt,
                "go_bp": _as_list(row.get("go_bp")),
                "go_mf": _as_list(row.get("go_mf")),
                "go_cc": _as_list(row.get("go_cc")),
                "ground_truth": "",
            })
    return out


def load_eval_samples(args):
    """Yield normalized eval samples from the held-out CAFA5 split.

    Calls the paper's loader DIRECTLY (BioReason-Pro/bioreason2/dataset/cafa5/load.py)
    with eval.py's exact args (split_go_aspects=True, return_as_chat_template=True,
    seed=23, val_split_ratio=0.1) so the (protein, aspect) expansion and the
    train/val split are reproduced bit-for-bit. Requires the gated wanglab/cafa5
    dataset to be present in --cafa5_cache_dir (snapshot_download once access lands);
    until then load_dataset raises a clear network/permission error here.

    Each yielded dict carries everything the scorer + generator need:
      protein_id, sequence, go_aspect (long form), prompt (chat list),
      go_bp/go_mf/go_cc (GT term lists), ground_truth (text, fallback scorer mode).
    """
    from bioreason2.dataset.cafa5.load import load_cafa5_dataset

    _, val_ds, _ = load_cafa5_dataset(
        dataset=args.cafa5_dataset,
        dataset_name=args.cafa5_dataset_name,
        cache_dir=args.cafa5_cache_dir,
        max_length=args.max_protein_len,
        seed=args.seed,
        val_split_ratio=args.val_split_ratio,
        return_as_chat_template=True,
        split_go_aspects=True,
        interpro_dataset_name=args.interpro_dataset_name,
        include_go_defs=args.include_go_defs,
        include_protein_function_summary=args.include_protein_function_summary,
        interpro_in_prompt=args.interpro_in_prompt,
        ppi_in_prompt=args.ppi_in_prompt,
    )
    if not val_ds or len(val_ds) == 0:
        raise ValueError("CAFA5 validation split empty — check dataset/config/cache.")
    val_ds = val_ds.shuffle(seed=args.seed)
    n = len(val_ds) if args.max_samples <= 0 else min(args.max_samples, len(val_ds))

    out = []
    for i in range(n):
        s = val_ds[i]
        out.append({
            "protein_id": s.get("protein_id", f"unknown_{i}"),
            "sequence": s.get("sequence", ""),
            "go_aspect": s.get("go_aspect", "all"),
            "prompt": s.get("prompt"),
            "go_bp": _as_list(s.get("go_bp")),
            "go_mf": _as_list(s.get("go_mf")),
            "go_cc": _as_list(s.get("go_cc")),
            "ground_truth": s.get("ground_truth_go_terms", ""),
        })
    return out


def make_record(sample, response_text):
    """Assemble the eval.py-schema prediction record (works for both scorer modes)."""
    return {
        "protein_id": sample["protein_id"],
        "go_aspect": sample["go_aspect"],
        "generated_response": response_text,
        "success": True,
        "protein_sequence": sample["sequence"],
        # reasoning_mode ground truth (the official scorer path):
        "go_bp": sample.get("go_bp", []),
        "go_mf": sample.get("go_mf", []),
        "go_cc": sample.get("go_cc", []),
        # text ground truth (non-reasoning fallback) + predictions for convenience:
        "ground_truth": sample.get("ground_truth", ""),
        "predicted_go_terms": sorted(set(_GO_RE.findall(response_text))),
    }


def build_model(args):
    """Load the checkpoint via our XPU BioReasonModel.

    Two modes:
      - Full ckpt (default): base/SFT/RL weights, no LoRA (--ckpt_dir only).
      - Trained-adapter eval: --adapter_path points at a saved epoch_<N>/adapter dir
        (adapter_model.safetensors + adapter_config.json from our LoRA-GRPO run) and
        --proj_dir at the epoch_<N> dir holding protein_projection.pt/go_projection.pt.
        The frozen backbone still comes from --ckpt_dir (the SFT base). This is the
        path that measures OUR RL uplift vs the SFT 0.414 baseline.
    """
    import torch
    from torchtune.dev.bioreason.model import BioReasonModel

    # Fail fast on a length-mismatched ESM3 cache: keys are sha1(seq[:max_protein_len]),
    # so a cache built at a different length KeyErrors mid-eval. _load_esm3_cache only
    # checks the model name, so cross-check the sidecar's max_protein_len here.
    if args.esm3_cache_path:
        sidecar = args.esm3_cache_path + ".json"
        if os.path.exists(sidecar):
            cache_mpl = json.load(open(sidecar)).get("max_protein_len")
            if cache_mpl is not None and int(cache_mpl) != int(args.max_protein_len):
                raise ValueError(
                    f"ESM3 cache max_protein_len={cache_mpl} != --max_protein_len="
                    f"{args.max_protein_len}. Use the matching cache or re-encode."
                )

    device = torch.device("xpu") if (hasattr(torch, "xpu") and torch.xpu.is_available()) \
        else torch.device("cpu")
    _adapter = getattr(args, "adapter_path", None)
    model = BioReasonModel(
        ckpt_dir=args.ckpt_dir,
        device=device,
        dtype=torch.bfloat16,
        esm3_cache_path=args.esm3_cache_path,   # faithful 2048 cache (None → live ESM3)
        enable_lora=bool(_adapter),
        adapter_path=_adapter,                  # None → no LoRA (full-ckpt eval)
    )
    # Trained projections live in the saved epoch dir, not the SFT ckpt_dir. When
    # --proj_dir is given, overlay protein_projection.pt / go_projection.pt from there
    # (the RL run trains the projectors alongside the adapter). Without it the model
    # keeps the SFT-base projections loaded from ckpt_dir.
    _proj = getattr(args, "proj_dir", None)
    if _proj:
        import torch as _t
        for _name, _mod in (("protein_projection", model.protein_projection),
                            ("go_projection", model.go_projection)):
            _p = os.path.join(_proj, f"{_name}.pt")
            if os.path.exists(_p):
                _mod.load_state_dict(_t.load(_p, map_location=device), strict=True)
                _mod.to(device=device, dtype=torch.bfloat16)
                print(f"[eval] overlaid trained {_name} from {_p}", flush=True)
            else:
                print(f"[eval] WARNING: --proj_dir set but {_p} missing; "
                      f"keeping SFT-base {_name}", flush=True)
    model.eval()
    return model, device


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inject_go_pred", action=argparse.BooleanOptionalAction, default=True,
                    help="Inject the go_pred (GO-GPT predictions) column into the prompt as "
                         "go_speculations via the paper's _format_reasoning_prompt — the model "
                         "was TRAINED on this. Default True. --no-inject_go_pred reproduces the "
                         "old cold-prompt path (scored 0.41, below GO-GPT's 0.54).")
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--adapter_path", default=None,
                    help="dir with adapter_model.safetensors + adapter_config.json from a "
                         "LoRA-GRPO run. Set => enable_lora=True, backbone=W_base+adapter. "
                         "This is the OUR-RL-uplift eval path (vs SFT 0.414).")
    ap.add_argument("--proj_dir", default=None,
                    help="epoch_<N> dir holding trained protein_projection.pt/go_projection.pt "
                         "(overlaid on top of the SFT-base projections). Usually the parent of "
                         "--adapter_path.")
    ap.add_argument("--out", required=True, help="dir for per-sample prediction JSONs")
    ap.add_argument("--esm3_cache_path", default=None,
                    help="ESM3 cache .pt (must be encoded at --max_protein_len)")
    ap.add_argument("--local_parquet", default=None,
                    help="LOCAL parquet dir/file (e.g. datasets/bioreason_rl). When set, "
                         "renders the paper-faithful per-aspect prompt over local rows "
                         "(no gated wanglab/cafa5 needed). NOTE: RL train set → optimistic "
                         "F_max; use for the input-fidelity A/B, not the strict 73.6%% claim.")
    ap.add_argument("--cafa5_dataset", default="wanglab/cafa5")
    ap.add_argument("--cafa5_dataset_name", default="cafa5_reasoning")
    ap.add_argument("--cafa5_cache_dir", default=None,
                    help="HF datasets cache for wanglab/cafa5 (snapshot_download target)")
    ap.add_argument("--interpro_dataset_name", default="interpro_metadata")
    ap.add_argument("--include_go_defs", action="store_true", default=False)
    # BooleanOptionalAction so the text-context flags can be DISABLED for the
    # text-ablation diagnostic (--no-interpro_in_prompt etc.). Default True matches
    # the paper's eval (interpro+ppi+function text in the prompt).
    ap.add_argument("--include_protein_function_summary",
                    action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--interpro_in_prompt",
                    action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--ppi_in_prompt",
                    action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--val_split_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--max_samples", type=int, default=-1)
    ap.add_argument("--max_protein_len", type=int, default=2048)
    ap.add_argument("--num_go_tokens", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--num_shards", type=int, default=1,
                    help="split proteins across N processes/tiles (strided)")
    ap.add_argument("--shard_id", type=int, default=0, help="this shard's index [0,N)")
    ap.add_argument("--max_num_seqs", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0)  # greedy eval
    ap.add_argument("--enable_thinking", action="store_true", default=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    import torch

    # XPU-required vLLM env, mirroring torchtune.dev.rl.vllm_backend._init_vllm_tp1:
    # ZE_AFFINITY_MASK already selects one tile, so vLLM must NOT spawn a V1
    # EngineCore subprocess (it would hang on the already-initialized XPU), and
    # torch.compile must be disabled for the engine. Set BEFORE importing vllm.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    from vllm import LLM, SamplingParams

    model, device = build_model(args)

    # Free the BioReasonModel backbone BEFORE constructing vLLM: build_prompt_embeds
    # only uses self._embed + projections + ESM3/GO caches, never self.backbone (vLLM
    # owns the transformer forward at eval). Dropping it avoids a redundant ~8 GiB
    # backbone copy resident on the same tile as vLLM's own copy.
    if hasattr(model, "backbone"):
        del model.backbone
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.synchronize()
        import gc
        gc.collect()

    # In-process TP=1 vLLM over the same backbone (prompt_embeds path), greedy.
    # Flags match the recipe's proven XPU server-mode init (vllm_backend.py:212).
    llm = LLM(
        model=args.ckpt_dir,
        tensor_parallel_size=1,
        enforce_eager=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.7,
        max_model_len=args.max_protein_len + 512 + args.num_go_tokens + args.max_new_tokens,
        max_num_seqs=args.max_num_seqs,
        disable_custom_all_reduce=True,
        enable_sleep_mode=False,
        enable_prompt_embeds=True,
        trust_remote_code=True,
    )
    sp = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,      # 0 → greedy
        top_k=-1,
        detokenize=True,
    )

    samples = load_local_parquet_samples(args) if args.local_parquet \
        else load_eval_samples(args)

    t0 = time.perf_counter()
    n = 0
    for s in samples:
        seq = s["sequence"]
        # BioReasonModel.tokenizer is the raw HF tokenizer (apply_chat_template + encode).
        prompt_string = build_prompt_string(s, model.tokenizer, args.enable_thinking)
        input_ids = build_input_ids(
            prompt_string, seq[:args.max_protein_len], model.tokenizer, args.num_go_tokens,
        ).to(device).unsqueeze(0)
        with torch.no_grad():
            # go_aspects=["all"]: the ckpt ships a single go_embedding.pt("all"); the
            # asked aspect is conveyed in the prompt text, not the GO embedding.
            pe = model.build_prompt_embeds(input_ids, [seq[:args.max_protein_len]],
                                           go_aspects=["all"])
        out = llm.generate([{"prompt_embeds": pe[0]}], sampling_params=sp)
        resp = out[0].outputs[0].text if out and out[0].outputs else ""

        rec = make_record(s, resp)
        rec["input_prompt"] = prompt_string
        fn = f"{s['protein_id']}_{aspect_code(s['go_aspect'])}_k00.json"
        with open(os.path.join(args.out, fn), "w") as f:
            json.dump(rec, f, indent=2)
        n += 1
        if n % 50 == 0:
            print(f"[eval] {n} samples, {n/(time.perf_counter()-t0):.2f}/s", flush=True)

    print(f"[eval] DONE: {n} prediction JSONs → {args.out}", flush=True)
    print("[eval] score with: python BioReason-Pro/evals/cafa_evals.py "
          f"--input_dir {args.out} "
          "--ontology BioReason-Pro/bioreason2/dataset/go-basic.obo "
          "--ia_file BioReason-Pro/data/IA.txt "
          "--reasoning_mode True --final_answer_only False --threads 0", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

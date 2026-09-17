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

REGRESSION GATE: after ANY change to prompt-construction logic in this file
(inject_go_pred / interpro_in_prompt / ppi_in_prompt / include_protein_function_
summary and their call sites), run
    bash experiments/bioreason/check_eval_harness_sanity.sh
before trusting a new F_max number. It re-scores the published SFT checkpoint at
default (faithful) flags and fails loudly if the result falls outside its known-
good band — the same bug class (a silently-cold prompt corrupting F_max by ~0.3)
recurred twice in a sibling BioReason project for lack of exactly this check. The
CLI defaults themselves are pinned at the unit-test layer by
tests/torchtune/dev/rl/test_eval_cafa_fmax_flag_defaults.py.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def build_native_input_ids(row, protein_seq, tokenizer, protein_token_id, go_token_id,
                           num_go_tokens, inject_go_pred=True,
                           interpro_in_prompt=True, ppi_in_prompt=True,
                           keep_list_prefix=False, add_uniprot_summary=False):
    """Build prompt input_ids EXACTLY as the native SFT trained (dataset_sft).

    Reuses BioReasonSFTDataset._build_prompt_ids verbatim (via __new__, no data load) so
    the eval prompt is bit-identical to training: text(_build_prompt_text) + "\nProtein: "
    + [protein_token_id]*(len(seq)+2) + "\nGO graph: " + [go_token_id]*num_go_tokens +
    "\nReasoning:\n". This is what makes the F_max number valid for our checkpoint (the
    chat-template/<|protein_pad|> path is for the published HF model, NOT our native ckpt).

    interpro_in_prompt / ppi_in_prompt: text ablation (default True = unchanged prior
    behavior), forwarded to BioReasonSFTDataset._build_prompt_text.
    add_uniprot_summary: MUST match whatever the checkpoint was trained with (parity
    contract confound #5) — appends " Summarize in UniProt format." to the prompt. Default
    False preserves prior behavior for every checkpoint trained before this flag existed.
    """
    import torch
    from torchtune.dev.bioreason.dataset_sft import BioReasonSFTDataset
    ds = BioReasonSFTDataset.__new__(BioReasonSFTDataset)  # no __init__ / no data load
    ds.tokenizer = tokenizer
    ds.protein_token_id = int(protein_token_id)
    ds.go_token_id = int(go_token_id)
    ds.num_go_tokens = int(num_go_tokens)
    ds.inject_go_pred = bool(inject_go_pred)
    ds.interpro_in_prompt = bool(interpro_in_prompt)
    ds.ppi_in_prompt = bool(ppi_in_prompt)
    ds.keep_list_prefix = bool(keep_list_prefix)
    ds.add_uniprot_summary = bool(add_uniprot_summary)
    ids = ds._build_prompt_ids(row, protein_seq)
    return torch.tensor(ids, dtype=torch.long)


def build_native_prompt_text(row, tokenizer, interpro_in_prompt=True, ppi_in_prompt=True,
                             keep_list_prefix=False, add_uniprot_summary=False):
    """Return the native SFT text prompt used to build the integer-id input layout."""
    from torchtune.dev.bioreason.dataset_sft import BioReasonSFTDataset

    ds = BioReasonSFTDataset.__new__(BioReasonSFTDataset)
    ds.tokenizer = tokenizer
    ds.interpro_in_prompt = bool(interpro_in_prompt)
    ds.ppi_in_prompt = bool(ppi_in_prompt)
    ds.keep_list_prefix = bool(keep_list_prefix)
    ds.add_uniprot_summary = bool(add_uniprot_summary)
    return ds._build_prompt_text(
        row,
        interpro_in_prompt=ds.interpro_in_prompt,
        ppi_in_prompt=ds.ppi_in_prompt,
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
    # NATIVE-PROMPT path: build inputs EXACTLY as the native SFT was trained
    # (dataset_sft._build_prompt_text + integer-id placeholder layout), one sample PER
    # PROTEIN (training was per-protein, not per-aspect). The reasoning-mode scorer
    # regexes ALL GO terms out of the single generated_response, so one generation per
    # protein covers every aspect. This guarantees eval inputs bit-match training inputs.
    if getattr(args, "native_prompt", False):
        out = []
        for i in idxs:
            row = df.iloc[i]
            seq = (row.get("sequence", "") or "")[: args.max_protein_len]
            present = [a for a in ("MF", "BP", "CC")
                       if _as_list(row.get(ASPECT_TO_COLUMN[a]))]
            if not present:
                continue
            out.append({
                "protein_id": row.get("protein_id", f"unknown_{i}"),
                "sequence": seq,
                "go_aspect": "all",            # per-protein; scorer reads go_bp/mf/cc lists
                "row": {k: row[k] for k in row.index},  # raw row for native prompt build
                "go_bp": _as_list(row.get("go_bp")),
                "go_mf": _as_list(row.get("go_mf")),
                "go_cc": _as_list(row.get("go_cc")),
                "ground_truth": "",
            })
        return out

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
        # _format_reasoning_prompt does plain `if example.get("go_mf"):` truthy checks —
        # correct for HF `datasets` examples (native Python lists) but our rows come from
        # pandas.iloc, where go_mf/go_cc/go_bp are numpy arrays: `if array:` raises
        # "truth value of an array... is ambiguous". Coerce via the same _as_list already
        # used for the aspect-presence check above (a data-shape adapter, not a change to
        # the paper's logic — same fix already landed in a sibling eval driver, see
        # memory/project_bioreason_gopred_eval_fix_20260724.md).
        for _col in ("go_mf", "go_cc", "go_bp"):
            row_d[_col] = _as_list(row_d.get(_col))
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

    # ROOT CAUSE (2026-09-03): the vLLM-HTTP client path (--vllm_http_url) was documented
    # as "backbone-free CPU-only" (see load_backbone below and every launcher's comments)
    # but this line put it on XPU whenever XPU was available regardless. The client's own
    # compute -- an embedding-table lookup + a small MLP projection for build_prompt_embeds
    # -- is genuinely cheap CPU work; there was never a real need for it to touch the GPU.
    # Putting it on XPU meant every client process opened a real Level-Zero context on
    # tiles the vLLM servers are ACTIVELY generating on. ZE_AFFINITY_MASK scoping (see the
    # 2N/4N launchers) only restricts WHICH tiles are visible -- it does not stop the
    # client from creating a context and competing for shared driver-level resources
    # (command queues, event pools) with the server's own context on those same tiles.
    # This is a resource race, not a deterministic bug, which is exactly why evals have
    # been "sometimes hangs, sometimes not" for dozens of prior runs rather than reliably
    # reproducing: intermittent UR_RESULT_ERROR_OUT_OF_RESOURCES client errors and vLLM
    # EngineCore crashes under load, root-caused after two ZE_AFFINITY_MASK-only "fixes"
    # (this session) failed to make the 2-node harness reliable. Force CPU whenever the
    # client doesn't own a backbone -- there is no other XPU work on this path.
    device = torch.device("cpu") if getattr(args, "vllm_http_url", None) else (
        torch.device("xpu") if (hasattr(torch, "xpu") and torch.xpu.is_available())
        else torch.device("cpu")
    )
    _adapter = getattr(args, "adapter_path", None)
    # 32B can't fit one tile for generation -> shard the backbone across N tiles via HF
    # device_map. backbone_device_map="auto" lets HF spread layers over all tiles VISIBLE
    # to this process; the launcher restricts visibility with ZE_AFFINITY_MASK to exactly
    # the N tiles this shard owns (e.g. "0,1"), so "auto" => those N tiles only.
    _dmap = getattr(args, "backbone_device_map", None)
    # Auto-detect projector_output_norm from the saved projection .pt: a LayerNorm-capped
    # projector (model_native projector_output_norm=True) saves a "3.weight"/"3.bias" pair
    # (LayerNorm at Sequential index 3). The eval-side projector arch MUST match or the
    # strict load_state_dict below fails. Detect from proj_dir (falls back to ckpt_dir).
    _proj_norm = bool(getattr(args, "projector_output_norm", False))
    _pd = getattr(args, "proj_dir", None) or args.ckpt_dir
    _pp = os.path.join(_pd, "protein_projection.pt") if _pd else None
    if _pp and os.path.exists(_pp):
        _keys = torch.load(_pp, map_location="cpu").keys()
        if any(k.startswith("3.") for k in _keys):
            _proj_norm = True
            print(f"[eval] auto-detected projector LayerNorm (index-3 keys) -> projector_output_norm=True", flush=True)
    model = BioReasonModel(
        ckpt_dir=args.ckpt_dir,
        device=device,
        dtype=torch.bfloat16,
        esm3_cache_path=args.esm3_cache_path,   # faithful 2048 cache (None → live ESM3)
        enable_lora=bool(_adapter),
        adapter_path=_adapter,                  # None → no LoRA (full-ckpt eval)
        backbone_device_map=_dmap,
        projector_output_norm=_proj_norm,
        disable_protein_splice=bool(getattr(args, "disable_protein_splice", False)),
        disable_go_splice=bool(getattr(args, "disable_go_splice", False)),
        # vLLM-HTTP client mode: a separate vLLM server owns the transformer, so the
        # client only needs _embed + projections for build_prompt_embeds — skip the
        # ~65 GiB backbone load (lets 6 client shards co-reside with the servers).
        load_backbone=not bool(getattr(args, "vllm_http_url", None)),
    )
    if args.disable_protein_splice:
        print("[eval] ABLATION: protein splice DISABLED (placeholder embeds kept)", flush=True)
    if args.disable_go_splice:
        print("[eval] ABLATION: GO splice DISABLED (placeholder embeds kept)", flush=True)
    # NATIVE-PROMPT: our SFT spliced INTEGER ids (151643/151644), not the HF special
    # tokens. Override the model's placeholder ids so build_prompt_embeds matches the
    # native input_ids (else `input_ids == self.protein_token_id` finds 0 -> count
    # mismatch). Must equal the training config's protein_token_id/go_token_id.
    if getattr(args, "native_prompt", False):
        model.protein_token_id = int(args.protein_token_id)
        model.go_token_id = int(args.go_token_id)
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


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser. Factored out of main() so tests can inspect flag
    defaults (e.g. that the go_pred/InterPro/PPI/function-summary injection
    flags default True) WITHOUT running the eval pipeline itself. A sibling
    BioReason eval script silently regressed to a cold, un-injected prompt
    when these defaulted the wrong way — see the flags' own help text below —
    and there was no automated check to catch it before a real training run
    consumed the broken prompt. tests/torchtune/dev/rl pins these defaults."""
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
    ap.add_argument("--add_uniprot_summary",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="native-prompt path only (--native_prompt): MUST match whatever "
                         "the checkpoint was trained with (dataset_sft's add_uniprot_summary "
                         "flag). Default False for backward compat with every checkpoint "
                         "trained before this flag existed.")
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
    ap.add_argument("--k", type=int, default=1,
                    help="Number of independent completions to sample per protein. k=1 (default) "
                         "is the historic single-sample behaviour and writes to --out unchanged. "
                         "k>1 writes k SIBLING replicate dirs <out>/k00 .. <out>/k{N-1}, each "
                         "containing one _k00.json per protein, which is exactly the layout "
                         "rescore_by_group_frequency.py consumes (--replicate_dir <out>/k00 ...). "
                         "That indirection is deliberate: score_fmax_scored.py calls "
                         "ce.select_best_from_k_samples when it sees _k01+ IN ONE DIR, which picks "
                         "the highest-F1 sample AGAINST GROUND TRUTH -- oracle selection that would "
                         "invalidate the number. Keeping each sample in its own dir makes that "
                         "impossible by construction and reuses the already-validated rescorer "
                         "(it reproduces the published 0.6737/0.6417 exactly) instead of adding a "
                         "second scoring path. Motivation: frequency ranking (count/k as the "
                         "confidence) is worth +0.0349 F_max offline, ~2x the +-0.016 noise floor, "
                         "but F_max's threshold sweep is inert without confidences -- see "
                         "memory/project_bioreason_freq_ranking_beats_flat_confidence_20260915.md. "
                         "COST: k-sample eval is ~k x the generation work.")
    ap.add_argument("--repetition_penalty", type=float, default=1.0,
                    help="vLLM repetition_penalty (1.0=off). WARNING: on the --vllm_http_url "
                         "prompt_embeds path (no real prompt_token_ids -- the prompt comes from "
                         "ESM3/GO embeddings), any value != 1.0 crashes the vLLM EngineCore with "
                         "an XPU 'scatter gather kernel index out of bounds' assertion in "
                         "apply_penalties()'s prompt-token bin-count pass (job 8759264, "
                         "2026-08-16: 3/3 servers, ~all requests failed after the first). "
                         "This is NOT a config tuning knob for this eval harness until that vLLM "
                         "bug is fixed upstream -- leave at 1.0. Kept only so a future retry "
                         "against a patched vLLM doesn't require re-adding this plumbing.")
    ap.add_argument("--enable_thinking", action="store_true", default=True)
    ap.add_argument("--no_vllm", action="store_true",
                    help="generate via native HF backbone.generate(inputs_embeds=...) + KV "
                         "cache instead of in-process vLLM. REQUIRED for 32B (vLLM TP=1 "
                         "OOMs: 62 GiB weights > one tile budget). Slower but fits.")
    ap.add_argument("--vllm_http_url", default=None,
                    help="Generate via a vLLM OpenAI HTTP server (e.g. http://localhost:8001) "
                         "using prompt_embeds over /v1/completions. This is the 32B vLLM path: "
                         "the server (launch_vllm_http_32b_tp2.sh, TP=2 --enable-prompt-embeds) "
                         "hosts the sharded backbone; the client builds prompt_embeds "
                         "(backbone-free) and POSTs them. Mutually exclusive with --no_vllm and "
                         "the in-process LLM() path. Uses VLLMClient.generate_from_embeds.")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="Number of samples to process concurrently within this shard "
                         "process, ONLY on the --vllm_http_url path. build_prompt_embeds is "
                         "CPU-only on this path (backbone deleted before the client loop; "
                         "see pbs_1n_eval_vllm_tp2.sh's client launch comment) and the actual "
                         "generation happens server-side, so a thread pool here lets multiple "
                         "HTTP requests be in flight at once against a server's MAX_NUM_SEQS "
                         "budget, instead of one request/shard at a time. ROOT-CAUSED "
                         "2026-08-17: prior evals saw ZERO speedup from raising vLLM's "
                         "MAX_NUM_SEQS (2->16) because this client loop only ever had 1 "
                         "in-flight request per shard regardless of the server's setting --"
                         "see memory/project_bioreason_vllm_max_num_seqs_16_validated_20260816.md. "
                         "Ignored (forced to 1) on --no_vllm / in-process LLM() paths -- those "
                         "are GPU-resident per-process and NOT safe to parallelize this way.")
    ap.add_argument("--vllm_max_model_len", type=int, default=8192,
                    help="MUST match --max-model-len on the vLLM server (launch_vllm_http_32b_"
                         "tp2.sh default 4808). Used to clamp the per-request max_tokens to "
                         "(max_model_len - actual_prompt_len) — a fixed max_new_tokens overflows "
                         "vLLM's own budget check on long prompts (protein+text+GO exceeding the "
                         "assumed 2048+512+200 split), producing 'max_tokens must be at least 1, "
                         "got <negative>' 400s that silently drop that shard's remaining samples.")
    ap.add_argument("--require_full_max_new_tokens", action="store_true",
                    help="Fail a sample instead of silently reducing max_new_tokens when "
                         "the vLLM context budget is too small for an apples-to-apples eval.")
    ap.add_argument("--backbone_device_map", default=None,
                    help="HF device_map for the backbone (e.g. 'auto'). Shards the 32B "
                         "across the tiles VISIBLE to this process (set via ZE_AFFINITY_MASK "
                         "to the N tiles this shard owns). Use with --no_vllm for 32B.")
    ap.add_argument("--native_prompt", action="store_true",
                    help="build inputs via dataset_sft (native SFT training layout: text + "
                         "integer-id placeholders), NOT the HF chat-template/<|protein_pad|> "
                         "path. REQUIRED to eval a native-SFT checkpoint. One sample/protein.")
    ap.add_argument("--native_keep_list_prefix", action="store_true",
                    help="Match keep_list_prefix=true training: place GO terms\\n immediately "
                         "after the native Reasoning header.")
    ap.add_argument("--disable_protein_splice", action="store_true", default=False,
                    help="ABLATION (Exp 2): keep protein placeholder tokens but do NOT write "
                         "projected ESM3 features. Isolates the protein-embedding modality's "
                         "F_max contribution (seqlen/format held constant).")
    ap.add_argument("--disable_go_splice", action="store_true", default=False,
                    help="ABLATION: same as --disable_protein_splice for the GO modality.")
    ap.add_argument("--protein_token_id", type=int, default=151643,
                    help="integer protein placeholder id (native_prompt). Must match the "
                         "training config (Qwen3 reserved-gap id).")
    ap.add_argument("--go_token_id", type=int, default=151644,
                    help="integer GO placeholder id (native_prompt). Must match training.")
    return ap


def validate_k_sampling(k: int, temperature: float, no_vllm: bool) -> None:
    """Reject --k configurations that would silently produce a degenerate group.

    Pure and separately callable so the guard can be tested by CALLING it. An earlier
    version of this check lived inline in main() and its test asserted on source text;
    a mutation that disabled the guard entirely still passed, because the substrings it
    grepped for also appeared in the adjacent check and in the error message. Behaviour
    is the only thing worth asserting on.

    Raises:
        SystemExit: on k < 1, on k > 1 without sampling, or on k > 1 with --no_vllm.
    """
    if k < 1:
        raise SystemExit(f"[eval] --k must be >= 1, got {k}")
    if k == 1:
        return
    if temperature <= 0.0:
        # k identical greedy completions make count/k == 1.0 for every term -- exactly
        # the flat-confidence baseline this path exists to escape, but reported as a
        # k-sample run.
        raise SystemExit(
            f"[eval] --k {k} requires --temperature > 0 (got {temperature}). Greedy "
            "sampling returns the same completion k times, so count/k is 1.0 for every "
            "term and frequency ranking degenerates to the flat-1.0 baseline it is meant "
            "to replace. Pass e.g. --temperature 0.7, or leave --k 1 for greedy eval."
        )
    if no_vllm:
        # That branch calls backbone.generate(do_sample=False, num_beams=1) -- greedy
        # REGARDLESS of --temperature. Wiring do_sample through is doable but unexercised
        # by any current launcher (32B evals go over --vllm_http_url), so refuse loudly
        # rather than ship an untested sampling path returning k identical strings.
        raise SystemExit(
            "[eval] --k > 1 is not supported on the --no_vllm path: that branch hardcodes "
            "backbone.generate(do_sample=False), so it would return k identical greedy "
            "completions no matter what --temperature says. Use --vllm_http_url (the 32B "
            "path) or the in-process vLLM path."
        )


def apply_k_provenance(rec: dict, k_index: int, k_total: int, temperature: float,
                       n_real: int) -> dict:
    """Stamp k-sample provenance, and mark padded placeholders unsuccessful.

    Pure and separately callable so the padding contract can be tested by CALLING it
    (see validate_k_sampling for why source-grep tests are not trusted here).

    `n_real` is how many completions the server actually returned; indices >= n_real are
    empty padding, not predictions. `make_record` hardcodes ``success: True``, and
    ``rescore_by_group_frequency.load_replicate`` only drops ``success=False`` rows --
    so a padded row would be KEPT as a sample that voted for zero terms, while
    ``build_arms`` fixes the denominator at ``g = len(replicate_dirs)``. Every frequency
    in that group would come out scaled by ``n_real/k``, silently, and only in the
    ``freq`` arm -- the arm under test. Marking it unsuccessful makes the loader drop the
    row, which drops the key from the replicate intersection and so drops the whole
    protein: the rescorer's existing "lose the protein rather than bias it" contract,
    not a new one.
    """
    rec["k_index"] = int(k_index)
    rec["k_total"] = int(k_total)
    rec["eval_temperature"] = float(temperature)
    if k_index >= n_real:
        rec["success"] = False
        rec["k_padded"] = True
    return rec


def k_output_dirs(out: str, k: int) -> list:
    """Replicate dirs for a k-sample eval.

    k=1 returns [out] unchanged so every existing launcher, resume-skip check and
    scoring invocation is untouched. k>1 returns sibling <out>/k00 .. <out>/k{N-1},
    each holding one _k00.json per protein -- the layout rescore_by_group_frequency.py
    consumes, and the one layout in which ce.select_best_from_k_samples (which picks the
    best sample AGAINST GROUND TRUTH) cannot fire.
    """
    if k == 1:
        return [out]
    return [os.path.join(out, f"k{i:02d}") for i in range(k)]


def _sp_with_n(sp, k: int):
    """Clone an in-process vLLM SamplingParams with n=k.

    Built by copy so every other field (max_tokens, temperature, repetition_penalty,
    detokenize) stays exactly as the single-sample path set it -- re-listing them here
    would silently drift the two paths apart the next time one is edited.
    """
    import copy
    sp2 = copy.copy(sp)
    sp2.n = k
    return sp2


def main() -> int:
    ap = build_arg_parser()
    args = ap.parse_args()

    validate_k_sampling(args.k, args.temperature, bool(args.no_vllm))

    _out_dirs = k_output_dirs(args.out, args.k)
    for _d in _out_dirs:
        os.makedirs(_d, exist_ok=True)
    import torch

    LLM = SamplingParams = None
    _http_client = None
    if args.vllm_http_url:
        # vLLM-HTTP client mode: a separate OpenAI api_server (TP=2, --enable-prompt-embeds)
        # owns the transformer. We only build prompt_embeds locally (backbone-free) and POST
        # them via VLLMClient.generate_from_embeds. No in-process vLLM, no LLM import here.
        from torchtune.dev.rl.vllm_client import VLLMClient
        # pool_maxsize must cover this process's own --concurrency (one shared Session, one
        # HTTPAdapter per scheme) or requests/urllib3's default cap of 10 connections/host
        # silently serializes past that point even though the server has far more headroom
        # (vLLM itself reports ~160-180x safe concurrency on these TP=4 servers).
        _pool_size = max(10, max(1, args.concurrency) + 2)
        _http_client = VLLMClient(
            base_url=args.vllm_http_url, connection_timeout=1800.0, pool_maxsize=_pool_size
        )
        print(f"[eval] vLLM-HTTP client connected to {args.vllm_http_url} "
              f"(api={_http_client._api_type}, model={_http_client._model_name})", flush=True)
    elif not args.no_vllm:
        # XPU-required vLLM env, mirroring torchtune.dev.rl.vllm_backend._init_vllm_tp1:
        # ZE_AFFINITY_MASK already selects one tile, so vLLM must NOT spawn a V1
        # EngineCore subprocess (it would hang on the already-initialized XPU), and
        # torch.compile must be disabled for the engine. Set BEFORE importing vllm.
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ["TORCH_COMPILE_DISABLE"] = "1"
        from vllm import LLM, SamplingParams

    model, device = build_model(args)

    # ── Native HF-generate path (--no_vllm) ────────────────────────────────────
    # vLLM TP=1 cannot host a 32B (62 GiB weights > one 64 GiB tile budget once its
    # KV preallocation is added -> XPU OOM, 0 preds, job 8573774). The BioReasonModel
    # backbone IS an HF AutoModelForCausalLM with native .generate(inputs_embeds=...)
    # + KV cache, so we generate directly on it — no vLLM, no TP, no enable_prompt_embeds
    # risk. Slower per-sample than vLLM but the only path that fits a 32B at TP=1 on XPU.
    # KEEP model.backbone (do NOT free it — it does the generation here).
    if args.vllm_http_url:
        # vLLM-HTTP client mode: no in-process backbone (load_backbone=False) and no
        # in-process LLM — the server owns the transformer. Just carry the client through.
        llm = None
        sp = None
        backbone = None
    elif args.no_vllm:
        llm = None
        sp = None
        backbone = model.backbone
        backbone.eval()
    else:
        # Free the BioReasonModel backbone BEFORE constructing vLLM: build_prompt_embeds
        # only uses self._embed + projections + ESM3/GO caches, never self.backbone (vLLM
        # owns the transformer forward at eval). Dropping it avoids a redundant backbone
        # copy resident on the same tile as vLLM's own copy.
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
    sp = None if (args.no_vllm or args.vllm_http_url) else SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,      # 0 → greedy
        top_k=-1,
        repetition_penalty=args.repetition_penalty,
        detokenize=True,
    )

    samples = load_local_parquet_samples(args) if args.local_parquet \
        else load_eval_samples(args)

    # Native-prompt mode needs a TORCHTUNE qwen3 tokenizer (dataset_sft._build_prompt_ids
    # uses tok.encode(add_bos=,add_eos=)+tok.bos_id, NOT the HF API). Built once; the HF
    # model.tokenizer is still used for DECODING the generated ids.
    _tt_tok = None
    if getattr(args, "native_prompt", False):
        from torchtune.models.qwen3 import qwen3_tokenizer
        _tt_tok = qwen3_tokenizer(
            path=os.path.join(args.ckpt_dir, "vocab.json"),
            merges_file=os.path.join(args.ckpt_dir, "merges.txt"),
            max_seq_len=args.max_protein_len + args.num_go_tokens + 4096,
        )

    t0 = time.perf_counter()
    n = 0

    # ROOT-CAUSED 2026-08-17: this loop used to be strictly sequential (one HTTP request in
    # flight per shard process, regardless of the vLLM server's --max-num-seqs setting).
    # Raising MAX_NUM_SEQS 2->16 measured ZERO speedup because the server was never sent more
    # than 1 concurrent request per shard to begin with -- see
    # memory/project_bioreason_vllm_max_num_seqs_16_validated_20260816.md. On the
    # --vllm_http_url path, build_prompt_embeds is CPU-only (backbone is deleted before this
    # loop runs -- see pbs_1n_eval_vllm_tp2.sh's client launch comment) and the actual
    # generation work happens server-side, so a thread pool here is safe: while one thread
    # blocks on its HTTP call, others can build embeds and issue their own requests, actually
    # exercising the server's concurrency budget. NOT enabled on --no_vllm / in-process LLM()
    # -- those hold GPU-resident state per-process and are not safe to parallelize this way.
    _concurrency = max(1, args.concurrency) if args.vllm_http_url else 1

    def _process_one(s):
        """Runs one sample end-to-end: build embeds, generate, write JSON. Returns
        (protein_id, ok: bool) for progress tracking; never raises (isolates one
        bad protein from the rest of the shard, matching the historic single-threaded
        per-sample try/except — job 8680415 lost ~47 samples/shard before this existed)."""
        seq = s["sequence"]
        # Resume across walltime-limited slots: skip proteins already predicted.
        # Filename must match the write below (protein_id + aspect_code + _k00.json).
        # Under k>1 the protein is only done when ALL k replicate dirs have it: a protein
        # present in 3 of 8 dirs would otherwise be scored as a 3-sample group, silently
        # mixing group sizes across the eval set and biasing count/k.
        _fn = f"{s['protein_id']}_{aspect_code(s['go_aspect'])}_k00.json"
        if all(os.path.exists(os.path.join(_d, _fn)) for _d in _out_dirs):
            return s["protein_id"], True
        try:
            _effective_max_new_tokens = args.max_new_tokens
            if getattr(args, "native_prompt", False):
                # Bit-identical to training: native text + integer-id placeholder layout.
                prompt_string = build_native_prompt_text(
                    s["row"],
                    _tt_tok,
                    interpro_in_prompt=getattr(args, "interpro_in_prompt", True),
                    ppi_in_prompt=getattr(args, "ppi_in_prompt", True),
                    keep_list_prefix=getattr(args, "native_keep_list_prefix", False),
                    add_uniprot_summary=getattr(args, "add_uniprot_summary", False),
                )
                input_ids = build_native_input_ids(
                    s["row"], seq[:args.max_protein_len], _tt_tok,
                    args.protein_token_id, args.go_token_id, args.num_go_tokens,
                    inject_go_pred=getattr(args, "inject_go_pred", True),
                    interpro_in_prompt=getattr(args, "interpro_in_prompt", True),
                    ppi_in_prompt=getattr(args, "ppi_in_prompt", True),
                    keep_list_prefix=getattr(args, "native_keep_list_prefix", False),
                    add_uniprot_summary=getattr(args, "add_uniprot_summary", False),
                ).to(device).unsqueeze(0)
            else:
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
            if args.no_vllm:
                # Native HF greedy decode on the (KV-cached) backbone. pe is [1, P, H] on
                # device; HF generate accepts inputs_embeds directly. Greedy (do_sample=False),
                # decode only the NEW tokens (HF returns only generated ids for inputs_embeds).
                # With device_map sharding, the input embedding layer may live on a specific
                # tile — move pe to wherever the backbone expects its input (its embed device),
                # not blindly to `device`. get_input_embeddings().weight.device is authoritative.
                _in_dev = backbone.get_input_embeddings().weight.device
                with torch.no_grad():
                    gen_ids = backbone.generate(
                        inputs_embeds=pe.to(device=_in_dev, dtype=next(backbone.parameters()).dtype),
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        num_beams=1,
                        pad_token_id=model.tokenizer.pad_token_id
                            if model.tokenizer.pad_token_id is not None
                            else model.tokenizer.eos_token_id,
                        use_cache=True,
                    )
                # With inputs_embeds (no input_ids), HF returns ONLY the generated token ids.
                # Always k==1 here: --k > 1 is rejected up front on this greedy-only path.
                resps = [model.tokenizer.decode(gen_ids[0], skip_special_tokens=True)]
            elif args.vllm_http_url:
                # vLLM-HTTP: POST prompt_embeds ([P, H] on CPU) to the OpenAI server, get back
                # completion token IDs, decode with the HF tokenizer (same as the no_vllm path).
                # Greedy (temperature=0) to match device_map parity; stop at EOS server-side.
                # CLAMP max_tokens to the server's actual remaining budget: a fixed max_new_tokens
                # overflows vLLM's (max_model_len - prompt_len) check on long prompts, producing a
                # 400 ("max_tokens must be at least 1, got <negative>") that used to kill the
                # whole shard (job 8680415, N=280 — some proteins' protein+text+GO prompt exceeds
                # the assumed split). Mirror the no_vllm path's effective cap instead of trusting
                # a constant.
                _eos = (model.tokenizer.eos_token_id
                        if model.tokenizer.eos_token_id is not None else None)
                _prompt_len = int(pe[0].shape[0])
                _budget = args.vllm_max_model_len - _prompt_len - 8  # small safety margin
                if args.require_full_max_new_tokens and _budget < args.max_new_tokens:
                    raise RuntimeError(
                        f"vLLM parity budget too small: prompt_tokens={_prompt_len}, "
                        f"max_model_len={args.vllm_max_model_len}, "
                        f"requested_new_tokens={args.max_new_tokens}"
                    )
                _req_max_tokens = max(1, min(args.max_new_tokens, _budget))
                _effective_max_new_tokens = _req_max_tokens
                print(
                    f"[eval] vllm_budget protein_id={s['protein_id']} prompt_tokens="
                    f"{_prompt_len} requested={args.max_new_tokens} effective={_req_max_tokens}",
                    flush=True,
                )
                if _budget < 1:
                    print(f"[eval] WARNING: prompt_len={_prompt_len} >= vllm_max_model_len="
                          f"{args.vllm_max_model_len} for {s['protein_id']} — skipping (no "
                          f"budget for any generation)", flush=True)
                    resps = [""] * args.k
                else:
                    # n=args.k: the server samples k completions from ONE prompt in a single
                    # request, so the prompt_embeds payload (~41 MiB for BioReason) crosses
                    # the wire once, not k times.
                    comp_ids = _http_client.generate_from_embeds(
                        [pe[0]],
                        n=args.k,
                        max_tokens=_req_max_tokens,
                        temperature=args.temperature,      # 0 → greedy
                        top_k=0,
                        stop_token_ids=[_eos] if _eos is not None else None,
                        repetition_penalty=args.repetition_penalty,
                    )
                    resps = [
                        model.tokenizer.decode(c, skip_special_tokens=True) if c else ""
                        for c in (comp_ids or [])
                    ]
            else:
                out = llm.generate(
                    [{"prompt_embeds": pe[0]}],
                    sampling_params=(sp if args.k == 1 else _sp_with_n(sp, args.k)),
                )
                resps = [o.text for o in out[0].outputs] if out and out[0].outputs else []

            # The server can legitimately return fewer than k sequences (a finish-reason
            # edge, a dropped candidate). Pad rather than write a short group: a missing
            # replicate file would make this protein's resume-skip fail forever, and a
            # ragged group silently changes the count/k denominator per protein.
            _n_real = len(resps)
            if _n_real < args.k:
                print(f"[eval] WARNING: k={args.k} requested but {_n_real} completions "
                      f"returned for {s['protein_id']} — padding with empty "
                      f"(success=False; this protein will be DROPPED from the scored set. "
                      f"Delete its files under {args.out} and re-run to recover it.)",
                      flush=True)
                resps = list(resps) + [""] * (args.k - _n_real)

            fn = f"{s['protein_id']}_{aspect_code(s['go_aspect'])}_k00.json"
            for _i, _d in enumerate(_out_dirs):
                rec = make_record(s, resps[_i])
                rec["input_prompt"] = prompt_string
                rec["native_keep_list_prefix"] = bool(
                    getattr(args, "native_keep_list_prefix", False)
                )
                rec["requested_max_new_tokens"] = int(args.max_new_tokens)
                rec["effective_max_new_tokens"] = int(_effective_max_new_tokens)
                rec["generation_backend"] = (
                    "vllm_http" if args.vllm_http_url
                    else "hf_generate" if args.no_vllm else "vllm"
                )
                if args.k > 1:
                    apply_k_provenance(rec, _i, args.k, args.temperature, _n_real)
                # Write to a temp file + atomic rename: with --concurrency>1, multiple threads
                # write to the SAME dir concurrently, and a partial/interleaved write to
                # the final filename would corrupt a JSON that a concurrent resume-skip check
                # (os.path.exists) might read before this write completes.
                _tmp = os.path.join(_d, f".{fn}.tmp{os.getpid()}.{_i}")
                with open(_tmp, "w") as f:
                    json.dump(rec, f, indent=2)
                os.replace(_tmp, os.path.join(_d, fn))
            return s["protein_id"], True
        except Exception as e:  # noqa: BLE001
            print(f"[eval] SAMPLE FAILED protein_id={s.get('protein_id')}: "
                  f"{type(e).__name__}: {e}", flush=True)
            return s.get("protein_id"), False

    if _concurrency <= 1:
        for s in samples:
            _, ok = _process_one(s)
            if ok:
                n += 1
                if n % 50 == 0:
                    print(f"[eval] {n} samples, {n/(time.perf_counter()-t0):.2f}/s", flush=True)
    else:
        print(f"[eval] dispatching with concurrency={_concurrency}", flush=True)
        with ThreadPoolExecutor(max_workers=_concurrency) as pool:
            futures = [pool.submit(_process_one, s) for s in samples]
            for fut in as_completed(futures):
                _, ok = fut.result()
                if ok:
                    n += 1
                    if n % 50 == 0:
                        print(f"[eval] {n} samples, {n/(time.perf_counter()-t0):.2f}/s",
                              flush=True)

    if args.k > 1:
        print(f"[eval] DONE: {n} proteins x k={args.k} → {len(_out_dirs)} replicate dirs "
              f"under {args.out}", flush=True)
        print("[eval] score with (frequency ranking; do NOT point a scorer at the parent "
              "dir): python experiments/bioreason/rescore_by_group_frequency.py "
              f"--replicate_dir {' '.join(_out_dirs)} "
              f"--out_root {os.path.join(args.out, 'arms')}", flush=True)
    else:
        print(f"[eval] DONE: {n} prediction JSONs → {args.out}", flush=True)
        print("[eval] score with: python BioReason-Pro/evals/cafa_evals.py "
              f"--input_dir {args.out} "
              "--ontology BioReason-Pro/bioreason2/dataset/go-basic.obo "
              "--ia_file BioReason-Pro/data/IA.txt "
              "--reasoning_mode True --final_answer_only False --threads 0", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

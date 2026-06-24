#!/usr/bin/env python3
"""Offline ESM3 pre-encode for BioReason GRPO.

ESM3 is FROZEN during RL and the dataset has a bounded, fixed set of protein
sequences, yet the training loop re-runs the ~1.4B fp32 ESM3 encoder every step
on the critical path (and keeps it resident, ~5.5 GiB/tile). This script runs
ESM3 ONCE over every unique (truncated) sequence and writes a keyed cache so the
training process can (a) never load ESM3 and (b) replace the per-step encode with
a dict lookup. Mirrors the existing ``go_embedding.pt`` precompute pattern.

Output:
  <out>/esm3_cache.pt        {sha1(seq_trunc): tensor[L+2, embedding_dim] bf16}
  <out>/esm3_cache.pt.json   sidecar {max_protein_len, embedding_dim,
                                      esm3_model_name, n_seqs}

The key/truncation MUST match BioReasonRLDataset (sequence[:max_protein_len]) and
BioReasonModel.esm3_cache_key (sha1). Idempotent + resumable: an existing partial
cache is loaded and only missing keys are computed.

Run on ONE tile of any hold (single-process, no distributed):
  ZE_AFFINITY_MASK=0 python experiments/bioreason/precompute_esm3_cache.py \
      --data_dir /lus/flare/projects/ModCon/ngetty/datasets/bioreason_rl \
      --max_protein_len 128

Needs the BioReason deps on PYTHONPATH (BIOREASON_DEPS) and `module load frameworks`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import torch


def _esm3_cache_key(sequence: str) -> str:
    """Must match BioReasonModel.esm3_cache_key."""
    return hashlib.sha1(sequence.encode("ascii", "ignore")).hexdigest()


def _list_parquet(data_dir: str) -> list[str]:
    """Find parquet shards without glob (DAOS/dfuse hangs on glob.glob)."""
    data_sub = os.path.join(data_dir, "data")
    root = data_sub if os.path.isdir(data_sub) else data_dir
    files = [
        os.path.join(root, fn)
        for fn in sorted(os.listdir(root))
        if fn.endswith(".parquet")
    ]
    if not files:
        raise FileNotFoundError(f"No .parquet under {root}")
    return files


def _load_unique_sequences(data_dir: str, max_protein_len: int) -> list[str]:
    import pyarrow.parquet as pq

    seen: set[str] = set()
    uniq: list[str] = []
    for path in _list_parquet(data_dir):
        col = pq.read_table(path, columns=["sequence"]).column("sequence").to_pylist()
        for s in col:
            if not s:
                continue
            t = s[:max_protein_len]  # mirror dataset.py truncation EXACTLY
            if t not in seen:
                seen.add(t)
                uniq.append(t)
    return uniq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir",
        default="/lus/flare/projects/ModCon/ngetty/datasets/bioreason_rl",
    )
    ap.add_argument("--out", default=None, help="cache .pt path (default <data_dir>/esm3_cache.pt)")
    ap.add_argument("--max_protein_len", type=int, default=128)
    ap.add_argument("--esm3_model_name", default="esm3_sm_open_v1")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--flush_every", type=int, default=500,
                    help="checkpoint the partial cache to disk every N new seqs")
    args = ap.parse_args()

    out_path = args.out or os.path.join(args.data_dir, "esm3_cache.pt")
    sidecar = out_path + ".json"

    device = (
        torch.device("xpu") if (hasattr(torch, "xpu") and torch.xpu.is_available())
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"[precompute] device={device} out={out_path}", flush=True)

    uniq = _load_unique_sequences(args.data_dir, args.max_protein_len)
    print(f"[precompute] {len(uniq)} unique truncated sequences "
          f"(max_protein_len={args.max_protein_len})", flush=True)

    # Resume: load any existing partial cache.
    cache: dict[str, torch.Tensor] = {}
    if os.path.exists(out_path):
        cache = torch.load(out_path, map_location="cpu")
        print(f"[precompute] resuming: {len(cache)} keys already cached", flush=True)

    todo = [s for s in uniq if _esm3_cache_key(s) not in cache]
    print(f"[precompute] {len(todo)} sequences to encode", flush=True)

    if todo:
        # Reuse the model wrapper's path/env setup: it inserts BIOREASON_SRC/DEPS
        # onto sys.path, sets INFRA_PROVIDER, and installs the unsloth-avoidance
        # shim — exactly what's needed before importing ESM3 (bioreason2).
        from torchtune.dev.bioreason.model import _ensure_paths
        _ensure_paths()
        from bioreason2.models.protein_encoder import create_protein_encoder

        enc = create_protein_encoder(args.esm3_model_name, inference_mode=True)
        enc.model.to(device=device)
        embedding_dim = int(enc.embedding_dim)

        t0 = time.perf_counter()
        for i, seq in enumerate(todo):
            # encode_sequences(seqs, batch_idx_map, batch_size) -> list per batch
            # item; one seq, one batch slot -> [L+2, embedding_dim].
            raw = enc.encode_sequences([seq], [0], 1)
            feat = raw[0].detach().to(torch.bfloat16).cpu().contiguous()
            cache[_esm3_cache_key(seq)] = feat
            if (i + 1) % args.log_every == 0:
                rate = (i + 1) / max(time.perf_counter() - t0, 1e-6)
                print(f"[precompute] {i + 1}/{len(todo)} ({rate:.1f} seq/s)", flush=True)
            if (i + 1) % args.flush_every == 0:
                _atomic_save(cache, out_path)
                print(f"[precompute] flushed partial cache ({len(cache)} keys)", flush=True)
    else:
        # Nothing to do, but we may still need embedding_dim for the sidecar.
        embedding_dim = int(next(iter(cache.values())).shape[-1])

    _atomic_save(cache, out_path)
    meta = {
        "max_protein_len": args.max_protein_len,
        "embedding_dim": embedding_dim,
        "esm3_model_name": args.esm3_model_name,
        "n_seqs": len(cache),
    }
    with open(sidecar, "w") as f:
        json.dump(meta, f, indent=2)

    size_gib = os.path.getsize(out_path) / 1e9
    print(f"[precompute] DONE: {len(cache)} keys, {size_gib:.2f} GiB → {out_path}", flush=True)
    print(f"[precompute] sidecar: {meta}", flush=True)
    return 0


def _atomic_save(obj, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


if __name__ == "__main__":
    sys.exit(main())

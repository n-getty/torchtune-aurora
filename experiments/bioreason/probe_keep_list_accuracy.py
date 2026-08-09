#!/usr/bin/env python3
"""Measure keep-list decisions from BioReason prediction JSON files.

This is intentionally inference-free.  It compares the GO terms present in the
prompt's ``initial GO term speculations`` block with the terms emitted by a
checkpoint, and scores the resulting binary keep/drop decisions against the
ground-truth GO columns.  That makes it useful before spending another XPU
allocation on a full F_max evaluation.
"""
import argparse
import json
import os
import re
import sys
from collections.abc import Iterable
from typing import Dict, List, Set, Union

GO_ID = re.compile(r"GO:\d{7}")
SPECULATIONS = re.compile(
    r"initial GO term speculations:\s*(.*?)\n\nReason about the function",
    re.IGNORECASE | re.DOTALL,
)
KEEP_LIST = re.compile(r"(?:^|\n)GO terms:\s*\n(?P<body>.*?)(?:\n\n|$)", re.DOTALL)


Number = Union[int, float]


def _terms(value: object) -> Set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return set(GO_ID.findall(value))
    if isinstance(value, Iterable):
        result: Set[str] = set()
        for item in value:
            result.update(_terms(item))
        return result
    return set()


def prompt_candidates(prompt: str) -> Set[str]:
    match = SPECULATIONS.search(prompt)
    return _terms(match.group(1)) if match else set()


def emitted_keep_list(response: str) -> Set[str]:
    """Parse only the candidates-first bare-ID block, not reasoning text."""
    match = KEEP_LIST.search(response)
    if not match:
        return set()
    return set(
        term for line in match.group("body").splitlines()
        for term in [line.strip()]
        if GO_ID.fullmatch(term)
    )


def measure(record: dict) -> Dict[str, Number]:
    candidates = prompt_candidates(record.get("input_prompt", ""))
    truth = _terms(record.get("go_mf")) | _terms(record.get("go_cc")) | _terms(record.get("go_bp"))
    emitted = emitted_keep_list(record.get("generated_response", ""))
    true_candidates = candidates & truth
    false_candidates = candidates - truth
    true_positives = len(emitted & true_candidates)
    false_positives = len(emitted & false_candidates)
    false_negatives = len(true_candidates - emitted)
    true_negatives = len(false_candidates - emitted)
    return {
        "candidates": len(candidates),
        "ground_truth_candidates": len(true_candidates),
        "emitted_candidates": len(emitted),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
        "candidate_accuracy": (true_positives + true_negatives) / len(candidates)
        if candidates else 1.0,
        "precision": true_positives / (true_positives + false_positives)
        if true_positives + false_positives else 1.0,
        "recall": true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives else 1.0,
    }


def measure_sets(candidates: Set[str], truth: Set[str], emitted: Set[str]) -> Dict[str, Number]:
    emitted = emitted & candidates
    true_candidates = candidates & truth
    false_candidates = candidates - truth
    true_positives = len(emitted & true_candidates)
    false_positives = len(emitted & false_candidates)
    false_negatives = len(true_candidates - emitted)
    true_negatives = len(false_candidates - emitted)
    return {
        "candidates": len(candidates),
        "ground_truth_candidates": len(true_candidates),
        "emitted_candidates": len(emitted),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
        "candidate_accuracy": (true_positives + true_negatives) / len(candidates)
        if candidates else 1.0,
        "precision": true_positives / (true_positives + false_positives)
        if true_positives + false_positives else 1.0,
        "recall": true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives else 1.0,
    }


def _prediction_files(root: str) -> List[str]:
    files: List[str] = []
    for directory, _, names in os.walk(root):
        files.extend(os.path.join(directory, name) for name in names if name.endswith(".json"))
    return sorted(files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction_dir", help="Directory of eval JSON files")
    parser.add_argument("--ckpt_dir", help="Checkpoint directory for live generation")
    parser.add_argument("--proj_dir", help="Projector directory for live generation")
    parser.add_argument("--num_proteins", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable output")
    args = parser.parse_args()

    if args.ckpt_dir or args.proj_dir:
        if not args.ckpt_dir:
            parser.error("--ckpt_dir is required for live generation")
        return _run_live_probe(args)
    if not args.prediction_dir:
        parser.error("--prediction_dir is required unless --ckpt_dir/--proj_dir is set")

    totals: Dict[str, Number] = {key: 0 for key in (
        "candidates", "ground_truth_candidates", "emitted_candidates",
        "true_positives", "false_positives", "false_negatives", "true_negatives",
    )}
    records = 0
    for path in _prediction_files(args.prediction_dir):
        try:
            with open(path, encoding="utf-8") as handle:
                result = measure(json.load(handle))
        except (OSError, json.JSONDecodeError, TypeError) as error:
            print(f"warning: skipping {path}: {error}", file=sys.stderr)
            continue
        records += 1
        for key in totals:
            totals[key] += result[key]

    if not records:
        parser.error(f"no readable JSON prediction files under {args.prediction_dir}")
    tp = totals["true_positives"]
    fp = totals["false_positives"]
    fn = totals["false_negatives"]
    tn = totals["true_negatives"]
    candidates = totals["candidates"]
    summary = {
        "records": records,
        **totals,
        "candidate_accuracy": (tp + tn) / candidates if candidates else 1.0,
        "precision": tp / (tp + fp) if tp + fp else 1.0,
        "recall": tp / (tp + fn) if tp + fn else 1.0,
    }
    summary["gates"] = (
        "CLEAR_80" if summary["candidate_accuracy"] >= 0.8
        else "CLEAR_60" if summary["candidate_accuracy"] >= 0.6
        else "BELOW_60"
    )
    if args.json:
        print(json.dumps(summary, sort_keys=True))
    else:
        print(f"records: {records}")
        print(f"candidate accuracy: {summary['candidate_accuracy']:.4f}")
        print(f"precision: {summary['precision']:.4f}")
        print(f"recall: {summary['recall']:.4f}")
        print(f"gates: {summary['gates']}")
        print(f"candidates: {candidates} (GT={totals['ground_truth_candidates']}, emitted={totals['emitted_candidates']})")
    return 0


def _run_live_probe(args) -> int:
    """Generate a small native-prompt sample using the checkpoint under test."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import eval_cafa_fmax as ecf
    import probe_generation_health as health

    # Importing the dataset class here keeps the offline JSON mode lightweight.
    from torchtune.dev.bioreason.dataset_sft import BioReasonSFTDataset

    eval_args = ecf.build_arg_parser().parse_args([
        "--ckpt_dir", args.ckpt_dir,
        "--proj_dir", args.proj_dir or args.ckpt_dir,
        "--out", "/tmp/probe_keep_list_unused",
        "--native_prompt", "--no_vllm",
        "--backbone_device_map", "auto",
        "--local_parquet", os.environ.get(
            "BIOREASON_PROBE_PARQUET",
            "/lus/flare/projects/ModCon/ngetty/datasets/bioreason_rl",
        ),
        "--esm3_cache_path", os.environ.get(
            "BIOREASON_PROBE_ESM3_CACHE",
            "/lus/flare/projects/ModCon/ngetty/datasets/bioreason_rl/esm3_cache_2048.pt",
        ),
    ])
    model, device = ecf.build_model(eval_args)
    model.backbone.eval()
    samples = ecf.load_local_parquet_samples(eval_args)
    from torchtune.models.qwen3 import qwen3_tokenizer
    tokenizer = qwen3_tokenizer(
        path=os.path.join(eval_args.ckpt_dir, "vocab.json"),
        merges_file=os.path.join(eval_args.ckpt_dir, "merges.txt"),
        max_seq_len=eval_args.max_protein_len + eval_args.num_go_tokens + 4096,
    )
    totals = {key: 0 for key in (
        "candidates", "ground_truth_candidates", "emitted_candidates",
        "true_positives", "false_positives", "false_negatives", "true_negatives",
    )}
    checked = 0
    for sample in samples:
        if checked >= args.num_proteins:
            break
        row = sample["row"]
        candidates = set(BioReasonSFTDataset._gopred_terms(row))
        truth = set(BioReasonSFTDataset._gt_terms(row))
        input_ids = ecf.build_native_input_ids(
            row, sample["sequence"][:eval_args.max_protein_len], tokenizer,
            eval_args.protein_token_id, eval_args.go_token_id, eval_args.num_go_tokens,
            inject_go_pred=True, interpro_in_prompt=True, ppi_in_prompt=True,
        ).to(device).unsqueeze(0)
        response = health._generate_variant(
            model, model.backbone, device, input_ids, sample["sequence"],
            eval_args.max_protein_len, eval_args.max_new_tokens,
            False, 0.0, None, False,
        )
        result = measure_sets(candidates, truth, emitted_keep_list(response))
        for key in totals:
            totals[key] += result[key]
        checked += 1
    if not checked:
        raise RuntimeError("no proteins available for live keep-list probe")
    candidates = totals["candidates"]
    accuracy = (totals["true_positives"] + totals["true_negatives"]) / candidates
    print(f"checked proteins: {checked}")
    print(f"candidate accuracy: {accuracy:.4f}")
    print(f"precision: {totals['true_positives'] / (totals['true_positives'] + totals['false_positives']) if totals['true_positives'] + totals['false_positives'] else 1.0:.4f}")
    print(f"recall: {totals['true_positives'] / (totals['true_positives'] + totals['false_negatives']) if totals['true_positives'] + totals['false_negatives'] else 1.0:.4f}")
    print(f"gates: {'CLEAR_80' if accuracy >= 0.8 else 'CLEAR_60' if accuracy >= 0.6 else 'BELOW_60'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

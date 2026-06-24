#!/usr/bin/env python3
"""Unweighted F_max scorer — reuses the paper's cafa_evals processing verbatim.

The paper's cafa_evals.py main() hard-requires the weighted-F column 'f_w', which
cafaeval only emits when an IA file is supplied. This thin wrapper runs the IDENTICAL
processing (process_json_data in reasoning_mode -> CAFA-format files -> cafa_eval) but
reads the UNWEIGHTED best-F ('f') so a baseline can be produced without IA.txt. When
IA.txt IS available, prefer the paper's run_cafa_eval.sh for the weighted number.

Usage:
  PYTHONPATH=$BIOREASON_DEPS:$BIOREASON_SRC python score_fmax_unweighted.py \
    --input_dir <pred_json_dir> --ontology <go-basic.obo> [--ia_file IA.txt]
"""
import argparse, os, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--ontology", required=True)
    ap.add_argument("--ia_file", default=None)
    ap.add_argument("--reasoning_mode", default="True")
    ap.add_argument("--final_answer_only", default="False",
                    help="extract GO terms only AFTER </think> (the post-reasoning "
                         "final answer) instead of from the whole response")
    args = ap.parse_args()

    bsrc = os.environ.get("BIOREASON_SRC", "/lus/flare/projects/ModCon/ngetty/BioReason-Pro")
    sys.path.insert(0, os.path.join(bsrc, "evals"))
    sys.path.insert(0, bsrc)
    import cafa_evals as ce
    from cafaeval.evaluation import cafa_eval

    reasoning = args.reasoning_mode.lower() in ("1", "true", "yes")
    final_only = args.final_answer_only.lower() in ("1", "true", "yes")
    preds, gts = ce.process_json_data(args.input_dir, reasoning_mode=reasoning,
                                      final_answer_only=final_only, go_dag=None)
    if not preds or not gts:
        print(f"[score] NO predictions/ground-truth (preds={len(preds)}, gts={len(gts)})")
        return 1

    pred_dir = os.path.join(args.input_dir, "_cafa_pred")
    os.makedirs(pred_dir, exist_ok=True)
    pred_tsv = os.path.join(pred_dir, "llm_predictions.tsv")
    gt_tsv = os.path.join(args.input_dir, "ground_truth.tsv")
    ce.create_cafa_prediction_file(preds, pred_tsv)
    ce.create_cafa_ground_truth_file(gts, gt_tsv)

    # th_step=0.99 because every predicted term has score 1.0 (binary predictions).
    if args.ia_file and os.path.exists(args.ia_file):
        eval_df, best = cafa_eval(args.ontology, pred_dir, gt_tsv, args.ia_file, th_step=0.99)
        weighted = True
    else:
        eval_df, best = cafa_eval(args.ontology, pred_dir, gt_tsv, th_step=0.99)
        weighted = False

    fdf = best.get("f")
    if fdf is None:
        print("[score] ERROR: no best-F dataframe")
        return 1
    fdf = fdf.reset_index()
    print("\n=== F_max (unweighted) by namespace ===")
    per_ns = {}
    for _, r in fdf.iterrows():
        per_ns[r["ns"]] = float(r["f"])
        print(f"  {r['ns']:25s}: {r['f']:.4f}")
    overall = sum(per_ns.values()) / len(per_ns) if per_ns else 0.0
    print(f"  {'OVERALL MEAN F_max':25s}: {overall:.4f}")
    print(f"\n[score] proteins_with_preds={len(preds)} proteins_with_gt={len(gts)} "
          f"weighted={weighted}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

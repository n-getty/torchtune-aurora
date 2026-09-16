#!/bin/bash
#PBS -N br_eval_grpo
#PBS -A ModCon
#PBS -q debug
#PBS -l select=2
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/logs/pbs_2n_eval_grpo_ckpt.out
#
# PHASE 3 — held-out CAFA F_max eval of a GRPO LoRA checkpoint.
#
# Modeled on pbs_2n_eval_v8_trend.sh. The difference: an SFT/merged checkpoint is
# evaluated with --ckpt_dir alone, but a GRPO run saves a LoRA adapter + GRPO-trained
# projectors, so this passes ADAPTER_DIR through to pbs_2n_eval_vllm_tp2.sh's optional
# adapter path (--ckpt_dir <v8 SFT base> --adapter_path <adapter> --proj_dir <parent>).
#
# USAGE:
#   CKPT=/path/to/epoch_0_step100 REP=1 qsub -v CKPT,REP $0
#   CKPT=... REP=2 N=150 qsub -v CKPT,REP,N $0
#   CKPT=... REP=1 N=500 qsub -v CKPT,REP,N $0      # high-power endpoint arm
#
# N: request cap on proteins. Empirical yield is ~84.7% (127 of 150 -- the pipeline
# stamps the real number as `achievable_ceiling` in coverage.txt; do NOT re-derive it
# from the parquet, see memory/feedback_use_the_pipelines_reported_ceiling_not_a_re_
# derived_one_20260916.md). Cost is ~5.2 min fixed + ~2.5 s/protein, so N=500 (~424
# scorable) is ~23 min and fits the 60-min debug queue.
#
# Why you may want N>150: the paired-reward statistic has se ~= 0.10/sqrt(n). At n=127
# that is 0.0089, enough to detect the paper's whole RL gain (+0.0222, t=2.5) but NOT a
# half-size one (+0.011 -> t=1.2). N=500 gives se 0.0049 and t=2.3 even at half size.
# See memory/project_bioreason_paper_rl_gain_paired_calibration_20260916.md.
#
# CKPT is a checkpoint dir containing adapter/ + protein_projection.pt + go_projection.pt.
# Point it at an rsync'd SNAPSHOT, never a live training output_dir: the trainer
# overwrites its checkpoint dir in place as it goes, so evaluating the live copy both
# races the writer and leaves you unable to say which step you measured.
#
# REP exists because of the eval noise floor. Two clean evals of the SAME checkpoint,
# same seed, same 127 proteins, differing only in server topology, returned F_max
# 0.6737 vs 0.6896 -- a 0.0159 spread
# (memory/project_bioreason_eval_run_to_run_noise_0016_fmax_20260905.md). Any single
# reading below ~0.016 of another is indistinguishable from noise. Run REP=1,2,3 at the
# endpoints and report mean +- spread; a lone number is not a result.

set -eo pipefail
TT=/lus/flare/projects/ModCon/ngetty/torchtune

[ -n "${CKPT:-}" ] || { echo "FATAL: CKPT not set — usage: CKPT=<ckpt dir> REP=1 qsub -v CKPT,REP $0"; exit 2; }
[ -d "$CKPT" ]     || { echo "FATAL: CKPT dir does not exist: $CKPT"; exit 2; }

REP=${REP:-1}
_CKPT_TAG=$(basename "${CKPT%/}")

# The base the GRPO run trained on top of (same value the GRPO config's
# base_model_path resolves from). The adapter is meaningless against any other base.
export EPOCH=${EPOCH:-/lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/runs/sft_qwen3_32b_lora_r128_lrsched_v8_step1550_snapshot}
export ADAPTER_DIR="${CKPT%/}/adapter"

# OUT is eval_out/$TAG (pbs_2n_eval_vllm_tp2.sh:87), so anything NOT in TAG silently
# overwrites a previous arm on the same checkpoint. EVAL_K was not in TAG: running the
# k=8 frequency-ranked arm on a checkpoint already scored at k=1 destroyed the k=1 F_max
# AND the coverage.txt line recording that it *was* k=1. The k!=1 suffix is conditional
# so every existing k=1 path (step-0 baseline x3, step-20) keeps its exact name.
# See memory/feedback_eval_tag_is_internally_derived_k8_would_clobber_k1_20260916.md
_K_SUFFIX=""
[ "${EVAL_K:-1}" != "1" ] && _K_SUFFIX="_k${EVAL_K}"
# N has the same clobber hazard as EVAL_K and for the same reason: the draw is
# df.sample(random_state=7).head(N), so a larger N is a NESTED SUPERSET of a smaller one
# -- comparable, but a DIFFERENT protein population. Writing an N=500 run over an N=150
# dir would destroy the 127-protein arm the step-0/20/40 trend is built on, and
# coverage.txt's `requested_max_samples` line with it. Conditional again, so every
# existing N=150 path keeps its exact name.
_N_SUFFIX=""
[ "${N:-150}" != "150" ] && _N_SUFFIX="_n${N}"
export TAG="${TAG:-grpo_${_CKPT_TAG}_rep${REP}${_K_SUFFIX}${_N_SUFFIX}}"
export N=${N:-150}
export SEED=7
export PARITY=0
export MAX_NEW_TOKENS=5000
export TEST=/lus/flare/projects/ModCon/ngetty/datasets/bioreason_pro_test/test-00000-of-00001.parquet
export CACHE=/lus/flare/projects/ModCon/ngetty/datasets/bioreason_pro_test/esm3_cache_2048.pt
export OBO=/lus/flare/projects/ModCon/ngetty/BioReason-Pro/bioreason2/dataset/go-basic.obo
export IA=/lus/flare/projects/ModCon/ngetty/datasets/bioreason_pro_test/IA.txt
export RESUME=0
# MUST match training. The GRPO config sets add_uniprot_summary: true (inherited from
# the v8 SFT distribution); evaluating without it is a prompt-distribution mismatch.
export ADD_UNIPROT_SUMMARY=1

export TILES_PER_SERVER=4
# 2 engines/node, NOT the 3/node default: 3/node is the confirmed banned:1 trigger
# (16/16 recorded crashes), while 1/node and 2/node both ran clean 127/127 and faster.
export NSERVERS_PER_NODE=2
export CLIENTS_PER_SERVER=3
export MAX_NUM_SEQS=16
export CONCURRENCY=12

[ -f "$ADAPTER_DIR/adapter_model.safetensors" ] || { echo "FATAL: no adapter_model.safetensors at $ADAPTER_DIR"; exit 2; }
[ -f "${CKPT%/}/protein_projection.pt" ] || { echo "FATAL: no protein_projection.pt at $CKPT — GRPO trains the projectors; evaluating without them measures the wrong model"; exit 2; }
[ -f "$EPOCH/config.json" ] || { echo "FATAL: base checkpoint missing: $EPOCH"; exit 2; }
[ -f "$CACHE" ] || { echo "FATAL: layer-(-1) ESM3 cache missing: $CACHE"; exit 2; }

echo "=== GRPO eval: ckpt=$CKPT rep=$REP tag=$TAG base=$EPOCH ==="
exec bash "$TT/experiments/bioreason/pbs_2n_eval_vllm_tp2.sh"

#!/bin/bash
#PBS -N br_eval_step0
#PBS -A ModCon
#PBS -q debug
#PBS -l select=2
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
# Directory (trailing slash) => PBS names each file br_eval_step0.o<jobid>, so concurrent
# reps cannot overwrite each other's log. Was a fixed filename.
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/logs/
#
# PHASE 3 — the GRPO run's **step-0** F_max baseline. Three repeats of this job are the
# left endpoint that every "did GRPO move F_max?" claim is differenced against.
#
# WHY THIS EVALUATES NO ADAPTER (the load-bearing fact — do not "fix" it by adding one)
# ------------------------------------------------------------------------------------
# The GRPO config sets `resume_from_checkpoint: False` and the SFT lineage it builds on is
# a full-FT *merge*, so there is no prior adapter to resume. GRPO therefore starts from a
# FRESHLY INITIALIZED LoRA adapter, created by `get_peft_model(..., init_lora_weights=
# "gaussian")` (torchtune/dev/bioreason/model.py:394). Verified against the installed
# peft 0.19.1: the "gaussian" branch of `LoraLayer.reset_lora_parameters` draws lora_A from
# a normal and calls `nn.init.zeros_(lora_B.weight)`. Since a LoRA contributes
# `scale * (B @ A)` and B is exactly zero, the step-0 adapter contributes EXACTLY ZERO to
# every projection — W_eff == W_base, bit-identically.
#
# So step 0 == the bare v8 step1550 SFT merge, and the correct step-0 eval passes no
# adapter at all. pbs_2n_eval_vllm_tp2.sh makes this explicit: with ADAPTER_DIR unset,
# every adapter arg is empty and the invocation is byte-identical to a base eval
# (see its comment at :126). Passing a zero-effect adapter instead would measure the same
# model while adding a staging step that can fail — strictly worse.
#
# Corollary worth keeping: this baseline is NOT specific to the GRPO run. It is the v8
# step1550 F_max. If a v8-trend eval of step 1550 already exists at these exact settings,
# it is a valid rep and this job is redundant for that rep.
#
# WHY THREE REPEATS
# -----------------
# Two clean evals of the SAME checkpoint, same seed, same 127 proteins, differing only in
# server topology, returned 0.6737 vs 0.6896 — a 0.0159 spread
# (memory/project_bioreason_eval_run_to_run_noise_0016_fmax_20260905.md). A single reading
# cannot establish an endpoint. Report mean ± spread across reps; a lone number is not a
# result, and any endpoint delta under ~0.016 is not distinguishable from noise.
#
# USAGE (one at a time — the debug queue allows 1 running + 1 queued per user):
#   REP=1 qsub -v REP experiments/bioreason/pbs_2n_eval_grpo_step0.sh
#   REP=2 qsub -v REP experiments/bioreason/pbs_2n_eval_grpo_step0.sh
#   REP=3 qsub -v REP experiments/bioreason/pbs_2n_eval_grpo_step0.sh
# `qsub -v REP` is required: a caller-side `REP=1 qsub ...` prefix does NOT propagate into
# the job environment.
#
# REP is not cosmetic — TAG derives OUT=eval_out/$TAG and LOGDIR=eval_logs/$TAG
# (pbs_2n_eval_vllm_tp2.sh:72-73). Two reps sharing a TAG would overwrite each other's
# scores and silently leave one reading where there should be two.

set -eo pipefail
TT=/lus/flare/projects/ModCon/ngetty/torchtune

[ -n "${REP:-}" ] || { echo "FATAL: REP not set — usage: REP=1 qsub -v REP $0"; exit 2; }
case "$REP" in ''|*[!0-9]*) echo "FATAL: REP must be an integer, got '$REP'"; exit 2;; esac

# The base the GRPO run trains on top of — same value the GRPO config's base_model_path
# resolves from (recipes/configs/dev/production/bioreason_32b_lora_grpo_hsdp_xpu.yaml:33,
# which names the /tmp staging copy; this is the Lustre source that copy is made from).
export EPOCH=${EPOCH:-/lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/runs/sft_qwen3_32b_lora_r128_lrsched_v8_step1550_snapshot}

# Deliberately NOT exported / deliberately empty. See the header.
unset ADAPTER_DIR

# N goes in TAG for the same reason EVAL_K does in pbs_2n_eval_grpo_ckpt.sh: OUT is
# eval_out/$TAG, the draw is sample(random_state=7).head(N), and a larger N is a nested
# SUPERSET -- comparable, but a different protein population. An N=500 run written over
# the N=150 dir would destroy the 127-protein baseline the whole step-0/20/40 trend is
# differenced against (and it is the n=3 baseline, so there is no cheap way back).
# Conditional, so the existing grpo_step0_rep{1,2,3} paths keep their exact names.
_N_SUFFIX=""
[ "${N:-150}" != "150" ] && _N_SUFFIX="_n${N}"

# EPOCH is overridable but the default TAG does NOT encode which base was scored.
# A `EPOCH=<step1050> REP=5 N=500 qsub` would therefore write a step1050 reading
# into `grpo_step0_rep5_n500`, a name that reads as another step1550 step-0 rep,
# and it would be pooled with the n=4 step-0 baseline by anyone reading names
# rather than provenance. (fmax_endpoint_contrast.py's gate would catch the pool
# via coverage.txt's base_ckpt, but a name that lies is a trap regardless.)
# So: a non-default EPOCH must be accompanied by an explicit TAG.
_DEFAULT_EPOCH=/lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/runs/sft_qwen3_32b_lora_r128_lrsched_v8_step1550_snapshot
if [ "${EPOCH%/}" != "${_DEFAULT_EPOCH}" ] && [ -z "${TAG:-}" ]; then
    echo "FATAL: EPOCH overridden to ${EPOCH} but no TAG given."
    echo "       The default TAG (grpo_step0_rep${REP}${_N_SUFFIX}) does not name the base and"
    echo "       would be mistaken for a step1550 step-0 rep. Pass an explicit TAG, e.g."
    echo "       TAG=step0_base1050_rep${REP}${_N_SUFFIX}"
    exit 2
fi
export TAG="${TAG:-grpo_step0_rep${REP}${_N_SUFFIX}}"
export N=${N:-150}
export SEED=7
export PARITY=0
export MAX_NEW_TOKENS=5000
export TEST=/lus/flare/projects/ModCon/ngetty/datasets/bioreason_pro_test/test-00000-of-00001.parquet
export CACHE=/lus/flare/projects/ModCon/ngetty/datasets/bioreason_pro_test/esm3_cache_2048.pt
export OBO=/lus/flare/projects/ModCon/ngetty/BioReason-Pro/bioreason2/dataset/go-basic.obo
export IA=/lus/flare/projects/ModCon/ngetty/datasets/bioreason_pro_test/IA.txt
export RESUME=0
# MUST match training: the GRPO config sets add_uniprot_summary: true (inherited from the
# v8 SFT prompt distribution). Evaluating without it is a prompt-distribution mismatch and
# the number is not comparable to anything.
export ADD_UNIPROT_SUMMARY=1

export TILES_PER_SERVER=4
# 2 engines/node, NOT the 3/node default: 3/node is the confirmed banned:1 trigger
# (16/16 recorded crashes); 1/node and 2/node both ran clean 127/127 and faster
# (memory/project_banned1_engine_density_probe_20260905.md).
export NSERVERS_PER_NODE=2
export CLIENTS_PER_SERVER=3
export MAX_NUM_SEQS=16
export CONCURRENCY=12

[ -f "$EPOCH/config.json" ] || { echo "FATAL: base checkpoint missing: $EPOCH"; exit 2; }
[ -f "$CACHE" ]            || { echo "FATAL: layer-(-1) ESM3 cache missing: $CACHE"; exit 2; }

echo "=== GRPO step-0 baseline eval: rep=$REP tag=$TAG base=$EPOCH (NO adapter, by design) ==="
exec bash "$TT/experiments/bioreason/pbs_2n_eval_vllm_tp2.sh"

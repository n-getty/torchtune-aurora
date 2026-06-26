#!/bin/bash
# Self-chaining go_pred RL on debug-scaling (1h walltime cap). Each link runs ~CHAIN_STEPS
# steps of go_pred RL resuming from the latest saved adapter, saves every SAVE_EVERY steps,
# then submits the NEXT link (depend=afterany) resuming the adapter it just wrote — until
# the cumulative step budget CHAIN_TARGET is reached. This replaces the 12h capacity run
# (which never got scheduled) with a sequence of schedulable 1h jobs.
#
# Submit the FIRST link with:
#   qsub -v CHAIN_TARGET=120,CHAIN_STEPS=24,SAVE_EVERY=8,LINK=1 \
#        experiments/bioreason/chain_gopred_debugscaling.sh
# Each link auto-submits the next; the chain stops when cumulative steps >= CHAIN_TARGET.
#
# Resume semantics: the LoRA adapter (+ trained projections) carries the learning across
# links; the AdamW optimizer state resets each link (adapter-only resume). For GRPO with a
# near-on-policy update that's acceptable — the chain is ~equivalent to periodic opt restarts.
#
#PBS -l select=2
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -A ModCon
#PBS -q debug-scaling
#PBS -N br_gopred_chain
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/chain_gopred_debugscaling.out

set -o pipefail
TT=/lus/flare/projects/ModCon/ngetty/torchtune
cd "$TT"

# ---- chain params (env-injected via qsub -v) ----
CHAIN_TARGET=${CHAIN_TARGET:-120}   # cumulative go_pred RL steps to reach across all links
CHAIN_STEPS=${CHAIN_STEPS:-24}      # steps per 1h link (~10min boot + ~50min @ ~113s/step)
SAVE_EVERY=${SAVE_EVERY:-8}         # checkpoint cadence (a walltime kill loses <= SAVE_EVERY)
LINK=${LINK:-1}                     # 1-based link index
# Cumulative steps already completed BEFORE this link (set by the predecessor).
DONE_STEPS=${DONE_STEPS:-0}
# Shared output dir across the whole chain (adapter overwrites epoch_0 each save).
CHAIN_OUT=${CHAIN_OUT:-$TT/outputs/bioreason_gopred_chain_20260626}
CHAIN_STATE=$CHAIN_OUT/chain_state
mkdir -p "$CHAIN_STATE"

echo "=== go_pred CHAIN link=$LINK start $(date) job=${PBS_JOBID} ===" | tee -a "$CHAIN_STATE/chain.log"
echo "    DONE_STEPS=$DONE_STEPS CHAIN_STEPS=$CHAIN_STEPS TARGET=$CHAIN_TARGET out=$CHAIN_OUT" | tee -a "$CHAIN_STATE/chain.log"
echo "    nodes: $(cat $PBS_NODEFILE | sort -u | tr '\n' ' ')" | tee -a "$CHAIN_STATE/chain.log"

# ---- resume: find the latest adapter this chain has written (if any) ----
RESUME=""
if [ -f "$CHAIN_OUT/epoch_0/adapter/adapter_model.safetensors" ]; then
    RESUME="$CHAIN_OUT/epoch_0/adapter"
    echo "    RESUME from $RESUME" | tee -a "$CHAIN_STATE/chain.log"
else
    echo "    FRESH start (no prior adapter) — link 1 trains from the SFT base" | tee -a "$CHAIN_STATE/chain.log"
fi

# ---- training env (go_pred, raised seq budget; matches batch_prod_rl_4n_gopred.sh) ----
export ENABLE_LORA=1
export NSTEPS=$CHAIN_STEPS
# Match the VALIDATED 2N go_pred smoke envelope (G=8, bs=2) — NOT the 4N HSDP G=24 config.
export GRPO_SAMPLES=${GRPO_SAMPLES:-8}
export BATCH_SIZE=${BATCH_SIZE:-2}
export FORWARD_BATCH_SIZE=${FORWARD_BATCH_SIZE:-1}
export MAX_GEN_TOKENS=${MAX_GEN_TOKENS:-1024}
export REF_FORWARD_BATCH_SIZE=${REF_FORWARD_BATCH_SIZE:-16}
export TORCHTUNE_USE_CHUNKED_LOSS=${TORCHTUNE_USE_CHUNKED_LOSS:-0}
export TORCHTUNE_VLLM_STOP_TOKENS=${TORCHTUNE_VLLM_STOP_TOKENS:-1}
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-7168}
export MAX_SEQ_LEN=${MAX_SEQ_LEN:-6144}

# Use the 2N SERVER launcher (1 train node + 1 vLLM node, single replicate) — the path the
# go_pred smoke validated. The HSDP launcher needs >=3 nodes; this fits select=2 cleanly.
# The 2N server launcher only forwards EXTRA_OVERRIDES, so pass resume/save/output there.
RESUME_OVR=""
[ -n "$RESUME" ] && RESUME_OVR="lora_adapter_path=$RESUME"
export WSYNC_PATH=${WSYNC_PATH:-$CHAIN_OUT/wsync/weight_update.raw}
export EXTRA_OVERRIDES="dataset.inject_go_pred=true max_seq_len=${MAX_SEQ_LEN} vllm_max_model_len=${VLLM_MAX_MODEL_LEN} save_every_n_steps=${SAVE_EVERY} output_dir=${CHAIN_OUT} ${RESUME_OVR} ${EXTRA_OVERRIDES:-}"

t0=$(date +%s)
bash "$TT/experiments/bioreason/run_bioreason_2node_server.sh"
RC=$?
dt=$(( $(date +%s) - t0 ))
echo "=== link=$LINK train rc=$RC dt=${dt}s $(date) ===" | tee -a "$CHAIN_STATE/chain.log"

# ---- decide whether to chain the next link ----
# We can't know exactly how many steps completed before a walltime kill; estimate from the
# train log's last METRICS step= and add to DONE_STEPS. Conservative: if no steps detected,
# don't advance (avoids an infinite chain on a boot-failing link).
# The recipe DiskLogger writes "$CHAIN_OUT/logs/log_*.txt" with lines "Step N | ..."; the
# 2N server LAUNCHER log "run_bioreason_2node_*.log" mirrors them as "METRICS step=N".
# Read BOTH formats from BOTH sources so the chain advances regardless of which is present.
_max_step() { grep -hoE "Step [0-9]+ \||METRICS step=[0-9]+" "$@" 2>/dev/null | grep -oE "[0-9]+" | sort -n | tail -1; }
LAST_STEP=$(_max_step "$CHAIN_OUT"/logs/log_*.txt)
LAST_STEP=${LAST_STEP:-0}
if [ "$LAST_STEP" -eq 0 ]; then
    _ll=$(ls -t $TT/experiments/bioreason/run_bioreason_2node_*.log 2>/dev/null | head -1)
    [ -n "$_ll" ] && LAST_STEP=$(_max_step "$_ll")
    LAST_STEP=${LAST_STEP:-0}
fi
NEW_DONE=$(( DONE_STEPS + LAST_STEP ))
echo "    link=$LINK completed ~$LAST_STEP steps (cumulative ~$NEW_DONE / $CHAIN_TARGET)" | tee -a "$CHAIN_STATE/chain.log"

if [ "$NEW_DONE" -ge "$CHAIN_TARGET" ]; then
    echo "=== CHAIN COMPLETE: ~$NEW_DONE >= $CHAIN_TARGET steps. Final adapter: $CHAIN_OUT/epoch_0/adapter ===" | tee -a "$CHAIN_STATE/chain.log"
    echo "$NEW_DONE" > "$CHAIN_STATE/cumulative_steps"
    exit $RC
fi
if [ "$LAST_STEP" -eq 0 ]; then
    echo "=== link=$LINK made NO progress (boot/crash). NOT auto-chaining (avoid infinite loop). Inspect + resubmit manually. ===" | tee -a "$CHAIN_STATE/chain.log"
    exit $RC
fi

# Submit the next link, resuming the adapter this link wrote. depend=afterany so it queues
# now and starts whenever debug-scaling frees (max_run=1 means at most one of ours runs).
NEXT=$(( LINK + 1 ))
echo "$NEW_DONE" > "$CHAIN_STATE/cumulative_steps"
nextjob=$(qsub -W depend=afterany:${PBS_JOBID} \
     -v CHAIN_TARGET=$CHAIN_TARGET,CHAIN_STEPS=$CHAIN_STEPS,SAVE_EVERY=$SAVE_EVERY,LINK=$NEXT,DONE_STEPS=$NEW_DONE,CHAIN_OUT=$CHAIN_OUT,GRPO_SAMPLES=$GRPO_SAMPLES,BATCH_SIZE=$BATCH_SIZE,MAX_GEN_TOKENS=$MAX_GEN_TOKENS \
     "$TT/experiments/bioreason/chain_gopred_debugscaling.sh" 2>&1)
echo "=== submitted next link=$NEXT: $nextjob ===" | tee -a "$CHAIN_STATE/chain.log"
exit $RC

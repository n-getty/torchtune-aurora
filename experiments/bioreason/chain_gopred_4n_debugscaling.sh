#!/bin/bash
# Self-chaining go_pred RL on debug-scaling (1h walltime cap) — 4-NODE HSDP variant.
# 4 nodes = 3 training replicas (dp_replicate=3) + 1 shared vLLM node => 3x distinct
# prompts/step at ~the same per-step time as 2N (gen is on the shared vLLM node either way),
# so a given data exposure is reached in ~1/3 the steps. Uses the HSDP launcher (needs >=3
# nodes) + async (HW-validated 4N dp_replicate=3, 2026-06-23, ~30% step win, staleness=1).
# Each link runs ~CHAIN_STEPS steps resuming the latest adapter, saves every SAVE_EVERY,
# then submits the next link (depend=afterany) until cumulative >= CHAIN_TARGET.
#
# Submit the FIRST link with:
#   qsub -v CHAIN_TARGET=80,CHAIN_STEPS=18,SAVE_EVERY=6,LINK=1 \
#        experiments/bioreason/chain_gopred_4n_debugscaling.sh
#
# Resume: the LoRA adapter (+ trained projections) carries learning across links; AdamW
# optimizer state resets each link (adapter-only resume) — fine for near-on-policy GRPO.
#
#PBS -l select=4
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -A ModCon
#PBS -q debug-scaling
#PBS -N br_gopred_4n_chain
#PBS -j oe
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/chain_gopred_4n_debugscaling.out

set -o pipefail
TT=/lus/flare/projects/ModCon/ngetty/torchtune
cd "$TT"

# ---- chain params (env-injected via qsub -v) ----
CHAIN_TARGET=${CHAIN_TARGET:-80}    # cumulative steps; 4N's 3x distinct prompts means fewer
                                    # steps reach the same data exposure as the 2N target 120.
CHAIN_STEPS=${CHAIN_STEPS:-18}      # steps per 1h link (4N step ~similar to 2N; ~10min boot)
SAVE_EVERY=${SAVE_EVERY:-6}         # checkpoint cadence (a walltime kill loses <= SAVE_EVERY)
LINK=${LINK:-1}                     # 1-based link index
# Cumulative steps already completed BEFORE this link (set by the predecessor).
DONE_STEPS=${DONE_STEPS:-0}
# Shared output dir across the whole chain (adapter overwrites epoch_0 each save).
CHAIN_OUT=${CHAIN_OUT:-$TT/outputs/bioreason_gopred_4n_chain_20260626}
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
# fbs=2 (NOT 1): num_seqs=bs*G=16 -> 8 backward chunks instead of 16, halving the per-step
# FSDP allgather+reduce-scatter collective count. The smoke/link-1 used fbs=1 -> grpo=33.5s
# (~2x the 45-53s baseline's ~17s at fbs=4); fbs=2 recovers ~half of that. Conservative vs
# fbs=4 because the go_pred prompt (~3400 tok vs baseline ~2250) raises backward activation
# memory; link-1 peaked at only 56/64 GiB at fbs=1 so fbs=2 has headroom. If a link OOMs,
# the chain's 0-progress guard halts it -> drop back to fbs=1.
export FORWARD_BATCH_SIZE=${FORWARD_BATCH_SIZE:-2}
export MAX_GEN_TOKENS=${MAX_GEN_TOKENS:-1024}
export REF_FORWARD_BATCH_SIZE=${REF_FORWARD_BATCH_SIZE:-16}
export TORCHTUNE_USE_CHUNKED_LOSS=${TORCHTUNE_USE_CHUNKED_LOSS:-0}
export TORCHTUNE_VLLM_STOP_TOKENS=${TORCHTUNE_VLLM_STOP_TOKENS:-1}

# ── PROMPT-LENGTH lever (the real gen cost driver) ──────────────────────────────
# The 57.5s vLLM gen is dominated by PREFILL of the protein placeholders (len(seq)+2 per
# protein). max_protein_len 2048 -> 128 cuts prompt ~3480 -> ~1560 tok (~halves prefill).
# JUSTIFIED: the fidelity A/B (memory project_bioreason_replication_gaps) showed protein
# 128 vs 2048 is F_max-NEUTRAL (0.4303 vs 0.4170) — the model predicts GO from the TEXT
# context (interpro/ppi/go_pred), not the long protein embedding. The 128 ESM3 cache is
# already built (datasets/bioreason_rl/esm3_cache.pt). go_pred TEXT (p95 ~1960 tok) is NOT
# truncated: prompt ~= 130(prot)+200(go)+1960(text)+80 ~= 2370 -> max_seq_len=4096 safe.
export MAX_PROTEIN_LEN=${MAX_PROTEIN_LEN:-128}
export ESM3_CACHE_PATH=${ESM3_CACHE_PATH:-/lus/flare/projects/ModCon/ngetty/datasets/bioreason_rl/esm3_cache.pt}
# max_seq_len=4608: measured 0/800 rows truncate at protein=128 (max untruncated prompt 4356;
# the go_pred TEXT tail, not protein, sets this). vs the old 6144 (protein=2048). vLLM ctx =
# 4608 prompt + 1024 gen = 5632. Still ~halves prefill vs the old protein-2048/6144 envelope.
export MAX_SEQ_LEN=${MAX_SEQ_LEN:-4608}
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-5632}

# ── ENGINE UTILIZATION lever ────────────────────────────────────────────────────
# Default seqs_per_engine=4 fans bsz=16 over only 4 of the 12 vLLM tiles (8 idle). =2 ->
# 8 engines (2 seqs each), spreading the decode across more tiles. (=1 -> 12 engines but
# 16/12 is uneven.) Cuts the throughput-bound part of gen; the longest single seq still
# sets a floor.
export TORCHTUNE_VLLM_SEQS_PER_ENGINE=${TORCHTUNE_VLLM_SEQS_PER_ENGINE:-2}

# ── VARLEN (no-grad ref/rollout forwards) ───────────────────────────────────────
# HONEST SCOPE: BioReason's backbone is HF AutoModelForCausalLM(attn_implementation=sdpa),
# so torchtune's attention_utils varlen only engages if HF SDPA routes through it; and even
# then it speeds only OUR HF forwards (ref_fwd ~9s + grpo policy fwd), NOT the 57.5s vLLM
# gen (vLLM has its own kernels). NOGRAD_BYPASS is required for var-len prompts (CLAUDE.md).
# Low-risk to set; verify 'varlen=engaged' in the log — if it stays 'requested-but-skipped'
# the HF backbone isn't wired to it (a known gap) and this is a harmless no-op.
export TORCHTUNE_USE_IPEX_VARLEN=${TORCHTUNE_USE_IPEX_VARLEN:-1}
export TORCHTUNE_VARLEN_NOGRAD_BYPASS=${TORCHTUNE_VARLEN_NOGRAD_BYPASS:-1}

# ASYNC generation (default ON): overlaps the ~67s vLLM gen behind the backward pass, so the
# step time floor is ~max(gen,train) instead of gen+train. HW-validated 2026-06-23 (~30% win,
# staleness=1, ratios~1.0, IS-corrected via GRPOLoss). The async YAML sets loss=GRPOLoss +
# always_compute_rollout_logprobs=true (the required combo). Set USE_ASYNC=0 for the sync path.
USE_ASYNC=${USE_ASYNC:-1}
if [ "$USE_ASYNC" = "1" ]; then
    export CONFIG=${CONFIG:-recipes/configs/dev/production/bioreason_4b_lora_grpo_2node_server_xpu_async.yaml}
fi

# Use the HSDP launcher (4 nodes -> 3 train replicas dp_replicate=3 + 1 vLLM node). It reads
# RESUME_ADAPTER / OUTPUT_DIR / SAVE_EVERY_N_STEPS as ENV (not EXTRA_OVERRIDES) and forwards
# EXTRA_OVERRIDES for the rest. dp_replicate is derived from node count by the launcher.
export RESUME_ADAPTER="$RESUME"
export OUTPUT_DIR="$CHAIN_OUT"
export SAVE_EVERY_N_STEPS="$SAVE_EVERY"
export WSYNC_PATH=${WSYNC_PATH:-$CHAIN_OUT/wsync/weight_update.raw}
# CONSTANT LR (lr_scheduler=null): the config's cosine warmup (num_warmup_steps=50) ramps lr
# from ~6e-8 over 50 steps, but each chain link is ~18 steps and resets the scheduler — so a
# warmup-per-link would keep lr near-zero forever and the chain would barely train. Constant
# lr=3e-6 (the config's peak) gives every link the full learning rate. lr_scheduler=null is
# tolerated by the recipe (see feedback_grpo_step_based_resume).
# esm3_cache_path and dataset.max_protein_len MUST agree (cache keyed by truncated seq). 128.
# vllm_max_model_len is passed in EXTRA_OVERRIDES because the HSDP launcher only sends
# --max-model-len to the vLLM SERVER, not to the recipe config (recipe-side prompt truncation).
export EXTRA_OVERRIDES="dataset.inject_go_pred=true dataset.max_protein_len=${MAX_PROTEIN_LEN} esm3_cache_path=${ESM3_CACHE_PATH} max_seq_len=${MAX_SEQ_LEN} vllm_max_model_len=${VLLM_MAX_MODEL_LEN} lr_scheduler=null ${EXTRA_OVERRIDES:-}"

# Retry-once-on-boot-flake: the Aurora vLLM EngineCore boot flake ("WorkerProc init failed",
# "Failed core proc(s): {}") fails FAST (<BOOT_WINDOW, before any training step) and clears on
# a relaunch. A real failure is slow (after boot). So if the launcher exits non-zero in under
# BOOT_WINDOW with 0 steps trained, retry once in-job (same nodes; the flake is per-spawn).
BOOT_WINDOW=${BOOT_WINDOW:-600}
t0=$(date +%s)
bash "$TT/experiments/bioreason/run_bioreason_Nnode_hsdp.sh"
RC=$?
dt=$(( $(date +%s) - t0 ))
if [ $RC -ne 0 ] && [ $dt -lt $BOOT_WINDOW ]; then
    echo "=== link=$LINK FAST-FAIL rc=$RC dt=${dt}s (<${BOOT_WINDOW}s) — likely vLLM boot flake, retrying once ===" | tee -a "$CHAIN_STATE/chain.log"
    sleep 20
    t0=$(date +%s)
    bash "$TT/experiments/bioreason/run_bioreason_Nnode_hsdp.sh"
    RC=$?
    dt=$(( $(date +%s) - t0 ))
fi
echo "=== link=$LINK train rc=$RC dt=${dt}s $(date) ===" | tee -a "$CHAIN_STATE/chain.log"

# ---- decide whether to chain the next link ----
# We can't know exactly how many steps completed before a walltime kill; estimate from the
# train log's last METRICS step= and add to DONE_STEPS. Conservative: if no steps detected,
# don't advance (avoids an infinite chain on a boot-failing link).
# The recipe DiskLogger writes "$CHAIN_OUT/logs/log_*.txt" with lines "Step N | ..."; the
# HSDP LAUNCHER log "run_bioreason_Nnode_*.log" mirrors them as "METRICS step=N".
# Read BOTH formats from BOTH sources so the chain advances regardless of which is present.
_max_step() { grep -hoE "Step [0-9]+ \||METRICS step=[0-9]+" "$@" 2>/dev/null | grep -oE "[0-9]+" | sort -n | tail -1; }
LAST_STEP=$(_max_step "$CHAIN_OUT"/logs/log_*.txt)
LAST_STEP=${LAST_STEP:-0}
if [ "$LAST_STEP" -eq 0 ]; then
    _ll=$(ls -t $TT/experiments/bioreason/run_bioreason_Nnode_*.log 2>/dev/null | head -1)
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
     -v CHAIN_TARGET=$CHAIN_TARGET,CHAIN_STEPS=$CHAIN_STEPS,SAVE_EVERY=$SAVE_EVERY,LINK=$NEXT,DONE_STEPS=$NEW_DONE,CHAIN_OUT=$CHAIN_OUT,GRPO_SAMPLES=$GRPO_SAMPLES,BATCH_SIZE=$BATCH_SIZE,FORWARD_BATCH_SIZE=$FORWARD_BATCH_SIZE,MAX_GEN_TOKENS=$MAX_GEN_TOKENS,USE_ASYNC=$USE_ASYNC \
     "$TT/experiments/bioreason/chain_gopred_4n_debugscaling.sh" 2>&1)
echo "=== submitted next link=$NEXT: $nextjob ===" | tee -a "$CHAIN_STATE/chain.log"
exit $RC

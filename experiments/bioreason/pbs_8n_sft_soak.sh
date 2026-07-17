#!/bin/bash
#PBS -N br_sft_8n_soak
#PBS -A ModCon
#PBS -q capacity
#PBS -l select=8
#PBS -l walltime=06:00:00
#PBS -l filesystems=home:flare
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/logs/pbs_8n_sft_soak.out
#PBS -e /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/logs/pbs_8n_sft_soak.err
#
# BioReason Qwen3-32B SFT — 8-NODE HSDP CAPABILITY SOAK (checkpoint-resume, 6h segments).
# =====================================================================================
# PURPOSE: the multi-hundred-step endurance run that the 15-step prod soak (job prodsoak_174804,
# 2026-07-15) could not cover. Closes the two gates before a full production run:
#   (1) no banned:1 / OOM / compile-recompile churn over hundreds of steps (was 15-step-clean),
#   (2) produce a real multi-epoch checkpoint worth an F_max eval (the accelerated stack's
#       bucketbs sample-ordering + compile numerics have NOT been eval-validated yet).
#
# ★ CORRECTNESS: points at the stage1norm config (NOT sft_bioreason_qwen3_32B_xpu.yaml, which is
#   the legacy Stage-2/LoRA config that CAUSED the regression). stage1norm =
#     - enable_lora=false + freeze_backbone=true  (LoRA breaks generation from spliced embeds:
#       ':' collapse — project_bioreason_sft_lora_splice_collapse_20260630)
#     - layer-37 ESM3 cache                        (final-layer norm ~10600 swamps text -> ':';
#       feedback_bioreason_esm3_layer37_scale_fix)
#     - projector_output_norm=true                 (LayerNorm bounds proj norm ~sqrt(H)~71, kills
#       the 721->1629 over-amplify trap; over-train safe for a LONG run)
#   Last smoke on this config: 0/7 ':' collapse, 7/7 real protein-conditioned text.
#
# ★ ACCELERATION (all HW-validated 2026-07-15, ~1.6x compounded, config-default in stage1norm):
#     - TORCHTUNE_USE_XPU_FLASH=1  native SYCL-TLA fused flash (10%->~20% MFU, ~2x)
#     - compile.model=true         per-layer backbone compile (+5.8% seq6144)
#     - pad_buckets=[4096,6144] bucket_batch_sizes=[2,1]  (~1.49x throughput, plain-causal, NO flex)
#   These live in the config; the launcher does NOT re-inject them (avoids config-drift). It only
#   sets the flash env (read at import) + the distributed topology + resume/checkpoint cadence.
#
# ★ TRANSPORT = ofi + pmi-KVS (NOT mpi): the 32B HSDP per-layer intra-node param all-gather is
#   ~2.3x faster under ofi (A/B job 8673012). Recovers the entire 1N->2N tax; 2N-ofi ~= 1N. ofi
#   REQUIRES CCL_KVS_MODE=pmi. See project_bioreason_sft_hsdp_2n_forward_allgather_tax_20260714.
#
# SEGMENT MATH (8N; HSDP = dp_shard=12 within node, dp_replicate=8 across nodes = 8 DATA-parallel
#   replicas). tqdm "it" is an OPT-STEP (=ga microbatches; recipe pbar total=len(dl)//ga). At
#   seq6144 the opt-step is ~78s STEADY (per-mb fwd 4.6 + bwd 14.1 ~18.7s x ga=4 + optim; measured
#   prod soak 07-15 + 2N-ofi transport A/B). Bucketbs makes the effective rate better: 79% of the
#   corpus trains in the 4096 bucket at bs2 (~10.7s/sample), only the 21% tail at seq6144 bs1.
#   Opt-steps/epoch: validated 4N reference = ~609/epoch at dp=48 -> ~305/epoch at 8N dp=96.
#   => ~305 x 78s ~= 6.6h/epoch — SLIGHTLY OVER a 6h walltime. So a 6h segment gets ~0.9 epoch;
#   checkpoint-resume (BIOREASON_RESUME=1) carries the remainder into segment 2. NOT one-epoch-
#   per-segment. Use walltime=06:00:00 + SAVE_EVERY=50 so a segment always lands >=1 checkpoint.
#   (Confirm the real opt-steps/epoch from segment-1's tqdm total before planning epoch count —
#    the 305 is derived, not yet measured at 8N with the bucket sampler.)
#   Milestone gate: FIRST beat the 0.30 echo-go_pred floor (proof of protein denoising), likely
#   within ~1 epoch; then push epochs toward the 4B-pro ~0.66 F_max (multi-epoch = several segments).
#
# USAGE:
#   qsub experiments/bioreason/pbs_8n_sft_soak.sh                 # segment 1 (fresh)
#   BIOREASON_RESUME=1 qsub -v BIOREASON_RESUME experiments/bioreason/pbs_8n_sft_soak.sh   # seg 2+
#   (override EPOCHS / SAVE_EVERY / MAX_STEPS via qsub -v as needed)
#
# Recipe:  recipes/dev/sft_bioreason_distributed_xpu.py
# Config:  recipes/configs/dev/production/sft_bioreason_qwen3_32B_stage1norm_xpu.yaml
# Wrapper: experiments/bioreason/_sft_rank_wrapper.sh
set -eo pipefail
PROJDIR=/lus/flare/projects/ModCon/ngetty/torchtune

module load frameworks/2025.3.1 2>/dev/null || module load frameworks 2>/dev/null || true
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_DATASETS_CACHE=/lus/flare/projects/ModCon/ngetty/hf_datasets_cache
export HF_HOME=/lus/flare/projects/ModCon/ngetty/hf_cache
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ftp_proxy; export no_proxy="*" NO_PROXY="*"
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export PYTHONUNBUFFERED=1

# Production multi-node CCL row. gc:0.99 + userfaultfd + default alloc = validated.
export CCL_PROCESS_LAUNCHER=pmix CCL_ATL_TRANSPORT=${CCL_ATL_TRANSPORT:-ofi} CCL_KVS_MODE=${CCL_KVS_MODE:-pmi} CCL_KVS_USE_MPI_RANKS=${CCL_KVS_USE_MPI_RANKS:-0}
export CCL_CONFIGURATION=cpu_gpu_dpcpp CCL_KVS_CONNECTION_TIMEOUT=600 CCL_OP_SYNC=1
export CCL_WORKER_COUNT=1 CCL_ALLREDUCE=ring CCL_CHUNK_SIZE=16777216
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=65536 FI_PROVIDER=cxi GLOO_SOCKET_IFNAME=hsn0
unset PYTORCH_ALLOC_CONF XPU_USM_ALLOC_SO
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.99
export FI_MR_CACHE_MONITOR=userfaultfd

# Native SYCL-TLA fused flash attention (config-default stack; read at import, propagates via pmix).
# Override TORCHTUNE_USE_XPU_FLASH=0 only to A/B the old math path.
export TORCHTUNE_USE_XPU_FLASH=${TORCHTUNE_USE_XPU_FLASH:-1}
export TORCHTUNE_USE_XPU_FLEX=${TORCHTUNE_USE_XPU_FLEX:-0}   # bucketbs is plain-causal; NO flex
export TORCHTUNE_BIOREASON_TIMING=${TORCHTUNE_BIOREASON_TIMING:-1}

NPROC=${NPROC:-12}
RECIPE="${PROJDIR}/recipes/dev/sft_bioreason_distributed_xpu.py"
CONFIG=${CONFIG:-recipes/configs/dev/production/sft_bioreason_qwen3_32B_stage1norm_xpu.yaml}
WRAPPER="${PROJDIR}/experiments/bioreason/_sft_rank_wrapper.sh"; chmod +x "${WRAPPER}"

# Persistent output dir across segments (checkpoints + resume_state live here).
OUTDIR=${OUTDIR:-/lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/runs/sft_qwen3_32b_8n_soak}
SAVE_EVERY=${SAVE_EVERY:-50}              # checkpoint cadence (opt-steps) — ~3x within a 6h/epoch segment
EPOCHS=${EPOCHS:-1}                        # 1 epoch/segment; bump + resume for multi-epoch
BIOREASON_RESUME=${BIOREASON_RESUME:-0}   # set 1 for segments after the first
mkdir -p "${OUTDIR}"

# Node discovery + explicit deduped hostfile (raw $PBS_NODEFILE miscounts WORLD under PALS).
UNIQUE_NODES=($(awk '!seen[$0]++' "${PBS_NODEFILE}"))
NNODES=${#UNIQUE_NODES[@]}
export WORLD=$((NPROC*NNODES))
DP_REPLICATE=${DP_REPLICATE:-${NNODES}}    # one shard group (12 tiles) per node
HOSTFILE="${OUTDIR}/hostfile_$(echo ${PBS_JOBID:-$$} | tr -dc 0-9 | tail -c6).txt"
: > "${HOSTFILE}"; for n in "${UNIQUE_NODES[@]}"; do echo "${n}" >> "${HOSTFILE}"; done

NODE0=$(echo "${UNIQUE_NODES[0]}" | cut -d. -f1)
NODE0_ADDR=$(getent hosts "${NODE0}" | awk 'NR==1 {print $1}'); [ -z "${NODE0_ADDR}" ] && NODE0_ADDR="${NODE0}"
LAST4=$(echo "${PBS_JOBID:-$$}" | tr -dc '0-9' | tail -c 4)
export MASTER_ADDR="${NODE0_ADDR}"; export MASTER_PORT=$(( 29500 + (10#${LAST4:-0} % 400) ))

echo "=== BioReason Qwen3-32B SFT ${NNODES}N HSDP SOAK | resume=${BIOREASON_RESUME} epochs=${EPOCHS} save_every=${SAVE_EVERY} ==="
echo "    config=${CONFIG}"
echo "    NNODES=${NNODES} WORLD=${WORLD} DP_REPLICATE=${DP_REPLICATE} OUTDIR=${OUTDIR} MASTER=${MASTER_ADDR}:${MASTER_PORT}"
echo "    flash=${TORCHTUNE_USE_XPU_FLASH} flex=${TORCHTUNE_USE_XPU_FLEX} transport=${CCL_ATL_TRANSPORT} kvs=${CCL_KVS_MODE}"
echo "    hostfile:"; cat "${HOSTFILE}"
cd "${PROJDIR}"

SEGLOG="${OUTDIR}/segment_$(date +%Y%m%d_%H%M%S).log"
mpiexec --pmi=pmix --hostfile "${HOSTFILE}" -n ${WORLD} -ppn ${NPROC} \
    --cpu-bind depth --depth 8 \
    --env WORLD=${WORLD} --env MASTER_ADDR=${MASTER_ADDR} --env MASTER_PORT=${MASTER_PORT} \
    bash "${WRAPPER}" "${RECIPE}" --config "${CONFIG}" \
        output_dir="${OUTDIR}" \
        epochs=${EPOCHS} \
        data_parallel_shard_dim=${NPROC} data_parallel_replicate_dim=${DP_REPLICATE} \
        save_every_n_steps=${SAVE_EVERY} \
        ${MAX_STEPS:+max_steps_per_epoch=${MAX_STEPS}} \
        bioreason_resume=${BIOREASON_RESUME} \
    2>&1 | tee -a "${SEGLOG}"
rc=${PIPESTATUS[0]}

# Run-health gate: flag a degraded segment (banned:1, OOM, recompile churn) before trusting it.
echo "=== 8N soak segment exit rc=${rc} $(date) ==="
echo "--- engage (flash + bucket sampler + compile), first ranks ---"
grep -hE "xpu_flash=|Per-bucket batch sizing ENABLED|Compiling backbone" "${SEGLOG}" 2>/dev/null | head -4 || true
echo "--- step times (steady after compile amortizes in step 1) ---"
grep -hoE "[0-9.]+s/it\]" "${SEGLOG}" 2>/dev/null | tail -6 || true
echo "--- loss (monotonic?) ---"
grep -hoE "Loss: [0-9.]+" "${SEGLOG}" 2>/dev/null | awk '!seen[$0]++' | tail -8 || true
echo "--- banned/OOM/recompile scan (expect NONE) ---"
grep -hE "banned|OutOfMemory|PDE|Recompil|recompil|cache_size_limit|Traceback" "${SEGLOG}" 2>/dev/null | head -5 || echo "NONE"
echo "=== resume next epoch/segment with: BIOREASON_RESUME=1 qsub -v BIOREASON_RESUME ${BASH_SOURCE##*/} ==="
exit ${rc}

#!/bin/bash
#PBS -N agpt2b_7n_hsdp
#PBS -A ModCon
#PBS -q debug-scaling
#PBS -l select=8
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -o /lus/flare/projects/ModCon/ngetty/torchtune/experiments/auroragpt_2b_bakeoff/logs/run_agpt2b_7n_hsdp_8node.out
#PBS -e /lus/flare/projects/ModCon/ngetty/torchtune/experiments/auroragpt_2b_bakeoff/logs/run_agpt2b_7n_hsdp_8node.err
#
# AGPT-2B GRPO - 7-replica HSDP across 8 nodes (debug-scaling).
#
# TOPOLOGY:
#   Nodes 0-6 -> 7 training replicas: dp_replicate=7 x dp_shard=12 (world=84).
#                Dense llama3 FSDP1 HYBRID_SHARD within each node; gradients
#                all-reduced across the 7 replicas (native HYBRID_SHARD).
#                Launched via mpiexec --pmi=pmix (production multi-node row of
#                the CLAUDE.md decision table).
#   Node 7    -> 1 dedicated vLLM pool: 12 HTTP servers (tiles 0-11), SHARED by
#                all 7 replicas. Each replica's shard-leader POSTs its DISTINCT
#                prompt slice and broadcasts node-locally over the gloo dp_shard PG.
#
# Config:  recipes/configs/dev/production/auroragpt_2b_grpo_7n_gsm8k_hsdp_xpu.yaml
# Recipe:  recipes/dev/grpo_full_finetune_distributed_xpu.py  (base)
# Wrapper: experiments/auroragpt_2b_bakeoff/_agpt2b_7n_train_rank_wrapper.sh
#
# v1 SCOPE: weight sync OFF (frozen policy). This run validates the data-parallel
# rollout + cross-replica grad-allreduce topology. Smoke first (NSTEPS=5), then
# scale NSTEPS for the full run.
#
# Useful overrides:
#   NSTEPS=5            VLLM_ONLY=1 (validate vLLM startup, skip training)
#   BATCH_SIZE=4        GRPO_SAMPLES=16   MAX_GEN_TOKENS=512
#   VLLM_MAX_NUM_SEQS=64

set -eo pipefail

PROJDIR=/lus/flare/projects/ModCon/ngetty/torchtune
cd "${PROJDIR}"

TS=$(date +%Y%m%d_%H%M%S)
LOGDIR=${PROJDIR}/experiments/auroragpt_2b_bakeoff/logs/agpt2b_7n_hsdp_${TS}
mkdir -p "${LOGDIR}"
LOG=${LOGDIR}/launcher.log

# --- Model + config -------------------------------------------------------
MODEL_PATH=${MODEL_PATH:-/lus/flare/projects/ModCon/ngetty/torchtune/experiments/agpt2b_sft/logs/mathmix_2n_full_20260615_043457/run_out/epoch_2}
CONFIG=${CONFIG:-${PROJDIR}/recipes/configs/dev/production/auroragpt_2b_grpo_7n_gsm8k_hsdp_xpu.yaml}

# --- RL envelope (defaults MUST match the YAML) ---------------------------
NSTEPS=${NSTEPS:-5}
# Periodic step checkpointing → resumable recipe_state every N steps (and at the
# final step). Matches the YAML default; set =0/empty to disable.
SAVE_EVERY_N_STEPS=${SAVE_EVERY_N_STEPS:-50}
# Resume: set RESUME_FROM=<output_dir>/epoch_<N> (the dir holding the saved policy
# weights + recipe_state.pt) to continue a finished/interrupted run. Empty =
# fresh run (default, identical to before). When set, the launcher injects
# resume_from_checkpoint=true + the checkpointer paths and uses RESUME_FROM as the
# base_model_path so training continues from those weights.
RESUME_FROM=${RESUME_FROM:-}
# Per-replica prompts/step. batch_size=4 (64 seqs/rank at G=16) made torch_resv
# grow ~5 GiB/step from fragmentation → tile-pinned banned:1 at step ~11 (job
# 8544991 mem-probe). 2 halves the per-rank peak; global distinct prompts/step =
# 2 × dp_replicate (= 14 at 7 replicas) — still well above the old DP=1 regime.
BATCH_SIZE=${BATCH_SIZE:-2}
# G=16 + batch_size=2/4 fragments PyTorch reserved to the tile ceiling on the
# UNLUCKY replica (data-dependent: long-completion replicas reserve 62 GiB while
# others sit at 47 — any one pinning kills the job). G=8 halves the per-prompt
# rollout/logprob buffer volume that fragments. BioReason run 41 precedent:
# G=8/fbs=4 is the VALIDATED-clean envelope (job 8545037, 40 steps, ~30 GiB resv).
# G=16 (with bs=2 → 32 seqs/rank) re-triggers per-replica fragmentation → banned:1
# at step ~12 (job 8545177). Per-rank buffer volume = bs × G; keep it at 16 seqs/rank.
# The 14-distinct-prompts/step data-parallelism win is from bs×replicas, NOT G.
GRPO_SAMPLES=${GRPO_SAMPLES:-8}
FORWARD_BATCH_SIZE=${FORWARD_BATCH_SIZE:-4}
REF_FORWARD_BATCH_SIZE=${REF_FORWARD_BATCH_SIZE:-4}
MAX_GEN_TOKENS=${MAX_GEN_TOKENS:-512}

# --- vLLM topology --------------------------------------------------------
VLLM_DP=${VLLM_DP:-12}
VLLM_BASE_PORT=${VLLM_BASE_PORT:-8001}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-1024}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-64}
VLLM_GPU_MEM=${VLLM_GPU_MEM:-0.70}
VLLM_STARTUP_TIMEOUT=${VLLM_STARTUP_TIMEOUT:-1800}

# --- Train tiles per node (FULL node = 12; dp_shard must equal this) -------
NPROC=${NPROC:-12}

# --- Fast-path env (match 2N production) -----------------------------------
export TORCHTUNE_USE_CHUNKED_LOSS=${TORCHTUNE_USE_CHUNKED_LOSS:-0}
export TORCHTUNE_PINNED_CPU_BUF=${TORCHTUNE_PINNED_CPU_BUF:-1}
export TORCHTUNE_USE_IPEX_VARLEN=${TORCHTUNE_USE_IPEX_VARLEN:-1}
export TORCHTUNE_VARLEN_NOGRAD_BYPASS=${TORCHTUNE_VARLEN_NOGRAD_BYPASS:-0}

# --- Weight sync (XCCL FSDP1 server path; rank 0 → shared 12-server pool) ---
# Mirrors the 2N-validated transport: gloo cross-node (CXI-MR-leak-safe), xccl
# intra. Only rank 0 gathers + broadcasts; the path applies the Llama Q/K
# un-permute (model_type LLAMA3). interval=1 = every step (on-policy, ratios≈1.0).
export WSYNC_CROSS_METHOD=${WSYNC_CROSS_METHOD:-gloo}
export WSYNC_INTRA_METHOD=${WSYNC_INTRA_METHOD:-xccl}
export VLLM_WSYNC_INTERVAL=${VLLM_WSYNC_INTERVAL:-1}

# ============================================================
# Node discovery: (N-1) train + 1 vLLM, derived from the nodefile.
# Works for any allocation >= 3 nodes: 4-node -> 3 train (dp_replicate=3) + 1 vLLM;
# 8-node -> 7 train (dp_replicate=7) + 1 vLLM. dp_replicate = #train nodes; one
# node = one shard group of NPROC ranks.
# ============================================================
if [[ -z "${PBS_NODEFILE:-}" ]]; then
    echo "ERROR: PBS_NODEFILE not set. Run from a held PBS job." | tee -a "${LOG}"
    exit 1
fi
# Keep FULL FQDN hostnames (e.g. xNNNN....hsn.cm.aurora.alcf.anl.gov). mpiexec
# --pmi=pmix / PALS RPC-launch FAILS with bare short names ("Couldn't send RPC
# launch ... Resource temporarily unavailable"); it needs FQDNs in the hostfile.
# SSH, hostname -i, and curl all work with the FQDN too. See
# memory/feedback_pbs_mpiexec_use_pbs_nodefile.md.
UNIQUE_NODES=($(awk '!seen[$0]++' "${PBS_NODEFILE}"))
NTOTAL=${#UNIQUE_NODES[@]}
if [ "${NTOTAL}" -lt 3 ]; then
    echo "ERROR: Need >=3 nodes (>=2 train + 1 vLLM). Got ${NTOTAL}: ${UNIQUE_NODES[*]}" | tee -a "${LOG}"
    exit 1
fi
NTRAIN_NODES=$((NTOTAL - 1))
DP_REPLICATE=${DP_REPLICATE:-${NTRAIN_NODES}}
WORLD=$((NPROC * NTRAIN_NODES))
TRAIN_NODES=("${UNIQUE_NODES[@]:0:${NTRAIN_NODES}}")
VLLM_NODE="${UNIQUE_NODES[${NTRAIN_NODES}]}"
if [ "${DP_REPLICATE}" -ne "${NTRAIN_NODES}" ]; then
    echo "ERROR: DP_REPLICATE=${DP_REPLICATE} must equal #train nodes=${NTRAIN_NODES}" | tee -a "${LOG}"
    exit 1
fi

VLLM_NODE_IP=$(ssh "${VLLM_NODE}" "hostname -i" 2>/dev/null | awk '{print $1}')
if [[ -z "${VLLM_NODE_IP}" ]]; then
    echo "ERROR: hostname -i failed on vLLM node ${VLLM_NODE}" | tee -a "${LOG}"
    exit 1
fi

echo "=== AGPT-2B 7N HSDP (8-node) ===" | tee -a "${LOG}"
echo "  TS=${TS}  LOGDIR=${LOGDIR}" | tee -a "${LOG}"
echo "  Train nodes (7): ${TRAIN_NODES[*]}" | tee -a "${LOG}"
echo "  vLLM node:       ${VLLM_NODE} (${VLLM_NODE_IP})" | tee -a "${LOG}"
echo "  CONFIG=${CONFIG}" | tee -a "${LOG}"
echo "  MODEL=${MODEL_PATH}" | tee -a "${LOG}"
echo "  world=${WORLD} (dp_replicate=${DP_REPLICATE} x dp_shard=${NPROC})" | tee -a "${LOG}"
echo "  batch_size=${BATCH_SIZE} (global distinct prompts/step = ${BATCH_SIZE}x${DP_REPLICATE}=$((BATCH_SIZE*DP_REPLICATE)))" | tee -a "${LOG}"
echo "  G=${GRPO_SAMPLES} fbs=${FORWARD_BATCH_SIZE} ref_fbs=${REF_FORWARD_BATCH_SIZE} max_gen=${MAX_GEN_TOKENS} steps=${NSTEPS}" | tee -a "${LOG}"
echo "  VLLM_DP=${VLLM_DP} max_num_seqs=${VLLM_MAX_NUM_SEQS} gpu_mem=${VLLM_GPU_MEM}" | tee -a "${LOG}"

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ftp_proxy
export no_proxy="*"
export NO_PROXY="*"

# ============================================================
# Verify model exists on vLLM node + all train nodes
# ============================================================
for node in "${VLLM_NODE}" "${TRAIN_NODES[@]}"; do
    if ! ssh "${node}" "test -f '${MODEL_PATH}/config.json' || test -f '${MODEL_PATH}/model.safetensors'" 2>/dev/null; then
        echo "ERROR: model not found at ${MODEL_PATH} on ${node}" | tee -a "${LOG}"
        exit 1
    fi
done
echo "Model verified on all nodes." | tee -a "${LOG}"

# ============================================================
# Stage model to /tmp on the vLLM node (avoid 12-way Lustre reads)
# ============================================================
VLLM_MODEL_LOCAL="/tmp/models/$(basename "${MODEL_PATH}")"
echo "Staging model to ${VLLM_NODE}:${VLLM_MODEL_LOCAL} ..." | tee -a "${LOG}"
ssh "${VLLM_NODE}" "
mkdir -p /tmp/models
if [ -f '${VLLM_MODEL_LOCAL}/config.json' ] || [ -f '${VLLM_MODEL_LOCAL}/model.safetensors' ]; then
    echo '  Already staged.'
else
    echo '  Copying from Lustre to /tmp ...'
    cp -r '${MODEL_PATH}' /tmp/models/
    echo '  Done.'
fi
" 2>&1 | tee -a "${LOG}"

# ============================================================
# PYTHONPATH (train + vLLM)
# ============================================================
source recipes/dev/_aurora_paths.sh
VLLM_CUSTOMIZATION="${PROJDIR}/recipes/dev/_usercustomize_vllm"
VLLM_PYTHONPATH="$(aurora_pythonpath "${PROJDIR}" "${VLLM_CUSTOMIZATION}")"
_AURORA_USER_SITE="/home/${USER}/.local/aurora/frameworks/2025.3.1/lib/python3.12/site-packages"
TRAIN_PYTHONPATH="$(aurora_pythonpath "${PROJDIR}"):${_AURORA_USER_SITE}"

# ============================================================
# Pre-launch cleanup on vLLM node
# ============================================================
echo "Cleaning stale vLLM on ${VLLM_NODE}..." | tee -a "${LOG}"
ssh "${VLLM_NODE}" "
pkill -9 -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true
pkill -9 -f 'vllm.v1.engine' 2>/dev/null || true
pkill -9 VLLM 2>/dev/null || true
sleep 2
bash ${PROJDIR}/recipes/dev/clean_tiles.sh --kill 2>&1 | tail -10 || true
sleep 3
mkdir -p /tmp/torchtune
" 2>&1 | tee -a "${LOG}" || true

# ============================================================
# Build the shared 12-URL pool (all train shard-leaders use this)
# ============================================================
VLLM_URLS=""
for ((r=0; r<VLLM_DP; r++)); do
    PORT=$((VLLM_BASE_PORT + r))
    VLLM_URLS="${VLLM_URLS:+${VLLM_URLS},}http://${VLLM_NODE_IP}:${PORT}"
done
echo "vLLM URLs: ${VLLM_URLS}" | tee -a "${LOG}"

# ============================================================
# Launch ${VLLM_DP} vLLM HTTP servers on the vLLM node (one per tile)
# ============================================================
echo "Starting ${VLLM_DP} vLLM HTTP servers on ${VLLM_NODE}..." | tee -a "${LOG}"
ssh "${VLLM_NODE}" "bash -s" <<EOF 2>&1 | tee -a "${LOG}"
set -o pipefail
cd ${PROJDIR}
module purge 2>/dev/null || true
module load frameworks/2025.3.1 2>/dev/null || module load frameworks 2>/dev/null || true
export PATH=\$(echo "\$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:\$//')
unset VIRTUAL_ENV
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ftp_proxy
unset PYTHONNOUSERSITE
export PYTHONPATH='${VLLM_PYTHONPATH}'
export no_proxy='*'
export NO_PROXY='*'
export VLLM_SERVER_DEV_MODE=1
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TORCH_COMPILE_DISABLE=1
unset PYTORCH_ALLOC_CONF
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export FI_PROVIDER=cxi
export CCL_KVS_IFACE=lo
export RAYON_NUM_THREADS=1
export TOKIO_WORKER_THREADS=1
export RUST_BACKTRACE=1
mkdir -p /tmp/torchtune
> /tmp/torchtune/vllm_pids.txt
for i in \$(seq 0 $((VLLM_DP-1))); do
    PORT=\$((${VLLM_BASE_PORT} + i))
    LOG_R=/tmp/torchtune/vllm_http_tile\${i}.log
    echo "[VLLM_NODE] Launching tile \${i} on port \${PORT}"
    setsid nohup env ZE_AFFINITY_MASK=\${i} PYTHONUNBUFFERED=1 python3 -m vllm.entrypoints.openai.api_server \\
        --model '${VLLM_MODEL_LOCAL}' \\
        --tensor-parallel-size 1 \\
        --port \${PORT} \\
        --host 0.0.0.0 \\
        --enforce-eager \\
        --dtype bfloat16 \\
        --gpu-memory-utilization ${VLLM_GPU_MEM} \\
        --max-model-len ${VLLM_MAX_MODEL_LEN} \\
        --max-num-seqs ${VLLM_MAX_NUM_SEQS} \\
        --distributed-executor-backend mp \\
        --worker-extension-cls torchtune.dev.vllm_weight_sync_worker.WeightSyncFromFileExtension \\
        > "\${LOG_R}" 2>&1 < /dev/null &
    echo \$! >> /tmp/torchtune/vllm_pids.txt
done
echo "[VLLM_NODE] Launched ${VLLM_DP} tiles, PIDs:"
cat /tmp/torchtune/vllm_pids.txt
EOF

# ============================================================
# Cleanup trap
# ============================================================
cleanup() {
    if [ "${VLLM_LEAVE_RUNNING:-0}" = "1" ]; then
        echo "Cleanup: VLLM_LEAVE_RUNNING=1, leaving vLLM tiles alive." | tee -a "${LOG}"
        return 0
    fi
    echo "Cleaning up..." | tee -a "${LOG}"
    ssh "${VLLM_NODE}" "
pkill -9 -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true
pkill -9 -f 'vllm.v1.engine' 2>/dev/null || true
pkill -9 VLLM 2>/dev/null || true
" 2>/dev/null || true
    for node in "${TRAIN_NODES[@]}"; do
        ssh "${node}" "pkill -9 -f 'grpo_full_finet[u]ne_distributed_xpu' 2>/dev/null" 2>/dev/null || true
    done
    echo "Cleanup done." | tee -a "${LOG}"
}
trap cleanup EXIT

# ============================================================
# Wait for all ${VLLM_DP} servers healthy (single persistent watcher SSH)
# ============================================================
echo "Waiting for ${VLLM_DP} vLLM servers (${VLLM_STARTUP_TIMEOUT}s timeout)..." | tee -a "${LOG}"
ssh "${VLLM_NODE}" "VLLM_DP=${VLLM_DP} VLLM_BASE_PORT=${VLLM_BASE_PORT} TIMEOUT=${VLLM_STARTUP_TIMEOUT} bash -s" <<'WATCH' | tee -a "${LOG}"
mapfile -t PIDS < /tmp/torchtune/vllm_pids.txt
if [ "${#PIDS[@]}" -ne "${VLLM_DP}" ]; then
    echo "FATAL: PID file has ${#PIDS[@]} entries, expected ${VLLM_DP}"
    exit 1
fi
TIMEOUT=${TIMEOUT:-600}
declare -a READY
for i in $(seq 0 $((VLLM_DP-1))); do READY[i]=0; done
DEADLINE=$(( $(date +%s) + TIMEOUT ))
while :; do
    all=1
    for i in $(seq 0 $((VLLM_DP-1))); do
        [ "${READY[i]}" -eq 1 ] && continue
        if ! kill -0 "${PIDS[i]}" 2>/dev/null; then
            echo "FATAL tile $i: PID ${PIDS[i]} died during startup"
            tail -60 /tmp/torchtune/vllm_http_tile${i}.log 2>/dev/null || echo "(no log)"
            exit 1
        fi
        port=$((VLLM_BASE_PORT + i))
        if curl --noproxy '*' -s --max-time 2 -o /dev/null "http://localhost:${port}/health"; then
            echo "  Tile $i healthy on port ${port}"
            READY[i]=1
        else
            all=0
        fi
    done
    if [ "$all" -eq 1 ]; then echo "All ${VLLM_DP} vLLM servers ready."; exit 0; fi
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then
        echo "FATAL: not all tiles ready within ${TIMEOUT}s"
        for i in $(seq 0 $((VLLM_DP-1))); do
            [ "${READY[i]}" -eq 1 ] && continue
            echo "--- tile $i (port $((VLLM_BASE_PORT+i))) ---"
            tail -40 /tmp/torchtune/vllm_http_tile${i}.log 2>/dev/null || echo "(no log)"
        done
        exit 1
    fi
    sleep 3
done
WATCH
WATCH_RC=${PIPESTATUS[0]}
if [ "${WATCH_RC}" -ne 0 ]; then
    echo "ERROR: vLLM startup failed (${WATCH_RC}); aborting." | tee -a "${LOG}"
    exit "${WATCH_RC}"
fi

# ============================================================
# Cross-node connectivity preflight from EACH train node
# ============================================================
echo "Preflight: cross-node /health from all 7 train nodes..." | tee -a "${LOG}"
PREFLIGHT_FAIL=0
for tnode in "${TRAIN_NODES[@]}"; do
    for ((r=0; r<VLLM_DP; r++)); do
        PORT=$((VLLM_BASE_PORT + r))
        URL="http://${VLLM_NODE_IP}:${PORT}/health"
        if ! ssh "${tnode}" "unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ftp_proxy; curl --noproxy '*' -s --max-time 5 -o /dev/null -w '%{http_code}' '${URL}'" 2>&1 | grep -q '^200$'; then
            echo "  PREFLIGHT FAIL: ${tnode} -> tile ${r}" | tee -a "${LOG}"
            PREFLIGHT_FAIL=1
        fi
    done
done
if [ "${PREFLIGHT_FAIL}" -ne 0 ]; then
    echo "ERROR: cross-node connectivity preflight failed; aborting." | tee -a "${LOG}"
    exit 1
fi
echo "Preflight OK: all tiles reachable from all train nodes." | tee -a "${LOG}"

# ============================================================
# EngineCore warmup on every tile (a real generate call)
# ============================================================
echo "EngineCore warmup on ${VLLM_DP} tiles..." | tee -a "${LOG}"
WARMUP_FAIL=0
for ((r=0; r<VLLM_DP; r++)); do
    PORT=$((VLLM_BASE_PORT + r))
    RESP=$(ssh "${VLLM_NODE}" "unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ftp_proxy; \
        curl --noproxy '*' -sf --max-time 30 -X POST 'http://localhost:${PORT}/v1/completions' \
        -H 'Content-Type: application/json' \
        -d '{\"model\": \"${VLLM_MODEL_LOCAL}\", \"prompt\": \"1+1=\", \"max_tokens\": 1}' 2>&1")
    if echo "${RESP}" | grep -q '"choices"'; then
        echo "  Tile ${r}: EngineCore OK" | tee -a "${LOG}"
    else
        echo "  Tile ${r}: EngineCore FAIL: ${RESP:0:200}" | tee -a "${LOG}"
        WARMUP_FAIL=1
    fi
done
if [ "${WARMUP_FAIL}" -ne 0 ]; then
    echo "ERROR: vLLM EngineCore warmup failed on >=1 tile; aborting." | tee -a "${LOG}"
    exit 1
fi
echo "vLLM EngineCore warmup PASSED." | tee -a "${LOG}"

# Test hook: validate vLLM startup only.
if [ "${VLLM_ONLY:-0}" = "1" ]; then
    echo "VLLM_ONLY=1: vLLM validated; skipping training. Exiting 0." | tee -a "${LOG}"
    if [ "${VLLM_LEAVE_RUNNING:-0}" = "1" ]; then trap - EXIT; fi
    exit 0
fi

# ============================================================
# Launch training: mpiexec --pmi=pmix across the 7 train nodes
# ============================================================
# Hostfile MUST match PBS_NODEFILE's exact format: one FQDN line per node, NO
# ":N" slot suffix. A constructed "host:N" file fails PALS' PMIx RPC handshake
# ("Couldn't send RPC launch ... Resource temporarily unavailable") even with
# correct FQDNs. Ranks-per-node is set by `-ppn ${NPROC}`, not by the hostfile.
# See memory/feedback_pbs_mpiexec_use_pbs_nodefile.md. We can't pass
# $PBS_NODEFILE directly because it also contains the vLLM node — so we emit the
# first ${NTRAIN_NODES} lines verbatim (same FQDN format, no suffix).
HOSTFILE=${LOGDIR}/hostfile.txt
> "${HOSTFILE}"
for n in "${TRAIN_NODES[@]}"; do echo "${n}" >> "${HOSTFILE}"; done
echo "Hostfile (plain FQDN, no :N suffix; -ppn ${NPROC} sets ranks/node):" | tee -a "${LOG}"
cat "${HOSTFILE}" | tee -a "${LOG}"

WRAPPER=${PROJDIR}/experiments/auroragpt_2b_bakeoff/_agpt2b_7n_train_rank_wrapper.sh
chmod +x "${WRAPPER}" 2>/dev/null || true

# Master = first train node.
NODE0_ADDR=$(ssh "${TRAIN_NODES[0]}" "hostname -i" 2>/dev/null | awk '{print $1}')
LAST4=$(echo "${PBS_JOBID:-$$}" | tr -dc '0-9' | tail -c 4)
MASTER_PORT=$(( 29500 + ( 10#${LAST4:-0} % 400 ) ))
export MASTER_ADDR=${NODE0_ADDR}
export MASTER_PORT

# Full multinode CCL block (production multi-node row from CLAUDE.md).
export CCL_PROCESS_LAUNCHER=pmix
export CCL_ATL_TRANSPORT=mpi
export CCL_KVS_MODE=mpi
export CCL_KVS_USE_MPI_RANKS=1
export CCL_CONFIGURATION=cpu_gpu_dpcpp
export CCL_KVS_CONNECTION_TIMEOUT=600
export CCL_OP_SYNC=1
export FI_PROVIDER=cxi
export CCL_WORKER_COUNT=1
export CCL_ALLREDUCE=ring
export CCL_CHUNK_SIZE=16777216
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=65536
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
unset XPU_USM_ALLOC_SO
# Allocator config: the HSDP 8N banned:1 at step 11 is NOT an XPU/CCL leak — the
# mem-probe (job 8544991) showed torch_resv climbing MONOTONICALLY ~5 GiB/step
# (20.7→41→46→52→57→62.2 GiB ceiling) while torch_alloc stayed FLAT at ~13 GiB
# and external CCL stayed flat at ~2 GiB. That is PyTorch reserved-pool growth
# from FRAGMENTATION of the larger, variable-length GSM8K buffers at batch_size=4
# (64 seqs/rank vs the 2N path's 16) — pinning the 64 GiB tile by step ~6.
# `max_split_size_mb:512` WORSENS this (forces splits → unreusable fragments), so
# it is removed; gc:0.8 reclaims the freeable cached blocks (torch_alloc drops to
# 8.7 GiB POST-BWD, so blocks ARE collectable) before reserved pins the tile.
# Paired with batch_size=2 (below) to halve per-rank peak. See
# memory/project_agpt2b_7n_hsdp_launcher.md.
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8
export GLOO_SOCKET_IFNAME=hsn0
# Enable l0_free / external-CCL mem probes so that if banned:1 recurs we get the
# leak evidence (which tile, growth rate) instead of guessing. Cheap; rank-0 log.
export TORCHTUNE_MEM_PROBE=${TORCHTUNE_MEM_PROBE:-1}

# Load the frameworks module on the mom node BEFORE mpiexec so the frameworks
# python3 (with torch/XPU) is on PATH; mpiexec --pmi=pmix propagates this env to
# all 36 ranks. Without this the ranks inherit system python3 and die with
# "ModuleNotFoundError: No module named 'torch'". Strip any myenv shadow.
module load frameworks/2025.3.1 2>/dev/null || module load frameworks 2>/dev/null || true
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV

export PYTHONNOUSERSITE=1
export PYTHONPATH="${TRAIN_PYTHONPATH}"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/lus/flare/projects/ModCon/ngetty/hf_datasets_cache}
export HF_HOME=${HF_HOME:-/lus/flare/projects/ModCon/ngetty/hf_cache}
export PYTHONUNBUFFERED=1
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ftp_proxy

# Forward wsync transport selectors to the ranks (weight_sync.py reads
# WSYNC_CROSS_METHOD from os.environ). mpiexec --pmi=pmix propagates the launching
# env, but export them explicitly alongside the rest for clarity.
export NSTEPS MODEL_PATH CONFIG VLLM_URLS PROJDIR WORLD
export WSYNC_CROSS_METHOD WSYNC_INTRA_METHOD VLLM_WSYNC_INTERVAL

# save_every_n_steps override (0/empty disables periodic step checkpointing).
_SAVE_OVERRIDE=""
if [ -n "${SAVE_EVERY_N_STEPS}" ] && [ "${SAVE_EVERY_N_STEPS}" != "0" ]; then
    _SAVE_OVERRIDE="save_every_n_steps=${SAVE_EVERY_N_STEPS}"
fi

# Resume overrides: when RESUME_FROM points at an epoch_<N> dir, continue training
# from the saved POLICY weights + recipe_state. Keep base_model_path at the ORIGINAL
# init — that keeps the tokenizer and the FROZEN ref model (ref_checkpointer reads
# ${base_model_path}/model.safetensors) resolving correctly. Override ONLY the
# policy checkpointer to read the resumed weights + recipe_state from RESUME_FROM.
# The policy save writes model.safetensors (verified job 8545510 epoch_0), which is
# the config default — so checkpoint_files needs no override. Note: a step-based run
# (epochs=1) writes every step-save into epoch_0 (epochs_run stays 0), so the latest
# state lives at ${output_dir}/epoch_0/recipe_state.pt (steps_run = last saved step).
_RESUME_OVERRIDE=""
_BASE_MODEL="${MODEL_PATH}"
if [ -n "${RESUME_FROM}" ]; then
    echo "RESUME_FROM=${RESUME_FROM} → continuing training from this checkpoint." | tee -a "${LOG}"
    _RESUME_OVERRIDE="resume_from_checkpoint=true checkpointer.checkpoint_dir=${RESUME_FROM} checkpointer.recipe_checkpoint=${RESUME_FROM}/recipe_state.pt"
fi

EXTRA="output_dir=${LOGDIR}/run_out vllm_weight_sync_interval=${VLLM_WSYNC_INTERVAL} ${_SAVE_OVERRIDE} ${_RESUME_OVERRIDE} ${EXTRA_OVERRIDES:-}"
TRAIN_LOG=${LOGDIR}/train_mpiexec.log

echo "Launching mpiexec --pmi=pmix -n ${WORLD} -ppn ${NPROC} ..." | tee -a "${LOG}"
echo "  MASTER=${MASTER_ADDR}:${MASTER_PORT}  train log: ${TRAIN_LOG}" | tee -a "${LOG}"

mpiexec \
    --pmi=pmix \
    -n ${WORLD} \
    -ppn ${NPROC} \
    --hostfile "${HOSTFILE}" \
    --cpu-bind depth --depth 8 \
    bash "${WRAPPER}" \
        ${PROJDIR}/recipes/dev/grpo_full_finetune_distributed_xpu.py \
        --config ${CONFIG} \
        base_model_path=${_BASE_MODEL} \
        num_steps=${NSTEPS} \
        data_parallel_replicate_dim=${DP_REPLICATE} \
        batch_size=${BATCH_SIZE} \
        grpo_samples=${GRPO_SAMPLES} \
        forward_batch_size=${FORWARD_BATCH_SIZE} \
        ref_forward_batch_size=${REF_FORWARD_BATCH_SIZE} \
        max_generated_tokens=${MAX_GEN_TOKENS} \
        vllm_max_num_seqs=${VLLM_MAX_NUM_SEQS} \
        "vllm_url=${VLLM_URLS}" \
        log_peak_memory_stats=true \
        ${EXTRA} \
    > "${TRAIN_LOG}" 2>&1
RC=$?

echo "=== AGPT-2B 7N HSDP: mpiexec rc=${RC} at $(date) (train log: ${TRAIN_LOG}) ===" | tee -a "${LOG}"
exit ${RC}

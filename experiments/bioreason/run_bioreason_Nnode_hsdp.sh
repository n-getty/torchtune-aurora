#!/bin/bash
# BioReason 4B GRPO — N-node HSDP (centralized vLLM).
#
# TOPOLOGY (N total PBS nodes):
#   Nodes 0..N-2 -> (N-1) training replicas: dp_replicate=(N-1) x dp_shard=12
#                   (world=12*(N-1)). One node = one shard group of 12 ranks.
#                   Launched via mpiexec --pmi=pmix (production multi-node row of
#                   the CLAUDE.md decision table).
#   Node N-1     -> 1 CENTRALIZED vLLM pool: 12 prompt_embeds HTTP servers
#                   (tiles 0-11, ports 8001-8012), --enable-prompt-embeds
#                   --tensor-parallel-size 1, SHARED by all (N-1) replicas.
#                   vLLM is NOT distributed — single VLLM_NODE_IP.
#
#   4 nodes = 3 train (dp_replicate=3) + 1 vLLM; world=36, dp_shard=12.
#
# This launcher fuses:
#   SOURCE A — experiments/bioreason/run_bioreason_2node_server.sh
#              (centralized prompt_embeds vLLM node + BioReason env/staging/wsync)
#   SOURCE B — experiments/auroragpt_2b_bakeoff/run_agpt2b_7n_hsdp_8node.sh
#              (multi-node node split + mpiexec --pmi=pmix + production CCL env)
#
# Wire format: train side POSTs base64(torch.save(prompt_embeds_bf16)) to
#   /v1/completions. Weight sync: XCCL/gloo to the shared 12-server pool.
#
# Usage (from a held PBS job with >=3 nodes, login or one of the held nodes):
#   bash experiments/bioreason/run_bioreason_Nnode_hsdp.sh
set -o pipefail

TT_DIR="/lus/flare/projects/ModCon/ngetty/torchtune"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG="${SCRIPT_DIR}/run_bioreason_Nnode_$(date +%Y%m%d_%H%M%S).log"

echo "=== BioReason 4B N-Node HSDP (centralized vLLM) ===" | tee "${LOG}"
echo "Date: $(date)  Host: $(hostname)" | tee -a "${LOG}"

# ============================================================
# Configuration
# ============================================================
VLLM_DP=${VLLM_DP:-12}                  # one HTTP server per tile (vLLM node)
VLLM_BASE_PORT=${VLLM_BASE_PORT:-8001}
VLLM_GPU_MEM=${VLLM_GPU_MEM:-0.70}      # validated 2026-04-29 for DP=12 BioReason 4B
MODEL_SRC=${MODEL_SRC:-/lus/flare/projects/ModCon/ngetty/models/bioreason-pro-sft}
MODEL_PATH=${MODEL_PATH:-/tmp/torchtune/$(basename "${MODEL_SRC}")}
NSTEPS=${NSTEPS:-5}
GRPO_SAMPLES=${GRPO_SAMPLES:-4}
# forward_batch_size: default 1 (safe envelope for the UR:40-in-backward wall).
FORWARD_BATCH_SIZE=${FORWARD_BATCH_SIZE:-1}
# batch_size = distinct prompts per step (per replica). Global distinct prompts/step
# = BATCH_SIZE x DP_REPLICATE.
BATCH_SIZE=${BATCH_SIZE:-2}
# Default ref_fbs to cover the full B*G group (one allgather), unless overridden.
# REF_FORWARD_BATCH_SIZE MUST be >= BATCH_SIZE*GRPO_SAMPLES or the no-grad ref fwd
# does num_seqs sequential FSDP allgathers (documented ~500x slowdown).
REF_FORWARD_BATCH_SIZE=${REF_FORWARD_BATCH_SIZE:-$(( BATCH_SIZE * GRPO_SAMPLES ))}
MAX_GEN_TOKENS=${MAX_GEN_TOKENS:-2048}
# vLLM's max_model_len MUST cover prompt ctx + MAX_GEN_TOKENS. Default 6144.
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-6144}
# =1 single backward, =0 chunked (default; bounds per-backward L0 footprint).
TORCHTUNE_USE_CHUNKED_LOSS=${TORCHTUNE_USE_CHUNKED_LOSS:-0}
VLLM_STARTUP_TIMEOUT=${VLLM_STARTUP_TIMEOUT:-1800}

# Checkpoint/resume knobs (needed for the 4N→8N continue plan):
#   SAVE_EVERY_N_STEPS — save the LoRA adapter+projectors every N steps (0/empty=off).
#   OUTPUT_DIR         — where epoch_<N>/adapter is written (default = config's output_dir).
#   RESUME_ADAPTER     — dir holding adapter_model.safetensors to RESUME the policy LoRA
#                        from (e.g. the 4N run's saved adapter). Empty = fresh init.
SAVE_EVERY_N_STEPS=${SAVE_EVERY_N_STEPS:-}
OUTPUT_DIR=${OUTPUT_DIR:-}
RESUME_ADAPTER=${RESUME_ADAPTER:-}
# Assemble the resume/save/output overrides into EXTRA_OVERRIDES (passed to the recipe).
_CKPT_OVERRIDES=""
[ -n "${SAVE_EVERY_N_STEPS}" ] && _CKPT_OVERRIDES="${_CKPT_OVERRIDES} save_every_n_steps=${SAVE_EVERY_N_STEPS}"
[ -n "${OUTPUT_DIR}" ]        && _CKPT_OVERRIDES="${_CKPT_OVERRIDES} output_dir=${OUTPUT_DIR}"
[ -n "${RESUME_ADAPTER}" ]    && _CKPT_OVERRIDES="${_CKPT_OVERRIDES} lora_adapter_path=${RESUME_ADAPTER}"
EXTRA_OVERRIDES="${_CKPT_OVERRIDES} ${EXTRA_OVERRIDES:-}"

# Train tiles per node (FULL node = 12; dp_shard must equal this).
NPROC=${NPROC:-12}

# Config selection: ENABLE_LORA=1 picks the PEFT-LoRA variant (frozen backbone +
# adapters + projectors), else the full-FT default. An explicit CONFIG= overrides.
if [ -n "${CONFIG:-}" ]; then
  CONFIG="${CONFIG}"
elif [ "${ENABLE_LORA:-0}" = "1" ]; then
  CONFIG=recipes/configs/dev/production/bioreason_4b_lora_grpo_2node_server_xpu.yaml
else
  CONFIG=recipes/configs/dev/production/bioreason_4b_grpo_2node_server_xpu.yaml
fi

# ============================================================
# Node discovery: (N-1) train + 1 vLLM, derived from the nodefile.
# Keep FULL FQDN hostnames — mpiexec --pmi=pmix / PALS RPC-launch FAILS with bare
# short names; it needs FQDNs in the hostfile. SSH/hostname -i/curl all work with
# the FQDN too. See memory/feedback_pbs_mpiexec_use_pbs_nodefile.md.
# ============================================================
if [[ -z "${PBS_NODEFILE:-}" ]]; then
    echo "ERROR: PBS_NODEFILE not set. Run from a held PBS job." | tee -a "${LOG}"
    exit 1
fi

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

# HSN IP of the FIRST train node for XCCL weight sync: vLLM workers connect to this
# address via Slingshot. hostname -i returns the management IP (10.115.x); we need
# hsn0 (10.112.x).
TRAIN_NODE0="${TRAIN_NODES[0]}"
TRAIN_NODE0_IP=$(ssh "${TRAIN_NODE0}" "hostname -i" 2>/dev/null | awk '{print $1}')
TRAIN_NODE_HSN_IP=$(ssh "${TRAIN_NODE0}" "ip -4 addr show hsn0 2>/dev/null | grep 'inet ' | awk '{print \$2}' | cut -d'/' -f1 | head -1")
if [[ -z "${TRAIN_NODE_HSN_IP}" ]]; then
    echo "WARNING: Could not get hsn0 IP for ${TRAIN_NODE0}; falling back to ${TRAIN_NODE0_IP}" | tee -a "${LOG}"
    TRAIN_NODE_HSN_IP="${TRAIN_NODE0_IP}"
fi

echo "=== BioReason N-Node HSDP ===" | tee -a "${LOG}"
echo "  Train nodes (${NTRAIN_NODES}): ${TRAIN_NODES[*]}" | tee -a "${LOG}"
echo "  vLLM node:       ${VLLM_NODE} (${VLLM_NODE_IP}, DP=${VLLM_DP} HTTP servers)" | tee -a "${LOG}"
echo "  TRAIN_NODE_HSN_IP: ${TRAIN_NODE_HSN_IP} (for XCCL weight sync)" | tee -a "${LOG}"
echo "  CONFIG=${CONFIG}" | tee -a "${LOG}"
echo "  world=${WORLD} (dp_replicate=${DP_REPLICATE} x dp_shard=${NPROC})" | tee -a "${LOG}"
echo "  batch_size=${BATCH_SIZE} (global distinct prompts/step = ${BATCH_SIZE}x${DP_REPLICATE}=$((BATCH_SIZE*DP_REPLICATE)))" | tee -a "${LOG}"
echo "  G=${GRPO_SAMPLES} fbs=${FORWARD_BATCH_SIZE} ref_fbs=${REF_FORWARD_BATCH_SIZE} max_gen=${MAX_GEN_TOKENS} steps=${NSTEPS}" | tee -a "${LOG}"

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ftp_proxy
export no_proxy="*"
export NO_PROXY="*"

# ============================================================
# Prepare PYTHONPATH (vLLM + bioreason deps)
# ============================================================
cd "${TT_DIR}"
source recipes/dev/_aurora_paths.sh
VLLM_CUSTOMIZATION="${TT_DIR}/recipes/dev/_usercustomize_vllm"
# BioReason source/deps paths — overridable via env so other developers can
# point at their own checkout without editing this script. The model wrapper
# reads BIOREASON_SRC / BIOREASON_DEPS at first model-construction time.
BIOREASON_SRC=${BIOREASON_SRC:-/flare/ModCon/ngetty/BioReason-Pro}
BIOREASON_DEPS=${BIOREASON_DEPS:-/lus/flare/projects/ModCon/ngetty/bioreason_deps}
export BIOREASON_SRC BIOREASON_DEPS
VLLM_PYTHONPATH="$(aurora_pythonpath "${TT_DIR}" "${VLLM_CUSTOMIZATION}")"
WORKER_EXT="torchtune.dev.vllm_weight_sync_worker.WeightSyncFromFileExtension"
TRAIN_PYTHONPATH="${BIOREASON_DEPS}:$(aurora_pythonpath "${TT_DIR}")"

# ============================================================
# Stage model to /tmp on all nodes (avoid 12-way Lustre reads)
# ============================================================
echo "Staging ${MODEL_SRC} to ${MODEL_PATH} on all ${NTOTAL} nodes..." | tee -a "${LOG}"
for node in "${UNIQUE_NODES[@]}"; do
    if ! ssh "${node}" "test -f '${MODEL_PATH}/config.json'" 2>/dev/null; then
        echo "  Copying to ${node}..." | tee -a "${LOG}"
        ssh "${node}" "mkdir -p $(dirname "${MODEL_PATH}") && cp -r ${MODEL_SRC} ${MODEL_PATH}" &
    else
        echo "  Already staged on ${node}" | tee -a "${LOG}"
    fi
done
wait
for node in "${UNIQUE_NODES[@]}"; do
    if ! ssh "${node}" "test -f '${MODEL_PATH}/config.json'" 2>/dev/null; then
        echo "FATAL: staging failed on ${node}" | tee -a "${LOG}"; exit 1
    fi
done

# ============================================================
# Pre-launch cleanup on VLLM_NODE
#
# A bare `pkill -9 vllm...api_server` leaves the EngineCore subproc + VllmWorker
# children alive (spawned via VLLM_WORKER_MULTIPROC_METHOD=spawn). These orphans
# hold ~52 GiB of L0 device memory each, so the next launch sees free=10 GiB and
# EngineCore aborts. We pkill all four vLLM process families + clean_tiles.sh
# --kill (fuser-detect /dev/dri holders) + sleep for L0 page release.
# Reference: memory/feedback_vllm_orphan_engine_core.md.
# ============================================================
TT_DIR_REMOTE="${TT_DIR}"
echo "Cleaning stale vLLM on ${VLLM_NODE}..." | tee -a "${LOG}"
ssh "${VLLM_NODE}" "
pkill -9 -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true
pkill -9 -f 'vllm.v1.engine' 2>/dev/null || true
pkill -9 -f 'from multiprocessing' 2>/dev/null || true
pkill -9 -f 'VLLM::' 2>/dev/null || true
# EngineCore subprocs use prctl(PR_SET_NAME, 'VLLM::EngineCore') so /proc/<pid>/comm
# is TRUNCATED to 15 chars: 'VLLM::EngineCor'. 'pkill VLLM' matches comm by prefix
# and catches both VLLM::EngineCor and VLLM::Worker subprocs.
pkill -9 VLLM 2>/dev/null || true
sleep 2
bash ${TT_DIR_REMOTE}/recipes/dev/clean_tiles.sh --kill 2>&1 | tail -20 || true
sleep 3
rm -f /dev/shm/vllm* 2>/dev/null || true
rm -f /dev/shm/torchtune/weight_update.raw 2>/dev/null || true
mkdir -p /dev/shm/torchtune /tmp/torchtune
" 2>&1 | tee -a "${LOG}" || true

# Cleanup on each TRAIN_NODE too (raw_bytes weight sync writes /dev/shm/torchtune/...)
for node in "${TRAIN_NODES[@]}"; do
    ssh "${node}" "
pkill -9 -f 'grpo_bioreason_distributed_xpu' 2>/dev/null || true
pkill -9 -f 'torch.distributed.run' 2>/dev/null || true
sleep 2
bash ${TT_DIR_REMOTE}/recipes/dev/clean_tiles.sh --kill 2>&1 | tail -20 || true
sleep 3
rm -f /dev/shm/torchtune/weight_update.raw 2>/dev/null || true
mkdir -p /dev/shm/torchtune /tmp/torchtune
" 2>&1 | tee -a "${LOG}" || true
done

# Shared-FS weight sync file (used by all nodes). Stale file from a prior run
# would be loaded once by vLLM on first sync attempt before the new file lands.
# WSYNC_PATH is overridable so two 4N jobs (e.g. an A/B and a prod run) can run
# CONCURRENTLY without clobbering each other's weight_update.raw on Lustre. Default
# = the shared path (single-job behavior unchanged).
WSYNC_PATH=${WSYNC_PATH:-/lus/flare/projects/ModCon/ngetty/torchtune/outputs/wsync/weight_update.raw}
rm -f "${WSYNC_PATH}" 2>/dev/null || true
mkdir -p "$(dirname "${WSYNC_PATH}")"

# ============================================================
# Pre-launch tile-memory verification on VLLM_NODE
# Catches orphaned EngineCore subprocs holding L0 device memory that survived the
# pkill+clean_tiles pass above. Without this the next launch sees free=10 GiB on
# the affected tile and dies at EngineCore init.
# ============================================================
echo "Verifying tile memory before launch on ${VLLM_NODE}..." | tee -a "${LOG}"
TILE_CHECK=$(ssh "${VLLM_NODE}" "bash ${TT_DIR_REMOTE}/recipes/dev/clean_tiles.sh --check" 2>&1)
echo "${TILE_CHECK}" | tee -a "${LOG}"
if echo "${TILE_CHECK}" | grep -q 'FULL'; then
    echo "WARNING: tiles below 20 GiB free after first cleanup pass. Re-running clean_tiles --kill..." | tee -a "${LOG}"
    ssh "${VLLM_NODE}" "bash ${TT_DIR_REMOTE}/recipes/dev/clean_tiles.sh --kill" 2>&1 | tail -30 | tee -a "${LOG}"
    sleep 5
    TILE_CHECK2=$(ssh "${VLLM_NODE}" "bash ${TT_DIR_REMOTE}/recipes/dev/clean_tiles.sh --check" 2>&1)
    echo "${TILE_CHECK2}" | tee -a "${LOG}"
    if echo "${TILE_CHECK2}" | grep -q 'FULL'; then
        echo "ERROR: tiles still FULL after second cleanup pass; aborting before vLLM would fail at EngineCore init." | tee -a "${LOG}"
        exit 1
    fi
    echo "Recovered: tiles now CLEAN/USABLE." | tee -a "${LOG}"
fi

# ============================================================
# Build the shared ${VLLM_DP}-URL pool (all train shard-leaders use this)
# ============================================================
VLLM_URLS=""
for ((r=0; r<VLLM_DP; r++)); do
    PORT=$((VLLM_BASE_PORT + r))
    VLLM_URLS="${VLLM_URLS:+${VLLM_URLS},}http://${VLLM_NODE_IP}:${PORT}"
done
echo "vLLM URLs: ${VLLM_URLS}" | tee -a "${LOG}"

# ============================================================
# Launch ${VLLM_DP} vLLM prompt_embeds HTTP servers on VLLM_NODE (one per tile)
#
# Each child is started with `setsid nohup ... &` and the SSH heredoc does NOT
# `wait` on them. The SSH session returns as soon as launches are dispatched; the
# children survive any subsequent SSH death (own session, ignore SIGHUP). The
# persistent watcher SSH below is the synchronization point.
# ============================================================
echo "Starting ${VLLM_DP} vLLM prompt_embeds HTTP servers on ${VLLM_NODE}..." | tee -a "${LOG}"
ssh "${VLLM_NODE}" "bash -s" <<EOF | tee -a "${LOG}"
set -o pipefail
cd ${TT_DIR}
module purge 2>/dev/null || true
module load frameworks/2025.3.1 2>/dev/null || module load frameworks 2>/dev/null || true
export PATH=\$(echo "\$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:\$//')
unset VIRTUAL_ENV
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ftp_proxy
# NOTE: PYTHONNOUSERSITE must NOT be set for vLLM workers — it disables
# site.execusercustomize(), which loads _usercustomize_vllm from PYTHONPATH.
# Without that patch vllm's registry _run_in_subprocess SIGSEGV's on XPU during
# model architecture inspection, killing every tile before EngineCore spawns.
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
export INFRA_PROVIDER=local
mkdir -p /tmp/torchtune
> /tmp/torchtune/vllm_pids.txt
for i in \$(seq 0 $((VLLM_DP-1))); do
    PORT=\$((${VLLM_BASE_PORT} + i))
    LOG_R=/tmp/torchtune/vllm_http_tile\${i}.log
    echo "[VLLM_NODE] Launching tile \${i} on port \${PORT}"
    setsid nohup env ZE_AFFINITY_MASK=\${i} PYTHONUNBUFFERED=1 python3 -m vllm.entrypoints.openai.api_server \\
        --model '${MODEL_PATH}' \\
        --tensor-parallel-size 1 \\
        --port \${PORT} \\
        --host 0.0.0.0 \\
        --enforce-eager \\
        --dtype bfloat16 \\
        --gpu-memory-utilization ${VLLM_GPU_MEM} \\
        --max-model-len ${VLLM_MAX_MODEL_LEN} \\
        --enable-prompt-embeds \\
        --distributed-executor-backend mp \\
        --worker-extension-cls ${WORKER_EXT} \\
        > "\${LOG_R}" 2>&1 < /dev/null &
    echo \$! >> /tmp/torchtune/vllm_pids.txt
done
echo "[VLLM_NODE] Launched ${VLLM_DP} tiles, PIDs:"
cat /tmp/torchtune/vllm_pids.txt
EOF
VLLM_LAUNCH_RC=$?
if [ "${VLLM_LAUNCH_RC}" -ne 0 ]; then
    echo "ERROR: vLLM launch SSH returned non-zero (${VLLM_LAUNCH_RC})." | tee -a "${LOG}"
    exit "${VLLM_LAUNCH_RC}"
fi

# ============================================================
# Cleanup trap
# ============================================================
cleanup() {
    if [ "${VLLM_LEAVE_RUNNING:-0}" = "1" ]; then
        echo "Cleanup: VLLM_LEAVE_RUNNING=1, leaving vLLM tiles alive on ${VLLM_NODE}." | tee -a "${LOG}"
        return 0
    fi
    echo "Cleaning up..." | tee -a "${LOG}"
    ssh "${VLLM_NODE}" "
pkill -9 -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true
pkill -9 -f 'vllm.v1.engine' 2>/dev/null || true
pkill -9 -f 'VLLM::' 2>/dev/null || true
pkill -9 VLLM 2>/dev/null || true
" 2>/dev/null || true
    for node in "${TRAIN_NODES[@]}"; do
        ssh "${node}" "pkill -9 -f 'grpo_bioreason_distributed_xpu' 2>/dev/null" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    echo "Cleanup done." | tee -a "${LOG}"
}
trap cleanup EXIT

# ============================================================
# Wait for all ${VLLM_DP} vLLM servers healthy (single persistent watcher SSH).
# Fast-fails if any vLLM PID dies during startup.
# ============================================================
echo "Waiting for ${VLLM_DP} vLLM servers (${VLLM_STARTUP_TIMEOUT}s timeout, single watcher SSH)..." | tee -a "${LOG}"
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
            echo "--- tail /tmp/torchtune/vllm_http_tile${i}.log ---"
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
            echo "--- tile $i (port $((VLLM_BASE_PORT+i))), PID ${PIDS[i]} alive=$(kill -0 ${PIDS[i]} 2>/dev/null && echo yes || echo no) ---"
            tail -40 /tmp/torchtune/vllm_http_tile${i}.log 2>/dev/null || echo "(no log)"
        done
        exit 1
    fi
    sleep 3
done
WATCH
WATCH_RC=${PIPESTATUS[0]}
if [ "${WATCH_RC}" -ne 0 ]; then
    echo "ERROR: vLLM startup watcher exited ${WATCH_RC}." | tee -a "${LOG}"
    exit "${WATCH_RC}"
fi

# ============================================================
# Cross-node connectivity preflight from EACH train node.
# Catches proxy/firewall regressions BEFORE blowing the PG-init 600s timeout.
# ============================================================
echo "Preflight: cross-node /health from all ${NTRAIN_NODES} train nodes..." | tee -a "${LOG}"
PREFLIGHT_FAIL=0
for tnode in "${TRAIN_NODES[@]}"; do
    for ((r=0; r<VLLM_DP; r++)); do
        PORT=$((VLLM_BASE_PORT + r))
        URL="http://${VLLM_NODE_IP}:${PORT}/health/"
        if ! ssh "${tnode}" "unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ftp_proxy; curl --noproxy '*' -L -s --max-time 5 -o /dev/null -w '%{http_code}' '${URL}'" 2>&1 | grep -q '^200$'; then
            echo "  PREFLIGHT FAIL: ${tnode} -> tile ${r} (${URL})" | tee -a "${LOG}"
            PREFLIGHT_FAIL=1
        fi
    done
done
if [ "${PREFLIGHT_FAIL}" -ne 0 ]; then
    echo "ERROR: cross-node connectivity preflight failed; aborting before PG-init timeout would kill ranks." | tee -a "${LOG}"
    exit 1
fi
echo "Preflight OK: all ${VLLM_DP} tiles reachable from all train nodes." | tee -a "${LOG}"

# Test hook: VLLM_ONLY=1 stops here (used by hardening validation tests).
if [ "${VLLM_ONLY:-0}" = "1" ]; then
    echo "VLLM_ONLY=1: vLLM startup validated; skipping training. Exiting 0." | tee -a "${LOG}"
    if [ "${VLLM_LEAVE_RUNNING:-0}" = "1" ]; then
        echo "VLLM_LEAVE_RUNNING=1: clearing EXIT trap; tiles will remain alive after this script returns." | tee -a "${LOG}"
        trap - EXIT
    fi
    exit 0
fi

# ============================================================
# Launch training: mpiexec --pmi=pmix across the (N-1) train nodes.
#
# Hostfile MUST match PBS_NODEFILE's exact format: one FQDN line per node, NO
# ":N" slot suffix. A constructed "host:N" file fails PALS' PMIx RPC handshake
# even with correct FQDNs. Ranks-per-node is set by `-ppn ${NPROC}`. We can't
# pass $PBS_NODEFILE directly because it also contains the vLLM node — so we emit
# the first ${NTRAIN_NODES} lines verbatim (same FQDN format, no suffix).
# ============================================================
HOSTFILE="${SCRIPT_DIR}/hostfile_$(date +%Y%m%d_%H%M%S).txt"
> "${HOSTFILE}"
for n in "${TRAIN_NODES[@]}"; do echo "${n}" >> "${HOSTFILE}"; done
echo "Hostfile (plain FQDN, no :N suffix; -ppn ${NPROC} sets ranks/node):" | tee -a "${LOG}"
cat "${HOSTFILE}" | tee -a "${LOG}"

WRAPPER=${TT_DIR}/experiments/bioreason/_bioreason_train_rank_wrapper.sh
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
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8
export GLOO_SOCKET_IFNAME=hsn0
export TORCHTUNE_MEM_PROBE=${TORCHTUNE_MEM_PROBE:-1}

# Load the frameworks module on the mom node BEFORE mpiexec so the frameworks
# python3 (with torch/XPU) is on PATH; mpiexec --pmi=pmix propagates this env to
# all ranks. Strip any myenv shadow.
module load frameworks/2025.3.1 2>/dev/null || module load frameworks 2>/dev/null || true
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV

export PYTHONNOUSERSITE=1
export PYTHONPATH="${TRAIN_PYTHONPATH}"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ftp_proxy

# BioReason wsync + fast-path env (weight_sync.py reads these from os.environ).
# mpiexec --pmi=pmix propagates the launching env; export BEFORE mpiexec so they
# reach all ranks.
# WORLD is LOAD-BEARING: the rank wrapper resolves WORLD_SIZE as
# ${PMI_SIZE:-${PALS_NRANKS:-${WORLD:-...}}}, and on this PALS PMI_SIZE/PALS_NRANKS
# are often EMPTY → it falls back to ${WORLD}, which MUST be exported here or every
# rank dies "could not resolve WORLD_SIZE" (the 4N boot-1 failure). Matches AGPT-2B
# launcher's `export ... WORLD`.
export WORLD
export BIOREASON_SRC BIOREASON_DEPS
export TORCHTUNE_WEIGHT_SYNC_PATH=${WSYNC_PATH}
export TORCHTUNE_XCCL_HOST=${TRAIN_NODE_HSN_IP}
export TORCHTUNE_PINNED_CPU_BUF=${TORCHTUNE_PINNED_CPU_BUF:-1}
export TORCHTUNE_USE_IPEX_VARLEN=${TORCHTUNE_USE_IPEX_VARLEN:-1}
export TORCHTUNE_USE_CHUNKED_LOSS=${TORCHTUNE_USE_CHUNKED_LOSS}
export WSYNC_CROSS_METHOD=${WSYNC_CROSS_METHOD:-gloo}
export WSYNC_INTRA_METHOD=${WSYNC_INTRA_METHOD:-xccl}
# WSYNC_TOPOLOGY=node_fanout (real 2-hop): 1 cross-node send to vLLM replica 0 +
# intra-node fanout to ranks 1..11. Validated production default for BioReason.
export WSYNC_TOPOLOGY=${WSYNC_TOPOLOGY:-node_fanout}

# AB_BANDS: when set (e.g. AB_BANDS="0 1"), run the SAME vLLM pool through one
# mpiexec leg per value of TORCHTUNE_VLLM_REPLICA_BANDS — same nodes, back-to-back,
# so the band-fix timing comparison is immune to Aurora's ~1.8x node-to-node
# variance. Empty (default) = single normal run (bands ON via recipe default).
AB_BANDS=${AB_BANDS:-}

_run_leg() {
    # $1 = leg label (used in log filename); inherits TORCHTUNE_VLLM_REPLICA_BANDS
    local _leg="$1"
    local _train_log="${SCRIPT_DIR}/train_mpiexec_$(date +%Y%m%d_%H%M%S)_${_leg}.log"
    echo "Launching mpiexec leg=${_leg} (REPLICA_BANDS=${TORCHTUNE_VLLM_REPLICA_BANDS:-1}) -n ${WORLD} -ppn ${NPROC} ..." | tee -a "${LOG}"
    echo "  MASTER=${MASTER_ADDR}:${MASTER_PORT}  train log: ${_train_log}" | tee -a "${LOG}"
    mpiexec \
        --pmi=pmix \
        -n ${WORLD} \
        -ppn ${NPROC} \
        --hostfile "${HOSTFILE}" \
        --cpu-bind depth --depth 8 \
        bash "${WRAPPER}" \
            ${TT_DIR}/recipes/dev/grpo_bioreason_distributed_xpu.py \
            --config ${CONFIG} \
            base_model_path=${MODEL_PATH} \
            num_steps=${NSTEPS} \
            data_parallel_replicate_dim=${DP_REPLICATE} \
            batch_size=${BATCH_SIZE} \
            grpo_samples=${GRPO_SAMPLES} \
            forward_batch_size=${FORWARD_BATCH_SIZE} \
            ref_forward_batch_size=${REF_FORWARD_BATCH_SIZE} \
            max_generated_tokens=${MAX_GEN_TOKENS} \
            "vllm_url=${VLLM_URLS}" \
            log_peak_memory_stats=true \
            ${EXTRA_OVERRIDES:-} \
        > "${_train_log}" 2>&1
    local _rc=$?
    echo "=== leg=${_leg} mpiexec rc=${_rc} at $(date) (train log: ${_train_log}) ===" | tee -a "${LOG}"
    return ${_rc}
}

if [ -n "${AB_BANDS}" ]; then
    RC=0
    for _b in ${AB_BANDS}; do
        export TORCHTUNE_VLLM_REPLICA_BANDS=${_b}
        _lbl=$([ "${_b}" = "0" ] && echo "bandsOFF" || echo "bandsON")
        _run_leg "${_lbl}"
        _legrc=$?
        [ ${_legrc} -ne 0 ] && RC=${_legrc}
    done
    echo "=== BioReason N-Node HSDP A/B done; worst rc=${RC} at $(date) ===" | tee -a "${LOG}"
    exit ${RC}
elif [ -n "${AB_STOP}" ]; then
    # AB_STOP="0 1": same vLLM pool, leg per TORCHTUNE_VLLM_STOP_TOKENS value
    # (0=vLLM never stops at EOS -> every rollout to max_tokens; 1=stop at EOS).
    # The real generation lever (bands A/B showed gen is max_gen-bound, not dispatch).
    RC=0
    for _s in ${AB_STOP}; do
        export TORCHTUNE_VLLM_STOP_TOKENS=${_s}
        _lbl=$([ "${_s}" = "0" ] && echo "stopOFF" || echo "stopON")
        _run_leg "${_lbl}"
        _legrc=$?
        [ ${_legrc} -ne 0 ] && RC=${_legrc}
    done
    echo "=== BioReason N-Node HSDP STOP A/B done; worst rc=${RC} at $(date) ===" | tee -a "${LOG}"
    exit ${RC}
else
    _run_leg "single"
    RC=$?
    echo "=== BioReason N-Node HSDP: mpiexec rc=${RC} at $(date) ===" | tee -a "${LOG}"
    exit ${RC}
fi

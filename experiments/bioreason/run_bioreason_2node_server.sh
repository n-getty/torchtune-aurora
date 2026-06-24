#!/bin/bash
# BioReason 4B GRPO — 2-node asymmetric (HTTP vLLM + prompt_embeds)
# Phase 2 launcher. See:
#   - configs: recipes/configs/dev/production/bioreason_4b_grpo_2node_server_xpu.yaml
#   - prototype: memory/project_bioreason_phase2_prototype.md
#
# Topology:
#   Node 0 (TRAIN_NODE): 11 training ranks (FSDP1 SHARD_GRAD_OP / ZeRO-2)
#   Node 1 (VLLM_NODE):  12 vLLM HTTP servers, ports 8001-8012, ZE_AFFINITY_MASK=0..11
#                         (--enable-prompt-embeds, --tensor-parallel-size 1)
#
# Wire format: train side POSTs base64(torch.save(prompt_embeds_bf16)) to /v1/completions.
# Weight sync: raw_bytes path → /dev/shm/torchtune/weight_update.raw + /collective_rpc.
#              vLLM only loads backbone (Qwen3-4B); recipe filters state_dict() to backbone.*
#
# Usage (from a held 2-node PBS job, login or one of the held nodes):
#   bash experiments/bioreason/run_bioreason_2node_server.sh
set -o pipefail

TT_DIR="/lus/flare/projects/ModCon/ngetty/torchtune"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG="${SCRIPT_DIR}/run_bioreason_2node_$(date +%Y%m%d_%H%M%S).log"

echo "=== BioReason 4B 2-Node Server Mode ===" | tee "${LOG}"
echo "Date: $(date)  Host: $(hostname)" | tee -a "${LOG}"

# ============================================================
# Configuration
# ============================================================
VLLM_DP=${VLLM_DP:-12}                  # one HTTP server per tile
VLLM_BASE_PORT=${VLLM_BASE_PORT:-8001}
VLLM_GPU_MEM=${VLLM_GPU_MEM:-0.70}      # validated 2026-04-29 for DP=12 BioReason 4B
TRAIN_TILES=${TRAIN_TILES:-12}          # 12 train ranks on 1 node (full node; the old
                                        # 11 was a stale "1 spare" default with no functional
                                        # dependency — AGPT-2B/Qwen use all 12).
MODEL_SRC=${MODEL_SRC:-/lus/flare/projects/ModCon/ngetty/models/bioreason-pro-sft}
MODEL_PATH=${MODEL_PATH:-/tmp/torchtune/$(basename "${MODEL_SRC}")}
NSTEPS=${NSTEPS:-5}
GRPO_SAMPLES=${GRPO_SAMPLES:-4}
FORWARD_BATCH_SIZE=${FORWARD_BATCH_SIZE:-4}
# batch_size = distinct prompts per step. In server single-replica mode all train ranks
# see the SAME batch (sampler_replicas=1) — distinct prompts come from batch_size>1 (rank 0
# generates all B*G and broadcasts), NOT from rank count. Widen this to reduce the per-step
# reward variance + strengthen batch-level advantage. ref_forward_batch_size MUST be
# >= BATCH_SIZE*GRPO_SAMPLES (see below) or the no-grad ref fwd does num_seqs sequential
# FSDP allgathers (documented ~500x slowdown). YAML-only before; now a launcher knob.
BATCH_SIZE=${BATCH_SIZE:-1}
# Default ref_fbs to cover the full B*G group (one allgather), unless overridden.
REF_FORWARD_BATCH_SIZE=${REF_FORWARD_BATCH_SIZE:-$(( BATCH_SIZE * GRPO_SAMPLES ))}
MAX_GEN_TOKENS=${MAX_GEN_TOKENS:-1024}
# vLLM's max_model_len MUST cover prompt ctx (<=1024) + MAX_GEN_TOKENS, else
# generation is truncated/rejected. Auto-derive so it tracks MAX_GEN_TOKENS
# (avoids the launcher-config-drift trap where a stale 2048 default silently
# capped a 2048-gen run). Override explicitly with VLLM_MAX_MODEL_LEN=.
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-$(( 1024 + MAX_GEN_TOKENS ))}
# Config selection: ENABLE_LORA=1 picks the PEFT-LoRA variant (frozen backbone +
# adapters + projectors), else the full-FT default. An explicit CONFIG= overrides.
if [ -n "${CONFIG:-}" ]; then
  CONFIG="${CONFIG}"
elif [ "${ENABLE_LORA:-0}" = "1" ]; then
  CONFIG=recipes/configs/dev/production/bioreason_4b_lora_grpo_2node_server_xpu.yaml
else
  CONFIG=recipes/configs/dev/production/bioreason_4b_grpo_2node_server_xpu.yaml
fi

RDZV_PORT=$((29400 + RANDOM % 1000))
MASTER_PORT=$((20000 + RANDOM % 20000))

# ============================================================
# Node discovery
# ============================================================
if [[ -z "${PBS_NODEFILE:-}" ]]; then
    echo "ERROR: PBS_NODEFILE not set. Run from a held PBS job." | tee -a "${LOG}"
    exit 1
fi

UNIQUE_NODES=($(cut -d'.' -f1 "${PBS_NODEFILE}" | awk '!seen[$0]++'))
if [ "${#UNIQUE_NODES[@]}" -lt 2 ]; then
    echo "ERROR: Need 2 nodes. Got ${#UNIQUE_NODES[@]}: ${UNIQUE_NODES[*]}" | tee -a "${LOG}"
    exit 1
fi

# Convention: first node = TRAIN, second node = VLLM.
TRAIN_NODE="${UNIQUE_NODES[0]}"
VLLM_NODE="${UNIQUE_NODES[1]}"

TRAIN_NODE_IP=$(ssh "${TRAIN_NODE}" "hostname -i" 2>/dev/null | head -1)
VLLM_NODE_IP=$(ssh "${VLLM_NODE}" "hostname -i" 2>/dev/null | head -1)
if [[ -z "${TRAIN_NODE_IP}" || -z "${VLLM_NODE_IP}" ]]; then
    echo "ERROR: hostname -i failed. TRAIN=${TRAIN_NODE_IP} VLLM=${VLLM_NODE_IP}" | tee -a "${LOG}"
    exit 1
fi

# HSN IP for XCCL weight sync: vLLM workers connect to this address via Slingshot.
# hostname -i above returns the management IP (10.115.x); we need hsn0 (10.112.x).
TRAIN_NODE_HSN_IP=$(ssh "${TRAIN_NODE}" "ip -4 addr show hsn0 2>/dev/null | grep 'inet ' | awk '{print \$2}' | cut -d'/' -f1 | head -1")
if [[ -z "${TRAIN_NODE_HSN_IP}" ]]; then
    echo "WARNING: Could not get hsn0 IP for ${TRAIN_NODE}; falling back to ${TRAIN_NODE_IP}" | tee -a "${LOG}"
    TRAIN_NODE_HSN_IP="${TRAIN_NODE_IP}"
fi
echo "TRAIN_NODE_HSN_IP: ${TRAIN_NODE_HSN_IP} (for XCCL weight sync)" | tee -a "${LOG}"

echo "Train node:  ${TRAIN_NODE} (IP=${TRAIN_NODE_IP}, ${TRAIN_TILES} ranks)" | tee -a "${LOG}"
echo "vLLM node:   ${VLLM_NODE}  (IP=${VLLM_NODE_IP}, DP=${VLLM_DP} HTTP servers)" | tee -a "${LOG}"
echo "Config:      ${CONFIG}" | tee -a "${LOG}"
echo "Steps: ${NSTEPS}, G=${GRPO_SAMPLES}, FBS=${FORWARD_BATCH_SIZE}, max_gen=${MAX_GEN_TOKENS}" | tee -a "${LOG}"

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
# Stage model to both nodes
# ============================================================
echo "Staging ${MODEL_SRC} to ${MODEL_PATH} on both nodes..." | tee -a "${LOG}"
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
# Adopted from experiments/multinode_32b/run_32b_3node_24way.sh + clean_tiles.sh.
# A bare `pkill -9 vllm.entrypoints.openai.api_server` leaves the EngineCore
# subproc and its VllmWorker children alive (they were spawned via
# VLLM_WORKER_MULTIPROC_METHOD=spawn). These orphans hold ~52 GiB of L0 device
# memory each, so the next launch sees free=10 GiB and EngineCore aborts with
# "No available memory for the cache blocks". We must:
#   1. pkill all four vLLM process families (api_server, v1.engine, multiprocessing
#      forkserver, VLLM:: workers).
#   2. Use clean_tiles.sh --kill to fuser-detect any process still holding /dev/dri
#      handles (catches edge cases where the proc name changed or pkill missed it).
#   3. Sleep a full 5s for L0 driver to release pages before launching new servers.
# Reference: memory/feedback_vllm_orphan_engine_core.md.
# ============================================================
TT_DIR_REMOTE="${TT_DIR}"
echo "Cleaning stale vLLM on ${VLLM_NODE}..." | tee -a "${LOG}"
ssh "${VLLM_NODE}" "
pkill -9 -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true
pkill -9 -f 'vllm.v1.engine' 2>/dev/null || true
pkill -9 -f 'from multiprocessing' 2>/dev/null || true
pkill -9 -f 'VLLM::' 2>/dev/null || true
# EngineCore subprocs use prctl(PR_SET_NAME, 'VLLM::EngineCore') so their /proc/<pid>/comm
# is TRUNCATED to 15 chars: 'VLLM::EngineCor' (no trailing 'e'). Both -f cmdline and
# 'pkill EngineCore' (full-name match) miss them. 'pkill VLLM' matches comm by prefix
# and catches both VLLM::EngineCor and VLLM::Worker subprocs.
pkill -9 VLLM 2>/dev/null || true
sleep 2
bash ${TT_DIR_REMOTE}/recipes/dev/clean_tiles.sh --kill 2>&1 | tail -20 || true
sleep 3
rm -f /dev/shm/vllm* 2>/dev/null || true
rm -f /dev/shm/torchtune/weight_update.raw 2>/dev/null || true
mkdir -p /dev/shm/torchtune /tmp/torchtune
" 2>&1 | tee -a "${LOG}" || true

# Cleanup on TRAIN_NODE too (raw_bytes weight sync writes /dev/shm/torchtune/...)
ssh "${TRAIN_NODE}" "
pkill -9 -f 'grpo_bioreason_distributed_xpu' 2>/dev/null || true
pkill -9 -f 'torch.distributed.run' 2>/dev/null || true
sleep 2
bash ${TT_DIR_REMOTE}/recipes/dev/clean_tiles.sh --kill 2>&1 | tail -20 || true
sleep 3
rm -f /dev/shm/torchtune/weight_update.raw 2>/dev/null || true
mkdir -p /dev/shm/torchtune /tmp/torchtune
" 2>&1 | tee -a "${LOG}" || true

# Shared-FS weight sync file (used by both nodes). Stale file from a prior run
# would be loaded once by vLLM on first sync attempt before the new file lands.
rm -f /lus/flare/projects/ModCon/ngetty/torchtune/outputs/wsync/weight_update.raw 2>/dev/null || true
mkdir -p /lus/flare/projects/ModCon/ngetty/torchtune/outputs/wsync

# ============================================================
# Pre-launch tile-memory verification
# Catches the v2 failure mode (orphaned EngineCore subprocs holding L0 device
# memory that survived the pkill+clean_tiles pass above). Without this, the
# next launch sees free=10 GiB on the affected tile and dies at EngineCore
# init with an empty failure dict ("Failed core proc(s): {}").
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
# Launch ${VLLM_DP} vLLM HTTP servers on VLLM_NODE
# ============================================================
VLLM_URLS=""
echo "Starting ${VLLM_DP} vLLM HTTP servers on ${VLLM_NODE}..." | tee -a "${LOG}"
for ((r=0; r<VLLM_DP; r++)); do
    PORT=$((VLLM_BASE_PORT + r))
    if [ -n "${VLLM_URLS}" ]; then
        VLLM_URLS="${VLLM_URLS},http://${VLLM_NODE_IP}:${PORT}"
    else
        VLLM_URLS="http://${VLLM_NODE_IP}:${PORT}"
    fi
done

# Launch all 12 servers in a single SSH session.
#
# CRITICAL: each child is started with `setsid nohup … &` and the SSH heredoc
# does NOT `wait` on them. The SSH session returns as soon as launches are
# dispatched; the children survive any subsequent SSH death because they're
# in their own session and ignore SIGHUP. The previous design used a
# foreground `wait` to keep SSH alive — but that meant any SSH drop during
# the long startup (v8: tile 4 SIGHUP'd ~65s in) killed every still-loading
# tile. The persistent watcher SSH below is the new synchronization point;
# vLLM child lifetimes are decoupled from SSH lifetimes.
#
# PIDs are written to /tmp/torchtune/vllm_pids.txt so the watcher can fast-fail
# if any child dies during startup (kill -0 check).
ssh "${VLLM_NODE}" "bash -s" <<EOF | tee -a "${LOG}"
set -o pipefail
cd ${TT_DIR}
module purge 2>/dev/null || true
module load frameworks/2025.3.1 2>/dev/null || module load frameworks 2>/dev/null || true
export PATH=\$(echo "\$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:\$//')
unset VIRTUAL_ENV
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ftp_proxy
# NOTE: PYTHONNOUSERSITE must NOT be set for vLLM workers — it disables
# site.execusercustomize(), which is the only thing that loads
# _usercustomize_vllm/usercustomize.py from PYTHONPATH. Without that patch,
# vllm.model_executor.models.registry's _run_in_subprocess SIGSEGV's on XPU
# during model architecture inspection (Qwen3ForCausalLM), killing every tile
# at startup before EngineCore even spawns. Train side keeps NOUSERSITE=1
# (line ~416) to avoid the math_verify user-site override.
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
    setsid nohup env ZE_AFFINITY_MASK=\${i} python3 -m vllm.entrypoints.openai.api_server \\
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

echo "vLLM URLs: ${VLLM_URLS}" | tee -a "${LOG}"

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
    ssh "${TRAIN_NODE}" "pkill -9 -f 'grpo_bioreason_distributed_xpu' 2>/dev/null" 2>/dev/null || true
    wait 2>/dev/null || true
    echo "Cleanup done." | tee -a "${LOG}"
}
trap cleanup EXIT

# ============================================================
# Wait for all ${VLLM_DP} vLLM servers to be healthy.
#
# Single persistent watcher SSH (NOT a polling storm of fresh SSHs). The
# previous design opened ~600 fresh SSH connections during a 900s startup,
# which is itself a likely contributor to the parent SSH drops we've seen.
# Now: one SSH that runs a local poll loop and emits one line per tile
# transition + fast-fails if any vLLM PID dies during startup.
#
# Timeout is 300s (was 900s); cold vLLM startup is ≤90s in v6/v7, so 300s is
# 3× margin. The fast-fail PID check reports failures within ~3s either way.
# ============================================================
echo "Waiting for ${VLLM_DP} vLLM servers (300s timeout, single watcher SSH)..." | tee -a "${LOG}"
ssh "${VLLM_NODE}" "VLLM_DP=${VLLM_DP} VLLM_BASE_PORT=${VLLM_BASE_PORT} bash -s" <<'WATCH' | tee -a "${LOG}"
mapfile -t PIDS < /tmp/torchtune/vllm_pids.txt
if [ "${#PIDS[@]}" -ne "${VLLM_DP}" ]; then
    echo "FATAL: PID file has ${#PIDS[@]} entries, expected ${VLLM_DP}"
    exit 1
fi
declare -a READY
for i in $(seq 0 $((VLLM_DP-1))); do READY[i]=0; done
DEADLINE=$(( $(date +%s) + 300 ))
while :; do
    all=1
    for i in $(seq 0 $((VLLM_DP-1))); do
        [ "${READY[i]}" -eq 1 ] && continue
        # Fast-fail: if the child died during startup, abort with the tail.
        if ! kill -0 "${PIDS[i]}" 2>/dev/null; then
            echo "FATAL tile $i: PID ${PIDS[i]} died during startup"
            echo "--- tail /tmp/torchtune/vllm_http_tile${i}.log ---"
            tail -60 /tmp/torchtune/vllm_http_tile${i}.log 2>/dev/null || echo "(no log)"
            exit 1
        fi
        port=$((VLLM_BASE_PORT + i))
        if curl --noproxy '*' -s --max-time 2 -o /dev/null "http://localhost:${port}/health"; then
            elapsed=$(( $(date +%s) - (DEADLINE - 300) ))
            echo "  Tile $i healthy on port ${port} (${elapsed}s)"
            READY[i]=1
        else
            all=0
        fi
    done
    if [ "$all" -eq 1 ]; then
        echo "All ${VLLM_DP} vLLM servers ready."
        exit 0
    fi
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then
        echo "FATAL: not all tiles ready within 300s"
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
# Cross-node connectivity preflight
#
# The localhost loop above only proves vLLM tiles are reachable from VLLM_NODE.
# v1 (16:30 hold) hung 600s because TRAIN_NODE inherited HTTP_PROXY=alcf and
# Python `requests` routed cross-node /health probes through the proxy. Catch
# any remaining proxy/firewall regression here, BEFORE blowing the PG init
# 600s timeout (which kills ranks 1-N).
# ============================================================
echo "Preflight: cross-node /health from ${TRAIN_NODE} to all ${VLLM_DP} vLLM tiles..." | tee -a "${LOG}"
PREFLIGHT_FAIL=0
for ((r=0; r<VLLM_DP; r++)); do
    PORT=$((VLLM_BASE_PORT + r))
    URL="http://${VLLM_NODE_IP}:${PORT}/health/"
    # -L follows the 307 redirect from /health/ → /health (which is what
    # the in-Python `requests.Session` does too; rank 0 will see 200).
    if ! ssh "${TRAIN_NODE}" "unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ftp_proxy; curl --noproxy '*' -L -s --max-time 5 -o /dev/null -w '%{http_code}' '${URL}'" 2>&1 | grep -q '^200$'; then
        echo "  PREFLIGHT FAIL: tile ${r} (${URL}) unreachable from TRAIN_NODE" | tee -a "${LOG}"
        PREFLIGHT_FAIL=1
    fi
done
if [ "${PREFLIGHT_FAIL}" -ne 0 ]; then
    echo "ERROR: cross-node connectivity preflight failed; aborting before PG-init timeout would kill ranks." | tee -a "${LOG}"
    exit 1
fi
echo "Preflight OK: all ${VLLM_DP} tiles reachable cross-node." | tee -a "${LOG}"

# Test hook: VLLM_ONLY=1 stops here (used by hardening validation tests).
# The trap will tear down vLLM tiles on exit. Pair with `trap - EXIT` if you
# want the tiles to survive for inspection.
if [ "${VLLM_ONLY:-0}" = "1" ]; then
    echo "VLLM_ONLY=1: vLLM startup validated; skipping training. Exiting 0." | tee -a "${LOG}"
    if [ "${VLLM_LEAVE_RUNNING:-0}" = "1" ]; then
        echo "VLLM_LEAVE_RUNNING=1: clearing EXIT trap; tiles will remain alive after this script returns." | tee -a "${LOG}"
        trap - EXIT
    fi
    exit 0
fi

# ============================================================
# Launch training on TRAIN_NODE (single node, 11 ranks)
# ============================================================
echo "" | tee -a "${LOG}"
echo "Starting ${TRAIN_TILES}-rank training on ${TRAIN_NODE}..." | tee -a "${LOG}"

TRAIN_LOG="/tmp/torchtune/train_node0.log"
TRAIN_PID_FILE="/tmp/torchtune/train_pid.txt"
TRAIN_EXIT_FILE="/tmp/torchtune/train_exit.txt"

# Detached launch: setsid+nohup so the python survives any SSH parent drop.
# Same pattern as the vLLM-side hardening (lines 208-274). v5 (8462595) lost
# a 2-node hold to "Connection to TRAIN_NODE closed by remote host" mid-step-2;
# the foreground `ssh ... <<EOF & wait` pattern dies with the parent and
# SIGHUPs all 11 train ranks. With setsid+nohup the python is in its own
# session and ignores SIGHUP; the watcher SSH below only observes.
ssh "${TRAIN_NODE}" "bash -s" <<EOF 2>&1 | tee -a "${LOG}"
set -e
mkdir -p /tmp/torchtune /dev/shm/torchtune
rm -f ${TRAIN_PID_FILE} ${TRAIN_EXIT_FILE}
> ${TRAIN_LOG}
setsid nohup bash -c '
cd ${TT_DIR}
module purge 2>/dev/null || true
module load frameworks/2025.3.1 2>/dev/null || module load frameworks 2>/dev/null || true
export PATH=\$(echo "\$PATH" | tr ":" "\n" | grep -v myenv | tr "\n" ":" | sed "s/:\\\$//")
unset VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export INFRA_PROVIDER=local
export CCL_PROCESS_LAUNCHER=none
export CCL_OP_SYNC=1
export CCL_WORKER_COUNT=1
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=65536
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export PYTORCH_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.95
export TORCH_COMPILE_DISABLE=1
export TORCHTUNE_USE_CHUNKED_LOSS=${TORCHTUNE_USE_CHUNKED_LOSS:-1}
export TORCHTUNE_PINNED_CPU_BUF=1
export TORCHTUNE_VLLM_FANOUT_MAX=${TORCHTUNE_VLLM_FANOUT_MAX:-}
export TORCHTUNE_USE_IPEX_VARLEN=${TORCHTUNE_USE_IPEX_VARLEN:-1}
export BIOREASON_SRC=${BIOREASON_SRC}
export BIOREASON_DEPS=${BIOREASON_DEPS}
export TORCHTUNE_WEIGHT_SYNC_PATH=/lus/flare/projects/ModCon/ngetty/torchtune/outputs/wsync/weight_update.raw
export TORCHTUNE_XCCL_HOST=${TRAIN_NODE_HSN_IP}
export GLOO_SOCKET_IFNAME=hsn0
export WSYNC_CROSS_METHOD=${WSYNC_CROSS_METHOD:-gloo}
export WSYNC_INTRA_METHOD=${WSYNC_INTRA_METHOD:-xccl}
# WSYNC_TOPOLOGY=node_fanout (real 2-hop): 1 cross-node send to vLLM replica 0
# + intra-node fanout to ranks 1..11 over XeLink/loopback. Cuts cross-NIC
# traffic from 12x to 1x for DP=12. Validated production default for BioReason
# 2-node DP=12 (5/5 G=8 + 10/10 G=4 clean, hold 8462930). Override to
# replica_fanout for the legacy 32B 2-node baseline if needed.
export WSYNC_TOPOLOGY=${WSYNC_TOPOLOGY:-node_fanout}
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${TRAIN_PYTHONPATH}"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ftp_proxy
export no_proxy="*"
export NO_PROXY="*"
mkdir -p /tmp/torchtune /dev/shm/torchtune
python3 -m torch.distributed.run \
    --standalone \
    --nproc_per_node=${TRAIN_TILES} \
    --redirects 3 --tee 3 \
    --log-dir /tmp/torchtune/torchelastic_logs \
    recipes/dev/grpo_bioreason_distributed_xpu.py \
    --config ${CONFIG} \
    base_model_path=${MODEL_PATH} \
    num_steps=${NSTEPS} \
    batch_size=${BATCH_SIZE} \
    grpo_samples=${GRPO_SAMPLES} \
    forward_batch_size=${FORWARD_BATCH_SIZE} \
    ref_forward_batch_size=${REF_FORWARD_BATCH_SIZE} \
    max_generated_tokens=${MAX_GEN_TOKENS} \
    vllm_url="${VLLM_URLS}" \
    log_peak_memory_stats=true ${EXTRA_OVERRIDES:-} > ${TRAIN_LOG} 2>&1 < /dev/null
echo \$? > ${TRAIN_EXIT_FILE}
' < /dev/null > /dev/null 2>&1 &
echo \$! > ${TRAIN_PID_FILE}
echo "[TRAIN_NODE] dispatched detached training, PID=\$(cat ${TRAIN_PID_FILE})"
EOF
DISPATCH_RC=$?
if [ "${DISPATCH_RC}" -ne 0 ]; then
    echo "ERROR: train dispatch SSH returned ${DISPATCH_RC}." | tee -a "${LOG}"
    exit "${DISPATCH_RC}"
fi

# Persistent watcher SSH. Tails TRAIN_LOG into the launcher LOG and polls the
# detached python PID. On watcher drop, retries up to 3× — the training process
# is decoupled (setsid+nohup) and unaffected by watcher death.
echo "Watching detached training on ${TRAIN_NODE}..." | tee -a "${LOG}"
TRAIN_EXIT=99
WATCH_TRIES=0
while [ "${WATCH_TRIES}" -lt 3 ]; do
    WATCH_TRIES=$((WATCH_TRIES + 1))
    ssh "${TRAIN_NODE}" "TRAIN_LOG=${TRAIN_LOG} TRAIN_PID_FILE=${TRAIN_PID_FILE} TRAIN_EXIT_FILE=${TRAIN_EXIT_FILE} bash -s" <<'WATCH' 2>&1 | tee -a "${LOG}"
TPID=$(cat ${TRAIN_PID_FILE} 2>/dev/null)
if [ -z "${TPID}" ]; then
    echo "WATCH: no PID file; training never dispatched"
    exit 98
fi
( tail -n +1 -F ${TRAIN_LOG} 2>/dev/null ) &
TAIL_PID=$!
trap "kill ${TAIL_PID} 2>/dev/null" EXIT
while kill -0 ${TPID} 2>/dev/null; do
    sleep 5
done
sleep 3
kill ${TAIL_PID} 2>/dev/null
EC=$(cat ${TRAIN_EXIT_FILE} 2>/dev/null)
echo "WATCH: training PID ${TPID} exited (exit_code=${EC:-unknown})"
exit ${EC:-97}
WATCH
    WATCH_RC=${PIPESTATUS[0]}
    # If the recorded training exit code exists, training is truly done.
    EXIT_FILE_EXISTS=$(ssh "${TRAIN_NODE}" "test -f ${TRAIN_EXIT_FILE} && echo yes || echo no" 2>/dev/null)
    if [ "${EXIT_FILE_EXISTS}" = "yes" ]; then
        TRAIN_EXIT=$(ssh "${TRAIN_NODE}" "cat ${TRAIN_EXIT_FILE}" 2>/dev/null || echo "${WATCH_RC}")
        break
    fi
    echo "WARN: watcher SSH dropped (rc=${WATCH_RC}); training still running on ${TRAIN_NODE}; reattaching (try ${WATCH_TRIES}/3)..." | tee -a "${LOG}"
    sleep 5
done

echo "=== BioReason 2-Node Server: exit=${TRAIN_EXIT} at $(date) ===" | tee -a "${LOG}"
exit "${TRAIN_EXIT}"

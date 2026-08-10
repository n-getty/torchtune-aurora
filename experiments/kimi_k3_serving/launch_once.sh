#!/usr/bin/env bash
set -uo pipefail

# ONE start per allocation. Size from measured free memory; never retry in place.
#
# WHY: retrying a failed start on the same nodes is what produced every
# unrecoverable `banned: 1` GPU fault in this investigation. A failed vLLM
# start leaves 5-9 GiB/tile allocated with NO surviving process holding it, so
# each retry begins with less memory than the last, and the run that finally
# fits then takes a GPU page fault after ~6 requests. Observed 3 times across
# 2 node sets (jobs 8746183, 8746327), with and without chunked prefill -- I
# initially misattributed it to bad hardware, then to --no-enable-chunked-
# prefill, before the retry pattern became the obvious common factor. The one
# clean 65.20 tok/s run was a first-start on untouched nodes.
#
# So: measure first, compute the config from what is actually free, start once.
# If it still fails, get a fresh allocation -- do NOT start again here.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
JOB=${1:?usage: launch_once.sh <jobid> <max_num_seqs> [tag]}
SEQS=${2:?usage: launch_once.sh <jobid> <max_num_seqs> [tag]}
TAG=${3:-once}
PYTHON=${PYTHON:-/flare/ModCon/ngetty/venvs/kimi-k3-xpu-framework/bin/python}
MODEL=/tmp/ngetty/AuroraGPT/prism_models
B=$SCRIPT_DIR/logs/k3_${TAG}_${JOB%%.*}
mkdir -p "$B"

qstat -f -w "$JOB" 2>/dev/null | tr -d '\n\t' | grep -oE 'exec_host = [^ ]+' \
  | sed 's/exec_host = //' | tr '+' '\n' | sed -E 's#/.*##' | sort -u > "$B/nodefile"
mapfile -t NODES < "$B/nodefile"
[[ ${#NODES[@]} -eq 3 ]] || { echo "ERROR: expected 3 nodes, got ${#NODES[@]}" >&2; exit 1; }
HEAD=${NODES[0]}
echo "nodes: ${NODES[*]}"

echo "=== mounting DAOS ==="
PBS_NODEFILE=$B/nodefile LOG_DIR=$B "$SCRIPT_DIR/mount_daos_models_all_nodes.sh" >/dev/null 2>&1
for n in "${NODES[@]}"; do
    ssh -o BatchMode=yes "$n" "test -f $MODEL/config.json" \
        || { echo "ERROR: model not mounted on $n" >&2; exit 1; }
done
echo "mounts verified"

echo "=== measuring free memory (this is what sizes the config) ==="
MINFREE=$(ssh -o BatchMode=yes "$HEAD" "ZE_FLAT_DEVICE_HIERARCHY=FLAT $PYTHON -c \"
import torch
print(min(torch.xpu.mem_get_info(i)[0] for i in range(torch.xpu.device_count())) / 1024**3)
\"" 2>/dev/null | tail -1)
[[ -n "$MINFREE" ]] || { echo "ERROR: could not measure free memory" >&2; exit 1; }
echo "worst-tile free: ${MINFREE} GiB"

# A node set that starts with memory already consumed has, every time, been one
# that later faulted. Refuse it rather than burning 15 minutes on a load.
read -r VERDICT UTIL BLOCKS <<<"$("$PYTHON" - "$MINFREE" "$SEQS" <<'PY'
import sys
free, seqs = float(sys.argv[1]), int(sys.argv[2])
total, weights, page_gib = 63.98, 47.45, 221184 * 24 / 1024**3
if free < 60:
    print("REFUSE 0 0")
    raise SystemExit
# Leave 1 GiB of slack under the measured floor so a small drift during load
# does not trip request_memory's hard check.
util = min(0.92, (free - 1.0) / total)
# 3 blocks per sequence at 32-in/512-out (576 tokens, 192-token blocks), then
# whatever is left after weights goes to activations.
need = seqs * 3
budget = total * util - weights
blocks = min(int((budget - 1.5) / page_gib), max(need + 60, need))
print(f"{'OK' if blocks >= need else 'TOOSMALL'} {util:.3f} {blocks}")
PY
)"
echo "verdict=$VERDICT util=$UTIL blocks=$BLOCKS (need $((SEQS * 3)) for c=$SEQS)"
case "$VERDICT" in
    REFUSE)
        echo "ERROR: nodes already have memory consumed (${MINFREE} GiB free)." >&2
        echo "       Every fault so far came from such a node set. Get a fresh one." >&2
        exit 2 ;;
    TOOSMALL)
        echo "ERROR: cannot fit c=$SEQS in the available memory." >&2
        exit 3 ;;
esac

echo "=== starting server (ONE attempt) ==="
ssh -o BatchMode=yes "$HEAD" "cd $SCRIPT_DIR && nohup env \
  K3_JOB_ID=$JOB K3_NODEFILE=$B/nodefile K3_CACHE_ROOT=/tmp/k3_hf_${TAG}_${JOB%%.*} \
  RAY_CGRAPH_get_timeout=3600 RAY_CGRAPH_submit_timeout=3600 \
  VLLM_KIMI_XPU_KDA_VECTORIZED=1 VLLM_KIMI_XPU_CONV1D_VECTORIZED=1 \
  VLLM_KIMI_XPU_KDA_TRITON=0 VLLM_KIMI_XPU_CAUSAL_CONV1D_TRITON=0 \
  VLLM_XPU_ALLOW_TRITON_SAMPLER=0 \
  MODEL=$MODEL TP=32 PP=1 EP=1 \
  MAX_MODEL_LEN=2048 MAX_NUM_SEQS=$SEQS MAX_BATCHED_TOKENS=4096 \
  GPU_MEM_UTIL=$UTIL SERVED_MODEL_NAME=Kimi-K3 LOG_DIR=$B/server \
  $SCRIPT_DIR/serve_k3.sh --model $MODEL --tp 32 \
  --max-model-len 2048 --max-num-seqs $SEQS --max-num-batched-tokens 4096 \
  --gpu-memory-utilization $UTIL --served-model-name Kimi-K3 --ep \
  --no-async-scheduling --diagnostic-blocks $BLOCKS \
  >$B/launcher.log 2>&1 & echo launcher_pid=\$!"
echo "RUN_BASE=$B"

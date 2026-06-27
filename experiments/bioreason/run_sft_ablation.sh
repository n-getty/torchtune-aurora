#!/bin/bash
# One BioReason SFT ablation cell: run N steps on this node, no checkpoint save, parse
# the median time_per_step_s / tokens_per_second over the measured window.
#
# Usage (on a held node, ABSOLUTE path):
#   TAG=hsdp_shard4 NPROC=12 STEPS=20 OVERRIDES="data_parallel_shard_dim=4 data_parallel_replicate_dim=3" \
#     bash /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/run_sft_ablation.sh
#
# Env:
#   TAG        label for the log dir (required)
#   NPROC      tiles to use (default 12 = full node)
#   STEPS      max optimizer steps (default 20; warmup 5, measure the rest)
#   WARMUP     warmup steps to discard (default 5)
#   OVERRIDES  extra key=val recipe overrides (the lever under test)

set -eo pipefail   # no `set -u` (venv activate touches unbound vars)

PROJDIR=/lus/flare/projects/ModCon/ngetty/torchtune
CACHE=/lus/flare/projects/ModCon/ngetty/datasets/bioreason_sft_reasoning/esm3_cache_2048.pt
CONFIG="${PROJDIR}/recipes/configs/dev/smoke/sft_bioreason_gemma4_31B_smoke_xpu.yaml"
RECIPE="${PROJDIR}/recipes/dev/sft_bioreason_distributed_xpu.py"
TAG="${TAG:?set TAG}"
NPROC="${NPROC:-12}"
STEPS="${STEPS:-20}"
WARMUP="${WARMUP:-5}"

# ── Environment (frameworks + BioReason deps + offline HF) ────────────────────
module load frameworks/2025.3.1 2>/dev/null || module load frameworks 2>/dev/null || true
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v myenv | tr '\n' ':' | sed 's/:$//')
unset VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export BIOREASON_SRC=/lus/flare/projects/ModCon/ngetty/BioReason-Pro
export BIOREASON_DEPS=/lus/flare/projects/ModCon/ngetty/bioreason_deps
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export HF_DATASETS_CACHE=/lus/flare/projects/ModCon/ngetty/hf_datasets_cache
export HF_HOME=/lus/flare/projects/ModCon/ngetty/hf_cache
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ftp_proxy
export no_proxy="*" NO_PROXY="*"
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
unset XPU_USM_ALLOC_SO PYTORCH_ALLOC_CONF
export PYTHONUNBUFFERED=1
# Single-node CCL row (torchrun --standalone)
export CCL_PROCESS_LAUNCHER=none CCL_ATL_TRANSPORT=ofi CCL_OP_SYNC=1 CCL_WORKER_COUNT=1
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=65536 FI_PROVIDER=cxi CCL_KVS_IFACE=lo

TS=$(date +%H%M%S)
LOGDIR="${PROJDIR}/experiments/bioreason/logs/ablate_${TAG}_${TS}"
mkdir -p "${LOGDIR}"
cd "${PROJDIR}"

echo "=== ABLATION ${TAG} | host=$(hostname) NPROC=${NPROC} STEPS=${STEPS} ===" | tee "${LOGDIR}/run.log"
echo "  OVERRIDES: ${OVERRIDES:-<none>}" | tee -a "${LOGDIR}/run.log"

# save_every_n_steps > STEPS so no checkpoint is written during the ablation.
PYTHONPATH="${BIOREASON_DEPS}:${BIOREASON_SRC}:${PROJDIR}" \
python3 -m torch.distributed.run \
    --standalone --nproc_per_node=${NPROC} \
    --redirects 3 --tee 3 --log-dir "${LOGDIR}/torchelastic" \
    "${RECIPE}" --config "${CONFIG}" \
    output_dir="${LOGDIR}/run_out" \
    esm3_cache_path="${CACHE}" \
    max_steps_per_epoch=${STEPS} \
    save_every_n_steps=100000 \
    ${OVERRIDES:-} \
    2>&1 | tee -a "${LOGDIR}/run.log"
rc=${PIPESTATUS[0]}
echo "=== ablation ${TAG} exit rc=${rc} ===" | tee -a "${LOGDIR}/run.log"

# ── Parse median step time + tokens/s over the measured window ────────────────
python3 - "$LOGDIR/run.log" "$WARMUP" "$TAG" <<'PYEOF' | tee -a "${LOGDIR}/run.log"
import re, sys, statistics
log, warmup, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
txt = open(log, errors="ignore").read()
steps = re.findall(r"time_per_step_s['\"]?:\s*([0-9.]+)", txt)
toks  = re.findall(r"tokens_per_second_per_gpu['\"]?:\s*([0-9.]+)", txt)
peak  = re.findall(r"peak memory (?:reserved|active)[^0-9]*([0-9.]+)\s*GiB", txt)
st = [float(x) for x in steps][warmup:]
tk = [float(x) for x in toks][warmup:]
def med(a): return round(statistics.median(a), 3) if a else None
print(f"ABLATE_RESULT tag={tag} n={len(st)} "
      f"median_step_s={med(st)} median_tok_s_per_gpu={med(tk)} "
      f"peak_gib={max([float(x) for x in peak]) if peak else None}")
PYEOF
exit ${rc}

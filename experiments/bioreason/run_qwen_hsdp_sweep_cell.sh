#!/bin/bash
# One HSDP shard-width cell for the Qwen3-32B BioReason SFT. Runs N steps at a given
# (shard, replicate) and reports median step time + peak mem. No checkpoint save.
#
# Usage (on a held node):
#   TAG=shard2 DP_SHARD=2 DP_REPLICATE=6 STEPS=12 \
#     bash /lus/flare/projects/ModCon/ngetty/torchtune/experiments/bioreason/run_qwen_hsdp_sweep_cell.sh
#
# Memory model (LoRA, AC on, seq~4-8K) — smaller shard = more replication = faster IF it fits:
#   shard=2/rep=6 ≈ 43 GiB/tile (aggressive)  shard=3/rep=4 ≈ 33  shard=12/rep=1 ≈ baseline
set -eo pipefail   # no set -u

PROJDIR=/lus/flare/projects/ModCon/ngetty/torchtune
RECIPE="${PROJDIR}/recipes/dev/sft_bioreason_distributed_xpu.py"
CONFIG="${PROJDIR}/recipes/configs/dev/smoke/sft_bioreason_qwen3_32B_smoke_xpu.yaml"
CACHE=/lus/flare/projects/ModCon/ngetty/datasets/bioreason_sft_reasoning/esm3_cache_2048.pt
TAG="${TAG:?set TAG}"
DP_SHARD="${DP_SHARD:?set DP_SHARD}"
DP_REPLICATE="${DP_REPLICATE:?set DP_REPLICATE}"
NPROC="${NPROC:-12}"
STEPS="${STEPS:-12}"
WARMUP="${WARMUP:-3}"

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
# Validated 32B-FSDP2 memory/MR strategy: default alloc + OFI dereg (no banned:1).
export XPU_USM_ALLOC_SO=${XPU_USM_ALLOC_SO:-}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-garbage_collection_threshold:0.99}
export FI_MR_CACHE_MONITOR=${FI_MR_CACHE_MONITOR:-userfaultfd}
export PYTHONUNBUFFERED=1
export CCL_PROCESS_LAUNCHER=none CCL_ATL_TRANSPORT=ofi CCL_OP_SYNC=1 CCL_WORKER_COUNT=1
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=65536 FI_PROVIDER=cxi CCL_KVS_IFACE=lo

TS=$(date +%H%M%S)
LOGDIR="${PROJDIR}/experiments/bioreason/logs/qwen_hsdp_${TAG}_${TS}"
mkdir -p "${LOGDIR}"; cd "${PROJDIR}"
echo "=== QWEN HSDP ${TAG} | shard=${DP_SHARD} replicate=${DP_REPLICATE} NPROC=${NPROC} host=$(hostname) ===" | tee "${LOGDIR}/run.log"

PYTHONPATH="${BIOREASON_DEPS}:${BIOREASON_SRC}:${PROJDIR}" \
python3 -m torch.distributed.run \
    --standalone --nproc_per_node=${NPROC} \
    --redirects 3 --tee 3 --log-dir "${LOGDIR}/torchelastic" \
    "${RECIPE}" --config "${CONFIG}" \
    output_dir="${LOGDIR}/run_out" \
    esm3_cache_path="${CACHE}" \
    max_steps_per_epoch=${STEPS} \
    save_every_n_steps=100000 \
    data_parallel_shard_dim=${DP_SHARD} \
    data_parallel_replicate_dim=${DP_REPLICATE} \
    2>&1 | tee -a "${LOGDIR}/run.log"
rc=${PIPESTATUS[0]}
echo "=== qwen hsdp ${TAG} exit rc=${rc} ===" | tee -a "${LOGDIR}/run.log"

python3 - "$LOGDIR/run.log" "$WARMUP" "$TAG" "$DP_SHARD" "$DP_REPLICATE" <<'PYEOF' | tee -a "${LOGDIR}/run.log"
import re, sys, statistics
log, warmup, tag, shard, rep = sys.argv[1:6]
txt = open(log, errors="ignore").read()
st = [float(x) for x in re.findall(r"time_per_step_s['\"]?:\s*([0-9.]+)", txt)][int(warmup):]
tk = [float(x) for x in re.findall(r"tokens_per_second_per_gpu['\"]?:\s*([0-9.]+)", txt)][int(warmup):]
crash = bool(re.search(r"banned: 1|could not create a memory|not allocated yet|OutOfMemory|UR_RESULT", txt))
def med(a): return round(statistics.median(a), 2) if a else None
print(f"HSDP_RESULT tag={tag} shard={shard} replicate={rep} n={len(st)} "
      f"median_step_s={med(st)} median_tok_s_per_gpu={med(tk)} crashed={crash}")
PYEOF
exit ${rc}

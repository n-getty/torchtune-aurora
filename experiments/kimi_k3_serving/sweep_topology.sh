#!/usr/bin/env bash
set -euo pipefail

# Run a reproducible vLLM serve sweep against an already-running server.
# Restart the server between cells when changing EP or batched-token settings.

SERVER_URL=${SERVER_URL:-http://127.0.0.1:8000}
MODEL=${MODEL:?Set MODEL to the served model name/path}
TOKENIZER=${TOKENIZER:-$MODEL}
SERVED_MODEL=${SERVED_MODEL:-$MODEL}
BLOCKS=${BLOCKS:-none}
OUT=${OUT:-results.tsv}
PROMPTS=${PROMPTS:-128}
INPUT_LEN=${INPUT_LEN:-1024}
OUTPUT_LEN=${OUTPUT_LEN:-512}
REQUEST_RATE=${REQUEST_RATE:-inf}
REPEATS=${REPEATS:-3}
CONCURRENCY_LIST=${CONCURRENCY_LIST:-${MAX_SEQS_LIST:-"1 2 4"}}
SERVER_MAX_NUM_SEQS=${SERVER_MAX_NUM_SEQS:?Set to the running server's max_num_seqs}
SERVER_MAX_BATCHED_TOKENS=${SERVER_MAX_BATCHED_TOKENS:?Set to the running server's max_num_batched_tokens}
SERVER_EP=${SERVER_EP:-0}
SERVER_TP=${SERVER_TP:-32}
SERVER_PP=${SERVER_PP:-1}
SERVER_GPU_MEM_UTIL=${SERVER_GPU_MEM_UTIL:?Set to the running server's gpu_memory_utilization}
SERVER_MAX_MODEL_LEN=${SERVER_MAX_MODEL_LEN:?Set to the running server's max_model_len}
DAOS_PATH=${DAOS_PATH:-NA}
BLOCK_PROFILE=${BLOCK_PROFILE:?Set to the production block_profile.json path}
VLLM_SRC=${VLLM_SRC:-/flare/ModCon/ngetty/vllm-xpu-src}
RUN_DIR=${RUN_DIR:-$(dirname "$OUT")/runs/$(date +%Y%m%d_%H%M%S)}
SERVER_LOG=${SERVER_LOG:-}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HEALTH_SCANNER="$SCRIPT_DIR/check_k3_serving_health.sh"

command -v vllm >/dev/null || { echo "ERROR: vllm is not on PATH" >&2; exit 1; }
export no_proxy="${no_proxy:+$no_proxy,}localhost,127.0.0.1"
export NO_PROXY="$no_proxy"
curl --noproxy "*" --fail --silent "$SERVER_URL/health" >/dev/null || { echo "ERROR: server is unhealthy" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"
mkdir -p "$RUN_DIR"
HEADER='timestamp\tjob_id\tnodes\tmodel\ttp\tpp\tep\tvllm_commit\tvllm_diff_sha256\tframework_versions\tdaos_path\tblock_profile\tblocks\tgpu_memory_utilization\tmax_model_len\tserver_max_num_seqs\tserver_max_num_batched_tokens\tmax_concurrency\trepeat\trc\tfailures\trequest_throughput\toutput_tok_s\tpeak_output_tok_s\tmean_ttft_ms\tmean_tpot_ms\tresult_log\tserver_log'
if [[ ! -s "$OUT" ]]; then
    printf '%b\n' "$HEADER" >"$OUT"
elif [[ "$(head -1 "$OUT")" != "$HEADER" ]]; then
    echo "ERROR: existing results schema differs; choose a new OUT path: $OUT" >&2
    exit 2
fi
NODES=$(sort -u "${PBS_NODEFILE:?PBS_NODEFILE required}" | tr '\n' ',')
JOB_ID=${PBS_JOBID:-NA}
VLLM_COMMIT=$(git -C "$VLLM_SRC" rev-parse HEAD 2>/dev/null || printf NA)
VLLM_DIFF_SHA256=$(git -C "$VLLM_SRC" diff --binary HEAD 2>/dev/null | sha256sum | awk '{print $1}')
export MODEL SERVER_TP SERVER_PP SERVER_EP BLOCKS DAOS_PATH BLOCK_PROFILE VLLM_COMMIT VLLM_DIFF_SHA256
export SERVED_MODEL
FRAMEWORK_VERSIONS=$(${PYTHON:-python} - <<'PY'
import importlib.metadata
import json
packages = ("torch", "transformers", "ray", "vllm", "triton")
versions = {}
for package in packages:
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        versions[package] = None
print(json.dumps(versions, sort_keys=True))
PY
)
export FRAMEWORK_VERSIONS NODES
RUN_METADATA="$RUN_DIR/metadata.json"
${PYTHON:-python} - "$RUN_METADATA" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

metadata = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "job_id": os.environ.get("PBS_JOBID", "NA"),
    "nodes": os.environ.get("NODES", ""),
    "model": os.environ["MODEL"],
    "model_source": os.environ["MODEL"],
    "served_model": os.environ["SERVED_MODEL"],
    "framework_versions": os.environ["FRAMEWORK_VERSIONS"],
    "server": {
        "tp": int(os.environ["SERVER_TP"]),
        "pp": int(os.environ["SERVER_PP"]),
        "ep": int(os.environ["SERVER_EP"]),
        "blocks": os.environ["BLOCKS"],
    },
    "vllm_commit": os.environ.get("VLLM_COMMIT", "NA"),
    "vllm_diff_sha256": os.environ.get("VLLM_DIFF_SHA256", "NA"),
    "daos_path": os.environ["DAOS_PATH"],
    "block_profile": os.environ["BLOCK_PROFILE"],
}
with open(sys.argv[1], "w") as handle:
    json.dump(metadata, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

metric() {
    local label=$1
    local log=$2
    awk -F: -v label="$label" '$1 ~ "^[[:space:]]*" label "[[:space:]]*$" {gsub(/[[:space:]]/, "", $2); print $2; exit}' "$log"
}

for concurrency in $CONCURRENCY_LIST; do
    for repeat in $(seq 1 "$REPEATS"); do
    result_log="$RUN_DIR/serve_${concurrency}_repeat_${repeat}.log"
    # --trust-remote-code is required for models whose tokenizer ships custom
    # code (K3: tokenization_kimi.py). The bench client loads the tokenizer in
    # its OWN process to build the random dataset, so the server having been
    # started with trust_remote_code=True does not cover it -- without this the
    # very first cell dies in get_tokenizer() before issuing a single request.
    args=(bench serve --backend openai --base-url "$SERVER_URL" --model "$SERVED_MODEL"
        --tokenizer "$TOKENIZER" ${TRUST_REMOTE_CODE:+--trust-remote-code}
        --num-prompts "$PROMPTS" --dataset-name random --random-input-len "$INPUT_LEN"
        --random-output-len "$OUTPUT_LEN" --request-rate "${REQUEST_RATE:-inf}"
        --max-concurrency "$concurrency")
    start=$(date +%s)
    set +e
    vllm "${args[@]}" >"$result_log" 2>&1
    rc=$?
    set -e
    output_tok_s=$(metric 'Output token throughput \\(tok/s\\)' "$result_log")
    peak_output_tok_s=$(metric 'Peak output token throughput \\(tok/s\\)' "$result_log")
    request_throughput=$(metric 'Request throughput \\(req/s\\)' "$result_log")
    mean_ttft_ms=$(metric 'Mean TTFT \\(ms\\)' "$result_log")
    mean_tpot_ms=$(metric 'Mean TPOT \\(ms\\)' "$result_log")
    failures=0
    [[ $rc -eq 0 ]] || failures=1
    [[ -n "${output_tok_s:-}" && -n "${mean_ttft_ms:-}" && -n "${mean_tpot_ms:-}" ]] || failures=1
    # A vLLM bench client can report a clean rc + real numbers while the
    # server it hit already crashed mid-cell (e.g. the request that tripped
    # it failed, but earlier requests in the same window succeeded and
    # dominate the average) -- do not trust a throughput number without
    # scanning the SERVER's own log for the K3 crash signatures, not just
    # the bench client's exit code. See check_k3_serving_health.sh.
    if [[ -n "$SERVER_LOG" && -f "$SERVER_LOG" && -x "$HEALTH_SCANNER" ]]; then
        if ! "$HEALTH_SCANNER" "$SERVER_LOG" >"$RUN_DIR/health_${concurrency}_repeat_${repeat}.log" 2>&1; then
            echo "server health DEGRADED during concurrency=$concurrency repeat=$repeat; see $RUN_DIR/health_${concurrency}_repeat_${repeat}.log"
            failures=1
        fi
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date -Is)" "$JOB_ID" "$NODES" "$MODEL" "$SERVER_TP" "$SERVER_PP" \
        "$SERVER_EP" "$VLLM_COMMIT" "$VLLM_DIFF_SHA256" "$FRAMEWORK_VERSIONS" \
        "$DAOS_PATH" "$BLOCK_PROFILE" "$BLOCKS" "$SERVER_GPU_MEM_UTIL" "$SERVER_MAX_MODEL_LEN" \
        "$SERVER_MAX_NUM_SEQS" "$SERVER_MAX_BATCHED_TOKENS" "$concurrency" "$repeat" "$rc" \
        "$failures" "${request_throughput:-NA}" "${output_tok_s:-NA}" \
        "${peak_output_tok_s:-NA}" "${mean_ttft_ms:-NA}" "${mean_tpot_ms:-NA}" \
        "$result_log" "${SERVER_LOG:-NA}" >>"$OUT"
    echo "cell concurrency=$concurrency repeat=$repeat server_seqs=$SERVER_MAX_NUM_SEQS server_tokens=$SERVER_MAX_BATCHED_TOKENS server_ep=$SERVER_EP rc=$rc elapsed=$(( $(date +%s) - start ))s"
    if [[ $failures -ne 0 ]]; then
        tail -40 "$result_log"
        exit "${rc:-1}"
    fi
    done
done

${PYTHON:-python} - "$OUT" "$REPEATS" <<'PY'
import csv
import math
import sys

path, repeats = sys.argv[1], int(sys.argv[2])
rows = list(csv.DictReader(open(path), delimiter="\t"))
for concurrency in sorted({row["max_concurrency"] for row in rows}):
    values = [
        float(row["output_tok_s"])
        for row in rows
        if row["max_concurrency"] == concurrency
        and row["repeat"] in {str(index) for index in range(1, repeats + 1)}
        and row["failures"] == "0"
        and row["output_tok_s"] not in {"", "NA"}
    ]
    if len(values) < repeats:
        raise SystemExit(f"missing successful repeats for concurrency={concurrency}")
    mean = sum(values) / len(values)
    spread = (max(values) - min(values)) / mean if mean else math.inf
    if spread > 0.05:
        raise SystemExit(
            f"throughput spread exceeds 5% for concurrency={concurrency}: {values}"
        )
print("repeat_spread_check=pass")
PY

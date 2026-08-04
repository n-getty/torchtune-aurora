#!/usr/bin/env bash
set -euo pipefail

# Run a reproducible vLLM serve sweep against an already-running server.
# Restart the server between cells when changing EP or batched-token settings.

SERVER_URL=${SERVER_URL:-http://127.0.0.1:8000}
MODEL=${MODEL:?Set MODEL to the served model name/path}
TOKENIZER=${TOKENIZER:-$MODEL}
SERVED_MODEL=${SERVED_MODEL:-$MODEL}
BLOCKS=${BLOCKS:?Set BLOCKS to the server's num_gpu_blocks_override}
OUT=${OUT:-results.tsv}
PROMPTS=${PROMPTS:-128}
INPUT_LEN=${INPUT_LEN:-1024}
OUTPUT_LEN=${OUTPUT_LEN:-512}
REQUEST_RATE=${REQUEST_RATE:-inf}
MAX_SEQS_LIST=${MAX_SEQS_LIST:-"16 32 64"}
SERVER_MAX_BATCHED_TOKENS=${SERVER_MAX_BATCHED_TOKENS:?Set to the running server's max_num_batched_tokens}
SERVER_EP=${SERVER_EP:-0}
RUN_DIR=${RUN_DIR:-$(dirname "$OUT")/runs/$(date +%Y%m%d_%H%M%S)}
SERVER_LOG=${SERVER_LOG:-}

command -v vllm >/dev/null || { echo "ERROR: vllm is not on PATH" >&2; exit 1; }
export no_proxy="${no_proxy:+$no_proxy,}localhost,127.0.0.1"
export NO_PROXY="$no_proxy"
curl --noproxy "*" --fail --silent "$SERVER_URL/health" >/dev/null || { echo "ERROR: server is unhealthy" >&2; exit 1; }
mkdir -p "$(dirname "$OUT")"
mkdir -p "$RUN_DIR"
if [[ ! -s "$OUT" ]]; then
    printf 'timestamp\tnodes\tmodel\tblocks\tmax_num_seqs\tserver_max_num_batched_tokens\tserver_ep\trc\toutput_tok_s\tpeak_output_tok_s\tmean_ttft_ms\tmean_tpot_ms\tresult_log\tserver_log\n' >"$OUT"
fi
NODES=$(sort -u "${PBS_NODEFILE:?PBS_NODEFILE required}" | tr '\n' ',')

metric() {
    local label=$1
    local log=$2
    awk -F: -v label="$label" '$1 ~ "^[[:space:]]*" label "[[:space:]]*$" {gsub(/[[:space:]]/, "", $2); print $2; exit}' "$log"
}

for seqs in $MAX_SEQS_LIST; do
    result_log="$RUN_DIR/serve_${seqs}.log"
    args=(bench serve --backend openai --base-url "$SERVER_URL" --model "$SERVED_MODEL"
        --tokenizer "$TOKENIZER"
        --num-prompts "$PROMPTS" --dataset-name random --random-input-len "$INPUT_LEN"
        --random-output-len "$OUTPUT_LEN" --request-rate "${REQUEST_RATE:-inf}"
        --max-concurrency "$seqs")
    start=$(date +%s)
    set +e
    vllm "${args[@]}" >"$result_log" 2>&1
    rc=$?
    set -e
    output_tok_s=$(metric 'Output token throughput \\(tok/s\\)' "$result_log")
    peak_output_tok_s=$(metric 'Peak output token throughput \\(tok/s\\)' "$result_log")
    mean_ttft_ms=$(metric 'Mean TTFT \\(ms\\)' "$result_log")
    mean_tpot_ms=$(metric 'Mean TPOT \\(ms\\)' "$result_log")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$(date -Is)" "$NODES" "$MODEL" "$BLOCKS" "$seqs" \
        "$SERVER_MAX_BATCHED_TOKENS" "$SERVER_EP" "$rc" "${output_tok_s:-NA}" \
        "${peak_output_tok_s:-NA}" "${mean_ttft_ms:-NA}" "${mean_tpot_ms:-NA}" \
        "$result_log" "${SERVER_LOG:-NA}" >>"$OUT"
    echo "cell seqs=$seqs server_tokens=$SERVER_MAX_BATCHED_TOKENS server_ep=$SERVER_EP rc=$rc elapsed=$(( $(date +%s) - start ))s"
    if [[ $rc -ne 0 ]]; then
        tail -40 "$result_log"
        exit "$rc"
    fi
done

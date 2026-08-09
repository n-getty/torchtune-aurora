#!/usr/bin/env bash
set -uo pipefail

# Push K3 to >100 output tok/s by going wide on concurrency.
#
# Why this shape, from what is now measured on hardware (job 8745911):
#
#   * The decode step is FIXED-COST: 8x the work costs 1.22x the wall time
#     (16.9s -> 20.7s for 16 steps), i.e. ~1.29 s/step at c=8 and tokens-per-
#     step == concurrency. Throughput is therefore ~linear in concurrency, and
#     100 tok/s needs c ~= 130 if the step time holds.
#   * The KV pool is the thing that used to stop us going wide: 11 of 32 ranks
#     mis-profile their free memory and vLLM clamps everyone to the smallest,
#     leaving 69 blocks / 10,093 tokens. --diagnostic-blocks bypasses that.
#     Healthy ranks computed 1331-1724 blocks, so 1200 is inside what the
#     hardware genuinely has (5.93 GiB/rank) with margin under the 1331 floor.
#   * PREFILL is the trap: kda.py's "vectorized" prefill batches across
#     sequences but still loops per token, so a 1024-token prompt is ~70k
#     einsum groups per layer-pass and never finishes. Decode IS batched.
#     So: SHORT prompts, LONG generations -- which is also the regime where
#     output tok/s is the honest metric.
#
# At in=32 / out=512, each sequence needs 3 blocks (576 tokens), so c=192 needs
# ~110,592 tokens against the 230,400 the 1200-block pool provides. 2x margin.
#
# Known ceiling to watch: vLLM-on-XPU has died after an exact number of
# completions in past runs (89 with async scheduling off). c=192 x 2 rounds =
# 384 requests crosses that several times over, so the ladder runs
# HIGH-CONCURRENCY FIRST (see LADDER below) and reports each rung separately,
# continuing past a failure. That way the rung that decides the >100 tok/s
# question runs on a fresh engine, and a death costs only the cheaper rungs.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_BASE=${RUN_BASE:?Set RUN_BASE}
URL=${URL:-http://127.0.0.1:8000}
MODEL=${MODEL:-Kimi-K3}
PYTHON=${PYTHON:-/flare/ModCon/ngetty/venvs/kimi-k3-xpu-framework/bin/python}
TOKENIZER=${TOKENIZER:-/tmp/ngetty/AuroraGPT/prism_models}
INPUT_LEN=${INPUT_LEN:-32}
OUTPUT_LEN=${OUTPUT_LEN:-512}
# DESCENDING on purpose. vLLM-on-XPU has died after an exact number of
# completions in past runs (89 with async scheduling off, 29 with it on -- see
# memory project_vllm_xpu_hard_request_count_limit_20260807). Any ladder that
# reaches c=192 crosses that count many times over, so if the engine is going
# to die it will die partway through. Running high-concurrency FIRST means the
# rung that decides whether we clear 100 tok/s executes on a fresh engine, and
# a later death costs only the less important low-c rungs.
LADDER=${LADDER:-"256 128 64 32"}
SERVER_LOG=${SERVER_LOG:-$RUN_BASE/server/server.log}
OUT=${OUT:-$RUN_BASE/highconc_results.tsv}

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1"
export PATH="$(dirname "$PYTHON"):$PATH"

curl --noproxy '*' --fail --silent "$URL/health" >/dev/null || {
    echo "ERROR: server unhealthy" >&2
    exit 1
}

# Correctness before any throughput claim. A big KV pool fails late and quietly.
echo "=== correctness gate ==="
"$PYTHON" "$SCRIPT_DIR/probe_batched_nan_logprobs.py" \
    --url "$URL" --model "$MODEL" --max-tokens 24 \
    --out "$RUN_BASE/highconc_correctness.json" || {
    echo "CORRECTNESS FAILED -- refusing to report throughput" >&2
    exit 1
}

printf 'concurrency\tprompts\trc\toutput_tok_s\tmean_ttft_ms\tmean_tpot_ms\tcompleted\tlog\n' >"$OUT"

for concurrency in $LADDER; do
    prompts=$((concurrency * 2))
    log="$RUN_BASE/highconc_c${concurrency}.log"
    echo "=== c=$concurrency prompts=$prompts in=$INPUT_LEN out=$OUTPUT_LEN ==="
    start=$(date +%s)
    "$PYTHON" -m vllm.entrypoints.cli.main bench serve \
        --backend openai --base-url "$URL" --model "$MODEL" \
        --tokenizer "$TOKENIZER" --trust-remote-code \
        --num-prompts "$prompts" --dataset-name random \
        --random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN" \
        --request-rate inf --max-concurrency "$concurrency" \
        --ignore-eos \
        >"$log" 2>&1
    rc=$?
    grab() { awk -F: -v k="$1" '$0 ~ k {gsub(/[^0-9.]/,"",$2); print $2; exit}' "$log"; }
    tok_s=$(grab 'Output token throughput')
    ttft=$(grab 'Mean TTFT')
    tpot=$(grab 'Mean TPOT')
    done_n=$(grab 'Successful requests')
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$concurrency" "$prompts" "$rc" "${tok_s:-NA}" "${ttft:-NA}" \
        "${tpot:-NA}" "${done_n:-NA}" "$log" >>"$OUT"
    echo "  rc=$rc output_tok_s=${tok_s:-NA} elapsed=$(( $(date +%s) - start ))s"
    # Keep climbing even if a rung dies: lower rungs are still real numbers.
done

echo
echo "=== results ==="
column -t -s $'\t' "$OUT"
if [[ -x "$SCRIPT_DIR/check_k3_serving_health.sh" && -f "$SERVER_LOG" ]]; then
    "$SCRIPT_DIR/check_k3_serving_health.sh" "$SERVER_LOG" \
        >"$RUN_BASE/health_after_highconc.log" 2>&1 \
        && echo "post-run health: GREEN" || echo "post-run health: DEGRADED"
fi
best=$(awk -F'\t' 'NR>1 && $4!="NA" {if ($4+0>m) m=$4+0} END {print m+0}' "$OUT")
echo "best output tok/s = $best  (target 100; Qwen-480B reference 106.19)"

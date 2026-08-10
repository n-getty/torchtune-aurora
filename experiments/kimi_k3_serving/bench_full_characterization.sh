#!/usr/bin/env bash
set -uo pipefail

# Full K3 throughput characterization from ONE server start.
#
# Answers three separate questions that are easy to conflate:
#
#   A. Peak aggregate throughput  -- how many tok/s can the box emit at all,
#      which is the number comparable to Qwen3-Coder-480B's 106.19 tok/s.
#   B. Single-user experience     -- what one interactive user actually feels.
#      These diverge hard: at c=128 the box did 65.20 tok/s aggregate but
#      TPOT was 1909 ms, i.e. ~0.5 tok/s per user. A headline aggregate number
#      alone is misleading, so both are always reported.
#   C. Realistic shapes           -- does the peak survive longer prompts, or
#      was it an artifact of a 32-token prompt?
#
# Ordering is deliberate:
#   * The correctness gate runs first; no throughput number is reported if the
#     model is not producing sane output.
#   * Cells run cheapest-first WITHIN each phase but the phases themselves run
#     peak-first, because vLLM-on-XPU has died after a fixed number of
#     completions in past runs. If the engine dies partway, the cells that
#     matter most are already on disk.
#   * Every cell appends to the TSV immediately, and a failed cell does not
#     abort the suite.
#
# LONG-PROMPT WARNING: KDA's prefill path in kda.py loops per token (batched
# across sequences, serial across tokens), so a 1024-token prompt is ~70k
# einsum groups per layer pass. The 1024-in cells are expected to be very slow
# and may not finish; they are placed last and time-boxed for that reason.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_BASE=${RUN_BASE:?Set RUN_BASE}
URL=${URL:-http://127.0.0.1:8000}
MODEL=${MODEL:-Kimi-K3}
PYTHON=${PYTHON:-/flare/ModCon/ngetty/venvs/kimi-k3-xpu-framework/bin/python}
TOKENIZER=${TOKENIZER:-/tmp/ngetty/AuroraGPT/prism_models}
MAX_SEQS=${MAX_SEQS:?Set MAX_SEQS to the server max_num_seqs}
NODES=${NODES:-3}
OUT=${OUT:-$RUN_BASE/characterization.tsv}
CELL_TIMEOUT=${CELL_TIMEOUT:-2400}
SERVER_LOG=${SERVER_LOG:-$RUN_BASE/server/server.log}

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1"
export PATH="$(dirname "$PYTHON"):$PATH"
# K3's remote-code tokenizer imports a sibling encoding_k3.py that
# transformers resolves against the CLIENT's sys.path, not the model dir.
export PYTHONPATH="$TOKENIZER${PYTHONPATH:+:$PYTHONPATH}"

curl --noproxy '*' --fail --silent "$URL/health" >/dev/null || {
    echo "ERROR: server unhealthy" >&2; exit 1; }

echo "=== correctness gate (blocks all throughput reporting) ==="
"$PYTHON" "$SCRIPT_DIR/probe_batched_nan_logprobs.py" \
    --url "$URL" --model "$MODEL" --max-tokens 24 \
    --out "$RUN_BASE/correctness.json" || {
    echo "CORRECTNESS FAILED -- not reporting throughput" >&2; exit 1; }

if [[ ! -s "$OUT" ]]; then
    printf 'phase\tconcurrency\tprompts\tinput_len\toutput_len\trc\tsuccessful\tfailed\toutput_tok_s\tper_user_tok_s\tmean_ttft_ms\tmean_tpot_ms\tduration_s\n' >"$OUT"
fi

cell() {
    local phase=$1 conc=$2 prompts=$3 ilen=$4 olen=$5
    local log="$RUN_BASE/cell_${phase}_c${conc}_i${ilen}_o${olen}.log"
    echo "--- $phase: c=$conc prompts=$prompts in=$ilen out=$olen ---"
    timeout "$CELL_TIMEOUT" "$PYTHON" -m vllm.entrypoints.cli.main bench serve \
        --backend openai --base-url "$URL" --model "$MODEL" \
        --tokenizer "$TOKENIZER" --trust-remote-code \
        --num-prompts "$prompts" --dataset-name random \
        --random-input-len "$ilen" --random-output-len "$olen" \
        --request-rate inf --max-concurrency "$conc" --ignore-eos \
        >"$log" 2>&1
    local rc=$?
    g() { awk -F: -v k="$1" '$0 ~ k {gsub(/[^0-9.]/,"",$2); print $2; exit}' "$log"; }
    local tok_s ttft tpot ok bad dur
    tok_s=$(g 'Output token throughput'); ttft=$(g 'Mean TTFT'); tpot=$(g 'Mean TPOT')
    ok=$(g 'Successful requests'); bad=$(g 'Failed requests'); dur=$(g 'Benchmark duration')
    # Per-user tok/s is the number an interactive user feels: 1000/TPOT.
    local per_user="NA"
    [[ -n "${tpot:-}" ]] && per_user=$("$PYTHON" -c "print(f'{1000.0/$tpot:.3f}')" 2>/dev/null || echo NA)
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$phase" "$conc" "$prompts" "$ilen" "$olen" "$rc" "${ok:-0}" "${bad:-NA}" \
        "${tok_s:-NA}" "$per_user" "${ttft:-NA}" "${tpot:-NA}" "${dur:-NA}" >>"$OUT"
    echo "    rc=$rc ok=${ok:-0} failed=${bad:-NA} aggregate=${tok_s:-NA} tok/s  per_user=${per_user} tok/s  tpot=${tpot:-NA} ms"
}

# ---- Phase A: peak aggregate. Highest concurrency first (engine-death risk).
# NOTE: max_num_seqs >= 192 takes an unrecoverable banned:1 GPU fault after
# exactly 6 requests (5 runs, jobs 8746183/8746327/8746385; independent of
# chunked prefill, block count, and node freshness). Keep MAX_SEQS at 128.
echo "=== PHASE A: peak aggregate throughput (32-in / 512-out) ==="
if [[ $MAX_SEQS -ge 192 ]]; then
    echo "WARNING: MAX_SEQS=$MAX_SEQS is at/above the banned:1 ceiling of 192." >&2
fi
for c in "$MAX_SEQS" $((MAX_SEQS / 2)) $((MAX_SEQS / 4)); do
    [[ $c -ge 1 ]] || continue
    cell peak "$c" $((c * 2)) 32 512
done

# ---- Phase B: single user. Cheap, and the number most often missing.
echo "=== PHASE B: single-user latency (c=1) ==="
cell single 1 4 32 512
cell single 1 4 256 512

# ---- Phase C: realistic shapes at a mid concurrency that is known to work.
echo "=== PHASE C: realistic prompt/output shapes ==="
MID=$((MAX_SEQS / 2)); [[ $MID -ge 1 ]] || MID=1
cell shape "$MID" $((MID * 2)) 256 512
cell shape "$MID" "$MID" 1024 512   # slow: per-token KDA prefill; may time out

echo
echo "=== RESULTS (nodes=$NODES, max_num_seqs=$MAX_SEQS) ==="
column -t -s $'\t' "$OUT"
if [[ -x "$SCRIPT_DIR/check_k3_serving_health.sh" && -f "$SERVER_LOG" ]]; then
    "$SCRIPT_DIR/check_k3_serving_health.sh" "$SERVER_LOG" \
        >"$RUN_BASE/health_after.log" 2>&1 \
        && echo "post-run health: GREEN" || echo "post-run health: DEGRADED"
fi
"$PYTHON" - "$OUT" <<'PY'
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1]), delimiter='\t')
        if r['output_tok_s'] not in ('NA', '')]
if not rows:
    print("no successful cells"); raise SystemExit
best = max(rows, key=lambda r: float(r['output_tok_s']))
print(f"\nPEAK AGGREGATE: {best['output_tok_s']} tok/s at c={best['concurrency']} "
      f"(in={best['input_len']}, out={best['output_len']}) -- target 100, "
      f"Qwen-480B reference 106.19")
single = [r for r in rows if r['concurrency'] == '1']
if single:
    s = single[0]
    print(f"SINGLE USER:    {s['output_tok_s']} tok/s, TPOT {s['mean_tpot_ms']} ms")
    print(f"  -> at peak concurrency each user sees {best['per_user_tok_s']} tok/s")
PY

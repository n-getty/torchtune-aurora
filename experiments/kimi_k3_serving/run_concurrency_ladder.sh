#!/usr/bin/env bash
set -euo pipefail

# Concurrency ladder on `vllm bench serve`, sized so every cell finishes.
#
# WHY NOT sweep_topology.sh DIRECTLY: it applies one PROMPTS value to every
# concurrency cell. The 480B config (PROMPTS=128, OUTPUT_LEN=512) is fine at
# c=32 -- 4 rounds -- but at c=1 it is 128 SEQUENTIAL 512-token generations.
# At K3's ~1.0-1.3 s/decode-step that cell alone is ~19 hours, so the sweep
# appears to hang at "0/128" and burns the whole allocation without producing
# a single number. Observed 2026-08-09 on job 8745911.
#
# Here PROMPTS scales with concurrency (ROUNDS full batches per cell), so each
# cell is the same wall-clock cost regardless of where it sits on the ladder,
# and the ladder measures exactly what Finding 2 predicts: whether the step is
# fixed-cost (throughput ~ linear in concurrency) or not.
#
# Each cell is a separate sweep_topology.sh invocation appending to one TSV.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_BASE=${RUN_BASE:?Set RUN_BASE}
LADDER=${LADDER:-"1 4 16 32"}
ROUNDS=${ROUNDS:-2}
export INPUT_LEN=${INPUT_LEN:-256}
export OUTPUT_LEN=${OUTPUT_LEN:-64}
export REPEATS=${REPEATS:-1}
export OUT=${OUT:-$RUN_BASE/ladder_results.tsv}

for concurrency in $LADDER; do
    prompts=$((concurrency * ROUNDS))
    echo "=== ladder cell: concurrency=$concurrency prompts=$prompts "\
"in=$INPUT_LEN out=$OUTPUT_LEN ==="
    # A failing cell must not abort the ladder: a c=32 OOM/timeout is itself a
    # result, and the lower cells that already succeeded stay on disk.
    if ! CONCURRENCY_LIST="$concurrency" PROMPTS="$prompts" \
        RUN_DIR="$RUN_BASE/ladder_c${concurrency}" \
        "$SCRIPT_DIR/run_step1_sweep.sh"; then
        echo "=== cell concurrency=$concurrency FAILED (continuing) ==="
    fi
done

echo "=== ladder complete ==="
column -t -s $'\t' "$OUT" 2>/dev/null | cut -c1-40,300-460 || cat "$OUT"

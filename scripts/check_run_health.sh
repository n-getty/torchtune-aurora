#!/usr/bin/env bash
# check_run_health.sh — RUN-HEALTH GATE for torchtune Aurora/XPU GRPO runs.
#
# WHY THIS EXISTS (motivating incident, 2026-06-17):
#   A Qwen3-4B full-FT GRPO run reported 274s/step (5x too slow) and the conclusion
#   "LoRA wins on step time" was WRONG. The dense run had silently taken the
#   CHUNKED_BACKWARD path, which lacks the `_orig_reduce_scatter_tensor` bypass, so
#   every reduce_scatter went through the gloo CPU-bounce (D2H -> gloo AllReduce ->
#   H2D, ~130s/backward). CPU tests passed throughout: this was a MEASUREMENT-VALIDITY
#   failure, not a code-correctness one. The fix is a gate that refuses to let a runtime
#   number be trusted until the execution mode is verified healthy.
#   See memory/project_lora_vs_fullft_4b_parity_20260617.md and docs/RESULTS_DISCIPLINE.md.
#
# USAGE:
#   scripts/check_run_health.sh <logfile>                 # single-log verdict
#   scripts/check_run_health.sh --compare <logA> <logB>   # A/B path/transport parity
#   scripts/check_run_health.sh --baseline <size> <secs> [<logfile>]  # monotonicity check
#
# EXIT CODES:
#   0  = GREEN     (safe to trust the runtime number)
#   1  = DEGRADED  (silent degraded mode detected; do NOT trust the number)
#   2  = usage / file error
#
# Dependency-free: bash + grep + awk only. Runs on a login node.
# Robust to torchelastic per-rank prefixes ("[default0]:") and tee'd double-lines.

set -o pipefail

# ----------------------------------------------------------------------------
# Monotonicity baseline table.
#   Known-good steady-state step times (seconds) by model size / topology.
#   SOURCE: docs/status.md "Where we are (one-page stock-take)" + Current Status.
#   These are MONOTONICITY ANCHORS: a smaller model at the same topology cannot be
#   slower than a larger one. Advisory only (baselines drift) -> WARN, never FAIL.
# Format: "label|max_plausible_secs|note"
# ----------------------------------------------------------------------------
baseline_lookup() {
    # $1 = size token (case-insensitive substring match against keys below)
    local key
    key=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')
    case "$key" in
        *agpt*2b*|*2b*)        echo "AGPT-2B|20|~13s/step 2N GSM8K (status.md 2026-06-13)";;
        *3b*)                  echo "Qwen2.5-3B|30|~21s/step 10+2 SHM (status.md)";;
        *4b*)                  echo "dense-4B|75|must be < 32B-2N ceiling (33-67s); AGPT-2B 13s, 3B 21s, LoRA-4B ~54.5s (status.md)";;
        *8b*)                  echo "Qwen3-8B|60|~27s colocate / varies (status.md)";;
        *30b*|*a3b*|*moe*)     echo "Qwen3-30B-A3B|70|54.8s/step G=8 (status.md)";;
        *32b*)                 echo "Qwen3-32B-2N|80|33-67s/step 2N (status.md)";;
        *)                     echo "";;
    esac
}

# ----------------------------------------------------------------------------
# Normalize a log to plain content lines:
#   - strip torchelastic rank prefix "[defaultN]:"
#   - collapse tee'd exact-duplicate consecutive lines
# Emits to stdout. We do NOT dedup non-adjacent lines (different ranks legitimately
# repeat markers); grep -c callers below count distinct events where it matters.
# ----------------------------------------------------------------------------
normalize() {
    sed -E 's/^\[[a-zA-Z]+[0-9]+\]:[[:space:]]?//' "$1"
}

RED=""; GRN=""; YEL=""; RST=""
if [ -t 1 ]; then RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; RST=$'\033[0m'; fi

# ----------------------------------------------------------------------------
# Single-log analysis.  Sets globals: VERDICT (GREEN|DEGRADED), and prints bullets.
# ----------------------------------------------------------------------------
analyze_log() {
    local LOG="$1"
    local degraded=0
    local -a findings=()
    local -a notes=()

    local norm
    norm=$(normalize "$LOG")

    # --- grpo_step path (one-shot rank-0 line) -------------------------------
    # SOURCE: grpo_full_finetune_distributed_xpu.py ~3675 "grpo_step path: %s ..."
    local pathline
    pathline=$(printf '%s\n' "$norm" | grep -m1 "grpo_step path:")
    local gpath="(none)"
    local ep_degree=""
    if [ -n "$pathline" ]; then
        gpath=$(printf '%s' "$pathline" | sed -E 's/.*grpo_step path:[[:space:]]*([A-Z_]+).*/\1/')
        ep_degree=$(printf '%s' "$pathline" | grep -oE 'ep_degree=[0-9]+' | head -1 | cut -d= -f2)
        notes+=("grpo_step path: ${gpath} (ep_degree=${ep_degree:-?}) :: $(printf '%s' "$pathline" | sed -E 's/.*grpo_step path: //')")
    else
        notes+=("grpo_step path: NOT EMITTED (standalone LoRA recipe, or run never reached grpo_step)")
    fi

    # --- gloo CPU-bounce reduce_scatter (THE incident) -----------------------
    # The module-level patch "Patched dist.reduce_scatter_tensor -> gloo" is installed
    # in BOTH dense and LoRA runs (it is EP infrastructure), so its presence ALONE is
    # NOT degradation. The smoking gun is v206 (the non-HSDP gloo PG actually built for
    # the CPU-bounce on a single-replicate run) COMBINED with a path that eats it.
    # SOURCE: distributed.py:929 (patch) ; grpo recipe:613 (v206 PG init) ;
    #         recipe:3796 bypass exists ONLY on SINGLE_BACKWARD, NOT on CHUNKED_BACKWARD.
    local patched v206 cnt_patched mn_colocate
    patched=$(printf '%s\n' "$norm" | grep -c "Patched dist.reduce_scatter_tensor")
    v206=$(printf '%s\n' "$norm" | grep -c "v206:.*_xpu_reduce_scatter_via_allreduce CPU-bounce")
    # Multi-node colocate marker: the recipe builds an explicit cross-node FSDP
    # dp_shard mesh and registers the world-sized gloo PG for reduce_scatter. On
    # a TRUE multi-node run native XCCL reduce_scatter leaks CXI MR handles
    # (banned:1) — the gloo CPU-bounce is the CORRECT, intended path there, not
    # the single-node 274s incident. Detect it so we annotate rather than fail.
    mn_colocate=$(printf '%s\n' "$norm" | grep -c "Multi-node colocate.*built explicit 1D dp_shard mesh")
    cnt_patched=$patched

    if [ "$v206" -gt 0 ]; then
        # CPU-bounce PG is live. Whether it CORRUPTS timing depends on path + EP.
        if [ "${ep_degree:-1}" != "" ] && [ "${ep_degree:-1}" -gt 1 ] 2>/dev/null; then
            notes+=("reduce_scatter: gloo CPU-bounce PG active (v206) on EP run (ep_degree=${ep_degree}) -- EXPECTED for EP; not a timing bug.")
        elif [ "$mn_colocate" -gt 0 ]; then
            notes+=("reduce_scatter: gloo CPU-bounce PG active (v206) on MULTI-NODE colocate run -- EXPECTED (native XCCL reduce_scatter leaks CXI handles cross-node); correct path, not the single-node 274s incident. Step-time reflects the gloo bounce; ACCURACY metrics are unaffected.")
        elif [ "$gpath" = "SINGLE_BACKWARD" ]; then
            notes+=("reduce_scatter: gloo CPU-bounce PG present (v206) but SINGLE_BACKWARD bypasses it (recipe:3796) -- timing OK.")
        elif [ "$gpath" = "CHUNKED_BACKWARD" ]; then
            degraded=1
            findings+=("GLOO CPU-BOUNCE reduce_scatter ACTIVE on CHUNKED_BACKWARD non-EP run (v206 PG built, ep_degree=${ep_degree:-1}).")
            findings+=("  -> The CHUNKED_BACKWARD path has NO _orig_reduce_scatter_tensor bypass; every reduce_scatter")
            findings+=("     goes D2H->gloo-AllReduce->H2D, adding ~130s/backward (2s/layer x ~64 layers).")
            findings+=("  -> This is EXACTLY the 2026-06-17 274s/step incident. The step-time number is CORRUPTED.")
            findings+=("  -> Note: gloo reduce_scatter is expected ONLY on EP runs; on non-EP it corrupts timing.")
            findings+=("  -> Fix: run SINGLE_BACKWARD (TORCHTUNE_USE_CHUNKED_LOSS=1) or use the bypass on chunked.")
        else
            # PACKED or unknown path with v206 active on non-EP: suspicious, flag.
            degraded=1
            findings+=("GLOO CPU-BOUNCE reduce_scatter PG active (v206) on non-EP path '${gpath}'; bypass status unknown -- treat timing as SUSPECT.")
        fi
    else
        if [ "$cnt_patched" -gt 0 ]; then
            notes+=("reduce_scatter: patch installed ($cnt_patched ranks) but no v206 CPU-bounce PG built -> native XCCL reduce_scatter in use (healthy).")
        else
            notes+=("reduce_scatter: no gloo CPU-bounce markers -> native XCCL (healthy).")
        fi
    fi

    # --- varlen requested-but-skipped (silent no-op) -------------------------
    # SOURCE: torchtune/modules/attention_utils.py:65 "varlen=requested-but-skipped (%s)"
    local varlen_skip varlen_eng
    varlen_skip=$(printf '%s\n' "$norm" | grep -c "varlen=requested-but-skipped")
    varlen_eng=$(printf '%s\n' "$norm" | grep -c "varlen=engaged\|varlen=no-grad-only\|varlen no-grad bypass ENGAGED")
    if [ "$varlen_skip" -gt 0 ]; then
        degraded=1
        findings+=("VARLEN requested-but-skipped ($varlen_skip occurrences) -- TORCHTUNE_USE_IPEX_VARLEN was set but the fast path silently no-op'd.")
        findings+=("  -> Any 'varlen speedup' claim for this run is invalid. Grep 'varlen=requested-but-skipped' for the reason (mask present / packing / non-XPU).")
    elif [ "$varlen_eng" -gt 0 ]; then
        notes+=("varlen: engaged ($varlen_eng markers).")
    fi

    # --- banned:1 / PDE / SIGABRT (runtime crash) ----------------------------
    # SOURCE: known XPU L0 crash signatures (CLAUDE.md empty_cache/banned notes,
    #         distributed.py:851 UR_RESULT_ERROR_OUT_OF_RESOURCES).
    local crash
    crash=$(printf '%s\n' "$norm" | grep -iE "banned:[[:space:]]?1|UR_RESULT_ERROR_OUT_OF_RESOURCES|SIGABRT|signal 6|urEventWait|PDE page-fault|page fault" | grep -viE "^#|will |would |could |may |after ~|detect|comment|see |when this" | grep -ivE "distributed.py:|recipe:|\.py:[0-9]" | head -5)
    if [ -n "$crash" ]; then
        degraded=1
        findings+=("RUNTIME CRASH signature(s) detected (banned:1 / UR:40 / SIGABRT):")
        while IFS= read -r cl; do [ -n "$cl" ] && findings+=("    $cl"); done <<< "$crash"
    fi

    # --- empty_cache in loop (UR-handle leak) --------------------------------
    # SOURCE: CLAUDE.md "NEVER call empty_cache() in FSDP training loops".
    # device_empty_cache is a no-op on XPU by design, so a literal call in the loop
    # would show as repeated "empty_cache ... start/done" across steps. Flag if many.
    local ec
    ec=$(printf '%s\n' "$norm" | grep -c "empty_cache gen[0-9].*start")
    if [ "$ec" -gt 4 ]; then
        notes+=("empty_cache: $ec serialized empty_cache markers seen (per-gen serialization, expected on some paths; only a concern if it correlates with banned:1).")
    fi

    # --- TIMING completeness -------------------------------------------------
    # SOURCE: recipe:4644/4771/4951 "TIMING step=%d ..."
    local ntiming
    ntiming=$(printf '%s\n' "$norm" | grep "TIMING step=" | sort -u | wc -l)
    if [ "$ntiming" -eq 0 ]; then
        degraded=1
        findings+=("NO 'TIMING step=' lines -- the run never completed a step. Any step-time number is fabricated/partial.")
    else
        # Report the steady-state step time (median-ish: drop step 0 if >1).
        local steptimes
        steptimes=$(printf '%s\n' "$norm" | grep "TIMING step=" | grep -oE 'total=[0-9.]+s' | grep -oE '[0-9.]+' | sort -n)
        local nsteps
        nsteps=$(printf '%s\n' "$steptimes" | grep -c .)
        local typ
        typ=$(printf '%s\n' "$steptimes" | awk '{a[NR]=$1} END{if(NR==0)exit; print a[int((NR+1)/2)]}')
        notes+=("TIMING: $ntiming distinct step lines; typical total=${typ}s.")
        LAST_TYP_STEP="$typ"
    fi

    # --- emit ----------------------------------------------------------------
    if [ "$degraded" -eq 1 ]; then
        VERDICT="DEGRADED"
        echo "${RED}===================== DEGRADED =====================${RST}"
        echo "Log: $LOG"
        echo "${RED}DEGRADED FINDINGS:${RST}"
        for f in "${findings[@]}"; do echo "  - $f"; done
    else
        VERDICT="GREEN"
        echo "${GRN}======================= GREEN ======================${RST}"
        echo "Log: $LOG"
    fi
    if [ "${#notes[@]}" -gt 0 ]; then
        echo "Notes:"
        for n in "${notes[@]}"; do echo "  . $n"; done
    fi
}

# ----------------------------------------------------------------------------
# Monotonicity check (advisory).
# ----------------------------------------------------------------------------
monotonicity_check() {
    local size="$1" secs="$2"
    local row; row=$(baseline_lookup "$size")
    if [ -z "$row" ]; then
        echo "${YEL}[monotonicity] no baseline row for size '$size' (known: 2b/3b/4b/8b/30b/32b); skipping.${RST}"
        return 0
    fi
    local label maxs note
    label=$(printf '%s' "$row" | cut -d'|' -f1)
    maxs=$(printf '%s' "$row" | cut -d'|' -f2)
    note=$(printf '%s' "$row" | cut -d'|' -f3)
    # integer-ish comparison via awk
    local over
    over=$(awk -v s="$secs" -v m="$maxs" 'BEGIN{print (s > m) ? 1 : 0}')
    if [ "$over" = "1" ]; then
        echo "${YEL}[monotonicity] WARN: ${label} measured ${secs}s/step > plausible ceiling ~${maxs}s.${RST}"
        echo "${YEL}              ${note}${RST}"
        echo "${YEL}              A smaller/equal model cannot exceed a larger one's step time -- INVESTIGATE before trusting this number.${RST}"
    else
        echo "[monotonicity] OK: ${label} ${secs}s/step within plausible bound (<= ~${maxs}s). (${note})"
    fi
}

# ----------------------------------------------------------------------------
# Compare mode: assert both legs took same grpo_step path AND same RS transport.
# ----------------------------------------------------------------------------
compare_logs() {
    local A="$1" B="$2"
    [ -f "$A" ] || { echo "ERROR: missing $A" >&2; exit 2; }
    [ -f "$B" ] || { echo "ERROR: missing $B" >&2; exit 2; }

    extract_path() {
        normalize "$1" | grep -m1 "grpo_step path:" | sed -E 's/.*grpo_step path:[[:space:]]*([A-Z_]+).*/\1/'
    }
    # transport state: ACTIVE if v206 CPU-bounce PG built on non-EP; else XCCL.
    extract_transport() {
        local n; n=$(normalize "$1")
        local v206 ep
        v206=$(printf '%s\n' "$n" | grep -c "v206:.*_xpu_reduce_scatter_via_allreduce CPU-bounce")
        ep=$(printf '%s\n' "$n" | grep -m1 "grpo_step path:" | grep -oE 'ep_degree=[0-9]+' | cut -d= -f2)
        local path; path=$(printf '%s\n' "$n" | grep -m1 "grpo_step path:" | sed -E 's/.*grpo_step path:[[:space:]]*([A-Z_]+).*/\1/')
        if [ "${v206:-0}" -gt 0 ]; then
            if [ "${ep:-1}" -gt 1 ] 2>/dev/null; then echo "gloo-CPU-bounce(EP-expected)";
            elif [ "$path" = "SINGLE_BACKWARD" ]; then echo "XCCL(bypassed)";
            else echo "gloo-CPU-bounce(ACTIVE)"; fi
        else
            echo "XCCL"
        fi
    }

    local pA pB tA tB
    pA=$(extract_path "$A"); pB=$(extract_path "$B")
    tA=$(extract_transport "$A"); tB=$(extract_transport "$B")
    [ -z "$pA" ] && pA="(none)"; [ -z "$pB" ] && pB="(none)"

    echo "==================== A/B COMPARE ===================="
    echo "  A: $A"
    echo "       grpo_step path : ${pA}"
    echo "       RS transport   : ${tA}"
    echo "  B: $B"
    echo "       grpo_step path : ${pB}"
    echo "       RS transport   : ${tB}"
    echo "----------------------------------------------------"

    local fail=0
    if [ "$pA" != "$pB" ]; then
        fail=1
        echo "${RED}MISMATCH: grpo_step path differs ('${pA}' vs '${pB}').${RST}"
    fi
    if [ "$tA" != "$tB" ]; then
        fail=1
        echo "${RED}MISMATCH: reduce_scatter transport differs ('${tA}' vs '${tB}').${RST}"
    fi

    if [ "$fail" -eq 1 ]; then
        echo "${RED}A/B INVALID: legs ran under different execution modes -- step-time comparison is apples-to-oranges.${RST}"
        echo "${RED}This is the exact 2026-06-17 mistake (LoRA bypassed gloo, dense did not). Re-run both legs in the same mode.${RST}"
        return 1
    fi
    echo "${GRN}A/B parity OK: both legs same path + same transport. Comparison is valid.${RST}"
    return 0
}

# ----------------------------------------------------------------------------
# Arg parsing
# ----------------------------------------------------------------------------
[ $# -lt 1 ] && { sed -n '5,18p' "$0"; exit 2; }

case "$1" in
    --compare)
        [ $# -ge 3 ] || { echo "usage: $0 --compare <logA> <logB>" >&2; exit 2; }
        compare_logs "$2" "$3"; exit $?;;
    --baseline)
        [ $# -ge 3 ] || { echo "usage: $0 --baseline <size> <secs> [<logfile>]" >&2; exit 2; }
        SIZE="$2"; SECS="$3"; shift 3
        monotonicity_check "$SIZE" "$SECS"
        if [ $# -ge 1 ]; then
            [ -f "$1" ] || { echo "ERROR: no such file: $1" >&2; exit 2; }
            VERDICT=""; analyze_log "$1"
            [ "$VERDICT" = "DEGRADED" ] && exit 1
        fi
        exit 0;;
    -h|--help)
        sed -n '5,18p' "$0"; exit 0;;
    *)
        LOG="$1"
        [ -f "$LOG" ] || { echo "ERROR: no such file: $LOG" >&2; exit 2; }
        VERDICT=""; LAST_TYP_STEP=""
        analyze_log "$LOG"
        # If a --baseline-style size is inferrable from the path, hint monotonicity.
        if [ -n "$LAST_TYP_STEP" ]; then
            sz=$(printf '%s' "$LOG" | grep -oiE '[0-9]+b|a3b|moe' | head -1)
            [ -n "$sz" ] && { echo; monotonicity_check "$sz" "$LAST_TYP_STEP"; }
        fi
        [ "$VERDICT" = "DEGRADED" ] && exit 1
        exit 0;;
esac

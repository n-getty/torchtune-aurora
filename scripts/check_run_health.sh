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
#   scripts/check_run_health.sh --preflight <config.yaml>  # PRE-LAUNCH gate (run BEFORE mpiexec)
#
# --preflight reads the resolved config YAML AND the launcher's effective env-var
# overrides (GRPO_SAMPLES, MAX_GEN_TOKENS, FORWARD_BATCH_SIZE, REF_FORWARD_BATCH_SIZE,
# LORA_USE_RUNTIME, VLLM_WORKER_EXT, ...) and REFUSES known-bad launch points that
# CLAUDE.md / memory document as banned:1 boundaries or silent-degradation traps —
# BEFORE a node-hour is spent. It encodes prose knowledge as an executable assertion.
#
# EXIT CODES:
#   0  = GREEN     (safe to trust the runtime number / safe to launch)
#   1  = DEGRADED  (silent degraded mode detected; do NOT trust the number)
#                  OR REFUSED (preflight found a documented known-bad launch point)
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
# PRE-LAUNCH preflight gate.
#
# Turns the validated-envelope knowledge that lives in prose (CLAUDE.md tables +
# memory/*.md) into an executable assertion that fires BEFORE mpiexec/torchrun, so
# a node-hour is never spent on a launch point already documented as banned:1 or as
# a silent-degradation trap.
#
# Effective-value model: a YAML value is the DEFAULT; the launcher overrides many of
# them on the CLI from env vars (grpo_samples=${GRPO_SAMPLES}, ...). So preflight reads
# the YAML for the baseline AND honors the same env vars the launcher uses, evaluating
# the EFFECTIVE launch point (env override wins, exactly as the recipe sees it).
#
# Checks are DATA-DRIVEN: each is a small function appended to PF_FINDINGS as
# "SEVERITY|message". REFUSE => exit 1 (unless an explicit override env is set);
# WARN => printed loudly but exit 0. Add a new check by writing one pf_check_* fn
# and calling it from run_preflight().
#
# Dependency-free: bash + grep + awk + sed (login-node python3 is 3.6; we do not
# rely on it). YAML is parsed with grep/sed for the flat scalar keys we care about.
# ----------------------------------------------------------------------------

# yaml_scalar <file> <key> : echo the scalar value of a top-level-ish `key: value`
# line (strips inline `# comments`, surrounding quotes, whitespace). Matches the
# first non-comment occurrence. Good enough for the flat RL keys in these configs.
yaml_scalar() {
    local file="$1" key="$2"
    grep -E "^[[:space:]]*${key}:[[:space:]]" "$file" 2>/dev/null \
        | grep -vE "^[[:space:]]*#" \
        | head -1 \
        | sed -E "s/^[[:space:]]*${key}:[[:space:]]*//; s/[[:space:]]*#.*\$//; s/^[\"']//; s/[\"']\$//; s/[[:space:]]*\$//"
}

# eff <env_var_name> <yaml_value> : effective value = env override if set & non-empty,
# else the YAML default. Mirrors the launcher's `key=${VAR:-default}` precedence.
eff() {
    local envname="$1" yamlval="$2" envval
    envval="$(printf '%s' "${!envname-}")"
    if [ -n "$envval" ]; then printf '%s' "$envval"; else printf '%s' "$yamlval"; fi
}

# is_int <s> : true if s is a non-negative integer.
is_int() { case "$1" in ''|*[!0-9]*) return 1;; *) return 0;; esac; }

PF_FINDINGS=()
pf_add() { PF_FINDINGS+=("$1|$2"); }   # severity|message

# --- Check 1: G x max_gen banned:1 boundary (LoRA 4B/2N) ---------------------
# SOURCE: CLAUDE.md "config's paper G=24/max_gen=512 is the documented banned:1
#   boundary"; YAML header MEMORY BOUNDARY note; memory project_lora_grpo_4b_envelope_20260505
#   ("G=24 max_gen=512 banned:1 step 1 (IPC eviction ceiling)").
# Validated-safe: G<=16 with max_gen<=384.
pf_check_g_maxgen_boundary() {
    local g="$1" mg="$2"
    is_int "$g" || return 0
    is_int "$mg" || return 0
    if [ "$g" -ge 24 ] && [ "$mg" -ge 512 ]; then
        if [ "${PREFLIGHT_ALLOW_BANNED:-0}" = "1" ]; then
            pf_add WARN "G=${g} x max_generated_tokens=${mg} is the DOCUMENTED banned:1 boundary (LoRA 4B/2N, IPC-handle eviction at step 0->1). Proceeding only because PREFLIGHT_ALLOW_BANNED=1. Validated-safe envelope is G<=16, max_gen<=384."
        else
            pf_add REFUSE "G=${g} x max_generated_tokens=${mg} is the DOCUMENTED banned:1 boundary for LoRA 4B/2N (OOM at the step 0->1 vLLM+ref_fwd IPC-handle eviction; lora_status.md / project_lora_grpo_4b_envelope_20260505). Use G=8/max_gen=384 (validated-safe ~52-53s/step). To force, set PREFLIGHT_ALLOW_BANNED=1."
        fi
    fi
}

# --- Check 2: fbs lowered without ref_forward_batch_size set ------------------
# SOURCE: CLAUDE.md "ref_forward_batch_size sharp edge"; memory
#   feedback_ref_forward_batch_size_default_trap ("0.2s->100s, 500x").
# If the launcher drops forward_batch_size BELOW the YAML default and does NOT set
# ref_forward_batch_size explicitly, ref_fwd runs num_seqs sequential FSDP-allgather
# cycles. We can only see "set explicitly" via the env var the launcher would pass.
pf_check_ref_fbs_trap() {
    local fbs="$1" fbs_yaml="$2" ref_set="$3" g="$4" bs="$5"
    is_int "$fbs" || return 0
    is_int "$fbs_yaml" || return 0
    if [ "$fbs" -lt "$fbs_yaml" ] && [ "$ref_set" != "1" ]; then
        local want="(>= grpo_samples x batch_size)"
        if is_int "$g" && is_int "$bs"; then want=">= $((g * bs))"; fi
        pf_add WARN "forward_batch_size lowered to ${fbs} (YAML default ${fbs_yaml}) but ref_forward_batch_size is NOT set explicitly. THE TRAP: ref_forward_batch_size defaults to fbs, so ref-fwd inflates to num_seqs sequential FSDP-allgather cycles (validated 0.2s -> 100s, 500x). Set ref_forward_batch_size ${want} in YAML or pass REF_FORWARD_BATCH_SIZE."
    fi
}

# NOTE: the "server mode missing --worker-extension-cls" assertion deliberately does
# NOT live here. VLLM_WORKER_EXT is set by _vllm_env_setup.sh and is only in scope on
# the remote vLLM node, AFTER this config-time preflight runs — a config-reading gate
# cannot observe it without synthesizing a pass (which defeats the check). The guard
# lives at the single source of truth instead: experiments/lora_grpo/_vllm_env_setup.sh
# asserts VLLM_WORKER_EXT is non-empty on the merged/delta path, protecting every launch
# site and every fork that sources it. See feedback_dense_4b_launcher_missing_worker_extension.

# --- Check 4: large fbs/gen_batch with ZeRO-3 => many backward chunks ---------
# SOURCE: the 274s artifact (RESULTS_DISCIPLINE.md / project_lora_vs_fullft_4b_parity).
# reshard_after_forward:true (ZeRO-3) + fbs>=2 means num_seqs/fbs backward chunks,
# each paying an FSDP allgather/reduce-scatter pair; on a non-bypassed chunked path
# this is exactly how 274s/step appeared. Advisory (depends on transport at runtime).
pf_check_chunk_inflation() {
    local fbs="$1" gbs="$2" reshard="$3" g="$4" bs="$5"
    is_int "$fbs" || return 0
    local zero3=0
    case "$reshard" in true|True|TRUE|1) zero3=1;; esac
    if [ "$zero3" = "1" ] && [ "$fbs" -ge 2 ]; then
        local nseq="?" nchunks="?"
        if is_int "$g" && is_int "$bs"; then nseq=$((g * bs)); nchunks=$(( (nseq + fbs - 1) / fbs )); fi
        pf_add WARN "fbs=${fbs} with reshard_after_forward (ZeRO-3) => ~${nchunks} backward chunks (num_seqs=${nseq}), each an FSDP allgather/reduce-scatter pair. If the chunked path does NOT bypass the gloo reduce_scatter, expect inflated step time (this is the 274s/step artifact, RESULTS_DISCIPLINE.md). Confirm grpo_step path + RS transport post-run with check_run_health.sh <log>."
    fi
}

run_preflight() {
    local CFG="$1"
    [ -f "$CFG" ] || { echo "ERROR: --preflight needs a config YAML; no such file: $CFG" >&2; exit 2; }

    # YAML baselines (defaults).
    local y_g y_mg y_fbs y_reffbs y_bs y_mode y_publish y_runtime y_reshard
    y_g=$(yaml_scalar "$CFG" grpo_samples)
    y_mg=$(yaml_scalar "$CFG" max_generated_tokens)
    y_fbs=$(yaml_scalar "$CFG" forward_batch_size)
    y_reffbs=$(yaml_scalar "$CFG" ref_forward_batch_size)
    y_bs=$(yaml_scalar "$CFG" batch_size)
    y_mode=$(yaml_scalar "$CFG" vllm_mode)
    y_gbs=$(yaml_scalar "$CFG" gen_batch_size)
    # lora.publish_mode is nested; grep it leniently (commented-out -> empty).
    y_publish=$(grep -E "^[[:space:]]*publish_mode:[[:space:]]" "$CFG" 2>/dev/null | grep -vE "^[[:space:]]*#" | head -1 | sed -E "s/^[[:space:]]*publish_mode:[[:space:]]*//; s/[[:space:]]*#.*\$//; s/^[\"']//; s/[\"']\$//; s/[[:space:]]*\$//")
    y_runtime=$(grep -E "^[[:space:]]*use_runtime_lora:[[:space:]]" "$CFG" 2>/dev/null | grep -vE "^[[:space:]]*#" | head -1 | sed -E "s/^[[:space:]]*use_runtime_lora:[[:space:]]*//; s/[[:space:]]*#.*$//; s/[[:space:]]*$//")
    y_reshard=$(yaml_scalar "$CFG" reshard_after_forward)
    [ -z "$y_reshard" ] && y_reshard=$(yaml_scalar "$CFG" reshard_after_fwd)

    # Effective values (env override wins, mirroring the launcher).
    local g mg fbs bs mode publish gbs runtime ref_set
    g=$(eff GRPO_SAMPLES "$y_g")
    mg=$(eff MAX_GEN_TOKENS "$y_mg")
    fbs=$(eff FORWARD_BATCH_SIZE "$y_fbs")
    bs=$(eff BATCH_SIZE "$y_bs")
    gbs=$(eff GEN_BATCH_SIZE "$y_gbs")
    mode=$(eff VLLM_MODE "$y_mode")
    [ -z "$mode" ] && mode="server"

    # LORA_USE_RUNTIME env drives both publish mode and the vLLM stack (see launcher).
    runtime="${LORA_USE_RUNTIME:-}"
    # publish_mode: explicit env wins; else launcher derives from LORA_USE_RUNTIME; else YAML.
    if [ -n "${LORA_PUBLISH_MODE:-}" ]; then
        publish="${LORA_PUBLISH_MODE}"
    elif [ "$runtime" = "1" ]; then
        publish="runtime"
    elif [ "$runtime" = "0" ]; then
        publish="merged"
    else
        publish="$y_publish"
    fi
    # (worker-extension-cls is verified in _vllm_env_setup.sh, not here — see the note
    # above pf_check_chunk_inflation. VLLM_WORKER_EXT is not in scope at config time.)

    # ref_forward_batch_size "set explicitly": env REF_FORWARD_BATCH_SIZE present, OR
    # the YAML carries a non-empty value (the recipe reads it from YAML too).
    ref_set=0
    [ -n "${REF_FORWARD_BATCH_SIZE:-}" ] && ref_set=1
    [ -n "$y_reffbs" ] && ref_set=1

    echo "==================== PREFLIGHT ====================="
    echo "Config: $CFG"
    echo "Effective launch point (env override > YAML default):"
    echo "  grpo_samples=${g:-?}  max_generated_tokens=${mg:-?}  batch_size=${bs:-?}"
    echo "  forward_batch_size=${fbs:-?} (yaml ${y_fbs:-?})  ref_fwd_set=${ref_set}  gen_batch_size=${gbs:-?}"
    echo "  vllm_mode=${mode:-?}  publish=${publish:-?}  LORA_USE_RUNTIME=${runtime:-unset}"
    echo "  reshard_after_forward=${y_reshard:-?}"
    echo "----------------------------------------------------"

    PF_FINDINGS=()
    pf_check_g_maxgen_boundary  "$g" "$mg"
    pf_check_ref_fbs_trap       "$fbs" "$y_fbs" "$ref_set" "$g" "$bs"
    pf_check_chunk_inflation    "$fbs" "$gbs" "$y_reshard" "$g" "$bs"

    local refuse=0 warn=0 f sev msg
    if [ "${#PF_FINDINGS[@]}" -eq 0 ]; then
        echo "${GRN}PREFLIGHT GREEN: no documented known-bad launch points. Safe to launch.${RST}"
        return 0
    fi
    for f in "${PF_FINDINGS[@]}"; do
        sev="${f%%|*}"; msg="${f#*|}"
        if [ "$sev" = "REFUSE" ]; then
            refuse=1
            echo "${RED}REFUSE: ${msg}${RST}"
        else
            warn=1
            echo "${YEL}WARN:   ${msg}${RST}"
        fi
    done
    echo "----------------------------------------------------"
    if [ "$refuse" -eq 1 ]; then
        echo "${RED}PREFLIGHT REFUSED: at least one documented banned/known-bad launch point. NOT launching.${RST}"
        return 1
    fi
    echo "${YEL}PREFLIGHT: warnings only (no refusals). Launch permitted; heed the warnings above.${RST}"
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
    --preflight)
        [ $# -ge 2 ] || { echo "usage: $0 --preflight <config.yaml>" >&2; exit 2; }
        run_preflight "$2"; exit $?;;
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

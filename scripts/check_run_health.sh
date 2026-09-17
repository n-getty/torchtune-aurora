#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
    # NOTE: order matters — bash `case` takes the first match, and substrings like "2b"/"3b"
    # are contained in "32b"/"a3b". More-specific (longer) size tokens MUST be checked before
    # their shorter substrings, or e.g. a 32B run silently matches the "*2b*" (AGPT-2B) branch
    # and gets held to a 20s ceiling instead of 32B's 80s (found via a live false-positive WARN
    # on job 8754767, a 32B run, 2026-08-14).
    case "$key" in
        *32b*)                 echo "Qwen3-32B-2N|80|33-67s/step 2N (status.md)";;
        *30b*|*a3b*|*moe*)     echo "Qwen3-30B-A3B|70|54.8s/step G=8 (status.md)";;
        *agpt*2b*|*2b*)        echo "AGPT-2B|20|~13s/step 2N GSM8K (status.md 2026-06-13)";;
        *3b*)                  echo "Qwen2.5-3B|30|~21s/step 10+2 SHM (status.md)";;
        *4b*)                  echo "dense-4B|75|must be < 32B-2N ceiling (33-67s); AGPT-2B 13s, 3B 21s, LoRA-4B ~54.5s (status.md)";;
        *8b*)                  echo "Qwen3-8B|60|~27s colocate / varies (status.md)";;
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

    # Multi-node launchers write the actual rank output to a separate timestamped
    # training log and leave only the path + mpiexec status in the wrapper log.
    # Fold that log in before classifying the run; otherwise a rank OOM can be
    # invisible here and an unrelated nearby SFT metric file can produce GREEN.
    local train_log
    train_log=$(printf '%s\n' "$norm" | sed -nE 's/.*train log:[[:space:]]*([^ )]+).*/\1/p' | tail -1)
    if [ -n "$train_log" ] && [ -f "$train_log" ] && [ "$train_log" != "$LOG" ]; then
        norm="$norm"$'\n'"$(normalize "$train_log")"
        notes+=("Nested training log folded in: $train_log")
    fi

    local nonzero_exit runtime_oom
    nonzero_exit=$(printf '%s\n' "$norm" | grep -iE '(^|[[:space:]])(mpiexec|[^ ]+ gate) rc=[1-9][0-9]*|exit rc=[1-9][0-9]*' | head -5)
    if [ -n "$nonzero_exit" ]; then
        degraded=1
        findings+=("NONZERO launcher/training exit detected:")
        while IFS= read -r line; do [ -n "$line" ] && findings+=("    $line"); done <<< "$nonzero_exit"
    fi
    runtime_oom=$(printf '%s\n' "$norm" | grep -iE 'OutOfMemoryError|out of memory|OOM kill' | grep -viE '^#|document|comment|known|would|could|may' | head -5)
    if [ -n "$runtime_oom" ]; then
        degraded=1
        findings+=("RUNTIME OOM detected:")
        while IFS= read -r line; do [ -n "$line" ] && findings+=("    $line"); done <<< "$runtime_oom"
    fi

    # SFT recipes emit per-step timing ("Step N | ... time_per_step_s:...") only
    # to the DiskLogger file, NOT to the stdout/cell log this gate is usually
    # pointed at. If the given log has no timing of either format, fold in a
    # sibling DiskLogger metric file so SFT runs get a real verdict instead of a
    # false DEGRADED. Search common locations relative to the given log.
    if [ -z "$train_log" ] && ! printf '%s\n' "$norm" | grep -qE "TIMING step=|time_per_step_s:[0-9.]+"; then
        local logdir metricf all_candidates log_mtime best_metricf best_delta delta mtime
        logdir=$(dirname "$LOG")
        # A multi-segment run dir accumulates one logs/log_*.txt per segment/topology
        # (e.g. a 4N segment then a resumed 16N segment side by side). Picking the first
        # glob match, OR always picking the globally-newest metric file, is WRONG — an
        # older segment log being (re-)checked would fold in a LATER segment's numbers.
        # The correct match is the metric file whose mtime is CLOSEST to $LOG's mtime
        # (each segment writes its own log_<epoch>.txt at roughly the same wall-clock
        # window as its segment_*.log). Found live 2026-08-14: checking a 4N segment log
        # folded in the newer 16N metric log under a naive "always newest" fix — the
        # opposite bug from the original "always first glob match".
        log_mtime=$(stat -c '%Y' "$LOG" 2>/dev/null || echo 0)
        all_candidates=$(ls "$logdir"/run_out/logs/log_*.txt "$logdir"/logs/log_*.txt \
            "$logdir"/*/run_out/logs/log_*.txt 2>/dev/null)
        best_metricf=""; best_delta=""
        for metricf in $all_candidates; do
            [ -f "$metricf" ] || continue
            grep -qE "time_per_step_s:[0-9.]+" "$metricf" || continue
            mtime=$(stat -c '%Y' "$metricf" 2>/dev/null || echo 0)
            delta=$(( mtime > log_mtime ? mtime - log_mtime : log_mtime - mtime ))
            if [ -z "$best_delta" ] || [ "$delta" -lt "$best_delta" ]; then
                best_delta="$delta"; best_metricf="$metricf"
            fi
        done
        if [ -n "$best_metricf" ]; then
            norm="$norm"$'\n'"$(normalize "$best_metricf")"
            notes+=("SFT metric log folded in: $best_metricf")
        fi
    fi

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
    # Counted with grep -c (same idiom as every other probe here), NOT `grep -q` in an
    # elif. Under `set -o pipefail` a `printf | grep -q` pipeline is unreliable in this
    # script: grep -q exits at the first match and printf then dies on SIGPIPE, so the
    # pipeline status can be non-zero even though the pattern matched. That made the
    # branch silently not-taken and reported a healthy run (job 8827944) as DEGRADED
    # while the very same grep matched when run standalone. grep -c consumes all input,
    # so there is no early exit and no SIGPIPE.
    local bypass_active
    bypass_active=$(printf '%s\n' "$norm" | grep -c "chunked backward: non-EP reduce_scatter bypass ACTIVE")
    cnt_patched=$patched

    if [ "$v206" -gt 0 ]; then
        # CPU-bounce PG is live. Whether it CORRUPTS timing depends on path + EP.
        if [ "${ep_degree:-1}" != "" ] && [ "${ep_degree:-1}" -gt 1 ] 2>/dev/null; then
            notes+=("reduce_scatter: gloo CPU-bounce PG active (v206) on EP run (ep_degree=${ep_degree}) -- EXPECTED for EP; not a timing bug.")
        elif [ "$mn_colocate" -gt 0 ]; then
            notes+=("reduce_scatter: gloo CPU-bounce PG active (v206) on MULTI-NODE colocate run -- EXPECTED (native XCCL reduce_scatter leaks CXI handles cross-node); correct path, not the single-node 274s incident. Step-time reflects the gloo bounce; ACCURACY metrics are unaffected.")
        elif [ "$gpath" = "SINGLE_BACKWARD" ]; then
            notes+=("reduce_scatter: gloo CPU-bounce PG present (v206) but SINGLE_BACKWARD bypasses it (recipe:3796) -- timing OK.")
        elif [ "$gpath" = "CHUNKED_BACKWARD" ] && [ "$bypass_active" -gt 0 ]; then
            # CHUNKED_BACKWARD *with* the bypass marker is healthy. This ordering is
            # load-bearing: before 2026-09-16 the BioReason recipe emitted no
            # "grpo_step path:" line, so healthy chunked BioReason runs fell through
            # to the bypass_active branch below. Now that the subclass emits the line
            # (grpo_bioreason_distributed_xpu.py ~3455), an unguarded CHUNKED_BACKWARD
            # branch here would flag every one of them DEGRADED. Trust the ACTIVE
            # marker -- it is printed at the swap site, not inferred.
            notes+=("reduce_scatter: gloo CPU-bounce PG present (v206) on CHUNKED_BACKWARD, but the chunked-backward bypass logged ACTIVE (native XCCL) -- timing OK.")
        elif [ "$gpath" = "CHUNKED_BACKWARD" ]; then
            degraded=1
            findings+=("GLOO CPU-BOUNCE reduce_scatter ACTIVE on CHUNKED_BACKWARD non-EP run (v206 PG built, ep_degree=${ep_degree:-1}).")
            findings+=("  -> The CHUNKED_BACKWARD path has NO _orig_reduce_scatter_tensor bypass; every reduce_scatter")
            findings+=("     goes D2H->gloo-AllReduce->H2D, adding ~130s/backward (2s/layer x ~64 layers).")
            findings+=("  -> This is EXACTLY the 2026-06-17 274s/step incident. The step-time number is CORRUPTED.")
            findings+=("  -> Note: gloo reduce_scatter is expected ONLY on EP runs; on non-EP it corrupts timing.")
            findings+=("  -> Fix: run SINGLE_BACKWARD (TORCHTUNE_USE_CHUNKED_LOSS=1) or use the bypass on chunked.")
        elif [ "$bypass_active" -gt 0 ]; then
            # The BioReason recipe does not emit a "grpo_step path:" line, so $gpath is
            # '(none)' and the branches above cannot classify it. But the chunked bypass
            # prints its own unambiguous marker when it swaps in the native
            # _orig_reduce_scatter_tensor. Trust that marker: it is emitted at the swap
            # site itself, not inferred. Without this branch every healthy BioReason 2N
            # run is reported DEGRADED (observed on job 8827805, whose log carries the
            # ACTIVE marker and completed two clean steps).
            notes+=("reduce_scatter: gloo CPU-bounce PG present (v206) but the chunked-backward bypass logged ACTIVE (native XCCL) -- timing OK.")
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

    # --- ignored-trainable grad sync: collective-mismatch early warning ------
    # SOURCE: grpo_bioreason_distributed_xpu.py::_sync_ignored_trainable_grads
    #   "Averaged %d/%d ignored trainable gradients across %d x %d HSDP ranks"
    # Job 8826889 (2026-09-14, dp_replicate=1) hung 1800s and completed ZERO steps
    # because that function called all_reduce inside a loop whose iteration count
    # depended on local grad presence: rank 0 had 0 grads -> 0 collectives -> fell
    # through, peers blocked -> gloo timeout -> Exit_status=143.
    #
    # The "Averaged 0" / "Averaged 0/N" line is the visible precursor. It is emitted
    # BEFORE the peers time out, so catching it turns a 30-minute hang into an
    # immediate verdict. Also catch the explicit disagreement error the fixed code
    # now logs when ranks differ about which params carry grads.
    # See memory/bugs/project_bioreason_sync_ignored_grads_deadlock_dp_replicate1_20260914.md
    grad_zero=$(printf '%s\n' "$norm" | grep -cE "Averaged 0(/[0-9]+)? ignored trainable gradients")
    grad_disagree=$(printf '%s\n' "$norm" | grep -c "GRAD PRESENCE DISAGREEMENT")
    if [ "$grad_zero" -gt 0 ]; then
        degraded=1
        findings+=("IGNORED-GRAD SYNC averaged ZERO gradients ($grad_zero occurrences) while replicated trainables exist.")
        findings+=("  -> rank 0 found no grads for FSDP-ignored trainable params. Under the pre-2026-09-14 code this")
        findings+=("     DEADLOCKS the other ranks in all_reduce (1800s gloo timeout, Exit_status=143, zero steps).")
        findings+=("     Even with the fixed rank-independent path, 0 grads on rank 0 means its LoRA grads went missing")
        findings+=("     -- the update is not what you think it is. Investigate before trusting this run.")
    fi
    if [ "$grad_disagree" -gt 0 ]; then
        degraded=1
        findings+=("GRAD PRESENCE DISAGREEMENT across ranks ($grad_disagree occurrences) -- some ranks lost grads for")
        findings+=("  replicated params. Averaging fell back to contributing-ranks-only; the effective batch differs")
        findings+=("  from the nominal one. Do not trust this run's updates.")
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

    # --- allocator plateau headroom (banned:1 RISK INDICATOR) ----------------
    # SOURCE: memory/project_bioreason_banned1_predicted_by_plateau_headroom_20260917.
    # On BioReason 32B 2N B4/G8 fbs=2 the rank-0 "post-ref-fwd alloc=... resv=..." probe
    # orders the three banned:1 deaths below the one clean 100-step arm:
    #     60.09 GiB resv (3.91 GiB headroom) -> 100/100 clean
    #     62.41 (1.58)  -> died step 13
    #     63.55/63.57 (0.45/0.43) -> died step 9 (x2, different node pairs)
    #
    # THE THRESHOLD IS n=1 ON THE SURVIVOR SIDE. Other clean G=8 arms plateaued just as
    # high (62.80, 63.11, 60.87) but are CENSORED -- they ended at their planned NSTEPS
    # of 4-7 and log a checkpoint save plus a clean PG abort, so they are not survivals.
    # And two G=2-envelope arms faulted at 58.72 (5.28 free) and 51.29 (12.71 free),
    # which this check calls fine. Hence: a NOTE either way, and worded as risk.
    #
    # THE MECHANISM IS BETTER SUPPORTED THAN THE THRESHOLD. With
    # PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 armed (the launcher sets it), a
    # step whose transient peak exceeds remaining headroom makes the caching allocator
    # RELEASE cached segments -- resv drops 16-19 GiB with alloc flat on the probe line
    # immediately BEFORE the first fault, in 3/3 G=8 deaths and 0/2 G=2 deaths. That is
    # the empty_cache()-equivalent L0 UR-handle poisoning reached declaratively, with
    # nobody calling empty_cache(). It is post-hoc (one probe before the fault);
    # experiments/bioreason/check_alloc_headroom.sh --reclaim classifies a dead arm by it.
    #
    # Reported as a NOTE, never as DEGRADED: it concerns the arm's FUTURE, not the
    # validity of numbers already produced. A short clean arm at low headroom has
    # perfectly citable numbers -- it just should not be extended on that basis.
    # Tile size is read from the log when present rather than assumed.
    local hr_resv hr_tile hr_free hr_n
    hr_n=$(printf '%s\n' "$norm" | grep -c "post-ref-fwd alloc=")
    if [ "${hr_n:-0}" -ge 3 ]; then
        # Max over probes, NOT a fixed probe index.
        #
        # This read `sed -n '3p'` until 2026-09-17, on the theory that the plateau has
        # settled by step 2. That theory is false and it failed in the FALSE-SAFE
        # direction. A/B control arm 8834067 probed 48.95 / 54.85 / 54.86 / 62.34 --
        # +7.48 higher at probe 4, three steps after it had supposedly settled -- and the
        # sibling gate (experiments/bioreason/check_alloc_headroom.sh, same bug) announced
        # "SAFE ... 9.14 free" for a run actually at 1.66. On a banned:1 early-warning
        # check that is the failure direction that costs node-hours.
        #
        # What governs the plateau is the longest rollout generated so far, not the step
        # index: the jump lands on the step that first hits the max_gen cap. Most arms hit
        # the cap at step 0-1, so their plateau locked immediately and the fixed index
        # looked sound. (Only 2 of 4 large late jumps across 30 arms coincide with a new
        # running max length, so treat that as a plausible driver, not an established
        # rule -- max-over-probes does not depend on it being true.)
        #
        # CALIBRATION-SAFE, checked before changing a fitted threshold: on all seven arms
        # the 2.0 constant was fitted to, max differs from probe 3 by <= 0.02
        # (60.09/60.11, 62.41/62.42, 63.57/63.57, 63.55/63.55, 62.80/62.81, 63.11/63.11,
        # 60.87/60.87). Every fitted verdict is preserved; only late-jumping arms change,
        # and they change from a wrong answer to a right one. See
        # memory/feedback_a_plateau_fitted_on_a_prefix_is_a_prefix_not_a_plateau.md.
        #
        # NB the post-fault probe on a dead arm reads LOW (44-46, the reclaim drop), so
        # taking the max cannot be fooled by it -- whereas taking the LAST probe would
        # invert the gate on exactly the runs that died.
        hr_resv=$(printf '%s\n' "$norm" | grep -oE "post-ref-fwd alloc=[0-9.]+ GiB resv=[0-9.]+ GiB" \
                  | grep -oE "resv=[0-9.]+" | sed 's/resv=//' | sort -g | tail -1)
        hr_tile=$(printf '%s\n' "$norm" | grep -oE "total capacity[^0-9]*([0-9.]+) ?GiB" \
                  | grep -oE "[0-9.]+" | head -1)
        hr_tile=${hr_tile:-64}
        if [ -n "$hr_resv" ]; then
            hr_free=$(awk -v t="$hr_tile" -v r="$hr_resv" 'BEGIN{printf "%.2f", t-r}')
            if awk -v f="$hr_free" 'BEGIN{exit !(f < 2.0)}'; then
                notes+=("ALLOCATOR HEADROOM: plateau resv=${hr_resv} GiB of ${hr_tile} GiB leaves only ${hr_free} GiB. All 3 observed B4/G8 fbs=2 banned:1 deaths sat below the ~2 GiB line (0.43/0.45/1.59) vs the one clean 100-step arm at 3.91 -- but n=1 on the survivor side (the other high-plateau clean arms are CENSORED at 4-7 planned steps, not survivals), so this is elevated risk, not a verdict. If extending this arm, treat the risk as real; if it already died, run experiments/bioreason/check_alloc_headroom.sh --reclaim on it to confirm the allocator mode. Retrying on another node pair was tried 3x and failed 3x; lower the footprint instead.")
            else
                notes+=("ALLOCATOR HEADROOM: plateau resv=${hr_resv} GiB of ${hr_tile} GiB, ${hr_free} GiB free (>=2 GiB; the 100/100 clean arm ran at 3.91). NOT an all-clear -- two G=2-envelope arms faulted at 5.28 and 12.71 GiB free via a different mechanism (no reclaim drop).")
            fi
        fi
    fi

    # --- TIMING completeness -------------------------------------------------
    # GRPO recipes: recipe:4644/4771/4951 "TIMING step=%d ... total=%.1fs"
    # SFT recipes (full_finetune/lora_finetune_distributed_xpu): the DiskLogger
    # emits "Step %d | loss:... time_per_step_s:%f ..." instead. Support both so
    # an SFT run is not falsely flagged DEGRADED for lacking GRPO TIMING lines.
    local ntiming nsft
    ntiming=$(printf '%s\n' "$norm" | grep "TIMING step=" | sort -u | wc -l)
    nsft=$(printf '%s\n' "$norm" | grep -E "time_per_step_s:[0-9.]+" | sort -u | wc -l)
    if [ "$ntiming" -gt 0 ]; then
        # GRPO format: step time is the `total=...s` field.
        local steptimes
        steptimes=$(printf '%s\n' "$norm" | grep "TIMING step=" | grep -oE 'total=[0-9.]+s' | grep -oE '[0-9.]+' | sort -n)
        local typ
        typ=$(printf '%s\n' "$steptimes" | awk '{a[NR]=$1} END{if(NR==0)exit; print a[int((NR+1)/2)]}')
        notes+=("TIMING: $ntiming distinct step lines; typical total=${typ}s.")
        LAST_TYP_STEP="$typ"
    elif [ "$nsft" -gt 0 ]; then
        # SFT format: step time is the `time_per_step_s:` field. Median over all
        # steps (the first 1-3 carry torch.compile warmup; median is robust to
        # those when enough steps exist).
        local steptimes typ
        steptimes=$(printf '%s\n' "$norm" | grep -oE 'time_per_step_s:[0-9.]+' | grep -oE '[0-9.]+' | sort -n)
        typ=$(printf '%s\n' "$steptimes" | awk '{a[NR]=$1} END{if(NR==0)exit; print a[int((NR+1)/2)]}')
        notes+=("TIMING (SFT): $nsft distinct step lines; median time_per_step_s=${typ}s.")
        LAST_TYP_STEP="$typ"
    else
        degraded=1
        findings+=("NO 'TIMING step=' or 'time_per_step_s:' lines -- the run never completed a step. Any step-time number is fabricated/partial.")
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
# Per-step phase extraction, PAIRED WITH ROLLOUT LENGTH.
#
# WHY (motivating incident, 2026-09-15): forward_batch_size=3 was REJECTED on a raw
# backward delta of +7.7% (223.9s vs the fbs=2 baseline's 207.9s). But that leg's
# rollouts were 13.5% longer (len_mean 1249.6 vs 1100.8). Per token it INVERTS:
# 0.1792 vs 0.1889 s/tok, i.e. fbs=3 is 5.1% FASTER. A real throughput win was thrown
# away because a raw phase time was compared across two runs with different rollout
# lengths. The same confound had been correctly caught on another cell hours earlier
# and still slipped through here -- which is exactly why it has to be mechanical.
# See memory/project_bioreason_fbs3_rejection_was_length_confounded_20260915.md.
#
# GRPO rollout length is SAMPLED, so it varies run to run even at identical config.
# Nearly every phase time (bwd, grpo, ref_fwd, vllm decode) is ~linear in tokens.
# Therefore a raw s/step comparison between two GRPO legs is meaningless unless
# len_mean matches; the token-normalized number is the only valid one.
#
# Emission order per step (verified on job 8828343):
#   GENTIMING ... -> BIOREASON_DIAG step=N ... len_mean=L -> grpo_step bwd= -> TIMING step=N
# so we buffer GENTIMING, latch len_mean at the DIAG, and attach the rest.
#
# Echoes one "step|len_mean|bwd|total|gen|grpo|vllm|ref_fwd" record per step.
# ----------------------------------------------------------------------------
extract_steps() {
    normalize "$1" | awk '
        /GENTIMING/ {
            if (match($0, /vllm=[0-9.]+/))    { v = substr($0, RSTART+5, RLENGTH-5) }
            if (match($0, /ref_fwd=[0-9.]+/)) { r = substr($0, RSTART+8, RLENGTH-8) }
            next
        }
        /BIOREASON_DIAG step=/ {
            if (match($0, /step=[0-9]+/))     { s = substr($0, RSTART+5, RLENGTH-5) }
            if (match($0, /len_mean=[0-9.]+/)){ L = substr($0, RSTART+9, RLENGTH-9) }
            vllm = v; ref = r; bwd = ""
            next
        }
        /grpo_step bwd=/ {
            if (bwd == "" && match($0, /bwd=[0-9.]+/)) { bwd = substr($0, RSTART+4, RLENGTH-4) }
            next
        }
        /TIMING step=/ {
            if (match($0, /step=[0-9]+/))  { ts = substr($0, RSTART+5, RLENGTH-5) }
            if (match($0, /total=[0-9.]+/)){ tot = substr($0, RSTART+6, RLENGTH-6) }
            if (match($0, /[^_]gen=[0-9.]+/)) { g = substr($0, RSTART+5, RLENGTH-5) }
            if (match($0, /grpo=[0-9.]+/)) { gr = substr($0, RSTART+5, RLENGTH-5) }
            if (L != "") {
                printf "%s|%s|%s|%s|%s|%s|%s|%s\n", ts, L, bwd, tot, g, gr, vllm, ref
            }
            L = ""; bwd = ""
            next
        }
    '
}

# Last WARM step (i.e. not step 0 -- cold pays Triton JIT + vLLM graph capture).
# Falls back to the only step present if a log has just one.
warm_step_record() {
    local recs; recs=$(extract_steps "$1")
    [ -z "$recs" ] && return 1
    local n; n=$(printf '%s\n' "$recs" | wc -l)
    if [ "$n" -ge 2 ]; then printf '%s\n' "$recs" | tail -1
    else printf '%s\n' "$recs" | tail -1; fi
}

# ----------------------------------------------------------------------------
# Mean per-token cost over ALL warm steps of a leg, with the within-leg spread.
#
# Why this is separate from (and more trustworthy than) the single-step table:
# a delta smaller than a leg's own step-to-step spread is not a measurement. On
# BioReason 32B 2N the observed within-leg bwd spread is 11-13.5% across only
# 3-4 steps, so a "+13.8%" from n=1 and a "+2.5%" from n=3 are the same null
# result seen twice. Prints n, mean, and spread so the reader cannot miss it.
#
# Echoes "n|mean_bwd_sktok|spread_bwd_pct|mean_grpo_sktok|spread_grpo_pct".
# ----------------------------------------------------------------------------
warm_mean_record() {
    local recs; recs=$(extract_steps "$1")
    [ -z "$recs" ] && return 1
    # Drop step 0 (cold: Triton JIT + vLLM graph capture) when >1 step exists.
    local n; n=$(printf '%s\n' "$recs" | wc -l)
    [ "$n" -ge 2 ] && recs=$(printf '%s\n' "$recs" | tail -n +2)
    printf '%s\n' "$recs" | awk -F'|' '
        $2 > 0 && $3 != "" {
            b = $3/($2/1000.0); sb += b; nb++
            if (mnb == 0 || b < mnb) mnb = b
            if (b > mxb) mxb = b
        }
        $2 > 0 && $6 != "" {
            g = $6/($2/1000.0); sg += g; ng++
            if (mng == 0 || g < mng) mng = g
            if (g > mxg) mxg = g
        }
        END {
            if (nb == 0) exit 1
            printf "%d|%.2f|%.1f|%.2f|%.1f\n", nb, sb/nb, (mnb>0 ? (mxb/mnb-1)*100 : 0),
                   (ng ? sg/ng : 0), (mng>0 ? (mxg/mng-1)*100 : 0)
        }'
}

# Print the all-warm-steps comparison. Never fatal on its own -- it is a
# readability aid whose job is to stop a sub-noise delta being cited as a result.
compare_warm_mean() {
    local ma mb
    ma=$(warm_mean_record "$1") || ma=""
    mb=$(warm_mean_record "$2") || mb=""
    [ -z "$ma" ] && return 0
    [ -z "$mb" ] && return 0

    local na mba spa mqa spq_a
    local nb mbb spb mqb spq_b
    IFS='|' read -r na mba spa mqa spq_a <<<"$ma"
    IFS='|' read -r nb mbb spb mqb spq_b <<<"$mb"

    echo ""
    echo "  ---- ALL WARM STEPS (mean s/ktok; the single-step table above is n=1) ----"
    printf "  %-10s %6s %12s %12s %10s\n" "phase" "n" "A mean" "B mean" "mean d%"
    awk -v na="$na" -v nb="$nb" -v a="$mba" -v b="$mbb" 'BEGIN{
        printf "  %-10s %3d/%-2d %12.2f %12.2f %+9.1f%%\n", "bwd", na, nb, a, b, (a>0?(b-a)/a*100:0) }'
    awk -v na="$na" -v nb="$nb" -v a="$mqa" -v b="$mqb" 'BEGIN{
        if (a>0 && b>0) printf "  %-10s %3d/%-2d %12.2f %12.2f %+9.1f%%\n", "grpo", na, nb, a, b, (b-a)/a*100 }'
    printf "  within-leg spread  : A bwd %.1f%%  B bwd %.1f%%\n" "$spa" "$spb"

    # The gate: is the observed delta even bigger than the legs' own noise?
    awk -v a="$mba" -v b="$mbb" -v sa="$spa" -v sb="$spb" -v na="$na" -v nb="$nb" '
        BEGIN{
            d = (a>0) ? (b-a)/a*100 : 0; if (d<0) d = -d
            noise = (sa>sb) ? sa : sb
            if (na < 3 || nb < 3)
                printf "  VERDICT: UNDERPOWERED -- only %d/%d warm steps. Need >=3 per leg.\n", na, nb
            if (d <= noise)
                printf "  VERDICT: WASH -- |delta| %.1f%% is inside the within-leg spread %.1f%%. NOT a result.\n", d, noise
            else
                printf "  VERDICT: delta %.1f%% exceeds within-leg spread %.1f%% -- readable, still check n.\n", d, noise
        }'
}

# ----------------------------------------------------------------------------
# Length-normalized A/B of the phase timers. Prints raw AND per-1k-token deltas,
# and marks the raw column unusable when the two legs' len_mean disagree.
# Returns 1 if the raw numbers must not be cited (length skew over threshold).
# ----------------------------------------------------------------------------
LEN_SKEW_PCT_MAX=5

compare_normalized() {
    local A="$1" B="$2"
    local ra rb
    ra=$(warm_step_record "$A") || ra=""
    rb=$(warm_step_record "$B") || rb=""

    if [ -z "$ra" ] || [ -z "$rb" ]; then
        # Non-BioReason logs have no len_mean. Say so rather than silently skipping:
        # a comparison with no length control is not a clean comparison, it is an
        # unverified one.
        echo "${YEL}[length-norm] SKIPPED: no BIOREASON_DIAG len_mean in $([ -z "$ra" ] && echo A)$([ -z "$ra" ] && [ -z "$rb" ] && echo " and ")$([ -z "$rb" ] && echo B).${RST}"
        echo "${YEL}              Raw phase times are only comparable if you have INDEPENDENTLY"
        echo "              confirmed both legs produced the same token count. GRPO rollout"
        echo "              length is sampled and varies run to run.${RST}"
        return 0
    fi

    local sa la ba ta ga qa va fa
    local sb lb bb tb gb qb vb fb
    IFS='|' read -r sa la ba ta ga qa va fa <<<"$ra"
    IFS='|' read -r sb lb bb tb gb qb vb fb <<<"$rb"

    echo "------------------- LENGTH-NORMALIZED -------------------"
    printf "  warm step compared : A=step%s  B=step%s\n" "$sa" "$sb"
    printf "  len_mean (tok/seq) : A=%s  B=%s\n" "$la" "$lb"

    local skew
    skew=$(awk -v a="$la" -v b="$lb" 'BEGIN{ if(a>0) printf "%.1f", (b-a)/a*100; else print "nan" }')
    printf "  rollout-length skew: %+.1f%% (threshold +/-%s%%)\n" "$skew" "$LEN_SKEW_PCT_MAX"

    local confounded=0
    awk -v s="$skew" -v m="$LEN_SKEW_PCT_MAX" 'BEGIN{ exit !(s<-m || s>m) }' && confounded=1

    printf "\n  %-10s %10s %10s %8s  |  %12s %12s %8s\n" \
        "phase" "A raw(s)" "B raw(s)" "raw d%" "A s/ktok" "B s/ktok" "norm d%"
    _row() {
        local name="$1" av="$2" bv="$3"
        [ -z "$av" ] && return 0
        [ -z "$bv" ] && return 0
        awk -v n="$name" -v av="$av" -v bv="$bv" -v la="$la" -v lb="$lb" 'BEGIN{
            rawd = (av>0) ? (bv-av)/av*100 : 0;
            na = av/(la/1000.0); nb = bv/(lb/1000.0);
            nd = (na>0) ? (nb-na)/na*100 : 0;
            printf "  %-10s %10.1f %10.1f %+7.1f%%  |  %12.4f %12.4f %+7.1f%%\n", n, av, bv, rawd, na, nb, nd;
        }'
    }
    _row "bwd"     "$ba" "$bb"
    _row "grpo"    "$qa" "$qb"
    _row "ref_fwd" "$fa" "$fb"
    _row "vllm"    "$va" "$vb"
    _row "gen"     "$ga" "$gb"
    _row "total"   "$ta" "$tb"

    # ------------------------------------------------------------------
    # ALL WARM STEPS, not just the last one. The single-step table above is
    # the shape that produced FOUR wrong verdicts on this workload (three on
    # fbs=3, then lensort on 2026-09-15, read as a "+13.8% regression" from
    # one step and settling at +2.5% -- a wash -- once a second warm step
    # landed). The within-leg bwd spread here is 11-13.5%, so any single-step
    # delta below that is unreadable noise. This block exists so the spread is
    # always in front of you next to the delta.
    # ------------------------------------------------------------------
    compare_warm_mean "$A" "$B"

    if [ "$confounded" -eq 1 ]; then
        echo ""
        echo "${RED}RAW COLUMN IS CONFOUNDED -- DO NOT CITE IT.${RST}"
        echo "${RED}  len_mean differs by ${skew}% (> ${LEN_SKEW_PCT_MAX}%). Phase time is ~linear in tokens,"
        echo "  so the raw delta is measuring the rollout-length draw, not the change under test."
        echo "  Read the 's/ktok' columns. This is the 2026-09-15 fbs=3 mistake: a raw +7.7%"
        echo "  'regression' was a -5.1% per-token WIN.${RST}"
        echo "${YEL}  To get a citable raw number, re-run with more steps so length averages out,"
        echo "  or fix the sampled length (max_gen_tokens / temperature) across legs.${RST}"
        return 1
    fi
    echo "${GRN}  Lengths match within ${LEN_SKEW_PCT_MAX}% -- raw and normalized agree; either is citable.${RST}"
    return 0
}

# ----------------------------------------------------------------------------
# Compare mode: assert both legs took same grpo_step path AND same RS transport,
# AND that any phase-time delta survives rollout-length normalization.
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

    # A leg whose binary predates the diagnostic emits NOTHING. That is BLIND,
    # not DIFFERENT -- and conflating the two is how a valid A/B (2026-09-16
    # async probe) was reported INVALID: leg A ran 09-15 21:47, the BioReason
    # subclass only gained the `grpo_step path` line at 09-16 01:25, so the
    # checker read an absence as a mismatch and told the reader to re-run two
    # node-hours of HW. Blind legs fall through to the config-derived inference
    # below; only a genuine PATH-vs-PATH disagreement is a mismatch.
    #
    # Config-derived fallback: the path is a pure function of the launch point
    # (matching grpo_step's own branch), so when the line is missing we can
    # still decide parity from the config echo -- with the verdict clearly
    # labelled INFERRED so no reader mistakes it for an observation.
    infer_path() {
        local n; n=$(normalize "$1")
        local chunked packing
        chunked=$(printf '%s\n' "$n" | grep -m1 -oE 'TORCHTUNE_USE_CHUNKED_LOSS=[0-9]+' | cut -d= -f2)
        packing=$(printf '%s\n' "$n" | grep -m1 -oE 'packing=(True|False)' | cut -d= -f2)
        if [ "$packing" = "True" ]; then echo "PACKED"
        elif [ "${chunked:-0}" = "1" ]; then echo "SINGLE_BACKWARD"
        else echo "CHUNKED_BACKWARD"; fi
    }
    # One path-determining knob, or empty if this leg never echoed it.
    # EMPTY MEANS UNKNOWN, NOT ZERO. An older leg omits the newer echoes, so
    # comparing a missing field against a present one as if both were values
    # reproduces the very blind-vs-fail conflation this block exists to fix
    # (caught 2026-09-16: empty-vs-'0' on TORCHTUNE_USE_CHUNKED_LOSS turned an
    # agreeing pair into a MISMATCH one level below the one just fixed).
    path_field() {
        local n="$1" f="$2"
        case "$f" in
            chunked) printf '%s\n' "$n" | grep -m1 -oE 'TORCHTUNE_USE_CHUNKED_LOSS=[0-9]+' | cut -d= -f2;;
            packing) printf '%s\n' "$n" | grep -m1 -oE 'packing=(True|False)' | cut -d= -f2;;
            fbs)     printf '%s\n' "$n" | grep -m1 -oE '^forward_batch_size: [0-9]+' | awk '{print $2}';;
            bs)      printf '%s\n' "$n" | grep -m1 -oE '^batch_size: [0-9]+' | awk '{print $2}';;
            g)       printf '%s\n' "$n" | grep -m1 -oE '^grpo_samples: [0-9]+' | awk '{print $2}';;
        esac
    }

    local pA pB tA tB blindA=0 blindB=0
    pA=$(extract_path "$A"); pB=$(extract_path "$B")
    tA=$(extract_transport "$A"); tB=$(extract_transport "$B")
    [ -z "$pA" ] && { pA="$(infer_path "$A")"; blindA=1; }
    [ -z "$pB" ] && { pB="$(infer_path "$B")"; blindB=1; }

    echo "==================== A/B COMPARE ===================="
    echo "  A: $A"
    echo "       grpo_step path : ${pA}$([ "$blindA" -eq 1 ] && echo '  [INFERRED from config -- leg emitted no diagnostic]')"
    echo "       RS transport   : ${tA}"
    echo "  B: $B"
    echo "       grpo_step path : ${pB}$([ "$blindB" -eq 1 ] && echo '  [INFERRED from config -- leg emitted no diagnostic]')"
    echo "       RS transport   : ${tB}"
    echo "----------------------------------------------------"

    local fail=0
    if [ "$blindA" -eq 1 ] || [ "$blindB" -eq 1 ]; then
        local nA nB f vA vB conflict=0 unknown=0 shown_a="" shown_b=""
        nA=$(normalize "$A"); nB=$(normalize "$B")
        echo "${YEL}BLIND: at least one leg predates the 'grpo_step path' diagnostic.${RST}"
        for f in chunked packing fbs bs g; do
            vA=$(path_field "$nA" "$f"); vB=$(path_field "$nB" "$f")
            shown_a="${shown_a}${f}=${vA:-?} "; shown_b="${shown_b}${f}=${vB:-?} "
            # Only two PRESENT values that DISAGREE are a conflict. A field absent
            # from one leg is unknown -- it weakens the inference, it is not evidence
            # of a difference.
            if [ -n "$vA" ] && [ -n "$vB" ] && [ "$vA" != "$vB" ]; then conflict=1
            elif [ -z "$vA" ] || [ -z "$vB" ]; then unknown=1; fi
        done
        echo "       A config: ${shown_a}"
        echo "       B config: ${shown_b}"
        if [ "$conflict" -eq 1 ]; then
            fail=1
            echo "${RED}MISMATCH: path-determining config differs between legs.${RST}"
        elif [ "$pA" != "$pB" ]; then
            fail=1
            echo "${RED}MISMATCH: grpo_step path differs ('${pA}' vs '${pB}').${RST}"
        else
            echo "${YEL}No observed config field disagrees => same path by construction.${RST}"
            [ "$unknown" -eq 1 ] && echo "${YEL}(Some fields marked '?' were never echoed by one leg -- unknown, not zero.)${RST}"
            echo "${YEL}Parity accepted on INFERENCE, not on an observation. Prefer re-running${RST}"
            echo "${YEL}the older leg on a binary that emits the line before citing this A/B.${RST}"
        fi
    elif [ "$pA" != "$pB" ]; then
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
    echo "${GRN}Execution-mode parity OK: both legs same path + same transport.${RST}"

    # Mode parity is necessary but NOT sufficient. The second way an A/B lies is
    # rollout-length skew (2026-09-15 fbs=3). Check it before declaring validity.
    compare_normalized "$A" "$B" || fail=1

    echo "----------------------------------------------------"
    if [ "$fail" -eq 1 ]; then
        echo "${RED}A/B INVALID: cite the per-token column only, or re-run length-matched.${RST}"
        return 1
    fi
    echo "${GRN}A/B VALID: same execution mode, lengths matched. Comparison is citable.${RST}"
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

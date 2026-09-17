# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""CPU-safe tests for scripts/check_run_health.sh — the RUN-HEALTH GATE.

This test makes the gate itself non-rotting: the gate must be tested like any
other code, or it silently stops catching the degraded modes it was built for.

Motivating incident (2026-06-17): a dense 4B GRPO run reported 274s/step because
it silently took the CHUNKED_BACKWARD path with the gloo CPU-bounce reduce_scatter
active (no bypass). CPU tests passed throughout — it was a MEASUREMENT-VALIDITY
failure. See memory/project_lora_vs_fullft_4b_parity_20260617.md.

These fixtures encode the EXACT log marker strings the recipe emits, so if a
marker is renamed in the recipe without updating the gate, this test catches it.
"""
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "scripts" / "check_run_health.sh"


# --- Fixtures: minimal but realistic log content (torchelastic-prefixed) -----

# A clean dense run: SINGLE_BACKWARD path bypasses the gloo patch, native XCCL.
CLEAN_DENSE = """\
[default0]:Patched dist.reduce_scatter_tensor → gloo CPU-AllReduce+scatter (XPU v57)
[default0]:Patched FSDP2 _get_gradient_divide_factors for XPU (force SUM reduction)
[default0]:varlen=engaged
[default0]:grpo_step path: SINGLE_BACKWARD (TORCHTUNE_USE_CHUNKED_LOSS=1, fbs=2, num_seqs=64, num_chunks=1, ep_degree=1)
[default0]:Rank 0: single-backward backward start
[default0]:TIMING step=0  total=22.1s  gen=10.4s  grpo=11.0s  clip=0.1s  opt=0.1s  other=0.5s
[default0]:TIMING step=1  total=21.8s  gen=10.1s  grpo=11.0s  clip=0.0s  opt=0.1s  other=0.6s
[default0]:TIMING step=2  total=21.9s  gen=10.2s  grpo=11.0s  clip=0.0s  opt=0.1s  other=0.6s
"""

# A clean LoRA run (standalone recipe): never emits grpo_step path, patch installed
# but no v206 PG built -> native XCCL. This is the GREEN incident leg.
CLEAN_LORA = """\
[default0]:=== Qwen3-4B LoRA-GRPO 2-Node Server Mode ===
[default0]:Patched dist.reduce_scatter_tensor → gloo CPU-AllReduce+scatter (XPU v57)
[default0]:varlen=engaged
[default0]:TIMING step=0  total=54.7s  gen=16.0s  grpo=10.0s  clip=0.0s  opt=0.1s
[default0]:TIMING step=1  total=54.5s  gen=16.0s  grpo=10.0s  clip=0.0s  opt=0.1s
"""

# THE incident: dense CHUNKED_BACKWARD with v206 CPU-bounce PG active on non-EP.
DEGRADED_GLOO = """\
[default0]:Patched dist.reduce_scatter_tensor → gloo CPU-AllReduce+scatter (XPU v57)
[default5]:v206: non-HSDP gloo PG initialized (world=11) for _xpu_reduce_scatter_via_allreduce CPU-bounce path
[default0]:grpo_step path: CHUNKED_BACKWARD (TORCHTUNE_USE_CHUNKED_LOSS=0, fbs=2, num_seqs=64, num_chunks=32, ep_degree=1)
[default0]:TIMING step=0  total=274.5s  gen=26.4s  grpo=242.8s  clip=0.0s  opt=0.3s  other=5.0s
[default0]:TIMING step=1  total=274.0s  gen=24.4s  grpo=245.3s  clip=0.0s  opt=0.1s  other=4.1s
"""

# EP run: gloo reduce_scatter IS expected here (ep_degree>1) -> not degraded.
CLEAN_EP = """\
[default0]:Patched dist.reduce_scatter_tensor → gloo CPU-AllReduce+scatter (XPU v57)
[default0]:v206: non-HSDP gloo PG initialized (world=11) for _xpu_reduce_scatter_via_allreduce CPU-bounce path
[default0]:grpo_step path: CHUNKED_BACKWARD (TORCHTUNE_USE_CHUNKED_LOSS=0, fbs=1, num_seqs=8, num_chunks=8, ep_degree=8)
[default0]:TIMING step=0  total=200.0s  gen=20.0s  grpo=175.0s  clip=0.0s  opt=0.3s  other=4.7s
"""

# varlen requested but silently skipped.
DEGRADED_VARLEN = """\
[default0]:grpo_step path: SINGLE_BACKWARD (TORCHTUNE_USE_CHUNKED_LOSS=1, fbs=2, num_seqs=64, num_chunks=1, ep_degree=1)
[default0]:varlen=requested-but-skipped (mask is not None)
[default0]:TIMING step=0  total=22.1s  gen=10.4s  grpo=11.0s  clip=0.1s  opt=0.1s  other=0.5s
"""

# banned:1 crash.
DEGRADED_BANNED = """\
[default0]:grpo_step path: SINGLE_BACKWARD (TORCHTUNE_USE_CHUNKED_LOSS=1, fbs=2, num_seqs=64, num_chunks=1, ep_degree=1)
[default3]:RuntimeError: banned:1 device PDE page-fault at step 11
"""

# No TIMING lines at all: run never completed a step.
DEGRADED_NOTIMING = """\
[default0]:grpo_step path: SINGLE_BACKWARD (TORCHTUNE_USE_CHUNKED_LOSS=1, fbs=2, num_seqs=64, num_chunks=1, ep_degree=1)
[default0]:Rank 0: single-backward backward start
"""

# tee'd double-lines (both bare and prefixed) — gate must still classify GREEN.
CLEAN_TEED = """\
[default0]:grpo_step path: SINGLE_BACKWARD (TORCHTUNE_USE_CHUNKED_LOSS=1, fbs=2, num_seqs=64, num_chunks=1, ep_degree=1)
grpo_step path: SINGLE_BACKWARD (TORCHTUNE_USE_CHUNKED_LOSS=1, fbs=2, num_seqs=64, num_chunks=1, ep_degree=1)
[default0]:TIMING step=0  total=22.1s  gen=10.4s  grpo=11.0s  clip=0.1s  opt=0.1s
TIMING step=0  total=22.1s  gen=10.4s  grpo=11.0s  clip=0.1s  opt=0.1s
"""


def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content)
    return str(p)


def _run(*args):
    """Run the gate; return (returncode, combined_output)."""
    proc = subprocess.run(
        ["bash", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return proc.returncode, proc.stdout + proc.stderr


def test_script_exists_and_executable():
    assert SCRIPT.exists(), f"gate script missing at {SCRIPT}"


def test_clean_dense_is_green(tmp_path):
    log = _write(tmp_path, "clean_dense.log", CLEAN_DENSE)
    rc, out = _run(log)
    assert rc == 0, out
    assert "GREEN" in out
    assert "SINGLE_BACKWARD" in out


def test_clean_lora_is_green(tmp_path):
    log = _write(tmp_path, "clean_lora.log", CLEAN_LORA)
    rc, out = _run(log)
    assert rc == 0, out
    assert "GREEN" in out
    # patch installed but no v206 -> healthy XCCL note
    assert "native XCCL" in out


def test_degraded_gloo_chunked_is_flagged(tmp_path):
    log = _write(tmp_path, "degraded_gloo.log", DEGRADED_GLOO)
    rc, out = _run(log)
    assert rc == 1, out
    assert "DEGRADED" in out
    assert "GLOO CPU-BOUNCE" in out
    assert "CHUNKED_BACKWARD" in out


def test_ep_gloo_is_not_flagged(tmp_path):
    # gloo reduce_scatter on an EP run (ep_degree>1) is EXPECTED, not degraded.
    log = _write(tmp_path, "clean_ep.log", CLEAN_EP)
    rc, out = _run(log)
    assert rc == 0, out
    assert "GREEN" in out
    assert "EXPECTED for EP" in out


def test_varlen_skip_is_flagged(tmp_path):
    log = _write(tmp_path, "varlen.log", DEGRADED_VARLEN)
    rc, out = _run(log)
    assert rc == 1, out
    assert "VARLEN requested-but-skipped" in out


def test_banned_crash_is_flagged(tmp_path):
    log = _write(tmp_path, "banned.log", DEGRADED_BANNED)
    rc, out = _run(log)
    assert rc == 1, out
    assert "RUNTIME CRASH" in out


def test_no_timing_is_flagged(tmp_path):
    log = _write(tmp_path, "notiming.log", DEGRADED_NOTIMING)
    rc, out = _run(log)
    assert rc == 1, out
    assert "never completed a step" in out


def test_teed_double_lines_still_green(tmp_path):
    log = _write(tmp_path, "teed.log", CLEAN_TEED)
    rc, out = _run(log)
    assert rc == 0, out
    assert "GREEN" in out


def test_compare_matching_paths_passes(tmp_path):
    a = _write(tmp_path, "a.log", CLEAN_DENSE)
    b = _write(tmp_path, "b.log", CLEAN_DENSE)
    rc, out = _run("--compare", a, b)
    assert rc == 0, out
    assert "parity OK" in out


def test_compare_mismatch_path_fails(tmp_path):
    # THE incident A/B: LoRA (no path / XCCL) vs dense (CHUNKED / gloo-active).
    a = _write(tmp_path, "lora.log", CLEAN_LORA)
    b = _write(tmp_path, "dense.log", DEGRADED_GLOO)
    rc, out = _run("--compare", a, b)
    assert rc == 1, out
    assert "MISMATCH" in out
    assert "INVALID" in out


def test_compare_transport_mismatch_fails(tmp_path):
    # Same path label but different transport (bypassed vs active) must FAIL.
    same_path_xccl = CLEAN_DENSE.replace("SINGLE_BACKWARD", "CHUNKED_BACKWARD").replace(
        "num_chunks=1", "num_chunks=32"
    )
    a = _write(tmp_path, "a.log", same_path_xccl)  # CHUNKED, no v206 -> XCCL
    b = _write(tmp_path, "b.log", DEGRADED_GLOO)  # CHUNKED, v206 -> gloo active
    rc, out = _run("--compare", a, b)
    assert rc == 1, out
    assert "transport differs" in out


def test_monotonicity_warns_on_implausible(tmp_path):
    rc, out = _run("--baseline", "4b", "274")
    assert rc == 0, out  # advisory: warn, never fail
    assert "WARN" in out
    assert "monotonicity" in out


def test_monotonicity_ok_on_plausible(tmp_path):
    rc, out = _run("--baseline", "4b", "20")
    assert rc == 0, out
    assert "OK" in out


def test_monotonicity_32b_not_misclassified_as_2b(tmp_path):
    """Regression: bash `case` takes the first match, and "2b" is a substring of
    "32b" — a naive pattern ordering silently classifies every 32B run as
    AGPT-2B (20s ceiling) instead of Qwen3-32B-2N (80s ceiling), producing a
    false WARN on legitimate 32B step times. Found live on job 8754767
    (2026-08-14): a healthy 77.5s/step 32B run WARNed as "AGPT-2B ... > ~20s".
    """
    rc, out = _run("--baseline", "32b", "77.5")
    assert rc == 0, out
    assert "OK" in out
    assert "Qwen3-32B-2N" in out
    assert "AGPT-2B" not in out


def test_monotonicity_a3b_not_misclassified_as_3b(tmp_path):
    """Same substring-collision class as the 32b/2b case above: "3b" is a
    substring of "a3b", which must classify as Qwen3-30B-A3B (70s ceiling),
    not Qwen2.5-3B (30s ceiling)."""
    rc, out = _run("--baseline", "a3b", "60")
    assert rc == 0, out
    assert "OK" in out
    assert "Qwen3-30B-A3B" in out
    assert "Qwen2.5-3B" not in out


def test_real_incident_logs_if_present():
    """If the real incident logs are on disk, the gate must classify them right."""
    deg = REPO_ROOT / "experiments/lora_grpo/dense_baseline_chunked_20260617_205501.log"
    grn = REPO_ROOT / "experiments/lora_grpo/baseline_pathA_g8_20260617_194844.log"
    if deg.exists():
        rc, out = _run(str(deg))
        assert rc == 1, out
        assert "DEGRADED" in out
    if grn.exists():
        rc, out = _run(str(grn))
        assert rc == 0, out
        assert "GREEN" in out


# --- SFT log-format support (full_finetune / lora_finetune_distributed_xpu) ---
# SFT recipes emit "Step N | ... time_per_step_s:..." via the DiskLogger instead
# of the GRPO "TIMING step=...total=...s" stdout line. The gate must classify
# these GREEN and read the step time from time_per_step_s, not false-DEGRADE for
# lacking GRPO TIMING lines (see memory project_sft_no_sync_zero2_2n_validated).

# DiskLogger metric file content (what the SFT recipe writes to run_out/logs).
SFT_METRIC = """\
Step 1 | loss:6.40 lr:5e-07 time_per_step_s:70.2 tokens_per_second_per_gpu:27 grad_norm:97.5
Step 2 | loss:1.58 lr:1e-06 time_per_step_s:21.6 tokens_per_second_per_gpu:91 grad_norm:131.0
Step 3 | loss:0.94 lr:1.5e-06 time_per_step_s:21.5 tokens_per_second_per_gpu:107 grad_norm:19.9
Step 4 | loss:1.11 lr:2e-06 time_per_step_s:21.5 tokens_per_second_per_gpu:194 grad_norm:13.2
"""


def test_sft_metric_log_is_green(tmp_path):
    """A run pointed directly at the SFT DiskLogger metric file is GREEN."""
    log = _write(tmp_path, "log_123.txt", SFT_METRIC)
    rc, out = _run(log)
    assert rc == 0, out
    assert "GREEN" in out
    assert "time_per_step_s" in out  # used the SFT timing extractor


def test_sft_cell_log_folds_in_sibling_metric(tmp_path):
    """A cell.log lacking timing must auto-discover the sibling run_out metric."""
    # Stdout-style log with no timing lines (the usual gate target for SFT).
    cell = tmp_path / "cell.log"
    cell.write_text("=== AGPT-2B SFT ===\nvarlen=no-grad-only\n0|0: 1/19\n")
    # Sibling DiskLogger metric file at run_out/logs/log_*.txt.
    mdir = tmp_path / "run_out" / "logs"
    mdir.mkdir(parents=True)
    (mdir / "log_123.txt").write_text(SFT_METRIC)
    rc, out = _run(str(cell))
    assert rc == 0, out
    assert "GREEN" in out
    assert "folded in" in out


def test_sft_no_timing_anywhere_is_degraded(tmp_path):
    """An SFT log with no timing and no sibling metric is still DEGRADED."""
    cell = _write(tmp_path, "empty_cell.log", "=== SFT ===\nvarlen=no-grad-only\n")
    rc, out = _run(cell)
    assert rc == 1, out
    assert "DEGRADED" in out


def test_wrapper_folds_nested_training_log_and_detects_oom(tmp_path):
    train = _write(
        tmp_path,
        "train.log",
        "[rank0]: torch.OutOfMemoryError: XPU out of memory\n",
    )
    wrapper = _write(
        tmp_path,
        "wrapper.log",
        f"train log: {train}\n=== leg=single mpiexec rc=143 ===\n",
    )

    rc, out = _run(wrapper)

    assert rc == 1, out
    assert "DEGRADED" in out
    assert "Nested training log folded in" in out
    assert "RUNTIME OOM" in out
    assert "NONZERO" in out


SFT_METRIC_SLOW = """\
Step 1 | loss:1.10 lr:1e-4 time_per_step_s:150.0 tokens_per_second_per_gpu:60 grad_norm:2.0
Step 2 | loss:1.05 lr:1e-4 time_per_step_s:151.0 tokens_per_second_per_gpu:59 grad_norm:1.8
"""


def test_multi_segment_dir_folds_in_temporally_matching_metric_log(tmp_path):
    """A run dir with TWO segment logs (e.g. a 4N run, then a later 16N resume) must
    fold in the metric file whose mtime is closest to the segment log being checked —
    not the first glob match (old bug) and not always the newest file (a naive fix that
    trades one wrong-match bug for another). Regression for a live 2026-08-14 incident:
    checking an EARLIER (4N) segment log incorrectly folded in a LATER (16N) metric log
    under a first-attempted "always pick newest" fix."""
    import os
    import time

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    old_seg = tmp_path / "segment_old.log"
    old_seg.write_text("=== old topology ===\n")
    old_metric = logs_dir / "log_100.txt"
    old_metric.write_text(SFT_METRIC)  # fast steps (~21s)

    # Force a real, orderable mtime gap between the two segment/metric pairs.
    t0 = time.time() - 1000
    os.utime(old_seg, (t0, t0))
    os.utime(old_metric, (t0 + 1, t0 + 1))

    new_seg = tmp_path / "segment_new.log"
    new_seg.write_text("=== new topology ===\n")
    new_metric = logs_dir / "log_200.txt"
    new_metric.write_text(SFT_METRIC_SLOW)  # slower steps (~150s)
    t1 = time.time()
    os.utime(new_seg, (t1, t1))
    os.utime(new_metric, (t1 + 1, t1 + 1))

    # Checking the OLD segment must fold in the OLD (fast, ~21s) metric log.
    rc, out = _run(str(old_seg))
    assert rc == 0, out
    assert "log_100.txt" in out
    assert "log_200.txt" not in out

    # Checking the NEW segment must fold in the NEW (slow, ~150s) metric log.
    rc, out = _run(str(new_seg))
    assert rc == 0, out
    assert "log_200.txt" in out
    assert "log_100.txt" not in out


def test_zero_ignored_grads_is_degraded(tmp_path):
    """`Averaged 0 ignored trainable gradients` must FAIL the gate.

    Job 8826889 (2026-09-14, dp_replicate=1) hung 1800s and completed ZERO steps:
    `_sync_ignored_trainable_grads` called all_reduce inside a loop whose iteration
    count depended on local grad presence, so rank 0 (no grads) issued zero collectives
    while its 11 peers blocked. This line is the visible precursor, emitted BEFORE the
    peers time out — catching it turns a 30-minute hang into an immediate verdict.
    Both the legacy ("Averaged 0 ...") and current ("Averaged 0/904 ...") forms count.
    """
    for line in (
        "INFO: Averaged 0 ignored trainable gradients across 1 x 12 HSDP ranks",
        "INFO: Averaged 0/904 ignored trainable gradients across 1 x 12 HSDP ranks",
    ):
        cell = _write(
            tmp_path,
            "zero_grads.log",
            "TIMING step=0  total=100.0s  gen=1.0s  grpo=1.0s  clip=0.0s  opt=0.0s  other=98.0s\n"
            + line
            + "\n",
        )
        rc, out = _run(cell)
        assert rc == 1, f"expected DEGRADED for {line!r}:\n{out}"
        assert "IGNORED-GRAD SYNC averaged ZERO" in out, out


def test_all_ignored_grads_present_stays_green(tmp_path):
    """The healthy form must NOT trip the rule (no false positives on good runs)."""
    cell = _write(
        tmp_path,
        "good_grads.log",
        "TIMING step=0  total=100.0s  gen=1.0s  grpo=1.0s  clip=0.0s  opt=0.0s  other=98.0s\n"
        "INFO: Averaged 904/904 ignored trainable gradients across 1 x 12 HSDP ranks\n",
    )
    rc, out = _run(cell)
    assert rc == 0, out
    assert "GREEN" in out


def test_grad_presence_disagreement_is_degraded(tmp_path):
    """Ranks disagreeing about which params carry grads means the effective batch is
    not the nominal one — the fixed code averages over contributors and logs this."""
    cell = _write(
        tmp_path,
        "disagree.log",
        "TIMING step=0  total=100.0s  gen=1.0s  grpo=1.0s  clip=0.0s  opt=0.0s  other=98.0s\n"
        "ERROR: GRAD PRESENCE DISAGREEMENT across ranks for 12/904 ignored trainable params\n",
    )
    rc, out = _run(cell)
    assert rc == 1, out
    assert "GRAD PRESENCE DISAGREEMENT" in out


def test_chunked_bypass_active_marker_clears_gloo_finding(tmp_path):
    """An explicit bypass-ACTIVE marker must clear the gloo CPU-bounce finding.

    The BioReason recipe emits no `grpo_step path:` line, so the gate's path-based
    branches cannot classify it and it fell through to "bypass status unknown" ->
    DEGRADED. Job 8827805 hit exactly this: two clean steps, native XCCL bypass logged
    ACTIVE, yet reported DEGRADED. A gate that cries wolf on healthy runs gets ignored.

    The marker is printed at the swap site itself (not inferred), so it is trustworthy.
    """
    cell = _write(
        tmp_path,
        "bypass_active.log",
        "TIMING step=1  total=990.0s  gen=448.9s  grpo=527.4s  clip=0.1s  opt=0.1s  other=13.4s\n"
        "v206: _xpu_reduce_scatter_via_allreduce CPU-bounce PG built\n"
        "chunked backward: non-EP reduce_scatter bypass ACTIVE (native XCCL; avoids gloo CPU-bounce)\n",
    )
    rc, out = _run(cell)
    assert rc == 0, out
    assert "GREEN" in out


def test_chunked_backward_without_bypass_still_degraded(tmp_path):
    """The 2026-06-17 incident form must remain DEGRADED (no weakening of the gate)."""
    cell = _write(
        tmp_path,
        "chunked_no_bypass.log",
        "TIMING step=0  total=274.0s  gen=1.0s  grpo=270.0s  clip=0.0s  opt=0.0s  other=3.0s\n"
        "v206: _xpu_reduce_scatter_via_allreduce CPU-bounce PG built\n"
        "grpo_step path: CHUNKED_BACKWARD\n",
    )
    rc, out = _run(cell)
    assert rc == 1, out
    assert "CHUNKED_BACKWARD" in out


def test_chunked_backward_with_bypass_marker_is_green(tmp_path):
    """CHUNKED_BACKWARD *plus* the bypass marker is healthy -- the BioReason 2N form.

    Until 2026-09-16 the BioReason subclass overrode `grpo_step` without carrying over
    the base recipe's one-shot "grpo_step path:" line, so every BioReason run reported
    `NOT EMITTED` and the gate's central discriminator was blind -- `--compare`'s
    path-match assertion could not fire at all. Adding the line fixed that, but it also
    routed these runs into the CHUNKED_BACKWARD branch, which unconditionally
    DEGRADED. Both halves are needed: the path line for visibility, this branch
    ordering so visibility does not cost a false positive.

    The pairing is the point -- the two pre-existing tests cover marker-without-path and
    path-without-marker, and neither would have caught this.
    """
    cell = _write(
        tmp_path,
        "chunked_with_bypass.log",
        "TIMING step=1  total=665.9s  gen=399.3s  grpo=381.0s  clip=0.2s  opt=0.1s  other=3.4s\n"
        "v206: _xpu_reduce_scatter_via_allreduce CPU-bounce PG built\n"
        "chunked backward: non-EP reduce_scatter bypass ACTIVE (native XCCL; avoids gloo CPU-bounce)\n"
        "grpo_step path: CHUNKED_BACKWARD (TORCHTUNE_USE_CHUNKED_LOSS=0, fbs=2, "
        "num_seqs=32, num_chunks=16, ep_degree=0, multimodal=True, enable_packing=False)\n",
    )
    rc, out = _run(cell)
    assert rc == 0, out
    assert "GREEN" in out
    assert "CHUNKED_BACKWARD" in out, (
        "the path must still be REPORTED even when it is healthy -- "
        "docs/RESULTS_DISCIPLINE.md requires recording it alongside the number"
    )


def test_bioreason_recipe_emits_grpo_step_path(tmp_path):
    """AST-free source guard: the BioReason override must keep emitting the path line.

    This is the defect's root, not its symptom. The gate can only classify what the
    recipe prints; an override that silently drops the line re-blinds it, and nothing
    else in the suite would notice because the gate would just say "NOT EMITTED" and
    pass. Pin it at the source.
    """
    recipe = REPO_ROOT / "recipes" / "dev" / "grpo_bioreason_distributed_xpu.py"
    src = recipe.read_text()
    assert "grpo_step path: %s" in src, (
        f"{recipe.name} no longer emits the 'grpo_step path:' diagnostic. "
        "check_run_health.sh reports NOT EMITTED and --compare cannot assert that two "
        "A/B legs took the same path."
    )
    for token in ("PACKED", "SINGLE_BACKWARD", "CHUNKED_BACKWARD"):
        assert token in src, f"{recipe.name} path diagnostic lost the {token} branch"


def test_unknown_path_without_bypass_marker_still_degraded(tmp_path):
    """No path line AND no bypass marker => still suspect. Absence of evidence is not
    evidence of a bypass."""
    cell = _write(
        tmp_path,
        "unknown_no_marker.log",
        "TIMING step=0  total=274.0s  gen=1.0s  grpo=270.0s  clip=0.0s  opt=0.0s  other=3.0s\n"
        "v206: _xpu_reduce_scatter_via_allreduce CPU-bounce PG built\n",
    )
    rc, out = _run(cell)
    assert rc == 1, out
    assert "bypass status unknown" in out


# ---------------------------------------------------------------------------
# Rollout-length normalization in --compare.
#
# Motivating incident (2026-09-15): forward_batch_size=3 was REJECTED on a raw
# backward delta of +7.7% (223.9s vs the fbs=2 baseline's 207.9s). That leg's
# rollouts were 13.5% longer. Per token the sign INVERTS -- fbs=3 is faster. A
# real throughput win was discarded because a raw phase time was compared across
# two GRPO runs with different sampled rollout lengths. The same confound had been
# caught correctly on another cell hours earlier and still slipped through, which
# is why it is enforced here rather than in prose.
# See memory/project_bioreason_fbs3_rejection_was_length_confounded_20260915.md.
#
# The per-step emission order the extractor depends on (verified on job 8828343):
#   GENTIMING -> BIOREASON_DIAG ... len_mean= -> grpo_step bwd= -> TIMING step=
# ---------------------------------------------------------------------------


def _bioreason_log(steps):
    """Build a BioReason GRPO log. steps = [(step, len_mean, bwd, total, gen, grpo)]."""
    out = ["Patched dist.reduce_scatter_tensor → gloo CPU-AllReduce+scatter (XPU v57)\n"]
    for step, len_mean, bwd, total, gen, grpo in steps:
        out.append(
            f"Rank 0: GENTIMING vllm={gen - 80.0:.1f}s policy_fwd=0.0s ref_fwd=80.0s\n"
        )
        out.append(
            f"BIOREASON_DIAG step={step} n=384 go_emit=0.8 nonzero_rew=0.8 "
            f"mean_pred=6.0 mean_tp=1.0 len_mean={len_mean} len_max=2000 "
            f"trunc_rate=0.000 stop_rate=1.000 group_std=0.15 batch_std=0.20\n"
        )
        out.append(f"Rank 0: grpo_step bwd={bwd}s\n")
        out.append(
            f"TIMING step={step}  total={total}s  gen={gen}s  grpo={grpo}s  "
            f"clip=0.2s  opt=0.1s  other=3.3s\n"
        )
    return "".join(out)


# The real incident pair, warm step only differing in rollout length.
FBS2_LOG = _bioreason_log([(0, 1216.1, 207.9, 677.4, 361.2, 307.2),
                           (1, 1100.8, 185.7, 513.0, 233.8, 275.7)])
FBS3_LOG = _bioreason_log([(0, 1238.8, 223.9, 1041.0, 704.3, 327.7),
                           (1, 1249.6, 204.5, 666.0, 361.5, 300.9)])


def test_compare_flags_rollout_length_skew(tmp_path):
    """THE incident: +13.5% longer rollouts make the raw backward delta uncitable."""
    a = _write(tmp_path, "fbs2.log", FBS2_LOG)
    b = _write(tmp_path, "fbs3.log", FBS3_LOG)
    rc, out = _run("--compare", a, b)
    assert rc == 1, out
    assert "CONFOUNDED" in out, out
    assert "13.5" in out, out
    # Execution mode matched -- the failure must be attributed to length, not mode.
    assert "MISMATCH" not in out, out


def test_compare_normalized_sign_inverts(tmp_path):
    """Raw says fbs=3 is slower; per token it is faster. Both must be shown."""
    a = _write(tmp_path, "fbs2.log", FBS2_LOG)
    b = _write(tmp_path, "fbs3.log", FBS3_LOG)
    _, out = _run("--compare", a, b)
    # The per-step table's bwd row carries BOTH a raw and a normalized column
    # (the "|" separator). The all-warm-steps aggregate below it also emits a
    # "bwd" row, so select on the separator rather than the prefix alone.
    bwd = [ln for ln in out.splitlines()
           if ln.strip().startswith("bwd") and "|" in ln]
    assert len(bwd) == 1, out
    # "bwd  185.7  204.5  +10.1%  |  168.6955  163.6524  -3.0%"
    raw_pct, norm_pct = [f for f in bwd[0].split() if f.endswith("%")]
    assert raw_pct.startswith("+"), bwd[0]
    assert norm_pct.startswith("-"), bwd[0]


def test_compare_uses_warm_step_not_cold(tmp_path):
    """Step 0 pays Triton JIT + vLLM graph capture. The verdict must use the warm step."""
    a = _write(tmp_path, "fbs2.log", FBS2_LOG)
    b = _write(tmp_path, "fbs3.log", FBS3_LOG)
    _, out = _run("--compare", a, b)
    assert "A=step1  B=step1" in out, out
    # The cold backward (207.9/223.9) must not appear in the compared row.
    bwd = [ln for ln in out.splitlines() if ln.strip().startswith("bwd")][0]
    assert "185.7" in bwd and "204.5" in bwd, bwd
    assert "207.9" not in bwd and "223.9" not in bwd, bwd


def test_compare_length_matched_legs_pass(tmp_path):
    """Within the 5% threshold, raw and normalized agree -- the A/B is citable."""
    a = _write(tmp_path, "a.log", _bioreason_log([(0, 1200.0, 200.0, 600.0, 300.0, 300.0),
                                                  (1, 1100.0, 185.0, 510.0, 230.0, 275.0)]))
    b = _write(tmp_path, "b.log", _bioreason_log([(0, 1210.0, 201.0, 604.0, 302.0, 302.0),
                                                  (1, 1122.0, 170.0, 495.0, 228.0, 260.0)]))
    rc, out = _run("--compare", a, b)
    assert rc == 0, out
    assert "A/B VALID" in out, out
    assert "CONFOUNDED" not in out, out


def test_compare_without_len_mean_says_skipped_not_silent(tmp_path):
    """A non-BioReason A/B still passes on mode parity, but must SAY it is unnormalized.
    Silently omitting the check would read as 'lengths verified' when nothing was."""
    a = _write(tmp_path, "a.log", CLEAN_DENSE)
    b = _write(tmp_path, "b.log", CLEAN_DENSE)
    rc, out = _run("--compare", a, b)
    assert rc == 0, out
    assert "SKIPPED" in out, out


def test_compare_mode_mismatch_still_wins_over_length(tmp_path):
    """A mode mismatch is fatal regardless of lengths -- don't let the new check mask it."""
    a = _write(tmp_path, "lora.log", CLEAN_LORA)
    b = _write(tmp_path, "dense.log", DEGRADED_GLOO)
    rc, out = _run("--compare", a, b)
    assert rc == 1, out
    assert "MISMATCH" in out, out


# ---------------------------------------------------------------------------
# ALL-WARM-STEPS aggregate: a delta smaller than a leg's own step-to-step
# spread is not a measurement.
#
# Motivating incident (2026-09-15, the FOURTH wrong sign on this workload):
# length-sorted policy chunks (TORCHTUNE_SORT_POLICY_CHUNKS_BY_LENGTH=1) was
# read off ONE warm step per leg as "+13.8% bwd, a regression", complete with a
# mechanism story. A second warm step landed at 182.0s (vs 203.7s) and the
# verdict collapsed to +2.5% -- a wash. The within-leg bwd spread on BioReason
# 32B 2N is 11-13.5% across 3-4 steps, so ANY single-step delta below that is
# unreadable. Worse, which step pair gets compared depends on when you run the
# script: the same two logs gave +13.8% and +0.3% twenty minutes apart.
# See memory/project_bioreason_lensort_removes_padding_but_slows_bwd_20260915.md.
# ---------------------------------------------------------------------------

# Two legs, length-matched, where the LAST warm step flatters B but the mean
# over all warm steps is a wash inside each leg's own spread.
NOISY_A = _bioreason_log([(0, 1200.0, 220.0, 700.0, 340.0, 330.0),
                          (1, 1200.0, 200.0, 620.0, 300.0, 300.0),
                          (2, 1200.0, 170.0, 560.0, 280.0, 260.0),
                          (3, 1200.0, 190.0, 600.0, 300.0, 290.0)])
NOISY_B = _bioreason_log([(0, 1200.0, 222.0, 704.0, 342.0, 332.0),
                          (1, 1200.0, 205.0, 628.0, 303.0, 305.0),
                          (2, 1200.0, 196.0, 610.0, 300.0, 296.0),
                          (3, 1200.0, 172.0, 566.0, 282.0, 262.0)])


def test_compare_reports_all_warm_steps_not_just_one(tmp_path):
    """The n=1 table must be accompanied by an n>1 mean, or the reader sees one draw."""
    a = _write(tmp_path, "a.log", NOISY_A)
    b = _write(tmp_path, "b.log", NOISY_B)
    _, out = _run("--compare", a, b)
    assert "ALL WARM STEPS" in out, out
    # 3 warm steps per leg after dropping cold step 0.
    assert "3/3" in out, out


def test_compare_calls_sub_spread_delta_a_wash(tmp_path):
    """THE incident: B's last warm step looks much better; the mean is inside the noise."""
    a = _write(tmp_path, "a.log", NOISY_A)
    b = _write(tmp_path, "b.log", NOISY_B)
    _, out = _run("--compare", a, b)
    assert "VERDICT: WASH" in out, out
    assert "within-leg spread" in out, out


def test_compare_flags_underpowered_legs(tmp_path):
    """Fewer than 3 warm steps per leg cannot resolve a ~12% spread. Say so."""
    a = _write(tmp_path, "a.log", NOISY_A)
    b = _write(tmp_path, "b.log", _bioreason_log(
        [(0, 1200.0, 222.0, 704.0, 342.0, 332.0),
         (1, 1200.0, 205.0, 628.0, 303.0, 305.0)]))
    _, out = _run("--compare", a, b)
    assert "UNDERPOWERED" in out, out
    assert "3/1" in out, out


def test_compare_real_effect_is_not_called_a_wash(tmp_path):
    """Mutation guard: a delta well ABOVE the spread must NOT be dismissed.
    Without this, a verdict that always prints WASH would pass the tests above."""
    a = _write(tmp_path, "a.log", _bioreason_log(
        [(0, 1200.0, 300.0, 700.0, 340.0, 330.0),
         (1, 1200.0, 200.0, 620.0, 300.0, 300.0),
         (2, 1200.0, 201.0, 621.0, 300.0, 301.0),
         (3, 1200.0, 199.0, 619.0, 300.0, 299.0)]))
    b = _write(tmp_path, "b.log", _bioreason_log(
        [(0, 1200.0, 300.0, 700.0, 340.0, 330.0),
         (1, 1200.0, 100.0, 520.0, 300.0, 200.0),
         (2, 1200.0, 101.0, 521.0, 300.0, 201.0),
         (3, 1200.0, 99.0, 519.0, 300.0, 199.0)]))
    _, out = _run("--compare", a, b)
    assert "VERDICT: WASH" not in out, out
    assert "exceeds within-leg spread" in out, out


def test_compare_self_comparison_is_a_zero_delta_wash(tmp_path):
    """A leg against itself must report 0.0% and never a directional verdict."""
    a = _write(tmp_path, "a.log", NOISY_A)
    _, out = _run("--compare", a, a)
    assert "VERDICT: WASH" in out, out
    mean_rows = [ln for ln in out.splitlines()
                 if ln.strip().startswith("bwd") and "/" in ln]
    assert mean_rows, out
    assert "+0.0%" in mean_rows[0], mean_rows[0]


# ---------------------------------------------------------------------------
# BLIND vs FAIL in --compare (2026-09-16).
#
# The `grpo_step path:` diagnostic reached the BioReason subclass on 09-16 at
# 01:25. An A/B whose baseline leg started 09-15 21:47 therefore has a leg that
# CANNOT emit the line. The gate read that absence as a path DIFFERENCE and
# declared a perfectly valid 1.27x async result INVALID, telling the reader to
# re-run two node-hours of HW.
#
# A gate needs three answers, not two: pass, fail, and "I cannot see." These
# pin the third, and — equally important — pin that adding it did not blunt the
# first two. See memory/feedback_gate_must_distinguish_blind_from_fail.md.
# ---------------------------------------------------------------------------

# Same launch point, expressed two ways: the older leg predates BOTH the
# `grpo_step path` line and the `TORCHTUNE_USE_CHUNKED_LOSS=`/`packing=` echoes.
_BLIND_OLD_LEG = """\
[default0]:Patched dist.reduce_scatter_tensor → gloo CPU-AllReduce+scatter (XPU v57)
[default5]:v206: non-HSDP gloo PG initialized (world=11) for _xpu_reduce_scatter_via_allreduce CPU-bounce path
[default0]:batch_size: 4
[default0]:forward_batch_size: 2
[default0]:grpo_samples: 8
[default0]:TIMING step=0  total=600.0s  gen=320.0s  grpo=270.0s  clip=0.1s  opt=0.1s  other=3.5s
[default0]:TIMING step=1  total=598.0s  gen=318.0s  grpo=269.0s  clip=0.1s  opt=0.1s  other=3.5s
"""

_SIGHTED_NEW_LEG = """\
[default0]:Patched dist.reduce_scatter_tensor → gloo CPU-AllReduce+scatter (XPU v57)
[default5]:v206: non-HSDP gloo PG initialized (world=11) for _xpu_reduce_scatter_via_allreduce CPU-bounce path
[default0]:batch_size: 4
[default0]:forward_batch_size: 2
[default0]:grpo_samples: 8
[default0]:grpo_step path: CHUNKED_BACKWARD (TORCHTUNE_USE_CHUNKED_LOSS=0, fbs=2, num_seqs=32, num_chunks=16, ep_degree=1, multimodal=True, enable_packing=False)
[default0]:TIMING step=0  total=470.0s  gen=185.0s  grpo=278.0s  clip=0.1s  opt=0.1s  other=3.3s
[default0]:TIMING step=1  total=466.0s  gen=183.0s  grpo=277.0s  clip=0.1s  opt=0.1s  other=3.3s
"""


def test_compare_blind_leg_with_agreeing_config_is_not_a_mismatch(tmp_path):
    """THE regression: a leg that predates the diagnostic is BLIND, not DIFFERENT.

    Both legs ran the same launch point; only one can say so. The gate must
    infer, label the inference, and accept mode parity.
    """
    a = _write(tmp_path, "old.log", _BLIND_OLD_LEG)
    b = _write(tmp_path, "new.log", _SIGHTED_NEW_LEG)
    _, out = _run("--compare", a, b)
    assert "grpo_step path differs" not in out, out
    assert "BLIND" in out, out
    assert "INFERRED" in out, out
    assert "parity OK" in out, out


def test_compare_blind_leg_absent_field_is_unknown_not_zero(tmp_path):
    """The fix needed the same fix one level down.

    Comparing config fields, an absent field read as a VALUE ("" vs "0") turned
    the agreeing pair straight back into a MISMATCH. Missing must mean unknown.
    """
    a = _write(tmp_path, "old.log", _BLIND_OLD_LEG)      # no CHUNKED_LOSS echo
    b = _write(tmp_path, "new.log", _SIGHTED_NEW_LEG)    # TORCHTUNE_USE_CHUNKED_LOSS=0
    _, out = _run("--compare", a, b)
    assert "path-determining config differs" not in out, out
    # The unknown must be surfaced, not silently treated as agreement.
    assert "?" in out, out
    assert "unknown, not zero" in out, out


def test_compare_blind_leg_with_conflicting_config_still_fails(tmp_path):
    """Negative control: relaxing the blind case must not blunt a real conflict.

    Same blind leg, but its OBSERVED fbs genuinely differs. Two present values
    that disagree are still a mismatch.
    """
    a = _write(tmp_path, "old.log",
               _BLIND_OLD_LEG.replace("forward_batch_size: 2", "forward_batch_size: 3"))
    b = _write(tmp_path, "new.log", _SIGHTED_NEW_LEG)
    rc, out = _run("--compare", a, b)
    assert rc == 1, out
    assert "path-determining config differs" in out, out
    assert "INVALID" in out, out


def test_compare_two_sighted_legs_that_differ_still_fail(tmp_path):
    """Negative control: the sighted path is untouched by the blind fallback."""
    a = _write(tmp_path, "a.log", _SIGHTED_NEW_LEG)
    b = _write(tmp_path, "b.log",
               _SIGHTED_NEW_LEG.replace("CHUNKED_BACKWARD", "SINGLE_BACKWARD"))
    rc, out = _run("--compare", a, b)
    assert rc == 1, out
    assert "grpo_step path differs" in out, out
    # A sighted pair must never be routed through the blind branch.
    assert "BLIND" not in out, out


def test_compare_blind_parity_is_labelled_as_inference_not_observation(tmp_path):
    """A gate that cannot see must SAY so, or the next reader over-trusts it.

    The accept path is deliberately hedged: it must tell the reader the verdict
    rests on config inference and recommend re-running the older leg.
    """
    a = _write(tmp_path, "old.log", _BLIND_OLD_LEG)
    b = _write(tmp_path, "new.log", _SIGHTED_NEW_LEG)
    _, out = _run("--compare", a, b)
    assert "INFERENCE, not on an observation" in out, out

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

import pytest

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
    b = _write(tmp_path, "b.log", DEGRADED_GLOO)   # CHUNKED, v206 -> gloo active
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

#!/usr/bin/env python3
"""Torch-based reproducer for rbdgx3's intermittent
'CUDA-capable device(s) is/are busy or unavailable' fault.

This mirrors the exact shape of probe that has historically triggered the
fault in this project's own investigations (see memory:
project_bioreason_gopred_stage2_epoch2_gpu_incidents_20260725,
feedback_h200_gpu_topology_pix_vs_sys) -- a single `torch.randn(...,
device='cuda:0')` call in a brand-new process with `CUDA_VISIBLE_DEVICES`
pinned to one physical GPU. The stdlib ctypes/libcuda reproducer
(cuda_gpu_repro.py) exercises only the raw CUDA Driver API and came back
35/35 clean; this script instead goes through PyTorch's full CUDA
runtime init path (cudaSetDevice, context creation, caching allocator,
cuBLAS/cuDNN handle lazy-init on first matmul) in case the fault is
specific to something in that stack rather than the bare driver API.

Requires: a Python environment with `torch` installed (this repo already
has one at nemo_rl_work/nemo-rl/.venv on rbdgx3). This is intentionally
the ONE dependency added relative to the ctypes version -- kept otherwise
minimal (no other imports beyond stdlib) so it's still easy to hand to
someone else who already has any torch+CUDA venv on the box.

Usage:
  python3 torch_gpu_repro.py                       # all GPUs, 5 trials each
  python3 torch_gpu_repro.py --gpus 2 --trials 10
  python3 torch_gpu_repro.py --gpus 2,3,6 --trials 20 --sleep-between 2

Exit code: 0 if every trial on every requested GPU succeeded, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time


def _worker(gpu_index: int) -> dict:
    """Runs INSIDE the fresh subprocess launched by _run_subprocess_trial.
    Deliberately mirrors the minimal historical repro shape: import torch,
    allocate a tensor directly on 'cuda:0' (CUDA_VISIBLE_DEVICES already
    pins this to the real physical GPU), do one matmul (exercises cuBLAS
    handle init, not just allocation), sync, and report."""
    import torch

    result = {"gpu": gpu_index, "torch_version": torch.__version__}
    try:
        x = torch.randn(1024, 1024, device="cuda:0")
        y = torch.randn(1024, 1024, device="cuda:0")
        z = x @ y
        torch.cuda.synchronize()
        result["ok"] = True
        result["stage"] = "done"
        result["checksum"] = z.sum().item()
    except Exception as e:  # noqa: BLE001 - intentionally broad, this is a probe
        result["ok"] = False
        result["stage"] = "torch_cuda_op"
        result["exception_type"] = type(e).__name__
        result["message"] = str(e)[:2000]
    return result


def _run_subprocess_trial(python_bin: str, gpu_index: int, timeout_s: float) -> dict:
    """Fresh interpreter per trial -- required to faithfully reproduce a
    fresh-process fault. A process that already has a working CUDA context
    on this GPU tells you nothing about whether a NEW process would
    succeed on it a moment later."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    t0 = time.time()
    try:
        proc = subprocess.run(
            [python_bin, os.path.abspath(__file__), "--_worker", str(gpu_index)],
            capture_output=True, text=True, timeout=timeout_s, env=env,
        )
    except subprocess.TimeoutExpired:
        return {
            "gpu": gpu_index, "ok": False, "stage": "subprocess_timeout",
            "exception_type": "TIMEOUT",
            "message": f"worker did not finish within {timeout_s}s (possible hang, not a clean error)",
            "elapsed_s": round(time.time() - t0, 2),
        }

    elapsed = round(time.time() - t0, 2)
    out_lines = proc.stdout.strip().splitlines()
    json_line = out_lines[-1] if out_lines else ""
    try:
        result = json.loads(json_line)
    except (json.JSONDecodeError, IndexError):
        return {
            "gpu": gpu_index, "ok": False, "stage": "worker_crash",
            "exception_type": "WORKER_CRASHED",
            "message": (proc.stderr.strip() or proc.stdout.strip() or "no output")[-2000:],
            "returncode": proc.returncode,
            "elapsed_s": elapsed,
        }
    result["elapsed_s"] = elapsed
    return result


def _detect_visible_gpu_indices() -> list[int]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout
        return [int(line.strip()) for line in out.splitlines() if line.strip()]
    except Exception as e:
        print(f"[warn] could not query nvidia-smi for GPU indices ({e}); falling back to 0-7", file=sys.stderr)
        return list(range(8))


def _default_python_bin() -> str:
    """Prefer the currently-running interpreter (it must already have
    torch, since we import it in _worker) unless the caller overrides
    --python-bin to point at a different venv."""
    return sys.executable


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpus", default="all", help="Comma-separated GPU indices, or 'all' (default).")
    parser.add_argument("--trials", type=int, default=5, help="Fresh-process trials per GPU (default 5).")
    parser.add_argument("--sleep-between", type=float, default=1.0, help="Seconds to sleep between trials (default 1.0).")
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-trial subprocess timeout in seconds (default 60).")
    parser.add_argument("--python-bin", default=None, help="Path to a python with torch installed (default: current interpreter).")
    parser.add_argument("--_worker", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._worker is not None:
        print(json.dumps(_worker(args._worker)))
        return

    python_bin = args.python_bin or _default_python_bin()

    if args.gpus == "all":
        gpu_list = _detect_visible_gpu_indices()
    else:
        gpu_list = [int(x) for x in args.gpus.split(",") if x.strip() != ""]

    print("Torch CUDA fresh-process reproducer")
    print(f"Python interpreter for workers: {python_bin}")
    print(f"GPUs to test: {gpu_list}")
    print(f"Trials per GPU: {args.trials}")
    print("=" * 72)

    all_results: dict[int, list[dict]] = {g: [] for g in gpu_list}
    any_failure = False

    for gpu in gpu_list:
        print(f"\n--- GPU {gpu} ---")
        for trial in range(1, args.trials + 1):
            result = _run_subprocess_trial(python_bin, gpu, args.timeout)
            all_results[gpu].append(result)
            status = "OK  " if result["ok"] else "FAIL"
            extra = f"  -- {result.get('exception_type')}: {result.get('message')}" if not result["ok"] else ""
            print(f"  trial {trial}/{args.trials}: {status}  stage={result.get('stage')}  {result.get('elapsed_s', '?')}s{extra}")
            if not result["ok"]:
                any_failure = True
            if trial < args.trials:
                time.sleep(args.sleep_between)

    print("\n" + "=" * 72)
    print("SUMMARY")
    for gpu in gpu_list:
        results = all_results[gpu]
        n_ok = sum(1 for r in results if r["ok"])
        verdict = "STABLE" if n_ok == len(results) else "FAULTY"
        print(f"  GPU {gpu}: {n_ok}/{len(results)} trials OK  -> {verdict}")

    out_path = "/tmp/torch_gpu_repro_results.json"
    try:
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nFull results written to {out_path}")
    except OSError:
        pass

    sys.exit(1 if any_failure else 0)


if __name__ == "__main__":
    main()

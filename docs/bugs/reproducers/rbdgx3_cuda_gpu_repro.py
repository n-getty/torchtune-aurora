#!/usr/bin/env python3
"""Minimal, dependency-free reproducer for an intermittent
'CUDA-capable device(s) is/are busy or unavailable' fault
(cudaErrorDevicesUnavailable / CUDA_ERROR_NOT_READY-class failure)
seen on specific GPUs on this node.

Symptom being reproduced: a brand-new process's FIRST CUDA call on an
affected GPU fails, even though `nvidia-smi` reports that GPU as 0% util,
0 MiB used, 0 ECC errors, normal P-state -- i.e. `nvidia-smi` does not
surface this fault. It has previously affected GPU indices 2, 3, and 6
on this node (never reproduced on 0, 1, 4, 5, 7) and always clears
(temporarily) after a full machine reboot.

Design goals:
  - Zero third-party dependencies: stdlib Python3 (ctypes) + the CUDA
    driver library that ships with the NVIDIA driver itself
    (libcuda.so.1 / libcuda.so / libcuda.dylib), NOT the CUDA toolkit,
    NOT PyTorch, NOT nvcc. Nothing to pip install, no venv to activate.
  - Uses the raw CUDA Driver API (cuInit/cuDeviceGet/cuCtxCreate/
    cuMemAlloc/cuMemFree/cuCtxDestroy) -- the same set of calls any
    higher-level framework (PyTorch, TensorFlow, JAX) performs on first
    use of a device, so a failure here is a faithful stand-in for "any
    CUDA app's first touch of this GPU will fail."
  - Runs each trial in a FRESH subprocess. The original fault is a
    fresh-process, first-CUDA-call phenomenon; reusing one process
    across trials (or across GPUs) does not reproduce it faithfully,
    because a process that got a working context on GPU A does not
    tell you whether a NEW process would succeed on GPU A five
    minutes later.

Usage:
  python3 cuda_gpu_repro.py                  # test all visible GPUs, 5 trials each
  python3 cuda_gpu_repro.py --gpus 2         # test only GPU 2
  python3 cuda_gpu_repro.py --gpus 2,3,6 --trials 10
  python3 cuda_gpu_repro.py --gpus all --trials 3

Exit code: 0 if every trial on every requested GPU succeeded, 1 otherwise.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import os
import subprocess
import sys
import time

_LIB_CANDIDATES = ["libcuda.so.1", "libcuda.so", "cuda.dll", "libcuda.dylib"]

# A handful of CUDA driver-API result codes worth naming explicitly;
# anything else is reported numerically (see CUDA's cuda.h for the full list).
_CUDA_ERROR_NAMES = {
    0: "CUDA_SUCCESS",
    100: "CUDA_ERROR_NO_DEVICE",
    101: "CUDA_ERROR_INVALID_DEVICE",
    201: "CUDA_ERROR_INVALID_CONTEXT",
    709: "CUDA_ERROR_CONTEXT_IS_DESTROYED",
    800: "CUDA_ERROR_NOT_PERMITTED",
    801: "CUDA_ERROR_NOT_SUPPORTED",
    802: "CUDA_ERROR_SYSTEM_NOT_READY",
    46: "CUDA_ERROR_DEVICES_UNAVAILABLE",  # the one we're hunting
    999: "CUDA_ERROR_UNKNOWN",
}


def _load_libcuda():
    last_err = None
    for name in _LIB_CANDIDATES:
        try:
            return ctypes.CDLL(name)
        except OSError as e:
            last_err = e
    raise RuntimeError(
        f"Could not load the CUDA driver library ({_LIB_CANDIDATES}). "
        f"This means the NVIDIA driver itself is not installed/loadable, "
        f"a different failure mode than what this script targets. "
        f"Last error: {last_err}"
    )


def _err_name(code: int) -> str:
    return _CUDA_ERROR_NAMES.get(code, f"CUDA_ERROR_{code}")


def _cuda_get_error_string(libcuda, code: int) -> str:
    buf = ctypes.c_char_p()
    rc = libcuda.cuGetErrorString(code, ctypes.byref(buf))
    if rc == 0 and buf.value:
        return buf.value.decode("utf-8", "replace")
    return "<no description available>"


def probe_one_gpu(gpu_index: int) -> dict:
    """Run in a fresh subprocess. Returns a JSON-serializable result dict.

    Performs the minimal sequence any CUDA application performs on first
    touch of a device: init -> get device handle -> create a context ->
    allocate device memory -> free it -> destroy the context. Any failure
    in this chain is reported with the exact driver-API call and error code.
    """
    libcuda = _load_libcuda()

    def call(name, *args):
        fn = getattr(libcuda, name)
        return fn(*args)

    rc = call("cuInit", ctypes.c_uint(0))
    if rc != 0:
        return {
            "gpu": gpu_index, "ok": False, "stage": "cuInit",
            "code": rc, "name": _err_name(rc),
            "message": _cuda_get_error_string(libcuda, rc),
        }

    device = ctypes.c_int()
    rc = call("cuDeviceGet", ctypes.byref(device), ctypes.c_int(gpu_index))
    if rc != 0:
        return {
            "gpu": gpu_index, "ok": False, "stage": "cuDeviceGet",
            "code": rc, "name": _err_name(rc),
            "message": _cuda_get_error_string(libcuda, rc),
        }

    context = ctypes.c_void_p()
    rc = call("cuCtxCreate_v2", ctypes.byref(context), ctypes.c_uint(0), device)
    if rc != 0:
        return {
            "gpu": gpu_index, "ok": False, "stage": "cuCtxCreate",
            "code": rc, "name": _err_name(rc),
            "message": _cuda_get_error_string(libcuda, rc),
        }

    # 16 MiB allocation -- large enough to not be a trivial no-op, small
    # enough to never be an OOM confound on a GPU with >100 GiB free.
    nbytes = 16 * 1024 * 1024
    dptr = ctypes.c_void_p()
    rc = call("cuMemAlloc_v2", ctypes.byref(dptr), ctypes.c_size_t(nbytes))
    if rc != 0:
        call("cuCtxDestroy_v2", context)
        return {
            "gpu": gpu_index, "ok": False, "stage": "cuMemAlloc",
            "code": rc, "name": _err_name(rc),
            "message": _cuda_get_error_string(libcuda, rc),
        }

    rc = call("cuMemsetD8_v2", dptr, ctypes.c_ubyte(0xAB), ctypes.c_size_t(nbytes))
    memset_ok = rc == 0
    memset_err = None if memset_ok else {"code": rc, "name": _err_name(rc)}

    call("cuMemFree_v2", dptr)
    call("cuCtxDestroy_v2", context)

    return {
        "gpu": gpu_index, "ok": memset_ok, "stage": "cuMemsetD8" if not memset_ok else "done",
        "code": 0 if memset_ok else memset_err["code"],
        "name": "CUDA_SUCCESS" if memset_ok else memset_err["name"],
        "message": "" if memset_ok else "memset after alloc failed",
    }


def _run_subprocess_trial(gpu_index: int, timeout_s: float = 30.0) -> dict:
    """Spawn `python3 THIS_SCRIPT --_worker <gpu_index>` and parse its
    single-line JSON stdout. A fresh interpreter per trial is required to
    faithfully reproduce a fresh-process fault (see module docstring)."""
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--_worker", str(gpu_index)],
            capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return {
            "gpu": gpu_index, "ok": False, "stage": "subprocess_timeout",
            "code": -1, "name": "TIMEOUT",
            "message": f"worker did not finish within {timeout_s}s (possible hang, not a clean error)",
            "elapsed_s": round(time.time() - t0, 2),
        }

    elapsed = round(time.time() - t0, 2)
    out = proc.stdout.strip().splitlines()
    json_line = out[-1] if out else ""
    try:
        result = json.loads(json_line)
    except (json.JSONDecodeError, IndexError):
        return {
            "gpu": gpu_index, "ok": False, "stage": "worker_crash",
            "code": proc.returncode, "name": "WORKER_CRASHED",
            "message": (proc.stderr.strip() or proc.stdout.strip() or "no output")[-1000:],
            "elapsed_s": elapsed,
        }
    result["elapsed_s"] = elapsed
    return result


def _detect_visible_gpu_indices() -> list[int]:
    """Ask nvidia-smi for the real physical GPU count/index set, so
    `--gpus all` does the right thing without requiring the caller to
    already know the node's topology."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout
        return [int(line.strip()) for line in out.splitlines() if line.strip()]
    except Exception as e:
        print(f"[warn] could not query nvidia-smi for GPU indices ({e}); "
              f"falling back to 0-7", file=sys.stderr)
        return list(range(8))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpus", default="all", help="Comma-separated GPU indices, or 'all' (default).")
    parser.add_argument("--trials", type=int, default=5, help="Fresh-process trials per GPU (default 5).")
    parser.add_argument("--sleep-between", type=float, default=1.0, help="Seconds to sleep between trials (default 1.0).")
    parser.add_argument("--_worker", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._worker is not None:
        # Internal worker mode: run exactly one probe and print JSON to stdout.
        result = probe_one_gpu(args._worker)
        print(json.dumps(result))
        return

    if args.gpus == "all":
        gpu_list = _detect_visible_gpu_indices()
    else:
        gpu_list = [int(x) for x in args.gpus.split(",") if x.strip() != ""]

    print(f"CUDA driver-API fresh-process reproducer")
    print(f"GPUs to test: {gpu_list}")
    print(f"Trials per GPU: {args.trials}")
    print(f"Python: {sys.version.split()[0]}  Platform: {sys.platform}")
    print("=" * 72)

    all_results: dict[int, list[dict]] = {g: [] for g in gpu_list}
    any_failure = False

    for gpu in gpu_list:
        print(f"\n--- GPU {gpu} ---")
        for trial in range(1, args.trials + 1):
            result = _run_subprocess_trial(gpu)
            all_results[gpu].append(result)
            status = "OK  " if result["ok"] else "FAIL"
            print(f"  trial {trial}/{args.trials}: {status}  "
                  f"stage={result.get('stage')}  code={result.get('code')} "
                  f"({result.get('name')})  {result.get('elapsed_s', '?')}s"
                  + (f"  -- {result.get('message')}" if not result["ok"] else ""))
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

    out_path = "/tmp/cuda_gpu_repro_results.json"
    try:
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nFull results written to {out_path}")
    except OSError:
        pass

    sys.exit(1 if any_failure else 0)


if __name__ == "__main__":
    main()

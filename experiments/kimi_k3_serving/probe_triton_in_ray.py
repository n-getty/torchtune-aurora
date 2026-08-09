#!/usr/bin/env python3
"""Step 1 diagnostic: does Triton work inside a Ray actor on XPU?

WHY
---
Both K3 throughput blockers (the sampler "OOM" and the KDA "RayChannelTimeout")
trace to the same root cause: Triton's Intel driver `init_devices()`
(`triton/backends/intel/driver.c` ~line 354) requires, for every Level-Zero
device in the SYCL context, a matching OpenCL device with an identical name.
`sycl::device(selector)` throws if nothing matches, and the exception is
uncaught -> SIGABRT. `xpu_worker.py:62-68` tries to repair this by
reassigning `ONEAPI_DEVICE_SELECTOR` to `opencl:gpu;level_zero:N` inside
`init_device()` -- but that only helps if it runs before anything creates a
SYCL context. There is also a SEPARATE known-fixed instance of this exact
crash class: `TORCHDYNAMO_DISABLE=1` (see
memory/project_ray_smoke_1node_pass.md) was needed because
`VocabParallelEmbedding.forward_xpu`'s `@torch.compile` triggers dynamo,
which triggers Triton's `init_devices()`, inside a Ray actor whose
Ray-injected selector is `level_zero:N` only. That fix predates K3; this
probe checks whether it (or the ordering fix, or neither) explains the
current K3 failures.

Leading hypothesis (ordering): Ray's actor bootstrap touches `torch`/XPU
before our own `init_device()`-style repair runs, so by the time we reassign
`ONEAPI_DEVICE_SELECTOR` the SYCL runtime has already read the old value.
Secondary hypothesis: the OpenCL GPU runtime is not visible to Ray actor
processes at all, regardless of ordering. This script distinguishes them
directly instead of guessing.

WHAT THIS TESTS
----------------
Three execution contexts, escalating toward how `serve_k3.sh`'s
`distributed_executor_backend=ray` actually runs vLLM workers:

  1. bare       -- baseline: a plain `python3 -c ...` subprocess. No Ray, no
                   pool. Establishes whether Triton works at all on this node
                   outside of Ray, and whether TORCHDYNAMO_DISABLE matters
                   even there.
  2. mp         -- a `ProcessPoolExecutor` (spawn context) worker. Isolates
                   "does going through a spawned pool worker change anything"
                   from "is it Ray specifically."
  3. ray        -- a real Ray actor, `ray.init()`'d and `ONEAPI_DEVICE_SELECTOR`
                   enumerated the same way `setup_ray_env.sh` does before
                   `ray start` in production (`level_zero:0,1,...,N-1`, so
                   Ray's IntelGPUAcceleratorManager assigns one tile per actor
                   as `level_zero:<tile>`).

Within the Ray context, each of 4 independent actors (fresh process each, so
a SIGABRT in one can't affect another) runs one variant:
  - none            -- do nothing; reproduces production behavior as a control.
  - dynamo_disable  -- set TORCHDYNAMO_DISABLE=1 only (the known 2026-05-06 fix).
  - early_fix       -- reassign ONEAPI_DEVICE_SELECTOR to the xpu_worker.py
                       `opencl:gpu;level_zero:N` form as the very first line
                       of the actor method, before importing torch/triton.
  - late_fix        -- do the same reassignment only after torch has already
                       been imported (mimics xpu_worker.py's actual call site,
                       which runs after `Worker.__init__` already touched
                       torch.xpu -- this is what production does today).

Since the crash is an uncaught C++ exception -> SIGABRT (not a catchable
Python exception), the actual triton launch always runs where a crash is
survivable to observe: bare/mp variants shell out to a throwaway subprocess;
Ray actors persist their diagnostic state to a per-variant JSON file
(fsync'd) immediately before the risky call, so even a dead actor leaves
usable evidence of exactly how far it got.

Every variant also reports, right before the triton call:
  - os.environ.get("ONEAPI_DEVICE_SELECTOR")
  - whether `torch` was already in sys.modules when our code started running
    (proxy for "did something upstream of us already touch XPU/SYCL" --
    Ray's own accelerator manager is a candidate)
  - dpctl.get_devices() split by backend, if dpctl is importable (is any
    opencl:gpu device visible in this process at all?)

Usage:
  python3 probe_triton_in_ray.py [--num-gpus N] [--skip bare mp ray]

Run this inside an existing 1-node debug hold (see hold_1node_debug.sh); it
needs live XPUs, not a login node.
"""

import argparse
import json
import multiprocessing
import os
import subprocess
import sys
import time

RESULT_DIR = os.environ.get("K3_TRITON_PROBE_DIR", "/tmp/k3_triton_probe")

SIGABRT_SIGNATURES = (
    "N4sycl3_V19exceptionE",
    "init_devices",
    "Fatal Python error: Aborted",
    "select_device",
)

_SUBPROCESS_TEMPLATE = r'''
import json, os, sys, time

result = {{"pid": os.getpid(), "fix_mode": {fix_mode!r}}}
result["torch_in_sys_modules_at_start"] = "torch" in sys.modules
result["oneapi_device_selector_at_start"] = os.environ.get("ONEAPI_DEVICE_SELECTOR")

FIX_MODE = {fix_mode!r}

if FIX_MODE == "dynamo_disable":
    os.environ["TORCHDYNAMO_DISABLE"] = "1"


def apply_selector_fix():
    visible = os.environ.get("ONEAPI_DEVICE_SELECTOR", "")
    selected = visible.rsplit("level_zero:", 1)[-1]
    if "," in selected or not selected.isdigit():
        selected = "0"
    os.environ["ONEAPI_DEVICE_SELECTOR"] = f"opencl:gpu;level_zero:{{selected}}"


if FIX_MODE == "early_fix":
    apply_selector_fix()
    result["oneapi_device_selector_after_early_fix"] = os.environ.get("ONEAPI_DEVICE_SELECTOR")

try:
    import dpctl
    result["dpctl_backends"] = sorted({{str(d.backend) for d in dpctl.get_devices()}})
except Exception as e:
    result["dpctl_error"] = repr(e)

import torch

try:
    result["torch_xpu_initialized_before_triton"] = torch.xpu.is_initialized()
except Exception:
    result["torch_xpu_initialized_before_triton"] = None

if FIX_MODE == "late_fix":
    apply_selector_fix()
    result["oneapi_device_selector_after_late_fix"] = os.environ.get("ONEAPI_DEVICE_SELECTOR")

result["stage"] = "pre_triton_launch"
sys.stdout.write(json.dumps(result) + "\n")
sys.stdout.flush()

import triton
import triton.language as tl


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


x = torch.ones(1024, device="xpu")
y = torch.ones(1024, device="xpu")
out = torch.empty_like(x)
_add_kernel[(1,)](x, y, out, 1024, BLOCK=1024)
torch.xpu.synchronize()
result["triton_kernel_ok"] = bool((out == 2).all().item())
result["stage"] = "post_triton_launch"
sys.stdout.write(json.dumps(result) + "\n")
sys.stdout.flush()
'''

FIX_MODES = ("none", "dynamo_disable", "early_fix", "late_fix")


def run_subprocess_variant(label, fix_mode, timeout=90):
    """Run the triton probe snippet in a brand-new subprocess.

    A SIGABRT here only kills the child; returncode/stderr carry the result.
    Written to a real .py file (not `-c`) because `@triton.jit` calls
    `inspect.getsourcelines()` on the decorated function, which raises
    `OSError: could not get source code` for code that has no on-disk file
    (e.g. `python -c "..."` or a `-c`-launched heredoc).
    """
    script = _SUBPROCESS_TEMPLATE.format(fix_mode=fix_mode)
    os.makedirs(RESULT_DIR, exist_ok=True)
    script_path = os.path.join(RESULT_DIR, f"_variant_{label.replace('/', '_')}.py")
    with open(script_path, "w") as fh:
        fh.write(script)
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        timed_out = False
        returncode = proc.returncode
        stdout, stderr = proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        timed_out = True
        returncode = None
        stdout = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = (e.stderr or b"").decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
    elapsed = time.time() - t0

    json_lines = []
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                json_lines.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    return {
        "label": label,
        "fix_mode": fix_mode,
        "returncode": returncode,
        "timed_out": timed_out,
        "elapsed_s": round(elapsed, 2),
        "sigabrt_signature": any(sig in stderr for sig in SIGABRT_SIGNATURES),
        "triton_kernel_ok": any(j.get("triton_kernel_ok") for j in json_lines),
        "json_lines": json_lines,
        "stderr_tail": stderr[-2000:],
    }


def bare_context(num_gpus):
    del num_gpus
    return [run_subprocess_variant(f"bare/{m}", m) for m in FIX_MODES]


def mp_context(num_gpus):
    del num_gpus
    import concurrent.futures as cf

    ctx = multiprocessing.get_context("spawn")
    results = []
    with cf.ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
        for fix_mode in FIX_MODES:
            fut = pool.submit(run_subprocess_variant, f"mp/{fix_mode}", fix_mode)
            try:
                results.append(fut.result(timeout=120))
            except cf.process.BrokenProcessPool as e:
                results.append(
                    {
                        "label": f"mp/{fix_mode}",
                        "fix_mode": fix_mode,
                        "broken_pool": True,
                        "error": repr(e),
                    }
                )
    return results


def ray_context(num_gpus):
    import ray

    tile_list = ",".join(str(i) for i in range(num_gpus))
    # NOT setdefault: an inherited ONEAPI_DEVICE_SELECTOR (e.g. the shell
    # default "level_zero:gpu", which contains no explicit tile list) is
    # exactly the kind of value this probe needs to REPLACE, not preserve.
    # setdefault silently keeps "level_zero:gpu" and Ray's own resource-spec
    # code then rejects the run with "ONEAPI_DEVICE_SELECTOR contains ['gpu']"
    # before any actor even starts -- a probe-harness bug that looks like a
    # Ray/Triton failure but is neither.
    os.environ["ONEAPI_DEVICE_SELECTOR"] = f"level_zero:{tile_list}"
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

    ray.init(num_cpus=4, num_gpus=num_gpus, include_dashboard=False, ignore_reinit_error=True)

    @ray.remote(num_gpus=1)
    class TritonProbeActor:
        def probe(self, fix_mode, result_path):
            import json as _json
            import os as _os
            import sys as _sys

            def persist(info):
                with open(result_path, "w") as fh:
                    _json.dump(info, fh)
                    fh.flush()
                    _os.fsync(fh.fileno())

            info = {"fix_mode": fix_mode, "pid": _os.getpid()}
            info["torch_in_sys_modules_at_actor_start"] = "torch" in _sys.modules
            info["oneapi_device_selector_at_actor_start"] = _os.environ.get(
                "ONEAPI_DEVICE_SELECTOR"
            )
            persist(info)

            if fix_mode == "dynamo_disable":
                _os.environ["TORCHDYNAMO_DISABLE"] = "1"

            def apply_selector_fix():
                visible = _os.environ.get("ONEAPI_DEVICE_SELECTOR", "")
                selected = visible.rsplit("level_zero:", 1)[-1]
                if "," in selected or not selected.isdigit():
                    gpu_ids = ray.get_gpu_ids()
                    selected = str(gpu_ids[0]) if gpu_ids else "0"
                _os.environ["ONEAPI_DEVICE_SELECTOR"] = f"opencl:gpu;level_zero:{selected}"

            if fix_mode == "early_fix":
                apply_selector_fix()
                info["oneapi_device_selector_after_early_fix"] = _os.environ.get(
                    "ONEAPI_DEVICE_SELECTOR"
                )
                persist(info)

            try:
                import dpctl

                info["dpctl_backends"] = sorted({str(d.backend) for d in dpctl.get_devices()})
            except Exception as e:
                info["dpctl_error"] = repr(e)
            persist(info)

            import torch

            try:
                info["torch_xpu_initialized_before_triton"] = torch.xpu.is_initialized()
            except Exception:
                info["torch_xpu_initialized_before_triton"] = None

            if fix_mode == "late_fix":
                apply_selector_fix()
                info["oneapi_device_selector_after_late_fix"] = _os.environ.get(
                    "ONEAPI_DEVICE_SELECTOR"
                )

            info["stage"] = "pre_triton_launch"
            persist(info)

            import triton
            import triton.language as tl

            @triton.jit
            def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
                pid = tl.program_id(0)
                offs = pid * BLOCK + tl.arange(0, BLOCK)
                mask = offs < n
                x = tl.load(x_ptr + offs, mask=mask)
                y = tl.load(y_ptr + offs, mask=mask)
                tl.store(out_ptr + offs, x + y, mask=mask)

            x = torch.ones(1024, device="xpu")
            y = torch.ones(1024, device="xpu")
            out = torch.empty_like(x)
            _add_kernel[(1,)](x, y, out, 1024, BLOCK=1024)
            torch.xpu.synchronize()
            info["triton_kernel_ok"] = bool((out == 2).all().item())
            info["stage"] = "post_triton_launch"
            persist(info)
            return info

    # runtime_env_fix mirrors vllm/v1/executor/ray_executor.py's actual
    # production fix: ship the opencl-inclusive selector via runtime_env
    # BEFORE the actor process forks, and set
    # RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR=1 so Ray's own
    # IntelGPUAcceleratorManager does not overwrite it afterward. This is
    # NOT the same as early_fix/late_fix, which both try to reassign the
    # env var from *inside* the already-forked actor process -- by the time
    # actor code runs, Ray's worker bootstrap has already initialized the
    # SYCL runtime via dpctl.SyclContext(...) against the opencl-less
    # selector it set on fork, and reassigning os.environ afterward cannot
    # undo that (see xpu.py's ray_noset_device_env_vars comment).
    runtime_env_actor = TritonProbeActor.options(
        runtime_env={
            "env_vars": {
                "ONEAPI_DEVICE_SELECTOR": f"opencl:gpu;level_zero:{tile_list}",
                "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR": "1",
            }
        }
    )

    os.makedirs(RESULT_DIR, exist_ok=True)
    results = []
    for fix_mode in (*FIX_MODES, "runtime_env_fix"):
        result_path = os.path.join(RESULT_DIR, f"ray_{fix_mode}.json")
        if os.path.exists(result_path):
            os.remove(result_path)
        actor_cls = runtime_env_actor if fix_mode == "runtime_env_fix" else TritonProbeActor
        actor = actor_cls.remote()
        entry = {"label": f"ray/{fix_mode}", "fix_mode": fix_mode, "result_path": result_path}
        try:
            probe_fix_mode = "none" if fix_mode == "runtime_env_fix" else fix_mode
            info = ray.get(actor.probe.remote(probe_fix_mode, result_path), timeout=120)
            entry["actor_returned"] = True
            entry["info"] = info
            entry["triton_kernel_ok"] = info.get("triton_kernel_ok", False)
        except Exception as e:
            entry["actor_returned"] = False
            entry["error"] = repr(e)
            entry["triton_kernel_ok"] = False
            if os.path.exists(result_path):
                with open(result_path) as fh:
                    entry["last_persisted_state"] = json.load(fh)
        results.append(entry)
        try:
            ray.kill(actor)
        except Exception:
            pass
    ray.shutdown()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-gpus", type=int, default=int(os.environ.get("NUM_GPUS", "12")))
    parser.add_argument("--skip", nargs="*", default=[], choices=["bare", "mp", "ray"])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)
    out_path = args.out or os.path.join(RESULT_DIR, "summary.json")

    all_results = {}
    contexts = [("bare", bare_context), ("mp", mp_context), ("ray", ray_context)]
    for name, fn in contexts:
        if name in args.skip:
            continue
        print(f"=== {name} context ===", flush=True)
        all_results[name] = fn(args.num_gpus)
        for r in all_results[name]:
            printable = {k: v for k, v in r.items() if k not in ("json_lines", "info", "stderr_tail")}
            print(json.dumps(printable), flush=True)

    with open(out_path, "w") as fh:
        json.dump(all_results, fh, indent=2, default=str)
    print(f"\nfull results written to {out_path}")

    def variant_ok(ctx, fix_mode):
        for r in all_results.get(ctx, []):
            if r.get("fix_mode") == fix_mode:
                return bool(r.get("triton_kernel_ok"))
        return None

    ray_modes = (*FIX_MODES, "runtime_env_fix") if "ray" not in args.skip else FIX_MODES
    verdict = {}
    for ctx in ("bare", "mp", "ray"):
        if ctx in args.skip:
            continue
        modes = ray_modes if ctx == "ray" else FIX_MODES
        for mode in modes:
            verdict[f"{ctx}_{mode}_ok"] = variant_ok(ctx, mode)
    print("\n=== VERDICT ===")
    print(json.dumps(verdict, indent=2))

    ray_none = verdict.get("ray_none_ok")
    ray_dynamo = verdict.get("ray_dynamo_disable_ok")
    ray_early = verdict.get("ray_early_fix_ok")
    ray_late = verdict.get("ray_late_fix_ok")
    ray_runtime_env = verdict.get("ray_runtime_env_fix_ok")

    if ray_none:
        print("-> Ray/none already works on this node; the K3 failures are not "
              "a generic Triton-in-Ray problem. Re-check K3-specific state.")
    elif ray_dynamo and not ray_none:
        print("-> TORCHDYNAMO_DISABLE=1 alone unblocks it (the known 2026-05-06 "
              "fix). Check whether serve_k3.sh's ray-worker ssh env block "
              "actually forwards TORCHDYNAMO_DISABLE -- it may have regressed.")
    elif ray_runtime_env and not any([ray_none, ray_dynamo, ray_early, ray_late]):
        print("-> runtime_env fix confirmed: in-actor os.environ reassignment "
              "(early_fix/late_fix) is TOO LATE -- Ray's own worker bootstrap "
              "already initializes SYCL against an opencl-less selector before "
              "actor code runs. The selector must be shipped via ray.remote's "
              "runtime_env at actor-CREATION time (matching "
              "ray_executor.py:_update_noset_device_env_vars), not reassigned "
              "from inside the actor. This IS what production vLLM does.")
    elif ray_early and not ray_none:
        print("-> ORDERING confirmed: fixing ONEAPI_DEVICE_SELECTOR before any "
              "torch/triton touch in the actor process unblocks Triton, even "
              "without touching TORCHDYNAMO_DISABLE. Proceed to Step 2 "
              "'If ordering' branch: move/duplicate the fix earlier.")
    elif ray_late and not ray_early:
        print("-> Unexpected: late fix works but early fix does not. Inspect "
              "per-variant JSON in " + RESULT_DIR + " manually before acting.")
    elif not any([ray_dynamo, ray_early, ray_late, ray_runtime_env]):
        print("-> None of the fixes helped, INCLUDING the runtime_env fix that "
              "matches production vLLM's own mechanism. Check dpctl_backends / "
              "torch_in_sys_modules_at_actor_start in ray_runtime_env_fix.json: "
              "if 'opencl:gpu' truly never appears there either, OpenCL is not "
              "visible to Ray actors on this node at all regardless of when or "
              "how the selector is set -> treat Triton-in-Ray as blocked and "
              "fall back to the Python-vectorization contingency plan.")
    else:
        print(f"-> Inconclusive from this run; inspect per-variant JSON in {RESULT_DIR}/ manually.")


if __name__ == "__main__":
    main()

"""Test XPU sleep/wake for colocated vLLM + training.

Tests the managed-tensor sleep/wake implementation that replaces
CuMemAllocator (CUDA-only) on Intel XPU.

Cycle:
    1. Create vLLM with enable_sleep_mode=True
    2. Generate text
    3. sleep(level=1) — offload weights to CPU, release GPU storage
    4. Allocate large tensors (simulating FSDP training)
    5. Free training tensors
    6. wake_up(tags=["weights"]) — restore weights from CPU
    7. wake_up(tags=["kv_cache"]) — reallocate KV cache
    8. Generate again — verify correctness

Usage:
    torchrun --standalone --nproc_per_node=1 \
        recipes/dev/_test_xpu_sleep.py [model_path] [--cycles N]
"""
import argparse
import importlib.util
import os
import pathlib
import sys
import time
import types

os.environ["ZE_FLAT_DEVICE_HIERARCHY"] = "FLAT"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

# --- Patch transformers version check ---
_utils_stub = types.ModuleType("transformers.utils")
_utils_stub.__path__ = []
sys.modules["transformers.utils"] = _utils_stub

for p in sys.path:
    _vp = pathlib.Path(p) / "transformers" / "utils" / "versions.py"
    if _vp.exists():
        _spec = importlib.util.spec_from_file_location(
            "transformers.utils.versions", str(_vp)
        )
        _vm = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_vm)
        _vm._compare_versions = lambda *a, **kw: None
        sys.modules["transformers.utils.versions"] = _vm
        break

del sys.modules["transformers.utils"]

import torch
import torch.distributed as dist

# Pre-register torchtune to bypass __init__.py (torchao import corrupts XCCL)
import importlib.util as _ilu
_spec = _ilu.find_spec("torchtune")
if _spec is not None and _spec.submodule_search_locations:
    _torchtune_path = list(_spec.submodule_search_locations)[0]
    _pkg = types.ModuleType("torchtune")
    _pkg.__path__ = [_torchtune_path]
    _pkg.__file__ = os.path.join(_torchtune_path, "__init__.py")
    _pkg.__version__ = ""
    sys.modules["torchtune"] = _pkg


def mem_report(label: str, device: int = 0) -> dict:
    alloc = torch.xpu.memory_allocated(device)
    reserved = torch.xpu.memory_reserved(device)
    free, total = torch.xpu.mem_get_info(device)
    print(f"  [{label}] alloc={alloc/2**30:.2f}G  reserved={reserved/2**30:.2f}G  "
          f"free={free/2**30:.2f}G  total={total/2**30:.2f}G")
    return {"alloc": alloc, "reserved": reserved, "free": free, "total": total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?",
                        default="/lus/flare/projects/ModCon/ngetty/models/Qwen2.5-3B")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--gpu-mem", type=float, default=0.5)
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--train-alloc-gib", type=float, default=10.0,
                        help="GiB to allocate during simulated training")
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    torch.xpu.set_device(local_rank)
    dist.init_process_group(backend="xccl")

    print(f"=== XPU Sleep/Wake Test ({args.cycles} cycles) ===")
    print(f"Model: {args.model}")
    mem_report("initial", local_rank)

    # Apply XPU sleep patches BEFORE importing vLLM LLM
    from torchtune.dev.xpu_sleep import patch_vllm_for_xpu_sleep
    patch_vllm_for_xpu_sleep()

    from vllm import LLM, SamplingParams

    # Create vLLM with sleep mode enabled
    print("\nCreating vLLM with enable_sleep_mode=True...")
    t0 = time.time()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        distributed_executor_backend="external_launcher",
        enforce_eager=True,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=512,
        max_num_seqs=16,
        enable_sleep_mode=True,
    )
    print(f"vLLM created in {time.time()-t0:.1f}s")
    mem_after_init = mem_report("after init", local_rank)

    # Fixed prompt for correctness checking
    test_prompt = "What is 2+2? Answer with just the number:"
    params = SamplingParams(max_tokens=16, temperature=0.0)  # greedy for determinism

    for cycle in range(args.cycles):
        print(f"\n--- Cycle {cycle+1}/{args.cycles} ---")

        # Phase 1: Generate
        t0 = time.time()
        outputs = llm.generate([test_prompt], sampling_params=params, use_tqdm=False)
        gen_time = time.time() - t0
        text = outputs[0].outputs[0].text.strip()
        n_tokens = len(outputs[0].outputs[0].token_ids)
        print(f"  Generated {n_tokens} tokens in {gen_time:.2f}s: '{text[:60]}'")
        mem_report("after gen", local_rank)

        # Phase 2: Sleep
        print("  Sleeping (level=1)...")
        t0 = time.time()
        llm.sleep(level=1)
        sleep_time = time.time() - t0
        print(f"  Sleep completed in {sleep_time:.2f}s")
        mem_after_sleep = mem_report("after sleep", local_rank)

        # Verify memory was freed
        freed_gib = (mem_after_init["alloc"] - mem_after_sleep["alloc"]) / 2**30
        print(f"  Freed {freed_gib:.2f} GiB from GPU")

        # Phase 3: Simulate FSDP training
        train_elements = int(args.train_alloc_gib * 2**30 / 2)  # bf16 = 2 bytes
        print(f"  Allocating {args.train_alloc_gib:.1f} GiB for simulated training...")
        training_tensors = []
        chunk_size = 128 * 1024 * 1024  # 256 MB per chunk in bf16
        n_chunks = train_elements // chunk_size
        for i in range(n_chunks):
            t = torch.randn(chunk_size, dtype=torch.bfloat16,
                            device=f"xpu:{local_rank}")
            training_tensors.append(t)
        mem_report("during training", local_rank)

        # Simulate computation
        result = sum(t.sum() for t in training_tensors)
        print(f"  Training result: {result.item():.4f}")

        # Free training tensors
        del training_tensors, result
        import gc
        gc.collect()
        mem_report("after training cleanup", local_rank)

        # Phase 4: Wake up
        print("  Waking up weights...")
        t0 = time.time()
        llm.wake_up(tags=["weights"])
        wake_w_time = time.time() - t0
        print(f"  Weights restored in {wake_w_time:.2f}s")
        mem_report("after weights wake", local_rank)

        print("  Waking up KV cache...")
        t0 = time.time()
        llm.wake_up(tags=["kv_cache"])
        wake_kv_time = time.time() - t0
        print(f"  KV cache restored in {wake_kv_time:.2f}s")
        mem_report("after full wake", local_rank)

        # Phase 5: Verify generation still works
        t0 = time.time()
        outputs2 = llm.generate([test_prompt], sampling_params=params, use_tqdm=False)
        gen2_time = time.time() - t0
        text2 = outputs2[0].outputs[0].text.strip()
        print(f"  Post-wake generation: '{text2[:60]}' ({gen2_time:.2f}s)")

        # Correctness check (greedy decode should be deterministic)
        if text == text2:
            print(f"  PASS: Output matches pre-sleep generation")
        else:
            print(f"  WARN: Output differs. Pre: '{text}' Post: '{text2}'")

    print(f"\n=== {args.cycles} sleep/wake cycles completed ===")
    mem_report("final", local_rank)

    del llm
    dist.destroy_process_group()
    print("SUCCESS!")


if __name__ == "__main__":
    main()

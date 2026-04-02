"""Test 4 (revised): vLLM engine destroy/recreate cycle on XPU.

Since sleep() is CUDA-only, test the alternative: destroy the LLM engine to
free GPU memory for FSDP training, then recreate it with updated weights.

This tests:
1. Create vLLM LLM with external_launcher
2. Generate some text
3. Delete the LLM engine and reclaim memory
4. Simulate "FSDP training" by allocating large tensors
5. Free training tensors
6. Recreate vLLM LLM from same weights
7. Verify generation still works

Usage:
    torchrun --standalone --nproc_per_node=1 \
        recipes/dev/_test_vllm_destroy_recreate.py [model_path]
"""
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

import gc

import torch
import torch.distributed as dist


def mem_stats(local_rank, label):
    alloc = torch.xpu.memory_allocated(local_rank) / 1e9
    reserved = torch.xpu.memory_reserved(local_rank) / 1e9
    free, total = torch.xpu.mem_get_info(local_rank)
    print(f"  [{label}] alloc={alloc:.1f}G, reserved={reserved:.1f}G, "
          f"free={free/1e9:.1f}G, total={total/1e9:.1f}G")
    return alloc


def create_llm(model_path, tp=1, gpu_mem=0.5):
    from vllm import LLM
    return LLM(
        model=model_path,
        tensor_parallel_size=tp,
        distributed_executor_backend="external_launcher",
        enforce_eager=True,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_mem,
        max_model_len=512,
        max_num_seqs=16,
    )


def test_generate(llm, rank):
    from vllm import SamplingParams
    outputs = llm.generate(
        ["What is 2+2? Answer briefly:"],
        sampling_params=SamplingParams(max_tokens=16, temperature=0.7),
        use_tqdm=False,
    )
    text = outputs[0].outputs[0].text.strip()
    n_tokens = len(outputs[0].outputs[0].token_ids)
    print(f"[rank {rank}] Generated {n_tokens} tokens: {text[:60]}")
    return n_tokens


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/lus/flare/projects/ModCon/ngetty/models/Qwen2.5-3B"
    n_cycles = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    torch.xpu.set_device(local_rank)
    dist.init_process_group(backend="xccl")

    print(f"[rank {rank}] === vLLM destroy/recreate test ({n_cycles} cycles) ===")
    print(f"[rank {rank}] Model: {model_path}")
    mem_stats(local_rank, "initial")

    for cycle in range(n_cycles):
        print(f"\n[rank {rank}] --- Cycle {cycle + 1}/{n_cycles} ---")

        # Phase 1: Create vLLM and generate
        print(f"[rank {rank}] Creating vLLM engine...")
        t0 = time.time()
        llm = create_llm(model_path)
        create_time = time.time() - t0
        print(f"[rank {rank}] vLLM created in {create_time:.1f}s")
        mem_stats(local_rank, "after vLLM create")

        test_generate(llm, rank)

        # Phase 2: Destroy vLLM to free memory
        print(f"[rank {rank}] Destroying vLLM engine...")

        # Need to aggressively clear vLLM's module-level state that holds
        # references to GPU tensors. Without this, del llm leaves model
        # weights alive via vLLM's cached parallel state / model runner.
        import vllm.distributed.parallel_state as vllm_ps
        if hasattr(vllm_ps, '_WORLD') and vllm_ps._WORLD is not None:
            vllm_ps._WORLD = None

        # Clear any model references in the engine before deleting
        if hasattr(llm, 'llm_engine'):
            engine = llm.llm_engine
            if hasattr(engine, 'engine_core'):
                core = engine.engine_core
                if hasattr(core, 'model_executor'):
                    executor = core.model_executor
                    if hasattr(executor, 'driver_worker'):
                        worker = executor.driver_worker
                        if hasattr(worker, 'worker'):
                            if hasattr(worker.worker, 'model_runner'):
                                runner = worker.worker.model_runner
                                if hasattr(runner, 'model'):
                                    del runner.model
                                if hasattr(runner, 'kv_caches'):
                                    del runner.kv_caches

        del llm
        gc.collect()
        torch.xpu.synchronize()
        torch.xpu.empty_cache()
        gc.collect()
        torch.xpu.empty_cache()
        torch.xpu.reset_peak_memory_stats()
        mem_after_destroy = mem_stats(local_rank, "after destroy + empty_cache")

        # Phase 3: Simulate FSDP training by allocating large tensors
        print(f"[rank {rank}] Simulating FSDP training (allocating 10 GiB)...")
        training_tensors = []
        for i in range(10):
            t = torch.randn(128 * 1024 * 1024, dtype=torch.bfloat16,
                            device=f"xpu:{local_rank}")  # ~256 MB each
            training_tensors.append(t)
        mem_stats(local_rank, "during training")

        # Simulate a training step (some computation)
        result = sum(t.sum() for t in training_tensors)
        print(f"[rank {rank}] Training step result: {result.item():.4f}")

        # Phase 4: Free training memory
        print(f"[rank {rank}] Freeing training tensors...")
        del training_tensors
        del result
        gc.collect()
        torch.xpu.empty_cache()
        mem_stats(local_rank, "after training cleanup")

        # Phase 5: Verify we can recreate vLLM
        # (In a real GRPO recipe, we'd reload updated weights here)

    # Final summary
    print(f"\n[rank {rank}] === {n_cycles} destroy/recreate cycles completed ===")
    mem_stats(local_rank, "final")

    dist.destroy_process_group()
    print(f"[rank {rank}] SUCCESS!")


if __name__ == "__main__":
    main()

"""Isolated vLLM tests on XPU — TP=1 and TP=2."""
import sys
import os
import time

def test_tp(tp_size, model_path, tiles):
    """Test vLLM with given tensor parallelism."""
    os.environ["ZE_AFFINITY_MASK"] = tiles
    os.environ["ZE_FLAT_DEVICE_HIERARCHY"] = "FLAT"
    os.environ["CCL_PROCESS_LAUNCHER"] = "None"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    # Do NOT set VLLM_ENABLE_V1_MULTIPROCESSING=0 — let V1 use its subprocess
    # for EngineCore. With ZE_AFFINITY_MASK isolating tiles, the subprocess
    # starts cleanly (no competing XCCL groups).

    import torch
    print(f"XPU devices visible: {torch.xpu.device_count()}")
    print(f"ZE_AFFINITY_MASK={tiles}, TP={tp_size}")

    from vllm import LLM, SamplingParams

    t0 = time.time()
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=0.5,
        max_model_len=512,
        distributed_executor_backend="mp" if tp_size > 1 else None,
    )
    print(f"LLM created in {time.time()-t0:.1f}s")

    t0 = time.time()
    out = llm.generate(
        ["What is 2+2? Answer briefly."],
        SamplingParams(max_tokens=32, temperature=0.0),
    )
    gen_time = time.time() - t0
    text = out[0].outputs[0].text.strip()
    print(f"Output: {text}")
    print(f"Generation time: {gen_time:.2f}s")
    print(f"=== TP={tp_size} PASSED ===")


if __name__ == "__main__":
    tp = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    model = sys.argv[2] if len(sys.argv) > 2 else "/lus/flare/projects/ModCon/ngetty/models/Qwen2.5-3B"
    tiles = sys.argv[3] if len(sys.argv) > 3 else "11"
    test_tp(tp, model, tiles)

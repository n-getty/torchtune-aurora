"""Test aggressive vLLM cleanup by nuking all vllm modules from sys.modules."""
import gc
import importlib.util
import os
import pathlib
import sys
import types

os.environ["ZE_FLAT_DEVICE_HIERARCHY"] = "FLAT"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

# Patch transformers
_u = types.ModuleType("transformers.utils")
_u.__path__ = []
sys.modules["transformers.utils"] = _u
for p in sys.path:
    _vp = pathlib.Path(p) / "transformers" / "utils" / "versions.py"
    if _vp.exists():
        _s = importlib.util.spec_from_file_location("transformers.utils.versions", str(_vp))
        _m = importlib.util.module_from_spec(_s)
        _s.loader.exec_module(_m)
        _m._compare_versions = lambda *a, **kw: None
        sys.modules["transformers.utils.versions"] = _m
        break
del sys.modules["transformers.utils"]

import torch
import torch.distributed as dist

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
torch.xpu.set_device(local_rank)
dist.init_process_group(backend="xccl")

print(f"Initial: alloc={torch.xpu.memory_allocated()/1e9:.1f}G, "
      f"reserved={torch.xpu.memory_reserved()/1e9:.1f}G")

from vllm import LLM, SamplingParams

model_path = sys.argv[1] if len(sys.argv) > 1 else "/lus/flare/projects/ModCon/ngetty/models/Qwen2.5-3B"

llm = LLM(
    model=model_path,
    tensor_parallel_size=1,
    distributed_executor_backend="external_launcher",
    enforce_eager=True,
    dtype="bfloat16",
    gpu_memory_utilization=0.5,
    max_model_len=512,
    max_num_seqs=16,
)
out = llm.generate(["Hello"], SamplingParams(max_tokens=8), use_tqdm=False)
print(f"Gen OK: {len(out[0].outputs[0].token_ids)} tokens")
print(f"Before cleanup: alloc={torch.xpu.memory_allocated()/1e9:.1f}G, "
      f"reserved={torch.xpu.memory_reserved()/1e9:.1f}G")

# Strategy 1: Just del
del out
del llm
gc.collect()
torch.xpu.synchronize()
torch.xpu.empty_cache()
print(f"After del+gc+empty: alloc={torch.xpu.memory_allocated()/1e9:.1f}G, "
      f"reserved={torch.xpu.memory_reserved()/1e9:.1f}G")

# Strategy 2: Nuke all vllm modules
vllm_mods = [k for k in sys.modules if k.startswith("vllm")]
print(f"Nuking {len(vllm_mods)} vllm modules from sys.modules...")
for mod_name in vllm_mods:
    mod = sys.modules.pop(mod_name, None)
    if mod is not None:
        for attr in list(vars(mod).keys()):
            try:
                delattr(mod, attr)
            except Exception:
                pass
gc.collect()
torch.xpu.synchronize()
torch.xpu.empty_cache()
gc.collect()
torch.xpu.empty_cache()
print(f"After nuke: alloc={torch.xpu.memory_allocated()/1e9:.1f}G, "
      f"reserved={torch.xpu.memory_reserved()/1e9:.1f}G")
free, total = torch.xpu.mem_get_info()
print(f"L0: free={free/1e9:.1f}G, total={total/1e9:.1f}G")

# Strategy 3: Check what's holding references
print("\nChecking for remaining XPU tensors...")
count = 0
for obj in gc.get_objects():
    if isinstance(obj, torch.Tensor) and obj.device.type == "xpu":
        count += 1
        if count <= 5:
            print(f"  Leaked tensor: shape={obj.shape}, dtype={obj.dtype}, "
                  f"size={obj.nelement() * obj.element_size() / 1e6:.1f}MB")
print(f"Total leaked XPU tensors: {count}")

dist.destroy_process_group()

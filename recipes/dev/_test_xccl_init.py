"""Minimal test: torch.distributed.init_process_group with XCCL."""
import os, sys, torch

# Pre-register torchtune to bypass __init__.py
import types as _types
import importlib.util as _imp_util
if "torchtune" not in sys.modules:
    _spec = _imp_util.find_spec("torchtune")
    if _spec is not None and _spec.submodule_search_locations:
        _torchtune_path = list(_spec.submodule_search_locations)[0]
        _pkg = _types.ModuleType("torchtune")
        _pkg.__path__ = [_torchtune_path]
        _pkg.__file__ = os.path.join(_torchtune_path, "__init__.py")
        _pkg.__version__ = ""
        sys.modules["torchtune"] = _pkg

import torchao  # noqa — torchtune.__init__ normally imports this

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
device = torch.device(f"xpu:{local_rank}")
torch.xpu.set_device(local_rank)
print(f"Rank {local_rank}: device={device}")

torch.distributed.init_process_group(
    backend="xccl",
    device_id=device,
)
print(f"Rank {local_rank}: PG initialized, world_size={torch.distributed.get_world_size()}")

# Simple allreduce test
t = torch.ones(10, device=device)
torch.distributed.all_reduce(t)
print(f"Rank {local_rank}: allreduce OK, sum={t[0].item()}")

torch.distributed.destroy_process_group()
print(f"Rank {local_rank}: Done")

"""Check free memory on each XPU tile."""
import os, subprocess, sys
for i in range(12):
    env = {**os.environ, "ZE_AFFINITY_MASK": str(i)}
    r = subprocess.run(
        [sys.executable, "-c",
         "import torch; t=torch.xpu.get_device_properties(0).total_memory; "
         "torch.xpu.empty_cache(); "
         "a=torch.empty(1, device='xpu:0'); "  # trigger runtime init
         "f=t - torch.xpu.memory_allocated(0); "
         f"print(f'Tile {i}: {{f/1024**3:.1f}} GiB free / {{t/1024**3:.1f}} GiB total')"],
        env=env, capture_output=True, text=True, timeout=15
    )
    print(r.stdout.strip() if r.returncode == 0 else f"Tile {i}: ERROR - {r.stderr[-200:]}")

"""Quick tile memory check for Aurora XPU."""
import os, subprocess, sys
for i in range(12):
    env = {**os.environ, "ZE_AFFINITY_MASK": str(i)}
    r = subprocess.run(
        [sys.executable, "-c",
         "import torch; p=torch.xpu.get_device_properties(0); "
         f"print(f'Tile {i}: {{p.total_memory/1024**3:.1f}} GiB total')"],
        env=env, capture_output=True, text=True, timeout=10
    )
    print(r.stdout.strip() if r.returncode == 0 else f"Tile {i}: ERROR")

# Setting up newer PyTorch (XPU nightly) for torchtune RL on Aurora

> **Goal:** stand up a separate venv on Aurora that runs torchtune's GRPO recipes
> on the upstream PyTorch XPU nightly (currently `torch-2.12.0.devYYYYMMDD+xpu`,
> shipping as 2.13 once it cuts), so we can compare against our
> `frameworks/2025.3.1` baseline (`torch 2.10.0a0+git449b176`).
>
> This is **not** a replacement for `module load frameworks/2025.3.1` — keep both
> available and switch by activating the appropriate venv.

Adapted from `saforem2/torchtitan/.../running-with-newer-pytorch.md` with the
Aurora-specific fixes we already made (torchcomms via nightly index, mpi4py from
source, `ezpz@fix/remove-ipex-imports`).

> [!NOTE]
> We will use the alias `uvi`:
>
> ```bash
> alias uvi='uv pip install --no-cache --link-mode=copy'
> ```

## Target install location

Put the venv outside the repo so a `git clean` cannot wipe it:

```bash
export TT_DIR=/lus/flare/projects/ModCon/ngetty/torchtune
export VENV_ROOT=/lus/flare/projects/ModCon/ngetty/venvs
export VENV=${VENV_ROOT}/torchtune-pt-nightly-xpu
mkdir -p ${VENV_ROOT}
```

## 1. Load modules and export environment variables

**Do NOT load `frameworks/2025.3.1`** — the nightly install brings its own torch,
oneccl, and torchcomms. Mixing the framework's pre-built wheels with the nightly
index causes silent ABI mismatches.

```bash
module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
```

For the GRPO recipes we additionally need our standard CCL block (only when
running multi-node — single-node SSH+standalone uses `none`/`ofi`, see
`CLAUDE.md` launcher table). Add them to the launcher script, not here.

## 2. Create venv

Fresh Python 3.12 venv, no system-site-packages (we want torch from the nightly
index, not from `/opt/aurora/.../frameworks/`):

```bash
uv venv --python=3.12 ${VENV}
source ${VENV}/bin/activate
```

## 3. Install PyTorch + torchcomms from XPU nightly

```bash
uvi torch torchvision torchaudio torchdata torchcomms \
    --pre \
    --index-url https://download.pytorch.org/whl/nightly/xpu \
    --upgrade
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.xpu.is_available(), torch.xpu.device_count())"
# expect: 2.12.0.devYYYYMMDD+xpu  True  12
```

## 4. Install torchtune RL dependencies

The RL recipes need: vLLM, transformers, math_verify, omegaconf, sentencepiece,
tiktoken, huggingface_hub, datasets, safetensors, plus distributed glue (mpi4py,
ezpz). Some need source builds on Aurora.

```bash
# Core RL stack (PyPI is fine for these)
uvi vllm transformers math_verify omegaconf sentencepiece tiktoken \
    huggingface_hub safetensors datasets blobfile tokenizers \
    pyarrow tqdm psutil tyro tensorboard wandb

# mpi4py from source against Cray MPICH (PyPI wheel links wrong MPI)
CC=$(which gcc) CXX=$(which g++) MPICC=$(which mpicc) \
    uv pip install --no-cache "git+https://github.com/mpi4py/mpi4py"

# ezpz on the branch that strips IPEX imports (IPEX is incompatible with nightly torch)
uvi "git+https://github.com/saforem2/ezpz@fix/remove-ipex-imports"
```

## 5. Remove Intel's MPI runtime

`impi-rt` arrives as a transitive dep and clashes with Cray MPI on Aurora:

```bash
uv pip uninstall impi-rt
```

## 6. Install torchtune itself (editable)

```bash
cd ${TT_DIR}
uv pip install --no-cache -e .
```

If editable install pulls in torch and clobbers the nightly, force the resolver
to skip torch:

```bash
uv pip install --no-cache -e . --no-deps
# then re-add only what's actually missing
```

## 7. Sanity checks before training

Verify on a login node (no XPU init):

```bash
python -c "
import torch, torchtune, vllm
print('torch:', torch.__version__)
print('torchtune:', torchtune.__version__)
print('vllm:', vllm.__version__)
"
```

On a held compute node, verify XPU + distributed:

```bash
python -c "
import torch, torch.distributed as dist
import torchtune.dev.rl.distributed as ttd
print('xpu count:', torch.xpu.device_count())
print('xccl available:', dist.is_xccl_available() if hasattr(dist, 'is_xccl_available') else 'unknown')
"
```

## 8. (Optional) Yeet venv to /tmp on each node

Significantly faster startup, especially with vLLM's many small files:

```bash
ezpz yeet-env
deactivate
source /tmp/.venv/bin/activate
```

## 9. Run the comparison RL workload

Use the same launcher we used against the `frameworks/2025.3.1` baseline, but
have it activate the nightly venv instead of `module load frameworks/...`:

```bash
# In experiments/comparison/run_0_6b_comparison.sh, replace:
#   module load frameworks/2025.3.1
# with:
#   module load oneapi/release/2025.3.1 hdf5 pti-gpu
#   source ${VENV}/bin/activate

NTILES=6 GRPO_SAMPLES=8 NSTEPS=20 \
    bash ${TT_DIR}/experiments/comparison/run_0_6b_comparison.sh
```

## Known compatibility risks for our codebase

| Component | Risk | Mitigation |
|-----------|------|------------|
| **IPEX `varlen_attention`** (`TORCHTUNE_USE_IPEX_VARLEN=1`) | IPEX is NOT being installed (incompatible with nightly torch). Recipes will silently fall back to PyTorch SDPA. | Expect to lose the 19% gain documented in `docs/reports/bioreason_ipex_varlen_20260430.md`. Run baseline comparison **with IPEX disabled** for fairness. |
| **vLLM XPU** | vLLM's XPU backend on PyPI may pin to a specific torch ABI (commonly 2.7–2.10). Nightly 2.12 likely breaks the wheel. | Try first; if it fails, install vLLM from source against the nightly torch, or run a fresh wheel from `vllm` nightly. |
| **`oneccl_bind_pt`** | The frameworks module ships its own `oneccl_bind_pt`. Nightly torch may use `torchcomms` instead and skip oneccl. | Verify `dist.init_process_group(backend='xccl')` still works. If not, use `torchcomms` backend per saforem2's docs. |
| **`torch.xpu.empty_cache()` UR leak** (`docs/bugs/intel_xpu_resource_leak_bug_report.md`) | Tied to oneCCL + L0 driver. Status on torch 2.12 unknown. | Keep `device_empty_cache` no-op guard in place; re-validate on 2.12 with a long-horizon FSDP run. |
| **CCL env vars** (`CCL_WORKER_COUNT=1`, no `CCL_REDUCE_SCATTER=ring`, etc.) | These are for the framework-bundled oneCCL. Nightly torch's collective backend may differ. | Test multi-node carefully; profile AllGather/ReduceScatter before assuming the same envelope holds. |
| **FSDP2 + per-chunk loop** (`TORCHTUNE_USE_CHUNKED_LOSS=1`) | Should work — pure PyTorch. | No expected change, but verify 32B doesn't OOM differently on 2.12. |
| **`init_xccl_communicator` in vLLM worker extension** (`vllm_weight_sync_worker.py`) | Calls into `oneccl_bind_pt` directly. May break under torchcomms. | SHM weight sync path (`load_weights_from_shm`) is independent — keep that as fallback. |

## Comparison test plan (for next session)

1. **Smoke**: `Qwen3-0.6B sum-of-digits 6-tile G=2 NSTEPS=10` — does it run end-to-end?
2. **Apples-to-apples**: rerun the three rows of `experiments/comparison/results_qwen3_0_6b.csv` (6t G=2, 6t G=8, 12t G=8) on torch 2.12, report:
   - step time (gen / GRPO / wsync breakdown)
   - questions/s, completions/s
   - mean accuracy steps 6–10
3. **Bigger**: try `BioReason 4B 1-node G=4` and `Qwen3-32B 2-node` to see if the
   nightly fixes any of the L0/CCL pain points (UR leak, IPC handle accumulation,
   banned:1 PDE).
4. **Decide**: keep both venvs, or migrate. Likely outcome: nightly stays the
   experiment branch until the next Aurora `frameworks/` module ships with a
   matching torch.

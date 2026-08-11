# Hand-edited wheel patches

`vllm_xpu_kernels` is installed as a pinned wheel (`0.1.7`) in
`/flare/ModCon/ngetty/venvs/kimi-k3-xpu-framework`, not an editable checkout
like `vllm-xpu-src`. At least one file in it has been hand-edited in place
inside `site-packages` and was never captured anywhere durable — a `pip
install --force-reinstall` or a fresh venv build silently loses the change.

The original unpatched 0.1.7 wheel is not available on public PyPI (`pip
download vllm_xpu_kernels==0.1.7` fails; only 0.0.1/0.1.3.1/0.1.12.x are
public) and no local copy of the pristine wheel was found, so this captures
the **current patched state**, not a diff against upstream.

## `vllm_xpu_kernels_0.1.7_fused_moe_interface.py`

Snapshot of `vllm_xpu_kernels/fused_moe_interface.py` as installed, mtime
2026-08-05. The patch: a `VLLM_XPU_DETERMINISTIC_MOE_GATHER` env-gated Python
fallback for the final MoE combine step, alongside the default native
`torch.ops._moe_C.moe_gather` kernel call —

```python
if os.getenv("VLLM_XPU_DETERMINISTIC_MOE_GATHER", "0") == "1":
    output.zero_()
    for row_idx in range(unpermuted_row_to_permuted_row.shape[0]):
        for slot_idx in range(unpermuted_row_to_permuted_row.shape[1]):
            row = int(unpermuted_row_to_permuted_row[row_idx, slot_idx].item())
            if 0 <= row < int(expert_first_token_offset[-1].item()):
                output[row_idx].add_(
                    gemm2_output[row]
                    * topk_weights[row_idx, slot_idx].to(gemm2_output.dtype)
                )
else:
    torch.ops._moe_C.moe_gather(output, gemm2_output, topk_weights,
                                unpermuted_row_to_permuted_row,
                                expert_first_token_offset, num_experts)
```

Used throughout the K3 investigation (see `RESULTS.md`, entries referencing
`VLLM_XPU_DETERMINISTIC_MOE_GATHER`) to isolate whether observed
non-determinism/degeneration came from the native kernel's row gather versus
elsewhere in the pipeline.

### This snapshot carries MORE than the gather toggle — it is load-bearing

The description above understates the file. The snapshot also contains the
**`situ` activation support K3 cannot run without** (`config.hidden_act ==
"situ"`), which the stock 0.1.7 wheel does not have. Measured 2026-08-11:

| file | `grep -c situ` | sha256 |
|---|---:|---|
| this snapshot | 7 | `93b0e7d3…` |
| K3 venv (`kimi-k3-xpu-framework`) | 7 | `93b0e7d3…` (identical) |
| nightly venv (`torchtune-pt-nightly-xpu`) | **0** | `78e79a97…` |

So copying this file is **mandatory**, not optional, for any venv that will
serve K3 — including the torch-2.11 venv proposed for XPU graph capture,
whose `vllm_xpu_kernels` is the same version `0.1.7` but has zero `situ`
support. Reverting it does not produce a slow-but-correct run; it produces a
model that cannot execute its own activation.

`serve_k3.sh` asserts this sha256 at job start, but only against the venv
`PYTHON=` points at.

### Reapplying after a venv rebuild

```bash
cp experiments/kimi_k3_serving/wheel_patches/vllm_xpu_kernels_0.1.7_fused_moe_interface.py \
   /flare/ModCon/ngetty/venvs/kimi-k3-xpu-framework/lib/python3.12/site-packages/vllm_xpu_kernels/fused_moe_interface.py
```

Verify the target venv is still on `vllm_xpu_kernels==0.1.7` first
(`pip show vllm_xpu_kernels`) — a version bump may have moved the
`moe_gather` call site or changed the function signature, in which case this
file needs re-diffing against the new version, not blind overwrite.

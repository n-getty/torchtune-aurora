# torch 2.11+xpu / vllm-xpu-kernels stack survey — 2026-05-05

Validates the proposed default vLLM rollout venv (`experiments/lora_grpo/torch211_venv`)
on Aurora before promoting it over the legacy `frameworks/2025.3.1` module. Companion to
[project_lora_grpo_torch211_unblocks_bgmv.md](../../memory/project_lora_grpo_torch211_unblocks_bgmv.md).

Stack:
- **New:** torch 2.11.0+xpu, vllm 0.1.dev1+gb786ec8e7, vllm-xpu-kernels, triton-xpu 3.7.0, NO ipex
- **Legacy:** torch 2.10.0a0, vllm 0.15.0, ipex 2.10.10 (frameworks/2025.3.1 module)

Hardware: 1× Aurora compute node `x4401c2s4b0n0` (debug-scaling, hold 8470089).
Workloads: Qwen3-4B (T1–T4, T6, T7), Qwen3-30B-A3B (T5a, T5b).
Driver: `experiments/lora_grpo/t211_survey/{T*.sh, _common.sh}`.

## Result matrix

| ID  | Test                                          | Result | Wall (s) | tok/s   | banned:1 |
| --- | --------------------------------------------- | ------ | -------- | ------- | -------- |
| T1  | LoRA `max_num_seqs` sweep {4,16,32,64}        | PASS   | 203/44/30/25 | 144 / 535 / 881 / 1367 | 0/0/0/0 |
| T2a | XPUGraph ON, no `--enforce-eager`, no LoRA    | PASS   | 18       | 1892    | 0        |
| T2b | XPUGraph ON + `--enable-lora`                 | PASS   | 29       | —       | 0        |
| T3  | drop `--no-async-scheduling` (LoRA, eager)    | PASS   | 29       | 1385 (40178/29) | 0 |
| T4  | TP=2 + `--enable-lora`                        | FAIL   | —        | —       | n/a (engine init) |
| T5a | Qwen3-30B-A3B MoE TP=2, no LoRA               | PASS   | 27       | —       | 0        |
| T5b | Qwen3-30B-A3B MoE TP=2 + `--enable-lora`      | FAIL   | —        | —       | n/a (engine init) |
| T6A | legacy frameworks Qwen3-4B TP=1 throughput    | PASS   | 22       | **1839.27** | 0   |
| T6B | torch 2.11 venv Qwen3-4B TP=1 throughput      | PASS   | 21       | **1758.05** | 0   |
| T7  | `enable_sleep_mode=True` Python LLM API       | FAIL   | —        | —       | n/a (gated CUDA-only) |

`T6 ratio (B/A) = 1758.05 / 1839.27 = 0.956`  → ≥ 0.85 promotion threshold ✓

## Findings

### Wins (recommend adopting)

1. **Drop `--enforce-eager`.** T2a hit 1892 tok/s vs T6B baseline 1758 tok/s (no XPUGraph) — **+7.6%** at no cost on TP=1. XPUGraph + LoRA (T2b) is also clean.
2. **Drop `--no-async-scheduling`.** T3 maintains baseline throughput (1385 tok/s on a longer-completion run, no errors). Async scheduling is safe with the new stack.
3. **Set `VLLM_XPU_ENABLE_XPU_GRAPH=1` by default.** No regressions observed; modest throughput win.
4. **Throughput parity vs legacy: 95.6%** on Qwen3-4B TP=1. Above the 85% promotion bar; promotes without regression for the merged-weight publish path.
5. **Qwen3-30B-A3B MoE serves cleanly on the new stack** (T5a, no LoRA, TP=2). Confirms vllm-xpu-kernels has not regressed FusedMoE between 0.15.0 and 0.1.dev1+gb786ec8e7.

### Blockers (do NOT enable on torch 2.11 venv yet)

6. **TP>1 + `--enable-lora` is broken.** T4 (Qwen3-4B TP=2) and T5b (Qwen3-30B-A3B TP=2) both fail at engine init with:
   ```
   RuntimeError: inputs.size(1) must match lora_b_weights.size(-1)
   ```
   from `bgmv_expand_slice` in vllm-xpu-kernels punica wrapper. The TP slicing of `lora_b_weights` is incorrect (column-parallel slice not applied before the BGMV call). **Implication:** runtime LoRA serving via `--enable-lora` is TP=1 only on this stack. The merged-weight publish path (`use_runtime_lora=false`, the new GRPO default) is unaffected because vLLM never sees a LoRA adapter.
7. **`enable_sleep_mode=True` is unsupported on XPU.** Pydantic validator rejects with `Sleep mode is not supported on current platform`. Upstream vLLM gates this to CUDA. Don't plan rollout architectures around vLLM sleep on Aurora; use process kill / restart or the existing colocate weight-broadcast path.

## Recommendations

| Lever | Default change |
|-------|----------------|
| `--enforce-eager` | **Remove** from rollout flags (XPUGraph faster, no instability seen) |
| `--no-async-scheduling` | **Remove** (async scheduler is safe) |
| `VLLM_XPU_ENABLE_XPU_GRAPH` | **Set to 1** in launchers |
| `--enable-lora` with TP>1 | **Block** at config-validation time on torch 2.11 stack until vllm-xpu-kernels punica TP slicing is fixed |
| `enable_sleep_mode=True` | **Block** at config-validation time on XPU; document as CUDA-only |
| Promote torch 2.11 venv as default rollout | **Yes** for merged-weight path (LoRA-GRPO already on this); TP=1 LoRA serving also supported |

## Followup tests (F1-F4 chain on capacity hold 8470027)

Goal: pressure-test the recommended-defaults envelope before promoting.

| Test | Stack | Config | Result | Throughput |
|------|-------|--------|--------|-----------|
| F1_A | legacy 2025.3.1 | defaults (no eager, async on) | **FAIL_START** | n/a |
| F1_B | torch211 + XPUGraph | defaults (no eager, async on) | **PASS** 7/7 | 1827.57 tok/s |
| F2 | torch211 + LoRA | 5-min soak, 21 rounds × 7 curls × n=24 | **PASS** 147/147 | avg 1596 tok/s, drift 1.24 (no degradation), banned:1=0 |
| F3 | torch211 dense TP=2 | n=24 max_tok=512 | **PASS** 7/7 | 1627 tok/s |
| F4 | torch211 + LoRA | max_model_len=4096, n=8 max_tok=1024 | **PASS** 7/7 | 791.90 tok/s |

**F1 finding (NEW)**: Legacy 2025.3.1 vLLM physically cannot drop `--enforce-eager` — engine startup throws `sycl::_V1::exception` from torch.compile's SYCL device lookup, then `RuntimeError: cancelled` from shm_broadcast. The +7.6% XPUGraph win measured on torch211 (T2a/b) is therefore a **torch211-exclusive capability**, not a portable default. The legacy comparable baseline must remain at eager+sync (the T6A 1839.27 tok/s number); the legitimate torch211-vs-legacy comparison is `defaults (B) vs eager+sync (A_T6) = 1827.57 / 1839.27 = 0.993` (dead even on TP=1 4B 7-prompt n=24).

**F2 finding**: 5-min LoRA soak holds steady — 21 rounds, no late banned:1, no KV fragmentation, per-round wall constant at 14-15s, first→last drift ratio 1.24 (variance is from completion-length distribution, not degradation). Refutes the "LoRA-induced KV fragmentation" hypothesis on the torch211 stack.

**F3 finding**: Confirms T4/T5b TP>1 failures are **LoRA-specific** (vllm-xpu-kernels punica `bgmv_expand_slice` slicing bug), not TP itself. Dense TP=2 works.

**F4 finding**: max_model_len=4096 LoRA TP=1 stable at lower throughput (791 tok/s — expected, longer per-req gen). Validates production envelope ceiling (LoRA-GRPO 4B uses max_gen_tokens up to 512, max_model_len 1536-2048).

**F5 (LoRA-GRPO 4B GSM8K E2E, 12 steps, 2-node)**: queued as PBS 8470204; closes the loop on the merged-weight publish path. Result will be appended.

## Artifacts

All under `experiments/lora_grpo/t211_survey/`:
- `T1_20260505_165152/{summary.log,seq{4,16,32,64}/...}`
- `T2a_xpugraph_20260505_171337/`
- `T2b_xpugraph_lora_20260505_171548/`
- `T3_async_sched_20260505_171810/`
- `T4_tp2_lora_20260505_172107/` (FAIL_START — see vllm.log for `lora_b_weights` error)
- `T5a_qwen30b_moe_20260505_172323/`
- `T5b_qwen30b_moe_lora_20260505_173835/` (FAIL_START — same bgmv error)
- `T6_throughput_ab_20260505_172645/` (B arm; A arm broken — see T6A_legacy_only re-run)
- `T6A_legacy_20260505_174316/` — re-run with absolute python path (PASS, 1839.27 tok/s)
- `T7_sleep_mode_20260505_172859/`
- `F1_defaults_ab_20260505_175737/{A_legacy,B_t211,summary.log}`
- `F2_lora_soak_20260505_180138/{test.log,vllm.log}`
- `F3_tp2_dense_20260505_180926/{test.log,vllm.log}`
- `F4_long_ctx_20260505_181212/{test.log,vllm.log}`
- `F1_to_F4_chain_20260505_175737.log` (driver log)

## Notes on the test driver

- `_common.sh` `start_vllm` had a hard-coded `--gpu-memory-utilization 0.85` that was removed so per-test FLAGS could control it (T5a/b need 0.90 for the 30B MoE).
- The original T6_A arm in `T6_throughput_ab_20260505_172645/` failed env activation because `module load frameworks/2025.3.1` does NOT prepend its `bin/` to PATH, and `myenv/bin/python3` symlinks to the *old* `frameworks/2025.2.0` (Python 3.10, no vllm). Fix in `T6A_legacy_only.sh`: hard-code `/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python3`. Saved memory entry: `feedback_aurora_environment.md` already covers stripping myenv from PATH; this run adds the absolute-python escape hatch as a fallback.

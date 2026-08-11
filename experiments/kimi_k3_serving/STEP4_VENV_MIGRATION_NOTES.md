# Step 4 — torch-2.11 venv for XPU graph capture: what is verified, on 2026-08-11

Free desk-work findings, gathered while the Step 1 profiling run loaded
weights. **No node time was spent on any of this.** Nothing here says the
migration is worth doing — that is Step 1's call. It says what it would cost
if the trace says to do it.

## Verified

| claim | status | evidence |
|---|---|---|
| nightly venv has torch 2.11 + XPUGraph | **CONFIRMED** | `torch 2.11.0+xpu`, `hasattr(torch.xpu,'XPUGraph') == True` |
| K3 venv inherits 2.10 from `frameworks/2025.3.1` | **CONFIRMED** | `torch 2.10.0a0+git449b176`, `supports_xpu_graph() == False` |
| both venvs point at the SAME vLLM source tree | **CONFIRMED** | both `vllm 0.1.dev1+gefb4cdf2b.xpu` → `/flare/ModCon/ngetty/vllm-xpu-src` (K3 venv) and `/lus/flare/.../vllm-xpu-src` (nightly) — same tree via two paths |
| same `vllm_xpu_kernels` version | **CONFIRMED** | `0.1.7` in both |
| nightly venv has **no** `situ` support | **CONFIRMED** | `grep -c situ fused_moe_interface.py`: K3 venv **7**, nightly **0**. sha256 differs (`93b0e7d3…` vs `78e79a97…`) |
| the wheel-patch snapshot matches the live K3 file | **CONFIRMED** | both sha256 `93b0e7d305f15d453766582fd1232294eb661986798eaee615fc15d6aad7d846` |
| ray drift | **CONFIRMED** | 2.53.0 → 2.55.1 |
| **transformers 5.7.0 vs K3 remote code** | **PASSES — the top risk is not a blocker** | see below |

## The top-listed risk did not materialize

The plan called transformers 5.7.0 (nightly) vs K3's remote code "the most
likely breakage". Tested in isolation, both entry points vLLM uses at startup
load cleanly under 5.7.0:

```
transformers 5.7.0
CONFIG OK:    KimiK3Config   model_type=kimi_k3
TOKENIZER OK: TikTokenTokenizer   vocab=163840
```

Control, same code in the K3 venv (transformers 4.57.1): identical results.

So the `pip install transformers==4.57.1` fallback is **not needed** on this
evidence. Caveat: this exercises `AutoConfig` and `AutoTokenizer` only.
`modeling_kimi_k3.py` / `modeling_kimi_linear.py` are NOT imported — K3 runs
`--model-impl vllm`, so the HF modeling code is not on the serving path, but
anything that does import it (e.g. an HF reference-logit check) is untested
under 5.7.0.

## One plan correction

The plan says K3's remote code was "written against 4.56.2". The K3 venv
actually ships **transformers 4.57.1**, so the known-good baseline is 4.57.1,
and the fallback pin in the plan (`transformers==4.57.1`) happens to be the
right version for a different reason than stated.

## What remains genuinely untested

1. **`situ` is mandatory and absent.** K3 requires `hidden_act="situ"`;
   copying the patched `fused_moe_interface.py` into the nightly venv is not
   optional. The sha256 assertion is already in `serve_k3.sh` and would catch
   a missed copy — but only for the venv it is pointed at, and `PYTHON=` is
   still hardcoded at `serve_k3.sh:36`, so pointing it at the nightly venv is
   itself an unmade change.
2. **Module shadowing.** `setup_ray_env.sh frameworks` and the ssh block load
   `frameworks/2025.3.1` on every remote node; its `LD_LIBRARY_PATH` /
   `PYTHONPATH` prepends can shadow the venv's torch. The Step 0 worker echo
   now prints `torch=…` per rank, so this is *detectable* — it has not been
   *tested*.
3. **triton 3.6.0 → 3.4.0/3.7.0.** The nightly venv has both
   `pytorch-triton-xpu 3.4.0` and `triton-xpu 3.7.0` installed; the K3 venv
   has `triton 3.6.0`. Which one wins at import is not established, and the
   Triton cache is known to mismatch across venvs
   (`memory/feedback_triton_cache_mismatch_across_venvs.md`).
4. **The capture gates themselves.** `world_size_across_dp > 1`
   (`xpu.py:216`) is 32 here and refuses capture regardless of torch version.
   The Step 0 echo confirms this is what currently trips
   (`world_size_across_dp=32 cudagraph_mode=NONE`), so even a perfect venv
   migration needs the escape hatch in Step 5.3 before anything captures.

## Live confirmation from the Step 1 run (job 8748640)

The worker echo added in Step 0 reports, from inside each of the 32 ranks:

```
K3_WORKER_GATES rank=15 local_rank=3 VLLM_XPU_ENABLE_XPU_GRAPH=0
  VLLM_KIMI_XPU_KDA_VECTORIZED=1 VLLM_KIMI_XPU_CONV1D_VECTORIZED=1
  VLLM_KIMI_XPU_KDA_CHUNKED=0 VLLM_KIMI_XPU_KDA_TRITON=0
  VLLM_KIMI_XPU_CAUSAL_CONV1D_TRITON=0 VLLM_XPU_ALLOW_TRITON_SAMPLER=0
  TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1
  torch=2.10.0a0+git449b176 has_XPUGraph=False supports_xpu_graph=False
  world_size_across_dp=32 cudagraph_mode=NONE enforce_eager=True
```

This is the first time the K3 stack has stated its own resolved configuration
per rank. It confirms both Step 0 fixes landed (the vectorized paths are on by
default and reached the workers) and that all four Step 5 capture gates are
currently tripped, with `supports_xpu_graph=False` and
`world_size_across_dp=32` being the two that a venv migration alone would not
both clear.

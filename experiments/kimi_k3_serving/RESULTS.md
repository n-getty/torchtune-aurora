# Kimi-K3 Serving Results

Record raw logs, a passing health check, and the exact PBS node list before
reporting a throughput number.

| Date | Gate | Model | Nodes | TP/PP | EP | Load source | Blocks | Output tok/s | Health log | Notes |
| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | --- | --- |
| 2026-08-03 | G0 | Qwen3-30B-A3B | x4301c1s7b0n0 | 4/1 | 0 | local `/tmp` staging | 512 | 45.87 | `gate0_8730773_30b_retry/server.log` | 8/8 requests, 128 input + 64 output tokens, concurrency 2, mean TTFT 195.06 ms; `GATE0 PASS` |
| 2026-08-03 | G0 | Qwen3-30B-A3B | x4302c2s1b0n0 | 4/1 | 0 | local `/tmp` staging | 512 | 46.90 | `gate0_8730913_manual/server.log` | Health 200; repeat benchmark 8/8 requests, 128 input + 64 output tokens, concurrency 2, mean TTFT 129.34 ms; load 12.95 s. Fixed PBS FQDN hostname resolution in `serve_k3.sh`. |
| 2026-08-03 | G0 | Qwen3-30B-A3B | x4310c4s0b0n0 | 4/1 | 0 | local `/tmp` staging | 512 | 45.00 | `gate0_8731217/server/server.log` | `debug` job 8731217; health 200, 8/8 requests, 128 input + 64 output tokens, concurrency 2, mean TTFT 198.48 ms; load 11.21 s. |
| 2026-08-03 | G0-smoke | Qwen3-0.6B | x4112c7s2b0n0+x4112c7s7b0n0 | 4/1 | 0 | Lustre | 128 | 66.80 | `two_node_8730821_clean/server/server.log` | Ray cross-node smoke; 4/4 requests, 32 input + 16 output tokens, concurrency 2, mean TTFT 131.45 ms; load 11.08 s |
| 2026-08-03 | G0-smoke | Qwen3-0.6B | x4407c6s0b0n0+x4407c6s1b0n0 | 4/1 | 0 | Lustre | 128 | 83.99 | `two_node_8730943_clean/server.log` | Ray cross-node smoke; health 200, 4/4 requests, 32 input + 16 output tokens, concurrency 2, mean TTFT 44.59 ms; repeat benchmark 0.76 s. |
| 2026-08-03 | G0-scale | Qwen3-30B-A3B | x4310c4s7b0n0+x4310c4s0b0n0 | 8/1 | 0 | Lustre | 256 | 36.96 | `qwen30_tp8_8730965/server.log` | Ray cross-node validation; health 200, load 521.65 s, 4/4 requests, 128 input + 64 output tokens, concurrency 2, mean TTFT 402.41 ms. |
| 2026-08-03 | G0-scale-smoke | Qwen3-30B-A3B | x4306c1s7b0n0+x4306c1s5b0n0 | 8/1 | 0 | Lustre | 256 | 19.83 | `qwen30_tp8_8731236_attempt2/server/server.log` | `debug` job 8731236; health 200, load 534.0 s, direct OpenAI `/v1/completions` 4/4, 64 output tokens each, sequential aggregate 19.83 tok/s (latencies 3.16/3.23/3.25/3.27 s). Initial `vllm bench serve` retry used a malformed `/lus` path; direct API measurement is retained. |
| 2026-08-03 | G0-scale-sweep | Qwen3-30B-A3B | x4016c5s1b0n0+x4016c5s6b0n0 | 8/1 | 0 | Lustre | 256 | 32.99 | `qwen30_sweep_8731284_attempt2/server/server.log` | `debug` job 8731284; aliased server model; 4/4 requests at max concurrency 2, 128 input + 64 output tokens, mean TTFT 473.89 ms, mean TPOT 53.86 ms; raw row `runs/20260803_230717/serve_2.log`. |
| 2026-08-03 | G0-scale-sweep | Qwen3-30B-A3B | x4016c5s1b0n0+x4016c5s6b0n0 | 8/1 | 0 | Lustre | 256 | 66.73 | `qwen30_sweep_8731284_attempt2/server/server.log` | `debug` job 8731284; aliased server model; 4/4 requests at max concurrency 4, 128 input + 64 output tokens, mean TTFT 357.07 ms, mean TPOT 54.98 ms; raw row `runs/20260803_230717/serve_4.log`. |
| 2026-08-03 | G0-smoke | Qwen3-0.6B | x4310c4s0b0n0+x4200c5s1b0n0 | 4/1 | 0 | Lustre | 128 | 56.25 | `two_node_8731169_attempt3/server/server.log` | `debug` job 8731169; health 200 with proxy bypass, 4/4 requests, 32 input + 16 output tokens, concurrency 2, mean TTFT 133.25 ms; load 4.51 s. |
| 2026-08-03 | EP-local-staging-smoke | Qwen3-30B-A3B | x4219c6s4b0n0 | 4/1 | 1 | local `/tmp` staging | 512 | — | `qwen30_ep1_8731362_local/server/server.log` | `debug` job 8731362; EP initialized and loaded in 6.79 s from a 57G local copy after Lustre-backed EP loads stalled at shard 0; health 200 and 4/4 direct `/v1/completions` requests returned 16 tokens each. Correctness/startup smoke only, not a throughput benchmark. |
| 2026-08-04 | EP-local-staging-smoke | Qwen3-30B-A3B | x4304c6s6b0n0 | 4/1 | 1 | local `/tmp` staging | 512 | — | `qwen30_ep1_8731448/launcher.log` | `debug` job 8731448; stage-only completed, health 200, and 4/4 direct `/v1/completions` requests returned 16 tokens each (latencies 812/621/619/669 ms). Correctness/startup smoke only, not a throughput benchmark. |

EP A/B note: the matching Qwen3-30B TP=8 EP-on leg (`8731016`) was not
health-checked or benchmarked. After more than eight minutes it had loaded
only 2/16 shards (first shard 281 seconds), with no error observed; see
`qwen30_ep8_8731016/metadata`. It is intentionally excluded from throughput
comparisons.

The fast EP preflight (`8731127`) intentionally used Qwen3-8B and was rejected
before engine startup because it is dense (`num_experts` is absent). The
launcher now rejects dense models before starting Ray; see
`qwen8_ep4_8731127/metadata`.

The Qwen3-30B TP=8 EP-on attempt (`8731310`) passed the nested MoE config
preflight and reached vLLM engine initialization, but remained at 0/16
checkpoint shards after more than three minutes. It was stopped without a
health check or throughput measurement; see
`qwen30_ep8_8731310_attempt4/launcher.log`.

The one-node Qwen3-30B TP=4 EP-on attempt (`8731340`) isolated the same issue
from Ray: all four workers initialized EP placement with 32/128 local experts,
then remained at 0/16 checkpoint shards for more than two minutes. It was
stopped without health or throughput measurement; see
`qwen30_ep1_8731340/launcher.log`.

The one-node EP local-staging smoke (`8731362`) succeeded: a 57G copy of the
checkpoint under `/tmp` loaded in 6.79 seconds, the server returned health 200,
and four direct completion requests returned 16 tokens each. This points to
concurrent EP checkpoint reads from Lustre as the cause of the earlier shard-0
stall; it does not establish a throughput result.
|  | G1 |  |  |  |  |  |  |  |  |  |
|  | G2 |  |  |  |  |  |  |  |  |  |
|  | G3 |  |  |  |  |  |  |  |  |  |
|  | G4 |  |  |  |  |  |  |  |  |  |
| 2026-08-04 | EP-local-staging-smoke | Qwen3-30B-A3B | x4219c1s7b0n0 | 4/1 | 1 | local `/tmp` staging | 512 | — | `ep_local_8731465/server.log` | `debug` job 8731465; stage-only passed, EP ranks 0-3 loaded all 16 shards, health 200, and direct `/v1/completions` returned HTTP 200 with 16 completion tokens. Correctness/startup smoke only, not a throughput benchmark. |
| 2026-08-04 | G3-startup | Kimi-Linear-48B-A3B-Instruct | x4220c4s5b0n0 | 2/1 | 0 | Lustre | 128 | — | `kimi_linear_tp2_8731537/launcher.log` | `debug` job 8731537; source vLLM `efb4cdf`, XPU kernels 0.1.7, all 20 shards loaded in 155.33 s, KV cache initialized, and `/health` returned 200. Fixed-prompt `/v1/completions` timed out after 90 s with no response; G3 generation/parity and throughput remain unpassed. |
| 2026-08-04 | G3-kda-fallback-smoke | Kimi-Linear-48B-A3B-Instruct | x4216c5s0b0n0 | 2/1 | 0 | Lustre | 64 | — | `kimi_linear_kda_8731574_retry1/server.log` | `debug` job 8731574; patched XPU correctness-first KDA recurrence loaded all 20 shards and passed `/health`; deterministic `/v1/completions` returned `" a city in the"` (4 tokens) and `" a city"` (2 tokens), with no worker traceback. Not a Gate 3 pass: no HF logits parity or throughput measurement yet. |
| 2026-08-04 | G3-parity | Kimi-Linear-48B-A3B-Instruct | x4311c4s3b0n0 | 2/1 | 0 | Lustre | 64 | — | `kimi_parity_8731702/server.log` | `debug` job 8731702; HF eager reference and XPU vLLM agree on fixed prompt `Aurora is`: HF argmax token 261 (`" a"`), top-10 logits 12.75/11.375; vLLM output top-logprob gap for tokens 261/276 is 1.375, matching the HF logit gap exactly. Health and one-token completion passed. Gate 3 throughput still unmeasured. |
| 2026-08-04 | G3-throughput | Kimi-Linear-48B-A3B-Instruct | x4117c4s3b0n0 | 2/1 | 0 | Lustre | 64 | 24.10 | `kimi_tp2_bench_8731733/server.log` | `debug` job 8731733; health 200; four deterministic requests, 12 prompt + 16 output tokens each, concurrency 2. Warm repeat: 64 output tokens / 2.65522 s = 24.10 aggregate tok/s (individual latencies 1.307–1.319 s). Cold-inclusive batch: 64 / 12.4712 s = 5.13 tok/s. Python KDA fallback; not a production performance result. |
| 2026-08-04 | K3-checkpoint | Kimi-K3 | login node | — | — | local Lustre | — | — | `verify_checkpoint.py` | 96 shards, 1,560,936,091,448 shard bytes, 497,220 indexed tensors; no symlinks or incomplete downloads. |
| 2026-08-04 | K3-XPU-preflight | Kimi-K3 | x4104c2s6b0n0 | — | — | local Lustre | — | — | `k3_xpu_preflight.out` | `debug` job 8732078, exit 0; config propagation, XPU SiTU execution, and tiny routed fused-MoE call passed, BF16 max error 0.0016615987, output shape (2, 32). Architecture/runtime preflight only; not Gate 4 serving evidence. |
| 2026-08-04 | K3-XPU-preflight-corrected | Kimi-K3 | x4311c3s5b0n0 | — | — | local Lustre | — | — | `k3_xpu_preflight_8732128/stdout.log` | `debug` job 8732128, exit 0; corrected BF16 GEMM2 probe passed after fixing the synthetic weight orientation, latent-MoE config assertions passed, SiTU max error 0.0037150383, fused-MoE output shape (2, 32). Architecture/runtime preflight only; not Gate 4 serving evidence. |
| 2026-08-04 | K3-XPU-import-fallback | Kimi-K3 | x4117c4s1b0n0 | — | — | local Lustre | — | — | `k3_xpu_preflight_8732264/stdout.log` | `debug` job 8732264, exit 0; current-source rerun passed K3 config resolution, direct model-class import, SiTU formula (max BF16 error 0.00172668695), and tiny routed fused-MoE call. The XPU-only KDA fallback avoids the native FLA KDA import segfault. Architecture/runtime preflight only; no model construction, weight loading, health check, or throughput yet. |
| 2026-08-04 | DAOS-login-inventory | Kimi-K3 | login node | — | — | DAOS pool | — | — | `daos container list AuroraGPT` | Read-only login-node audit passed: pool `AuroraGPT` is `Degraded` with rebuild complete and 253 TB NVMe free; `AuroraGPT:serving_models` is absent, so the workflow correctly refused container creation or ingest. |
| 2026-08-04 | K3-tensor-headers | Kimi-K3 | login node | — | — | local Lustre | — | — | `verify_tensor_headers.py` | Header-only safetensors validation passed: all 92 latent down/norm/up sets and all 494,592 routed MXFP4 tensors have the expected K3 shapes; no tensor payloads were loaded. |
| 2026-08-04 | K3-loader-mapping | Kimi-K3 | login node | — | — | checkpoint index | — | — | `verify_loader_mapping.py` | Offline namespace mapping passed: all 494,592 routed expert sources map to the expected `w13`/`w2` loader families, with all three latent projection/norm names present after `language_model.` normalization. |
| 2026-08-04 | K3-construction-harness | Kimi-K3 | x4300c2s0b0n0 | 1/1 | — | local Lustre config | — | — | `k3_model_construction_8732374/stdout.log` | Superseded harness record; the corrected meta-device construction verifier passed in `debug` job 8732374. |
| 2026-08-04 | K3-construction-progress | Kimi-K3 | x4117c4s1b0n0 / x4300c2s0b0 | 1/1 | — | local Lustre config | — | — | `k3_model_construction_8732350/`, `8732353/`, `8732362/` | Superseded progress record; full construction later passed in `debug` job `8732374`. |
| 2026-08-04 | K3-construction-runtime | Kimi-K3 | x4300c2s0b0n0 | 1/1 | — | local Lustre config | — | — | `k3_model_construction_8732353/` | Runtime construction reached all 93 layers, XPU MXFP4 expert selection, and Triton MLA initialization; the remaining failure was verifier-only parameter-count logic, later corrected by job `8732374`. |
| 2026-08-04 | K3-loader-contract-review | Kimi-K3 | login node | — | — | checkpoint index + patched loader | — | — | `verify_loader_mapping.py` | Reviewed K3 w1/w3→packed w13, w2→packed w2, and compressed-tensors block-scale loader paths. Contract review only; full payload loading remains unverified. |
| 2026-08-04 | K3-construction-pass | Kimi-K3 | x4300c2s0b0n0 | 1/1 | — | local Lustre config | — | — | `k3_model_construction_8732374/stdout.log` | `debug` job 8732374 exited 0: full meta-device construction passed with 2,406 parameters, 368 expert parameters, and 3 latent families. |
| 2026-08-04 | K3-payload-samples | Kimi-K3 | login node | — | — | local Lustre | — | — | `verify_payload_samples.py` | Read and validated 5 real tensors: latent down/norm/up plus packed MXFP4 w1/w2; expected shapes and finite payloads passed. |
| 2026-08-04 | K3-loader-samples | Kimi-K3 | x4117c4s3b0n0 | 1/1 | — | local Lustre | — | — | `k3_loader_samples.out` | `debug` job 8732404 exited 0; production `model.load_weights` accepted all 5 real tensors, including latent projections/norm and packed MXFP4 expert weights. |
| 2026-08-04 | K3-offline-verifiers-rerun | Kimi-K3 | login node | — | — | local Lustre + checkpoint index | — | — | `verify_*.py` | Rerun passed checkpoint integrity, loader contract, tensor headers, loader mapping, and real payload samples: 96 shards, 497,220 indexed tensors, 494,592 routed tensors, 92 latent layers. |
| 2026-08-04 | K3-topology-guard | Kimi-K3 | harness | — | — | PBS allocation | — | — | `serve_k3.sh` | TP=32 now refuses allocations with fewer than three nodes; remaining full-load/G1/G2/G4 runs require topology not available in `debug`. |
| 2026-08-04 | K3-TP32-submission | Kimi-K3 | pending `debug-scaling` job 8732626 | 32/1 | — | local Lustre | 64 | — | `k3_tp32_8732626/` | Valid three-node TP=32 serving job accepted by PBS queue `debug-scaling`; waiting for an execution host, so loading/health/generation remain pending. |
| 2026-08-04 | K3-TP32-EP-load-partial | Kimi-K3 | x4018c1s0b0n0+x4018c1s1b0n0+x4018c1s2b0n0 | 32/1 | 1 | Lustre | 64 | — | `k3_interactive_8732891_ep_gate/server/server.log` | Recoverable hold `8732891`; native vLLM source and XPU venv verified, K3 architecture/tokenizer/Ray TP=32 initialization passed, and EP-aware loading reached 37/96 shards in 11:52 before deliberate stop. No health, generation, or throughput claim; DAOS was not used. |
| 2026-08-04 | K3-TP32-EP-load-partial-2 | Kimi-K3 | x4504c3s0b0n0+x4504c3s1b0n0+x4504c4s2b0n0 | 32/1 | 1 | Lustre | 64 | — | `k3_interactive_8733039_ep_lustre/server.log` | Recoverable `debug-scaling` hold `8733039`; DAOS agent socket and dfuse launcher were absent on all three nodes. Startup, tokenizer, native K3 resolution, Ray world size 32/backend xccl, and weight loading passed; reached 80/96 shards at 38:28 before the one-hour PBS walltime expired. No health, generation, correctness, or throughput claim. |
| 2026-08-04 | K3-capacity-hold | Kimi-K3 | deleted `capacity` job 8733329 | 32/1 | — | Lustre baseline / DAOS pending | — | — | `hold_3node_capacity.sh` | Superseded recoverable three-node hold; the original 168-hour request was deleted explicitly. The hold is now bounded to 3 hours, enough for the observed approximately 1-hour full-load estimate plus health/correctness smoke and an initial benchmark. |
| 2026-08-04 | K3-DAOS-hold | Kimi-K3 | pending `debug-scaling` job 8733408 | 32/1 | — | DAOS `AuroraGPT:prism_models` | — | — | `hold_3node.sh` | DAOS-enabled three-node one-hour hold submitted with `filesystems=flare:home:daos_user_fs`; uses the existing documented `prism_models` container and will run `/flare/ModCon/ngetty/BaseMM_PRISM/scripts/setup_daos_models.sh mount` before serving. |

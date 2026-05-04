# vLLM MoE Serving Benchmark — Qwen3-30B-A3B vs Qwen3-32B Dense
**Date**: 2026-05-04  
**Node**: Aurora, 1 node (12 XPU tiles), frameworks/2025.3.1 (vLLM 0.15.0)  
**Purpose**: Find optimal vLLM serving config for GRPO rollout generation; compare MoE vs dense; evaluate EP flags

## Methodology

GRPO-mode burst latency: send G concurrent requests, measure wall time until **all** complete. This mirrors `generate_trajectory()` exactly — a synchronous burst of G sequences, not sustained throughput. Client: `experiments/ep_parallelism/vllm_moe_latency_client.py` (threading, stdlib only).

Sweep: batch (G) = 2, 4, 8 × max_tokens = 128, 256. 2 runs per point, best reported.  
Input length: 128 tokens. `ignore_eos=True` for fair comparison.

## Configs tested

| Config | Model | Tiles | TP | DP | EP flag |
|--------|-------|-------|----|----|---------|
| A | Qwen3-30B-A3B (MoE) | 4 | 4 | 1 | none |
| B | Qwen3-30B-A3B | 4 | 4 | 1 | `--enable-expert-parallel` |
| C | Qwen3-30B-A3B | 2 | 2 | 1 | none |
| D | Qwen3-30B-A3B | 2 | 2 | 1 | `--enable-expert-parallel` |
| E | Qwen3-30B-A3B | 12 | 4 | 3 | `--enable-expert-parallel` (AllToAll fires) |
| H | Qwen3-30B-A3B | 12 | 4×3 | indep | none, 3 servers + round-robin proxy |
| I | Qwen3-30B-A3B | 12 | 4×3 | indep | EP per instance (AgRs), 3 servers + proxy |
| J | Qwen3-32B (dense) | 4 | 4 | 1 | none |

## Results (G=4, max_tokens=256 — GRPO operating point)

| Config | best_s | vs A | tok/s @ G=8,256 |
|--------|--------|------|-----------------|
| **A: MoE TP=4, no EP** | **11.3s** | — | **173** |
| B: MoE TP=4, EP dp=1 | 11.6s | +3% | 168 |
| C: MoE TP=2, no EP | 11.7s | +4% | 160 |
| D: MoE TP=2, EP dp=1 | 12.2s | +8% | 154 |
| **E: MoE TP=4, EP dp=3 (AllToAll)** | **17.9s** | **+58%** | **114** |
| H proxy: 3× TP=4 no EP | 12.2s | +8% | 156 |
| H single: 1-of-3 TP=4 | 12.5s | +11% | 155 |
| I proxy: 3× TP=4 EP | 12.1s | +7% | 152 |
| I single: 1-of-3 TP=4 EP | 13.7s | +21% | 141 |
| **J: Dense 32B TP=4** | **13.5s** | **+19%** | **151** |

Full data (all batch/token combos):

```
config                 tp_dp      ep      batch  max_tokens  best_s  avg_s  tok/s
MoE_tp4_noep          tp4dp1     no      2      128         5.5     5.5    46
MoE_tp4_noep          tp4dp1     no      2      256         10.8    10.8   47
MoE_tp4_noep          tp4dp1     no      4      128         5.7     5.7    89
MoE_tp4_noep          tp4dp1     no      4      256         11.3    11.3   90
MoE_tp4_noep          tp4dp1     no      8      128         6.0     6.1    169
MoE_tp4_noep          tp4dp1     no      8      256         11.8    11.8   173
MoE_tp4_ep_dp1        tp4dp1     yes_dp1 2      128         5.6     5.7    45
MoE_tp4_ep_dp1        tp4dp1     yes_dp1 2      256         11.1    11.1   46
MoE_tp4_ep_dp1        tp4dp1     yes_dp1 4      128         5.8     5.8    88
MoE_tp4_ep_dp1        tp4dp1     yes_dp1 4      256         11.6    11.6   88
MoE_tp4_ep_dp1        tp4dp1     yes_dp1 8      128         6.1     6.2    166
MoE_tp4_ep_dp1        tp4dp1     yes_dp1 8      256         12.1    12.2   168
MoE_tp2_noep          tp2dp1     no      2      128         5.4     5.5    47
MoE_tp2_noep          tp2dp1     no      2      256         11.0    11.0   47
MoE_tp2_noep          tp2dp1     no      4      128         5.9     5.9    87
MoE_tp2_noep          tp2dp1     no      4      256         11.7    11.7   87
MoE_tp2_noep          tp2dp1     no      8      128         6.3     6.4    159
MoE_tp2_noep          tp2dp1     no      8      256         12.7    12.8   160
MoE_tp2_ep_dp1        tp2dp1     yes_dp1 2      128         5.7     5.8    44
MoE_tp2_ep_dp1        tp2dp1     yes_dp1 2      256         11.4    11.4   45
MoE_tp2_ep_dp1        tp2dp1     yes_dp1 4      128         6.1     6.1    85
MoE_tp2_ep_dp1        tp2dp1     yes_dp1 4      256         12.2    12.2   84
MoE_tp2_ep_dp1        tp2dp1     yes_dp1 8      128         6.7     6.8    151
MoE_tp2_ep_dp1        tp2dp1     yes_dp1 8      256         13.2    13.3   154
MoE_tp4_ep_dp3        tp4dp3     yes_dp3 2      128         8.9     8.9    29
MoE_tp4_ep_dp3        tp4dp3     yes_dp3 2      256         17.3    17.4   29
MoE_tp4_ep_dp3        tp4dp3     yes_dp3 4      128         9.2     9.2    56
MoE_tp4_ep_dp3        tp4dp3     yes_dp3 4      256         17.9    18.0   57
MoE_tp4_ep_dp3        tp4dp3     yes_dp3 8      128         9.2     9.2    111
MoE_tp4_ep_dp3        tp4dp3     yes_dp3 8      256         17.9    17.9   114
MoE_3xtp4_noep_proxy  tp4_3indep no      2      128         6.1     6.2    42
MoE_3xtp4_noep_proxy  tp4_3indep no      2      256         11.9    11.9   43
MoE_3xtp4_noep_proxy  tp4_3indep no      4      128         6.0     6.2    83
MoE_3xtp4_noep_proxy  tp4_3indep no      4      256         12.2    12.6   81
MoE_3xtp4_noep_proxy  tp4_3indep no      8      128         6.6     6.7    153
MoE_3xtp4_noep_proxy  tp4_3indep no      8      256         13.1    13.2   156
MoE_1xtp4_noep_single tp4_1of3   no      2      128         6.4     6.4    40
MoE_1xtp4_noep_single tp4_1of3   no      2      256         12.7    12.8   40
MoE_1xtp4_noep_single tp4_1of3   no      4      128         6.5     6.6    78
MoE_1xtp4_noep_single tp4_1of3   no      4      256         12.5    12.6   81
MoE_1xtp4_noep_single tp4_1of3   no      8      128         6.9     7.0    147
MoE_1xtp4_noep_single tp4_1of3   no      8      256         13.2    13.2   155
MoE_3xtp4_ep_proxy    tp4_3indep yes_dp1 2      128         5.9     5.9    43
MoE_3xtp4_ep_proxy    tp4_3indep yes_dp1 2      256         11.5    11.7   44
MoE_3xtp4_ep_proxy    tp4_3indep yes_dp1 4      128         6.0     6.3    81
MoE_3xtp4_ep_proxy    tp4_3indep yes_dp1 4      256         12.1    12.6   81
MoE_3xtp4_ep_proxy    tp4_3indep yes_dp1 8      128         6.8     6.8    150
MoE_3xtp4_ep_proxy    tp4_3indep yes_dp1 8      256         13.4    13.5   152
MoE_1xtp4_ep_single   tp4_1of3   yes_dp1 2      128         6.8     6.8    38
MoE_1xtp4_ep_single   tp4_1of3   yes_dp1 2      256         13.4    13.4   38
MoE_1xtp4_ep_single   tp4_1of3   yes_dp1 4      128         6.9     6.9    74
MoE_1xtp4_ep_single   tp4_1of3   yes_dp1 4      256         13.7    13.8   74
MoE_1xtp4_ep_single   tp4_1of3   yes_dp1 8      128         7.3     7.4    139
MoE_1xtp4_ep_single   tp4_1of3   yes_dp1 8      256         14.5    14.5   141
Dense32B_tp4          tp4dp1     no      2      128         6.5     6.7    38
Dense32B_tp4          tp4dp1     no      2      256         12.9    13.0   39
Dense32B_tp4          tp4dp1     no      4      128         6.8     6.8    75
Dense32B_tp4          tp4dp1     no      4      256         13.5    13.5   76
Dense32B_tp4          tp4dp1     no      8      128         6.9     6.9    148
Dense32B_tp4          tp4dp1     no      8      256         13.5    13.6   151
```

## Key findings

### 1. Config A (TP=4, no EP) is optimal — production config is correct
Batch scaling is nearly free: G=2→G=8 at 256 tok adds only 1s (+9%) while delivering 4× the throughput. Do not change the production vLLM launch.

### 2. Config E (native DP+EP AllToAll) is catastrophically slow — +58%
vLLM 0.15.0 `--data-parallel-size N` engines are linked: `has_unfinished_dp()` fires a blocking `all_reduce(ReduceOp.MAX)` every 32 scheduler steps to detect when all DP engines are idle. For GRPO's burst-and-wait pattern (send G reqs, block until all complete), this all-reduce lands at exactly the wrong time. **Never use `--data-parallel-size` with EP for GRPO.**

### 3. EP dp=1 (AgRs dispatch) is a uniform ~3% regression
With `data_parallel_size=1`, `--enable-expert-parallel` changes the weight layout but the AllToAll dispatch does not fire (`use_all2all = is_ep_communicator and dp > 1` in vLLM 0.15.0). The AgRs path adds overhead with no routing benefit. Do not add `--enable-expert-parallel` to production vLLM launches.

### 4. TP=4 beats TP=2; gap widens with batch size
At G=8, 256 tok: TP=4 (11.8s) vs TP=2 (12.7s) = 8% faster. MoE with TP=4 has lower per-rank expert load despite the AllReduce cost; TP=2 is memory-bandwidth constrained at larger batches.

### 5. MoE (A) is 16–19% faster than Dense 32B (J) at same hardware
A: G=4, 256 tok → 11.3s vs J: 13.5s (+19%). This is real but far below the theoretical speedup (~5–10×) implied by active-parameter count (~3B vs 32B active per token). The gap is explained by bandwidth-bound inference: at small batch sizes the decode phase is limited by HBM bandwidth, not FLOP count. The vLLM XPU Unquantized MoE backend likely reads all expert weights (not just active ones) to perform routing efficiently, eliminating most of the bandwidth savings. A truly sparse XPU MoE kernel could recover the theoretical 5–10× advantage.

### 6. Independent multi-instance (H proxy) adds ~8% overhead
3× separate TP=4 servers with round-robin proxy: G=4, 256 tok → 12.2s vs A 11.3s. The proxy scheduling + imperfect load distribution erases most of the per-instance load reduction benefit. Not worth the complexity given A's already excellent batch scaling.

## GRPO implications

The 73s gen time in EP=16 training is **not** vLLM serving latency — pure serving for G=4 at 256 tok is 11.3s. The remaining ~62s is wsync_gather overlap (70s, deferred) and logprobs/trajectory computation. vLLM serving is not the bottleneck.

The current EP=16 production vLLM config (TP=4, no EP, one instance per node, G/2 requests per instance) is optimal for single-node serving. Total 2-node serving latency: ~11s for G=2 per instance ≈ consistent with observed 73s (generation timer includes logprobs + gather overhead).

## Tooling

- Latency client: `experiments/ep_parallelism/vllm_moe_latency_client.py`
- Round-robin proxy: `experiments/ep_parallelism/proxy_round_robin.py`
- Full matrix script: `experiments/ep_parallelism/hold_vllm_moe_bench.sh` (A–J, 1.5h slot)
- Lite script: `experiments/ep_parallelism/bench_lite.sh` (A–D, fits 1h debug slot after 600s→1200s timeout fix)
- Final script: `experiments/ep_parallelism/bench_final.sh` (E, H, I, J, 1h debug slot)

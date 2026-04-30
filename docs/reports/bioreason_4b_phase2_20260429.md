# BioReason 4B GRPO — Phase 2 (2-node asymmetric) status (2026-04-29)

## TL;DR

**Phase 2 end-to-end pipeline VALIDATED.** Two-node asymmetric topology (11 train ranks on TRAIN_NODE + 12 vLLM HTTP servers on VLLM_NODE) trains BioReason-Pro 4B GRPO with working multimodal prompt_embeds wire and shared-FS raw_bytes weight sync.

- **Run 49** (job 8457145, debug-scaling): 5/5 steps clean. Weight sync silently failed (pointed at node-local `/dev/shm`, vLLM tile log showed `Not found` errors); training completed because raw_bytes failures are non-fatal background-thread exceptions.
- **Run 50** (same job, after fix): 5/5 steps clean. **4 consecutive successful weight syncs** confirmed via vLLM tile logs (`load_weights_from_raw: 399 params in ~9s`).

## Topology

```
TRAIN_NODE (x4719c5s2b0n0)            VLLM_NODE (x4719c5s4b0n0)
─────────────────────────────         ──────────────────────────────
11× FSDP1 SHARD_GRAD_OP ranks         12× vLLM HTTP servers
  (training_pg, single-node CCL)        (DP=12, ports 8001-8012)
  ignored=[_embed,                      ZE_AFFINITY_MASK=0..11
           protein_encoder,             --enable-prompt-embeds
           go_encoder]                  --tensor-parallel-size 1
                                        WeightSyncFromFileExtension
       │                                       ▲
       │  prompt_embeds (POST)                 │
       │  base64(torch.save(bf16))             │
       └──────────── HTTP ────────────────────►│
                                               │
       ┌────────────── HTTP /collective_rpc ──►│
       │  load_weights_from_raw(path)
       │
   weight_update.raw on shared FS:
   /lus/flare/.../outputs/wsync/weight_update.raw
       (write side: rank 0; read side: all 12 vLLM workers)
```

## Run 50 timing & wsync

| Step | total | gen | grpo | wsync wait | wsync save | wsync load | KL | grad_norm |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 52.3s | 29.7s | 17.3s | 5.2s | 6.3s | 10.7s | 0.0025 | 5.4 |
| 1 | 95.8s | 33.9s | 14.1s | 47.6s | 8.0s | 8.9s | 0.0023 | — |
| 2 | 92.4s | 32.6s | 13.5s | 46.2s | 7.1s | 8.5s | 0.0016 | 5.3 |
| 3 | 88.5s | 31.3s | 13.8s | 43.4s | 7.2s | 8.8s | 0.0020 | 4.1 |
| 4 | 87.3s | 32.2s | 15.0s | 40.0s | — | — | 0.0027 | 0.01 |

- **Save side** (rank 0, DAOS write): 6.3-8.0s for 8.5 GB → 1.0-1.3 GB/s.
- **Load side** (vLLM tile read+apply): 8.5-10.7s, dominated by file read (~8.6s) not param load (~0.2s).
- **Filter**: 399 backbone-only params (Qwen3-4B), aux modules (ESM3 / GO / projector / embed) excluded — those live on the train side only.
- **KL evolves** (0.0016-0.0027 across steps) and grad_norm varies meaningfully (0.01-5.4) — confirms vLLM is generating from the synced weights, not stale init.

## What this validates (vs `bioreason_4b_status_20260429.md`)

The earlier status report ("Recommendation order: A first, A+allocator, B if A insufficient, **C if A and B both fail**") proposed multi-node BioReason as the architectural escape from single-node memory pressure. Phase 2 is a different escape: keep training on a single node, but move vLLM (which was contending for the same XPU tiles) to its own node entirely. This:

1. Removes the 11+1 colocation pressure (vLLM no longer pins one tile + 12-15 GiB on TRAIN_NODE).
2. Decouples train-side and infer-side allocator state — no shared L0 IPC handle pool, no banned:1 from vLLM weight reload landing on train-side cached pages.
3. Lets vLLM run **DP=12** instead of TP=1 on a single tile — generation throughput scales with rollouts.
4. The wire format (HTTP prompt_embeds) means vLLM doesn't need any of the BioReason multimodal stack — it sees plain Qwen3-4B and ignores the aux `.pt` files in the checkpoint.

## Implementation pieces (final)

### Recipe — `recipes/dev/grpo_full_finetune_distributed_xpu.py`
- **FSDP1 wrap gate** (line 2874): extended from `dedicated_rank` to also cover `vllm_mode == "server"`. In server mode, all `world_size` ranks are training ranks (no rank reserved for vLLM); `_wsync_pg = None` (no torch.distributed PG, weights ship over HTTP).
- **Backbone-only filter** (line 4026): `_accept_and_rename` strips `_fsdp_wrapped_module.` and `_checkpoint_wrapped_module.` prefixes, accepts only `backbone.*`, then strips the `backbone.` prefix to match vLLM's Qwen3 namespace.
- **FSDP1 weight gather gate** (line 4034): relaxed from `_dp_replicate > 1` to `_use_fsdp1` alone — BioReason runs FSDP1 with `dp_replicate=1` (pure shard).
- **Weight sync path** (line 2020): env-overridable. Default `/dev/shm/torchtune/weight_update.raw` (node-local tmpfs, fastest single-node). Override with `TORCHTUNE_WEIGHT_SYNC_PATH` for cross-node — must point at a shared filesystem (DAOS `/lus/flare`) so VLLM_NODE can read what TRAIN_NODE writes.

### Config — `recipes/configs/dev/production/bioreason_4b_grpo_2node_server_xpu.yaml`
- `vllm_mode: server`, `vllm_weight_sync: true`, `vllm_weight_sync_method: raw_bytes`.
- `vllm_url` filled in at runtime by launcher (multi-URL DP=12 round-robin via `_vllm_clients` ThreadPoolExecutor).
- G=4, fbs=4, max_seq_len=1024, max_generated_tokens=1024 — matches single-node baseline (run 41/42).

### Launcher — `experiments/bioreason/run_bioreason_2node_server.sh`
- Discovers TRAIN_NODE/VLLM_NODE from `$PBS_NODEFILE` (first/second unique node).
- Stages `bioreason-pro-sft` to `/tmp/torchtune/` on both nodes (~15 GB/s DAOS).
- Cleans stale vLLM (`pkill -9 -f vllm.entrypoints.openai.api_server` etc.) and shared-FS wsync file.
- SSH-spawns 12 vLLM HTTP servers on VLLM_NODE in a single bash heredoc (each with `ZE_AFFINITY_MASK=$i`, `--enable-prompt-embeds`, `--worker-extension-cls torchtune.dev.vllm_weight_sync_worker.WeightSyncFromFileExtension`).
- Health-check loop polls `/health` on each port (timeout 900s).
- Launches `torch.distributed.run --standalone --nproc_per_node=11` on TRAIN_NODE.
- `export TORCHTUNE_WEIGHT_SYNC_PATH=/lus/flare/.../outputs/wsync/weight_update.raw`.

### vLLM worker extension — `torchtune/dev/vllm_weight_sync_worker.py`
- `load_weights_from_raw(path)`: reads 8-byte header len + JSON header + raw tensor bytes (BF16 stored as int16 bit pattern, reinterpreted on load). `frombuffer` is zero-copy from the mmap'd file. ~9s for 399 params / ~8.5 GB on DAOS read.

## Open optimization (before capacity-job long-horizon run)

**`other=40-47s` per step is wsync-wait blocking next gen.** The async-overlap design intends for `_save_raw_bytes` + HTTP POSTs to all 12 servers to run in a background thread that completes before next gen starts. Empirically that's not happening:

- save side (6-8s) + load side (8-10s) = ~17s of work per server, but `other` shows ~45s blocking
- 12 servers × ~10s sequential load would be ~120s; ~45s suggests partial parallelization

Either:
- `_wait_for_sync_complete` is called eagerly at start of next step (gating gen on sync done), or
- the ThreadPoolExecutor fan-out POST in `_post_weights_to_vllm` waits for all 12 sequentially before returning.

Investigate before the 4h capacity run (job 8456716). Single-node baseline was 45s/step; 95s/step with working wsync is acceptable for a 5-step proof but doubles wall for the long-horizon run.

## Launcher gotcha (worth recording)

`experiments/bioreason/run_bioreason_2node_server.sh` uses `bash -s <<EOF &` for the vLLM SSH heredoc + a cleanup `trap EXIT`. If the parent shell gets HUP'd (transient ssh exit), the trap fires and kills both vLLM and the not-yet-launched training. **Working pattern**:

```bash
ssh -f TRAIN_NODE "nohup /tmp/wrapper.sh </dev/null >/dev/null 2>&1 &"
```
where `wrapper.sh` exec's the launcher with `PBS_NODEFILE` exported. Plain `setsid + disown` from `ssh "..."` does NOT survive the controlling ssh closing — the child SSH connection inside the launcher inherits the dying parent.

## Job context

| Job | Queue | Nodes | Walltime | Status | Used for |
|---|---|---|---|---|---|
| 8457145 | debug-scaling | 2 | 1h | Done (R→E at 1:00) | Run 49 + Run 50 (Phase 2 validation) |
| 8456716 | capacity | 2 | 4h | Queued | Long-horizon (50+ steps) once async wsync overlap is verified |

## What's next

1. **Diagnose `other=45s` wsync-wait** — read `_wait_for_sync_complete` and `_post_weights_to_vllm` paths; either the gate is eager or the fan-out is sequential. Fix should bring step time back to ~50s.
2. **Capacity job (8456716) long-horizon** — once async overlap restored, run 50+ steps to confirm no slow-growth memory leaks (the single-node baseline showed 20/20 clean, but cross-node DAOS file churn is a new dimension).
3. **Reward signal sanity** — run 50 saw rewards 0.015-0.038 with successes=0 across all 5 steps. KL is moving (so the policy is updating), but 5 steps is way too short to see reward dynamics. Long-horizon run will tell us if the GRPO loop is actually pushing in a useful direction.

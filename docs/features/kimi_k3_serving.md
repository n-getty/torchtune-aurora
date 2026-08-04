# Kimi-K3 Serving on Aurora

This feature tracks the infrastructure-first path for serving text-only
`moonshotai/Kimi-K3` on Aurora XPU. The executable harness is under
`experiments/kimi_k3_serving/`; it does not submit PBS jobs or modify the
adjacent vLLM source tree.

## Gates

- **G0:** Qwen3-30B-A3B, TP=4, single node completes `vllm bench serve`.
- **G1:** Qwen3-Coder-480B, TP=32, three nodes loads from DAOS faster than
  staging.
- **G2:** TP=32 reproduces the existing ~106 output tok/s baseline and records
  EP on/off at serving concurrency.
- **G3:** PASS — Kimi-Linear-48B-A3B-Instruct has fixed-prompt eager-reference
  parity and a recorded TP=2 XPU throughput result.
- **G4:** Kimi-K3 generates correct text at TP=32 on three or more nodes and
  has a same-node, health-checked throughput result.

## Workflow

1. Build the dedicated stack with `experiments/kimi_k3_serving/build_stack.sh`.
   If the planned kernel source checkout is unavailable, it installs the
   vLLM-pinned `vllm-xpu-kernels==0.1.7` wheel.
   On Aurora, launch with the framework Python, the editable vLLM source on
   `PYTHONPATH`, and `/opt/aurora/26.26.0/oneapi/2025.3/lib` on
   `LD_LIBRARY_PATH`; otherwise the XPU kernel wheel cannot load its SYCL ABI
   and vLLM falls back to `UnspecifiedPlatform`.
2. After the venv has generated its Triton cache, precompile it with
   `VENV=/flare/ModCon/ngetty/venvs/vllm-serve-xpu \
   experiments/kimi_k3_serving/precompile_triton_zebin.sh`. The helper accepts
   `TRITON_CACHE_DIR` when the cache is outside the venv and refuses to run
   against a missing cache.
3. Use a recoverable hold for iterative serving. `submit_debug.sh` is limited
   to short one- or two-node smoke tests. For TP=32 K3 work, submit
   `hold_3node.sh` with `daos_user_fs` for DAOS validation, or
   `hold_3node_capacity.sh` through `submit_capacity.sh` for a longer Lustre
   run. The DAOS hold uses the existing documented `AuroraGPT:prism_models`
   container; do not create `serving_models` when `prism_models` is available.
   The debug-scaling DAOS hold provides a
   three-node, three-hour hold with enough walltime for full checkpoint
   loading and initial validation. End a
   hold explicitly with `qdel -W force JOBID`; do not wait for PBS to reclaim
   it naturally.
4. Ingest models with `daos_ingest.sh` after reviewing the dfuse mount.
   For Kimi-K3, first run
   `python3 experiments/kimi_k3_serving/verify_checkpoint.py PATH`; it
   requires all 96 shards, the published index size, no symlinks, and no
   incomplete Hub downloads. Then run
   `python experiments/kimi_k3_serving/verify_tensor_headers.py PATH` with
   the K3 framework environment to validate latent-MoE and MXFP4 headers
   without loading payloads. Set `VERIFY_CHECKPOINT=1` on `daos_ingest.sh`
   or `serve_k3.sh` to enforce the checkpoint gate automatically.
5. Start a server with `serve_k3.sh`. `--blocks` is mandatory because Aurora's
   per-tile memory reporting makes automatic KV sizing unreliable. Set
   `--served-model-name` to a short identifier and pass that identifier as
   `SERVED_MODEL` to the benchmark client; keep `TOKENIZER` pointed at the
   local checkpoint directory.
6. Run `sweep_topology.sh` against a healthy server. It writes one TSV row per
   concurrency cell with output/peak throughput, mean TTFT/TPOT, node IDs,
   benchmark log, and optional server log; append verified rows to `RESULTS.md`.

Run `daos_probe.sh --login` on a login node to inventory the `AuroraGPT` pool
and verify whether `serving_models` exists. Login-node DAOS and existing dfuse
mounts are supported. Run the probe without `--login` inside the same `debug`
allocation before serving to verify compute-node dfuse access. The probe never
creates or modifies a container; the current inventory shows that
`AuroraGPT:serving_models` still needs to be created before model ingest.

For the first node allocation, run `gate0_smoke.sh` with the Qwen3-30B-A3B
default. It waits for `/health`, runs eight short random requests, and leaves
server and benchmark logs under the gate directory.

One-node servers use vLLM's direct `mp` executor and do not start Ray. Two-node
servers use Ray for cross-node worker placement. Restart the server when
changing EP or batching parameters; the benchmark client cannot reconfigure a
running server.

### EP checkpoint loading

On Qwen3-30B-A3B, EP initialization itself works on XPU, but concurrent EP
checkpoint reads from the Lustre model path stalled at shard 0 in both one-node
and two-node tests. A one-node TP=4/EP=1 smoke succeeded after copying the 57G
checkpoint to local `/tmp`: weights loaded in 6.79 seconds, `/health` returned
200, and four direct completion requests succeeded. Until a faster shared
checkpoint path is validated, use the long-lived capacity hold for K3 EP
iteration rather than staging the 1.56 TB checkpoint into node-local `/tmp`.
The default lazy loader is retained for local storage; use
`--safetensors-load-strategy eager` for the Lustre checkpoint because it avoids
network-filesystem random reads. For multi-node Ray launches,
`serve_k3.sh --stage-model` copies the checkpoint independently to `/tmp` on
every allocated node and writes a completion marker only after the copy is
complete; a remote worker must never be pointed at the launcher node's local
path. Use `serve_k3.sh --stage-only --model PATH` inside an allocation to
validate all node-local copies and markers without starting vLLM. The bounded
`experiments/kimi_k3_serving/stage_only_capacity.sh` helper runs that check on
one capacity node when the `debug` queue is unavailable. Server metadata also
records `executor_ready` and `server_start`; these are orchestration markers,
not health checks. Treat `/health` and request results as the authoritative
serving evidence.

The vLLM model dispatch, KDA/XPU path, MLA path, and SiTU/MXFP4 kernel work are
owned by `/flare/ModCon/ngetty/vllm-xpu-src` and `vllm-xpu-kernels`; they are
not silently duplicated in torchtune. The current external XPU fused-MoE
wrapper implements K3's beta-scaled SiTU-GLU formula and passes the recorded
real-XPU preflight. Full MXFP4 checkpoint loading remains a separate K3 gate.

The current Kimi-Linear XPU experiment includes a correctness-first KDA
recurrence fallback in the editable vLLM source at
`/flare/ModCon/ngetty/vllm-xpu-src/vllm/model_executor/layers/kda.py`. It is
useful for isolating model correctness from the unavailable Triton KDA path,
but its Python recurrence is not a production throughput implementation. The
follow-up `G3-parity` run matches the HF fixed-prompt next-token ranking and
logit gap, and `G3-throughput` records 24.10 aggregate output tok/s warm on
TP=2; together these satisfy Gate 3 with an explicit fallback-performance
caveat. The K3 `debug` preflight (`8732264`) now also passes direct model-class
import, SiTU, and routed fused-MoE checks, but does not establish Gate 4.

## Safety

Use `experiments/kimi_k3_serving/submit_debug.sh` only for short smoke
allocations. Use `experiments/kimi_k3_serving/submit_capacity.sh` for the
recoverable three-node K3 hold. Explicitly force-delete failed or superseded
holds with `qdel -W force JOBID` instead of allowing natural PBS reclaim.

When launching over SSH, `serve_k3.sh` discovers the actual PBS aux nodefile
if the manually exported `PBS_NODEFILE` suffix is stale, and waits briefly for
PBS to publish that file after allocation start. The authoritative
allocation remains `qstat -f JOBID` → `exec_host`.

Use `CCL_PROCESS_LAUNCHER=none` and `CCL_ATL_TRANSPORT=ofi` for Ray/SSH
launches. Keep `--enforce-eager`, `TORCH_COMPILE_DISABLE=1`, and explicit
`num_gpu_blocks_override` until a measured experiment justifies changing them.
The launcher bounds SSH setup, verifies remote Ray workers remain alive before
starting vLLM, and stops local and remote Ray processes when the server exits.

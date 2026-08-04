# MoE Dense-Parity Investigation: Reframed Results

This report separates kernel parity, EP scaling, and capacity/value. The
validated Qwen3-30B-A3B versus Qwen3-4B comparison remains the Tier 1
reference; no new hardware result is inferred by this document.

The schema-valid kernel-parity control is recorded in
`experiments/qwen3_moe/kernel_parity_manifest.yaml`; the historical baseline
manifest remains available for detailed run provenance.

## Decision table

| Question | Current answer | Evidence/status |
| --- | --- | --- |
| Highest repeatable MoE per-GPU MFU | Qwen3-30B-A3B EP8, local batch 1 | 3.52% validated baseline |
| Highest aggregate throughput | EP16 candidate is not promoted | 2,739.41 tok/s aggregate, but 0.826% MFU and 46.94% scaling efficiency |
| Best EP scaling point | EP8 node-local baseline | EP16 larger-token-volume study pending |
| Closest strict dense-parity result | EP8 batch 1 | 729.53 tok/s/GPU; parity threshold remains 95% of both dense metrics |
| Best capacity/value result | Not established | Larger-model candidate selection and two-node control pending |

The winner fields above are mirrored by `decision_categories` in each
standalone manifest. A pending winner is intentionally not replaced by an
experimental partial result.
The EP scaling report also requires explicit `repeatable: true` or
`independent_repeats >= 2` before populating any performance winner category;
single sealed EP8/EP16 legs remain measured rows but cannot become promotion
winners.

## EP8/EP16 scaling evidence

| Configuration | Token/GEMM evidence | Communication evidence | Throughput/MFU | Decision |
| --- | --- | --- | --- | --- |
| EP8, local batch 1 | Validated baseline; detailed routed-token and grouped-GEMM artifact pending | Fresh node-local measurement pending | 729.53 tok/s/GPU; 3.52% MFU | Current repeatable MoE MFU winner |
| EP8, local batch 2 | True larger-token-volume result exists; projection and imbalance breakdown pending | Canonical AG/RS attribution pending | ~803.13 tok/s/GPU; strict parity not met | Diagnostic frontier, not promoted |
| EP16, tested workload | Existing candidate does not establish expert-shape scaling attribution | Cross-node collective breakdown pending | 2,739.41 aggregate tok/s; 0.826% MFU; 46.94% efficiency | Rejected for this workload/regime |
| EP16, larger token volume | Not measured | Not measured | Not measured | Required before generalizing EP16 rejection |

The July 28 batch-2 diagnostic logs retain the best steady-state frontier
(`803.13 tok/s/GPU` in `experiments/qwen3_moe/runs/qwen3_30b_a3b_sft_ep8_seq4096_synthetic/logs/log_1785255348.txt`),
but the launcher terminated with rank-local `SIGSEGV` during distributed
teardown and did not produce a sealed measurement artifact. The later
metadata-only and combine A/B logs likewise terminate with `SIGSEGV`; they are
diagnostic evidence only and cannot replace the fresh-node promotion gate.
No hardware artifact currently measures
`TORCHTUNE_EP_FUSED_ALLTOALL_ROUTING=1`, so fused routing remains an opt-in
candidate despite passing the CPU/distributed equivalence tests.

When model-state batch scaling is memory-limited, the manifest now provides
`experiments/qwen3_moe/benchmark_routed_token_volume.py` as a separate
synthetic routed-token diagnostic. It varies expert GEMM token volume and
imbalance without loading model state or running collectives. Its output is
explicitly `synthetic_routed_token_diagnostic` with `promotion_artifact: false`;
it can explain expert-compute sensitivity but cannot establish EP scaling,
MFU, or dense parity.
Use `--execution-path both` on a device with `torch._grouped_mm` to time the
grouped-MM and expert-loop implementations at the same routed-token volumes.
That mode first compares their outputs for every volume and aborts on a shape or
numeric mismatch; each timing row records the corresponding equivalence result.
The benchmark still reports no promotion metric, and its CPU fallback remains
the expert-loop path only.
The JSON also records the selected seed and dtype so repeated diagnostics can
use identical synthetic inputs and distinguish numerical-tolerance changes
from workload changes.
Paired output also reports `grouped_mm_speedup`, defined as expert-loop mean
seconds divided by grouped-MM mean seconds; values below one indicate that
grouped-MM is slower for that synthetic volume.

A CPU-only expert-loop sweep on July 29 (8 experts, model dimension 64,
hidden dimension 128, five timed iterations) illustrates the intended use of
this diagnostic: balanced volumes of 256, 512, 1024, and 2048 routed tokens
measured approximately 0.751, 0.840, 1.002, and 1.315 ms respectively, while
tokens/second increased from about 0.34M to 1.56M as launch overhead was
amortized. Increasing the synthetic imbalance factor from 1 to 4 changed the
reported imbalance ratio from 1.0 to about 1.6, but did not establish a
hardware expert-kernel conclusion. These CPU numbers are diagnostic only and
must not be used as XPU MFU, EP scaling, or dense-parity evidence.

The EP8/EP16 comparison is therefore a measurement plan, not a claim that
EP16 is intrinsically unsuitable. The pending artifacts must report local and
global routed tokens, tokens per local expert, zero-token experts, grouped-GEMM
shapes and projection timings, and, when aligned grouped GEMM is used, both
pre-padding routed counts and aligned compute counts. The resulting
`padding_tokens` and `padding_fraction` fields separate alignment waste from
routing imbalance. Artifacts must also report collective identity, memory, and
device health. Each rank artifact also records host, PBS job identity, local
rank, device type/index, source revision, uncommitted-change state, and the
declared warmup/measurement/steady-state window. The sealing helper validates
the required MoE timings and collectives independently for every rank; step
phase timing remains rank-0 aggregated because the recipe intentionally records
that opt-in timing on the metric rank only. When the Qwen3 padded-BMM expert path is active, the
`padded_bmm` record additionally reports `max_count`, `dense_compute_tokens`,
and `dense_to_routed_ratio`; this is the relevant compute-waste measure for
severely skewed expert loads and must not be conflated with alignment padding.
The path-specific expert timing contract is `sequential_expert_compute` plus
the `sequential_expert_gate`, `sequential_expert_up`, and
`sequential_expert_down` projection timers for sequential execution,
`padded_bmm` for padded-BMM execution, and the three `grouped_gemm_*`
projection timers for grouped-MM execution. These timings are opt-in
attribution data and remain outside throughput promotion windows.
The canonical AllToAll path uses CPU-vectorized routing metadata construction
by default (`TORCHTUNE_EP_CPU_VECTOR_ROUTING_METADATA=1`), avoiding Python list
expansion for every layer while retaining the established CPU collective and
single accelerator materialization step. Device-side metadata remains a
separate opt-in A/B (`TORCHTUNE_EP_DEVICE_ROUTING_METADATA=1`).
The canonical CPU-vectorized path also packs the send permutation, receive
permutation, and local expert counts into one host-to-device transfer by
default (`TORCHTUNE_EP_PACK_ROUTING_METADATA_TRANSFER=1`); this is separately
controlled and labeled for matched A/B artifacts.
The existing CPU-only router diagnostic
(`experiments/qwen3_moe/bench_qwen3_router.py`) reported, on a representative
256-token, hidden-dim-64, 128-expert, top-8 shape, `0.002627 s` for stable full
sorting versus `0.001368 s` for the tie-safe top-k path (`1.92x` selector
speedup), with equal route counts and zero maximum route-score error. This is
hypothesis-strengthening diagnostic evidence only: it does not measure XPU
kernel time, end-to-end MFU, communication, or dense parity. The required
Aurora A/B must retain `TORCHTUNE_MOE_TOPK_ROUTING` and
`TORCHTUNE_MOE_UNSTABLE_EXPERT_GROUPING` as separate controls and compare the
same canonical model, topology, batch, and measurement window.
Collective records retain the legacy `scope` field and add an explicit
`locality` field (`node_local`, `cross_node`, or `unknown`); canonical EP8 and
EP16 launchers set the first two values respectively. Canonical AllToAll
artifacts must include forward and backward dispatch/combine events; otherwise
the communication table would omit the backward routing cost.

Once rank artifacts are sealed, aggregate them with
`torchtune.modules.moe.aggregate_measurement_files` and pass the result to
`torchtune.modules.moe.summarize_ep_scaling_artifact`. The summary is the
canonical table input: it reports global routed assignments, local-rank token
counts, grouped-GEMM compute/padding ratios, collective duration by name and
locality, phase fractions after the declared warmup window, and peak memory.
It deliberately does not derive throughput, MFU, or device health from an
incomplete artifact. Routed counts are assignments, so top-k routing counts a
selected token once per selected expert rather than once per input token.
When step timing is enabled, the recipe records local/global token counts and
per-GPU/aggregate token rates directly in each completed step record; the
summary returns those rates after the same warmup discard used for phase
attribution, selecting exactly the declared `measurement_steps` after warmup.
Sealing rejects artifacts with fewer post-warmup records or inconsistent window
metadata across ranks, so throughput tables do not depend on parsing metric
logs or silently include extra steps.

For a completed EP8/EP16 pair, pass the two summaries to
`torchtune.modules.moe.compare_ep_scaling_summaries`. It emits one row per EP
degree with per-GPU and aggregate throughput, expected aggregate throughput,
topology-aware scaling efficiency, total communication duration, dispatch and
combine AllToAll duration, routing-metadata AllGather duration, routing metadata
materialization and permutation timings, communication duration by locality,
expert compute token totals, peak memory, and completion/device-health markers.
The
expected aggregate rate is the EP8 aggregate rate multiplied by the
`world_size` ratio; efficiency is measured aggregate rate divided by that
expectation. If either summary lacks a positive `world_size` or aggregate rate,
efficiency is `null` rather than inferred. This table is diagnostic only and
does not promote EP16 or alter the strict Tier 1 dense-parity threshold.
The comparison rejects EP8/EP16 pairs that mix model, sequence, optimizer,
expert execution, routing, transport, or recorded optimization overrides;
only EP degree, world size, topology, and expected locality may differ.
The canonical report also requires complete control metadata, so sparse or
hand-assembled summaries cannot bypass this validation.

Same-topology optimization A/Bs use
`experiments/qwen3_moe/report_optimization_ab.py`. The tool requires matching
model, checkpoint, topology, sequence, batch, optimizer, and execution-path
metadata; callers must explicitly name the varying control (for example,
`TORCHTUNE_MOE_TOPK_ROUTING`). It reports throughput, step-time, and memory
deltas but remains pending independent hardware repeat until promoted.

The canonical report command is:

```bash
python experiments/qwen3_moe/report_ep_scaling.py \
  --ep8 experiments/qwen3_moe/measurements/ep8_<jobid>.rank*.json \
  --ep16 experiments/qwen3_moe/measurements/ep16_<jobid>.rank*.json \
  --json-output experiments/qwen3_moe/reports/ep_scaling_<jobid>.json \
  --markdown-output experiments/qwen3_moe/reports/ep_scaling_<jobid>.md
```

Run it only after every rank artifact is sealed; the canonical report requires
the complete integer rank set `0..world_size-1` for each EP degree and is
fail-closed on missing gates, inconsistent provenance, or incomplete rank
sets.

## Gap attribution

| Gap category | Current classification | Evidence | What closes the gap |
| --- | --- | --- | --- |
| Communication | Pending measurement | EP16 is rejected in the tested regime, but dispatch/combine fractions are not yet isolated in a canonical AG/RS run | EP8-versus-EP16 collective timing and scaling-efficiency table |
| Expert compute | Pending measurement | Grouped-GEMM instrumentation is available; no promotion-window attribution yet | Per-projection GEMM duration, shapes, and achieved expert compute |
| Attention | Instrumented, hardware measurement pending | Qwen3 and Gemma4 MoE transformer layers now record opt-in `attention` phase timing | Fresh-node representative-layer or phase timing in the sealed artifact |
| Optimizer | Inferred from experimental evidence | July 28 diagnostic batch-2 runs reported approximately 103.79 s optimizer time, but used AllToAll and are not parity evidence | Matched-optimizer canonical run with optimizer-step timing |
| Pipeline bubble | Inferred/rejected for current workload | PP2 is rejected for the tested Qwen3 workload; no capacity-track PP result exists | Stage-aware PP timing with pipeline bubble and activation-transfer metrics |

`measured`, `inferred`, and `pending` classifications are preserved in the
manifest `gap_attribution` fields so incomplete attribution cannot be read as
a hardware conclusion.

The tie-safe top-k router remains an opt-in A/B candidate. It selects from the
same activation-dtype probability tensor as the reference path, preserving
ownership for BF16/FP16 near-ties while replacing the full stable sort with a
tie-repaired top-k selection. It does not yet remove the full softmax
allocation. CPU route and gradient equivalence tests pass; no hardware
throughput benefit is claimed until a matched fresh-node A/B run.
The CPU-only diagnostic `experiments/qwen3_moe/bench_qwen3_router.py` reports
reference-versus-candidate selector time, route-count equivalence, and BF16
score error; route-count mismatches are reported rather than hidden. Its output
is explicitly marked diagnostic and is not a promotion artifact.

The opt-in step-timing record now includes both `attention` and `non_expert`
phases for Qwen3-30B-A3B and Gemma4-26B-A4B transformer layers. Activation
checkpointing may execute these regions more than once during a training step;
the timing summary therefore reports invocation counts instead of assuming one
sample per layer. These timings remain excluded from promotion windows.

## Evaluation tracks

### `kernel_parity`

MoE is Qwen3-30B-A3B (about 3.35B active parameters/token), dense is
Qwen3-4B, and sequence length, tiles, optimizer, measurement window, and
fresh-node requirements must match. Promote only after independent repeats
at or above 95% of both matched dense throughput and MFU.

### `ep_scaling`

Compare EP8 and EP16 on Qwen3-30B-A3B across the safe local-batch ladder and,
when memory permits, a true larger local batch. Record routed-token ownership,
router, expert forward, expert GEMM shapes/time, canonical dispatch/combine
AllToAll collectives, optional AG/RS control collectives, memory, and device
health. Memory snapshots include forward, backward, optimizer, and the
post-optimizer `steady_state` phase so allocator headroom can be compared
without conflating optimizer allocation with retained steady-state memory.
Optional step timing also decomposes model forward, backward, manual
gradient release, optimizer, and total-step wall time. Do not use gradient
accumulation as a token-volume proxy.

Use `torchtune.modules.moe.measurement.summarize_step_timings` for the
canonical warmup-discarded phase means and phase fractions when comparing EP8
and EP16 artifacts.

Step timing is captured once per completed optimizer step independently of the
metric logger cadence, so increasing `log_every_n_steps` cannot remove timing
samples from the measurement artifact.

Opt-in MoE measurement now separates Qwen grouped expert compute into gate,
up, and down projection timings (`grouped_gemm_gate`, `grouped_gemm_up`, and
`grouped_gemm_down`). The timing boundaries are disabled for promotion runs and
are intended to distinguish expert GEMM cost from routing and collective cost.

The canonical opt-in measurement launchers are now available for both legs:
`run_native_ep8_measurement.pbs` measures node-local EP8 and
`run_native_ep16_measurement.pbs` measures two-node EP16. Neither launcher is
a promotion run; both require fresh-node collective health, the two-step
optimizer/memory smoke, and semantic completion before their artifacts are
interpreted. Rank artifacts are sealed only after the measured launcher exits
successfully; the aggregator rejects unsealed or failed measurement windows.
Canonical measurement launchers additionally require at least one completed
step containing both `attention` and `non_expert` timing phases. A run with
only generic forward/backward timing therefore cannot be mistaken for a
complete attribution result; generic artifact export remains unaffected.
They also require router, expert-forward, final-scatter, path-appropriate
expert-compute attribution, and the canonical dispatch/combine AllToAll
collective records, including backward dispatch and combine AllToAll events.
Grouped-MM artifacts additionally carry gate/up/down projection timing.
This keeps the sealed EP8/EP16 artifacts sufficient for the token/expert/
communication scaling table rather than merely proving that the training loop
completed.
The EP8 and EP16 measurement launchers default to the same compile-disabled
policy; compile A/Bs must set the same explicit override on both legs.
When measurement is enabled, AllToAll artifacts additionally record opt-in
dispatch/combine pack and unpack timings, including backward pack/unpack
phases. These buckets are diagnostic attribution only and are not included in
promotion windows or aggregate expert-compute totals.
For the seq4096 Qwen workload, both canonical launchers force
`checkpoint_experts=true`, matching the validated memory-stability run. This
is a recompute-for-memory gate, not a throughput claim: disabling it would
reopen the known XPU reserved-memory ratchet and invalidate the two-step smoke
and steady-state comparison. The EP scaling manifest records this requirement
so an artifact cannot silently mix the unstable checkpoint policy with the
validated reference.
An opt-in `TORCHTUNE_EP_FUSED_ALLTOALL_ROUTING=1` A/B now fuses dispatch and
combine packing, AllToAll, and permutation work under custom autograd
boundaries. It is CPU-proven against the unfused AllToAll path for forward
values and input/expert gradients, but remains disabled by default and is not
a promotion result until a fresh-node liveness, memory, and throughput repeat
passes. Its control value is sealed in `environment_overrides` and included in
optimization-profile provenance. Fused collective timing uses the same
dispatch/combine metric names as the unfused path, with one event per
collective direction.
Both canonical launchers now measure the validated grouped-MM + AllToAll
transport (`TORCHTUNE_EP_ALL2ALL=1`) and record an explicit optimization
profile and routing-index mode in every artifact. AG/RS remains an explicit
control rather than the implicit parity baseline.
The canonical profile includes compact routing indices, index-select routed
packing, single-row/in-place AG/RS autograd anchoring, in-place route-score
weighting, in-place final scatter, and in-place grouped SwiGLU multiplication.
Aggregation fails closed on missing/empty optimization attribution or invalid
`TORCHTUNE_EP_ALL2ALL` metadata.
Canonical measurement launchers preserve caller-supplied optimization overrides
with `${VAR:-default}` assignments, so A/B controls are effective rather than
silently overwritten during launcher setup.
This includes transport, collective-buffer, metadata-transfer, grad-release,
expert execution, recomputation, and anchor controls.
The standalone EP8 launcher now derives `source_revision` and clean/dirty
worktree state when callers do not provide them, and supplies explicit default
routing, pipeline, optimization-profile, and measurement-window metadata.
Device health, gate status, and semantic completion remain caller-owned and are
not fabricated by the launcher.
Before entering a collective preflight, both launchers validate that batch and
microbatch values are positive integers, the batch is divisible by the
microbatch, and the warmup plus measurement window fits within the declared
step count. EP8 explicitly exports the same step count used by its measurement
run, preventing the sealed artifact window from drifting from the executed
training length.

The executable pending-result template is
`experiments/qwen3_moe/ep_scaling_manifest.yaml`; it defines the EP8/EP16
batch sweep, AG/RS transport, measurement window, and required hardware gates.
Before submitting an allocation, validate the complete three-track graph with
`torchtune.modules.moe.validate_evaluation_manifest`; this checks the
standalone manifests, launchers, and report paths in
`experiments/qwen3_moe/evaluation_tracks_manifest.yaml`.

New measurement artifacts also record `router_semantics` so router A/B results
cannot be combined across incompatible routing implementations. Historical
artifacts without this optional field remain readable, but new canonical
launcher output uses `probability_topk_v2`.

### `capacity_value`

Select a larger native MoE for which comparable dense active computation does
not fit on one Aurora node. Report capacity/value separately unless active
computation and topology are genuinely matched. Keep checkpoint writes off
until native EP and pipeline reconstruction are production-safe.

The current Gemma4 candidate now has an executable, but unrun, two-node EP16
measurement path: `experiments/qwen3_moe/capacity_value_gemma4_manifest.yaml`,
`recipes/configs/dev/production/gemma4_26b_a4b_sft_ep16_seq4096_synthetic_xpu.yaml`,
and `experiments/qwen3_moe/run_capacity_gemma4_ep16_measurement.pbs`. The local
Gemma4-26B-A4B checkpoint is present with the native two-shard HF layout. The
manifest records Gemma4-31B as a separate dense capacity control and labels
the comparison `capacity_value_only`; this is not strict active-compute parity.
No capacity result is established until the fresh-node collective preflight,
two-step smoke, stable memory/loss, eight-step measurement window, and sealed
artifact gates pass on Aurora.

Capacity/value controls are compared with
`torchtune.modules.moe.compare_capacity_value_results` or
`experiments/qwen3_moe/report_capacity_value.py`. The input JSON must explicitly
provide model size, active parameters/token, topology, throughput, MFU,
active-FLOP efficiency, memory, communication, expert-compute, and stability metrics for
both controls. EP artifact/report inputs derive communication and expert-compute
fractions from sealed phase timings when omitted; explicit overrides remain
supported. Dense controls still require explicit fractions because they have no
MoE expert phase. Both controls must also use the same positive sequence length,
node count, and world size; all numeric metrics must be finite and nonnegative,
with communication and expert-compute fractions bounded to `[0,1]`. The
generated result is always labeled `capacity_value_only`
with `parity_claim_allowed: false`, even when active computation happens to
match.

The dense control can be converted from its rank-zero metric log with
`--dense-log`; the command requires explicit `--dense-nodes`,
`--dense-world-size`, `--dense-sequence-length`,
`--dense-active-flop-efficiency`, `--dense-mfu-percent`,
`--dense-communication-fraction`, and `--dense-expert-compute-fraction` values.
Those values are not inferred from a dense log, because dense execution has no
MoE expert-compute phase and a capacity comparison must expose, rather than
hide, any modeling assumption.

For the canonical Gemma4 run, the MoE side can be converted directly from a
sealed EP scaling report row with `report_capacity_value.py`'s
`--moe-ep-scaling-report` mode. The adapter requires explicit active-FLOP,
MFU, communication, and expert-compute inputs; it derives throughput, topology,
model identity, and completion state only from the sealed EP row.
It also accepts repeated `--moe-ep-artifact` arguments to aggregate the
canonical sealed Gemma EP16 rank artifacts directly, with complete-rank and
completion-gate validation before comparison.

The canonical input shape is:

```bash
python experiments/qwen3_moe/report_capacity_value.py \
  --moe-ep-artifact <each-gemma4-ep16-rank-json> \
  --moe-ep-degree 16 --moe-nodes 2 \
  --moe-total-parameters 25.2 \
  --moe-active-parameters-per-token 3.8 \
  --moe-active-flop-efficiency <explicit-value> \
  --moe-mfu-percent <explicit-value> \
  --dense-artifact <gemma4-31b-dense-json> \
  --dense-active-flop-efficiency <explicit-value> \
  --dense-mfu-percent <explicit-value> \
  --dense-communication-fraction <explicit-value> \
  --dense-expert-compute-fraction <explicit-value> \
  --output <capacity-report-json>
```

The remaining placeholders must be replaced with measured or explicitly
documented values; EP communication and expert-compute fractions are derived
from sealed phase timings when available.

The canonical dense launcher also writes a provenance artifact. Prefer
`--dense-artifact` when available; it preserves the retained measurement records,
topology, checkpoint identity, source revision, and uncommitted-change state.
The artifact mode still requires the explicit dense MFU, active-FLOP, communication,
and expert-compute arguments because those values are not inferred from throughput.

Pipeline attribution is similarly explicit: the measurement summary exposes
`pipeline_bubble` and activation-transfer phase means only when the recipe
records those names. Missing phases remain `pending`; pipeline degree or stage
count alone is not used to estimate a bubble.

The EP gradient-release path also has an opt-in streaming candidate controlled
by `TORCHTUNE_EP_GRAD_RELEASE_STREAMING=1`. It reduces, shards, accumulates,
and frees one FSDP parameter before collecting the next, avoiding the flattened
bucket and multi-parameter transient that caused the seq4096 gradient-release
ceiling. CPU/gloo equivalence passes for fresh and accumulated gradients and
both legacy/nonlegacy collective modes. The candidate is enabled only in the
new Gemma4 capacity launcher; it is unbenchmarked on XPU and is not enabled in
the validated Qwen3 promotion launchers.

## Current conclusions

- EP8 batch 1 is the validated MoE baseline at 729.53 tok/s/GPU and 3.52% MFU.
- Qwen3-30B-A3B EP16 and PP2 remain rejected for the tested regime, not for
  all larger models or token volumes.
- EP8 batch 2 is approximately 803.13 tok/s/GPU but does not meet strict dense
  parity.
- The AG/RS combine path now has an opt-in/A-B-controlled in-place autograd
  anchor (`TORCHTUNE_EP_INPLACE_AG_ANCHOR=1`) that avoids a second full
  combine-buffer allocation; it remains unbenchmarked on XPU.
- The same anchor can use a single gathered row (`TORCHTUNE_EP_SINGLE_ROW_AG_ANCHOR=1`)
  and scalar element instead of reducing the full gathered-token buffer to
  zero or broadcasting a hidden-width row; this also remains unbenchmarked on
  XPU.
- The AG/RS anchor now has an opt-in-controlled zero-cost custom-autograd
  identity (`TORCHTUNE_EP_ZERO_COST_AG_ANCHOR=1`) that retains the AllGather
  backward edge without a forward add or zero-anchor allocation. The scalar
  anchor remains available with `TORCHTUNE_EP_ZERO_COST_AG_ANCHOR=0`; neither
  path has a validated XPU throughput result yet.
- Collective receive buffers now default to uninitialized allocation
  (`TORCHTUNE_EP_UNINITIALIZED_COLLECTIVE_BUFFERS=1`) because AG/RS and routing
  metadata operations fully overwrite them; set it to `0` for the zero-filled
  A/B path. This remains unbenchmarked on XPU.
- The production AG/RS gloo metadata path now gathers counts into the CPU
  receive buffer and transfers them once for device-side vectorized metadata,
  removing a redundant device-buffer copy; this remains unbenchmarked on XPU.
- CPU-bounce AG/RS outputs now copy directly into their preallocated device
  destinations (`TORCHTUNE_EP_DIRECT_CPU_COPY=1`), avoiding temporary device
  tensors from `.to(input.device)`; the legacy path remains available for A/B.
- The current evidence does not establish strict active-compute parity,
  useful cross-node MoE scaling, or a capacity advantage. Those claims require
  the corresponding track's hardware gates and semantic completion markers.
- The factored-optimizer plus `checkpoint_experts=false` A/B is rejected:
  job `8710874` took 561.98 seconds for step 1 and reached 63.447 GiB
  reserved before being terminated before the second forward. The stable
  `up_only` checkpoint policy remains authoritative at 803.129 tok/s/GPU.
- The AG/RS combine accumulation now uses row-wise `index_add_` instead of
  expanding token indices across the hidden dimension; CPU EP equivalence passes,
  but this remains unbenchmarked on fresh Aurora nodes. The matched legacy
  `scatter_add_` path is available with `TORCHTUNE_EP_INDEX_ADD_COMBINE=0`;
  canonical artifact profiles record the setting as `index_add_combine_on/off`.
- The native AllToAll boundary now reuses already-contiguous send and backward
  buffers under `TORCHTUNE_MOE_ALLTOALL_CONDITIONAL_CONTIGUOUS=1`, avoiding
  redundant contiguous copies while preserving the legacy copy path at `0`.
  CPU forward/backward equivalence and strict EP control-signature tests pass;
  no fresh-node matched A/B exists, so this remains unpromoted.
- Strict EP8-versus-EP16 report comparison now treats the complete canonical
  optimization surface as immutable, including metadata transport, collective
  buffer allocation, routing/index packing, EP anchors, recomputation,
  gradient-release, native FSDP reduction, and optimizer controls. This closes
  a provenance gap where recorded but unlisted overrides could otherwise be
  mixed across scaling legs.
- Final routed-output accumulation now has a matched row-wise `index_add_` A/B
  candidate (`TORCHTUNE_MOE_INDEX_ADD_FINAL_SCATTER=1`) for MoE layers without
  a shared expert; `0` retains the legacy widened-index `scatter_add_` path.
  CPU value/gradient and shared-expert fallback tests pass, but fresh Aurora
  throughput and memory evidence is still required.
- Qwen padded-BMM expert packing now has an opt-in vectorized A/B candidate
  (`TORCHTUNE_MOE_VECTOR_PACKING=1`) that replaces the per-expert Python slice
  copies with one indexed pack and one indexed unpack while preserving the
  expert-major layout. It is mutually exclusive with grouped-MM; the canonical
  launchers fail fast instead of recording a misleading no-op A/B. CPU
  uneven/zero-count ordering checks pass; no XPU throughput or memory evidence
  exists yet, so it is not a promotion setting.
- The next unclosed implementation target is fused routed-token packing,
  AllToAll, and expert-major unpack/combine. It requires CPU equivalence and
  a fresh-node smoke before becoming a promotion candidate.
- The current incremental change vectorizes AG/RS local-expert metadata
  construction, removing per-expert scalar reads and Python tensor concatenation
  without changing collective transport or scatter-combine ordering. It is not
  yet a throughput claim until a fresh-node A/B validates it. CPU reference
  checks cover random, all-zero, and uneven expert-count matrices.
- The vectorized AG/RS metadata builder now reuses canonical integer routing
  counts and performs prefix subtraction in its cumulative buffers, avoiding
  two metadata temporaries. Int32 legacy inputs remain supported without
  mutation. This is covered by CPU reference checks and remains unbenchmarked
  on XPU.
- Rank token offsets in the AG/RS metadata builder are now scaled and reshaped
  in place, avoiding one repeated temporary per routed MoE layer. CPU reference
  equivalence passes; XPU impact remains unbenchmarked.
- The AllToAll metadata builder now constructs destination and expert
  permutations directly from grouped count ranges, avoiding per-token expert-ID
  tensors and stable sorts. Distributed CPU forward/backward equivalence
  coverage passes, but this diagnostic
  AllToAll path remains separate from canonical AG/RS promotion evidence and is
  unbenchmarked on XPU.
- The AG/RS and diagnostic AllToAll metadata builders now expand contiguous
  permutation ranges from group bases and counts without materializing a
  per-token group-ID tensor; the final range offsets are added in place to the
  expansion buffer to avoid a second full-length result allocation. Direct
  zero/uneven-group and reference equivalence tests pass; this remains
  unbenchmarked on XPU and is not a promotion claim.
- The vectorized AllToAll metadata path now constructs interleaved send and
  owned-expert index vectors directly from `arange` views instead of building
  them through repeat-interleave/multiply/add chains. Reference and distributed
  forward/backward equivalence remain green; this is unbenchmarked on XPU and
  remains outside promotion claims. A July 29 CPU microbenchmark of the
  topology-only helpers measured approximately `2.8–3.0x` lower construction
  time across EP8/EP16 and 128/256-expert shapes; this is diagnostic only and
  does not imply an end-to-end throughput or MFU gain.
- Each `ExpertParallel` instance now caches the immutable interleaved send and
  owned-expert index vectors by device and EP topology. Cached and uncached
  metadata remain equivalent for random, zero, and uneven counts, with rank
  separation covered on CPU. The cache is unbenchmarked on XPU and remains
  outside promotion claims pending a fresh-node matched metadata A/B.
- The Qwen3 router has an opt-in `TORCHTUNE_MOE_TOPK_ROUTING=1` candidate:
  sorted `topk` over the activation-dtype probability tensor handles rows whose
  kth and (k+1)th scores are distinct, while boundary ties and `top_k ==
  num_experts` use the existing stable full sort. CPU random, BF16 near-tie,
  exact-tie, all-expert, input/gate-gradient, and route-pairing equivalence
  tests pass. The candidate retains the full softmax allocation and only targets
  selector work; it remains disabled by canonical launchers and unbenchmarked on
  XPU. The tie check may synchronize and must demonstrate a net gain in a
  fresh-node A/B before promotion.
- Expert-grouping order is separately controlled by
  `TORCHTUNE_MOE_UNSTABLE_EXPERT_GROUPING=1`, allowing the grouping-sort cost
  to be measured independently from top-k selection. CPU route-pairing and
  gradient coverage pass; canonical launchers keep it disabled and no XPU
  result is claimed.
- The Qwen and generic token-choice routers now reuse one flattened
  selected-expert view for token counting and grouping sort. This removes
  redundant view construction without changing routing values, stable
  tie-breaking, or route ordering; it remains unbenchmarked on XPU and is not a
  promotion claim.
- A second unbenchmarked optimization keeps routed-token ownership indices
  one-dimensional through dispatch and grouped-MM padding, avoiding hidden
  hidden-width metadata copies. Its final scatter uses the same broadcast index
  view as before, and weighted duplicate-route CPU coverage passes. Set
  `TORCHTUNE_MOE_WIDE_ROUTING_INDICES=1` for a matched pre-optimization A/B;
  neither mode is promoted until a fresh-node measurement validates it.
- A third unbenchmarked optimization uses in-place final scatter-add into the
  existing zero output buffer when no shared expert is present. Shared-expert
  paths retain the out-of-place operation; CPU forward and gradient coverage
  passes.
- A separate final-scatter candidate uses row-wise `index_add_` for compact
  indices and no shared expert, avoiding the hidden-width index expansion. Set
  `TORCHTUNE_MOE_INDEX_ADD_FINAL_SCATTER=0` for the matched legacy A/B; shared
  expert and wide-index paths retain their safe fallback.
- Measurement artifacts now report opt-in `final_scatter` timing alongside router,
  expert, and collective phases, so the A/B can be attributed without changing
  promotion windows.
- A fourth unbenchmarked optimization packs grouped-MoE routed tokens with
  `index_select`/`gather` and masks only alignment padding, avoiding the full
  routed-token and routing-index sentinel concatenation. Set
  `TORCHTUNE_MOE_INDEX_SELECT_PACKING=0` for the matched legacy A/B; compact
  and wide-index CPU equivalence and gradient coverage pass.
- Padded-BMM vector packing now adds token positions in place to its repeated
  expert-offset buffer, avoiding a second full routed-token row-index
  allocation and reuses its expert-count prefix/base buffers in place.
  Existing CPU zero/uneven-count and forward/backward equivalence tests remain
  the correctness evidence; XPU impact is unbenchmarked and no promotion claim
  is made.
- The generic sequential expert fallback now transfers rounded token counts
  once before its expert loop instead of synchronizing once per expert. The
  Qwen3-specific sequential path already had this behavior; both remain
  unbenchmarked on XPU.
- Route-score weighting now scales the fresh gathered-token buffer in place,
  preserving router and input gradients while avoiding a second routed-token
  activation allocation. This remains unbenchmarked on XPU.
- The grouped Qwen3 SwiGLU helpers now reuse the activated gate buffer for the
  elementwise gate/up product, reducing one intermediate hidden-state allocation
  in both grouped-MM and the validated sequential expert path while preserving
  forward and backward results. This remains unbenchmarked on XPU.
- Integer routed-token counts now bypass redundant rounding and recasting in
  the active Qwen3 and generic grouped expert paths; floating legacy inputs
  retain normalization. This remains unbenchmarked on XPU.
- Grouped-token sum/item shape hints now run only while compiling; eager
  promotion forwards no longer perform that device-host scalar synchronization.
  This remains unbenchmarked on XPU.
- Grouped-GEMM measurement now infers hidden width from the expert layout, so
  Qwen3 HF-native `[expert, hidden, model]` weights no longer report the model
  width as hidden width. This corrects instrumentation only and remains
  unbenchmarked for throughput impact.
- The factored optimizer now reuses its FP32 gradient buffer for the normalized
  update rather than allocating a separate normalized-gradient tensor, while
  preserving the original normalized-denominator epsilon-clamp order. The
  one- and three-step reference, state-shape, and tiny-gradient tests pass; the
  change remains unbenchmarked and must be measured with the canonical optimizer
  control before promotion.
- July 28 experimental batch-2 runs show optimizer time can exceed 100 seconds
  per step, but those runs use `TORCHTUNE_EP_ALL2ALL=1` and are not valid AG/RS
  parity evidence. Metadata-only variants still hit XPU `SIGSEGV`, so no
  throughput or MFU claim is attached to them.
- Measurement aggregation now rejects an artifact when its resolved optimizer
  metadata disagrees with `TORCHTUNE_MOE_OPTIMIZER_COMPONENT`, preventing an
  AdamW/factored-optimizer A/B from being silently combined or mislabeled.

The canonical control metadata lives in
`experiments/qwen3_moe/evaluation_tracks_manifest.yaml`. Optional measurement
is enabled with `TORCHTUNE_MOE_MEASURE=1`; promotion windows remain unaffected
when it is disabled.

Grouped-MM measurement now records separate opt-in gate, up, and down
projection boundaries (`grouped_gemm_gate`, `grouped_gemm_up`, and
`grouped_gemm_down`) in addition to aggregate expert-forward timing. These
boundaries are implemented in both the generic grouped expert path and the
production Qwen3 HF-layout path, including the `up_only` recomputation branch.
They are disabled with the rest of the measurement collector and have CPU-safe
mocked-kernel coverage; no XPU performance claim is attached.

The production Qwen3 grouped-MM path retains its existing device-side count
handling and does not add new host synchronization for timing. The sequential
fallback still materializes counts for its per-expert loop by design; it is the
validated Qwen seq4096 promotion path, while grouped-MM remains explicit-only.

Opt-in measurement now also records sequential expert gate, up, and down
projection durations inside the aggregate `sequential_expert_compute` bucket.
This improves expert-compute attribution for the EP scaling table without
changing disabled-path execution or promotion windows.

Canonical AG/RS custom autograd boundaries likewise avoid constructing timing
contexts when measurement is disabled. Forward and backward collective
semantics are unchanged; CPU-safe coverage exercises both custom autograd
directions, while no XPU performance claim is attached until a fresh-node A/B.

The collector now returns one shared immutable no-op context for disabled
phase and collective timing. This removes per-call Python context allocation
and disabled locality validation across router, expert, dispatch, combine,
and scatter instrumentation without changing enabled artifact contents.

`MoE.forward` also branches around all phase timing boundaries when measurement
is disabled, so the canonical promotion path does not pay context entry/exit
overhead for router, EP dispatch/combine, expert execution, or final scatter.
The enabled path retains the same timing names and artifact behavior.

Disabled collectors now lazily allocate their measurement record only when it
is explicitly accessed. This avoids per-layer timing dictionaries and event
lists in the default promotion configuration while preserving the existing
record/export API and enabled measurement artifacts.

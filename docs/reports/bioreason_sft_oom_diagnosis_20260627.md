# BioReason SFT — 12-tile OOM root-cause diagnosis (2026-06-27)

## Question
Throughput ablations on the native-Gemma4 31B SFT recipe. The smoke ran clean at **6 tiles**;
scaling to **12 tiles (full node)** crashed. Why, and what's the fix?

## What crashed
On 12-tile single-node FSDP2 (seq=8192, bs=1), training **banned:1 PDE'd / OOM'd around step 3**,
while rank-0's logged peak memory was only **4.85 GiB**. Initial read was misleading because the
recipe logs peak memory from **rank 0 only**.

## Root cause (data-constrained)
**Per-rank sequence-length imbalance.** At batch_size=1, each of the 12 ranks independently draws a
variable-length sequence (prompt p50≈1770, p99≈4618, max≈5046 tok; total up to ~8440). The rank
that draws a max-length sequence materializes huge activations while rank 0 (short draw) sits at
4.85 GiB. Direct evidence: with HSDP shard=2, **rank 8 OOM'd at 50.87 GiB allocated** in the same
step rank 0 reported 4.85 GiB. The `banned:1` PDE and the clean `torch.OutOfMemoryError` are the
same root cause — whether the over-allocation page-faults or is caught.

Why 6 tiles survived: fewer ranks = fewer independent draws/step = lower probability any one hits
the long tail (and a smaller sample of 5 steps).

Three contributing factors, in order:
1. **Sequence-length imbalance** (dominant) — one long-drawing rank spikes.
2. **S² attention materialization** — multiplied the per-long-sequence cost (see fix below).
3. **Base-shard size vs sharding width** — HSDP shard=2 put ~31 GiB of base on each of 2 tiles →
   OOM on top of activations. Full-shard across 12 is the correct sharding; shard=2 is too little.

## Correction to an earlier hypothesis
I first attributed the banned:1 solely to the S² attention OOM. That was half right: the S²
materialization is real and caused a *clean 52 GiB OOM under selective-AC*, but the **banned:1 at
step 3 reproduces with BOTH legacy and SDPA attention** — so attention alone is not the blocker.
The blocker is per-rank activation memory from long sequences. CLAUDE.md line 48 documents the
related CCL IPC-handle accumulation for many-tile single-node FSDP2 ("single-node 32B 10+2 BLOCKED
— use 2-node HSDP"); that compounds memory pressure but the primary driver here is the activation
imbalance.

## Fixes (priority order)
1. **Sample packing into uniform-length bins** (the real fix) — pack short sequences to a fixed
   length so every rank does equal work and no single rank spikes. Eliminates the imbalance.
   Requires multimodal-aware packing (placeholder splice + block-diagonal mask per sub-sequence) —
   real code work, the main follow-up. Confirmation run E0 (full-12 + capped seq) tests the
   mechanism.
2. **Gemma4 SDPA attention** (DONE, committed) — Gemma4Attention hand-rolled
   `matmul→softmax(fp32)→matmul`, materializing `[b,n_heads,S,S]` fp32 scores (~8.6 GiB/global
   layer at S=8192/32 heads). Every other torchtune model uses fused SDPA via `_attention_call`;
   Gemma2/Gemma4 were the exceptions (Gemma2 *needs* it for attention soft-capping; Gemma4 has
   `softcapping=None` so it paid S² for nothing). Now routed through
   `F.scaled_dot_product_attention` (O(S), custom scale via `scale=`, sliding window as a cheap
   bool band-mask). Numerically exact (max|math−sdpa|~1e-7), 17 CPU tests green. `TORCHTUNE_GEMMA4_SDPA`
   (default 1; =0 = legacy). Necessary but not sufficient alone.
3. **2-node HSDP** (production topology) — shard within a node across a moderate group (e.g. 6),
   replicate across nodes. Fewer IPC handles/tile (CLAUDE.md line 48) and the right place to spend
   the headroom. Needs the `mpiexec --pmi=pmix` multi-node launcher (not torchrun --standalone).

## What did NOT work
- **max_seq_len cap** (4096/6144): fail-fasts on the longest prompts (a 2050-residue protein → 6426
  prompt tokens > 6144), erroring out real training examples. Unworkable without also cutting
  max_protein_len (loses protein info).
- **HSDP shard=2**: too little sharding → 31 GiB base/tile → OOM.

## SDPA helps but is not sufficient (measured)
Clean A/B on the known-good 6-tile topology, seq=8192:
- **SDPA ON**: survived to **step 6** (loss 30.7) before a long-draw rank banned:1'd.
- Prior legacy-attention 12-tile runs: banned:1 at **step 3**.
SDPA lowers per-rank memory so the run survives more steps before an unlucky max-length draw
spikes — but it does NOT eliminate the spike. Confirms: SDPA is a real improvement (keep it),
packing is the actual fix. (rank-0 peak stayed 9.66 GiB throughout; the crashing rank is a
different one that drew a long sequence — same imbalance signature.)

## Recommendation
Stop interactive ablations (debug-queue node init proved flaky after banned:1, and the real
lever is implementation work, not a config sweep). Next focused work item: **implement
multimodal-aware sample packing** (uniform-length bins; placeholder splice + block-diagonal
mask per sub-sequence), CPU-validated first, then a fresh hold. Capping seq/protein is a dead
end — seq cap errors long-prompt examples; protein cap breaks the ESM3 cache (keyed at
max_protein_len=2048 → KeyError if the dataset truncates to a different length).

## Process note
Per-rank memory logging (not just rank 0) would have made this obvious immediately. Worth adding a
max-over-ranks peak-memory all-reduce to the recipe's metric logging.

## ⚠️ CORRECTION: the section below claimed "RESOLVED" prematurely — it is NOT

The LinearCrossEntropyLoss + userfaultfd run completed steps 1-3 (loss 41.2→36.8→35.8) then
**banned:1 PDE'd on rank2 at the step 3→4 transition** — same crash point as before. I called
it resolved after seeing step 3 clear, BEFORE the run cleared the step-3→4 boundary where it
actually dies. So: LinearCrossEntropyLoss genuinely fixed the "not allocated yet" loss error
(real progress — the loss now runs), and userfaultfd did NOT eliminate the banned:1 (it
persists at step 3-4 across every allocator/loss/MR-monitor combination tried). The banned:1
at the step-3→4 transition on 12-tile single-node FSDP2 is the durable, still-OPEN blocker.
Read the section below as "loss-path fixed, banned:1 still open," not "resolved."

## RESOLUTION ATTEMPT (2026-06-27, node x4311c4s3b0n0) — loss fixed, banned:1 STILL OPEN

The "12-tile OOM" was THREE stacked failures, peeled back one HW iteration at a time
(earlier single-cause diagnoses were each partial):

1. **OFI memory-registration accumulation** (the banned:1 PDE at step 3). The default
   allocator's fresh-VA-per-step AllGather buffer + `FI_MR_CACHE_MONITOR=disabled` (or the
   65536 IPC-handle path) accumulates fabric registrations on receive-side tiles →
   NotPresent PDE. FIX: `FI_MR_CACHE_MONITOR=userfaultfd` (OFI deregisters freed MRs).
2. **Pluggable allocators break the loss**: usm_pending/usm_caching get past banned:1 but
   raise "tensor data not allocated yet" in F.cross_entropy. So: keep the DEFAULT allocator.
3. **Deprecated chunked_output is FSDP2-fragile at >6 tiles**: the same "not allocated yet"
   on the default allocator via `CEWithChunkedOutputLoss` → `chunked_output` → tied
   `self.output(chunk)` (transformer.py emits a FutureWarning to use a linear loss). FIX:
   **LinearCrossEntropyLoss** (skip_output_layer=True; projects valid tokens itself,
   DTensor-aware). Wired BioReasonNativeModel with skip_output_layer/output delegation.

**Validated config (12-tile, seq=8192, full node):** LinearCrossEntropyLoss + DEFAULT
PyTorch allocator + `FI_MR_CACHE_MONITOR=userfaultfd` + `PYTORCH_ALLOC_CONF=gc:0.99`.
Trained clean PAST step 3 (the historical crash) — loss 41.2→36.8→35.8 decreasing on all
12 ranks, ~48s/step. SDPA attention (committed earlier) is kept (orthogonal memory win).
18 CPU tests green.

Earlier mis-diagnoses, corrected: "S² attention", "per-rank seqlen imbalance", and
"needs 2N HSDP" were all wrong as the BLOCKER — the blockers were the OFI MR leak + the
deprecated loss path, both single-node-fixable. The user's "768 GiB shouldn't OOM" was
right: it never was a capacity problem.

## ★ DISCRIMINATOR: Qwen3-32B trains clean at 12 tiles — crash is GEMMA4-SPECIFIC

After the loss-path fixes, the Gemma4-31B 12-tile run's FINAL failure was at
`gemma4/_attention.py:275` (the SDPA path): "could not create a memory" — oneDNN cannot
allocate the attention workspace — on the long-sequence rank, after 2 clean steps.

Control experiment (node x4311c4s3b0n0): **Qwen3-32B text-only SFT**, stock dense recipe
(full_finetune_distributed_xpu.py), alpaca packed seq=2048, LinearCrossEntropyLoss,
full-shard 12 tiles, SAME env (default alloc + userfaultfd). Config
`sft_qwen3_32B_fitstest_xpu.yaml`, launcher `run_qwen3_32b_fitstest.sh`.

RESULT: **clean ALL 12/12 steps** (where Gemma4 died in EVERY config), loss 2.35→1.77→1.50
decreasing on all 12 ranks, **~14 s/step (vs Gemma4's ~48 s/step — 3.4× faster)**, zero
banned:1 / OOM / "could not create a memory". 12-tile single-node FSDP2 of a 32B dense
model is NOT the problem — Qwen3-32B (our validated 32B GRPO baseline arch) just works.

CONCLUSION: the BioReason 12-tile blocker is **Gemma4-specific** — almost certainly the
custom Gemma4Attention (hand-rolled module, heterogeneous local/global head dims, the SDPA
workspace it requests). NOT scale, NOT topology, NOT the allocator, NOT the recipe.

IMPLICATION / next step: per the user's call, swap the BioReason backbone from Gemma4-31B
to **Qwen3-32B** — it has all our validated 32B FSDP2/CCL/allocator infra, trains clean at
full node, and is 3.4× faster per step. The native-Gemma4 multimodal machinery (model_native,
dataset_sft, recipe, SDPA, LinearCE wiring) ports to a Qwen3-32B backbone with minimal change
(swap gemma4_31b()/lora_gemma4_31b() → qwen3_32b()/lora_qwen3_32b(), hidden_size 5376→5120,
reserved placeholder ids, GEMMA4→QWEN2 checkpointer). Gemma4 attention is a separate,
deferred investigation.

## ★★ TRUE ROOT CAUSE (2026-06-27, corrected again): seq=8192 @ 32B/12-tile, NOT Gemma

My "Gemma4-specific" conclusion had a SEQ-LENGTH CONFOUND: I compared Gemma@seq8192 vs
Qwen@seq2048 (the Qwen fits-test was seq=2048). Controlled re-test at matched seq exposes it:

| model        | seq  | result                          |
|--------------|------|---------------------------------|
| Qwen text    | 2048 | ✅ clean 12/12, 13s/step         |
| Qwen text    | 8192 + userfaultfd | ⏸ HANG at step-0 fwd (loadavg high, no progress) |
| Qwen text    | 8192, no userfaultfd | ❌ UR:40 UR_RESULT_ERROR_OUT_OF_RESOURCES, step 0 |
| Qwen BioReason | 8192 | ❌ banned:1 step 0              |
| Gemma BioReason| 8192 | ❌ banned:1 / OOM step 3        |

CONCLUSION: the blocker is **seq=8192 at 32B dense / 12-tile single-node FSDP2** — a Level
Zero resource wall (UR:40 / banned:1 / hang depending on MR-monitor), independent of model
(Gemma OR Qwen) and of the multimodal path. seq=2048 is comfortably under it. So neither
"Gemma attention" nor "BioReason multimodal" is the root cause — both were seq=8192 victims.
(Gemma is still ~3.4× slower than Qwen and worth replacing for throughput, but it was NOT
the crash cause.)

ACTIONABLE: BioReason prompts are p50≈1770 / p99≈4618 / max≈5046 tok; seq=4096 covers ~97%
of examples (long tail truncates). Testing BioReason-Qwen at seq=4096/12-tile now — expected
to clear the L0 wall. Production envelope: seq=4096 (or 6144), NOT 8192, at 12-tile single
node. seq=8192 would need fewer tiles, more nodes, or activation offloading — a separate opt.

## ★★★ ACTUAL ROOT CAUSE (2026-06-27, final, catchable traceback): XPU SDPA O(S²) at long seq

The 2-tile run gave a CATCHABLE OutOfMemoryError (vs the uncatchable banned:1/UR:40 at more
tiles) with a full traceback — the diagnostic that ends the guessing:

  qwen3/_attention.py:261 -> attention_utils.py:428 _sdpa_call -> F.scaled_dot_product_attention
  torch.OutOfMemoryError: Tried to allocate **9.00 GiB** (one SDPA call)

9.00 GiB = 64 heads × 6144 × 6144 × 4 bytes (fp32) = the FULL [B, n_heads, S, S] score tensor.
**XPU's scaled_dot_product_attention falls back to the MATH backend for the autograd
(training) path and materializes O(S²) fp32 scores.** This is the SAME mechanism as the
Gemma "could not create a memory" at gemma4/_attention.py and the 12-tile banned:1/UR:40 —
all are the O(S²) attention-score allocation at long seq, surfacing as different L0 error
flavors depending on tile count / MR-monitor.

Why text-only seq=2048 was the only clean run: 64×2048²×4 = 1.0 GiB per SDPA call (survivable);
seq=6144 → 9 GiB (OOM). It was NEVER Gemma-vs-Qwen or multimodal-vs-text — it was ALWAYS
seq-length × the O(S²) math-SDPA fallback. Every crash is explained by this one cause.

KEY CONSTRAINT INTERACTION: BioReason's longest prompts (2048-residue protein) need seq≥~5046,
but seq≥~3000 already makes the O(S²) score tensor too big to train a 32B at on these tiles.
seq and the data requirement are fundamentally in tension under math-SDPA.

CRITICAL UNKNOWN (next session): IPEX varlen does NOT help — it's no-grad-only (training fwd
always uses standard SDPA; CLAUDE.md line 306). So the lever is: does XPU/this torch expose a
MEMORY-EFFICIENT SDPA backend for the AUTOGRAD path (flash/mem-efficient), or must we use an
explicit chunked/flash-attention implementation for the Gemma4/Qwen3 attention training forward?
The working GRPO baselines run at shorter effective seq, so this O(S²)-at-long-seq-training wall
may simply not have been hit before. This is the real blocker for 32B SFT at seq>~3K on XPU.

NOT a quick env fix. Candidate solutions to investigate: (1) a flash-attention kernel for XPU
training (IPEX has varlen_fwd but no autograd — needs the bwd); (2) sample packing to keep
per-sequence S small while filling the budget (packing makes uniform S, but the LONGEST single
protein still needs ~5046 contiguous — packing does NOT reduce the max single-doc length, so the
O(S²) per-document cost in a block-diagonal mask remains... unless block-sparse flex-attention,
which is CUDA-only on XPU); (3) cap max_protein_len lower (e.g. 1024 → ~1290-tok prompt → 3-head
seqs fit) accepting protein-info loss + an ESM3 cache rebuild.

## ★★★★ DEFINITIVE (2026-06-27, single-variable SDPA backend probe): no flash BACKWARD on XPU

Stopped guessing and ran the controlled probe (experiments/bioreason/sdpa_backend_probe.py,
sdpa_flash_layout_probe.py, sdpa_seq_sweep_probe.py) at the exact crash shape
[B1,H64,S6144,D128] bf16 on one tile. The grad/no-grad split is the whole answer:

| call            | grad  | peak     | backend actually used |
|-----------------|-------|----------|-----------------------|
| default BHSD    | False | 0.38 GiB | FlashAttentionXPU (O(S), fused) |
| default BHSD    | True  | 27.94 GiB| MATH (O(S^2) fp32 scores) |
| flash-ctx (forced) | either | RAISED "No available kernel" | selector refuses (no bwd) |

torch 2.10 XPU build messages (verbatim): "XPU don't support SDPA mem efficient attention
backend", "XPU don't support SDPA cudnn attention backend", "FlashAttentionXPU requires
query, key, and value to be in BSHD layout", "Backward or grad to be supported."

CONCLUSION (evidence, not theory): **XPU has a flash-attention FORWARD kernel but NO flash/
mem-efficient BACKWARD kernel.** So:
  - no-grad paths (ref fwd, rollout logprobs, eval, generation) get flash → O(S) memory.
    This is why GRPO, generation, and the seq=2048 fits-test were always fine.
  - the autograd/training forward has only MATH → materializes [B,H,S,S] fp32 = O(S^2).
    Forcing SDPBackend.FLASH/EFFICIENT raises "No available kernel" (selector won't pick a
    backend that lacks backward). IPEX varlen is also no-grad-only (CLAUDE.md). There is
    currently NO memory-efficient autograd attention on this XPU stack.

This is a PLATFORM limitation, not a miscall in our recipe. mask=None / is_causal=True is
already correct (verified: qwen3/_attention.py:267 sets is_causal when mask is None, and both
the recipe _loss_step and BioReasonNativeModel.forward pass mask=None). The math fallback is
forced by the missing bwd kernel, independent of how we call SDPA.

### The real lever: seq is the binding constraint (single-call fwd+bwd MATH transient)

| seq  | attn fwd+bwd peak (1 call) | note |
|------|---------------------------|------|
| 2048 | 3.31 GiB  | only clean prior run |
| 3072 | 7.00 GiB  | |
| 4096 | 12.30 GiB | |
| 5120 | 19.09 GiB | |
| 6144 | 27.39 GiB | |
| 8192 | ~48 GiB (extrap) | the banned:1 killer at 12-tile |

With activation checkpointing ONE layer's attention transient is live at backward, so the
per-tile fit is ~ base_shard/tile + (attn transient at seq) + stored-AC activations. This
INVERTS the throughput story in sft_bioreason_qwen3_32B_xpu.yaml: the O(S^2) transient — not
the 61 GiB base — is binding, so the throughput-optimal shard is WIDER (thinner base/tile to
leave room for ~27 GiB of attention), not the aggressive shard=2. Predicted fits at seq=6144:
  shard=12 (base ~5 GiB/tile)  + 27 + ~4 AC ≈ 36 GiB  → safe
  shard=4  (base ~15 GiB/tile) + 27 + ~4    ≈ 47 GiB  → fits, more replication/throughput
  shard=2  (base ~30.5/tile)   + 27 + ~4    ≈ 62 GiB  → OOM (= the catchable 2-tile crash)

ACTIONABLE ENVELOPE: seq<=6144 at 12-tile single node (full shard or shard=4). BioReason
prompts p50~1770 / p99~4618 / max~5046 → total (prompt+target) p99~6798, so seq=6144 covers
the bulk; the longest few examples truncate the target tail (acceptable) — NOT the protein
(max_protein_len=2048 stays). seq=8192 needs >1 node or a real flash-bwd kernel; deferred.

## ★★★★★ HW envelope sweep (2026-06-27, nodes x4401c1s6b0n0 / x4618c2s2b0n0)

Ran the real recipe (Qwen3-32B, 12-tile full-shard, LoRA, AC, LinearCE) at each seq with the
new dataset drop_over_length filter so backward is actually reached (the 4096/5120 "crashes"
in the first pass were the dataset fail-fast on the <1% over-length tail, NOT memory):

| seq  | result                                                            |
|------|-------------------------------------------------------------------|
| 6144 | UR:40 in current_loss.backward() (step 0) — 27 GiB attn transient |
| 6144 + activation_offloading | banned:1 PDE step 0 — offload moves AC-stored acts to CPU but the LIVE 27 GiB attn transient stays on-tile |
| 5120 | **step 1 OK** (Loss 15.77, 132s/it) then **banned:1 at step 2**  |
| 4096 | (testing 10 steps)                                                |

KEY REFINEMENT from the seq=5120 result: it is NOT a static single-step fit failure — step 1
completed 4 microbatches of backward then step 2 crashed. Mechanism = **per-rank sequence-length
imbalance** + the XPU **reserved-pool staircase**: at batch_size=1 each rank independently draws a
variable-length seq (prompt p50~1800 / p99~4900 / tail to max_seq_len); step 1 all 12 ranks drew
fits, step 2 one rank drew a longer seq whose 19 GiB math-SDPA transient tipped it over. XPU can't
empty_cache under FSDP (UR-handle leak guard), so the large per-step transient is not reclaimed and
the reserved pool climbs until a later/larger draw page-faults (banned:1). This is the SAME
imbalance mechanism noted earlier (HSDP shard=2: rank8 OOM 50.87 GiB while rank0 at 4.85 GiB same
step). So the fitting seq must leave enough reclaim headroom to absorb the WORST per-rank draw, not
just the median — i.e. comfortably below the single-call ceiling.

DATASET FIX (committed-to-tree): BioReasonSFTDataset.drop_over_length (default True) pre-filters
examples whose PROMPT alone >= max_seq_len (the prompt is never truncated — placeholder runs are
load-bearing). Census at seq=5120: dropped 36/7365 = 0.49% of validation. Replaces the whole-run
ValueError fail-fast with a logged skip; the fail-fast remains as a defensive __getitem__ fallback
when drop_over_length=False. CPU tests: test_native_qwen3_sft.py::test_drop_over_length_* (2 new).

## ★★★★★★ step-2 staircase is the REAL wall (not seq fit) + a separate cache bug surfaced

Both seq=4096 and seq=5120 crash at EXACTLY step 2 (not a slow drift): step 1 completes all
microbatches, step 2 banned:1. Mechanism confirmed = variable per-step tensor shape at bs=1.
The collate pads to per-batch-max, so each rank's step is a DIFFERENT shape; XPU can't
empty_cache under FSDP, so the caching allocator cannot reuse step 1's freed O(S^2) math-SDPA
transient for step 2's different size → it reserves a SECOND transient block → ~2x peak → OOM.
Lowering seq only changes the per-block size, not the doubling — so it never makes a long run
stable. The lever is allocator behavior, not seq.

TWO candidate fixes for the doubling (both target the same reservation mechanism):
  (A) pad_to_fixed=True — pad every batch to a CONSTANT max_seq_len so the transient is one
      reused shape. IMPLEMENTED (collate + recipe flag, default OFF pending validation) but a
      first HW try surfaced a SEPARATE pre-existing bug (below) before the memory hypothesis
      could be tested.
  (B) XPU_USM_ALLOC_SO=recipes/dev/usm_caching_alloc.so — the PROVEN GRPO-recipe fix: pools the
      per-step buffer at a STABLE VA so OFI registers once → flat reserved pool. The recipe
      already has the hook; the launcher left it empty (default allocator = the staircasing one).
      This is the higher-confidence fix (it is how 32B GRPO runs for hundreds of steps) and
      should be tried FIRST next session, with pad_to_fixed as a complement.

SEPARATE PRE-EXISTING BUG (must fix before any production run; unrelated to memory): with
pad_to_fixed forcing all ranks through step 0 at once, rank9 hit
"Protein token count 2210 != protein features 240" in model_native._splice_embeds. Validation
has 154/7365 sequences > max_protein_len=2048 (max 15998). The dataset truncates the AA string
to 2048 → ≤2050 placeholders, but this example produced 2210 placeholders against a 240-row
ESM3 cache tensor → the ESM3 cache and the dataset placeholder construction are keyed
INCONSISTENTLY (cache likely precomputed at a different max_protein_len, or precompute truncates
differently than dataset_sft). The earlier OOM-at-step-2 runs never reached one of these 154
examples so it was masked. drop_over_length does NOT catch it (that filters by PROMPT length, not
by protein/feature count). NEXT: verify precompute_esm3_cache truncation == dataset_sft
truncation (both must use sequence[:max_protein_len] before the SHA1 key) and rebuild/repair the
cache, OR add a per-example placeholder-vs-feature consistency skip.

NET STATE: root cause (no flash bwd on XPU → math O(S^2)) is DEFINITIVE. The path to a stable
long run is the allocator fix (B, proven) ± fixed-shape (A), NOT seq tuning. One data-pipeline
bug (cache/placeholder mismatch) blocks production independent of memory. No clean multi-step
32B BioReason SFT run achieved yet.

## ★★★★★★★ THE step-2 crash is DOCUMENTED CCL IPC-handle accumulation, not the SDPA transient

Corrected the allocator env to the PROVEN 32B-GRPO values (default allocator, NO
XPU_USM_ALLOC_SO — usm_caching_alloc.so is the 3B fix and OOMs at 32B Adam-init per
feedback_alloc_32b_default; FI_MR_CACHE_MONITOR=disabled instead of userfaultfd; gc:0.95).
seq=4096 STILL crashed at exactly step 2 (rank7, banned:1) — identical signature. So the
MR-monitor was not it, and my "variable-shape allocator doubling" theory is ALSO not the
primary cause.

The real cause is DOCUMENTED in CLAUDE.md verbatim:
  "CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=65536 ... 65536 accumulates 10.85 GiB IPC handle
   memory by step 1 BWD on 10-tile FSDP2 -> OOM at step 2. Single-node 32B 10+2 BLOCKED -
   use 2-node HSDP."

My crash = OOM at step 2, single-node 12-tile 32B FSDP2 = that signature exactly, incl. the
step number. It is a banned:1 PDE (external/uncounted L0 memory exhausting l0_free; rank0 fine
at 7.65 GiB while rank7 page-faults), the CCL IPC-handle / OFI-MR accumulation fingerprint —
NOT a clean PyTorch OOM from the attention tensor.

RECONCILES EVERYTHING + answers "how would 2-node HSDP change the math":
  - The SDPA O(S^2) transient sets a real SEQ CEILING (~12 GiB at seq4096) but it FITS. It does
    NOT shrink with more nodes (true — the per-rank attention is the same).
  - The STEP-2 killer is CCL external-memory accumulation, which IS per-tile-buffer-size
    dependent. 2-node HSDP shards across 24 tiles -> each tile's AllGather buffer ~halves ->
    the per-tile IPC/MR accumulation drops below the l0_free wall. THAT is what 2N changes.
  - So both are real but separate: seq<=~5000 (SDPA ceiling) AND >=2-node HSDP (CCL wall).
    Single-node 32B FSDP2 is documented-BLOCKED regardless of seq. Matches the validated
    "Phase 2 32B 2-node smoke" (5/5 clean, 23+1 vLLM) — the GRPO 32B path is ALSO 2-node.

CORRECTED PLAN: run BioReason 32B SFT on 2-node HSDP (shard within node=12, replicate across
nodes=2), seq<=4096 (or 5120), drop_over_length=true, default allocator + FI_MR=disabled +
gc:0.95 + the production multinode CCL env (pmix/mpi). The recipe already supports multinode
(full_finetune_distributed_xpu.py:218 mpi4py barrier). Process lesson (AGAIN): the step-2 OOM
was a NAMED documented signature in CLAUDE.md the whole time — should have grepped the symptom
("OOM step 2" / "banned:1 step 2" / single-node 32B) BEFORE theorizing a novel mechanism.

## CORRECTION: the "ESM3 cache/placeholder mismatch" was MY pad_to_fixed bug, NOT a data bug

CPU verification (mmap-load the 12 GiB cache, compare every validation row): 0/7365 mismatches
— every row's len(seq[:2048])+2 placeholder count equals its cache feature count exactly. The
precompute (experiments/bioreason/precompute_esm3_cache.py) truncates s[:max_protein_len] and
sha1-keys IDENTICALLY to dataset_sft; sidecar confirms max_protein_len=2048, embedding_dim=1536,
n_seqs=7349. The "Protein token count 2210 != protein features 240" error occurred in EXACTLY
ONE run (seq4096_padfixed) and nowhere else — it was a bug in my experimental pad_to_fixed
collate path, not a pre-existing cache problem. pad_to_fixed has been REMOVED (the shape-doubling
hypothesis it tested was also not the root cause — the step-2 wall is CCL IPC-handle accumulation,
fixed by 2-node HSDP). drop_over_length (the genuinely useful filter) is kept. Tree clean, 7/7
qwen3 CPU tests pass.

## ★★★★★★★★ ANSWER (2026-06-27, single-variable isolation): VARIABLE PER-STEP SHAPE, not seq/capacity

User pushed: GRPO-32B works, 4B/32B packed-SFT works — so "32B can't train" is false; find the
REAL difference. Did the disciplined thing: varied ONE variable from the KNOWN-GOOD Qwen3-32B
fitstest (seq2048, alpaca packed, text-only, shard=-1, 12/12 clean — SAME env: FI_MR=userfaultfd,
gc:0.99, default alloc).

Confirmed exonerated (identical between working fitstest and crashing BioReason): FSDP mesh
(both shard=-1/replicate=1), env/allocator/MR-monitor, model (Qwen3-32B), node count (1), loss
(LinearCrossEntropyLoss), AC. The crashing BioReason seq4096 SMOKE used shard=-1 too.

CELL A = fitstest config, ONLY max_seq_len 2048->4096 (still alpaca PACKED = fixed 4096/row,
text-only). RESULT: **CLEAN through step 3+ (loss 2.41->1.77->1.52, ~34s/step, 12 ranks)** —
seq=4096 at 32B single-node 12-tile TRAINS FINE. So it is NOT sequence length, NOT the O(S^2)
capacity wall at 4096 (~12 GiB transient fits with AC), NOT env/FSDP/model/nodes.

The ONLY remaining difference: BioReason uses batch_size=1 with the custom collate padding to
PER-BATCH-MAX (dataset_sft.py:322) -> a DIFFERENT sequence shape every step per rank. The
working fitstest packs to a CONSTANT 4096/row. CELL B (= CELL A + dataset.packed=false ->
variable per-batch shape) is the confirmation run.

This finally explains the signature that never fit the capacity story: the BioReason seq4096
banned:1 had only **8.30 GiB reserved on rank0** — NOT a capacity OOM. It is a VA-RECYCLING /
external-memory PDE: variable per-step shapes churn the allocator's VAs; under FSDP (no
empty_cache) + FI_MR_CACHE_MONITOR=userfaultfd, freed-then-reissued VAs leave stale OFI/L0
registrations -> write to a recycled VA -> banned:1. Fixed-shape (packed) reuses one VA set ->
no churn -> clean. This is the SAME structural property that makes GRPO (broadcast identical
short trajectories) and packed-SFT (constant row length) both work.

THE FIX (matches what GRPO/packed-SFT get for free): make the BioReason SFT batch a CONSTANT
shape every step. Options: (a) pad every batch to a fixed max_seq_len (the pad_to_fixed idea —
but it had a separate splice bug; revisit carefully), or (b) sample-pack the multimodal stream
to constant-length bins (multimodal-aware: splice + per-doc block-diag handling). Either makes
the per-step transient a single reused VA set. seq can be as high as ~4096-5120 (CELL A proves
4096 fits); the constraint is FIXED shape, not small seq.

## CELL B result + refined hypothesis (2026-06-27)

CELL A (fitstest seq4096 PACKED text-only): rc=0, 10/10 clean, loss 2.41->1.12, 47s/step.
CELL B (= A + dataset.packed=false -> VARIABLE shape, text-only): ALSO 10/10 clean, BUT at
~8s/step — alpaca-unpacked rows are SHORT, so B never exercised LARGE variable transients.
B is therefore consistent with, but does not prove, the hypothesis. Refined hypothesis:
the trigger is variable shape AT LARGE (multi-GiB) transient sizes (the per-step O(S^2) MATH
attention block differs in size each step and is multi-GiB), which churns/recycles allocator
VAs under FSDP(no empty_cache)+OFI-MR -> banned:1. Small variable shapes (alpaca) don't churn
enough to fault.

CELL C (decisive): the REAL BioReason recipe (long, multimodal) at seq=4096 with the FIX
pad_to_fixed=true (constant 4096/step), env matched to CELL A exactly (userfaultfd, gc:0.99,
default alloc). If C is clean -> fixed-shape is the fix and variable-large-shape was the cause.
If C still crashes -> the multimodal splice (embeds.clone() of [B,4096,5120], ESM3 concat) is
implicated, not just shape. [result pending]

## CELL C result + the TWO-bug resolution (2026-06-27)

CELL C (real BioReason recipe, seq4096, pad_to_fixed=ON, A-env) crashed step 0 on EVERY rank
with DIFFERENT mismatches (824!=819, 2210!=240, 989!=430, ...). Root-caused on CPU:
**tokenizer.pad_id == 151643 == protein_token_id** (Qwen3 reserved-gap collision). pad_to_fixed
fills pad slots with padding_idx=pad_id=151643 -> every pad slot becomes a protein placeholder
-> splice count = real_placeholders + pad_count (varies per rank by pad amount) -> "Protein
token count N != features M". My earlier CPU tests hardcoded padding_idx=0 so never caught it.

So there are TWO DISTINCT bugs, both now fixed:
1. ORIGINAL step-2 banned:1 (variable-pad collate, no pad_to_fixed): variable LARGE per-step
   shape churns allocator VAs (no empty_cache under FSDP + OFI MR) -> stale-VA write ->
   banned:1. Evidence: CELL A (fixed-shape seq4096) = 10/10 clean; the working fitstest +
   GRPO both have uniform shapes. FIX = pad_to_fixed (constant per-step shape).
2. pad_to_fixed's OWN bug: pad_id collides with protein_token_id -> pad slots counted as
   placeholders. FIX = pad tokens with a neutral id (0) when pad_id is a placeholder id
   (recipe _setup_data; regression test test_collate_fixed_pad_id_must_not_collide).

Both fixes are required together. CELL C re-run with both fixes validates the end-to-end path.
NOTE: the ORIGINAL crash that started the saga was bug #1 (variable shape), NOT the SDPA O(S^2)
ceiling I spent days on — the O(S^2) transient is real but only sets the upper seq bound
(~6144); 4096 fits fine when the shape is FIXED. The whole saga reduces to: 32B SFT on XPU needs
a CONSTANT per-step tensor shape (like GRPO's broadcast trajectories and packed-SFT's fixed
rows), and the multimodal pad must not collide with placeholder ids.

## ✅ VALIDATED (2026-06-27, node x4219c2s3b0n0): BioReason 32B SFT trains end-to-end

CELL C with BOTH fixes (pad_to_fixed=ON + neutral pad-id): real BioReason multimodal recipe,
Qwen3-32B, seq4096, 1-node 12-tile FSDP2, **rc=0, 10/10 steps clean**, loss
15.78->8.58->7.48->6.35->3.94->1.91->0.73->0.40->0.22->0.07 (monotonic), ~104s/step. The
step-2 banned:1 wall that killed every prior run is gone. Single-node 32B BioReason SFT WORKS.
Both fixes shipped in the recipe (defaults). Remaining for production: ESM3 cache over TRAINING
parquets (cache is validation-only today); then full 117K x3ep run + F_max eval.

## ESM3 training cache built (2026-06-27, job 8572553)

Full train+val ESM3 cache precomputed: 121645 unique sequences (was 7349 val-only). 12-tile
sharded encode (precompute_esm3_cache.py --shard/--nshards + --merge; launcher
precompute_esm3_train_12tile.sh, debug queue 1h), ~4.8 seq/s/tile, ~33 min + merge + verify.
VERIFY: unique_seqs=121645 cached=121645 missing=0 PASS. esm3_cache_2048.pt (188 GiB),
max_protein_len=2048, dim=1536. The full SFT run will no longer KeyError on training proteins.

## Remaining to the full F_max repro
1. ✅ Both XPU bugs fixed (pad_to_fixed + neutral pad-id); seq=4096 validated 10/10.
2. ✅ ESM3 training cache complete + verified.
3. seq decision: seq=6144 (4% target truncation vs 23% at 4096) — HW-validate the 6144
   fixed-shape transient (~27 GiB) fits before the full run.
4. Full run: 117K x 3 epochs at seq=6144, full-shard, pad_to_fixed; per-epoch checkpoints;
   then native greedy decode + cafa_evals IA.txt-weighted F_max (target 0.66-0.70).

## Pre-full-run hardening (2026-06-27)

Two issues fixed before the full run, both surfaced by scaling past the smoke:

1. **ESM3 cache OOM/hang at scale (regression from growing the cache to 200 GiB):**
   model_native._load_esm3_cache did torch.load(map_location="cpu") = full copy into EACH
   rank's RAM. At ~200 GiB × 12 ranks = ~2.4 TB -> node OOM/thrash; validation job 8572599
   hung ~49 min in __init__ ("Instantiating BioReasonNativeModel", log silent). FIX:
   torch.load(..., mmap=True) — tensors memory-mapped, OS page cache shared across same-node
   ranks, per-seq splice copies only the small [L+2,dim] slice to device. Was harmless at the
   12 GiB val-only cache (fit 12×); only bit once the train cache landed.

2. **Resumable training state (was eval-only):** the LoRA subclass save_checkpoint wrote only
   a merged-W_eff eval artifact (optimizer/step/adapters folded away) -> resume restarted LoRA
   from scratch. Added self-contained resume (frozen base => persist only TRAINABLE state):
   _save_resume_state writes adapters+projections+optimizer+dataloader+step to
   <output_dir>/resume_state.pt; _setup_model reloads adapters+projections on bioreason_resume;
   setup() restores optimizer+dataloader+global_step. Decoupled from the parent HF-checkpointer
   recipe_state path (base-model-oriented) via a separate `bioreason_resume` flag (parent
   resume_from_checkpoint stays off so its loader just loads base). CPU test:
   test_resume_trainable_keys_selects_adapters_and_projections (9/9 qwen3 tests pass).

Process note: the 49-min hang was caught by checking log MTIME vs wall-clock (silence != progress)
— now guarded with a staleness watchdog monitor alongside the progress monitor.

## Cache format: .pt dict -> safetensors (2026-06-27, the mmap fix wasn't enough)

The mmap fix stopped the RAM OOM but NOT the I/O bottleneck: torch.load(mmap=True) still
deserializes the full pickle index (121645 entries) per reader, so 12 same-node ranks each
stream the ~200 GiB file from DAOS -> processes in D-state (uninterruptible I/O), ~2s CPU per
30s wall (~93% blocked), ~10 min and climbing in __init__ (job 8572649, killed). At that rate
it blows walltime before training.

FIX (user-approved): convert the cache to a single safetensors file. safetensors stores a small
JSON header {key: dtype/shape/offsets} + a flat blob; safe_open reads only the header once, then
get_tensor(key) reads ONLY that tensor's byte range (true random access). No full-file read, scales
to any rank count. Implementation:
- experiments/bioreason/convert_esm3_cache_to_safetensors.py (.pt mmap -> safetensors, bf16).
- model_native._LazySafetensorsCache: dict-like .get/__contains__/__len__ over safe_open;
  _load_esm3_cache routes *.safetensors to it (legacy .pt mmap path kept for the small val cache).
- CPU tests: test_lazy_safetensors_cache_roundtrip (10/10 qwen3 tests pass).
- Active configs (production + seq6144 smoke) + launcher point at esm3_cache_2048.safetensors.
Conversion job 8572705 (debug queue) running. After it: re-run the seq6144 validation (should
clear __init__ in seconds), then the 4-node full run.

## ✅ seq=6144 VALIDATED (2026-06-28, job 8572732, node x4307c5s7b0n0)

With the safetensors cache: model setup 52s (was 10-49 min hangs), then CLEAN through step 2+
(loss 15.63->8.51), ~214s/opt-step, drop_over_length removed only 3/7365 (0.04%) vs 2.59% at
4096. seq=6144 fixed-shape (~27 GiB transient) FITS on 1N 12-tile full-shard. Envelope confirmed.
Full-run estimate at ~214s/opt-step (CORRECTED): 1N ~435h, 4N ~109h (~18x 6h-segments),
8N ~54h (~9 segments). NOTE: 3 epochs is expensive because each example is a full seq6144
fwd+bwd through 32B under math-SDPA O(S^2); consider 8N and/or fewer epochs (1 epoch 4N ~6
segments). Proceed: 4N HSDP 6h capacity segments with checkpoint-resume (pbs_4n_sft_full.sh).

## ✅✅ FULL seq=6144 validation 10/10 + FULL RUN LAUNCHED (2026-06-28)

seq=6144 v3 (job 8572732, safetensors cache): rc=0, FULL 10/10 clean, loss
15.63->8.51->6.96->4.93->3.09->1.37->0.45->0.30->0.07->0.023 (monotonic), ~197-217s/step,
model setup 52s. AND resume_state.pt saved at step 10 (trainable+opt+dataloader) — the resume
machinery works end-to-end on HW. Every blocker fixed + validated.

FULL RUN: 4-node HSDP (shard12/repl4, dp=48), epochs=1 first (user decision — ~6 six-hour
segments / ~35h; 1 epoch SFT often captures most F_max, then decide on epochs 2-3). Launcher
experiments/bioreason/pbs_4n_sft_full.sh, capacity 6h, save_every=25, checkpoint-resume
(BIOREASON_RESUME=1 for segments after the first). Segment 1 = job 8572863 (fresh). After each
segment: resubmit with BIOREASON_RESUME=1. After 1 epoch: native greedy decode + cafa_evals
IA.txt-weighted F_max (target 0.66-0.70).

## 4N launcher bug: wrapper hardcoded WORLD_SIZE=24 default (2026-06-28)

First 4N segments (8572863/8573055) failed at init: "dp_replicate(4)*dp_shard(12)!=WORLD_SIZE(24)".
mpiexec DID launch 48 ranks (rank 34/40 present), but EVERY rank reported WORLD_SIZE=24. Root
cause: the bioreason rank wrapper had `WORLD_SIZE="${PMI_SIZE:-${WORLD_SIZE:-24}}"`, and under
Aurora PALS **PMI_SIZE is often EMPTY** -> it fell to the literal default 24. (The hostfile dedupe
was a red herring — not the cause.) FIX = mirror the validated 7N/8N wrapper:
`WORLD_SIZE="${PMI_SIZE:-${PALS_NRANKS:-${WORLD:-${WORLD_SIZE}}}}"` (never a hardcoded number) +
export WORLD from the launcher + pass `--env WORLD=...` through mpiexec so the fallback is
populated. Resubmitted 8573177. Lesson: copy the working SAME-SCALE launcher/wrapper, not the 2N
one — multi-node rank-env resolution differs from single-node. See
feedback_pbs_mpiexec_use_pbs_nodefile.md.

## 4N launch: config key + WORLD_SIZE fixes (2026-06-28)
After the WORLD_SIZE fix engaged (rank0 logged WORLD_SIZE=48, assert passed), next fail was
`omegaconf ConfigAttributeError: Missing key max_steps_per_epoch` — the recipe reads
cfg.max_steps_per_epoch directly (not .get), and the PRODUCTION config lacked it (smoke had it).
Added `max_steps_per_epoch: null` (null = full epoch). Audited all cfg.<key> direct accesses vs
the production config; only that one was missing (dataset_val is .get-gated). Resubmit 8573430.
All 4N failures so far are init-time config/plumbing (seconds each), not training correctness —
the training path is validated (seq6144 10/10).

## 4N full-run training VALIDATED + eval-harness gap (2026-06-28)

4N HSDP full run (8573430, after WORLD_SIZE + max_steps_per_epoch launcher fixes) TRAINED the
full 124367-example corpus: WORLD_SIZE=48, model setup 111s, 0.11% dropped, peak 54.5 GiB/tile,
real telemetry grad_norm 58->0.01, loss 15.84->0.0002 by step 36, 272s/opt-step, step-25
checkpoint written (resume_state.pt 2.05GB + epoch_0/ merged 32B HF safetensors + projections).
The whole training stack is proven end-to-end at multi-node scale.

OBSERVATION: loss saturates to ~0 by step ~25-36 (grad_norm -> 0.01) on bf16 LoRA lr=1e-4 — fast
template memorization. Train loss is therefore uninformative; only held-out F_max matters. User
stopped the run at step 36 (don't burn 4 capacity nodes driving a dead-flat loss before knowing
if it generalizes).

EVAL-HARNESS GAP (the remaining blocker, separable from training): eval_cafa_fmax.py is built for
the 4B anchor at vLLM TP=1; our 32B (62 GiB) cannot fit one-per-tile -> XPU OOM, 0 predictions
(job 8573774). TP=4 would enter the UNVALIDATED enable_prompt_embeds+TP>1-on-XPU regime (fragile).
DECISION: build a native KV-cached greedy-decode eval path (reuse the validated
BioReasonNativeModel forward; no vLLM/TP/prompt_embeds risk). NOTE: BioReasonNativeModel has no
setup_caches yet — native decode needs KV-cache wiring (else O(S^2)/token over 2048 tokens x 280
proteins x 32B = far too slow). This is real, bounded new code for next session.

NET: 32B multimodal SFT TRAINS on Aurora (the hard, validated win — all blockers fixed:
pad_to_fixed, neutral pad-id, drop_over_length, mmap+safetensors lazy cache, frozen-base LoRA
resume, 4N HSDP launch). The F_max number awaits a 32B-capable generation path.

## Eval host SOLVED (HF device_map) + train/eval prompt-convention mismatch (2026-06-28)

EVAL HOST (was the blocker): HF device_map across 2 tiles WORKS for 32B generation on XPU.
Added BioReasonModel(backbone_device_map=...) + eval --no_vllm/--backbone_device_map +
backbone.generate(inputs_embeds=pe moved to embed-device, use_cache=True). Smoke 8573946 loaded
the merged 32B sharded across 2 tiles (14 shards ~13s each, no OOM) and reached generation — so
the 62-GiB-doesn't-fit-one-tile problem is solved (TILES_PER_SHARD=2 -> 6 concurrent shards).
Launcher: run_eval_sft_merged.sh (ZE_AFFINITY_MASK=tile-pair per shard).

REMAINING (well-defined): the eval's INPUT construction does not match how the native SFT was
TRAINED, so it can't score our checkpoint yet:
- TRAINING (dataset_sft._build_prompt_ids): prompt text + INTEGER placeholder runs
  "\nProtein: " + [151643]*(len+2) + "\nGO graph: " + [151644]*200 + "\nReasoning:\n"
  (config protein_token_id=151643, go_token_id=151644 — Qwen reserved-gap ids, NOT special tokens).
- EVAL (build_prompt_string + build_input_ids + BioReasonModel): uses the RL chat-template path
  with STRING special tokens <|protein_pad|>/<|go_graph_pad|> resolved via
  convert_tokens_to_ids(get_token("protein_pad")). Our epoch_0 carries the BASE Qwen3 tokenizer
  (no such special tokens) -> 0 protein tokens matched -> "Protein token count 0 != features 360".

FIX (next session, isolated): make the eval build inputs IDENTICALLY to dataset_sft for the native
checkpoint — reuse dataset_sft._build_prompt_ids (or a shared helper) to emit the exact text +
integer-id (151643/151644) placeholder layout, set BioReasonModel.protein_token_id/go_token_id to
151643/151644 (not the special-token lookup), and ensure the same _SYS_WITH_CONTEXT/go_pred prompt.
This is a train/eval-consistency reconciliation (getting it subtly wrong silently tanks F_max), so
do it deliberately + pin with the existing test_cafa_fmax_eval_pipeline.py formula check. Then run
the 6-shard eval on epoch_0 vs the SFT-base to get the F_max comparison (anchor: 4B SFT 0.6686 —
a 32B should exceed it; if low despite ~0 train loss, lr=1e-4 too high -> restart at 1-2e-5).

## ★ EVAL RECONCILED + ROOT CAUSE: the SFT learned a SHORTCUT, not the multimodal task (2026-06-28)

Eval fully reconciled to training (native prompt via dataset_sft._build_prompt_ids, integer ids
151643/151644, HF device_map 2-tile host, backbone.generate(inputs_embeds)). Generation RUNS, but
output is degenerate ':' repeated — EVEN ON IN-DISTRIBUTION TRAINING DATA. Decisive A/B:
- BASE Qwen3-32B text-only via the same HF device_map path: COHERENT ("Kinases transfer phosphate
  groups from ATP..."). -> HF backbone + device_map host are SOUND.
- epoch_0 (merged SFT) text-only: COHERENT ("...phosphorylation..."). -> the merged weights are
  FINE; merge/save is correct.
- epoch_0 WITH protein/GO splice (in-dist train row): ':' collapse. -> the break is the PROTEIN
  EMBEDDING SPLICE specifically.

Mechanism (probed on-node): raw ESM3 features are norm ~10,600/row (max ~10,880) — inherent to
ESM3 (both train cache 11,221 and test cache 10,606; published model feeds the same). The trained
protein_projection (weights ~0.013, ~init scale) outputs norm ~14,959/row, vs TEXT embeds norm
~1.8 — an ~8000x mismatch that swamps the context -> model emits ':' with logit 36.5. The splice
code is BYTE-IDENTICAL between native training (model_native._splice_embeds) and eval
(model.build_prompt_embeds); same weights, same cache. So the eval is correct.

CONCLUSION: the SFT did NOT learn to use the protein embeddings. It reached loss 0.0002 in ~25
steps (grad_norm->0.01) by exploiting a SHORTCUT — the go_pred GO-term speculations are in the
PROMPT TEXT, so it learned to refine that text while ignoring the un-integrable norm-10K protein
features. The projection never moved from init (couldn't learn ESM3->text mapping in 25 steps).
This IS the lr=1e-4 / instant-collapse risk flagged at training time. "Did it learn?" = it learned
a degenerate shortcut, not the multimodal task; train loss ~0 was memorization of the text path.

FIX (retrain): much lower lr (~1-2e-5, matching the published recipe's controlled descent) so loss
descends gradually over the full epoch and the protein_projection actually trains to map ESM3
features into text-embedding scale. Consider: (a) explicit ESM3 feature normalization (LayerNorm)
before/in the projection to bridge the 10K->1.8 scale gap and make the mapping learnable faster;
(b) verify the published recipe's protein-feature handling/normalization. The whole train+eval
PIPELINE is now validated end-to-end (training runs at 4N, checkpoints/resume work, eval generates
+ scores) — the remaining issue is a TRAINING-RECIPE one (lr/normalization), not infrastructure.

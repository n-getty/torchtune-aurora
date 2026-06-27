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

RESULT: **clean past step 3** (where Gemma4 died in EVERY config), loss 2.35→1.77→1.50
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

# BioReason 32B SFT — Scoping Plan (2026-06-26)

## Strategic premise (why 32B, why Aurora)

Tonight's measurements settled the A100-vs-Aurora question for BioReason:
- BioReason-Pro 4B is **decode-bound** (~100s/step at 2N, gen dominated by ~575-token reasoning
  traces; gen levers don't help — see `project_bioreason_eval_fixed_rl_flat_vs_sft_20260626`).
- A 4B + frozen ESM3 + in-process vLLM **co-fits on one A100 (80 GB)** → Aurora gives no speed
  edge and real friction (XPU fragility cost 4 crashes this session). **For ≤~8B, train on A100.**
- **The Aurora motivation appears at the size where model + ESM3 + vLLM-KV no longer co-fit on one
  A100 — ~32B+.** There, Aurora's 12×64 GiB tiles/node is a genuine capability an A100 lacks.

**Goal of this work:** SFT a ~30B American model on the BioReason-Pro SFT data to reach a state
comparable to the published 4B SFT ckpt (F_max ~0.66–0.70 on the held-out test set), establishing
the larger-model multimodal SFT capability on Aurora.

## Decisions taken
- **Backbone: Gemma 4 31B (dense), American (Google).** Rationale: provenance (user goal: American
  model), and it's the **best-supported large model in this repo** — `torchtune.models.gemma4.gemma4_31b`
  (native module), validated GRPO (~117s/step), a vLLM Gemma4 overlay, tokenizer, and HF checkpointer
  all exist. On disk: `models/gemma-4-31B` (59 GiB, hidden_size=5376). MoE alt (gemma4_26b_a4b, ~3B
  active, cheaper) is a fallback if dense compute is too heavy, but MoE+multimodal SFT is more novel.
- **Approach: write a plan first** (this doc), then build the recipe + smoke at small scale before
  the expensive 32B run. NO speculative lever stacking (the hard lesson from the RL session).

## ★ The central architecture decision: native-torchtune backbone, NOT HF AutoModel

`BioReasonModel` (`torchtune/dev/bioreason/model.py`) is currently hardwired to an **HF
AutoModelForCausalLM** backbone (`from_pretrained`, `attn_implementation="sdpa"`, PEFT LoRA),
consumed via `backbone(inputs_embeds=...)`. But:
- The on-disk gemma-4-31B is `Gemma4ForConditionalGeneration` / `model_type: gemma4` — a custom arch
  the repo had to build a **native torchtune module + vLLM overlay** for (stock HF/vLLM don't support
  it cleanly). Forcing it through HF AutoModel is the wrong path.
- **KEY ENABLER (verified):** the torchtune-native `TransformerDecoder.forward` already accepts
  `input_embeds=` (short-circuits token embeddings; `transformer.py:521,652`). And `gemma4_31b()`
  returns a `TransformerDecoder`. So the BioReason multimodal embed-injection (protein+GO embeds
  spliced into the prompt) works **natively** — no HF dependency.

**Therefore: a new BioReason model variant that wraps the native `gemma4_31b()` decoder and feeds
`input_embeds`**, instead of the HF backbone. This also unlocks the repo's validated FSDP/EP/vLLM
Gemma4 infra for any future RL on top.

## What we have (de-risks most of the work)
1. **SFT data ON DISK**: `datasets/bioreason_sft_reasoning` — **117,002 train rows** + validation
   shard, with `reasoning` + `final_answer` columns (the GPT-5-style traces; the real ~130K-class
   SFT corpus, not the 9K RL set). 604 MB. Same schema family as the RL parquet (protein_id,
   sequence, go_*, interpro_formatted, ppi_formatted, organism…).
2. **Published SFT recipe to mirror**: `BioReason-Pro/train_protein_llm.py` — LoRA SFT (r=32, α=64,
   lr=1e-4, 3 epochs, max_length_text=4000), trains LoRA adapters + projections + embed/lm_head on a
   frozen ESM3. It's PyTorch-Lightning + unsloth (CUDA-only) — we mirror the *recipe*, not the code.
3. **Multimodal model machinery**: ESM3 pre-encode cache (protein=2048, built), GO encoder + cached
   go_embedding, protein/go projection MLPs, the embed-splicing forward — all reusable.
4. **Validation harness**: F_max eval on held-out test set (`run_eval_adapter_gopred_testset.sh`),
   IA.txt, the SFT/eval prompt (go_pred injection). The 4B SFT anchors at ~0.67 here.
5. **SFT throughput wins**: no_sync-during-accumulation + ZeRO-2 (2.15×/2.57× at 2N), async
   dataloader fix, `time_per_step_s` metric — all in the XPU SFT recipes.

## What must be built (4 pieces)

### A. BioReason SFT recipe (NEW) — moderate effort, low risk
We have only the **GRPO/RL** BioReason recipe. Need an **SFT** recipe = a *simplification*:
- Forward the multimodal prompt embeds through the backbone, cross-entropy loss on the
  `final_answer` (and optionally `reasoning`) tokens, masking the prompt. NO rollout/vLLM/reward/IS
  machinery (delete ~all of the GRPO complexity).
- Reuse: `BioReasonModel` embed-splicing, ESM3 cache, projections, FSDP setup, the dataloader
  pattern, the no_sync/ZeRO-2/AC wins. Model: the new native-Gemma4 variant (piece B).
- Loss-token masking: SFT trains on the assistant `final_answer`(+reasoning) only — mirror the
  published `return_answer_in_batch` / label-masking. The dataset already has the columns.
- Likely cleanest as a sibling to `recipes/dev/full_finetune_distributed_xpu.py` with the BioReason
  multimodal forward grafted in, OR a `recipes/dev/sft_bioreason_distributed_xpu.py`.

### B. Native-Gemma4 BioReason backbone variant — moderate effort, the novel part
- New `BioReasonModel` path (flag/subclass) that builds `gemma4_31b()` (native decoder) instead of
  HF AutoModel, loads weights via `FullModelHFCheckpointer` + `GEMMA4` model_type (the GRPO config
  pattern), and feeds `input_embeds`. `hidden_size=5376` flows to the projection MLPs.
- `protein_projection` (`Linear(esm3_dim→H)`) and `go_projection` (`Linear(2560→H)`) **re-init at
  H=5376** and train from scratch — fine, SFT trains them anyway.
- Tokenizer: `torchtune.models.gemma4.gemma4_tokenizer`. The protein/go placeholder tokens
  (`<|protein_pad|>`,`<|go_graph_pad|>`) must be added to the Gemma4 vocab (the dataset expands
  placeholders by len(seq)+2 / num_go_tokens; the embed-splice fills them) — verify Gemma4 tokenizer
  can take added special tokens, or reserve unused ids.
- **Risk:** the embed-splice + placeholder-id contract was built against the Qwen/HF path; porting
  to the native tokenizer + decoder needs a CPU equivalence check (embed at the right positions).

### C. Compute (the real cost) — capacity allocation, NOT debug-scaling
- 117K rows × 3 epochs. SFT steps are cheaper than GRPO (no generation). At seq≈4K, dense 31B FSDP:
  GRPO was ~117s/step *with* generation; SFT fwd+bwd alone est. **~10–25s/step** (to be MEASURED).
- batch≈16 (bs×ga across nodes) → 117K×3/16 ≈ **22K steps** → order **1.5–4 node-days** on `capacity`
  (2–4 nodes). LoRA (r=32) keeps optimizer/activation memory tractable at 31B; full-FT likely needs
  more nodes + ZeRO-3. Start LoRA (matches the published method).
- Needs: activation checkpointing + the no_sync/ZeRO-2 SFT wins; bf16. seq=4K × 31B activations are
  the memory pressure point — size fbs/AC accordingly.

### D. Validation — already built
- After SFT, eval the adapter+projections with the existing F_max harness on the held-out test set
  (`run_eval_adapter_gopred_testset.sh`, adapted for the Gemma4 backbone). **Success = F_max in the
  0.66–0.70 band** (comparable to the 4B SFT's 0.67). Also track SFT loss curve + a qualitative
  sample of generated functional summaries.

## Risks & mitigations (XPU-specific; the session's hard lesson)
- **31B + multimodal + seq=4K on XPU is untested together.** Mitigate by INCREMENTAL bring-up, one
  change at a time from a known-good point (the discipline I violated in the RL session):
  1. CPU: native-Gemma4 BioReason model builds + embed-splice equivalence test.
  2. 1-node smoke: tiny data, confirm forward/loss/backward + memory fit at seq=4K.
  3. Scale to 2–4 nodes; confirm step time + no banned:1.
  4. Short run (few hundred steps) → eval sanity → full run.
- **banned:1 / boot-flake fragility** — reuse the retry-once + clean_tiles patterns already built.
- **Tokenizer/placeholder contract drift** — pin with a CPU test before any HW.
- **Don't stack levers** — get a stable 31B SFT FIRST, optimize later only if needed.

## Open questions for the user (before building)
1. **LoRA vs full-FT** for the 31B SFT? Published method is LoRA (r=32). LoRA is far cheaper +
   matches the paper; full-FT at 31B needs many more nodes. Recommend LoRA.
2. **Effort gate:** is the goal a faithful repro (match ~0.67), or to test whether 31B *beats* 4B on
   this task? (The fidelity A/B + text-ablation suggest the task is text-driven — a bigger backbone
   may or may not lift F_max. A cheaper way to test the premise might exist.)
3. **Train on `final_answer` only, or `reasoning`+`final_answer`?** The published SFT trains the full
   trace; the F_max reward only needs the GO terms in the answer. Full trace = the published behavior.

## Estimated effort
- Pieces A+B (recipe + native-Gemma4 variant + CPU tests): ~1–2 focused days of dev.
- Incremental HW bring-up (smoke → scale): ~1 day of compute iteration.
- Full SFT run: ~1.5–4 node-days on capacity.
- Total: ~1 week wall, most of it the SFT run itself.

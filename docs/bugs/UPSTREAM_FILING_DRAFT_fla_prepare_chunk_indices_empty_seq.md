# `prepare_chunk_indices` mis-attributes chunk ownership when a zero-length sequence is present

**Target**: fla-org/flash-linear-attention (github.com/fla-org/flash-linear-attention) —
`fla/ops/utils/index.py` (or wherever `prepare_chunk_indices`/`prepare_chunk_offsets`
currently live upstream; vendored copy in this repo is
`vllm/model_executor/layers/fla/ops/index.py`, copied into vLLM per the file's own
SPDX header, "Copyright (c) 2023-2025, Songlin Yang, Yu Zhang").
**Filer**: ngetty / ALCF ModCon (TorchTune RL on Aurora, via vLLM's vendored copy).
**Found via**: Kimi-K3 XPU serving investigation (`experiments/kimi_k3_serving/`).

## TL;DR

`prepare_chunk_indices` infers each output row's owning-sequence index by **counting**
chunk-start markers (`indices.eq(0).cumsum(0) - 1`) instead of deriving it directly from
`cu_seqlens`. A zero-length sequence contributes zero chunks, so it is never counted as a
chunk start and is silently skipped — every later sequence's inferred index then shifts
down by one. Any kernel that uses that index to look up per-sequence bounds (e.g.
`bos, eos = cu_seqlens[i_n], cu_seqlens[i_n + 1]`) reads the wrong sequence's token range.
On Intel XPU this manifests as an out-of-bounds memory access (a GPU page fault); on CUDA
it would silently read/write the wrong sequence's data rather than crash, which is the
more dangerous failure mode.

We hit this via the `chunk_kda` prefill path (`fla/ops/kda.py`'s `chunk_kda_fwd` →
`chunk_gated_delta_rule_fwd_h` → `chunk_o.py`'s `chunk_gla_fwd_o_gk`), but the buggy
function is shared by every chunked kernel in the library
(`fla/ops/index.py`'s docstring-less call sites: `wy_fast.py`, `chunk_delta_h.py`,
`chunk_o.py`, `solve_tril.py`, `cumsum.py`, `kda.py`, `chunk_scaled_dot_kkt.py` in the
vendored tree — likely a similar set upstream), so the bug is not KDA-specific.

## Root cause

Current (buggy) implementation, as vendored:

```python
@tensor_cache
def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    indices = torch.cat(
        [
            torch.arange(n)
            for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
        ]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)
```

For each sequence, `torch.arange(n_chunks)` produces `[0, 1, ..., n_chunks-1]` (or an
empty tensor if `n_chunks == 0`). Concatenated across all sequences, `indices.eq(0)`
marks every position where a NEW sequence's chunk-0 starts. `cumsum(0) - 1` then counts
how many chunk-starts have been seen so far, using that count as the sequence index.

This is correct **only if every sequence contributes at least one chunk**. If sequence
`i` has zero length (`n_chunks[i] == 0`), it contributes zero rows and therefore no
`indices.eq(0)` marker — so the count never increments for it, and every sequence after
it is attributed to `count - 1` instead of `count`.

**Minimal repro** (pure tensor arithmetic, no GPU needed):

```python
import torch

def cdiv(a, b):
    return -(-a // b)

def prepare_lens(cu_seqlens):
    return cu_seqlens[1:] - cu_seqlens[:-1]

def prepare_chunk_indices_buggy(cu_seqlens, chunk_size):
    lens = prepare_lens(cu_seqlens)
    n_chunks = cdiv(lens, chunk_size)
    indices = torch.cat([torch.arange(n) for n in n_chunks.tolist()])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)

# seq0: length 0 (empty). seq1: length 3. seq2: length 5. chunk_size=64 (1 chunk each).
cu_seqlens = torch.tensor([0, 0, 3, 8], dtype=torch.int32)
print(prepare_chunk_indices_buggy(cu_seqlens, 64))
# tensor([[0, 0],
#         [1, 0]], dtype=torch.int32)
# WRONG: real sequences are index 1 and index 2, not 0 and 1.
```

Expected output: `[[1, 0], [2, 0]]` (sequence 1's chunk 0, sequence 2's chunk 0).
Actual output: `[[0, 0], [1, 0]]` — off by one for every sequence after the empty one.

## Downstream impact

Every consumer that combines `chunk_indices[i, 0]` (the sequence index) with `cu_seqlens`
to compute per-sequence bounds is affected, e.g. (paraphrased from the vendored
`chunk_o.py`):

```python
i_n, i_t = tl.load(chunk_indices + i_c * 2), tl.load(chunk_indices + i_c * 2 + 1)
bos, eos = tl.load(cu_seqlens + i_n), tl.load(cu_seqlens + i_n + 1)
```

With the off-by-one `i_n`, `bos`/`eos` bound the WRONG sequence's tokens. On our XPU
target this produces an out-of-bounds load — a GPU page fault (Intel Level Zero
`banned: 1`, `access: 0 (Read)`). On CUDA, the same off-by-one would not fault (both
indices are in-bounds of the same underlying buffer) — it would silently compute the
wrong sequence's output, which is a correctness bug we believe is currently undetected
upstream because it requires a **non-trailing** zero-length sequence to manifest, and
most batching code places padding at the end (a trailing empty sequence does NOT trigger
this — see below).

## When this does and doesn't trigger

- **Triggers**: any zero-length sequence NOT at the end of `cu_seqlens` (i.e., at least
  one real, nonzero-length sequence follows it).
- **Does NOT trigger**: a trailing zero-length sequence (common padding convention —
  nothing after it needs to be correctly attributed), or a batch with no zero-length
  sequences at all (the common case, which is presumably why this has gone unnoticed).

We separately confirmed (in vLLM's own KDA integration) that production traffic only
ever pads with trailing zero-length entries, so this specific manifestation is not
reachable in that caller — but the bug is general to any caller of
`prepare_chunk_indices` that can produce a non-trailing empty sequence (e.g. chunked
prefill with heterogeneous scheduling, or any test harness enumerating edge cases).

## Suggested fix

Derive the sequence index directly via `repeat_interleave` over each sequence's own
chunk count, rather than inferring it by counting chunk-start markers:

```python
@tensor_cache
def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    n_chunks = triton.cdiv(prepare_lens(cu_seqlens), chunk_size)
    indices = torch.cat([torch.arange(n) for n in n_chunks.tolist()])
    seq_indices = torch.repeat_interleave(
        torch.arange(n_chunks.numel(), dtype=indices.dtype), n_chunks.cpu()
    )
    return torch.stack([seq_indices, indices], 1).to(cu_seqlens)
```

A sequence contributing zero chunks then correctly contributes zero rows to
`seq_indices` too — there is no inference step left that can misattribute it. Verified
against the repro above: emits `[[1, 0], [2, 0]]` as expected. Also verified against
several other cases (control with no empty sequence, multi-chunk sequences, trailing
empty sequence, multiple interleaved empty sequences) — matches the buggy
implementation's output everywhere except where the bug actually manifests.

We checked `prepare_chunk_offsets` (same file) for the same class of bug and it is
**not** affected — it derives offsets via a direct `cumsum` over real per-sequence chunk
counts, so an empty sequence correctly contributes a zero-width gap rather than being
silently skipped.

## Hardware confirmation (Intel XPU)

Reproduced standalone via `chunk_kda` directly (not just the isolated tensor-arithmetic
repro above): calling `chunk_kda(..., cu_seqlens=[0, 0, 3, 8])` (i.e. `seq_lens=[0,3,5]`)
faults with `Segmentation fault from GPU ..., level: 1 (PDE), access: 0 (Read),
banned: 1` on every attempt; calling it with the leading empty sequence dropped
(`cu_seqlens=[0, 3, 8]`, i.e. `seq_lens=[3,5]`) — identical inputs otherwise — passes
cleanly and reproducibly (2/2). This isolates the fault to exactly this function; we did
not need to modify anything else to confirm the mechanism.

## What we're asking

- Confirm whether this reproduces against the current upstream `fla` release (we're
  working against a vendored copy inside vLLM, commit/version noted in vLLM's own
  `fla/ops/kda.py` header as "Copyright (c) 2023-2025, Songlin Yang, Yu Zhang" — no
  more precise upstream commit pin is recorded in the vendored tree).
- If confirmed, we'd appreciate either a fix landing upstream (so vLLM's next vendor-sync
  picks it up) or guidance on whether the suggested fix above matches your preferred
  approach, since we plan to carry the local patch in vLLM's vendored copy in the
  meantime and would rather not diverge silently.

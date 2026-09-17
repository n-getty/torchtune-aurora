"""Convert vLLM's per-choice sampled-token logprobs into a trainer-shaped ``pi_old``.

WHY THIS EXISTS. Enabling async gen/train overlap takes the entire vLLM HTTP wait off
the critical path (measured at BioReason 32B 2N: ``vllm`` 233.1s -> 0.0s on all 12 ranks)
but simultaneously ADDS a ~95s rollout policy forward, because ``GRPOLoss`` needs a real
``pi_old`` and the recipe's only source for one is a second no-grad forward over the full
``[B*G, P+C]`` batch. Net measured: 1.282x length-normalized instead of ~1.60x. The
sampler that produced the rollout already computed exactly that number and can return it
for free. This module is the alignment layer between the wire format and the trainer's
tensor contract.

EXACTNESS. ``rlhf.batched_logits_to_logprobs`` computes ``log_softmax(logits / T)`` and
both ``pi_logprobs`` call sites pass ``self._temperature``. vLLM's default
``logprobs_mode="raw_logprobs"`` is UNSCALED, so using it as ``pi_old`` is a systematic
mismatch (at V=151936, T=0.8: mean |dlogp| 0.404 nats, worst IS-ratio error 4.09x), not
drift. ``_logprobs_engine_kwargs`` in ``vllm_backend.py`` sets ``processed_logprobs`` at
all four ``LLM(...)`` sites; at T=0.8 / top_p=1.0 / no top_k / no penalties
``apply_top_k_top_p`` early-returns, so processed == the trainer's formula exactly.
``require_processed_mode()`` below is the runtime assertion of that precondition -- do not
call the builder without it, because the failure is silent and biases every ratio.

NO SHIFT. This is the alignment trap. In the logits path, position t predicts token t+1,
so the recipe slices ``[:, ctx-1:-1]``. vLLM's ``token_logprobs[j]`` is the logprob OF the
sampled token ``comp[j]`` -- already the gathered value, no shift. Applying the logits
path's off-by-one here would misalign every row while still producing a plausibly-shaped
tensor with plausible magnitudes. The equivalence test pins this.

FALLBACK, NOT REPAIR. Any row whose logprobs are missing, misaligned, or non-finite is
reported by ``rows_needing_fallback``. The caller must then run the policy forward for the
whole batch. Partially filling a batch -- some rows from vLLM, some recomputed -- is
deliberately not offered: the two sources can differ by recompute noise (measured ratios
up to 1.0739 on a healthy run), so a mixed batch would carry a per-row systematic
difference correlated with whichever rows happened to fail, which is a confound that no
downstream statistic could detect.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "PROCESSED_LOGPROBS_MODE",
    "require_processed_mode",
    "rows_needing_fallback",
    "build_behavior_logprobs",
]

PROCESSED_LOGPROBS_MODE = "processed_logprobs"

# Padded response positions are overwritten by the recipe
# (``logprobs.masked_fill_(response_padding_masks, 1.0)``), so the fill value here is
# never read. 1.0 matches that sentinel so a dump of the intermediate tensor is not
# misread as a real logprob -- a positive "logprob" is self-evidently a padding marker.
PAD_FILL = 1.0


def require_processed_mode(mode: Optional[str]) -> None:
    """Fail loudly unless the serving engine is in ``processed_logprobs`` mode.

    Args:
        mode: the engine's configured ``logprobs_mode``. ``None`` means the field could
            not be determined, which is treated as a failure rather than assumed good:
            vLLM's own default is the wrong mode, so "unknown" is much more likely to be
            raw than processed.

    Raises:
        ValueError: if the mode is anything other than ``processed_logprobs``.
    """
    if mode != PROCESSED_LOGPROBS_MODE:
        raise ValueError(
            f"vLLM behavior-policy logprobs require logprobs_mode="
            f"{PROCESSED_LOGPROBS_MODE!r}, got {mode!r}. The default "
            f"('raw_logprobs') is UNSCALED while the trainer computes "
            f"log_softmax(logits / temperature); at temperature != 1.0 that is a "
            f"systematic pi_old mismatch, not numerical drift (worst IS-ratio error "
            f"4.09x measured at V=151936, T=0.8). Refusing to build pi_old."
        )


def rows_needing_fallback(
    logprobs: Sequence[Optional[Sequence[float]]],
    completions: Sequence[Sequence[int]],
    max_generated_tokens: int,
) -> list[int]:
    """Indices of rows whose vLLM logprobs cannot be trusted as ``pi_old``.

    A row fails if the server returned nothing for it, if the logprob list disagrees in
    length with the token list it is supposed to annotate, or if any value inside the
    used span is non-finite.

    The length comparison is made against ``min(len(comp), max_generated_tokens)`` --
    the SAME truncation the recipe applies when it writes completions into
    ``query_responses`` -- so a row that vLLM over-generated is not flagged merely for
    carrying logprobs past the cap.

    Args:
        logprobs: per-row logprob lists as returned by
            ``VLLMClient.generate_from_embeds(return_logprobs=True)``; an element may be
            ``None``.
        completions: per-row generated token ids, same ordering.
        max_generated_tokens: the response-length cap.

    Returns:
        Sorted list of row indices requiring the policy-forward fallback. Empty means
        every row is usable.
    """
    bad: list[int] = []
    n = max(len(logprobs), len(completions))
    for i in range(n):
        lp = logprobs[i] if i < len(logprobs) else None
        comp = completions[i] if i < len(completions) else []
        if lp is None:
            bad.append(i)
            continue
        used = min(len(comp), max_generated_tokens)
        # A row whose logprobs are SHORTER than the tokens they annotate cannot be
        # aligned at all. Longer is tolerated only when the excess is exactly the
        # over-generation the recipe itself truncates away.
        if len(lp) < used or len(comp) != len(lp):
            bad.append(i)
            continue
        if any(v is None or not _finite(v) for v in lp[:used]):
            bad.append(i)
    return bad


def _finite(v: float) -> bool:
    return v == v and v not in (float("inf"), float("-inf"))


def build_behavior_logprobs(
    logprobs: Sequence[Optional[Sequence[float]]],
    completions: Sequence[Sequence[int]],
    num_seqs: int,
    max_generated_tokens: int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Pack vLLM's per-row logprob lists into the trainer's ``[num_seqs, C]`` tensor.

    Position ``j`` of row ``i`` carries the sampler's logprob of the token the recipe
    wrote at ``query_responses[i, context_length + j]``. No shift is applied -- see the
    module docstring; this is the one place an off-by-one would be invisible.

    Call ``rows_needing_fallback`` FIRST and fall back for the whole batch if it returns
    anything. This function assumes every row is usable and will raise rather than guess.

    Args:
        logprobs: per-row logprob lists, aligned 1:1 with ``completions``.
        completions: per-row generated token ids.
        num_seqs: number of rows the trajectory expects (``B*G``).
        max_generated_tokens: response-length cap; the returned width.
        device: destination device. Defaults to CPU, matching the recipe's CPU-assembly
            path -- the consumer moves it.
        dtype: destination dtype. float32 matches ``batched_logits_to_logprobs``.

    Returns:
        ``[num_seqs, max_generated_tokens]`` tensor, padded with :data:`PAD_FILL`.

    Raises:
        ValueError: if the inputs disagree with ``num_seqs``, or if any row would need
            the fallback (call ``rows_needing_fallback`` first).
    """
    if len(completions) != num_seqs:
        raise ValueError(
            f"expected {num_seqs} completions, got {len(completions)} -- a short "
            f"batch here would silently train on zero-filled pi_old rows"
        )
    if len(logprobs) != num_seqs:
        raise ValueError(
            f"expected {num_seqs} logprob rows, got {len(logprobs)}"
        )
    bad = rows_needing_fallback(logprobs, completions, max_generated_tokens)
    if bad:
        raise ValueError(
            f"{len(bad)} row(s) cannot supply behavior logprobs (first few: "
            f"{bad[:8]}); call rows_needing_fallback() and take the policy-forward "
            f"path for the WHOLE batch rather than mixing sources"
        )

    out = torch.full(
        (num_seqs, max_generated_tokens), PAD_FILL, dtype=dtype, device=device
    )
    for i, (lp, comp) in enumerate(zip(logprobs, completions)):
        used = min(len(comp), max_generated_tokens)
        if used:
            out[i, :used] = torch.tensor(
                list(lp[:used]), dtype=dtype, device=device
            )
    return out

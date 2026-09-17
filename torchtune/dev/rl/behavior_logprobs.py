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
    "fit_behavior_logprobs_width",
    "broadcast_behavior_logprobs",
    "audit_behavior_logprobs",
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


def fit_behavior_logprobs_width(
    behavior: torch.Tensor,
    width: int,
) -> torch.Tensor:
    """Narrow a ``[num_seqs, cap]`` behavior tensor to the batch's ACTUAL response width.

    WHY THIS EXISTS (a real crash, not a defensive nicety). ``build_behavior_logprobs``
    returns the width it is told, which is ``max_generated_tokens`` -- the *cap*. The
    trainer's ``responses`` tensor is instead the *actual* longest generation in the
    batch, which equals the cap only when some row ran to the limit. On job 8833972 steps
    0 and 1 both had such a row (width 3072) and looked fine; step 2's longest generation
    was 1743 and the run died in ``logprobs.masked_fill_(response_padding_masks, 1.0)``
    with "expanded size of the tensor (3072) must match the existing size (1743)".

    The bug survived the audit because ``TORCHTUNE_BLP_AUDIT`` forces the recompute
    branch: disabling the shortcut in order to measure it also disabled its plumbing, so
    the audited steps exercised numerics the production path never reached. Hence the
    loud error below rather than a silent reshape -- a width that is *larger* than the
    cap means the caller's assumptions are wrong, and quietly padding it would push the
    inconsistency downstream to where it is unreadable.

    Args:
        behavior: ``[num_seqs, cap]`` tensor from :func:`build_behavior_logprobs`.
        width: the trainer's actual response width (``responses.shape[1]``).

    Returns:
        ``behavior`` itself when the widths already agree, else a ``[num_seqs, width]``
        narrowed view. Trailing columns are dropped: they are :data:`PAD_FILL` sentinels
        beyond every row's generation, never real logprobs.

    Raises:
        ValueError: if ``width`` exceeds the tensor's width, which no correct caller can
            produce -- the trainer cannot generate past its own cap.
    """
    cap = behavior.shape[1]
    if width == cap:
        return behavior
    if width > cap:
        raise ValueError(
            f"trainer response width {width} exceeds the behavior-logprob width {cap}; "
            f"pi_old would be zero-padded on positions that carry real tokens. This "
            f"means max_generated_tokens disagrees with the generated batch -- fix the "
            f"caller rather than padding here"
        )
    return behavior[:, :width]


def audit_behavior_logprobs(
    behavior: torch.Tensor,
    recomputed: torch.Tensor,
    *,
    padding_mask: Optional[torch.Tensor] = None,
) -> dict:
    """Quantify how far vLLM's sampler logprobs sit from the trainer's own recompute.

    WHY AN AUDIT RATHER THAN A TEST. The CPU equivalence tests pin the *alignment*
    contract (no shift, correct packing) against synthetic inputs. They cannot speak to
    numerical agreement, because that depends on things only present on hardware: vLLM
    runs different kernels, a different TP sharding, and a different batch composition
    than the trainer's forward. Agreement is therefore an empirical question, and the
    failure mode is silent -- a systematically-biased ``pi_old`` still yields IS ratios
    near 1.0, still trains, and still produces a plausible loss curve. Nothing
    downstream distinguishes "ratios are 1.0 because the policy has not moved" from
    "ratios are 1.0 because both sides are wrong in the same direction".

    HOW TO READ THE RESULT. ``ratio_p99`` is the number that matters: it is
    ``exp(|dlogp|)``, the multiplicative error this substitution injects into the
    importance weight of a typical worst-case token. For scale, the known-acceptable
    recompute noise between two trainer forwards on a healthy run reached 1.0739, and
    the unscaled-``raw_logprobs`` bug this feature guards against sits at 4.09x. A p99
    near the former is the substitution working; anything approaching the latter means
    the mode flag did not reach the server.

    Args:
        behavior: ``[num_seqs, C]`` logprobs from the vLLM sampler.
        recomputed: ``[num_seqs, C]`` logprobs from the trainer's policy forward.
        padding_mask: optional ``[num_seqs, C]`` bool, True at positions to EXCLUDE.
            Padded positions carry :data:`PAD_FILL` sentinels in one tensor and real
            (arbitrary) values in the other, so including them would dominate the
            statistics with a difference that means nothing.

    Returns:
        dict with ``n_compared``, ``max_abs``, ``mean_abs``, ``p99_abs``,
        ``ratio_max``, ``ratio_p99``, and ``bias`` (signed mean, positive when vLLM
        reports the token as MORE likely than the trainer does -- a bias term is what
        separates a systematic offset from symmetric kernel noise).

    Raises:
        ValueError: if the two tensors disagree in shape.
    """
    if behavior.shape != recomputed.shape:
        raise ValueError(
            f"shape mismatch: behavior {tuple(behavior.shape)} vs recomputed "
            f"{tuple(recomputed.shape)} -- these must be the same [num_seqs, C] grid"
        )

    b = behavior.detach().to(torch.float32).flatten()
    r = recomputed.detach().to(torch.float32).flatten()

    keep = torch.isfinite(b) & torch.isfinite(r)
    if padding_mask is not None:
        keep &= ~padding_mask.detach().flatten().bool()

    n = int(keep.sum().item())
    if n == 0:
        # Not an error: a fully-padded or fully-fallback batch legitimately has nothing
        # to compare. Report it as such rather than emitting NaN statistics that would
        # read as a catastrophic disagreement.
        return {
            "n_compared": 0,
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "p99_abs": 0.0,
            "ratio_max": 1.0,
            "ratio_p99": 1.0,
            "bias": 0.0,
        }

    diff = b[keep] - r[keep]
    absd = diff.abs()
    p99 = torch.quantile(absd, 0.99).item() if n >= 100 else absd.max().item()

    return {
        "n_compared": n,
        "max_abs": absd.max().item(),
        "mean_abs": absd.mean().item(),
        "p99_abs": p99,
        "ratio_max": float(torch.exp(absd.max()).item()),
        "ratio_p99": float(torch.exp(torch.tensor(p99)).item()),
        "bias": diff.mean().item(),
    }


# Sentinel row prepended to the broadcast payload. Row 0 column 0 carries 1.0 when the
# leader HAS usable behavior logprobs and 0.0 when the whole batch must fall back. It
# travels INSIDE the broadcast tensor rather than as a separate collective because a
# second collective is a second place the ranks can disagree about whether to call it.
_HAVE_ROW = 1


def broadcast_behavior_logprobs(
    behavior_logprobs: Optional[torch.Tensor],
    *,
    num_seqs: int,
    max_generated_tokens: int,
    is_leader: bool,
    broadcast_fn,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Optional[torch.Tensor]:
    """Give every training rank the leader's behavior logprobs, or give them all ``None``.

    THE HAZARD THIS EXISTS FOR. Only the shard leader issues the vLLM HTTP request, so
    only the leader can build ``pi_old`` from the response -- exactly the provenance of
    ``query_responses``, which is why that tensor is broadcast
    (``_broadcast_query_responses``). If the logprobs are NOT broadcast, followers see
    ``None`` and take the policy-forward branch while the leader skips it. That is not a
    crash and not a wrong number: it is **divergent collective participation**, i.e. a
    hang or a wrong-shaped all-reduce, which at 2N costs an allocation to diagnose.

    The fallback decision must also be identical on every rank, so it is carried in the
    payload rather than decided locally. A rank that decided for itself could disagree
    with the leader about whether a batch was usable and reintroduce the same divergence.

    Args:
        behavior_logprobs: the leader's ``[num_seqs, max_generated_tokens]`` tensor, or
            ``None`` if any row needed the fallback. Ignored on followers.
        num_seqs: rows the trajectory expects (``B*G``).
        max_generated_tokens: response-length cap; the tensor width.
        is_leader: whether this rank produced the values.
        broadcast_fn: callable taking the payload tensor and broadcasting it in place
            from the leader. The caller supplies this so the two branches of
            ``_broadcast_query_responses`` (node-local gloo CPU bounce under HSDP,
            world/``_training_pg`` otherwise) are mirrored exactly -- this function must
            not re-derive that routing.
        device: device for the payload buffer.
        dtype: payload dtype; matches the tensor being carried.

    Returns:
        The logprobs tensor on every rank, or ``None`` on every rank.

    Raises:
        ValueError: if the leader passes a tensor of the wrong shape (a follower would
            allocate the declared shape and silently mis-parse a different one).
    """
    if is_leader and behavior_logprobs is not None:
        expected = (num_seqs, max_generated_tokens)
        if tuple(behavior_logprobs.shape) != expected:
            raise ValueError(
                f"leader's behavior logprobs have shape "
                f"{tuple(behavior_logprobs.shape)}, expected {expected}; followers "
                f"pre-allocate the expected shape, so a mismatch here is a silent "
                f"misparse on every other rank"
            )

    payload = torch.zeros(
        (num_seqs + _HAVE_ROW, max_generated_tokens), dtype=dtype, device=device
    )
    if is_leader and behavior_logprobs is not None:
        payload[0, 0] = 1.0
        payload[_HAVE_ROW:] = behavior_logprobs.to(device=payload.device, dtype=dtype)

    payload = broadcast_fn(payload)

    if float(payload[0, 0]) != 1.0:
        return None
    return payload[_HAVE_ROW:].clone()

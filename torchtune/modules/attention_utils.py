# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from collections import OrderedDict
from typing import Callable, Optional, Union

import torch

from torch import nn
from torchtune.utils._import_guard import _SUPPORTS_FLEX_ATTENTION
from torchtune.utils._logging import get_logger, log_once

# Opt-in IPEX varlen_attention path (XPU only).
# Set TORCHTUNE_USE_IPEX_VARLEN=1 to route SDPA calls through
# intel_extension_for_pytorch.llm.functional.varlen_attention. Benchmarked
# (BioReason 4B shapes B=8 S=1536, Qwen3-4B GQA 32q/8kv head_dim=128):
#   - PyTorch SDPA optimized: 130.7 ms/36-layer, peak 1057 MiB
#   - IPEX varlen persistent buf: 103.3 ms/36-layer, peak 384 MiB (resv +0)
# Constraints: mask must be None or causal-only (no arbitrary boolean masks);
# dropout must be 0 (no IPEX dropout support); inputs must already be GQA-expanded.
_USE_IPEX_VARLEN = os.environ.get("TORCHTUNE_USE_IPEX_VARLEN", "0") == "1"
_ipex_varlen_attention = None
if _USE_IPEX_VARLEN:
    try:
        from intel_extension_for_pytorch.llm.functional import (
            varlen_attention as _ipex_varlen_attention,
        )
    except ImportError:
        _USE_IPEX_VARLEN = False

# Opt-in compiled flex_attention training path (XPU only).
# Set TORCHTUNE_USE_XPU_FLEX=1 to route causal SDPA calls through a compiled
# torch.nn.attention.flex_attention on XPU. UNLIKE the IPEX varlen path above,
# flex has an autograd kernel, so this engages on the TRAINING (grad-enabled)
# forward — it removes the O(S^2) [B,H,S,S] fp32 score materialization that the
# XPU math-SDPA backend does (validated seq6144: 1.13 GiB vs 5.6 GiB math S^2),
# which is what forces batch_size=1 on 32B (bs>1 OOMs in math SDPA). Numerically
# equivalent to math SDPA within the bf16 noise floor (fwd max|d| 0.0156 < bf16
# floor 0.0307; grad max|d| dq/dk/dv 0.016-0.06).
#
# NOTE: deliberately independent of _SUPPORTS_FLEX_ATTENTION (which is CUDA-only)
# — widening that guard would activate the existing no-kernel_options compiled
# flex (which fails an XPU Triton static_assert) and flip packed_block_causal_mask
# onto hardcoded device="cuda". This path lives entirely inside _sdpa_call.
#
# The compiled flex ONLY compiles on the XPU Triton backend when kernel_options
# force EQUAL backward block sizes (BLOCK_M1=N1=M2=N2); the default config fails
# `static_assert(BLOCK_M2 % BLOCK_N2 == 0)`. Block size is env-tunable for a PVC
# sweep (correctness is block-independent; step-time is not).
# Opt-in native SYCL-TLA flash-attention training path (XPU only).
# Set TORCHTUNE_USE_XPU_FLASH=1 to route causal SDPA calls through the compiled
# SYCL-TLA fused flash kernel that already ships in frameworks/2025.3.1
# (libtorch-xpu-ops-sycltla-mha_{fwd,bwd}.so). UNLIKE the XPU_FLEX path, this is
# the VENDOR fused kernel with a native BACKWARD — no Triton, no compile. It has
# BOTH passes for bf16, head_dim {64,96,128,192,256}, causal (validated GATE 0,
# job 8659275): peak fwd 1.07 / bwd 1.50 GiB vs math 5.92 / 8.24 at [1,40,4096,128]
# (~5.5x memory cut), which is what removes the O(S^2) [B,H,S,S] fp32 score
# materialization that forces batch_size=1 on 32B.
#
# TWO requirements empirically established (GATE 0):
#   1. Must FORCE the FLASH backend: torch's XPU auto-dispatch (_fused_sdp_choice)
#      never selects flash even when it is viable — it falls to math. We wrap the
#      call in torch.nn.attention.sdpa_kernel([SDPBackend.FLASH_ATTENTION]).
#   2. q/k/v must be in BSHD MEMORY layout (shape [B,H,S,D] with the strides of a
#      transpose(1,2) over a contiguous [B,S,H,D]). Standard C-contiguous [B,H,S,D]
#      is rejected with "No available kernel". We coerce with
#      .transpose(1,2).contiguous().transpose(1,2) (no-op cost for the already-
#      transposed q; one small copy for GQA-flattened k/v).
# Guards: XPU only, mask=None, is_causal=True, dropout==0 (the fused kernel rejects
# dropout>0 and any attn_mask). Numerically equal to math within the bf16 floor.
_USE_XPU_FLASH = os.environ.get("TORCHTUNE_USE_XPU_FLASH", "0") == "1"
_xpu_flash_sdpa_kernel = None
_xpu_flash_backend = None
if _USE_XPU_FLASH:
    try:
        from torch.nn.attention import (
            sdpa_kernel as _xpu_flash_sdpa_kernel,
            SDPBackend as _xpu_flash_SDPBackend,
        )

        _xpu_flash_backend = _xpu_flash_SDPBackend.FLASH_ATTENTION
    except ImportError:
        _USE_XPU_FLASH = False

_USE_XPU_FLEX = os.environ.get("TORCHTUNE_USE_XPU_FLEX", "0") == "1"
_XPU_FLEX_BLOCK = int(os.environ.get("TORCHTUNE_XPU_FLEX_BLOCK", "64"))
_XPU_FLEX_MASK_CACHE_MAX = max(
    1, int(os.environ.get("TORCHTUNE_XPU_FLEX_MASK_CACHE_MAX", "4"))
)
_xpu_flex_attention = None
_xpu_flex_create_block_mask = None
_xpu_flex_BlockMask = None
if _USE_XPU_FLEX:
    try:
        from torch.nn.attention.flex_attention import (
            BlockMask as _xpu_flex_BlockMask,
            create_block_mask as _xpu_flex_create_block_mask,
            flex_attention as _xpu_flex_attention,
        )
    except ImportError:
        _USE_XPU_FLEX = False

_log: logging.Logger = get_logger()

# One-shot per-worker log: surfaces whether IPEX varlen actually engaged on
# the first SDPA call. Dense Qwen3 GRPO passes an explicit causal mask, so
# TORCHTUNE_USE_IPEX_VARLEN=1 will report "requested-but-skipped (mask is not
# None)" on that path even though the env var is set.
_VARLEN_LOG_DONE: bool = False


def _log_varlen_status_once(mask, is_causal: bool, dropout_p: float, device_type: str) -> None:
    global _VARLEN_LOG_DONE
    if _VARLEN_LOG_DONE:
        return
    _VARLEN_LOG_DONE = True
    if not _USE_IPEX_VARLEN:
        _log.info("varlen=disabled (TORCHTUNE_USE_IPEX_VARLEN unset)")
        return
    if _ipex_varlen_attention is None:
        _log.info("varlen=disabled (intel_extension_for_pytorch import failed)")
        return
    reasons = []
    if mask is not None:
        reasons.append("mask is not None")
    if not is_causal:
        reasons.append("is_causal=False")
    if dropout_p != 0.0:
        reasons.append(f"dropout_p={dropout_p}")
    if device_type != "xpu":
        reasons.append(f"device={device_type}")
    if reasons:
        _log.info("varlen=requested-but-skipped (%s)", ", ".join(reasons))
    elif torch.is_grad_enabled():
        # Training forward: varlen has no autograd kernel; falls back to SDPA.
        # No-grad paths (ref fwd, rollout logprobs) will use varlen.
        _log.info("varlen=no-grad-only (training fwd uses SDPA; ref/rollout use varlen)")
    else:
        _log.info("varlen=engaged")


def _reset_varlen_log_for_testing() -> None:
    """Reset the one-shot varlen log flag. Test use only."""
    global _VARLEN_LOG_DONE
    _VARLEN_LOG_DONE = False


# ── Compiled flex_attention (XPU training path) ──────────────────────────────
# Built once at import when TORCHTUNE_USE_XPU_FLEX=1. The compiled callable is
# wrapped in torch.compiler.disable(recursive=False) so it stays compiled even if
# an outer torch.compile(model) is active (nested compile is unsupported), mirror-
# ing compile_friendly_flex_attention. Equal-block kernel_options are baked in
# (required for the XPU Triton backend to compile the backward — see the flag doc).
# TORCHTUNE_XPU_FLEX_AUTOTUNE=1: let Inductor search block configs (mode=
# "max-autotune") instead of pinning our equal-block kernel_options. The pinned
# equal-block was chosen only because it COMPILES on the XPU backend; it is not
# perf-tuned for PVC (block-64 measured ~1.5x slower/opt-step than math SDPA).
# When autotune is on we do NOT pass kernel_options (autotune picks the blocks).
_XPU_FLEX_AUTOTUNE = os.environ.get("TORCHTUNE_XPU_FLEX_AUTOTUNE", "0") == "1"
_xpu_flex_compiled = None
if _USE_XPU_FLEX and _xpu_flex_attention is not None:
    _XPU_FLEX_KERNEL_OPTS = {
        "BLOCK_M1": _XPU_FLEX_BLOCK,
        "BLOCK_N1": _XPU_FLEX_BLOCK,
        "BLOCK_M2": _XPU_FLEX_BLOCK,
        "BLOCK_N2": _XPU_FLEX_BLOCK,
    }
    try:
        if _XPU_FLEX_AUTOTUNE:
            _xpu_flex_attention_compiled = torch.compile(
                _xpu_flex_attention, mode="max-autotune"
            )

            @torch.compiler.disable(recursive=False)
            def _xpu_flex_compiled(q, k, v, block_mask):  # noqa: F811
                # No kernel_options: Inductor autotunes the block config.
                return _xpu_flex_attention_compiled(q, k, v, block_mask=block_mask)

        else:
            _xpu_flex_attention_compiled = torch.compile(_xpu_flex_attention)

            @torch.compiler.disable(recursive=False)
            def _xpu_flex_compiled(q, k, v, block_mask):  # noqa: F811
                return _xpu_flex_attention_compiled(
                    q, k, v, block_mask=block_mask, kernel_options=_XPU_FLEX_KERNEL_OPTS
                )

    except Exception as e:  # pragma: no cover - hardware/toolchain dependent
        _log.info("XPU flex compile setup failed (%s); disabling.", str(e)[:120])
        _USE_XPU_FLEX = False

# FIFO-bounded causal BlockMask cache. BlockMask is broadcast over batch/head
# (B=None, H=None) so the mask depends only on the sequence length S; keying on
# (S, device) is correct. Bucketing keeps the key set finite (a handful of S).
_xpu_flex_mask_cache: "OrderedDict" = OrderedDict()
_XPU_FLEX_LOG_DONE: bool = False


def _reset_xpu_flex_log_for_testing() -> None:
    """Reset the one-shot XPU-flex log flag + mask cache. Test use only."""
    global _XPU_FLEX_LOG_DONE
    _XPU_FLEX_LOG_DONE = False
    _xpu_flex_mask_cache.clear()


def _log_xpu_flex_status_once(
    mask, is_causal: bool, dropout_p: float, device_type: str
) -> None:
    global _XPU_FLEX_LOG_DONE
    if _XPU_FLEX_LOG_DONE:
        return
    _XPU_FLEX_LOG_DONE = True
    if not _USE_XPU_FLEX:
        _log.info("xpu_flex=disabled (TORCHTUNE_USE_XPU_FLEX unset)")
        return
    if _xpu_flex_compiled is None:
        _log.info("xpu_flex=disabled (flex_attention import/compile failed)")
        return
    reasons = []
    if mask is not None:
        reasons.append("mask is not None")
    if not is_causal:
        reasons.append("is_causal=False")
    if dropout_p != 0.0:
        reasons.append(f"dropout_p={dropout_p}")
    if device_type != "xpu":
        reasons.append(f"device={device_type}")
    if reasons:
        _log.info("xpu_flex=requested-but-skipped (%s)", ", ".join(reasons))
    else:
        _log.info("xpu_flex=engaged (block=%d, grad=%s)", _XPU_FLEX_BLOCK, torch.is_grad_enabled())


_XPU_FLASH_LOG_DONE: bool = False


def _reset_xpu_flash_log_for_testing() -> None:
    """Reset the one-shot XPU-flash log flag. Test use only."""
    global _XPU_FLASH_LOG_DONE
    _XPU_FLASH_LOG_DONE = False


def _log_xpu_flash_status_once(
    mask, is_causal: bool, dropout_p: float, device_type: str
) -> None:
    global _XPU_FLASH_LOG_DONE
    if _XPU_FLASH_LOG_DONE:
        return
    _XPU_FLASH_LOG_DONE = True
    if not _USE_XPU_FLASH:
        _log.info("xpu_flash=disabled (TORCHTUNE_USE_XPU_FLASH unset)")
        return
    if _xpu_flash_sdpa_kernel is None:
        _log.info("xpu_flash=disabled (sdpa_kernel import failed)")
        return
    reasons = []
    if mask is not None:
        reasons.append("mask is not None")
    if not is_causal:
        reasons.append("is_causal=False")
    if dropout_p != 0.0:
        reasons.append(f"dropout_p={dropout_p}")
    if device_type != "xpu":
        reasons.append(f"device={device_type}")
    if reasons:
        _log.info("xpu_flash=requested-but-skipped (%s)", ", ".join(reasons))
    else:
        _log.info("xpu_flash=engaged (grad=%s)", torch.is_grad_enabled())


def _to_bshd_memory(t: torch.Tensor) -> torch.Tensor:
    """Coerce a [B, H, S, D] tensor to BSHD memory layout.

    The SYCL-TLA flash kernel requires q/k/v to be in BSHD memory (shape stays
    [B, H, S, D] but the underlying storage is that of a contiguous [B, S, H, D]
    transposed on dims 1<->2). Standard C-contiguous [B, H, S, D] is rejected with
    "No available kernel" (GATE 0, job 8659275).

    If ``t`` is already such a view (already-BSHD-contiguous), the round-trip is a
    cheap no-op-ish reallocation; for a GQA-flattened contiguous-BHSD k/v it is one
    copy of the (small) tensor. We detect the already-correct case to skip the copy.
    """
    # A tensor is already in BSHD memory iff its [B,S,H,D] view is contiguous.
    b, h, s, d = t.shape
    if t.transpose(1, 2).is_contiguous():
        return t
    return t.transpose(1, 2).contiguous().transpose(1, 2)


def _xpu_flash_call(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dropout_p: float
) -> torch.Tensor:
    """Native SYCL-TLA fused flash causal attention on XPU (fwd + bwd).

    q, k, v arrive as [B, H, S, D] (already GQA-expanded). We coerce each to BSHD
    memory and force the FLASH backend (auto-dispatch never picks it). Returns
    [B, H, S, D] to match the SDPA caller.
    """
    q = _to_bshd_memory(q)
    k = _to_bshd_memory(k)
    v = _to_bshd_memory(v)
    with _xpu_flash_sdpa_kernel([_xpu_flash_backend]):
        return nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True
        )


def _xpu_flex_causal_mask(seq_len: int, device: torch.device):
    """Fetch/build a cached causal BlockMask for the given seq_len on device.

    B=None, H=None so the mask broadcasts over batch and heads; only S varies.
    FIFO-bounded so variable-seqlen workloads do not accumulate masks forever.
    """
    key = (seq_len, str(device))
    bm = _xpu_flex_mask_cache.get(key)
    if bm is None:
        def _causal(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        bm = _xpu_flex_create_block_mask(
            _causal, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device
        )
        _xpu_flex_mask_cache[key] = bm
        _xpu_flex_mask_cache.move_to_end(key)
        while len(_xpu_flex_mask_cache) > _XPU_FLEX_MASK_CACHE_MAX:
            _xpu_flex_mask_cache.popitem(last=False)
    return bm


def xpu_packed_block_causal_mask(
    seq_lens: list[list[int]],
    max_seq_len: int,
    device: torch.device,
):
    """Build a block-diagonal (document) causal BlockMask for token PACKING on XPU.

    This is the XPU analogue of :func:`packed_block_causal_mask` (which hardcodes
    ``device="cuda"`` and is gated on the CUDA-only ``_SUPPORTS_FLEX_ATTENTION``). Each row
    of ``seq_lens`` is that pack's per-document lengths (summing to ``max_seq_len``); a query
    attends a key iff they are in the SAME document AND causal (q_idx >= kv_idx). Returns a
    BlockMask consumable by the compiled flex kernel (the ``_sdpa_call`` BlockMask branch).

    Requires ``TORCHTUNE_USE_XPU_FLEX=1`` (the create_block_mask import). Raises otherwise.
    """
    if _xpu_flex_create_block_mask is None:
        raise RuntimeError(
            "xpu_packed_block_causal_mask requires TORCHTUNE_USE_XPU_FLEX=1 (flex import)."
        )
    B = len(seq_lens)
    # document_ids[b, pos] = which document position `pos` belongs to in pack b.
    document_ids = torch.zeros((B, max_seq_len), dtype=torch.int32, device=device)
    for b, lens in enumerate(seq_lens):
        off = 0
        for doc_id, L in enumerate(lens):
            end = min(off + int(L), max_seq_len)
            if end > off:
                document_ids[b, off:end] = doc_id
            off = end
            if off >= max_seq_len:
                break

    def _block_causal(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (document_ids[b, q_idx] == document_ids[b, kv_idx])

    # B set (per-pack document layout differs); H broadcasts.
    return _xpu_flex_create_block_mask(
        _block_causal, B=B, H=None, Q_LEN=max_seq_len, KV_LEN=max_seq_len, device=device
    )


def _xpu_flex_call(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Compiled flex_attention causal call on XPU.

    q, k, v arrive as [B, H, S, D] already GQA-expanded (flex's expected layout),
    so no repacking is needed. Returns [B, H, S, D] to match the SDPA caller.
    """
    seq_len = q.shape[2]
    block_mask = _xpu_flex_causal_mask(seq_len, q.device)
    return _xpu_flex_compiled(q, k, v, block_mask)


def _compute_maskfree_causal(
    env_set: bool,
    device_type: str,
    packing_enabled: bool,
    query_responses: torch.Tensor,
    context_length: int,
    pad_id: int,
) -> "tuple[bool, Optional[str]]":
    """Returns (use_maskfree, skip_reason_or_None).

    Evaluates whether the maskfree causal forward path should be used for the
    current batch (TORCHTUNE_MASKFREE_CAUSAL guard logic). Safe only when XPU,
    no packing, and no prompt-side padding (right-padded responses are fine
    because causal attention cannot see later positions).
    """
    if not env_set or device_type != "xpu" or packing_enabled:
        reason = (
            "env not set" if not env_set
            else "device != xpu" if device_type != "xpu"
            else "packing enabled"
        )
        return False, reason
    has_prompt_pad = (query_responses[:, :context_length] == pad_id).any().item()
    if has_prompt_pad:
        return False, "prompt padding detected"
    return True, None


if _SUPPORTS_FLEX_ATTENTION:
    from torch.nn.attention.flex_attention import (
        BlockMask,
        create_block_mask as create_block_causal_mask_flex,
        flex_attention,
    )

    def compile_flex_attention():
        try:
            return torch.compile(flex_attention)
        except Exception as e:
            # It may fail on some combinations of hardware/versions. Using max-autotune fixes this issue.
            # Context: https://github.com/pytorch/torchtune/issues/2113
            _log.info(
                f"Compiling flex_attention failed with error '{e}'. Retrying with mode='max-autotune'."
            )
            try:
                return torch.compile(flex_attention, mode="max-autotune")
            except Exception as e:
                _log.info(
                    f"Compiling flex_attention failed with error: '{e}', "
                    "Updating your pytorch version to nightlies may solve it, or you can set"
                    "in your config dataset.packed=False to avoid using flex attention."
                )
                raise

    flex_attention_compiled = compile_flex_attention()

    # We cannot do nested compile, but flex attention only has perf benefits
    # when compiled. To insulate it from the compiler, we wrap it with
    # compiler.disable so that it can be used regardless of whether the model
    # is compiled or not, and flex attention always remains compiled.
    @torch.compiler.disable(recursive=False)
    def compile_friendly_flex_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_mask: BlockMask,
    ) -> torch.Tensor:
        return flex_attention_compiled(q, k, v, block_mask=block_mask)

    _MaskType = Union[torch.Tensor, BlockMask]
else:
    _MaskType = torch.Tensor


def _get_document_ids_from_seq_lens(
    seq_lens: list[torch.Tensor],
) -> torch.Tensor:
    """
    Convert a batch tensor of seq lens into integer IDs denoting sample ownership.
    For example, seq_lens = [2, 3, 1] would return [0, 0, 1, 1, 1, 2].

    Args:
        seq_lens (list[torch.Tensor]): Sequence lengths of samples in each pack in the batch,
            shape (batch_size, n), where n is the max number of sequences in a pack and can vary
            across packs.

    Returns:
        Tensor: Document IDs of shape (batch_size, max_seq_len).
    """
    batch_size = len(seq_lens)
    batch_document_ids = []
    for sample_idx in range(batch_size):
        # We assume seq lens sum to max seq lens, so document_ids should be of
        # shape (max_seq_len, )
        document_ids = torch.cat(
            [
                torch.full((seq_len,), i, dtype=torch.long, device=seq_len.device)
                for i, seq_len in enumerate(seq_lens[sample_idx])
            ]
        )
        batch_document_ids.append(document_ids)
    batch_document_ids = torch.stack(batch_document_ids)
    return batch_document_ids


def create_block_causal_mask(seq_lens: list[torch.Tensor]) -> torch.Tensor:
    """
    Given a batch tensor of seq lens defining the lengths of samples in each pack,
    Construct a 2D block causal mask for each pack in the batch. For example, if
    a single sample's seq_lens is [3, 2, 1], the mask would be::

        mask = [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]

    Args:
        seq_lens (list[torch.Tensor]): Sequence lengths of samples in each pack in the batch,
            shape (batch_size, n), where n is the max number of sequences in a pack and can vary
            across packs.


    Returns:
        Tensor: Block causal mask of shape (batch_size, max_seq_len, max_seq_len).
    """
    batch_block_attn_masks = []
    batch_size = len(seq_lens)
    for sample_idx in range(batch_size):
        block_attn_masks = [
            torch.tril(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=seq_len.device)
            )
            for i, seq_len in enumerate(seq_lens[sample_idx])
        ]

        batch_block_attn_masks.append(torch.block_diag(*block_attn_masks))
    return torch.stack(batch_block_attn_masks)


def packed_block_causal_mask(
    seq_lens: list[torch.Tensor],
) -> _MaskType:
    """
    Create a block causal document mask for a batch of packed sequences. If
    flex attention is supported by the current hardware, block causal logic and
    passing this into :func:`torch.nn.attention.flex_attention.create_block_mask`.
    The resultant BlockMask is a compressed representation of the full block causal
    mask. If on an older version, a standard 2D block causal mask is created and returned.

    Args:
        seq_lens (list[torch.Tensor]): Sequence lengths of samples in each pack in the batch,
            shape (batch_size, n), where n is the max number of sequences in a pack and can vary
            across packs.

    Returns:
        _MaskType: BlockMask or Tensor if torch version < 2.5.0.
    """
    if _SUPPORTS_FLEX_ATTENTION:
        document_ids = _get_document_ids_from_seq_lens(seq_lens)
        batch_size, max_seq_len = document_ids.shape
        document_ids = document_ids.to("cuda")

        # Instead of passing a tensor mask, flex attention requires a mask_mod function
        # that determines which elements of QK^T should be included in the attention
        # computation prior to the softmax. For sample packing, we need both the
        # logic for both causal mask and document mask. See PyTorch's official
        # blog post for more details: https://pytorch.org/blog/flexattention/#mask-mods
        def mask_mod(b, h, q_idx, kv_idx):
            """
            Defines the logic of a block causal mask by combining both a standard causal mask
            and a block diagonal document mask.

            See :func:`~torchtune.modules.attention_utils.create_block_causal_mask`
            for an illustration.
            """
            causal_mask = q_idx >= kv_idx
            document_mask = document_ids[b, q_idx] == document_ids[b, kv_idx]
            return causal_mask & document_mask

        return create_block_causal_mask_flex(
            mask_mod,
            batch_size,
            None,
            max_seq_len,
            max_seq_len,
            device="cuda",
        )
    else:
        return create_block_causal_mask(seq_lens=seq_lens)


def _sdpa_or_flex_attention() -> Callable:
    """
    Helper function to decide when to call flex attention or SDPA. It will use
    flex attention if ALL of the following conditions are met, otherwise it will
    default to SDPA:
    - torch version >= 2.5.0
    - we are sample packing, therefore mask is a BlockMask
    - torch.cuda.get_device_capability() >= (7, 5)
    """

    # Persistent per-shape output buffer cache for IPEX varlen path.
    # Reusing the output tensor across calls means 0 allocator delta per attention
    # call (validated 2026-04-30 micro-bench), which is the goal of this path.
    #
    # BOUNDED (2026-06-24): the caches are keyed by (b, h, s, d, dtype, device).
    # On a FIXED-seqlen workload (the original microbench / SFT) only one key ever
    # appears, so the original unbounded dict was fine.  On VARIABLE-seqlen RL
    # (GSM8K rollouts: prompt+completion length differs nearly every step) `s`
    # changes per step, so a brand-new ~14.7 MiB output buffer (shape [b*s, h, d],
    # bf16) was cached EVERY step and NEVER evicted — a ~0.44 GiB/step live-memory
    # leak in the no-grad ref forward (root-caused via the no-FSDP colocate creep
    # bisect, jobs 8558600/8558618/8558640).  empty_cache cannot reclaim it (the
    # buffers are live, referenced by these module-level dicts).
    #
    # Fix: cap each cache with simple FIFO eviction.
    #
    # DEFAULT CAP = 1 (corrected 2026-06-25). The original cap=8 was set against a
    # mistaken ~14.7 MiB-per-shape / ~0.44 GiB/step estimate. A leak census on the
    # 4B-LoRA-2N server soak (job 8561648) measured the TRUE per-generation
    # footprint at ~2.5 GiB: each ref-forward generation retains ~36 distinct
    # (total_tokens, n_heads, head_dim) bf16 buffers (≈69 MiB each), NOT one
    # 14.7 MiB buffer. At cap=8 that is up to ~20 GiB of seqlen-keyed generations
    # held on the tile — which OOMs (banned:1 PDE) around step 6 on a 64 GiB tile.
    # cap=1 keeps only the CURRENT generation (flat ~2.5 GiB working set, validated
    # bit-flat 1.43→4.28→4.46→4.25→4.45→4.35→4.33 GiB across steps 0-6 vs the
    # baseline staircase 3.97→7.01→9.46→12.11→14.67→CRASH). On variable-seqlen RL
    # (GSM8K) `s` changes every step so cross-step reuse is ~nil anyway; on
    # FIXED-seqlen workloads raise the cap via TORCHTUNE_VARLEN_CACHE_MAX to recover
    # consecutive-same-shape reuse. See docs/reports/colocate_pagefault_investigation_20260625.md.
    _varlen_out_cache: "OrderedDict" = OrderedDict()
    _varlen_alibi_cache: dict = {}
    _varlen_seqlens_cache: dict = {}
    _varlen_census_seen: set = set()  # diag: keys already logged (TORCHTUNE_LEAK_CENSUS)
    _varlen_cache_max = max(1, int(os.environ.get("TORCHTUNE_VARLEN_CACHE_MAX", "1")))

    def _varlen_cache_evict(key: tuple) -> None:
        # FIFO: keep the cap most-recently-INSERTED keys. Evict the same key from
        # all three caches together (they share the cache_key) so they never drift.
        _varlen_out_cache.move_to_end(key)
        while len(_varlen_out_cache) > _varlen_cache_max:
            old, _ = _varlen_out_cache.popitem(last=False)
            _varlen_alibi_cache.pop(old, None)
            _varlen_seqlens_cache.pop(old, None)

    def _ipex_varlen_call(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        # q,k,v come in as [B, H, S, D] (already GQA-expanded by attention.py).
        # varlen wants [total_tokens, H, D] packed.
        b, h, s, d = q.shape
        # transpose to [B, S, H, D] then flatten batch+seq
        q_packed = q.transpose(1, 2).contiguous().view(b * s, h, d)
        k_packed = k.transpose(1, 2).contiguous().view(b * s, h, d)
        v_packed = v.transpose(1, 2).contiguous().view(b * s, h, d)

        cache_key = (b, h, s, d, q.dtype, str(q.device))
        # Under autograd (training fwd), allocate fresh to avoid version-counter
        # conflicts across chunks in the chunked grpo_step backward loop.
        # No-grad paths (ref fwd, rollout logprobs) reuse the persistent buffer.
        # NOTE: this call is only reached on no-grad paths (the SDPA caller gates it
        # with `not torch.is_grad_enabled()`), but keep the grad-enabled fresh-alloc
        # branch as defensive code in case the call site changes.
        if torch.is_grad_enabled():
            out = torch.empty_like(q_packed)
            alibi = torch.zeros(h, dtype=torch.float32, device=q.device)
            seqlens = torch.arange(0, b * s + 1, s, dtype=torch.int32, device=q.device)
        else:
            out = _varlen_out_cache.get(cache_key)
            if out is None or out.shape != q_packed.shape:
                out = torch.empty_like(q_packed)
                _varlen_out_cache[cache_key] = out
            alibi = _varlen_alibi_cache.get(cache_key)
            if alibi is None:
                alibi = torch.zeros(h, dtype=torch.float32, device=q.device)
                _varlen_alibi_cache[cache_key] = alibi
            seqlens = _varlen_seqlens_cache.get(cache_key)
            if seqlens is None:
                seqlens = torch.arange(0, b * s + 1, s, dtype=torch.int32, device=q.device)
                _varlen_seqlens_cache[cache_key] = seqlens
            # Bound all three caches together (FIFO) so variable-seqlen RL does not
            # accumulate one ~14.7 MiB buffer per distinct (b,s) forever.
            _varlen_cache_evict(cache_key)
            # Diagnostic (TORCHTUNE_LEAK_CENSUS=1): expose the true cache occupancy
            # + per-buffer bytes so the (b,h,s,d)-keyed dict size is unambiguous —
            # resolves whether the per-step (N,32,128)x36 retention is IN this cache
            # (1 buffer/key) or aliased downstream. One log per distinct key.
            if os.environ.get("TORCHTUNE_LEAK_CENSUS", "0") == "1":
                _nb = out.numel() * out.element_size()
                if cache_key not in _varlen_census_seen:
                    _varlen_census_seen.add(cache_key)
                    _log.info(
                        "VARLEN_CACHE diag: entries=%d cap=%d this_key=%s buf=%.1f MiB",
                        len(_varlen_out_cache), _varlen_cache_max,
                        (b, h, s, d), _nb / 1024**2,
                    )

        softmax_scale = 1.0 / (d ** 0.5)
        _ipex_varlen_attention(
            q_packed, k_packed, v_packed, out,
            seqlens, seqlens,
            alibi,
            s, s,
            0.0, softmax_scale,
            False, True, False, None,
        )
        # Return [B, H, S, D] to match SDPA caller expectation (which transposes back).
        return out.view(b, s, h, d).transpose(1, 2)

    # Create SDPA Call
    def _sdpa_call(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[_MaskType],
        dropout_p: float,
        is_causal: bool,
    ) -> torch.Tensor:
        # XPU token-PACKING branch (highest precedence): a BlockMask means sample packing
        # with a block-diagonal (document) mask — doc A must not attend doc B. The flash
        # and varlen branches below are causal-ONLY (mask=None) and cannot express this, so
        # packing MUST route through compiled flex (which takes an arbitrary BlockMask and
        # has an autograd kernel). Gated on XPU + flex-available; the flash-causal path
        # (mask=None) is unaffected. See BioReasonPackedSFTDataset / packing scope memo.
        if (
            _USE_XPU_FLEX
            and _xpu_flex_compiled is not None
            and _xpu_flex_BlockMask is not None
            and isinstance(mask, _xpu_flex_BlockMask)
            and dropout_p == 0.0
            and q.device.type == "xpu"
        ):
            return _xpu_flex_compiled(q, k, v, mask)

        # Native SYCL-TLA fused flash branch (XPU, causal-only, no mask, no dropout).
        # This is the VENDOR fused kernel with a native backward — it removes the
        # O(S^2) score materialization that forces batch_size=1, at ~5.5x less peak
        # memory than math (GATE 0). Preferred over flex (Triton, ~1.5x slower) and
        # varlen (no autograd) when available. Takes precedence over both.
        _log_xpu_flash_status_once(mask, is_causal, dropout_p, q.device.type)
        if (
            _USE_XPU_FLASH
            and _xpu_flash_sdpa_kernel is not None
            and mask is None
            and is_causal
            and dropout_p == 0.0
            and q.device.type == "xpu"
        ):
            return _xpu_flash_call(q, k, v, dropout_p)

        # Compiled flex_attention branch (XPU, causal-only, no mask, no dropout).
        # UNLIKE varlen below, flex has an autograd kernel, so this engages on the
        # grad-enabled TRAINING forward — removing the O(S^2) score materialization
        # that forces batch_size=1. Takes precedence over varlen when both are set.
        _log_xpu_flex_status_once(mask, is_causal, dropout_p, q.device.type)
        if (
            _USE_XPU_FLEX
            and _xpu_flex_compiled is not None
            and mask is None
            and is_causal
            and dropout_p == 0.0
            and q.device.type == "xpu"
        ):
            return _xpu_flex_call(q, k, v)

        # IPEX varlen branch: only valid for causal-only, no mask, no dropout, on XPU,
        # and only for no-grad paths (ref fwd, rollout logprobs).
        # torch_ipex::varlen_fwd has no registered autograd kernel; running it under
        # torch.is_grad_enabled() uses PyTorch's autograd fallthrough which may produce
        # silently incorrect gradients and has been observed to trigger banned:1 PDE GPU
        # faults on Aurora (WS8.5, 2026-05-02). Training forward uses standard SDPA.
        _log_varlen_status_once(mask, is_causal, dropout_p, q.device.type)
        if (
            _USE_IPEX_VARLEN
            and _ipex_varlen_attention is not None
            and mask is None
            and is_causal
            and dropout_p == 0.0
            and q.device.type == "xpu"
            and not torch.is_grad_enabled()
        ):
            return _ipex_varlen_call(q, k, v)

        # shape: [b, 1, s, s]
        if mask is not None:
            mask = mask[:, None, :, :]

        # Flash attention from https://pytorch.org/blog/accelerating-large-language-models/
        return nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=dropout_p, is_causal=is_causal
        )

    if not _SUPPORTS_FLEX_ATTENTION:
        return _sdpa_call

    # Create Flex Attention Call
    def _attention_call(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[_MaskType],
        dropout_p: float,
        is_causal: bool,
    ) -> torch.Tensor:
        # Flex attention uses the BlockMask
        # (https://github.com/pytorch/pytorch/blob/main/torch/nn/attention/flex_attention.py#L168)
        # instead of a traditional boolean tensor mask. If this is passed in,
        # we assume the user wants to use flex attention instead of traditional SDPA.
        # This will use flash attention under the hood with support for custom masks.
        # Currently, it is used when sample packing is enabled (see torchtune.datasets.PackedDataset)
        if isinstance(mask, BlockMask):
            if not torch.compiler.is_compiling():
                log_once(
                    _log,
                    "Using flex attention for attention computation since a BlockMask was passed in.",
                    level=logging.DEBUG,
                )
            if dropout_p > 0.0:
                raise ValueError(
                    "Flex attention does not support dropout. Please set dropout to 0.0."
                )
            return compile_friendly_flex_attention(
                q,
                k,
                v,
                block_mask=mask,
            )
        else:
            # If mask is a standard boolean tensor or None, then use SDPA
            return _sdpa_call(q, k, v, mask, dropout_p, is_causal)

    return _attention_call


def kv_offset_mask_flex(b, h, q_idx, kv_idx, offset):
    """
    Mask mod for autoregressive generation to be used by flex attention. See https://pytorch.org/blog/flexattention/#mask-mods.

    This mask mod can be passed to :func:`~torch.nn.attention.flex_attention.create_block_mask` to create a BlockMask
    to generate a single token where all past tokens are unmasked.

    Example::
        >>> from torch.nn.attention.flex_attention import create_block_mask
        >>> current_token_idx, input_tokens, token_to_generate = 3, 5, 8
        >>> total_response_length = input_tokens + tokens_to_generate
        >>> create_block_mask(
        >>>     mask_mod=partial(kv_offset_mask_flex, offset=current_token_idx),
        >>>     B=1,
        >>>     H=None,
        >>>     Q_LEN=1,
        >>>     KV_LEN=total_response_length,
        >>> )
    """
    return kv_idx <= offset


def causal_mask_flex(b, h, q_idx, kv_idx):
    """
    Mask mod for a standard causal mask to be used by flex attention. See https://pytorch.org/blog/flexattention/#mask-mods.

    This mask mod can be passed to :func:`~torch.nn.attention.flex_attention.create_block_mask` to create a BlockMask
    equivalent of a causal mask.

    Example::
        >>> # Construct a causal mask for prefill stage of autoregressive generation
        >>> from torch.nn.attention.flex_attention import create_block_mask
        >>> bsz, input_tokens, token_to_generate = 2, 3, 5
        >>> total_response_length = input_tokens + tokens_to_generate
        >>> create_block_mask(
        >>>     mask_mod=causal_mask_flex
        >>>     B=bsz,
        >>>     H=None,
        >>>     Q_LEN=input_tokens,
        >>>     KV_LEN=total_response_length,
        >>> )

    """

    return q_idx >= kv_idx

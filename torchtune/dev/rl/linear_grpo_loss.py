# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor

from torchtune.modules.loss import RLLoss


class LinearGRPOLoss(nn.Module, RLLoss):
    """Memory efficient GRPO loss that incrementally computes loss for chunks of tokens
    by masking ignored tokens, calculating logits and then applying GRPO loss. Combines
    the linear projection with the GRPO calculation for futher memory savings.
    """

    def __init__(
        self,
        num_output_chunks: int = 8,
        epsilon: float = 0.1,
        kl_coeff: float = 0.1,
        ignore_index: int = -100,
        mask_pre_projection: bool = True,
        temperature: float = 1.0,
    ):
        super().__init__()
        """
        Args:
            num_output_chunks (int): Number of chunks to split the output tensor into. Default is 8.
            ignore_index (int): Index to ignore in the target tensor. Default is -100.
            mask_pre_projection (bool): Whether to mask the output tensor before projection, avoiding
                computing it for tokens that will be ignored during CE anyway. Default is True.
            temperature (float): Softmax temperature applied to logits before computing logprobs,
                matching the rollout/recipe temperature (rlhf.logits_to_logprobs divides logits by
                it). The logprob of token t is ``log_softmax(logits / temperature)[t]``, equal to
                ``-cross_entropy(logits / temperature, t)``. Default 1.0 (no scaling). MUST match
                the recipe's ``temperature`` or the policy logprobs are computed under the wrong
                distribution.
        """
        self.linear_projection = None
        # FSDP2 tied-embedding support. For a TiedLinear output (qwen2_5 / qwen3 with
        # tie_word_embeddings) the projection weight lives in model.tok_embeddings,
        # owned by the ROOT FSDP2 unit. set_model_output captures that root here so
        # forward() can unshard() it before the per-chunk projection (a no-op under
        # default AllGather prefetch where the root stays resident; the fix when the
        # root reshards, e.g. disable_prefetch=True). None for untied / no-FSDP.
        self._fsdp_root = None
        self.num_output_chunks = num_output_chunks
        self.epsilon = epsilon
        self.kl_coeff = kl_coeff
        self.ignore_index = ignore_index
        self.mask_pre_projection = mask_pre_projection
        self.temperature = temperature

    def apply_compile_strategy(self, *args, **kwargs):
        """Applies compile only to the compute_grpo_loss function.
        If compiling CE + chunking operation together, memory requirement is higher."""
        self.chunked_grpo_loss = torch.compile(self.chunked_grpo_loss, *args, **kwargs)
        return self

    def set_model_output(self, model: nn.Module) -> None:
        """Modify model output to match the expected input for the loss function.

        Two residency mechanisms, selected by whether the output projection is tied.
        HW-validated 2026-06-24 on 2-rank FSDP2 (Aurora XPU): the chunked-vocab
        projection + tied-grad are bit-exact to a single-process full-weight
        reference (``probe_tied_grad.py``: max_abs_diff=0.0).

        - TIED (``model.output`` is a ``TiedLinear``, not an ``nn.Module``): the
          projection weight physically lives in ``model.tok_embeddings``. Under
          torchtune's FSDP2 sharding, ``tok_embeddings`` is NOT its own unit — it
          stays in the **root** FSDP2 unit, which (with default AllGather prefetch,
          ``reshard_after_forward=None``) keeps its own params RESIDENT through the
          forward. So ``F.linear(hidden, tok_embeddings.weight)`` post-forward sees
          the FULL weight, and FSDP2's autograd accumulates the grad from BOTH the
          input-embedding use and this unembed use into the one tied param. We
          capture the root unit (``_fsdp_root``) and ``unshard()`` it once in
          ``forward`` — a no-op under default prefetch, and the correct fix if the
          root ever reshards (e.g. ``disable_prefetch=True``). We do NOT make
          ``tok_embeddings`` a separate ``custom_sharded_layers`` unit: that would
          reshard it mid-forward and break the in-forward tied unembed used by
          generation/ref forwards.

        - UNTIED (``model.output`` is a real ``nn.Linear``): capture it directly.
          Under FSDP it is its own ``fully_shard`` unit (via
          ``custom_sharded_layers=['output']``); because ``skip_output_layer=True``
          makes ``unembed`` skip it in forward, calling it post-forward fires FSDP2's
          pre-forward all-gather hook -> projects -> reshards. No root unshard needed.
        """
        # The loss may handle the output projection. If true, the model should skip it.
        model.skip_output_layer = True
        from torch.distributed.fsdp import FSDPModule
        from torchtune.modules.tied_linear import TiedLinear

        if isinstance(model.output, TiedLinear):
            tied_module = model.tok_embeddings
            # Closure reads tied_module.weight at call time (full weight post-forward).
            self.linear_projection = lambda x: F.linear(x, tied_module.weight)
            # Root FSDP2 unit owns tok_embeddings.weight; unshard() it in forward so
            # the projection sees the full weight even if the root resharded.
            self._fsdp_root = model if isinstance(model, FSDPModule) else None
        else:
            self.linear_projection = model.output
            # Untied: model.output is its own FSDP2 unit; calling it fires the
            # all-gather hook. No root unshard needed.
            self._fsdp_root = None

    def chunked_grpo_loss(
        self,
        hidden_chunk: torch.Tensor,  # [B*G, chunk_size, H]
        targets_chunk: torch.Tensor,  # [B*G, chunk_size]
        ref_logprobs_chunk: torch.Tensor,  # [B*G, chunk_size]
        advantages: torch.Tensor,  # [B*G]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logits_chunk = self.linear_projection(hidden_chunk)
        if isinstance(pi_logits_chunk, DTensor):
            pi_logits_chunk = pi_logits_chunk.full_tensor()
        if isinstance(targets_chunk, DTensor):
            targets_chunk = targets_chunk.full_tensor()
        if isinstance(ref_logprobs_chunk, DTensor):
            ref_logprobs_chunk = ref_logprobs_chunk.full_tensor()

        # CE -> token logprob = log_softmax(logits / T)[t] = -cross_entropy(logits / T, t).
        # Temperature must match the recipe/rollout temperature so policy logprobs come
        # from the same distribution the standard rlhf.logits_to_logprobs path uses.
        pi_logits_flat = pi_logits_chunk.reshape(-1, pi_logits_chunk.size(-1))
        targets_flat = targets_chunk.reshape(-1)
        pi_logits_flat = pi_logits_flat.float()
        if self.temperature != 1.0:
            pi_logits_flat = pi_logits_flat / self.temperature
        pi_logprobs_chunk = -F.cross_entropy(
            pi_logits_flat, targets_flat, reduction="none"
        )
        pi_logprobs_chunk = pi_logprobs_chunk.view_as(targets_chunk)

        # Detach
        pi_logprobs_detached = pi_logprobs_chunk.detach()
        ref_logprobs_detached = ref_logprobs_chunk.detach()

        # KL term — k3 estimator exp(d) - d - 1, d = ref - pi. Clamp to [-10, 10]
        # (exp(10)=22026) and scrub NaN: on long generations a rare token gets a
        # very negative pi logprob -> d ~ +80 -> exp(d) overflows to Inf -> the
        # reduction NaNs and poisons the whole step (mirrors GRPOSimpleLoss /
        # GRPOLoss hardening in loss.py:291-294; hit on BioReason 2048-gen step 4).
        # nan_to_num runs first because clamp alone does not filter inf-inf=nan.
        _kl_diff = ref_logprobs_detached - pi_logprobs_chunk
        _kl_diff = torch.nan_to_num(_kl_diff, nan=0.0, posinf=10.0, neginf=-10.0)
        _kl_diff = _kl_diff.clamp(min=-10.0, max=10.0)
        per_token_kl = torch.exp(_kl_diff) - _kl_diff - 1

        # Policy term
        per_token_policy_loss = (
            torch.exp(pi_logprobs_chunk - pi_logprobs_detached) * advantages[:, None]
        )

        # Total per-token loss
        per_token_loss = -(per_token_policy_loss - self.kl_coeff * per_token_kl)

        return per_token_loss, per_token_policy_loss, per_token_kl, pi_logprobs_chunk

    def forward(
        self,
        pi_old_outputs: torch.Tensor,
        pi_outputs: torch.Tensor,
        ref_outputs: torch.Tensor,
        advantages: torch.Tensor,
        padding_masks: Optional[torch.Tensor] = None,  # [B*G, response_length]
    ) -> tuple[torch.Tensor, ...]:
        """
        Args:
            pi_old_outputs (torch.Tensor): Hidden state of the model, pre projection. Shape ``[bsz, seq_len, emb_dim]``
            pi_outputs (torch.Tensor): Labels for the model. Shape ``[bsz, seq_len]``
            ref_outputs (torch.Tensor): Reference logprobs for KL loss. Shape ``[bsz, seq_len, vocab_size]``
            advantages (torch.Tensor): Advantages for KL loss. Shape ``[bsz, seq_len, vocab_size?]``
            padding_masks (Optional[torch.Tensor]): Mask for padding tokens. Shape ``[bsz, seq_len]``

        Returns:
            tuple[torch.Tensor, ...]: loss, policy_loss, kl_loss, ratios, clipfrac, pi_logprobs
        """
        # TIED-EMBEDDING FSDP2 residency: ensure the root unit (which owns the tied
        # tok_embeddings.weight) is unsharded before the chunk loop, so the closure
        # linear_projection reads the FULL weight per chunk. Under default AllGather
        # prefetch the root stays resident through forward, so this is a no-op
        # (HW-validated bit-exact); it is the correctness fix only if the root
        # resharded (disable_prefetch=True). We do NOT reshard() before backward —
        # FSDP2 auto-reshards in the post-backward hook, and the weight must stay
        # unsharded across all chunks' graphs until the single backward(). hasattr-
        # guarded so CPU / no-FSDP / untied (the equivalence tests set
        # linear_projection directly, bypassing set_model_output) is unaffected.
        if self._fsdp_root is not None and hasattr(self._fsdp_root, "unshard"):
            self._fsdp_root.unshard()

        # Chunk along sequence dimension
        hidden_chunks = pi_old_outputs.tensor_split(self.num_output_chunks, dim=1)
        target_chunks = pi_outputs.tensor_split(self.num_output_chunks, dim=1)
        ref_logprobs_chunks = ref_outputs.tensor_split(self.num_output_chunks, dim=1)

        # Default to all-ones mask if padding_masks is None
        if padding_masks is None:
            padding_masks = torch.ones_like(pi_outputs, dtype=torch.bool)
        padding_masks_chunks = padding_masks.tensor_split(self.num_output_chunks, dim=1)

        # Initialize accumulators
        batch_size = advantages.numel()
        device = pi_old_outputs.device

        total_loss_sum = torch.zeros(batch_size, device=device)
        total_policy_sum = torch.zeros(batch_size, device=device)
        total_kl_sum = torch.zeros(batch_size, device=device)
        total_token_count = torch.zeros(batch_size, device=device)
        pi_logprobs_list = []  # Collect pi_logprobs for each chunk

        # Process each chunk
        for chunk_idx in range(self.num_output_chunks):
            (
                per_token_loss_chunk,
                per_token_policy_loss_chunk,
                per_token_kl_chunk,
                pi_logprobs_chunk,
            ) = self.chunked_grpo_loss(
                hidden_chunks[chunk_idx],
                target_chunks[chunk_idx],
                ref_logprobs_chunks[chunk_idx],
                advantages,
            )

            # Accumulate with padding mask applied
            padding_masks_chunk = padding_masks_chunks[chunk_idx]
            total_loss_sum += (per_token_loss_chunk * padding_masks_chunk).sum(dim=1)
            with torch.no_grad():
                total_policy_sum += (
                    per_token_policy_loss_chunk * padding_masks_chunk
                ).sum(dim=1)
                total_kl_sum += (per_token_kl_chunk * padding_masks_chunk).sum(dim=1)
                total_token_count += padding_masks_chunk.sum(dim=1)

            # Store pi_logprobs for this chunk
            pi_logprobs_list.append(pi_logprobs_chunk)

        # Concatenate pi_logprobs across all chunks
        pi_logprobs = torch.cat(pi_logprobs_list, dim=1)  # [B*G, response_length]

        # Compute mean losses per sequence, then average over batch
        total_token_count = total_token_count.clamp(min=1e-9)
        loss = (total_loss_sum / total_token_count).mean()
        with torch.no_grad():
            policy_loss = (total_policy_sum / total_token_count).mean()
            kl_loss = (total_kl_sum / total_token_count).mean()

        # Dummy values for unused metrics
        ratios = torch.tensor(1.0, device=device)
        clipfrac = torch.tensor(0.0, device=device)

        return loss, policy_loss, kl_loss, ratios, clipfrac, pi_logprobs

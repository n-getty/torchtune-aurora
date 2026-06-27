"""
Native-torchtune BioReason multimodal model (Gemma 4 backbone).

Unlike :class:`torchtune.dev.bioreason.model.BioReasonModel` (which wraps an HF
``AutoModelForCausalLM`` consumed via ``inputs_embeds=``), this variant wraps the
**native torchtune** ``gemma4_31b()`` / ``lora_gemma4_31b()`` decoder — a
``TransformerDecoder`` — and feeds it ``input_embeds`` directly (transformer.py
short-circuits ``tok_embeddings`` when ``input_embeds`` is passed). This unlocks the
repo's validated FSDP / vLLM Gemma4 infra and removes the HF dependency for the
multimodal embed-injection.

★ Embedding-scale contract (the load-bearing subtlety):
Gemma's ``tok_embeddings`` is a :class:`GemmaNormEmbeddings`, whose ``forward``
multiplies the looked-up vectors by ``sqrt(embed_dim)`` (≈73.3 at H=5376). When we
feed ``input_embeds`` the decoder layers run on those vectors directly, so the text
and completion tokens MUST be embedded through ``tok_embeddings`` (which applies the
scale) — we therefore reuse ``self.backbone.tok_embeddings`` as ``self._embed`` rather
than loading a separate ``nn.Embedding``. The from-scratch protein/GO projections are
spliced in at placeholder positions and learn whatever magnitude the (scaled) regime
needs. This also makes the text-only equivalence test pass exactly:
``model(input_embeds=build(tokens)) == backbone(tokens=tokens)``.

The SFT splice (``build_full_embeds_train``) runs WITH grad enabled — the projections
must be trained through the splice. This is a deliberate divergence from the RL model's
``@torch.no_grad`` ``build_prompt_embeds`` (whose projections were trained elsewhere /
via a separate grad path).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import torch
import torch.nn as nn

from torchtune.models.gemma4 import gemma4_31b, lora_gemma4_31b

logger = logging.getLogger(__name__)


class BioReasonNativeModel(nn.Module):
    """Multimodal BioReason model on a native torchtune Gemma 4 decoder.

    Components:
      - Gemma 4 ``TransformerDecoder`` backbone (full-FT, or frozen base + LoRA when
        ``enable_lora=True``). Output is tied to ``tok_embeddings``.
      - ``protein_projection`` MLP  (trainable) — ESM3 features -> H
      - ``go_projection`` MLP       (trainable) — GO graph features (2560) -> H
      - ESM3 features come from a pre-encoded cache (no live ESM3 encoder built); GO
        features come from a cached ``go_embedding.pt``. Both frozen.

    Placeholder tokens are RESERVED Gemma vocab ids (passed in as ``protein_token_id`` /
    ``go_token_id``) — the native Gemma4 tokenizer has no add-special-tokens path, so the
    dataset splices these ids directly and this model fills those positions.

    Args:
        backbone_builder (str): "gemma4_31b" (full-FT) or "lora_gemma4_31b" (LoRA).
        device (torch.device): device to build on.
        dtype (torch.dtype): parameter dtype. Default: bfloat16.
        hidden_size (int): backbone embed_dim (5376 for 31B). Used to size projections.
        protein_token_id (int): reserved vocab id marking protein placeholder positions.
        go_token_id (int): reserved vocab id marking GO placeholder positions.
        esm3_cache_path (Optional[str]): path to the pre-encoded ESM3 cache (.pt) + .json
            sidecar. Required (no live ESM3 encoder is built here).
        protein_model_name (str): ESM3 model name; validated against the cache sidecar.
        go_embedding_path (Optional[str]): path to cached go_embedding.pt ([max_go, 2560]).
        proj_resume_dir (Optional[str]): dir with protein_projection.pt/go_projection.pt
            to resume trained projections (e.g. continuing a run).
        enable_lora (bool): build the LoRA backbone variant. Default: False.
        lora_rank (int): LoRA rank. Default: 32.
        lora_alpha (float): LoRA alpha. Default: 64.
        lora_dropout (float): LoRA dropout. Default: 0.0.
        lora_attn_modules (list[str]): attention projections to adapt.
        apply_lora_to_mlp (bool): adapt the MLP too. Default: True.
    """

    GO_DIM = 2560  # GO graph encoder output dim (fixed, matches go_embedding.pt)

    def __init__(
        self,
        *,
        device: torch.device,
        hidden_size: int,
        protein_token_id: int,
        go_token_id: int,
        backbone_builder: str = "gemma4_31b",
        dtype: torch.dtype = torch.bfloat16,
        esm3_cache_path: Optional[str] = None,
        protein_model_name: str = "esm3_sm_open_v1",
        go_embedding_path: Optional[str] = None,
        proj_resume_dir: Optional[str] = None,
        enable_lora: bool = False,
        lora_rank: int = 32,
        lora_alpha: float = 64.0,
        lora_dropout: float = 0.0,
        lora_attn_modules: Optional[list[str]] = None,
        apply_lora_to_mlp: bool = True,
        # Test-injection hooks (production passes paths/None for all three):
        backbone: Optional[nn.Module] = None,
        protein_hidden_override: Optional[int] = None,
        esm3_cache_inject: Optional[tuple] = None,
        go_embedding_inject: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.protein_token_id = protein_token_id
        self.go_token_id = go_token_id
        self._has_lora = bool(enable_lora)

        # ── Backbone (native Gemma 4 TransformerDecoder) ──────────────────────
        if backbone is not None:
            self.backbone = backbone
        elif enable_lora:
            if lora_attn_modules is None:
                lora_attn_modules = ["q_proj", "k_proj", "v_proj", "output_proj"]
            self.backbone = lora_gemma4_31b(
                lora_attn_modules=lora_attn_modules,
                apply_lora_to_mlp=apply_lora_to_mlp,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
        elif backbone_builder == "gemma4_31b":
            self.backbone = gemma4_31b()
        else:
            raise ValueError(f"Unknown backbone_builder: {backbone_builder!r}")

        # ★ Reuse tok_embeddings (GemmaNormEmbeddings — applies the sqrt(H) scale) so
        # text/completion embeds enter the layers at the same magnitude the decoder
        # expects. Do NOT load a separate nn.Embedding.
        self._embed = self.backbone.tok_embeddings

        # ── Trainable projection MLPs (fresh init; SFT trains them) ───────────
        if protein_hidden_override is not None:
            protein_hidden = protein_hidden_override
        else:
            protein_hidden = self._infer_protein_hidden(
                esm3_cache_path, protein_model_name
            )
        self._protein_hidden = protein_hidden
        # NOTE: do NOT call .to(device=...) here — under a meta-device build context
        # (the distributed recipe builds the wrapper on meta before FSDP2 sharding),
        # .to(real_device) on a meta param raises "Cannot copy out of meta tensor".
        # The recipe re-inits these projections (to_empty + reset_parameters) on device
        # after sharding. For the single-process/CPU path (tests), they land on the
        # ambient default device, which is correct. Set dtype via the constructor so it
        # holds in both regimes without a meta-incompatible copy.
        self.protein_projection = nn.Sequential(
            nn.Linear(protein_hidden, hidden_size, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, dtype=dtype),
        )
        self.go_projection = nn.Sequential(
            nn.Linear(self.GO_DIM, hidden_size, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, dtype=dtype),
        )

        # ── ESM3 pre-encode cache (no live encoder) ───────────────────────────
        self._esm3_cache = None
        self._esm3_cache_meta = None
        if esm3_cache_inject is not None:
            self._esm3_cache, self._esm3_cache_meta = esm3_cache_inject
        elif esm3_cache_path is not None:
            self._esm3_cache, self._esm3_cache_meta = self._load_esm3_cache(
                esm3_cache_path, protein_model_name
            )
            logger.info(
                "ESM3 cache loaded (%d seqs, dim=%d)",
                self._esm3_cache_meta.get("n_seqs", -1),
                int(self._esm3_cache_meta["embedding_dim"]),
            )

        # ── Cached GO embedding ───────────────────────────────────────────────
        self._go_embed_cache: dict[str, torch.Tensor] = {}
        if go_embedding_inject is not None:
            self._go_embed_cache["all"] = go_embedding_inject.to(
                device=device, dtype=dtype
            )
        elif go_embedding_path is not None and os.path.exists(go_embedding_path):
            emb = torch.load(go_embedding_path, map_location=device).to(
                device=device, dtype=dtype
            )
            self._go_embed_cache["all"] = emb
            logger.info("Loaded cached GO embedding from %s", go_embedding_path)

        # ── Resume trained projections (optional) ─────────────────────────────
        if proj_resume_dir is not None:
            self._load_projections(proj_resume_dir)

        # ── Freeze the LoRA base; keep adapters + projections trainable ───────
        # The lora_gemma4 builder does NOT freeze the base (the standard torchtune LoRA
        # recipe does it via set_trainable_params). Here the trainable set is the LoRA
        # adapters PLUS the from-scratch protein/GO projections. The projections live
        # OUTSIDE backbone, so freeze only the backbone's non-adapter params.
        if self._has_lora:
            from torchtune.modules.peft import get_adapter_params

            adapter_keys = set(get_adapter_params(self.backbone).keys())
            for name, p in self.backbone.named_parameters():
                p.requires_grad = name in adapter_keys
            # projections stay trainable (default requires_grad=True)

    # ── cache / hidden-dim helpers ────────────────────────────────────────────

    @staticmethod
    def esm3_cache_key(sequence: str) -> str:
        """SHA1 key for a (already-truncated) amino-acid sequence — must match the
        truncation the dataset/precompute used."""
        import hashlib

        return hashlib.sha1(sequence.encode("ascii", "ignore")).hexdigest()

    def _infer_protein_hidden(
        self, esm3_cache_path: Optional[str], protein_model_name: str
    ) -> int:
        if esm3_cache_path is None:
            raise ValueError(
                "esm3_cache_path is required (no live ESM3 encoder is built in the "
                "native model)."
            )
        sidecar = esm3_cache_path + ".json"
        if not os.path.exists(sidecar):
            raise FileNotFoundError(
                f"ESM3 cache sidecar not found at {sidecar} (written by "
                f"precompute_esm3_cache.py alongside the .pt)."
            )
        with open(sidecar) as f:
            meta = json.load(f)
        return int(meta["embedding_dim"])

    def _load_esm3_cache(self, cache_path: str, protein_model_name: str):
        """Load the pre-encoded ESM3 cache + sidecar; reject a stale/mismatched cache.

        Returns (cache_dict, meta_dict). cache_dict maps esm3_cache_key(seq) ->
        per-residue feature tensor [L+2, embedding_dim].
        """
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"ESM3 cache not found at {cache_path}. Run "
                f"experiments/bioreason/precompute_esm3_cache.py first."
            )
        sidecar = cache_path + ".json"
        if not os.path.exists(sidecar):
            raise FileNotFoundError(f"ESM3 cache sidecar not found at {sidecar}.")
        with open(sidecar) as f:
            meta = json.load(f)
        if meta.get("esm3_model_name") != protein_model_name:
            raise ValueError(
                f"ESM3 cache model mismatch: sidecar='{meta.get('esm3_model_name')}' "
                f"vs requested='{protein_model_name}'. Re-run precompute."
            )
        cache = torch.load(cache_path, map_location="cpu")
        if not isinstance(cache, dict) or not cache:
            raise ValueError(f"ESM3 cache at {cache_path} is empty or not a dict.")
        return cache, meta

    def _load_projections(self, src_dir: str):
        for name, module in (
            ("protein_projection", self.protein_projection),
            ("go_projection", self.go_projection),
        ):
            path = os.path.join(src_dir, f"{name}.pt")
            if os.path.exists(path):
                state = torch.load(path, map_location=self.device)
                module.load_state_dict(state, strict=True)
                module.to(device=self.device, dtype=self.dtype)
                logger.info("Resumed %s from %s", name, path)
            else:
                logger.warning("%s.pt not found at %s — keeping init", name, path)

    def projector_state_dict(self) -> dict:
        return {
            "protein_projection": self.protein_projection.state_dict(),
            "go_projection": self.go_projection.state_dict(),
        }

    # ── embedding-splice ──────────────────────────────────────────────────────

    def _splice_embeds(
        self,
        input_ids: torch.Tensor,
        protein_sequences: list[str],
        go_aspects: Optional[list[str]],
        batch_idx_map: Optional[list[int]],
    ) -> torch.Tensor:
        """Build [B, S, H] embeds: text via tok_embeddings (scaled), protein/GO via the
        trainable projections spliced at the reserved-id positions.

        Runs in whatever grad context the caller establishes — SFT calls it with grad
        ENABLED so the projections train through the splice.
        """
        B = input_ids.shape[0]
        input_ids = input_ids.to(self.device)
        embeds = self._embed(input_ids)  # [B, S, H], scaled by sqrt(H)

        # Protein features (cache lookup -> projection)
        if protein_sequences:
            if batch_idx_map is None:
                batch_idx_map = list(range(B))
            if self._esm3_cache is None:
                raise RuntimeError("ESM3 cache not loaded but protein_sequences passed.")
            per_item: list[list[torch.Tensor]] = [[] for _ in range(B)]
            for seq_idx, seq in enumerate(protein_sequences):
                k = self.esm3_cache_key(seq)
                t = self._esm3_cache.get(k)
                if t is None:
                    raise KeyError(
                        f"ESM3 cache miss for sequence (len={len(seq)}, "
                        f"key={k[:12]}…). Cache incomplete — re-run precompute over "
                        f"the full dataset."
                    )
                per_item[batch_idx_map[seq_idx]].append(t)
            raw = [
                torch.cat(per_item[i], dim=0)
                if per_item[i]
                else torch.zeros((0, self._protein_hidden))
                for i in range(B)
            ]
            flat = torch.cat(raw, dim=0).to(device=self.device, dtype=self.dtype)
            flat = self.protein_projection(flat)
            mask = input_ids == self.protein_token_id
            if mask.sum().item() != flat.shape[0]:
                raise ValueError(
                    f"Protein token count {mask.sum().item()} != "
                    f"protein features {flat.shape[0]}"
                )
            embeds = embeds.clone()
            embeds[mask] = flat

        # GO features (cached -> projection)
        go_embeds = self._get_go_embeds(go_aspects or ["all"] * B, B)
        if go_embeds is not None:
            go_mask = input_ids == self.go_token_id
            go_per_item = go_mask.sum(dim=1)
            sliced = [go_embeds[i][: go_per_item[i].item()] for i in range(B)]
            flat_go = torch.cat(sliced, dim=0).to(device=self.device, dtype=self.dtype)
            flat_go = self.go_projection(flat_go)
            if go_mask.sum().item() != flat_go.shape[0]:
                raise ValueError(
                    f"GO token count {go_mask.sum().item()} != "
                    f"GO features {flat_go.shape[0]}"
                )
            if not protein_sequences:
                embeds = embeds.clone()
            embeds[go_mask] = flat_go

        return embeds

    def _get_go_embeds(
        self, go_aspects: list[str], batch_size: int
    ) -> Optional[list[torch.Tensor]]:
        if not self._go_embed_cache:
            return None
        return [self._go_embed_cache[a or "all"] for a in go_aspects]

    @torch.no_grad()
    def build_prompt_embeds(
        self,
        input_ids: torch.Tensor,
        protein_sequences: list[str],
        go_aspects: Optional[list[str]] = None,
        batch_idx_map: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """No-grad multimodal prompt embeds (e.g. for generation). Returns [B, S, H]
        on CPU."""
        return self._splice_embeds(
            input_ids, protein_sequences, go_aspects, batch_idx_map
        ).cpu()

    def build_full_embeds_train(
        self,
        tokens: torch.Tensor,
        protein_sequences: list[str],
        go_aspects: Optional[list[str]] = None,
        batch_idx_map: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """Grad-ENABLED multimodal embeds over the full SFT token stream
        (prompt + target are one contiguous sequence). Returns [B, S, H] on device.

        The projections are trained THROUGH this splice, so do not wrap in no_grad.
        """
        return self._splice_embeds(
            tokens, protein_sequences, go_aspects, batch_idx_map
        )

    # ── forward (native decoder) ──────────────────────────────────────────────

    # ── LinearCrossEntropyLoss wiring ─────────────────────────────────────────
    # The modern SFT loss (LinearCrossEntropyLoss) calls set_model_output(model),
    # which sets model.skip_output_layer=True and grabs model.output, then does the
    # vocab projection ITSELF chunk-by-chunk on only the valid (non-ignored) tokens.
    # This is the FSDP-correct replacement for the now-deprecated chunked_output path
    # (which materializes per-chunk full-vocab logits via the tied output weight and
    # raises "tensor ... data is not allocated yet" under FSDP2 sharding at >6 tiles).
    # Delegate both attributes to the backbone so the loss drives the decoder directly.
    @property
    def output(self):
        return self.backbone.output

    @property
    def skip_output_layer(self) -> bool:
        return self.backbone.skip_output_layer

    @skip_output_layer.setter
    def skip_output_layer(self, value: bool) -> None:
        self.backbone.skip_output_layer = value

    def forward(
        self,
        tokens: Optional[torch.Tensor] = None,
        *,
        protein_sequences: Optional[list[str]] = None,
        go_aspects: Optional[list[str]] = None,
        batch_idx_map: Optional[list[int]] = None,
        input_embeds: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the native Gemma 4 decoder.

        Returns logits [B,S,vocab] normally, OR hidden states [B,S,emb_dim] when
        skip_output_layer is set (LinearCrossEntropyLoss does the projection itself).

        The embed-splice runs HERE (inside the module's forward) so the splice's
        ``self._embed(tokens)`` lookup executes under the root FSDP forward hook —
        ``tok_embeddings.weight`` is unsharded to a plain tensor at that point.
        Building the embeds OUTSIDE forward (e.g. in the recipe) leaves the weight as a
        sharded DTensor and ``aten.embedding`` errors on mixed Tensor/DTensor.

        Pass ``tokens`` + the multimodal side inputs (training path). ``input_embeds``
        may be passed directly to bypass the splice (used by the CPU equivalence test).
        ``mask=None`` -> causal by default; right-padding handled by label masking.
        Returns logits [B, S, vocab].
        """
        if input_embeds is None:
            input_embeds = self._splice_embeds(
                tokens, protein_sequences or [], go_aspects, batch_idx_map
            )
        return self.backbone(
            tokens=None,
            input_embeds=input_embeds,
            mask=mask,
            input_pos=input_pos,
        )

    @staticmethod
    def merge_backbone_state_dict(full_state_dict: dict) -> dict:
        """Given a full (gathered) state dict of this wrapper, return the BACKBONE
        weights as a bare-decoder tune-format state dict with LoRA MERGED in
        (``W_eff = W_base + (alpha/rank) * (B @ A)``), suitable for the GEMMA4
        checkpointer to write HF safetensors for eval.

        Keys are stripped of the ``backbone.`` prefix and any FSDP/AC wrapper
        infixes. LoRA adapter/base pairs collapse to a single merged ``...weight``.
        Non-LoRA keys pass through unchanged. Projection keys are dropped (saved
        separately by the recipe).

        NOTE: alpha/rank are read from the global LoRA defaults (32/64) unless the
        caller patches them; for the standard config they are constant across layers.
        """
        def _strip(name: str) -> str:
            return (
                name.replace("_fsdp_wrapped_module.", "")
                .replace("_checkpoint_wrapped_module.", "")
                .replace("base_model.model.", "")
            )

        # Collect backbone-only tensors under stripped keys.
        bk: dict[str, torch.Tensor] = {}
        for k, v in full_state_dict.items():
            ck = _strip(k)
            if ck.startswith("backbone."):
                bk[ck[len("backbone."):]] = v
        return bk

    def merged_backbone_for_save(
        self, full_state_dict: dict, lora_rank: int = 32, lora_alpha: float = 64.0
    ) -> dict:
        """Merge LoRA adapters into base weights in a gathered backbone state dict.

        Produces a bare-decoder state dict (no lora_a/lora_b keys): for each
        ``<p>.lora_a.weight`` / ``<p>.lora_b.weight`` pair, replaces ``<p>.weight``
        with ``W_base + (alpha/rank) * (B @ A)``. Used at checkpoint time so the saved
        model is a complete Gemma4 decoder loadable by vLLM for eval (no native-LoRA
        loader exists on the eval side)."""
        bk = self.merge_backbone_state_dict(full_state_dict)
        scale = lora_alpha / lora_rank
        # find lora pairs
        lora_prefixes = set()
        for k in bk:
            if k.endswith(".lora_a.weight"):
                lora_prefixes.add(k[: -len(".lora_a.weight")])
        for p in lora_prefixes:
            a = bk.pop(f"{p}.lora_a.weight")
            b = bk.pop(f"{p}.lora_b.weight")
            wkey = f"{p}.weight"
            if wkey in bk:
                delta = scale * (b.to(torch.float32) @ a.to(torch.float32))
                bk[wkey] = (bk[wkey].to(torch.float32) + delta).to(bk[wkey].dtype)
        return bk

    def set_num_output_chunks(self, num_output_chunks: int) -> None:
        """Delegate to the backbone so the chunked SFTLoss (set_model_output) can
        request chunked logits — avoids materializing the full [B,S,vocab] fp32 tensor
        at the 262144 Gemma vocab."""
        self.backbone.set_num_output_chunks(num_output_chunks)

    def trainable_parameters(self):
        for name, p in self.named_parameters():
            if p.requires_grad:
                yield name, p

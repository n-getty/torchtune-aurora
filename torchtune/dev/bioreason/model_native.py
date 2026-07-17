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
from torchtune.models.qwen3 import qwen3_32b, lora_qwen3_32b

# Backbone registry: name -> (dense_builder, lora_builder). The wrapper is
# backbone-agnostic — it reuses backbone.tok_embeddings for the splice and feeds
# input_embeds, which works for any TransformerDecoder. Gemma4 applies a sqrt(H)
# embedding scale inside tok_embeddings (GemmaNormEmbeddings); Qwen3 uses a plain
# nn.Embedding (no scale). Either way the splice uses the SAME embedding the decoder
# would, so text/target tokens enter at the correct magnitude for that backbone.
_BACKBONES = {
    "gemma4_31b": (gemma4_31b, lora_gemma4_31b),
    "qwen3_32b": (qwen3_32b, lora_qwen3_32b),
}


class _LazySafetensorsCache:
    """Lazy, scalable ESM3 feature cache backed by a single safetensors file.

    Mirrors the dict `.get(key)` / `.__getitem__` / `in` / `len` API the embed-splice
    uses, but reads ONLY the requested tensor's bytes per lookup (safe_open get_tensor),
    never the whole file. This is what lets 12 same-node ranks share one ~200 GiB cache
    without each deserializing the full pickle (the .pt path was ~93% I/O-blocked at scale
    and blew walltime; see feedback_esm3_cache_mmap_required_at_scale + the safetensors
    conversion). The handle is opened once; the key set comes from the header (cheap).
    """

    def __init__(self, path: str):
        from safetensors import safe_open

        self._path = path
        # framework="pt", device="cpu": get_tensor returns a CPU torch.Tensor slice.
        self._f = safe_open(path, framework="pt", device="cpu")
        self._keys = set(self._f.keys())

    def get(self, key, default=None):
        if key not in self._keys:
            return default
        return self._f.get_tensor(key)

    def __getitem__(self, key):
        return self._f.get_tensor(key)

    def __contains__(self, key):
        return key in self._keys

    def __len__(self):
        return len(self._keys)

    def keys(self):
        return self._keys

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
        freeze_backbone (bool): Stage-1 alignment mode — freeze the ENTIRE backbone and
            train ONLY the protein/GO projections (no LoRA). Ignored when enable_lora=True
            (LoRA already freezes the base). Default: False. See the trainable-param
            policy note in __init__ for the two-stage (LLaVA-style) rationale.
        freeze_projector (bool): Stage-2 only (enable_lora=True) — freeze the protein/GO
            projections at their loaded (Stage-1-aligned) values so only the LoRA adapters
            train. Prevents the projector over-amplification trap on a long run (a trainable
            projector drifts norm 721->1044 in 10 steps once loss saturates). Default: False.
        projector_output_norm (bool): append a LayerNorm to each projector so the spliced
            feature magnitude is BOUNDED (per-row norm ~sqrt(H)) and cannot over-amplify.
            Fixes the frozen-backbone over-amplification trap (proj-out norm 721->1629 ->
            ':' collapse) without LoRA — the cleanest capability path, since the base backbone
            already reasons coherently from a correctly-scaled splice (LoRA itself breaks
            splice-generation; see the recipe notes). Default: False.
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
        freeze_backbone: bool = False,
        freeze_projector: bool = False,
        projector_output_norm: bool = False,
        lora_rank: int = 32,
        lora_alpha: float = 64.0,
        lora_dropout: float = 0.0,
        lora_attn_modules: Optional[list[str]] = None,
        apply_lora_to_mlp: bool = True,
        disable_protein_splice: bool = False,
        disable_go_splice: bool = False,
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
        self._freeze_backbone = bool(freeze_backbone)
        self._freeze_projector = bool(freeze_projector)
        self._projector_output_norm = bool(projector_output_norm)
        # Modality-ablation switches (Exp 2, embed-on/off): keep placeholder tokens but skip
        # writing projected features. Isolate the protein / GO modality's contribution.
        self.disable_protein_splice = bool(disable_protein_splice)
        self.disable_go_splice = bool(disable_go_splice)

        # ── Backbone (native TransformerDecoder: gemma4_31b or qwen3_32b) ─────
        if backbone is not None:
            self.backbone = backbone
        elif backbone_builder in _BACKBONES:
            dense_builder, lora_builder = _BACKBONES[backbone_builder]
            if enable_lora:
                if lora_attn_modules is None:
                    lora_attn_modules = ["q_proj", "k_proj", "v_proj", "output_proj"]
                self.backbone = lora_builder(
                    lora_attn_modules=lora_attn_modules,
                    apply_lora_to_mlp=apply_lora_to_mlp,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                )
            else:
                self.backbone = dense_builder()
        else:
            raise ValueError(
                f"Unknown backbone_builder: {backbone_builder!r} "
                f"(known: {sorted(_BACKBONES)})"
            )

        # Reuse the backbone's own tok_embeddings for the splice. Feeding input_embeds
        # bypasses this layer in forward, but using it keeps text/target tokens at the
        # exact magnitude the decoder expects (Gemma4: sqrt(H)-scaled; Qwen3: plain).
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
        # projector_output_norm: append a LayerNorm to each projector so the spliced
        # feature magnitude is BOUNDED and cannot over-amplify. Without it, a frozen-backbone
        # projector trained past loss-saturation drifts its output per-row norm 721 -> 1629
        # and re-breaks the splice -> ':' collapse (HW-confirmed). LayerNorm pins per-element
        # output to ~unit (per-row norm ~sqrt(H) ~71, well below the 721 the base tolerated
        # AND bounded). Matches the published go_graph_encoder, which uses LayerNorm. The
        # recipe's post-meta reset_parameters loop re-inits it (weight=1, bias=0).
        def _proj(in_dim):
            layers = [
                nn.Linear(in_dim, hidden_size, dtype=dtype),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size, dtype=dtype),
            ]
            if self._projector_output_norm:
                layers.append(nn.LayerNorm(hidden_size, dtype=dtype))
            return nn.Sequential(*layers)

        self.protein_projection = _proj(protein_hidden)
        self.go_projection = _proj(self.GO_DIM)

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

        # ── Trainable-param policy (three regimes) ────────────────────────────
        # The published BioReason recipe is TWO-STAGE (LLaVA-style alignment):
        #   Stage 1 (freeze_backbone=True, enable_lora=False): backbone FULLY FROZEN,
        #     train ONLY protein/GO projections. With the backbone unable to change,
        #     the only way to reduce loss is to make the spliced protein/GO features
        #     informative — there is no text shortcut to fall into. This is the fix
        #     for the epoch_0 failure (single-stage LoRA learned the go_pred text
        #     shortcut at lr=1e-4 in ~25 steps; the projection stayed at init ->
        #     norm-10600 ESM3 feats swamped context -> ':' collapse at gen).
        #   Stage 2 (enable_lora=True): LoRA-finetune the backbone on top of the
        #     Stage-1-aligned projections (loaded via proj_resume_dir). Base frozen
        #     except adapters; projections continue training.
        #   Full-FT (neither): everything trainable (legacy; not the published path).
        if self._has_lora:
            # Stage 2: freeze base, keep LoRA adapters + projections trainable. The
            # lora_gemma4 builder does NOT freeze the base (the standard torchtune LoRA
            # recipe does it via set_trainable_params). The projections live OUTSIDE
            # backbone, so freeze only the backbone's non-adapter params.
            from torchtune.modules.peft import get_adapter_params

            adapter_keys = set(get_adapter_params(self.backbone).keys())
            for name, p in self.backbone.named_parameters():
                p.requires_grad = name in adapter_keys
            # Projections: trainable by default, BUT freeze_projector locks them at the
            # Stage-1-aligned values. Needed for a long Stage-2 run: the go_pred text leak
            # drives loss to ~0 fast, after which a trainable projector OVER-AMPLIFIES
            # (proj-out norm 721->1044 in just 10 Stage-2 steps) and re-breaks the scale
            # -> ':' collapse. Freezing keeps the validated norm-721 alignment while only
            # the LoRA backbone learns to reason from the spliced features.
            if self._freeze_projector:
                for proj in (self.protein_projection, self.go_projection):
                    for p in proj.parameters():
                        p.requires_grad = False
        elif self._freeze_backbone:
            # Stage 1: freeze the ENTIRE backbone; only projections train.
            for p in self.backbone.parameters():
                p.requires_grad = False
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
        # safetensors path (PREFERRED at scale): lazy per-key reads via safe_open. The
        # .pt-dict path does NOT scale — even torch.load(mmap=True) must deserialize the
        # whole pickle (121645-entry index) per reader, so 12 same-node ranks each stream
        # ~200 GiB from DAOS = ~93% I/O-blocked, blowing walltime (job 8572649). The lazy
        # cache reads only the small header once + each [L+2,dim] slice on demand. Convert
        # with experiments/bioreason/convert_esm3_cache_to_safetensors.py. See
        # feedback_esm3_cache_mmap_required_at_scale.
        if cache_path.endswith(".safetensors"):
            cache = _LazySafetensorsCache(cache_path)
            if len(cache) == 0:
                raise ValueError(f"ESM3 safetensors cache at {cache_path} is empty.")
            return cache, meta
        # Legacy .pt dict path: mmap defers raw storage bytes (avoids the 12×-RAM OOM that
        # wedged a node), but still deserializes the full pickle index — slow at scale.
        # Kept for the small validation-only cache; prefer safetensors for the train cache.
        cache = torch.load(cache_path, map_location="cpu", mmap=True)
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
        # Ablation (disable_protein_splice, Exp 2 embed-on/off): keep placeholder tokens but
        # skip writing projected ESM3 features — the plain placeholder embedding stays. Holds
        # seqlen/format constant to isolate the protein-embedding modality's contribution.
        if protein_sequences and not getattr(self, "disable_protein_splice", False):
            if batch_idx_map is None:
                batch_idx_map = list(range(B))
            if self._esm3_cache is None:
                raise RuntimeError("ESM3 cache not loaded but protein_sequences passed.")
            per_item: list[list[torch.Tensor]] = [[] for _ in range(B)]
            import os as _os
            import time as _time

            _timing = _os.environ.get("TORCHTUNE_BIOREASON_TIMING") == "1"
            _t0 = _time.perf_counter() if _timing else 0.0
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
            if _timing:
                logger.info(
                    "[bioreason-timing] esm3_cache_read %d proteins in %.3fs",
                    len(protein_sequences),
                    _time.perf_counter() - _t0,
                )
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

        # GO features (cached -> projection). disable_go_splice: same ablation for GO.
        # go_aspects is per-DOCUMENT (one entry per protein/GO doc). Without packing that is
        # one doc per batch row (len == B); with packing several docs share a row and
        # batch_idx_map[doc_idx] gives the owning row. Assemble GO features per document and
        # scatter into the row's GO placeholders left-to-right, mirroring the protein path —
        # otherwise a 2-doc pack has 2*num_go_tokens placeholders but only one doc's features
        # ("GO token count 400 != GO features 200", HW job 8673929).
        _go_aspects = go_aspects if go_aspects is not None else (["all"] * B)
        go_embeds = self._get_go_embeds(_go_aspects, len(_go_aspects))
        if go_embeds is not None and not getattr(self, "disable_go_splice", False):
            go_mask = input_ids == self.go_token_id
            go_per_item = go_mask.sum(dim=1)  # GO tokens present in each batch ROW
            _go_bmap = batch_idx_map
            if _go_bmap is None:
                _go_bmap = list(range(len(go_embeds)))
            # Group per-document GO caches by owning batch row (document order), then take
            # exactly the row's GO-token count from the front of the row's concatenated
            # caches. Each doc contributes its num_go_tokens (<=200) prefix; a row with no
            # GO tokens (text-only) slices to length 0. Without packing this reduces to the
            # prior per-row `go_embeds[i][:count]`. batch_idx_map carries the doc->row map so
            # a multi-doc pack fills all its GO blocks (HW bug 8673929: 2-doc pack had 400
            # placeholders but the old per-row path supplied only one doc's 200 features).
            go_per_row: list[list[torch.Tensor]] = [[] for _ in range(B)]
            for doc_idx, ge in enumerate(go_embeds):
                go_per_row[_go_bmap[doc_idx]].append(ge)
            row_go = []
            for i in range(B):
                cat = (
                    torch.cat(go_per_row[i], dim=0)
                    if go_per_row[i]
                    else torch.zeros((0, self.GO_DIM))
                )
                row_go.append(cat[: int(go_per_item[i].item())])
            flat_go = torch.cat(row_go, dim=0).to(device=self.device, dtype=self.dtype)
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

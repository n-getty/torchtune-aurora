"""
BioReason-Pro model wrapper for TorchTune GRPO training on XPU.

Wraps the three-component multimodal stack (ESM3 protein encoder, GO graph encoder,
Qwen3-4B LLM backbone) into a single nn.Module whose forward() accepts inputs_embeds
so the GRPO recipe can use it identically to a text-only model after embeddings are
pre-computed by build_prompt_embeds().
"""

from __future__ import annotations

import os
import sys
import fnmatch
import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_DEFAULT_BIOREASON_SRC = "/flare/ModCon/ngetty/BioReason-Pro"
_DEFAULT_BIOREASON_DEPS = "/lus/flare/projects/ModCon/ngetty/bioreason_deps"
_DEFAULT_PROJDIR = "/lus/flare/projects/ModCon/ngetty/torchtune"


def _resolve_bioreason_paths() -> tuple[str, str, str]:
    """Resolve BioReason source/deps/project paths from env vars with defaults.

    Honours BIOREASON_SRC, BIOREASON_DEPS, BIOREASON_PROJDIR. Resolution is lazy
    (called from _ensure_paths) so the module imports cleanly on environments
    without the BioReason checkout, e.g. CI and other dev machines.
    """
    src = os.environ.get("BIOREASON_SRC", _DEFAULT_BIOREASON_SRC)
    deps = os.environ.get("BIOREASON_DEPS", _DEFAULT_BIOREASON_DEPS)
    projdir = os.environ.get("BIOREASON_PROJDIR", _DEFAULT_PROJDIR)
    return src, deps, projdir


# Lazy: populated on first _ensure_paths() call. Plain assignment keeps the
# module-level names for any downstream code that reads them after init.
_BIOREASON_SRC = _DEFAULT_BIOREASON_SRC
_BIOREASON_DEPS = _DEFAULT_BIOREASON_DEPS
_PROJDIR = _DEFAULT_PROJDIR


def _ensure_paths():
    global _BIOREASON_SRC, _BIOREASON_DEPS, _PROJDIR
    _BIOREASON_SRC, _BIOREASON_DEPS, _PROJDIR = _resolve_bioreason_paths()
    for label, path in (("BIOREASON_SRC", _BIOREASON_SRC), ("BIOREASON_DEPS", _BIOREASON_DEPS)):
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"{label}={path!r} does not exist. Set the env var to a valid "
                f"BioReason checkout (or unset to use the default)."
            )
    for p in [_BIOREASON_DEPS, _BIOREASON_SRC]:
        if p not in sys.path:
            sys.path.insert(0, p)

    # ESM3's data_root() calls snapshot_download when INFRA_PROVIDER is not set.
    # Setting INFRA_PROVIDER makes data_root() return Path("") (relative to CWD).
    # We create the expected structure under PROJDIR:
    #   data/weights/esm3_sm_open_v1.pth  → symlink to protein_model/pytorch_model.bin
    #   data/uniref90_and_mgnify90_residue_annotations_gt_1k_proteins.csv  (minimal)
    # Must be set BEFORE any esm.* import since data_root() is @cached.
    if "INFRA_PROVIDER" not in os.environ:
        os.environ["INFRA_PROVIDER"] = "local"

    # Prevent bioreason2/models/__init__.py from importing unsloth by
    # pre-registering the package in sys.modules without executing __init__.
    import types
    if "bioreason2" not in sys.modules:
        pkg = types.ModuleType("bioreason2")
        pkg.__path__ = [f"{_BIOREASON_SRC}/bioreason2"]
        pkg.__package__ = "bioreason2"
        sys.modules["bioreason2"] = pkg
    if "bioreason2.models" not in sys.modules:
        pkg = types.ModuleType("bioreason2.models")
        pkg.__path__ = [f"{_BIOREASON_SRC}/bioreason2/models"]
        pkg.__package__ = "bioreason2.models"
        sys.modules["bioreason2.models"] = pkg


class BioReasonModel(nn.Module):
    """
    Multimodal model for GRPO RL training on Aurora XPU.

    Components:
      - ESM3 protein encoder (frozen during RL — embeddings are static per sequence)
      - GO graph encoder (frozen — output cached from go_embedding.pt at load time)
      - protein_projection MLP  (trainable)
      - go_projection MLP       (trainable)
      - Qwen3-4B LLM backbone   (trainable via DDP, no LoRA for simplicity)

    The forward() method accepts inputs_embeds (pre-computed by build_prompt_embeds)
    and returns logits, matching the interface expected by the GRPO recipe.

    For rollout generation, use build_prompt_embeds() + vLLM(enable_prompt_embeds=True).
    For training forward, use build_full_embeds() which extends prompt embeds with
    completion token embeddings.
    """

    def __init__(
        self,
        ckpt_dir: str,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        protein_model_name: str = "esm3_sm_open_v1",
        attn_implementation: str = "sdpa",
        go_obo_path: Optional[str] = None,
        precomputed_go_path: Optional[str] = None,
    ):
        super().__init__()
        _ensure_paths()

        self.device = device
        self.dtype = dtype
        self._ckpt_dir = ckpt_dir

        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from bioreason2.models.special_tokens import get_all_special_tokens, get_token

        cfg = AutoConfig.from_pretrained(ckpt_dir, trust_remote_code=True)
        self.hidden_size = cfg.hidden_size
        self.vocab_size = cfg.vocab_size

        # ── LLM backbone ──────────────────────────────────────────────────────
        logger.info("Loading Qwen3 LLM backbone...")
        self.backbone = AutoModelForCausalLM.from_pretrained(
            ckpt_dir,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        ).to(device)

        # ── Tokenizer + special token IDs ─────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": get_all_special_tokens()}
        )
        self.protein_token_id = self.tokenizer.convert_tokens_to_ids(
            get_token("protein_pad")
        )
        self.go_token_id = self.tokenizer.convert_tokens_to_ids(get_token("go_graph_pad"))

        # Local embedding layer for computing prompt_embeds outside vLLM.
        # Loaded from checkpoint safetensors — same weights as backbone.embed_tokens.
        self._embed = self._load_embed_layer(ckpt_dir, cfg)

        # ── Projectors (trainable) ────────────────────────────────────────────
        from bioreason2.models.protein_encoder import create_protein_encoder

        self.protein_encoder = create_protein_encoder(
            protein_model_name, inference_mode=True
        )
        # ESM3 stays in float32: it has fp32_autocast_context internally for
        # numerical stability in structure ops. Output embeddings are cast to
        # self.dtype in build_prompt_embeds() before the projection layer.
        self.protein_encoder.model.to(device=device)
        protein_hidden = self.protein_encoder.embedding_dim

        self.protein_projection = nn.Sequential(
            nn.Linear(protein_hidden, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        ).to(device=device, dtype=dtype)

        self.go_projection = nn.Sequential(
            nn.Linear(2560, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        ).to(device=device, dtype=dtype)

        # ── GO encoder (optional, usually frozen with cached output) ──────────
        self.go_encoder = None
        self._go_embed_cache: dict[str, torch.Tensor] = {}

        if go_obo_path and precomputed_go_path:
            from bioreason2.models.go_graph_encoder import create_go_graph_encoder_pipeline
            self.go_encoder = create_go_graph_encoder_pipeline(
                go_obo_path=go_obo_path,
                precomputed_embeddings_path=precomputed_go_path,
                embeddings_load_to=str(device),
            )

        # ── Load checkpoint weights for projectors / GO ───────────────────────
        self._load_custom_weights(ckpt_dir)

        # Freeze ESM3 and GO encoder during RL
        self._freeze_encoders()

    # ── Weight loading helpers ────────────────────────────────────────────────

    def _load_embed_layer(self, ckpt_dir: str, cfg) -> nn.Embedding:
        from safetensors import safe_open
        emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size, dtype=self.dtype)
        key = "model.embed_tokens.weight"
        # Try single file, then shards. Use os.listdir + fnmatch (NOT the stdlib
        # glob module) to avoid hangs on DAOS/dfuse mounts (see CLAUDE.md
        # "Critical Platform Constraints"). The regression test asserts the
        # forbidden substring is absent from this module.
        try:
            shard_names = sorted(
                fn for fn in os.listdir(ckpt_dir)
                if fnmatch.fnmatch(fn, "model-*.safetensors")
            )
        except FileNotFoundError:
            shard_names = []
        candidates = [os.path.join(ckpt_dir, "model.safetensors")] + [
            os.path.join(ckpt_dir, n) for n in shard_names
        ]
        for path in candidates:
            if not os.path.exists(path):
                continue
            with safe_open(path, framework="pt", device="cpu") as f:
                if key in f.keys():
                    emb.weight.data = f.get_tensor(key)
                    logger.info(f"Loaded embed_tokens from {path}")
                    break
        return emb.to(self.device)

    def _load_custom_weights(self, ckpt_dir: str):
        def _load(name, module):
            path = os.path.join(ckpt_dir, f"{name}.pt")
            if os.path.exists(path):
                state = torch.load(path, map_location=self.device)
                module.load_state_dict(state, strict=True)
                module.to(device=self.device, dtype=self.dtype)
                logger.info(f"Loaded {name} from {path}")
            else:
                logger.warning(f"{name}.pt not found at {path} — using random init")

        _load("protein_projection", self.protein_projection)
        _load("go_projection", self.go_projection)
        if self.go_encoder is not None:
            _load("go_encoder", self.go_encoder)

        # Pre-computed GO embedding (avoids encoder forward during training)
        go_emb_path = os.path.join(ckpt_dir, "go_embedding.pt")
        if os.path.exists(go_emb_path):
            emb = torch.load(go_emb_path, map_location=self.device).to(
                device=self.device, dtype=self.dtype
            )
            self._go_embed_cache["all"] = emb
            logger.info(f"Loaded cached GO embedding from {go_emb_path}")

    def _freeze_encoders(self):
        for p in self.protein_encoder.model.parameters():
            p.requires_grad = False
        if self.go_encoder is not None:
            for p in self.go_encoder.parameters():
                p.requires_grad = False
        # GO projection is trainable; protein projection is trainable.
        logger.info("ESM3 and GO encoder frozen (RL trains projectors + backbone)")

    # ── Embedding computation ─────────────────────────────────────────────────

    @torch.no_grad()
    def build_prompt_embeds(
        self,
        input_ids: torch.Tensor,
        protein_sequences: list[str],
        go_aspects: Optional[list[str]] = None,
        batch_idx_map: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """
        Build multimodal prompt embeddings for vLLM generation.

        Returns: [B, ctx_len, hidden_size] on CPU (vLLM expects CPU tensors).
        """
        B = input_ids.shape[0]
        input_ids = input_ids.to(self.device)

        # Text token embeddings
        embeds = self._embed(input_ids)  # [B, ctx_len, H]

        # Protein embeddings
        if protein_sequences:
            if batch_idx_map is None:
                batch_idx_map = list(range(B))
            raw = self.protein_encoder.encode_sequences(
                protein_sequences, batch_idx_map, B
            )
            # ESM3 per_residue_embedding includes BOS and EOS tokens (+2 per sequence).
            # SFT was trained with placeholders for the BOS/EOS too — see upstream
            # PLProcessor (processing_pl.py:184-185, num_protein_tokens = seq_len + 2).
            # dataset.py inserts len(seq)+2 protein_pad tokens, so fill all of them
            # with the full unstripped ESM3 features.
            flat = torch.cat(raw, dim=0).to(device=self.device, dtype=self.dtype)
            flat = self.protein_projection(flat)
            mask = input_ids == self.protein_token_id
            if mask.sum().item() != flat.shape[0]:
                raise ValueError(
                    f"Protein token count {mask.sum().item()} != "
                    f"protein features {flat.shape[0]}"
                )
            embeds[mask] = flat

        # GO embeddings
        go_embeds = self._get_go_embeds(go_aspects or ["all"] * B, B)
        if go_embeds is not None:
            go_mask = input_ids == self.go_token_id
            # go_embedding.pt has shape [max_go_tokens, 2560].  Slice each batch item's
            # embedding to the number of GO placeholder tokens actually present.
            go_per_item = go_mask.sum(dim=1)  # [B] tokens per item
            sliced = [go_embeds[i][:go_per_item[i].item()] for i in range(B)]
            flat_go = torch.cat(sliced, dim=0).to(device=self.device, dtype=self.dtype)
            flat_go = self.go_projection(flat_go)
            if go_mask.sum().item() != flat_go.shape[0]:
                raise ValueError(
                    f"GO token count {go_mask.sum().item()} != "
                    f"GO features {flat_go.shape[0]}"
                )
            embeds[go_mask] = flat_go

        return embeds.cpu()  # vLLM expects CPU

    def build_full_embeds(
        self,
        prompt_embeds: torch.Tensor,
        completion_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extend prompt_embeds with completion token embeddings for training forward.

        Args:
            prompt_embeds: [B, ctx_len, H] — from build_prompt_embeds (on CPU)
            completion_ids: [B, comp_len] — generated completion token IDs

        Returns: [B, ctx_len + comp_len, H] on self.device
        """
        prompt_embeds = prompt_embeds.to(device=self.device, dtype=self.dtype)
        comp_embeds = self._embed(completion_ids.to(self.device))
        return torch.cat([prompt_embeds, comp_embeds], dim=1)

    def _get_go_embeds(
        self, go_aspects: list[str], batch_size: int
    ) -> Optional[list[torch.Tensor]]:
        if not self._go_embed_cache and self.go_encoder is None:
            return None
        result = []
        for aspect in go_aspects:
            key = aspect or "all"
            if key not in self._go_embed_cache:
                if self.go_encoder is None:
                    return None
                self._go_embed_cache[key] = self.go_encoder(key).detach()
            result.append(self._go_embed_cache[key])
        return result

    # ── Standard nn.Module forward (inputs_embeds path) ──────────────────────

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass using pre-computed inputs_embeds.
        Returns logits [B, seq_len, vocab_size].
        """
        out = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            **kwargs,
        )
        return out.logits

    def trainable_parameters(self):
        """Yield (name, param) for trainable parameters (backbone + projectors)."""
        for name, p in self.named_parameters():
            if p.requires_grad:
                yield name, p

    def projector_state_dict(self) -> dict:
        """State dict for trainable projectors (for weight sync to vLLM workers)."""
        return {
            "protein_projection": self.protein_projection.state_dict(),
            "go_projection": self.go_projection.state_dict(),
        }

    def vllm_param_iter(self):
        """Yield (hf_name, param) for LLM backbone params only — used for vLLM weight sync.

        ESM3, GO encoder, protein_projection, and go_projection are excluded because
        vLLM receives pre-computed prompt_embeds and never runs the encoders/projectors.
        The backbone uses native HF parameter names (no 'backbone.' prefix).
        """
        for name, param in self.backbone.named_parameters():
            yield name, param

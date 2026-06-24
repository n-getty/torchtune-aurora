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
import json
import fnmatch
import hashlib
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
      - Qwen3-4B LLM backbone   (full-FT by default; frozen base + PEFT-LoRA
        adapters when ``enable_lora=True`` — matches the published BioReason-Pro
        RL recipe, which trains GO encoder + projections + LoRA adapters only)

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
        enable_lora: bool = False,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        esm3_cache_path: Optional[str] = None,
        adapter_path: Optional[str] = None,
        proj_resume_dir: Optional[str] = None,
    ):
        super().__init__()
        _ensure_paths()

        self.device = device
        self.dtype = dtype
        self._ckpt_dir = ckpt_dir
        self._has_lora = bool(enable_lora)
        # When an ESM3 pre-encode cache is provided, the (frozen) ESM3 encoder is
        # NOT built — its ~5.5 GiB fp32 footprint never enters the process and the
        # per-step encode is replaced by a dict lookup. Loaded below after tokenizer.
        self._esm3_cache_path = esm3_cache_path
        self._esm3_cache = None
        self._esm3_cache_meta = None

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

        # ── PEFT-LoRA on the HF backbone (published BioReason-Pro RL recipe) ───
        # The backbone is an HF AutoModelForCausalLM, so we wrap it with PEFT
        # (NOT torchtune's LoRALinear, which only builds from scratch and cannot
        # retrofit an HF module). get_peft_model freezes the base and marks only
        # the adapters trainable; the projectors (built below) stay trainable.
        # Config matches bioreason2/utils/save_grpo_ckpt.py.
        if self._has_lora:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                init_lora_weights="gaussian",
                bias="none",
                task_type="CAUSAL_LM",
            )
            # autocast_adapter_dtype=False keeps adapters in the base dtype (bf16).
            # PEFT defaults to True (fp32 adapters), but FSDP1's flat-param requires
            # UNIFORM dtype across a wrapped module — fp32 adapters + bf16 base/
            # projectors crash with "Must flatten tensors with uniform dtype but got
            # torch.bfloat16 and torch.float32". bf16 adapters + bf16 AdamW is the
            # same regime the full-FT BioReason path already trains in successfully.
            self.backbone = get_peft_model(
                self.backbone, lora_config, autocast_adapter_dtype=False
            )
            # RESUME: load trained adapter weights into the fresh PEFT adapter (e.g.
            # to continue a 4N run at 8N). adapter_path is a dir holding
            # adapter_model.safetensors (what save_checkpoint writes). We load the
            # tensors into the EXISTING adapter (set_peft_model_state_dict) rather than
            # PeftModel.from_pretrained so the LoraConfig/dtype/FSDP-readiness already
            # set above is preserved. Without this, an 8N "resume" would silently
            # restart from the gaussian init (zero learning carried over).
            if adapter_path is not None:
                import os as _os
                from safetensors.torch import load_file as _load_sft
                from peft import set_peft_model_state_dict as _set_peft_sd
                _af = _os.path.join(adapter_path, "adapter_model.safetensors")
                if not _os.path.isfile(_af):
                    raise FileNotFoundError(
                        f"adapter_path set but {_af} not found — cannot resume LoRA."
                    )
                _sd = _load_sft(_af)
                _res = _set_peft_sd(self.backbone, _sd)
                _missing = getattr(_res, "unexpected_keys", None) or []
                logger.info(
                    "PEFT-LoRA adapter RESUMED from %s (%d tensors, unexpected=%d)",
                    _af, len(_sd), len(_missing),
                )
            # LOAD-BEARING with gradient checkpointing: the frozen base produces
            # no input grads, so without this hook the adapter grads are dropped
            # (silent zero-learning). See model risks in the plan.
            self.backbone.enable_input_require_grads()
            logger.info(
                "PEFT-LoRA applied to backbone (r=%d, alpha=%d, dropout=%.3f)",
                lora_rank, lora_alpha, lora_dropout,
            )

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

        # ── Protein encoder (ESM3) OR its pre-encoded cache ───────────────────
        if self._esm3_cache_path is not None:
            # Cached path: skip building ESM3 entirely (frees ~5.5 GiB fp32/tile
            # and removes the per-step encoder forward). build_prompt_embeds()
            # looks up raw per-residue features from the cache and applies the
            # (live, trainable) protein_projection. embedding_dim comes from the
            # cache sidecar since there is no encoder to query.
            self.protein_encoder = None
            self._esm3_cache, self._esm3_cache_meta = self._load_esm3_cache(
                self._esm3_cache_path, protein_model_name
            )
            protein_hidden = int(self._esm3_cache_meta["embedding_dim"])
            logger.info(
                "ESM3 pre-encode cache loaded (%d seqs, dim=%d) — ESM3 encoder NOT built",
                self._esm3_cache_meta.get("n_seqs", -1), protein_hidden,
            )
        else:
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
        # RESUME the TRAINED projections (protein_projection.pt / go_projection.pt) from
        # a prior run's epoch dir, overlaying the SFT-base projections just loaded from
        # ckpt_dir. Without this, resuming a run (adapter_path set) would restart the
        # TRAINABLE projectors from the SFT base — silently discarding their RL training
        # (the LoRA adapter would resume but the projectors would not). proj_resume_dir
        # is normally the same epoch dir that holds adapter_path's adapter/.
        if proj_resume_dir is not None:
            self._load_custom_weights(proj_resume_dir)
            logger.info("Resumed trained projections from %s", proj_resume_dir)

        # Freeze ESM3 and GO encoder during RL
        self._freeze_encoders()

    # ── ESM3 pre-encode cache helpers ─────────────────────────────────────────

    @staticmethod
    def esm3_cache_key(sequence: str) -> str:
        """Stable key for a (already-truncated) amino-acid sequence.

        The caller MUST pass the same truncation the dataset applies
        (``sequence[:max_protein_len]``) so the key matches what the precompute
        script wrote. SHA1 keeps the on-disk dict compact and avoids storing long
        AA strings as keys.
        """
        return hashlib.sha1(sequence.encode("ascii", "ignore")).hexdigest()

    def _load_esm3_cache(self, cache_path: str, protein_model_name: str):
        """Load the pre-encoded ESM3 cache + sidecar; assert config-match.

        Returns (cache_dict, meta_dict). cache_dict maps esm3_cache_key(seq) ->
        per-residue feature tensor [L+2, embedding_dim]. The sidecar JSON
        (``<cache>.json``) records max_protein_len / embedding_dim /
        esm3_model_name / n_seqs so a stale cache is rejected loudly rather than
        producing silently-wrong embeddings.
        """
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"ESM3 cache not found at {cache_path}. Run "
                f"experiments/bioreason/precompute_esm3_cache.py first."
            )
        sidecar = cache_path + ".json"
        if not os.path.exists(sidecar):
            raise FileNotFoundError(
                f"ESM3 cache sidecar not found at {sidecar} (written by the "
                f"precompute script alongside the .pt)."
            )
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
                    # Cast to self.dtype: some checkpoints store embed_tokens as fp32
                    # (e.g. bioreason-pro-rl) and assigning the raw tensor would make
                    # `embeds` fp32 while protein/GO features are bf16 — `embeds[mask]
                    # = flat` then fails with a dtype mismatch in build_prompt_embeds.
                    emb.weight.data = f.get_tensor(key).to(self.dtype)
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
        # protein_encoder is None when the ESM3 pre-encode cache is used (encoder
        # not built) — nothing to freeze on the protein side then.
        if self.protein_encoder is not None:
            for p in self.protein_encoder.model.parameters():
                p.requires_grad = False
        if self.go_encoder is not None:
            for p in self.go_encoder.parameters():
                p.requires_grad = False
        # GO projection is trainable; protein projection is trainable.
        _bk = "LoRA adapters" if getattr(self, "_has_lora", False) else "full backbone"
        logger.info("ESM3 and GO encoder frozen (RL trains projectors + %s)", _bk)

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
            if self._esm3_cache is not None:
                # Cache lookup replaces the live ESM3 forward. Build the SAME
                # per-batch-item `raw` list encode_sequences would return: for each
                # batch item, concatenate the cached features of its sequence(s)
                # (in batch_idx_map order). Sequences are keyed by the SAME
                # truncation the dataset applied. A miss is a hard error (the cache
                # must be complete) — never silently zero-fill.
                _per_item: list[list[torch.Tensor]] = [[] for _ in range(B)]
                for _seq_idx, _seq in enumerate(protein_sequences):
                    _k = self.esm3_cache_key(_seq)
                    _t = self._esm3_cache.get(_k)
                    if _t is None:
                        raise KeyError(
                            f"ESM3 cache miss for sequence (len={len(_seq)}, "
                            f"key={_k[:12]}…). Cache is incomplete — re-run "
                            f"precompute_esm3_cache.py over the full dataset."
                        )
                    _per_item[batch_idx_map[_seq_idx]].append(_t)
                raw = [
                    torch.cat(_per_item[i], dim=0) if _per_item[i]
                    else torch.zeros((0, int(self._esm3_cache_meta["embedding_dim"])))
                    for i in range(B)
                ]
            else:
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

    @staticmethod
    def _peft_name_to_hf(name: str) -> Optional[str]:
        """Translate a PEFT backbone param name to the clean HF name vLLM expects.

        PEFT renames a wrapped linear's weight to e.g.
        ``base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight``
        and adds adapter params ``...q_proj.lora_A.default.weight`` /
        ``...lora_B.default.weight``. vLLM's ``load_weights`` expects the original
        HF name ``model.layers.0.self_attn.q_proj.weight``.

        Returns the clean HF name for base/non-target params, or ``None`` for
        adapter-only params (which are never shipped to vLLM — the merged delta
        rides on ``base_layer.weight`` after ``merge_adapter()``).
        """
        if ".lora_A." in name or ".lora_B." in name or ".lora_embedding_" in name:
            return None
        # Strip the PEFT wrapper prefix: PeftModel.base_model.model -> original root.
        if name.startswith("base_model.model."):
            name = name[len("base_model.model."):]
        # Remove the ``.base_layer`` infix inserted around wrapped linears.
        name = name.replace(".base_layer.", ".")
        return name

    def lora_delta_map(self) -> dict:
        """Return {clean_hf_name: delta_weight} for every LoRA-target backbone module.

        Eager dict form — materializes ALL ~398 fp32 deltas at once. Prefer
        ``lora_delta_iter()`` in the colocate merge: holding the full dict for the
        whole load_weights loop is a large per-step fp32 transient that, with no
        empty_cache under colocate, fragments the allocator → reserved staircase →
        banned:1 (observed +11 GiB/step, 2026-06-18). Kept for the server path /
        tests where the dict is convenient.
        """
        return dict(self.lora_delta_iter())

    def lora_delta_iter(self):
        """Yield (clean_hf_name, delta_weight) one LoRA-target at a time.

        ``delta = sum_a scale_a * (B_a @ A_a)`` via PEFT's non-mutating
        ``get_delta_weight`` (fp32). Streaming so the caller can add it to the base,
        load it into vLLM, and free it before computing the next — bounding the
        per-step transient to one layer's delta instead of all 398. The frozen base
        is never mutated (avoids the bf16 merge/unmerge drift). Must be called with
        full (summoned) params under FSDP; in colocate the model is unsharded so
        it's a plain read.
        """
        if not self._has_lora:
            return
        for mod_name, module in self.backbone.named_modules():
            if not (hasattr(module, "base_layer") and hasattr(module, "get_delta_weight")):
                continue
            active = getattr(module, "active_adapters", [])
            if not active:
                continue
            hf_name = self._peft_name_to_hf(f"{mod_name}.base_layer.weight")
            if hf_name is None:
                continue
            delta = None
            for a in active:
                d = module.get_delta_weight(a)
                delta = d if delta is None else delta + d
            yield hf_name, delta

    def vllm_param_iter(self):
        """Yield (hf_name, param) for LLM backbone params only — used for vLLM weight sync.

        ESM3, GO encoder, protein_projection, and go_projection are excluded because
        vLLM receives pre-computed prompt_embeds and never runs the encoders/projectors.
        The backbone uses native HF parameter names (no 'backbone.' prefix).

        When LoRA is active, PEFT-renamed params are translated back to clean HF
        names and adapter params (lora_A/lora_B) are skipped — callers merge the
        adapter into ``base_layer.weight`` (``merge_adapter()``) before iterating
        so the yielded base weight already carries W_eff.
        """
        if self._has_lora:
            for name, param in self.backbone.named_parameters():
                hf_name = self._peft_name_to_hf(name)
                if hf_name is None:
                    continue
                yield hf_name, param
        else:
            for name, param in self.backbone.named_parameters():
                yield name, param

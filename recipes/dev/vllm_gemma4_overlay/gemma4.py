"""Gemma4 model for vLLM 0.10.1.

Ported from vLLM's gemma3.py with adaptations for Gemma4 architecture:
- Heterogeneous head dimensions (local=256, global=512)
- K=V for global attention layers (no v_proj)
- Partial rotary embeddings (25% on global layers)
- Per-layer scalar buffers
- HF weight prefix model.language_model.

Cache strategy: ALL layers use LOCAL dims (num_kv_heads=16, head_dim=256)
for the vLLM Attention module and KV cache, ensuring uniform cache specs.
- Local layers: use vLLM Attention normally.
- Global layers: bypass vLLM Attention for compute. Pack their real
  (4 heads × 512 dim) KV data as (8 heads × 256 dim) into the first 8
  of 16 cache head slots via reshape_and_cache_flash. Compute attention
  with F.scaled_dot_product_attention at real dims (4 heads, 512 dim).
  For decode, read from cache and unpack back to (4, 512).
"""
from collections.abc import Iterable
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from vllm.attention import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import GeluAndMul
# NOTE: NOT using vLLM's GemmaRMSNorm — it does x*(1+w) which is wrong for
# Gemma4. Using our Gemma4RMSNorm (x*w) defined below instead.
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.interfaces import SupportsLoRA, SupportsPP
from vllm.model_executor.models.utils import (AutoWeightsLoader,
                                               extract_layer_index,
                                               is_pp_missing_parameter,
                                               make_empty_intermediate_tensors_factory,
                                               make_layers, maybe_prefix)

logger = init_logger(__name__)


class Gemma4RMSNorm(nn.Module):
    """Gemma4-style RMSNorm: x * weight (NOT x * (1+weight) like GemmaRMSNorm).

    Gemma4 checkpoints store norm weights meant for direct multiplication.
    GemmaRMSNorm (used by Gemma1/2/3) adds 1 to the weights, which is wrong
    for Gemma4 where e.g. k_norm weight ≈ 0.06-0.12 encodes attention scaling.
    """

    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if self.with_scale:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        variance = x_float.pow(2).mean(-1, keepdim=True) + self.eps
        x_normed = x_float * torch.pow(variance, -0.5)
        if self.with_scale:
            out = x_normed * self.weight.float()
        else:
            out = x_normed
        return out.type_as(x)


class Gemma4MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_activation: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_activation != "gelu_pytorch_tanh":
            raise ValueError(
                "Gemma4 uses `gelu_pytorch_tanh` as the hidden activation. "
                f"Got `{hidden_activation}`.")
        self.act_fn = GeluAndMul(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Gemma4Attention(nn.Module):
    """Gemma4 attention with heterogeneous head dimensions.

    Local (sliding) layers: head_dim=256, num_kv_heads=16
    Global (full) layers: head_dim=512, num_kv_heads=4, k_eq_v=True

    Cache strategy: ALL layers use LOCAL dims (16 kv_heads, 256 head_dim)
    for the vLLM Attention module, ensuring uniform KV cache specs.

    Local layers: pass through to vLLM Attention normally (IPEX FlashAttention).
    Global layers: use a parallel KV cache with real dims (4 kv_heads, 512 head_dim)
      and call IPEX flash_attn_varlen_func directly. This avoids the slow
      per-sequence F.scaled_dot_product_attention loop and uses the fused SYCL
      kernel (which has pre-compiled XeTLA support for head_dim=512).
      The parallel cache reuses the same block_table from vLLM's block manager.
    """

    # Cache dims used by ALL layers (local layer dims)
    CACHE_NUM_KV_HEADS = 16  # before TP
    CACHE_HEAD_DIM = 256

    def __init__(self,
                 config,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 head_dim: int,
                 max_position_embeddings: int,
                 is_sliding: bool,
                 k_eq_v: bool = False,
                 partial_rotary_factor: float = 1.0,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "") -> None:
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.is_sliding = is_sliding
        self.is_global = not is_sliding
        self.k_eq_v = k_eq_v

        # Real (architectural) head dimensions for this layer type
        self.real_head_dim = head_dim
        self.real_num_heads = num_heads
        self.real_num_kv_heads = num_kv_heads

        # TP partitioning for REAL dims
        assert num_heads % tp_size == 0
        self.num_heads = num_heads // tp_size
        self.real_num_kv_heads_tp = max(1, num_kv_heads // tp_size)

        # Cache (uniform) dims with TP
        assert self.CACHE_NUM_KV_HEADS % tp_size == 0 or \
            tp_size % self.CACHE_NUM_KV_HEADS == 0
        self.cache_num_kv_heads_tp = max(1, self.CACHE_NUM_KV_HEADS // tp_size)

        # QKV split sizes (real dims)
        self.q_size = self.num_heads * self.real_head_dim
        self.kv_size = self.real_num_kv_heads_tp * self.real_head_dim
        # Scaling: 1.0 (NOT 1/sqrt(head_dim)). Gemma4 bakes attention scaling
        # into the k_norm learned weights (≈0.06-0.12), so no additional
        # scaling is needed in the attention computation.
        self.scaling = 1.0

        # QKV projection uses REAL head dimensions (matches HF weights)
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.real_head_dim,
            num_heads,
            num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            num_heads * self.real_head_dim,
            hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Norms use REAL head_dim.
        # Gemma4RMSNorm (x * w) NOT GemmaRMSNorm (x * (1+w)).
        # k_norm weights ≈ 0.06-0.12 encode attention scaling (HF scaling=1.0).
        self.q_norm = Gemma4RMSNorm(self.real_head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma4RMSNorm(self.real_head_dim, eps=config.rms_norm_eps)
        # v_norm: RMSNorm without learned scale (with_scale=False)
        self.v_norm = Gemma4RMSNorm(self.real_head_dim, eps=config.rms_norm_eps,
                                     with_scale=False)

        sliding_window = config.sliding_window if self.is_sliding else None

        # RoPE configuration
        rope_params = config.rope_parameters
        if self.is_sliding:
            rope_cfg = rope_params.get("sliding_attention", {})
            self.rope_theta = rope_cfg.get("rope_theta", 10000.0)
            rope_scaling = {"rope_type": "default"}
            rotary_dim = self.real_head_dim
        else:
            rope_cfg = rope_params.get("full_attention", {})
            self.rope_theta = rope_cfg.get("rope_theta", 1000000.0)
            rope_scaling = {"rope_type": "default"}
            rotary_dim = int(self.real_head_dim * partial_rotary_factor)

        # RoPE uses REAL head_dim
        self.rotary_emb = get_rope(
            self.real_head_dim,
            rotary_dim=rotary_dim,
            max_position=max_position_embeddings,
            base=self.rope_theta,
            is_neox_style=True,
            rope_scaling=rope_scaling,
        )

        # Attention module uses UNIFORM LOCAL cache dimensions for ALL layers.
        # For local layers, this matches their real dims.
        # For global layers, this is used only for cache allocation/management;
        # actual attention is computed manually.
        self.attn = Attention(self.num_heads,
                              self.CACHE_HEAD_DIM,  # 256
                              self.scaling,
                              num_kv_heads=self.cache_num_kv_heads_tp,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              per_layer_sliding_window=sliding_window,
                              prefix=f"{prefix}.attn")

        # For global layers: parallel KV cache with real dims (lazy-allocated)
        if self.is_global:
            self._global_kv_cache = None  # Allocated on first forward

    def _forward_local(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Local (sliding) layers: standard vLLM Attention path."""
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # q/k/v norms at real dims
        q = q.unflatten(-1, (self.num_heads, self.real_head_dim))
        q = self.q_norm(q)
        q = q.flatten(-2, -1)
        k = k.unflatten(-1, (self.real_num_kv_heads_tp, self.real_head_dim))
        k = self.k_norm(k)
        k = k.flatten(-2, -1)
        # v_norm: RMSNorm without learned scale
        v = v.unflatten(-1, (self.real_num_kv_heads_tp, self.real_head_dim))
        v = self.v_norm(v)
        v = v.flatten(-2, -1)

        # RoPE at real dims (v is not rotated)
        q, k = self.rotary_emb(positions, q, k)

        # Directly pass to vLLM Attention (dims match: 16/tp kv_heads, 256 head_dim)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    def _ensure_global_cache(self, device, dtype):
        """Lazy-allocate the parallel KV cache for global layers.

        Uses the same num_blocks and block_size as the main cache, but with
        real global dims (real_num_kv_heads_tp, real_head_dim=512) instead of
        uniform local dims (cache_num_kv_heads_tp, 256).

        Memory overhead: ~4% of total cache (10 global layers × 2048 bytes/token
        vs 60 layers × 4096 bytes/token in the main cache).
        """
        # Get num_blocks from the main cache allocated to this layer
        main_kv = self.attn.kv_cache[0]  # [2, num_blocks, block_size, kv_heads, head_dim]
        num_blocks = main_kv.shape[1]
        block_size = main_kv.shape[2]
        # Allocate parallel cache with real global dims
        self._global_kv_cache = torch.zeros(
            2, num_blocks, block_size,
            self.real_num_kv_heads_tp, self.real_head_dim,
            dtype=dtype, device=device,
        )
        logger.info(
            "Allocated global parallel cache: shape=%s, %.2f GiB",
            list(self._global_kv_cache.shape),
            self._global_kv_cache.nbytes / (1024**3),
        )

    def _forward_global(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Global (full) layers: IPEX FlashAttention with parallel cache.

        Uses a separate KV cache with real global dims (4/tp kv_heads, 512 head_dim)
        and calls the IPEX flash_attn_varlen_func kernel directly. This avoids the
        slow per-sequence F.scaled_dot_product_attention loop.

        The parallel cache reuses the same block_table from vLLM's block manager.
        """
        num_tokens = hidden_states.shape[0]
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # K=V: for global layers, v = raw k_proj output (BEFORE norm and RoPE).
        # v from QKV split already has k_proj weights loaded (k→v copy in load_weights).

        # q/k/v norms at real dims
        q = q.unflatten(-1, (self.num_heads, self.real_head_dim))
        q = self.q_norm(q)
        q = q.flatten(-2, -1)
        k = k.unflatten(-1, (self.real_num_kv_heads_tp, self.real_head_dim))
        k = self.k_norm(k)
        k = k.flatten(-2, -1)
        # v_norm: RMSNorm without learned scale (applied to raw k_proj output)
        v = v.unflatten(-1, (self.real_num_kv_heads_tp, self.real_head_dim))
        v = self.v_norm(v)
        v = v.flatten(-2, -1)

        # RoPE at real dims (partial rotary for global layers; v is not rotated)
        q, k = self.rotary_emb(positions, q, k)

        # --- Get forward context and metadata ---
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            # Profiling run — return zeros through o_proj for correct shape
            dummy = torch.zeros(
                num_tokens, self.num_heads * self.real_head_dim,
                dtype=hidden_states.dtype, device=hidden_states.device,
            )
            output, _ = self.o_proj(dummy)
            return output

        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[self.attn.layer_name]

        # --- Lazy-allocate parallel cache on first real forward ---
        if self._global_kv_cache is None:
            self._ensure_global_cache(hidden_states.device, hidden_states.dtype)

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Reshape K, V to (num_tokens, real_kv_heads_tp, real_head_dim)
        k_real = k.view(num_tokens, self.real_num_kv_heads_tp, self.real_head_dim)
        v_real = v.view(num_tokens, self.real_num_kv_heads_tp, self.real_head_dim)

        # --- Write to parallel cache via IPEX reshape_and_cache_flash ---
        global_key_cache = self._global_kv_cache[0]
        global_value_cache = self._global_kv_cache[1]

        import intel_extension_for_pytorch as ipex
        ipex.llm.modules.PagedAttention.reshape_and_cache_flash(
            k_real[:num_actual_tokens],
            v_real[:num_actual_tokens],
            global_key_cache,
            global_value_cache,
            attn_metadata.slot_mapping[:num_actual_tokens],
        )

        # --- Compute attention via IPEX flash_attn_varlen_func ---
        q_real = q.view(num_tokens, self.num_heads, self.real_head_dim)
        output = torch.zeros_like(q_real[:num_actual_tokens])

        # Build cu_seqlens_k from seq_lens if not available
        seq_lens = attn_metadata.seq_lens
        cu_seqlens_k = torch.zeros(seq_lens.shape[0] + 1, dtype=torch.int32,
                                   device=hidden_states.device)
        torch.cumsum(seq_lens, dim=0, out=cu_seqlens_k[1:])

        ipex.llm.modules.PagedAttention.flash_attn_varlen_func(
            output,                                    # out
            q_real[:num_actual_tokens].contiguous(),   # query
            global_key_cache,                          # key_cache
            global_value_cache,                        # value_cache
            attn_metadata.query_start_loc.int(),       # cu_seqlens_q
            cu_seqlens_k,                              # cu_seqlens_kv
            int(attn_metadata.max_query_len),          # max_seqlen_q
            int(attn_metadata.max_seq_len),            # max_seqlen_kv
            self.scaling,                              # scale
            True,                                      # is_causal (full attention)
            attn_metadata.block_table,                 # block_tables
            None,                                      # alibi_slopes
        )

        # Reshape output: (num_actual_tokens, num_heads, real_head_dim) → flat
        if num_actual_tokens < num_tokens:
            full_output = torch.zeros(
                num_tokens, self.num_heads * self.real_head_dim,
                dtype=hidden_states.dtype, device=hidden_states.device,
            )
            full_output[:num_actual_tokens] = output.reshape(
                num_actual_tokens, self.num_heads * self.real_head_dim,
            )
            attn_output_flat = full_output
        else:
            attn_output_flat = output.reshape(
                num_tokens, self.num_heads * self.real_head_dim,
            )

        result, _ = self.o_proj(attn_output_flat)
        return result

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        if self.is_global:
            return self._forward_global(positions, hidden_states)
        else:
            return self._forward_local(positions, hidden_states)


class Gemma4DecoderLayer(nn.Module):

    def __init__(
        self,
        config,
        layer_idx: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        is_sliding = config.layer_types[layer_idx] == "sliding_attention"

        # Per-layer head_dim and num_kv_heads
        if is_sliding:
            head_dim = config.head_dim
            num_kv_heads = config.num_key_value_heads
            k_eq_v = False
            partial_rotary_factor = 1.0
        else:
            head_dim = config.global_head_dim
            num_kv_heads = config.num_global_key_value_heads
            k_eq_v = config.attention_k_eq_v
            rope_cfg = config.rope_parameters.get("full_attention", {})
            partial_rotary_factor = rope_cfg.get("partial_rotary_factor", 0.25)

        self.self_attn = Gemma4Attention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            is_sliding=is_sliding,
            k_eq_v=k_eq_v,
            partial_rotary_factor=partial_rotary_factor,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = Gemma4MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_activation=config.hidden_activation,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = Gemma4RMSNorm(config.hidden_size,
                                             eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(config.hidden_size,
                                                      eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size,
                                                       eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size,
                                                        eps=config.rms_norm_eps)

        # Per-layer scalar (post-residual scaling, non-trainable)
        self.register_buffer(
            "layer_scalar",
            torch.ones(1, dtype=torch.float32),
            persistent=True,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Gemma4 uses layer_scalar to scale the ENTIRE output (residual included).
        # This means we CANNOT use the fused norm+residual-add pattern (which
        # stores residual separately and adds later). Instead, do explicit
        # residual additions so layer_scalar can multiply everything.

        # Reconstruct full hidden state from fused-norm split
        if residual is not None:
            hidden_states = hidden_states + residual

        # --- Attention block (matches HF Gemma4) ---
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # --- Feedforward block ---
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Scale ENTIRE output (residual + attn + FFN) by per-layer scalar
        hidden_states = hidden_states * self.layer_scalar.to(hidden_states.dtype)

        # Return with residual=None so the next layer and final norm
        # don't try to add a stale residual
        return hidden_states, None


@support_torch_compile
class Gemma4Model(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Gemma4DecoderLayer(
                config,
                layer_idx=extract_layer_index(prefix),
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers")
        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Normalize the embedding by sqrt(hidden_size) (Gemma convention)
        normalizer = self.config.hidden_size**0.5
        self.register_buffer("normalizer",
                             torch.tensor(normalizer),
                             persistent=False)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids) * self.normalizer

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for layer in self.layers[self.start_layer:self.end_layer]:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                **kwargs,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        # residual is None (already incorporated in decoder layers)
        if residual is not None:
            hidden_states = hidden_states + residual
        hidden_states = self.norm(hidden_states)
        return hidden_states

class Gemma4ForCausalLM(nn.Module, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # HF prefix remap: model.language_model.X → model.X
    # (only strip the inner "language_model." segment)
    _HF_STRIP = "language_model."

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        super().__init__()
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.model = Gemma4Model(vllm_config=vllm_config,
                                 prefix=maybe_prefix(prefix, "model"))
        # TEMPORARILY disable softcapping for debugging (raw logits are too large,
        # capping to 30.0 makes all tokens equal)
        self.logits_processor = LogitsProcessor(
            config.vocab_size, soft_cap=config.final_logit_softcapping)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds, **kwargs)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.model.embed_tokens, hidden_states,
                                       sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        """Load weights with HF prefix remapping and k_eq_v handling.

        HF weights: model.language_model.layers.0.self_attn.q_proj.weight
        After remap: model.layers.0.self_attn.q_proj.weight
        Our params:  model.layers.0.self_attn.qkv_proj.weight
        """
        # Collect k_proj weights for k_eq_v layers to copy into v shard later
        k_proj_cache: dict[int, torch.Tensor] = {}

        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        # Also include buffers (for layer_scalar)
        for bname, buf in self.named_buffers():
            if bname not in params_dict:
                params_dict[bname] = buf
        loaded_params: set[str] = set()

        skip_prefixes = [
            "model.vision_tower.", "model.multi_modal_projector.",
            "model.embed_vision.", "vision_tower.",
            "multi_modal_projector.", "embed_vision.",
        ]
        if self.config.tie_word_embeddings:
            skip_prefixes.append("lm_head.")

        for name, loaded_weight in weights:
            # Remap HF: model.language_model.X → model.X
            name = name.replace(self._HF_STRIP, "")

            # Skip vision/multimodal/lm_head weights
            if any(name.startswith(p) for p in skip_prefixes):
                continue

            # For k_eq_v global layers: v_proj doesn't exist in HF checkpoint.
            # Cache k_proj to copy into v shard after all weights loaded.
            if "k_proj" in name:
                layer_idx = self._extract_layer_idx(name)
                if layer_idx is not None and self._is_keqv_layer(layer_idx):
                    k_proj_cache[layer_idx] = loaded_weight.clone()

            for (param_name, shard_name, shard_id) in stacked_params_mapping:
                if shard_name not in name:
                    continue
                stacked_name = name.replace(shard_name, param_name)
                if stacked_name.endswith(".bias") and stacked_name not in params_dict:
                    continue
                if is_pp_missing_parameter(stacked_name, self):
                    continue
                if stacked_name not in params_dict:
                    continue
                param = params_dict[stacked_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(stacked_name)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remap FP8 kv-scale names
                remapped = maybe_remap_kv_scale_name(name, params_dict)
                if remapped is not None:
                    name = remapped
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        # Check for unloaded params
        all_params = set(params_dict.keys())
        unloaded = all_params - loaded_params
        unloaded = {p for p in unloaded if not p.startswith("model.vision_tower")}
        if unloaded:
            logger.warning("Unloaded params (%d): %s", len(unloaded),
                           sorted(list(unloaded))[:10])
        else:
            logger.info("All %d params loaded OK", len(loaded_params))

        # Copy k shard → v shard directly in QKV weight for k_eq_v layers.
        # After TP sharding, global QKV layout is [q_tp | k_tp | v_tp]:
        #   q_tp: num_heads/tp * head_dim (e.g., 8*512 = 4096)
        #   k_tp: num_kv_heads/tp * head_dim (e.g., 2*512 = 1024)
        #   v_tp: same as k_tp
        # We copy param[q_size+k_size : q_size+2*k_size] = param[q_size : q_size+k_size]
        for layer_idx in k_proj_cache:
            qkv_name = f"model.layers.{layer_idx}.self_attn.qkv_proj.weight"
            if qkv_name not in params_dict:
                continue
            param = params_dict[qkv_name]
            # Read the QKV dimensions from the attention module
            attn_mod = None
            for mod_name, mod in self.named_modules():
                if mod_name == f"model.layers.{layer_idx}.self_attn":
                    attn_mod = mod
                    break
            if attn_mod is None:
                continue
            q_size = attn_mod.q_size  # num_heads/tp * real_head_dim
            kv_size = attn_mod.kv_size  # num_kv_heads/tp * real_head_dim
            # Direct copy: k shard → v shard
            with torch.no_grad():
                k_shard = param.data[q_size:q_size + kv_size, :].clone()
                param.data[q_size + kv_size:q_size + 2 * kv_size, :] = k_shard
            logger.info("layer=%d k→v direct copy: q=%d kv=%d shape=%s",
                        layer_idx, q_size, kv_size, list(param.shape))
            loaded_params.add(qkv_name)

        return loaded_params

    def _extract_layer_idx(self, name: str) -> Optional[int]:
        parts = name.split(".")
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    pass
        return None

    def _is_keqv_layer(self, layer_idx: int) -> bool:
        if layer_idx >= len(self.config.layer_types):
            return False
        return (self.config.layer_types[layer_idx] == "full_attention"
                and self.config.attention_k_eq_v)

#!/usr/bin/env python3
"""Extract fixed-prompt eager-HF logits for Kimi-Linear parity checks."""

import argparse
import json
import importlib

import torch
import torch.nn.functional as F


def install_cpu_reference_shims() -> None:
    conv_module = importlib.import_module("fla.modules.conv.causal_conv1d")

    def eager_conv(
        x,
        weight=None,
        bias=None,
        residual=None,
        initial_state=None,
        output_final_state=False,
        activation=None,
        backend=None,
        cu_seqlens=None,
        **kwargs,
    ):
        if initial_state is not None or cu_seqlens is not None:
            raise ValueError("reference shim only supports a single unpadded prompt")
        width = weight.shape[-1]
        output = F.conv1d(
            x.transpose(1, 2),
            weight.unsqueeze(1),
            bias=bias,
            padding=width - 1,
            groups=x.shape[-1],
        ).transpose(1, 2)[:, : x.shape[1]]
        if activation in ("silu", "swish"):
            output = F.silu(output)
        return output, None

    conv_module.causal_conv1d = eager_conv

    kda_module = importlib.import_module("fla.ops.kda")
    gate_module = importlib.import_module("fla.ops.kda.gate")
    norm_module = importlib.import_module("fla.modules.fused_norm_gate")

    def eager_gate(g, a_log, head_k_dim, g_bias=None, beta=1.0, threshold=20.0):
        g = g.view(*g.shape[:-1], -1, head_k_dim).float()
        g = g + (g_bias.view(1, 1, -1, 1) if g_bias is not None else 0)
        scaled = g * beta
        softplus = torch.where(scaled > threshold, g, torch.log1p(scaled.exp()) / beta)
        return -a_log.float().exp() * softplus

    def eager_recurrent(
        q,
        k,
        v,
        g,
        beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        use_beta_sigmoid_in_kernel=False,
        allow_neg_eigval=False,
        **kwargs,
    ):
        dtype = v.dtype
        if use_qk_l2norm_in_kernel:
            q = q / torch.sqrt((q.float() * q.float()).sum(-1, keepdim=True) + 1e-6)
            k = k / torch.sqrt((k.float() * k.float()).sum(-1, keepdim=True) + 1e-6)
        if use_beta_sigmoid_in_kernel:
            beta = beta.sigmoid()
            if allow_neg_eigval:
                beta = beta * 2
        q, k, v, g, beta = [x.transpose(1, 2).float() for x in (q, k, v, g, beta)]
        batch, heads, tokens, key_dim = q.shape
        value_dim = v.shape[-1]
        state = torch.zeros(batch, heads, key_dim, value_dim) if initial_state is None else initial_state.float().clone()
        output = torch.empty(batch, heads, tokens, value_dim)
        for token in range(tokens):
            state = state * g[:, :, token, :, None].exp()
            delta = v[:, :, token] - torch.einsum("bhkv,bhk->bhv", state, k[:, :, token])
            state = state + torch.einsum("bhk,bhv->bhkv", beta[:, :, token, None] * k[:, :, token], delta)
            output[:, :, token] = torch.einsum("bhk,bhkv->bhv", q[:, :, token] * key_dim**-0.5, state)
        return output.transpose(1, 2).to(dtype), state if output_final_state else None

    kda_module.fused_kda_gate = eager_gate
    gate_module.fused_kda_gate = eager_gate
    kda_module.fused_recurrent_kda = eager_recurrent
    kda_module.chunk_kda = eager_recurrent

    def eager_norm(x, g, weight, bias=None, activation="swish", **kwargs):
        output = x.float() * torch.rsqrt((x.float() * x.float()).mean(-1, keepdim=True) + kwargs.get("eps", 1e-6))
        if weight is not None:
            output = output * weight
        if activation in ("swish", "silu"):
            output = output * g * torch.sigmoid(g)
        elif activation == "sigmoid":
            output = output * torch.sigmoid(g)
        return output.to(x.dtype)

    norm_module.rms_norm_gated = eager_norm


def dequantize_mxfp4(weight_packed: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
    """MXFP4 dequant: E2M1 4-bit values (2 packed per uint8 byte), E8M0 (power-
    of-2) per-32-element-group scale. This compressed_tensors==0.13.0 install
    only implements MXFP4Compressor.compress_weight, not decompress_weight
    (raises NotImplementedError("MXFP4 Decompression is currently not
    supported")) -- vLLM never needs dense dequant since its Marlin kernel
    consumes the packed format directly. Reimplemented here from the
    CompressionFormat.mxfp4_pack_quantized encoding documented in
    compressed_tensors/compressors/quantized_compressors/fp4_quantized.py
    (unpack_fp4_from_uint8, kE2M1ToFloat) and cross-checked against vLLM's own
    `weight_scale.view(torch.float8_e8m0fnu)` reinterpretation
    (marlin_utils_fp4.py) rather than re-deriving the exponent bias by hand.
    group_size=32 confirmed from CompressedTensorsW4A16Mxfp4 (weight-only,
    schemes/compressed_tensors_w4a16_mxfp4.py) and from this checkpoint's own
    shapes: a [3072,1792] packed tensor (1792*2=3584 unpacked cols) pairs with
    a [3072,112] scale tensor (3584/32=112 groups)."""
    from compressed_tensors.compressors.quantized_compressors.fp4_quantized import (
        kE2M1ToFloat,
    )

    m, n_packed = weight_packed.shape
    n = n_packed * 2
    group_size = 32

    a_flat = weight_packed.flatten()
    high = (a_flat & 0xF0) >> 4
    low = a_flat & 0x0F
    combined = torch.stack((low, high), dim=1).flatten()
    signs = (combined & 0x08).to(torch.bool)
    abs_vals = (combined & 0x07).to(torch.long)
    kE2M1 = kE2M1ToFloat.to(device=weight_packed.device)
    values = kE2M1[abs_vals] * torch.where(signs, -1.0, 1.0)
    unpacked = values.reshape(m, n).float()

    scale = weight_scale.view(torch.float8_e8m0fnu).float()  # [m, n // group_size]
    scale_expanded = scale.repeat_interleave(group_size, dim=1)
    return (unpacked * scale_expanded).to(torch.bfloat16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Aurora is")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    install_cpu_reference_shims()
    import transformers.utils as transformers_utils

    transformers_utils.auto_docstring = lambda obj=None, **kwargs: (lambda f: f) if obj is None else obj
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from compressed_tensors.linear.compressed_linear import CompressedLinear

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Force the remote-code modeling module to import/cache before from_pretrained
    # runs, so its _init_weights (isinstance(module, nn.Linear) -- CompressedLinear
    # IS an nn.Linear subclass) can be patched. HF's post-load
    # _initialize_missing_keys pass re-inits every nn.Linear it finds, including
    # CompressedLinear modules whose .weight was deliberately deleted by
    # CompressedLinear.from_linear() in favor of compressed params
    # (weight_packed/weight_scale/...). Skip those; this is a logits comparison
    # against real checkpoint weights, not training.
    def _patch_init_weights(cls):
        _orig_init_weights = cls._init_weights

        def _init_weights_skip_compressed(self, module):
            if isinstance(module, CompressedLinear):
                return
            return _orig_init_weights(self, module)

        cls._init_weights = _init_weights_skip_compressed

    # K3 nests a KimiLinearModel backbone (modeling_kimi_linear.py) inside the
    # outer conditional-generation wrapper (modeling_kimi_k3.py) -- both define
    # their own _init_weights, and HF calls whichever module actually contains
    # the CompressedLinear, so both must be patched.
    outer_cls = get_class_from_dynamic_module(
        "modeling_kimi_k3.KimiK3ForConditionalGeneration", args.model
    )
    _patch_init_weights(outer_cls)
    inner_cls = get_class_from_dynamic_module(
        "modeling_kimi_linear.KimiLinearForCausalLM", args.model
    )
    _patch_init_weights(inner_cls)
    import sys as _sys
    for _name, _mod in list(_sys.modules.items()):
        if _name.endswith("modeling_kimi_linear") and hasattr(_mod, "KimiPreTrainedModel"):
            _patch_init_weights(_mod.KimiPreTrainedModel)
        if _name.endswith("modeling_kimi_k3") and hasattr(_mod, "KimiK3PreTrainedModel"):
            _patch_init_weights(_mod.KimiK3PreTrainedModel)

    # K3's checkpoint stores self_attn.A_log as a flat [128] tensor, but every
    # OTHER per-head KDA tensor (dt_bias, b_proj, q/k/v_proj) is consistently
    # sized for num_heads=96 (config's own value), and vLLM's KDA module
    # allocates A_log as [1,1,96,1] and loads it via a plain narrow(0, 0, 96)
    # sharded_weight_loader with no shape assertion -- so production vLLM has
    # always silently read only elements [0:96] and dropped [96:128], at every
    # TP degree tested (1/2/4/8, since local_num_heads*tp_size == 96 in all
    # cases and no rank's shard window ever reaches index 96). Replicate that
    # exact truncation here so the HF reference sees what vLLM actually serves,
    # rather than tripping transformers' stricter shape-checked loader.
    import transformers.modeling_utils as modeling_utils

    _orig_load_parameter_into_model = modeling_utils._load_parameter_into_model

    def _load_parameter_into_model_truncate_a_log(model, param_name, tensor):
        if param_name.endswith("self_attn.A_log") and tensor.numel() != 96:
            tensor = tensor.reshape(-1)[:96].contiguous()
        return _orig_load_parameter_into_model(model, param_name, tensor)

    modeling_utils._load_parameter_into_model = _load_parameter_into_model_truncate_a_log

    # This compressed_tensors install (0.13.0) has no MXFP4 decompress_weight
    # (raises NotImplementedError -- vLLM's Marlin kernel never needs dense
    # dequant). Patch in the manual dequantizer above.
    #
    # Also: MXFP4PackedCompressor inherits NVFP4PackedCompressor.
    # compression_param_info() unmodified, and that method's returned dict has
    # ONLY "weight_packed" -- no "weight_scale" entry, even though every MXFP4
    # checkpoint tensor ships a weight_scale sibling on disk.
    # CompressedLinear.from_linear() registers exactly the params
    # compression_param_info() names (compressed_linear.py:81-85), so without
    # this patch weight_scale is never created as a module parameter slot at
    # all and the checkpoint's weight_scale tensor has nowhere to load into.
    from compressed_tensors.compressors.quantized_compressors.fp4_quantized import (
        MXFP4PackedCompressor,
    )

    _orig_compression_param_info = MXFP4PackedCompressor.compression_param_info

    def _mxfp4_compression_param_info(self, weight_shape, quantization_args=None):
        output = _orig_compression_param_info(self, weight_shape, quantization_args)
        group_size = 32
        output["weight_scale"] = (
            torch.Size((weight_shape[0], weight_shape[1] // group_size)),
            torch.uint8,
        )
        return output

    MXFP4PackedCompressor.compression_param_info = _mxfp4_compression_param_info

    def _mxfp4_decompress_weight(self, compressed_data, quantization_args=None):
        if "weight_scale" not in compressed_data:
            raise KeyError(
                f"weight_scale missing; compressed_data keys={list(compressed_data)}, "
                f"weight_packed.shape={compressed_data['weight_packed'].shape}"
            )
        return dequantize_mxfp4(
            compressed_data["weight_packed"], compressed_data["weight_scale"]
        )

    MXFP4PackedCompressor.decompress_weight = _mxfp4_decompress_weight

    # ROOT CAUSE of weight_scale disappearing (traced through 5+ layers of the
    # HF/compressed_tensors loading pipeline): CompressedTensorsHfQuantizer.
    # _process_model_before_weight_loading() calls apply_quantization_config()
    # (correctly converts nn.Linear -> CompressedLinear, registering
    # weight_packed + weight_scale param slots per compression_param_info())
    # and THEN unconditionally calls self.compressor.compress_model(model)
    # whenever quantization_status == COMPRESSED -- which is ALWAYS true for
    # an already-compressed checkpoint like this one (K3's own config.json
    # says "quantization_status": "compressed"). compress_model() re-derives
    # each module's params from its CURRENT state_dict via
    # quant_compressor.compress(), which looks for a key literally named
    # "weight" to trigger real compression (base.py:94) -- but at this point,
    # BEFORE any checkpoint weights are loaded, the module only has
    # weight_packed (empty/meta) and weight_scale (empty/meta), no "weight".
    # So compress() falls through to its `else` branch for both params
    # (base.py:126-134), which then explicitly DROPS weight_scale via
    # `if name.endswith("weight_scale") and self._skip_scale(): continue`
    # (base.py:131) -- and _skip_scale() returns True for `isinstance(self,
    # NVFP4PackedCompressor)`, which MXFP4PackedCompressor inherits without
    # override. compress_model() then deletes every existing param on the
    # module and re-registers only what compress() returned -- i.e. only
    # weight_packed survives. Confirmed via a full-model sweep: 288/288
    # CompressedLinear modules ended up missing weight_scale, and a hook on
    # _load_parameter_into_model showed it was never even called for
    # weight_scale (proving the drop happens before checkpoint loading, not
    # during it) -- a minimal from_linear()-only repro outside the full
    # from_pretrained pipeline registers weight_scale correctly, isolating the
    # bug to this specific pre-load re-compression pass. This pass appears to
    # exist for the run_compressed=False / QAT re-quantization path (see the
    # mirrored, correctly-gated decompress_model() call in
    # _process_model_after_weight_loading(), gated on `not run_compressed`)
    # but compress_model() itself has no such gate. Simplest correct fix for
    # loading an ALREADY-compressed checkpoint for inference (never
    # re-quantizing): skip the compress_model() call entirely.
    import transformers.quantizers.quantizer_compressed_tensors as _hf_ct_quantizer

    def _process_model_before_weight_loading_no_recompress(self, model, **kwargs):
        from compressed_tensors.quantization import apply_quantization_config

        apply_quantization_config(model, self.compressor.quantization_config, self.run_compressed)

    _hf_ct_quantizer.CompressedTensorsHfQuantizer._process_model_before_weight_loading = (
        _process_model_before_weight_loading_no_recompress
    )

    # K3's `quantization_config.ignore` regex list (self_attn/shared_experts/
    # mlp.(gate|up|down)_proj/lm_head/vision_tower/mm_projector) was written for
    # vLLM's own more permissive per-scheme matching. compressed_tensors'
    # apply_quantization_config wraps EVERY nn.Linear not matching `ignore` in
    # CompressedLinear regardless of whether the checkpoint actually shipped
    # packed params for it -- and this checkpoint has several small residual/
    # gate/norm Linears (mlp_res_proj, self_attention_res_proj,
    # output_attn_res_proj, block_sparse_moe.gate, routed_expert_*) stored as
    # plain BF16 .weight with no weight_packed sibling. Wrapping those crashes
    # both at _init_weights time (AttributeError: no .weight) and at direct
    # `.weight` access time (K3's _apply_attn_res reads .weight without
    # calling forward(), so CompressedLinear's lazy decompress never fires).
    # Extend `ignore` with every module prefix the checkpoint itself proves is
    # plain, rather than guessing a fixed name list.
    import re as _re
    from copy import deepcopy
    from pathlib import Path as _Path

    from safetensors import safe_open as _safe_open

    def _plain_module_prefixes(model_path: str) -> list:
        index_path = _Path(model_path) / "model.safetensors.index.json"
        if index_path.exists():
            keys = list(json.loads(index_path.read_text())["weight_map"].keys())
        else:
            with _safe_open(str(_Path(model_path) / "model.safetensors"), framework="pt") as f:
                keys = list(f.keys())
        weight_keys = [k for k in keys if k.endswith(".weight")]
        packed_prefixes = {k[: -len(".weight_packed")] for k in keys if k.endswith(".weight_packed")}
        plain_prefixes = sorted(
            {k[: -len(".weight")] for k in weight_keys} - packed_prefixes
        )
        return plain_prefixes

    import compressed_tensors.quantization as _ct_quant
    import compressed_tensors.quantization.lifecycle.apply as _ct_apply

    _orig_apply_quant_config = _ct_apply.apply_quantization_config

    def _apply_quantization_config_extend_ignore(model, config, run_compressed=False):
        if config is not None:
            extra = [f"re:^{_re.escape(p)}$" for p in _plain_module_prefixes(args.model)]
            config = deepcopy(config)
            config.ignore = list(config.ignore or []) + extra
        return _orig_apply_quant_config(model, config, run_compressed)

    # transformers' CompressedTensorsHfQuantizer does
    # `from compressed_tensors.quantization import apply_quantization_config`
    # INSIDE its method body (a fresh lookup on every call, not a stale
    # imported reference) -- that resolves against the `compressed_tensors.
    # quantization` PACKAGE's re-exported binding, not the
    # `lifecycle.apply` submodule attribute. Patch both; patching only the
    # submodule silently does nothing because the package-level name is a
    # separate binding created at package-import time.
    _ct_apply.apply_quantization_config = _apply_quantization_config_extend_ignore
    _ct_quant.apply_quantization_config = _apply_quantization_config_extend_ignore

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    # K3's language_model backbone (modeling_kimi_k3.py:919) is constructed
    # from `config.text_config`, a SEPARATE KimiLinearConfig object nested
    # inside the top-level KimiK3Config -- the `attn_implementation="eager"`
    # kwarg above only lands on the top-level config, so KimiMLAAttention
    # (which reads its own nested config's `_attn_implementation`) still
    # resolves to the checkpoint's default flash_attention_2 and crashes with
    # no flash-attn kernel installed on this CPU-only login node. Force both.
    for cfg in (model.config, getattr(model.config, "text_config", None)):
        if cfg is not None:
            cfg._attn_implementation = "eager"
    if hasattr(model, "language_model"):
        model.language_model.config._attn_implementation = "eager"
        for module in model.language_model.modules():
            if hasattr(module, "config") and hasattr(module.config, "_attn_implementation"):
                module.config._attn_implementation = "eager"
    model.eval()
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids
    with torch.inference_mode():
        logits = model(input_ids=input_ids, use_cache=False).logits[0, -1].float()
    values, indices = torch.topk(logits, args.top_k)
    print(json.dumps({
        "prompt": args.prompt,
        "input_ids": input_ids.tolist(),
        "top_ids": indices.tolist(),
        "top_logits": values.tolist(),
        "argmax": int(indices[0]),
    }))


if __name__ == "__main__":
    main()

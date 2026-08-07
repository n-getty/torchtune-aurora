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
        **kwargs,
    ):
        dtype = v.dtype
        if use_qk_l2norm_in_kernel:
            q = q / torch.sqrt((q.float() * q.float()).sum(-1, keepdim=True) + 1e-6)
            k = k / torch.sqrt((k.float() * k.float()).sum(-1, keepdim=True) + 1e-6)
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
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.config._attn_implementation = "eager"
    model.config.attn_implementation = "eager"
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

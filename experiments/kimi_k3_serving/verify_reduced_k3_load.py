#!/usr/bin/env python3
"""Construct a reduced K3 and run load_weights over its safetensors.

This is the de-risk step for the reduced-K3 harness: it catches every naming
and shape error BEFORE an allocation is spent on serving. verify_model_
construction.py only builds the module (and hardcodes the full 93-layer /
896-expert dims); this additionally drives the real load_weights path and
asserts that every parameter the model declares was actually populated.

The loader is strict (default_loader.py fails on any uninitialized param), so
"loads cleanly" is a meaningful bar -- but note it is a bar on NAMES and
SHAPES, not on numerics. A model built from random weights loads fine and
still computes nonsense; that is expected and fine, because the harness exists
to reproduce a decode-shape defect, not to produce good text.

Runs on a login node (meta device + 1-rank gloo, no XPU required) as long as
PYTHONPATH points at the vllm worktree:

    export PYTHONPATH=/flare/ModCon/ngetty/vllm-xpu-src:$PYTHONPATH
    python3 verify_reduced_k3_load.py /tmp/reduced-k3
"""

import argparse
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="reduced K3 checkpoint directory")
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="only construct the module, do not run load_weights",
    )
    args = parser.parse_args()

    from safetensors.torch import load_file

    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.models.kimi_k3 import KimiK3ForConditionalGeneration

    model_config = ModelConfig(
        model=args.model, trust_remote_code=False, dtype="bfloat16"
    )
    vllm_config = VllmConfig(model_config=model_config)
    text_config = model_config.hf_text_config

    assert model_config.hf_config.model_type == "kimi_k3"
    assert text_config.model_type == "kimi_linear"
    # The K3-only structures the harness exists to exercise. If any of these is
    # absent the reduced model is not a K3 surrogate and the run proves nothing.
    assert text_config.routed_expert_hidden_size is not None, "latent MoE missing"
    assert text_config.routed_expert_hidden_size < text_config.hidden_size
    assert text_config.hidden_act == "situ", text_config.hidden_act
    assert getattr(text_config, "attn_res_block_size", None) is not None
    kda_flags = [
        text_config.is_kda_layer(i) for i in range(text_config.num_hidden_layers)
    ]
    assert any(kda_flags) and not all(kda_flags), (
        f"need BOTH attention types, got kda={kda_flags}"
    )
    print(
        f"config OK: layers={text_config.num_hidden_layers} "
        f"kda={[i for i, v in enumerate(kda_flags) if v]} "
        f"mla={[i for i, v in enumerate(kda_flags) if not v]} "
        f"experts={text_config.num_experts} "
        f"latent={text_config.routed_expert_hidden_size}/{text_config.hidden_size}"
    )

    with NamedTemporaryFile(prefix="reduced_k3_", delete=True) as init_file:
        with set_current_vllm_config(vllm_config):
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=Path(init_file.name).as_uri(),
                local_rank=0,
                backend="gloo",
            )
            initialize_model_parallel(1, 1, backend="gloo")
            try:
                if args.skip_load:
                    with torch.device("meta"):
                        model = KimiK3ForConditionalGeneration(vllm_config=vllm_config)
                    declared = dict(model.named_parameters())
                    print(f"construction OK: {len(declared)} parameters on meta")
                    return 0

                # Build directly on CPU, NOT on meta + to_empty(). vLLM attaches
                # per-parameter loader metadata via set_weight_attrs (e.g.
                # A_log gets a `weight_loader`, kda.py:332), and to_empty()
                # REPLACES every Parameter object, silently dropping those
                # attributes -- load_weights then dies with "'Parameter' object
                # has no attribute 'weight_loader'". That failure is an artifact
                # of the scaffold, not a defect in the model or checkpoint.
                with torch.device("cpu"):
                    model = KimiK3ForConditionalGeneration(vllm_config=vllm_config)
                declared = dict(model.named_parameters())
                print(f"construction OK: {len(declared)} parameters on cpu")

                weights = load_file(str(Path(args.model) / "model.safetensors"))
                print(f"checkpoint has {len(weights)} tensors")

                loaded = model.load_weights(
                    (name, tensor) for name, tensor in weights.items()
                )
            finally:
                destroy_model_parallel()
                destroy_distributed_environment()

    loaded = set(loaded or ())
    declared_names = set(declared)
    missing = sorted(declared_names - loaded)
    extra = sorted(loaded - declared_names)

    print(f"load_weights returned {len(loaded)} loaded names")
    if missing:
        print(f"\nFAIL: {len(missing)} declared parameters were NOT loaded")
        for name in missing[:20]:
            print(f"    {name}")
        if len(missing) > 20:
            print(f"    ... and {len(missing) - 20} more")
        return 1
    if extra:
        print(f"\nnote: {len(extra)} loaded names are not plain parameters")
        for name in extra[:10]:
            print(f"    {name}")

    print("\nREDUCED_K3_LOAD_PASS - every declared parameter was populated")
    print("Names and shapes are correct. This says nothing about numerics:")
    print("the weights are random, so the model will emit garbage by design.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

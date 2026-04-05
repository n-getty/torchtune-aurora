# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from torchtune.models.gemma4._component_builders import gemma4
from torchtune.models.gemma4._tokenizer import Gemma4Tokenizer
from torchtune.modules import TransformerDecoder


def gemma4_31b() -> TransformerDecoder:
    """
    Builder for creating the Gemma 4 31B model.

    Returns:
        TransformerDecoder: Instantiation of Gemma 4 31B model.
    """
    return gemma4(
        vocab_size=262_144,
        num_layers=60,
        num_heads=32,
        embed_dim=5376,
        intermediate_dim=21504,
        max_seq_len=8192,
        local_head_dim=256,
        local_num_kv_heads=16,
        local_rope_base=10_000.0,
        sliding_window_size=1024,
        global_head_dim=512,
        global_num_kv_heads=4,
        global_rope_base=1_000_000.0,
        global_partial_rotary_factor=0.25,
        global_k_eq_v=True,
        attn_dropout=0.0,
        norm_eps=1e-6,
        final_capping_value=30.0,
    )


def gemma4_tokenizer(
    tokenizer_json_path: str,
    tokenizer_config_json_path: str,
    max_seq_len: Optional[int] = None,
    truncation_type: str = "right",
) -> Gemma4Tokenizer:
    """
    Builder for the Gemma 4 tokenizer.

    Args:
        tokenizer_json_path: Path to tokenizer.json
        tokenizer_config_json_path: Path to tokenizer_config.json
        max_seq_len: Maximum sequence length. Default: None
        truncation_type: Truncation type. Default: "right"

    Returns:
        Gemma4Tokenizer: Instantiation of the Gemma 4 tokenizer.
    """
    return Gemma4Tokenizer(
        tokenizer_json_path=tokenizer_json_path,
        tokenizer_config_json_path=tokenizer_config_json_path,
        max_seq_len=max_seq_len,
        truncation_type=truncation_type,
    )

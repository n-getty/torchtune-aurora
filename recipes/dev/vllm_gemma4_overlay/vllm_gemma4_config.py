"""Custom Gemma4TextConfig for vLLM 0.10.1.

transformers 4.57.6 has no Gemma4 config class. This provides one that
vLLM's _CONFIG_REGISTRY can use to load google/gemma-4-31B config.json.

The HF config.json has top-level model_type="gemma4" with all text model
fields nested inside text_config. This class flattens them.
"""
import json
import os
from transformers import PretrainedConfig


class Gemma4TextConfig(PretrainedConfig):
    model_type = "gemma4_text"

    def __init__(
        self,
        # Text model params (from text_config in config.json)
        vocab_size=262144,
        hidden_size=5376,
        intermediate_size=21504,
        num_hidden_layers=60,
        num_attention_heads=32,
        num_key_value_heads=16,
        num_global_key_value_heads=4,
        head_dim=256,
        global_head_dim=512,
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=262144,
        rms_norm_eps=1e-6,
        attention_bias=False,
        attention_dropout=0.0,
        attention_k_eq_v=True,
        sliding_window=1024,
        final_logit_softcapping=30.0,
        tie_word_embeddings=True,
        layer_types=None,
        rope_parameters=None,
        # Passthrough
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_global_key_value_heads = num_global_key_value_heads
        self.head_dim = head_dim
        self.global_head_dim = global_head_dim
        self.hidden_activation = hidden_activation
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.attention_k_eq_v = attention_k_eq_v
        self.sliding_window = sliding_window
        self.final_logit_softcapping = final_logit_softcapping

        # Gemma convention: scaling = sqrt(head_dim)
        self.query_pre_attn_scalar = head_dim

        # Default layer types: 5 sliding + 1 full, repeated 10x
        if layer_types is None:
            pattern = ["sliding_attention"] * 5 + ["full_attention"]
            self.layer_types = pattern * 10
        else:
            self.layer_types = layer_types

        # RoPE parameters (nested dict)
        if rope_parameters is None:
            self.rope_parameters = {
                "sliding_attention": {
                    "rope_theta": 10000.0,
                    "rope_type": "default",
                },
                "full_attention": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1000000.0,
                    "rope_type": "proportional",
                },
            }
        else:
            self.rope_parameters = rope_parameters

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load from config.json, extracting text_config fields."""
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        if os.path.isfile(config_path):
            with open(config_path) as f:
                config_dict = json.load(f)

            # Gemma4 wraps text config inside top-level multimodal config
            text_config = config_dict.get("text_config", {})

            # Merge top-level tie_word_embeddings if not in text_config
            if "tie_word_embeddings" not in text_config:
                text_config["tie_word_embeddings"] = config_dict.get(
                    "tie_word_embeddings", True
                )

            # Filter to only known init params
            import inspect
            valid_params = set(inspect.signature(cls.__init__).parameters.keys())
            valid_params.discard("self")
            valid_params.discard("kwargs")
            filtered = {k: v for k, v in text_config.items() if k in valid_params}

            config = cls(**filtered)

            # Set architectures for vLLM model dispatch
            config.architectures = ["Gemma4ForCausalLM"]

            return config

        # Fallback to standard HF loading
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

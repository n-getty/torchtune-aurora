from torchtune.models.qwen3._tokenizer import Qwen3Tokenizer
from torchtune.models.qwen3_moe._component_builders import qwen3_moe
from torchtune.modules.transformer import TransformerDecoder


def qwen3_30b_a3b() -> TransformerDecoder:
    """Qwen3-30B-A3B: 48 layers, 128 experts/layer, top-8, ~3.3B active params."""
    return qwen3_moe(
        vocab_size=151936,
        num_layers=48,
        num_heads=32,
        num_kv_heads=4,
        embed_dim=2048,
        moe_intermediate_dim=768,
        num_experts=128,
        experts_per_token=8,
        max_seq_len=40960,
        head_dim=128,
        attn_dropout=0.0,
        norm_eps=1e-6,
        rope_base=1_000_000.0,
        tie_word_embeddings=False,
        norm_topk_prob=True,
    )


def qwen3_moe_tokenizer(
    path: str,
    merges_file: str,
    max_seq_len: int = 40960,
) -> Qwen3Tokenizer:
    return Qwen3Tokenizer(
        path=path,
        merges_file=merges_file,
        max_seq_len=max_seq_len,
    )

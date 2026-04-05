# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Optional

from torchtune.data import Message, truncate
from torchtune.modules.transforms import Transform
from torchtune.modules.transforms.tokenizers._hf_tokenizer import HuggingFaceBaseTokenizer
from torchtune.modules.transforms.tokenizers._utils import ModelTokenizer


class Gemma4Tokenizer(ModelTokenizer, Transform):
    """
    Tokenizer for Gemma 4 models using the HuggingFace tokenizer.json format.

    Wraps HuggingFaceBaseTokenizer and adds the properties required by torchtune
    recipes (pad_id, stop_tokens, vocab_size).

    Args:
        tokenizer_json_path (str): Path to tokenizer.json
        tokenizer_config_json_path (str): Path to tokenizer_config.json
        max_seq_len (Optional[int]): Maximum sequence length. Default: None
        truncation_type (str): Truncation type ("left" or "right"). Default: "right"
    """

    def __init__(
        self,
        tokenizer_json_path: str,
        tokenizer_config_json_path: str,
        max_seq_len: Optional[int] = None,
        truncation_type: str = "right",
    ):
        self._base = HuggingFaceBaseTokenizer(
            tokenizer_json_path=tokenizer_json_path,
            tokenizer_config_json_path=tokenizer_config_json_path,
        )
        self.max_seq_len = max_seq_len
        self.truncation_type = truncation_type
        self.special_tokens = {}

    @property
    def eos_id(self) -> int:
        return self._base.eos_id

    @property
    def bos_id(self) -> int:
        return self._base.bos_id

    @property
    def pad_id(self) -> int:
        return 0  # Gemma convention: pad_id = 0

    @property
    def stop_tokens(self) -> list[int]:
        return [self.eos_id]

    @property
    def vocab_size(self) -> int:
        return self._base.tokenizer.get_vocab_size()

    def encode(
        self, text: str, add_bos: bool = True, add_eos: bool = True
    ) -> list[int]:
        return self._base.encode(text, add_bos=add_bos, add_eos=add_eos)

    def decode(self, token_ids: list[int]) -> str:
        return self._base.decode(token_ids)

    def tokenize_messages(
        self,
        messages: list[Message],
        **kwargs: dict[str, Any],
    ) -> tuple[list[int], list[bool]]:
        """
        Tokenize messages for Gemma 4. Uses a simple concatenation approach
        suitable for GRPO training (prompts are plain text, not chat-formatted).
        """
        tokens = []
        mask = []
        for message in messages:
            content = message.text_content
            toks = self._base.encode(content, add_bos=len(tokens) == 0, add_eos=False)
            tokens.extend(toks)
            mask.extend([message.masked] * len(toks))

        # Add EOS
        tokens.append(self.eos_id)
        mask.append(False)

        if self.max_seq_len is not None:
            tokens = truncate(
                tokens=tokens,
                max_seq_len=self.max_seq_len,
                eos_id=self.eos_id,
                truncation_type=self.truncation_type,
            )
            mask = truncate(
                tokens=mask,
                max_seq_len=self.max_seq_len,
                eos_id=True,
                truncation_type=self.truncation_type,
            )

        return tokens, mask

    def __call__(
        self, sample: dict[str, Any], **kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Process a sample with a 'messages' field."""
        messages = sample.pop("messages")
        tokens, mask = self.tokenize_messages(messages, **kwargs)
        sample["tokens"] = tokens
        sample["mask"] = mask
        return sample

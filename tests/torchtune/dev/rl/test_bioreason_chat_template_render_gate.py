# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Regression guard for the missing-``chat_template.jinja`` defect (2026-09-16).

19 of the 20 v8 SFT snapshots shipped without ``chat_template.jinja``. HF then falls
back to the ``chat_template`` key embedded in ``tokenizer_config.json``, which is a
*different* template -- upstream Qwen's, which collapses any non-string message content
to ``''``. GRPO's dataset passes a content *list*, so the rendered training prompt
becomes an empty user turn: no protein, no GO graph, no question. It does not crash;
it trains on nothing.

It is also invisible to eval, which builds prompts via ``build_native_input_ids`` and
never touches a chat template -- so a checkpoint can score correctly and still be
unusable as a GRPO base.

These tests are CPU-only and load no model. They pin the three things that let the
defect exist and the one thing that catches it:

1. the dataset really does pass a non-string content list (if that ever changes to a
   plain string, the fallback template stops being dangerous and this guard is moot);
2. the fallback template really does drop such content (the assumption that the two
   templates were interchangeable was the whole trap);
3. the gate script rejects a checkpoint whose template drops it, and accepts one whose
   template keeps it.
"""

import json
import os
import subprocess
import sys

import pytest

REPO = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
GATE = os.path.join(
    REPO, "experiments", "bioreason", "check_grpo_base_prompt_render.py"
)

# The exact template shipped in the good snapshots emits these two markers.
PROTEIN_PAD = "<|protein_pad|>"
GO_PAD = "<|go_graph_pad|>"

# A minimal stand-in for each of the two real templates. We do not copy the 32B
# snapshots' full jinja (huge, and the point is the *behavioural* difference), only
# the branch that distinguishes them.
GOOD_TEMPLATE = (
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message.role + '\\n' }}"
    "{%- if message.content is string %}{{- message.content }}"
    "{%- else %}"
    "{%- for part in message.content %}"
    "{%- if part.type == 'protein' %}{{- 'Protein: " + PROTEIN_PAD + "\\n\\n' }}"
    "{%- elif part.type == 'go_graph' %}{{- 'GO graph: " + GO_PAD + "\\n\\n' }}"
    "{%- elif part.type == 'text' %}{{- part.text }}"
    "{%- endif %}{%- endfor %}"
    "{%- endif %}"
    "{{- '<|im_end|>\\n' }}{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
)

# The dangerous one: identical except non-string content becomes ''.
BAD_TEMPLATE = (
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message.role + '\\n' }}"
    "{%- if message.content is string %}{%- set content = message.content %}"
    "{%- else %}{%- set content = '' %}{%- endif %}"
    "{{- content }}{{- '<|im_end|>\\n' }}{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
)


def test_dataset_passes_non_string_content_list():
    """The defect only bites because the user message content is a list, not a str.

    Pinning this by source inspection rather than by importing the dataset (which
    drags in ESM3 and a tokenizer). If someone flattens this to a plain string the
    fallback template becomes harmless -- and this test should be revisited, not
    silently deleted.
    """
    src = os.path.join(REPO, "torchtune", "dev", "bioreason", "dataset.py")
    with open(src) as f:
        text = f.read()
    assert '"type": "protein"' in text, (
        "dataset.py no longer builds a content-list user message with a protein part; "
        "re-derive whether the chat-template fallback is still dangerous"
    )
    assert '"type": "go_graph"' in text
    assert "apply_chat_template" in text


def _render(template, messages):
    from jinja2 import Environment
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    return env.from_string(template).render(
        messages=messages, add_generation_prompt=True
    )


MESSAGES = [
    {"role": "system", "content": "You are an expert in protein function prediction."},
    {
        "role": "user",
        "content": [
            {"type": "protein"},
            {"type": "go_graph"},
            {"type": "text", "text": "__BODY__"},
        ],
    },
]


def test_the_two_templates_are_not_interchangeable():
    """The assumption that cost us: 'the .jinja file is just a copy of the key'."""
    pytest.importorskip("jinja2")
    good = _render(GOOD_TEMPLATE, MESSAGES)
    bad = _render(BAD_TEMPLATE, MESSAGES)

    for marker in (PROTEIN_PAD, GO_PAD, "__BODY__"):
        assert marker in good, f"good template dropped {marker}"
        assert marker not in bad, f"bad template unexpectedly kept {marker}"

    # And the failure mode is specifically an EMPTY user turn -- not a crash, not a
    # partial prompt. That silence is why it went unnoticed.
    assert "<|im_start|>user\n<|im_end|>" in bad


def _write_fake_ckpt(tmp_path, name, *, jinja_template, embedded_template):
    """A directory AutoTokenizer can load, with a controllable chat template."""
    d = tmp_path / name
    d.mkdir()
    cfg = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "chat_template": embedded_template,
    }
    (d / "tokenizer_config.json").write_text(json.dumps(cfg))
    # Minimal WordLevel tokenizer.json -- the gate only renders, never tokenizes text.
    tok = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {"type": "WordLevel", "vocab": {"a": 0}, "unk_token": "a"},
    }
    (d / "tokenizer.json").write_text(json.dumps(tok))
    if jinja_template is not None:
        (d / "chat_template.jinja").write_text(jinja_template)
    return d


@pytest.mark.parametrize(
    "has_jinja,expect_rc",
    [
        (True, 0),  # good template present -> gate passes
        (False, 1),  # falls back to the content-dropping one -> gate must FAIL
    ],
    ids=["jinja_present_passes", "jinja_missing_fails"],
)
def test_gate_accepts_good_ckpt_and_rejects_empty_prompt_ckpt(
    tmp_path, has_jinja, expect_rc
):
    """The gate must be exercised against a BAD input, not only a good one.

    A checker only validated on the passing case can be a no-op and nobody notices.
    """
    pytest.importorskip("transformers")
    pytest.importorskip("jinja2")
    assert os.path.exists(GATE), f"gate script missing: {GATE}"

    ckpt = _write_fake_ckpt(
        tmp_path,
        "ckpt",
        jinja_template=GOOD_TEMPLATE if has_jinja else None,
        embedded_template=BAD_TEMPLATE,
    )
    proc = subprocess.run(
        [sys.executable, GATE, str(ckpt)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == expect_rc, (
        f"gate rc={proc.returncode}, expected {expect_rc}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    if expect_rc:
        assert "FAIL" in proc.stdout
        assert "EMPTY prompts" in proc.stdout
    else:
        assert "OK" in proc.stdout

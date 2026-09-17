# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Regression guards for BioReason no-grad forward batch selection."""

from pathlib import Path


RECIPE = (
    Path(__file__).parents[4] / "recipes" / "dev" / "grpo_bioreason_distributed_xpu.py"
)


def test_setup_reads_ref_forward_batch_size():
    source = RECIPE.read_text()
    setup = source.split("def setup", 1)[1].split("def ", 1)[0]

    assert "self._ref_forward_batch_size = cfg.get(" in setup
    assert '"ref_forward_batch_size", cfg.forward_batch_size' in setup


def test_generate_trajectory_uses_ref_forward_batch_size_for_no_grad_forwards():
    source = RECIPE.read_text()
    method = source.split("def generate_trajectory", 1)[1].split("def ", 1)[0]

    assert "ref_fwd_bs = self._ref_forward_batch_size" in method
    assert "range(0, num_seqs, ref_fwd_bs)" in method
    assert "fwd_bs = self._forward_batch_size" not in method


def test_generate_trajectory_trims_to_hsdp_shard_group():
    source = RECIPE.read_text()
    method = source.split("def generate_trajectory", 1)[1].split("def ", 1)[0]

    assert "self._gloo_dp_shard_pg" in method
    assert 'self._dp_mesh.get_group("dp_shard")' not in method
    assert "trim_query_responses_to_global_max(" in method


def test_chunk_width_trim_is_opt_in_and_covers_all_chunked_forwards():
    source = RECIPE.read_text()
    setup = source.split("def setup", 1)[1].split("def ", 1)[0]
    generate = source.split("def generate_trajectory", 1)[1].split("def ", 1)[0]
    grpo_step = source.split("def grpo_step", 1)[1].split("def ", 1)[0]

    assert 'os.environ.get("TORCHTUNE_TRIM_CHUNK_WIDTH", "0") == "1"' in setup
    assert generate.count("get_right_padded_response_length(") == 2
    assert generate.count("pad_response_logprobs(") == 2
    assert generate.count("get_descending_response_chunk_ranges(") == 2
    assert "responses.clone()" in generate
    assert "_chunk_total_length" in generate
    assert grpo_step.count("get_right_padded_response_length(") == 1
    assert grpo_step.count("pad_response_logprobs(") == 1
    assert grpo_step.count("get_descending_response_chunk_ranges(") == 1
    assert "_chunk_pi_logprobs[_row]" in grpo_step
    assert "trajectory.ref_logprobs[_rows, :_chunk_response_length]" in grpo_step
    assert "padding_masks=~_chunk_padding_masks" in grpo_step
    assert "_chunk_weight = len(_rows) / total_seqs" in grpo_step
    assert "_c_loss\n                            * _chunk_weight" in grpo_step

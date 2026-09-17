# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Regression guard for BioReason diagnostics process-group scoping."""

from pathlib import Path


RECIPE = (
    Path(__file__).parents[4] / "recipes" / "dev" / "grpo_bioreason_distributed_xpu.py"
)


def test_hsdp_diagnostics_reduce_over_replicas_not_world():
    source = RECIPE.read_text()
    method = source.split("def _log_bioreason_diagnostics", 1)[1].split("def ", 1)[0]

    assert 'self._dp_mesh.get_group("dp_replicate")' in method
    assert "group=pg" in method


def test_diagnostics_report_response_length_cdf():
    source = RECIPE.read_text()
    method = source.split("def _log_bioreason_diagnostics", 1)[1].split("def ", 1)[0]

    assert "BIOREASON_LENGTH_CDF" in method
    assert "length_cdf_counts" in method
    assert "group=pg" in method

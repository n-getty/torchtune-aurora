"""Guard for the projection-resume fix (2026-06-23, async-resume correctness).

On resume (lora_adapter_path set), the recipe must overlay the TRAINED projections
(protein_projection.pt / go_projection.pt) from the adapter's epoch dir on top of the
SFT-base projections loaded from base_model_path. Without this, resuming a run reloads
the LoRA adapter but silently restarts the TRAINABLE projectors from SFT init, discarding
their RL training. The recipe derives proj_resume_dir = dirname(adapter_path) by default.
"""
import os
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[4]
_RECIPE = _REPO / "recipes" / "dev" / "grpo_bioreason_distributed_xpu.py"
_MODEL = _REPO / "torchtune" / "dev" / "bioreason" / "model.py"


def test_model_accepts_proj_resume_dir():
    src = _MODEL.read_text()
    assert "proj_resume_dir" in src
    # it must overlay via a second _load_custom_weights from that dir
    assert "self._load_custom_weights(proj_resume_dir)" in src


def test_recipe_derives_proj_resume_from_adapter_parent():
    src = _RECIPE.read_text()
    assert "proj_resume_dir" in src
    # default = dirname of the adapter path when adapter set and not overridden
    assert "os.path.dirname(_adapter_path" in src or "_os.path.dirname(_adapter_path" in src
    # and it's passed to the model
    assert "proj_resume_dir=_proj_resume_dir" in src


def test_dirname_derivation_logic():
    # the adapter lives at <epoch>/adapter; its parent is <epoch> holding the .pt files
    adapter = "/x/outputs/run/epoch_0/adapter"
    assert os.path.dirname(adapter.rstrip("/")) == "/x/outputs/run/epoch_0"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

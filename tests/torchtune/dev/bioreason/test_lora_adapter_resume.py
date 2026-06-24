"""CPU-safe guard for the BioReason LoRA adapter-resume path (4N→8N continue).

To continue a multi-node HSDP run at a larger node count, the saved LoRA adapter
must be LOADED into the policy on the resume run — otherwise an "8N resume" silently
restarts from the gaussian init (zero learning carried over). This pins the wiring:
  - BioReasonModel.__init__ accepts adapter_path and loads adapter_model.safetensors
    into the existing PEFT adapter (set_peft_model_state_dict), erroring on a missing
    file (no silent fresh-init on a typo'd path).
  - the recipe reads cfg.lora_adapter_path and passes it to the POLICY only (ref stays
    the frozen full SFT model).
  - the HSDP launcher forwards RESUME_ADAPTER → lora_adapter_path and SAVE_EVERY_N_STEPS
    → save_every_n_steps.

Source/signature inspection only (a live round-trip needs a PEFT model + checkpoint).
"""
import inspect
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[4]


def test_bioreason_model_ctor_accepts_adapter_path():
    from torchtune.dev.bioreason.model import BioReasonModel
    assert "adapter_path" in inspect.signature(BioReasonModel.__init__).parameters


def test_model_loads_adapter_or_errors_loudly():
    src = (_REPO / "torchtune" / "dev" / "bioreason" / "model.py").read_text()
    # loads the saved safetensors into the existing adapter, not from_pretrained
    assert "set_peft_model_state_dict" in src
    assert "adapter_model.safetensors" in src
    # a missing adapter file must raise, never silently fall back to gaussian init
    assert "FileNotFoundError" in src


def test_recipe_wires_adapter_path_to_policy():
    src = (_REPO / "recipes" / "dev" / "grpo_bioreason_distributed_xpu.py").read_text()
    assert 'cfg.get("lora_adapter_path"' in src
    assert "adapter_path=_adapter_path" in src


def test_hsdp_launcher_forwards_save_and_resume():
    src = (_REPO / "experiments" / "bioreason" / "run_bioreason_Nnode_hsdp.sh").read_text()
    assert "lora_adapter_path=${RESUME_ADAPTER}" in src
    assert "save_every_n_steps=${SAVE_EVERY_N_STEPS}" in src


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

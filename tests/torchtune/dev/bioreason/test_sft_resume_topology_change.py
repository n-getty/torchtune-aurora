"""CPU-safe guard for resuming BioReason SFT into a DIFFERENT data_parallel topology
(e.g. 4N/dp_degree=48 -> 16N/dp_degree=192).

torchdata's StatefulDistributedSampler.state_dict() only saves a raw `yielded` count
into the CURRENT run's per-rank partition — torch's DistributedSampler derives that
partition from num_replicas+seed+epoch, so restoring `yielded` under a different
num_replicas silently replays a different, wrong subset of the data rather than "the
same training position". `bioreason_resume_reset_dataloader=True` must skip the
dataloader restore while still restoring model/optimizer/global_step normally.
"""
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
_RECIPE = _REPO / "recipes" / "dev" / "sft_bioreason_distributed_xpu.py"


def test_recipe_reads_reset_dataloader_flag():
    src = _RECIPE.read_text()
    assert "bioreason_resume_reset_dataloader" in src
    assert 'cfg.get("bioreason_resume_reset_dataloader"' in src


def test_reset_flag_skips_dataloader_load_state_dict():
    src = _RECIPE.read_text()
    # the skip branch must come before the normal restore attempt in source order
    skip_idx = src.index("_bioreason_resume_reset_dataloader:")
    # find the setup()-time branch (not the __init__ assignment)
    setup_skip_idx = src.index("if self._bioreason_resume_reset_dataloader:")
    restore_idx = src.index('elif blob.get("dataloader") is not None:')
    assert setup_skip_idx < restore_idx
    assert skip_idx < restore_idx


def test_reset_flag_does_not_gate_optimizer_or_step_restore():
    src = _RECIPE.read_text()
    # optimizer state restore must appear BEFORE the reset-dataloader conditional,
    # i.e. unconditional on the new flag
    opt_idx = src.index("training.load_from_full_optimizer_state_dict")
    flag_idx = src.index("if self._bioreason_resume_reset_dataloader:")
    assert opt_idx < flag_idx
    # global_step restore must appear AFTER, and not be inside the dataloader if/elif
    step_idx = src.index("self.global_step = int(blob.get(training.STEPS_KEY")
    assert step_idx > flag_idx


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))

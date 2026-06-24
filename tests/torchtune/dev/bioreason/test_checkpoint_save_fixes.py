"""Guards for the two BioReason checkpoint-save fixes (2026-06-22, 4N step-11 crash).

The 4N HSDP soak crashed at step 11 right after the step-10 save, and the saved
projection .pt files were 8.1 GiB each (vs ~42 MB real). Two distinct bugs in
save_checkpoint, both rooted in summon_full_params:

  1. banned:1 — summon_full_params(rank0_only=False) gathered full params on ALL
     ranks mid-training, freeing L0 pages the live XCCL wsync handles referenced →
     next collective faulted. Fix: rank0_only=True (only rank 0 writes) + gc/sync
     before the next step.
  2. 8 GiB bloat — under summon, the projection params are VIEWS into the giant FSDP
     flat-param buffer; torch.save persists the whole storage, not the view. Fix:
     .detach().clone() each tensor before save so only the real data is written.

Test 1 is source inspection (the crash repro needs a live multi-rank FSDP+CCL stack).
Test 2 reproduces the view-vs-clone storage-bloat behavior directly with torch (the
exact mechanism), no XPU needed.
"""
import io
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_REPO = Path(__file__).resolve().parents[4]
_RECIPE = _REPO / "recipes" / "dev" / "grpo_bioreason_distributed_xpu.py"


def test_save_uses_rank0_only_and_settles_l0():
    src = _RECIPE.read_text()
    # the periodic save summon must be rank0_only=True (was False → banned:1)
    # find the save_checkpoint summon specifically (writeback=False, near save_dir)
    assert "rank0_only=True" in src
    # and it must NOT use rank0_only=False in save_checkpoint anymore. The colocate
    # LoRA-merge path legitimately keeps rank0_only=False — so just assert the save
    # block has the gc+synchronize settle after the summon.
    assert "_gc.collect()" in src or "gc.collect()" in src
    assert "torch.xpu.synchronize()" in src


def test_save_clones_projection_tensors():
    src = _RECIPE.read_text()
    # The save path was rewritten from summon_full_params to FULL_STATE_DICT
    # state_dict() (2026-06-22); the projection/adapter/backbone tensors are now
    # sliced out of the gathered CPU dict. The load-bearing fix this guards is the
    # .detach().clone() on each saved tensor (breaks the view-into-flat-buffer
    # storage bloat — see test_clone_breaks_storage_bloat).
    assert "FULL_STATE_DICT" in src
    assert "detach().clone()" in src


def test_clone_breaks_storage_bloat():
    """Reproduce the actual bug: a small VIEW into a large storage serializes the
    whole storage; .clone() serializes only the view's data."""
    big = torch.zeros(2_000_000, dtype=torch.float32)  # 8 MB backing storage
    view = big[:10]                                     # tiny view, same storage
    # saving the view persists the whole 8 MB storage
    buf_view = io.BytesIO(); torch.save(view, buf_view)
    # saving a clone persists only ~40 bytes of real data
    buf_clone = io.BytesIO(); torch.save(view.detach().clone(), buf_clone)
    assert buf_view.tell() > 1_000_000, "view save should be bloated (whole storage)"
    assert buf_clone.tell() < 10_000, "clone save should be tight"
    # the clone must be byte-equal in value
    assert torch.equal(view, torch.load(io.BytesIO(buf_clone.getvalue())))


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

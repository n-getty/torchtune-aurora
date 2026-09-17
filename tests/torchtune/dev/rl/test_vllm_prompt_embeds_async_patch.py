"""CPU-safe regression tests for the vLLM prompt_embeds / async-scheduling patch.

Guards the backport of upstream vLLM PR #45673 that lives in
`recipes/dev/_usercustomize_vllm/usercustomize.py` (`_apply_prompt_embeds_patch`).

Background: vLLM 0.15.0 (the version in frameworks/2025.3.1) fails to refresh
`is_token_ids.gpu` on the async-scheduling pure-decode path in
`GPUModelRunner._prepare_input_ids`. With `--enable-prompt-embeds`, the stale mask
selects never-written rows of `input_ids`, feeding garbage indices to an embedding
gather -> out-of-bounds device read -> "banned: 1" GPU fault that kills the server.
This is what made BioReason 32B evals lose servers mid-run.

These tests use a fake GPUModelRunner so they run on a login node with no XPU, no
vLLM import, and no distributed init.
"""

import importlib.util
from pathlib import Path

import pytest

_USERCUSTOMIZE = (
    Path(__file__).resolve().parents[4]
    / "recipes"
    / "dev"
    / "_usercustomize_vllm"
    / "usercustomize.py"
)


def _load_module():
    """Import usercustomize.py by path without triggering its import hook."""
    spec = importlib.util.spec_from_file_location("_uc_under_test", _USERCUSTOMIZE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Buffer:
    """Stand-in for vLLM's CpuGpuBuffer, recording copy_to_gpu calls."""

    def __init__(self):
        self.copied_with = []

    def copy_to_gpu(self, n=None):
        self.copied_with.append(n)


class _InputBatch:
    def __init__(self, prev_sampled_token_ids):
        self.prev_sampled_token_ids = prev_sampled_token_ids


class _FakeRunner:
    """Minimal GPUModelRunner surface used by the patch."""

    def __init__(self, *, enable_prompt_embeds, prev_sampled_token_ids):
        self.enable_prompt_embeds = enable_prompt_embeds
        self.input_batch = _InputBatch(prev_sampled_token_ids)
        self.is_token_ids = _Buffer()
        self.calls = []

    def _prepare_input_ids(self, scheduler_output, total_num_scheduled_tokens,
                           cu_num_tokens):
        # Records that the original implementation still ran, and with what.
        self.calls.append(total_num_scheduled_tokens)
        return "orig-return"


def _patched_runner(**kwargs):
    """Patch a FRESH subclass per test.

    `_apply_prompt_embeds_patch` mutates the class it is handed. Patching the
    shared `_FakeRunner` would stack wrappers across tests and leak state
    between them (each wrapper adding another copy_to_gpu call), so give every
    test its own throwaway subclass.
    """
    mod = _load_module()

    class _Runner(_FakeRunner):
        pass

    mod._apply_prompt_embeds_patch(_Runner)
    return _Runner(**kwargs)


def test_refreshes_is_token_ids_on_async_decode_path():
    """The whole point: async path + prompt embeds must re-upload the mask.

    Without this, `is_token_ids.gpu` keeps the previous step's mask on a
    reordered pure-decode batch and the embedding gather reads garbage.
    """
    runner = _patched_runner(
        enable_prompt_embeds=True, prev_sampled_token_ids=object()
    )
    runner._prepare_input_ids("sched", 8, None)
    assert runner.is_token_ids.copied_with == [8], (
        "is_token_ids must be copied to GPU exactly once, sized to the batch"
    )


def test_original_implementation_still_runs():
    """The patch must wrap, never replace, the upstream method."""
    runner = _patched_runner(
        enable_prompt_embeds=True, prev_sampled_token_ids=object()
    )
    assert runner._prepare_input_ids("sched", 4, None) == "orig-return"
    assert runner.calls == [4]


@pytest.mark.parametrize(
    "enable_prompt_embeds,prev_sampled,reason",
    [
        (False, object(), "prompt embeds off -> is_token_ids is unused"),
        (True, None, "sync path already uploads unconditionally"),
        (False, None, "neither condition applies"),
    ],
)
def test_no_extra_copy_off_the_buggy_path(enable_prompt_embeds, prev_sampled, reason):
    """Don't pay an H2D copy on paths that were never broken."""
    runner = _patched_runner(
        enable_prompt_embeds=enable_prompt_embeds,
        prev_sampled_token_ids=prev_sampled,
    )
    runner._prepare_input_ids("sched", 8, None)
    assert runner.is_token_ids.copied_with == [], reason
    assert runner.calls == [8], "original must run regardless"


def test_keyword_call_still_refreshes():
    """vLLM 0.15.0 calls positionally, but a keyword call must not silently skip."""
    runner = _patched_runner(
        enable_prompt_embeds=True, prev_sampled_token_ids=object()
    )
    runner._prepare_input_ids(
        scheduler_output="sched", total_num_scheduled_tokens=8, cu_num_tokens=None
    )
    assert runner.is_token_ids.copied_with == [8]


def test_unrecognized_signature_skips_refresh_without_adding_a_failure():
    """If upstream reshapes the signature, the wrapper must not be what breaks.

    A patch that takes the engine down is worse than one that stops helping. The
    wrapper can't find the token count here, so it must skip the refresh and hand
    the call through unchanged — whatever the original then does is upstream's
    business, not a failure the patch introduced. (`_FakeRunner` mirrors the real
    signature, so it raises TypeError; the point is that the *wrapper* didn't.)
    """
    runner = _patched_runner(
        enable_prompt_embeds=True, prev_sampled_token_ids=object()
    )
    with pytest.raises(TypeError, match="missing 2 required positional arguments"):
        runner._prepare_input_ids("only-one-arg")
    # The wrapper skipped the refresh rather than guessing a wrong length.
    assert runner.is_token_ids.copied_with == []


def test_copy_failure_does_not_break_generation():
    """A failed refresh must not take down the engine; the original still runs."""

    class _Exploding(_Buffer):
        def copy_to_gpu(self, n=None):
            raise RuntimeError("simulated device failure")

    runner = _patched_runner(
        enable_prompt_embeds=True, prev_sampled_token_ids=object()
    )
    runner.is_token_ids = _Exploding()
    assert runner._prepare_input_ids("sched", 8, None) == "orig-return"


def test_embed_guard_is_opt_in_and_plumbed_through_ssh():
    """The guard costs a per-step D2H sync, so it must default OFF — but it must
    also be plumbed through the PBS wrapper's ssh env block, or setting it would
    silently do nothing on the compute nodes.

    That silent-no-op failure mode is exactly how the usercustomize memory patch
    went unapplied for months (PYTHONNOUSERSITE blocked it and nobody checked),
    so pin the plumbing rather than trusting it.
    """
    root = Path(__file__).resolve().parents[4]
    launcher = root / "experiments" / "bioreason" / "launch_vllm_http_32b_tp2.sh"
    wrapper = root / "experiments" / "bioreason" / "pbs_2n_eval_vllm_tp2.sh"
    if not launcher.exists() or not wrapper.exists():
        pytest.skip("launchers not present (experiments/ is gitignored)")
    assert "TORCHTUNE_VLLM_EMBED_GUARD:-0" in launcher.read_text(), "must default OFF"
    assert "TORCHTUNE_VLLM_EMBED_GUARD" in wrapper.read_text(), (
        "must be forwarded across the ssh hop, else enabling it is a silent no-op"
    )


def test_embed_guard_only_patches_when_env_set(monkeypatch):
    """Patch 6 must not attach unless explicitly requested."""
    mod = _load_module()
    assert hasattr(mod, "_apply_embed_guard_patch"), "guard helper must exist"
    # The gate lives in the import hook; assert the env check is the literal gate.
    src = _USERCUSTOMIZE.read_text()
    assert 'os.environ.get("TORCHTUNE_VLLM_EMBED_GUARD") == "1"' in src


def test_launcher_leaves_async_scheduling_at_vllm_default():
    """`--no-async-scheduling` must stay OPT-IN, because it was refuted as a fix.

    Job 8807697 reproduced banned:1 with async scheduling disabled AND this patch
    loaded on all 24 workers, so disabling it buys nothing and costs pipelining
    throughput. The escape hatch stays available for re-testing that path, but it
    must default to OFF. Flipping the default back to 1 should fail this test and
    force whoever does it to justify the regression.
    """
    launcher = (
        Path(__file__).resolve().parents[4]
        / "experiments"
        / "bioreason"
        / "launch_vllm_http_32b_tp2.sh"
    )
    if not launcher.exists():
        pytest.skip("launcher not present (experiments/ is gitignored)")
    text = launcher.read_text()
    assert "--no-async-scheduling" in text, "escape hatch must remain available"
    assert "EVAL_DISABLE_ASYNC_SCHED:-0" in text, (
        "must default to OFF — disabling async scheduling was refuted as a "
        "banned:1 fix by job 8807697"
    )

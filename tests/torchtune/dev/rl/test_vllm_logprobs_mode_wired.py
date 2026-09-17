"""Enforce that vLLM's ``logprobs_mode`` is actually set at every engine site.

STAGED -- copy into ``tests/torchtune/dev/rl/`` when
``apply_logprobs_mode_fix.py --apply`` lands. It is held here because the fix it
guards edits ``vllm_backend.py``, which the live 32B run imports.

Why this file exists at all: ``test_vllm_behavior_logprobs.py`` already pins the
raw-vs-processed logprob FORMULA with 15 passing tests, and it passed the entire
time no engine construction site set the flag. A test that pins a formula does not
prove the production path uses it. These tests check the CALL SITES.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

def _find_backend() -> pathlib.Path:
    """Walk up for the repo root rather than hardcoding a parent depth.

    A fixed ``parents[N]`` breaks the moment the file is run from a scratch dir,
    which is exactly how this test is exercised before it lands.
    """
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "torchtune" / "dev" / "rl" / "vllm_backend.py"
        if cand.exists():
            return cand
    raise RuntimeError(f"could not locate torchtune/dev/rl/vllm_backend.py above {here}")


BACKEND = _find_backend()


def _tree() -> ast.Module:
    return ast.parse(BACKEND.read_text())


def _llm_call_nodes(tree: ast.Module) -> list[ast.Call]:
    return [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "LLM"
    ]


def test_backend_defines_the_helper():
    tree = _tree()
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "_logprobs_engine_kwargs" in names, (
        "vllm_backend.py must define _logprobs_engine_kwargs; without it vLLM "
        "defaults to raw_logprobs and async GRPOLoss trains a ~4.09x-wrong IS ratio."
    )


def test_every_llm_construction_site_sets_logprobs_mode():
    """Each ``LLM(...)`` must get the mode, inline or via its kwargs dict.

    Two shapes exist in this file:
      1. ``LLM(..., **_logprobs_engine_kwargs(cfg))``
      2. ``llm_kwargs.update(_logprobs_engine_kwargs(cfg))`` then ``LLM(**llm_kwargs)``
    Shape 2 is checked by counting updates, since an AST cannot cheaply prove which
    dict flows into which call.
    """
    tree = _tree()
    calls = _llm_call_nodes(tree)
    assert calls, "no LLM(...) call found -- did the file move or get renamed?"

    src = BACKEND.read_text()
    inline = 0
    via_kwargs = 0
    for call in calls:
        kwargs_src = " ".join(ast.unparse(k) for k in call.keywords)
        if "_logprobs_engine_kwargs" in kwargs_src:
            inline += 1
        else:
            via_kwargs += 1

    updates = src.count("llm_kwargs.update(_logprobs_engine_kwargs(cfg))")
    assert updates >= via_kwargs, (
        f"{via_kwargs} LLM(**llm_kwargs) site(s) but only {updates} "
        "llm_kwargs.update(_logprobs_engine_kwargs(cfg)) line(s). Every engine "
        "construction path must set logprobs_mode -- a missed site is silent."
    )
    assert inline + via_kwargs == len(calls)


def test_lora_and_logprobs_kwargs_are_paired_everywhere():
    """Guard against a NEW engine site copying the LoRA line but not this one."""
    src = BACKEND.read_text()
    lora = src.count("_lora_engine_kwargs(cfg)")
    logprobs = src.count("_logprobs_engine_kwargs(cfg)")
    # the helper's own `def` line adds one occurrence to each count
    assert logprobs >= lora, (
        f"_lora_engine_kwargs appears {lora}x but _logprobs_engine_kwargs only "
        f"{logprobs}x. A new LLM() site was probably added with the LoRA kwargs "
        "and without the logprobs kwargs."
    )


def test_helper_defaults_to_processed_logprobs():
    src = BACKEND.read_text()
    assert '"vllm_logprobs_mode", "processed_logprobs"' in src, (
        "the default must be processed_logprobs; raw_logprobs are unscaled and "
        "are not log_softmax(logits / temperature)."
    )


def test_helper_fails_fast_when_async_is_enabled():
    """Async + unsettable mode must raise, not warn.

    The failure this guards against does not crash and does not warn -- it just
    trains on the wrong ratio. A warning would repeat that mistake.
    """
    src = BACKEND.read_text()
    assert "async_generation" in src and "RuntimeError" in src, (
        "_logprobs_engine_kwargs must raise when async_generation.enabled is true "
        "and processed_logprobs could not be set."
    )


@pytest.mark.parametrize("bad", ["raw_logprobs", "raw_logits", "processed_logits"])
def test_no_site_hardcodes_a_non_processed_mode(bad):
    src = BACKEND.read_text()
    assert f'logprobs_mode="{bad}"' not in src
    assert f"logprobs_mode='{bad}'" not in src

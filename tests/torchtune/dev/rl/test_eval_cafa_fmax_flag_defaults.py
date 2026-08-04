"""CPU test: eval_cafa_fmax.py's prompt-fidelity CLI flags default to faithful.

A sibling BioReason eval script (different codebase, same task) silently regressed
to a "cold" hand-rolled prompt that omitted go_pred/InterPro/PPI/UniProt-summary
context the checkpoint was actually trained with — corrupting F_max by ~0.3 (the
paper's own published checkpoint scored 0.40 instead of its real ~0.69-0.73) — and
it recurred TWICE because nothing automatically checked that these flags still
defaulted the right way after an eval-script edit. See
memory/project_bioreason_gopred_eval_fix_20260724.md.

This repo's eval_cafa_fmax.py already defaults all of these to True (with
BooleanOptionalAction --no-<flag> escape hatches for the ablations that
intentionally disable them). This test pins those defaults so a future edit can't
silently flip one without a test failure. It only inspects argparse.Action.default
via build_arg_parser() — it does not run the eval pipeline, needs no XPU/torch/
bioreason2 import (the module's top-level imports are stdlib-only).
"""
import importlib.util
import os

_HERE = os.path.dirname(__file__)
_EVAL_PATH = os.path.abspath(
    os.path.join(_HERE, "../../../../experiments/bioreason/eval_cafa_fmax.py")
)


def _load_eval_module():
    spec = importlib.util.spec_from_file_location("eval_cafa_fmax_under_test", _EVAL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _defaults(parser):
    # option_strings[0] is the primary flag (e.g. "--inject_go_pred"); for
    # BooleanOptionalAction, option_strings[-1] would instead be the
    # "--no-<flag>" alias.
    return {
        action.option_strings[0].lstrip("-"): action.default
        for action in parser._actions
        if action.option_strings
    }


def test_prompt_fidelity_flags_default_true():
    mod = _load_eval_module()
    parser = mod.build_arg_parser()
    defaults = _defaults(parser)

    faithful_flags = [
        "inject_go_pred",
        "interpro_in_prompt",
        "ppi_in_prompt",
        "include_protein_function_summary",
    ]
    for flag in faithful_flags:
        assert flag in defaults, f"expected flag --{flag} not found in parser"
        assert defaults[flag] is True, (
            f"--{flag} must default to True (faithful prompt) — got "
            f"{defaults[flag]!r}. A False default here silently reproduces the "
            f"cold-prompt eval bug (F_max off by ~0.3 on a known checkpoint)."
        )


def test_ablation_flags_default_off():
    """The modality-ablation flags must default to the FAITHFUL (non-ablated)
    setting — i.e. splice enabled — so a plain eval run is never accidentally
    an ablation."""
    mod = _load_eval_module()
    parser = mod.build_arg_parser()
    defaults = _defaults(parser)

    for flag in ("disable_protein_splice", "disable_go_splice"):
        assert flag in defaults, f"expected flag --{flag} not found in parser"
        assert defaults[flag] is False, (
            f"--{flag} must default to False (modality present) — got "
            f"{defaults[flag]!r}."
        )

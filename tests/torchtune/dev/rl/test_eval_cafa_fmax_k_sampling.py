"""CPU tests: eval_cafa_fmax.py's --k G-sample path.

Frequency ranking (confidence = count/k across k sampled completions) is worth
+0.0349 F_max offline -- ~2x the +-0.016 eval noise floor -- but it was only ever
demonstrated by re-scoring existing artifacts. The harness itself could not produce
the input: --temperature defaulted to 0.0 and `_k00` was hardcoded at both write
sites, so it emitted exactly one greedy sample per protein. See
memory/project_bioreason_eval_harness_cannot_emit_g_samples_20260915.md.

The failure modes this pins down are all SILENT -- each one produces a run that
looks complete and a number that looks real:

  - **Greedy k.** k>1 at temperature 0 returns the same completion k times, so
    every term's count/k is 1.0. That is byte-for-byte the flat-confidence baseline
    the feature exists to escape, dressed up as a k-sample result.
  - **Oracle selection.** score_fmax_scored.py calls ce.select_best_from_k_samples
    when it finds _k01+ in ONE directory, picking the highest-F1 sample AGAINST
    GROUND TRUTH. Writing k files into one dir would post a large fake gain. Hence
    sibling replicate dirs, each holding one _k00.json.
  - **Ragged groups.** A resume that skips a protein present in only some replicate
    dirs would score it as a smaller group, varying the count/k denominator across
    the eval set.

These tests only touch argparse and the module's pure helpers -- no XPU, no torch,
no model load (eval_cafa_fmax.py's top-level imports are stdlib-only).
"""
import importlib.util
import json
import os

import pytest

_HERE = os.path.dirname(__file__)
_EVAL_PATH = os.path.abspath(
    os.path.join(_HERE, "../../../../experiments/bioreason/eval_cafa_fmax.py")
)
_RESCORE_PATH = os.path.abspath(
    os.path.join(_HERE, "../../../../experiments/bioreason/rescore_by_group_frequency.py")
)


def _load_eval_module():
    spec = importlib.util.spec_from_file_location("eval_cafa_fmax_k_under_test", _EVAL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_rescorer():
    if not os.path.isfile(_RESCORE_PATH):
        return None
    spec = importlib.util.spec_from_file_location("rescore_group_freq_under_test",
                                                  _RESCORE_PATH)
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except Exception:
        return None
    return m


@pytest.fixture(scope="module")
def mod():
    assert os.path.isfile(_EVAL_PATH), f"missing {_EVAL_PATH}"
    return _load_eval_module()


def test_k_defaults_to_one_so_every_existing_launcher_is_unchanged(mod):
    parser = mod.build_arg_parser()
    k = [a for a in parser._actions if a.option_strings and a.option_strings[0] == "--k"]
    assert k, "--k flag not found"
    assert k[0].default == 1
    # And greedy stays the single-sample default: k=1 eval must not silently start sampling.
    temp = [a for a in parser._actions if a.option_strings
            and a.option_strings[0] == "--temperature"]
    assert temp[0].default == 0.0


def test_greedy_multisample_is_rejected_not_silently_degenerate(mod):
    """THE DEGENERATE-GROUP GUARD.

    k>1 at temperature 0 yields k identical completions, so count/k == 1.0 for every
    term -- indistinguishable from the flat-confidence status quo, but reported as a
    k-sample run. Must fail loudly instead.

    Asserted by CALLING the validator. An earlier version of this test grepped main()'s
    source for "args.k > 1" / "args.temperature"; deleting the guard outright still
    passed it, because those substrings also occur in the adjacent --no_vllm check and
    in the error message. A guard test that cannot fail is worse than no test -- it
    reports safety that isn't there.
    """
    with pytest.raises(SystemExit) as ei:
        mod.validate_k_sampling(k=8, temperature=0.0, no_vllm=False)
    assert "--temperature" in str(ei.value)

    # Negative temperature is the same degeneracy, not a separate case.
    with pytest.raises(SystemExit):
        mod.validate_k_sampling(k=8, temperature=-1.0, no_vllm=False)

    # And the guard must not fire on the configurations that ARE valid.
    mod.validate_k_sampling(k=8, temperature=0.7, no_vllm=False)   # sampled k-group
    mod.validate_k_sampling(k=1, temperature=0.0, no_vllm=False)   # greedy single
    mod.validate_k_sampling(k=1, temperature=0.0, no_vllm=True)    # greedy single, HF


def test_k_below_one_is_rejected(mod):
    for bad in (0, -1):
        with pytest.raises(SystemExit):
            mod.validate_k_sampling(k=bad, temperature=0.7, no_vllm=False)


def test_no_vllm_path_rejects_k_because_it_hardcodes_greedy(mod):
    """backbone.generate(do_sample=False) ignores --temperature entirely, so k>1 there
    would return k identical strings even at temperature 0.7 -- the temperature guard
    above cannot catch it."""
    with pytest.raises(SystemExit) as ei:
        mod.validate_k_sampling(k=8, temperature=0.7, no_vllm=True)
    assert "no_vllm" in str(ei.value)
    # The premise of that rejection: the branch really is hardcoded greedy. If someone
    # wires do_sample through, this assert fails and the guard should be revisited.
    assert "do_sample=False" in open(_EVAL_PATH).read()


def test_k_fans_out_into_sibling_dirs_never_k01_in_one_dir(mod):
    """Layout guard against ce.select_best_from_k_samples oracle contamination.

    Each replicate dir holds one _k00.json, which is also exactly what the already-
    validated rescore_by_group_frequency.py consumes via --replicate_dir.
    """
    assert mod.k_output_dirs("/tmp/ev", 1) == ["/tmp/ev"], (
        "k=1 must write to --out itself so existing launchers/scorers are untouched"
    )
    dirs = mod.k_output_dirs("/tmp/ev", 4)
    assert dirs == [
        "/tmp/ev/k00", "/tmp/ev/k01", "/tmp/ev/k02", "/tmp/ev/k03",
    ]
    assert len(set(dirs)) == 4, "replicate dirs must be distinct"
    # The only filename written is _k00.json -- no _k{i} in a filename anywhere, which
    # is what keeps select_best_from_k_samples from ever firing.
    src = open(_EVAL_PATH).read()
    assert '_k00.json"' in src
    assert "k{_i:02d}.json" not in src and "_k{i:02d}.json" not in src


def test_resume_skip_requires_all_replicates(mod):
    """A protein present in 3 of 8 replicate dirs must be re-run, not skipped: scoring
    it as a 3-sample group would vary the count/k denominator across the eval set."""
    src = open(_EVAL_PATH).read()
    fn = src.split("def _process_one")[1].split("\n        try:")[0]
    assert "all(" in fn and "_out_dirs" in fn, (
        "resume-skip must require the prediction in ALL replicate dirs"
    )


def test_short_completion_list_is_padded_not_written_ragged(mod):
    """A server returning fewer than k sequences must be padded, not written as a short
    group -- a missing replicate file makes this protein's resume-skip fail forever.

    The padded rows are then marked unsuccessful (see the denominator guard below), so the
    warning has to say the protein drops out; a warning that only says "padding with
    empty" reads as harmless and it isn't.
    """
    src = open(_EVAL_PATH).read()
    assert "_n_real < args.k" in src
    warn = src.split("padding with empty")[1].split("flush=True")[0]
    assert "DROPPED" in warn and "re-run" in warn, (
        "the pad warning must state that the protein leaves the scored set and how to "
        "recover it"
    )


def test_padded_samples_are_marked_unsuccessful_so_the_rescorer_drops_them(mod):
    """THE DENOMINATOR GUARD.

    make_record hardcodes success=True. rescore_by_group_frequency.load_replicate drops
    only success=False rows, and build_arms fixes g = len(replicate_dirs). So a padded
    empty placeholder written as successful is a sample that voted for zero terms against
    a full-k denominator: every frequency in that group comes out scaled by n_real/k, in
    the freq arm only -- the arm under test. Marking it unsuccessful drops the protein
    instead, which is the rescorer's existing contract.
    """
    # Real completions keep success=True and carry provenance.
    real = {"success": True}
    mod.apply_k_provenance(real, k_index=0, k_total=8, temperature=0.7, n_real=6)
    assert real["success"] is True
    assert real["k_index"] == 0 and real["k_total"] == 8
    assert real["eval_temperature"] == 0.7
    assert "k_padded" not in real

    # Index at/after the real count is padding -> unsuccessful, flagged.
    for idx in (6, 7):
        pad = {"success": True}
        mod.apply_k_provenance(pad, k_index=idx, k_total=8, temperature=0.7, n_real=6)
        assert pad["success"] is False, (
            f"k_index={idx} is padding (n_real=6) and must not vote in count/k"
        )
        assert pad["k_padded"] is True

    # Full group: nothing is padding.
    for idx in range(8):
        rec = {"success": True}
        mod.apply_k_provenance(rec, k_index=idx, k_total=8, temperature=0.7, n_real=8)
        assert rec["success"] is True and "k_padded" not in rec


def test_rescorer_actually_drops_a_padded_row(mod, tmp_path):
    """End-to-end on the interface: write one real + one padded record the way the eval
    harness does, and confirm the UNMODIFIED rescorer's loader keeps the first and drops
    the second. This is the claim the report makes ("--k output is consumable by
    rescore_by_group_frequency.py unchanged"); assert it against the real loader rather
    than reasoning about the field."""
    resc = _load_rescorer()
    if resc is None:
        pytest.skip("rescore_by_group_frequency.py not importable")

    def _write(d, response, k_index, n_real):
        os.makedirs(d, exist_ok=True)
        rec = {
            "protein_id": "P12345",
            "go_aspect": "biological_process",
            "generated_response": response,
            "success": True,
            "go_bp": ["GO:0006915"],
        }
        mod.apply_k_provenance(rec, k_index=k_index, k_total=2, temperature=0.7,
                               n_real=n_real)
        with open(os.path.join(d, "P12345_bp_k00.json"), "w") as f:
            json.dump(rec, f)

    d_real = str(tmp_path / "k00")
    d_pad = str(tmp_path / "k01")
    _write(d_real, "answer GO:0006915", k_index=0, n_real=1)
    _write(d_pad, "", k_index=1, n_real=1)

    assert len(resc.load_replicate(d_real)) == 1, "the real completion must load"
    assert resc.load_replicate(d_pad) == {}, (
        "the padded placeholder must be dropped by the rescorer's success filter -- "
        "otherwise it votes for zero terms against a denominator of 2, halving every "
        "frequency in this group"
    )


def test_k_records_carry_provenance(mod):
    """A scored k-run must never be mistakable for a greedy run after the fact."""
    src = open(_EVAL_PATH).read()
    for field in ("k_index", "k_total", "eval_temperature"):
        assert f'"{field}"' in src, f"k>1 records must stamp {field}"


def test_http_path_requests_k_in_one_call(mod):
    """generate_from_embeds takes n=, so k completions cost ONE ~41 MiB prompt_embeds
    payload rather than k round trips."""
    src = open(_EVAL_PATH).read()
    call = src.split("_http_client.generate_from_embeds(")[1].split(")")[0]
    assert "n=args.k" in call


def test_sp_with_n_preserves_every_other_sampling_field(mod):
    """The in-process vLLM path must not re-list sampling fields (they would drift from
    the k=1 path on the next edit) -- it clones."""
    class _FakeSP:
        def __init__(self):
            self.n = 1
            self.max_tokens = 2048
            self.temperature = 0.7
            self.repetition_penalty = 1.0
            self.detokenize = True

    sp = _FakeSP()
    sp2 = mod._sp_with_n(sp, 8)
    assert sp2.n == 8
    assert sp.n == 1, "must not mutate the caller's SamplingParams"
    for f in ("max_tokens", "temperature", "repetition_penalty", "detokenize"):
        assert getattr(sp2, f) == getattr(sp, f), f"{f} drifted during clone"


def test_k_scoring_hint_does_not_point_a_scorer_at_the_parent_dir(mod):
    """The parent dir contains k subdirs; pointing cafa_evals/score_fmax at it would
    either find nothing or, worse, glob across replicates."""
    src = open(_EVAL_PATH).read()
    tail = src.split("if args.k > 1:")[-1]
    assert "rescore_by_group_frequency.py" in tail
    assert "--replicate_dir" in tail

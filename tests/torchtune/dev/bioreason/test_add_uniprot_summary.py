# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# CPU regression test for the add_uniprot_summary parity fix (see
# experiments/bioreason/parity/PARITY_CONTRACT.md confound #5). rbdgx3 trains with
# add_uniprot_summary=True (BioReason-Pro's _format_reasoning_prompt/_add_uniprot_summary);
# Aurora's dataset_sft.py had no such field at all until this fix, which caused a real,
# reproducible loss-scale gap (~1.0 vs ~0.7 at the same training step, same LR/data) that
# was mistaken for a possible layer-(-1) alignment problem before being root-caused here.

_PROT_ID = 151643
_GO_ID = 151644


class _StubTok:
    """Minimal tokenizer: 1 id per whitespace token; deterministic, no weights."""
    bos_id = 1

    def encode(self, text, add_bos=False, add_eos=False):
        ids = ([self.bos_id] if add_bos else []) + [7] * len(text.split())
        if add_eos:
            ids += [2]
        return ids


def _ds_with(monkeypatch, rows, add_uniprot_summary):
    from torchtune.dev.bioreason import dataset_sft as sft
    monkeypatch.setattr(sft.BioReasonSFTDataset, "_load", lambda self, df: list(rows))
    return sft.BioReasonSFTDataset(
        data_files="unused", tokenizer=_StubTok(), max_seq_len=4096,
        max_protein_len=2048, num_go_tokens=5, protein_token_id=_PROT_ID,
        go_token_id=_GO_ID, train_on_reasoning=True, inject_go_pred=True,
        add_uniprot_summary=add_uniprot_summary,
    )


def _row():
    return {
        "organism": "Homo sapiens", "interpro_formatted": "IPR0001",
        "ppi_formatted": "P12345", "go_pred": "GO:0000001",
        "go_mf": ["GO:0000001"], "go_cc": [], "go_bp": [],
        "sequence": "MKT", "reasoning": "because of X",
        "final_answer": "- Functional Summary: does Y",
        "protein_function": "Catalyzes the hydrolysis of Z.",
    }


def test_default_off_is_byte_identical_to_prior_behavior(monkeypatch):
    ds = _ds_with(monkeypatch, [_row()], add_uniprot_summary=False)
    prompt = ds._build_prompt_text(ds.examples[0])
    target_ids = ds._build_target_ids(ds.examples[0])
    assert "Summarize in UniProt format" not in prompt
    # Target should be exactly reasoning + "\n" + final_answer, unmodified.
    final = ds.examples[0]["final_answer"]
    reasoning = ds.examples[0]["reasoning"]
    expected_target = f"{reasoning}\n{final}"
    expected_ids = ds.tokenizer.encode(expected_target, add_bos=False, add_eos=True)
    assert target_ids == expected_ids


def test_on_appends_uniprot_instruction_to_prompt(monkeypatch):
    ds = _ds_with(monkeypatch, [_row()], add_uniprot_summary=True)
    prompt = ds._build_prompt_text(ds.examples[0])
    assert prompt.endswith("Summarize in UniProt format.")


def test_on_inserts_uniprot_summary_line_into_target(monkeypatch):
    ds = _ds_with(monkeypatch, [_row()], add_uniprot_summary=True)
    ex = ds.examples[0]
    target_ids = ds._build_target_ids(ex)
    expected_final = ds._add_uniprot_summary_line(
        ex["final_answer"], ex["protein_function"]
    )
    expected_target = f"{ex['reasoning']}\n{expected_final}"
    expected_ids = ds.tokenizer.encode(expected_target, add_bos=False, add_eos=True)
    assert target_ids == expected_ids


def test_add_uniprot_summary_line_matches_reference_insertion_logic():
    from torchtune.dev.bioreason.dataset_sft import BioReasonSFTDataset

    final_answer = "- Functional Summary: does Y\n- Extra detail line"
    protein_function = "Catalyzes the hydrolysis of Z."
    out = BioReasonSFTDataset._add_uniprot_summary_line(final_answer, protein_function)
    # Verbatim port of BioReason-Pro's _add_uniprot_summary: insert as line 2.
    lines = final_answer.split("\n")
    expected = (
        lines[0] + "\n- UniProt Summary: " + protein_function.strip() + "\n"
        + "\n".join(lines[1:])
    )
    assert out == expected


def test_add_uniprot_summary_factory_kwarg_threads_through(monkeypatch):
    from torchtune.dev.bioreason.dataset_sft import bioreason_sft_dataset, BioReasonSFTDataset
    monkeypatch.setattr(BioReasonSFTDataset, "_load", lambda self, df: [_row()])
    ds = bioreason_sft_dataset(
        tokenizer=_StubTok(), data_files="unused", protein_token_id=_PROT_ID,
        go_token_id=_GO_ID, add_uniprot_summary=True,
    )
    assert ds.add_uniprot_summary is True


def test_build_native_prompt_text_accepts_add_uniprot_summary(monkeypatch):
    import sys, os
    sys.path.insert(0, os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "experiments", "bioreason",
    ))
    import eval_cafa_fmax as ecf

    prompt = ecf.build_native_prompt_text(_row(), _StubTok(), add_uniprot_summary=True)
    assert prompt.endswith("Summarize in UniProt format.")

    prompt_off = ecf.build_native_prompt_text(_row(), _StubTok(), add_uniprot_summary=False)
    assert "Summarize in UniProt format" not in prompt_off


def test_build_native_input_ids_accepts_add_uniprot_summary():
    import sys, os
    sys.path.insert(0, os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "experiments", "bioreason",
    ))
    import eval_cafa_fmax as ecf

    row = _row()
    ids_on = ecf.build_native_input_ids(
        row, row["sequence"], _StubTok(), _PROT_ID, _GO_ID, num_go_tokens=5,
        add_uniprot_summary=True,
    )
    ids_off = ecf.build_native_input_ids(
        row, row["sequence"], _StubTok(), _PROT_ID, _GO_ID, num_go_tokens=5,
        add_uniprot_summary=False,
    )
    # The uniprot instruction adds tokens to the prompt text, so the two ids sequences
    # must differ in length (proves the flag actually reaches _build_prompt_text, not
    # silently dropped along the way).
    assert len(ids_on) != len(ids_off)

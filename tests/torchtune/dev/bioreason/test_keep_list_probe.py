import importlib.util
from pathlib import Path


_SCRIPT = Path(__file__).parents[4] / "experiments" / "bioreason" / "probe_keep_list_accuracy.py"
_SPEC = importlib.util.spec_from_file_location("keep_list_probe", _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_keep_list_parser_ignores_reasoning_terms():
    response = "GO terms:\nGO:0000001\n\n<think>GO:0000002</think>"
    assert _MODULE.emitted_keep_list(response) == {"GO:0000001"}


def test_measure_scores_binary_candidate_decisions():
    record = {
        "input_prompt": "initial GO term speculations:\nBP: GO:0000001, GO:0000002\n\nReason about the function",
        "generated_response": "GO terms:\nGO:0000001\n\ntrace",
        "go_mf": [], "go_cc": [], "go_bp": ["GO:0000001"],
    }
    result = _MODULE.measure(record)
    assert result["candidates"] == 2
    assert result["true_positives"] == 1
    assert result["true_negatives"] == 1
    assert result["candidate_accuracy"] == 1.0


def test_measure_handles_old_checkpoint_without_keep_list():
    record = {
        "input_prompt": "initial GO term speculations: GO:0000001\n\nReason about the function",
        "generated_response": "<think>GO:0000001</think>",
        "go_mf": ["GO:0000001"], "go_cc": [], "go_bp": [],
    }
    result = _MODULE.measure(record)
    assert result["emitted_candidates"] == 0
    assert result["false_negatives"] == 1

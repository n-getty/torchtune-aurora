#!/usr/bin/env python3
"""Create a sealed-artifact EP scaling report."""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from torchtune.modules.moe.measurement import (
    aggregate_measurement_files,
    compare_ep_scaling_summaries,
    summarize_ep_scaling_artifact,
    validate_router_semantics,
)


def _warmup_steps(artifact: Mapping[str, Any]) -> int:
    metadata = artifact.get("common_metadata", {})
    window = metadata.get("measurement_window", {}) if isinstance(metadata, Mapping) else {}
    try:
        value = int(window.get("warmup_steps", 0))
    except (TypeError, ValueError) as error:
        raise ValueError("measurement_window.warmup_steps must be an integer") from error
    if value < 0:
        raise ValueError("measurement_window.warmup_steps must be non-negative")
    return value


def _validate_ep_leg_metadata(
    artifact: Mapping[str, Any], *, expected_ep: int, expected_locality: str
) -> None:
    """Reject a sealed artifact set whose metadata disagrees with its CLI leg."""
    metadata = artifact.get("common_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"EP{expected_ep} artifact has no common metadata")
    for field in ("source_revision", "model", "checkpoint"):
        value = metadata.get(field)
        if not isinstance(value, str) or not value.strip() or value == "unknown":
            raise ValueError(
                f"EP{expected_ep} artifact must record non-placeholder {field}"
            )
    validate_router_semantics(
        metadata["model"],
        metadata.get("router_semantics"),
        context=f"EP{expected_ep} artifact",
    )
    try:
        ep_degree = int(metadata["ep_degree"])
        world_size = int(metadata["world_size"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"EP{expected_ep} artifact must record integer ep_degree and world_size"
        ) from error
    if ep_degree != expected_ep:
        raise ValueError(
            f"EP{expected_ep} artifact metadata reports ep_degree={ep_degree}"
        )
    if world_size != expected_ep:
        raise ValueError(
            f"EP{expected_ep} artifact metadata reports world_size={world_size}"
        )
    topology = metadata.get("topology")
    if not isinstance(topology, Mapping):
        raise ValueError(f"EP{expected_ep} artifact must record topology")
    try:
        topology_ep = int(topology["ep"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"EP{expected_ep} artifact topology must record ep") from error
    if topology_ep != expected_ep:
        raise ValueError(
            f"EP{expected_ep} artifact topology reports ep={topology_ep}"
        )
    environment = metadata.get("environment_overrides")
    if not isinstance(environment, Mapping):
        raise ValueError(f"EP{expected_ep} artifact must record environment_overrides")
    transport = environment.get("TORCHTUNE_EP_ALL2ALL")
    if str(transport) != "1":
        raise ValueError(
            f"EP{expected_ep} artifact transport must be AllToAll "
            "(TORCHTUNE_EP_ALL2ALL=1)"
        )
    locality = str(environment.get("TORCHTUNE_MOE_COLLECTIVE_LOCALITY", ""))
    if locality != expected_locality:
        raise ValueError(
            f"EP{expected_ep} artifact locality must be {expected_locality!r}, "
            f"got {locality!r}"
        )
    for field, expected in (
        ("device_health", "green"),
        ("gate_status", "passed"),
        ("semantic_completion", "passed"),
        ("measurement_completion", "passed"),
    ):
        if metadata.get(field) != expected:
            raise ValueError(
                f"EP{expected_ep} artifact {field} must be {expected!r}"
            )
    worktree_state = metadata.get("uncommitted_change_state")
    if (
        not isinstance(worktree_state, str)
        or not worktree_state.strip()
        or worktree_state == "unknown"
    ):
        raise ValueError(
            f"EP{expected_ep} artifact must record non-placeholder "
            "uncommitted_change_state"
        )


def _validate_report_memory(
    summary: Mapping[str, Any], *, expected_ep: int
) -> None:
    """Require peak and steady-state allocator evidence for canonical reports."""
    for name in ("peak_memory", "steady_state_memory"):
        memory = summary.get(name)
        if not isinstance(memory, Mapping):
            raise ValueError(f"EP{expected_ep} artifact has no {name} summary")
        reserved = memory.get("max_reserved_bytes")
        allocated = memory.get("max_allocated_bytes")
        if reserved is None or allocated is None:
            raise ValueError(
                f"EP{expected_ep} artifact {name} must record reserved and allocated bytes"
            )
        for field, value in (("max_reserved_bytes", reserved), ("max_allocated_bytes", allocated)):
            try:
                value = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"EP{expected_ep} artifact {name}.{field} must be numeric"
                ) from error
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"EP{expected_ep} artifact {name}.{field} must be finite and non-negative"
                )


def _validate_report_timing(
    summary: Mapping[str, Any],
    *,
    expected_ep: int,
    expected_measurement_steps: int | None = None,
) -> None:
    """Reject a sealed EP summary without a usable measured window."""
    phase_timings = summary.get("phase_timings")
    if not isinstance(phase_timings, Mapping):
        raise ValueError(f"EP{expected_ep} artifact has no phase timing summary")
    try:
        sample_count = int(phase_timings["sample_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"EP{expected_ep} artifact must record timing sample_count"
        ) from error
    if sample_count < 1:
        raise ValueError(
            f"EP{expected_ep} artifact has no post-warmup step timing records"
        )
    if expected_measurement_steps is not None:
        try:
            expected_measurement_steps = int(expected_measurement_steps)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"EP{expected_ep} measurement_steps must be an integer"
            ) from error
        if expected_measurement_steps < 1:
            raise ValueError(
                f"EP{expected_ep} measurement_steps must be positive"
            )
        used_steps = phase_timings.get("measurement_steps_used")
        try:
            used_steps = int(used_steps)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"EP{expected_ep} artifact has no measurement_steps_used"
            ) from error
        if (
            used_steps != expected_measurement_steps
            or sample_count != expected_measurement_steps
        ):
            raise ValueError(
                f"EP{expected_ep} artifact recorded {used_steps} measurement steps; "
                f"expected {expected_measurement_steps}"
            )
    for field_name in (
        "mean_total_step_s",
        "mean_tokens_per_second_per_gpu",
        "mean_aggregate_tokens_per_second",
    ):
        value = phase_timings.get(field_name)
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"EP{expected_ep} artifact has no valid {field_name}"
            ) from error
        if not math.isfinite(numeric_value) or numeric_value <= 0:
            raise ValueError(
                f"EP{expected_ep} artifact has invalid {field_name}: {value!r}"
            )


def _validate_routing_metadata_attribution(
    summary: Mapping[str, Any], *, expected_ep: int
) -> None:
    """Require routing metadata collective and construction attribution."""
    communication = summary.get("communication")
    if not isinstance(communication, Mapping):
        raise ValueError(f"EP{expected_ep} artifact has no communication summary")
    by_collective = communication.get("by_collective")
    if not isinstance(by_collective, Mapping):
        raise ValueError(
            f"EP{expected_ep} artifact has no collective attribution summary"
        )
    metadata_collective = None
    for name, value in by_collective.items():
        if str(name).lower() in {
            "routing_metadata",
            "routing_metadata_allgather",
            "routing_metadata_all_gather",
        }:
            metadata_collective = value
            break
    if not isinstance(metadata_collective, Mapping):
        raise ValueError(
            f"EP{expected_ep} artifact is missing routing metadata collective timing"
        )
    try:
        collective_seconds = float(metadata_collective["duration_s"])
        collective_count = int(metadata_collective["count"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"EP{expected_ep} artifact has incomplete routing metadata collective timing"
        ) from error
    if not math.isfinite(collective_seconds) or collective_seconds < 0 or collective_count < 1:
        raise ValueError(
            f"EP{expected_ep} artifact has invalid routing metadata collective timing"
        )
    timing = summary.get("routing_metadata")
    if not isinstance(timing, Mapping):
        raise ValueError(f"EP{expected_ep} artifact has no routing metadata timing")
    for phase in ("materialization", "permutation"):
        seconds = timing.get(f"{phase}_seconds")
        count = timing.get(f"{phase}_event_count")
        if seconds is None or count is None:
            raise ValueError(
                f"EP{expected_ep} artifact is missing routing metadata {phase} timing"
            )
        try:
            seconds = float(seconds)
            count = int(count)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"EP{expected_ep} artifact has invalid routing metadata {phase} timing"
            ) from error
        if not math.isfinite(seconds) or seconds < 0 or count < 1:
            raise ValueError(
                f"EP{expected_ep} artifact has invalid routing metadata {phase} timing"
            )


def _validate_alltoall_attribution(
    summary: Mapping[str, Any], *, expected_ep: int, expected_locality: str
) -> None:
    """Require forward/backward dispatch and combine collectives in reports."""
    communication = summary.get("communication")
    if not isinstance(communication, Mapping):
        raise ValueError(f"EP{expected_ep} artifact has no communication summary")
    by_collective = communication.get("by_collective")
    if not isinstance(by_collective, Mapping):
        raise ValueError(
            f"EP{expected_ep} artifact has no collective attribution summary"
        )
    by_locality = communication.get("by_locality")
    if not isinstance(by_locality, Mapping):
        raise ValueError(
            f"EP{expected_ep} artifact has no collective locality summary"
        )
    observed_localities = {
        str(name)
        for name, value in by_locality.items()
        if isinstance(value, Mapping) and int(value.get("count", 0)) > 0
    }
    if observed_localities != {expected_locality}:
        raise ValueError(
            f"EP{expected_ep} artifact collective locality must be "
            f"{{{expected_locality!r}}}, got {sorted(observed_localities)!r}"
        )
    required = (
        "dispatch_alltoall",
        "combine_alltoall",
        "dispatch_backward_alltoall",
        "combine_backward_alltoall",
    )
    missing = [name for name in required if name not in by_collective]
    if missing:
        raise ValueError(
            f"EP{expected_ep} artifact is missing AllToAll collective timing: "
            + ", ".join(missing)
        )
    for name in required:
        value = by_collective[name]
        if not isinstance(value, Mapping):
            raise ValueError(
                f"EP{expected_ep} artifact has invalid {name} collective timing"
            )
        try:
            seconds = float(value["duration_s"])
            count = int(value["count"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"EP{expected_ep} artifact has incomplete {name} collective timing"
            ) from error
        if not math.isfinite(seconds) or seconds < 0 or count < 1:
            raise ValueError(
                f"EP{expected_ep} artifact has invalid {name} collective timing"
            )


def _validate_routing_phase_attribution(
    summary: Mapping[str, Any], *, expected_ep: int
) -> None:
    """Require pack/unpack timing for each canonical routing direction."""
    timing = summary.get("routing_phases")
    if not isinstance(timing, Mapping):
        raise ValueError(f"EP{expected_ep} artifact has no routing phase timing")
    required = (
        "dispatch_pack",
        "dispatch_unpack",
        "dispatch_backward_pack",
        "dispatch_backward_unpack",
        "combine_pack",
        "combine_unpack",
        "combine_backward_pack",
        "combine_backward_unpack",
    )
    for name in required:
        value = timing.get(name)
        if not isinstance(value, Mapping):
            raise ValueError(
                f"EP{expected_ep} artifact is missing routing phase timing {name}"
            )
        try:
            seconds = float(value["seconds"])
            count = int(value["event_count"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"EP{expected_ep} artifact has invalid routing phase timing {name}"
            ) from error
        if not math.isfinite(seconds) or seconds < 0 or count < 1:
            raise ValueError(
                f"EP{expected_ep} artifact has invalid routing phase timing {name}"
            )


def _decision_categories(rows: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    rows = list(rows)

    def is_repeatable(row: Mapping[str, Any]) -> bool:
        if row.get("repeatable") is True:
            return True
        try:
            return int(row.get("independent_repeats", 1)) >= 2
        except (TypeError, ValueError):
            return False

    def best(value_getter: Any, label: str) -> str:
        candidates = [
            row
            for row in rows
            if is_repeatable(row)
            and isinstance(value_getter(row), (int, float))
        ]
        if not candidates:
            return "pending_measurement"
        winner = max(candidates, key=lambda row: float(value_getter(row)))
        return f"ep{int(winner['ep_degree'])}_{label}"

    return {
        "highest_repeatable_moe_per_gpu_mfu": best(
            lambda row: row.get("model_metrics", {}).get("mfu_percent")
            if isinstance(row.get("model_metrics"), Mapping)
            else None,
            "mfu",
        ),
        "highest_aggregate_throughput": best(
            lambda row: row.get("aggregate_tokens_per_second"),
            "aggregate_throughput",
        ),
        "best_ep_scaling_point": best(
            lambda row: row.get("scaling_efficiency"), "scaling"
        ),
        "closest_strict_dense_parity": "inherited_kernel_parity_reference",
        "best_capacity_value_result": "pending_capacity_value_measurement",
    }


def _gap_attribution(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    rows = list(rows)

    def status(predicate: Any, evidence: str) -> dict[str, Any]:
        return {
            "status": "measured" if any(predicate(row) for row in rows) else "pending",
            "evidence": evidence,
        }

    communication = status(
        lambda row: isinstance(
            row.get("collective_communication_seconds"), (int, float)
        )
        or isinstance(row.get("communication_seconds"), (int, float)),
        "dispatch/combine AllToAll timing by name and locality; legacy total timing accepted",
    )
    communication["collective_seconds"] = [
        row.get("collective_communication_seconds")
        for row in rows
        if isinstance(row.get("collective_communication_seconds"), (int, float))
    ]
    communication["routing_metadata_collective_seconds"] = [
        row.get("routing_metadata_collective_seconds")
        for row in rows
        if isinstance(row.get("routing_metadata_collective_seconds"), (int, float))
    ]
    communication["pack_unpack_seconds"] = [
        row.get("routing_pack_unpack_seconds")
        for row in rows
        if isinstance(row.get("routing_pack_unpack_seconds"), (int, float))
    ]
    communication["pack_unpack_status"] = (
        "measured"
        if any(
            isinstance(row.get("routing_pack_unpack_seconds"), (int, float))
            for row in rows
        )
        else "pending"
    )
    return {
        "communication": communication,
        "expert_compute": status(
            lambda row: isinstance(row.get("expert_compute"), Mapping)
            and isinstance(row["expert_compute"].get("timing_seconds"), (int, float)),
            "grouped-GEMM timing and shape statistics",
        ),
        "attention": status(
            lambda row: isinstance(row.get("phase_timings"), Mapping)
            and isinstance(
                row["phase_timings"].get("phase_mean_s", {}).get("attention"),
                (int, float),
            ),
            "attention phase timing",
        ),
        "optimizer": status(
            lambda row: isinstance(row.get("phase_timings"), Mapping)
            and isinstance(
                row["phase_timings"].get("phase_mean_s", {}).get(
                    "optimizer_step_total"
                ),
                (int, float),
            ),
            "optimizer-step phase timing",
        ),
        "pipeline_bubble": status(
            lambda row: isinstance(row.get("phase_timings"), Mapping)
            and isinstance(
                row["phase_timings"].get("phase_mean_s", {}).get(
                    "pipeline_bubble"
                ),
                (int, float),
            ),
            "explicit pipeline bubble timing",
        ),
    }


def build_report(
    ep8_paths: Iterable[str], ep16_paths: Iterable[str]
) -> dict[str, Any]:
    """Aggregate two sealed EP artifacts and return a table-ready report."""
    aggregated: dict[int, dict[str, Any]] = {}
    for ep_degree, expected_locality, paths in (
        (8, "node_local", ep8_paths),
        (16, "cross_node", ep16_paths),
    ):
        artifact = aggregate_measurement_files(
            paths,
            require_complete_rank_set=True,
            require_router_semantics=True,
        )
        if artifact.get("rank_count", 0) < 1:
            raise ValueError(f"EP{ep_degree} artifact set is empty")
        _validate_ep_leg_metadata(
            artifact,
            expected_ep=ep_degree,
            expected_locality=expected_locality,
        )
        aggregated[ep_degree] = summarize_ep_scaling_artifact(
            artifact, warmup_steps=_warmup_steps(artifact)
        )
        window = artifact["common_metadata"]["measurement_window"]
        _validate_report_timing(
            aggregated[ep_degree],
            expected_ep=ep_degree,
            expected_measurement_steps=window["measurement_steps"],
        )
        _validate_routing_metadata_attribution(
            aggregated[ep_degree], expected_ep=ep_degree
        )
        _validate_alltoall_attribution(
            aggregated[ep_degree],
            expected_ep=ep_degree,
            expected_locality=expected_locality,
        )
        _validate_routing_phase_attribution(
            aggregated[ep_degree], expected_ep=ep_degree
        )
        _validate_report_memory(aggregated[ep_degree], expected_ep=ep_degree)
    comparison = compare_ep_scaling_summaries(
        aggregated, require_control_metadata=True
    )
    rows = comparison["rows"]
    repeatability = {
        str(ep_degree): {
            "independent_repeats": int(
                summary.get("independent_repeats", 1)
            ),
            "repeatable": bool(summary.get("repeatable", False)),
        }
        for ep_degree, summary in aggregated.items()
    }
    return {
        "result_class": "ep_scaling",
        "ep_summaries": {str(ep): summary for ep, summary in aggregated.items()},
        "comparison": comparison,
        "repeatability": {
            "status": (
                "passed"
                if all(value["repeatable"] for value in repeatability.values())
                else "pending_independent_hardware_repeat"
            ),
            "by_ep": repeatability,
            "winner_categories_require_repeatable": True,
        },
        "decision_categories": _decision_categories(rows),
        "gap_attribution": _gap_attribution(rows),
        "interpretation": {
            "status": "pending_larger_token_volume_control",
            "classification": "pending",
            "reason": (
                "A single EP8/EP16 pair cannot distinguish cross-node communication "
                "from undersized expert GEMMs. Repeat EP16 at larger true local "
                "batch/token volume before assigning a dominant blocker."
            ),
            "rules": {
                "expert_gemm_or_launch_overhead": (
                    "EP16 improves materially with larger token volume while its "
                    "communication fraction remains similar."
                ),
                "cross_node_communication": (
                    "EP16 remains poor after expert-compute efficiency improves, "
                    "with communication fraction remaining dominant."
                ),
                "mixed_or_workload_limited": (
                    "Both communication and expert compute improve but EP16 remains "
                    "below EP8; retain EP16 as workload-specific rejected."
                ),
            },
            "required_evidence": [
                "same model, sequence length, optimizer, and execution controls",
                "EP16 true larger local batch or synthetic routed-token diagnostic",
                "communication fraction by locality",
                "expert grouped-GEMM timing and shape statistics",
                "repeatable throughput, memory, and device-health gates",
            ],
        },
    }


def _format_value(value: Any) -> str:
    if value is None:
        return "pending"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def format_markdown(report: Mapping[str, Any]) -> str:
    """Format the comparison rows as a compact markdown table."""
    comparison = report["comparison"]
    rows = comparison["rows"]
    columns = """EP | World size | Batch | Microbatch | Accumulation | Transport | Routed assignments | Local assignments | Tokens/local expert | Zero-token experts | Imbalance ratio | Path | Expert compute s | Sequential gate s | Sequential up s | Sequential down s | Grouped gate s | Grouped up s | Grouped down s | Compute tokens | Padding tokens | Padding fraction | Expert compute fraction | Router s | Attention s | Non-expert s | Backward s | Grad release s | Optimizer s | Total step s | Communication fraction | MFU % | Active FLOP efficiency | tok/s/GPU | Aggregate tok/s | Expected aggregate | Efficiency | Dispatch AllToAll s (fwd+bwd) | Combine AllToAll s (fwd+bwd) | Dispatch pack s | Dispatch unpack s | Dispatch backward pack s | Dispatch backward unpack s | Combine pack s | Combine unpack s | Combine backward pack s | Combine backward unpack s | Routing metadata s | Metadata materialization s | Metadata permutation s | Collective communication s | Pack/unpack s | Communication s | Peak reserved bytes | Steady-state reserved bytes | Steady-state allocated bytes | Health | Completion""".split(" | ")
    lines = [
        "# EP Scaling Report",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---:" for _ in columns) + " |",
    ]
    for row in rows:
        memory = row.get("peak_memory", {})
        reserved = memory.get("max_reserved_bytes") if isinstance(memory, Mapping) else None
        steady_state_memory = row.get("steady_state_memory", {})
        if not isinstance(steady_state_memory, Mapping):
            steady_state_memory = {}
        expert_compute = row.get("expert_compute", {})
        if not isinstance(expert_compute, Mapping):
            expert_compute = {}
        timing_by_name = expert_compute.get("timing_by_name", {})
        if not isinstance(timing_by_name, Mapping):
            timing_by_name = {}

        def timing_seconds(name: str):
            value = timing_by_name.get(name)
            return value.get("seconds") if isinstance(value, Mapping) else None
        routed_tokens = row.get("routed_tokens", {})
        if not isinstance(routed_tokens, Mapping):
            routed_tokens = {}
        model_metrics = row.get("model_metrics", {})
        if not isinstance(model_metrics, Mapping):
            model_metrics = {}
        phase_timings = row.get("phase_timings", {})
        if not isinstance(phase_timings, Mapping):
            phase_timings = {}
        phase_means = phase_timings.get("phase_mean_s", {})
        if not isinstance(phase_means, Mapping):
            phase_means = {}
        phase_fractions = phase_timings.get("phase_fraction_of_total", {})
        if not isinstance(phase_fractions, Mapping):
            phase_fractions = {}
        routing_phases = row.get("routing_phases", {})
        if not isinstance(routing_phases, Mapping):
            routing_phases = {}

        def routing_seconds(name: str):
            value = routing_phases.get(name)
            return value.get("seconds") if isinstance(value, Mapping) else None
        total_step = phase_timings.get("mean_total_step_s")
        communication_fraction = (
            row.get("communication_seconds") / total_step
            if isinstance(row.get("communication_seconds"), (int, float))
            and isinstance(total_step, (int, float))
            and total_step > 0
            else None
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    _format_value(row.get("ep_degree")),
                    _format_value(row.get("world_size")),
                    _format_value(row.get("batch_size")),
                    _format_value(row.get("microbatch_size")),
                    _format_value(row.get("gradient_accumulation_steps")),
                    _format_value(row.get("transport")),
                    _format_value(routed_tokens.get("global_assignments")),
                    _format_value(routed_tokens.get("mean_local_assignments")),
                    _format_value(expert_compute.get("tokens_per_local_expert")),
                    _format_value(expert_compute.get("zero_token_experts")),
                    _format_value(expert_compute.get("max_expert_imbalance_ratio")),
                    _format_value(expert_compute.get("execution_path")),
                    _format_value(expert_compute.get("timing_seconds")),
                    _format_value(timing_seconds("sequential_expert_gate")),
                    _format_value(timing_seconds("sequential_expert_up")),
                    _format_value(timing_seconds("sequential_expert_down")),
                    _format_value(timing_seconds("grouped_gemm_gate")),
                    _format_value(timing_seconds("grouped_gemm_up")),
                    _format_value(timing_seconds("grouped_gemm_down")),
                    _format_value(expert_compute.get("compute_tokens")),
                    _format_value(expert_compute.get("padding_tokens")),
                    _format_value(expert_compute.get("padding_fraction")),
                    _format_value(phase_fractions.get("expert_forward")),
                    _format_value(phase_means.get("router")),
                    _format_value(phase_means.get("attention")),
                    _format_value(phase_means.get("non_expert")),
                    _format_value(phase_means.get("backward_total")),
                    _format_value(phase_means.get("manual_grad_release_total")),
                    _format_value(phase_means.get("optimizer_step_total")),
                    _format_value(total_step),
                    _format_value(communication_fraction),
                    _format_value(model_metrics.get("mfu_percent")),
                    _format_value(model_metrics.get("active_flop_efficiency")),
                    _format_value(row.get("tokens_per_second_per_gpu")),
                    _format_value(row.get("aggregate_tokens_per_second")),
                    _format_value(row.get("expected_aggregate_tokens_per_second")),
                    _format_value(row.get("scaling_efficiency")),
                    _format_value(row.get("dispatch_alltoall_seconds")),
                    _format_value(row.get("combine_alltoall_seconds")),
                    _format_value(routing_seconds("dispatch_pack")),
                    _format_value(routing_seconds("dispatch_unpack")),
                    _format_value(routing_seconds("dispatch_backward_pack")),
                    _format_value(routing_seconds("dispatch_backward_unpack")),
                    _format_value(routing_seconds("combine_pack")),
                    _format_value(routing_seconds("combine_unpack")),
                    _format_value(routing_seconds("combine_backward_pack")),
                    _format_value(routing_seconds("combine_backward_unpack")),
                    _format_value(row.get("routing_metadata_seconds")),
                    _format_value(
                        row.get("routing_metadata_materialization_seconds")
                    ),
                    _format_value(row.get("routing_metadata_permutation_seconds")),
                    _format_value(row.get("collective_communication_seconds")),
                    _format_value(row.get("routing_pack_unpack_seconds")),
                    _format_value(row.get("communication_seconds")),
                ]
            )
            + " | "
            + " | ".join(
                [
                    _format_value(reserved),
                    _format_value(steady_state_memory.get("max_reserved_bytes")),
                    _format_value(steady_state_memory.get("max_allocated_bytes")),
                    _format_value(row.get("device_health")),
                    _format_value(row.get("measurement_completion")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Decision Categories",
            "",
            "| Category | Result |",
            "| --- | --- |",
        ]
    )
    for name, value in report.get("decision_categories", {}).items():
        lines.append(f"| {name} | {_format_value(value)} |")
    lines.extend(
        [
            "",
            "## Gap Attribution",
            "",
            "| Gap | Status | Evidence |",
            "| --- | --- | --- |",
        ]
    )
    for name, value in report.get("gap_attribution", {}).items():
        if isinstance(value, Mapping):
            lines.append(
                f"| {name} | {_format_value(value.get('status'))} | "
                f"{_format_value(value.get('evidence'))} |"
            )
    lines.extend(
        [
            "",
            "Scaling efficiency is aggregate throughput divided by EP8 aggregate throughput scaled by world-size ratio.",
            "",
            "## Interpretation",
            "",
            f"Status: {_format_value(report.get('interpretation', {}).get('status'))}",
            "",
            report.get("interpretation", {}).get("reason", "pending"),
            "",
            "The report classifies expert-GEMM/launch overhead only after larger-token-volume evidence; otherwise the conclusion remains pending.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ep8", nargs="+", required=True, help="sealed EP8 rank artifacts")
    parser.add_argument("--ep16", nargs="+", required=True, help="sealed EP16 rank artifacts")
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    report = build_report(args.ep8, args.ep16)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(format_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

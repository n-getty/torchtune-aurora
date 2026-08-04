# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Validation for the three MoE experiment result classes."""

from pathlib import Path
from typing import Any, Mapping

RESULT_CLASSES = {"kernel_parity", "ep_scaling", "capacity_value"}
REQUIRED_METADATA = {
    "source_revision",
    "model",
    "hardware",
    "topology",
    "sequence_length",
    "batch_size",
    "optimizer",
    "measurement_window",
}
DECISION_CATEGORIES = {
    "highest_repeatable_moe_per_gpu_mfu",
    "highest_aggregate_throughput",
    "best_ep_scaling_point",
    "closest_strict_dense_parity",
    "best_capacity_value_result",
}
GAP_CATEGORIES = {
    "communication",
    "expert_compute",
    "attention",
    "optimizer",
    "pipeline_bubble",
}

EP_SCALING_REQUIRED_METRICS = {
    "local_tokens",
    "global_tokens",
    "tokens_per_local_expert",
    "expert_imbalance",
    "expert_execution_path",
    "expert_compute_timing",
    "grouped_gemm_gate",
    "grouped_gemm_up",
    "grouped_gemm_down",
    "router",
    "attention",
    "non_expert",
    "expert_forward",
    "final_scatter",
    "dispatch_alltoall",
    "combine_alltoall",
    "dispatch_backward_alltoall",
    "combine_backward_alltoall",
    "routing_metadata_allgather",
    "routing_metadata_materialization",
    "routing_metadata_permutation",
    "dispatch_pack",
    "dispatch_unpack",
    "dispatch_backward_pack",
    "dispatch_backward_unpack",
    "combine_pack",
    "combine_unpack",
    "combine_backward_pack",
    "combine_backward_unpack",
    "collective_communication",
    "routing_pack_unpack",
    "model_fwd_total",
    "backward_total",
    "manual_grad_release_total",
    "optimizer_step_total",
    "collective_locality",
    "total_step_s",
    "peak_memory",
    "steady_state_memory",
    "device_health",
    "per_gpu_throughput",
    "aggregate_throughput",
}
CAPACITY_VALUE_REQUIRED_METRICS = {
    "per_gpu_throughput",
    "aggregate_throughput",
    "mfu_percent",
    "active_flop_efficiency",
    "total_parameters",
    "peak_memory",
    "steady_state_memory",
    "communication_fraction",
    "expert_compute_fraction",
    "stability",
}


def _validate_repository_paths(manifest: Mapping[str, Any]) -> None:
    """Validate repository-relative executable paths when requested."""
    if manifest.get("validate_paths") is not True:
        return

    candidates: list[tuple[str, Any]] = []

    def collect(value: Any, key: str) -> None:
        if isinstance(value, Mapping):
            for child_key, child_value in value.items():
                collect(child_value, str(child_key))
        elif isinstance(value, list):
            for child_value in value:
                collect(child_value, key)
        elif (
            isinstance(value, str)
            and key in {
                "moe_config",
                "moe_launcher",
                "configuration",
                "measurement_configuration",
                "measurement_launcher",
                "promotion_evaluator",
                "comparison_report",
                "launcher",
                "driver",
            }
            and not value.startswith(("/", "<", "pending"))
        ):
            candidates.append((key, value))

    collect(manifest, "")
    repository_root = Path(__file__).resolve().parents[3]
    missing = [
        f"{key}={value}"
        for key, value in candidates
        if not (repository_root / value).is_file()
    ]
    if missing:
        raise ValueError(
            "manifest references missing repository paths: " + ", ".join(missing)
        )


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` when a benchmark manifest is incomplete."""
    result_class = manifest.get("result_class")
    if result_class not in RESULT_CLASSES:
        raise ValueError(f"result_class must be one of {sorted(RESULT_CLASSES)}")
    missing = sorted(REQUIRED_METADATA - set(manifest))
    if missing:
        raise ValueError(f"manifest missing required metadata: {', '.join(missing)}")
    if not isinstance(manifest["hardware"], Mapping):
        raise ValueError("hardware must be a mapping")
    if not isinstance(manifest["topology"], Mapping):
        raise ValueError("topology must be a mapping")
    if not isinstance(manifest["measurement_window"], Mapping):
        raise ValueError("measurement_window must be a mapping")
    _validate_repository_paths(manifest)
    window = manifest["measurement_window"]
    if window.get("warmup_steps", 0) < 0 or window.get("measurement_steps", 0) < 1:
        raise ValueError("measurement window requires warmup >= 0 and measurement > 0")
    if "environment" not in manifest:
        raise ValueError("manifest must record environment overrides")
    decisions = manifest.get("decision_categories")
    if not isinstance(decisions, Mapping):
        raise ValueError("manifest must record decision categories")
    missing_decisions = sorted(DECISION_CATEGORIES - set(decisions))
    if missing_decisions:
        raise ValueError(
            "decision categories missing: " + ", ".join(missing_decisions)
        )
    gaps = manifest.get("gap_attribution")
    if not isinstance(gaps, Mapping):
        raise ValueError("manifest must record gap attribution")
    missing_gaps = sorted(GAP_CATEGORIES - set(gaps))
    if missing_gaps:
        raise ValueError("gap attribution missing: " + ", ".join(missing_gaps))
    if "completion" not in manifest:
        raise ValueError("manifest must record completion and device health")
    completion = manifest["completion"]
    for key in ("semantic_completion", "device_health", "status"):
        if key not in completion:
            raise ValueError(f"completion missing {key}")

    if result_class == "kernel_parity":
        controls = manifest.get("controls")
        if not isinstance(controls, Mapping):
            raise ValueError("kernel_parity manifest must record controls")
        evaluator = controls.get("promotion_evaluator")
        if not isinstance(evaluator, str) or not evaluator:
            raise ValueError(
                "kernel_parity controls must record promotion_evaluator"
            )
        if controls.get("matched_sequence_length") is not True:
            raise ValueError("kernel_parity must match sequence length")
        if controls.get("matched_optimizer_policy") is not True:
            raise ValueError("kernel_parity must match optimizer policy")
        if controls.get("matched_measurement_window") is not True:
            raise ValueError("kernel_parity must match measurement window")

    if result_class == "ep_scaling":
        controls = manifest.get("controls")
        if not isinstance(controls, Mapping):
            raise ValueError("ep_scaling manifest must record controls")
        if controls.get("ep_degrees") != [8, 16]:
            raise ValueError("ep_scaling controls must compare EP8 and EP16")
        local_batch_sweep = controls.get("local_batch_sweep")
        if not isinstance(local_batch_sweep, list):
            raise ValueError("ep_scaling controls must record local_batch_sweep")
        if len(local_batch_sweep) < 2:
            raise ValueError(
                "ep_scaling local_batch_sweep must contain at least two true batch sizes"
            )
        try:
            normalized_batches = [int(batch) for batch in local_batch_sweep]
        except (TypeError, ValueError) as error:
            raise ValueError(
                "ep_scaling local_batch_sweep must contain positive integers"
            ) from error
        if any(batch < 1 for batch in normalized_batches):
            raise ValueError(
                "ep_scaling local_batch_sweep must contain positive integers"
            )
        if normalized_batches != sorted(set(normalized_batches)):
            raise ValueError(
                "ep_scaling local_batch_sweep must be sorted and unique"
            )
        if normalized_batches[0] != 1:
            raise ValueError("ep_scaling local_batch_sweep must start at batch 1")
        if controls.get("grad_release_policy_sweep") != [
            "native_fsdp",
            "streaming_manual",
        ]:
            raise ValueError(
                "ep_scaling controls must compare native_fsdp and streaming_manual"
            )
        if controls.get("gradient_accumulation_is_proxy") is not False:
            raise ValueError("ep_scaling must not use gradient accumulation as a proxy")
        gates = manifest.get("required_gates")
        if not isinstance(gates, list):
            raise ValueError("ep_scaling manifest must record required_gates")
        required_gates = {
            "fresh_node_collective_health",
            "two_step_forward_backward_optimizer_smoke",
            "finite_loss_and_stable_memory",
            "semantic_completion",
            "sealed_measurement_artifacts",
        }
        if not required_gates.issubset(gates):
            raise ValueError("ep_scaling required_gates are incomplete")
        synthetic = controls.get("synthetic_token_volume_diagnostic")
        if not isinstance(synthetic, Mapping):
            raise ValueError(
                "ep_scaling must record synthetic token-volume diagnostic"
            )
        if synthetic.get("promotion_artifact") is not False:
            raise ValueError(
                "synthetic token-volume diagnostic must not be a promotion artifact"
            )
        if not isinstance(synthetic.get("driver"), str) or not synthetic["driver"]:
            raise ValueError(
                "synthetic token-volume diagnostic must record a driver"
            )
        output_contract = synthetic.get("output_contract")
        if not isinstance(output_contract, list):
            raise ValueError(
                "synthetic token-volume diagnostic must record output_contract"
            )
        required_output_fields = {
            "rows",
            "grouped_gemm",
            "volume_scaling",
            "promotion_artifact_false",
        }
        if not required_output_fields.issubset(output_contract):
            raise ValueError(
                "synthetic token-volume diagnostic output_contract is incomplete"
            )
        controlled_overrides = controls.get("controlled_overrides")
        if not isinstance(controlled_overrides, list):
            raise ValueError("ep_scaling must record controlled_overrides")
        required_overrides = {
            "TORCHTUNE_EP_INDEX_ADD_COMBINE",
            "TORCHTUNE_MOE_INDEX_ADD_FINAL_SCATTER",
            "TORCHTUNE_MOE_NATIVE_FSDP_GRAD_REDUCE",
            "TORCHTUNE_EP_GRAD_RELEASE_STREAMING",
            "TORCHTUNE_MOE_ALLTOALL_CONDITIONAL_CONTIGUOUS",
            "TORCHTUNE_MOE_TOPK_ROUTING",
            "TORCHTUNE_MOE_UNSTABLE_EXPERT_GROUPING",
        }
        if not required_overrides.issubset(controlled_overrides):
            raise ValueError("ep_scaling controlled_overrides are incomplete")
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, Mapping):
            raise ValueError("ep_scaling must record artifacts")
        if artifacts.get("required_execution_path") != "controlled_by_launcher":
            raise ValueError(
                "ep_scaling requires controlled_by_launcher execution path"
            )
        required_metrics = set(manifest.get("required_metrics", []))
        if not EP_SCALING_REQUIRED_METRICS.issubset(required_metrics):
            raise ValueError("ep_scaling required_metrics are incomplete")

    if result_class == "capacity_value":
        controls = manifest.get("controls")
        if not isinstance(controls, Mapping) or not controls.get("dense_control"):
            raise ValueError("capacity_value manifest must record a dense control")
        moe_config = controls.get("moe_config")
        moe_launcher = controls.get("moe_launcher")
        if not isinstance(moe_config, str) or not moe_config:
            raise ValueError("capacity_value must record an executable MoE config")
        if not isinstance(moe_launcher, str) or not moe_launcher:
            raise ValueError("capacity_value must record an executable MoE launcher")
        alltoall_contiguous = manifest.get("environment", {}).get(
            "TORCHTUNE_MOE_ALLTOALL_CONDITIONAL_CONTIGUOUS"
        )
        if alltoall_contiguous not in {"0", "1"}:
            raise ValueError(
                "capacity_value must record a binary AllToAll contiguity control"
            )
        dense_control = controls["dense_control"]
        if not isinstance(dense_control, Mapping):
            raise ValueError("capacity_value dense_control must record model provenance")
        for field in (
            "model",
            "checkpoint",
            "comparison_label",
            "measurement_configuration",
            "measurement_launcher",
            "measurement_artifact",
        ):
            if not isinstance(dense_control.get(field), str) or not dense_control[field]:
                raise ValueError(
                    f"capacity_value dense_control must record {field}"
                )
        if dense_control.get("comparison_label") != "capacity_value_only":
            raise ValueError(
                "capacity_value dense_control must be labeled capacity_value_only"
            )
        if controls.get("capacity_label_required_when_unmatched") is not True:
            raise ValueError("capacity_value must require capacity labeling when unmatched")
        if manifest.get("parity_claim_allowed") is not False:
            raise ValueError("capacity_value cannot allow an unqualified parity claim")
        required_metrics = set(manifest.get("required_metrics", []))
        if not CAPACITY_VALUE_REQUIRED_METRICS.issubset(required_metrics):
            raise ValueError("capacity_value required_metrics are incomplete")
        gates = manifest.get("required_gates")
        if not isinstance(gates, list) or not {
            "public_or_local_checkpoint_and_config",
            "native_grouped_gemm_and_ep_compatibility",
            "two_node_memory_and_stability_smoke",
            "semantic_completion",
            "sealed_measurement_artifacts",
        }.issubset(gates):
            raise ValueError("capacity_value required_gates are incomplete")
        provenance = manifest.get("artifact_provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("capacity_value must record artifact provenance")
        for field in ("execution_path", "launcher", "checkpoint_identity"):
            if not isinstance(provenance.get(field), str) or not provenance[field]:
                raise ValueError(
                    f"capacity_value artifact provenance must record {field}"
                )


def validate_evaluation_manifest(
    manifest: Mapping[str, Any], *, manifest_path: str | Path | None = None
) -> None:
    """Validate the top-level manifest that connects the three MoE tracks."""
    if manifest.get("result_classes") != [
        "kernel_parity",
        "ep_scaling",
        "capacity_value",
    ]:
        raise ValueError("evaluation manifest must list all three result classes")
    if sorted(manifest.get("decision_categories", [])) != sorted(DECISION_CATEGORIES):
        raise ValueError("evaluation manifest decision categories are incomplete")
    if sorted(manifest.get("gap_categories", [])) != sorted(GAP_CATEGORIES):
        raise ValueError("evaluation manifest gap categories are incomplete")
    tracks = manifest.get("tracks")
    if not isinstance(tracks, Mapping):
        raise ValueError("evaluation manifest must record tracks")
    repository_root = Path(__file__).resolve().parents[3]

    def require_path(value: Any, field: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"evaluation manifest {field} must be a path")
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = repository_root / candidate
        if not candidate.is_file():
            raise ValueError(f"evaluation manifest references missing {field}: {value}")
        return candidate

    expected_classes = {
        "kernel_parity": "kernel_parity",
        "ep_scaling": "ep_scaling",
        "capacity_value": "capacity_value",
    }
    for track_name, result_class in expected_classes.items():
        track = tracks.get(track_name)
        if not isinstance(track, Mapping):
            raise ValueError(f"evaluation manifest missing {track_name} track")
        standalone_path = require_path(
            track.get("standalone_manifest"),
            f"{track_name}.standalone_manifest",
        )
        try:
            import yaml

            standalone_manifest = yaml.safe_load(standalone_path.read_text())
        except (OSError, ValueError) as error:
            raise ValueError(
                f"evaluation manifest cannot load {track_name}.standalone_manifest"
            ) from error
        if not isinstance(standalone_manifest, Mapping):
            raise ValueError(f"{track_name} standalone manifest must be a mapping")
        if standalone_manifest.get("result_class") != result_class:
            raise ValueError(
                f"{track_name} standalone manifest has the wrong result_class"
            )
        validate_manifest(standalone_manifest)

    kernel = tracks["kernel_parity"]
    require_path(kernel.get("manifest"), "kernel_parity.manifest")
    require_path(kernel.get("promotion_evaluator"), "kernel_parity.promotion_evaluator")
    require_path(
        kernel.get("optimization_ab_report"),
        "kernel_parity.optimization_ab_report",
    )

    scaling = tracks["ep_scaling"]
    launchers = scaling.get("measurement_launchers")
    if not isinstance(launchers, Mapping) or not launchers:
        raise ValueError("ep_scaling track must record measurement_launchers")
    for name, launcher in launchers.items():
        require_path(launcher, f"ep_scaling.measurement_launchers.{name}")
    measurement_artifact = scaling.get("measurement_artifact")
    if not isinstance(measurement_artifact, Mapping):
        raise ValueError("ep_scaling track must record measurement_artifact")
    scaling_required_metrics = set(scaling.get("required_metrics", []))
    if not EP_SCALING_REQUIRED_METRICS.issubset(scaling_required_metrics):
        raise ValueError(
            "evaluation manifest ep_scaling required_metrics are incomplete"
        )
    require_path(
        measurement_artifact.get("comparison_report"),
        "ep_scaling.measurement_artifact.comparison_report",
    )

    capacity = tracks["capacity_value"]
    require_path(
        capacity.get("dense_measurement_launcher"),
        "capacity_value.dense_measurement_launcher",
    )
    require_path(capacity.get("comparison_report"), "capacity_value.comparison_report")

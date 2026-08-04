# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Opt-in, non-promoting MoE measurement helpers."""

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Optional

import torch

_COLLECTIVE_LOCALITIES = {"node_local", "cross_node", "unknown"}


def validate_router_semantics(
    model: Any, router_semantics: Any, *, context: str = "measurement artifact"
) -> None:
    """Validate the router marker for model families with fixed semantics."""
    if not isinstance(router_semantics, str) or not router_semantics.strip():
        raise ValueError(f"{context} requires non-empty router_semantics")
    model_name = str(model).lower()
    expected = None
    if "qwen3" in model_name:
        expected = "probability_topk_v2"
    elif "gemma4" in model_name:
        expected = "sigmoid_argsort_v1"
    if expected is not None and router_semantics != expected:
        raise ValueError(
            f"{context} router_semantics must be {expected!r} for model {model!r}; "
            f"got {router_semantics!r}"
        )
_BINARY_ENVIRONMENT_OVERRIDES = (
    "TORCHTUNE_EP_INDEX_ADD_COMBINE",
    "TORCHTUNE_MOE_INDEX_ADD_FINAL_SCATTER",
    "TORCHTUNE_MOE_VECTOR_PACKING",
    "TORCHTUNE_EP_ROWWISE_ALLTOALL_UNPERMUTE",
    "TORCHTUNE_EP_UNINITIALIZED_ALLTOALL_BUFFERS",
    "TORCHTUNE_EP_FUSED_ALLTOALL_ROUTING",
    "TORCHTUNE_EP_CPU_VECTOR_ROUTING_METADATA",
    "TORCHTUNE_EP_PACK_ROUTING_METADATA_TRANSFER",
    "TORCHTUNE_MOE_ALLTOALL_CONDITIONAL_CONTIGUOUS",
)


def _validate_binary_environment_overrides(
    environment_overrides: Mapping[str, Any], path: str
) -> None:
    for override_name in _BINARY_ENVIRONMENT_OVERRIDES:
        override_value = environment_overrides.get(override_name)
        if override_value is not None and str(override_value) not in {"0", "1"}:
            raise ValueError(
                f"measurement artifact {override_name} must be 0 or 1: {path}"
            )


class _NoOpTiming:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


_NO_OP_TIMING = _NoOpTiming()


def measurement_enabled() -> bool:
    """Return whether optional MoE measurement is enabled."""
    return os.environ.get("TORCHTUNE_MOE_MEASURE", "0") == "1"


def synchronize_measurement_device() -> None:
    """Synchronize the active accelerator for an opt-in timing boundary."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.synchronize()
    elif hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def token_statistics(
    counts: Iterable[int], *, ep_degree: Optional[int] = None
) -> dict[str, Any]:
    """Summarize routed token ownership, including zero-token experts."""
    values = [int(value) for value in counts]
    if any(value < 0 for value in values):
        raise ValueError("expert token counts must be non-negative")
    total = sum(values)
    nonzero = [value for value in values if value]
    mean = total / len(values) if values else 0.0
    result: dict[str, Any] = {
        "expert_count": len(values),
        "total_tokens": total,
        "nonzero_experts": len(nonzero),
        "zero_token_experts": len(values) - len(nonzero),
        "min_tokens_per_expert": min(values, default=0),
        "max_tokens_per_expert": max(values, default=0),
        "mean_tokens_per_expert": mean,
        "imbalance_ratio": (max(values) / mean) if mean else 0.0,
        "counts": values,
    }
    if ep_degree is not None:
        if ep_degree < 1 or len(values) % ep_degree:
            raise ValueError("expert count must be divisible by ep_degree")
        local = len(values) // ep_degree
        result.update(
            {
                "ep_degree": ep_degree,
                "local_experts": local,
                "local_token_count": total / ep_degree,
                "tokens_per_local_expert": total / (ep_degree * local),
            }
        )
    return result


def synthetic_expert_token_counts(
    total_tokens: int,
    expert_count: int,
    *,
    imbalance_factor: float = 1.0,
) -> list[int]:
    """Create deterministic routed-token counts for GEMM volume diagnostics.

    This workload has no model state and must not be used as a training or
    promotion result. ``imbalance_factor`` scales linearly from the first to
    the last expert, allowing expert-shape sensitivity to be measured without
    changing checkpoint memory.
    """
    if total_tokens < 0:
        raise ValueError("total_tokens must be non-negative")
    if expert_count < 1:
        raise ValueError("expert_count must contain at least one expert")
    if not math.isfinite(imbalance_factor) or imbalance_factor < 1:
        raise ValueError("imbalance_factor must be finite and at least one")
    if total_tokens == 0:
        return [0] * expert_count
    weights = [
        1.0
        + (imbalance_factor - 1.0) * index / max(expert_count - 1, 1)
        for index in range(expert_count)
    ]
    weight_total = sum(weights)
    raw = [total_tokens * weight / weight_total for weight in weights]
    counts = [int(value) for value in raw]
    remainder = total_tokens - sum(counts)
    for index in sorted(
        range(expert_count), key=lambda item: raw[item] - counts[item], reverse=True
    )[:remainder]:
        counts[index] += 1
    return counts


def grouped_gemm_statistics(
    counts: Iterable[int],
    *,
    model_dim: int,
    hidden_dim: int,
    routed_counts: Optional[Iterable[int]] = None,
    alignment: Optional[int] = None,
    stage: Optional[str] = None,
) -> dict[str, Any]:
    """Describe per-expert grouped-GEMM shapes for a routed batch.

    ``counts`` is the count consumed by the grouped GEMM. When aligned padding
    is used, ``routed_counts`` preserves the pre-padding ownership so the
    artifact can distinguish routing imbalance from alignment overhead.
    """
    values = [int(value) for value in counts]
    if model_dim < 1 or hidden_dim < 1:
        raise ValueError("GEMM dimensions must be positive")
    stats = token_statistics(values)
    stats.update(
        {
            "model_dim": model_dim,
            "hidden_dim": hidden_dim,
            "up_projection_shapes": [[value, model_dim, hidden_dim] for value in values],
            "down_projection_shapes": [[value, hidden_dim, model_dim] for value in values],
            "active_expert_gemm_count": sum(value > 0 for value in values),
        }
    )
    if routed_counts is not None:
        routed_values = [int(value) for value in routed_counts]
        if len(routed_values) != len(values):
            raise ValueError("routed_counts must match grouped GEMM expert count")
        if any(value < 0 for value in routed_values):
            raise ValueError("routed expert token counts must be non-negative")
        padded_tokens = sum(values) - sum(routed_values)
        if padded_tokens < 0:
            raise ValueError("grouped GEMM counts cannot be below routed counts")
        stats.update(
            {
                "routed_counts": routed_values,
                "routed_tokens": sum(routed_values),
                "compute_tokens": sum(values),
                "padding_tokens": padded_tokens,
                "padding_fraction": (
                    padded_tokens / sum(values) if sum(values) else 0.0
                ),
            }
        )
        if alignment is not None:
            if alignment < 1:
                raise ValueError("GEMM alignment must be positive")
            stats["alignment"] = alignment
    if stage is not None:
        if not stage.strip():
            raise ValueError("GEMM measurement stage must be non-empty")
        stats["stage"] = stage
    return stats


def padded_bmm_statistics(
    counts: Iterable[int],
    *,
    model_dim: int,
    hidden_dim: int,
    projection_count: int = 3,
) -> dict[str, Any]:
    """Describe dense padded-BMM work for the sequential-expert layout."""
    values = [int(value) for value in counts]
    if model_dim < 1 or hidden_dim < 1 or projection_count < 1:
        raise ValueError("BMM dimensions and projection count must be positive")
    if any(value < 0 for value in values):
        raise ValueError("expert token counts must be non-negative")
    max_count = max(values, default=0)
    routed_tokens = sum(values)
    dense_tokens = len(values) * max_count
    return {
        "stage": "padded_bmm",
        "model_dim": model_dim,
        "hidden_dim": hidden_dim,
        "projection_count": projection_count,
        "counts": values,
        "max_count": max_count,
        "expert_count": len(values),
        "routed_tokens": routed_tokens,
        "dense_compute_tokens": dense_tokens,
        "padding_tokens": dense_tokens - routed_tokens,
        "padding_fraction": (
            (dense_tokens - routed_tokens) / dense_tokens if dense_tokens else 0.0
        ),
        "dense_to_routed_ratio": dense_tokens / routed_tokens if routed_tokens else 0.0,
    }


@dataclass
class MemorySnapshot:
    phase: str
    step: Optional[int] = None
    microbatch: Optional[int] = None
    allocated_bytes: Optional[int] = None
    reserved_bytes: Optional[int] = None
    free_bytes: Optional[int] = None
    total_bytes: Optional[int] = None


@dataclass
class MoEMeasurement:
    timings_s: dict[str, float] = field(default_factory=dict)
    timing_counts: dict[str, int] = field(default_factory=dict)
    routed_tokens: list[dict[str, Any]] = field(default_factory=list)
    grouped_gemm: list[dict[str, Any]] = field(default_factory=list)
    collectives: list[dict[str, Any]] = field(default_factory=list)
    memory: list[MemorySnapshot] = field(default_factory=list)

    def add_timing(self, name: str, duration_s: float) -> None:
        self.timings_s[name] = self.timings_s.get(name, 0.0) + float(duration_s)
        self.timing_counts[name] = self.timing_counts.get(name, 0) + 1

    def record_collective(
        self,
        name: str,
        duration_s: float,
        *,
        scope: str,
        backend: str,
        locality: str = "unknown",
    ) -> None:
        self.collectives.append(
            {
                "name": name,
                "duration_s": float(duration_s),
                "scope": scope,
                "backend": backend,
                "locality": locality,
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class MoEMeasurementCollector:
    """Low-overhead opt-in collector for one measurement step."""

    def __init__(self, *, enabled: Optional[bool] = None) -> None:
        self.enabled = measurement_enabled() if enabled is None else enabled
        self._record: Optional[MoEMeasurement] = None

    @property
    def record(self) -> MoEMeasurement:
        if self._record is None:
            self._record = MoEMeasurement()
        return self._record

    def time(self, name: str):
        if not self.enabled:
            return _NO_OP_TIMING
        return _Timing(self, name)

    def collective(self, name: str, *, scope: str, backend: str):
        """Time one collective and retain its transport identity when enabled."""
        if not self.enabled:
            return _NO_OP_TIMING
        locality = os.environ.get("TORCHTUNE_MOE_COLLECTIVE_LOCALITY", "unknown")
        if locality not in _COLLECTIVE_LOCALITIES:
            raise ValueError(
                "TORCHTUNE_MOE_COLLECTIVE_LOCALITY must be node_local, "
                "cross_node, or unknown"
            )
        return _CollectiveTiming(self, name, scope, backend, locality)

    def record_tokens(
        self, counts: Iterable[int], *, ep_degree: Optional[int] = None
    ) -> None:
        if self.enabled:
            if isinstance(counts, torch.Tensor):
                counts = counts.detach().cpu().tolist()
            self.record.routed_tokens.append(token_statistics(counts, ep_degree=ep_degree))

    def record_gemm(
        self,
        counts: Iterable[int],
        *,
        model_dim: int,
        hidden_dim: int,
        routed_counts: Optional[Iterable[int]] = None,
        alignment: Optional[int] = None,
        stage: Optional[str] = None,
    ) -> None:
        if self.enabled:
            self.record.grouped_gemm.append(
                grouped_gemm_statistics(
                    counts,
                    model_dim=model_dim,
                    hidden_dim=hidden_dim,
                    routed_counts=routed_counts,
                    alignment=alignment,
                    stage=stage,
                )
            )

    def record_padded_bmm(
        self,
        counts: Iterable[int],
        *,
        model_dim: int,
        hidden_dim: int,
        projection_count: int = 3,
    ) -> None:
        if self.enabled:
            self.record.grouped_gemm.append(
                padded_bmm_statistics(
                    counts,
                    model_dim=model_dim,
                    hidden_dim=hidden_dim,
                    projection_count=projection_count,
                )
            )

    def snapshot_memory(
        self,
        phase: str,
        device: torch.device | str,
        *,
        step: Optional[int] = None,
        microbatch: Optional[int] = None,
    ) -> None:
        if not self.enabled:
            return
        device = torch.device(device)
        if device.type == "xpu" and torch.xpu.is_available():
            free, total = torch.xpu.mem_get_info(device)
            allocated = torch.xpu.memory_allocated(device)
            reserved = torch.xpu.memory_reserved(device)
        elif device.type == "cuda" and torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(device)
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
        else:
            return
        self.record.memory.append(
            MemorySnapshot(
                phase,
                step,
                microbatch,
                allocated,
                reserved,
                free,
                total,
            )
        )


def snapshot_model_measurements(
    model: torch.nn.Module,
    phase: str,
    device: torch.device | str,
    *,
    step: Optional[int] = None,
    microbatch: Optional[int] = None,
) -> None:
    """Capture a phase snapshot for every enabled MoE module in ``model``."""
    for module in model.modules():
        collector = getattr(module, "measurement", None)
        if collector is not None and getattr(collector, "enabled", False):
            collector.snapshot_memory(
                phase,
                device,
                step=step,
                microbatch=microbatch,
            )


def export_model_measurements(
    model: torch.nn.Module,
    path: str,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    step_timings: Optional[Iterable[Mapping[str, Any]]] = None,
) -> None:
    """Write enabled MoE measurements for one rank to a JSON artifact."""
    records = {}
    for name, module in model.named_modules():
        collector = getattr(module, "measurement", None)
        if collector is not None and getattr(collector, "enabled", False):
            records[name or "<root>"] = collector.record.as_dict()
    if not records:
        return
    payload: dict[str, Any] = {
        "metadata": dict(metadata or {}),
        "records": records,
    }
    if step_timings is not None:
        payload["step_timings"] = list(step_timings)
    destination = os.fspath(path)
    os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
    temporary = f"{destination}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")
    os.replace(temporary, destination)


def mark_measurement_artifacts_complete(
    paths: Iterable[os.PathLike[str] | str],
    *,
    metadata_updates: Optional[Mapping[str, Any]] = None,
    minimum_global_step: Optional[int] = None,
    require_measurement_records: bool = False,
    require_passed_gates: bool = False,
    require_provenance: bool = False,
    require_execution_path: bool = False,
    expected_execution_path: Optional[str] = None,
    require_step_timing: bool = False,
    require_throughput_metrics: bool = False,
    require_declared_measurement_window: bool = False,
    required_step_phases: Iterable[str] = ("attention", "non_expert"),
    required_memory_phases: Iterable[str] = (),
    require_moe_metrics: bool = False,
    required_moe_timings: Optional[Iterable[str]] = None,
    required_collectives: Iterable[str] = (
        "dispatch_alltoall",
        "combine_alltoall",
    ),
) -> None:
    """Seal rank artifacts only after the measured launcher exits successfully."""
    if minimum_global_step is not None and minimum_global_step < 1:
        raise ValueError("minimum_global_step must be positive")
    valid_execution_paths = {"grouped_mm", "padded_bmm", "sequential"}
    required_memory_phases = {
        str(phase) for phase in required_memory_phases if str(phase).strip()
    }
    if (
        expected_execution_path is not None
        and expected_execution_path not in valid_execution_paths
    ):
        raise ValueError(
            f"expected_execution_path is invalid: {expected_execution_path!r}"
        )
    destinations = [os.fspath(path) for path in paths]
    if not destinations:
        raise ValueError("cannot seal an empty measurement artifact set")
    if len(destinations) != len(set(destinations)):
        raise ValueError("measurement artifact set contains duplicate paths")
    payloads = []
    for destination in destinations:
        with open(destination, encoding="utf-8") as input_file:
            payload = json.load(input_file)
        if not isinstance(payload, Mapping) or not isinstance(
            payload.get("metadata"), Mapping
        ):
            raise ValueError(f"measurement artifact has no metadata mapping: {destination}")
        if require_provenance:
            metadata = payload["metadata"]
            for name in ("model", "checkpoint", "source_revision"):
                value = metadata.get(name)
                if not isinstance(value, str) or not value.strip() or value == "unknown":
                    raise ValueError(
                        f"measurement artifact {name} must be non-placeholder: "
                        f"{destination}"
                    )
            uncommitted = metadata.get("uncommitted_change_state")
            if uncommitted not in {"clean", "dirty"}:
                raise ValueError(
                    "measurement artifact uncommitted_change_state must be "
                    f"'clean' or 'dirty': {destination}"
                )
        if require_moe_metrics:
            metadata = payload["metadata"]
            if "model" in metadata or "router_semantics" in metadata:
                validate_router_semantics(
                    metadata.get("model"),
                    metadata.get("router_semantics"),
                    context=f"canonical measurement artifact {destination}",
                )
        if minimum_global_step is not None:
            try:
                global_step = int(payload["metadata"]["global_step"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"measurement artifact has no valid global_step: {destination}"
                ) from error
            if global_step < minimum_global_step:
                raise ValueError(
                    f"measurement artifact ended at global_step={global_step}, "
                    f"expected at least {minimum_global_step}: {destination}"
                )
        payloads.append((destination, payload))

    required_phases = {
        str(phase) for phase in required_step_phases if str(phase).strip()
    }
    declared_paths = {
        payload.get("metadata", {}).get("expert_execution_path")
        for _, payload in payloads
        if isinstance(payload.get("metadata"), Mapping)
        and payload["metadata"].get("expert_execution_path") is not None
    }
    declared_path = (
        next(iter(declared_paths))
        if len(declared_paths) == 1
        and next(iter(declared_paths)) in valid_execution_paths
        else None
    )
    timing_execution_path = expected_execution_path or declared_path
    if required_moe_timings is None:
        required_timings = {
            "router",
            "expert_forward",
            "final_scatter",
        }
        if timing_execution_path is None:
            required_timings.update(
                {"grouped_gemm_gate", "grouped_gemm_up", "grouped_gemm_down"}
            )
    else:
        required_timings = {
            str(name) for name in required_moe_timings if str(name).strip()
        }
    path_timings = {
        "sequential": {
            "sequential_expert_compute",
            "sequential_expert_gate",
            "sequential_expert_up",
            "sequential_expert_down",
        },
        "padded_bmm": {"padded_bmm"},
        "grouped_mm": {
            "grouped_gemm_gate",
            "grouped_gemm_up",
            "grouped_gemm_down",
        },
    }
    if timing_execution_path is not None:
        required_timings.update(path_timings[timing_execution_path])
    required_collective_names = {
        str(name) for name in required_collectives if str(name).strip()
    }
    if require_step_timing and not required_phases:
        raise ValueError("required_step_phases must not be empty")
    if require_throughput_metrics and not require_step_timing:
        raise ValueError("require_throughput_metrics requires require_step_timing")
    if require_declared_measurement_window and not require_step_timing:
        raise ValueError(
            "require_declared_measurement_window requires require_step_timing"
        )
    if require_moe_metrics and not required_timings:
        raise ValueError("required_moe_timings must not be empty")

    if require_step_timing:
        step_records = []
        for destination, payload in payloads:
            step_timings = payload.get("step_timings", [])
            if not isinstance(step_timings, list):
                raise ValueError(
                    f"measurement artifact step_timings is not a list: {destination}"
                )
            for step in step_timings:
                if not isinstance(step, Mapping):
                    raise ValueError(
                        f"measurement artifact has non-mapping step timing: {destination}"
                    )
                step_records.append(step)
        observed_phases = {
            str(phase)
            for step in step_records
            for phase in (step.get("timings_s", {}) or {})
        }
        missing_phases = sorted(required_phases - observed_phases)
        if not step_records:
            raise ValueError("measurement artifacts have no step timing records")
        if missing_phases:
            raise ValueError(
                "measurement artifacts missing required step phases: "
                + ", ".join(missing_phases)
            )
        if require_declared_measurement_window:
            declared_window = [
                payload.get("metadata", {}).get("measurement_window")
                for _, payload in payloads
            ]
            if any(not isinstance(window, Mapping) for window in declared_window):
                raise ValueError(
                    "measurement artifacts must declare measurement_window"
                )
            normalized_windows = []
            for window in declared_window:
                try:
                    normalized_windows.append(
                        {
                            "warmup_steps": int(window["warmup_steps"]),
                            "measurement_steps": int(window["measurement_steps"]),
                            "steady_state_steps": int(window["steady_state_steps"]),
                        }
                    )
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        "measurement_window must declare integer warmup_steps, "
                        "measurement_steps, and steady_state_steps"
                    ) from error
            if any(window != normalized_windows[0] for window in normalized_windows[1:]):
                raise ValueError(
                    "measurement artifacts have inconsistent measurement_window"
                )
            try:
                warmup_steps = normalized_windows[0]["warmup_steps"]
                measurement_steps = normalized_windows[0]["measurement_steps"]
            except (IndexError, KeyError) as error:
                raise ValueError("measurement_window is empty") from error
            if warmup_steps < 0 or measurement_steps < 1:
                raise ValueError("measurement_window has invalid step counts")
            unique_steps = sorted(
                {int(step.get("step", 0)) for step in step_records}
            )
            available_steps = unique_steps[warmup_steps:]
            if len(available_steps) < measurement_steps:
                raise ValueError(
                    "measurement artifacts have fewer post-warmup step timing "
                    f"records than declared measurement_steps={measurement_steps}"
                )
        if require_throughput_metrics:
            required_fields = (
                "local_tokens",
                "global_tokens",
                "tokens_per_second_per_gpu",
                "aggregate_tokens_per_second",
            )
            for step in step_records:
                for field_name in required_fields:
                    value = step.get(field_name)
                    try:
                        numeric_value = float(value)
                    except (TypeError, ValueError) as error:
                        raise ValueError(
                            "measurement step is missing valid "
                            f"{field_name} throughput metric"
                        ) from error
                    if not math.isfinite(numeric_value) or numeric_value < 0:
                        raise ValueError(
                            "measurement step has invalid "
                            f"{field_name} throughput metric"
                        )

    for destination, payload in payloads:
        if required_memory_phases:
            records_payload = payload.get("records", {})
            if not isinstance(records_payload, Mapping):
                raise ValueError(
                    f"measurement artifact records is not a mapping: {destination}"
                )
            observed_memory_phases = {
                str(snapshot.get("phase"))
                for record in records_payload.values()
                if isinstance(record, Mapping)
                for snapshot in (record.get("memory", []) or [])
                if isinstance(snapshot, Mapping) and snapshot.get("phase") is not None
            }
            missing_memory_phases = sorted(
                required_memory_phases - observed_memory_phases
            )
            if missing_memory_phases:
                raise ValueError(
                    "measurement artifact missing required memory phases "
                    f"({', '.join(missing_memory_phases)}): {destination}"
                )
        if require_moe_metrics:
            records_payload = payload.get("records", {})
            if not isinstance(records_payload, Mapping):
                raise ValueError(
                    f"measurement artifact records is not a mapping: {destination}"
                )
            records = [
                record
                for record in records_payload.values()
                if isinstance(record, Mapping)
            ]
            observed_timings = {
                str(name)
                for record in records
                for name in (record.get("timings_s", {}) or {})
            }
            observed_collectives = {
                str(event.get("name"))
                for record in records
                for event in (record.get("collectives", []) or [])
                if isinstance(event, Mapping) and event.get("name") is not None
            }
            missing_timings = sorted(required_timings - observed_timings)
            missing_collectives = sorted(
                required_collective_names - observed_collectives
            )
            if missing_timings:
                raise ValueError(
                    "measurement artifact missing required MoE timings "
                    f"({', '.join(missing_timings)}): {destination}"
                )
            if missing_collectives:
                raise ValueError(
                    "measurement artifact missing required collectives "
                    f"({', '.join(missing_collectives)}): {destination}"
                )

    updates = dict(metadata_updates or {})
    updates["measurement_completion"] = "passed"
    for destination, payload in payloads:
        records = payload.get("records")
        if require_measurement_records and (
            not isinstance(records, Mapping) or not records
        ):
            raise ValueError(
                f"measurement artifact has no enabled MoE records: {destination}"
            )
        if require_passed_gates:
            effective_metadata = {**payload["metadata"], **updates}
            for name, expected in (
                ("device_health", "green"),
                ("gate_status", "passed"),
                ("semantic_completion", "passed"),
            ):
                if effective_metadata.get(name) != expected:
                    raise ValueError(
                        f"measurement artifact {name} must be {expected!r}: "
                        f"{destination}"
                    )
        execution_path = payload["metadata"].get("expert_execution_path")
        if require_execution_path and execution_path not in valid_execution_paths:
            raise ValueError(
                "measurement artifact requires a valid expert_execution_path: "
                f"{destination}"
            )
        if (
            expected_execution_path is not None
            and execution_path != expected_execution_path
        ):
            raise ValueError(
                "measurement artifact expert_execution_path disagrees with expected "
                f"{expected_execution_path!r}: {destination}"
            )
        environment_overrides = payload["metadata"].get("environment_overrides", {})
        if isinstance(environment_overrides, Mapping):
            _validate_binary_environment_overrides(environment_overrides, destination)
            grouped = str(
                environment_overrides.get("TORCHTUNE_MOE_GROUPED_EXPERTS", "0")
            ) == "1"
            sequential = str(
                environment_overrides.get("TORCHTUNE_MOE_SEQUENTIAL_EXPERTS", "0")
            ) == "1"
            inferred_path = "grouped_mm" if grouped else (
                "sequential" if sequential else "padded_bmm"
            )
            if execution_path is not None and execution_path != inferred_path:
                raise ValueError(
                    "measurement artifact expert_execution_path disagrees with "
                    f"environment_overrides: {destination}"
                )
    for destination, payload in payloads:
        metadata = {**payload["metadata"], **updates}
        sealed = {**payload, "metadata": metadata}
        temporary = f"{destination}.seal.tmp.{os.getpid()}"
        with open(temporary, "w", encoding="utf-8") as output:
            json.dump(sealed, output, indent=2, sort_keys=True)
            output.write("\n")
        os.replace(temporary, destination)


class _Timing:
    def __init__(self, collector: MoEMeasurementCollector, name: str) -> None:
        self.collector = collector
        self.name = name
        self.start = 0.0

    def __enter__(self):
        if self.collector.enabled:
            synchronize_measurement_device()
            self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.collector.enabled:
            synchronize_measurement_device()
            self.collector.record.add_timing(self.name, time.perf_counter() - self.start)
        return False


class _CollectiveTiming:
    def __init__(
        self,
        collector: MoEMeasurementCollector,
        name: str,
        scope: str,
        backend: str,
        locality: str,
    ) -> None:
        self.collector = collector
        self.name = name
        self.scope = scope
        self.backend = backend
        self.locality = locality
        self.start = 0.0

    def __enter__(self):
        if self.collector.enabled:
            synchronize_measurement_device()
            self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.collector.enabled:
            synchronize_measurement_device()
            duration = time.perf_counter() - self.start
            self.collector.record.record_collective(
                self.name,
                duration,
                scope=self.scope,
                backend=self.backend,
                locality=self.locality,
            )
        return False


def aggregate_rank_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate rank records without assuming equal expert ownership."""
    records = list(records)
    if not records:
        return {
            "rank_count": 0,
            "total_tokens": 0,
            "max_rank_tokens": 0,
            "min_rank_tokens": 0,
            "rank_token_imbalance_ratio": 0.0,
            "timings_s": {},
            "timing_counts": {},
            "step_timings": [],
            "collectives": [],
            "collective_locality": {},
            "grouped_gemm": [],
            "memory": [],
        }
    token_totals = [
        sum(int(item.get("total_tokens", 0)) for item in record.get("routed_tokens", []))
        for record in records
    ]
    timing_names = {
        name for record in records for name in record.get("timings_s", {})
    }
    timing_counts = {
        name
        for record in records
        for name in record.get("timing_counts", {})
    }
    step_timings: list[Mapping[str, Any]] = []
    for record in records:
        step_timings.extend(record.get("step_timings", []))
    collective_totals: dict[tuple[str, str, str, str], dict[str, float]] = {}
    for record in records:
        collective_events = record.get("collectives", [])
        if not isinstance(collective_events, list):
            raise ValueError("collective records must be lists")
        for event in collective_events:
            if not isinstance(event, Mapping):
                raise ValueError("collective events must be mappings")
            locality = str(event.get("locality", "unknown"))
            if locality not in _COLLECTIVE_LOCALITIES:
                raise ValueError(
                    f"invalid collective locality: {locality}"
                )
            key = (
                str(event.get("name", "unknown")),
                str(event.get("scope", "unknown")),
                str(event.get("backend", "unknown")),
                locality,
            )
            aggregate = collective_totals.setdefault(
                key, {"duration_s": 0.0, "count": 0}
            )
            aggregate["duration_s"] += float(event.get("duration_s", 0.0))
            aggregate["count"] += 1
    locality_totals: dict[str, dict[str, float]] = {}
    for (name, scope, backend, locality), totals in collective_totals.items():
        del name, scope, backend
        aggregate = locality_totals.setdefault(
            locality, {"duration_s": 0.0, "count": 0}
        )
        aggregate["duration_s"] += totals["duration_s"]
        aggregate["count"] += totals["count"]
    grouped_gemm_totals: dict[tuple[int, int, int, Optional[str]], dict[str, Any]] = {}
    for record in records:
        for index, stats in enumerate(record.get("grouped_gemm", [])):
            key = (
                index,
                int(stats.get("model_dim", 0)),
                int(stats.get("hidden_dim", 0)),
                stats.get("stage"),
            )
            aggregate = grouped_gemm_totals.setdefault(
                key,
                {
                    "rank_count": 0,
                    "total_tokens": 0,
                    "active_expert_gemm_count": 0,
                    "zero_token_experts": 0,
                    "max_tokens_per_expert": 0,
                    "counts": [],
                },
            )
            if stats.get("stage") is not None:
                aggregate["stage"] = stats["stage"]
            counts = [int(value) for value in stats.get("counts", [])]
            if not aggregate["counts"]:
                aggregate["counts"] = [0] * len(counts)
            if len(aggregate["counts"]) != len(counts):
                raise ValueError("grouped-GEMM records have inconsistent expert counts")
            aggregate["rank_count"] += 1
            aggregate["total_tokens"] += int(stats.get("total_tokens", sum(counts)))
            aggregate["active_expert_gemm_count"] += int(
                stats.get("active_expert_gemm_count", sum(value > 0 for value in counts))
            )
            aggregate["zero_token_experts"] += int(stats.get("zero_token_experts", 0))
            aggregate["max_tokens_per_expert"] = max(
                aggregate["max_tokens_per_expert"],
                int(stats.get("max_tokens_per_expert", max(counts, default=0))),
            )
            aggregate["counts"] = [
                previous + value
                for previous, value in zip(aggregate["counts"], counts)
            ]
            if "routed_counts" in stats:
                routed_counts = [int(value) for value in stats["routed_counts"]]
                if len(routed_counts) != len(counts):
                    raise ValueError("grouped-GEMM routed counts have inconsistent expert counts")
                if "routed_counts" not in aggregate:
                    aggregate["routed_counts"] = [0] * len(routed_counts)
                aggregate["routed_counts"] = [
                    previous + value
                    for previous, value in zip(aggregate["routed_counts"], routed_counts)
                ]
                aggregate["routed_tokens"] = aggregate.get("routed_tokens", 0) + int(
                    stats.get("routed_tokens", sum(routed_counts))
                )
                aggregate["compute_tokens"] = aggregate.get("compute_tokens", 0) + int(
                    stats.get("compute_tokens", sum(counts))
                )
                aggregate["padding_tokens"] = aggregate.get("padding_tokens", 0) + int(
                    stats.get("padding_tokens", 0)
                )
            if "dense_compute_tokens" in stats:
                aggregate["dense_compute_tokens"] = aggregate.get(
                    "dense_compute_tokens", 0
                ) + int(stats["dense_compute_tokens"])
                aggregate["dense_routed_tokens"] = aggregate.get(
                    "dense_routed_tokens", 0
                ) + int(stats.get("routed_tokens", sum(counts)))
                aggregate["dense_padding_tokens"] = aggregate.get(
                    "dense_padding_tokens", 0
                ) + int(stats.get("padding_tokens", 0))
                aggregate["max_count"] = max(
                    aggregate.get("max_count", 0), int(stats.get("max_count", 0))
                )
    for aggregate in grouped_gemm_totals.values():
        if "routed_tokens" in aggregate:
            routed_tokens = aggregate["routed_tokens"]
            compute_tokens = aggregate.get("compute_tokens", 0)
            aggregate["padding_fraction"] = (
                aggregate["padding_tokens"] / compute_tokens
                if compute_tokens
                else 0.0
            )
            aggregate["compute_to_routed_ratio"] = (
                compute_tokens / routed_tokens if routed_tokens else 0.0
            )
        if "dense_compute_tokens" in aggregate:
            dense_compute_tokens = aggregate["dense_compute_tokens"]
            dense_routed_tokens = aggregate.get("dense_routed_tokens", 0)
            aggregate["dense_padding_fraction"] = (
                aggregate["dense_padding_tokens"] / dense_compute_tokens
                if dense_compute_tokens
                else 0.0
            )
            aggregate["dense_to_routed_ratio"] = (
                dense_compute_tokens / dense_routed_tokens
                if dense_routed_tokens
                else 0.0
            )
    memory_totals: dict[tuple[Any, Any, Any, str], dict[str, Any]] = {}
    for record in records:
        for snapshot in record.get("memory", []):
            if isinstance(snapshot, Mapping):
                phase = str(snapshot.get("phase", "unknown"))
                step = snapshot.get("step")
                microbatch = snapshot.get("microbatch")
                values = snapshot
            else:
                phase = str(getattr(snapshot, "phase", "unknown"))
                step = getattr(snapshot, "step", None)
                microbatch = getattr(snapshot, "microbatch", None)
                values = asdict(snapshot)
            key = (step, microbatch, phase, str(values.get("device", "unknown")))
            aggregate = memory_totals.setdefault(
                key,
                {
                    "phase": phase,
                    "step": step,
                    "microbatch": microbatch,
                    "rank_count": 0,
                    "max_allocated_bytes": 0,
                    "max_reserved_bytes": 0,
                    "min_free_bytes": None,
                    "max_total_bytes": 0,
                },
            )
            aggregate["rank_count"] += 1
            allocated = values.get("allocated_bytes")
            reserved = values.get("reserved_bytes")
            free = values.get("free_bytes")
            total = values.get("total_bytes")
            if allocated is not None:
                aggregate["max_allocated_bytes"] = max(
                    aggregate["max_allocated_bytes"], int(allocated)
                )
            if reserved is not None:
                aggregate["max_reserved_bytes"] = max(
                    aggregate["max_reserved_bytes"], int(reserved)
                )
            if free is not None:
                aggregate["min_free_bytes"] = (
                    int(free)
                    if aggregate["min_free_bytes"] is None
                    else min(aggregate["min_free_bytes"], int(free))
                )
            if total is not None:
                aggregate["max_total_bytes"] = max(
                    aggregate["max_total_bytes"], int(total)
                )
    return {
        "rank_count": len(records),
        "total_tokens": sum(token_totals),
        "max_rank_tokens": max(token_totals),
        "min_rank_tokens": min(token_totals),
        "rank_token_imbalance_ratio": max(token_totals) / min(token_totals)
        if min(token_totals)
        else 0.0,
        "timings_s": {
            name: sum(float(record.get("timings_s", {}).get(name, 0.0)) for record in records)
            for name in timing_names
        },
        "timing_counts": {
            name: sum(int(record.get("timing_counts", {}).get(name, 0)) for record in records)
            for name in timing_counts
        },
        "step_timings": step_timings,
        "collectives": [
            {
                "name": name,
                "scope": scope,
                "backend": backend,
                "locality": locality,
                **totals,
            }
            for (name, scope, backend, locality), totals in sorted(
                collective_totals.items()
            )
        ],
        "collective_locality": {
            locality: totals
            for locality, totals in sorted(locality_totals.items())
        },
        "grouped_gemm": [
            {
                "event_index": index,
                "model_dim": model_dim,
                "hidden_dim": hidden_dim,
                **totals,
            }
            for (index, model_dim, hidden_dim, stage), totals in sorted(
                grouped_gemm_totals.items(),
                key=lambda item: (
                    item[0][0],
                    item[0][1],
                    item[0][2],
                    item[0][3] or "",
                ),
            )
        ],
        "memory": [
            totals
            for _, totals in sorted(
                memory_totals.items(),
                key=lambda item: (
                    item[0][0] is None,
                    item[0][0],
                    item[0][1] is None,
                    item[0][1],
                    item[0][2],
                ),
            )
        ],
    }


def summarize_step_timings(
    step_timings: Iterable[Mapping[str, Any]],
    *,
    warmup_steps: int = 0,
    measurement_steps: Optional[int] = None,
) -> dict[str, Any]:
    """Summarize measured step phases with a consistent warmup policy.

    Accepts either direct step records or the rank-wrapped ``step_timings``
    returned by :func:`aggregate_measurement_files`. Phase fractions use the
    sum of each phase divided by the sum of ``total_step_s`` over the retained
    samples, so overlapping or omitted sub-phases remain visible rather than
    being normalized away.
    """
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if measurement_steps is not None and measurement_steps < 1:
        raise ValueError("measurement_steps must be positive when provided")
    records: list[Mapping[str, Any]] = []
    for item in step_timings:
        if not isinstance(item, Mapping):
            raise ValueError("step timing entries must be mappings")
        nested = item.get("steps")
        if nested is None:
            records.append(item)
        else:
            if not isinstance(nested, list):
                raise ValueError("rank-wrapped step timings must contain a list")
            records.extend(nested)
    ordered = sorted(records, key=lambda item: int(item.get("step", 0)))
    unique_steps = sorted({int(item.get("step", 0)) for item in ordered})
    selected_steps = unique_steps[warmup_steps:]
    if measurement_steps is not None:
        if len(selected_steps) < measurement_steps:
            raise ValueError(
                "step timing entries have fewer post-warmup steps than "
                f"measurement_steps={measurement_steps}"
            )
        selected_steps = selected_steps[:measurement_steps]
    discarded_steps = set(unique_steps[:warmup_steps])
    selected_step_set = set(selected_steps)
    retained = [
        item
        for item in ordered
        if int(item.get("step", 0)) in selected_step_set
    ]
    totals: list[float] = []
    local_token_totals: list[float] = []
    global_token_totals: list[float] = []
    throughput_per_gpu: list[float] = []
    aggregate_throughput: list[float] = []
    phase_totals: dict[str, float] = {}
    phase_counts: dict[str, int] = {}
    for item in retained:
        total = item.get("total_step_s", item.get("time_per_step_s"))
        if total is None:
            raise ValueError("step timing entry has no total_step_s")
        total = float(total)
        if not math.isfinite(total) or total < 0:
            raise ValueError("step timing totals must be finite and non-negative")
        totals.append(total)
        local_tokens = item.get("local_tokens")
        global_tokens = item.get("global_tokens")
        if local_tokens is not None:
            local_tokens = float(local_tokens)
            if not math.isfinite(local_tokens) or local_tokens < 0:
                raise ValueError("step local token counts must be finite and non-negative")
            local_token_totals.append(local_tokens)
        if global_tokens is not None:
            global_tokens = float(global_tokens)
            if not math.isfinite(global_tokens) or global_tokens < 0:
                raise ValueError("step global token counts must be finite and non-negative")
            global_token_totals.append(global_tokens)
        per_gpu = item.get("tokens_per_second_per_gpu")
        aggregate = item.get("aggregate_tokens_per_second")
        if per_gpu is not None:
            per_gpu = float(per_gpu)
            if not math.isfinite(per_gpu) or per_gpu < 0:
                raise ValueError("per-GPU throughput must be finite and non-negative")
            throughput_per_gpu.append(per_gpu)
        if aggregate is not None:
            aggregate = float(aggregate)
            if not math.isfinite(aggregate) or aggregate < 0:
                raise ValueError("aggregate throughput must be finite and non-negative")
            aggregate_throughput.append(aggregate)
        phases = item.get("timings_s", {})
        if not isinstance(phases, Mapping):
            raise ValueError("step timing phases must be a mapping")
        for name, value in phases.items():
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise ValueError("step phase timings must be finite and non-negative")
            phase_totals[str(name)] = phase_totals.get(str(name), 0.0) + value
            phase_counts[str(name)] = phase_counts.get(str(name), 0) + 1
    total_sum = sum(totals)
    sample_count = len(retained)
    return {
        "warmup_steps_discarded": len(discarded_steps),
        "measurement_steps_requested": measurement_steps,
        "measurement_steps_used": len(selected_steps),
        "sample_count": sample_count,
        "step_start": min((int(item.get("step", 0)) for item in retained), default=None),
        "step_end": max((int(item.get("step", 0)) for item in retained), default=None),
        "mean_total_step_s": total_sum / sample_count if sample_count else 0.0,
        "mean_local_tokens": (
            sum(local_token_totals) / len(local_token_totals)
            if local_token_totals
            else None
        ),
        "mean_global_tokens": (
            sum(global_token_totals) / len(global_token_totals)
            if global_token_totals
            else None
        ),
        "mean_tokens_per_second_per_gpu": (
            sum(throughput_per_gpu) / len(throughput_per_gpu)
            if throughput_per_gpu
            else None
        ),
        "mean_aggregate_tokens_per_second": (
            sum(aggregate_throughput) / len(aggregate_throughput)
            if aggregate_throughput
            else None
        ),
        "phase_mean_s": {
            name: total / phase_counts[name] for name, total in sorted(phase_totals.items())
        },
        "phase_fraction_of_total": {
            name: total / total_sum if total_sum else 0.0
            for name, total in sorted(phase_totals.items())
        },
        "phase_sample_counts": dict(sorted(phase_counts.items())),
    }


def summarize_pipeline_timings(
    phase_timings: Mapping[str, Any], metadata: Mapping[str, Any]
) -> dict[str, Any]:
    """Summarize explicitly recorded pipeline bubble and transfer phases.

    The helper intentionally does not estimate a bubble from pipeline degree
    or stage count. A capacity result may report pipeline attribution only when
    the launcher/recipe records the corresponding phase names.
    """
    if not isinstance(phase_timings, Mapping):
        raise ValueError("phase_timings must be a mapping")
    phase_means = phase_timings.get("phase_mean_s", {})
    if not isinstance(phase_means, Mapping):
        phase_means = {}

    def phase_seconds(names: set[str]) -> Optional[float]:
        values = []
        for name, value in phase_means.items():
            if str(name).lower() in names:
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError) as error:
                    raise ValueError("pipeline phase timings must be numeric") from error
                if not math.isfinite(numeric_value) or numeric_value < 0:
                    raise ValueError(
                        "pipeline phase timings must be finite and non-negative"
                    )
                values.append(numeric_value)
        return sum(values) if values else None

    bubble_seconds = phase_seconds({"pipeline_bubble", "bubble"})
    activation_transfer_seconds = phase_seconds(
        {
            "activation_transfer",
            "activation_transfer_forward",
            "activation_transfer_backward",
            "pipeline_activation_transfer",
        }
    )
    total_seconds = phase_timings.get("mean_total_step_s")
    try:
        total_seconds = float(total_seconds)
    except (TypeError, ValueError):
        total_seconds = None
    if total_seconds is not None and (
        not math.isfinite(total_seconds) or total_seconds < 0
    ):
        raise ValueError("pipeline total step time must be finite and non-negative")
    degree = metadata.get("pipeline_parallel_degree", metadata.get("pp"))
    try:
        degree = int(degree) if degree is not None else None
    except (TypeError, ValueError) as error:
        raise ValueError("pipeline parallel degree must be an integer") from error
    return {
        "pipeline_parallel_degree": degree,
        "bubble_seconds": bubble_seconds,
        "activation_transfer_seconds": activation_transfer_seconds,
        "bubble_fraction": (
            bubble_seconds / total_seconds
            if bubble_seconds is not None and total_seconds and total_seconds > 0
            else None
        ),
        "activation_transfer_fraction": (
            activation_transfer_seconds / total_seconds
            if activation_transfer_seconds is not None
            and total_seconds
            and total_seconds > 0
            else None
        ),
        "timing_recorded": bubble_seconds is not None
        or activation_transfer_seconds is not None,
    }


def summarize_ep_scaling_artifact(
    artifact: Mapping[str, Any], *, warmup_steps: int = 0
) -> dict[str, Any]:
    """Build a table-ready EP scaling summary from an aggregated artifact.

    Routed counts are assignment counts, not unique input-token counts: a
    top-k router contributes one assignment per selected expert.  The summary
    keeps that distinction explicit and does not infer throughput, MFU, or
    device health from partial artifacts.
    """
    if not isinstance(artifact, Mapping):
        raise ValueError("EP scaling artifact must be a mapping")
    records = artifact.get("records", {})
    if not isinstance(records, Mapping):
        raise ValueError("EP scaling artifact records must be a mapping")

    module_records = [
        record for record in records.values() if isinstance(record, Mapping)
    ]
    # Aggregated module records intentionally do not retain the raw routed
    # layer list. Their total_tokens remains the authoritative global count.
    routed_layers = [
        {
            "module": name,
            "global_routed_tokens": int(record.get("total_tokens", 0)),
            "rank_count": int(record.get("rank_count", 0)),
            "local_routed_tokens": (
                float(record.get("total_tokens", 0))
                / int(record.get("rank_count", 1))
                if int(record.get("rank_count", 0))
                else 0.0
            ),
            "max_rank_tokens": int(record.get("max_rank_tokens", 0)),
            "min_rank_tokens": int(record.get("min_rank_tokens", 0)),
            "rank_token_imbalance_ratio": float(
                record.get("rank_token_imbalance_ratio", 0.0)
            ),
        }
        for name, record in records.items()
        if isinstance(record, Mapping) and "total_tokens" in record
    ]

    gemm_events = [
        event
        for record in module_records
        for event in record.get("grouped_gemm", [])
        if isinstance(event, Mapping)
    ]
    gemm_by_stage: dict[str, dict[str, float]] = {}
    for event in gemm_events:
        stage = str(event.get("stage", "unspecified"))
        summary = gemm_by_stage.setdefault(
            stage,
            {
                "event_count": 0,
                "compute_tokens": 0,
                "routed_tokens": 0,
                "padding_tokens": 0,
                "zero_token_experts": 0,
                "active_expert_gemm_count": 0,
                "max_tokens_per_expert": 0,
                "_counts": [],
                "_routed_counts": [],
                "_rank_count": 0,
            },
        )
        summary["event_count"] += 1
        summary["_rank_count"] += int(event.get("rank_count", 1))
        event_counts = [int(value) for value in event.get("counts", [])]
        if not summary["_counts"]:
            summary["_counts"] = [0] * len(event_counts)
        if len(summary["_counts"]) != len(event_counts):
            raise ValueError("grouped-GEMM summary has inconsistent expert counts")
        summary["_counts"] = [
            previous + value
            for previous, value in zip(summary["_counts"], event_counts)
        ]
        event_routed_counts = [
            int(value) for value in event.get("routed_counts", event_counts)
        ]
        if not summary["_routed_counts"]:
            summary["_routed_counts"] = [0] * len(event_routed_counts)
        if len(summary["_routed_counts"]) != len(event_routed_counts):
            raise ValueError(
                "grouped-GEMM summary has inconsistent routed expert counts"
            )
        summary["_routed_counts"] = [
            previous + value
            for previous, value in zip(
                summary["_routed_counts"], event_routed_counts
            )
        ]
        for key in (
            "compute_tokens",
            "routed_tokens",
            "padding_tokens",
            "zero_token_experts",
            "active_expert_gemm_count",
        ):
            summary[key] += int(event.get(key, 0))
        summary["max_tokens_per_expert"] = max(
            summary["max_tokens_per_expert"],
            int(event.get("max_tokens_per_expert", 0)),
        )
    for summary in gemm_by_stage.values():
        routed = summary["routed_tokens"]
        compute = summary["compute_tokens"]
        counts = summary.pop("_counts")
        routed_counts = summary.pop("_routed_counts")
        rank_count = summary.pop("_rank_count")
        expert_count = len(counts)
        total_count = sum(routed_counts)
        mean_count = total_count / expert_count if expert_count else 0.0
        summary["padding_fraction"] = (
            summary["padding_tokens"] / compute if compute else 0.0
        )
        summary["compute_to_routed_ratio"] = compute / routed if routed else 0.0
        summary["expert_count"] = expert_count
        summary["routed_counts"] = routed_counts
        summary["max_routed_tokens_per_expert"] = max(routed_counts, default=0)
        summary["zero_routed_token_experts"] = sum(
            value == 0 for value in routed_counts
        )
        summary["tokens_per_local_expert"] = (
            total_count / (expert_count * rank_count)
            if expert_count and rank_count
            else 0.0
        )
        summary["expert_imbalance_ratio"] = (
            max(routed_counts, default=0) / mean_count if mean_count else 0.0
        )

    collective_summary = [
        event
        for record in module_records
        for event in record.get("collectives", [])
        if isinstance(event, Mapping)
    ]
    collective_by_name: dict[str, dict[str, float]] = {}
    collective_by_locality: dict[str, dict[str, float]] = {}
    for event in collective_summary:
        duration = float(event.get("duration_s", 0.0))
        name = str(event.get("name", "unknown"))
        locality = str(event.get("locality", "unknown"))
        count = int(event.get("count", 1))
        for destination, key in (
            (collective_by_name, name),
            (collective_by_locality, locality),
        ):
            totals = destination.setdefault(key, {"duration_s": 0.0, "count": 0})
            totals["duration_s"] += duration
            totals["count"] += count

    measurement_window = artifact.get("common_metadata", {}).get(
        "measurement_window", {}
    )
    declared_measurement_steps = None
    if isinstance(measurement_window, Mapping) and "measurement_steps" in measurement_window:
        declared_measurement_steps = int(measurement_window["measurement_steps"])
    phase_timings = summarize_step_timings(
        artifact.get("step_timings", []),
        warmup_steps=warmup_steps,
        measurement_steps=declared_measurement_steps,
    )
    metadata = artifact.get("common_metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    transport_override = metadata.get("environment_overrides", {})
    if not isinstance(transport_override, Mapping):
        transport_override = {}
    transport = (
        "alltoall"
        if str(transport_override.get("TORCHTUNE_EP_ALL2ALL", "1")) == "1"
        else "allgather_reduce_scatter"
    )
    execution_path = metadata.get("expert_execution_path")
    model_metric_names = (
        "mfu_percent",
        "active_flop_efficiency",
        "total_parameters",
        "active_parameters_per_token",
    )
    model_metrics: dict[str, float] = {}
    for name in model_metric_names:
        if name not in metadata or metadata[name] is None:
            continue
        try:
            value = float(metadata[name])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} metadata must be finite and non-negative") from error
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} metadata must be finite and non-negative")
        model_metrics[name] = value
    timing_totals: dict[str, float] = {}
    for record in module_records:
        for name, value in record.get("timings_s", {}).items():
            timing_totals[str(name)] = timing_totals.get(str(name), 0.0) + float(value)
    timing_counts = {
        str(name): sum(
            int(record.get("timing_counts", {}).get(name, 0))
            for record in module_records
        )
        for name in {
            name
            for record in module_records
            for name in record.get("timing_counts", {})
        }
    }
    expert_timing_names = {
        "sequential": (
            "sequential_expert_compute",
            "sequential_expert_gate",
            "sequential_expert_up",
            "sequential_expert_down",
        ),
        "grouped_mm": ("grouped_gemm_gate", "grouped_gemm_up", "grouped_gemm_down"),
        "padded_bmm": ("padded_bmm",),
    }
    selected_timing_names = expert_timing_names.get(str(execution_path), ())
    aggregate_timing_name = {
        "sequential": "sequential_expert_compute",
        "padded_bmm": "padded_bmm",
    }.get(str(execution_path))
    expert_timing = {
        "by_timing": {
            name: {
                "seconds": timing_totals.get(name, 0.0),
                "event_count": timing_counts.get(name, 0),
            }
            for name in selected_timing_names
            if name in timing_totals or name in timing_counts
        },
        "total_seconds": timing_totals.get(aggregate_timing_name, 0.0)
        if aggregate_timing_name is not None
        else sum(timing_totals.get(name, 0.0) for name in selected_timing_names),
        "event_count": timing_counts.get(aggregate_timing_name, 0)
        if aggregate_timing_name is not None
        else sum(timing_counts.get(name, 0) for name in selected_timing_names),
    }
    if execution_path == "padded_bmm":
        expert_timing["by_timing"] = {
            "padded_bmm": {
                "seconds": timing_totals.get("padded_bmm", 0.0),
                "event_count": timing_counts.get("padded_bmm", 0),
            }
        }
    routing_metadata_timing = {
        "materialization_seconds": (
            timing_totals.get("routing_metadata_materialization")
            if "routing_metadata_materialization" in timing_totals
            or "routing_metadata_materialization" in timing_counts
            else None
        ),
        "materialization_event_count": timing_counts.get(
            "routing_metadata_materialization", 0
        ),
        "permutation_seconds": (
            timing_totals.get("routing_metadata_permutation")
            if "routing_metadata_permutation" in timing_totals
            or "routing_metadata_permutation" in timing_counts
            else None
        ),
        "permutation_event_count": timing_counts.get(
            "routing_metadata_permutation", 0
        ),
    }
    routing_phase_timing = {
        name: {
            "seconds": timing_totals.get(name, 0.0),
            "event_count": timing_counts.get(name, 0),
        }
        for name in (
            "dispatch_pack",
            "dispatch_unpack",
            "dispatch_backward_pack",
            "dispatch_backward_unpack",
            "combine_pack",
            "combine_unpack",
            "combine_backward_pack",
            "combine_backward_unpack",
        )
        if name in timing_totals or name in timing_counts
    }
    memory = [
        snapshot
        for record in module_records
        for snapshot in record.get("memory", [])
        if isinstance(snapshot, Mapping)
    ]
    peak_memory = {
        "max_allocated_bytes": max(
            (
                int(item.get("max_allocated_bytes", item.get("allocated_bytes", 0)))
                for item in memory
            ),
            default=0,
        ),
        "max_reserved_bytes": max(
            (
                int(item.get("max_reserved_bytes", item.get("reserved_bytes", 0)))
                for item in memory
            ),
            default=0,
        ),
        "min_free_bytes": min(
            (
                int(
                    item["min_free_bytes"]
                    if "min_free_bytes" in item
                    else item["free_bytes"]
                )
                for item in memory
                if (
                    item.get("min_free_bytes")
                    if "min_free_bytes" in item
                    else item.get("free_bytes")
                )
                is not None
            ),
            default=None,
        ),
    }
    memory_by_phase: dict[str, dict[str, Optional[int]]] = {}
    for item in memory:
        phase = str(item.get("phase", "unknown"))
        phase_summary = memory_by_phase.setdefault(
            phase,
            {
                "max_allocated_bytes": 0,
                "max_reserved_bytes": 0,
                "min_free_bytes": None,
            },
        )
        phase_summary["max_allocated_bytes"] = max(
            int(phase_summary["max_allocated_bytes"] or 0),
            int(item.get("max_allocated_bytes", item.get("allocated_bytes", 0))),
        )
        phase_summary["max_reserved_bytes"] = max(
            int(phase_summary["max_reserved_bytes"] or 0),
            int(item.get("max_reserved_bytes", item.get("reserved_bytes", 0))),
        )
        free = item.get("min_free_bytes", item.get("free_bytes"))
        if free is not None:
            phase_summary["min_free_bytes"] = (
                int(free)
                if phase_summary["min_free_bytes"] is None
                else min(int(phase_summary["min_free_bytes"]), int(free))
            )
    return {
        "metadata": dict(metadata),
        "transport": transport,
        "model_metrics": model_metrics,
        "token_semantics": "global routed assignments; top-k selections counted separately",
        "routed_tokens": {
            "layer_count": len(routed_layers),
            "layers": routed_layers,
        },
        "grouped_gemm": {
            "event_count": len(gemm_events),
            "by_stage": dict(sorted(gemm_by_stage.items())),
        },
        "expert_compute": {
            "execution_path": execution_path,
            "timing": expert_timing,
        },
        "communication": {
            "by_collective": dict(sorted(collective_by_name.items())),
            "by_locality": dict(sorted(collective_by_locality.items())),
        },
        "routing_metadata": routing_metadata_timing,
        "routing_phases": routing_phase_timing,
        "phase_timings": phase_timings,
        "pipeline": summarize_pipeline_timings(phase_timings, metadata),
        "throughput": {
            "tokens_per_second_per_gpu": phase_timings[
                "mean_tokens_per_second_per_gpu"
            ],
            "aggregate_tokens_per_second": phase_timings[
                "mean_aggregate_tokens_per_second"
            ],
            "mean_local_tokens": phase_timings["mean_local_tokens"],
            "mean_global_tokens": phase_timings["mean_global_tokens"],
        },
        "peak_memory": peak_memory,
        "memory_by_phase": memory_by_phase,
        "steady_state_memory": memory_by_phase.get("steady_state", {}),
    }


def compare_ep_scaling_summaries(
    summaries: Mapping[int | str, Mapping[str, Any]],
    *,
    baseline_ep_degree: int = 8,
    require_control_metadata: bool = False,
) -> dict[str, Any]:
    """Build a comparable EP table from already validated summaries.

    Scaling efficiency is only computed when both summaries expose positive
    ``world_size`` metadata.  The expected aggregate throughput then scales
    the baseline by the world-size ratio, matching the EP8-per-node reference
    convention without silently assuming a topology.  Missing values remain
    ``None`` rather than becoming performance claims.
    """
    if not isinstance(summaries, Mapping) or not summaries:
        raise ValueError("EP scaling summaries must be a non-empty mapping")
    normalized: dict[int, Mapping[str, Any]] = {}
    for key, summary in summaries.items():
        try:
            ep_degree = int(key)
        except (TypeError, ValueError) as error:
            raise ValueError("EP scaling summary keys must be integer EP degrees") from error
        if ep_degree < 1 or not isinstance(summary, Mapping):
            raise ValueError("EP scaling summaries must use positive integer degrees")
        normalized[ep_degree] = summary
    if baseline_ep_degree not in normalized:
        raise ValueError(f"missing baseline EP{baseline_ep_degree} summary")

    control_fields = (
        "model",
        "checkpoint",
        "source_revision",
        "uncommitted_change_state",
        "sequence_length",
        "batch_size",
        "microbatch_size",
        "gradient_accumulation_steps",
        "optimizer",
        "optimization_profile",
        "routing_index_mode",
        "expert_execution_path",
    )
    controlled_environment = (
        "TORCHTUNE_EP_ALL2ALL",
        "TORCHTUNE_MOE_ALLTOALL_CONDITIONAL_CONTIGUOUS",
        "TORCHTUNE_EP_INPLACE_AG_ANCHOR",
        "TORCHTUNE_EP_SINGLE_ROW_AG_ANCHOR",
        "TORCHTUNE_EP_ZERO_COST_AG_ANCHOR",
        "TORCHTUNE_EP_UNINITIALIZED_COLLECTIVE_BUFFERS",
        "TORCHTUNE_EP_ROWWISE_ALLTOALL_UNPERMUTE",
        "TORCHTUNE_EP_UNINITIALIZED_ALLTOALL_BUFFERS",
        "TORCHTUNE_EP_FUSED_ALLTOALL_ROUTING",
        "TORCHTUNE_EP_CPU_METADATA_TRANSFER",
        "TORCHTUNE_EP_DIRECT_CPU_COPY",
        "TORCHTUNE_EP_DEVICE_ROUTING_METADATA",
        "TORCHTUNE_EP_CPU_VECTOR_ROUTING_METADATA",
        "TORCHTUNE_EP_PACK_ROUTING_METADATA_TRANSFER",
        "TORCHTUNE_EP_INDEX_ADD_COMBINE",
        "TORCHTUNE_MOE_INDEX_SELECT_PACKING",
        "TORCHTUNE_MOE_INDEX_ADD_FINAL_SCATTER",
        "TORCHTUNE_MOE_INPLACE_ROUTE_WEIGHTING",
        "TORCHTUNE_MOE_INPLACE_FINAL_SCATTER",
        "TORCHTUNE_MOE_INPLACE_SWIGLU",
        "TORCHTUNE_MOE_TOPK_ROUTING",
        "TORCHTUNE_MOE_UNSTABLE_EXPERT_GROUPING",
        "TORCHTUNE_MOE_GROUPED_EXPERTS",
        "TORCHTUNE_MOE_SEQUENTIAL_EXPERTS",
        "TORCHTUNE_MOE_GROUPED_RECOMPUTE_PREACT",
        "TORCHTUNE_MOE_VECTOR_PACKING",
        "TORCHTUNE_EP_GRAD_RELEASE_XCCL",
        "TORCHTUNE_EP_GRAD_RELEASE_STREAMING",
        "TORCHTUNE_MOE_NATIVE_FSDP_GRAD_REDUCE",
        "TORCHTUNE_MOE_OPTIMIZER_COMPONENT",
        "TORCHTUNE_MOE_OPTIMIZER_FUSED",
    )

    def control_signature(summary: Mapping[str, Any]) -> dict[str, Any]:
        metadata = summary.get("metadata", {})
        if not isinstance(metadata, Mapping):
            return {}
        signature = {
            field: metadata[field]
            for field in control_fields
            if field in metadata and metadata[field] is not None
        }
        environment = metadata.get("environment_overrides", {})
        if isinstance(environment, Mapping):
            for name in controlled_environment:
                if name in environment:
                    signature[name] = str(environment[name])
        return signature

    baseline_signature = control_signature(normalized[baseline_ep_degree])
    if require_control_metadata:
        required_fields = set(control_fields)
        required_environment = set(controlled_environment)
        for ep_degree, summary in normalized.items():
            metadata = summary.get("metadata", {})
            environment = (
                metadata.get("environment_overrides", {})
                if isinstance(metadata, Mapping)
                else {}
            )
            missing_fields = sorted(required_fields - set(control_signature(summary)))
            missing_environment = sorted(
                required_environment - set(environment)
                if isinstance(environment, Mapping)
                else required_environment
            )
            if missing_fields or missing_environment:
                missing = missing_fields + missing_environment
                raise ValueError(
                    f"EP{ep_degree} summary is missing control metadata: "
                    + ", ".join(missing)
                )
    for ep_degree, summary in normalized.items():
        if ep_degree == baseline_ep_degree:
            continue
        candidate_signature = control_signature(summary)
        mismatches = [
            field
            for field in sorted(set(baseline_signature) & set(candidate_signature))
            if baseline_signature[field] != candidate_signature[field]
        ]
        if mismatches:
            raise ValueError(
                "EP scaling summaries use incompatible controls: "
                + ", ".join(mismatches)
            )

    def numeric(value: Any) -> Optional[float]:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) and result >= 0 else None

    def world_size(summary: Mapping[str, Any]) -> Optional[float]:
        return numeric(summary.get("metadata", {}).get("world_size"))

    baseline = normalized[baseline_ep_degree]
    baseline_throughput = numeric(
        baseline.get("throughput", {}).get("aggregate_tokens_per_second")
    )
    baseline_world_size = world_size(baseline)
    rows = []
    for ep_degree in sorted(normalized):
        summary = normalized[ep_degree]
        metadata = summary.get("metadata", {})
        if not isinstance(metadata, Mapping):
            metadata = {}
        throughput = summary.get("throughput", {})
        if not isinstance(throughput, Mapping):
            throughput = {}
        aggregate = numeric(throughput.get("aggregate_tokens_per_second"))
        current_world_size = world_size(summary)
        expected = (
            baseline_throughput * current_world_size / baseline_world_size
            if baseline_throughput is not None
            and baseline_world_size is not None
            and current_world_size is not None
            and baseline_world_size > 0
            else None
        )
        efficiency = (
            aggregate / expected
            if aggregate is not None and expected is not None and expected > 0
            else None
        )
        communication = summary.get("communication", {})
        by_locality = (
            communication.get("by_locality", {})
            if isinstance(communication, Mapping)
            else {}
        )
        communication_seconds = sum(
            numeric(value.get("duration_s")) or 0.0
            for value in by_locality.values()
            if isinstance(value, Mapping)
        )
        by_collective = communication.get("by_collective", {})
        if not isinstance(by_collective, Mapping):
            by_collective = {}

        def collective_seconds(names: set[str]) -> float:
            return sum(
                numeric(value.get("duration_s")) or 0.0
                for name, value in by_collective.items()
                if str(name).lower() in names and isinstance(value, Mapping)
            )

        dispatch_seconds = collective_seconds(
            {
                "dispatch",
                "dispatch_alltoall",
                "dispatch_all_to_all",
                "dispatch_backward_alltoall",
                "dispatch_backward_all_to_all",
            }
        )
        combine_seconds = collective_seconds(
            {
                "combine",
                "combine_alltoall",
                "combine_all_to_all",
                "combine_backward_alltoall",
                "combine_backward_all_to_all",
            }
        )
        routing_metadata_seconds = collective_seconds(
            {
                "routing_metadata",
                "routing_metadata_allgather",
                "routing_metadata_all_gather",
            }
        )
        routing_phases = summary.get("routing_phases", {})
        if not isinstance(routing_phases, Mapping):
            routing_phases = {}

        def routing_phase_seconds(names: set[str]) -> Optional[float]:
            values = [
                numeric(value.get("seconds"))
                for name, value in routing_phases.items()
                if str(name) in names and isinstance(value, Mapping)
            ]
            if not values or any(value is None for value in values):
                return None
            return sum(value for value in values if value is not None)

        dispatch_pack_seconds = routing_phase_seconds(
            {"dispatch_pack", "dispatch_unpack"}
        )
        dispatch_backward_pack_seconds = routing_phase_seconds(
            {"dispatch_backward_pack", "dispatch_backward_unpack"}
        )
        combine_pack_seconds = routing_phase_seconds(
            {"combine_pack", "combine_unpack"}
        )
        combine_backward_pack_seconds = routing_phase_seconds(
            {"combine_backward_pack", "combine_backward_unpack"}
        )
        routing_pack_unpack_values = (
            dispatch_pack_seconds,
            dispatch_backward_pack_seconds,
            combine_pack_seconds,
            combine_backward_pack_seconds,
        )
        routing_pack_unpack_seconds = (
            sum(value for value in routing_pack_unpack_values if value is not None)
            if all(value is not None for value in routing_pack_unpack_values)
            else None
        )
        grouped_gemm = summary.get("grouped_gemm", {})
        stages = (
            grouped_gemm.get("by_stage", {})
            if isinstance(grouped_gemm, Mapping)
            else {}
        )
        grouped_expert_compute = {
            "event_count": sum(
                int(value.get("event_count", 0))
                for value in stages.values()
                if isinstance(value, Mapping)
            ),
            "compute_tokens": sum(
                int(value.get("compute_tokens", 0))
                for value in stages.values()
                if isinstance(value, Mapping)
            ),
            "routed_tokens": sum(
                int(value.get("routed_tokens", 0))
                for value in stages.values()
                if isinstance(value, Mapping)
            ),
            "padding_tokens": sum(
                int(value.get("padding_tokens", 0))
                for value in stages.values()
                if isinstance(value, Mapping)
            ),
            "zero_token_experts": sum(
                int(value.get("zero_routed_token_experts", 0))
                for value in stages.values()
                if isinstance(value, Mapping)
            ),
            "tokens_per_local_expert": (
                sum(
                    float(value.get("tokens_per_local_expert", 0.0))
                    for value in stages.values()
                    if isinstance(value, Mapping)
                )
                / len([value for value in stages.values() if isinstance(value, Mapping)])
                if any(isinstance(value, Mapping) for value in stages.values())
                else None
            ),
            "max_expert_imbalance_ratio": max(
                (
                    float(value.get("expert_imbalance_ratio", 0.0))
                    for value in stages.values()
                    if isinstance(value, Mapping)
                ),
                default=0.0,
            ),
            "padding_fraction": (
                sum(
                    int(value.get("padding_tokens", 0))
                    for value in stages.values()
                    if isinstance(value, Mapping)
                )
                / sum(
                    int(value.get("compute_tokens", 0))
                    for value in stages.values()
                    if isinstance(value, Mapping)
                )
                if sum(
                    int(value.get("compute_tokens", 0))
                    for value in stages.values()
                    if isinstance(value, Mapping)
                )
                else 0.0
            ),
        }
        reported_expert_compute = summary.get("expert_compute", {})
        if not isinstance(reported_expert_compute, Mapping):
            reported_expert_compute = {}
        timing = reported_expert_compute.get("timing", {})
        if not isinstance(timing, Mapping):
            timing = {}
        expert_compute = {
            **grouped_expert_compute,
            "execution_path": reported_expert_compute.get("execution_path"),
            "timing_seconds": numeric(timing.get("total_seconds")),
            "timing_event_count": int(timing.get("event_count", 0)),
            "timing_by_name": dict(timing.get("by_timing", {}))
            if isinstance(timing.get("by_timing", {}), Mapping)
            else {},
            "by_stage": dict(stages),
        }
        routed = summary.get("routed_tokens", {})
        routed_layers = (
            routed.get("layers", []) if isinstance(routed, Mapping) else []
        )
        routed_layers = [
            item for item in routed_layers if isinstance(item, Mapping)
        ]
        routed_tokens = {
            "layer_count": len(routed_layers),
            "global_assignments": sum(
                int(item.get("global_routed_tokens", 0)) for item in routed_layers
            ),
            "mean_local_assignments": (
                sum(float(item.get("local_routed_tokens", 0.0)) for item in routed_layers)
                / len(routed_layers)
                if routed_layers
                else None
            ),
            "max_rank_tokens": max(
                (int(item.get("max_rank_tokens", 0)) for item in routed_layers),
                default=0,
            ),
            "min_rank_tokens": min(
                (int(item.get("min_rank_tokens", 0)) for item in routed_layers),
                default=0,
            ),
            "max_imbalance_ratio": max(
                (float(item.get("rank_token_imbalance_ratio", 0.0)) for item in routed_layers),
                default=0.0,
            ),
        }
        rows.append(
            {
                "ep_degree": ep_degree,
                "world_size": current_world_size,
                "batch_size": metadata.get("batch_size"),
                "microbatch_size": metadata.get("microbatch_size"),
                "gradient_accumulation_steps": metadata.get(
                    "gradient_accumulation_steps"
                ),
                "transport": summary.get("transport"),
                "tokens_per_second_per_gpu": numeric(
                    throughput.get("tokens_per_second_per_gpu")
                ),
                "aggregate_tokens_per_second": aggregate,
                "expected_aggregate_tokens_per_second": expected,
                "scaling_efficiency": efficiency,
                "communication_seconds": communication_seconds,
                "collective_communication_seconds": (
                    dispatch_seconds + combine_seconds
                ),
                "routing_metadata_collective_seconds": routing_metadata_seconds,
                "routing_pack_unpack_seconds": routing_pack_unpack_seconds,
                "routing_dispatch_pack_unpack_seconds": dispatch_pack_seconds,
                "routing_dispatch_backward_pack_unpack_seconds": dispatch_backward_pack_seconds,
                "routing_combine_pack_unpack_seconds": combine_pack_seconds,
                "routing_combine_backward_pack_unpack_seconds": combine_backward_pack_seconds,
                "dispatch_alltoall_seconds": dispatch_seconds,
                "combine_alltoall_seconds": combine_seconds,
                "routing_metadata_seconds": routing_metadata_seconds,
                "routing_metadata_materialization_seconds": numeric(
                    summary.get("routing_metadata", {}).get(
                        "materialization_seconds"
                    )
                    if isinstance(summary.get("routing_metadata"), Mapping)
                    else None
                ),
                "routing_metadata_permutation_seconds": numeric(
                    summary.get("routing_metadata", {}).get("permutation_seconds")
                    if isinstance(summary.get("routing_metadata"), Mapping)
                    else None
                ),
                "routing_phases": dict(summary.get("routing_phases", {}))
                if isinstance(summary.get("routing_phases"), Mapping)
                else {},
                "communication_by_locality": dict(by_locality)
                if isinstance(by_locality, Mapping)
                else {},
                "phase_timings": dict(summary.get("phase_timings", {}))
                if isinstance(summary.get("phase_timings", {}), Mapping)
                else {},
                "routed_tokens": routed_tokens,
                "model_metrics": dict(summary.get("model_metrics", {}))
                if isinstance(summary.get("model_metrics", {}), Mapping)
                else {},
                "expert_compute": expert_compute,
                "peak_memory": summary.get("peak_memory", {}),
                "steady_state_memory": summary.get("steady_state_memory", {}),
                "device_health": metadata.get("device_health"),
                "measurement_completion": metadata.get("measurement_completion"),
            }
        )
    return {
        "baseline_ep_degree": baseline_ep_degree,
        "rows": rows,
        "scaling_efficiency_definition": (
            "aggregate throughput divided by baseline aggregate throughput scaled "
            "by world-size ratio"
        ),
    }


def compare_optimization_summaries(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any], *,
    varying_controls: Iterable[str] = (),
) -> dict[str, Any]:
    """Compare two same-topology optimization A/B summaries.

    Unlike EP scaling, this comparison permits only explicitly declared
    controls to vary. It is intended for router, packing, or kernel A/Bs and
    reports improvements without turning a single run into a promotion claim.
    """
    if not isinstance(baseline, Mapping) or not isinstance(candidate, Mapping):
        raise ValueError("optimization summaries must be mappings")
    baseline_metadata = baseline.get("metadata", {})
    candidate_metadata = candidate.get("metadata", {})
    if not isinstance(baseline_metadata, Mapping) or not isinstance(
        candidate_metadata, Mapping
    ):
        raise ValueError("optimization summaries must contain metadata mappings")
    required_metadata = (
        "model", "checkpoint", "world_size", "sequence_length", "batch_size",
        "optimizer", "topology", "optimization_profile", "routing_index_mode",
        "expert_execution_path", "device_health", "gate_status",
        "semantic_completion", "measurement_completion",
    )
    for name, metadata in (("baseline", baseline_metadata), ("candidate", candidate_metadata)):
        missing = [field for field in required_metadata if field not in metadata]
        if missing:
            raise ValueError(
                f"{name} optimization summary missing metadata: {', '.join(missing)}"
            )
    varying = set(varying_controls)
    for control in varying:
        if not isinstance(control, str) or not control.strip():
            raise ValueError("varying_controls must contain non-empty strings")
    environment_names = (
        "TORCHTUNE_MOE_TOPK_ROUTING",
        "TORCHTUNE_MOE_UNSTABLE_EXPERT_GROUPING",
        "TORCHTUNE_MOE_VECTOR_PACKING",
        "TORCHTUNE_EP_CPU_VECTOR_ROUTING_METADATA",
        "TORCHTUNE_EP_PACK_ROUTING_METADATA_TRANSFER",
    )
    control_names = (
        "model", "checkpoint", "world_size", "sequence_length", "batch_size",
        "optimizer", "topology", "optimization_profile", "routing_index_mode",
        "expert_execution_path", *environment_names,
    )
    immutable_controls = {
        "model",
        "checkpoint",
        "world_size",
        "sequence_length",
        "batch_size",
        "optimizer",
        "topology",
        "expert_execution_path",
    }
    unknown_controls = varying - set(control_names)
    if unknown_controls:
        raise ValueError(
            "varying_controls contains unknown controls: "
            + ", ".join(sorted(unknown_controls))
        )
    invalid_varying = varying & immutable_controls
    if invalid_varying:
        raise ValueError(
            "optimization A/B controls cannot vary: "
            + ", ".join(sorted(invalid_varying))
        )
    for name, metadata in (("baseline", baseline_metadata), ("candidate", candidate_metadata)):
        if not isinstance(metadata["topology"], Mapping):
            raise ValueError(f"{name} optimization summary topology must be a mapping")
        environment = metadata.get("environment_overrides")
        if not isinstance(environment, Mapping):
            raise ValueError(
                f"{name} optimization summary environment_overrides must be a mapping"
            )
        if metadata["device_health"] != "green":
            raise ValueError(f"{name} optimization summary device_health must be green")
        for field in ("gate_status", "semantic_completion", "measurement_completion"):
            if metadata[field] != "passed":
                raise ValueError(
                    f"{name} optimization summary {field} must be passed"
                )
    mismatches = []
    for name in control_names:
        if name in environment_names:
            left = baseline_metadata["environment_overrides"].get(name)
            right = candidate_metadata["environment_overrides"].get(name)
        else:
            left = baseline_metadata.get(name)
            right = candidate_metadata.get(name)
        if left != right and name not in varying:
            mismatches.append(name)
    if mismatches:
        raise ValueError(
            "optimization summaries use incompatible controls: "
            + ", ".join(sorted(mismatches))
        )
    differing_controls = []
    for name in control_names:
        if name in environment_names:
            left = baseline_metadata["environment_overrides"].get(name)
            right = candidate_metadata["environment_overrides"].get(name)
        else:
            left = baseline_metadata.get(name)
            right = candidate_metadata.get(name)
        if left != right:
            differing_controls.append(name)
    undeclared_differences = set(differing_controls) - varying
    if undeclared_differences:
        raise ValueError(
            "optimization summaries differ in undeclared controls: "
            + ", ".join(sorted(undeclared_differences))
        )
    if not differing_controls:
        raise ValueError("optimization A/B summaries must differ in a declared control")

    def metric(summary: Mapping[str, Any], *paths: str) -> Optional[float]:
        value: Any = summary
        for path in paths:
            if not isinstance(value, Mapping):
                return None
            value = value.get(path)
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    baseline_throughput = metric(baseline, "throughput", "tokens_per_second_per_gpu")
    candidate_throughput = metric(candidate, "throughput", "tokens_per_second_per_gpu")
    baseline_step = metric(baseline, "phase_timings", "mean_total_step_s")
    candidate_step = metric(candidate, "phase_timings", "mean_total_step_s")
    throughput_delta = (
        candidate_throughput - baseline_throughput
        if baseline_throughput is not None and candidate_throughput is not None
        else None
    )
    step_delta = (
        candidate_step - baseline_step
        if baseline_step is not None and candidate_step is not None
        else None
    )
    return {
        "result_class": "optimization_ab",
        "varying_controls": sorted(varying),
        "baseline": dict(baseline),
        "candidate": dict(candidate),
        "delta": {
            "tokens_per_second_per_gpu": throughput_delta,
            "tokens_per_second_per_gpu_fraction": (
                throughput_delta / baseline_throughput
                if throughput_delta is not None and baseline_throughput
                else None
            ),
            "mean_total_step_s": step_delta,
            "mean_total_step_s_fraction": (
                step_delta / baseline_step
                if step_delta is not None and baseline_step
                else None
            ),
            "peak_reserved_bytes": (
                metric(candidate, "peak_memory", "max_reserved_bytes")
                - metric(baseline, "peak_memory", "max_reserved_bytes")
                if metric(candidate, "peak_memory", "max_reserved_bytes") is not None
                and metric(baseline, "peak_memory", "max_reserved_bytes") is not None
                else None
            ),
        },
        "candidate_improves_throughput": (
            candidate_throughput > baseline_throughput
            if baseline_throughput is not None and candidate_throughput is not None
            else None
        ),
        "promotion_status": "pending_independent_hardware_repeat",
    }


def compare_capacity_value_results(
    moe: Mapping[str, Any],
    dense: Mapping[str, Any],
    *,
    require_canonical_metadata: bool = False,
) -> dict[str, Any]:
    """Compare larger-model MoE and dense controls without claiming parity.

    Both inputs are result summaries rather than rank artifacts. The function
    requires explicit active-compute and topology metadata, and preserves
    missing metrics as errors because a capacity/value table must not silently
    substitute a strict-parity metric.
    """
    if not isinstance(moe, Mapping) or not isinstance(dense, Mapping):
        raise ValueError("capacity/value results must be mappings")

    required = (
        "model",
        "total_parameters",
        "active_parameters_per_token",
        "sequence_length",
        "topology",
        "per_gpu_throughput",
        "aggregate_throughput",
        "mfu_percent",
        "active_flop_efficiency",
        "peak_memory",
        "communication_fraction",
        "expert_compute_fraction",
        "stability",
    )

    def validate_result(name: str, result: Mapping[str, Any]) -> dict[str, Any]:
        missing = [field for field in required if field not in result]
        if missing:
            raise ValueError(
                f"{name} capacity/value result missing: {', '.join(missing)}"
            )
        stability = result["stability"]
        if stability not in {"passed", "failed", "pending"} and stability is not True:
            raise ValueError(f"{name} stability must be passed, failed, or pending")
        topology = result["topology"]
        if not isinstance(topology, Mapping):
            raise ValueError(f"{name} topology must be a mapping")
        try:
            sequence_length = int(result["sequence_length"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} sequence_length must be a positive integer") from error
        if sequence_length < 1:
            raise ValueError(f"{name} sequence_length must be a positive integer")
        for field in (
            "total_parameters",
            "active_parameters_per_token",
            "per_gpu_throughput",
            "aggregate_throughput",
            "mfu_percent",
            "active_flop_efficiency",
            "communication_fraction",
            "expert_compute_fraction",
        ):
            try:
                value = float(result[field])
            except (TypeError, ValueError) as error:
                raise ValueError(f"{name} {field} must be finite and non-negative") from error
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} {field} must be finite and non-negative")
        for field in ("communication_fraction", "expert_compute_fraction"):
            if float(result[field]) > 1:
                raise ValueError(f"{name} {field} must be between 0 and 1")
        for field in ("nodes", "world_size"):
            if field not in topology:
                raise ValueError(f"{name} topology must record {field}")
            try:
                value = int(topology[field])
            except (TypeError, ValueError) as error:
                raise ValueError(f"{name} topology {field} must be positive") from error
            if value < 1:
                raise ValueError(f"{name} topology {field} must be positive")
        memory = result["peak_memory"]
        if not isinstance(memory, Mapping):
            raise ValueError(f"{name} peak_memory must be a mapping")
        memory_values = [
            memory.get("max_reserved_bytes"),
            memory.get("max_reserved_gib"),
        ]
        if not any(value is not None for value in memory_values):
            raise ValueError(
                f"{name} peak_memory must record max_reserved_bytes or max_reserved_gib"
            )
        for value in memory_values:
            if value is not None:
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"{name} peak_memory values must be finite and non-negative"
                    ) from error
                if not math.isfinite(numeric_value) or numeric_value < 0:
                    raise ValueError(
                        f"{name} peak_memory values must be finite and non-negative"
                    )
        if require_canonical_metadata:
            steady_state = result.get("steady_state_memory")
            if not isinstance(steady_state, Mapping):
                raise ValueError(
                    f"{name} steady_state_memory must be a mapping for canonical reports"
                )
            steady_values = [
                steady_state.get("max_reserved_bytes"),
                steady_state.get("max_reserved_gib"),
            ]
            if not any(value is not None for value in steady_values):
                raise ValueError(
                    f"{name} steady_state_memory must record reserved memory"
                )
            for value in steady_values:
                if value is not None:
                    try:
                        numeric_value = float(value)
                    except (TypeError, ValueError) as error:
                        raise ValueError(
                            f"{name} steady_state_memory values must be finite and non-negative"
                        ) from error
                    if not math.isfinite(numeric_value) or numeric_value < 0:
                        raise ValueError(
                            f"{name} steady_state_memory values must be finite and non-negative"
                        )
        return dict(result)

    moe_result = validate_result("MoE", moe)
    dense_result = validate_result("dense", dense)
    if require_canonical_metadata:
        for name, result in (("MoE", moe_result), ("dense", dense_result)):
            model = result.get("model")
            if not isinstance(model, str) or not model.strip() or model == "unknown":
                raise ValueError(
                    f"{name} capacity/value model must be non-placeholder"
                )
            if result["stability"] != "passed" and result["stability"] is not True:
                raise ValueError(
                    f"{name} capacity/value stability must be 'passed'"
                )
            for field in (
                "total_parameters",
                "active_parameters_per_token",
                "per_gpu_throughput",
                "aggregate_throughput",
                "mfu_percent",
                "active_flop_efficiency",
            ):
                if float(result[field]) <= 0:
                    raise ValueError(
                        f"{name} capacity/value {field} must be positive"
                    )
            for field in ("checkpoint", "source_revision", "uncommitted_change_state"):
                value = result.get(field)
                if not isinstance(value, str) or not value.strip() or value == "unknown":
                    raise ValueError(
                        f"{name} capacity/value {field} must be non-placeholder"
                    )
            for field in (
                "batch_size",
                "microbatch_size",
                "gradient_accumulation_steps",
            ):
                try:
                    value = int(result[field])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"{name} capacity/value {field} must be a positive integer"
                    ) from error
                if value < 1:
                    raise ValueError(
                        f"{name} capacity/value {field} must be a positive integer"
                    )
            if not isinstance(result.get("optimizer"), str) or not result["optimizer"].strip():
                raise ValueError(f"{name} capacity/value optimizer must be non-empty")
            if not isinstance(result.get("environment_overrides"), Mapping):
                raise ValueError(
                    f"{name} capacity/value environment_overrides must be a mapping"
                )
        for field in (
            "batch_size",
            "microbatch_size",
            "gradient_accumulation_steps",
            "optimizer",
            "environment_overrides",
        ):
            if moe_result[field] != dense_result[field]:
                raise ValueError(
                    f"capacity/value controls must use the same {field}"
                )
        for field in ("source_revision", "uncommitted_change_state"):
            if moe_result[field] != dense_result[field]:
                raise ValueError(
                    f"capacity/value controls must use the same {field}"
                )
    if moe_result["sequence_length"] != dense_result["sequence_length"]:
        raise ValueError("capacity/value controls must use the same sequence_length")
    for field in ("nodes", "world_size"):
        if moe_result["topology"][field] != dense_result["topology"][field]:
            raise ValueError(
                "capacity/value controls must use matching topology "
                f"for {field}"
            )

    def numeric_delta(field: str) -> dict[str, Optional[float]]:
        try:
            moe_value = float(moe_result[field])
            dense_value = float(dense_result[field])
        except (KeyError, TypeError, ValueError):
            return {"moe_minus_dense": None, "moe_over_dense": None}
        if not math.isfinite(moe_value) or not math.isfinite(dense_value):
            return {"moe_minus_dense": None, "moe_over_dense": None}
        return {
            "moe_minus_dense": moe_value - dense_value,
            "moe_over_dense": moe_value / dense_value if dense_value else None,
        }

    def peak_memory(result: Mapping[str, Any]) -> Optional[float]:
        memory = result.get("peak_memory")
        if not isinstance(memory, Mapping):
            return None
        value = memory.get("max_reserved_bytes")
        if value is not None:
            return float(value) / (1024**3)
        value = memory.get("max_reserved_gib")
        if value is not None:
            return float(value)
        return None

    def steady_state_memory(result: Mapping[str, Any]) -> Optional[float]:
        memory = result.get("steady_state_memory")
        if not isinstance(memory, Mapping):
            return None
        value = memory.get("max_reserved_bytes")
        if value is not None:
            return float(value) / (1024**3)
        value = memory.get("max_reserved_gib")
        if value is not None:
            return float(value)
        return None

    def pipeline(result: Mapping[str, Any]) -> Mapping[str, Any]:
        value = result.get("pipeline")
        if isinstance(value, Mapping):
            return value
        measurement = result.get("measurement")
        if isinstance(measurement, Mapping) and isinstance(
            measurement.get("pipeline"), Mapping
        ):
            return measurement["pipeline"]
        return {"timing_recorded": False}

    memory_delta = None
    moe_memory = peak_memory(moe_result)
    dense_memory = peak_memory(dense_result)
    if moe_memory is not None and dense_memory is not None:
        memory_delta = {
            "moe_minus_dense": moe_memory - dense_memory,
            "moe_over_dense": moe_memory / dense_memory if dense_memory else None,
        }
    steady_state_delta = None
    moe_steady_state = steady_state_memory(moe_result)
    dense_steady_state = steady_state_memory(dense_result)
    if moe_steady_state is not None and dense_steady_state is not None:
        steady_state_delta = {
            "moe_minus_dense": moe_steady_state - dense_steady_state,
            "moe_over_dense": (
                moe_steady_state / dense_steady_state if dense_steady_state else None
            ),
        }
    moe_mfu = moe_result.get("mfu_percent")
    moe_throughput = moe_result.get("aggregate_throughput")
    dense_throughput = dense_result.get("aggregate_throughput")
    decision_categories = {
        "highest_repeatable_moe_per_gpu_mfu": (
            "moe" if isinstance(moe_mfu, (int, float)) else "pending_measurement"
        ),
        "highest_aggregate_throughput": (
            "moe"
            if isinstance(moe_throughput, (int, float))
            and isinstance(dense_throughput, (int, float))
            and moe_throughput >= dense_throughput
            else "dense"
            if isinstance(moe_throughput, (int, float))
            and isinstance(dense_throughput, (int, float))
            else "pending_measurement"
        ),
        "best_ep_scaling_point": (
            "moe"
            if isinstance(moe_result.get("scaling_efficiency"), (int, float))
            else "pending_measurement"
        ),
        "closest_strict_dense_parity": "not_applicable_capacity_comparison",
        "best_capacity_value_result": (
            "capacity_value_candidate"
            if moe_result.get("stability") == "passed"
            and dense_result.get("stability") == "passed"
            else "pending_measurement"
        ),
    }
    gap_attribution = {}
    for name, field in (
        ("communication", "communication_fraction"),
        ("expert_compute", "expert_compute_fraction"),
    ):
        provenance = {}
        for control, result in (("MoE", moe_result), ("dense", dense_result)):
            result_provenance = result.get("fraction_provenance")
            source = (
                result_provenance.get(name)
                if isinstance(result_provenance, Mapping)
                else None
            )
            provenance[control] = source or "explicit_override"
        if all(source == "sealed_phase_timing" for source in provenance.values()):
            evidence = "sealed phase timing for both controls"
        elif all(source == "explicit_override" for source in provenance.values()):
            evidence = f"explicit {field} values for both controls"
        else:
            evidence = (
                f"mixed provenance: MoE {provenance['MoE']}, "
                f"dense {provenance['dense']}"
            )
        gap_attribution[name] = {
            "status": (
                "measured"
                if isinstance(moe_result.get(field), (int, float))
                and isinstance(dense_result.get(field), (int, float))
                else "pending"
            ),
            "evidence": evidence,
            "provenance": provenance,
        }
    gap_attribution.update(
        {
            "attention": {
                "status": "pending",
                "evidence": "no dedicated attention fraction in capacity result schema",
            },
            "optimizer": {
                "status": "pending",
                "evidence": "no dedicated optimizer fraction in capacity result schema",
            },
            "pipeline_bubble": {
                "status": (
                    "measured"
                    if pipeline(moe_result).get("timing_recorded")
                    and pipeline(dense_result).get("timing_recorded")
                    else "pending"
                ),
                "evidence": "explicit pipeline timing for both controls",
            },
        }
    )
    return {
        "result_class": "capacity_value",
        "comparison_label": "capacity_value_only",
        "parity_claim_allowed": False,
        "active_compute_matched": (
            moe_result["active_parameters_per_token"]
            == dense_result["active_parameters_per_token"]
        ),
        "rows": {"moe": moe_result, "dense": dense_result},
        "comparison_metrics": {
            "per_gpu_throughput": numeric_delta("per_gpu_throughput"),
            "aggregate_throughput": numeric_delta("aggregate_throughput"),
            "mfu_percent": numeric_delta("mfu_percent"),
            "active_flop_efficiency": numeric_delta("active_flop_efficiency"),
            "communication_fraction": numeric_delta("communication_fraction"),
            "expert_compute_fraction": numeric_delta("expert_compute_fraction"),
            "scaling_efficiency": numeric_delta("scaling_efficiency"),
            "peak_memory": memory_delta,
            "steady_state_memory": steady_state_delta,
        },
        "pipeline": {"moe": dict(pipeline(moe_result)), "dense": dict(pipeline(dense_result))},
        "capacity_advantage": {
            "moe_total_parameters": moe_result["total_parameters"],
            "dense_total_parameters": dense_result["total_parameters"],
            "moe_fits": moe_result.get("fits_allocation"),
            "dense_fits": dense_result.get("fits_allocation"),
            "moe_model_larger": (
                moe_result["total_parameters"] > dense_result["total_parameters"]
                if isinstance(moe_result["total_parameters"], (int, float))
                and isinstance(dense_result["total_parameters"], (int, float))
                else None
            ),
        },
        "decision_categories": decision_categories,
        "gap_attribution": gap_attribution,
    }


def evaluate_kernel_parity(
    moe: Mapping[str, Any],
    dense: Mapping[str, Any],
    *,
    threshold_percent: float = 95.0,
    require_canonical_metadata: bool = False,
) -> dict[str, Any]:
    """Evaluate strict Tier 1 throughput/MFU parity.

    The result is promotable only when both metrics clear the threshold and
    the MoE result has at least two independent repeats. Repeatability is
    intentionally separate from numerical threshold checks so a single
    promising run cannot become a parity claim.
    """
    if not isinstance(moe, Mapping) or not isinstance(dense, Mapping):
        raise ValueError("kernel-parity results must be mappings")
    if not math.isfinite(threshold_percent) or not 0 < threshold_percent <= 100:
        raise ValueError("threshold_percent must be finite and in (0, 100]")

    required = (
        "throughput_tokens_per_second_per_gpu",
        "mfu_percent",
        "sequence_length",
        "topology",
        "optimizer",
        "measurement_window",
        "status",
    )

    def validate(name: str, result: Mapping[str, Any]) -> dict[str, Any]:
        missing = [field for field in required if field not in result]
        if missing:
            raise ValueError(f"{name} kernel-parity result missing: {', '.join(missing)}")
        for field in ("throughput_tokens_per_second_per_gpu", "mfu_percent"):
            try:
                value = float(result[field])
            except (TypeError, ValueError) as error:
                raise ValueError(f"{name} {field} must be finite and positive") from error
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} {field} must be finite and positive")
        try:
            sequence_length = int(result["sequence_length"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} sequence_length must be positive") from error
        if sequence_length < 1:
            raise ValueError(f"{name} sequence_length must be positive")
        if not isinstance(result["topology"], Mapping):
            raise ValueError(f"{name} topology must be a mapping")
        if not isinstance(result["measurement_window"], Mapping):
            raise ValueError(f"{name} measurement_window must be a mapping")
        if not isinstance(result["optimizer"], str) or not result["optimizer"]:
            raise ValueError(f"{name} optimizer must be non-empty")
        if not isinstance(result["status"], str) or not result["status"]:
            raise ValueError(f"{name} status must be non-empty")
        return dict(result)

    moe_result = validate("MoE", moe)
    dense_result = validate("dense", dense)
    if require_canonical_metadata:
        expected_models = {"MoE": "Qwen3-30B-A3B", "dense": "Qwen3-4B"}
        for name, result in (("MoE", moe_result), ("dense", dense_result)):
            model = result.get("model")
            if model != expected_models[name]:
                raise ValueError(
                    f"{name} kernel-parity model must be {expected_models[name]!r}"
                )
            for field in ("source_revision", "checkpoint", "uncommitted_change_state"):
                value = result.get(field)
                if not isinstance(value, str) or not value.strip() or value == "unknown":
                    raise ValueError(
                        f"{name} kernel-parity {field} must be non-placeholder"
                    )
            for field, expected in (
                ("device_health", "green"),
                ("semantic_completion", "passed"),
                ("measurement_completion", "passed"),
            ):
                if result.get(field) != expected:
                    raise ValueError(
                        f"{name} kernel-parity {field} must be {expected!r}"
                    )
            topology = result["topology"]
            for field in ("nodes", "world_size", "participating_tiles"):
                try:
                    value = int(topology[field])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"{name} kernel-parity topology must record {field}"
                    ) from error
                if value < 1:
                    raise ValueError(
                        f"{name} kernel-parity topology {field} must be positive"
                    )
            try:
                active_parameters = float(result["active_parameters_per_token"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"{name} active_parameters_per_token must be positive"
                ) from error
            if not math.isfinite(active_parameters) or active_parameters <= 0:
                raise ValueError(
                    f"{name} active_parameters_per_token must be positive"
                )
            for field in ("batch_size", "microbatch_size", "gradient_accumulation_steps"):
                try:
                    value = int(result[field])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"{name} kernel-parity must record {field}"
                    ) from error
                if value < 1:
                    raise ValueError(
                        f"{name} kernel-parity {field} must be positive"
                    )
    if moe_result["sequence_length"] != dense_result["sequence_length"]:
        raise ValueError("kernel-parity controls must use the same sequence_length")
    if moe_result["optimizer"] != dense_result["optimizer"]:
        raise ValueError("kernel-parity controls must use the same optimizer")
    if moe_result["measurement_window"] != dense_result["measurement_window"]:
        raise ValueError("kernel-parity controls must use the same measurement_window")
    if require_canonical_metadata:
        for field in ("batch_size", "microbatch_size", "gradient_accumulation_steps"):
            if moe_result[field] != dense_result[field]:
                raise ValueError(
                    f"kernel-parity controls must use the same {field}"
                )
        for field in ("source_revision", "uncommitted_change_state"):
            if moe_result[field] != dense_result[field]:
                raise ValueError(
                    f"kernel-parity controls must use the same {field}"
                )

    fraction = threshold_percent / 100.0
    throughput_threshold = (
        float(dense_result["throughput_tokens_per_second_per_gpu"]) * fraction
    )
    mfu_threshold = float(dense_result["mfu_percent"]) * fraction
    throughput_ratio = (
        float(moe_result["throughput_tokens_per_second_per_gpu"])
        / float(dense_result["throughput_tokens_per_second_per_gpu"])
    )
    mfu_ratio = float(moe_result["mfu_percent"]) / float(dense_result["mfu_percent"])
    try:
        repeat_count = int(moe_result.get("independent_repeats", 1))
    except (TypeError, ValueError) as error:
        raise ValueError("MoE independent_repeats must be an integer") from error
    if repeat_count < 1:
        raise ValueError("MoE independent_repeats must be positive")
    repeatable = repeat_count >= 2
    meets_threshold = (
        float(moe_result["throughput_tokens_per_second_per_gpu"]) >= throughput_threshold
        and float(moe_result["mfu_percent"]) >= mfu_threshold
    )
    return {
        "result_class": "kernel_parity",
        "threshold_percent": threshold_percent,
        "thresholds": {
            "throughput_tokens_per_second_per_gpu": throughput_threshold,
            "mfu_percent": mfu_threshold,
        },
        "ratios": {
            "throughput": throughput_ratio,
            "mfu": mfu_ratio,
        },
        "meets_threshold": meets_threshold,
        "independent_repeats": repeat_count,
        "repeatable": repeatable,
        "promoted": meets_threshold and repeatable,
        "rows": {"moe": moe_result, "dense": dense_result},
    }


def aggregate_measurement_files(
    paths: Iterable[os.PathLike[str] | str],
    *,
    require_complete_rank_set: bool = False,
    require_router_semantics: bool = False,
) -> dict[str, Any]:
    """Aggregate rank-local JSON measurement artifacts deterministically.

    ``require_complete_rank_set`` additionally requires integer ranks exactly
    covering ``0..world_size-1``. It is intended for canonical benchmark
    reports; generic callers may aggregate a deliberately partial diagnostic
    set by leaving it disabled.
    ``require_router_semantics`` requires the versioned router marker emitted
    by new canonical artifacts while allowing historical generic fixtures.
    """
    ordered_paths = sorted(os.fspath(path) for path in paths)
    if not ordered_paths:
        return {
            "rank_count": 0,
            "rank_files": [],
            "metadata_by_rank": {},
            "common_metadata": {},
            "records": {},
            "records_by_pipeline_stage": {},
            "collective_locality": {},
            "step_timings": [],
        }

    rank_records: dict[str, dict[str, Any]] = {}
    rank_step_timings: dict[str, list[Mapping[str, Any]]] = {}
    metadata_by_rank: dict[str, Mapping[str, Any]] = {}
    module_names_by_stage: dict[str, set[str]] = {}
    reference_measurement_window: Optional[dict[str, int]] = None
    saw_measurement_window = False
    invariant_metadata = (
        "source_revision",
        "uncommitted_change_state",
        "world_size",
        "ep_degree",
        "global_step",
        "sequence_length",
        "model",
        "checkpoint",
        "batch_size",
        "gradient_accumulation_steps",
        "optimizer",
        "topology",
        "environment_overrides",
        "optimization_profile",
        "routing_index_mode",
        "expert_execution_path",
        "device_health",
        "gate_status",
        "semantic_completion",
        "measurement_completion",
    )
    optional_invariant_metadata = ("microbatch_size", "router_semantics")
    if require_router_semantics:
        invariant_metadata = (*invariant_metadata, "pipeline_stage")
    reference_metadata: Optional[dict[str, Any]] = None
    for path in ordered_paths:
        with open(path, encoding="utf-8") as input_file:
            payload = json.load(input_file)
        if not isinstance(payload, Mapping) or not isinstance(
            payload.get("records"), Mapping
        ):
            raise ValueError(f"measurement artifact has no records mapping: {path}")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError(f"measurement artifact metadata is not a mapping: {path}")
        if require_router_semantics:
            router_semantics = metadata.get("router_semantics")
            if not isinstance(router_semantics, str) or not router_semantics.strip():
                raise ValueError(
                    "canonical measurement artifact requires non-empty "
                    f"router_semantics: {path}"
                )
        rank_value = metadata.get("rank")
        if rank_value is None:
            raise ValueError(f"measurement artifact metadata has no rank: {path}")
        missing_metadata = [
            name for name in invariant_metadata if name not in metadata
        ]
        if missing_metadata:
            raise ValueError(
                f"measurement artifact metadata missing {', '.join(missing_metadata)}: {path}"
            )
        measurement_window = metadata.get("measurement_window")
        normalized_window: Optional[dict[str, int]] = None
        if measurement_window is not None:
            saw_measurement_window = True
            if not isinstance(measurement_window, Mapping):
                raise ValueError(
                    f"measurement artifact measurement_window is not a mapping: {path}"
                )
            try:
                normalized_window = {
                    "warmup_steps": int(measurement_window["warmup_steps"]),
                    "measurement_steps": int(measurement_window["measurement_steps"]),
                    "steady_state_steps": int(measurement_window["steady_state_steps"]),
                }
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    "measurement artifact measurement_window must contain integer "
                    f"warmup_steps, measurement_steps, and steady_state_steps: {path}"
                ) from error
            if (
                normalized_window["warmup_steps"] < 0
                or normalized_window["measurement_steps"] < 1
                or normalized_window["steady_state_steps"] < 0
            ):
                raise ValueError(
                    f"measurement artifact measurement_window has invalid values: {path}"
                )
            if reference_measurement_window is None:
                reference_measurement_window = normalized_window
            elif normalized_window != reference_measurement_window:
                raise ValueError(
                    "measurement window metadata differs across ranks: "
                    f"{path}"
                )
        elif saw_measurement_window:
            raise ValueError(
                "measurement artifact measurement_window is missing while other "
                f"ranks provide it: {path}"
            )
        for name in ("optimization_profile", "routing_index_mode"):
            value = metadata[name]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"measurement artifact metadata {name} must be non-empty: {path}"
                )
        execution_path = metadata.get("expert_execution_path")
        if execution_path not in {
            "grouped_mm",
            "padded_bmm",
            "sequential",
        }:
            raise ValueError(
                "measurement artifact expert_execution_path is missing or invalid: "
                f"{path}"
            )
        environment_overrides = metadata["environment_overrides"]
        if not isinstance(environment_overrides, Mapping):
            raise ValueError(
                f"measurement artifact environment_overrides is not a mapping: {path}"
            )
        grouped = str(
            environment_overrides.get("TORCHTUNE_MOE_GROUPED_EXPERTS", "0")
        ) == "1"
        sequential = str(
            environment_overrides.get("TORCHTUNE_MOE_SEQUENTIAL_EXPERTS", "0")
        ) == "1"
        inferred_execution_path = "grouped_mm" if grouped else (
            "sequential" if sequential else "padded_bmm"
        )
        if execution_path != inferred_execution_path:
            raise ValueError(
                "measurement artifact expert_execution_path disagrees with "
                f"environment_overrides: {path}"
            )
        transport_override = environment_overrides.get("TORCHTUNE_EP_ALL2ALL")
        if transport_override is not None and str(transport_override) not in {"0", "1"}:
            raise ValueError(
                f"measurement artifact TORCHTUNE_EP_ALL2ALL must be 0 or 1: {path}"
            )
        _validate_binary_environment_overrides(environment_overrides, path)
        locality_override = environment_overrides.get(
            "TORCHTUNE_MOE_COLLECTIVE_LOCALITY"
        )
        if locality_override is not None:
            locality_override = str(locality_override)
            if locality_override not in _COLLECTIVE_LOCALITIES:
                raise ValueError(
                    "measurement artifact TORCHTUNE_MOE_COLLECTIVE_LOCALITY "
                    f"is invalid: {path}"
                )
            for record in payload["records"].values():
                if not isinstance(record, Mapping):
                    raise ValueError(
                        f"measurement module record is not a mapping: {path}"
                    )
                collective_events = record.get("collectives", [])
                if not isinstance(collective_events, list):
                    raise ValueError(
                        f"measurement collective records are not a list: {path}"
                    )
                for event in collective_events:
                    if not isinstance(event, Mapping):
                        raise ValueError(
                            f"measurement collective event is not a mapping: {path}"
                        )
                    event_locality = str(event.get("locality", "unknown"))
                    if event_locality != locality_override:
                        raise ValueError(
                            "measurement collective locality disagrees with "
                            "TORCHTUNE_MOE_COLLECTIVE_LOCALITY: "
                            f"{path}"
                        )
        optimizer_override = environment_overrides.get(
            "TORCHTUNE_MOE_OPTIMIZER_COMPONENT"
        )
        if optimizer_override is not None and str(optimizer_override) != str(
            metadata["optimizer"]
        ):
            raise ValueError(
                "measurement optimizer metadata disagrees with "
                f"TORCHTUNE_MOE_OPTIMIZER_COMPONENT: {path}"
            )
        gate_values = {
            "device_health": "green",
            "gate_status": "passed",
            "semantic_completion": "passed",
        }
        invalid_gates = [
            name
            for name, expected in gate_values.items()
            if metadata[name] != expected
        ]
        if invalid_gates:
            raise ValueError(
                f"measurement artifact gates not passed ({', '.join(invalid_gates)}): {path}"
            )
        if metadata["measurement_completion"] != "passed":
            raise ValueError(
                f"measurement artifact has not been sealed ({path})"
            )
        current_metadata = {
            name: metadata[name] for name in invariant_metadata
        }
        if normalized_window is not None:
            current_metadata["measurement_window"] = normalized_window
        for name in optional_invariant_metadata:
            if name in metadata:
                current_metadata[name] = metadata[name]
        if reference_metadata is None:
            reference_metadata = current_metadata
        elif current_metadata != reference_metadata:
            raise ValueError(f"measurement invariant metadata differs across ranks: {path}")
        rank = str(rank_value)
        if rank in rank_records:
            raise ValueError(f"duplicate measurement rank {rank}: {path}")
        records = dict(payload["records"])
        current_step_timings = payload.get("step_timings", [])
        if not isinstance(current_step_timings, list):
            raise ValueError(f"measurement step_timings is not a list: {path}")
        names = set(records)
        stage = str(metadata.get("pipeline_stage", "0"))
        expected_names = module_names_by_stage.setdefault(stage, names)
        if names != expected_names:
            raise ValueError(f"measurement modules differ across ranks: {path}")
        rank_records[rank] = records
        rank_step_timings[rank] = current_step_timings
        metadata_by_rank[rank] = dict(metadata)

    if saw_measurement_window and len(metadata_by_rank) > 1:
        missing_window_ranks = [
            rank
            for rank, metadata in metadata_by_rank.items()
            if "measurement_window" not in metadata
        ]
        if missing_window_ranks:
            raise ValueError(
                "measurement window metadata missing for ranks: "
                + ", ".join(sorted(missing_window_ranks))
            )

    if require_complete_rank_set:
        world_sizes = {
            int(metadata["world_size"])
            for metadata in metadata_by_rank.values()
        }
        if len(world_sizes) != 1 or next(iter(world_sizes), 0) < 1:
            raise ValueError(
                "complete rank-set validation requires one positive world_size"
            )
        world_size = next(iter(world_sizes))
        try:
            observed_ranks = {int(rank) for rank in rank_records}
        except ValueError as error:
            raise ValueError(
                "complete rank-set validation requires integer rank metadata"
            ) from error
        expected_ranks = set(range(world_size))
        if observed_ranks != expected_ranks:
            missing_ranks = sorted(expected_ranks - observed_ranks)
            unexpected_ranks = sorted(observed_ranks - expected_ranks)
            raise ValueError(
                "measurement artifact rank set is incomplete: "
                f"missing={missing_ranks}, unexpected={unexpected_ranks}"
            )

    def rank_sort_key(rank: str) -> tuple[int, int | str]:
        try:
            return (0, int(rank))
        except ValueError:
            return (1, rank)

    ordered_ranks = sorted(rank_records, key=rank_sort_key)
    ranks_by_stage: dict[str, list[str]] = {}
    for rank in ordered_ranks:
        stage = str(metadata_by_rank[rank].get("pipeline_stage", "0"))
        ranks_by_stage.setdefault(stage, []).append(rank)

    records_by_stage: dict[str, dict[str, dict[str, Any]]] = {}
    for stage, stage_ranks in sorted(ranks_by_stage.items()):
        records_by_stage[stage] = {
            name: aggregate_rank_records(
                [rank_records[rank][name] for rank in stage_ranks]
            )
            for name in sorted(module_names_by_stage[stage])
        }
    aggregate_records = (
        records_by_stage[next(iter(records_by_stage))]
        if len(records_by_stage) == 1
        else {}
    )
    aggregate_locality: dict[str, dict[str, float]] = {}
    for stage_records in records_by_stage.values():
        for record in stage_records.values():
            for locality, totals in record["collective_locality"].items():
                aggregate = aggregate_locality.setdefault(
                    locality, {"duration_s": 0.0, "count": 0}
                )
                aggregate["duration_s"] += totals["duration_s"]
                aggregate["count"] += totals["count"]
    return {
        "rank_count": len(rank_records),
        "rank_files": ordered_paths,
        "metadata_by_rank": {
            rank: metadata_by_rank[rank] for rank in ordered_ranks
        },
        "common_metadata": reference_metadata or {},
        "records": aggregate_records,
        "records_by_pipeline_stage": records_by_stage,
        "collective_locality": aggregate_locality,
        "step_timings": [
            {"rank": rank, "steps": rank_step_timings[rank]}
            for rank in ordered_ranks
            if rank_step_timings[rank]
        ],
    }

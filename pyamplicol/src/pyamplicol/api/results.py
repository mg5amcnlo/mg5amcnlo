# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Literal, TypeAlias, cast

from pyamplicol.config import (
    BenchmarkConfig,
    ConfigClamp,
    GenerationConfig,
    RunConfig,
)

from .requests import ProcessRequest, ProcessSet


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(entry) for key, entry in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(entry) for entry in value)
    return value


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    frozen = _freeze(value)
    assert isinstance(frozen, Mapping)
    return frozen


def _exact_decimal_sum(values: tuple[Decimal, ...]) -> Decimal:
    if not values:
        return Decimal(0)
    if not all(value.is_finite() for value in values):
        return sum(values, start=Decimal(0))

    def finite_exponent(value: Decimal) -> int:
        exponent = value.as_tuple().exponent
        if not isinstance(exponent, int):
            raise ValueError("finite Decimal has a non-integral exponent")
        return exponent

    common_exponent = min(finite_exponent(value) for value in values)
    total = 0
    for value in values:
        sign, digits, _ = value.as_tuple()
        exponent = finite_exponent(value)
        coefficient = int("".join(str(digit) for digit in digits) or "0")
        if sign:
            coefficient = -coefficient
        total += coefficient * 10 ** (exponent - common_exponent)
    if total == 0:
        return Decimal(0)
    digits = tuple(int(character) for character in str(abs(total)))
    return Decimal((int(total < 0), digits, common_exponent))


@dataclass(frozen=True, slots=True)
class ExternalParticle:
    index: int
    label: int
    name: str
    pdg_id: int
    state: Literal["incoming", "outgoing"]
    momentum_slot: int

    def __post_init__(self) -> None:
        for name in ("index", "label", "momentum_slot"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"external particle {name} must be non-negative")
        if not self.name:
            raise ValueError("external particle name must not be empty")
        if isinstance(self.pdg_id, bool) or not isinstance(self.pdg_id, int):
            raise ValueError("external particle pdg_id must be an integer")
        if self.state not in ("incoming", "outgoing"):
            raise ValueError("external particle state must be 'incoming' or 'outgoing'")


@dataclass(frozen=True, slots=True)
class HelicityConfiguration:
    id: str
    index: int
    values: tuple[int, ...]
    computed: bool
    structural_zero: bool
    representative_id: str
    coefficient: float

    def __post_init__(self) -> None:
        if not self.id or not self.representative_id:
            raise ValueError("helicity IDs must not be empty")
        if isinstance(self.index, bool) or self.index < 0:
            raise ValueError("helicity index must be non-negative")
        values = tuple(self.values)
        if not all(
            isinstance(value, int) and not isinstance(value, bool) for value in values
        ):
            raise ValueError("helicity values must be integers")
        if not math.isfinite(self.coefficient):
            raise ValueError("helicity coefficient must be finite")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class ColorFlow:
    id: str
    index: int
    word: tuple[int, ...]
    computed: bool
    representative_id: str
    coefficient: float

    def __post_init__(self) -> None:
        if not self.id or not self.representative_id:
            raise ValueError("color-flow IDs must not be empty")
        if isinstance(self.index, bool) or self.index < 0:
            raise ValueError("color-flow index must be non-negative")
        word = tuple(self.word)
        if not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in word
        ):
            raise ValueError("color-flow words must contain non-negative integers")
        if not math.isfinite(self.coefficient):
            raise ValueError("color-flow coefficient must be finite")
        object.__setattr__(self, "word", word)


@dataclass(frozen=True, slots=True)
class ContractedColorComponent:
    id: str
    index: int
    description: str

    def __post_init__(self) -> None:
        if not self.id or not self.description:
            raise ValueError("contracted-color metadata must not be empty")
        if isinstance(self.index, bool) or self.index < 0:
            raise ValueError("contracted-color index must be non-negative")


@dataclass(frozen=True, slots=True)
class ModelParameter:
    name: str
    kind: str
    default_real: float
    default_imaginary: float
    mutable: bool

    def __post_init__(self) -> None:
        if not self.name or not self.kind:
            raise ValueError("model parameter name and kind must not be empty")
        if not math.isfinite(self.default_real) or not math.isfinite(
            self.default_imaginary
        ):
            raise ValueError("model parameter defaults must be finite")


ColorComponent: TypeAlias = ColorFlow | ContractedColorComponent


@dataclass(frozen=True, slots=True)
class ReductionGroup:
    id: str
    representative_helicity_id: str
    representative_color_id: str
    physical_helicity_ids: tuple[str, ...]
    physical_color_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.id or not self.representative_helicity_id:
            raise ValueError("reduction group and representative IDs must not be empty")
        if not self.representative_color_id:
            raise ValueError("reduction color representative ID must not be empty")
        helicities = tuple(self.physical_helicity_ids)
        colors = tuple(self.physical_color_ids)
        if not helicities or not colors:
            raise ValueError("reduction groups require physical helicity and color IDs")
        if len(set(helicities)) != len(helicities) or len(set(colors)) != len(colors):
            raise ValueError("reduction group member IDs must be unique")
        if self.representative_helicity_id not in helicities:
            raise ValueError("helicity representative must be a physical group member")
        if self.representative_color_id not in colors:
            raise ValueError("color representative must be a physical group member")
        object.__setattr__(self, "physical_helicity_ids", helicities)
        object.__setattr__(self, "physical_color_ids", colors)


@dataclass(frozen=True, slots=True)
class PhysicsReduction:
    kind: Literal["lc-diagonal", "contracted-color"]
    groups: tuple[ReductionGroup, ...]

    def __post_init__(self) -> None:
        if self.kind not in ("lc-diagonal", "contracted-color"):
            raise ValueError("unsupported physics reduction kind")
        groups = tuple(self.groups)
        if not all(isinstance(group, ReductionGroup) for group in groups):
            raise TypeError("physics reduction contains invalid groups")
        if len({group.id for group in groups}) != len(groups):
            raise ValueError("physics reduction group IDs must be unique")
        object.__setattr__(self, "groups", groups)


@dataclass(frozen=True, slots=True)
class GenerationPlan:
    """Dry-run result with concrete requests, coverage, and effective settings."""

    concrete_processes: tuple[ProcessRequest, ...]
    estimated_coverage: Mapping[str, object]
    requested_settings: RunConfig | GenerationConfig
    effective_settings: RunConfig | GenerationConfig
    adjustments: tuple[ConfigClamp, ...] = ()
    unsupported_features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        concrete_processes = tuple(self.concrete_processes)
        if not all(
            isinstance(process, ProcessRequest) for process in concrete_processes
        ):
            raise TypeError("concrete_processes must contain ProcessRequest objects")
        for name in ("requested_settings", "effective_settings"):
            if not isinstance(getattr(self, name), (RunConfig, GenerationConfig)):
                raise TypeError(f"{name} must be RunConfig or GenerationConfig")
        adjustments = tuple(self.adjustments)
        if not all(isinstance(adjustment, ConfigClamp) for adjustment in adjustments):
            raise TypeError("adjustments must contain ConfigClamp objects")
        unsupported_features = tuple(self.unsupported_features)
        if not all(
            isinstance(feature, str) and feature for feature in unsupported_features
        ):
            raise ValueError("unsupported_features must contain non-empty strings")
        object.__setattr__(self, "concrete_processes", concrete_processes)
        object.__setattr__(
            self, "estimated_coverage", _freeze_mapping(self.estimated_coverage)
        )
        object.__setattr__(self, "adjustments", adjustments)
        object.__setattr__(self, "unsupported_features", unsupported_features)


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Location and process inventory of a successfully written artifact."""

    output: Path
    processes: ProcessSet
    mode: Literal["error", "append", "replace"]
    schema_version: int = 3
    files: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        try:
            output = Path(os.fspath(self.output)).expanduser().resolve(strict=False)
            files = tuple(
                Path(os.fspath(path)).expanduser().resolve(strict=False)
                for path in self.files
            )
        except TypeError as exc:
            raise ValueError("generation result paths must be path-like") from exc
        object.__setattr__(self, "output", output)
        if self.mode not in ("error", "append", "replace"):
            raise ValueError(f"invalid generation mode {self.mode!r}")
        if self.schema_version != 3:
            raise ValueError("generated artifacts must use schema version 3")
        object.__setattr__(self, "files", files)


@dataclass(frozen=True, slots=True)
class WarmUpResult:
    """Metrics from warming one on-the-fly selector family at one f64 point."""

    elapsed_seconds: float
    query_count: int
    warmed_query_count: int
    current_rss_bytes: int | None
    peak_rss_bytes: int | None
    already_warm: bool

    def __post_init__(self) -> None:
        if (
            isinstance(self.elapsed_seconds, bool)
            or not isinstance(self.elapsed_seconds, (int, float))
            or not math.isfinite(float(self.elapsed_seconds))
            or self.elapsed_seconds < 0.0
        ):
            raise ValueError("warm-up elapsed_seconds must be finite and non-negative")
        object.__setattr__(self, "elapsed_seconds", float(self.elapsed_seconds))
        for name in ("query_count", "warmed_query_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"warm-up {name} must be a non-negative integer")
        if self.warmed_query_count > self.query_count:
            raise ValueError("warm-up warmed_query_count may not exceed query_count")
        for name in ("current_rss_bytes", "peak_rss_bytes"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(
                    f"warm-up {name} must be a non-negative integer or null"
                )
        if (
            self.current_rss_bytes is not None
            and self.peak_rss_bytes is not None
            and self.current_rss_bytes > self.peak_rss_bytes
        ):
            raise ValueError("warm-up current RSS may not exceed peak RSS")
        if not isinstance(self.already_warm, bool):
            raise TypeError("warm-up already_warm must be a boolean")
        if self.already_warm and self.warmed_query_count != 0:
            raise ValueError("an already-warm selector family cannot warm new queries")


@dataclass(frozen=True, slots=True)
class ProcessPhysics:
    """Physical axes, reduction metadata, and selectors for one process.

    LC processes expose physical color flows; NLC/full processes expose one
    contracted color component. ``selector_capabilities`` states which axes may
    be restricted without changing the generated artifact.
    """

    process_id: str
    process: str
    color_accuracy: Literal["lc", "nlc", "full"]
    helicity_coverage: str
    color_coverage: str
    color_kind: str
    structural_zero_helicity_count: int
    external_particles: tuple[ExternalParticle, ...]
    helicities: tuple[HelicityConfiguration, ...]
    color_flows: tuple[ColorFlow, ...]
    contracted_color_components: tuple[ContractedColorComponent, ...]
    reduction: PhysicsReduction
    model_parameters: tuple[ModelParameter, ...]
    selector_capabilities: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.process_id or not self.process:
            raise ValueError("process ID and expression must not be empty")
        if self.color_accuracy not in ("lc", "nlc", "full"):
            raise ValueError("color accuracy must be 'lc', 'nlc', or 'full'")
        for name in ("helicity_coverage", "color_coverage", "color_kind"):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        if (
            isinstance(self.structural_zero_helicity_count, bool)
            or self.structural_zero_helicity_count < 0
        ):
            raise ValueError("structural-zero helicity count must be non-negative")
        particles = tuple(self.external_particles)
        helicities = tuple(self.helicities)
        color_flows = tuple(self.color_flows)
        contracted = tuple(self.contracted_color_components)
        reduction = self.reduction
        parameters = tuple(self.model_parameters)
        selector_capabilities = tuple(self.selector_capabilities)
        expected = (
            ("external_particles", particles, ExternalParticle),
            ("helicities", helicities, HelicityConfiguration),
            ("color_flows", color_flows, ColorFlow),
            ("contracted_color_components", contracted, ContractedColorComponent),
            ("model_parameters", parameters, ModelParameter),
        )
        for name, values, expected_type in expected:
            if not all(isinstance(value, expected_type) for value in values):
                raise TypeError(f"{name} contains invalid metadata")
        if not helicities:
            raise ValueError("physics metadata requires physical helicities")
        particle_indices = tuple(particle.index for particle in particles)
        if len(set(particle_indices)) != len(particles):
            raise ValueError("external particle indices must be unique")
        for name, identifiers in (
            ("helicities", tuple(value.id for value in helicities)),
            ("color_flows", tuple(value.id for value in color_flows)),
            (
                "contracted_color_components",
                tuple(value.id for value in contracted),
            ),
        ):
            if len(set(identifiers)) != len(identifiers):
                raise ValueError(f"{name} IDs must be unique")
        parameter_names = tuple(parameter.name for parameter in parameters)
        if len(set(parameter_names)) != len(parameter_names):
            raise ValueError("model parameter names must be unique")
        if self.color_accuracy == "lc" and not color_flows:
            raise ValueError("LC physics metadata requires physical color flows")
        if self.color_accuracy != "lc" and len(contracted) != 1:
            raise ValueError("NLC/full physics metadata requires one color component")
        if not isinstance(reduction, PhysicsReduction):
            raise TypeError("reduction must be PhysicsReduction metadata")
        expected_reduction = (
            "lc-diagonal" if self.color_accuracy == "lc" else "contracted-color"
        )
        if reduction.kind != expected_reduction:
            raise ValueError("reduction kind does not match color accuracy")
        physical_helicity_ids = {value.id for value in helicities}
        physical_color_ids = {value.id for value in color_flows} | {
            value.id for value in contracted
        }
        for group in reduction.groups:
            if not set(group.physical_helicity_ids) <= physical_helicity_ids:
                raise ValueError("reduction references unknown physical helicities")
            if not set(group.physical_color_ids) <= physical_color_ids:
                raise ValueError("reduction references unknown physical colors")
        if not all(isinstance(value, str) and value for value in selector_capabilities):
            raise ValueError("selector_capabilities must contain non-empty strings")
        if len(set(selector_capabilities)) != len(selector_capabilities):
            raise ValueError("selector_capabilities must be unique")
        object.__setattr__(self, "external_particles", particles)
        object.__setattr__(self, "helicities", helicities)
        object.__setattr__(self, "color_flows", color_flows)
        object.__setattr__(self, "contracted_color_components", contracted)
        object.__setattr__(self, "model_parameters", parameters)
        object.__setattr__(self, "selector_capabilities", selector_capabilities)

    @property
    def helicity_ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self.helicities)

    @property
    def color_flow_ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self.color_flows)

    @property
    def color_ids(self) -> tuple[str, ...]:
        components: tuple[ColorComponent, ...] = (
            self.color_flows
            if self.color_accuracy == "lc"
            else self.contracted_color_components
        )
        return tuple(item.id for item in components)


@dataclass(frozen=True, slots=True)
class ResolvedEvaluation:
    """Physical values shaped ``(point, helicity, color)``.

    :meth:`total` explicitly sums the helicity and color axes for each point and
    reproduces the compatibility summed evaluation.
    """

    values: tuple[tuple[tuple[complex | Decimal, ...], ...], ...]
    helicity_ids: tuple[str, ...]
    color_ids: tuple[str, ...]
    color_accuracy: Literal["lc", "nlc", "full"] = "lc"

    def __post_init__(self) -> None:
        values = tuple(
            tuple(
                tuple(
                    entry if isinstance(entry, Decimal) else complex(entry)
                    for entry in colors
                )
                for colors in helicities
            )
            for helicities in self.values
        )
        helicity_ids = tuple(self.helicity_ids)
        color_ids = tuple(self.color_ids)
        expected_colors = len(color_ids)
        if self.color_accuracy != "lc" and expected_colors != 1:
            raise ValueError("NLC/full resolved output has one contracted color axis")
        for point in values:
            if len(point) != len(helicity_ids):
                raise ValueError(
                    "resolved values do not match the physical helicity dimension"
                )
            if any(len(colors) != expected_colors for colors in point):
                raise ValueError(
                    "resolved values do not match the contracted color dimension"
                )
        if self.color_accuracy not in ("lc", "nlc", "full"):
            raise ValueError("color_accuracy must be 'lc', 'nlc', or 'full'")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "helicity_ids", helicity_ids)
        object.__setattr__(self, "color_ids", color_ids)

    @property
    def color_flow_ids(self) -> tuple[str, ...]:
        return self.color_ids if self.color_accuracy == "lc" else ()

    @property
    def accuracy(self) -> Literal["lc", "nlc", "full"]:
        """Compatibility alias for :attr:`color_accuracy`."""

        return self.color_accuracy

    @property
    def shape(self) -> tuple[int, int, int]:
        return (len(self.values), len(self.helicity_ids), len(self.color_ids))

    def total(self) -> tuple[complex | Decimal, ...]:
        """Sum all non-point axes while preserving decimal precision."""

        totals: list[complex | Decimal] = []
        for point in self.values:
            entries = tuple(entry for helicity in point for entry in helicity)
            if entries and isinstance(entries[0], Decimal):
                if not all(isinstance(entry, Decimal) for entry in entries):
                    raise TypeError("resolved values must not mix Decimal and complex")
                totals.append(
                    _exact_decimal_sum(tuple(cast(Decimal, entry) for entry in entries))
                )
            else:
                totals.append(sum((complex(entry) for entry in entries), start=0j))
        return tuple(totals)


@dataclass(frozen=True, slots=True)
class BenchmarkStatistics:
    """Distribution summary for repeated benchmark measurements."""

    standard_deviation: float
    standard_error: float
    relative_standard_error: float

    def __post_init__(self) -> None:
        for name in (
            "standard_deviation",
            "standard_error",
            "relative_standard_error",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"benchmark {name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class BenchmarkComponentTiming:
    """Mean per-point time and uncertainty for one runtime-profile component."""

    mean_seconds_per_point: float
    uncertainty: BenchmarkStatistics
    sample_count: int

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.mean_seconds_per_point)
            or self.mean_seconds_per_point < 0
        ):
            raise ValueError(
                "benchmark component timing must be finite and non-negative"
            )
        if not isinstance(self.uncertainty, BenchmarkStatistics):
            raise TypeError(
                "benchmark component uncertainty must be BenchmarkStatistics"
            )
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count < 1
        ):
            raise ValueError("benchmark component sample_count must be positive")


@dataclass(frozen=True, slots=True)
class BenchmarkStageTiming:
    """Per-point timings for one native evaluator stage.

    The leaf/backend/output-gather fields are internal attribution, not
    additional top-level phases. A full-stage leaf gather is owned by the
    evaluator-call envelope; a composed selected-chunk leaf gather is owned by
    the input-pack envelope.
    """

    stage_index: int
    input_pack_time: BenchmarkComponentTiming | None = None
    evaluator_call_time: BenchmarkComponentTiming | None = None
    output_assign_time: BenchmarkComponentTiming | None = None
    leaf_input_pack_time: BenchmarkComponentTiming | None = None
    backend_call_time: BenchmarkComponentTiming | None = None
    evaluator_output_gather_time: BenchmarkComponentTiming | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.stage_index, bool)
            or not isinstance(self.stage_index, int)
            or self.stage_index < 1
        ):
            raise ValueError("benchmark stage_index must be positive")
        timings = (
            self.input_pack_time,
            self.evaluator_call_time,
            self.output_assign_time,
            self.leaf_input_pack_time,
            self.backend_call_time,
            self.evaluator_output_gather_time,
        )
        if not any(timing is not None for timing in timings):
            raise ValueError("benchmark stage timing must contain at least one value")
        if any(
            timing is not None and not isinstance(timing, BenchmarkComponentTiming)
            for timing in timings
        ):
            raise TypeError("benchmark stage timings must be BenchmarkComponentTiming")


@dataclass(frozen=True, slots=True)
class BenchmarkProfileCounters:
    """Mean native profile work counts with an explicit normalization basis.

    Movement and materialization fields are means per profiled phase-space
    point. Backend-call, allocation, and compiled Direct-Arena fields are means
    per profiled runtime call. The native repeated profiler reports integer
    totals; normalizing before aggregating samples keeps these values comparable
    across batch sizes and repetition counts.
    """

    sample_count: int
    normalization: Literal["mean_per_profiled_point_or_runtime_call_v1"] = (
        "mean_per_profiled_point_or_runtime_call_v1"
    )
    native_input_components_per_point: float | None = None
    native_input_pack_bytes_per_point: float | None = None
    native_input_crossing_bytes_per_point: float | None = None
    state_components_per_point: float | None = None
    state_clear_components_per_point: float | None = None
    source_components_per_point: float | None = None
    momentum_components_per_point: float | None = None
    model_parameter_components_per_point: float | None = None
    stage_input_copy_components_per_point: float | None = None
    stage_leaf_input_copy_components_per_point: float | None = None
    stage_evaluator_output_gather_components_per_point: float | None = None
    stage_output_assign_components_per_point: float | None = None
    amplitude_input_copy_components_per_point: float | None = None
    amplitude_leaf_input_copy_components_per_point: float | None = None
    amplitude_evaluator_output_gather_components_per_point: float | None = None
    amplitude_output_remap_components_per_point: float | None = None
    reduction_input_components_per_point: float | None = None
    selector_gather_points_per_point: float | None = None
    selector_gather_bytes_per_point: float | None = None
    selector_scatter_values_per_point: float | None = None
    resolved_materialized_components_per_point: float | None = None
    total_materialized_values_per_point: float | None = None
    final_output_copy_values_per_point: float | None = None
    native_input_container_allocations_per_call: float | None = None
    evaluator_backend_calls_per_call: float | None = None
    compiled_direct_arena_engines_per_call: float | None = None
    compiled_direct_arena_calls_per_call: float | None = None
    compiled_direct_arena_boundary_input_bytes_per_call: float | None = None
    compiled_direct_arena_boundary_current_output_bytes_per_call: float | None = None
    compiled_direct_arena_boundary_amplitude_output_bytes_per_call: float | None = None
    recurrence_momentum_scalar_values_per_point: float | None = None
    recurrence_schedule_executions_per_call: float | None = None
    recurrence_replay_schedule_executions_per_call: float | None = None
    recurrence_union_schedule_executions_per_call: float | None = None
    recurrence_union_source_rows_per_call: float | None = None
    recurrence_replay_output_values_per_point: float | None = None
    recurrence_source_calls_per_call: float | None = None
    recurrence_source_rows_per_call: float | None = None
    recurrence_contribution_calls_per_call: float | None = None
    recurrence_contribution_rows_per_call: float | None = None
    recurrence_finalization_calls_per_call: float | None = None
    recurrence_finalization_rows_per_call: float | None = None
    recurrence_closure_calls_per_call: float | None = None
    recurrence_closure_rows_per_call: float | None = None
    observed_scratch_reallocations_per_call: float | None = None
    native_output_allocations_per_call: float | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count < 1
        ):
            raise ValueError("benchmark profile counter sample_count must be positive")
        if self.normalization != "mean_per_profiled_point_or_runtime_call_v1":
            raise ValueError("benchmark profile counter normalization is invalid")
        for name in (
            "native_input_components_per_point",
            "native_input_pack_bytes_per_point",
            "native_input_crossing_bytes_per_point",
            "state_components_per_point",
            "state_clear_components_per_point",
            "source_components_per_point",
            "momentum_components_per_point",
            "model_parameter_components_per_point",
            "stage_input_copy_components_per_point",
            "stage_leaf_input_copy_components_per_point",
            "stage_evaluator_output_gather_components_per_point",
            "stage_output_assign_components_per_point",
            "amplitude_input_copy_components_per_point",
            "amplitude_leaf_input_copy_components_per_point",
            "amplitude_evaluator_output_gather_components_per_point",
            "amplitude_output_remap_components_per_point",
            "reduction_input_components_per_point",
            "selector_gather_points_per_point",
            "selector_gather_bytes_per_point",
            "selector_scatter_values_per_point",
            "resolved_materialized_components_per_point",
            "total_materialized_values_per_point",
            "final_output_copy_values_per_point",
            "native_input_container_allocations_per_call",
            "evaluator_backend_calls_per_call",
            "compiled_direct_arena_engines_per_call",
            "compiled_direct_arena_calls_per_call",
            "compiled_direct_arena_boundary_input_bytes_per_call",
            "compiled_direct_arena_boundary_current_output_bytes_per_call",
            "compiled_direct_arena_boundary_amplitude_output_bytes_per_call",
            "recurrence_momentum_scalar_values_per_point",
            "recurrence_schedule_executions_per_call",
            "recurrence_replay_schedule_executions_per_call",
            "recurrence_union_schedule_executions_per_call",
            "recurrence_union_source_rows_per_call",
            "recurrence_replay_output_values_per_point",
            "recurrence_source_calls_per_call",
            "recurrence_source_rows_per_call",
            "recurrence_contribution_calls_per_call",
            "recurrence_contribution_rows_per_call",
            "recurrence_finalization_calls_per_call",
            "recurrence_finalization_rows_per_call",
            "recurrence_closure_calls_per_call",
            "recurrence_closure_rows_per_call",
            "observed_scratch_reallocations_per_call",
            "native_output_allocations_per_call",
        ):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, (float, int))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError(
                    f"benchmark profile counter {name} must be finite and non-negative"
                )


@dataclass(frozen=True, slots=True)
class BenchmarkTimingBreakdown:
    """Typed aggregate of bounded native Rusticol profile samples.

    Internal leaf/backend/output-gather/remap fields explain their enclosing
    work but are not additive top-level phases. Leaf gathering is owned by the
    evaluator envelope for full stages and by the input-pack envelope for
    composed selected-chunk paths. The recurrence schedule is an inclusive
    top-level phase; its source-kernel, contribution-kernel, finalization, and
    closure fields are sub-attribution and must not be added to it.
    """

    sample_count: int
    execution_mode: Literal["compiled", "eager", "recurrence", "on-the-fly"] = (
        "compiled"
    )
    wall_time: BenchmarkComponentTiming | None = None
    source_fill_time: BenchmarkComponentTiming | None = None
    momentum_setup_time: BenchmarkComponentTiming | None = None
    stage_input_pack_time: BenchmarkComponentTiming | None = None
    stage_evaluator_call_time: BenchmarkComponentTiming | None = None
    output_assign_time: BenchmarkComponentTiming | None = None
    amplitude_input_pack_time: BenchmarkComponentTiming | None = None
    amplitude_evaluator_call_time: BenchmarkComponentTiming | None = None
    reduction_time: BenchmarkComponentTiming | None = None
    other_core_time: BenchmarkComponentTiming | None = None
    eager_execution_time: BenchmarkComponentTiming | None = None
    eager_initialize_time: BenchmarkComponentTiming | None = None
    eager_gather_time: BenchmarkComponentTiming | None = None
    eager_kernel_call_time: BenchmarkComponentTiming | None = None
    eager_invocation_scatter_time: BenchmarkComponentTiming | None = None
    eager_finalization_time: BenchmarkComponentTiming | None = None
    eager_scatter_finalization_time: BenchmarkComponentTiming | None = None
    eager_closure_time: BenchmarkComponentTiming | None = None
    eager_copy_out_time: BenchmarkComponentTiming | None = None
    stages: tuple[BenchmarkStageTiming, ...] = ()
    native_input_pack_time: BenchmarkComponentTiming | None = None
    native_input_crossing_time: BenchmarkComponentTiming | None = None
    orchestration_time: BenchmarkComponentTiming | None = None
    state_prepare_time: BenchmarkComponentTiming | None = None
    state_clear_time: BenchmarkComponentTiming | None = None
    momentum_input_setup_time: BenchmarkComponentTiming | None = None
    model_parameter_setup_time: BenchmarkComponentTiming | None = None
    total_materialization_time: BenchmarkComponentTiming | None = None
    final_output_copy_time: BenchmarkComponentTiming | None = None
    selector_planner_time: BenchmarkComponentTiming | None = None
    selector_gather_time: BenchmarkComponentTiming | None = None
    selector_scatter_time: BenchmarkComponentTiming | None = None
    recurrence_momentum_fill_time: BenchmarkComponentTiming | None = None
    recurrence_union_source_fill_time: BenchmarkComponentTiming | None = None
    recurrence_schedule_time: BenchmarkComponentTiming | None = None
    recurrence_source_kernel_time: BenchmarkComponentTiming | None = None
    recurrence_contribution_kernel_time: BenchmarkComponentTiming | None = None
    recurrence_finalization_time: BenchmarkComponentTiming | None = None
    recurrence_closure_time: BenchmarkComponentTiming | None = None
    recurrence_replay_output_mapping_time: BenchmarkComponentTiming | None = None
    counters: BenchmarkProfileCounters | None = None
    stage_leaf_input_pack_time: BenchmarkComponentTiming | None = None
    stage_backend_call_time: BenchmarkComponentTiming | None = None
    stage_evaluator_output_gather_time: BenchmarkComponentTiming | None = None
    amplitude_leaf_input_pack_time: BenchmarkComponentTiming | None = None
    amplitude_backend_call_time: BenchmarkComponentTiming | None = None
    amplitude_evaluator_output_gather_time: BenchmarkComponentTiming | None = None
    amplitude_output_remap_time: BenchmarkComponentTiming | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count < 1
        ):
            raise ValueError("benchmark timing breakdown sample_count must be positive")
        if self.execution_mode not in {
            "compiled",
            "eager",
            "recurrence",
            "on-the-fly",
        }:
            raise ValueError(
                "benchmark timing breakdown execution_mode must be compiled, eager, "
                "recurrence, or on-the-fly"
            )
        for name in (
            "wall_time",
            "native_input_pack_time",
            "native_input_crossing_time",
            "orchestration_time",
            "state_prepare_time",
            "state_clear_time",
            "source_fill_time",
            "momentum_input_setup_time",
            "momentum_setup_time",
            "model_parameter_setup_time",
            "stage_input_pack_time",
            "stage_evaluator_call_time",
            "stage_leaf_input_pack_time",
            "stage_backend_call_time",
            "stage_evaluator_output_gather_time",
            "output_assign_time",
            "amplitude_input_pack_time",
            "amplitude_evaluator_call_time",
            "amplitude_leaf_input_pack_time",
            "amplitude_backend_call_time",
            "amplitude_evaluator_output_gather_time",
            "amplitude_output_remap_time",
            "reduction_time",
            "total_materialization_time",
            "final_output_copy_time",
            "selector_planner_time",
            "selector_gather_time",
            "selector_scatter_time",
            "other_core_time",
            "eager_execution_time",
            "eager_initialize_time",
            "eager_gather_time",
            "eager_kernel_call_time",
            "eager_invocation_scatter_time",
            "eager_finalization_time",
            "eager_scatter_finalization_time",
            "eager_closure_time",
            "eager_copy_out_time",
            "recurrence_momentum_fill_time",
            "recurrence_union_source_fill_time",
            "recurrence_schedule_time",
            "recurrence_source_kernel_time",
            "recurrence_contribution_kernel_time",
            "recurrence_finalization_time",
            "recurrence_closure_time",
            "recurrence_replay_output_mapping_time",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, BenchmarkComponentTiming):
                raise TypeError(
                    f"benchmark timing breakdown {name} must be "
                    "BenchmarkComponentTiming or null"
                )
        stages = tuple(self.stages)
        if not all(isinstance(stage, BenchmarkStageTiming) for stage in stages):
            raise TypeError("benchmark timing breakdown stages are invalid")
        indices = tuple(stage.stage_index for stage in stages)
        if len(indices) != len(set(indices)):
            raise ValueError("benchmark timing breakdown stage indices must be unique")
        if self.counters is not None and not isinstance(
            self.counters, BenchmarkProfileCounters
        ):
            raise TypeError(
                "benchmark timing breakdown counters must be "
                "BenchmarkProfileCounters or null"
            )
        object.__setattr__(self, "stages", stages)


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Per-point timings and uncertainty across independent measured blocks.

    ``sample_count`` is the number of timed blocks. Each block averages
    ``repetitions_per_sample`` runtime calls of ``effective_config.batch_size``
    points. For native f64 evaluation, ``wall_time_per_point`` is measured by
    Rusticol around repeated core evaluations of an already packed momentum
    buffer; caller-language conversion and adapter overhead are excluded.
    For compiled/eager execution, ``evaluator_time_per_point`` is the relevant
    evaluator envelope measured by the bounded native profiler. For recurrence
    and on-the-fly execution, it is the inclusive recurrence schedule measured
    by the paired profiled pass. ``evaluator_total_time_per_point`` is the
    accumulated warmed headline evaluator duration divided by its authenticated
    measured-point count; it is distinct from both wall-sample statistics and
    execution attribution.
    ``interrupted`` marks a valid partial result computed only from timing blocks
    that finished before sampling was interrupted.
    """

    requested_config: BenchmarkConfig
    effective_config: BenchmarkConfig
    sample_count: int
    wall_time_per_point: float
    evaluator_time_per_point: float | None
    uncertainty: BenchmarkStatistics
    environment: Mapping[str, object]
    repetitions_per_sample: int = 1
    evaluator_uncertainty: BenchmarkStatistics | None = None
    process_id: str | None = None
    process_expression: str | None = None
    timing_breakdown: BenchmarkTimingBreakdown | None = None
    interrupted: bool = False
    evaluator_total_time_per_point: float | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count < 1
        ):
            raise ValueError("benchmark sample_count must be positive")
        if not isinstance(self.interrupted, bool):
            raise TypeError("benchmark interrupted flag must be a boolean")
        if not math.isfinite(self.wall_time_per_point) or self.wall_time_per_point < 0:
            raise ValueError("benchmark wall_time_per_point must be non-negative")
        if self.evaluator_time_per_point is not None and (
            not math.isfinite(self.evaluator_time_per_point)
            or self.evaluator_time_per_point < 0
        ):
            raise ValueError("benchmark evaluator_time_per_point must be non-negative")
        if self.evaluator_total_time_per_point is not None and (
            not math.isfinite(self.evaluator_total_time_per_point)
            or self.evaluator_total_time_per_point <= 0
        ):
            raise ValueError(
                "benchmark evaluator_total_time_per_point must be positive"
            )
        if (
            isinstance(self.repetitions_per_sample, bool)
            or not isinstance(self.repetitions_per_sample, int)
            or self.repetitions_per_sample < 1
        ):
            raise ValueError("benchmark repetitions_per_sample must be positive")
        if self.evaluator_uncertainty is not None and not isinstance(
            self.evaluator_uncertainty, BenchmarkStatistics
        ):
            raise TypeError(
                "benchmark evaluator_uncertainty must be BenchmarkStatistics or null"
            )
        for name in ("process_id", "process_expression"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"benchmark {name} must be a non-empty string or null")
        if self.timing_breakdown is not None and not isinstance(
            self.timing_breakdown, BenchmarkTimingBreakdown
        ):
            raise TypeError(
                "benchmark timing_breakdown must be BenchmarkTimingBreakdown or null"
            )
        object.__setattr__(self, "environment", _freeze_mapping(self.environment))

    @property
    def evaluation_count(self) -> int:
        """Return the number of timed runtime evaluations."""

        return self.sample_count * self.repetitions_per_sample

    @property
    def evaluated_point_count(self) -> int:
        """Return the number of phase-space point evaluations timed."""

        return self.evaluation_count * self.effective_config.batch_size


__all__ = [
    "BenchmarkComponentTiming",
    "BenchmarkProfileCounters",
    "BenchmarkResult",
    "BenchmarkStageTiming",
    "BenchmarkStatistics",
    "BenchmarkTimingBreakdown",
    "ColorComponent",
    "ColorFlow",
    "ContractedColorComponent",
    "ExternalParticle",
    "GenerationPlan",
    "GenerationResult",
    "HelicityConfiguration",
    "ModelParameter",
    "PhysicsReduction",
    "ProcessPhysics",
    "ReductionGroup",
    "ResolvedEvaluation",
    "WarmUpResult",
]

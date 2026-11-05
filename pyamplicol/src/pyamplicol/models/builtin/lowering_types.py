# SPDX-License-Identifier: 0BSD
"""Built-in-SM diagnostics for symbolic lowering results."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TensorNetworkProbe:
    engine: str
    tensor_names: tuple[str, ...]
    expression: str
    output_structure: str
    output_rank: int
    output_size: int
    nonzero_entries: int
    max_abs_entry: float
    weighted_checksum: tuple[float, float]
    first_nonzero_entries: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class ColorAlgebraProbe:
    engine: str
    input_expression: str
    simplified_expression: str


@dataclass(frozen=True)
class RecursionLoweringPlan:
    engine: str
    current_count: int
    interaction_count: int
    amplitude_count: int
    source_current_count: int
    vertex_kind_counts: tuple[tuple[int, int], ...]
    tensor_route_vertex_kinds: tuple[int, ...]
    color_order: tuple[int, ...]
    expression: str
    expression_length: int
    expression_truncated: bool
    first_assignments: tuple[str, ...]


@dataclass(frozen=True)
class VertexLoweringStep:
    index: int
    vertex_kind: int
    backend: str
    tensor_names: tuple[str, ...]
    expression_head: str
    full_tensor_network_ready: bool
    result_current: str
    left_current: str
    right_current: str


@dataclass(frozen=True)
class VertexLoweringReport:
    total_interactions: int
    full_tensor_network_ready: bool
    backend_counts: tuple[tuple[str, int], ...]
    ready_vertex_kind_counts: tuple[tuple[int, int], ...]
    pending_vertex_kind_counts: tuple[tuple[int, int], ...]
    tensor_names: tuple[str, ...]
    first_steps: tuple[VertexLoweringStep, ...]


@dataclass(frozen=True)
class TensorNetworkBlueprint:
    engine: str
    status: str
    current_count: int
    interaction_count: int
    amplitude_count: int
    expression_built: bool
    expression_executed: bool
    full_me_tensor_network_ready: bool
    propagator_lowering_ready: bool
    ready_interactions: int
    pending_interactions: int
    placeholder_vertex_kinds: tuple[int, ...]
    registered_tensor_names: tuple[str, ...]
    current_leaf_count: int
    parametric_external_current_count: int
    parametric_source_current_parameter_count: int
    parametric_current_momentum_count: int
    parametric_momentum_parameter_count: int
    parametric_parameter_count: int
    expression: str | None
    expression_length: int | None
    expression_truncated: bool | None
    executed_expression: str | None
    executed_expression_length: int | None
    executed_expression_truncated: bool | None
    execution_time_s: float | None


@dataclass(frozen=True)
class SymbolicLoweringReport:
    tensor_library: str
    tensor_network_probe: TensorNetworkProbe
    color_algebra_probe: ColorAlgebraProbe
    recursion_plan: RecursionLoweringPlan | None = None
    vertex_lowering: VertexLoweringReport | None = None
    tensor_network_blueprint: TensorNetworkBlueprint | None = None
    full_me_tensor_network_ready: bool = False

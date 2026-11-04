# SPDX-License-Identifier: 0BSD
"""Frozen stage compilation records and runtime-model adapter."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from ..models._physics_ir import PropagatorIR
from ..models.base import Model


@dataclass(frozen=True)
class GenericStageInputComponent:
    kind: str
    source_id: int
    component: int
    global_component: int
    parameter_index: int
    real_valued: bool = False

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "source_id": self.source_id,
            "component": self.component,
            "global_component": self.global_component,
            "parameter_index": self.parameter_index,
            "real_valued": self.real_valued,
        }


@dataclass(frozen=True)
class GenericStageOutputSlot:
    value_slot_id: int
    current_id: int
    variant: str
    component_start: int
    component_stop: int
    output_start: int
    output_stop: int
    selector_domain_ids: tuple[int, ...] = field(
        default=(),
        repr=False,
        compare=False,
    )
    color_selector_domain_ids: tuple[int, ...] = field(
        default=(),
        repr=False,
        compare=False,
    )

    def to_json_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "value_slot_id": self.value_slot_id,
            "current_id": self.current_id,
            "variant": self.variant,
            "component_start": self.component_start,
            "component_stop": self.component_stop,
            "output_start": self.output_start,
            "output_stop": self.output_stop,
        }
        if self.color_selector_domain_ids:
            result["color_selector_domain_ids"] = list(
                self.color_selector_domain_ids
            )
        return result


@dataclass(frozen=True)
class GenericCompiledStageBlueprint:
    stage_index: int
    stage_kind: str
    subset_size: int | None
    evaluator_label: str
    parameter_layout: str
    output_length: int
    output_slots: tuple[GenericStageOutputSlot, ...]
    input_value_slot_ids: tuple[int, ...]
    output_value_slot_ids: tuple[int, ...]
    interaction_ids: tuple[int, ...]
    input_components: tuple[GenericStageInputComponent, ...]
    parameter_count: int
    value_parameter_count: int
    momentum_parameter_count: int
    model_parameter_count: int
    real_valued_inputs: tuple[int, ...]
    expression_ready: bool
    blockers: tuple[str, ...]
    first_output_previews: tuple[str, ...]
    evaluation_groups_by_current: tuple[tuple[int, tuple[int, ...]], ...] = field(
        default=(),
        repr=False,
        compare=False,
    )
    fanout_chunk_size: int | None = None
    fanout_evaluation_occurrences_before: int | None = None
    fanout_evaluation_occurrences_after: int | None = None
    selector_output_partitions: tuple[tuple[int, int], ...] = field(
        default=(),
        repr=False,
        compare=False,
    )
    parameter_symbols: tuple[Any, ...] = field(
        default=(),
        repr=False,
        compare=False,
    )
    output_expressions: tuple[Any, ...] = field(
        default=(),
        repr=False,
        compare=False,
    )
    symbolica_functions: tuple[tuple[Any, tuple[Any, ...], Any], ...] = field(
        default=(),
        repr=False,
        compare=False,
    )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "stage_index": self.stage_index,
            "stage_kind": self.stage_kind,
            "subset_size": self.subset_size,
            "evaluator_label": self.evaluator_label,
            "parameter_layout": self.parameter_layout,
            "output_length": self.output_length,
            "output_slots": [slot.to_json_dict() for slot in self.output_slots],
            "input_value_slot_ids": list(self.input_value_slot_ids),
            "output_value_slot_ids": list(self.output_value_slot_ids),
            "interaction_ids": list(self.interaction_ids),
            "input_components": [
                component.to_json_dict() for component in self.input_components
            ],
            "parameter_count": self.parameter_count,
            "value_parameter_count": self.value_parameter_count,
            "momentum_parameter_count": self.momentum_parameter_count,
            "model_parameter_count": self.model_parameter_count,
            "real_valued_inputs": list(self.real_valued_inputs),
            "expression_ready": self.expression_ready,
            "blockers": list(self.blockers),
            "first_output_previews": list(self.first_output_previews),
            "symbolica_function_count": len(self.symbolica_functions),
            "evaluation_fanout": {
                "strategy": (
                    "shared-evaluation-affinity"
                    if self.fanout_chunk_size is not None
                    else "natural-current-order"
                ),
                "chunk_size": self.fanout_chunk_size,
                "evaluation_occurrences_before": (
                    self.fanout_evaluation_occurrences_before
                ),
                "evaluation_occurrences_after": (
                    self.fanout_evaluation_occurrences_after
                ),
            },
        }


@dataclass(frozen=True)
class GenericStageCompilerBlueprint:
    kind: str
    runtime_available: bool
    parameter_count: int
    value_parameter_count: int
    momentum_parameter_count: int
    model_parameter_count: int
    real_valued_inputs: tuple[int, ...]
    stage_count: int
    stages: tuple[GenericCompiledStageBlueprint, ...]
    amplitude_stage: GenericCompiledStageBlueprint
    expression_ready: bool
    blockers: tuple[str, ...]
    parameter_symbols: tuple[Any, ...] = field(
        default=(),
        repr=False,
        compare=False,
    )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "runtime_available": self.runtime_available,
            "parameter_count": self.parameter_count,
            "value_parameter_count": self.value_parameter_count,
            "momentum_parameter_count": self.momentum_parameter_count,
            "model_parameter_count": self.model_parameter_count,
            "real_valued_inputs": list(self.real_valued_inputs),
            "stage_count": self.stage_count,
            "stages": [stage.to_json_dict() for stage in self.stages],
            "amplitude_stage": self.amplitude_stage.to_json_dict(),
            "expression_ready": self.expression_ready,
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class _StageLocalInputs:
    parameter_symbols: tuple[Any, ...]
    input_components: tuple[GenericStageInputComponent, ...]
    value_symbols: tuple[Any, ...] | Mapping[int, tuple[Any, ...]]
    momentum_symbols: tuple[Any, ...] | Mapping[int, tuple[Any, ...]]
    model_parameter_symbols: Mapping[str, Any]
    value_parameter_count: int
    momentum_parameter_count: int
    model_parameter_count: int
    real_valued_inputs: tuple[int, ...]


StageEvaluatorCompiler = Callable[
    [GenericCompiledStageBlueprint, tuple[Any, ...], tuple[int, ...]],
    dict[str, object],
]
StageBlueprintProgress = Callable[[str, int, int], None]
StageBlueprintConsumer = Callable[
    [GenericCompiledStageBlueprint, int, int],
    None,
]


class _RuntimeParameterizedModel:
    """Overlay runtime parameters while delegating every model decision.

    This proxy deliberately does not inherit any concrete model. Models with a
    specialized runtime-parameter implementation can still provide
    ``with_runtime_parameters`` and bypass it.
    """

    def __init__(self, base: Model, parameters: Mapping[str, Any]) -> None:
        self._base_model = base
        self._runtime_parameters = parameters
        self.name = base.name
        self.particles = base.particles
        self.vertices = base.vertices

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_model, name)

    def mass(self, pdg: int) -> Any:
        particle = self._base_model.particle(pdg)
        name = f"particle.{int(particle.pdg)}.mass"
        return self._runtime_parameters.get(name, self._base_model.mass(pdg))

    def width(self, pdg: int) -> Any:
        particle = self._base_model.particle(pdg)
        name = f"particle.{int(particle.pdg)}.width"
        return self._runtime_parameters.get(name, self._base_model.width(pdg))

    def propagator_lowering_rule(self, particle_id: int, chirality: int = 0) -> Any:
        return self._base_model.propagator_lowering_rule(particle_id, chirality)

    def propagator_component_expression(
        self,
        particle_id: int,
        value: Sequence[Any],
        momentum: Sequence[Any],
        *,
        chirality: int = 0,
        propagator: PropagatorIR | None = None,
    ) -> tuple[Any, ...]:
        return cast(
            tuple[Any, ...],
            type(self._base_model).propagator_component_expression(
                self,
                particle_id,
                value,
                momentum,
                chirality=chirality,
                propagator=propagator,
            ),
        )

    def is_chiral_eligible(self, pdg: int) -> bool:
        return self._base_model.is_chiral_eligible(pdg)

    def with_runtime_parameters(
        self,
        parameters: Mapping[str, Any],
    ) -> _RuntimeParameterizedModel:
        return _RuntimeParameterizedModel(self._base_model, parameters)

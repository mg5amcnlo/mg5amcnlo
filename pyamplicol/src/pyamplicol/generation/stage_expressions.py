# SPDX-License-Identifier: 0BSD
"""Symbolic expression construction for current and amplitude stages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..models._physics_ir import ContractionIR, PropagatorIR
from ..models.base import Model
from ..models.expressions import _exact_complex_expression
from .contracts import (
    runtime_coupling_parameter_names as _runtime_coupling_parameter_names,
)
from .dag_types import GenericDAG
from .stage_parameters import (
    _amplitude_stage_model_parameter_records,
    _contract_components,
    _coupling,
    _current_stage_model_parameter_records,
    _dict,
    _expression_previews,
    _global_stage_inputs,
    _list,
    _model_symbolica_functions,
    _momentum_components,
    _project_components,
    _runtime_coupling,
    _specialize_stage_symbolica_functions,
    _stage_input_momentum_slot_ids,
    _stage_local_inputs,
    _sum_components,
    _value_components,
)
from .stage_types import (
    GenericCompiledStageBlueprint,
    GenericStageOutputSlot,
    _RuntimeParameterizedModel,
)


def _compile_current_stage_blueprint(
    dag: GenericDAG,
    model: Model,
    stage: dict[str, Any],
    *,
    value_slots: dict[int, dict[str, Any]],
    current_slots: dict[int, dict[str, Any]],
    momentum_slots: dict[int, dict[str, Any]],
    global_value_component_count: int,
    global_momentum_parameter_count: int,
    model_parameter_records: Sequence[dict[str, Any]],
    global_parameter_symbols: Sequence[Any],
    global_value_symbols: Sequence[Any],
    global_momentum_symbols: Sequence[Any],
    global_model_parameter_symbols: Mapping[str, Any],
    global_real_valued_inputs: Sequence[int],
    stage_local_parameter_layout: bool,
) -> GenericCompiledStageBlueprint:
    blockers: list[str] = []
    outputs: list[Any] = []
    output_slots: list[GenericStageOutputSlot] = []
    interactions_compacted = bool(stage.get("interactions_compacted", False))
    interactions = (
        []
        if interactions_compacted
        else [_dict(item) for item in _list(stage["interactions"])]
    )
    interaction_ids = (
        tuple(int(value) for value in _list(stage.get("interaction_ids", [])))
        if interactions_compacted
        else tuple(int(interaction["interaction_id"]) for interaction in interactions)
    )
    input_value_slot_ids = tuple(
        int(value) for value in _list(stage["input_value_slot_ids"])
    )
    input_momentum_slot_ids = (
        tuple(int(value) for value in _list(stage.get("input_momentum_slot_ids", [])))
        if interactions_compacted
        else _stage_input_momentum_slot_ids(interactions)
    )
    output_slot_ids = tuple(
        int(value) for value in _list(stage["output_value_slot_ids"])
    )
    output_slots_by_current: dict[int, list[dict[str, Any]]] = {}
    for slot_id in output_slot_ids:
        slot = value_slots[int(slot_id)]
        output_slots_by_current.setdefault(int(slot["current_id"]), []).append(slot)
    stage_model_parameter_records = (
        _current_stage_model_parameter_records(
            model,
            model_parameter_records,
            dag=dag,
            interactions=interactions,
            interaction_ids=interaction_ids,
            output_slots_by_current=output_slots_by_current,
            current_slots=current_slots,
        )
        if stage_local_parameter_layout
        else model_parameter_records
    )
    local_inputs = (
        _stage_local_inputs(
            value_slot_ids=input_value_slot_ids,
            momentum_slot_ids=input_momentum_slot_ids,
            value_slots=value_slots,
            momentum_slots=momentum_slots,
            global_value_component_count=global_value_component_count,
            global_momentum_parameter_count=global_momentum_parameter_count,
            model_parameter_records=stage_model_parameter_records,
        )
        if stage_local_parameter_layout
        else _global_stage_inputs(
            parameter_symbols=global_parameter_symbols,
            value_symbols=global_value_symbols,
            momentum_symbols=global_momentum_symbols,
            model_parameter_symbols=global_model_parameter_symbols,
            value_parameter_count=global_value_component_count,
            momentum_parameter_count=len(global_momentum_symbols),
            model_parameter_count=len(model_parameter_records),
            real_valued_inputs=global_real_valued_inputs,
        )
    )
    stage_model = (
        model.with_runtime_parameters(local_inputs.model_parameter_symbols)
        if hasattr(model, "with_runtime_parameters")
        else _RuntimeParameterizedModel(model, local_inputs.model_parameter_symbols)
    )
    interactions_by_result: dict[int, list[dict[str, Any] | int]] = {}
    if interactions_compacted:
        for interaction_id in interaction_ids:
            interactions_by_result.setdefault(
                int(dag.interactions[interaction_id].result_id),
                [],
            ).append(interaction_id)
    else:
        for interaction in interactions:
            interactions_by_result.setdefault(
                int(interaction["result_current_id"]),
                [],
            ).append(interaction)
    value_components_by_slot_id = {
        int(slot_id): _value_components(
            value_slots[int(slot_id)],
            local_inputs.value_symbols,
        )
        for slot_id in input_value_slot_ids
    }
    momentum_components_by_slot_id = {
        int(slot_id): _momentum_components(
            int(slot_id),
            local_inputs.momentum_symbols,
            momentum_slots,
            by_slot_id=True,
        )
        for slot_id in input_momentum_slot_ids
    }
    momentum_components_by_mask = {
        int(slot["momentum_mask"]): momentum_components_by_slot_id[int(slot_id)]
        for slot_id, slot in momentum_slots.items()
        if int(slot_id) in momentum_components_by_slot_id
    }
    input_value_slot_by_current_id = {
        int(value_slots[slot_id]["current_id"]): int(slot_id)
        for slot_id in input_value_slot_ids
    }
    momentum_slot_by_mask = {
        int(slot["momentum_mask"]): int(slot_id)
        for slot_id, slot in momentum_slots.items()
    }
    compact_coupling_cache: dict[
        tuple[int, tuple[int, ...], tuple[float, ...]],
        tuple[Any, Any],
    ] = {}
    compact_evaluation_cache: dict[int, tuple[Any, ...]] = {}
    evaluation_groups_by_current = tuple(
        (
            current_id,
            tuple(
                sorted(
                    {
                        (
                            int(interaction.evaluation_group_id)
                            if interaction.evaluation_group_id is not None
                            else -(interaction.id + 1)
                        )
                        for interaction_item in interactions_by_result[current_id]
                        for interaction in (
                            dag.interactions[
                                _interaction_item_id(
                                    interaction_item,
                                    compacted=interactions_compacted,
                                )
                            ],
                        )
                        if interaction.evaluation_factor != (0.0, 0.0)
                    }
                )
            ),
        )
        for current_id in sorted(interactions_by_result)
    )

    for current_id in sorted(interactions_by_result):
        current_slot = current_slots[current_id]
        dimension = int(current_slot["dimension"])
        total = tuple(0j for _ in range(dimension))
        for interaction_item in interactions_by_result[current_id]:
            interaction_id = _interaction_item_id(
                interaction_item,
                compacted=interactions_compacted,
            )
            try:
                contribution = _compact_interaction_contribution(
                    dag,
                    stage_model,
                    interaction_id,
                    value_components_by_slot_id=value_components_by_slot_id,
                    input_value_slot_by_current_id=input_value_slot_by_current_id,
                    momentum_components_by_slot_id=momentum_components_by_slot_id,
                    momentum_slot_by_mask=momentum_slot_by_mask,
                    model_parameter_symbols=local_inputs.model_parameter_symbols,
                    coupling_cache=compact_coupling_cache,
                    evaluation_cache=compact_evaluation_cache,
                )
            except ValueError as error:
                blockers.append(f"interaction {interaction_id}: {error}")
                continue
            total = _sum_components(total, contribution)
        result_slots = output_slots_by_current.get(current_id, ())
        for slot in result_slots:
            variant = str(slot["variant"])
            try:
                components = (
                    stage_model.propagator_component_expression(
                        int(current_slot["particle_id"]),
                        total,
                        momentum_components_by_mask[int(current_slot["momentum_mask"])],
                        chirality=int(current_slot["chirality"]),
                        propagator=PropagatorIR.from_json_dict(
                            _dict(slot["propagator"])
                        ),
                    )
                    if variant == "propagated"
                    else total
                )
            except ValueError as error:
                blockers.append(f"value slot {slot['value_slot_id']}: {error}")
                continue
            start = len(outputs)
            outputs.extend(components)
            output_slots.append(
                GenericStageOutputSlot(
                    value_slot_id=int(slot["value_slot_id"]),
                    current_id=current_id,
                    variant=variant,
                    component_start=int(slot["component_start"]),
                    component_stop=int(slot["component_stop"]),
                    output_start=start,
                    output_stop=len(outputs),
                )
            )

    specialized_outputs, symbolica_functions = _specialize_stage_symbolica_functions(
        outputs,
        _model_symbolica_functions(stage_model),
    )
    return GenericCompiledStageBlueprint(
        stage_index=int(stage["stage_index"]),
        stage_kind=str(stage["stage_kind"]),
        subset_size=int(stage["subset_size"]),
        evaluator_label=(
            f"generic_stage_{int(stage['stage_index'])}_subset_{int(stage['subset_size'])}"
        ),
        parameter_layout=(
            "stage-local-value-momentum"
            if stage_local_parameter_layout
            else "global-value-momentum"
        ),
        output_length=len(outputs),
        output_slots=tuple(output_slots),
        input_value_slot_ids=input_value_slot_ids,
        output_value_slot_ids=output_slot_ids,
        interaction_ids=interaction_ids,
        input_components=local_inputs.input_components,
        parameter_count=len(local_inputs.parameter_symbols),
        value_parameter_count=local_inputs.value_parameter_count,
        momentum_parameter_count=local_inputs.momentum_parameter_count,
        model_parameter_count=local_inputs.model_parameter_count,
        real_valued_inputs=local_inputs.real_valued_inputs,
        expression_ready=not blockers,
        blockers=tuple(blockers),
        first_output_previews=_expression_previews(specialized_outputs),
        evaluation_groups_by_current=evaluation_groups_by_current,
        parameter_symbols=local_inputs.parameter_symbols,
        output_expressions=specialized_outputs,
        symbolica_functions=symbolica_functions,
    )


def _compile_amplitude_stage_blueprint(
    model: Model,
    stage: dict[str, Any],
    *,
    value_slots: dict[int, dict[str, Any]],
    global_value_component_count: int,
    global_momentum_parameter_count: int,
    model_parameter_records: Sequence[dict[str, Any]],
    global_parameter_symbols: Sequence[Any],
    global_value_symbols: Sequence[Any],
    global_model_parameter_symbols: Mapping[str, Any],
    global_real_valued_inputs: Sequence[int],
    stage_local_parameter_layout: bool,
) -> GenericCompiledStageBlueprint:
    blockers: list[str] = []
    outputs: list[Any] = []
    output_slots: list[GenericStageOutputSlot] = []
    input_value_slot_ids = tuple(
        sorted(
            {
                int(root[side]["value_slot_id"])
                for root in (_dict(item) for item in _list(stage["roots"]))
                for side in ("left_value_slot", "right_value_slot")
            }
        )
    )
    local_inputs = (
        _stage_local_inputs(
            value_slot_ids=input_value_slot_ids,
            momentum_slot_ids=(),
            value_slots=value_slots,
            momentum_slots={},
            global_value_component_count=global_value_component_count,
            global_momentum_parameter_count=global_momentum_parameter_count,
            model_parameter_records=(
                _amplitude_stage_model_parameter_records(
                    model_parameter_records,
                    roots=tuple(_dict(item) for item in _list(stage["roots"])),
                )
                if stage_local_parameter_layout
                else model_parameter_records
            ),
        )
        if stage_local_parameter_layout
        else _global_stage_inputs(
            parameter_symbols=global_parameter_symbols,
            value_symbols=global_value_symbols,
            momentum_symbols=(),
            model_parameter_symbols=global_model_parameter_symbols,
            value_parameter_count=global_value_component_count,
            momentum_parameter_count=0,
            model_parameter_count=len(model_parameter_records),
            real_valued_inputs=global_real_valued_inputs,
        )
    )
    stage_model = (
        model.with_runtime_parameters(local_inputs.model_parameter_symbols)
        if hasattr(model, "with_runtime_parameters")
        else _RuntimeParameterizedModel(model, local_inputs.model_parameter_symbols)
    )
    for root in (_dict(item) for item in _list(stage["roots"])):
        try:
            output = _amplitude_root_expression(
                stage_model,
                root,
                value_symbols=local_inputs.value_symbols,
                model_parameter_symbols=local_inputs.model_parameter_symbols,
                value_slots=value_slots,
            )
        except (TypeError, ValueError) as error:
            blockers.append(f"amplitude root {root['root_id']}: {error}")
            continue
        start = len(outputs)
        outputs.append(output)
        output_slots.append(
            GenericStageOutputSlot(
                value_slot_id=-1,
                current_id=-1,
                variant="amplitude-root",
                component_start=int(root["output_index"]),
                component_stop=int(root["output_index"]) + 1,
                output_start=start,
                output_stop=len(outputs),
            )
        )
    return GenericCompiledStageBlueprint(
        stage_index=0,
        stage_kind=str(stage["stage_kind"]),
        subset_size=None,
        evaluator_label="generic_amplitude_stage",
        parameter_layout=(
            "stage-local-value-momentum"
            if stage_local_parameter_layout
            else "global-value-momentum"
        ),
        output_length=len(outputs),
        output_slots=tuple(output_slots),
        input_value_slot_ids=input_value_slot_ids,
        output_value_slot_ids=(),
        interaction_ids=(),
        input_components=local_inputs.input_components,
        parameter_count=len(local_inputs.parameter_symbols),
        value_parameter_count=local_inputs.value_parameter_count,
        momentum_parameter_count=local_inputs.momentum_parameter_count,
        model_parameter_count=local_inputs.model_parameter_count,
        real_valued_inputs=local_inputs.real_valued_inputs,
        expression_ready=not blockers,
        blockers=tuple(blockers),
        first_output_previews=_expression_previews(outputs),
        parameter_symbols=local_inputs.parameter_symbols,
        output_expressions=tuple(outputs),
    )


def _interaction_contribution(
    dag: GenericDAG,
    model: Model,
    interaction: dict[str, Any],
    *,
    value_components_by_slot_id: Mapping[int, tuple[Any, ...]],
    momentum_components_by_slot_id: Mapping[int, tuple[Any, ...]],
    model_parameter_symbols: Mapping[str, Any],
) -> tuple[Any, ...]:
    left_slot = _dict(interaction["left_value_slot"])
    right_slot = _dict(interaction["right_value_slot"])
    left = value_components_by_slot_id[int(left_slot["value_slot_id"])]
    right = value_components_by_slot_id[int(right_slot["value_slot_id"])]
    momenta = _dict(interaction["momentum_slots"])
    left_current = dag.currents[int(interaction["left_current_id"])]
    right_current = dag.currents[int(interaction["right_current_id"])]
    result_current = dag.currents[int(interaction["result_current_id"])]
    components = model.vertex_component_expression(
        int(interaction["vertex_kind"]),
        left,
        right,
        result_particle_id=int(result_current.index.particle_id),
        result_chirality=int(result_current.index.chirality),
        left_chirality=int(left_current.index.chirality),
        right_chirality=int(right_current.index.chirality),
        coupling=_runtime_coupling(interaction, model_parameter_symbols),
        left_momentum=momentum_components_by_slot_id[int(momenta["left"])],
        right_momentum=momentum_components_by_slot_id[int(momenta["right"])],
    )
    color_weight = _coupling(interaction.get("color_weight"))
    if color_weight == (1.0, 0.0):
        return components
    weight = _exact_complex_expression(complex(*color_weight))
    return tuple(weight * component for component in components)


def _compact_interaction_contribution(
    dag: GenericDAG,
    model: Model,
    interaction_id: int,
    *,
    value_components_by_slot_id: Mapping[int, tuple[Any, ...]],
    input_value_slot_by_current_id: Mapping[int, int],
    momentum_components_by_slot_id: Mapping[int, tuple[Any, ...]],
    momentum_slot_by_mask: Mapping[int, int],
    model_parameter_symbols: Mapping[str, Any],
    coupling_cache: dict[
        tuple[int, tuple[int, ...], tuple[float, ...]],
        tuple[Any, Any],
    ],
    evaluation_cache: dict[int, tuple[Any, ...]],
) -> tuple[Any, ...]:
    interaction = dag.interactions[interaction_id]
    evaluation_group_id = interaction.evaluation_group_id
    canonical_components = (
        None
        if evaluation_group_id is None
        else evaluation_cache.get(evaluation_group_id)
    )
    evaluation_factor = complex(*interaction.evaluation_factor)
    if evaluation_factor == 0j:
        # Authenticated numerical zero-current relations propagate an exact
        # zero attachment into downstream interactions.  Do not evaluate a
        # kernel whose contribution is certified to be zero.
        return tuple(
            0j for _component in range(dag.currents[interaction.result_id].dimension)
        )
    if canonical_components is None:
        left_current = dag.currents[interaction.left_id]
        right_current = dag.currents[interaction.right_id]
        result_current = dag.currents[interaction.result_id]
        left = value_components_by_slot_id[
            input_value_slot_by_current_id[interaction.left_id]
        ]
        right = value_components_by_slot_id[
            input_value_slot_by_current_id[interaction.right_id]
        ]
        coupling_key = (
            int(interaction.vertex_kind),
            interaction.vertex_particles,
            interaction.coupling,
        )
        coupling = coupling_cache.get(coupling_key)
        if coupling is None:
            resolved_coupling = list(interaction.coupling)
            names = _runtime_coupling_parameter_names(
                interaction.vertex_kind,
                interaction.vertex_particles,
                interaction.coupling,
                model=model,
            )
            for index, name in enumerate(names):
                if isinstance(name, str) and name in model_parameter_symbols:
                    resolved_coupling[index] = model_parameter_symbols[name]
            coupling = (resolved_coupling[0], resolved_coupling[1])
            coupling_cache[coupling_key] = coupling
        components = model.vertex_component_expression(
            int(interaction.vertex_kind),
            left,
            right,
            result_particle_id=int(result_current.index.particle_id),
            result_chirality=int(result_current.index.chirality),
            left_chirality=int(left_current.index.chirality),
            right_chirality=int(right_current.index.chirality),
            coupling=coupling,
            left_momentum=momentum_components_by_slot_id[
                momentum_slot_by_mask[left_current.index.momentum_mask]
            ],
            right_momentum=momentum_components_by_slot_id[
                momentum_slot_by_mask[right_current.index.momentum_mask]
            ],
        )
        if evaluation_factor == 1.0 + 0.0j:
            canonical_components = components
        else:
            exact_factor = _exact_complex_expression(evaluation_factor)
            canonical_components = tuple(
                component / exact_factor for component in components
            )
        if evaluation_group_id is not None:
            evaluation_cache[evaluation_group_id] = canonical_components
    color_weight = complex(*interaction.color_weight)
    if color_weight == evaluation_factor == 1.0 + 0.0j:
        return canonical_components
    weight = _exact_complex_expression(color_weight) * _exact_complex_expression(
        evaluation_factor
    )
    return tuple(weight * component for component in canonical_components)


def _amplitude_root_expression(
    model: Model,
    root: dict[str, Any],
    *,
    value_symbols: Sequence[Any] | Mapping[int, tuple[Any, ...]],
    model_parameter_symbols: Mapping[str, Any],
    value_slots: dict[int, dict[str, Any]],
) -> Any:
    left_value_slot = _dict(root["left_value_slot"])
    right_value_slot = _dict(root["right_value_slot"])
    left = _value_components(left_value_slot, value_symbols)
    right = _value_components(right_value_slot, value_symbols)
    left_value_metadata = _amplitude_value_slot_metadata(
        left_value_slot,
        value_slots,
    )
    right_value_metadata = _amplitude_value_slot_metadata(
        right_value_slot,
        value_slots,
    )
    kind = str(root["kind"])
    contraction_ir = _root_contraction_ir(root)
    coupling = _runtime_coupling(root, model_parameter_symbols)
    color_weight = _coupling(root.get("color_weight"))
    if kind == "direct-contraction":
        _validate_direct_contraction_metadata(
            contraction_ir,
            left_value_slot,
            right_value_slot,
            left_value_metadata,
            right_value_metadata,
        )
        return _apply_amplitude_color_weight(
            color_weight,
            _contract_components(contraction_ir, left, right),
        )
    if kind == "vertex-closure":
        _validate_scalar_projection(contraction_ir)
        vertex_kind = root.get("vertex_kind")
        particles = root.get("vertex_particles")
        if (
            vertex_kind is None
            or not isinstance(particles, list)
            or len(particles) != 3
        ):
            raise ValueError("vertex closure is missing vertex metadata")
        components = model.vertex_component_expression(
            int(vertex_kind),
            left,
            right,
            result_particle_id=int(particles[2]),
            result_chirality=0,
            left_chirality=int(left_value_metadata["chirality"]),
            right_chirality=int(right_value_metadata["chirality"]),
            coupling=coupling,
        )
        return _apply_amplitude_color_weight(
            color_weight,
            _project_components(contraction_ir, components),
        )
    raise ValueError(f"unsupported amplitude root kind {kind!r}")


def _root_contraction_ir(root: Mapping[str, Any]) -> ContractionIR:
    payload = root.get("contraction_ir")
    if not isinstance(payload, Mapping):
        raise TypeError("amplitude root contraction_ir must be a mapping")
    contraction_ir = ContractionIR.from_json_dict(payload)
    if root.get("contraction") != contraction_ir.name:
        raise ValueError(
            "amplitude root contraction projection does not match ContractionIR"
        )
    return contraction_ir


def _amplitude_value_slot_metadata(
    reference: Mapping[str, Any],
    value_slots: Mapping[int, dict[str, Any]],
) -> dict[str, Any]:
    try:
        value_slot_id = int(reference["value_slot_id"])
        return value_slots[value_slot_id]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("amplitude root references an unknown value slot") from error


def _validate_direct_contraction_metadata(
    contraction_ir: ContractionIR,
    left_reference: Mapping[str, Any],
    right_reference: Mapping[str, Any],
    left_value_slot: Mapping[str, Any],
    right_value_slot: Mapping[str, Any],
) -> None:
    dimension = len(contraction_ir.coefficients)
    for side, reference, value_slot, basis in (
        (
            "left",
            left_reference,
            left_value_slot,
            contraction_ir.left_basis,
        ),
        (
            "right",
            right_reference,
            right_value_slot,
            contraction_ir.right_basis,
        ),
    ):
        if (
            int(reference.get("dimension", -1)) != dimension
            or int(value_slot.get("dimension", -1)) != dimension
        ):
            raise ValueError(
                f"direct contraction {side} value slot dimension does not match "
                "ContractionIR"
            )
        propagator = value_slot.get("propagator")
        if not isinstance(propagator, Mapping) or propagator.get("basis") != basis:
            raise ValueError(
                f"direct contraction {side} value slot basis does not match "
                "ContractionIR"
            )

    left_chirality = int(left_value_slot["chirality"])
    right_chirality = int(right_value_slot["chirality"])
    relation_matches = (
        contraction_ir.chirality_relation == "any"
        or (
            contraction_ir.chirality_relation == "equal"
            and left_chirality == right_chirality
        )
        or (
            contraction_ir.chirality_relation == "opposite"
            and left_chirality == -right_chirality
        )
    )
    if not relation_matches:
        raise ValueError("direct contraction violates ContractionIR chirality relation")


def _validate_scalar_projection(contraction_ir: ContractionIR) -> None:
    if (
        len(contraction_ir.coefficients) != 1
        or contraction_ir.left_basis != "scalar"
        or contraction_ir.right_basis != "scalar"
        or contraction_ir.chirality_relation != "any"
    ):
        raise ValueError("vertex closure does not declare a scalar projection IR")


def _apply_amplitude_color_weight(
    color_weight: tuple[Any, Any],
    expression: Any,
) -> Any:
    if color_weight == (1.0, 0.0):
        return expression
    return _exact_complex_expression(complex(*color_weight)) * expression


def _interaction_item_id(
    item: dict[str, Any] | int,
    *,
    compacted: bool,
) -> int:
    if compacted:
        if not isinstance(item, int):
            raise TypeError("compacted stage interaction must be an integer id")
        return item
    return int(_dict(item)["interaction_id"])

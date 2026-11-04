# SPDX-License-Identifier: 0BSD
"""Neutral schema-v3 runtime layout assembly for generated DAGs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..color import ColorContractionPlan, ColorGroupDescriptor
from ..models.base import Model
from .contracts import RuntimeExpressionSchema
from .dag_types import CurrentNode, GenericDAG, InteractionNode
from .physics_metadata import (
    build_resolved_physics_payload,
    build_runtime_model_parameter_records,
    build_runtime_normalization,
)
from .runtime_amplitudes import build_runtime_amplitude_stage_layout

RUNTIME_PHYSICS_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class RuntimeStageLayout:
    """One runtime stage plus indexes retained for eager table lowering."""

    record: dict[str, object]
    interactions: tuple[InteractionNode, ...]
    evaluation_groups: tuple[tuple[InteractionNode, ...], ...]
    input_value_slot_by_current: dict[int, int]
    output_current_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RuntimeSchemaLayout:
    """Owned runtime schema plus the indexes used to assemble it.

    The eager lane consumes these indexes directly so it never has to parse the
    freshly constructed JSON-shaped schema back into Python lookup tables.
    """

    runtime_schema: dict[str, object]
    current_slots: list[dict[str, object]]
    value_slots_by_id: tuple[dict[str, object], ...]
    value_slot_ids_by_current_variant: dict[tuple[int, str], int]
    momentum_slot_by_mask: dict[int, int]
    stages: tuple[RuntimeStageLayout, ...]
    amplitude_stage: dict[str, object]
    color_contraction: ColorContractionPlan | None
    color_contraction_descriptors: tuple[ColorGroupDescriptor, ...]
    model_parameters: list[dict[str, object]]


@dataclass(frozen=True, slots=True)
class _SourceDescriptor:
    source_ir: Any
    applied_crossing: Any
    expected_states: frozenset[tuple[int, int, object]]
    crossing: str
    source_ir_payload: dict[str, object]
    applied_crossing_payload: dict[str, object]


@dataclass(frozen=True, slots=True)
class _ValueSlotLayout:
    records: list[dict[str, object]]
    by_current_variant: dict[tuple[int, str], dict[str, object]]
    slot_ids_by_current_variant: dict[tuple[int, str], int]
    input_slot_id_by_current: dict[int, int]
    result_slot_ids_by_current: dict[int, tuple[int, ...]]


def build_runtime_expression_schema(
    dag: GenericDAG,
    model: Model,
    *,
    process_id: str | None = None,
    color_contraction: str = "direct",
) -> RuntimeExpressionSchema:
    return RuntimeExpressionSchema.from_mapping(
        build_runtime_schema(
            dag,
            model,
            process_id=process_id,
            color_contraction=color_contraction,
        )
    )


def build_runtime_schema(
    dag: GenericDAG,
    model: Model,
    *,
    process_id: str | None = None,
    color_contraction: str = "direct",
) -> dict[str, object]:
    """Build the stage compiler and runtime metadata for one concrete process."""

    return build_runtime_schema_layout(
        dag,
        model,
        process_id=process_id,
        color_contraction=color_contraction,
    ).runtime_schema


def build_runtime_schema_layout(
    dag: GenericDAG,
    model: Model,
    *,
    process_id: str | None = None,
    color_contraction: str = "direct",
) -> RuntimeSchemaLayout:
    """Build runtime metadata while retaining its typed construction indexes."""

    public_process_id = process_id or dag.process.key

    color_state_payloads: dict[object, dict[str, object]] = {}
    current_slots = _current_slots(
        dag,
        color_state_payloads=color_state_payloads,
    )
    interactions_by_size, interaction_inputs = _interaction_indexes(dag)
    amplitude_inputs = {
        current_id
        for root in dag.amplitude_roots
        for current_id in (root.left_id, root.right_id)
    }
    propagators_by_current = _propagators_by_current(dag, model)
    propagator_payloads: dict[tuple[int, int], dict[str, object]] = {}
    value_slot_layout = _value_slots(
        dag,
        current_slots=current_slots,
        interaction_inputs=interaction_inputs,
        amplitude_inputs=amplitude_inputs,
        propagators_by_current=propagators_by_current,
        propagator_payloads=propagator_payloads,
    )
    value_slot_records = value_slot_layout.records
    value_slots = value_slot_layout.by_current_variant
    momentum_slots = _momentum_slots(dag)
    momentum_slot_by_mask = {
        int(slot["momentum_mask"]): int(slot["momentum_slot_id"])
        for slot in momentum_slots
    }
    amplitude_stage_layout = build_runtime_amplitude_stage_layout(
        dag,
        model,
        current_slots=current_slots,
        value_slots=value_slots,
        color_contraction=color_contraction,
    )
    amplitude_stage = amplitude_stage_layout.record
    model_parameters = build_runtime_model_parameter_records(
        dag,
        model,
        amplitude_stage=amplitude_stage,
    )
    stage_layouts = _stage_layouts(
        dag,
        input_value_slot_id_by_current=(value_slot_layout.input_slot_id_by_current),
        result_value_slot_ids_by_current=(value_slot_layout.result_slot_ids_by_current),
        momentum_slot_by_mask=momentum_slot_by_mask,
        interactions_by_size=interactions_by_size,
    )
    stages = [stage.record for stage in stage_layouts]
    external_particles = _external_particles(dag)
    normalization = build_runtime_normalization(dag, model)
    physics = build_resolved_physics_payload(
        dag,
        model,
        process_id=public_process_id,
        amplitude_stage=amplitude_stage,
        model_parameters=model_parameters,
        normalization=normalization,
    )
    value_count = (
        int(value_slot_records[-1]["component_stop"]) if value_slot_records else 0
    )
    momentum_count = 4 * len(momentum_slots)
    source_records = _source_records(
        dag,
        model,
        current_slots=current_slots,
        value_slots=value_slots,
        color_state_payloads=color_state_payloads,
    )

    runtime_schema: dict[str, object] = {
        "schema_version": 1,
        "kind": "pyamplicol-runtime-expression-schema",
        "process_key": public_process_id,
        "process": dag.process.process,
        "color_accuracy": dag.process.color_accuracy,
        "external_particles": external_particles,
        "momentum_conventions": _momentum_conventions(dag),
        "model": _model_payload(dag, model),
        "normalization": normalization,
        "parameter_layout": {
            "value_component_count": value_count,
            "momentum_parameter_count": momentum_count,
            "model_parameter_count": len(model_parameters),
            "parameter_count_if_flattened": (
                value_count + momentum_count + len(model_parameters)
            ),
            "momentum_components_real": True,
            "model_parameters_real": True,
        },
        "model_parameters": model_parameters,
        "physics": physics,
        "current_storage": {
            "component_count": (
                int(current_slots[-1]["component_stop"]) if current_slots else 0
            ),
            "number_type": "complex",
            "current_slots": current_slots,
        },
        "value_storage": {
            "component_count": value_count,
            "number_type": "complex",
            "value_slots": value_slot_records,
        },
        "source_fill": {
            "source_count": len(dag.sources),
            "sources": source_records,
        },
        "momentum_slots": momentum_slots,
        "stages": stages,
        "amplitude_stage": amplitude_stage,
    }
    if dag.lc_topology_replay is not None:
        runtime_schema["lc_topology_replay"] = (
            dag.lc_topology_replay.to_runtime_manifest()
        )
    if dag.color_topology_replay is not None:
        runtime_schema["color_topology_replay"] = (
            dag.color_topology_replay.to_runtime_manifest()
        )
    if dag.helicity_recurrence is not None:
        runtime_schema["helicity_recurrence"] = {
            **dag.helicity_recurrence.to_runtime_manifest(),
            **(
                {}
                if dag.helicity_materialization is None
                else {
                    "materialization": (
                        dag.helicity_materialization.to_runtime_manifest()
                    )
                }
            ),
        }
    return RuntimeSchemaLayout(
        runtime_schema=runtime_schema,
        current_slots=current_slots,
        value_slots_by_id=tuple(value_slot_records),
        value_slot_ids_by_current_variant=(
            value_slot_layout.slot_ids_by_current_variant
        ),
        momentum_slot_by_mask=momentum_slot_by_mask,
        stages=stage_layouts,
        amplitude_stage=amplitude_stage,
        color_contraction=amplitude_stage_layout.color_contraction,
        color_contraction_descriptors=(amplitude_stage_layout.contraction_descriptors),
        model_parameters=model_parameters,
    )


def _current_slots(
    dag: GenericDAG,
    *,
    color_state_payloads: dict[object, dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    offset = 0
    slots: list[dict[str, object]] = []
    color_payloads = {} if color_state_payloads is None else color_state_payloads
    for current in dag.currents:
        index = current.index
        start = offset
        offset += current.dimension
        color_state = color_payloads.get(index.color_state)
        if color_state is None:
            color_state = index.color_state.to_json_dict()
            color_payloads[index.color_state] = color_state
        slots.append(
            {
                "current_id": current.id,
                "component_start": start,
                "component_stop": offset,
                "dimension": current.dimension,
                "is_source": current.is_source,
                "particle_id": index.particle_id,
                "external_mask": index.external_mask,
                "external_labels": list(index.external_labels),
                "momentum_mask": index.momentum_mask,
                # Decimal conversion is capped by Python for very wide
                # full-colour helicity masks.  The runtime contract accepts
                # canonical hexadecimal integers and does not interpret this
                # diagnostic-only field on the hot path.
                "helicity_ancestry": hex(index.helicity_ancestry),
                "chirality": index.chirality,
                "spin_state": _spin_state(index.spin_state),
                "flavour_flow": list(index.flavour_flow),
                "quantum_number_flow": [
                    [name, expression] for name, expression in index.quantum_number_flow
                ],
                "color_state": color_state,
                "auxiliary_kind": index.auxiliary_kind,
            }
        )
    return slots


def _value_slots(
    dag: GenericDAG,
    *,
    current_slots: Sequence[Mapping[str, object]],
    interaction_inputs: set[int],
    amplitude_inputs: set[int],
    propagators_by_current: Sequence[Any],
    propagator_payloads: dict[tuple[int, int], dict[str, object]],
) -> _ValueSlotLayout:
    records: list[dict[str, object]] = []
    by_current_variant: dict[tuple[int, str], dict[str, object]] = {}
    slot_ids_by_current_variant: dict[tuple[int, str], int] = {}
    input_slot_id_by_current: dict[int, int] = {}
    result_slot_ids_by_current: dict[int, tuple[int, ...]] = {}
    offset = 0

    def add(
        current: CurrentNode,
        variant: str,
        applies_propagator: bool,
    ) -> int:
        nonlocal offset
        current_slot = current_slots[current.id]
        propagator = propagators_by_current[current.id]
        propagator_key = (
            current.index.particle_id,
            current.index.chirality,
        )
        propagator_payload = propagator_payloads.get(propagator_key)
        if propagator_payload is None:
            propagator_payload = propagator.to_json_dict()
            propagator_payloads[propagator_key] = propagator_payload
        start = offset
        offset += current.dimension
        slot_id = len(records)
        record: dict[str, object] = {
            "value_slot_id": slot_id,
            "current_id": current.id,
            "variant": variant,
            "component_start": start,
            "component_stop": offset,
            "dimension": current.dimension,
            "current_component_start": current_slot["component_start"],
            "current_component_stop": current_slot["component_stop"],
            "is_source": current.is_source,
            "applies_propagator": applies_propagator,
            "particle_id": current.index.particle_id,
            "external_mask": current.index.external_mask,
            "external_labels": current_slot["external_labels"],
            "momentum_mask": current.index.momentum_mask,
            "chirality": current.index.chirality,
            "propagator": propagator_payload,
            "used_as_interaction_input": current.id in interaction_inputs,
            "used_as_amplitude_input": current.id in amplitude_inputs,
        }
        records.append(record)
        key = (current.id, variant)
        by_current_variant[key] = record
        slot_ids_by_current_variant[key] = slot_id
        return slot_id

    for current in dag.currents:
        if current.is_source:
            source_slot_id = add(current, "source", False)
            result_slot_ids_by_current[current.id] = (source_slot_id,)
            if current.id in interaction_inputs:
                input_slot_id_by_current[current.id] = source_slot_id
            continue
        propagator = propagators_by_current[current.id]
        needs_propagated = (
            current.id in interaction_inputs and propagator.applies_propagator
        )
        needs_unpropagated = current.id in amplitude_inputs
        unpropagated_slot_id: int | None = None
        if (
            needs_unpropagated
            or not needs_propagated
            or (current.id in interaction_inputs and not propagator.applies_propagator)
        ):
            unpropagated_slot_id = add(current, "unpropagated", False)
        propagated_slot_id: int | None = None
        if needs_propagated:
            propagated_slot_id = add(current, "propagated", True)
        result_slot_ids = tuple(
            slot_id
            for slot_id in (unpropagated_slot_id, propagated_slot_id)
            if slot_id is not None
        )
        if not result_slot_ids:
            raise ValueError(f"result current {current.id} has no runtime value slots")
        result_slot_ids_by_current[current.id] = result_slot_ids
        if current.id in interaction_inputs:
            input_slot_id = (
                propagated_slot_id
                if propagator.applies_propagator
                else unpropagated_slot_id
            )
            if input_slot_id is None:
                variant = (
                    "propagated" if propagator.applies_propagator else "unpropagated"
                )
                raise ValueError(
                    f"interaction current {current.id} has no {variant} "
                    "runtime value slot"
                )
            input_slot_id_by_current[current.id] = input_slot_id
    return _ValueSlotLayout(
        records=records,
        by_current_variant=by_current_variant,
        slot_ids_by_current_variant=slot_ids_by_current_variant,
        input_slot_id_by_current=input_slot_id_by_current,
        result_slot_ids_by_current=result_slot_ids_by_current,
    )


def _momentum_slots(dag: GenericDAG) -> list[dict[str, object]]:
    masks = sorted(
        {current.index.momentum_mask for current in dag.currents},
        key=lambda mask: (mask.bit_count(), mask),
    )
    incoming = {leg.label for leg in dag.process.initial_legs}
    result: list[dict[str, object]] = []
    for slot_id, mask in enumerate(masks):
        labels = _mask_labels(mask)
        start = 4 * slot_id
        result.append(
            {
                "momentum_slot_id": slot_id,
                "momentum_mask": mask,
                "external_labels": list(labels),
                "component_start": start,
                "component_stop": start + 4,
                "component_order": ["E", "px", "py", "pz"],
                "real_valued": True,
                "crossed_incoming_labels": [
                    label for label in labels if label in incoming
                ],
                "construction": (
                    "sum all-outgoing momenta for external labels; negate "
                    "physical incoming momenta first"
                ),
            }
        )
    return result


def _stage_layouts(
    dag: GenericDAG,
    *,
    input_value_slot_id_by_current: Mapping[int, int],
    result_value_slot_ids_by_current: Mapping[int, tuple[int, ...]],
    momentum_slot_by_mask: Mapping[int, int],
    interactions_by_size: Mapping[int, Sequence[InteractionNode]],
) -> tuple[RuntimeStageLayout, ...]:
    stages: list[RuntimeStageLayout] = []
    for stage_index, size in enumerate(sorted(interactions_by_size), start=1):
        interactions = tuple(interactions_by_size[size])
        input_current_ids: set[int] = set()
        output_current_ids: set[int] = set()
        input_value_slot_ids: set[int] = set()
        output_value_slot_ids: set[int] = set()
        momentum_ids: set[int] = set()
        evaluation_groups: dict[tuple[str, int], list[InteractionNode]] = {}
        for interaction in interactions:
            left_id = interaction.left_id
            right_id = interaction.right_id
            result_id = interaction.result_id
            input_current_ids.add(left_id)
            input_current_ids.add(right_id)
            output_current_ids.add(result_id)
            input_value_slot_ids.add(input_value_slot_id_by_current[left_id])
            input_value_slot_ids.add(input_value_slot_id_by_current[right_id])
            output_value_slot_ids.update(result_value_slot_ids_by_current[result_id])
            momentum_ids.add(
                momentum_slot_by_mask[dag.currents[left_id].index.momentum_mask]
            )
            momentum_ids.add(
                momentum_slot_by_mask[dag.currents[right_id].index.momentum_mask]
            )
            momentum_ids.add(
                momentum_slot_by_mask[dag.currents[result_id].index.momentum_mask]
            )
            group_key = (
                ("group", int(interaction.evaluation_group_id))
                if interaction.evaluation_group_id is not None
                else ("interaction", interaction.id)
            )
            if interaction.evaluation_factor != (0.0, 0.0):
                evaluation_groups.setdefault(group_key, []).append(interaction)

        ordered_output_current_ids = tuple(sorted(output_current_ids))
        record: dict[str, object] = {
            "stage_index": stage_index,
            "stage_kind": "current-combine",
            "subset_size": size,
            "input_current_ids": sorted(input_current_ids),
            "output_current_ids": list(ordered_output_current_ids),
            "input_value_slot_ids": sorted(input_value_slot_ids),
            "output_value_slot_ids": sorted(output_value_slot_ids),
            "input_momentum_slot_ids": sorted(momentum_ids),
            "interaction_count": len(interactions),
            "interaction_evaluation_count": len(evaluation_groups),
            "interaction_ids": [interaction.id for interaction in interactions],
            "interactions_compacted": True,
            "interactions": [],
        }
        stages.append(
            RuntimeStageLayout(
                record=record,
                interactions=interactions,
                evaluation_groups=tuple(
                    tuple(group) for group in evaluation_groups.values()
                ),
                input_value_slot_by_current={
                    current_id: input_value_slot_id_by_current[current_id]
                    for current_id in input_current_ids
                },
                output_current_ids=ordered_output_current_ids,
            )
        )
    return tuple(stages)


def _interaction_indexes(
    dag: GenericDAG,
) -> tuple[dict[int, list[InteractionNode]], set[int]]:
    by_size: dict[int, list[InteractionNode]] = {}
    inputs: set[int] = set()
    for interaction in dag.interactions:
        size = len(dag.currents[interaction.result_id].index.external_labels)
        by_size.setdefault(size, []).append(interaction)
        inputs.add(interaction.left_id)
        inputs.add(interaction.right_id)
    return by_size, inputs


def _propagators_by_current(dag: GenericDAG, model: Model) -> list[Any]:
    cache: dict[tuple[int, int], Any] = {}
    result: list[Any] = []
    for current in dag.currents:
        key = (current.index.particle_id, current.index.chirality)
        propagator = cache.get(key)
        if propagator is None:
            propagator = model._propagator_ir(*key)
            cache[key] = propagator
        result.append(propagator)
    return result


def _source_records(
    dag: GenericDAG,
    model: Model,
    *,
    current_slots: Sequence[Mapping[str, object]],
    value_slots: Mapping[tuple[int, str], Mapping[str, object]],
    color_state_payloads: Mapping[object, dict[str, object]],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    source_start = 0
    legs_by_label = {leg.label: leg for leg in dag.process.legs}
    descriptors: dict[tuple[int, bool], _SourceDescriptor] = {}
    for source_index, current_id in enumerate(dag.sources):
        current = dag.currents[current_id]
        source_leg_label = current.source_leg_label
        leg = (
            legs_by_label.get(source_leg_label)
            if source_leg_label is not None
            else None
        )
        descriptor_key = (
            current.index.particle_id,
            bool(leg is not None and leg.is_initial),
        )
        descriptor = descriptors.get(descriptor_key)
        if descriptor is None:
            descriptor = _source_descriptor(
                model,
                particle_id=current.index.particle_id,
                crossed=descriptor_key[1],
            )
            descriptors[descriptor_key] = descriptor
        records.append(
            _source_record(
                current,
                leg=leg,
                descriptor=descriptor,
                current_id=current_id,
                source_index=source_index,
                source_start=source_start,
                current_slot=current_slots[current_id],
                value_slot=value_slots[(current_id, "source")],
                color_state_payload=color_state_payloads[current.index.color_state],
            )
        )
        source_start += current.dimension
    return records


def _source_descriptor(
    model: Model,
    *,
    particle_id: int,
    crossed: bool,
) -> _SourceDescriptor:
    source_ir = model._source_ir(particle_id)
    applied_crossing = (
        source_ir.crossing if crossed else type(source_ir.crossing).identity()
    )
    expected_states = frozenset(
        (state.helicity, state.chirality, state.spin_state)
        for state in (
            applied_crossing.apply(declared_state)
            for declared_state in source_ir.states
        )
    )
    return _SourceDescriptor(
        source_ir=source_ir,
        applied_crossing=applied_crossing,
        expected_states=expected_states,
        crossing=(
            "negate-incoming-momentum"
            if applied_crossing.momentum_transform == "negate-four-momentum"
            else "identity"
        ),
        source_ir_payload=source_ir.to_json_dict(),
        applied_crossing_payload=applied_crossing.to_json_dict(),
    )


def _source_record(
    current: CurrentNode,
    *,
    leg: Any,
    descriptor: _SourceDescriptor,
    current_id: int,
    source_index: int,
    source_start: int,
    current_slot: Mapping[str, object],
    value_slot: Mapping[str, object],
    color_state_payload: dict[str, object],
) -> dict[str, object]:
    if current.source_helicity is None:
        raise ValueError(f"source current {current_id} has no source helicity")
    current_state = (
        int(current.source_helicity),
        int(current.index.chirality),
        current.index.spin_state,
    )
    if current_state not in descriptor.expected_states:
        raise ValueError(
            f"source current {current_id} state {current_state!r} is not declared "
            f"by model source metadata"
        )
    source_ir = descriptor.source_ir
    return {
        "source_id": source_index,
        "current_id": current_id,
        "current_component_start": current_slot["component_start"],
        "current_component_stop": current_slot["component_stop"],
        "value_slot": _value_slot_ref(value_slot),
        "source_parameter_start": source_start,
        "source_parameter_stop": source_start + current.dimension,
        "leg_label": current.source_leg_label,
        "input_momentum_slot": None if leg is None else leg.label - 1,
        "side": None if leg is None else leg.side,
        "crossing": descriptor.crossing,
        "physical_pdg": None if leg is None else leg.pdg,
        "outgoing_pdg": current.index.particle_id,
        "particle_id": current.index.particle_id,
        "anti_particle_id": source_ir.identity.anti_pdg_label,
        "source_kind": "external-wavefunction",
        "wavefunction_kind": source_ir.wavefunction_family,
        "source_orientation": source_ir.identity.orientation,
        "source_basis": source_ir.basis,
        "source_ir": descriptor.source_ir_payload,
        "applied_crossing": descriptor.applied_crossing_payload,
        "source_helicity": current.source_helicity,
        "chirality": current.index.chirality,
        "spin_state": current_slot["spin_state"],
        "dimension": current.dimension,
        "helicity_ancestry": current_slot["helicity_ancestry"],
        "color_state": color_state_payload,
    }


def _model_payload(dag: GenericDAG, model: Model) -> dict[str, object]:
    particle_ids = {current.index.particle_id for current in dag.currents}
    vertex_kinds = {interaction.vertex_kind for interaction in dag.interactions}
    vertex_kinds.update(
        root.vertex_kind for root in dag.amplitude_roots if root.vertex_kind is not None
    )
    return {
        "name": model.name,
        "particles": [
            {
                "pdg": particle.pdg,
                "anti_pdg": particle.anti_pdg,
                "spin": particle.spin,
                "dimension": particle.dimension,
                "color_rep": particle.color_rep,
                "mass": particle.mass,
                "width": particle.width,
                "mass_parameter": _runtime_particle_parameter_name(
                    model,
                    particle.pdg,
                    kind="mass",
                ),
                "width_parameter": _runtime_particle_parameter_name(
                    model,
                    particle.pdg,
                    kind="width",
                ),
                "charge": particle.charge,
            }
            for particle in sorted(model.particles.values(), key=lambda item: item.pdg)
            if particle.pdg in particle_ids or particle.anti_pdg in particle_ids
        ],
        "vertices": [
            {
                "kind": vertex.kind,
                "particles": list(vertex.particles),
                "coupling": list(vertex.coupling),
                "lowering": model.vertex_lowering_rule(vertex.kind).to_json_dict(),
            }
            for vertex in model.vertices
            if vertex.kind in vertex_kinds
        ],
    }


def _runtime_particle_parameter_name(
    model: Model,
    pdg: int,
    *,
    kind: str,
) -> str | None:
    provider = getattr(model, f"runtime_{kind}_parameter_name", None)
    if not callable(provider):
        return None
    name = provider(int(pdg))
    return None if name is None else str(name)


def _external_particles(dag: GenericDAG) -> list[dict[str, object]]:
    return [
        {
            "label": leg.label,
            "index": leg.label - 1,
            "side": leg.side,
            "role": "initial" if leg.is_initial else "final",
            "particle": leg.particle,
            "outgoing_particle": leg.outgoing_particle,
            "pdg": leg.pdg,
            "outgoing_pdg": leg.outgoing_pdg,
            "statistics": leg.statistics,
            "wavefunction_family": leg.wavefunction_family,
            "color_role": leg.color_role,
            "source_orientation": leg.source_orientation,
            "momentum_slot": leg.label - 1,
            "momentum_components": ["E", "px", "py", "pz"],
        }
        for leg in dag.process.legs
    ]


def _momentum_conventions(dag: GenericDAG) -> dict[str, object]:
    incoming = [leg.label for leg in dag.process.initial_legs]
    return {
        "input_shape": ["batch", len(dag.process.legs), 4],
        "component_order": ["E", "px", "py", "pz"],
        "input_momenta": "physical external four-momenta in process order",
        "incoming_labels": incoming,
        "final_state_labels": [leg.label for leg in dag.process.final_legs],
        "all_outgoing_convention": {
            "crossed_incoming_labels": incoming,
            "operation": "negate incoming four-vectors before current/source use",
        },
        "metric": "mostly-minus",
    }


def _input_value_slot(
    current: CurrentNode,
    model: Model,
    value_slots: Mapping[tuple[int, str], Mapping[str, object]],
) -> Mapping[str, object]:
    if current.is_source:
        variant = "source"
    else:
        propagator = model._propagator_ir(
            current.index.particle_id,
            current.index.chirality,
        )
        variant = "propagated" if propagator.applies_propagator else "unpropagated"
    try:
        return value_slots[(current.id, variant)]
    except KeyError as exc:
        raise ValueError(
            f"interaction current {current.id} has no {variant} runtime value slot"
        ) from exc


def _result_value_slots(
    current: CurrentNode,
    value_slots: Mapping[tuple[int, str], Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    result = tuple(
        value_slots[(current.id, variant)]
        for variant in ("unpropagated", "propagated")
        if (current.id, variant) in value_slots
    )
    if result:
        return result
    if current.is_source:
        return (value_slots[(current.id, "source")],)
    raise ValueError(f"result current {current.id} has no runtime value slots")


def _value_slot_ref(slot: Mapping[str, object]) -> dict[str, object]:
    return {
        key: slot[key]
        for key in (
            "value_slot_id",
            "current_id",
            "variant",
            "component_start",
            "component_stop",
            "dimension",
        )
    }


def _spin_state(value: object) -> object:
    return list(value) if isinstance(value, tuple) else value


def _mask_labels(mask: int) -> tuple[int, ...]:
    return tuple(index + 1 for index in range(mask.bit_length()) if mask & (1 << index))


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("runtime schema value must be an object")
    return value


def _sequence(value: object) -> Sequence[object]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise TypeError("runtime schema value must be an array")
    return value


def _mapping_sequence(value: object) -> tuple[Mapping[str, object], ...]:
    return tuple(_mapping(item) for item in _sequence(value))


__all__ = [
    "RUNTIME_PHYSICS_SCHEMA_VERSION",
    "RuntimeSchemaLayout",
    "build_runtime_expression_schema",
    "build_runtime_schema",
    "build_runtime_schema_layout",
]

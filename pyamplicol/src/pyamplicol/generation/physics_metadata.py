# SPDX-License-Identifier: 0BSD
"""Strict runtime-physics-v1 metadata derived from one compiled DAG."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import product
from typing import Any

from ..models.base import Model
from .contracts import runtime_coupling_parameter_names
from .dag_types import GenericDAG
from .helicity_replay import (
    HELICITY_RECURRENCE_CONTRACT_VERSION,
    HELICITY_RECURRENCE_PROOF_ALGORITHM,
    RUNTIME_SELECTOR_PROVENANCE,
)
from .runtime_amplitudes import build_runtime_amplitude_metadata

_HELICITY_WEIGHT_TOLERANCE = 1.0e-12
NATIVE_REDUCTION_GROUPS_EXTENSION_KEY = "native_reduction_groups"
EAGER_PLAN_V3_REDUCTION_GROUPS_KIND = "pyamplicol-eager-plan-v3-reduction-groups"
EAGER_PLAN_V3_REDUCTION_GROUPS_SCHEMA_VERSION = 1
EAGER_PLAN_V3_REDUCTION_GROUPS_STORAGE_ABI = "pacbin-v1"
EAGER_PLAN_V3_REDUCTION_GROUPS_RUNTIME_LAYOUT_ABI = "pyamplicol-eager-runtime-layout-v1"
EAGER_PLAN_V3_REDUCTION_GROUPS_CONTAINER_PATH = "eager-runtime.pacbin"
EAGER_PLAN_V3_REDUCTION_GROUPS_MEMBER = "reductions/groups.bin"
EAGER_PLAN_V3_REDUCTION_ENTRIES_MEMBER = "reductions/entries.bin"
_NORMALIZATION_EXTENSION_KEYS = (
    "color_accuracy",
    "color_factor",
    "average_factor",
    "identical_factor",
    "global_coupling_factor",
    "qcd_coupling_power",
    "electroweak_coupling_power",
    "couplings_in_stage_evaluators",
    "coupling_policy",
)


def build_resolved_physics_from_dag(
    dag: GenericDAG,
    model: Model,
    *,
    process_id: str | None = None,
    native_eager_plan_v3_reduction_groups: bool = False,
    runtime_model_parameters: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Build strict public physics without constructing runtime storage layouts."""

    amplitude_metadata = build_runtime_amplitude_metadata(dag, model)
    model_parameters = (
        build_runtime_model_parameter_records(
            dag,
            model,
            amplitude_stage=amplitude_metadata,
        )
        if runtime_model_parameters is None
        else list(runtime_model_parameters)
    )
    return build_resolved_physics_payload(
        dag,
        model,
        process_id=process_id or dag.process.key,
        amplitude_stage=amplitude_metadata,
        model_parameters=model_parameters,
        normalization=build_runtime_normalization(dag, model),
        native_eager_plan_v3_reduction_groups=(native_eager_plan_v3_reduction_groups),
    )


def build_resolved_physics_payload(
    dag: GenericDAG,
    model: Model,
    *,
    process_id: str,
    amplitude_stage: Mapping[str, object],
    model_parameters: Sequence[Mapping[str, object]],
    normalization: Mapping[str, object],
    native_eager_plan_v3_reduction_groups: bool = False,
) -> dict[str, object]:
    """Build strict public axes and exact representative-reuse metadata."""

    coherent_groups = _mapping_sequence(amplitude_stage.get("coherent_groups", ()))
    helicities, group_helicities, structural_zero_count = _helicity_metadata(
        dag,
        model,
        coherent_groups,
        _mapping_sequence(amplitude_stage.get("roots", ())),
    )
    color_components, group_colors = _color_metadata(dag, coherent_groups)
    if native_eager_plan_v3_reduction_groups:
        reduction_groups: list[dict[str, object]] = []
        native_reduction_extension = {
            NATIVE_REDUCTION_GROUPS_EXTENSION_KEY: {
                "kind": EAGER_PLAN_V3_REDUCTION_GROUPS_KIND,
                "schema_version": EAGER_PLAN_V3_REDUCTION_GROUPS_SCHEMA_VERSION,
                "storage_abi": EAGER_PLAN_V3_REDUCTION_GROUPS_STORAGE_ABI,
                "runtime_layout_abi": (
                    EAGER_PLAN_V3_REDUCTION_GROUPS_RUNTIME_LAYOUT_ABI
                ),
                "container_path": EAGER_PLAN_V3_REDUCTION_GROUPS_CONTAINER_PATH,
                "group_member": EAGER_PLAN_V3_REDUCTION_GROUPS_MEMBER,
                "entry_member": EAGER_PLAN_V3_REDUCTION_ENTRIES_MEMBER,
                "group_count": len(coherent_groups),
            }
        }
    else:
        helicity_ids = {
            tuple(_integer(value) for value in _sequence(record["values"])): str(
                record["id"]
            )
            for record in helicities
        }
        reduction_groups = _reduction_groups(
            coherent_groups,
            group_helicities=group_helicities,
            group_colors=group_colors,
            helicity_ids=helicity_ids,
        )
        native_reduction_extension = {}
    lc_color_complete = (
        dag.color_coverage == "complete"
        and not dag.selected_color_sector_ids
        and not dag.color_plan.truncated
        and len(color_components) == _expected_lc_physical_color_count(dag)
    )
    is_lc = dag.process.color_accuracy == "lc"
    return {
        "schema_version": 1,
        "kind": "pyamplicol-resolved-physics",
        "process_id": process_id,
        "process": dag.process.process,
        "color_accuracy": dag.process.color_accuracy,
        "coverage": {
            "helicities": dag.helicity_coverage,
            "color": ("complete" if is_lc and lc_color_complete else "selected")
            if is_lc
            else "contracted",
            "color_kind": "physical-lc-flows" if is_lc else "contracted-color",
            "structural_zero_helicity_count": structural_zero_count,
        },
        "external_particles": _external_particles(dag),
        "helicities": helicities,
        "color_components": color_components,
        "reduction": {
            "kind": "lc-diagonal" if is_lc else "contracted-color",
            "groups": reduction_groups,
        },
        "model_parameters": _public_model_parameters(model_parameters),
        "selectors": {
            "helicity": True,
            "color_flow": is_lc,
            "contracted_color": False,
        },
        "extensions": {
            "process_key": dag.process.key,
            "normalization": {
                key: normalization[key]
                for key in _NORMALIZATION_EXTENSION_KEYS
                if key in normalization
            },
            "selected_source_helicities": {
                str(label): helicity
                for label, helicity in dag.selected_source_helicities
            },
            **(
                {}
                if dag.lc_topology_replay is None
                else {"lc_topology_replay": dag.lc_topology_replay.to_json_dict()}
            ),
            **(
                {}
                if dag.color_topology_replay is None
                else {
                    "color_topology_replay": (dag.color_topology_replay.to_json_dict())
                }
            ),
            **_runtime_selector_extension(dag),
            **native_reduction_extension,
        },
    }


def build_runtime_model_parameter_records(
    dag: GenericDAG,
    model: Model,
    *,
    amplitude_stage: Mapping[str, object],
) -> list[dict[str, object]]:
    """Build the shared runtime parameter catalog used by execution and physics."""

    records: list[dict[str, object]] = []
    seen: set[str] = set()

    def add(name: str, kind: str, default: float, **metadata: object) -> None:
        if name in seen:
            return
        seen.add(name)
        records.append(
            {
                "name": name,
                "kind": kind,
                "parameter_index": len(records),
                "default": float(default),
                **metadata,
            }
        )

    def add_complex(
        name: str,
        value: object,
        *,
        kind: str,
        **metadata: object,
    ) -> None:
        real, imaginary = _complex_pair(value, name)
        for component, default in (("real", real), ("imag", imaginary)):
            add(
                f"{name}.{component}",
                kind,
                default,
                runtime_name=name,
                complex_component=component,
                **metadata,
            )

    for raw_name, value in sorted(
        model.runtime_normalization_parameter_defaults().items()
    ):
        add(str(raw_name), "normalization", float(value))
    defaults_provider = getattr(model, "runtime_parameter_defaults", None)
    if callable(defaults_provider):
        type_provider = getattr(model, "runtime_parameter_type", None)
        for raw_name, value in sorted(defaults_provider().items()):
            name = str(raw_name)
            declared = (
                str(type_provider(name)).lower()
                if callable(type_provider)
                else "complex"
            )
            if declared == "complex":
                add_complex(name, value, kind="external_parameter_component")
            else:
                real, imaginary = _complex_pair(value, name)
                if imaginary != 0.0:
                    raise ValueError(
                        "real runtime model parameter "
                        f"{name!r} has an imaginary default"
                    )
                add(name, "external_parameter", real, parameter_type=declared)
        _add_derived_parameter_records(
            dag,
            model,
            amplitude_stage=amplitude_stage,
            add_complex=add_complex,
        )
        return records

    for particle in sorted(model.particles.values(), key=lambda item: item.pdg):
        if float(particle.mass) != 0.0:
            add(
                f"particle.{particle.pdg}.mass",
                "particle_mass",
                float(particle.mass),
                pdg=particle.pdg,
            )
        if float(particle.width) != 0.0:
            add(
                f"particle.{particle.pdg}.width",
                "particle_width",
                float(particle.width),
                pdg=particle.pdg,
            )
    for kind, particles, coupling in _coupling_signatures(dag, amplitude_stage):
        names = runtime_coupling_parameter_names(
            kind,
            particles,
            coupling,
            model=model,
        )
        for component, name in enumerate(names):
            if name is None:
                continue
            add(
                name,
                "coupling_component",
                float(coupling[component]),
                vertex_kind=kind,
                vertex_particles=list(particles),
                component=component,
            )
    return records


def build_runtime_normalization(
    dag: GenericDAG,
    model: Model,
) -> dict[str, object]:
    """Build the shared runtime normalization payload for one DAG."""

    return dict(model.runtime_normalization_payload(dag))


def _add_derived_parameter_records(
    dag: GenericDAG,
    model: Model,
    *,
    amplitude_stage: Mapping[str, object],
    add_complex: Any,
) -> None:
    defaults_provider = getattr(model, "runtime_derived_parameter_defaults_for", None)
    if not callable(defaults_provider):
        return
    used = _used_coupling_parameter_names(dag, model, amplitude_stage)
    values = defaults_provider(tuple(sorted(used)))
    domains_provider = getattr(model, "runtime_derived_parameter_domains_for", None)
    domains = (
        domains_provider(tuple(sorted(used))) if callable(domains_provider) else {}
    )
    for raw_name, value in sorted(values.items()):
        name = str(raw_name)
        domain = str(domains.get(raw_name, domains.get(name, "complex")))
        if domain not in {"real", "imaginary", "complex"}:
            raise ValueError(f"unsupported runtime parameter domain {domain!r}")
        add_complex(
            name,
            value,
            kind="derived_parameter_component",
            derived=True,
            complex_domain=domain,
        )


def _used_coupling_parameter_names(
    dag: GenericDAG,
    model: Model,
    amplitude_stage: Mapping[str, object],
) -> set[str]:
    names = {
        name
        for interaction in dag.interactions
        for name in runtime_coupling_parameter_names(
            interaction.vertex_kind,
            interaction.vertex_particles,
            interaction.coupling,
            model=model,
        )
        if name is not None
    }
    for root in _mapping_sequence(amplitude_stage["roots"]):
        raw_names = root.get("coupling_parameter_names")
        if isinstance(raw_names, list):
            names.update(str(name) for name in raw_names if isinstance(name, str))
    runtime_particle_names = getattr(
        model,
        "runtime_parameter_names_for_particle",
        None,
    )
    if callable(runtime_particle_names):
        for particle_id in {current.index.particle_id for current in dag.currents}:
            names.update(str(name) for name in runtime_particle_names(int(particle_id)))
    return names


def _coupling_signatures(
    dag: GenericDAG,
    amplitude_stage: Mapping[str, object],
) -> tuple[tuple[int, tuple[int, ...], tuple[float, ...]], ...]:
    signatures = {
        (
            interaction.vertex_kind,
            tuple(interaction.vertex_particles),
            tuple(interaction.coupling),
        )
        for interaction in dag.interactions
    }
    for root in _mapping_sequence(amplitude_stage["roots"]):
        kind = root.get("vertex_kind")
        particles = root.get("vertex_particles")
        coupling = root.get("coupling")
        if (
            isinstance(kind, int)
            and isinstance(particles, list)
            and isinstance(coupling, list)
        ):
            signatures.add(
                (
                    kind,
                    tuple(int(pdg) for pdg in particles),
                    tuple(float(value) for value in coupling),
                )
            )
    return tuple(sorted(signatures))


def _runtime_selector_extension(dag: GenericDAG) -> dict[str, object]:
    """Emit independent runtime contracts for both selector axes."""

    recurrence = dag.helicity_recurrence
    helicity_complete = (
        dag.helicity_coverage == "complete" and not dag.selected_source_helicities
    )
    recurrence_payload: dict[str, object] | None
    if not helicity_complete:
        recurrence_payload = None
    elif recurrence is None:
        recurrence_payload = {
            "contract_version": HELICITY_RECURRENCE_CONTRACT_VERSION,
            "proof_algorithm": HELICITY_RECURRENCE_PROOF_ALGORITHM,
            "status": "unavailable",
            "proof_counts": {},
        }
    else:
        recurrence_payload = {
            "contract_version": HELICITY_RECURRENCE_CONTRACT_VERSION,
            "proof_algorithm": HELICITY_RECURRENCE_PROOF_ALGORITHM,
            "status": "available",
            "proof_counts": recurrence.proof_counts(),
            **(
                {}
                if dag.helicity_materialization is None
                else {
                    "execution": (
                        "materialized-recurrence-retained-proof-graph"
                        if dag.helicity_materialization.strategy
                        == "retained-proof-graph"
                        else "materialized-recurrence-quotient"
                    ),
                    "materialization_strategy": (dag.helicity_materialization.strategy),
                    "proof_current_count": (
                        dag.helicity_materialization.proof_current_count
                    ),
                    "proof_amplitude_count": (
                        dag.helicity_materialization.proof_root_count
                    ),
                    "materialized_current_count": (
                        dag.helicity_materialization.materialized_current_count
                    ),
                    "materialized_amplitude_count": (
                        dag.helicity_materialization.materialized_root_count
                    ),
                }
            ),
        }
    is_lc = dag.process.color_accuracy == "lc"
    color_complete = (
        is_lc and dag.color_coverage == "complete" and not dag.selected_color_sector_ids
    )
    specialized_axes = [
        *([] if helicity_complete else ["helicity"]),
        *([] if not is_lc or color_complete else ["color_flow"]),
    ]
    selector_payload: dict[str, object] = {
        "kind": "pyamplicol-runtime-selectors",
        "contract_version": 1,
        "provenance": RUNTIME_SELECTOR_PROVENANCE,
        "axes": {
            "helicity": {
                "generation_coverage": dag.helicity_coverage,
                "generation_selection": {
                    str(label): int(helicity)
                    for label, helicity in dag.selected_source_helicities
                },
                "runtime_contract": (
                    "complete-reusable"
                    if helicity_complete
                    else "generation-specialized"
                ),
            },
            "color_flow": {
                "generation_coverage": (dag.color_coverage if is_lc else "contracted"),
                "generation_selection": list(dag.selected_color_sector_ids),
                "runtime_contract": (
                    "complete-reusable"
                    if color_complete
                    else "generation-specialized"
                    if is_lc
                    else "contracted-color"
                ),
            },
        },
        "generation_specialized_axes": specialized_axes,
    }
    if recurrence_payload is not None:
        selector_payload["helicity_recurrence"] = recurrence_payload
    return {"runtime_selectors": selector_payload}


def _helicity_metadata(
    dag: GenericDAG,
    model: Model,
    groups: Sequence[Mapping[str, object]],
    roots: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], dict[int, tuple[tuple[int, ...], ...]], int]:
    if dag.helicity_materialization is not None:
        return _materialized_helicity_metadata(dag, model, groups, roots)

    represented: dict[tuple[int, ...], tuple[tuple[int, ...], bool]] = {}
    group_members: dict[int, tuple[tuple[int, ...], ...]] = {}
    for group in groups:
        group_id = _integer(group["group_id"])
        vector = tuple(_integer(value) for value in _sequence(group["helicities"]))
        weight = _number(group.get("helicity_weight", 1.0))
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"invalid helicity weight {weight!r} in group {group_id}")
        members = [vector]
        if weight > 1.0 + _HELICITY_WEIGHT_TOLERANCE:
            if not math.isclose(
                weight,
                2.0,
                rel_tol=_HELICITY_WEIGHT_TOLERANCE,
                abs_tol=_HELICITY_WEIGHT_TOLERANCE,
            ):
                raise ValueError(
                    f"unsupported helicity reuse weight {weight!r} in group {group_id}"
                )
            flipped = tuple(-value for value in vector)
            if flipped != vector:
                members.append(flipped)
        group_members[group_id] = tuple(members)
        for index, member in enumerate(members):
            candidate = (vector, index == 0)
            previous = represented.setdefault(member, candidate)
            if previous != candidate:
                raise ValueError(
                    f"inconsistent helicity representative for physical state {member}"
                )

    possible = _possible_helicity_vectors(dag, model)
    structural_zeros = sorted(set(possible) - represented.keys())
    for vector in structural_zeros:
        represented[vector] = (vector, False)

    identifiers = {vector: _helicity_id(vector) for vector in represented}
    records: list[dict[str, object]] = []
    for index, vector in enumerate(sorted(represented)):
        representative, computed = represented[vector]
        structural_zero = vector in structural_zeros
        records.append(
            {
                "id": identifiers[vector],
                "index": index,
                "values": list(vector),
                "computed": computed and not structural_zero,
                "structural_zero": structural_zero,
                "representative_id": identifiers[representative],
                "coefficient": 0.0 if structural_zero else 1.0,
            }
        )
    return records, group_members, len(structural_zeros)


def _materialized_helicity_metadata(
    dag: GenericDAG,
    model: Model,
    groups: Sequence[Mapping[str, object]],
    roots: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], dict[int, tuple[tuple[int, ...], ...]], int]:
    materialization = dag.helicity_materialization
    recurrence = dag.helicity_recurrence
    if materialization is None or recurrence is None:
        raise ValueError("materialized helicity metadata requires its proof")

    group_by_root_id = {
        _integer(root["dag_root_id"]): _integer(root["coherent_group_id"])
        for root in roots
    }
    domain_by_id = {domain.id: domain for domain in recurrence.selector_domains}
    route_vectors_by_group: dict[int, set[tuple[int, ...]]] = {
        _integer(group["group_id"]): set() for group in groups
    }
    route_coefficients: dict[tuple[int, ...], float] = {}
    for route in materialization.amplitude_routes:
        group_id = group_by_root_id.get(route.materialized_root_id)
        if group_id is None:
            raise ValueError(
                "helicity route refers to an absent materialized amplitude root"
            )
        coefficient = route.factor[0] ** 2 + route.factor[1] ** 2
        for domain_id in route.selector_domain_ids:
            domain = domain_by_id.get(domain_id)
            if domain is None or not domain.complete:
                raise ValueError(
                    "amplitude route refers to a non-complete selector domain"
                )
            by_label = dict(domain.source_states)
            vector = tuple(by_label.get(int(leg.label), 0) for leg in dag.process.legs)
            route_vectors_by_group[group_id].add(vector)
            previous = route_coefficients.setdefault(vector, coefficient)
            if not math.isclose(
                previous,
                coefficient,
                rel_tol=_HELICITY_WEIGHT_TOLERANCE,
                abs_tol=_HELICITY_WEIGHT_TOLERANCE,
            ):
                raise ValueError(
                    f"inconsistent recurrence factor for physical helicity {vector}"
                )

    represented: dict[tuple[int, ...], tuple[tuple[int, ...], bool, float]] = {}
    group_members: dict[int, tuple[tuple[int, ...], ...]] = {}
    for group in groups:
        group_id = _integer(group["group_id"])
        representative = tuple(
            _integer(value) for value in _sequence(group["helicities"])
        )
        members = tuple(sorted(route_vectors_by_group[group_id]))
        if not members:
            raise ValueError(
                f"materialized helicity group {group_id} has no physical routes"
            )
        group_members[group_id] = members
        for member in members:
            candidate = (
                representative,
                member == representative,
                route_coefficients[member],
            )
            previous = represented.setdefault(member, candidate)
            if previous != candidate:
                raise ValueError(
                    f"inconsistent helicity representative for physical state {member}"
                )

    possible = _possible_helicity_vectors(dag, model)
    structural_zeros = sorted(set(possible) - represented.keys())
    expected_zero_domains = {
        tuple(
            dict(domain_by_id[domain_id].source_states).get(int(leg.label), 0)
            for leg in dag.process.legs
        )
        for domain_id in recurrence.structural_zero_selector_domain_ids
    }
    if set(structural_zeros) != expected_zero_domains:
        raise ValueError(
            "materialized helicity routes do not match certified structural zeros"
        )
    for vector in structural_zeros:
        represented[vector] = (vector, False, 0.0)

    identifiers = {vector: _helicity_id(vector) for vector in represented}
    records: list[dict[str, object]] = []
    for index, vector in enumerate(sorted(represented)):
        representative, computed, coefficient = represented[vector]
        structural_zero = vector in structural_zeros
        records.append(
            {
                "id": identifiers[vector],
                "index": index,
                "values": list(vector),
                "computed": computed and not structural_zero,
                "structural_zero": structural_zero,
                "representative_id": identifiers[representative],
                "coefficient": 0.0 if structural_zero else coefficient,
            }
        )
    return records, group_members, len(structural_zeros)


def _possible_helicity_vectors(
    dag: GenericDAG,
    model: Model,
) -> tuple[tuple[int, ...], ...]:
    selected = dict(dag.selected_source_helicities)
    per_leg: list[tuple[int, ...]] = []
    for leg in dag.process.legs:
        if leg.outgoing_pdg is None:
            per_leg.append((0,))
            continue
        particle_id = int(leg.outgoing_pdg)
        source_ir = model._source_ir(particle_id)
        values: set[int] = set()
        for declared_state in source_ir.states:
            state = (
                source_ir.crossing.apply(declared_state)
                if leg.is_initial
                else declared_state
            )
            values.add(int(state.helicity))
        if leg.label in selected:
            requested = selected[leg.label]
            values.intersection_update((requested,))
            if not values:
                raise ValueError(
                    f"selected helicity {requested} is unavailable for external "
                    f"leg {leg.label}"
                )
        per_leg.append(tuple(sorted(values)))
    return tuple(tuple(values) for values in product(*per_leg))


def _color_metadata(
    dag: GenericDAG,
    groups: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], dict[int, tuple[str, ...]]]:
    if dag.process.color_accuracy != "lc":
        identifier = "color:contracted"
        return [
            {
                "kind": "contracted-color",
                "id": identifier,
                "index": 0,
                "description": "coherent sparse contraction of the full color basis",
            }
        ], {_integer(group["group_id"]): (identifier,) for group in groups}

    records: dict[str, dict[str, object]] = {}
    members_by_group: dict[int, tuple[str, ...]] = {}
    for group in groups:
        group_id = _integer(group["group_id"])
        word = tuple(_integer(value) for value in _sequence(group["color_word"]))
        identifier = _color_id(word)
        members = [identifier]
        records.setdefault(
            identifier,
            {
                "kind": "lc-flow",
                "id": identifier,
                "word": list(word),
                "computed": True,
                "representative_id": identifier,
                "coefficient": 1.0,
            },
        )
        helicity_weight = _number(group.get("helicity_weight", 1.0))
        all_sector_weight = _number(group.get("all_sector_weight", helicity_weight))
        if word and all_sector_weight > helicity_weight + _HELICITY_WEIGHT_TOLERANCE:
            reflected = (word[0], *reversed(word[1:]))
            reflected_id = _color_id(reflected)
            if reflected_id != identifier:
                members.append(reflected_id)
                records.setdefault(
                    reflected_id,
                    {
                        "kind": "lc-flow",
                        "id": reflected_id,
                        "word": list(reflected),
                        "computed": False,
                        "representative_id": identifier,
                        "coefficient": 1.0,
                    },
                )
        members_by_group[group_id] = tuple(members)
    if dag.lc_topology_replay is not None:
        _add_replayed_lc_color_components(dag, records)
    components = [records[key] for key in sorted(records)]
    for index, component in enumerate(components):
        component["index"] = index
    return components, members_by_group


def _add_replayed_lc_color_components(
    dag: GenericDAG,
    records: dict[str, dict[str, object]],
) -> None:
    replay = dag.lc_topology_replay
    if replay is None:
        return
    materialized = set(replay.materialized_sector_ids)
    for sector in dag.color_plan.sectors:
        word = tuple(sector.word_labels or sector.color_words[0])
        identifier = _color_id(word)
        representative = dag.color_plan.sector(replay.representative_for(sector.id))
        if representative is None:
            raise ValueError(
                f"LC replay representative for sector {sector.id} is missing"
            )
        representative_word = tuple(
            representative.word_labels or representative.color_words[0]
        )
        representative_id = _color_id(representative_word)
        records.setdefault(
            identifier,
            {
                "kind": "lc-flow",
                "id": identifier,
                "word": list(word),
                "computed": int(sector.id) in materialized,
                "representative_id": representative_id,
                "coefficient": 1.0,
            },
        )
        if not (
            dag.color_plan.trace_reflections_folded
            and sector.kind == "single-trace"
            and len(sector.trace_labels) > 2
        ):
            continue
        reflected = (word[0], *reversed(word[1:]))
        reflected_id = _color_id(reflected)
        if reflected_id == identifier:
            continue
        records.setdefault(
            reflected_id,
            {
                "kind": "lc-flow",
                "id": reflected_id,
                "word": list(reflected),
                "computed": False,
                "representative_id": representative_id,
                "coefficient": 1.0,
            },
        )


def _reduction_groups(
    groups: Sequence[Mapping[str, object]],
    *,
    group_helicities: Mapping[int, tuple[tuple[int, ...], ...]],
    group_colors: Mapping[int, tuple[str, ...]],
    helicity_ids: Mapping[tuple[int, ...], str],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for group in groups:
        group_id = _integer(group["group_id"])
        helicities = group_helicities[group_id]
        colors = group_colors[group_id]
        representative = tuple(
            _integer(value) for value in _sequence(group["helicities"])
        )
        if representative not in helicities:
            raise ValueError(
                f"helicity representative is not a member of group {group_id}"
            )
        records.append(
            {
                "id": f"reduction:{group_id}",
                "representative_helicity_id": helicity_ids[representative],
                "representative_color_id": colors[0],
                "physical_helicity_ids": [
                    helicity_ids[helicity] for helicity in helicities
                ],
                "physical_color_ids": list(colors),
            }
        )
    return records


def _public_model_parameters(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    parameters: dict[str, dict[str, object]] = {}
    for record in records:
        raw_kind = str(record.get("kind", ""))
        if raw_kind == "runtime_control":
            continue
        name = str(record.get("runtime_name", record.get("name", "")))
        if not name:
            raise ValueError("runtime model parameter has no name")
        kind = _public_parameter_kind(raw_kind)
        parameter = parameters.setdefault(
            name,
            {
                "name": name,
                "kind": kind,
                "default_real": 0.0,
                "default_imaginary": 0.0,
                "mutable": kind != "derived",
            },
        )
        if parameter["kind"] != kind:
            raise ValueError(f"runtime parameter {name!r} has inconsistent kinds")
        component = str(record.get("complex_component", "real"))
        field = "default_imaginary" if component == "imag" else "default_real"
        parameter[field] = _number(record.get("default", 0.0))
    return list(parameters.values())


def _public_parameter_kind(kind: str) -> str:
    if kind == "normalization":
        return "normalization"
    if kind == "particle_mass":
        return "mass"
    if kind == "particle_width":
        return "width"
    if kind == "coupling_component":
        return "coupling"
    if kind in {"external_parameter", "external_parameter_component"}:
        return "external"
    if kind == "derived_parameter_component":
        return "derived"
    raise ValueError(f"unsupported public model-parameter kind {kind!r}")


def _external_particles(dag: GenericDAG) -> list[dict[str, object]]:
    return [
        {
            "index": index,
            "label": leg.label,
            "particle": leg.particle,
            "pdg": leg.pdg,
            "role": "initial" if leg.is_initial else "final",
            "momentum_slot": index,
            "momentum_components": ["E", "px", "py", "pz"],
        }
        for index, leg in enumerate(dag.process.legs)
    ]


def _expected_lc_physical_color_count(dag: GenericDAG) -> int:
    if dag.process.color_accuracy != "lc":
        return 0
    count = 0
    for sector in dag.color_plan.sectors:
        count += 1
        if (
            dag.color_plan.trace_reflections_folded
            and sector.kind == "single-trace"
            and len(sector.trace_labels) > 2
        ):
            reflected = (sector.trace_labels[0], *reversed(sector.trace_labels[1:]))
            if reflected != sector.trace_labels:
                count += 1
    return count


def _helicity_id(values: Sequence[int]) -> str:
    return "h:" + ",".join(f"{int(value):+d}" for value in values)


def _color_id(word: Sequence[int]) -> str:
    if word:
        return "flow:" + ",".join(str(int(label)) for label in word)
    return "flow:singlet"


def _complex_pair(value: object, name: str) -> tuple[float, float]:
    if isinstance(value, complex):
        return float(value.real), float(value.imag)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        if len(value) != 2:
            raise ValueError(
                f"runtime model parameter {name!r} must have two components"
            )
        return float(value[0]), float(value[1])
    if not isinstance(value, str | int | float):
        raise ValueError(f"runtime model parameter {name!r} is not numeric")
    return float(value), 0.0


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("runtime physics record must be a mapping")
    return {str(key): item for key, item in value.items()}


def _mapping_sequence(value: object) -> tuple[Mapping[str, object], ...]:
    return tuple(_mapping(item) for item in _sequence(value))


def _sequence(value: object) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("runtime physics field must be a sequence")
    return value


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("runtime physics integer field is invalid")
    return value


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("runtime physics numeric field is invalid")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("runtime physics numeric field must be finite")
    return result


__all__ = [
    "EAGER_PLAN_V3_REDUCTION_ENTRIES_MEMBER",
    "EAGER_PLAN_V3_REDUCTION_GROUPS_CONTAINER_PATH",
    "EAGER_PLAN_V3_REDUCTION_GROUPS_KIND",
    "EAGER_PLAN_V3_REDUCTION_GROUPS_MEMBER",
    "EAGER_PLAN_V3_REDUCTION_GROUPS_RUNTIME_LAYOUT_ABI",
    "EAGER_PLAN_V3_REDUCTION_GROUPS_SCHEMA_VERSION",
    "EAGER_PLAN_V3_REDUCTION_GROUPS_STORAGE_ABI",
    "NATIVE_REDUCTION_GROUPS_EXTENSION_KEY",
    "build_resolved_physics_from_dag",
    "build_resolved_physics_payload",
    "build_runtime_model_parameter_records",
    "build_runtime_normalization",
]

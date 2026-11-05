# SPDX-License-Identifier: 0BSD
"""Amplitude grouping and color-contraction runtime records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..color import (
    ColorContractionPlan,
    ColorGroupDescriptor,
    build_color_contraction_plan,
    build_symmetric_group_color_contraction_plan,
)
from ..color.plan_types import LCColorSectorReplayPartition
from ..models.base import Model
from .contracts import runtime_coupling_parameter_names
from .dag_types import AmplitudeRoot, CurrentNode, GenericDAG


@dataclass(frozen=True, slots=True)
class RuntimeAmplitudeStageLayout:
    """Runtime amplitude record plus its typed colour-contraction input."""

    record: dict[str, object]
    color_contraction: ColorContractionPlan | None
    contraction_descriptors: tuple[ColorGroupDescriptor, ...]


def build_runtime_amplitude_stage(
    dag: GenericDAG,
    model: Model,
    *,
    current_slots: Sequence[Mapping[str, object]],
    value_slots: Mapping[tuple[int, str], Mapping[str, object]],
    color_contraction: str = "direct",
) -> dict[str, object]:
    """Build evaluator roots and physical coherent-amplitude metadata."""

    return build_runtime_amplitude_stage_layout(
        dag,
        model,
        current_slots=current_slots,
        value_slots=value_slots,
        color_contraction=color_contraction,
    ).record


def build_runtime_amplitude_stage_layout(
    dag: GenericDAG,
    model: Model,
    *,
    current_slots: Sequence[Mapping[str, object]],
    value_slots: Mapping[tuple[int, str], Mapping[str, object]],
    color_contraction: str = "direct",
) -> RuntimeAmplitudeStageLayout:
    """Build an amplitude record while retaining its authenticated plan."""

    metadata, contraction, contraction_descriptors = (
        _build_runtime_amplitude_metadata_layout(
            dag,
            model,
            color_contraction=color_contraction,
        )
    )
    metadata_roots = _mapping_sequence(metadata["roots"])
    roots: list[dict[str, object]] = []
    for root, record in zip(dag.amplitude_roots, metadata_roots, strict=True):
        roots.append(
            {
                "output_index": record["output_index"],
                "root_id": record["root_id"],
                "dag_root_id": record["dag_root_id"],
                "kind": record["kind"],
                "left_current_id": record["left_current_id"],
                "right_current_id": record["right_current_id"],
                "left_slot": _current_slot_ref(current_slots[root.left_id]),
                "right_slot": _current_slot_ref(current_slots[root.right_id]),
                "left_value_slot": _value_slot_ref(
                    _amplitude_value_slot(dag.currents[root.left_id], value_slots)
                ),
                "right_value_slot": _value_slot_ref(
                    _amplitude_value_slot(dag.currents[root.right_id], value_slots)
                ),
                "vertex_kind": record["vertex_kind"],
                "vertex_particles": record["vertex_particles"],
                "coupling": record["coupling"],
                "coupling_parameter_names": record["coupling_parameter_names"],
                "color_weight": record["color_weight"],
                "color_sector_id": record["color_sector_id"],
                "contraction": record["contraction"],
                "contraction_ir": record["contraction_ir"],
                "coherent_group_id": record["coherent_group_id"],
                "helicity_weight": record["helicity_weight"],
                "all_sector_weight": record["all_sector_weight"],
            }
        )
    return RuntimeAmplitudeStageLayout(
        record={**metadata, "roots": roots},
        color_contraction=contraction,
        contraction_descriptors=contraction_descriptors,
    )


def build_runtime_amplitude_metadata(
    dag: GenericDAG,
    model: Model,
    *,
    color_contraction: str = "direct",
) -> dict[str, object]:
    """Build slot-free amplitude and coherent-group metadata for one DAG."""

    metadata, _, _ = _build_runtime_amplitude_metadata_layout(
        dag,
        model,
        color_contraction=color_contraction,
    )
    return metadata


def _build_runtime_amplitude_metadata_layout(
    dag: GenericDAG,
    model: Model,
    *,
    color_contraction: str,
) -> tuple[
    dict[str, object],
    ColorContractionPlan | None,
    tuple[ColorGroupDescriptor, ...],
]:
    if color_contraction not in {"direct", "symmetric-group-fft"}:
        raise ValueError(
            f"unsupported runtime colour contraction {color_contraction!r}"
        )

    group_ids, descriptors = _amplitude_groups(dag, dag.amplitude_roots)
    replay = _color_topology_replay_amplitudes(dag, descriptors)
    contraction_descriptors = (
        descriptors if replay is None else replay.physical_descriptors
    )
    contraction = (
        build_color_contraction_plan(dag.color_plan, contraction_descriptors)
        if color_contraction == "direct"
        else build_symmetric_group_color_contraction_plan(
            dag.color_plan,
            contraction_descriptors,
        )
    )
    multiple_lc_sectors = _has_multiple_lc_root_sectors(dag)
    roots: list[dict[str, object]] = []
    group_weights: dict[int, tuple[float, float]] = {}
    for output_index, root in enumerate(dag.amplitude_roots):
        group_id = group_ids[root.id]
        all_sector_weight = _root_all_sector_weight(
            dag,
            root,
            has_multiple_lc_root_sectors=multiple_lc_sectors,
        )
        group_weights.setdefault(
            group_id,
            (float(root.helicity_weight), float(all_sector_weight)),
        )
        roots.append(
            {
                "output_index": output_index,
                "root_id": output_index,
                "dag_root_id": root.id,
                "kind": root.kind,
                "left_current_id": root.left_id,
                "right_current_id": root.right_id,
                "vertex_kind": root.vertex_kind,
                "vertex_particles": (
                    None
                    if root.vertex_particles is None
                    else list(root.vertex_particles)
                ),
                "coupling": list(root.coupling),
                "coupling_parameter_names": (
                    None
                    if root.vertex_kind is None or root.vertex_particles is None
                    else runtime_coupling_parameter_names(
                        root.vertex_kind,
                        root.vertex_particles,
                        root.coupling,
                        model=model,
                    )
                ),
                "color_weight": list(root.color_weight),
                "color_sector_id": _root_color_sector_id(dag, root),
                "contraction": root.contraction,
                "contraction_ir": root.contraction_ir.to_json_dict(),
                "coherent_group_id": group_id,
                "helicity_weight": root.helicity_weight,
                "all_sector_weight": all_sector_weight,
            }
        )

    coherent_groups = []
    for descriptor in descriptors:
        helicity_weight, all_sector_weight = group_weights[descriptor.group_id]
        coherent_groups.append(
            {
                "group_id": descriptor.group_id,
                "helicities": _helicity_vector(dag, descriptor.helicity_key),
                "color_sector_id": descriptor.sector_id,
                "color_word": list(descriptor.word),
                "helicity_weight": helicity_weight,
                "all_sector_weight": all_sector_weight,
            }
        )

    return (
        {
            "stage_kind": "amplitude-roots",
            "output_count": len(roots),
            "selected_color_sector_ids": None,
            "coherent_groups": coherent_groups,
            "color_topology_replay": (
                None if replay is None else replay.to_runtime_manifest(dag)
            ),
            "roots": roots,
            "final_reduction": {
                "status": (
                    "sparse-color-contraction"
                    if contraction is not None
                    else "coherent-leading-color-diagonal"
                ),
                "operation": (
                    "sum root outputs into coherent helicity/color amplitudes, "
                    "then apply the requested color contraction"
                ),
            },
            "color_contraction": (
                None if contraction is None else contraction.to_json_dict()
            ),
        },
        contraction,
        contraction_descriptors,
    )


@dataclass(frozen=True, slots=True)
class _ColorReplayGroupRoute:
    source_group_id: int
    target_group_id: int
    factor: tuple[float, float]


@dataclass(frozen=True, slots=True)
class _ColorReplayMapping:
    label_permutation: tuple[tuple[int, int], ...]
    group_routes: tuple[_ColorReplayGroupRoute, ...]


@dataclass(frozen=True, slots=True)
class _ColorReplayAmplitudePlan:
    physical_descriptors: tuple[ColorGroupDescriptor, ...]
    mappings: tuple[_ColorReplayMapping, ...]

    def to_runtime_manifest(self, dag: GenericDAG) -> dict[str, object]:
        groups = []
        for descriptor in self.physical_descriptors:
            groups.append(
                {
                    "group_id": descriptor.group_id,
                    "helicities": _helicity_vector(dag, descriptor.helicity_key),
                    "color_sector_id": descriptor.sector_id,
                    "color_word": list(descriptor.word),
                    "helicity_weight": descriptor.helicity_weight,
                }
            )
        return {
            "contract_version": 1,
            "physical_group_count": len(groups),
            "physical_groups": groups,
            "mappings": [
                {
                    "label_permutation": [
                        {
                            "representative_label": representative,
                            "sector_label": sector,
                        }
                        for representative, sector in mapping.label_permutation
                    ],
                    "group_routes": [
                        {
                            "source_group_id": route.source_group_id,
                            "target_group_id": route.target_group_id,
                            "factor": list(route.factor),
                        }
                        for route in mapping.group_routes
                    ],
                }
                for mapping in self.mappings
            ],
        }


def _color_topology_replay_amplitudes(
    dag: GenericDAG,
    materialized_descriptors: Sequence[ColorGroupDescriptor],
) -> _ColorReplayAmplitudePlan | None:
    """Derive the amplitude-level gather authenticated by the color proof.

    The generated DAG contains only representative sectors.  NLC/full color
    contraction is nevertheless defined over every physical sector, so the
    runtime must evaluate representatives at the proven external-label
    permutations, gather coherent complex amplitudes into the complete
    physical group domain, and only then apply the unchanged color metric.
    """

    certificate = dag.color_topology_replay
    if certificate is None or not certificate.optimized:
        return None
    materialized_by_sector: dict[int, list[ColorGroupDescriptor]] = {}
    for descriptor in materialized_descriptors:
        materialized_by_sector.setdefault(descriptor.sector_id, []).append(descriptor)
    expected_materialized = set(certificate.materialized_sector_ids)
    if set(materialized_by_sector) != expected_materialized:
        raise ValueError(
            "color topology replay materialized amplitude sectors do not match "
            "the proof certificate"
        )

    pending_routes: list[
        tuple[
            tuple[tuple[int, int], ...],
            int,
            tuple[object, ...],
            int,
            tuple[int, ...],
            float,
        ]
    ] = []
    for partition in certificate.partitions:
        source_descriptors = materialized_by_sector.get(
            int(partition.materialized_sector_id)
        )
        if not source_descriptors:
            raise ValueError(
                "color topology replay partition has no materialized amplitudes"
            )
        _append_color_replay_partition_routes(
            dag,
            partition,
            source_descriptors,
            pending_routes,
        )
    for sector_id in certificate.residual_sector_ids:
        source_descriptors = materialized_by_sector.get(int(sector_id))
        if not source_descriptors:
            raise ValueError(
                f"color topology replay residual sector {sector_id} is absent"
            )
        for source in source_descriptors:
            pending_routes.append(
                (
                    (),
                    source.group_id,
                    source.helicity_key,
                    int(sector_id),
                    source.word,
                    source.helicity_weight,
                )
            )

    physical_keys = sorted(
        {
            (helicity_key, sector_id, word, helicity_weight)
            for (
                _permutation,
                _source_group_id,
                helicity_key,
                sector_id,
                word,
                helicity_weight,
            ) in pending_routes
        },
        key=repr,
    )
    physical_id_by_key = {key: group_id for group_id, key in enumerate(physical_keys)}
    physical_descriptors = tuple(
        ColorGroupDescriptor(
            group_id=group_id,
            helicity_key=helicity_key,
            sector_id=sector_id,
            word=word,
            helicity_weight=helicity_weight,
        )
        for group_id, (helicity_key, sector_id, word, helicity_weight) in enumerate(
            physical_keys
        )
    )
    routes_by_mapping: dict[
        tuple[tuple[int, int], ...],
        list[_ColorReplayGroupRoute],
    ] = {}
    target_ids: list[int] = []
    for (
        permutation,
        source_group_id,
        helicity_key,
        sector_id,
        word,
        helicity_weight,
    ) in pending_routes:
        target_group_id = physical_id_by_key[
            (helicity_key, sector_id, word, helicity_weight)
        ]
        routes_by_mapping.setdefault(permutation, []).append(
            _ColorReplayGroupRoute(
                source_group_id=source_group_id,
                target_group_id=target_group_id,
                factor=(1.0, 0.0),
            )
        )
        target_ids.append(target_group_id)
    if sorted(target_ids) != list(range(len(physical_descriptors))):
        raise ValueError(
            "color topology replay amplitude gather is not a physical-group bijection"
        )
    mappings = tuple(
        _ColorReplayMapping(
            label_permutation=permutation,
            group_routes=tuple(
                sorted(
                    routes,
                    key=lambda route: (
                        route.target_group_id,
                        route.source_group_id,
                    ),
                )
            ),
        )
        for permutation, routes in sorted(routes_by_mapping.items())
    )
    if not mappings or any(not mapping.group_routes for mapping in mappings):
        raise ValueError("color topology replay amplitude gather is empty")
    return _ColorReplayAmplitudePlan(
        physical_descriptors=physical_descriptors,
        mappings=mappings,
    )


def _append_color_replay_partition_routes(
    dag: GenericDAG,
    partition: LCColorSectorReplayPartition,
    source_descriptors: Sequence[ColorGroupDescriptor],
    pending_routes: list[
        tuple[
            tuple[tuple[int, int], ...],
            int,
            tuple[object, ...],
            int,
            tuple[int, ...],
            float,
        ]
    ],
) -> None:
    for sector_id, permutation, weight, sign in zip(
        partition.active_sector_ids,
        partition.label_permutations,
        partition.weights,
        partition.signs,
        strict=True,
    ):
        if weight != 1.0 or sign != 1:
            raise ValueError(
                "color topology replay amplitude gather currently requires "
                "unit positive proof factors"
            )
        target_sector = dag.color_plan.sector(int(sector_id))
        if target_sector is None:
            raise ValueError(
                f"color topology replay target sector {sector_id} is absent"
            )
        target_word = tuple(target_sector.word_labels or target_sector.color_words[0])
        normalized_permutation = tuple(
            sorted(
                (int(left), int(right))
                for left, right in permutation
                if int(left) != int(right)
            )
        )
        for source in source_descriptors:
            pending_routes.append(
                (
                    normalized_permutation,
                    source.group_id,
                    _permute_helicity_key(source.helicity_key, normalized_permutation),
                    int(sector_id),
                    target_word,
                    source.helicity_weight,
                )
            )


def _permute_helicity_key(
    helicity_key: tuple[object, ...],
    permutation: Sequence[tuple[int, int]],
) -> tuple[object, ...]:
    by_label = dict(permutation)
    result: list[object] = []
    for item in helicity_key:
        if not isinstance(item, tuple) or len(item) < 5:
            raise ValueError(
                "color topology replay requires explicit external-label helicity keys"
            )
        result.append((by_label.get(int(item[0]), int(item[0])), *item[1:]))
    return tuple(sorted(result))


def _mapping_sequence(value: object) -> tuple[Mapping[str, object], ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise TypeError("runtime amplitude field must be an array")
    result: list[Mapping[str, object]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise TypeError("runtime amplitude record must be an object")
        result.append(item)
    return tuple(result)


def _amplitude_groups(
    dag: GenericDAG,
    roots: Sequence[AmplitudeRoot],
) -> tuple[dict[int, int], tuple[ColorGroupDescriptor, ...]]:
    ids: dict[tuple[object, ...], int] = {}
    result: dict[int, int] = {}
    descriptors: dict[int, ColorGroupDescriptor] = {}
    source_by_ancestry = {
        int(current.index.helicity_ancestry): (
            int(current.source_leg_label or 0),
            int(current.index.particle_id),
            int(current.index.chirality),
            current.index.spin_state,
            current.source_helicity,
        )
        for current in dag.currents
        if current.is_source
    }
    single_bit_ancestry = all(
        bit > 0 and bit & (bit - 1) == 0 for bit in source_by_ancestry
    )
    physical_sources_cache: dict[int, tuple[object, ...]] = {}
    materialized_source_keys = _materialized_root_source_keys(dag)

    def physical_sources(ancestry: int) -> tuple[object, ...]:
        cached = physical_sources_cache.get(ancestry)
        if cached is not None:
            return cached
        if single_bit_ancestry:
            sources: list[tuple[object, ...]] = []
            remaining = ancestry
            while remaining:
                bit = remaining & -remaining
                source = source_by_ancestry.get(bit)
                if source is not None:
                    sources.append(source)
                remaining ^= bit
            value: tuple[object, ...] = tuple(sorted(sources))
        else:
            value = tuple(
                sorted(
                    source
                    for bit, source in source_by_ancestry.items()
                    if ancestry & bit
                )
            )
        physical_sources_cache[ancestry] = value
        return value

    for root in roots:
        left = dag.currents[root.left_id].index
        right = dag.currents[root.right_id].index
        ancestry = int(left.helicity_ancestry | right.helicity_ancestry)
        source_key = materialized_source_keys.get(root.id) or physical_sources(ancestry)
        sector_id = _root_color_sector_id(dag, root)
        sector = dag.color_plan.sector(sector_id)
        word = (
            tuple(sector.word_labels or sector.color_words[0])
            if sector is not None and sector.color_words
            else ()
        )
        color_key = (
            left.color_state.accuracy,
            word or sector_id,
            tuple(
                sorted(
                    set(left.color_state.basis_key) | set(right.color_state.basis_key)
                )
            ),
            tuple(root.color_weight),
        )
        key = (source_key or ancestry, color_key)
        group_id = ids.setdefault(key, len(ids))
        result[root.id] = group_id
        descriptors.setdefault(
            group_id,
            ColorGroupDescriptor(
                group_id=group_id,
                helicity_key=source_key or (ancestry,),
                sector_id=sector_id,
                word=word,
                helicity_weight=float(root.helicity_weight),
            ),
        )
    return result, tuple(descriptors[index] for index in sorted(descriptors))


def _materialized_root_source_keys(
    dag: GenericDAG,
) -> dict[int, tuple[object, ...]]:
    materialization = dag.helicity_materialization
    recurrence = dag.helicity_recurrence
    if materialization is None or recurrence is None:
        return {}
    if materialization.strategy == "retained-proof-graph":
        # The retained graph keeps the original physical roots and their
        # ancestry. Preserve the established coherent grouping so selector
        # metadata cannot change cancellations in the summed workload.
        return {}
    domain_by_id = {domain.id: domain for domain in recurrence.selector_domains}
    routes_by_root: dict[int, list[object]] = {}
    for route in materialization.amplitude_routes:
        routes_by_root.setdefault(route.materialized_root_id, []).append(route)
    result: dict[int, tuple[object, ...]] = {}
    for root_id, routes in routes_by_root.items():
        first_domain_id = next(
            (domain_id for route in routes for domain_id in route.selector_domain_ids),
            None,
        )
        if first_domain_id is None:
            raise ValueError(
                f"materialized amplitude root {root_id} has no selector domain"
            )
        domain = domain_by_id.get(first_domain_id)
        if domain is None or not domain.complete:
            raise ValueError(
                f"materialized amplitude root {root_id} has an invalid selector domain"
            )
        result[root_id] = tuple(
            (label, 0, 0, 0, helicity) for label, helicity in domain.source_states
        )
    return result


def _helicity_vector(
    dag: GenericDAG,
    helicity_key: tuple[object, ...],
) -> list[int]:
    by_label: dict[int, int] = {}
    for item in helicity_key:
        if not isinstance(item, tuple) or len(item) < 5:
            continue
        by_label[int(item[0])] = int(item[4])
    return [by_label.get(leg.label, 0) for leg in dag.process.legs]


def _root_color_sector_id(dag: GenericDAG, root: AmplitudeRoot) -> int:
    if root.color_sector_id is not None:
        return int(root.color_sector_id)
    return int(dag.currents[root.left_id].index.color_state.sector_id)


def _root_all_sector_weight(
    dag: GenericDAG,
    root: AmplitudeRoot,
    *,
    has_multiple_lc_root_sectors: bool,
) -> float:
    weight = float(root.helicity_weight)
    if dag.process.color_accuracy != "lc":
        return weight
    if dag.color_plan.process.color_endpoints.pair_count != 0:
        return weight
    if not dag.color_plan.trace_reflections_folded:
        return weight
    if not has_multiple_lc_root_sectors:
        return weight
    sector = dag.color_plan.sector(_root_color_sector_id(dag, root))
    if sector is None or sector.kind != "single-trace":
        return weight
    trace = tuple(int(label) for label in sector.trace_labels)
    if len(trace) <= 2 or trace[1:] == tuple(reversed(trace[1:])):
        return weight
    return 2.0 * weight


def _has_multiple_lc_root_sectors(dag: GenericDAG) -> bool:
    if dag.process.color_accuracy != "lc":
        return False
    if dag.color_plan.process.color_endpoints.pair_count != 0:
        return False
    return len({_root_color_sector_id(dag, root) for root in dag.amplitude_roots}) > 1


def _current_slot_ref(slot: Mapping[str, object]) -> dict[str, object]:
    return {
        key: slot[key]
        for key in (
            "current_id",
            "component_start",
            "component_stop",
            "dimension",
        )
    }


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


def _amplitude_value_slot(
    current: CurrentNode,
    value_slots: Mapping[tuple[int, str], Mapping[str, object]],
) -> Mapping[str, object]:
    variant = "source" if current.is_source else "unpropagated"
    try:
        return value_slots[(current.id, variant)]
    except KeyError as exc:
        raise ValueError(
            f"amplitude current {current.id} has no {variant} runtime value slot"
        ) from exc


__all__ = [
    "build_runtime_amplitude_metadata",
    "build_runtime_amplitude_stage",
]

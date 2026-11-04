# SPDX-License-Identifier: 0BSD
"""Project proven process semantics into recurrence-builder logical records.

The projector is deliberately upstream of generic DAG construction.  It maps
only canonical process IR, a complete LC colour plan, an optional exact replay
proof, and a prepared recurrence-template catalog into the compact logical ABI
consumed by :mod:`pyamplicol.generation.recurrence_columnar`.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Final

from ..color.plan import (
    ColorTopologyReplayCertificate,
    GenericColorPlan,
    LCColorTopologyReplayPlan,
)
from ..color.plan_build import _ordered_open_line_blocks
from ..color.plan_types import _canonical_open_string_product_key
from ..models.base import Model, Vertex
from ..models.recurrence_template import (
    CurrentStateTemplateV1,
    EvaluatorBindingV1,
    RecurrenceTemplateCatalog,
    SourceTemplateV1,
)
from ..models.recurrence_template import (
    ExactComplexRationalV1 as ModelExactComplexRationalV1,
)
from ..processes.ir import CanonicalProcessIR, ProcessLegIR
from .recurrence_columnar import (
    ExactComplexRationalV1,
    RecurrenceBuilderLogicalInputV1,
    RecurrenceCouplingLimitV1,
    RecurrenceExternalLegV1,
    RecurrenceLCFlowLayout,
    RecurrenceLCOpenStringV1,
    RecurrenceNormalizationV1,
    RecurrenceParameterProjectionV1,
    RecurrencePhysicalLCSectorV1,
    RecurrencePublicLCFlowV1,
    RecurrenceReplayPartitionV1,
    RecurrenceReplayTargetV1,
    RecurrenceSelectedSourceCoverageV1,
    RecurrenceSemanticDigestV1,
    RecurrenceSemanticTemplateReferenceV1,
    RecurrenceSourceStateV1,
)
from .recurrence_fermion_pairing import (
    FermionPairingCatalogV1,
    build_recurrence_fermion_pairing_catalog_v1,
)

_TEMPLATE_SECTIONS: Final = (
    ("parameter", "parameters"),
    ("current-state", "current_states"),
    ("source", "sources"),
    ("quantum-flow", "quantum_flows"),
    ("transition", "transitions"),
    ("propagator", "propagators"),
    ("closure", "closures"),
    ("color-contraction", "color_contractions"),
    ("symmetry-proof", "symmetry_proofs"),
)
_CLOSURE_ANCHOR_PROOF_ALGORITHM_V2: Final = "canonical-lc-closure-anchor-v2"
_CLOSURE_ANCHOR_PROOF_ALGORITHM_V3: Final = "canonical-lc-closure-anchor-v3"
_CLOSURE_ANCHOR_PROOF_ALGORITHM_V4: Final = "canonical-lc-closure-anchor-v4"
_IDENTITY_REPLAY_PROOF_ALGORITHM: Final = (
    "canonical-recurrence-identity-materialization-v1"
)
_PURE_MASSLESS_ADJOINT_HELICITY_SUPPORT_ROLE: Final = (
    "helicity-support:pure-massless-adjoint-tree-v1"
)
_GLOBAL_HELICITY_FLIP_EQUIVALENCE_ROLE: Final = "helicity-equivalence:global-flip-v1"


class RecurrenceProjectionError(ValueError):
    """The supplied process semantics cannot be projected exactly."""


@dataclass(frozen=True, slots=True)
class RecurrenceGenerationSliceV1:
    """Explicit generation-time coverage retained by recurrence v1."""

    selected_public_flow_ids: tuple[int, ...] | None = None
    selected_source_helicities: tuple[tuple[int, int], ...] | None = None

    def __post_init__(self) -> None:
        if self.selected_public_flow_ids is not None:
            flows = tuple(sorted(int(value) for value in self.selected_public_flow_ids))
            if not flows or len(flows) != len(set(flows)):
                raise RecurrenceProjectionError(
                    "selected recurrence public flows must be nonempty and unique"
                )
            object.__setattr__(self, "selected_public_flow_ids", flows)
        if self.selected_source_helicities is not None:
            selected = tuple(
                sorted(
                    (int(label), int(helicity))
                    for label, helicity in self.selected_source_helicities
                )
            )
            labels = tuple(label for label, _ in selected)
            if not selected or len(labels) != len(set(labels)):
                raise RecurrenceProjectionError(
                    "selected recurrence source labels must be nonempty and unique"
                )
            object.__setattr__(self, "selected_source_helicities", selected)


def project_recurrence_process_v1(
    process: CanonicalProcessIR,
    color_plan: GenericColorPlan,
    template_catalog: RecurrenceTemplateCatalog,
    *,
    layout: RecurrenceLCFlowLayout,
    normalization: RecurrenceNormalizationV1,
    topology_replay: (
        LCColorTopologyReplayPlan | ColorTopologyReplayCertificate | None
    ) = None,
    generation_slice: RecurrenceGenerationSliceV1 | None = None,
    coupling_order_limits: Mapping[str, int] | None = None,
    process_support_mask: int = 1,
    model: Model | None = None,
    prefer_symmetric_group_closure_anchor: bool = False,
    contracted_physical_color_union: bool = False,
) -> RecurrenceBuilderLogicalInputV1:
    """Return deterministic recurrence-builder records for one LC process.

    Normalization is supplied as an exact, model-owned contract because it is
    process dependent and intentionally absent from the process-independent
    recurrence template catalog.
    """

    _validate_projection_roots(
        process,
        color_plan,
        template_catalog,
        layout,
        contracted_physical_color_union=contracted_physical_color_union,
    )
    if not isinstance(normalization, RecurrenceNormalizationV1):
        raise TypeError("recurrence projection requires exact normalization v1")
    if (
        isinstance(process_support_mask, bool)
        or not isinstance(process_support_mask, int)
        or process_support_mask <= 0
    ):
        raise RecurrenceProjectionError(
            "recurrence process support mask must be a positive integer bitmap"
        )
    selection = generation_slice or RecurrenceGenerationSliceV1()
    if contracted_physical_color_union and (
        topology_replay is not None
        or selection.selected_public_flow_ids is not None
        or selection.selected_source_helicities is not None
    ):
        raise RecurrenceProjectionError(
            "contracted physical-color union requires complete runtime flow and "
            "source coverage without replay"
        )
    projected_color_plan = (
        _physical_color_owner_plan(color_plan)
        if contracted_physical_color_union
        else color_plan
    )
    reflection_contract = _closure_reflection_contract(
        template_catalog,
        required=projected_color_plan.trace_reflections_folded,
    )

    template_ids, template_references = _project_template_references(template_catalog)
    external_legs, selected_sources = _project_external_legs(
        process,
        template_catalog,
        template_ids,
        selection.selected_source_helicities,
        process_support_mask,
    )
    physical_sectors, public_flows = _project_physical_sectors(
        projected_color_plan,
        process,
        external_legs,
        process_support_mask,
    )
    selected_flows = selection.selected_public_flow_ids
    if selected_flows is not None:
        available_flow_ids = {flow.flow_id for flow in public_flows}
        unknown = tuple(sorted(set(selected_flows) - available_flow_ids))
        if unknown:
            raise RecurrenceProjectionError(
                f"generation slice references unknown public LC flows {unknown!r}"
            )
    replay_partitions = _project_replay_partitions(
        process,
        projected_color_plan,
        physical_sectors,
        external_legs,
        topology_replay,
        layout,
    )
    if prefer_symmetric_group_closure_anchor:
        physical_sectors = _project_cyclic_trace_closure_anchors(
            physical_sectors,
            replay_partitions,
            external_legs,
            template_catalog,
        )
    fermion_pairing = build_recurrence_fermion_pairing_catalog_v1(
        process,
        template_catalog.current_states,
        quantum_flows=template_catalog.quantum_flows,
    )
    closure_reconstruction_digest = _project_closure_obligation_roots_digest(
        process,
        projected_color_plan,
        template_catalog,
        physical_sectors,
        fermion_pairing,
        reflection_contract=reflection_contract,
    )
    helicity_support = _project_helicity_support_proof(
        process,
        model,
        model_catalog_digest=template_catalog.header.compiled_model_digest,
        coupling_order_limits=coupling_order_limits,
    )
    global_helicity_flip = _project_global_helicity_flip_proof(
        process,
        external_legs,
        model,
        model_catalog_digest=template_catalog.header.compiled_model_digest,
        coupling_order_limits=coupling_order_limits,
        generation_selected=selection.selected_source_helicities is not None,
    )

    return RecurrenceBuilderLogicalInputV1(
        process_id=process.key,
        layout=layout,
        semantic_digests=(
            RecurrenceSemanticDigestV1("process", _digest(process.to_json_dict())),
            RecurrenceSemanticDigestV1(
                "model-catalog", template_catalog.header.compiled_model_digest
            ),
            RecurrenceSemanticDigestV1(
                "prepared-catalog", template_catalog.catalog_digest
            ),
            RecurrenceSemanticDigestV1(
                "color-plan", _digest(projected_color_plan.to_json_dict())
            ),
            RecurrenceSemanticDigestV1(
                "fermion-pairing-semantic", fermion_pairing.semantic_digest
            ),
            RecurrenceSemanticDigestV1(
                "fermion-pairing-topology", fermion_pairing.topology_digest
            ),
            RecurrenceSemanticDigestV1(
                "closure-reconstruction",
                closure_reconstruction_digest,
            ),
            *(() if helicity_support is None else (helicity_support,)),
            *(() if global_helicity_flip is None else (global_helicity_flip,)),
        ),
        external_legs=external_legs,
        physical_sectors=physical_sectors,
        public_flows=public_flows,
        semantic_template_references=template_references,
        normalization=normalization,
        fermion_pairing_catalog=fermion_pairing,
        replay_partitions=replay_partitions,
        selected_public_flow_ids=selected_flows,
        selected_source_coverage=selected_sources,
        coupling_limits=_project_coupling_limits(coupling_order_limits),
        parameter_projection=_project_parameters(template_catalog, template_ids),
        process_support_mask=process_support_mask,
    )


def _project_closure_obligation_roots_digest(
    process: CanonicalProcessIR,
    color_plan: GenericColorPlan,
    template_catalog: RecurrenceTemplateCatalog,
    sectors: tuple[RecurrencePhysicalLCSectorV1, ...],
    pairing: FermionPairingCatalogV1,
    *,
    reflection_contract: tuple[
        ModelExactComplexRationalV1 | None,
        str | None,
    ],
) -> str:
    """Bind the exact catalogs from which Rust derives closure obligations.

    This digest is deliberately not an expected-row list. Rust independently
    derives complete three-line traversal, fermion-pairing, and folded-trace
    obligation multisets from these authenticated roots and compares them with
    the closure rows it emits.
    """

    reflection_phase, reflection_proof_digest = reflection_contract
    return _digest(
        {
            "algorithm": "model-generic-closure-obligation-roots-v2",
            "color_plan_digest": _digest(color_plan.to_json_dict()),
            "fermion_pairing_semantic_digest": pairing.semantic_digest,
            "fermion_pairing_topology_digest": pairing.topology_digest,
            "model_catalog_digest": template_catalog.catalog_digest,
            "process_digest": _digest(process.to_json_dict()),
            "reflection_phase": (
                None if reflection_phase is None else reflection_phase.to_dict()
            ),
            "reflection_proof_digest": reflection_proof_digest,
            "sectors": tuple(
                {
                    "closure_proof_algorithm": sector.closure_proof_algorithm,
                    "closure_proof_digest": sector.closure_proof_digest,
                    "kind": sector.kind,
                    "open_strings": tuple(
                        (
                            line.fundamental_source_slot,
                            *line.adjoint_source_slots,
                            line.antifundamental_source_slot,
                        )
                        for line in sector.open_strings
                    ),
                    "sector_id": sector.sector_id,
                    "trace_word": sector.trace_source_slots,
                    "word": sector.word_source_slots,
                }
                for sector in sectors
            ),
        }
    )


def _closure_reflection_contract(
    template_catalog: RecurrenceTemplateCatalog,
    *,
    required: bool,
) -> tuple[ModelExactComplexRationalV1 | None, str | None]:
    proofs = tuple(
        proof
        for proof in template_catalog.symmetry_proofs
        if proof.proof_algorithm == "canonical-current-word-reversal-v1"
    )
    if not proofs:
        if required:
            raise RecurrenceProjectionError(
                "folded LC trace reflections require an exact model "
                "current-reflection proof"
            )
        return None, None
    phases = {proof.exact_phase for proof in proofs}
    if len(phases) != 1:
        raise RecurrenceProjectionError(
            "model current-reflection proofs carry inconsistent exact phases"
        )
    return next(iter(phases)), _digest(
        {
            "algorithm": "model-current-reflection-proof-set-v1",
            "proofs": tuple(proof.to_dict() for proof in proofs),
        }
    )


def _project_helicity_support_proof(
    process: CanonicalProcessIR,
    model: Model | None,
    *,
    model_catalog_digest: str,
    coupling_order_limits: Mapping[str, int] | None,
) -> RecurrenceSemanticDigestV1 | None:
    """Project a fail-closed exact theorem without constructing a process DAG."""

    if model is None:
        return None
    limits = {
        str(name).upper(): int(value)
        for name, value in (coupling_order_limits or {}).items()
        if int(value) >= 0
    }
    vertices = _reachable_vertex_inventory(
        process,
        model,
        coupling_order_limits=limits,
    )
    if not model.pure_massless_adjoint_helicity_zero_rule_is_proven(
        process,
        vertices,
    ):
        return None
    return RecurrenceSemanticDigestV1(
        _PURE_MASSLESS_ADJOINT_HELICITY_SUPPORT_ROLE,
        _digest(
            {
                "algorithm": _PURE_MASSLESS_ADJOINT_HELICITY_SUPPORT_ROLE,
                "coupling_order_limits": sorted(limits.items()),
                "model_catalog_digest": model_catalog_digest,
                "process": process.to_json_dict(),
                "reachable_vertices": [
                    {
                        "kind": int(vertex.kind),
                        "particles": [int(value) for value in vertex.particles],
                    }
                    for vertex in vertices
                ],
            }
        ),
    )


def _project_global_helicity_flip_proof(
    process: CanonicalProcessIR,
    external_legs: Sequence[RecurrenceExternalLegV1],
    model: Model | None,
    *,
    model_catalog_digest: str,
    coupling_order_limits: Mapping[str, int] | None,
    generation_selected: bool,
) -> RecurrenceSemanticDigestV1 | None:
    """Project the same fail-closed parity theorem used by DAG generation.

    The recurrence builder operates before a :class:`GenericDAG` exists, so it
    cannot consume the DAG post-pass directly.  Instead, bind the exact model
    predicate, reachable vertex inventory, and complete two-state external
    helicity axes into the process projection.  Generation-specialized
    helicity slices remain untouched because they need not contain both orbit
    members.
    """

    if model is None or generation_selected:
        return None
    if len(external_legs) != len(process.legs):
        raise RecurrenceProjectionError(
            "recurrence parity proof external-leg projection is incomplete"
        )

    external_axes: list[tuple[int, tuple[int, int]]] = []
    for process_leg, projected_leg in zip(process.legs, external_legs, strict=True):
        outgoing_pdg = process_leg.outgoing_pdg
        if outgoing_pdg is None:
            return None
        particle_id = int(outgoing_pdg)
        if model.mass(particle_id) != 0.0 or not (
            model.is_fundamental_colored_fermion(particle_id)
            or model.is_massless_adjoint_vector(particle_id)
        ):
            return None
        helicities = tuple(
            sorted(
                {int(state.public_helicity) for state in projected_leg.source_states}
            )
        )
        if (
            len(helicities) != 2
            or helicities[0] == 0
            or helicities[0] != -helicities[1]
        ):
            return None
        external_axes.append((int(projected_leg.source_slot), helicities))

    limits = {
        str(name).upper(): int(value)
        for name, value in (coupling_order_limits or {}).items()
        if int(value) >= 0
    }
    vertices = _reachable_vertex_inventory(
        process,
        model,
        coupling_order_limits=limits,
    )
    if not vertices or not model.global_helicity_flip_equivalence_is_proven(vertices):
        return None
    return RecurrenceSemanticDigestV1(
        _GLOBAL_HELICITY_FLIP_EQUIVALENCE_ROLE,
        _digest(
            {
                "algorithm": _GLOBAL_HELICITY_FLIP_EQUIVALENCE_ROLE,
                "coupling_order_limits": sorted(limits.items()),
                "external_helicity_axes": external_axes,
                "model_catalog_digest": model_catalog_digest,
                "process": process.to_json_dict(),
                "reachable_vertices": [
                    {
                        "kind": int(vertex.kind),
                        "particles": [int(value) for value in vertex.particles],
                    }
                    for vertex in vertices
                ],
            }
        ),
    )


def _reachable_vertex_inventory(
    process: CanonicalProcessIR,
    model: Model,
    *,
    coupling_order_limits: Mapping[str, int],
) -> tuple[Vertex, ...]:
    """Conservatively close the process species under allowed local vertices."""

    reachable = {
        int(leg.outgoing_pdg) for leg in process.legs if leg.outgoing_pdg is not None
    }
    used: set[Vertex] = set()
    vertices = tuple(model.iter_vertices(color_accuracy=process.color_accuracy))
    changed = True
    while changed:
        changed = False
        for vertex in vertices:
            left, right, result = (int(value) for value in vertex.particles)
            if left not in reachable or right not in reachable:
                continue
            orders = {
                str(name).upper(): int(value)
                for name, value in model.vertex_coupling_orders(vertex)
            }
            if any(
                orders.get(name, 0) > limit
                for name, limit in coupling_order_limits.items()
            ):
                continue
            used.add(vertex)
            if result not in reachable:
                reachable.add(result)
                changed = True
    return tuple(
        sorted(
            used,
            key=lambda vertex: (
                int(vertex.kind),
                tuple(int(value) for value in vertex.particles),
            ),
        )
    )


def _validate_projection_roots(
    process: CanonicalProcessIR,
    color_plan: GenericColorPlan,
    template_catalog: RecurrenceTemplateCatalog,
    layout: str,
    *,
    contracted_physical_color_union: bool,
) -> None:
    if not isinstance(process, CanonicalProcessIR):
        raise TypeError("recurrence projection requires CanonicalProcessIR")
    if not isinstance(color_plan, GenericColorPlan):
        raise TypeError("recurrence projection requires GenericColorPlan")
    if not isinstance(template_catalog, RecurrenceTemplateCatalog):
        raise TypeError("recurrence projection requires RecurrenceTemplateCatalog")
    if type(contracted_physical_color_union) is not bool:
        raise TypeError("contracted physical-color union opt-in must be boolean")
    if layout not in {
        "topology-replay",
        "all-flow-union",
        "contracted-color-union",
    }:
        raise RecurrenceProjectionError(
            f"unsupported recurrence flow layout {layout!r}"
        )
    if process.color_accuracy != color_plan.color_accuracy:
        raise RecurrenceProjectionError(
            "recurrence process and color-plan accuracies disagree"
        )
    if contracted_physical_color_union and (
        process.color_accuracy not in {"nlc", "full"}
        or layout != "all-flow-union"
        or color_plan.trace_reflections_folded
    ):
        raise RecurrenceProjectionError(
            "contracted physical-color union requires complete NLC/full color, "
            "an all-flow-union layout, and unfolded physical sectors"
        )
    if process.color_accuracy == "lc" and layout == "contracted-color-union":
        raise RecurrenceProjectionError(
            "LC recurrence requires topology-replay or all-flow-union"
        )
    if (
        process.color_accuracy in {"nlc", "full"}
        and layout != "contracted-color-union"
        and not contracted_physical_color_union
    ):
        raise RecurrenceProjectionError(
            "NLC/full recurrence requires the contracted-color-union strategy"
        )
    if process.color_accuracy not in {"lc", "nlc", "full"}:
        raise RecurrenceProjectionError(
            f"unsupported recurrence color accuracy {process.color_accuracy!r}"
        )
    if color_plan.process != process:
        raise RecurrenceProjectionError(
            "recurrence color plan does not belong to the supplied process IR"
        )
    if color_plan.truncated:
        raise RecurrenceProjectionError(
            "recurrence projection rejects truncated color plans"
        )
    if not color_plan.sectors:
        raise RecurrenceProjectionError(
            "recurrence projection requires physical LC sectors"
        )
    sector_ids = tuple(sorted(int(sector.id) for sector in color_plan.sectors))
    if sector_ids != tuple(range(len(sector_ids))):
        raise RecurrenceProjectionError(
            "recurrence projection requires dense physical LC sector IDs"
        )


def _physical_color_owner_plan(color_plan: GenericColorPlan) -> GenericColorPlan:
    """Return dense canonical tensor owners for one complete physical plan."""

    owner_by_key: dict[tuple[object, ...], int] = {}
    owners = []
    for sector in sorted(color_plan.sectors, key=lambda item: int(item.id)):
        sector_id = int(sector.id)
        if sector.kind == "open-lines":
            key: tuple[object, ...] = (
                "open-lines",
                _canonical_open_string_product_key(
                    (
                        line.fundamental_label,
                        line.adjoint_labels,
                        line.antifundamental_label,
                        line.singlet_labels,
                    )
                    for line in sector.open_color_lines
                ),
            )
        else:
            key = (sector.kind, sector_id)
        if owner_by_key.setdefault(key, sector_id) != sector_id:
            continue
        owners.append(replace(sector, id=len(owners)))
    if not owners:
        raise RecurrenceProjectionError(
            "contracted physical-color union has no canonical owner sectors"
        )
    return replace(color_plan, sectors=tuple(owners))


def _project_template_references(
    catalog: RecurrenceTemplateCatalog,
) -> tuple[
    dict[tuple[str, str], int],
    tuple[RecurrenceSemanticTemplateReferenceV1, ...],
]:
    binding_by_template: dict[str, EvaluatorBindingV1] = {}
    for binding in catalog.evaluator_bindings:
        for semantic_id in binding.semantic_template_ids:
            previous = binding_by_template.setdefault(semantic_id, binding)
            if previous != binding:
                raise RecurrenceProjectionError(
                    f"semantic template {semantic_id!r} has multiple callables"
                )

    numeric_ids: dict[tuple[str, str], int] = {}
    references: list[RecurrenceSemanticTemplateReferenceV1] = []
    for kind, attribute in _TEMPLATE_SECTIONS:
        records = tuple(
            sorted(getattr(catalog, attribute), key=lambda item: item.template_id)
        )
        for numeric_id, record in enumerate(records):
            key = (kind, record.template_id)
            numeric_ids[key] = numeric_id
            binding = binding_by_template.get(record.template_id)
            prepared_kernel_id = None if binding is None else binding.prepared_kernel_id
            references.append(
                RecurrenceSemanticTemplateReferenceV1(
                    kind=kind,
                    template_id=numeric_id,
                    semantic_digest=record.semantic_digest,
                    prepared_kernel_id=prepared_kernel_id,
                )
            )
    return numeric_ids, tuple(references)


def _project_external_legs(
    process: CanonicalProcessIR,
    catalog: RecurrenceTemplateCatalog,
    template_ids: Mapping[tuple[str, str], int],
    selected_helicities: tuple[tuple[int, int], ...] | None,
    support_mask: int,
) -> tuple[
    tuple[RecurrenceExternalLegV1, ...],
    tuple[RecurrenceSelectedSourceCoverageV1, ...] | None,
]:
    selected = dict(selected_helicities or ())
    known_labels = {int(leg.label) for leg in process.legs}
    unknown_labels = tuple(sorted(set(selected) - known_labels))
    if unknown_labels:
        raise RecurrenceProjectionError(
            f"generation slice references unknown source labels {unknown_labels!r}"
        )

    states_by_id = {record.template_id: record for record in catalog.current_states}
    states_by_execution_key: dict[tuple[object, ...], CurrentStateTemplateV1] = {}
    for record in catalog.current_states:
        key = _current_state_execution_key(record)
        previous = states_by_execution_key.setdefault(key, record)
        if previous.template_id != record.template_id:
            raise RecurrenceProjectionError(
                "recurrence current-state catalog contains duplicate execution "
                f"contracts {previous.template_id!r} and {record.template_id!r}"
            )
    source_rows: dict[
        int,
        list[tuple[SourceTemplateV1, CurrentStateTemplateV1]],
    ] = {}
    for source in catalog.sources:
        try:
            current = states_by_id[source.state_template_id]
        except KeyError as exc:
            raise RecurrenceProjectionError(
                f"source template {source.template_id!r} references a missing state"
            ) from exc
        source_rows.setdefault(int(current.particle_id), []).append((source, current))

    external: list[RecurrenceExternalLegV1] = []
    selected_coverage: list[RecurrenceSelectedSourceCoverageV1] = []
    for source_slot, leg in enumerate(process.legs):
        is_fermionic = _is_fermionic_process_leg(leg)
        if leg.pdg is None or leg.outgoing_pdg is None:
            raise RecurrenceProjectionError(
                f"external leg {leg.label} has no concrete physical/outgoing PDG"
            )
        candidates = source_rows.get(int(leg.outgoing_pdg), ())
        projected: list[
            tuple[
                SourceTemplateV1,
                CurrentStateTemplateV1,
                int,
                int,
                int,
                int,
                ExactComplexRationalV1,
            ]
        ] = []
        for source, current in candidates:
            if source.wavefunction_family != leg.wavefunction_family:
                continue
            helicity = int(source.helicity)
            chirality = int(current.chirality)
            spin_state = int(source.spin_state)
            crossing = _crossing_contract(source)
            momentum_sign = 1
            phase = ExactComplexRationalV1(1)
            if leg.is_initial:
                helicity *= crossing[0]
                chirality *= crossing[1]
                spin_state *= crossing[2]
                momentum_sign = crossing[3]
                phase = crossing[4]
            execution_key = _current_state_execution_key(
                current,
                chirality=chirality,
            )
            try:
                execution_current = states_by_execution_key[execution_key]
            except KeyError as exc:
                raise RecurrenceProjectionError(
                    "recurrence template catalog has no current-state contract for "
                    f"crossed source {source.template_id!r}: particle="
                    f"{current.particle_id}, chirality={chirality}"
                ) from exc
            projected.append(
                (
                    source,
                    execution_current,
                    helicity,
                    chirality,
                    spin_state,
                    momentum_sign,
                    phase,
                )
            )
        projected.sort(
            key=lambda item: (
                item[2],
                item[3],
                item[4],
                item[1].template_id,
                item[0].template_id,
            )
        )
        if not projected:
            raise RecurrenceProjectionError(
                "recurrence template catalog has no supported source semantics for "
                f"external leg {leg.label} (outgoing PDG {leg.outgoing_pdg})"
            )

        source_states = tuple(
            RecurrenceSourceStateV1(
                state_index=index,
                public_helicity=helicity,
                chirality=chirality,
                spin_state=spin_state,
                current_state_template_id=_template_id(
                    template_ids,
                    "current-state",
                    current.template_id,
                ),
                source_template_id=_template_id(
                    template_ids, "source", source.template_id
                ),
                momentum_sign=momentum_sign,
                crossing_phase=phase,
            )
            for index, (
                source,
                current,
                helicity,
                chirality,
                spin_state,
                momentum_sign,
                phase,
            ) in enumerate(projected)
        )
        external.append(
            RecurrenceExternalLegV1(
                source_slot=source_slot,
                public_label=int(leg.label),
                physical_pdg=int(leg.pdg),
                outgoing_pdg=int(leg.outgoing_pdg),
                is_initial=bool(leg.is_initial),
                is_fermionic=is_fermionic,
                source_states=source_states,
                momentum_mask=1 << source_slot,
                support_mask=support_mask,
            )
        )
        if selected_helicities is not None:
            requested = selected.get(int(leg.label))
            retained = tuple(
                index
                for index, (
                    _source,
                    _current,
                    helicity,
                    _chirality,
                    _spin_state,
                    _momentum_sign,
                    _phase,
                ) in enumerate(projected)
                if requested is None or helicity == requested
            )
            if not retained:
                available = tuple(sorted({item[2] for item in projected}))
                raise RecurrenceProjectionError(
                    f"selected helicity {requested} is unavailable for source "
                    f"label {leg.label}; available={available!r}"
                )
            selected_coverage.append(
                RecurrenceSelectedSourceCoverageV1(source_slot, retained)
            )

    return (
        tuple(external),
        None if selected_helicities is None else tuple(selected_coverage),
    )


def _current_state_execution_key(
    state: CurrentStateTemplateV1,
    *,
    chirality: int | None = None,
) -> tuple[object, ...]:
    """Identity of a current state after applying external crossing.

    Crossing may change the chiral representation used by subsequent
    recurrence transitions, but it does not change the model-owned particle,
    basis, colour, tensor, or parameter contracts.
    """

    return (
        state.particle_id,
        state.anti_particle_id,
        state.species_id,
        state.orientation,
        state.statistics,
        state.color_representation,
        state.basis,
        state.tensor_ordering,
        state.dimension,
        state.chirality if chirality is None else chirality,
        state.lc_color_shape_kind,
        state.auxiliary_kind,
        state.mass_parameter_id,
        state.width_parameter_id,
    )


def _is_fermionic_process_leg(leg: ProcessLegIR) -> bool:
    concrete_statistics = {"auxiliary", "boson", "fermion", "ghost"}
    concrete_families = {
        "auxiliary",
        "fermion",
        "ghost",
        "scalar",
        "spin2",
        "vector",
    }
    if (
        leg.statistics not in concrete_statistics
        or leg.wavefunction_family not in concrete_families
    ):
        raise RecurrenceProjectionError(
            "recurrence closure-anchor classification requires concrete external "
            f"statistics and spin metadata for leg {leg.label}"
        )
    statistics_is_fermionic = leg.statistics == "fermion"
    family_is_fermionic = leg.wavefunction_family == "fermion"
    if statistics_is_fermionic != family_is_fermionic:
        raise RecurrenceProjectionError(
            "recurrence closure-anchor classification found inconsistent "
            f"statistics and spin metadata for leg {leg.label}"
        )
    return statistics_is_fermionic


def _crossing_contract(
    source: SourceTemplateV1,
) -> tuple[int, int, int, int, ExactComplexRationalV1]:
    try:
        payload = json.loads(source.crossing)
        helicity = payload["helicity_factor"]
        chirality = payload["chirality_factor"]
        spin_state = payload["spin_state_factor"]
        momentum_transform = payload["momentum_transform"]
        phase = payload["phase"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RecurrenceProjectionError(
            f"source template {source.template_id!r} has malformed crossing semantics"
        ) from exc
    if (
        helicity not in {-1, 1}
        or chirality not in {-1, 1}
        or spin_state not in {-1, 1}
        or momentum_transform not in {"identity", "negate-four-momentum"}
        or not isinstance(phase, Mapping)
    ):
        raise RecurrenceProjectionError(
            f"source template {source.template_id!r} has unsupported crossing factors"
        )
    try:
        exact_phase = ExactComplexRationalV1(
            int(phase["real_numerator"]),
            int(phase["real_denominator"]),
            int(phase["imag_numerator"]),
            int(phase["imag_denominator"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RecurrenceProjectionError(
            f"source template {source.template_id!r} has malformed crossing phase"
        ) from exc
    momentum_sign = -1 if momentum_transform == "negate-four-momentum" else 1
    return int(helicity), int(chirality), int(spin_state), momentum_sign, exact_phase


def _project_physical_sectors(
    color_plan: GenericColorPlan,
    process: CanonicalProcessIR,
    external_legs: tuple[RecurrenceExternalLegV1, ...],
    support_mask: int,
) -> tuple[
    tuple[RecurrencePhysicalLCSectorV1, ...],
    tuple[RecurrencePublicLCFlowV1, ...],
]:
    slot_by_label = {
        int(leg.label): source_slot for source_slot, leg in enumerate(process.legs)
    }

    def slots(labels: Sequence[int], context: str) -> tuple[int, ...]:
        try:
            return tuple(slot_by_label[int(label)] for label in labels)
        except KeyError as exc:
            raise RecurrenceProjectionError(
                f"{context} references unknown external label {exc.args[0]}"
            ) from exc

    result: list[RecurrencePhysicalLCSectorV1] = []
    public_flows: list[RecurrencePublicLCFlowV1] = []
    identity_permutation = tuple(range(len(process.legs)))
    for sector in sorted(color_plan.sectors, key=lambda item: item.id):
        words = sector.color_words
        if not words:
            raise RecurrenceProjectionError(
                f"LC sector {sector.id} has no physical color word"
            )
        word = sector.word_labels or words[0]
        public_id = (
            "flow:" + ",".join(str(int(label)) for label in word)
            if word
            else "flow:singlet"
        )
        open_strings = tuple(
            RecurrenceLCOpenStringV1(
                fundamental_source_slot=slots(
                    (line.fundamental_label,), "LC fundamental endpoint"
                )[0],
                antifundamental_source_slot=slots(
                    (line.antifundamental_label,), "LC antifundamental endpoint"
                )[0],
                adjoint_source_slots=slots(
                    line.adjoint_labels, "LC open-string adjoints"
                ),
                singlet_source_slots=slots(
                    line.singlet_labels, "LC open-string singlets"
                ),
            )
            for line in sector.open_color_lines
        )
        trace_source_slots = slots(sector.trace_labels, "LC trace")
        singlet_source_slots = slots(sector.singlet_labels, "LC singlets")
        word_source_slots = slots(word, "LC color word")
        closure_word = sector.canonical_closure_traversal_word(process)
        closure_word_source_slots = slots(
            closure_word,
            "LC closure traversal",
        )
        physical_blocks = (
            _ordered_open_line_blocks(tuple(word), sector.open_color_lines)
            if sector.kind == "open-lines"
            else ()
        )
        closure_blocks = (
            _ordered_open_line_blocks(closure_word, sector.open_color_lines)
            if sector.kind == "open-lines"
            else ()
        )
        if physical_blocks is None or closure_blocks is None:
            raise RecurrenceProjectionError(
                f"LC sector {sector.id} open-line blocks cannot be reconstructed"
            )
        (
            closure_source_slot,
            closure_proof_algorithm,
            closure_proof_digest,
        ) = _closure_anchor_contract(
            sector_id=int(sector.id),
            sector_kind=sector.kind,
            open_string_count=len(open_strings),
            word_source_slots=word_source_slots,
            closure_word_source_slots=closure_word_source_slots,
            physical_block_source_slots=tuple(
                slots(block, "LC physical open-line block") for block in physical_blocks
            ),
            closure_block_source_slots=tuple(
                slots(block, "LC closure open-line block") for block in closure_blocks
            ),
            process_source_label_order=tuple(int(leg.label) for leg in process.legs),
            singlet_source_slots=singlet_source_slots,
            external_legs=external_legs,
        )
        result.append(
            RecurrencePhysicalLCSectorV1(
                sector_id=int(sector.id),
                public_id=public_id,
                kind=sector.kind,
                closure_source_slot=closure_source_slot,
                closure_proof_algorithm=closure_proof_algorithm,
                closure_proof_digest=closure_proof_digest,
                open_strings=open_strings,
                trace_source_slots=trace_source_slots,
                singlet_source_slots=singlet_source_slots,
                word_source_slots=word_source_slots,
                support_mask=support_mask,
            )
        )
        public_flows.append(
            RecurrencePublicLCFlowV1(
                flow_id=len(public_flows),
                public_id=public_id,
                construction_sector_id=int(sector.id),
                word_source_slots=word_source_slots,
                source_slot_permutation=identity_permutation,
            )
        )
        if not (
            color_plan.trace_reflections_folded
            and sector.kind == "single-trace"
            and len(word) > 2
        ):
            continue
        reflected = (word[0], *reversed(word[1:]))
        if reflected == tuple(word):
            continue
        public_flows.append(
            RecurrencePublicLCFlowV1(
                flow_id=len(public_flows),
                public_id="flow:" + ",".join(str(int(label)) for label in reflected),
                construction_sector_id=int(sector.id),
                word_source_slots=slots(reflected, "reflected LC public color word"),
                # Exact trace reflection changes only the ordered colour word.
                # External states and momenta remain attached to their public
                # labels; the reflection proof supplies the amplitude relation.
                source_slot_permutation=identity_permutation,
            )
        )
    return tuple(result), tuple(public_flows)


def _closure_anchor_contract(
    *,
    sector_id: int,
    sector_kind: str,
    open_string_count: int,
    word_source_slots: tuple[int, ...],
    closure_word_source_slots: tuple[int, ...],
    physical_block_source_slots: tuple[tuple[int, ...], ...],
    closure_block_source_slots: tuple[tuple[int, ...], ...],
    process_source_label_order: tuple[int, ...],
    singlet_source_slots: tuple[int, ...],
    external_legs: tuple[RecurrenceExternalLegV1, ...],
) -> tuple[int, str, str]:
    external_source_count = len(external_legs)
    if external_source_count <= 0:
        raise RecurrenceProjectionError(
            "a physical LC sector requires at least one external source"
        )
    if word_source_slots:
        if not closure_word_source_slots:
            raise RecurrenceProjectionError(
                "a coloured LC sector has an empty closure traversal"
            )
        if set(closure_word_source_slots) != set(word_source_slots):
            raise RecurrenceProjectionError(
                "LC closure traversal does not preserve the physical colour word"
            )
        closure_source_slot = closure_word_source_slots[-1]
        # Newly projected open-line sectors always carry the v3 structural
        # proof, including the no-rotation case.  The decoder retains v2 for
        # already encoded terminal-word contracts.
        proof_algorithm = (
            _CLOSURE_ANCHOR_PROOF_ALGORITHM_V3
            if sector_kind == "open-lines"
            else _CLOSURE_ANCHOR_PROOF_ALGORITHM_V2
        )
        policy = (
            "terminal-colour-word-endpoint"
            if proof_algorithm == _CLOSURE_ANCHOR_PROOF_ALGORITHM_V2
            else "minimum-coloured-source-slot-open-line-block-rotation"
        )
    else:
        if sector_kind != "singlet" or tuple(sorted(singlet_source_slots)) != tuple(
            range(external_source_count)
        ):
            raise RecurrenceProjectionError(
                "an LC sector without a colour word must be an all-singlet sector"
            )
        fermionic_source_slots = tuple(
            leg.source_slot for leg in external_legs if leg.is_fermionic
        )
        if fermionic_source_slots:
            policy = "minimum-fermionic-source-slot"
            closure_source_slot = min(fermionic_source_slots)
        else:
            policy = "minimum-source-slot"
            closure_source_slot = min(leg.source_slot for leg in external_legs)
        proof_algorithm = _CLOSURE_ANCHOR_PROOF_ALGORITHM_V2
    proof_payload: dict[str, object] = {
        "algorithm": proof_algorithm,
        "closure_source_slot": closure_source_slot,
        "component_order_policy": (
            "independent-open-string-block-permutation"
            if sector_kind == "open-lines" and open_string_count > 1
            else "exact-ordered-components"
        ),
        "external_source_count": external_source_count,
        "fermionic_source_slots": tuple(
            leg.source_slot for leg in external_legs if leg.is_fermionic
        ),
        "open_string_count": open_string_count,
        "policy": policy,
        "sector_id": sector_id,
        "sector_kind": sector_kind,
        "singlet_source_slots": singlet_source_slots,
        "word_source_slots": word_source_slots,
    }
    if proof_algorithm == _CLOSURE_ANCHOR_PROOF_ALGORITHM_V3:
        if not physical_block_source_slots or not closure_block_source_slots:
            raise RecurrenceProjectionError(
                "v3 LC closure anchor requires reconstructed open-line blocks"
            )
        distinguished_source_slot = min(word_source_slots)
        if distinguished_source_slot not in closure_block_source_slots[0]:
            raise RecurrenceProjectionError(
                "v3 LC closure traversal does not start with the block containing "
                "the minimum coloured source slot"
            )
        proof_payload.update(
            {
                "closure_block_source_slots": closure_block_source_slots,
                "closure_traversal_source_slots": closure_word_source_slots,
                "distinguished_source_slot": distinguished_source_slot,
                "distinguished_source_block": closure_block_source_slots[0],
                "physical_block_source_slots": physical_block_source_slots,
                "process_source_label_order": process_source_label_order,
                "selected_closure_block": closure_block_source_slots[-1],
            }
        )
    proof_digest = _digest(proof_payload)
    return closure_source_slot, proof_algorithm, proof_digest


def _project_cyclic_trace_closure_anchors(
    sectors: tuple[RecurrencePhysicalLCSectorV1, ...],
    replay_partitions: tuple[RecurrenceReplayPartitionV1, ...],
    external_legs: tuple[RecurrenceExternalLegV1, ...],
    template_catalog: RecurrenceTemplateCatalog,
) -> tuple[RecurrencePhysicalLCSectorV1, ...]:
    """Fix one authenticated adjoint trace source as the closure sink.

    This is deliberately an opt-in projection for the symmetric-group colour
    lane.  It uses only cyclic trace equivalence plus process/template
    contracts.  Any heterogeneous, open-line, fermionic, auxiliary, or
    replay-incompatible channel retains its established v2/v3 anchor.
    """

    # Replay and reflection folding choose their canonical owners independently.
    # A replay representative can therefore be the non-canonical member of a
    # reflection orbit whose owner belongs to another replay partition.  Keep
    # the established per-sector anchors until those two proofs are composed.
    if replay_partitions:
        return sectors

    state_template_id = _cyclic_trace_state_template_id(
        external_legs,
        template_catalog,
    )
    if state_template_id is None:
        return sectors

    candidates: dict[int, int] = {}
    full_source_domain = tuple(range(len(external_legs)))
    for sector in sectors:
        if (
            sector.kind != "single-trace"
            or sector.open_strings
            or sector.singlet_source_slots
            or not sector.trace_source_slots
            or sector.trace_source_slots != sector.word_source_slots
            or tuple(sorted(sector.word_source_slots)) != full_source_domain
        ):
            continue
        candidates[sector.sector_id] = min(sector.word_source_slots)

    # A replayed representative must keep the same distinguished source under
    # every exact target mapping.  Fall back for the complete partition when
    # any target cannot preserve that closure ancestry.
    rejected: set[int] = set()
    for partition in replay_partitions:
        partition_sector_ids = {target.sector_id for target in partition.targets}
        representative_anchor = candidates.get(partition.representative_sector_id)
        if representative_anchor is None or not partition_sector_ids.issubset(
            candidates
        ):
            rejected.update(partition_sector_ids)
            continue
        if any(
            target.source_slot_permutation[representative_anchor]
            != candidates[target.sector_id]
            for target in partition.targets
        ):
            rejected.update(partition_sector_ids)

    result: list[RecurrencePhysicalLCSectorV1] = []
    process_source_label_order = tuple(leg.public_label for leg in external_legs)
    fermionic_source_slots = tuple(
        leg.source_slot for leg in external_legs if leg.is_fermionic
    )
    for sector in sectors:
        closure_source_slot = candidates.get(sector.sector_id)
        if closure_source_slot is None or sector.sector_id in rejected:
            result.append(sector)
            continue
        proof_payload = {
            "algorithm": _CLOSURE_ANCHOR_PROOF_ALGORITHM_V4,
            "closure_source_slot": closure_source_slot,
            "component_order_policy": "cyclic-single-trace",
            "external_source_count": len(external_legs),
            "fermionic_source_slots": fermionic_source_slots,
            "open_string_count": 0,
            "policy": "minimum-source-slot-cyclic-single-trace",
            "process_source_label_order": process_source_label_order,
            "sector_id": sector.sector_id,
            "sector_kind": sector.kind,
            "singlet_source_slots": sector.singlet_source_slots,
            "state_template_id": state_template_id,
            "trace_source_slots": sector.trace_source_slots,
            "word_source_slots": sector.word_source_slots,
        }
        result.append(
            replace(
                sector,
                closure_source_slot=closure_source_slot,
                closure_proof_algorithm=_CLOSURE_ANCHOR_PROOF_ALGORITHM_V4,
                closure_proof_digest=_digest(proof_payload),
            )
        )
    return tuple(result)


def _cyclic_trace_state_template_id(
    external_legs: tuple[RecurrenceExternalLegV1, ...],
    catalog: RecurrenceTemplateCatalog,
) -> int | None:
    """Return one process-wide state ID only for the certified adjoint case."""

    if not external_legs or any(
        leg.is_fermionic or not leg.source_states for leg in external_legs
    ):
        return None
    state_ids = {
        state.current_state_template_id
        for leg in external_legs
        for state in leg.source_states
    }
    if len(state_ids) != 1:
        return None
    state_template_id = next(iter(state_ids))
    try:
        state = catalog.current_states[state_template_id]
    except IndexError:
        return None
    if (
        state.orientation != "self-conjugate"
        or state.statistics != "boson"
        or state.particle_id != state.anti_particle_id
        or state.color_representation != 8
        or state.lc_color_shape_kind != "adjoint-segment"
        or state.auxiliary_kind is not None
    ):
        return None

    bindings = {binding.resolver_key: binding for binding in catalog.evaluator_bindings}
    colors = {color.template_id: color for color in catalog.color_contractions}
    for closure in catalog.closures:
        if (
            closure.input_state_template_ids != (state.template_id, state.template_id)
            or closure.result_state_template_id is not None
            or closure.coupling_parameter_ids
            or closure.coupling_orders
            or closure.eligible_quantum_flow_template_ids
            or closure.output_factor_source != "none"
            or closure.equivalence_class != "direct-contraction"
        ):
            continue
        binding = bindings.get(closure.evaluator_resolver_key)
        color = colors.get(closure.color_contraction_template_id)
        if (
            binding is not None
            and binding.callable_kind == "rusticol-template"
            and binding.contract_kind == "closure"
            and binding.input_state_template_ids
            == (state.template_id, state.template_id)
            and binding.output_state_template_id is None
            and color is not None
            and color.input_representations == (8, 8)
            and color.output_representation is None
        ):
            return state_template_id
    return None


def _project_replay_partitions(
    process: CanonicalProcessIR,
    color_plan: GenericColorPlan,
    physical_sectors: tuple[RecurrencePhysicalLCSectorV1, ...],
    external_legs: tuple[RecurrenceExternalLegV1, ...],
    replay: LCColorTopologyReplayPlan | ColorTopologyReplayCertificate | None,
    layout: RecurrenceLCFlowLayout,
) -> tuple[RecurrenceReplayPartitionV1, ...]:
    if layout not in {"topology-replay", "contracted-color-union"}:
        return ()
    if replay is None:
        return ()
    if layout == "topology-replay" and not isinstance(
        replay, LCColorTopologyReplayPlan
    ):
        raise RecurrenceProjectionError(
            "LC topology-replay layout requires an LC replay plan"
        )
    if layout == "contracted-color-union" and not isinstance(
        replay, ColorTopologyReplayCertificate
    ):
        raise RecurrenceProjectionError(
            "contracted-color-union replay requires a color-generic certificate"
        )
    if isinstance(replay, ColorTopologyReplayCertificate) and (
        replay.color_accuracy != color_plan.color_accuracy
    ):
        raise RecurrenceProjectionError(
            "topology replay certificate color accuracy does not match the plan"
        )
    physical_ids = tuple(sorted(int(sector.id) for sector in color_plan.sectors))
    if replay.physical_sector_ids != physical_ids:
        raise RecurrenceProjectionError(
            "topology replay proof does not cover this physical LC plan"
        )

    source_slot_by_label = {
        int(leg.label): source_slot for source_slot, leg in enumerate(process.legs)
    }
    labels = set(source_slot_by_label)
    physical_sector_by_id = {sector.sector_id: sector for sector in physical_sectors}
    partitions: list[RecurrenceReplayPartitionV1] = []
    identity_materializations: dict[
        int, tuple[str, int, Mapping[str, object] | None]
    ] = {}
    for partition in replay.partitions:
        if partition.proof_algorithm is None or partition.proof_digest is None:
            raise RecurrenceProjectionError(
                "topology replay partition lacks an exact proof certificate"
            )
        representative_sector_id = int(partition.representative_sector_id)
        try:
            representative_sector = physical_sector_by_id[representative_sector_id]
        except KeyError as exc:
            raise RecurrenceProjectionError(
                "topology replay references a missing representative LC sector "
                f"{representative_sector_id}"
            ) from exc
        representative_anchor = representative_sector.closure_source_slot
        projected_targets: list[
            tuple[RecurrenceReplayTargetV1, bool, int, Mapping[str, object]]
        ] = []
        for sector_id, raw_mapping, weight, sign in zip(
            partition.active_sector_ids,
            partition.label_permutations,
            partition.weights,
            partition.signs,
            strict=True,
        ):
            mapping = {int(left): int(right) for left, right in raw_mapping}
            if not set(mapping).issubset(labels) or not set(mapping.values()).issubset(
                labels
            ):
                raise RecurrenceProjectionError(
                    "topology replay maps an unknown external label"
                )
            complete = {label: mapping.get(label, label) for label in labels}
            if set(complete.values()) != labels:
                raise RecurrenceProjectionError(
                    "topology replay external mapping is not a bijection"
                )
            permutation = tuple(
                source_slot_by_label[complete[int(leg.label)]] for leg in process.legs
            )
            for representative_slot, target_slot in enumerate(permutation):
                if (
                    process.legs[representative_slot].side
                    != process.legs[target_slot].side
                ):
                    raise RecurrenceProjectionError(
                        "topology replay maps an external source across the "
                        "initial/final crossing boundary"
                    )
            sector = color_plan.sector(int(sector_id))
            if sector is None:
                raise RecurrenceProjectionError(
                    f"topology replay references missing LC sector {sector_id}"
                )
            expected_weight = (
                2.0
                if isinstance(replay, LCColorTopologyReplayPlan)
                and color_plan.trace_reflections_folded
                and sector.kind == "single-trace"
                and len(sector.trace_labels) > 2
                else 1.0
            )
            if not math.isclose(
                float(weight),
                expected_weight,
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise RecurrenceProjectionError(
                    "topology replay carries an unsupported non-public-flow "
                    f"multiplicity {weight!r} for sector {sector_id}"
                )
            target_sector = physical_sector_by_id[int(sector_id)]
            target = RecurrenceReplayTargetV1(
                sector_id=int(sector_id),
                external_permutation=permutation,
                source_slot_permutation=permutation,
                # Rust derives and certifies source-state and closure
                # bijections from this exact slot permutation.
                amplitude_phase=ExactComplexRationalV1(1),
                fermion_sign=int(sign),
            )
            target_anchor = target_sector.closure_source_slot
            anchor_compatible = permutation[representative_anchor] == target_anchor
            source_replay = {
                "fermion_sign": int(sign),
                "partition_proof_algorithm": partition.proof_algorithm,
                "partition_proof_digest": partition.proof_digest,
                "representative_sector_id": representative_sector_id,
                "source_slot_permutation": permutation,
            }
            projected_targets.append(
                (target, anchor_compatible, int(expected_weight), source_replay)
            )

        if not isinstance(replay, LCColorTopologyReplayPlan) and any(
            not anchor_compatible for _, anchor_compatible, _, _ in projected_targets
        ):
            raise RecurrenceProjectionError(
                "topology replay does not map the representative closure "
                "anchor onto the target sector anchor"
            )
        representative_is_compatible = any(
            target.sector_id == representative_sector_id and anchor_compatible
            for target, anchor_compatible, _, _ in projected_targets
        )
        retained_targets = (
            tuple(
                target
                for target, anchor_compatible, _, _ in projected_targets
                if anchor_compatible
            )
            if representative_is_compatible
            else ()
        )
        if retained_targets:
            partitions.append(
                RecurrenceReplayPartitionV1(
                    representative_sector_id=representative_sector_id,
                    materialized_sector_id=int(partition.materialized_sector_id),
                    proof_algorithm=partition.proof_algorithm,
                    proof_digest=partition.proof_digest,
                    targets=retained_targets,
                )
            )
        for target, anchor_compatible, multiplicity, source_replay in projected_targets:
            if representative_is_compatible and anchor_compatible:
                continue
            target_sector = physical_sector_by_id[target.sector_id]
            mapped_word = tuple(
                target.source_slot_permutation[source_slot]
                for source_slot in representative_sector.word_source_slots
            )
            if mapped_word != target_sector.word_source_slots:
                raise RecurrenceProjectionError(
                    "topology replay does not map the representative LC word "
                    "onto the target sector word"
                )
            for representative_slot, target_slot in enumerate(
                target.source_slot_permutation
            ):
                representative_leg = external_legs[representative_slot]
                target_leg = external_legs[target_slot]
                representative_contracts = {
                    state.source_template_id: (
                        state.momentum_sign,
                        state.crossing_phase.canonical_key,
                    )
                    for state in representative_leg.source_states
                }
                target_contracts = {
                    state.source_template_id: (
                        state.momentum_sign,
                        state.crossing_phase.canonical_key,
                    )
                    for state in target_leg.source_states
                }
                if representative_contracts != target_contracts:
                    raise RecurrenceProjectionError(
                        "topology replay maps an external source onto one with "
                        "a different crossing contract"
                    )
            identity_materializations[target.sector_id] = (
                (
                    "closure-anchor-incompatible"
                    if representative_is_compatible
                    else "representative-closure-anchor-incompatible"
                ),
                multiplicity,
                source_replay,
            )

    if isinstance(replay, LCColorTopologyReplayPlan):
        for sector_id in replay.residual_sector_ids:
            sector = color_plan.sector(int(sector_id))
            if sector is None:
                raise RecurrenceProjectionError(
                    f"topology replay references missing residual LC sector {sector_id}"
                )
            multiplicity = (
                2
                if color_plan.trace_reflections_folded
                and sector.kind == "single-trace"
                and len(sector.trace_labels) > 2
                else 1
            )
            identity_materializations[int(sector_id)] = (
                "replay-plan-residual",
                multiplicity,
                None,
            )

    identity_permutation = tuple(range(len(process.legs)))
    for sector_id in sorted(identity_materializations):
        reason, multiplicity, source_replay = identity_materializations[sector_id]
        sector = physical_sector_by_id[sector_id]
        proof_payload = {
            "algorithm": _IDENTITY_REPLAY_PROOF_ALGORITHM,
            "closure_proof_algorithm": sector.closure_proof_algorithm,
            "closure_proof_digest": sector.closure_proof_digest,
            "closure_source_slot": sector.closure_source_slot,
            "identity_source_slot_permutation": identity_permutation,
            "policy": "materialize-physical-sector-with-identity-replay",
            "process_source_label_order": tuple(int(leg.label) for leg in process.legs),
            "public_flow_multiplicity": multiplicity,
            "reason": reason,
            "sector_id": sector_id,
            "source_replay": source_replay,
            "word_source_slots": sector.word_source_slots,
        }
        partitions.append(
            RecurrenceReplayPartitionV1(
                representative_sector_id=sector_id,
                materialized_sector_id=sector_id,
                proof_algorithm=_IDENTITY_REPLAY_PROOF_ALGORITHM,
                proof_digest=_digest(proof_payload),
                targets=(
                    RecurrenceReplayTargetV1(
                        sector_id=sector_id,
                        external_permutation=identity_permutation,
                        source_slot_permutation=identity_permutation,
                        amplitude_phase=ExactComplexRationalV1(1),
                        fermion_sign=1,
                    ),
                ),
            )
        )
    return tuple(partitions)


def _project_coupling_limits(
    limits: Mapping[str, int] | None,
) -> tuple[RecurrenceCouplingLimitV1, ...]:
    normalized = tuple(
        sorted(
            (str(raw_name).upper(), raw_maximum)
            for raw_name, raw_maximum in (limits or {}).items()
        )
    )
    names = tuple(name for name, _ in normalized)
    if len(names) != len(set(names)):
        raise RecurrenceProjectionError(
            "coupling-order names must be unique after canonicalization"
        )
    result: list[RecurrenceCouplingLimitV1] = []
    for name, raw_maximum in normalized:
        if not name:
            raise RecurrenceProjectionError("coupling-order names must be nonempty")
        if isinstance(raw_maximum, bool) or not isinstance(raw_maximum, int):
            raise RecurrenceProjectionError(
                f"coupling-order limit {name!r} must be an integer"
            )
        if raw_maximum < 0:
            raise RecurrenceProjectionError(
                f"coupling-order limit {name!r} must be nonnegative"
            )
        result.append(RecurrenceCouplingLimitV1(name, 0, raw_maximum))
    return tuple(result)


def _project_parameters(
    catalog: RecurrenceTemplateCatalog,
    template_ids: Mapping[tuple[str, str], int],
) -> tuple[RecurrenceParameterProjectionV1, ...]:
    runtime_parameters = tuple(
        sorted(
            (
                parameter
                for parameter in catalog.parameters
                if parameter.mutable or parameter.parameter_kind == "derived"
            ),
            key=lambda item: (item.name, item.template_id),
        )
    )
    names = tuple(parameter.name for parameter in runtime_parameters)
    if len(names) != len(set(names)):
        raise RecurrenceProjectionError(
            "recurrence catalog exposes duplicate runtime parameter names"
        )
    result: list[RecurrenceParameterProjectionV1] = []
    for parameter in runtime_parameters:
        components = (0,) if parameter.value_type == "real" else (0, 1)
        for component in components:
            result.append(
                RecurrenceParameterProjectionV1(
                    runtime_slot=len(result),
                    runtime_name=parameter.name,
                    parameter_template_id=_template_id(
                        template_ids, "parameter", parameter.template_id
                    ),
                    prepared_parameter_id=parameter.prepared_parameter_id,
                    component=component,
                )
            )
    return tuple(result)


def _template_id(
    template_ids: Mapping[tuple[str, str], int],
    kind: str,
    semantic_id: str,
) -> int:
    try:
        return template_ids[(kind, semantic_id)]
    except KeyError as exc:
        raise RecurrenceProjectionError(
            f"missing {kind} semantic template {semantic_id!r}"
        ) from exc


def _digest(payload: object) -> str:
    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise RecurrenceProjectionError(
            "recurrence projection input is not canonical JSON"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "RecurrenceGenerationSliceV1",
    "RecurrenceProjectionError",
    "project_recurrence_process_v1",
]

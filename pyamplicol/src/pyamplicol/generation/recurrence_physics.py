# SPDX-License-Identifier: 0BSD
"""Public physics and parameter metadata for compact recurrence artifacts.

The recurrence builder forks before :class:`GenericDAG`, so its public axes
must be derived from the authenticated process projection rather than from a
materialized DAG.  This module keeps that derivation small and deterministic.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from itertools import product
from types import SimpleNamespace
from typing import Any

from .._internal.versions import (
    RECURRENCE_BUILDER_INPUT_ABI,
    RECURRENCE_DIRECT_TEMPLATE_ABI,
    RECURRENCE_PLAN_ABI,
    RECURRENCE_RUNTIME_LAYOUT_ABI,
)
from ..color import (
    ColorContractionPlan,
    ColorGroupDescriptor,
    GenericColorPlan,
    build_color_contraction_plan,
    build_symmetric_group_color_contraction_plan,
    exact_color_contraction_factor,
)
from ..color.plan_types import _canonical_open_string_product_key
from ..models.base import Model, _runtime_particle_parameter_name
from ..models.recurrence_template import RecurrenceTemplateCatalog
from ..processes.ir import CanonicalProcessIR
from .recurrence_columnar import (
    ExactComplexRationalV1,
    RecurrenceBuilderLogicalInputV1,
    RecurrenceExternalLegV1,
    RecurrenceNormalizationV1,
    RecurrenceParameterProjectionV1,
    RecurrencePublicLCFlowV1,
)

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
_GLOBAL_HELICITY_FLIP_EQUIVALENCE_ROLE = "helicity-equivalence:global-flip-v1"
_ZERO_SECTOR_OWNER = 0xFFFF_FFFF


def build_recurrence_normalization(
    process: CanonicalProcessIR,
    model: Model,
) -> tuple[RecurrenceNormalizationV1, dict[str, object]]:
    """Bind the existing process normalization without constructing a DAG."""

    # Existing model implementations inspect only ``dag.process``.  Keep the
    # compatibility adapter local while preserving the established payload.
    payload = dict(
        model.runtime_normalization_payload(SimpleNamespace(process=process))
    )
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return (
        RecurrenceNormalizationV1(
            factor=ExactComplexRationalV1(1),
            convention="runtime-normalization-extension-v1",
            semantic_digest=hashlib.sha256(encoded).hexdigest(),
        ),
        payload,
    )


def build_recurrence_physics(
    process: CanonicalProcessIR,
    logical: RecurrenceBuilderLogicalInputV1,
    catalog: RecurrenceTemplateCatalog,
    *,
    process_id: str,
    resolved_helicities: Sequence[Sequence[int]],
    normalization: Mapping[str, object],
    selected_color_sector_ids: Sequence[int] | None,
    color_plan: GenericColorPlan | None = None,
) -> dict[str, object]:
    """Build strict ``pyamplicol-resolved-physics`` metadata pre-DAG."""

    if logical.process_id != process.key:
        raise ValueError("recurrence projection does not belong to the process IR")
    if process.color_accuracy != "lc" and color_plan is None:
        raise ValueError("contracted recurrence physics requires its color plan")
    if logical.layout != "all-flow-union" and not resolved_helicities:
        raise ValueError("fixed-source recurrence has no resolved helicities")

    possible_helicities = _possible_helicities(logical)
    resolved = {tuple(int(value) for value in row) for row in resolved_helicities}
    unknown = resolved - set(possible_helicities)
    if unknown:
        raise ValueError(
            "recurrence lowering returned helicities outside generated coverage: "
            f"{sorted(unknown)!r}"
        )
    # All-flow-union selects source states at runtime and deliberately has no
    # helicity-expanded destinations.  Every retained assignment is therefore
    # executable even though some may evaluate to zero dynamically.
    computed_helicities = (
        set(possible_helicities) if logical.layout == "all-flow-union" else resolved
    )
    parity_proof = next(
        (
            row
            for row in getattr(logical, "semantic_digests", ())
            if row.role == _GLOBAL_HELICITY_FLIP_EQUIVALENCE_ROLE
        ),
        None,
    )
    parity_reduced = parity_proof is not None and logical.layout != "all-flow-union"
    if parity_reduced:
        for values in resolved:
            flipped = tuple(-value for value in values)
            if flipped != values and flipped in resolved:
                raise ValueError(
                    "recurrence parity proof retained both members of global "
                    f"helicity-flip orbit {values!r} / {flipped!r}"
                )
    helicities = []
    structural_zero_count = 0
    parity_aliases: list[dict[str, str]] = []
    for index, values in enumerate(possible_helicities):
        identifier = _helicity_id(values)
        representative = values
        computed = values in computed_helicities
        if parity_reduced and not computed:
            flipped = tuple(-value for value in values)
            if flipped in resolved:
                representative = flipped
        structural_zero = not computed and representative == values
        structural_zero_count += int(structural_zero)
        representative_id = _helicity_id(representative)
        if representative != values:
            parity_aliases.append(
                {
                    "physical_id": identifier,
                    "representative_id": representative_id,
                }
            )
        helicities.append(
            {
                "id": identifier,
                "index": index,
                "values": list(values),
                "computed": computed,
                "structural_zero": structural_zero,
                "representative_id": representative_id,
                "coefficient": 0.0 if structural_zero else 1.0,
            }
        )
    if parity_reduced:
        nonzero_physical = len(helicities) - structural_zero_count
        if nonzero_physical != 2 * len(resolved) or len(parity_aliases) != len(
            resolved
        ):
            raise ValueError(
                "recurrence global-helicity-flip reduction does not form complete "
                "two-member nonzero orbits"
            )

    selected_flow_ids = (
        None
        if logical.selected_public_flow_ids is None
        else set(logical.selected_public_flow_ids)
    )
    selected_sector_ids = (
        None
        if selected_color_sector_ids is None
        else tuple(sorted({int(value) for value in selected_color_sector_ids}))
    )
    if (selected_flow_ids is None) != (selected_sector_ids is None):
        raise ValueError(
            "recurrence selected public flows and requested color sectors must "
            "be recorded together"
        )
    if selected_sector_ids == ():
        raise ValueError("recurrence selected color sectors cannot be empty")
    retained_flows = _retained_public_flows(logical)
    if not retained_flows:
        raise ValueError("recurrence physics has no retained public LC flow")
    labels_by_slot = {
        leg.source_slot: leg.public_label for leg in logical.external_legs
    }
    if process.color_accuracy == "lc":
        color_components = []
        for index, flow in enumerate(retained_flows):
            weight = _complex_factor(flow.reduction_weight)
            color_components.append(
                {
                    "kind": "lc-flow",
                    "id": flow.public_id,
                    "index": index,
                    "word": [labels_by_slot[slot] for slot in flow.word_source_slots],
                    "computed": True,
                    "representative_id": flow.public_id,
                    "coefficient": float(
                        weight.real * weight.real + weight.imag * weight.imag
                    ),
                }
            )
    else:
        color_components = [
            {
                "kind": "contracted-color",
                "id": "color:contracted",
                "index": 0,
                "description": (
                    "coherent sparse contraction of the complete ordered color basis"
                ),
            }
        ]

    selected_sources = {
        row.source_slot: tuple(row.source_state_indices)
        for row in logical.selected_source_coverage or ()
    }
    selected_source_helicities = {}
    for leg in logical.external_legs:
        retained = selected_sources.get(leg.source_slot)
        if retained is None:
            continue
        retained_helicities = {
            leg.source_states[index].public_helicity for index in retained
        }
        if len(retained_helicities) == 1:
            selected_source_helicities[str(leg.public_label)] = (
                retained_helicities.pop()
            )

    color_coverage = (
        ("complete" if selected_flow_ids is None else "selected")
        if process.color_accuracy == "lc"
        else "contracted"
    )
    helicity_coverage = (
        "complete" if logical.selected_source_coverage is None else "selected"
    )
    public_parameters = _public_model_parameters(catalog)
    return {
        "schema_version": 1,
        "kind": "pyamplicol-resolved-physics",
        "process_id": process_id,
        "process": process.process,
        "color_accuracy": process.color_accuracy,
        "coverage": {
            "helicities": helicity_coverage,
            "color": color_coverage,
            "color_kind": (
                "physical-lc-flows"
                if process.color_accuracy == "lc"
                else "contracted-color"
            ),
            "structural_zero_helicity_count": structural_zero_count,
        },
        "external_particles": [
            {
                "index": index,
                "label": int(leg.label),
                "particle": str(leg.particle),
                "pdg": int(leg.pdg),
                "role": "initial" if leg.is_initial else "final",
                "momentum_slot": index,
                "momentum_components": ["E", "px", "py", "pz"],
            }
            for index, leg in enumerate(process.legs)
        ],
        "helicities": helicities,
        "color_components": color_components,
        # The compact recurrence plan owns the high-cardinality destination
        # expansion.  Rusticol hydrates these groups while loading the plan.
        "reduction": {
            "kind": (
                "lc-diagonal" if process.color_accuracy == "lc" else "contracted-color"
            ),
            "groups": [],
        },
        "model_parameters": public_parameters,
        "selectors": {
            "helicity": True,
            "color_flow": process.color_accuracy == "lc",
            "contracted_color": False,
        },
        "extensions": {
            "process_key": process.key,
            "normalization": {
                key: normalization[key]
                for key in _NORMALIZATION_EXTENSION_KEYS
                if key in normalization
            },
            "selected_source_helicities": selected_source_helicities,
            "runtime_selectors": {
                "kind": "pyamplicol-runtime-selectors",
                "contract_version": 1,
                "provenance": RECURRENCE_PLAN_ABI,
                "axes": {
                    "helicity": {
                        "generation_coverage": helicity_coverage,
                        "generation_selection": selected_source_helicities,
                        "runtime_contract": (
                            "complete-reusable"
                            if helicity_coverage == "complete"
                            else "generation-specialized"
                        ),
                    },
                    "color_flow": {
                        "generation_coverage": color_coverage,
                        "generation_selection": list(selected_sector_ids or ()),
                        "runtime_contract": (
                            "complete-reusable"
                            if color_coverage == "complete"
                            else (
                                "generation-specialized"
                                if color_coverage == "selected"
                                else "contracted-color"
                            )
                        ),
                    },
                },
                "generation_specialized_axes": [
                    *([] if helicity_coverage == "complete" else ["helicity"]),
                    *(["color_flow"] if color_coverage == "selected" else []),
                ],
            },
            "recurrence_runtime_reduction": {
                "kind": "pyamplicol-recurrence-native-reduction-v2",
                "runtime_layout_abi": RECURRENCE_RUNTIME_LAYOUT_ABI,
                "container_path": "recurrence-runtime.pacbin",
                "plan_member_path": "schedule/recurrence-direct-schedule-v2.bin",
            },
            "recurrence": {
                "builder_input_abi": RECURRENCE_BUILDER_INPUT_ABI,
                "plan_abi": RECURRENCE_PLAN_ABI,
                "runtime_layout_abi": RECURRENCE_RUNTIME_LAYOUT_ABI,
                "direct_template_abi": RECURRENCE_DIRECT_TEMPLATE_ABI,
                "lc_flow_layout": logical.layout,
            },
            **(
                {}
                if not parity_reduced
                else {
                    "global_helicity_flip_reduction": {
                        "kind": "pyamplicol-global-helicity-flip-reduction-v1",
                        "proof_role": parity_proof.role,
                        "proof_digest": parity_proof.digest,
                        "physical_nonzero_helicity_count": (
                            len(helicities) - structural_zero_count
                        ),
                        "representative_helicity_count": len(resolved),
                        "aliases": parity_aliases,
                    }
                }
            ),
        },
    }


def build_recurrence_runtime_metadata(
    logical: RecurrenceBuilderLogicalInputV1,
    catalog: RecurrenceTemplateCatalog,
    model: Model,
    normalization: Mapping[str, object],
) -> dict[str, object]:
    """Return bounded source, parameter, and normalization runtime metadata."""

    return _build_recurrence_runtime_metadata(
        logical,
        catalog,
        model,
        normalization,
    )


def build_on_the_fly_public_metadata(
    process: CanonicalProcessIR,
    catalog: RecurrenceTemplateCatalog,
    *,
    process_id: str,
) -> dict[str, object]:
    """Return compact O(externals + parameters) public display metadata."""

    return {
        "schema_version": 1,
        "kind": "pyamplicol-on-the-fly-public-metadata",
        "process_id": process_id,
        "process": process.process,
        "color_accuracy": process.color_accuracy,
        "external_particles": [
            {
                "index": index,
                "label": int(leg.label),
                "particle": str(leg.particle),
                "pdg": int(leg.pdg),
                "role": "initial" if leg.is_initial else "final",
                "momentum_slot": index,
                "momentum_components": ["E", "px", "py", "pz"],
            }
            for index, leg in enumerate(process.legs)
        ],
        "model_parameters": _public_model_parameters(catalog),
    }


def build_on_the_fly_runtime_metadata(
    external_legs: Sequence[RecurrenceExternalLegV1],
    parameter_projection: Sequence[RecurrenceParameterProjectionV1],
    catalog: RecurrenceTemplateCatalog,
    model: Model,
    normalization: Mapping[str, object],
) -> dict[str, object]:
    """Return irreducible runtime support for the source-only OTF lane."""

    # The established recurrence implementation below already owns parameter
    # defaults, source masses, external-slot order, and normalization.  Feed it
    # only those shared structural rows and discard its recurrence-only source
    # replicas instead of introducing a second metadata implementation.
    support = _build_recurrence_runtime_metadata(
        SimpleNamespace(
            parameter_projection=tuple(parameter_projection),
            external_legs=tuple(external_legs),
            # This branch suppresses public flow projection; the on-the-fly
            # selector adapter derives that axis lazily from the compact seed.
            layout="contracted-color-union",
        ),
        catalog,
        model,
        normalization,
        include_source_templates=False,
    )
    return {
        name: support[name]
        for name in (
            "runtime_parameters",
            "prepared_parameter_defaults",
            "parameter_projection",
            "external_legs",
            "particle_masses",
            "normalization",
        )
    }


def _build_recurrence_runtime_metadata(
    logical: Any,
    catalog: RecurrenceTemplateCatalog,
    model: Model,
    normalization: Mapping[str, object],
    *,
    include_source_templates: bool = True,
) -> dict[str, object]:
    """Build shared prepared-runtime support and recurrence-only companions."""

    parameter_projection = logical.parameter_projection
    external_legs = logical.external_legs
    layout = logical.layout

    values_by_name: dict[str, complex] = {}
    for provider_name in (
        "runtime_parameter_defaults",
        "runtime_derived_parameter_defaults",
    ):
        provider = getattr(model, provider_name, None)
        if not callable(provider):
            continue
        for name, raw_value in provider().items():
            values_by_name[str(name)] = _complex_value(raw_value)

    prepared_defaults = [0j] * len(catalog.parameters)
    for parameter in catalog.parameters:
        prepared_id = parameter.prepared_parameter_id
        if prepared_id is None:
            continue
        if prepared_id >= len(prepared_defaults):
            raise ValueError(
                f"prepared parameter ID {prepared_id} exceeds catalog size"
            )
        value = values_by_name.get(parameter.name)
        if value is None and parameter.default_value is not None:
            value = _complex_factor(parameter.default_value)
        if value is None:
            raise ValueError(
                f"recurrence prepared parameter {parameter.name!r} has no default"
            )
        prepared_defaults[prepared_id] = value

    runtime_parameters = []
    for projection in sorted(
        parameter_projection, key=lambda item: item.runtime_slot
    ):
        parameter = catalog.parameters[projection.parameter_template_id]
        if parameter.name != projection.runtime_name:
            raise ValueError(
                "recurrence parameter projection name disagrees with its template"
            )
        value = values_by_name.get(parameter.name)
        if value is None and parameter.default_value is not None:
            value = _complex_factor(parameter.default_value)
        if value is None:
            raise ValueError(
                f"recurrence runtime parameter {parameter.name!r} has no default"
            )
        is_complex = parameter.value_type == "complex"
        component_name = "real" if projection.component == 0 else "imag"
        runtime_parameters.append(
            {
                "name": (
                    f"{parameter.name}.{component_name}"
                    if is_complex
                    else parameter.name
                ),
                "kind": (
                    "derived_parameter_component"
                    if parameter.parameter_kind == "derived"
                    else parameter.parameter_kind
                ),
                "parameter_index": projection.runtime_slot,
                "default": float(
                    value.real if projection.component == 0 else value.imag
                ),
                "runtime_name": parameter.name if is_complex else None,
                "complex_component": component_name if is_complex else None,
            }
        )

    state_index_by_id = {
        state.template_id: index for index, state in enumerate(catalog.current_states)
    }
    referenced_source_ids = sorted(
        {
            state.source_template_id
            for leg in external_legs
            for state in leg.source_states
        }
    )
    source_templates = []
    particle_masses: dict[int, float] = {}
    parameter_names = {parameter.name for parameter in catalog.parameters}
    for source_template_id in referenced_source_ids:
        source = catalog.sources[source_template_id]
        state = next(
            item
            for item in catalog.current_states
            if item.template_id == source.state_template_id
        )
        source_ir = model._source_ir(state.particle_id)
        particle = model.particle(state.particle_id)
        mass_parameter = _runtime_particle_parameter_name(
            source_ir.mass_parameter,
            particle_pdg=int(particle.pdg),
            kind="mass",
            available_names=parameter_names,
        )
        if include_source_templates:
            width_parameter = _runtime_particle_parameter_name(
                source_ir.width_parameter,
                particle_pdg=int(particle.pdg),
                kind="width",
                available_names=parameter_names,
            )
            try:
                crossing = _runtime_crossing(json.loads(source.crossing))
            except json.JSONDecodeError as exc:  # pragma: no cover - validated
                raise ValueError(
                    f"recurrence source {source.template_id!r} has malformed "
                    "crossing JSON"
                ) from exc
            source_ir_payload = source_ir.to_json_dict()
            # Built-in models expose particle masses directly rather than
            # through named SourceIR parameters. Prepared kernels use the
            # generic particle fallback; retain the same name here.
            source_ir_payload["mass_parameter"] = mass_parameter
            source_ir_payload["width_parameter"] = width_parameter
            if crossing != source_ir_payload["crossing"]:
                raise ValueError(
                    f"recurrence source {source.template_id!r} crossing disagrees "
                    "with its typed SourceIR"
                )
            source_templates.append(
                {
                    "source_template_id": source_template_id,
                    "current_state_template_id": state_index_by_id[state.template_id],
                    "dimension": state.dimension,
                    "helicity": source.helicity,
                    "chirality": state.chirality,
                    "spin_state": source.spin_state,
                    "source_ir": source_ir_payload,
                    "crossing": crossing,
                }
            )
        mass = float(model.mass(state.particle_id))
        if mass_parameter is not None:
            mass_value = values_by_name.get(mass_parameter)
            if mass_value is not None:
                if mass_value.imag != 0.0:
                    raise ValueError(
                        "recurrence source mass parameter must be real: "
                        f"{mass_parameter!r}"
                    )
                mass = float(mass_value.real)
        previous = particle_masses.setdefault(state.particle_id, mass)
        if previous != mass:
            raise ValueError(
                "recurrence particle "
                f"{state.particle_id} has inconsistent source masses"
            )

    return {
        "public_color_flows": (
            [
                {
                    "public_id": flow.public_id,
                    "construction_sector_id": flow.construction_sector_id,
                    "target_sector_id": target_sector_id,
                }
                for target_sector_id, flow in enumerate(_retained_public_flows(logical))
            ]
            if layout != "contracted-color-union"
            else []
        ),
        "runtime_parameters": runtime_parameters,
        "prepared_parameter_defaults": [
            [float(value.real), float(value.imag)] for value in prepared_defaults
        ],
        "parameter_projection": [
            {
                "runtime_slot": row.runtime_slot,
                "runtime_name": row.runtime_name,
                "parameter_template_id": row.parameter_template_id,
                "prepared_parameter_id": row.prepared_parameter_id,
                "component": row.component,
            }
            for row in parameter_projection
        ],
        "source_templates": source_templates,
        "external_legs": [
            {
                "source_slot": leg.source_slot,
                "public_label": leg.public_label,
                "physical_pdg": leg.physical_pdg,
                "outgoing_pdg": leg.outgoing_pdg,
                "is_initial": leg.is_initial,
            }
            for leg in external_legs
        ],
        "particle_masses": [
            {"outgoing_pdg": pdg, "mass": mass}
            for pdg, mass in sorted(particle_masses.items())
        ],
        "normalization": {
            key: normalization[key]
            for key in _NORMALIZATION_EXTENSION_KEYS
            if key in normalization
        },
    }


def build_recurrence_color_contraction(
    logical: RecurrenceBuilderLogicalInputV1,
    color_plan: GenericColorPlan,
    resolved_helicities: Sequence[Sequence[int]],
    amplitude_destinations: Sequence[tuple[int, int | None]],
    certified_structural_zero_sector_ids: Sequence[int],
    *,
    contraction: str = "direct",
) -> ColorContractionPlan | None:
    if color_plan.color_accuracy == "lc":
        return None
    if logical.layout != "contracted-color-union":
        raise ValueError("NLC/full recurrence requires contracted-color-union")
    if not logical.physical_sectors or not resolved_helicities:
        raise ValueError("contracted recurrence has an empty color/helicity domain")
    contraction_destinations = recurrence_color_contraction_destinations(
        logical,
        resolved_helicities,
        amplitude_destinations,
    )
    labels_by_slot = {
        leg.source_slot: leg.public_label for leg in logical.external_legs
    }
    sectors_by_id = {sector.sector_id: sector for sector in logical.physical_sectors}
    descriptors = []
    for group_id, (sector_id, helicity_id) in enumerate(contraction_destinations):
        try:
            sector = sectors_by_id[sector_id]
        except KeyError as exc:
            raise ValueError(
                f"contracted recurrence destination {group_id} references "
                f"unknown physical sector {sector_id}"
            ) from exc
        if helicity_id is None:
            raise ValueError(
                f"contracted recurrence destination {group_id} has no helicity"
            )
        try:
            helicity = resolved_helicities[helicity_id]
        except IndexError as exc:
            raise ValueError(
                f"contracted recurrence destination {group_id} references "
                f"unknown resolved helicity {helicity_id}"
            ) from exc
        descriptors.append(
            ColorGroupDescriptor(
                group_id=group_id,
                helicity_key=tuple(int(value) for value in helicity),
                sector_id=sector.sector_id,
                word=tuple(labels_by_slot[slot] for slot in sector.word_source_slots),
                helicity_weight=1.0,
            )
        )
    if not descriptors:
        raise ValueError("contracted recurrence has no nonzero amplitude destination")
    if contraction == "direct":
        contraction_plan = build_color_contraction_plan(
            color_plan,
            tuple(descriptors),
        )
    elif contraction in {"auto", "symmetric-group-fft"}:
        active_sector_ids = {descriptor.sector_id for descriptor in descriptors}
        contraction_plan = build_symmetric_group_color_contraction_plan(
            color_plan,
            tuple(descriptors),
            sector_owner_ids=recurrence_color_sector_owner_map(
                logical,
                active_sector_ids,
                set(certified_structural_zero_sector_ids),
            ),
        )
        if contraction == "auto" and not contraction_plan.supported:
            contraction_plan = build_color_contraction_plan(
                color_plan, tuple(descriptors)
            )
    else:
        raise ValueError(f"unknown color contraction method {contraction!r}")
    if contraction_plan is None or not contraction_plan.supported:
        reason = None if contraction_plan is None else contraction_plan.reason
        raise ValueError(
            f"could not build recurrence color contraction: {reason or 'unsupported'}"
        )
    return contraction_plan


def build_on_the_fly_color_contraction(
    color_plan: GenericColorPlan,
    *,
    contraction: str = "direct",
) -> tuple[ColorContractionPlan, tuple[int, ...], tuple[int, ...]]:
    """Build the one-component metric over OTF structural color selectors.

    Full/NLC color plans explicitly contain whole-open-string block-order
    replicas.  The native query decoder already erases that traversal-only
    ordering, so one runtime amplitude is requested for each canonical tensor
    owner and the color metric contracts those owner amplitudes into one public
    result.
    """

    if color_plan.color_accuracy not in {"nlc", "full"}:
        raise ValueError("on-the-fly color contraction requires NLC or full color")
    if color_plan.trace_reflections_folded:
        raise ValueError(
            "on-the-fly contracted color cannot fold trace reflections"
        )
    owner_by_sector = on_the_fly_color_sector_owner_map(color_plan)
    owner_sector_ids = tuple(
        sector_id
        for sector_id, owner_id in enumerate(owner_by_sector)
        if sector_id == owner_id
    )
    sector_by_id = {int(sector.id): sector for sector in color_plan.sectors}
    descriptors = tuple(
        ColorGroupDescriptor(
            group_id=group_id,
            # All structural selectors contribute to the same contracted
            # component.  A single component forces expanded v3 storage.
            helicity_key=(),
            sector_id=sector_id,
            word=tuple(sector_by_id[sector_id].color_words[0]),
            helicity_weight=1.0,
        )
        for group_id, sector_id in enumerate(owner_sector_ids)
    )
    if contraction == "direct":
        contraction_plan = build_color_contraction_plan(color_plan, descriptors)
    elif contraction in {"auto", "symmetric-group-fft"}:
        contraction_plan = build_symmetric_group_color_contraction_plan(
            color_plan,
            descriptors,
            sector_owner_ids=owner_by_sector,
        )
        if contraction == "auto" and not contraction_plan.supported:
            contraction_plan = build_color_contraction_plan(color_plan, descriptors)
    else:
        raise ValueError(f"unknown color contraction method {contraction!r}")
    if contraction_plan is None or not contraction_plan.supported:
        reason = None if contraction_plan is None else contraction_plan.reason
        raise ValueError(
            "could not build on-the-fly color contraction: "
            f"{reason or 'unsupported'}"
        )
    if contraction_plan.repeated_block is not None:
        raise ValueError(
            "one-component on-the-fly color contraction must use expanded storage"
        )
    return contraction_plan, owner_by_sector, owner_sector_ids


def on_the_fly_color_sector_owner_map(
    color_plan: GenericColorPlan,
) -> tuple[int, ...]:
    """Return each full/NLC sector's canonical structural-selector owner.

    Only permutations of complete open strings are aliases.  Reversing a
    string, trace orientation, and every other physical sector distinction are
    retained.  Owner sectors appear in the same order as the lazy compact LC
    selector axis, including an optional ordinary reference flow moved first.
    """

    sectors = tuple(sorted(color_plan.sectors, key=lambda item: int(item.id)))
    if tuple(int(sector.id) for sector in sectors) != tuple(range(len(sectors))):
        raise ValueError("on-the-fly physical color sectors are not densely numbered")
    if not sectors:
        raise ValueError("on-the-fly contracted color plan has no physical sectors")

    owner_by_key: dict[tuple[object, ...], int] = {}
    owners: list[int] = []
    for sector in sectors:
        sector_id = int(sector.id)
        if sector.kind == "open-lines":
            open_strings = _canonical_open_string_product_key(
                (
                    line.fundamental_label,
                    line.adjoint_labels,
                    line.antifundamental_label,
                    line.singlet_labels,
                )
                for line in sector.open_color_lines
            )
            key: tuple[object, ...] = ("open-lines", open_strings)
        else:
            key = (sector.kind, sector_id)
        owners.append(owner_by_key.setdefault(key, sector_id))
    return tuple(owners)


def recurrence_color_contraction_destinations(
    logical: RecurrenceBuilderLogicalInputV1,
    resolved_helicities: Sequence[Sequence[int]],
    amplitude_destinations: Sequence[tuple[int, int | None]],
) -> tuple[tuple[int, int | None], ...]:
    """Return the physical amplitude domain consumed by color contraction.

    A contracted topology-replay schedule materializes only one representative
    per certified partition.  Color contraction still acts on the complete
    physical color basis, so its destination domain is the dense
    sector-major/helicity-minor replay scratch rather than the smaller Direct
    schedule destination table.
    """

    if logical.layout != "contracted-color-union" or not logical.replay_partitions:
        return tuple(amplitude_destinations)
    return tuple(
        (sector.sector_id, helicity_id)
        for sector in sorted(logical.physical_sectors, key=lambda item: item.sector_id)
        for helicity_id in range(len(resolved_helicities))
    )


def recurrence_color_sector_owner_map(
    logical: RecurrenceBuilderLogicalInputV1,
    active_sector_ids: set[int],
    certified_structural_zero_sector_ids: set[int],
) -> tuple[int, ...]:
    """Return the independently derived canonical owner of every color sector.

    Contracted recurrence construction treats permutations of complete open
    strings as aliases of the same product of color tensors.  This map binds
    that reduction into the process artifact so the loader can reject a
    schedule that silently omits, duplicates, or changes an alias class.
    """

    sectors = tuple(sorted(logical.physical_sectors, key=lambda item: item.sector_id))
    if tuple(sector.sector_id for sector in sectors) != tuple(range(len(sectors))):
        raise ValueError("recurrence physical color sectors are not densely numbered")

    owner_by_key: dict[tuple[object, ...], int] = {}
    owners: list[int] = []
    for sector in sectors:
        if sector.kind == "open-lines":
            open_strings = _canonical_open_string_product_key(
                (
                    string.fundamental_source_slot,
                    string.adjoint_source_slots,
                    string.antifundamental_source_slot,
                    string.singlet_source_slots,
                )
                for string in sector.open_strings
            )
            key: tuple[object, ...] = ("open-lines", open_strings)
        else:
            # Trace orientation and singlet sectors are already canonicalized
            # by their construction-sector identity.  Rust does not alias them.
            key = (sector.kind, sector.sector_id)
        owner = owner_by_key.setdefault(key, sector.sector_id)
        owners.append(owner)
    active = set(active_sector_ids)
    unknown = active.difference(range(len(sectors)))
    if unknown:
        raise ValueError(
            f"recurrence color destinations reference unknown sectors {sorted(unknown)}"
        )
    certified = set(certified_structural_zero_sector_ids)
    unknown_certified = certified.difference(range(len(sectors)))
    if unknown_certified:
        raise ValueError(
            "recurrence structural-zero certificate references unknown sectors "
            f"{sorted(unknown_certified)}"
        )
    overlap = certified.intersection(active)
    if overlap:
        raise ValueError(
            "recurrence structural-zero certificate overlaps active sectors "
            f"{sorted(overlap)}"
        )
    members_by_owner: dict[int, set[int]] = {}
    for sector_id, owner_id in enumerate(owners):
        members_by_owner.setdefault(owner_id, set()).add(sector_id)
    result = []
    for _sector_id, owner_id in enumerate(owners):
        if owner_id in active:
            result.append(owner_id)
            continue
        active_members = members_by_owner[owner_id].intersection(active)
        if active_members:
            raise ValueError(
                f"recurrence color sector class owned by {owner_id} has active "
                f"non-owner sectors {sorted(active_members)}"
            )
        uncertified = members_by_owner[owner_id].difference(certified)
        if uncertified:
            raise ValueError(
                f"recurrence color sector class owned by {owner_id} has no active "
                "destination and no complete structural-zero certificate; "
                f"uncertified sectors={sorted(uncertified)}"
            )
        result.append(_ZERO_SECTOR_OWNER)
    return tuple(result)


def recurrence_exact_color_coefficients(
    color_plan: GenericColorPlan,
    contraction: ColorContractionPlan,
    group_sector_ids: Sequence[int],
) -> tuple[ExactComplexRationalV1, ...]:
    """Return exact symmetry-folded coefficients in compact entry order."""

    symmetric_group = contraction.symmetric_group_block
    if symmetric_group is not None:
        entries = (
            *symmetric_group.kernel_entries,
            *symmetric_group.residual_entries,
        )
        exact_weights = (
            *symmetric_group.kernel_exact_weights,
            *symmetric_group.residual_exact_weights,
        )
        result = []
        for entry, exact in zip(entries, exact_weights, strict=True):
            symmetry = int(entry.symmetry_factor)
            if symmetry not in {1, 2} or float(symmetry) != entry.symmetry_factor:
                raise ValueError(
                    "symmetric-group color contraction has a noncanonical "
                    "symmetry factor"
                )
            result.append(
                ExactComplexRationalV1(
                    real_numerator=exact.numerator * symmetry,
                    real_denominator=exact.denominator,
                )
            )
        return tuple(result)

    sector_by_id = {sector.id: sector for sector in color_plan.sectors}
    repeated = contraction.repeated_block
    entries = contraction.entries if repeated is None else repeated.entries
    component_count = 1 if repeated is None else repeated.component_count

    result = []
    for entry in entries:
        if repeated is None:
            left_group_id = entry.left_group_id
            right_group_id = entry.right_group_id
        else:
            left_group_id = repeated.component_group_ids[
                entry.left_group_index * component_count
            ]
            right_group_id = repeated.component_group_ids[
                entry.right_group_index * component_count
            ]
        try:
            left_sector = sector_by_id[group_sector_ids[left_group_id]]
            right_sector = sector_by_id[group_sector_ids[right_group_id]]
        except (IndexError, KeyError) as exc:
            raise ValueError(
                "recurrence color entry references an unknown physical sector"
            ) from exc
        exact = exact_color_contraction_factor(
            color_plan,
            left_sector,
            right_sector,
            accuracy=contraction.color_accuracy,
            full_col_acc=20,
        )
        symmetry = entry.symmetry_factor
        if symmetry not in {1.0, 2.0}:
            raise ValueError(
                "recurrence color contraction has a non-integral symmetry factor"
            )
        folded = exact * int(symmetry)
        result.append(
            ExactComplexRationalV1(
                real_numerator=folded.numerator,
                real_denominator=folded.denominator,
            )
        )
    return tuple(result)


def _retained_public_flows(
    logical: RecurrenceBuilderLogicalInputV1,
) -> tuple[RecurrencePublicLCFlowV1, ...]:
    selected = (
        None
        if logical.selected_public_flow_ids is None
        else set(logical.selected_public_flow_ids)
    )
    retained = tuple(
        flow
        for flow in logical.public_flows
        if selected is None or flow.flow_id in selected
    )
    if not retained:
        raise ValueError("recurrence metadata has no retained public LC flow")
    return retained


def recurrence_referenced_kernel_ids(
    logical: RecurrenceBuilderLogicalInputV1,
) -> frozenset[int]:
    """Return exactly the prepared kernels named by process template references."""

    return frozenset(
        reference.prepared_kernel_id
        for reference in logical.semantic_template_references
        if reference.prepared_kernel_id is not None
    )


def _possible_helicities(
    logical: RecurrenceBuilderLogicalInputV1,
) -> tuple[tuple[int, ...], ...]:
    selected = {
        row.source_slot: set(row.source_state_indices)
        for row in logical.selected_source_coverage or ()
    }
    choices = []
    for leg in logical.external_legs:
        retained = selected.get(leg.source_slot)
        choices.append(
            tuple(
                state.public_helicity
                for state in leg.source_states
                if retained is None or state.state_index in retained
            )
        )
    return tuple(sorted(product(*choices)))


def _public_model_parameters(
    catalog: RecurrenceTemplateCatalog,
) -> list[dict[str, object]]:
    result = []
    for parameter in sorted(catalog.parameters, key=lambda item: item.name):
        if parameter.parameter_kind == "constant":
            continue
        default = (
            0j
            if parameter.default_value is None
            else _complex_factor(parameter.default_value)
        )
        if parameter.name.startswith("normalization."):
            kind = "normalization"
        elif parameter.parameter_kind == "derived":
            kind = "derived"
        else:
            kind = "external"
        result.append(
            {
                "name": parameter.name,
                "kind": kind,
                "default_real": float(default.real),
                "default_imaginary": float(default.imag),
                "mutable": bool(parameter.mutable),
            }
        )
    return result


def _complex_factor(value: object) -> complex:
    return complex(
        int(value.real_numerator) / int(value.real_denominator),
        int(value.imag_numerator) / int(value.imag_denominator),
    )


def _runtime_crossing(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("recurrence source crossing must be an object")
    result = dict(value)
    phase = result.get("phase")
    if not isinstance(phase, Mapping):
        raise ValueError("recurrence source crossing phase must be exact")
    result["phase"] = [
        int(phase["real_numerator"]) / int(phase["real_denominator"]),
        int(phase["imag_numerator"]) / int(phase["imag_denominator"]),
    ]
    return result


def _complex_value(value: object) -> complex:
    if isinstance(value, tuple):
        return complex(*value)
    return complex(value)  # type: ignore[arg-type]


def _helicity_id(values: Sequence[int]) -> str:
    return "h:" + ",".join(f"{int(value):+d}" for value in values)


__all__ = [
    "build_on_the_fly_public_metadata",
    "build_on_the_fly_runtime_metadata",
    "build_recurrence_color_contraction",
    "build_recurrence_normalization",
    "build_recurrence_physics",
    "build_recurrence_runtime_metadata",
    "recurrence_color_contraction_destinations",
    "recurrence_color_sector_owner_map",
    "recurrence_exact_color_coefficients",
    "recurrence_referenced_kernel_ids",
]

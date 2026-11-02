# SPDX-License-Identifier: 0BSD
"""Entry point for compiling external UFO model IR."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace

from .._internal.physics.symbols import ModelSymbolRegistry, symbols
from . import compiler_symbolica as _sym
from .compiler_auxiliary_components import reduce_contact_auxiliary_components
from .compiler_color_flow import (
    compile_lc_color_transition_terms,
    synthesize_fundamental_fierz_auxiliaries,
)
from .compiler_contact_trees import (
    _compile_color_singlet_contact_trees,
    _compile_heft_colored_contact_trees,
    _deduplicate_contact_partials,
    _record_heft_contact_orbit_certificates,
)
from .compiler_contacts import (
    _build_contact_decomposition_proof,
    _contact_final_component_expressions,
    _contact_partial_component_expressions,
    _contact_term_has_literal_color_singlet,
    _fuse_contact_finals,
    _record_contact_decomposition_proofs,
    _validated_contact_decomposition_proof,
)
from .compiler_contractions import compile_contraction_records
from .compiler_gauge import compile_goldstone_partner_records
from .compiler_kernels import (
    _canonicalize_oriented_kernel_component,
    _fuse_oriented_kernels,
    _is_compile_time_zero_parameter,
    _is_single_structure_constant,
    _lc_color_normalization_power,
    _oriented_component_expressions,
    _remap_kernel_symbols,
    _replace_expression_symbols,
    _spin_axis_labels,
)
from .compiler_records import (
    _coupling,
    _mappings,
    _order,
    _parameter,
    _particle,
    _propagator,
    _resolve_coupling_records,
    _resolve_parameter_records,
    _sequence,
)
from .compiler_tensor_ordering import compile_tensor_ordering_metadata
from .contact_decomposition import (
    CompiledContactDecompositionSplit,
    CompiledContactOrbitStep,
    CompiledContactOrientationProof,
    compiled_contact_orbit_step,
)
from .contracts import (
    CompiledModelIR,
    CompiledOrientedKernel,
    CompiledParameterRecord,
    CompiledParticleRecord,
    CompiledPropagatorRecord,
    CompiledVertexTerm,
    compiled_particle_component_dimension,
)
from .tensors import (
    classify_trilinear_color_expression,
    normalize_color_expression,
    normalize_lorentz_expression,
    project_trilinear_color_expression,
)

_SUPPORTED_TRILINEAR_COLOR_PROJECTIONS = frozenset(
    {
        "singlet",
        "color-identity",
        "fundamental-generator",
        "adjoint-structure-constant",
    }
)


def compile_ufo_model_ir(model: Mapping[str, object]) -> CompiledModelIR:
    _sym._ensure_symbolica()
    model_name = str(model.get("name", "unnamed-model"))
    model_symbols = symbols.model(model_name)
    particles = tuple(_particle(item) for item in _mappings(model.get("particles")))
    particle_by_name = {particle.name: particle for particle in particles}
    lorentz_by_name = {
        str(item["name"]): item for item in _mappings(model.get("lorentz_structures"))
    }
    parameter_records = tuple(
        _parameter(item, model_symbols=model_symbols)
        for item in _mappings(model.get("parameters"))
    )
    parameter_records = _resolve_parameter_records(parameter_records, model_symbols)
    coupling_records = tuple(
        _coupling(item, model_symbols=model_symbols)
        for item in _mappings(model.get("couplings"))
    )
    coupling_records = _resolve_coupling_records(
        coupling_records,
        parameter_records,
        model_symbols,
    )
    coupling_by_name = {coupling.name: coupling for coupling in coupling_records}
    terms: list[CompiledVertexTerm] = []
    for vertex in _mappings(model.get("vertex_rules")):
        particle_names = tuple(str(value) for value in _sequence(vertex["particles"]))
        try:
            vertex_particles = tuple(particle_by_name[name] for name in particle_names)
        except KeyError as exc:
            raise ValueError(
                f"vertex {vertex.get('name')} refers to unknown particle {exc.args[0]}"
            ) from exc
        colors = tuple(particle.color for particle in vertex_particles)
        color_sources = tuple(
            str(value) for value in _sequence(vertex["color_structures"])
        )
        lorentz_names = tuple(
            str(value) for value in _sequence(vertex["lorentz_structures"])
        )
        coupling_matrix = _sequence(vertex["couplings"])
        if len(coupling_matrix) != len(color_sources):
            raise ValueError(
                f"vertex {vertex.get('name')} coupling rows do not match "
                "color structures"
            )
        normalized_colors = tuple(
            normalize_color_expression(source, colors) for source in color_sources
        )
        normalized_lorentz = []
        for name in lorentz_names:
            try:
                lorentz = lorentz_by_name[name]
            except KeyError as exc:
                raise ValueError(
                    f"vertex {vertex.get('name')} refers to unknown Lorentz "
                    f"structure {name}"
                ) from exc
            normalized_lorentz.append(
                normalize_lorentz_expression(
                    str(lorentz["structure"]),
                    tuple(int(value) for value in _sequence(lorentz["spins"])),
                    model_symbols=model_symbols,
                )
            )
        for color_index, row_value in enumerate(coupling_matrix):
            row = _sequence(row_value)
            if len(row) != len(lorentz_names):
                raise ValueError(
                    f"vertex {vertex.get('name')} coupling columns do not match "
                    "Lorentz structures"
                )
            for lorentz_index, coupling_value in enumerate(row):
                if coupling_value is None:
                    continue
                coupling_name = str(coupling_value)
                try:
                    coupling = coupling_by_name[coupling_name]
                except KeyError as exc:
                    raise ValueError(
                        f"vertex {vertex.get('name')} refers to unknown coupling "
                        f"{coupling_name}"
                    ) from exc
                source_lorentz = lorentz_by_name[lorentz_names[lorentz_index]]
                terms.append(
                    CompiledVertexTerm(
                        id=len(terms),
                        vertex=str(vertex["name"]),
                        particles=particle_names,
                        color_index=color_index,
                        lorentz_index=lorentz_index,
                        color_source=color_sources[color_index],
                        color_expression=normalized_colors[color_index].expression,
                        lorentz_name=lorentz_names[lorentz_index],
                        lorentz_source=str(source_lorentz["structure"]),
                        lorentz_expression=normalized_lorentz[lorentz_index].expression,
                        coupling=coupling.name,
                        coupling_expression=coupling.resolved_expression,
                        coupling_orders=coupling.orders,
                        lc_color_normalization_power=(
                            _lc_color_normalization_power(color_sources[color_index])
                        ),
                    )
                )
    propagators = tuple(
        _propagator(item, particles) for item in _mappings(model.get("propagators"))
    )
    terms = list(
        _record_contact_decomposition_proofs(
            terms,
            particles,
            model_symbols=model_symbols,
        )
    )
    terms = list(
        _record_heft_contact_orbit_certificates(
            terms,
            particles,
            model_symbols=model_symbols,
        )
    )
    oriented_kernels = _compile_oriented_kernels(
        terms,
        particles,
        parameter_records,
        propagators,
        model_symbols,
    )
    contact_particles, contact_kernels = _compile_four_point_contact_kernels(
        terms,
        particles,
        start_kind=len(oriented_kernels),
        model_symbols=model_symbols,
    )
    contact_particles, contact_kernels = _deduplicate_contact_partials(
        contact_particles,
        contact_kernels,
        terms,
        model_symbols=model_symbols,
    )
    contact_kernels = _fuse_contact_finals(
        contact_kernels,
        terms,
        model_symbols=model_symbols,
    )
    particles = (*particles, *contact_particles)
    oriented_kernels = (*oriented_kernels, *contact_kernels)
    tree_start_kind = max((kernel.kind for kernel in oriented_kernels), default=-1) + 1
    tree_particles, tree_kernels = _compile_color_singlet_contact_trees(
        terms,
        particles,
        start_kind=tree_start_kind,
        model_symbols=model_symbols,
    )
    particles = (*particles, *tree_particles)
    oriented_kernels = (*oriented_kernels, *tree_kernels)
    heft_start_kind = max((kernel.kind for kernel in oriented_kernels), default=-1) + 1
    heft_particles, heft_kernels = _compile_heft_colored_contact_trees(
        terms,
        particles,
        start_kind=heft_start_kind,
        model_symbols=model_symbols,
    )
    particles = (*particles, *heft_particles)
    oriented_kernels = (*oriented_kernels, *heft_kernels)
    particles, oriented_kernels = reduce_contact_auxiliary_components(
        particles,
        oriented_kernels,
        auxiliary_particles=(*tree_particles, *heft_particles),
        model_symbols=model_symbols,
    )
    (
        annotated_terms,
        oriented_kernels,
        tensor_orderings,
        current_orderings,
    ) = compile_tensor_ordering_metadata(
        terms,
        particles,
        oriented_kernels,
        parameter_records,
        propagators,
    )
    terms = list(annotated_terms)
    oriented_kernels = _annotate_oriented_kernel_color_projections(
        oriented_kernels,
        particles,
        terms,
    )
    particles, oriented_kernels = synthesize_fundamental_fierz_auxiliaries(
        particles,
        oriented_kernels,
        propagators,
        model_symbols=model_symbols,
    )
    (
        annotated_terms,
        oriented_kernels,
        tensor_orderings,
        current_orderings,
    ) = compile_tensor_ordering_metadata(
        terms,
        particles,
        oriented_kernels,
        parameter_records,
        propagators,
    )
    terms = list(annotated_terms)
    oriented_kernels = _annotate_oriented_kernel_evaluation_equivalence(
        oriented_kernels,
        particles,
        terms,
        model_symbols,
    )
    direct_contractions, closure_contractions = compile_contraction_records(
        particles,
        parameter_records,
        propagators,
    )
    goldstone_partners = compile_goldstone_partner_records(
        particles,
        parameter_records,
        propagators,
    )
    return CompiledModelIR(
        name=model_name,
        orders=tuple(_order(item) for item in _mappings(model.get("orders"))),
        parameters=parameter_records,
        particles=particles,
        couplings=coupling_records,
        propagators=propagators,
        vertex_terms=tuple(terms),
        oriented_kernels=oriented_kernels,
        direct_contractions=direct_contractions,
        closure_contractions=closure_contractions,
        tensor_orderings=tensor_orderings,
        current_orderings=current_orderings,
        goldstone_partners=goldstone_partners,
    )


def _annotate_oriented_kernel_color_projections(
    kernels: Sequence[CompiledOrientedKernel],
    particles: Sequence[CompiledParticleRecord],
    terms: Sequence[CompiledVertexTerm],
) -> tuple[CompiledOrientedKernel, ...]:
    particle_by_name = {particle.name: particle for particle in particles}
    term_by_id = {term.id: term for term in terms}
    annotated: list[CompiledOrientedKernel] = []
    for kernel in kernels:
        term = term_by_id.get(kernel.term_id)
        oriented_representations = tuple(
            particle_by_name[name].color for name in kernel.particles
        )
        if term is not None and term.valence == 3 and kernel.vertex == term.vertex:
            representations = tuple(
                particle_by_name[name].color for name in term.particles
            )
            certified_structure, coefficient = project_trilinear_color_expression(
                term.color_expression,
                representations,
            )
            if certified_structure not in _SUPPORTED_TRILINEAR_COLOR_PROJECTIONS:
                raise ValueError(
                    f"vertex {term.vertex!r} has unsupported trilinear color "
                    f"tensor {term.color_source!r}; pyAmpliCol could not prove it "
                    "as a singlet, identity, fundamental generator, or adjoint "
                    "structure constant"
                )
            structure, _ = classify_trilinear_color_expression(
                kernel.color_expression,
                kernel.color_source,
                oriented_representations,
                allow_source_fallback=True,
            )
            if structure not in {certified_structure, "generic-tensor"}:
                raise ValueError(
                    f"vertex {term.vertex!r} changed color-tensor family while "
                    "being oriented"
                )
            if structure == "generic-tensor":
                # Numeric prefactors and the dual representation carried by an
                # oriented result state can hide the family from the cheap
                # oriented recognizer.  The original trilinear term and the
                # compiler-owned source-leg permutation already certify it.
                structure = certified_structure
            recurrence_structure = certified_structure
            recurrence_proof_source = "exact-trilinear-symbolica-projection-v1"
            recurrence_provenance = (("vertex-term-id", str(term.id)),)
            tensor_role_representations = tuple(
                representations[source_leg]
                for source_leg in kernel.source_particle_legs
            )
        else:
            # Contact fragments are emitted only after compiler_contacts has
            # serialized a complete decomposition proof. Their compact source
            # is compiler-owned, so textual recognition is safe at this point.
            structure, coefficient = classify_trilinear_color_expression(
                kernel.color_expression,
                kernel.color_source,
                oriented_representations,
                allow_source_fallback=True,
            )
            if structure == "generic-tensor":
                raise ValueError(
                    f"compiler-generated kernel {kernel.vertex!r} has no proven "
                    "color-flow projection"
                )
            contact_contract = _contact_recurrence_color_contract(
                kernel,
                term_by_id,
            )
            if contact_contract is None:
                recurrence_structure = structure
                recurrence_proof_source = "compiler-generated-color-projection-v1"
                recurrence_provenance = ()
            else:
                (
                    recurrence_structure,
                    recurrence_proof_source,
                    recurrence_provenance,
                ) = contact_contract
            tensor_role_representations = (
                oriented_representations[0],
                oriented_representations[1],
                _dual_color_representation(oriented_representations[2]),
            )
        annotated_kernel = replace(
            kernel,
            color_projection_structure=structure,
            color_projection_coefficient=(
                float(coefficient.real),
                float(coefficient.imag),
            ),
            lc_recurrence_color_rule_kind=recurrence_structure,
        )
        recurrence_kernel = replace(
            annotated_kernel,
            color_projection_structure=recurrence_structure,
        )
        annotated_kernel = replace(
            annotated_kernel,
            lc_color_transition_terms=compile_lc_color_transition_terms(
                recurrence_kernel,
                oriented_representations,
                proof_source=recurrence_proof_source,
                provenance=recurrence_provenance,
                tensor_role_representations=tensor_role_representations,
            ),
        )
        annotated.append(annotated_kernel)
    return tuple(annotated)


def _dual_color_representation(representation: int) -> int:
    return -representation if abs(representation) == 3 else representation


def _contact_recurrence_color_contract(
    kernel: CompiledOrientedKernel,
    term_by_id: dict[int, CompiledVertexTerm],
) -> tuple[str, str, tuple[tuple[str, str], ...]] | None:
    is_partial = kernel.vertex.endswith("::contact-partial")
    is_final = kernel.vertex.endswith(("::contact-final", "::contact-final-fused"))
    if not is_partial and not is_final:
        # Balanced contact-tree lowering has its own compiler-owned singlet
        # auxiliaries and is not described by the four-point split proof.
        # Its trilinear color projection is authenticated by the generic
        # compiler-generated contract below.
        return None

    stage = "partial" if is_partial else "final"
    owner_matches = []
    orientation_matches = []
    fused_term_ids = set(kernel.term_ids or (kernel.term_id,))
    # Contact-partial deduplication and final fusion may retain an auxiliary
    # owned by a term that is no longer the fused kernel's primary term ID.
    # Match the authenticated auxiliary name and source-leg orientation against
    # the complete proof catalog instead of trusting that incidental owner.
    for term in term_by_id.values():
        if term.contact_decomposition_proof is None:
            continue
        proof = term.contact_decomposition_proof
        if proof.status != "proven":
            continue
        for split in proof.splits:
            auxiliary_name = f"__pyamplicol_contact_{term.id}_r{split.result_leg}"
            owns_auxiliary = (
                kernel.particles[2] == auxiliary_name
                if stage == "partial"
                else auxiliary_name in kernel.particles[:2]
            )
            if owns_auxiliary:
                owner_matches.append((term, proof, split))
            orientation_is_relevant = (
                owns_auxiliary if stage == "partial" else term.id in fused_term_ids
            )
            if not orientation_is_relevant:
                continue
            for orientation in split.orientations:
                if orientation.stage != stage:
                    continue
                expected_source_legs = (
                    orientation.input_legs[0],
                    orientation.input_legs[1],
                    -1 if stage == "partial" else split.result_leg,
                )
                if kernel.source_particle_legs == expected_source_legs:
                    orientation_matches.append((term, proof, split, orientation))
    if len(owner_matches) != 1 or not orientation_matches:
        raise ValueError(
            f"compiler-generated contact kernel {kernel.vertex!r} has "
            f"{len(owner_matches)} recurrence auxiliary owners and "
            f"{len(orientation_matches)} exact stage orientations"
        )
    term, proof, split = owner_matches[0]
    if any(
        candidate_split.decomposition_kind != split.decomposition_kind
        for _term, _proof, candidate_split, _orientation in orientation_matches
    ):
        raise ValueError(
            f"compiler-generated contact kernel {kernel.vertex!r} fuses "
            "incompatible recurrence color decompositions"
        )
    recurrence_structure = {
        "literal-color-singlet": "singlet",
        "two-structure-constants": "adjoint-structure-constant",
    }[split.decomposition_kind]
    return (
        recurrence_structure,
        f"{proof.algorithm}-recurrence-{stage}-v{proof.algorithm_version}",
        (
            ("contact-decomposition-kind", split.decomposition_kind),
            ("contact-result-leg", str(split.result_leg)),
            ("contact-stage", stage),
            ("vertex-term-id", str(term.id)),
        ),
    )


def _annotate_oriented_kernel_evaluation_equivalence(
    kernels: Sequence[CompiledOrientedKernel],
    particles: Sequence[CompiledParticleRecord],
    terms: Sequence[CompiledVertexTerm],
    model_symbols: ModelSymbolRegistry,
) -> tuple[CompiledOrientedKernel, ...]:
    """Prove exact signed/permuted kernel relations from lowered expressions.

    The comparison uses the concrete component expressions after resolving
    generated coupling aliases.  It is therefore independent of how a UFO
    author spelled or ordered the source Lorentz/color structures.  Only exact
    Symbolica canonical equality up to an overall sign and input exchange is
    recorded; kernels that do not pass that test remain in distinct classes.
    """

    particle_by_name = {particle.name: particle for particle in particles}
    term_by_id = {term.id: term for term in terms}
    annotated: list[CompiledOrientedKernel] = []
    for kernel in kernels:
        derived_couplings = {
            symbols.derived_coupling(model_symbols.model_name, term_id): _sym.E(
                term_by_id[term_id].coupling_expression
            )
            for name in kernel.runtime_parameters
            if name.startswith("derived_coupling_")
            for term_id in (int(name.rsplit("_", 1)[1]),)
            if term_id in term_by_id
        }
        coupling = _replace_expression_symbols(
            _sym.E(kernel.coupling_expression),
            derived_couplings,
        )
        # The algebraic colour-flow normalization is evaluated by the kernel,
        # so it must also participate in proofs of shared kernel evaluations.
        if kernel.lc_color_normalization_power:
            coupling *= _sym.E(f"2^(-{kernel.lc_color_normalization_power}/2)")
        dimensions = tuple(
            compiled_particle_component_dimension(particle)
            for particle in (
                particle_by_name[kernel.particles[0]],
                particle_by_name[kernel.particles[1]],
                particle_by_name[kernel.particles[2]],
            )
        )
        candidates: list[tuple[str, tuple[int, int], tuple[float, float], str]] = []
        components_by_input_order: dict[
            tuple[int, int], tuple[_sym.Expression, ...]
        ] = {}
        for input_order, swap_sides in (((0, 1), False), ((1, 0), True)):
            oriented_components = tuple(
                (
                    _replace_expression_symbols(
                        _remap_kernel_symbols(
                            _sym.E(component),
                            old_kind=kernel.kind,
                            new_kind=0,
                            model_symbols=model_symbols,
                            swap_sides=swap_sides,
                        ),
                        derived_couplings,
                    )
                    * coupling
                ).expand()
                for component in kernel.component_expressions
            )
            components_by_input_order[input_order] = oriented_components
            oriented_dimensions = (
                dimensions[input_order[0]],
                dimensions[input_order[1]],
                dimensions[2],
            )
            for sign in (1, -1):
                # Horner/factor collection optimizes evaluation, but is not a
                # unique algebraic normal form: mirrored kernels can retain
                # different factorizations of the same coupling polynomial.
                # Proof identities must use expanded expressions, while the
                # stored runtime components remain optimized and unchanged.
                component_strings = tuple(
                    (sign * component).expand().to_canonical_string()
                    for component in oriented_components
                )
                signature = json.dumps(
                    {
                        "input_dimensions": list(oriented_dimensions[:2]),
                        "output_dimension": oriented_dimensions[2],
                        "components": list(component_strings),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                candidates.append(
                    (
                        signature,
                        input_order,
                        (float(sign), 0.0),
                        hashlib.sha256(signature.encode("utf-8")).hexdigest(),
                    )
                )
        _signature, input_order, factor, class_digest = min(
            candidates,
            key=lambda candidate: candidate[0],
        )
        input_exchange_factor: tuple[float, float] | None = None
        if dimensions[0] == dimensions[1]:
            direct_components = components_by_input_order[(0, 1)]
            swapped_components = components_by_input_order[(1, 0)]
            zero = _sym.E("0").to_canonical_string()
            for exchange_sign in (1, -1):
                if all(
                    (swapped - exchange_sign * direct).expand().to_canonical_string()
                    == zero
                    for direct, swapped in zip(
                        direct_components,
                        swapped_components,
                        strict=True,
                    )
                ):
                    input_exchange_factor = (float(exchange_sign), 0.0)
                    break
        annotated.append(
            replace(
                kernel,
                evaluation_class=f"symbolica-sha256:{class_digest}",
                evaluation_factor=factor,
                evaluation_input_order=input_order,
                evaluation_input_exchange_factor=input_exchange_factor,
                evaluation_equivalence_verified=True,
            )
        )
    reference_factor_by_class: dict[str, complex] = {}
    normalized: list[CompiledOrientedKernel] = []
    for kernel in annotated:
        class_factor_complex = complex(*kernel.evaluation_factor)
        reference_factor = reference_factor_by_class.setdefault(
            kernel.evaluation_class,
            class_factor_complex,
        )
        relative_factor = class_factor_complex / reference_factor
        normalized.append(
            replace(
                kernel,
                evaluation_factor=(
                    float(relative_factor.real),
                    float(relative_factor.imag),
                ),
            )
        )
    return tuple(normalized)


def _compile_oriented_kernels(
    terms: Sequence[CompiledVertexTerm],
    particles: Sequence[CompiledParticleRecord],
    parameters: Sequence[CompiledParameterRecord],
    propagators: Sequence[CompiledPropagatorRecord],
    model_symbols: ModelSymbolRegistry,
) -> tuple[CompiledOrientedKernel, ...]:
    particle_by_name = {particle.name: particle for particle in particles}
    parameter_by_name = {parameter.name: parameter for parameter in parameters}
    propagator_by_name = {propagator.name: propagator for propagator in propagators}
    external_parameters = {
        parameter.name for parameter in parameters if parameter.nature == "external"
    }
    kernels: list[CompiledOrientedKernel] = []
    for term in terms:
        if term.valence != 3:
            continue
        oriented_result_particles: set[str] = set()
        for result_leg in range(3):
            result_source_name = term.particles[result_leg]
            if result_source_name in oriented_result_particles:
                continue
            oriented_result_particles.add(result_source_name)
            input_legs = tuple(leg for leg in range(3) if leg != result_leg)
            input_orders = (
                (input_legs,)
                if term.particles[input_legs[0]] == term.particles[input_legs[1]]
                else (input_legs, tuple(reversed(input_legs)))
            )
            for left_leg, right_leg in input_orders:
                result_source = particle_by_name[term.particles[result_leg]]
                try:
                    result_name = particle_by_name[result_source.antiname].name
                except KeyError as exc:
                    raise ValueError(
                        f"vertex {term.vertex} particle {result_source.name} refers to "
                        f"absent antiparticle {result_source.antiname}"
                    ) from exc
                ordered_components = _oriented_component_expressions(
                    term,
                    particle_by_name,
                    left_leg=left_leg,
                    right_leg=right_leg,
                    result_leg=result_leg,
                    kind=len(kernels),
                    model_symbols=model_symbols,
                    use_transverse_massless_yang_mills=(
                        _term_supports_transverse_massless_yang_mills(
                            term,
                            particle_by_name,
                            parameter_by_name,
                            propagator_by_name,
                        )
                    ),
                )
                coupling_symbols = set(
                    _sym.E(term.coupling_expression).get_all_symbols(False)
                )
                runtime_parameters = tuple(
                    sorted(
                        name
                        for name in external_parameters
                        if model_symbols.symbol(name) in coupling_symbols
                    )
                )
                kernels.append(
                    CompiledOrientedKernel(
                        kind=len(kernels),
                        term_id=term.id,
                        vertex=term.vertex,
                        particles=(
                            term.particles[left_leg],
                            term.particles[right_leg],
                            result_name,
                        ),
                        source_particle_legs=(left_leg, right_leg, result_leg),
                        component_expressions=tuple(
                            str(component) for component in ordered_components.values
                        ),
                        coupling_expression=term.coupling_expression,
                        coupling_orders=term.coupling_orders,
                        runtime_parameters=runtime_parameters,
                        color_source=term.color_source,
                        color_expression=term.color_expression,
                        lc_color_normalization_power=(
                            term.lc_color_normalization_power
                        ),
                        term_ids=(term.id,),
                        output_ordering_id=ordered_components.ordering_id,
                    )
                )
    return _fuse_oriented_kernels(
        kernels,
        model_name=model_symbols.model_name,
    )


def _term_supports_transverse_massless_yang_mills(
    term: CompiledVertexTerm,
    particles: Mapping[str, CompiledParticleRecord],
    parameters: Mapping[str, CompiledParameterRecord],
    propagators: Mapping[str, CompiledPropagatorRecord],
) -> bool:
    """Prove the field/propagator contract needed by transverse YM lowering."""

    if not _is_single_structure_constant(term.color_expression):
        return False
    for name in term.particles:
        particle = particles[name]
        propagator = (
            None
            if particle.propagator is None
            else propagators.get(particle.propagator)
        )
        if not (
            particle.spin == 3
            and particle.color == 8
            and particle.self_conjugate is True
            and _is_compile_time_zero_parameter(particle.mass, parameters)
            and particle.propagating
            and particle.ghost_number == 0
            and propagator is not None
            and propagator.particle == particle.name
            and not propagator.custom
        ):
            return False
    return True


def _compile_four_point_contact_kernels(
    terms: Sequence[CompiledVertexTerm],
    particles: Sequence[CompiledParticleRecord],
    *,
    start_kind: int,
    model_symbols: ModelSymbolRegistry,
) -> tuple[tuple[CompiledParticleRecord, ...], tuple[CompiledOrientedKernel, ...]]:
    """Lower momentum-independent four-point tensors through dense auxiliaries."""

    particle_by_name = {particle.name: particle for particle in particles}
    used_pdgs = {abs(particle.pdg_code) for particle in particles}
    next_pdg = max(9_000_000, max(used_pdgs, default=0) + 1)
    auxiliary_particles: list[CompiledParticleRecord] = []
    kernels: list[CompiledOrientedKernel] = []

    def allocate_pdg() -> int:
        nonlocal next_pdg
        while next_pdg in used_pdgs:
            next_pdg += 1
        result = next_pdg
        used_pdgs.add(result)
        next_pdg += 1
        return result

    for term in terms:
        if term.valence != 4:
            continue
        if "ufo_momentum_" in term.lorentz_expression:
            continue
        source_particles = tuple(particle_by_name[name] for name in term.particles)
        proof = _validated_contact_decomposition_proof(term)
        if (
            proof is None
            and term.contact_decomposition_proof is None
            and _contact_term_has_literal_color_singlet(term)
            and all(particle.color == 1 for particle in source_particles)
        ):
            # Preserve direct construction of uncolored test/compiler terms.
            # UFO compilation always serializes this trivial proof beforehand.
            proof = _build_contact_decomposition_proof(
                term,
                source_particles,
                particle_by_name,
                model_symbols=model_symbols,
            )
        if proof is None:
            # Leave the complete term unlowered so model preflight reports a
            # structured unsupported-contact-color-lowering error. Partial
            # orientation output would incorrectly make the term appear valid.
            continue
        for split in proof.splits:
            result_leg = split.result_leg
            source_result = source_particles[result_leg]
            remaining_leg = split.remaining_leg
            open_legs = split.open_legs
            component_axis_order = tuple(
                label
                for leg in open_legs
                for label in _spin_axis_labels(source_particles[leg].spin, leg + 1)
            )
            if component_axis_order != split.component_axis_order:
                raise ValueError(
                    f"contact decomposition component order mismatch for term "
                    f"{term.id} result leg {result_leg}"
                )
            auxiliary_name = f"__pyamplicol_contact_{term.id}_r{result_leg}"
            representative_indices = split.component_basis_order
            component_expansion = split.component_expansion
            auxiliary_dimension = len(representative_indices)
            auxiliary = CompiledParticleRecord(
                name=auxiliary_name,
                antiname=auxiliary_name,
                pdg_code=allocate_pdg(),
                spin=-1,
                color=split.auxiliary_color,
                mass="ZERO",
                width="ZERO",
                charge=0.0,
                quantum_numbers=(("electric_charge", "0"),),
                ghost_number=0,
                propagating=False,
                goldstoneboson=False,
                propagator=None,
                component_dimension=auxiliary_dimension,
                auxiliary_kind=(
                    f"ufo-contact:{term.id}:result-{result_leg}:"
                    + ",".join(str(leg) for leg in open_legs)
                ),
            )
            auxiliary_particles.append(auxiliary)

            partial_orientations = tuple(
                item for item in split.orientations if item.stage == "partial"
            )
            for orientation in partial_orientations:
                left_leg, right_leg = orientation.input_legs
                kind = start_kind + len(kernels)
                components = _contact_partial_component_expressions(
                    term,
                    particle_by_name,
                    left_leg=left_leg,
                    right_leg=right_leg,
                    open_legs=open_legs,
                    kind=kind,
                    model_symbols=model_symbols,
                )
                kernels.append(
                    CompiledOrientedKernel(
                        kind=kind,
                        term_id=term.id,
                        vertex=f"{term.vertex}::contact-partial",
                        particles=(
                            source_particles[left_leg].name,
                            source_particles[right_leg].name,
                            auxiliary.name,
                        ),
                        source_particle_legs=(left_leg, right_leg, -1),
                        component_expressions=tuple(
                            _canonicalize_oriented_kernel_component(
                                _sym.E(components[index])
                                * _sym.E(orientation.scalar_prefactor)
                            ).to_canonical_string()
                            for index in representative_indices
                        ),
                        coupling_expression="1",
                        coupling_orders=(),
                        runtime_parameters=(),
                        color_source="1",
                        color_expression="1",
                        lc_color_normalization_power=0,
                        term_ids=(),
                        contact_orbit_steps=_compiled_contact_orbit_steps(
                            term,
                            split,
                            orientation,
                        ),
                    )
                )

            result_name = particle_by_name[source_result.antiname].name
            final_orientations = tuple(
                item for item in split.orientations if item.stage == "final"
            )
            for orientation in final_orientations:
                auxiliary_on_left = orientation.input_legs[0] == -1
                left_name, right_name = (
                    (auxiliary.name, source_particles[remaining_leg].name)
                    if auxiliary_on_left
                    else (source_particles[remaining_leg].name, auxiliary.name)
                )
                kind = start_kind + len(kernels)
                derived_coupling = symbols.derived_coupling(
                    model_symbols.model_name,
                    term.id,
                )
                final_prefactor = (
                    _sym.E(orientation.scalar_prefactor) * derived_coupling
                )
                combined_color_source = (
                    f"{split.outer_color_source}*{split.final_color_source}"
                    if split.decomposition_kind == "two-structure-constants"
                    else term.color_source
                )
                kernels.append(
                    CompiledOrientedKernel(
                        kind=kind,
                        term_id=term.id,
                        vertex=f"{term.vertex}::contact-final",
                        particles=(left_name, right_name, result_name),
                        source_particle_legs=(
                            -1 if auxiliary_on_left else remaining_leg,
                            remaining_leg if auxiliary_on_left else -1,
                            result_leg,
                        ),
                        component_expressions=tuple(
                            (_sym.E(component) * final_prefactor).to_canonical_string()
                            for component in _contact_final_component_expressions(
                                source_particles,
                                auxiliary,
                                open_legs=open_legs,
                                remaining_leg=remaining_leg,
                                result_leg=result_leg,
                                kind=kind,
                                auxiliary_on_left=auxiliary_on_left,
                                component_expansion=component_expansion,
                                model_symbols=model_symbols,
                            )
                        ),
                        coupling_expression="1",
                        coupling_orders=term.coupling_orders,
                        runtime_parameters=(f"derived_coupling_{term.id}",),
                        color_source=combined_color_source,
                        color_expression=combined_color_source,
                        lc_color_normalization_power=(
                            split.outer_color_normalization_power
                            + split.final_color_normalization_power
                        ),
                        term_ids=(term.id,),
                        contact_orbit_steps=_compiled_contact_orbit_steps(
                            term,
                            split,
                            orientation,
                        ),
                    )
                )
    return tuple(auxiliary_particles), tuple(kernels)


def _compiled_contact_orbit_steps(
    term: CompiledVertexTerm,
    split: CompiledContactDecompositionSplit,
    orientation: CompiledContactOrientationProof,
) -> tuple[CompiledContactOrbitStep, ...]:
    certificate = term.contact_orbit_certificate
    if certificate is None:
        return ()
    return (compiled_contact_orbit_step(certificate, split, orientation),)

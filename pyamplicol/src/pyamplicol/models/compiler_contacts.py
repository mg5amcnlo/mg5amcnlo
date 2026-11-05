# SPDX-License-Identifier: 0BSD
"""Four-point contact component compression and fusion."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace

from .._internal.physics.symbols import ModelSymbolRegistry, symbols
from . import compiler_symbolica as _sym
from .compiler_kernels import (
    _as_expression,
    _canonicalize_oriented_kernel_component,
    _component_symbols,
    _function_arguments,
    _input_tensor_expression,
    _ordered_dense_tensor_components,
    _permutation_sign,
    _remap_kernel_symbols,
    _replace_expression_symbols,
    _spin_axis_labels,
    _spin_dimension,
    _spin_representations,
    _spin_slots,
)
from .compiler_records import _replace_evaluator_constants
from .contact_decomposition import (
    CONTACT_DECOMPOSITION_ALGORITHM,
    CONTACT_DECOMPOSITION_ALGORITHM_VERSION,
    CONTACT_ORBIT_ALGORITHM,
    CONTACT_ORBIT_ALGORITHM_VERSION,
    CONTACT_ORBIT_EVALUATOR_CLASS,
    CompiledContactDecompositionProof,
    CompiledContactDecompositionSplit,
    CompiledContactDummyIndexMapping,
    CompiledContactOrbitCertificate,
    CompiledContactOrientationProof,
    CompiledContactUnsupportedReason,
    canonical_contact_orbit_steps,
    contact_orbit_evaluator_class_is_certifiable,
)
from .contracts import (
    CompiledOrientedKernel,
    CompiledParticleRecord,
    CompiledVertexTerm,
    compiled_particle_component_dimension,
)


def _fuse_contact_finals(
    kernels: Sequence[CompiledOrientedKernel],
    terms: Sequence[CompiledVertexTerm],
    *,
    model_symbols: ModelSymbolRegistry,
) -> tuple[CompiledOrientedKernel, ...]:
    coupling_by_term = {term.id: term.coupling_expression for term in terms}
    groups: dict[tuple[object, ...], list[CompiledOrientedKernel]] = {}
    passthrough: list[CompiledOrientedKernel] = []
    for kernel in kernels:
        if not kernel.vertex.endswith("::contact-final"):
            passthrough.append(kernel)
            continue
        key = (
            kernel.particles,
            kernel.coupling_orders,
            kernel.color_source,
            kernel.lc_color_normalization_power,
        )
        groups.setdefault(key, []).append(kernel)

    fused: list[CompiledOrientedKernel] = []
    for members in groups.values():
        first = members[0]
        runtime_aliases: dict[_sym.Expression, _sym.Expression] = {}
        canonical_runtime_name: str | None = None
        if all(
            coupling_by_term.get(member.term_id) == coupling_by_term.get(first.term_id)
            for member in members
        ):
            canonical_runtime_name = f"derived_coupling_{first.term_id}"
            canonical_runtime = symbols.derived_coupling(
                model_symbols.model_name,
                first.term_id,
            )
            runtime_aliases = {
                symbols.derived_coupling(
                    model_symbols.model_name,
                    member.term_id,
                ): canonical_runtime
                for member in members
            }
        components = tuple(
            _canonicalize_oriented_kernel_component(
                sum(
                    (
                        _replace_expression_symbols(
                            _remap_kernel_symbols(
                                _sym.E(member.component_expressions[index]),
                                old_kind=member.kind,
                                new_kind=first.kind,
                                model_symbols=model_symbols,
                            ),
                            runtime_aliases,
                        )
                        for member in members
                    ),
                    _sym.E("0"),
                )
            ).to_canonical_string()
            for index in range(len(first.component_expressions))
        )
        fused.append(
            replace(
                first,
                vertex=first.vertex.replace(
                    "::contact-final",
                    "::contact-final-fused",
                ),
                component_expressions=components,
                runtime_parameters=(
                    (canonical_runtime_name,)
                    if canonical_runtime_name is not None
                    else tuple(
                        sorted(
                            {
                                name
                                for member in members
                                for name in member.runtime_parameters
                            }
                        )
                    )
                ),
                term_ids=tuple(
                    term_id
                    for member in members
                    for term_id in (member.term_ids or (member.term_id,))
                ),
                contact_orbit_steps=canonical_contact_orbit_steps(
                    *(member.contact_orbit_steps for member in members)
                ),
            )
        )
    return tuple(sorted((*passthrough, *fused), key=lambda kernel: kernel.kind))


def _contact_partial_component_expressions(
    term: CompiledVertexTerm,
    particle_by_name: Mapping[str, CompiledParticleRecord],
    *,
    left_leg: int,
    right_leg: int,
    open_legs: tuple[int, ...],
    kind: int,
    model_symbols: ModelSymbolRegistry,
) -> tuple[str, ...]:
    library = _sym.TensorLibrary.hep_lib_atom()
    expression = _sym.E(term.lorentz_expression)
    particles = tuple(particle_by_name[name] for name in term.particles)
    expression *= _input_tensor_expression(
        library,
        kind=kind,
        side="left",
        spin=particles[left_leg].spin,
        leg=left_leg + 1,
        components=_component_symbols(
            kind,
            "left",
            particles[left_leg].spin,
            model_symbols=model_symbols,
        ),
        model_symbols=model_symbols,
    )
    expression *= _input_tensor_expression(
        library,
        kind=kind,
        side="right",
        spin=particles[right_leg].spin,
        leg=right_leg + 1,
        components=_component_symbols(
            kind,
            "right",
            particles[right_leg].spin,
            model_symbols=model_symbols,
        ),
        model_symbols=model_symbols,
    )
    result = _execute_dense_tensor(
        expression,
        library,
        axis_labels=tuple(
            label
            for leg in open_legs
            for label in _spin_axis_labels(particles[leg].spin, leg + 1)
        ),
    )
    expected = math.prod(_spin_dimension(particles[leg].spin) for leg in open_legs)
    if len(result) != expected:
        raise ValueError(
            f"contact partial {term.vertex}/{term.id} produced {len(result)} "
            f"components, expected {expected}"
        )
    return tuple(
        _replace_evaluator_constants(
            _as_expression(result[index])
        ).to_canonical_string()
        for index in range(len(result))
    )


def _contact_final_component_expressions(
    particles: Sequence[CompiledParticleRecord],
    auxiliary: CompiledParticleRecord,
    *,
    open_legs: tuple[int, ...],
    remaining_leg: int,
    result_leg: int,
    kind: int,
    auxiliary_on_left: bool,
    component_expansion: tuple[tuple[int, int] | None, ...],
    model_symbols: ModelSymbolRegistry,
) -> tuple[str, ...]:
    library = _sym.TensorLibrary.hep_lib_atom()
    auxiliary_side = "left" if auxiliary_on_left else "right"
    physical_side = "right" if auxiliary_on_left else "left"
    auxiliary_symbols = tuple(
        model_symbols.kernel_component(kind, auxiliary_side, component)
        for component in range(compiled_particle_component_dimension(auxiliary))
    )
    expanded_auxiliary = tuple(
        _sym.E("0") if entry is None else entry[1] * auxiliary_symbols[entry[0]]
        for entry in component_expansion
    )
    expression = _contact_auxiliary_tensor_expression(
        library,
        kind=kind,
        side=auxiliary_side,
        particles=particles,
        open_legs=open_legs,
        components=expanded_auxiliary,
        model_symbols=model_symbols,
    )
    expression *= _input_tensor_expression(
        library,
        kind=kind,
        side=physical_side,
        spin=particles[remaining_leg].spin,
        leg=remaining_leg + 1,
        components=_component_symbols(
            kind,
            physical_side,
            particles[remaining_leg].spin,
            model_symbols=model_symbols,
        ),
        model_symbols=model_symbols,
    )
    result = _execute_dense_tensor(
        expression,
        library,
        axis_labels=_spin_axis_labels(particles[result_leg].spin, result_leg + 1),
    )
    expected = _spin_dimension(particles[result_leg].spin)
    if len(result) != expected:
        raise ValueError(
            f"contact final for source leg {result_leg} produced {len(result)} "
            f"components, expected {expected}"
        )
    return tuple(
        _replace_evaluator_constants(
            _as_expression(result[index])
        ).to_canonical_string()
        for index in range(len(result))
    )


def _contact_auxiliary_tensor_expression(
    library: _sym.TensorLibrary,
    *,
    kind: int,
    side: str,
    particles: Sequence[CompiledParticleRecord],
    open_legs: tuple[int, ...],
    components: Sequence[_sym.Expression],
    model_symbols: ModelSymbolRegistry,
) -> _sym.Expression:
    representations = tuple(
        representation
        for leg in open_legs
        for representation in _spin_representations(particles[leg].spin)
    )
    if not representations:
        if len(components) != 1:
            raise ValueError("scalar contact auxiliary must have one component")
        return components[0]
    slots = tuple(
        slot for leg in open_legs for slot in _spin_slots(particles[leg].spin, leg + 1)
    )
    name = _sym.TensorName(model_symbols.kernel_tensor_name(kind, side))
    library.register(_sym.Tensor.dense(name(*representations), components))
    return name(*slots).to_expression()


def _execute_dense_tensor(
    expression: _sym.Expression,
    library: _sym.TensorLibrary,
    *,
    axis_labels: Sequence[str],
) -> tuple[object, ...]:
    if not axis_labels and set(expression.get_all_symbols()) == set(
        expression.get_all_symbols(include_function_symbols=False)
    ):
        # A rank-zero expression with no function indeterminates contains no
        # tensor heads to contract. Spenso need not construct a tensor network.
        return (expression,)
    network = _sym.TensorNetwork(
        _sym.as_tensor(expression) if axis_labels else expression, library
    )
    network.execute(library=library)
    result = network.result_tensor(library)
    return _ordered_dense_tensor_components(result, axis_labels).values


def _contact_auxiliary_color(
    term: CompiledVertexTerm,
    particles: Sequence[CompiledParticleRecord],
    *,
    remaining_leg: int,
    result_leg: int,
) -> int:
    colors = tuple(particle.color for particle in particles)
    if all(color == 1 for color in colors):
        return 1
    if "f(" in term.color_source or "::f(" in term.color_source:
        return 8
    remaining = abs(colors[remaining_leg])
    result = abs(colors[result_leg])
    if remaining == 1:
        return int(colors[result_leg])
    if result == 1:
        return int(colors[remaining_leg])
    if remaining == result == 8:
        return 1
    return 1


def _compress_contact_components(
    components: Sequence[str],
) -> tuple[tuple[int, ...], tuple[tuple[int, int] | None, ...]]:
    representatives: list[_sym.Expression] = []
    representative_indices: list[int] = []
    expansion: list[tuple[int, int] | None] = []
    zero = _sym.E("0")
    for index, source in enumerate(components):
        expression = _sym.E(source)
        if str(expression) == "0":
            expansion.append(None)
            continue
        match: tuple[int, int] | None = None
        for basis_index, representative in enumerate(representatives):
            if str((expression - representative).expand()) == "0":
                match = (basis_index, 1)
                break
            if str((expression + representative).expand()) == "0":
                match = (basis_index, -1)
                break
        if match is None:
            match = (len(representatives), 1)
            representatives.append(expression)
            representative_indices.append(index)
        expansion.append(match)
    if not representatives:
        representatives.append(zero)
        representative_indices.append(0)
        expansion[0] = (0, 1)
    return tuple(representative_indices), tuple(expansion)


def _record_contact_decomposition_proofs(
    terms: Sequence[CompiledVertexTerm],
    particles: Sequence[CompiledParticleRecord],
    *,
    model_symbols: ModelSymbolRegistry,
) -> tuple[CompiledVertexTerm, ...]:
    """Attach explicit proof records before four-point contact lowering."""

    particle_by_name = {particle.name: particle for particle in particles}
    result: list[CompiledVertexTerm] = []
    for term in terms:
        if term.valence != 4 or "ufo_momentum_" in term.lorentz_expression:
            result.append(term)
            continue
        source_particles = tuple(particle_by_name[name] for name in term.particles)
        proof = _build_contact_decomposition_proof(
            term,
            source_particles,
            particle_by_name,
            model_symbols=model_symbols,
        )
        result.append(
            replace(
                term,
                contact_decomposition_proof=proof,
                contact_orbit_certificate=_build_contact_orbit_certificate(
                    term,
                    source_particles,
                    proof,
                ),
            )
        )
    return tuple(result)


def _build_contact_orbit_certificate(
    term: CompiledVertexTerm,
    source_particles: Sequence[CompiledParticleRecord],
    proof: CompiledContactDecompositionProof,
) -> CompiledContactOrbitCertificate | None:
    """Issue the deliberately narrow compiler-owned contact-orbit proof."""

    if (
        proof.status != "proven"
        or not proof.matches(term)
        or tuple(particle.name for particle in source_particles) != term.particles
        or not contact_orbit_evaluator_class_is_certifiable(
            color_source=term.color_source,
            color_expression=term.color_expression,
            lorentz_expression=term.lorentz_expression,
            decomposition_kinds=tuple(
                split.decomposition_kind for split in proof.splits
            ),
            particle_contracts=tuple(
                (
                    particle.spin,
                    particle.color,
                    particle.statistics,
                    particle.wavefunction_family,
                    particle.self_conjugate,
                )
                for particle in source_particles
            ),
        )
    ):
        return None
    equivalence_classes: dict[str, int] = {}
    leg_classes = tuple(
        equivalence_classes.setdefault(particle.name, len(equivalence_classes))
        for particle in source_particles
    )
    if len(leg_classes) != 4:
        raise ValueError("contact-orbit certificate requires exactly four particles")
    return CompiledContactOrbitCertificate(
        algorithm=CONTACT_ORBIT_ALGORITHM,
        algorithm_version=CONTACT_ORBIT_ALGORITHM_VERSION,
        term_id=term.id,
        vertex=term.vertex,
        particles=(
            source_particles[0].name,
            source_particles[1].name,
            source_particles[2].name,
            source_particles[3].name,
        ),
        color_expression=term.color_expression,
        lorentz_expression=term.lorentz_expression,
        coupling_expression=term.coupling_expression,
        evaluator_class=CONTACT_ORBIT_EVALUATOR_CLASS,
        physical_leg_equivalence_classes=(
            leg_classes[0],
            leg_classes[1],
            leg_classes[2],
            leg_classes[3],
        ),
        reconstruction_factor="1",
    )


def _build_contact_decomposition_proof(
    term: CompiledVertexTerm,
    source_particles: Sequence[CompiledParticleRecord],
    particle_by_name: Mapping[str, CompiledParticleRecord],
    *,
    model_symbols: ModelSymbolRegistry,
) -> CompiledContactDecompositionProof:
    result_legs: list[int] = []
    result_particle_names: set[str] = set()
    for result_leg, source_result in enumerate(source_particles):
        if source_result.name in result_particle_names:
            continue
        result_particle_names.add(source_result.name)
        result_legs.append(result_leg)

    literal_singlet = _contact_term_has_literal_color_singlet(term)
    if literal_singlet and any(particle.color != 1 for particle in source_particles):
        return _contact_proof(
            term,
            status="unsupported",
            unsupported_reasons=(
                _contact_unsupported_reason(
                    "literal-singlet-with-colored-legs",
                    "a literal color singlet cannot contract non-singlet external legs",
                    color_representations=",".join(
                        str(particle.color) for particle in source_particles
                    ),
                ),
            ),
        )

    splits: list[CompiledContactDecompositionSplit] = []
    unsupported: list[CompiledContactUnsupportedReason] = []
    for result_leg in result_legs:
        color_split = _four_point_contact_color_split(
            term,
            result_leg,
            model_symbols=model_symbols,
        )
        if color_split is None:
            unsupported.append(
                _unsupported_contact_color_split_reason(
                    term,
                    result_leg,
                    model_symbols=model_symbols,
                )
            )
            continue
        splits.append(
            _contact_decomposition_split_proof(
                term,
                source_particles,
                particle_by_name,
                result_leg=result_leg,
                color_split=color_split,
                model_symbols=model_symbols,
            )
        )
    if unsupported:
        return _contact_proof(
            term,
            status="unsupported",
            unsupported_reasons=tuple(unsupported),
        )
    return _contact_proof(term, status="proven", splits=tuple(splits))


def _contact_decomposition_split_proof(
    term: CompiledVertexTerm,
    source_particles: Sequence[CompiledParticleRecord],
    particle_by_name: Mapping[str, CompiledParticleRecord],
    *,
    result_leg: int,
    color_split: tuple[
        tuple[int, int],
        int,
        str,
        str,
        int,
        int,
        tuple[int, ...],
        tuple[int, ...],
        int,
        str,
    ],
    model_symbols: ModelSymbolRegistry,
) -> CompiledContactDecompositionSplit:
    (
        pair_legs,
        remaining_leg,
        outer_color_source,
        final_color_source,
        outer_color_power,
        final_color_power,
        outer_color_factor,
        final_color_factor,
        color_dummy,
        color_coefficient,
    ) = color_split
    open_legs = (
        min(remaining_leg, result_leg),
        max(remaining_leg, result_leg),
    )
    canonical_components = _contact_partial_component_expressions(
        term,
        particle_by_name,
        left_leg=pair_legs[0],
        right_leg=pair_legs[1],
        open_legs=open_legs,
        kind=term.id,
        model_symbols=model_symbols,
    )
    representative_indices, component_expansion = _compress_contact_components(
        canonical_components
    )
    assignment_multiplicity = sum(
        source_particles[leg].name == source_particles[remaining_leg].name
        for leg in range(4)
        if leg != result_leg
    )
    pair_orders = (
        (pair_legs,)
        if source_particles[pair_legs[0]].name == source_particles[pair_legs[1]].name
        else (pair_legs, tuple(reversed(pair_legs)))
    )
    canonical_outer_parity = (
        _permutation_sign(
            outer_color_factor,
            (color_dummy, pair_legs[0] + 1, pair_legs[1] + 1),
        )
        if outer_color_factor
        else 1
    )
    orientations: list[CompiledContactOrientationProof] = []
    for left_leg, right_leg in pair_orders:
        outer_parity = (
            _permutation_sign(
                outer_color_factor,
                (color_dummy, left_leg + 1, right_leg + 1),
            )
            if outer_color_factor
            else 1
        )
        orientations.append(
            CompiledContactOrientationProof(
                stage="partial",
                input_legs=(left_leg, right_leg),
                permutation_parity=outer_parity,
                scalar_prefactor=_canonical_scalar_prefactor(
                    _sym.E(str(outer_parity)) / canonical_outer_parity
                ),
            )
        )

    for auxiliary_on_left in (True, False):
        final_input_tokens = (
            (color_dummy, remaining_leg + 1)
            if auxiliary_on_left
            else (remaining_leg + 1, color_dummy)
        )
        final_parity = (
            _permutation_sign(
                final_color_factor,
                (result_leg + 1, *final_input_tokens),
            )
            if final_color_factor
            else 1
        )
        orientations.append(
            CompiledContactOrientationProof(
                stage="final",
                input_legs=(
                    (-1, remaining_leg) if auxiliary_on_left else (remaining_leg, -1)
                ),
                permutation_parity=final_parity,
                scalar_prefactor=_canonical_scalar_prefactor(
                    _sym.E(color_coefficient)
                    * canonical_outer_parity
                    * final_parity
                    / assignment_multiplicity
                ),
            )
        )

    dummy_mapping = (
        CompiledContactDummyIndexMapping(
            source_index=color_dummy,
            normalized_symbol=f"ufo_c_dummy_{abs(color_dummy)}_adjoint",
            outer_slot=outer_color_factor.index(color_dummy),
            final_slot=final_color_factor.index(color_dummy),
        )
        if outer_color_factor
        else None
    )
    return CompiledContactDecompositionSplit(
        decomposition_kind=(
            "two-structure-constants" if outer_color_factor else "literal-color-singlet"
        ),
        result_leg=result_leg,
        pair_legs=pair_legs,
        remaining_leg=remaining_leg,
        outer_color_source=outer_color_source,
        final_color_source=final_color_source,
        outer_color_factor=outer_color_factor,
        final_color_factor=final_color_factor,
        dummy_index_mapping=dummy_mapping,
        outer_color_normalization_power=outer_color_power,
        final_color_normalization_power=final_color_power,
        color_coefficient=color_coefficient,
        auxiliary_color=_contact_auxiliary_color(
            term,
            source_particles,
            remaining_leg=remaining_leg,
            result_leg=result_leg,
        ),
        open_legs=open_legs,
        component_axis_order=tuple(
            label
            for leg in open_legs
            for label in _spin_axis_labels(source_particles[leg].spin, leg + 1)
        ),
        component_basis_order=representative_indices,
        component_expansion=component_expansion,
        assignment_multiplicity=assignment_multiplicity,
        canonical_outer_parity=canonical_outer_parity,
        orientations=tuple(orientations),
    )


def _contact_proof(
    term: CompiledVertexTerm,
    *,
    status: str,
    splits: tuple[CompiledContactDecompositionSplit, ...] = (),
    unsupported_reasons: tuple[CompiledContactUnsupportedReason, ...] = (),
) -> CompiledContactDecompositionProof:
    return CompiledContactDecompositionProof(
        status=status,
        algorithm=CONTACT_DECOMPOSITION_ALGORITHM,
        algorithm_version=CONTACT_DECOMPOSITION_ALGORITHM_VERSION,
        term_id=term.id,
        vertex=term.vertex,
        particles=term.particles,
        color_index=term.color_index,
        lorentz_index=term.lorentz_index,
        original_color_source=term.color_source,
        normalized_color_expression=term.color_expression,
        lorentz_name=term.lorentz_name,
        original_lorentz_source=term.lorentz_source,
        normalized_lorentz_expression=term.lorentz_expression,
        splits=splits,
        unsupported_reasons=unsupported_reasons,
    )


def _contact_unsupported_reason(
    code: str,
    message: str,
    **context: str,
) -> CompiledContactUnsupportedReason:
    return CompiledContactUnsupportedReason(
        code=code,
        message=message,
        context=tuple(sorted(context.items())),
    )


def _unsupported_contact_color_split_reason(
    term: CompiledVertexTerm,
    result_leg: int,
    *,
    model_symbols: ModelSymbolRegistry,
) -> CompiledContactUnsupportedReason:
    common = {
        "normalized_color_expression": term.color_expression,
        "result_leg": str(result_leg),
    }
    factor_arguments = _function_arguments(term.color_expression, "::f")
    if len(factor_arguments) != 2:
        return _contact_unsupported_reason(
            "unsupported-color-factor-count",
            "contact color expression is not exactly two structure constants",
            factor_count=str(len(factor_arguments)),
            **common,
        )
    factors = _normalized_structure_constant_factors(term.color_expression)
    if len(factors) != 2:
        return _contact_unsupported_reason(
            "malformed-structure-constant-indices",
            "contact structure-constant indices are not normalized adjoint indices",
            **common,
        )
    if (
        _source_structure_constant_product_coefficient(
            term.color_source,
            model_symbols=model_symbols,
        )
        is None
    ):
        return _contact_unsupported_reason(
            "non-scalar-color-prefactor",
            "contact structure constants retain a non-scalar color factor",
            **common,
        )
    shared_dummies = set(value for value in factors[0] if value < 0) & set(
        value for value in factors[1] if value < 0
    )
    if len(shared_dummies) != 1:
        return _contact_unsupported_reason(
            "non-unique-shared-dummy",
            "contact structure constants must share exactly one adjoint dummy",
            shared_dummy_count=str(len(shared_dummies)),
            **common,
        )
    result_index = result_leg + 1
    if not any(result_index in factor for factor in factors):
        return _contact_unsupported_reason(
            "result-leg-absent-from-color-factors",
            "contact result leg is absent from both structure constants",
            **common,
        )
    return _contact_unsupported_reason(
        "unsupported-structure-constant-topology",
        "contact structure constants do not form a trivalent auxiliary split",
        **common,
    )


def _canonical_scalar_prefactor(expression: _sym.Expression) -> str:
    return str(expression.to_canonical_string())


def _contact_term_has_literal_color_singlet(term: CompiledVertexTerm) -> bool:
    return term.color_source in {"1", "UFO::{}::1"} or term.color_expression == "1"


def _validated_contact_decomposition_proof(
    term: CompiledVertexTerm,
) -> CompiledContactDecompositionProof | None:
    proof = term.contact_decomposition_proof
    if proof is None:
        return None
    if not proof.matches(term):
        raise ValueError(
            f"contact decomposition proof identity mismatch for term {term.id}"
        )
    if (
        proof.algorithm != CONTACT_DECOMPOSITION_ALGORITHM
        or proof.algorithm_version != CONTACT_DECOMPOSITION_ALGORITHM_VERSION
    ):
        raise ValueError(
            f"unsupported contact decomposition proof algorithm for term {term.id}: "
            f"{proof.algorithm}/v{proof.algorithm_version}"
        )
    return proof if proof.status == "proven" else None


def _four_point_contact_color_split(
    term: CompiledVertexTerm,
    result_leg: int,
    *,
    model_symbols: ModelSymbolRegistry | None = None,
) -> (
    tuple[
        tuple[int, int],
        int,
        str,
        str,
        int,
        int,
        tuple[int, ...],
        tuple[int, ...],
        int,
        str,
    ]
    | None
):
    active_symbols = model_symbols or symbols.model("contact_color_parser")
    factors = _normalized_structure_constant_factors(term.color_expression)
    color_coefficient = _source_structure_constant_product_coefficient(
        term.color_source,
        model_symbols=active_symbols,
    )
    if len(factors) == 2 and color_coefficient is not None:
        factors = (
            _source_structure_constant_factor_orientations(
                _function_arguments(term.color_source, "f"), factors
            )
            or ()
        )
    if len(factors) == 2 and color_coefficient is not None:
        shared_dummies = set(value for value in factors[0] if value < 0) & set(
            value for value in factors[1] if value < 0
        )
        if len(shared_dummies) == 1:
            dummy = next(iter(shared_dummies))
            result_index = result_leg + 1
            final_factor = next(
                (factor for factor in factors if result_index in factor),
                None,
            )
            if final_factor is not None:
                outer_factor = factors[1] if final_factor is factors[0] else factors[0]
                pair = tuple(value - 1 for value in outer_factor if value > 0)
                remaining = tuple(
                    value - 1
                    for value in final_factor
                    if value > 0 and value != result_index
                )
                if len(pair) == 2 and len(remaining) == 1:
                    canonical_f = "UFO::{}::f(1,2,3)"
                    return (
                        (pair[0], pair[1]),
                        remaining[0],
                        canonical_f,
                        canonical_f,
                        1,
                        1,
                        outer_factor,
                        final_factor,
                        dummy,
                        color_coefficient,
                    )

    if term.color_source in {"1", "UFO::{}::1"} or term.color_expression == "1":
        input_legs = tuple(leg for leg in range(4) if leg != result_leg)
        return (
            (input_legs[0], input_legs[1]),
            input_legs[2],
            term.color_source,
            "1",
            term.lc_color_normalization_power,
            0,
            (),
            (),
            -1,
            "1",
        )
    return None


def _source_structure_constant_product_coefficient(
    expression: str,
    *,
    model_symbols: ModelSymbolRegistry,
) -> str | None:
    """Return the explicit scalar multiplying two source UFO f tensors.

    The normalized Spenso expression may acquire a sign while canonicalizing
    antisymmetric structure-constant arguments. The decomposition records
    those permutations separately, so using the normalized scalar here would
    apply that sign twice.
    """

    _sym._ensure_symbolica()
    parsed = model_symbols.expression(expression)
    wildcards = tuple(
        _sym.E(f"ufo_contact_color_argument_{index}_") for index in range(3)
    )
    coefficient = parsed.replace(
        model_symbols.symbol("f")(*wildcards),
        _sym.E("1"),
        bottom_up=True,
        repeat=True,
    )
    if coefficient.get_all_symbols():
        return None
    return str(coefficient.to_canonical_string())


def _source_structure_constant_factor_orientations(
    source_factors: Sequence[Sequence[str]],
    normalized_factors: Sequence[tuple[int, ...]],
) -> tuple[tuple[int, int, int], ...] | None:
    """Recover source permutations after normalized tensor authentication.

    The source scalar coefficient must accompany source f-tensor orientations,
    independently of antisymmetric permutations chosen by the CAS printer.
    """

    try:
        source = tuple(
            tuple(int(argument.strip()) for argument in factor)
            for factor in source_factors
        )
    except ValueError:
        return None
    if any(len(factor) != 3 for factor in source):
        return None
    source_dummies = {value for factor in source for value in factor if value < 0}
    normalized_dummies = {
        value for factor in normalized_factors for value in factor if value < 0
    }
    if len(source_dummies) != len(normalized_dummies):
        return None
    dummy_map = dict(
        zip(sorted(source_dummies), sorted(normalized_dummies), strict=True)
    )
    oriented = tuple(
        tuple(dummy_map.get(value, value) for value in factor) for factor in source
    )
    if sorted(sorted(factor) for factor in oriented) != sorted(
        sorted(factor) for factor in normalized_factors
    ):
        return None
    return oriented  # type: ignore[return-value]


def _normalized_structure_constant_factors(
    expression: str,
    *,
    expected_count: int = 2,
) -> tuple[tuple[int, ...], ...]:
    """Return typed f-tensor index words from a normalized color monomial."""

    if expression.count("::f(") != expected_count:
        return ()
    result: list[tuple[int, ...]] = []
    for arguments in _function_arguments(expression, "::f"):
        indices: list[int] = []
        for argument in arguments:
            dummy = re.search(r"ufo_c_dummy_([0-9]+)_adjoint", argument)
            if dummy is not None:
                indices.append(-int(dummy.group(1)))
                continue
            external = re.search(r"ufo_c_([0-9]+)", argument)
            if external is None:
                return ()
            indices.append(int(external.group(1)))
        if len(indices) != 3:
            return ()
        result.append(tuple(indices))
    return tuple(result)

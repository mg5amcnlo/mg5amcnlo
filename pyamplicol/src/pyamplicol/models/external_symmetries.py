# SPDX-License-Identifier: 0BSD
"""Algebraic symmetry certificates for compiled external models.

The certificates in this module are deliberately derived from normalized
Lorentz/color tensors and exact coupling relations. UFO particle names,
vertex names, PDG assignments, and coupling-order labels are not proof input.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass

from .._internal.physics.symbols import ModelSymbolRegistry, symbols
from . import compiler_symbolica as _sym
from .contracts import (
    CompiledCouplingRecord,
    CompiledModelIR,
    CompiledOrientedKernel,
    CompiledParameterRecord,
    CompiledParticleRecord,
    CompiledPropagatorRecord,
    CompiledVertexTerm,
)
from .tensors import (
    _constant_expression_ratio,
    classify_trilinear_color_expression,
    normalize_color_expression,
    normalize_lorentz_expression,
)

_YM_THREE_VECTOR = (
    "-UFO::Metric(UFO::idx(1,1),UFO::idx(1,2))"
    "*UFO::P(UFO::idx(1,3),UFO::idx(1,2))"
    "-UFO::Metric(UFO::idx(1,1),UFO::idx(1,3))"
    "*UFO::P(UFO::idx(1,2),UFO::idx(1,1))"
    "-UFO::Metric(UFO::idx(1,2),UFO::idx(1,3))"
    "*UFO::P(UFO::idx(1,1),UFO::idx(1,3))"
    "+UFO::Metric(UFO::idx(1,1),UFO::idx(1,2))"
    "*UFO::P(UFO::idx(1,3),UFO::idx(1,1))"
    "+UFO::Metric(UFO::idx(1,1),UFO::idx(1,3))"
    "*UFO::P(UFO::idx(1,2),UFO::idx(1,3))"
    "+UFO::Metric(UFO::idx(1,2),UFO::idx(1,3))"
    "*UFO::P(UFO::idx(1,1),UFO::idx(1,2))"
)

_YM_FOUR_VECTOR_BASIS = (
    (
        "UFO::f(-1,1,2)*UFO::f(3,4,-1)",
        "-UFO::Metric(UFO::idx(1,1),UFO::idx(1,3))"
        "*UFO::Metric(UFO::idx(1,2),UFO::idx(1,4))"
        "+UFO::Metric(UFO::idx(1,1),UFO::idx(1,4))"
        "*UFO::Metric(UFO::idx(1,2),UFO::idx(1,3))",
    ),
    (
        "UFO::f(-1,1,3)*UFO::f(2,4,-1)",
        "-UFO::Metric(UFO::idx(1,1),UFO::idx(1,2))"
        "*UFO::Metric(UFO::idx(1,3),UFO::idx(1,4))"
        "+UFO::Metric(UFO::idx(1,1),UFO::idx(1,4))"
        "*UFO::Metric(UFO::idx(1,2),UFO::idx(1,3))",
    ),
    (
        "UFO::f(-1,1,4)*UFO::f(2,3,-1)",
        "-UFO::Metric(UFO::idx(1,1),UFO::idx(1,2))"
        "*UFO::Metric(UFO::idx(1,3),UFO::idx(1,4))"
        "+UFO::Metric(UFO::idx(1,1),UFO::idx(1,3))"
        "*UFO::Metric(UFO::idx(1,2),UFO::idx(1,4))",
    ),
)

_CONTACT_ORIGIN = re.compile(r"^ufo-contact:(?P<term>[0-9]+):")
_CONTRACTED_INDEX = re.compile(
    r"^ufo_(?P<family>[cls])_dummy_(?P<label>[0-9]+)"
    r"(?:_(?P<representation>[A-Za-z0-9_]+))?$"
)


@dataclass(frozen=True, slots=True)
class ExternalSymmetryCertificates:
    """Fail-closed model-level certificates consumed by DAG generation."""

    parity_kernel_kinds: frozenset[int]
    yang_mills_kernel_kinds: frozenset[int]
    yang_mills_adjoint_names: frozenset[str]
    adjoint_current_reflection_phases: tuple[
        tuple[int, tuple[float, float]], ...
    ]
    parity_kernel_digests: tuple[tuple[int, str], ...]
    yang_mills_kernel_digests: tuple[tuple[int, str], ...]
    yang_mills_adjoint_digests: tuple[tuple[str, str], ...]
    adjoint_current_reflection_digests: tuple[tuple[int, str], ...]

    def __post_init__(self) -> None:
        expected = (
            (self.parity_kernel_kinds, self.parity_kernel_digests),
            (self.yang_mills_kernel_kinds, self.yang_mills_kernel_digests),
            (self.yang_mills_adjoint_names, self.yang_mills_adjoint_digests),
            (
                frozenset(
                    kind for kind, _phase in self.adjoint_current_reflection_phases
                ),
                self.adjoint_current_reflection_digests,
            ),
        )
        for selectors, digests in expected:
            if len(digests) != len(selectors) or {
                selector for selector, _digest in digests
            } != set(selectors):
                raise ValueError("symmetry certificate digest selectors are incomplete")
            if any(
                len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                for _selector, digest in digests
            ):
                raise ValueError("symmetry certificate contains an invalid SHA-256")


def derive_external_symmetry_certificates(
    ir: CompiledModelIR,
) -> ExternalSymmetryCertificates:
    """Recognize exact vectorlike and Yang--Mills sectors in compiled IR."""

    _sym._ensure_symbolica()
    model_symbols = symbols.model(ir.name)
    particles = {particle.name: particle for particle in ir.particles}
    parameters = {parameter.name: parameter for parameter in ir.parameters}
    propagators = {propagator.name: propagator for propagator in ir.propagators}
    couplings = {coupling.name: coupling for coupling in ir.couplings}

    ym_three_reference = normalize_lorentz_expression(
        _YM_THREE_VECTOR,
        (3, 3, 3),
        model_symbols=model_symbols,
    ).expression
    ym_three_color_reference = normalize_color_expression(
        "UFO::f(1,2,3)",
        (8, 8, 8),
    ).expression
    ym_cubic_terms: dict[int, str] = {}
    ym_coupling_by_adjoint: dict[str, _sym.Expression] = {}
    vectorlike_terms: set[int] = set()

    for term in ir.vertex_terms:
        records = tuple(particles[name] for name in term.particles)
        ym_coupling = _yang_mills_cubic_effective_coupling(
            term,
            records,
            couplings=couplings,
            ym_three_color_reference=ym_three_color_reference,
            ym_three_reference=ym_three_reference,
            parameters=parameters,
            propagators=propagators,
        )
        if ym_coupling is not None:
            adjoint_name = records[0].name
            ym_cubic_terms[term.id] = adjoint_name
            previous = ym_coupling_by_adjoint.get(adjoint_name, _sym.E("0"))
            ym_coupling_by_adjoint[adjoint_name] = (
                previous + ym_coupling
            ).expand()
            continue
        if _is_vectorlike_gauge_term(
            term,
            records,
            model_symbols=model_symbols,
            parameters=parameters,
            propagators=propagators,
        ):
            vectorlike_terms.add(term.id)

    quartic_groups = _certified_yang_mills_quartic_groups(
        ir,
        particles,
        couplings,
        ym_coupling_by_adjoint,
        model_symbols=model_symbols,
        parameters=parameters,
        propagators=propagators,
    )
    ym_quartic_terms = {
        term_id
        for term_ids in quartic_groups.values()
        for term_id in term_ids
    }
    candidate_adjoint_names = frozenset(
        name
        for name in ym_coupling_by_adjoint
        if name in quartic_groups
    )

    parity_kinds: set[int] = set()
    yang_mills_kinds: set[int] = set()
    for kernel in ir.oriented_kernels:
        term_ids = frozenset(kernel.term_ids or (kernel.term_id,))
        if term_ids and term_ids <= vectorlike_terms:
            parity_kinds.add(kernel.kind)
        if term_ids and term_ids <= set(ym_cubic_terms):
            names = {ym_cubic_terms[term_id] for term_id in term_ids}
            if names <= candidate_adjoint_names:
                parity_kinds.add(kernel.kind)
                yang_mills_kinds.add(kernel.kind)
                continue
        if term_ids and term_ids <= ym_quartic_terms:
            group_names = {
                name
                for name, group_term_ids in quartic_groups.items()
                if term_ids <= group_term_ids
            }
            if group_names:
                parity_kinds.add(kernel.kind)
                yang_mills_kinds.add(kernel.kind)
                continue

        origin_term_id = _contact_origin_term_id(kernel.particles, particles)
        if origin_term_id is not None and origin_term_id in ym_quartic_terms:
            parity_kinds.add(kernel.kind)
            yang_mills_kinds.add(kernel.kind)

    reflection_phases = _certified_two_source_adjoint_current_reflections(
        ir,
        particles,
        parameters,
        model_symbols=model_symbols,
    )
    complete_adjoint_names = frozenset(
        name
        for name in candidate_adjoint_names
        if _yang_mills_sector_is_closed(
            name,
            ir.oriented_kernels,
            yang_mills_kernel_kinds=frozenset(yang_mills_kinds),
        )
    )
    kernels_by_kind = {kernel.kind: kernel for kernel in ir.oriented_kernels}
    kernel_digests = {
        kind: _kernel_contract_digest(kernels_by_kind[kind])
        for kind in parity_kinds | yang_mills_kinds | set(reflection_phases)
    }
    yang_mills_term_ids_by_name = {
        name: frozenset(
            {
                term_id
                for term_id, term_name in ym_cubic_terms.items()
                if term_name == name
            }
            | set(quartic_groups.get(name, ()))
        )
        for name in complete_adjoint_names
    }
    adjoint_digests = {
        name: _adjoint_sector_contract_digest(
            name,
            ir,
            particle=particles[name],
            propagator=(
                None
                if particles[name].propagator is None
                else propagators.get(particles[name].propagator)
            ),
            kernel_digests=kernel_digests,
            kernel_kinds=frozenset(yang_mills_kinds),
            term_ids=yang_mills_term_ids_by_name[name],
        )
        for name in complete_adjoint_names
    }

    return ExternalSymmetryCertificates(
        parity_kernel_kinds=frozenset(parity_kinds),
        yang_mills_kernel_kinds=frozenset(yang_mills_kinds),
        yang_mills_adjoint_names=complete_adjoint_names,
        adjoint_current_reflection_phases=tuple(sorted(reflection_phases.items())),
        parity_kernel_digests=tuple(
            (kind, kernel_digests[kind]) for kind in sorted(parity_kinds)
        ),
        yang_mills_kernel_digests=tuple(
            (kind, kernel_digests[kind]) for kind in sorted(yang_mills_kinds)
        ),
        yang_mills_adjoint_digests=tuple(sorted(adjoint_digests.items())),
        adjoint_current_reflection_digests=tuple(
            (kind, kernel_digests[kind]) for kind in sorted(reflection_phases)
        ),
    )


def _contract_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _kernel_contract_digest(kernel: CompiledOrientedKernel) -> str:
    """Bind a theorem to the exact oriented executable kernel contract."""

    return _contract_digest(
        {
            "kind": kernel.kind,
            "term_ids": list(kernel.term_ids or (kernel.term_id,)),
            "particles": list(kernel.particles),
            "source_particle_legs": list(kernel.source_particle_legs),
            "component_expressions": list(kernel.component_expressions),
            "coupling_expression": kernel.coupling_expression,
            "coupling_orders": [list(item) for item in kernel.coupling_orders],
            "runtime_parameters": list(kernel.runtime_parameters),
            "color_expression": kernel.color_expression,
            "color_projection_structure": kernel.color_projection_structure,
            "color_projection_coefficient": kernel.color_projection_coefficient,
            "lc_color_normalization_power": kernel.lc_color_normalization_power,
            "input_ordering_ids": list(kernel.input_ordering_ids),
            "output_ordering_id": kernel.output_ordering_id,
        }
    )


def _adjoint_sector_contract_digest(
    name: str,
    ir: CompiledModelIR,
    *,
    particle: CompiledParticleRecord,
    propagator: CompiledPropagatorRecord | None,
    kernel_digests: dict[int, str],
    kernel_kinds: frozenset[int],
    term_ids: frozenset[int],
) -> str:
    """Bind global Yang--Mills theorems to source, kernel, and color data."""

    current_orderings = tuple(
        ordering.to_json_dict()
        for ordering in ir.current_orderings
        if ordering.particle == name
    )
    terms = tuple(
        {
            "id": term.id,
            "particles": list(term.particles),
            "tensor_product_signature": _tensor_product_signature(
                term.color_expression,
                term.lorentz_expression,
            ),
            "coupling_expression": term.coupling_expression,
            "coupling_orders": [list(item) for item in term.coupling_orders],
            "backend": term.backend,
            "lc_color_normalization_power": term.lc_color_normalization_power,
            "source_ordering_ids": list(term.source_ordering_ids),
        }
        for term in ir.vertex_terms
        if term.id in term_ids
    )
    kernels = tuple(
        (kernel.kind, kernel_digests[kernel.kind])
        for kernel in ir.oriented_kernels
        if kernel.kind in kernel_kinds and name in kernel.particles
    )
    return _contract_digest(
        {
            "particle": {
                "name": particle.name,
                "antiname": particle.antiname,
                "spin": particle.spin,
                "color": particle.color,
                "mass": particle.mass,
                "width": particle.width,
                "quantum_numbers": [list(item) for item in particle.quantum_numbers],
                "ghost_number": particle.ghost_number,
                "propagating": particle.propagating,
                "goldstoneboson": particle.goldstoneboson,
                "component_dimension": particle.component_dimension,
                "auxiliary_kind": particle.auxiliary_kind,
                "statistics": particle.statistics,
                "wavefunction_family": particle.wavefunction_family,
                "color_role": particle.color_role,
                "self_conjugate": particle.self_conjugate,
                "source_orientation": particle.source_orientation,
            },
            "propagator": (
                None
                if propagator is None
                else {
                    "numerator": propagator.numerator,
                    "denominator": propagator.denominator,
                    "custom": propagator.custom,
                }
            ),
            "current_orderings": current_orderings,
            "vertex_terms": terms,
            "oriented_kernels": kernels,
        }
    )


def _yang_mills_sector_is_closed(
    source_name: str,
    kernels: tuple[CompiledOrientedKernel, ...],
    *,
    yang_mills_kernel_kinds: frozenset[int],
) -> bool:
    """Require every tree-reachable interaction to remain Yang--Mills.

    A certified cubic and quartic tensor is insufficient for global helicity
    and trace-reflection theorems when the same adjoint source can also produce
    another state.  Follow the trivalent compiled theory from two reachable
    currents, including synthetic contact auxiliaries, and fail closed as soon
    as any reachable kernel lacks the exact Yang--Mills certificate.
    """

    reachable = {source_name}
    changed = True
    while changed:
        changed = False
        for kernel in kernels:
            left_name, right_name, result_name = kernel.particles
            if left_name not in reachable or right_name not in reachable:
                continue
            if kernel.kind not in yang_mills_kernel_kinds:
                return False
            if result_name not in reachable:
                reachable.add(result_name)
                changed = True
    return True


def _yang_mills_cubic_effective_coupling(
    term: CompiledVertexTerm,
    particles: tuple[CompiledParticleRecord, ...],
    *,
    couplings: dict[str, CompiledCouplingRecord],
    ym_three_color_reference: str,
    ym_three_reference: str,
    parameters: dict[str, CompiledParameterRecord],
    propagators: dict[str, CompiledPropagatorRecord],
) -> _sym.Expression | None:
    if len(particles) != 3 or len({particle.name for particle in particles}) != 1:
        return None
    particle = particles[0]
    if not _is_compile_time_massless_adjoint_vector(
        particle,
        parameters,
        propagators,
    ):
        return None
    structure, _coefficient = classify_trilinear_color_expression(
        term.color_expression,
        term.color_source,
        tuple(item.color for item in particles),
    )
    if structure != "adjoint-structure-constant":
        return None
    color_ratio = _constant_expression_ratio_exact(
        _sym.E(term.color_expression),
        _sym.E(ym_three_color_reference),
    )
    lorentz_ratio = _constant_expression_ratio_exact(
        _sym.E(term.lorentz_expression),
        _sym.E(ym_three_reference),
    )
    coupling = couplings.get(term.coupling)
    if color_ratio is None or lorentz_ratio is None or coupling is None:
        return None
    effective = (
        _sym.E(coupling.expression) * color_ratio * lorentz_ratio
    ).expand()
    return None if effective == _sym.E("0") else effective


def _is_vectorlike_gauge_term(
    term: CompiledVertexTerm,
    particles: tuple[CompiledParticleRecord, ...],
    *,
    model_symbols: ModelSymbolRegistry,
    parameters: dict[str, CompiledParameterRecord],
    propagators: dict[str, CompiledPropagatorRecord],
) -> bool:
    if len(particles) != 3:
        return False
    vector_legs = tuple(
        index for index, particle in enumerate(particles) if particle.spin == 3
    )
    fermion_legs = tuple(
        index for index, particle in enumerate(particles) if particle.spin == 2
    )
    if len(vector_legs) != 1 or len(fermion_legs) != 2:
        return False
    vector = particles[vector_legs[0]]
    left_fermion, right_fermion = (particles[index] for index in fermion_legs)
    if not _is_compile_time_massless_adjoint_vector(
        vector,
        parameters,
        propagators,
    ):
        return False
    if sorted((left_fermion.color, right_fermion.color)) != [-3, 3]:
        return False
    if (
        left_fermion.antiname != right_fermion.name
        or right_fermion.antiname != left_fermion.name
    ):
        return False
    structure, _coefficient = classify_trilinear_color_expression(
        term.color_expression,
        term.color_source,
        tuple(item.color for item in particles),
    )
    if structure != "fundamental-generator":
        return False

    vector_leg = vector_legs[0] + 1
    references = tuple(
        normalize_lorentz_expression(
            "UFO::Gamma("
            f"UFO::idx(1,{vector_leg}),"
            f"UFO::idx(1,{output_leg + 1}),"
            f"UFO::idx(1,{input_leg + 1}))",
            tuple(particle.spin for particle in particles),
            model_symbols=model_symbols,
        ).expression
        for output_leg, input_leg in (
            fermion_legs,
            tuple(reversed(fermion_legs)),
        )
    )
    return any(
        _expressions_are_constant_multiples(term.lorentz_expression, reference)
        for reference in references
    )


def _certified_yang_mills_quartic_groups(
    ir: CompiledModelIR,
    particles: dict[str, CompiledParticleRecord],
    couplings: dict[str, CompiledCouplingRecord],
    cubic_couplings: dict[str, _sym.Expression],
    *,
    model_symbols: ModelSymbolRegistry,
    parameters: dict[str, CompiledParameterRecord],
    propagators: dict[str, CompiledPropagatorRecord],
) -> dict[str, frozenset[int]]:
    references = tuple(
        (
            normalize_color_expression(color, (8, 8, 8, 8)).expression,
            normalize_lorentz_expression(
                lorentz,
                (3, 3, 3, 3),
                model_symbols=model_symbols,
            ).expression,
        )
        for color, lorentz in _YM_FOUR_VECTOR_BASIS
    )
    terms_by_particles: dict[
        tuple[str, ...], list[CompiledVertexTerm]
    ] = defaultdict(list)
    for term in ir.vertex_terms:
        if len(term.particles) == 4:
            terms_by_particles[term.particles].append(term)

    certified: dict[str, frozenset[int]] = {}
    for particle_names, terms in terms_by_particles.items():
        if len(set(particle_names)) != 1:
            continue
        name = particle_names[0]
        particle = particles[name]
        cubic = cubic_couplings.get(name)
        if cubic is None or not _is_compile_time_massless_adjoint_vector(
            particle,
            parameters,
            propagators,
        ):
            continue
        quartic_coefficients = [_sym.E("0") for _ in references]
        valid = True
        for term in terms:
            matches = tuple(
                (index, ratio)
                for index, (color_reference, lorentz_reference) in enumerate(
                    references
                )
                if (
                    ratio := _tensor_product_ratio(
                        term.color_expression,
                        term.lorentz_expression,
                        color_reference,
                        lorentz_reference,
                    )
                )
                is not None
            )
            coupling = couplings.get(term.coupling)
            if len(matches) != 1 or coupling is None:
                valid = False
                break
            index, ratio = matches[0]
            quartic_coefficients[index] = (
                quartic_coefficients[index]
                + _sym.E(coupling.expression) * ratio
            ).expand()
        if not valid or not _has_yang_mills_coupling_relation(
            cubic,
            tuple(quartic_coefficients),
        ):
            continue
        certified[name] = frozenset(term.id for term in terms)
    return certified


def _has_yang_mills_coupling_relation(
    cubic: _sym.Expression,
    quartic_coefficients: tuple[_sym.Expression, ...],
) -> bool:
    if cubic == _sym.E("0") or not quartic_coefficients:
        return False
    expected = (_sym.E("1i") * cubic**2).expand()
    return all(
        (coefficient - expected).cancel().expand() == _sym.E("0")
        for coefficient in quartic_coefficients
    )


def _constant_expression_ratio_exact(
    target: _sym.Expression,
    reference: _sym.Expression,
) -> _sym.Expression | None:
    """Return an exact scalar tensor ratio without converting through float."""

    if reference == _sym.E("0"):
        return None
    ratio = (target / reference).cancel()
    if ratio.get_all_symbols(False):
        return None
    if (target - ratio * reference).expand() != _sym.E("0"):
        return None
    return ratio


def _tensor_product_ratio(
    color: str,
    lorentz: str,
    reference_color: str,
    reference_lorentz: str,
) -> _sym.Expression | None:
    _sym._ensure_symbolica()
    target_candidates = _canonical_tensor_product_expressions(
        (_sym.E(color) * _sym.E(lorentz)).expand()
    )
    reference_candidates = _canonical_tensor_product_expressions(
        (_sym.E(reference_color) * _sym.E(reference_lorentz)).expand()
    )
    for target in target_candidates:
        for reference in reference_candidates:
            ratio = _constant_expression_ratio_exact(target, reference)
            if ratio is not None and ratio != _sym.E("0"):
                return ratio
    return None


def _tensor_product_signature(color: str, lorentz: str) -> str:
    _sym._ensure_symbolica()
    expression = (_sym.E(color) * _sym.E(lorentz)).expand()
    return min(
        str(candidate.to_canonical_string())
        for candidate in _canonical_tensor_product_expressions(expression)
    )


def _canonical_tensor_product_expressions(
    expression: _sym.Expression,
) -> tuple[_sym.Expression, ...]:
    groups: dict[tuple[str, str], list[_sym.Expression]] = defaultdict(list)
    for symbol in expression.get_all_symbols():
        match = _CONTRACTED_INDEX.fullmatch(str(symbol))
        if match is None:
            continue
        key = (match.group("family"), match.group("representation") or "index")
        groups[key].append(symbol)
    if not groups:
        return (expression,)

    contracted_indices: list[tuple[_sym.Expression, int]] = []
    group_contracts: dict[int, tuple[str, str]] = {}
    for group, ((family, representation), source_symbols) in enumerate(
        sorted(groups.items())
    ):
        group_contracts[group] = (family, representation)
        contracted_indices.extend(
            (symbol, group)
            for symbol in source_symbols
        )

    canonical, external, dummy = expression.canonize_tensors(contracted_indices)
    if external:
        raise ValueError(
            "contracted UFO tensor indices unexpectedly became external during "
            "canonization"
        )
    group_offsets: dict[int, int] = defaultdict(int)
    for source, group_expression in dummy:
        group = int(str(group_expression))
        family, representation = group_contracts[group]
        target = symbols.canonical_tensor_index(
            family,
            representation,
            group_offsets[group],
        )
        group_offsets[group] += 1
        canonical = canonical.replace(source, target)

    if any(
        _CONTRACTED_INDEX.fullmatch(str(symbol))
        for symbol in canonical.get_all_symbols()
    ):
        raise ValueError(
            "Symbolica tensor canonization left a source-owned contracted index"
        )
    return (canonical,)


def _expressions_are_constant_multiples(target: str, reference: str) -> bool:
    target_expression = _sym.E(target)
    reference_expression = _sym.E(reference)
    if target_expression == reference_expression:
        return True
    return (
        _constant_expression_ratio(target_expression, reference_expression) is not None
    )


def _is_compile_time_massless_adjoint_vector(
    particle: CompiledParticleRecord,
    parameters: dict[str, CompiledParameterRecord],
    propagators: dict[str, CompiledPropagatorRecord],
) -> bool:
    propagator = (
        None
        if particle.propagator is None
        else propagators.get(particle.propagator)
    )
    return (
        particle.spin == 3
        and particle.color == 8
        and particle.self_conjugate is True
        and _parameter_is_compile_time_zero(particle.mass, parameters)
        and particle.propagating
        and particle.ghost_number == 0
        and propagator is not None
        and not propagator.custom
    )


def _parameter_is_compile_time_zero(
    name: str,
    parameters: dict[str, CompiledParameterRecord],
) -> bool:
    parameter = parameters.get(name)
    if parameter is None or parameter.nature != "internal":
        return False
    return (
        _sym.E(parameter.resolved_expression).expand().to_canonical_string()
        == _sym.E("0").to_canonical_string()
    )


def _certified_two_source_adjoint_current_reflections(
    ir: CompiledModelIR,
    particles: dict[str, CompiledParticleRecord],
    parameters: dict[str, CompiledParameterRecord],
    *,
    model_symbols: ModelSymbolRegistry,
) -> dict[int, tuple[float, float]]:
    """Prove local two-input current reflection from lowered expressions.

    The proof is intentionally narrower than a Yang--Mills model
    classification.  Every cubic kernel of a candidate gauge species must be
    exactly antisymmetric under exchanging its formal input currents and
    momenta, must carry an exact structure-constant color tensor, and must use
    the compiler's standard linear propagator.  One failed kernel disables the
    optimization for the complete species.
    """

    propagators = {propagator.name: propagator for propagator in ir.propagators}
    kernels_by_name: dict[str, list[CompiledOrientedKernel]] = defaultdict(list)
    for kernel in ir.oriented_kernels:
        if len(set(kernel.particles)) == 1:
            kernels_by_name[kernel.particles[0]].append(kernel)

    certified: dict[int, tuple[float, float]] = {}
    for name, kernels in kernels_by_name.items():
        particle = particles.get(name)
        if particle is None or not _is_compile_time_massless_adjoint_vector(
            particle,
            parameters,
            propagators,
        ):
            continue
        propagator = (
            None
            if particle.propagator is None
            else propagators.get(particle.propagator)
        )
        if propagator is None or propagator.custom:
            continue
        if not kernels or any(
            not _kernel_has_exact_adjoint_reflection(
                kernel,
                model_symbols=model_symbols,
            )
            for kernel in kernels
        ):
            continue
        certified.update({kernel.kind: (-1.0, 0.0) for kernel in kernels})
    return certified


def _kernel_has_exact_adjoint_reflection(
    kernel: CompiledOrientedKernel,
    *,
    model_symbols: ModelSymbolRegistry,
) -> bool:
    from .compiler_kernels import (
        _remap_kernel_symbols,
    )

    color_reference = normalize_color_expression(
        "UFO::f(1,2,3)",
        (8, 8, 8),
    ).expression
    color_ratio = _constant_expression_ratio(
        _sym.E(kernel.color_expression),
        _sym.E(color_reference),
    )
    if color_ratio is None or color_ratio == 0j:
        return False

    coupling = _sym.E(kernel.coupling_expression)
    zero = _sym.E("0").to_canonical_string()
    for component_source in kernel.component_expressions:
        component = _sym.E(component_source) * coupling
        direct = _remap_kernel_symbols(
            component,
            old_kind=kernel.kind,
            new_kind=0,
            model_symbols=model_symbols,
        )
        swapped = _remap_kernel_symbols(
            component,
            old_kind=kernel.kind,
            new_kind=0,
            model_symbols=model_symbols,
            swap_sides=True,
        )
        if (swapped + direct).expand().to_canonical_string() != zero:
            return False
    return True


def _contact_origin_term_id(
    particle_names: tuple[str, str, str],
    particles: dict[str, CompiledParticleRecord],
) -> int | None:
    for name in particle_names:
        auxiliary_kind = particles[name].auxiliary_kind
        if auxiliary_kind is None:
            continue
        match = _CONTACT_ORIGIN.match(auxiliary_kind)
        if match is not None:
            return int(match.group("term"))
    return None


__all__ = [
    "ExternalSymmetryCertificates",
    "derive_external_symmetry_certificates",
]

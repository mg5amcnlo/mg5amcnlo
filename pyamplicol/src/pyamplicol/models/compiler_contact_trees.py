# SPDX-License-Identifier: 0BSD
"""Color-singlet four-point contact-tree compilation."""

from __future__ import annotations

import math
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from itertools import product

from .._internal.physics.symbols import ModelSymbolRegistry, symbols
from . import compiler_symbolica as _sym
from .compiler_contacts import (
    _execute_dense_tensor,
    _normalized_structure_constant_factors,
    _source_structure_constant_factor_orientations,
    _source_structure_constant_product_coefficient,
)
from .compiler_kernels import (
    _as_expression,
    _canonicalize_oriented_kernel_component,
    _function_arguments,
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
    CONTACT_ORBIT_ALGORITHM,
    CONTACT_ORBIT_ALGORITHM_VERSION,
    HEFT_CONTACT_ORBIT_EVALUATOR_CLASS,
    CompiledContactOrbitCertificate,
    CompiledContactOrbitStep,
    canonical_contact_orbit_steps,
)
from .contracts import (
    CompiledOrientedKernel,
    CompiledParticleRecord,
    CompiledVertexTerm,
    _ContactTreeNode,
    compiled_particle_component_dimension,
)
from .tensors import normalize_color_expression, project_trilinear_color_expression

_DIRECT_DENSE_CONTACT_INPUT_WORK = 1_024
_MAX_STAGED_CONTACT_INPUT_WORK = 256


def _compile_color_singlet_contact_trees(
    terms: Sequence[CompiledVertexTerm],
    particles: Sequence[CompiledParticleRecord],
    *,
    start_kind: int,
    model_symbols: ModelSymbolRegistry,
) -> tuple[tuple[CompiledParticleRecord, ...], tuple[CompiledOrientedKernel, ...]]:
    """Lower arbitrary color-singlet contacts to balanced trivalent trees."""

    particle_by_name = {particle.name: particle for particle in particles}
    used_pdgs = {abs(particle.pdg_code) for particle in particles}
    next_pdg = max(9_000_000, max(used_pdgs, default=0) + 1)
    auxiliary_particles: list[CompiledParticleRecord] = []
    kernels: list[CompiledOrientedKernel] = []
    final_template_cache: dict[
        tuple[object, ...],
        tuple[int, tuple[_sym.Expression, ...]],
    ] = {}

    def allocate_pdg() -> int:
        nonlocal next_pdg
        while next_pdg in used_pdgs:
            next_pdg += 1
        result = next_pdg
        used_pdgs.add(result)
        next_pdg += 1
        return result

    for term in terms:
        if term.valence < 4 or not _contact_term_is_color_singlet(term):
            continue
        if term.valence == 4 and "ufo_momentum_" not in term.lorentz_expression:
            continue
        source_particles = tuple(particle_by_name[name] for name in term.particles)
        scalar_product_tree = all(particle.spin == 1 for particle in source_particles)
        oriented_result_particles: set[str] = set()
        for result_leg in range(term.valence):
            source_result = source_particles[result_leg]
            if source_result.name in oriented_result_particles:
                continue
            oriented_result_particles.add(source_result.name)
            input_legs = tuple(leg for leg in range(term.valence) if leg != result_leg)

            def build_node(
                legs: tuple[int, ...],
                *,
                source_particles: tuple[CompiledParticleRecord, ...] = source_particles,
                scalar_product_tree: bool = scalar_product_tree,
                term: CompiledVertexTerm = term,
                result_leg: int = result_leg,
            ) -> _ContactTreeNode:
                if len(legs) == 1:
                    leg = legs[0]
                    return _ContactTreeNode(
                        legs=legs,
                        particle=source_particles[leg],
                        physical_leg=leg,
                    )
                split = len(legs) // 2
                left_node = build_node(legs[:split])
                right_node = build_node(legs[split:])
                auxiliary_name = (
                    f"__pyamplicol_contact_tree_{term.id}_r{result_leg}_"
                    + "_".join(str(leg) for leg in legs)
                )
                auxiliary_dimension = (
                    1
                    if scalar_product_tree
                    else sum(
                        _spin_dimension(source_particles[leg].spin) + 4 for leg in legs
                    )
                )
                auxiliary = CompiledParticleRecord(
                    name=auxiliary_name,
                    antiname=auxiliary_name,
                    pdg_code=allocate_pdg(),
                    spin=-1,
                    color=1,
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
                        f"ufo-contact-tree:{term.id}:result-{result_leg}:"
                        + ",".join(str(leg) for leg in legs)
                    ),
                )
                auxiliary_particles.append(auxiliary)
                node = _ContactTreeNode(
                    legs=legs,
                    particle=auxiliary,
                    left=left_node,
                    right=right_node,
                )
                _append_contact_tree_partial_kernels(
                    kernels,
                    term=term,
                    node=node,
                    source_particles=source_particles,
                    scalar_product_tree=scalar_product_tree,
                    start_kind=start_kind,
                    model_symbols=model_symbols,
                )
                return node

            split = len(input_legs) // 2
            left_root = build_node(input_legs[:split])
            right_root = build_node(input_legs[split:])
            assignment_multiplicity = _contact_tree_assignment_multiplicity(
                input_legs,
                source_particles,
                left_root,
                right_root,
            )
            _append_contact_tree_final_kernels(
                kernels,
                term=term,
                source_particles=source_particles,
                left_node=left_root,
                right_node=right_root,
                result_leg=result_leg,
                result_name=particle_by_name[source_result.antiname].name,
                scalar_product_tree=scalar_product_tree,
                assignment_multiplicity=assignment_multiplicity,
                start_kind=start_kind,
                template_cache=final_template_cache,
                model_symbols=model_symbols,
            )
    return tuple(auxiliary_particles), tuple(kernels)


def _compile_heft_colored_contact_trees(
    terms: Sequence[CompiledVertexTerm],
    particles: Sequence[CompiledParticleRecord],
    *,
    start_kind: int,
    model_symbols: ModelSymbolRegistry,
) -> tuple[tuple[CompiledParticleRecord, ...], tuple[CompiledOrientedKernel, ...]]:
    """Lower only the scalar-HEFT one- and two-``f`` contact families."""

    particle_by_name = {particle.name: particle for particle in particles}
    used_pdgs = {abs(particle.pdg_code) for particle in particles}
    next_pdg = max(9_050_000, max(used_pdgs, default=0) + 1)
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
        source_particles = tuple(particle_by_name[name] for name in term.particles)
        topology = _heft_contact_color_topology(
            term,
            source_particles,
            model_symbols=model_symbols,
        )
        if topology is None:
            continue
        factors, source_coefficient = topology
        singlet_leg = next(
            index
            for index, particle in enumerate(source_particles)
            if particle.color == 1
        )
        colored_legs = tuple(
            index
            for index, particle in enumerate(source_particles)
            if particle.color == 8
        )

        def leaf(
            leg: int,
            *,
            source_particles: tuple[CompiledParticleRecord, ...] = source_particles,
        ) -> _ContactTreeNode:
            return _ContactTreeNode(
                legs=(leg,),
                particle=source_particles[leg],
                physical_leg=leg,
            )

        def combine(
            left: _ContactTreeNode,
            right: _ContactTreeNode,
            *,
            result_leg: int,
            label: str,
            factor: tuple[int, int, int] | None,
            left_token: int | None,
            right_token: int | None,
            output_token: int,
            term: CompiledVertexTerm = term,
            source_particles: tuple[CompiledParticleRecord, ...] = source_particles,
        ) -> _ContactTreeNode:
            legs = (*left.legs, *right.legs)
            auxiliary_name = (
                f"__pyamplicol_heft_contact_{term.id}_r{result_leg}_{label}"
            )
            auxiliary = CompiledParticleRecord(
                name=auxiliary_name,
                antiname=auxiliary_name,
                pdg_code=allocate_pdg(),
                spin=-1,
                color=8,
                mass="ZERO",
                width="ZERO",
                charge=0.0,
                quantum_numbers=(("electric_charge", "0"),),
                ghost_number=0,
                propagating=False,
                goldstoneboson=False,
                propagator=None,
                component_dimension=_heft_node_component_dimension(
                    legs,
                    source_particles,
                    include_momenta=term.valence == 4,
                ),
                auxiliary_kind=(
                    f"ufo-heft-contact-tree:{term.id}:result-{result_leg}:"
                    + ",".join(str(leg) for leg in legs)
                ),
            )
            auxiliary_particles.append(auxiliary)
            node = _ContactTreeNode(
                legs=legs,
                particle=auxiliary,
                left=left,
                right=right,
            )
            _append_heft_contact_partial_kernels(
                kernels,
                term=term,
                node=node,
                result_leg=result_leg,
                source_particles=source_particles,
                factor=factor,
                left_token=left_token,
                right_token=right_token,
                output_token=output_token,
                include_momenta=term.valence == 4,
                start_kind=start_kind,
                model_symbols=model_symbols,
            )
            return node

        oriented_result_particles: set[str] = set()
        for result_leg, source_result in enumerate(source_particles):
            if source_result.name in oriented_result_particles:
                continue
            oriented_result_particles.add(source_result.name)
            result_token = result_leg + 1 if source_result.color == 8 else None
            input_legs = tuple(leg for leg in range(term.valence) if leg != result_leg)

            if term.valence == 4:
                factor = factors[0]
                if result_token is None:
                    pair_legs = tuple(sorted(colored_legs[:2]))
                    pair_tokens = (pair_legs[0] + 1, pair_legs[1] + 1)
                    output_token = next(
                        token for token in factor if token not in set(pair_tokens)
                    )
                    left_root = combine(
                        leaf(pair_legs[0]),
                        leaf(pair_legs[1]),
                        result_leg=result_leg,
                        label="f",
                        factor=factor,
                        left_token=pair_tokens[0],
                        right_token=pair_tokens[1],
                        output_token=output_token,
                    )
                    remaining_leg = next(
                        leg for leg in input_legs if leg not in set(pair_legs)
                    )
                    right_root = leaf(remaining_leg)
                    left_token = output_token
                    right_token = remaining_leg + 1
                    final_factor = None
                else:
                    remaining_colored = tuple(
                        leg for leg in colored_legs if leg != result_leg
                    )
                    if len(remaining_colored) != 2:
                        raise ValueError("scalar-HEFT Hggg tree has no gluon pair")
                    standalone_leg, identity_leg = remaining_colored
                    left_root = leaf(standalone_leg)
                    right_root = combine(
                        leaf(identity_leg),
                        leaf(singlet_leg),
                        result_leg=result_leg,
                        label="identity",
                        factor=None,
                        left_token=identity_leg + 1,
                        right_token=None,
                        output_token=identity_leg + 1,
                    )
                    left_token = standalone_leg + 1
                    right_token = identity_leg + 1
                    final_factor = factor
            else:
                dummy = next(value for value in factors[0] if value < 0)
                if result_token is None:
                    roots: list[_ContactTreeNode] = []
                    for index, factor in enumerate(factors):
                        pair_legs = tuple(
                            sorted(value - 1 for value in factor if value > 0)
                        )
                        roots.append(
                            combine(
                                leaf(pair_legs[0]),
                                leaf(pair_legs[1]),
                                result_leg=result_leg,
                                label=f"f{index}",
                                factor=factor,
                                left_token=pair_legs[0] + 1,
                                right_token=pair_legs[1] + 1,
                                output_token=dummy,
                            )
                        )
                    left_root, right_root = roots
                    left_token = right_token = dummy
                    final_factor = None
                else:
                    result_factor = next(
                        factor for factor in factors if result_token in factor
                    )
                    outer_factor = next(
                        factor for factor in factors if factor != result_factor
                    )
                    outer_legs = tuple(
                        sorted(value - 1 for value in outer_factor if value > 0)
                    )
                    left_root = combine(
                        leaf(outer_legs[0]),
                        leaf(outer_legs[1]),
                        result_leg=result_leg,
                        label="outer-f",
                        factor=outer_factor,
                        left_token=outer_legs[0] + 1,
                        right_token=outer_legs[1] + 1,
                        output_token=dummy,
                    )
                    remaining_token = next(
                        value
                        for value in result_factor
                        if value > 0 and value != result_token
                    )
                    remaining_leg = remaining_token - 1
                    right_root = combine(
                        leaf(remaining_leg),
                        leaf(singlet_leg),
                        result_leg=result_leg,
                        label="identity",
                        factor=None,
                        left_token=remaining_token,
                        right_token=None,
                        output_token=remaining_token,
                    )
                    left_token = dummy
                    right_token = remaining_token
                    final_factor = result_factor

            assignment_multiplicity = _contact_tree_assignment_multiplicity(
                input_legs,
                source_particles,
                left_root,
                right_root,
            )
            _append_heft_contact_final_kernels(
                kernels,
                term=term,
                source_particles=source_particles,
                left_node=left_root,
                right_node=right_root,
                result_leg=result_leg,
                result_name=particle_by_name[source_result.antiname].name,
                factor=final_factor,
                left_token=left_token,
                right_token=right_token,
                output_token=result_token,
                source_coefficient=source_coefficient,
                assignment_multiplicity=assignment_multiplicity,
                start_kind=start_kind,
                model_symbols=model_symbols,
            )

    return tuple(auxiliary_particles), tuple(kernels)


def _record_heft_contact_orbit_certificates(
    terms: Sequence[CompiledVertexTerm],
    particles: Sequence[CompiledParticleRecord],
    *,
    model_symbols: ModelSymbolRegistry,
) -> tuple[CompiledVertexTerm, ...]:
    """Certify the exact physical-assignment orbit of supported HEFT trees."""

    particle_by_name = {particle.name: particle for particle in particles}
    certified: list[CompiledVertexTerm] = []
    for term in terms:
        source_particles = tuple(particle_by_name[name] for name in term.particles)
        if (
            _heft_contact_color_topology(
                term,
                source_particles,
                model_symbols=model_symbols,
            )
            is None
        ):
            certified.append(term)
            continue
        if term.contact_orbit_certificate is not None:
            raise ValueError(
                f"scalar-HEFT term {term.id} already has a contact-orbit certificate"
            )
        equivalence_classes: dict[str, int] = {}
        leg_classes = tuple(
            equivalence_classes.setdefault(particle.name, len(equivalence_classes))
            for particle in source_particles
        )
        certified.append(
            replace(
                term,
                contact_orbit_certificate=CompiledContactOrbitCertificate(
                    algorithm=CONTACT_ORBIT_ALGORITHM,
                    algorithm_version=CONTACT_ORBIT_ALGORITHM_VERSION,
                    term_id=term.id,
                    vertex=term.vertex,
                    particles=term.particles,
                    color_expression=term.color_expression,
                    lorentz_expression=term.lorentz_expression,
                    coupling_expression=term.coupling_expression,
                    evaluator_class=HEFT_CONTACT_ORBIT_EVALUATOR_CLASS,
                    physical_leg_equivalence_classes=leg_classes,
                    reconstruction_factor="1",
                ),
            )
        )
    return tuple(certified)


def _heft_contact_orbit_step(
    term: CompiledVertexTerm,
    *,
    stage: str,
    result_leg: int,
    left_covered_legs: tuple[int, ...],
    right_covered_legs: tuple[int, ...],
    source_particle_legs: tuple[int, int, int],
) -> CompiledContactOrbitStep:
    return CompiledContactOrbitStep(
        algorithm=CONTACT_ORBIT_ALGORITHM,
        algorithm_version=CONTACT_ORBIT_ALGORITHM_VERSION,
        term_id=term.id,
        stage=stage,
        result_leg=result_leg,
        left_covered_legs=left_covered_legs,
        right_covered_legs=right_covered_legs,
        source_particle_legs=source_particle_legs,
        reconstruction_factor="1",
    )


def _heft_contact_color_topology(
    term: CompiledVertexTerm,
    source_particles: Sequence[CompiledParticleRecord],
    *,
    model_symbols: ModelSymbolRegistry,
) -> tuple[tuple[tuple[int, int, int], ...], str] | None:
    colors = tuple(particle.color for particle in source_particles)
    if term.valence == 4 and colors.count(8) == 3 and colors.count(1) == 1:
        expected_count = 1
    elif term.valence == 5 and colors.count(8) == 4 and colors.count(1) == 1:
        expected_count = 2
    else:
        return None
    factors = _normalized_structure_constant_factors(
        term.color_expression,
        expected_count=expected_count,
    )
    if len(factors) != expected_count:
        return None
    colored_tokens = {
        index + 1
        for index, particle in enumerate(source_particles)
        if particle.color == 8
    }
    if expected_count == 1:
        if set(factors[0]) != colored_tokens or any(value < 1 for value in factors[0]):
            return None
    else:
        dummy_sets = tuple(
            {value for value in factor if value < 0} for factor in factors
        )
        if any(len(values) != 1 for values in dummy_sets):
            return None
        shared = dummy_sets[0] & dummy_sets[1]
        positives = tuple(value for factor in factors for value in factor if value > 0)
        if (
            len(shared) != 1
            or set(positives) != colored_tokens
            or len(positives) != len(set(positives))
            or any(len(factor) != 3 for factor in factors)
        ):
            return None
    source_factors = _function_arguments(term.color_source, "f")
    if len(source_factors) != expected_count:
        return None
    compact_source = re.sub(r"\s+", "", term.color_source)
    tensor = r"(?:UFO::(?:\{\}::)?)?f\(-?\d+,-?\d+,-?\d+\)"
    if (
        re.fullmatch(rf"{tensor}(?:\*{tensor}){{{expected_count - 1}}}", compact_source)
        is None
    ):
        return None
    coefficient = _source_structure_constant_product_coefficient(
        term.color_source,
        model_symbols=model_symbols,
    )
    if coefficient is None:
        return None
    oriented_factors = _source_structure_constant_factor_orientations(
        source_factors, factors
    )
    if oriented_factors is None:
        return None
    return oriented_factors, coefficient


def _heft_trilinear_color(
    particles: tuple[
        CompiledParticleRecord, CompiledParticleRecord, CompiledParticleRecord
    ],
) -> tuple[str, str, int]:
    representations = tuple(particle.color for particle in particles)
    colored_slots = tuple(
        index + 1
        for index, representation in enumerate(representations)
        if representation == 8
    )
    if len(colored_slots) == 3:
        source = "UFO::f(1,2,3)"
        expected = "adjoint-structure-constant"
        normalization_power = 1
    elif len(colored_slots) == 2 and set(representations) <= {1, 8}:
        source = f"UFO::Identity({colored_slots[0]},{colored_slots[1]})"
        expected = "color-identity"
        normalization_power = 0
    else:
        raise ValueError(
            f"scalar-HEFT contact node has unsupported colors {representations!r}"
        )
    expression = normalize_color_expression(source, representations).expression
    structure, coefficient = project_trilinear_color_expression(
        expression,
        representations,
    )
    if structure != expected or coefficient != 1.0 + 0.0j:
        raise ValueError(
            "scalar-HEFT contact node failed exact trilinear color projection"
        )
    return source, expression, normalization_power


def _heft_node_base_dimension(
    legs: Sequence[int],
    source_particles: Sequence[CompiledParticleRecord],
) -> int:
    return math.prod(_spin_dimension(source_particles[leg].spin) for leg in legs)


def _heft_node_component_dimension(
    legs: Sequence[int],
    source_particles: Sequence[CompiledParticleRecord],
    *,
    include_momenta: bool,
) -> int:
    base_dimension = _heft_node_base_dimension(legs, source_particles)
    return base_dimension * (1 + (4 * len(legs) if include_momenta else 0))


def _heft_node_base_index(
    node: _ContactTreeNode,
    source_particles: Sequence[CompiledParticleRecord],
    component_by_leg: Mapping[int, int],
) -> int:
    index = 0
    for leg in node.legs:
        dimension = _spin_dimension(source_particles[leg].spin)
        component = component_by_leg[leg]
        if not 0 <= component < dimension:
            raise ValueError("scalar-HEFT contact component is outside its basis")
        index = index * dimension + component
    return index


def _heft_node_base_expression(
    kind: int,
    side: str,
    node: _ContactTreeNode,
    source_particles: Sequence[CompiledParticleRecord],
    component_by_leg: Mapping[int, int],
    *,
    model_symbols: ModelSymbolRegistry,
) -> _sym.Expression:
    if node.is_leaf:
        if node.physical_leg is None:
            raise ValueError("scalar-HEFT contact leaf has no source leg")
        index = component_by_leg[node.physical_leg]
    else:
        index = _heft_node_base_index(node, source_particles, component_by_leg)
    return model_symbols.kernel_component(kind, side, index)


def _heft_node_momentum_weighted_expression(
    kind: int,
    side: str,
    node: _ContactTreeNode,
    source_particles: Sequence[CompiledParticleRecord],
    component_by_leg: Mapping[int, int],
    *,
    leg: int,
    momentum_component: int,
    model_symbols: ModelSymbolRegistry,
) -> _sym.Expression:
    base = _heft_node_base_expression(
        kind,
        side,
        node,
        source_particles,
        component_by_leg,
        model_symbols=model_symbols,
    )
    if node.is_leaf:
        if node.physical_leg != leg:
            raise ValueError("scalar-HEFT leaf momentum belongs to another source leg")
        return base * model_symbols.kernel_momentum(
            kind,
            side,
            momentum_component,
        )
    try:
        leg_position = node.legs.index(leg)
    except ValueError as error:
        raise ValueError(
            "scalar-HEFT auxiliary momentum belongs to another source node"
        ) from error
    base_dimension = _heft_node_base_dimension(node.legs, source_particles)
    base_index = _heft_node_base_index(node, source_particles, component_by_leg)
    jet_index = (
        base_dimension
        + (leg_position * 4 + momentum_component) * base_dimension
        + base_index
    )
    if jet_index >= compiled_particle_component_dimension(node.particle):
        raise ValueError("scalar-HEFT momentum jet is absent from an auxiliary")
    return model_symbols.kernel_component(kind, side, jet_index)


def _heft_contact_final_component_expressions(
    term: CompiledVertexTerm,
    source_particles: Sequence[CompiledParticleRecord],
    *,
    left_node: _ContactTreeNode,
    right_node: _ContactTreeNode,
    result_leg: int,
    kind: int,
    canonical_left_side: str,
    canonical_right_side: str,
    model_symbols: ModelSymbolRegistry,
) -> tuple[_sym.Expression, ...]:
    """Rebuild one HEFT Lorentz tensor linearly from authenticated node jets."""

    input_legs = tuple(leg for leg in range(term.valence) if leg != result_leg)
    if set((*left_node.legs, *right_node.legs)) != set(input_legs):
        raise ValueError("scalar-HEFT final nodes do not cover every input leg")
    component_placeholders = {
        leg: tuple(
            model_symbols.symbol(
                f"compiler::heft_term_{term.id}_result_{result_leg}_"
                f"leg_{leg}_component_{component}"
            )
            for component in range(_spin_dimension(source_particles[leg].spin))
        )
        for leg in input_legs
    }
    momentum_placeholders = {
        leg: tuple(
            model_symbols.symbol(
                f"compiler::heft_term_{term.id}_result_{result_leg}_"
                f"leg_{leg}_momentum_{component}"
            )
            for component in range(4)
        )
        for leg in input_legs
    }
    oracle_term = replace(term, color_source="1", color_expression="1")
    oracle_components = eager_color_singlet_vertex_term_components(
        oracle_term,
        source_particles,
        result_leg=result_leg,
        input_components=component_placeholders,
        input_momenta=momentum_placeholders,
        model_symbols=model_symbols,
    )
    zero = _sym.E("0")
    momentum_zeroes = {
        momentum: zero
        for values in momentum_placeholders.values()
        for momentum in values
    }
    component_ranges = tuple(
        range(_spin_dimension(source_particles[leg].spin)) for leg in input_legs
    )
    results: list[_sym.Expression] = []
    for source_component in oracle_components:
        expanded = source_component.expand()
        translated = zero
        recovered = zero
        for indices in product(*component_ranges):
            component_by_leg = dict(zip(input_legs, indices, strict=True))
            monomial = math.prod(
                component_placeholders[leg][component_by_leg[leg]] for leg in input_legs
            )
            coefficient = expanded.coefficient(monomial)
            if coefficient == zero:
                continue
            constant = _replace_expression_symbols(coefficient, momentum_zeroes)
            reconstructed_coefficient = constant
            left_base = _heft_node_base_expression(
                kind,
                canonical_left_side,
                left_node,
                source_particles,
                component_by_leg,
                model_symbols=model_symbols,
            )
            right_base = _heft_node_base_expression(
                kind,
                canonical_right_side,
                right_node,
                source_particles,
                component_by_leg,
                model_symbols=model_symbols,
            )
            translated += constant * left_base * right_base
            for leg in input_legs:
                node, side, other_base = (
                    (left_node, canonical_left_side, right_base)
                    if leg in left_node.legs
                    else (right_node, canonical_right_side, left_base)
                )
                for momentum_component, momentum in enumerate(
                    momentum_placeholders[leg]
                ):
                    momentum_coefficient = _replace_expression_symbols(
                        coefficient.coefficient(momentum),
                        momentum_zeroes,
                    )
                    if momentum_coefficient == zero:
                        continue
                    reconstructed_coefficient += momentum_coefficient * momentum
                    translated += (
                        momentum_coefficient
                        * _heft_node_momentum_weighted_expression(
                            kind,
                            side,
                            node,
                            source_particles,
                            component_by_leg,
                            leg=leg,
                            momentum_component=momentum_component,
                            model_symbols=model_symbols,
                        )
                        * other_base
                    )
            if (
                coefficient - reconstructed_coefficient
            ).expand().to_canonical_string() != "0":
                raise ValueError(
                    "scalar-HEFT contact Lorentz momentum degree exceeds one"
                )
            recovered += reconstructed_coefficient * monomial
        if (expanded - recovered).expand().to_canonical_string() != "0":
            raise ValueError(
                "scalar-HEFT contact Lorentz tensor is not multilinear in its inputs"
            )
        results.append(_replace_evaluator_constants(translated))
    return tuple(results)


def _append_heft_contact_partial_kernels(
    kernels: list[CompiledOrientedKernel],
    *,
    term: CompiledVertexTerm,
    node: _ContactTreeNode,
    result_leg: int,
    source_particles: Sequence[CompiledParticleRecord],
    factor: tuple[int, int, int] | None,
    left_token: int | None,
    right_token: int | None,
    output_token: int,
    include_momenta: bool,
    start_kind: int,
    model_symbols: ModelSymbolRegistry,
) -> None:
    if node.left is None or node.right is None:
        raise ValueError("scalar-HEFT contact partial is missing a child")
    if not node.left.is_leaf or not node.right.is_leaf:
        raise ValueError("scalar-HEFT contact partial must join two physical legs")
    orientations = (
        ((node.left, node.right),)
        if node.left.particle.name == node.right.particle.name
        else ((node.left, node.right), (node.right, node.left))
    )
    for actual_left, actual_right in orientations:
        kind = start_kind + len(kernels)
        swapped = actual_left is node.right
        canonical_left_side = "right" if swapped else "left"
        canonical_right_side = "left" if swapped else "right"
        left_dimension = _spin_dimension(node.left.particle.spin)
        right_dimension = _spin_dimension(node.right.particle.spin)
        left_components = tuple(
            model_symbols.kernel_component(kind, canonical_left_side, index)
            for index in range(left_dimension)
        )
        right_components = tuple(
            model_symbols.kernel_component(kind, canonical_right_side, index)
            for index in range(right_dimension)
        )
        base_components = tuple(
            left * right for left in left_components for right in right_components
        )
        encoded_components = list(base_components)
        if include_momenta:
            for side in (canonical_left_side, canonical_right_side):
                for momentum_component in range(4):
                    momentum = model_symbols.kernel_momentum(
                        kind,
                        side,
                        momentum_component,
                    )
                    encoded_components.extend(
                        base * momentum for base in base_components
                    )
        actual_left_token, actual_right_token = (
            (right_token, left_token) if swapped else (left_token, right_token)
        )
        sign = (
            1
            if factor is None
            else _permutation_sign(
                factor,
                (actual_left_token, actual_right_token, output_token),  # type: ignore[arg-type]
            )
        )
        particles = (
            actual_left.particle,
            actual_right.particle,
            node.particle,
        )
        source_particle_legs = (
            _contact_tree_source_leg(actual_left),
            _contact_tree_source_leg(actual_right),
            -1,
        )
        color_source, color_expression, color_power = _heft_trilinear_color(particles)
        kernels.append(
            CompiledOrientedKernel(
                kind=kind,
                term_id=term.id,
                vertex=f"{term.vertex}::contact-heft-tree-partial",
                particles=tuple(particle.name for particle in particles),
                source_particle_legs=source_particle_legs,
                component_expressions=tuple(
                    _canonicalize_oriented_kernel_component(
                        sign * component
                    ).to_canonical_string()
                    for component in encoded_components
                ),
                coupling_expression="1",
                coupling_orders=(),
                runtime_parameters=(),
                color_source=color_source,
                color_expression=color_expression,
                lc_color_normalization_power=color_power,
                term_ids=(),
                contact_orbit_steps=(
                    _heft_contact_orbit_step(
                        term,
                        stage="partial",
                        result_leg=result_leg,
                        left_covered_legs=actual_left.legs,
                        right_covered_legs=actual_right.legs,
                        source_particle_legs=source_particle_legs,
                    ),
                ),
            )
        )


def _append_heft_contact_final_kernels(
    kernels: list[CompiledOrientedKernel],
    *,
    term: CompiledVertexTerm,
    source_particles: Sequence[CompiledParticleRecord],
    left_node: _ContactTreeNode,
    right_node: _ContactTreeNode,
    result_leg: int,
    result_name: str,
    factor: tuple[int, int, int] | None,
    left_token: int,
    right_token: int | None,
    output_token: int | None,
    source_coefficient: str,
    assignment_multiplicity: int,
    start_kind: int,
    model_symbols: ModelSymbolRegistry,
) -> None:
    orientations = (
        ((left_node, right_node),)
        if left_node.particle.name == right_node.particle.name
        else ((left_node, right_node), (right_node, left_node))
    )
    canonical_kind = start_kind + len(kernels)
    components = _heft_contact_final_component_expressions(
        term,
        source_particles,
        left_node=left_node,
        right_node=right_node,
        result_leg=result_leg,
        kind=canonical_kind,
        canonical_left_side="left",
        canonical_right_side="right",
        model_symbols=model_symbols,
    )
    prefactor = (
        _sym.E(source_coefficient)
        * symbols.derived_coupling(model_symbols.model_name, term.id)
        / assignment_multiplicity
    )
    weighted = tuple(
        _canonicalize_oriented_kernel_component(component * prefactor)
        for component in components
    )
    result_particle = next(
        particle for particle in source_particles if particle.name == result_name
    )
    for actual_left, actual_right in orientations:
        kind = start_kind + len(kernels)
        swapped = actual_left is right_node
        actual_left_token, actual_right_token = (
            (right_token, left_token) if swapped else (left_token, right_token)
        )
        sign = (
            1
            if factor is None
            else _permutation_sign(
                factor,
                (actual_left_token, actual_right_token, output_token),  # type: ignore[arg-type]
            )
        )
        oriented_components = tuple(
            _canonicalize_oriented_kernel_component(
                sign
                * _remap_kernel_symbols(
                    component,
                    old_kind=canonical_kind,
                    new_kind=kind,
                    model_symbols=model_symbols,
                    swap_sides=swapped,
                )
            )
            for component in weighted
        )
        particles = (actual_left.particle, actual_right.particle, result_particle)
        source_particle_legs = (
            _contact_tree_source_leg(actual_left),
            _contact_tree_source_leg(actual_right),
            result_leg,
        )
        color_source, color_expression, color_power = _heft_trilinear_color(particles)
        kernels.append(
            CompiledOrientedKernel(
                kind=kind,
                term_id=term.id,
                vertex=f"{term.vertex}::contact-heft-tree-final",
                particles=tuple(particle.name for particle in particles),
                source_particle_legs=source_particle_legs,
                component_expressions=tuple(
                    component.to_canonical_string() for component in oriented_components
                ),
                coupling_expression="1",
                coupling_orders=term.coupling_orders,
                runtime_parameters=(f"derived_coupling_{term.id}",),
                color_source=color_source,
                color_expression=color_expression,
                lc_color_normalization_power=color_power,
                term_ids=(term.id,),
                contact_orbit_steps=(
                    _heft_contact_orbit_step(
                        term,
                        stage="final",
                        result_leg=result_leg,
                        left_covered_legs=actual_left.legs,
                        right_covered_legs=actual_right.legs,
                        source_particle_legs=source_particle_legs,
                    ),
                ),
            )
        )


def _append_contact_tree_partial_kernels(
    kernels: list[CompiledOrientedKernel],
    *,
    term: CompiledVertexTerm,
    node: _ContactTreeNode,
    source_particles: Sequence[CompiledParticleRecord],
    scalar_product_tree: bool,
    start_kind: int,
    model_symbols: ModelSymbolRegistry,
) -> None:
    if node.left is None or node.right is None:
        raise ValueError("contact tree partial node is missing a child")
    orientations = (
        ((node.left, node.right),)
        if node.left.particle.name == node.right.particle.name
        else ((node.left, node.right), (node.right, node.left))
    )
    for actual_left, actual_right in orientations:
        kind = start_kind + len(kernels)
        canonical_left_side = "left" if actual_left is node.left else "right"
        canonical_right_side = "right" if actual_right is node.right else "left"
        left_payload = _contact_tree_node_payload(
            kind,
            canonical_left_side,
            node.left,
            source_particles,
            scalar_product_tree=scalar_product_tree,
            model_symbols=model_symbols,
        )
        right_payload = _contact_tree_node_payload(
            kind,
            canonical_right_side,
            node.right,
            source_particles,
            scalar_product_tree=scalar_product_tree,
            model_symbols=model_symbols,
        )
        components = (
            (left_payload[0] * right_payload[0],)
            if scalar_product_tree
            else (*left_payload, *right_payload)
        )
        kernels.append(
            CompiledOrientedKernel(
                kind=kind,
                term_id=term.id,
                vertex=f"{term.vertex}::contact-tree-partial",
                particles=(
                    actual_left.particle.name,
                    actual_right.particle.name,
                    node.particle.name,
                ),
                source_particle_legs=(
                    _contact_tree_source_leg(actual_left),
                    _contact_tree_source_leg(actual_right),
                    -1,
                ),
                component_expressions=tuple(
                    component.to_canonical_string() for component in components
                ),
                coupling_expression="1",
                coupling_orders=(),
                runtime_parameters=(),
                color_source="1",
                color_expression="1",
                lc_color_normalization_power=0,
                term_ids=(),
            )
        )


def _append_contact_tree_final_kernels(
    kernels: list[CompiledOrientedKernel],
    *,
    term: CompiledVertexTerm,
    source_particles: Sequence[CompiledParticleRecord],
    left_node: _ContactTreeNode,
    right_node: _ContactTreeNode,
    result_leg: int,
    result_name: str,
    scalar_product_tree: bool,
    assignment_multiplicity: int,
    start_kind: int,
    template_cache: dict[
        tuple[object, ...],
        tuple[int, tuple[_sym.Expression, ...]],
    ],
    model_symbols: ModelSymbolRegistry,
) -> None:
    orientations = (
        ((left_node, right_node),)
        if left_node.particle.name == right_node.particle.name
        else ((left_node, right_node), (right_node, left_node))
    )
    canonical_kind = start_kind + len(kernels)
    if scalar_product_tree:
        left_payload = _contact_tree_node_payload(
            canonical_kind,
            "left",
            left_node,
            source_particles,
            scalar_product_tree=True,
            model_symbols=model_symbols,
        )
        right_payload = _contact_tree_node_payload(
            canonical_kind,
            "right",
            right_node,
            source_particles,
            scalar_product_tree=True,
            model_symbols=model_symbols,
        )
        canonical_components = (
            left_payload[0] * right_payload[0] * _sym.E(term.lorentz_expression),
        )
    else:
        template_key = (
            term.lorentz_expression,
            tuple(particle.spin for particle in source_particles),
            result_leg,
            left_node.legs,
            right_node.legs,
        )
        cached = template_cache.get(template_key)
        if cached is None:
            canonical_components = _contact_tree_final_component_expressions(
                term,
                source_particles,
                left_node=left_node,
                right_node=right_node,
                result_leg=result_leg,
                kind=canonical_kind,
                canonical_left_side="left",
                canonical_right_side="right",
                model_symbols=model_symbols,
            )
            template_cache[template_key] = canonical_kind, canonical_components
        else:
            template_kind, template_components = cached
            canonical_components = tuple(
                _remap_kernel_symbols(
                    component,
                    old_kind=template_kind,
                    new_kind=canonical_kind,
                    model_symbols=model_symbols,
                )
                for component in template_components
            )
    prefactor = (
        symbols.derived_coupling(model_symbols.model_name, term.id)
        / assignment_multiplicity
    )
    canonical_weighted_components = tuple(
        _canonicalize_oriented_kernel_component(component * prefactor)
        for component in canonical_components
    )
    for actual_left, actual_right in orientations:
        kind = start_kind + len(kernels)
        components = tuple(
            _remap_kernel_symbols(
                component,
                old_kind=canonical_kind,
                new_kind=kind,
                model_symbols=model_symbols,
                swap_sides=actual_left is right_node,
            )
            for component in canonical_weighted_components
        )
        kernels.append(
            CompiledOrientedKernel(
                kind=kind,
                term_id=term.id,
                vertex=f"{term.vertex}::contact-tree-final",
                particles=(
                    actual_left.particle.name,
                    actual_right.particle.name,
                    result_name,
                ),
                source_particle_legs=(
                    _contact_tree_source_leg(actual_left),
                    _contact_tree_source_leg(actual_right),
                    result_leg,
                ),
                component_expressions=tuple(
                    component.to_canonical_string() for component in components
                ),
                coupling_expression="1",
                coupling_orders=term.coupling_orders,
                runtime_parameters=(f"derived_coupling_{term.id}",),
                color_source=term.color_source,
                color_expression=term.color_expression,
                lc_color_normalization_power=term.lc_color_normalization_power,
                term_ids=(term.id,),
            )
        )


def _contact_tree_node_payload(
    kind: int,
    side: str,
    node: _ContactTreeNode,
    source_particles: Sequence[CompiledParticleRecord],
    *,
    scalar_product_tree: bool,
    model_symbols: ModelSymbolRegistry,
) -> tuple[_sym.Expression, ...]:
    if node.is_leaf:
        if node.physical_leg is None:
            raise ValueError("contact tree leaf has no source leg")
        dimension = _spin_dimension(source_particles[node.physical_leg].spin)
        components = tuple(
            model_symbols.kernel_component(kind, side, index)
            for index in range(dimension)
        )
        if scalar_product_tree:
            return components
        momenta = tuple(
            model_symbols.kernel_momentum(kind, side, index) for index in range(4)
        )
        return (*components, *momenta)
    dimension = compiled_particle_component_dimension(node.particle)
    return tuple(
        model_symbols.kernel_component(kind, side, index) for index in range(dimension)
    )


def _contact_tree_final_component_expressions(
    term: CompiledVertexTerm,
    source_particles: Sequence[CompiledParticleRecord],
    *,
    left_node: _ContactTreeNode,
    right_node: _ContactTreeNode,
    result_leg: int,
    kind: int,
    canonical_left_side: str,
    canonical_right_side: str,
    model_symbols: ModelSymbolRegistry,
) -> tuple[_sym.Expression, ...]:
    payload_by_leg = {
        **_contact_tree_payload_by_leg(
            kind,
            canonical_left_side,
            left_node,
            source_particles,
            model_symbols=model_symbols,
        ),
        **_contact_tree_payload_by_leg(
            kind,
            canonical_right_side,
            right_node,
            source_particles,
            model_symbols=model_symbols,
        ),
    }
    momentum_by_leg = {
        leg: momentum for leg, (_components, momentum) in payload_by_leg.items()
    }
    momentum_by_leg[result_leg] = tuple(
        -sum(
            (momentum[component] for momentum in momentum_by_leg.values()),
            _sym.E("0"),
        )
        for component in range(4)
    )
    library = _sym.TensorLibrary.hep_lib_atom()
    expression = _sym.E(term.lorentz_expression)
    minkowski = _sym.Representation.mink(4)
    for leg, momentum in momentum_by_leg.items():
        library.register(
            _sym.Tensor.dense(
                _sym.TensorName(model_symbols.ufo_momentum_tensor_name(leg + 1))(
                    minkowski
                ),
                momentum,
            )
        )
    input_work = math.prod(
        _spin_dimension(source_particles[leg].spin) for leg in payload_by_leg
    )
    if input_work > _DIRECT_DENSE_CONTACT_INPUT_WORK and sys.platform.startswith(
        "linux"
    ):
        result = _execute_contact_tensor_staged(
            expression,
            library,
            source_particles,
            payload_by_leg=payload_by_leg,
            result_leg=result_leg,
            kind=kind,
            model_symbols=model_symbols,
        )
    else:
        for leg, (components, _momentum) in sorted(payload_by_leg.items()):
            expression *= _contact_tree_physical_tensor_expression(
                library,
                kind=kind,
                leg=leg,
                spin=source_particles[leg].spin,
                components=components,
                model_symbols=model_symbols,
            )
        result = _execute_dense_tensor(
            expression,
            library,
            axis_labels=_spin_axis_labels(
                source_particles[result_leg].spin,
                result_leg + 1,
            ),
        )
    expected = _spin_dimension(source_particles[result_leg].spin)
    if len(result) != expected:
        raise ValueError(
            f"contact tree final {term.vertex}/{term.id} produced {len(result)} "
            f"components, expected {expected}"
        )
    return tuple(
        _replace_evaluator_constants(_as_expression(result[index]))
        for index in range(len(result))
    )


def _execute_contact_tensor_staged(
    expression: _sym.Expression,
    library: _sym.TensorLibrary,
    source_particles: Sequence[CompiledParticleRecord],
    *,
    payload_by_leg: Mapping[
        int,
        tuple[tuple[_sym.Expression, ...], tuple[_sym.Expression, ...]],
    ],
    result_leg: int,
    kind: int,
    model_symbols: ModelSymbolRegistry,
) -> tuple[_sym.Expression, ...]:
    """Contract high-rank inputs in bounded, model-independent stages."""

    dimensions = tuple(_spin_dimension(particle.spin) for particle in source_particles)
    input_legs = tuple(sorted(payload_by_leg, key=lambda leg: (-dimensions[leg], leg)))
    if set(input_legs) != set(range(len(source_particles))) - {result_leg}:
        raise ValueError("staged contact payload does not cover every input leg")
    for leg in input_legs:
        if len(payload_by_leg[leg][0]) != dimensions[leg]:
            raise ValueError(f"staged contact input leg {leg} has wrong dimension")
    remaining_legs = list(range(len(source_particles)))
    pending_legs = list(input_legs)
    step = 0
    while pending_legs:
        stage_legs: list[int] = []
        stage_work = 1
        while pending_legs:
            next_leg = pending_legs[0]
            next_work = stage_work * dimensions[next_leg]
            if stage_legs and next_work > _MAX_STAGED_CONTACT_INPUT_WORK:
                break
            stage_legs.append(pending_legs.pop(0))
            stage_work = next_work
        for leg in stage_legs:
            expression *= _contact_tree_physical_tensor_expression(
                library,
                kind=kind,
                leg=leg,
                spin=source_particles[leg].spin,
                components=payload_by_leg[leg][0],
                model_symbols=model_symbols,
            )
            remaining_legs.remove(leg)
        axis_labels = tuple(
            label
            for remaining_leg in remaining_legs
            for label in _spin_axis_labels(
                source_particles[remaining_leg].spin,
                remaining_leg + 1,
            )
        )
        dense = _execute_dense_tensor(
            expression,
            library,
            axis_labels=axis_labels,
        )
        expected = math.prod(dimensions[index] for index in remaining_legs)
        if len(dense) != expected:
            raise ValueError(
                f"staged contact contraction {step} produced {len(dense)} "
                f"components, expected {expected}"
            )
        if not pending_legs:
            return tuple(_as_expression(component) for component in dense)
        representations = tuple(
            representation
            for remaining_leg in remaining_legs
            for representation in _spin_representations(
                source_particles[remaining_leg].spin
            )
        )
        if not representations:
            if len(dense) != 1:
                raise ValueError("scalar staged contact must have one component")
            expression = _as_expression(dense[0])
            library = _sym.TensorLibrary.hep_lib_atom()
            continue
        slots = tuple(
            slot
            for remaining_leg in remaining_legs
            for slot in _spin_slots(
                source_particles[remaining_leg].spin,
                remaining_leg + 1,
            )
        )
        library = _sym.TensorLibrary.hep_lib_atom()
        name = _sym.TensorName(
            model_symbols.kernel_tensor_name(kind, f"contact_stage_{step}")
        )
        library.register(
            _sym.Tensor.dense(
                name(*representations),
                tuple(_as_expression(component) for component in dense),
            )
        )
        expression = name(*slots).to_expression()
        step += 1
    raise ValueError("staged contact contraction has no input legs")


def eager_color_singlet_vertex_term_components(
    term: CompiledVertexTerm,
    particles: Sequence[CompiledParticleRecord],
    *,
    result_leg: int,
    input_components: Mapping[int, Sequence[_sym.Expression]],
    input_momenta: Mapping[int, Sequence[_sym.Expression]],
    coupling: _sym.Expression | None = None,
    model_symbols: ModelSymbolRegistry | None = None,
) -> tuple[_sym.Expression, ...]:
    """Contract one original color-singlet n-ary term without its lowered tree."""

    _sym._ensure_symbolica()

    if not _contact_term_is_color_singlet(term):
        raise ValueError("the eager n-ary oracle currently requires a color singlet")
    if not 0 <= result_leg < term.valence:
        raise ValueError(f"result leg {result_leg} is outside valence {term.valence}")
    particle_by_name = {particle.name: particle for particle in particles}
    source_particles = tuple(particle_by_name[name] for name in term.particles)
    tensor_symbols = model_symbols or symbols.model("eager-contact-oracle")
    input_legs = set(range(term.valence)) - {result_leg}
    if set(input_components) != input_legs or set(input_momenta) != input_legs:
        raise ValueError(
            "eager n-ary inputs must provide every non-result leg exactly once"
        )

    library = _sym.TensorLibrary.hep_lib_atom()
    expression = _sym.E(term.lorentz_expression)
    momentum_by_leg: dict[int, tuple[_sym.Expression, ...]] = {}
    for leg in sorted(input_legs):
        components = tuple(input_components[leg])
        expected_dimension = _spin_dimension(source_particles[leg].spin)
        if len(components) != expected_dimension:
            raise ValueError(
                f"input leg {leg} has {len(components)} components, "
                f"expected {expected_dimension}"
            )
        momentum = tuple(input_momenta[leg])
        if len(momentum) != 4:
            raise ValueError(f"input leg {leg} momentum must have four components")
        momentum_by_leg[leg] = momentum
        expression *= _contact_tree_physical_tensor_expression(
            library,
            kind=term.id,
            leg=leg,
            spin=source_particles[leg].spin,
            components=components,
            model_symbols=tensor_symbols,
        )

    momentum_by_leg[result_leg] = tuple(
        -sum(
            (momentum[component] for momentum in momentum_by_leg.values()),
            _sym.E("0"),
        )
        for component in range(4)
    )
    minkowski = _sym.Representation.mink(4)
    for leg, momentum in momentum_by_leg.items():
        library.register(
            _sym.Tensor.dense(
                _sym.TensorName(tensor_symbols.ufo_momentum_tensor_name(leg + 1))(
                    minkowski
                ),
                momentum,
            )
        )
    if coupling is not None:
        expression *= coupling
    result = _execute_dense_tensor(
        expression,
        library,
        axis_labels=_spin_axis_labels(
            source_particles[result_leg].spin,
            result_leg + 1,
        ),
    )
    expected_dimension = _spin_dimension(source_particles[result_leg].spin)
    if len(result) != expected_dimension:
        raise ValueError(
            f"eager n-ary term {term.vertex}/{term.id} produced {len(result)} "
            f"components, expected {expected_dimension}"
        )
    return tuple(
        _replace_evaluator_constants(_as_expression(result[index]))
        for index in range(len(result))
    )


def _contact_tree_payload_by_leg(
    kind: int,
    side: str,
    node: _ContactTreeNode,
    source_particles: Sequence[CompiledParticleRecord],
    *,
    model_symbols: ModelSymbolRegistry,
) -> dict[int, tuple[tuple[_sym.Expression, ...], tuple[_sym.Expression, ...]]]:
    payload = _contact_tree_node_payload(
        kind,
        side,
        node,
        source_particles,
        scalar_product_tree=False,
        model_symbols=model_symbols,
    )
    result: dict[
        int, tuple[tuple[_sym.Expression, ...], tuple[_sym.Expression, ...]]
    ] = {}
    cursor = 0
    for leg in node.legs:
        dimension = _spin_dimension(source_particles[leg].spin)
        components = tuple(payload[cursor : cursor + dimension])
        cursor += dimension
        momentum = tuple(payload[cursor : cursor + 4])
        cursor += 4
        result[leg] = components, momentum
    if cursor != len(payload):
        raise ValueError("contact tree payload layout mismatch")
    return result


def _contact_tree_physical_tensor_expression(
    library: _sym.TensorLibrary,
    *,
    kind: int,
    leg: int,
    spin: int,
    components: Sequence[_sym.Expression],
    model_symbols: ModelSymbolRegistry,
) -> _sym.Expression:
    representations = _spin_representations(spin)
    if not representations:
        if len(components) != 1:
            raise ValueError("scalar contact input must have one component")
        return components[0]
    name = _sym.TensorName(model_symbols.contact_leg_tensor_name(kind, leg))
    library.register(_sym.Tensor.dense(name(*representations), components))
    return name(*_spin_slots(spin, leg + 1)).to_expression()


def _contact_tree_assignment_multiplicity(
    input_legs: Sequence[int],
    source_particles: Sequence[CompiledParticleRecord],
    left_root: _ContactTreeNode,
    right_root: _ContactTreeNode,
) -> int:
    species_counts: dict[str, int] = {}
    for leg in input_legs:
        name = source_particles[leg].name
        species_counts[name] = species_counts.get(name, 0) + 1
    permutations = math.prod(math.factorial(count) for count in species_counts.values())
    symmetry_nodes = (
        _contact_tree_same_input_node_count(left_root)
        + _contact_tree_same_input_node_count(right_root)
        + int(left_root.particle.name == right_root.particle.name)
    )
    symmetry_divisor = 2**symmetry_nodes
    if permutations % symmetry_divisor:
        raise ValueError("contact tree assignment symmetry is not integral")
    return permutations // symmetry_divisor


def _contact_tree_same_input_node_count(node: _ContactTreeNode) -> int:
    if node.is_leaf:
        return 0
    if node.left is None or node.right is None:
        raise ValueError("contact tree internal node is missing a child")
    return (
        int(node.left.particle.name == node.right.particle.name)
        + _contact_tree_same_input_node_count(node.left)
        + _contact_tree_same_input_node_count(node.right)
    )


def _contact_tree_source_leg(node: _ContactTreeNode) -> int:
    return -1 if node.physical_leg is None else node.physical_leg


def _contact_term_is_color_singlet(term: CompiledVertexTerm) -> bool:
    return term.color_source in {"1", "UFO::{}::1"} or term.color_expression == "1"


def _deduplicate_contact_partials(
    auxiliary_particles: Sequence[CompiledParticleRecord],
    kernels: Sequence[CompiledOrientedKernel],
    terms: Sequence[CompiledVertexTerm],
    *,
    model_symbols: ModelSymbolRegistry,
) -> tuple[tuple[CompiledParticleRecord, ...], tuple[CompiledOrientedKernel, ...]]:
    term_by_id = {term.id: term for term in terms}
    auxiliary_by_name = {particle.name: particle for particle in auxiliary_particles}
    representative_by_signature: dict[tuple[object, ...], tuple[str, int]] = {}
    replacement: dict[str, str] = {}
    retained: list[CompiledOrientedKernel] = []

    for kernel in kernels:
        if not kernel.vertex.endswith("::contact-partial"):
            retained.append(kernel)
            continue
        auxiliary = auxiliary_by_name[kernel.particles[2]]
        term = term_by_id[kernel.term_id]
        substitutions = {
            symbols.derived_coupling(
                model_symbols.model_name,
                int(name.rsplit("_", 1)[1]),
            ): _sym.E(term.coupling_expression)
            for name in kernel.runtime_parameters
            if name.startswith("derived_coupling_")
        }
        normalized_components = tuple(
            _replace_expression_symbols(
                _remap_kernel_symbols(
                    _sym.E(component),
                    old_kind=kernel.kind,
                    new_kind=0,
                    model_symbols=model_symbols,
                ),
                substitutions,
            ).to_canonical_string()
            for component in kernel.component_expressions
        )
        signature = (
            kernel.particles[:2],
            compiled_particle_component_dimension(auxiliary),
            auxiliary.color,
            kernel.coupling_orders,
            kernel.color_source,
            kernel.lc_color_normalization_power,
            normalized_components,
            auxiliary.auxiliary_kind if kernel.contact_orbit_steps else None,
        )
        representative = representative_by_signature.get(signature)
        if representative is None:
            representative_by_signature[signature] = (
                auxiliary.name,
                len(retained),
            )
            retained.append(kernel)
        else:
            representative_name, representative_index = representative
            replacement[auxiliary.name] = representative_name
            retained[representative_index] = replace(
                retained[representative_index],
                contact_orbit_steps=canonical_contact_orbit_steps(
                    retained[representative_index].contact_orbit_steps,
                    kernel.contact_orbit_steps,
                ),
            )

    if not replacement:
        return tuple(auxiliary_particles), tuple(kernels)

    rewritten = tuple(
        replace(
            kernel,
            particles=tuple(replacement.get(name, name) for name in kernel.particles),
        )
        for kernel in retained
    )
    particles = tuple(
        particle for particle in auxiliary_particles if particle.name not in replacement
    )
    return particles, rewritten

# SPDX-License-Identifier: 0BSD
"""Compile explicit component-ordering contracts for external UFO models."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from ._physics_ir import (
    TENSOR_ORDERING_CONTRACT_VERSION,
    CompiledCurrentOrderingRecord,
    TensorAxisIR,
    TensorIndexBindingIR,
    TensorOrderingIR,
)
from .contracts import (
    CompiledOrientedKernel,
    CompiledParameterRecord,
    CompiledParticleRecord,
    CompiledPropagatorRecord,
    CompiledVertexTerm,
    compiled_particle_component_dimension,
    compiled_particle_is_chiral_eligible,
)

_NORMALIZED_INDEX_PATTERN = re.compile(
    r"\bufo_(?P<family>[lsc])_(?:"
    r"dummy_(?P<dummy>[0-9]+)(?:_(?P<dummy_space>[a-z]+))?"
    r"|(?P<component>[0-9]+)_(?P<leg>[0-9]+)"
    r"|(?P<leg_only>[0-9]+))\b"
)
_NORMALIZED_INDEX_TOKEN_PATTERN = re.compile(r"\bufo_[lsc]_[A-Za-z0-9_]+\b")
_COLOR_DUMMY_SPACES = frozenset(
    {"adjoint", "fundamental", "antifundamental"}
)


@dataclass(frozen=True, slots=True)
class OrderedComponents:
    """Materialized values with their proven canonical storage contract."""

    ordering: TensorOrderingIR
    values: tuple[object, ...]

    def __post_init__(self) -> None:
        if len(self.values) != self.ordering.stored_size:
            raise ValueError(
                "materialized component count does not match its tensor ordering"
            )

    @property
    def ordering_id(self) -> str:
        return self.ordering.ordering_id


def identity_ordering_for_materialized_axes(
    expected_axis_labels: Sequence[str],
    extents: Sequence[int],
) -> TensorOrderingIR:
    labels = tuple(str(label) for label in expected_axis_labels)
    dimensions = tuple(int(extent) for extent in extents)
    if len(labels) != len(dimensions):
        raise ValueError("tensor axis labels and extents must have equal length")
    spaces = tuple(_space_from_axis_label(label) for label in labels)
    return TensorOrderingIR.identity(
        basis=_basis_for_spaces(spaces),
        axes=tuple(
            TensorAxisIR(name=f"axis-{index}", space=space, extent=extent)
            for index, (space, extent) in enumerate(
                zip(spaces, dimensions, strict=True)
            )
        ),
    )


def compile_vertex_index_bindings(
    term: CompiledVertexTerm,
    particle_by_name: Mapping[str, CompiledParticleRecord],
) -> tuple[TensorIndexBindingIR, ...]:
    """Record physical-leg and dummy provenance without making it identity."""

    descriptors: list[tuple[str, str, int | None, int | None, int | None]] = []
    seen: set[tuple[str, str, int | None, int | None, int | None]] = set()
    for origin, expression in (
        ("color", term.color_expression),
        ("lorentz", term.lorentz_expression),
    ):
        matches = tuple(_NORMALIZED_INDEX_PATTERN.finditer(expression))
        matched_spans = {match.span() for match in matches}
        for token in _NORMALIZED_INDEX_TOKEN_PATTERN.finditer(expression):
            if token.span() not in matched_spans:
                raise ValueError(
                    f"vertex term {term.id} contains malformed normalized tensor "
                    f"index {token.group(0)!r}"
                )
        for match in matches:
            family = match.group("family")
            component_text = match.group("component")
            leg_text = match.group("leg") or match.group("leg_only")
            dummy_text = match.group("dummy")
            source_component = (
                None if component_text is None else int(component_text)
            )
            source_leg = None if leg_text is None else int(leg_text)
            source_dummy = None if dummy_text is None else -int(dummy_text)
            _validate_index_source(
                term,
                origin=origin,
                family=family,
                source_component=source_component,
                source_leg=source_leg,
                source_dummy=source_dummy,
                dummy_space=match.group("dummy_space"),
                particle_by_name=particle_by_name,
            )
            space = _binding_space(
                family,
                source_leg=source_leg,
                dummy_space=match.group("dummy_space"),
                particles=term.particles,
                particle_by_name=particle_by_name,
            )
            descriptor = (
                origin,
                space,
                source_component,
                source_leg,
                source_dummy,
            )
            if descriptor not in seen:
                seen.add(descriptor)
                descriptors.append(descriptor)

    dummy_ordinals: dict[tuple[str, str, int], int] = {}
    next_dummy_by_space: dict[tuple[str, str], int] = {}
    bindings: list[TensorIndexBindingIR] = []
    for origin, space, component, leg, dummy in descriptors:
        if dummy is None:
            normalized_name = f"{origin}:{space}:leg-{leg}"
            if component is not None:
                normalized_name += f":component-{component}"
        else:
            key = (origin, space, dummy)
            if key not in dummy_ordinals:
                space_key = (origin, space)
                ordinal = next_dummy_by_space.get(space_key, 0)
                next_dummy_by_space[space_key] = ordinal + 1
                dummy_ordinals[key] = ordinal
            normalized_name = f"{origin}:{space}:dummy-{dummy_ordinals[key]}"
        bindings.append(
            TensorIndexBindingIR(
                origin=origin,
                space=space,
                source_component=component,
                source_leg=leg,
                source_dummy=dummy,
                normalized_name=normalized_name,
            )
        )
    return tuple(bindings)


def compile_tensor_ordering_metadata(
    terms: Sequence[CompiledVertexTerm],
    particles: Sequence[CompiledParticleRecord],
    kernels: Sequence[CompiledOrientedKernel],
    parameters: Sequence[CompiledParameterRecord],
    propagators: Sequence[CompiledPropagatorRecord],
) -> tuple[
    tuple[CompiledVertexTerm, ...],
    tuple[CompiledOrientedKernel, ...],
    tuple[TensorOrderingIR, ...],
    tuple[CompiledCurrentOrderingRecord, ...],
]:
    """Attach and cross-check every external model component-ordering reference."""

    particle_by_name = {particle.name: particle for particle in particles}
    parameters_by_name = {parameter.name: parameter for parameter in parameters}
    propagators_by_name = {propagator.name: propagator for propagator in propagators}
    contact_orderings = _contact_auxiliary_orderings(terms, particle_by_name)
    ordering_registry: dict[str, TensorOrderingIR] = {}

    def register(ordering: TensorOrderingIR) -> TensorOrderingIR:
        previous = ordering_registry.setdefault(ordering.ordering_id, ordering)
        if previous != ordering:
            raise ValueError(
                f"tensor ordering digest collision for {ordering.ordering_id!r}"
            )
        return previous

    full_ordering_by_particle: dict[str, TensorOrderingIR] = {}
    current_orderings: list[CompiledCurrentOrderingRecord] = []
    for particle in particles:
        full = register(
            contact_orderings.get(
                particle.name,
                _full_particle_ordering(particle),
            )
        )
        full_ordering_by_particle[particle.name] = full
        identity = tuple(range(full.stored_size))
        current_orderings.append(
            CompiledCurrentOrderingRecord(
                particle=particle.name,
                chirality=0,
                ordering_id=full.ordering_id,
                kernel_ordering_id=full.ordering_id,
                input_embedding=identity,
                result_projection=identity,
            )
        )
        if compiled_particle_is_chiral_eligible(
            particle,
            parameters=parameters_by_name,
            propagators=propagators_by_name,
        ):
            for chirality in (-1, 1):
                projected = register(
                    TensorOrderingIR.identity(
                        basis=f"weyl-chirality:{chirality:+d}",
                        axes=(
                            TensorAxisIR(
                                name="axis-0",
                                space="weyl-spinor",
                                extent=2,
                            ),
                        ),
                    )
                )
                input_embedding = (
                    (None, None, 0, 1)
                    if chirality == -1
                    else (0, 1, None, None)
                )
                result_projection = (0, 1) if chirality == -1 else (2, 3)
                current_orderings.append(
                    CompiledCurrentOrderingRecord(
                        particle=particle.name,
                        chirality=chirality,
                        ordering_id=projected.ordering_id,
                        kernel_ordering_id=full.ordering_id,
                        input_embedding=input_embedding,
                        result_projection=result_projection,
                    )
                )

    annotated_terms: list[CompiledVertexTerm] = []
    for term in terms:
        source_ordering_ids = tuple(
            full_ordering_by_particle[name].ordering_id for name in term.particles
        )
        index_bindings = compile_vertex_index_bindings(term, particle_by_name)
        if term.source_ordering_ids and term.source_ordering_ids != source_ordering_ids:
            raise ValueError(f"vertex term {term.id} has stale source tensor orderings")
        if term.index_bindings and term.index_bindings != index_bindings:
            raise ValueError(f"vertex term {term.id} has stale tensor index bindings")
        annotated_terms.append(
            replace(
                term,
                source_ordering_ids=source_ordering_ids,
                index_bindings=index_bindings,
            )
        )

    annotated_kernels: list[CompiledOrientedKernel] = []
    for kernel in kernels:
        input_ordering_ids = (
            full_ordering_by_particle[kernel.particles[0]].ordering_id,
            full_ordering_by_particle[kernel.particles[1]].ordering_id,
        )
        output_ordering_id = full_ordering_by_particle[
            kernel.particles[2]
        ].ordering_id
        if (
            kernel.input_ordering_ids
            and kernel.input_ordering_ids != input_ordering_ids
        ):
            raise ValueError(f"oriented kernel {kernel.kind} has stale input orderings")
        if (
            kernel.output_ordering_id
            and kernel.output_ordering_id != output_ordering_id
        ):
            raise ValueError(f"oriented kernel {kernel.kind} has stale output ordering")
        annotated_kernels.append(
            replace(
                kernel,
                input_ordering_ids=input_ordering_ids,
                output_ordering_id=output_ordering_id,
                term_ids=kernel.term_ids or (kernel.term_id,),
            )
        )

    return (
        tuple(annotated_terms),
        tuple(annotated_kernels),
        tuple(sorted(ordering_registry.values(), key=lambda item: item.ordering_id)),
        tuple(current_orderings),
    )


def _contact_auxiliary_orderings(
    terms: Sequence[CompiledVertexTerm],
    particle_by_name: Mapping[str, CompiledParticleRecord],
) -> dict[str, TensorOrderingIR]:
    result: dict[str, TensorOrderingIR] = {}
    for term in terms:
        proof = term.contact_decomposition_proof
        if proof is None or proof.status != "proven":
            continue
        for split in proof.splits:
            expected_axis_order = tuple(
                label
                for leg in split.open_legs
                for label in _spin_axis_labels(
                    particle_by_name[term.particles[leg]].spin,
                    leg + 1,
                )
            )
            if split.component_axis_order != expected_axis_order:
                raise ValueError(
                    f"contact term {term.id} result leg {split.result_leg} has "
                    "non-canonical component axes"
                )
            name = f"__pyamplicol_contact_{term.id}_r{split.result_leg}"
            spaces = tuple(
                _space_from_axis_label(label) for label in split.component_axis_order
            )
            ordering = TensorOrderingIR.create(
                basis=(
                    "contact-scalar"
                    if not spaces
                    else "contact:" + "x".join(spaces)
                ),
                axes=tuple(
                    TensorAxisIR(name=f"axis-{index}", space=space, extent=4)
                    for index, space in enumerate(spaces)
                ),
                component_basis=split.component_basis_order,
                component_expansion=split.component_expansion,
            )
            previous = result.setdefault(name, ordering)
            if previous != ordering:
                raise ValueError(
                    f"contact auxiliary {name!r} has conflicting tensor orderings"
                )
    return result


def _full_particle_ordering(particle: CompiledParticleRecord) -> TensorOrderingIR:
    expected_dimension = {-1: 1, 1: 1, 2: 4, 3: 4, 5: 16}.get(particle.spin)
    component_dimension = compiled_particle_component_dimension(particle)
    if expected_dimension is None or component_dimension != expected_dimension:
        basis = "opaque-auxiliary"
        if particle.auxiliary_kind:
            basis += f":{particle.auxiliary_kind.split(':', 1)[0]}"
        return TensorOrderingIR.identity(
            basis=basis,
            axes=(
                TensorAxisIR(
                    name="axis-0",
                    space="opaque-component",
                    extent=component_dimension,
                ),
            ),
        )
    spaces = {
        -1: (),
        1: (),
        2: ("bispinor",),
        3: ("lorentz-vector",),
        5: ("lorentz-vector", "lorentz-vector"),
    }.get(particle.spin)
    if spaces is None:
        raise ValueError(
            f"particle {particle.name!r} has unsupported tensor spin {particle.spin}"
        )
    return TensorOrderingIR.identity(
        basis=_basis_for_spaces(spaces),
        axes=tuple(
            TensorAxisIR(name=f"axis-{index}", space=space, extent=4)
            for index, space in enumerate(spaces)
        ),
    )


def _basis_for_spaces(spaces: Sequence[str]) -> str:
    values = tuple(spaces)
    if not values:
        return "scalar"
    if values == ("bispinor",):
        return "dirac"
    if values == ("lorentz-vector",):
        return "lorentz-vector"
    if values == ("lorentz-vector", "lorentz-vector"):
        return "lorentz-rank-2"
    return "tensor-product:" + "x".join(values)


def _space_from_axis_label(label: str) -> str:
    if re.fullmatch(r"ufo_s_[0-9]+_[0-9]+", label):
        return "bispinor"
    if re.fullmatch(r"ufo_l_[0-9]+_[0-9]+", label):
        return "lorentz-vector"
    raise ValueError(f"unsupported physical tensor axis label {label!r}")


def _spin_axis_labels(spin: int, leg: int) -> tuple[str, ...]:
    if spin in {-1, 1}:
        return ()
    if spin == 2:
        return (f"ufo_s_1_{leg}",)
    if spin == 3:
        return (f"ufo_l_1_{leg}",)
    if spin == 5:
        return (f"ufo_l_1_{leg}", f"ufo_l_2_{leg}")
    raise ValueError(f"unsupported tensor spin {spin} on source leg {leg}")


def _validate_index_source(
    term: CompiledVertexTerm,
    *,
    origin: str,
    family: str,
    source_component: int | None,
    source_leg: int | None,
    source_dummy: int | None,
    dummy_space: str | None,
    particle_by_name: Mapping[str, CompiledParticleRecord],
) -> None:
    if origin == "color" and family != "c":
        raise ValueError(
            f"vertex term {term.id} color expression contains a non-color index"
        )
    if origin == "lorentz" and family == "c":
        raise ValueError(
            f"vertex term {term.id} Lorentz expression contains a color index"
        )
    if source_dummy is not None:
        if family == "c":
            if dummy_space not in _COLOR_DUMMY_SPACES:
                raise ValueError(
                    f"vertex term {term.id} has unknown color dummy space "
                    f"{dummy_space!r}"
                )
        elif dummy_space is not None:
            raise ValueError(
                f"vertex term {term.id} has a typed non-color dummy index"
            )
        return
    if source_leg is None or source_leg > len(term.particles):
        raise ValueError(f"tensor index refers to absent source leg {source_leg}")
    particle = particle_by_name[term.particles[source_leg - 1]]
    if family == "c":
        if source_component is not None:
            raise ValueError(
                f"vertex term {term.id} color index has an invalid component ordinal"
            )
        if particle.color not in {-3, 3, 8}:
            raise ValueError(
                f"vertex term {term.id} color index refers to singlet particle "
                f"{particle.name!r}"
            )
        return
    expected_family = "s" if particle.spin == 2 else "l"
    label = f"ufo_{family}_{source_component}_{source_leg}"
    if family != expected_family or label not in _spin_axis_labels(
        particle.spin,
        source_leg,
    ):
        raise ValueError(
            f"vertex term {term.id} tensor index {label!r} is incompatible with "
            f"particle {particle.name!r} spin {particle.spin}"
        )


def _binding_space(
    family: str,
    *,
    source_leg: int | None,
    dummy_space: str | None,
    particles: Sequence[str],
    particle_by_name: Mapping[str, CompiledParticleRecord],
) -> str:
    if family == "l":
        return "lorentz-vector"
    if family == "s":
        return "bispinor"
    if source_leg is None:
        return "color-" + (dummy_space or "unknown")
    if source_leg < 1 or source_leg > len(particles):
        raise ValueError(f"tensor index refers to absent source leg {source_leg}")
    particle = particle_by_name[particles[source_leg - 1]]
    return {
        -3: "color-antifundamental",
        3: "color-fundamental",
        8: "color-adjoint",
    }[particle.color]


__all__ = [
    "TENSOR_ORDERING_CONTRACT_VERSION",
    "OrderedComponents",
    "compile_tensor_ordering_metadata",
    "compile_vertex_index_bindings",
    "identity_ordering_for_materialized_axes",
]

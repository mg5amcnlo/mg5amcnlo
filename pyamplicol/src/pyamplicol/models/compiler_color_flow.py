# SPDX-License-Identifier: 0BSD
"""Compiler-owned color-flow auxiliaries derived from proven model tensors."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import replace

from .._internal.physics.symbols import ModelSymbolRegistry
from . import compiler_symbolica as _sym
from .compiler_kernels import (
    _canonicalize_oriented_kernel_component,
    _remap_kernel_symbols,
)
from .contracts import (
    CompiledLCColorTransitionTerm,
    CompiledOrientedKernel,
    CompiledParticleRecord,
    CompiledPropagatorRecord,
    compiled_particle_component_dimension,
)
from .tensors import normalize_color_expression

_U1_SUBTRACTION_AUXILIARY = "u1-subtraction-color-flow-vector"


class _CompiledLCColorTransitionCompilation(tuple):
    """Transition tuple carrying compiler-owned closure companions.

    The oriented-kernel compiler call site predates recurrence closure
    contracts and assigns only the transition tuple. ``CompiledOrientedKernel``
    consumes this private companion immediately and persists the two catalogs
    as separate fields.
    """

    compiler_closure_terms: tuple[CompiledLCColorTransitionTerm, ...]

    def __new__(
        cls,
        transition_terms: tuple[CompiledLCColorTransitionTerm, ...],
        closure_terms: tuple[CompiledLCColorTransitionTerm, ...],
    ) -> _CompiledLCColorTransitionCompilation:
        instance = super().__new__(cls, transition_terms)
        instance.compiler_closure_terms = closure_terms
        return instance


def compile_lc_color_transition_terms(
    kernel: CompiledOrientedKernel,
    oriented_representations: tuple[int, int, int],
    *,
    proof_source: str,
    provenance: tuple[tuple[str, str], ...] = (),
    tensor_role_representations: tuple[int, int, int] | None = None,
) -> tuple[CompiledLCColorTransitionTerm, ...]:
    """Compile exact ordered-word operations from a certified local tensor.

    This function is part of model compilation. Runtime and recurrence-builder
    code consume its closed records and never repeat tensor-family inference.
    """

    structure = kernel.color_projection_structure
    if structure is None:
        raise ValueError(f"kernel {kernel.kind} has no certified color projection")
    tensor_representations = (
        oriented_representations
        if tensor_role_representations is None
        else tuple(int(value) for value in tensor_role_representations)
    )
    if tuple(abs(value) for value in tensor_representations) != tuple(
        abs(value) for value in oriented_representations
    ):
        raise ValueError(
            f"kernel {kernel.kind} tensor-role representations "
            f"{tensor_representations!r} do not match its oriented current shapes "
            f"{oriented_representations!r}"
        )
    shapes = tuple(_lc_color_shape(value) for value in oriented_representations)
    result_shape = shapes[2]

    def term(
        permutation: tuple[int, int],
        operation: str,
        result_component_kind: str | None,
        exact_factor_expression: str = "1",
        *,
        closure: bool = False,
    ) -> CompiledLCColorTransitionTerm:
        result_component_role = (
            "passive"
            if operation == "concatenate-join" and abs(output_representation) == 1
            else "active"
            if operation in {"concatenate-join", "inherit-left", "inherit-right"}
            else "none"
        )
        payload = {
            "color_expression": kernel.color_expression,
            "color_source": kernel.color_source,
            "exact_factor_expression": exact_factor_expression,
            "input_permutation": list(permutation),
            "input_shape_kinds": list(shapes[:2]),
            "kernel_kind": kernel.kind,
            "operation": operation,
            "oriented_representations": list(oriented_representations),
            "tensor_role_representations": list(tensor_representations),
            "proof_source": proof_source,
            "result_component_kind": result_component_kind,
            "result_component_role": result_component_role,
            "result_shape_kind": None if closure else result_shape,
            "source_particle_legs": list(kernel.source_particle_legs),
            "structure": structure,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        term_provenance = tuple(
            sorted(
                (
                    ("compiler-proof-source", proof_source),
                    (
                        "contract-kind",
                        "closure" if closure else "transition",
                    ),
                    ("kernel-kind", str(kernel.kind)),
                    (
                        "source-particle-legs",
                        ",".join(str(value) for value in kernel.source_particle_legs),
                    ),
                    *provenance,
                )
            )
        )
        return CompiledLCColorTransitionTerm(
            input_permutation=permutation,
            reverse_parent_mask=0,
            component_operation=operation,
            result_component_kind=result_component_kind,
            result_component_role=result_component_role,
            input_shape_kinds=(shapes[0], shapes[1]),
            result_shape_kind=None if closure else result_shape,
            exact_factor_expression=exact_factor_expression,
            proof_digest=digest,
            provenance=term_provenance,
        )

    def compiled(
        *transition_terms: CompiledLCColorTransitionTerm,
    ) -> tuple[CompiledLCColorTransitionTerm, ...]:
        left_representation, right_representation = oriented_representations[:2]
        if abs(left_representation) == abs(right_representation) == 1:
            closure_kind = None
        elif {left_representation, right_representation} == {3, -3}:
            closure_kind = "open-string"
        elif left_representation == right_representation == 8:
            closure_kind = "trace"
        else:
            closure_terms: tuple[CompiledLCColorTransitionTerm, ...] = ()
            return _CompiledLCColorTransitionCompilation(
                tuple(transition_terms),
                closure_terms,
            )
        closure_terms = (
            term(
                (0, 1),
                "close",
                closure_kind,
                closure=True,
            ),
        )
        return _CompiledLCColorTransitionCompilation(
            tuple(transition_terms),
            closure_terms,
        )

    left_representation, right_representation, output_representation = (
        oriented_representations
    )
    if structure == "singlet":
        if oriented_representations != (1, 1, 1):
            raise ValueError(
                "a colored literal-singlet transition requires explicit contact "
                f"provenance; kernel {kernel.kind} has {oriented_representations!r}"
            )
        return compiled(term((0, 1), "empty", None))
    if structure == "color-identity":
        colored_inputs = tuple(
            index
            for index, representation in enumerate(oriented_representations[:2])
            if abs(representation) != 1
        )
        if len(colored_inputs) == 0:
            return compiled(term((0, 1), "empty", None))
        if len(colored_inputs) == 1:
            operation = "inherit-left" if colored_inputs[0] == 0 else "inherit-right"
            return compiled(term((0, 1), operation, None))
        if len(colored_inputs) == 2 and abs(output_representation) == 1:
            if {left_representation, right_representation} == {3, -3}:
                permutation = (0, 1) if left_representation == 3 else (1, 0)
                result_component = "open-string"
            elif left_representation == right_representation == 8:
                permutation = (0, 1)
                result_component = "trace"
            else:
                raise ValueError(
                    f"kernel {kernel.kind} has unsupported identity contraction "
                    f"{oriented_representations!r}"
                )
            return compiled(
                term(
                    permutation,
                    "concatenate-join",
                    result_component,
                )
            )
        raise ValueError(
            f"kernel {kernel.kind} has unsupported color-identity orientation "
            f"{oriented_representations!r}"
        )
    if structure == "fundamental-generator":
        try:
            fundamental = tensor_representations.index(3)
            antifundamental = tensor_representations.index(-3)
            adjoint = tensor_representations.index(8)
        except ValueError as exc:
            raise ValueError(
                f"kernel {kernel.kind} fundamental generator has invalid oriented "
                f"tensor roles {tensor_representations!r} for current shapes "
                f"{oriented_representations!r}"
            ) from exc
        tensor_output_representation = tensor_representations[2]
        if tensor_output_representation == -3:
            permutation = (fundamental, adjoint)
            result_component = "open-string"
        elif tensor_output_representation == 3:
            permutation = (adjoint, antifundamental)
            result_component = "open-string"
        elif tensor_output_representation == 8:
            permutation = (fundamental, antifundamental)
            result_component = "adjoint-segment"
        else:
            raise ValueError(
                f"kernel {kernel.kind} fundamental generator has unsupported output "
                f"tensor representation {tensor_output_representation}"
            )
        if set(permutation) != {0, 1}:
            raise ValueError(
                f"kernel {kernel.kind} fundamental generator output is not slot 2"
            )
        return compiled(
            term(
                permutation,
                "concatenate-join",
                result_component,
            )
        )
    if structure == "adjoint-structure-constant":
        if oriented_representations != (8, 8, 8):
            raise ValueError(
                f"kernel {kernel.kind} structure constant has representations "
                f"{oriented_representations!r}"
            )
        return compiled(
            term((0, 1), "concatenate-join", "adjoint-segment"),
            term(
                (1, 0),
                "concatenate-join",
                "adjoint-segment",
                "-1",
            ),
        )
    raise ValueError(
        f"kernel {kernel.kind} has unsupported recurrence color projection "
        f"{structure!r}"
    )


def _lc_color_shape(representation: int) -> str:
    return {
        1: "singlet-forest",
        3: "fundamental-open-string",
        -3: "antifundamental-open-string",
        8: "adjoint-segment",
    }.get(representation) or _unsupported_lc_color_shape(representation)


def _unsupported_lc_color_shape(representation: int) -> str:
    raise ValueError(
        f"recurrence LC color witnesses do not support representation {representation}"
    )


def synthesize_fundamental_fierz_auxiliaries(
    particles: Sequence[CompiledParticleRecord],
    kernels: Sequence[CompiledOrientedKernel],
    propagators: Sequence[CompiledPropagatorRecord],
    *,
    model_symbols: ModelSymbolRegistry,
) -> tuple[tuple[CompiledParticleRecord, ...], tuple[CompiledOrientedKernel, ...]]:
    """Materialize the singlet term of a proven fundamental Fierz identity.

    A certified ``T^a_ij`` kernel implies the SU(3) color-flow decomposition
    into an ordinary line connection and a ``1 / N_c`` singlet subtraction.
    The synthetic current reuses the exact compiled Lorentz/coupling kernel;
    no particle name or PDG label participates in the proof.

    This milestone deliberately targets the supported QCD-like contract:
    massless self-conjugate adjoint vectors with a default propagator and
    fundamental fermion lines. Models outside that contract keep the existing
    exact sector-partition fallback instead of acquiring an unproven auxiliary.
    """

    particle_by_name = {particle.name: particle for particle in particles}
    propagator_by_particle = {
        propagator.particle: propagator for propagator in propagators
    }
    eligible_sources: set[str] = set()
    for kernel in kernels:
        source_name = _eligible_adjoint_source(
            kernel,
            particle_by_name,
            propagator_by_particle,
        )
        if source_name is not None:
            eligible_sources.add(source_name)
    if not eligible_sources:
        return tuple(particles), tuple(kernels)

    used_pdgs = {abs(particle.pdg_code) for particle in particles}
    next_pdg = max(9_100_000, max(used_pdgs, default=0) + 1)

    def allocate_pdg() -> int:
        nonlocal next_pdg
        while next_pdg in used_pdgs:
            next_pdg += 1
        result = next_pdg
        used_pdgs.add(result)
        next_pdg += 1
        return result

    quantum_number_names = sorted(
        {
            name
            for particle in particles
            for name, _expression in particle.quantum_numbers
        }
        | {"electric_charge"}
    )
    auxiliary_by_source: dict[str, CompiledParticleRecord] = {}
    for source_name in sorted(eligible_sources):
        source = particle_by_name[source_name]
        auxiliary_name = f"__pyamplicol_u1_subtraction_{source_name}"
        auxiliary_by_source[source_name] = CompiledParticleRecord(
            name=auxiliary_name,
            antiname=auxiliary_name,
            pdg_code=allocate_pdg(),
            spin=source.spin,
            color=1,
            mass=source.mass,
            width=source.width,
            charge=0.0,
            quantum_numbers=tuple((name, "0") for name in quantum_number_names),
            ghost_number=0,
            propagating=False,
            goldstoneboson=False,
            propagator=None,
            component_dimension=compiled_particle_component_dimension(source),
            auxiliary_kind=_U1_SUBTRACTION_AUXILIARY,
        )

    synthetic: list[CompiledOrientedKernel] = []
    next_kind = max((kernel.kind for kernel in kernels), default=-1) + 1
    for kernel in kernels:
        source_name = _kernel_adjoint_source(
            kernel,
            particle_by_name,
            eligible_sources,
        )
        if source_name is None:
            continue
        source_slot = kernel.particles.index(source_name)
        representations = tuple(
            particle_by_name[name].color for name in kernel.particles
        )
        if source_slot == 2 and representations[:2] != (3, -3):
            # Match the canonical oriented closure used by the LC recursion.
            # The reverse input ordering is an equivalent physical kernel, but
            # materializing both would create duplicate subtraction currents.
            continue

        auxiliary = auxiliary_by_source[source_name]
        synthetic_particles = list(kernel.particles)
        synthetic_particles[source_slot] = auxiliary.name
        synthetic_representations = tuple(
            auxiliary.color if index == source_slot else representations[index]
            for index in range(3)
        )
        colored_legs = tuple(
            index + 1
            for index, representation in enumerate(synthetic_representations)
            if abs(representation) != 1
        )
        if len(colored_legs) != 2:
            raise ValueError(
                f"Fierz auxiliary for kernel {kernel.kind} did not leave two "
                "fundamental color legs"
            )
        color_source = f"UFO::Identity({colored_legs[0]},{colored_legs[1]})"
        color_expression = normalize_color_expression(
            color_source,
            synthetic_representations,
        ).expression

        kind = next_kind + len(synthetic)
        coefficient = complex(*(kernel.color_projection_coefficient or (1.0, 0.0)))
        if source_slot == 2:
            fundamental_dimension = abs(representations[0])
            coefficient /= fundamental_dimension
        source_legs = list(kernel.source_particle_legs)
        source_legs[source_slot] = -1
        synthetic_kernel = CompiledOrientedKernel(
            kind=kind,
            term_id=kernel.term_id,
            vertex=f"{kernel.vertex}::u1-subtraction",
            particles=tuple(synthetic_particles),
            source_particle_legs=tuple(source_legs),
            component_expressions=tuple(
                _canonicalize_oriented_kernel_component(
                    _remap_kernel_symbols(
                        _sym.E(component),
                        old_kind=kernel.kind,
                        new_kind=kind,
                        model_symbols=model_symbols,
                    )
                ).to_canonical_string()
                for component in kernel.component_expressions
            ),
            coupling_expression=kernel.coupling_expression,
            coupling_orders=kernel.coupling_orders,
            runtime_parameters=kernel.runtime_parameters,
            color_source=color_source,
            color_expression=color_expression,
            color_projection_structure="color-identity",
            color_projection_coefficient=(
                float(coefficient.real),
                float(coefficient.imag),
            ),
            lc_color_normalization_power=kernel.lc_color_normalization_power,
            term_ids=kernel.term_ids,
        )
        synthetic.append(
            replace(
                synthetic_kernel,
                lc_color_transition_terms=compile_lc_color_transition_terms(
                    synthetic_kernel,
                    synthetic_representations,  # type: ignore[arg-type]
                    proof_source="fundamental-fierz-identity-v1",
                    provenance=(("source-kernel-kind", str(kernel.kind)),),
                ),
            )
        )

    return (
        (*particles, *auxiliary_by_source.values()),
        (*kernels, *synthetic),
    )


def _kernel_adjoint_source(
    kernel: CompiledOrientedKernel,
    particles: dict[str, CompiledParticleRecord],
    eligible_sources: set[str],
) -> str | None:
    if kernel.color_projection_structure != "fundamental-generator":
        return None
    matches = tuple(name for name in kernel.particles if name in eligible_sources)
    if len(matches) != 1:
        return None
    source_name = matches[0]
    representations = tuple(particles[name].color for name in kernel.particles)
    if representations.count(8) != 1:
        return None
    if sorted(abs(value) for value in representations if value != 8) != [3, 3]:
        return None
    return source_name


def _eligible_adjoint_source(
    kernel: CompiledOrientedKernel,
    particles: dict[str, CompiledParticleRecord],
    propagators: dict[str, CompiledPropagatorRecord],
) -> str | None:
    if kernel.color_projection_structure != "fundamental-generator":
        return None
    representations = tuple(particles[name].color for name in kernel.particles)
    adjoint_slots = tuple(
        index
        for index, representation in enumerate(representations)
        if representation == 8
    )
    if len(adjoint_slots) != 1:
        return None
    if sorted(abs(value) for value in representations if value != 8) != [3, 3]:
        return None
    if not all(
        particles[name].statistics == "fermion"
        for index, name in enumerate(kernel.particles)
        if index != adjoint_slots[0]
    ):
        return None
    source = particles[kernel.particles[adjoint_slots[0]]]
    propagator = propagators.get(source.name)
    if (
        source.spin != 3
        or source.mass.upper() != "ZERO"
        or source.width.upper() != "ZERO"
        or not source.self_conjugate
        or (propagator is not None and propagator.custom)
    ):
        return None
    return source.name


__all__ = [
    "synthesize_fundamental_fierz_auxiliaries",
]

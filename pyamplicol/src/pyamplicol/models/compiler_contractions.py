# SPDX-License-Identifier: 0BSD
"""Compile model-owned current-contraction records for external models."""

from __future__ import annotations

from collections.abc import Sequence

from ._physics_ir import ContractionIR
from .contracts import (
    CompiledClosureContractionRecord,
    CompiledDirectContractionRecord,
    CompiledParameterRecord,
    CompiledParticleRecord,
    CompiledPropagatorRecord,
    compiled_current_dimension,
    compiled_particle_component_dimension,
    compiled_particle_is_chiral_eligible,
)


def compile_contraction_records(
    particles: Sequence[CompiledParticleRecord],
    parameters: Sequence[CompiledParameterRecord],
    propagators: Sequence[CompiledPropagatorRecord],
) -> tuple[
    tuple[CompiledDirectContractionRecord, ...],
    tuple[CompiledClosureContractionRecord, ...],
]:
    """Derive concrete contraction states from explicit compiled metadata.

    This is deliberately a finite compiler step. Runtime generation consumes
    the resulting records and does not infer a contraction from component
    dimensions.
    """

    particles_by_name = {particle.name: particle for particle in particles}
    parameters_by_name = {parameter.name: parameter for parameter in parameters}
    propagators_by_name = {propagator.name: propagator for propagator in propagators}
    direct: list[CompiledDirectContractionRecord] = []
    closure: list[CompiledClosureContractionRecord] = []

    for left in particles:
        try:
            right = particles_by_name[left.antiname]
        except KeyError as exc:
            raise ValueError(
                f"particle {left.name!r} refers to absent antiparticle "
                f"{left.antiname!r}"
            ) from exc

        left_is_weyl = compiled_particle_is_chiral_eligible(
            left,
            parameters=parameters_by_name,
            propagators=propagators_by_name,
        )
        right_is_weyl = compiled_particle_is_chiral_eligible(
            right,
            parameters=parameters_by_name,
            propagators=propagators_by_name,
        )
        if left_is_weyl != right_is_weyl:
            states: tuple[tuple[int, int], ...] = ()
        elif left_is_weyl:
            states = ((-1, 1), (0, 0), (1, -1))
        else:
            states = ((0, 0),)
        for left_chirality, right_chirality in states:
            contraction = _compile_direct_contraction(
                left,
                right,
                left_chirality=left_chirality,
                right_chirality=right_chirality,
                parameters=parameters_by_name,
                propagators=propagators_by_name,
            )
            if contraction is not None:
                direct.append(
                    CompiledDirectContractionRecord(
                        left_particle=left.name,
                        left_chirality=left_chirality,
                        right_particle=right.name,
                        right_chirality=right_chirality,
                        contraction_ir=contraction,
                    )
                )

        scalar_projection = _compile_scalar_closure(left)
        if scalar_projection is not None:
            closure.append(
                CompiledClosureContractionRecord(
                    particle=left.name,
                    chirality=0,
                    contraction_ir=scalar_projection,
                )
            )

    return tuple(direct), tuple(closure)


def _compile_direct_contraction(
    left: CompiledParticleRecord,
    right: CompiledParticleRecord,
    *,
    left_chirality: int,
    right_chirality: int,
    parameters: dict[str, CompiledParameterRecord],
    propagators: dict[str, CompiledPropagatorRecord],
) -> ContractionIR | None:
    try:
        left_dimension = compiled_current_dimension(
            left,
            left_chirality,
            parameters=parameters,
            propagators=propagators,
        )
        right_dimension = compiled_current_dimension(
            right,
            right_chirality,
            parameters=parameters,
            propagators=propagators,
        )
    except ValueError:
        return None
    if left_dimension != right_dimension:
        return None

    if left_chirality != 0 or right_chirality != 0:
        if (
            left.statistics != "fermion"
            or right.statistics != "fermion"
            or left.wavefunction_family != "fermion"
            or right.wavefunction_family != "fermion"
            or left_dimension != 2
            or left_chirality != -right_chirality
        ):
            return None
        return ContractionIR(
            name="weyl",
            left_basis="weyl-chiral",
            right_basis="weyl-chiral",
            coefficients=((1.0, 0.0),) * 2,
            chirality_relation="opposite",
            metric_signature=None,
        )

    if (
        left.statistics == right.statistics == "boson"
        and left.wavefunction_family == right.wavefunction_family == "scalar"
        and left_dimension == 1
    ):
        return ContractionIR(
            name="scalar",
            left_basis="scalar",
            right_basis="scalar",
            coefficients=((1.0, 0.0),),
        )
    if (
        left.statistics == right.statistics == "fermion"
        and left.wavefunction_family == right.wavefunction_family == "fermion"
        and left_dimension == 4
    ):
        return ContractionIR(
            name="dirac",
            left_basis="dirac",
            right_basis="dirac",
            coefficients=((1.0, 0.0),) * 4,
        )
    if (
        left.statistics == right.statistics == "boson"
        and left.wavefunction_family == right.wavefunction_family == "vector"
        and left_dimension == 4
    ):
        return ContractionIR(
            name="lorentz",
            left_basis="lorentz-vector",
            right_basis="lorentz-vector",
            coefficients=(
                (1.0, 0.0),
                (-1.0, 0.0),
                (-1.0, 0.0),
                (-1.0, 0.0),
            ),
            metric_signature="mostly-minus",
        )
    if (
        left.auxiliary_kind
        == right.auxiliary_kind
        == "u1-subtraction-color-flow-vector"
        and left_dimension == 4
    ):
        basis = "auxiliary:u1-subtraction-color-flow-vector"
        return ContractionIR(
            name="lorentz",
            left_basis=basis,
            right_basis=basis,
            coefficients=(
                (1.0, 0.0),
                (-1.0, 0.0),
                (-1.0, 0.0),
                (-1.0, 0.0),
            ),
            metric_signature="mostly-minus",
        )
    if (
        left.statistics == right.statistics == "boson"
        and left.wavefunction_family == right.wavefunction_family == "spin2"
        and left_dimension == 16
    ):
        metric = (1.0, -1.0, -1.0, -1.0)
        return ContractionIR(
            name="lorentz-rank-2",
            left_basis="lorentz-rank-2",
            right_basis="lorentz-rank-2",
            coefficients=tuple(
                (metric[mu] * metric[nu], 0.0)
                for mu in range(4)
                for nu in range(4)
            ),
            metric_signature="mostly-minus-tensor-product",
        )
    return None


def _compile_scalar_closure(
    particle: CompiledParticleRecord,
) -> ContractionIR | None:
    try:
        dimension = compiled_particle_component_dimension(particle)
    except ValueError:
        return None
    if (
        particle.statistics != "boson"
        or particle.wavefunction_family != "scalar"
        or dimension != 1
    ):
        return None
    return ContractionIR(
        name="scalar",
        left_basis="scalar",
        right_basis="scalar",
        coefficients=((1.0, 0.0),),
        chirality_relation="any",
        metric_signature=None,
    )


__all__ = ["compile_contraction_records"]

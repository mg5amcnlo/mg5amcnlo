# SPDX-License-Identifier: 0BSD
"""Compile explicit gauge/Goldstone partner contracts from normalized UFO IR."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .contracts import (
    CompiledGoldstonePartnerRecord,
    CompiledParameterRecord,
    CompiledParticleRecord,
    CompiledPropagatorRecord,
    _goldstone_vector_quantum_contract_matches,
    _resolved_mass_expression,
)


def compile_goldstone_partner_records(
    particles: Sequence[CompiledParticleRecord],
    parameters: Sequence[CompiledParameterRecord],
    propagators: Sequence[CompiledPropagatorRecord],
) -> tuple[CompiledGoldstonePartnerRecord, ...]:
    """Resolve every declared Goldstone once, before executable model loading."""

    parameter_by_name: Mapping[str, CompiledParameterRecord] = {
        parameter.name: parameter for parameter in parameters
    }
    propagator_by_particle = {
        propagator.particle: propagator for propagator in propagators
    }
    vectors = tuple(
        particle
        for particle in particles
        if particle.spin == 3
        and particle.wavefunction_family == "vector"
        and particle.propagating
        and not particle.goldstoneboson
    )
    records: list[CompiledGoldstonePartnerRecord] = []
    for goldstone in particles:
        if not goldstone.goldstoneboson:
            continue
        if goldstone.spin != 1 or not goldstone.propagating:
            raise ValueError(
                f"Goldstone {goldstone.name!r} must be a propagating scalar"
            )
        candidates = tuple(
            vector
            for vector in vectors
            if _goldstone_vector_quantum_contract_matches(
                goldstone,
                vector,
                parameter_by_name,
            )
            and _resolved_mass_expression(vector, parameter_by_name) != "0"
        )
        if len(candidates) > 1:
            raise ValueError(
                f"Goldstone {goldstone.name!r} ambiguously matches vectors "
                f"{[vector.name for vector in candidates]!r}"
            )
        mass_expression = _resolved_mass_expression(
            goldstone,
            parameter_by_name,
        )
        if not candidates:
            records.append(
                CompiledGoldstonePartnerRecord(
                    goldstone=goldstone.name,
                    vector=None,
                    policy="explicit",
                    mass_expression=mass_expression,
                )
            )
            continue
        vector = candidates[0]
        propagator = propagator_by_particle.get(vector.name)
        custom = propagator is not None and propagator.custom
        records.append(
            CompiledGoldstonePartnerRecord(
                goldstone=goldstone.name,
                vector=vector.name,
                policy="model-supplied" if custom else "absorbed",
                mass_expression=mass_expression,
            )
        )
    return tuple(records)


__all__ = ["compile_goldstone_partner_records"]

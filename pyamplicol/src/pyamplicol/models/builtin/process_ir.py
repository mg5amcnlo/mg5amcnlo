# SPDX-License-Identifier: 0BSD
"""Legacy built-in-SM process IR construction."""

from __future__ import annotations

from collections import Counter

from ...processes.core_syntax import (
    canonical_process_key,
)
from ...processes.ir import (
    CanonicalProcessIR,
    ColorEndpointSummary,
    ColorRole,
    LegSide,
    ParticleStatistics,
    ProcessLegIR,
    ProcessSetEntryIR,
    ProcessSetIR,
    SourceOrientation,
    WavefunctionFamily,
)
from .process_catalog import (
    ANTI_PARTICLE,
    ANTIQUARKS,
    GLUONS,
    PDGS,
    QUARKS,
    SINGLETS,
)
from .process_enumeration import ProcessEnumerator
from .process_selection import enumerate_generic_process_set, enumerate_process_set
from .process_types import (
    BuiltinProcessOptions as ProcessOptions,
)
from .process_types import (
    BuiltinProcessSetEntry as ProcessSetEntry,
)

_VECTOR_NAMES = frozenset({"a", "z", "w+", "w-"})


def build_process_ir(
    process: str,
    *,
    color_accuracy: str = "lc",
    options: ProcessOptions | None = None,
) -> CanonicalProcessIR:
    parsed = ProcessEnumerator(options).parse(process)
    physical_initial = tuple(
        particle if particle in {"p", "j"} else ANTI_PARTICLE[particle]
        for particle in parsed.initial_state
    )
    canonical = f"{' '.join(physical_initial)} > {' '.join(parsed.rest)}"
    legs = tuple(
        [
            *(
                _leg_ir(
                    label=index + 1,
                    side="initial",
                    particle=particle,
                    outgoing_particle=parsed.initial_state[index],
                )
                for index, particle in enumerate(physical_initial)
            ),
            *(
                _leg_ir(
                    label=index + 3,
                    side="final",
                    particle=particle,
                    outgoing_particle=particle,
                )
                for index, particle in enumerate(parsed.rest)
            ),
        ]
    )
    counts = Counter(leg.color_role for leg in legs)
    return CanonicalProcessIR(
        process=canonical,
        key=canonical_process_key(canonical),
        color_accuracy=color_accuracy,
        legs=legs,
        color_endpoints=ColorEndpointSummary(
            fundamental_count=counts["fundamental"],
            antifundamental_count=counts["antifundamental"],
            pair_count=min(
                counts["fundamental"],
                counts["antifundamental"],
            ),
        ),
    )


def build_process_set_ir(
    process_string: str,
    *,
    color_accuracy: str = "lc",
    options: ProcessOptions | None = None,
    generic: bool = False,
    max_quark_pairs: int | None = None,
) -> ProcessSetIR:
    enumeration = (
        enumerate_generic_process_set(
            process_string,
            options=options,
            max_quark_pairs=max_quark_pairs,
        )
        if generic
        else enumerate_process_set(process_string, options=options)
    )
    entries = tuple(
        _process_set_entry_ir(entry, color_accuracy=color_accuracy)
        for entry in enumeration.entries
    )
    return ProcessSetIR(
        request=enumeration.request,
        color_accuracy=color_accuracy,
        entries=entries,
    )


def _process_set_entry_ir(
    entry: ProcessSetEntry,
    *,
    color_accuracy: str,
) -> ProcessSetEntryIR:
    ir = build_process_ir(
        entry.process,
        color_accuracy=color_accuracy,
        options=entry.enumeration.options,
    )
    return ProcessSetEntryIR(
        key=entry.key,
        process=entry.process,
        ir=ir,
        n_groups=len(entry.enumeration.groups),
        n_records=entry.enumeration.n_records,
        n_unique_processes=len(entry.enumeration.unique_processes),
    )


def _leg_ir(
    *,
    label: int,
    side: LegSide,
    particle: str,
    outgoing_particle: str,
) -> ProcessLegIR:
    statistics, wavefunction_family, color_role, source_orientation = _structural_roles(
        outgoing_particle
    )
    return ProcessLegIR(
        label=label,
        side=side,
        particle=particle,
        outgoing_particle=outgoing_particle,
        pdg=_pdg_or_none(particle),
        outgoing_pdg=_pdg_or_none(outgoing_particle),
        statistics=statistics,
        wavefunction_family=wavefunction_family,
        color_role=color_role,
        source_orientation=source_orientation,
    )


def _pdg_or_none(particle: str) -> int | None:
    if particle in {"p", "j"}:
        return None
    return int(PDGS[particle])


def _structural_roles(
    outgoing_particle: str,
) -> tuple[
    ParticleStatistics,
    WavefunctionFamily,
    ColorRole,
    SourceOrientation,
]:
    if outgoing_particle in {"p", "j"}:
        return "inclusive", "inclusive", "inclusive", "inclusive"

    pdg = abs(int(PDGS[outgoing_particle]))
    if (
        outgoing_particle in QUARKS
        or outgoing_particle in ANTIQUARKS
        or pdg in {11, 12, 13, 14, 15, 16}
    ):
        statistics: ParticleStatistics = "fermion"
        wavefunction_family: WavefunctionFamily = "fermion"
    elif outgoing_particle in GLUONS or outgoing_particle in _VECTOR_NAMES:
        statistics = "boson"
        wavefunction_family = "vector"
    elif outgoing_particle == "h" or outgoing_particle in SINGLETS:
        statistics = "boson"
        wavefunction_family = "scalar"
    else:
        raise ValueError(f"unknown built-in particle role for {outgoing_particle!r}")

    color_role: ColorRole
    if outgoing_particle in QUARKS:
        color_role = "fundamental"
    elif outgoing_particle in ANTIQUARKS:
        color_role = "antifundamental"
    elif outgoing_particle in GLUONS:
        color_role = "adjoint"
    else:
        color_role = "singlet"

    if ANTI_PARTICLE[outgoing_particle] == outgoing_particle:
        source_orientation: SourceOrientation = "self-conjugate"
    elif int(PDGS[outgoing_particle]) > 0:
        source_orientation = "particle"
    else:
        source_orientation = "antiparticle"

    return statistics, wavefunction_family, color_role, source_orientation


__all__ = ["build_process_ir", "build_process_set_ir"]

# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import itertools
import re
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from ..models.contracts import (
    CompiledModelIR,
    CompiledParameterRecord,
    CompiledParticleRecord,
    validate_color_representation,
)
from .conservation import ConservedCharges
from .core_syntax import (
    canonical_process_key,
    split_process_set,
)
from .ir import (
    CanonicalProcessIR,
    ColorEndpointSummary,
    ColorRole,
    LegSide,
    ParticleStatistics,
    ProcessLegIR,
    SourceOrientation,
    WavefunctionFamily,
)


@dataclass(frozen=True)
class ModelParticleCatalog:
    model_name: str
    particles: tuple[CompiledParticleRecord, ...]
    parameters: tuple[CompiledParameterRecord, ...] = ()

    def __post_init__(self) -> None:
        names = [particle.name for particle in self.particles]
        if len(names) != len(set(names)):
            raise ValueError("compiled model contains duplicate particle names")
        for particle in self.particles:
            validate_color_representation(
                particle.color,
                context=f"particle {particle.name!r}",
            )

    @property
    def by_name(self) -> dict[str, CompiledParticleRecord]:
        return {particle.name: particle for particle in self.particles}

    @property
    def external_particles(self) -> tuple[CompiledParticleRecord, ...]:
        return tuple(
            particle
            for particle in self.particles
            if particle.spin > 0
            and particle.ghost_number == 0
            and not particle.goldstoneboson
            and particle.propagating
            and particle.auxiliary_kind is None
        )

    def resolve(self, name: str, *, external: bool = True) -> CompiledParticleRecord:
        candidates = self.external_particles if external else self.particles
        exact = next(
            (particle for particle in candidates if particle.name == name), None
        )
        if exact is not None:
            return exact
        folded = [
            particle
            for particle in candidates
            if particle.name.casefold() == name.casefold()
        ]
        if len(folded) == 1:
            return folded[0]
        if len(folded) > 1:
            names = ", ".join(sorted(particle.name for particle in folded))
            raise ValueError(f"particle name {name!r} is case-ambiguous: {names}")
        raise ValueError(
            f"particle {name!r} is not an external state in {self.model_name}"
        )

    def antiparticle(self, particle: CompiledParticleRecord) -> CompiledParticleRecord:
        try:
            return self.by_name[particle.antiname]
        except KeyError as exc:
            raise ValueError(
                f"particle {particle.name!r} refers to absent antiparticle "
                f"{particle.antiname!r}"
            ) from exc

    def validate_process_role(self, particle: CompiledParticleRecord) -> None:
        """Reject colored roles unsupported by the current color planner."""

        representation = validate_color_representation(
            particle.color,
            context=f"particle {particle.name!r}",
        )
        supported_colored_role = (
            particle.color_role in {"fundamental", "antifundamental"}
            and particle.statistics == "fermion"
        ) or (
            particle.color_role == "adjoint"
            and particle.wavefunction_family == "vector"
        )
        if representation != 1 and not supported_colored_role:
            raise ValueError(
                f"particle {particle.name!r} has unsupported colored external-state "
                f"role (spin={particle.spin}, color={representation}); current color "
                "planning supports fundamental fermions and adjoint vectors"
            )

    def default_multiparticles(self) -> dict[str, tuple[str, ...]]:
        partons = tuple(
            particle.name
            for particle in self.external_particles
            if self._mass_is_compile_time_zero(particle.mass)
            and (
                (
                    particle.statistics == "fermion"
                    and particle.color_role in {"fundamental", "antifundamental"}
                )
                or (
                    particle.wavefunction_family == "vector"
                    and particle.color_role == "adjoint"
                )
            )
        )
        result = {} if not partons else {"p": partons, "j": partons}
        result["all"] = tuple(particle.name for particle in self.external_particles)
        return result

    def _mass_is_compile_time_zero(self, name: str) -> bool:
        if name.casefold() == "zero":
            return True
        parameter = next(
            (parameter for parameter in self.parameters if parameter.name == name),
            None,
        )
        if parameter is None or parameter.nature.casefold() != "internal":
            return False
        from ..models import compiler_symbolica as _sym

        _sym._ensure_symbolica()
        return (
            _sym.E(parameter.resolved_expression).expand().to_canonical_string()
            == _sym.E("0").to_canonical_string()
        )


def parse_multiparticle_definitions(
    definitions: Sequence[str],
    catalog: ModelParticleCatalog,
) -> dict[str, tuple[str, ...]]:
    result = dict(catalog.default_multiparticles())
    for definition in definitions:
        name, separator, raw_items = definition.partition("=")
        name = name.strip()
        if (
            not separator
            or not name
            or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name)
        ):
            raise ValueError(
                f"invalid multiparticle definition {definition!r}; "
                "expected NAME=ITEM,ITEM"
            )
        items = tuple(item.strip() for item in raw_items.split(",") if item.strip())
        if not items:
            raise ValueError(f"multiparticle {name!r} has no members")
        resolved = tuple(catalog.resolve(item).name for item in items)
        result[name] = tuple(dict.fromkeys(resolved))
    return result


def expand_model_processes(
    process_set: str,
    catalog: ModelParticleCatalog,
    *,
    multiparticles: Mapping[str, Sequence[str]] | None = None,
    charges: ConservedCharges | None = None,
) -> tuple[str, ...]:
    """Concrete processes of ``process_set``, one per multiset of each side.

    Side permutations describe the same matrix element, so each side is
    enumerated as a multiset. With ``charges``, processes violating a
    conserved additive charge of the model are dropped.
    """

    aliases = {
        name: tuple(values) for name, values in catalog.default_multiparticles().items()
    }
    if multiparticles is not None:
        aliases.update({name: tuple(values) for name, values in multiparticles.items()})
    expanded: dict[str, None] = {}
    for process in split_process_set(process_set):
        initial_text, separator, final_text = process.partition(">")
        if not separator or ">" in final_text:
            raise ValueError("invalid collision format; expected 'initial > final'")
        initial_slots = _expanded_slots(initial_text, aliases, catalog)
        final_slots = _expanded_slots(final_text, aliases, catalog)
        if len(initial_slots) != 2:
            raise ValueError(
                "pyAmpliCol processes require exactly two initial particles"
            )
        if not final_slots:
            raise ValueError("pyAmpliCol processes require at least one final particle")
        finals: dict[tuple[int, ...], list[tuple[str, ...]]] = {}
        for final in _side_multisets(final_slots):
            key = () if charges is None else charges.total(final)
            finals.setdefault(key, []).append(final)
        for initial in _side_multisets(initial_slots):
            key = () if charges is None else charges.total(initial)
            for final in finals.get(key, ()):
                concrete = f"{' '.join(initial)} > {' '.join(final)}"
                expanded.setdefault(concrete, None)
    return tuple(expanded)


def _side_multisets(slots: Sequence[tuple[str, ...]]) -> Iterator[tuple[str, ...]]:
    """Each particle multiset that fills ``slots`` once.

    Slots with the same options are filled by non-decreasing option index:
    the order in which a cartesian product meets a multiset first, unless
    slots with different but overlapping options interleave.
    """

    groups: dict[tuple[str, ...], list[int]] = {}
    for position, options in enumerate(slots):
        groups.setdefault(options, []).append(position)
    seen: set[tuple[str, ...]] = set()
    for choice in itertools.product(
        *(
            itertools.combinations_with_replacement(options, len(positions))
            for options, positions in groups.items()
        )
    ):
        assignment = [""] * len(slots)
        for positions, names in zip(groups.values(), choice, strict=True):
            for position, name in zip(positions, names, strict=True):
                assignment[position] = name
        key = tuple(sorted(assignment))
        if key not in seen:
            seen.add(key)
            yield tuple(assignment)


def build_model_process_ir(
    process: str,
    model: CompiledModelIR,
    *,
    color_accuracy: str = "lc",
) -> CanonicalProcessIR:
    catalog = ModelParticleCatalog(model.name, model.particles, model.parameters)
    initial_text, separator, final_text = process.partition(">")
    if not separator or ">" in final_text:
        raise ValueError("invalid collision format; expected 'initial > final'")
    initial = tuple(catalog.resolve(token) for token in initial_text.split())
    final = tuple(catalog.resolve(token) for token in final_text.split())
    if len(initial) != 2 or not final:
        raise ValueError(
            "process must have exactly two initial and at least one final particle"
        )
    outgoing_initial = tuple(catalog.antiparticle(particle) for particle in initial)
    canonical = f"{' '.join(particle.name for particle in initial)} > " + " ".join(
        particle.name for particle in final
    )
    legs = tuple(
        [
            *(
                _model_leg(
                    label=index + 1,
                    side="initial",
                    particle=particle,
                    outgoing_particle=outgoing_initial[index],
                    catalog=catalog,
                )
                for index, particle in enumerate(initial)
            ),
            *(
                _model_leg(
                    label=index + 3,
                    side="final",
                    particle=particle,
                    outgoing_particle=particle,
                    catalog=catalog,
                )
                for index, particle in enumerate(final)
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


def _expanded_slots(
    text: str,
    aliases: Mapping[str, Sequence[str]],
    catalog: ModelParticleCatalog,
) -> list[tuple[str, ...]]:
    slots: list[tuple[str, ...]] = []
    for token in text.split():
        count, name = _repetition(token)
        if name in aliases:
            options = tuple(catalog.resolve(item).name for item in aliases[name])
        else:
            options = (catalog.resolve(name).name,)
        slots.extend(options for _ in range(count))
    return slots


def _repetition(token: str) -> tuple[int, str]:
    match = re.fullmatch(r"(?:(\d+)\*)?(.+)", token)
    if match is None:
        raise ValueError(f"invalid process token {token!r}")
    count = int(match.group(1) or 1)
    if count < 1:
        raise ValueError(f"process repetition must be positive: {token!r}")
    return count, match.group(2)


def _model_leg(
    *,
    label: int,
    side: str,
    particle: CompiledParticleRecord,
    outgoing_particle: CompiledParticleRecord,
    catalog: ModelParticleCatalog,
) -> ProcessLegIR:
    catalog.validate_process_role(outgoing_particle)
    return ProcessLegIR(
        label=label,
        side=cast(LegSide, side),
        particle=particle.name,
        outgoing_particle=outgoing_particle.name,
        pdg=particle.pdg_code,
        outgoing_pdg=outgoing_particle.pdg_code,
        statistics=cast(ParticleStatistics, outgoing_particle.statistics),
        wavefunction_family=cast(
            WavefunctionFamily,
            outgoing_particle.wavefunction_family,
        ),
        color_role=cast(ColorRole, outgoing_particle.color_role),
        source_orientation=cast(
            SourceOrientation,
            outgoing_particle.source_orientation,
        ),
    )


__all__ = [
    "ModelParticleCatalog",
    "build_model_process_ir",
    "expand_model_processes",
    "parse_multiparticle_definitions",
]

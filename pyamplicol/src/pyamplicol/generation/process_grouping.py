# SPDX-License-Identifier: 0BSD
"""Flavour-profile canonicalization and process-group maps.

Two processes are candidates for one matrix element when their external legs
have the same particle types. A particle type is the spin, the SU(3) dimension,
the resolved mass and width and self-conjugacy; particles and antiparticles
share a type. The leg mapping between two candidates is derived from flavour
profiles: each flavour of a process is described by how many times it appears
as a particle and as an antiparticle in the initial and in the final state.
Sorting legs by profile, and only then by PDG, gives positions that follow any
flavour relabelling, which sorting by PDG alone does not.

A group map records, for each representative, the members whose matrix
element equals ``factor`` times the representative's at permuted momenta:
``ME_member(p) = factor * ME_representative(q)`` with
``q[r] = p[permutation[r]]``, so ``permutation`` maps representative legs to
member legs. ``conjugated`` members are related by charge conjugation, so LC
colour-flow words map with reversed order.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from fractions import Fraction
from typing import Any

PROCESS_GROUPS_PATH = "process-groups.json"
PROCESS_GROUPS_ROLE = "process-groups"
PROCESS_GROUPS_KIND = "pyamplicol-process-groups"
PROCESS_GROUPS_SCHEMA_VERSION = 1
MAX_RATIONAL = 99
"""Largest numerator and denominator of a recorded group factor."""

_ZERO_EXPRESSIONS = frozenset({"0", "0.0", "-0", "-0.0", "zero"})


class GroupingBasis(StrEnum):
    IDENTITY = "identity"
    FLAVOR_BLIND = "flavor-blind"
    NUMERICAL = "numerical"


# The least verified link decides the basis of a composed relation.
_BASIS_STRENGTH = {
    GroupingBasis.IDENTITY: 0,
    GroupingBasis.NUMERICAL: 1,
    GroupingBasis.FLAVOR_BLIND: 2,
}


class GroupingAnalysis(StrEnum):
    NONE = "none"
    NUMERICAL = "numerical"


@dataclass(frozen=True, order=True, slots=True)
class ParticleType:
    """Everything a flavour relabelling must preserve about one particle."""

    spin: int
    color: int
    mass: str
    width: str
    self_conjugate: bool


@dataclass(frozen=True, slots=True)
class GroupingParticle:
    pdg: int
    name: str
    type: ParticleType


ParticleTable = Mapping[int, GroupingParticle]


@dataclass(frozen=True, slots=True)
class GroupingProcess:
    process_id: str
    expression: str
    pdgs: tuple[int, ...]
    initial_count: int
    generated: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.initial_count < len(self.pdgs):
            raise ValueError(
                f"process {self.process_id!r} needs initial and final legs"
            )


def resolved_parameter_name(
    name: str,
    parameters: Mapping[str, object],
) -> str:
    """Return ``ZERO`` for compile-time zero parameters, otherwise ``name``."""

    if not name or name.strip().casefold() == "zero":
        return "ZERO"
    record = parameters.get(name)
    resolved = getattr(record, "resolved_expression", None)
    if isinstance(resolved, str) and resolved.strip().casefold() in _ZERO_EXPRESSIONS:
        return "ZERO"
    return name


def particle_table(
    particles: Iterable[object],
    parameters: Mapping[str, object],
) -> dict[int, GroupingParticle]:
    """Build the grouping view of compiled particle records.

    ``particles`` are ``CompiledParticleRecord``-like objects and
    ``parameters`` maps parameter names to records with a
    ``resolved_expression``, as in ``CompiledModelIR``.
    """

    table: dict[int, GroupingParticle] = {}
    for particle in particles:
        pdg = int(particle.pdg_code)  # type: ignore[attr-defined]
        name = str(particle.name)  # type: ignore[attr-defined]
        antiname = str(particle.antiname)  # type: ignore[attr-defined]
        declared = getattr(particle, "self_conjugate", None)
        self_conjugate = bool(declared) if declared is not None else name == antiname
        table[pdg] = GroupingParticle(
            pdg=pdg,
            name=name,
            type=ParticleType(
                spin=int(particle.spin),  # type: ignore[attr-defined]
                color=abs(int(particle.color)),  # type: ignore[attr-defined]
                mass=resolved_parameter_name(
                    str(particle.mass),  # type: ignore[attr-defined]
                    parameters,
                ),
                width=resolved_parameter_name(
                    str(particle.width),  # type: ignore[attr-defined]
                    parameters,
                ),
                self_conjugate=self_conjugate,
            ),
        )
    return table


@dataclass(frozen=True, slots=True)
class _Leg:
    index: int
    initial: bool
    pdg: int
    flavour: int
    orientation: int
    type: ParticleType


def _particle(table: ParticleTable, pdg: int) -> GroupingParticle:
    try:
        return table[pdg]
    except KeyError as error:
        raise ValueError(f"PDG code {pdg} is absent from the model") from error


def _legs(
    process: GroupingProcess,
    table: ParticleTable,
    *,
    conjugate: bool = False,
) -> tuple[_Leg, ...]:
    legs = []
    for index, pdg in enumerate(process.pdgs):
        particle = _particle(table, pdg)
        if particle.type.self_conjugate:
            flavour, orientation = abs(pdg), 0
        else:
            flavour, orientation = abs(pdg), (1 if pdg > 0 else -1)
            if conjugate:
                orientation = -orientation
        legs.append(
            _Leg(
                index=index,
                initial=index < process.initial_count,
                pdg=pdg if not conjugate or orientation == 0 else -pdg,
                flavour=flavour,
                orientation=orientation,
                type=particle.type,
            )
        )
    return tuple(legs)


Profile = tuple[int, int, int, int]


def _profiles(legs: Sequence[_Leg]) -> dict[int, Profile]:
    counts: dict[int, list[int]] = {}
    for leg in legs:
        slot = (0 if leg.initial else 2) + (1 if leg.orientation < 0 else 0)
        counts.setdefault(leg.flavour, [0, 0, 0, 0])[slot] += 1
    return {flavour: tuple(values) for flavour, values in counts.items()}  # type: ignore[misc]


def coarse_key(process: GroupingProcess, table: ParticleTable) -> tuple[Any, ...]:
    """Particle types of each side, with particles and antiparticles merged."""

    legs = _legs(process, table)
    return (
        process.initial_count,
        tuple(sorted(leg.type for leg in legs if leg.initial)),
        tuple(sorted(leg.type for leg in legs if not leg.initial)),
    )


def profile_key(
    process: GroupingProcess,
    table: ParticleTable,
    *,
    conjugate: bool = False,
) -> tuple[Any, ...]:
    """Equal for exactly the processes related by a type-preserving relabelling."""

    legs = _legs(process, table, conjugate=conjugate)
    types = {leg.flavour: leg.type for leg in legs}
    profiles = _profiles(legs)
    return (
        process.initial_count,
        len(legs),
        tuple(
            sorted((types[flavour], profile) for flavour, profile in profiles.items())
        ),
    )


def flavor_class_key(
    process: GroupingProcess,
    table: ParticleTable,
    *,
    consider_charge_conjugation: bool,
) -> tuple[Any, ...]:
    """Class of processes assumed equal by ``--flavor-blind``."""

    key = profile_key(process, table)
    if consider_charge_conjugation:
        key = min(key, profile_key(process, table, conjugate=True))
    return key


def _ordered_legs(
    legs: Sequence[_Leg],
    ranks: Mapping[int, int],
    profiles: Mapping[int, Profile],
) -> tuple[_Leg, ...]:
    return tuple(
        sorted(
            legs,
            key=lambda leg: (
                not leg.initial,
                leg.type,
                profiles[leg.flavour],
                ranks[leg.flavour],
                -leg.orientation,
                leg.index,
            ),
        )
    )


def _flavour_groups(legs: Sequence[_Leg]) -> dict[tuple[Any, ...], list[int]]:
    """Flavours with equal (type, profile), in a fixed order."""

    types = {leg.flavour: leg.type for leg in legs}
    profiles = _profiles(legs)
    groups: dict[tuple[Any, ...], list[int]] = {}
    for flavour in sorted(profiles):
        groups.setdefault((types[flavour], profiles[flavour]), []).append(flavour)
    return groups


def _profile_order(
    legs: Sequence[_Leg],
    assignment: Mapping[tuple[Any, ...], Sequence[int]],
) -> tuple[_Leg, ...]:
    ranks = {
        flavour: rank
        for flavours in assignment.values()
        for rank, flavour in enumerate(flavours)
    }
    return _ordered_legs(legs, ranks, _profiles(legs))


@dataclass(frozen=True, slots=True)
class LegMapping:
    """One candidate relation ``member -> representative``."""

    permutation: tuple[int, ...]
    conjugated: bool
    method: str


def _mapping(
    member: Sequence[_Leg],
    representative: Sequence[_Leg],
    *,
    conjugated: bool,
    method: str,
) -> LegMapping:
    permutation = [0] * len(representative)
    for member_leg, representative_leg in zip(member, representative, strict=True):
        if (
            member_leg.initial != representative_leg.initial
            or member_leg.type != representative_leg.type
            or member_leg.orientation != representative_leg.orientation
        ):
            raise ValueError("leg orders do not describe the same process shape")
        permutation[representative_leg.index] = member_leg.index
    return LegMapping(tuple(permutation), conjugated, method)


def profile_mappings(
    member: GroupingProcess,
    representative: GroupingProcess,
    table: ParticleTable,
    *,
    conjugated: bool = False,
    limit: int = 24,
) -> tuple[LegMapping, ...]:
    """Leg mappings induced by flavour relabellings, at most ``limit``.

    Flavours with the same type and profile can be paired in any order; each
    pairing is one mapping, the first one pairs them by increasing ``|PDG|``.
    """

    member_legs = _legs(member, table, conjugate=conjugated)
    representative_legs = _legs(representative, table)
    if profile_key(member, table, conjugate=conjugated) != profile_key(
        representative, table
    ):
        return ()
    representative_order = _profile_order(
        representative_legs, _flavour_groups(representative_legs)
    )
    member_groups = _flavour_groups(member_legs)
    keys = list(member_groups)
    choices = [itertools.permutations(member_groups[key]) for key in keys]
    mappings: list[LegMapping] = []
    for combination in itertools.product(*choices):
        member_order = _profile_order(
            member_legs, dict(zip(keys, combination, strict=True))
        )
        mapping = _mapping(
            member_order,
            representative_order,
            conjugated=conjugated,
            method="flavor-profile",
        )
        if mapping not in mappings:
            mappings.append(mapping)
        if len(mappings) >= limit:
            break
    return tuple(mappings)


def signed_pdg_mapping(
    member: GroupingProcess,
    representative: GroupingProcess,
    table: ParticleTable,
) -> LegMapping | None:
    """Pair legs by sorting each side by type and signed PDG."""

    if coarse_key(member, table) != coarse_key(representative, table):
        return None

    def order(process: GroupingProcess) -> tuple[_Leg, ...]:
        return tuple(
            sorted(
                _legs(process, table),
                key=lambda leg: (not leg.initial, leg.type, leg.pdg, leg.index),
            )
        )

    permutation = [0] * len(representative.pdgs)
    for member_leg, representative_leg in zip(
        order(member), order(representative), strict=True
    ):
        permutation[representative_leg.index] = member_leg.index
    return LegMapping(tuple(permutation), False, "signed-pdg")


def candidate_mappings(
    member: GroupingProcess,
    representative: GroupingProcess,
    table: ParticleTable,
    *,
    consider_charge_conjugation: bool,
    limit: int = 24,
) -> tuple[LegMapping, ...]:
    """All leg mappings worth testing numerically, most natural first."""

    mappings = list(profile_mappings(member, representative, table, limit=limit))
    if consider_charge_conjugation:
        mappings.extend(
            profile_mappings(
                member, representative, table, conjugated=True, limit=limit
            )
        )
    fallback = signed_pdg_mapping(member, representative, table)
    if fallback is not None and all(
        (fallback.permutation, False) != (mapping.permutation, mapping.conjugated)
        for mapping in mappings
    ):
        mappings.append(fallback)
    return tuple(mappings)


def flavor_blind_mapping(
    member: GroupingProcess,
    representative: GroupingProcess,
    table: ParticleTable,
    *,
    consider_charge_conjugation: bool,
) -> LegMapping:
    """The mapping assumed by ``--flavor-blind`` within one flavour class."""

    mappings = profile_mappings(member, representative, table, limit=1)
    if not mappings and consider_charge_conjugation:
        mappings = profile_mappings(
            member, representative, table, conjugated=True, limit=1
        )
    if not mappings:
        raise ValueError(
            f"{member.expression!r} and {representative.expression!r} are not "
            "related by a flavour relabelling"
        )
    return replace(mappings[0], method="flavor-blind")


def representative_key(
    process: GroupingProcess,
    table: ParticleTable,
) -> tuple[Any, ...]:
    """Canonical order of processes: type signature, then PDGs.

    Smaller ``|PDG|`` comes first and particles precede antiparticles, so
    ``d d~ > g g`` represents ``u u~ > g g`` and ``u g > u g`` represents
    ``u~ g > u~ g``.
    """

    legs = sorted(
        _legs(process, table),
        key=lambda leg: (
            not leg.initial,
            leg.type,
            abs(leg.pdg),
            leg.pdg < 0,
            leg.index,
        ),
    )
    return (
        coarse_key(process, table),
        tuple((leg.initial, abs(leg.pdg), leg.pdg < 0) for leg in legs),
        process.process_id,
    )


def flavor_blind_classes(
    processes: Sequence[GroupingProcess],
    table: ParticleTable,
    *,
    consider_charge_conjugation: bool,
) -> list[list[GroupingProcess]]:
    """Partition ``processes`` into flavour classes, representative first."""

    classes: dict[tuple[Any, ...], list[GroupingProcess]] = {}
    for process in processes:
        key = flavor_class_key(
            process,
            table,
            consider_charge_conjugation=consider_charge_conjugation,
        )
        classes.setdefault(key, []).append(process)
    result = []
    for members in classes.values():
        members.sort(key=lambda process: representative_key(process, table))
        result.append(members)
    return result


# --------------------------------------------------------------------------
# Relations and group maps


def identity_permutation(count: int) -> tuple[int, ...]:
    return tuple(range(count))


def compose_permutations(
    member_to_middle: Sequence[int],
    middle_to_representative: Sequence[int],
) -> tuple[int, ...]:
    """Compose ``member -> middle`` with ``middle -> representative``."""

    return tuple(member_to_middle[index] for index in middle_to_representative)


def invert_permutation(permutation: Sequence[int]) -> tuple[int, ...]:
    inverse = [0] * len(permutation)
    for representative_index, member_index in enumerate(permutation):
        inverse[member_index] = representative_index
    return tuple(inverse)


@dataclass(frozen=True, slots=True)
class GroupMember:
    process_id: str
    expression: str
    pdgs: tuple[int, ...]
    initial_count: int
    factor: Fraction = Fraction(1)
    permutation: tuple[int, ...] = ()
    conjugated: bool = False
    basis: GroupingBasis = GroupingBasis.IDENTITY
    generated: bool = True
    warning: str | None = None

    @classmethod
    def identity(cls, process: GroupingProcess) -> GroupMember:
        return cls(
            process_id=process.process_id,
            expression=process.expression,
            pdgs=process.pdgs,
            initial_count=process.initial_count,
            permutation=identity_permutation(len(process.pdgs)),
            generated=process.generated,
        )

    @property
    def process(self) -> GroupingProcess:
        return GroupingProcess(
            process_id=self.process_id,
            expression=self.expression,
            pdgs=self.pdgs,
            initial_count=self.initial_count,
            generated=self.generated,
        )

    def related(
        self,
        mapping: LegMapping,
        *,
        factor: Fraction,
        basis: GroupingBasis,
        warning: str | None = None,
    ) -> GroupMember:
        return replace(
            self,
            factor=Fraction(factor),
            permutation=tuple(mapping.permutation),
            conjugated=mapping.conjugated,
            basis=basis,
            warning=warning,
        )

    def through(self, middle: GroupMember) -> GroupMember:
        """Re-express ``self -> A`` through ``A -> B`` as ``self -> B``.

        ``middle`` is the relation of the old representative ``A`` to the new
        representative ``B``.
        """

        return replace(
            self,
            factor=self.factor * middle.factor,
            permutation=compose_permutations(self.permutation, middle.permutation),
            conjugated=self.conjugated != middle.conjugated,
            basis=max(self.basis, middle.basis, key=_BASIS_STRENGTH.__getitem__),
            warning=self.warning or middle.warning,
        )

    def inverted_to(self, representative: GroupMember) -> GroupMember:
        """Relation of ``representative`` to ``self``, the inverse of ``self``."""

        return replace(
            representative,
            factor=1 / self.factor,
            permutation=invert_permutation(self.permutation),
            conjugated=self.conjugated,
            basis=self.basis,
            warning=self.warning,
        )

    def as_dict(self) -> dict[str, object]:
        record: dict[str, object] = {
            "process_id": self.process_id,
            "process": self.expression,
            "pdgs": list(self.pdgs),
            "initial_count": self.initial_count,
            "factor": {
                "numerator": self.factor.numerator,
                "denominator": self.factor.denominator,
            },
            "permutation": list(self.permutation),
            "conjugated": self.conjugated,
            "basis": str(self.basis),
            "generated": self.generated,
        }
        if self.warning:
            record["warning"] = self.warning
        return record

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> GroupMember:
        factor = value["factor"]
        if not isinstance(factor, Mapping):
            raise ValueError("process-group factor must be an object")
        return cls(
            process_id=str(value["process_id"]),
            expression=str(value["process"]),
            pdgs=tuple(int(pdg) for pdg in value["pdgs"]),  # type: ignore[attr-defined]
            initial_count=int(value["initial_count"]),  # type: ignore[call-overload]
            factor=Fraction(
                int(factor["numerator"]),
                int(factor["denominator"]),
            ),
            permutation=tuple(int(i) for i in value["permutation"]),  # type: ignore[attr-defined]
            conjugated=bool(value["conjugated"]),
            basis=GroupingBasis(str(value["basis"])),
            generated=bool(value["generated"]),
            warning=(str(value["warning"]) if value.get("warning") else None),
        )


@dataclass(slots=True)
class ProcessGroup:
    """A representative and every member, the representative first."""

    members: list[GroupMember]

    @property
    def representative(self) -> GroupMember:
        return self.members[0]

    def as_dict(self) -> dict[str, object]:
        representative = self.representative
        return {
            "representative": {
                "process_id": representative.process_id,
                "process": representative.expression,
            },
            "members": [member.as_dict() for member in self.members],
        }


@dataclass(slots=True)
class ProcessGroupMap:
    """Process groups keyed by representative process ID."""

    groups: dict[str, ProcessGroup] = field(default_factory=dict)
    analyses: list[dict[str, object]] = field(default_factory=list)

    @classmethod
    def trivial(cls, processes: Iterable[GroupingProcess]) -> ProcessGroupMap:
        result = cls()
        for process in processes:
            result.add_group([GroupMember.identity(process)])
        return result

    def add_group(self, members: Sequence[GroupMember]) -> None:
        representative = members[0]
        if not representative.generated:
            raise ValueError(
                f"representative {representative.process_id!r} was not generated"
            )
        for member in members:
            if self.group_of(member.process_id) is not None:
                raise ValueError(
                    f"process {member.process_id!r} already belongs to a group"
                )
        self.groups[representative.process_id] = ProcessGroup(list(members))

    def group_of(self, process_id: str) -> ProcessGroup | None:
        for group in self.groups.values():
            if any(member.process_id == process_id for member in group.members):
                return group
        return None

    @property
    def representatives(self) -> tuple[GroupMember, ...]:
        return tuple(group.representative for group in self.groups.values())

    def members(self) -> Iterator[GroupMember]:
        for group in self.groups.values():
            yield from group.members

    def merge(
        self,
        member_representative: str,
        relation: GroupMember,
        table: ParticleTable,
    ) -> str:
        """Join two groups given ``relation``, the relation of one
        representative (``relation.process_id``) to another
        (``member_representative`` is the target). Returns the representative
        of the merged group, the canonically smallest generated one."""

        if relation.process_id == member_representative:
            raise ValueError("cannot merge a group with itself")
        source = self.groups.pop(relation.process_id)
        target = self.groups.pop(member_representative)
        source_key = representative_key(source.representative.process, table)
        target_key = representative_key(target.representative.process, table)
        if target_key <= source_key:
            keep, moved, link = target, source, relation
        else:
            keep, moved = source, target
            link = relation.inverted_to(target.representative)
        merged = list(keep.members)
        merged.extend(member.through(link) for member in moved.members)
        self.groups[keep.representative.process_id] = ProcessGroup(merged)
        return keep.representative.process_id

    def as_document(self) -> dict[str, object]:
        groups = sorted(
            self.groups.values(), key=lambda group: group.representative.process_id
        )
        return {
            "kind": PROCESS_GROUPS_KIND,
            "schema_version": PROCESS_GROUPS_SCHEMA_VERSION,
            "analyses": list(self.analyses),
            "groups": [group.as_dict() for group in groups],
        }

    def to_json(self) -> str:
        return json.dumps(self.as_document(), indent=2, sort_keys=False) + "\n"

    @classmethod
    def from_document(cls, document: Mapping[str, object]) -> ProcessGroupMap:
        if document.get("kind") != PROCESS_GROUPS_KIND:
            raise ValueError("document is not a pyAmpliCol process-group map")
        if document.get("schema_version") != PROCESS_GROUPS_SCHEMA_VERSION:
            raise ValueError("unsupported process-group map schema version")
        result = cls(analyses=[dict(entry) for entry in document.get("analyses", [])])  # type: ignore[attr-defined]
        for group in document.get("groups", []):  # type: ignore[attr-defined]
            members = [GroupMember.from_dict(member) for member in group["members"]]
            result.add_group(members)
        return result

    @classmethod
    def from_json(cls, text: str) -> ProcessGroupMap:
        return cls.from_document(json.loads(text))


def rational_factor(value: float, *, tolerance: float) -> Fraction | None:
    """Return ``value`` as a fraction with |numerator|, denominator <= 99."""

    if value == 0 or value != value:
        return None
    fraction = Fraction(value).limit_denominator(MAX_RATIONAL)
    if (
        fraction == 0
        or abs(fraction.numerator) > MAX_RATIONAL
        or abs(float(fraction) - value) > tolerance * abs(value)
    ):
        return None
    return fraction


__all__ = [
    "MAX_RATIONAL",
    "PROCESS_GROUPS_KIND",
    "PROCESS_GROUPS_PATH",
    "PROCESS_GROUPS_ROLE",
    "PROCESS_GROUPS_SCHEMA_VERSION",
    "GroupMember",
    "GroupingAnalysis",
    "GroupingBasis",
    "GroupingParticle",
    "GroupingProcess",
    "LegMapping",
    "ParticleTable",
    "ParticleType",
    "ProcessGroup",
    "ProcessGroupMap",
    "candidate_mappings",
    "coarse_key",
    "compose_permutations",
    "flavor_blind_classes",
    "flavor_blind_mapping",
    "flavor_class_key",
    "identity_permutation",
    "invert_permutation",
    "particle_table",
    "profile_key",
    "profile_mappings",
    "rational_factor",
    "representative_key",
    "resolved_parameter_name",
    "signed_pdg_mapping",
]

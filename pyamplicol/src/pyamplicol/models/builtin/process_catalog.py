# SPDX-License-Identifier: 0BSD
"""Legacy built-in Standard Model process catalog."""

from __future__ import annotations

from dataclasses import dataclass

from ...processes.core_syntax import (
    _anonymous_options,
    _split_repeat_token,
    _tokenize_side,
)
from .process_types import BuiltinParsedProcess as ParsedProcess
from .process_types import (
    BuiltinProcessEnumeration as ProcessEnumeration,
)
from .process_types import (
    BuiltinSubprocessRecord as SubprocessRecord,
)

QUARKS = frozenset({"d", "u", "s", "c", "b", "t"})
ANTIQUARKS = frozenset({f"{q}~" for q in QUARKS})
SINGLETS = frozenset(
    {
        "a",
        "z",
        "w+",
        "w-",
        "e+",
        "e-",
        "mu+",
        "mu-",
        "ta+",
        "ta-",
        "ve",
        "ve~",
        "vm",
        "vm~",
        "vt",
        "vt~",
        "h",
    }
)
GLUONS = frozenset({"g"})
ALL_COLOURED = QUARKS | ANTIQUARKS | GLUONS
PDGS = {
    "g": "21",
    "d": "1",
    "u": "2",
    "s": "3",
    "c": "4",
    "b": "5",
    "t": "6",
    "d~": "-1",
    "u~": "-2",
    "s~": "-3",
    "c~": "-4",
    "b~": "-5",
    "t~": "-6",
    "a": "22",
    "z": "23",
    "w+": "24",
    "w-": "-24",
    "e+": "-11",
    "e-": "11",
    "mu+": "-13",
    "mu-": "13",
    "ta+": "-15",
    "ta-": "15",
    "ve": "12",
    "ve~": "-12",
    "vm": "14",
    "vm~": "-14",
    "vt": "16",
    "vt~": "-16",
    "h": "25",
}
ANTI_PARTICLE = {
    "g": "g",
    "d": "d~",
    "u": "u~",
    "s": "s~",
    "c": "c~",
    "b": "b~",
    "t": "t~",
    "d~": "d",
    "u~": "u",
    "s~": "s",
    "c~": "c",
    "b~": "b",
    "t~": "t",
    "a": "a",
    "z": "z",
    "w+": "w-",
    "w-": "w+",
    "e+": "e-",
    "e-": "e+",
    "mu+": "mu-",
    "mu-": "mu+",
    "ta+": "ta-",
    "ta-": "ta+",
    "ve": "ve~",
    "ve~": "ve",
    "vm": "vm~",
    "vm~": "vm",
    "vt": "vt~",
    "vt~": "vt",
    "h": "h",
}
SORT_PARTICLES = {
    "g": 13,
    "d": 1,
    "u": 2,
    "s": 3,
    "c": 4,
    "b": 5,
    "t": 6,
    "d~": 7,
    "u~": 8,
    "s~": 9,
    "c~": 10,
    "b~": 11,
    "t~": 12,
    "a": 80,
    "z": 81,
    "w+": 82,
    "w-": 83,
    "e+": 84,
    "e-": 85,
    "mu+": 86,
    "mu-": 87,
    "ta+": 88,
    "ta-": 89,
    "ve": 90,
    "ve~": 91,
    "vm": 92,
    "vm~": 93,
    "vt": 94,
    "vt~": 95,
    "h": 96,
}
CHARGES3 = {
    "g": 0,
    "d": -1,
    "u": 2,
    "s": -1,
    "c": 2,
    "b": -1,
    "t": 2,
    "d~": 1,
    "u~": -2,
    "s~": 1,
    "c~": -2,
    "b~": 1,
    "t~": -2,
    "a": 0,
    "z": 0,
    "w+": 3,
    "w-": -3,
    "e+": 3,
    "e-": -3,
    "mu+": 3,
    "mu-": -3,
    "ta+": 3,
    "ta-": -3,
    "ve": 0,
    "ve~": 0,
    "vm": 0,
    "vm~": 0,
    "vt": 0,
    "vt~": 0,
    "h": 0,
}
FAMILY = {
    "g": 0,
    "d": 1,
    "u": 1,
    "s": 11,
    "c": 11,
    "b": 21,
    "t": 21,
    "d~": -1,
    "u~": -1,
    "s~": -11,
    "c~": -11,
    "b~": -21,
    "t~": -21,
    "a": 0,
    "z": 0,
    "w+": 0,
    "w-": 0,
    "e+": -31,
    "e-": 31,
    "mu+": -41,
    "mu-": 41,
    "ta+": -51,
    "ta-": 51,
    "ve": 31,
    "ve~": -31,
    "vm": 41,
    "vm~": -41,
    "vt": 51,
    "vt~": -51,
    "h": 0,
}


@dataclass(frozen=True)
class ParticleSelectionMetadata:
    name: str
    pdg: int
    anti_name: str
    charge3: int
    family: int
    color_rep: int
    is_fermion: bool
    is_antiparticle: bool

    @property
    def is_colored(self) -> bool:
        return self.color_rep != 1


def selection_metadata(name: str) -> ParticleSelectionMetadata:
    return _PARTICLE_SELECTION_METADATA[name]


def _metadata_color_rep(name: str) -> int:
    if name in QUARKS:
        return 3
    if name in ANTIQUARKS:
        return -3
    if name in GLUONS:
        return 8
    return 1


_PARTICLE_SELECTION_METADATA = {
    name: ParticleSelectionMetadata(
        name=name,
        pdg=int(PDGS[name]),
        anti_name=ANTI_PARTICLE[name],
        charge3=CHARGES3[name],
        family=FAMILY[name],
        color_rep=_metadata_color_rep(name),
        is_fermion=name in QUARKS
        or name in ANTIQUARKS
        or abs(int(PDGS[name])) in range(11, 17),
        is_antiparticle=name in ANTIQUARKS or int(PDGS[name]) < 0,
    )
    for name in PDGS
}


def process_uses_inclusive_labels(process: str) -> bool:
    parts = process.lower().replace("bar", "~").split(">")
    if len(parts) != 2:
        raise ValueError("invalid collision format; expected 'initial > final'")
    for token in (*_tokenize_side(parts[0].strip()), *_tokenize_side(parts[1].strip())):
        _, item = _split_repeat_token(token)
        if item in {"p", "j"}:
            return True
        if (
            item.startswith("[")
            and item.endswith("]")
            and any(option in {"p", "j"} for option in _anonymous_options(item))
        ):
            return True
    return False


def request_allows_charged_current(request: ParsedProcess) -> bool:
    """Return whether built-in request quantum numbers imply charged-current flow."""

    if "w+" in request.rest or "w-" in request.rest:
        return True
    non_qcd_rest = [
        particle
        for particle in request.rest
        if particle not in ALL_COLOURED and particle in CHARGES3
    ]
    return abs(sum(CHARGES3[particle] for particle in non_qcd_rest)) == 3


def concrete_processes_from_inclusive_enumeration(
    enumeration: ProcessEnumeration,
) -> tuple[str, ...]:
    processes: dict[str, None] = {}
    for group in enumeration.groups:
        for record in group.records:
            processes.setdefault(_record_to_physical_process(record), None)
    return tuple(sorted(processes, key=_concrete_process_sort_key))


def _record_to_physical_process(record: SubprocessRecord) -> str:
    initial = tuple(ANTI_PARTICLE[p] for p in record.process[:2])
    final = record.process[2:]
    return f"{' '.join(initial)} > {' '.join(final)}"


def _concrete_process_sort_key(process: str) -> tuple[object, ...]:
    initial, _, final = process.partition(">")
    initial_tokens = tuple(_tokenize_side(initial.strip()))
    final_tokens = tuple(_tokenize_side(final.strip()))
    q_qbar_first = not (
        len(initial_tokens) == 2
        and not initial_tokens[0].endswith("~")
        and initial_tokens[1].endswith("~")
    )
    return (
        q_qbar_first,
        tuple(SORT_PARTICLES.get(token, 999) for token in initial_tokens),
        tuple(SORT_PARTICLES.get(token, 999) for token in final_tokens),
        initial_tokens,
        final_tokens,
    )


__all__ = [
    "ALL_COLOURED",
    "ANTIQUARKS",
    "ANTI_PARTICLE",
    "CHARGES3",
    "FAMILY",
    "GLUONS",
    "PDGS",
    "QUARKS",
    "SINGLETS",
    "SORT_PARTICLES",
    "ParticleSelectionMetadata",
    "concrete_processes_from_inclusive_enumeration",
    "process_uses_inclusive_labels",
    "request_allows_charged_current",
    "selection_metadata",
]

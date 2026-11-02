# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .core_syntax import ProcessTuple

LegSide = Literal["initial", "final"]
# These values mirror canonical compiled-particle metadata. ``inclusive`` is
# reserved for the built-in compatibility aliases that do not identify a
# concrete model particle.
ParticleStatistics = Literal["boson", "fermion", "ghost", "auxiliary", "inclusive"]
WavefunctionFamily = Literal[
    "scalar",
    "fermion",
    "vector",
    "spin2",
    "ghost",
    "auxiliary",
    "inclusive",
]
ColorRole = Literal[
    "fundamental",
    "antifundamental",
    "adjoint",
    "singlet",
    "inclusive",
]
SourceOrientation = Literal[
    "particle",
    "antiparticle",
    "self-conjugate",
    "inclusive",
]

@dataclass(frozen=True)
class ProcessLegIR:
    """One external leg with both physical and all-outgoing conventions."""

    label: int
    side: LegSide
    particle: str
    outgoing_particle: str
    pdg: int | None
    outgoing_pdg: int | None
    statistics: ParticleStatistics
    wavefunction_family: WavefunctionFamily
    color_role: ColorRole
    source_orientation: SourceOrientation

    @property
    def is_initial(self) -> bool:
        return self.side == "initial"

    @property
    def is_final(self) -> bool:
        return self.side == "final"

    @property
    def is_colored(self) -> bool:
        return self.color_role not in {"singlet", "inclusive"}

    @property
    def is_singlet(self) -> bool:
        return self.color_role == "singlet"

    def to_json_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "side": self.side,
            "particle": self.particle,
            "outgoing_particle": self.outgoing_particle,
            "pdg": self.pdg,
            "outgoing_pdg": self.outgoing_pdg,
            "statistics": self.statistics,
            "wavefunction_family": self.wavefunction_family,
            "color_role": self.color_role,
            "source_orientation": self.source_orientation,
        }


@dataclass(frozen=True)
class ColorEndpointSummary:
    fundamental_count: int
    antifundamental_count: int
    pair_count: int

    @property
    def balanced(self) -> bool:
        return self.fundamental_count == self.antifundamental_count

    def to_json_dict(self) -> dict[str, object]:
        return {
            "fundamental_count": self.fundamental_count,
            "antifundamental_count": self.antifundamental_count,
            "pair_count": self.pair_count,
            "balanced": self.balanced,
        }


@dataclass(frozen=True)
class CanonicalProcessIR:
    """Canonical process data used before model-driven current generation."""

    process: str
    key: str
    color_accuracy: str
    legs: tuple[ProcessLegIR, ...]
    color_endpoints: ColorEndpointSummary

    @property
    def initial_legs(self) -> tuple[ProcessLegIR, ...]:
        return tuple(leg for leg in self.legs if leg.is_initial)

    @property
    def final_legs(self) -> tuple[ProcessLegIR, ...]:
        return tuple(leg for leg in self.legs if leg.is_final)

    @property
    def outgoing_particles(self) -> ProcessTuple:
        return tuple(leg.outgoing_particle for leg in self.legs)

    @property
    def outgoing_pdgs(self) -> tuple[int, ...]:
        return tuple(
            int(leg.outgoing_pdg) for leg in self.legs if leg.outgoing_pdg is not None
        )

    @property
    def initial_pdgs(self) -> tuple[int, ...]:
        return tuple(int(leg.pdg) for leg in self.initial_legs if leg.pdg is not None)

    @property
    def final_pdgs(self) -> tuple[int, ...]:
        return tuple(int(leg.pdg) for leg in self.final_legs if leg.pdg is not None)

    @property
    def fundamental_labels(self) -> tuple[int, ...]:
        return self._labels_with_color_role("fundamental")

    @property
    def antifundamental_labels(self) -> tuple[int, ...]:
        return self._labels_with_color_role("antifundamental")

    @property
    def adjoint_labels(self) -> tuple[int, ...]:
        return self._labels_with_color_role("adjoint")

    @property
    def singlet_labels(self) -> tuple[int, ...]:
        return self._labels_with_color_role("singlet")

    @property
    def has_inclusive_initial_state(self) -> bool:
        return any(leg.color_role == "inclusive" for leg in self.initial_legs)

    def _labels_with_color_role(self, color_role: ColorRole) -> tuple[int, ...]:
        return tuple(leg.label for leg in self.legs if leg.color_role == color_role)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "process": self.process,
            "key": self.key,
            "color_accuracy": self.color_accuracy,
            "legs": [leg.to_json_dict() for leg in self.legs],
            "outgoing_particles": list(self.outgoing_particles),
            "outgoing_pdgs": list(self.outgoing_pdgs),
            "color_endpoints": self.color_endpoints.to_json_dict(),
            "color_role_labels": {
                "fundamental": list(self.fundamental_labels),
                "antifundamental": list(self.antifundamental_labels),
                "adjoint": list(self.adjoint_labels),
                "singlet": list(self.singlet_labels),
            },
        }


@dataclass(frozen=True)
class ProcessSetEntryIR:
    key: str
    process: str
    ir: CanonicalProcessIR
    n_groups: int
    n_records: int
    n_unique_processes: int

    def to_json_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "process": self.process,
            "n_groups": self.n_groups,
            "n_records": self.n_records,
            "n_unique_processes": self.n_unique_processes,
            "ir": self.ir.to_json_dict(),
        }


@dataclass(frozen=True)
class ProcessSetIR:
    request: str
    color_accuracy: str
    entries: tuple[ProcessSetEntryIR, ...]

    @property
    def default_key(self) -> str:
        if not self.entries:
            raise ValueError("process set is empty")
        return self.entries[0].key

    def to_json_dict(self) -> dict[str, object]:
        return {
            "request": self.request,
            "color_accuracy": self.color_accuracy,
            "default_key": self.default_key,
            "entries": [entry.to_json_dict() for entry in self.entries],
        }


__all__ = [
    "CanonicalProcessIR",
    "ColorEndpointSummary",
    "ColorRole",
    "ParticleStatistics",
    "ProcessLegIR",
    "ProcessSetEntryIR",
    "ProcessSetIR",
    "SourceOrientation",
    "WavefunctionFamily",
]

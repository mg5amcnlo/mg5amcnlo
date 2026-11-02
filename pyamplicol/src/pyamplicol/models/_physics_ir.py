# SPDX-License-Identifier: 0BSD
"""Canonical, model-owned physics contracts used by generation and runtimes."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Literal, cast

ParticleOrientation = Literal["particle", "antiparticle", "self-conjugate"]
ParticleStatistics = Literal["boson", "fermion", "ghost", "auxiliary"]
WavefunctionFamily = Literal[
    "scalar",
    "fermion",
    "vector",
    "spin2",
    "ghost",
    "auxiliary",
]
MomentumTransform = Literal["identity", "negate-four-momentum"]
PropagatorKind = Literal[
    "identity",
    "scalar",
    "weyl-fermion",
    "dirac-fermion",
    "vector",
    "spin2",
    "custom",
    "unsupported",
]
PropagatorGauge = Literal[
    "feynman",
    "unitary",
    "de-donder",
    "fierz-pauli",
    "model-supplied",
]
PropagatorMassClass = Literal["massless", "massive", "not-applicable"]
GoldstonePolicy = Literal[
    "not-applicable",
    "absorbed",
    "explicit",
    "model-supplied",
]
TensorFlattening = Literal["row-major-last-axis-fastest"]
TENSOR_ORDERING_CONTRACT_VERSION = 1


def _strict_ir_fields(
    payload: Mapping[str, object],
    *,
    required: set[str],
    context: str,
) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{context} must be a mapping")
    fields = set(payload)
    missing = required - fields
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"{context} is missing required fields: {names}")
    unknown = fields - required
    if unknown:
        names = ", ".join(sorted(str(name) for name in unknown))
        raise ValueError(f"{context} has unknown fields: {names}")
    return payload


def _strict_ir_sequence(value: object, context: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{context} must be an array")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class TensorAxisIR:
    """One semantic tensor axis, independent of a source UFO index label."""

    name: str
    space: str
    extent: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("tensor axis name must be a non-empty string")
        if not isinstance(self.space, str) or not self.space:
            raise ValueError("tensor axis space must be a non-empty string")
        if isinstance(self.extent, bool) or not isinstance(self.extent, int):
            raise TypeError("tensor axis extent must be an integer")
        if self.extent <= 0:
            raise ValueError("tensor axis extent must be positive")

    def to_json_dict(self) -> dict[str, object]:
        return {"name": self.name, "space": self.space, "extent": self.extent}

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, object]) -> TensorAxisIR:
        fields = _strict_ir_fields(
            payload,
            required={"name", "space", "extent"},
            context="tensor axis IR",
        )
        name = fields["name"]
        space = fields["space"]
        extent = fields["extent"]
        if not isinstance(name, str) or not isinstance(space, str):
            raise TypeError("tensor axis name and space must be strings")
        if isinstance(extent, bool) or not isinstance(extent, int):
            raise TypeError("tensor axis extent must be an integer")
        return cls(name=name, space=space, extent=extent)


@dataclass(frozen=True, slots=True)
class TensorOrderingIR:
    """Content-addressed mapping from stored components to a canonical tensor."""

    ordering_id: str
    basis: str
    axes: tuple[TensorAxisIR, ...]
    flattening: TensorFlattening
    component_basis: tuple[int, ...]
    component_expansion: tuple[tuple[int, int] | None, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.basis, str) or not self.basis:
            raise ValueError("tensor ordering basis must be a non-empty string")
        if self.flattening != "row-major-last-axis-fastest":
            raise ValueError(f"unsupported tensor flattening {self.flattening!r}")
        if not all(isinstance(axis, TensorAxisIR) for axis in self.axes):
            raise TypeError("tensor ordering axes must contain TensorAxisIR records")
        axis_names = tuple(axis.name for axis in self.axes)
        if len(set(axis_names)) != len(axis_names):
            raise ValueError("tensor ordering axis names must be unique")
        if not self.component_basis:
            raise ValueError("tensor ordering component basis must not be empty")
        if any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in self.component_basis
        ):
            raise TypeError("tensor ordering component basis must contain integers")
        canonical_size = self.canonical_size
        if len(self.component_expansion) != canonical_size:
            raise ValueError(
                "tensor ordering expansion size does not match its canonical axes"
            )
        if len(set(self.component_basis)) != len(self.component_basis):
            raise ValueError("tensor ordering component basis must be unique")
        if any(index < 0 or index >= canonical_size for index in self.component_basis):
            raise ValueError("tensor ordering component basis index is out of range")
        referenced_slots: set[int] = set()
        for entry in self.component_expansion:
            if entry is None:
                continue
            if (
                not isinstance(entry, tuple)
                or len(entry) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in entry
                )
            ):
                raise TypeError(
                    "tensor ordering expansion entries must be (slot, sign) pairs"
                )
            slot, sign = entry
            if slot < 0 or slot >= self.stored_size:
                raise ValueError("tensor ordering expansion refers to an absent slot")
            if sign not in {-1, 1}:
                raise ValueError("tensor ordering expansion sign must be +/-1")
            referenced_slots.add(slot)
        if referenced_slots != set(range(self.stored_size)):
            raise ValueError(
                "tensor ordering expansion does not cover every stored slot"
            )
        for slot, canonical_component in enumerate(self.component_basis):
            if self.component_expansion[canonical_component] != (slot, 1):
                raise ValueError(
                    "tensor ordering component basis disagrees with its expansion"
                )
        expected_id = self.content_id(
            basis=self.basis,
            axes=self.axes,
            flattening=self.flattening,
            component_basis=self.component_basis,
            component_expansion=self.component_expansion,
        )
        if self.ordering_id != expected_id:
            raise ValueError(
                "tensor ordering ID does not match its semantic component contract"
            )

    @property
    def canonical_size(self) -> int:
        return math.prod(axis.extent for axis in self.axes) if self.axes else 1

    @property
    def stored_size(self) -> int:
        return len(self.component_basis)

    @classmethod
    def identity(
        cls,
        *,
        basis: str,
        axes: Sequence[TensorAxisIR],
    ) -> TensorOrderingIR:
        axis_tuple = tuple(axes)
        size = math.prod(axis.extent for axis in axis_tuple) if axis_tuple else 1
        component_basis = tuple(range(size))
        component_expansion = tuple((index, 1) for index in range(size))
        return cls.create(
            basis=basis,
            axes=axis_tuple,
            component_basis=component_basis,
            component_expansion=component_expansion,
        )

    @classmethod
    def create(
        cls,
        *,
        basis: str,
        axes: Sequence[TensorAxisIR],
        component_basis: Sequence[int],
        component_expansion: Sequence[tuple[int, int] | None],
        flattening: TensorFlattening = "row-major-last-axis-fastest",
    ) -> TensorOrderingIR:
        axis_tuple = tuple(axes)
        basis_tuple = tuple(component_basis)
        expansion_tuple = tuple(component_expansion)
        return cls(
            ordering_id=cls.content_id(
                basis=basis,
                axes=axis_tuple,
                flattening=flattening,
                component_basis=basis_tuple,
                component_expansion=expansion_tuple,
            ),
            basis=basis,
            axes=axis_tuple,
            flattening=flattening,
            component_basis=basis_tuple,
            component_expansion=expansion_tuple,
        )

    @staticmethod
    def content_id(
        *,
        basis: str,
        axes: Sequence[TensorAxisIR],
        flattening: TensorFlattening,
        component_basis: Sequence[int],
        component_expansion: Sequence[tuple[int, int] | None],
    ) -> str:
        payload = {
            "basis": basis,
            "axes": [axis.to_json_dict() for axis in axes],
            "flattening": flattening,
            "component_basis": list(component_basis),
            "component_expansion": [
                None if entry is None else list(entry) for entry in component_expansion
            ],
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return "tensor-ordering-sha256:" + hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()

    def to_json_dict(self) -> dict[str, object]:
        return {
            "ordering_id": self.ordering_id,
            "basis": self.basis,
            "axes": [axis.to_json_dict() for axis in self.axes],
            "flattening": self.flattening,
            "component_basis": list(self.component_basis),
            "component_expansion": [
                None if entry is None else list(entry)
                for entry in self.component_expansion
            ],
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, object]) -> TensorOrderingIR:
        fields = _strict_ir_fields(
            payload,
            required={
                "ordering_id",
                "basis",
                "axes",
                "flattening",
                "component_basis",
                "component_expansion",
            },
            context="tensor ordering IR",
        )
        ordering_id = fields["ordering_id"]
        basis = fields["basis"]
        flattening = fields["flattening"]
        if not all(
            isinstance(value, str) for value in (ordering_id, basis, flattening)
        ):
            raise TypeError("tensor ordering ID, basis, and flattening must be strings")
        axes = tuple(
            TensorAxisIR.from_json_dict(axis)
            for axis in _strict_ir_sequence(fields["axes"], "tensor ordering axes")
            if isinstance(axis, Mapping)
        )
        axis_values = _strict_ir_sequence(fields["axes"], "tensor ordering axes")
        if len(axes) != len(axis_values):
            raise TypeError("tensor ordering axes must contain mappings")
        component_basis_values = _strict_ir_sequence(
            fields["component_basis"], "tensor ordering component basis"
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in component_basis_values
        ):
            raise TypeError("tensor ordering component basis must contain integers")
        expansion_values = _strict_ir_sequence(
            fields["component_expansion"], "tensor ordering component expansion"
        )
        expansion: list[tuple[int, int] | None] = []
        for entry in expansion_values:
            if entry is None:
                expansion.append(None)
                continue
            pair = _strict_ir_sequence(entry, "tensor ordering expansion entry")
            if len(pair) != 2 or any(
                isinstance(value, bool) or not isinstance(value, int) for value in pair
            ):
                raise TypeError(
                    "tensor ordering expansion entry must be an integer pair"
                )
            expansion.append((pair[0], pair[1]))
        return cls(
            ordering_id=ordering_id,
            basis=basis,
            axes=axes,
            flattening=cast(TensorFlattening, flattening),
            component_basis=cast(tuple[int, ...], component_basis_values),
            component_expansion=tuple(expansion),
        )


@dataclass(frozen=True, slots=True)
class TensorIndexBindingIR:
    """Provenance for one normalized tensor index used by a UFO vertex term."""

    origin: str
    space: str
    source_component: int | None
    source_leg: int | None
    source_dummy: int | None
    normalized_name: str

    def __post_init__(self) -> None:
        for field_name in ("origin", "space", "normalized_name"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"tensor index binding {field_name} must not be empty")
        if (self.source_leg is None) == (self.source_dummy is None):
            raise ValueError(
                "tensor index binding must identify exactly one source leg or dummy"
            )
        for field_name in ("source_component", "source_leg", "source_dummy"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int)
            ):
                raise TypeError(f"tensor index binding {field_name} must be an integer")
        if self.source_component is not None and self.source_component < 0:
            raise ValueError("tensor index source component must be non-negative")
        if self.source_leg is not None and self.source_leg <= 0:
            raise ValueError("tensor index source leg must be positive")
        if self.source_dummy == 0:
            raise ValueError("tensor index source dummy must be nonzero")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "origin": self.origin,
            "space": self.space,
            "source_component": self.source_component,
            "source_leg": self.source_leg,
            "source_dummy": self.source_dummy,
            "normalized_name": self.normalized_name,
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, object]) -> TensorIndexBindingIR:
        fields = _strict_ir_fields(
            payload,
            required={
                "origin",
                "space",
                "source_component",
                "source_leg",
                "source_dummy",
                "normalized_name",
            },
            context="tensor index binding IR",
        )
        for field_name in ("origin", "space", "normalized_name"):
            if not isinstance(fields[field_name], str):
                raise TypeError(f"tensor index binding {field_name} must be a string")
        for field_name in ("source_component", "source_leg", "source_dummy"):
            value = fields[field_name]
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int)
            ):
                raise TypeError(f"tensor index binding {field_name} must be an integer")
        return cls(
            origin=cast(str, fields["origin"]),
            space=cast(str, fields["space"]),
            source_component=cast(int | None, fields["source_component"]),
            source_leg=cast(int | None, fields["source_leg"]),
            source_dummy=cast(int | None, fields["source_dummy"]),
            normalized_name=cast(str, fields["normalized_name"]),
        )


@dataclass(frozen=True, slots=True)
class CompiledCurrentOrderingRecord:
    """Map one stored current state to the full ordering used by kernels."""

    particle: str
    chirality: int
    ordering_id: str
    kernel_ordering_id: str
    input_embedding: tuple[int | None, ...]
    result_projection: tuple[int, ...]

    def __post_init__(self) -> None:
        for field_name in ("particle", "ordering_id", "kernel_ordering_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"current ordering {field_name} must not be empty")
        if isinstance(self.chirality, bool) or not isinstance(self.chirality, int):
            raise TypeError("current ordering chirality must be an integer")
        if any(
            value is not None
            and (isinstance(value, bool) or not isinstance(value, int))
            for value in self.input_embedding
        ):
            raise TypeError("current input embedding must contain integers or null")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in self.result_projection
        ):
            raise TypeError("current result projection must contain integers")

    @property
    def selector(self) -> tuple[str, int]:
        return self.particle, self.chirality

    def to_json_dict(self) -> dict[str, object]:
        return {
            "particle": self.particle,
            "chirality": self.chirality,
            "ordering_id": self.ordering_id,
            "kernel_ordering_id": self.kernel_ordering_id,
            "input_embedding": list(self.input_embedding),
            "result_projection": list(self.result_projection),
        }

    @classmethod
    def from_json_dict(
        cls, payload: Mapping[str, object]
    ) -> CompiledCurrentOrderingRecord:
        fields = _strict_ir_fields(
            payload,
            required={
                "particle",
                "chirality",
                "ordering_id",
                "kernel_ordering_id",
                "input_embedding",
                "result_projection",
            },
            context="compiled current ordering",
        )
        for field_name in ("particle", "ordering_id", "kernel_ordering_id"):
            if not isinstance(fields[field_name], str):
                raise TypeError(f"current ordering {field_name} must be a string")
        chirality = fields["chirality"]
        if isinstance(chirality, bool) or not isinstance(chirality, int):
            raise TypeError("current ordering chirality must be an integer")
        embedding = _strict_ir_sequence(
            fields["input_embedding"], "current input embedding"
        )
        projection = _strict_ir_sequence(
            fields["result_projection"], "current result projection"
        )
        return cls(
            particle=cast(str, fields["particle"]),
            chirality=chirality,
            ordering_id=cast(str, fields["ordering_id"]),
            kernel_ordering_id=cast(str, fields["kernel_ordering_id"]),
            input_embedding=cast(tuple[int | None, ...], embedding),
            result_projection=cast(tuple[int, ...], projection),
        )


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a boolean")
    return value


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("optional propagator metadata must be a string or null")
    return value


def _particle_orientation(value: object) -> ParticleOrientation:
    if value not in {"particle", "antiparticle", "self-conjugate"}:
        raise ValueError(f"invalid particle orientation {value!r}")
    return cast(ParticleOrientation, value)


def _propagator_kind(value: object) -> PropagatorKind:
    if value not in {
        "identity",
        "scalar",
        "weyl-fermion",
        "dirac-fermion",
        "vector",
        "spin2",
        "custom",
        "unsupported",
    }:
        raise ValueError(f"invalid propagator kind {value!r}")
    return cast(PropagatorKind, value)


def _optional_propagator_gauge(value: object) -> PropagatorGauge | None:
    if value is None:
        return None
    if value not in {
        "feynman",
        "unitary",
        "de-donder",
        "fierz-pauli",
        "model-supplied",
    }:
        raise ValueError(f"invalid propagator gauge {value!r}")
    return cast(PropagatorGauge, value)


def _goldstone_policy(value: object) -> GoldstonePolicy:
    if value not in {
        "not-applicable",
        "absorbed",
        "explicit",
        "model-supplied",
    }:
        raise ValueError(f"invalid propagator Goldstone policy {value!r}")
    return cast(GoldstonePolicy, value)


def _propagator_mass_class(value: object) -> PropagatorMassClass:
    if value not in {"massless", "massive", "not-applicable"}:
        raise ValueError(f"invalid propagator mass class {value!r}")
    return cast(PropagatorMassClass, value)


@dataclass(frozen=True, slots=True)
class ParticleIdentityIR:
    """Identity of one oriented model state, independent of SM taxonomy."""

    canonical_id: str
    species_id: str
    anti_canonical_id: str
    display_name: str
    anti_display_name: str
    pdg_label: int
    anti_pdg_label: int
    orientation: ParticleOrientation
    self_conjugate: bool

    def __post_init__(self) -> None:
        for field_name in (
            "canonical_id",
            "species_id",
            "anti_canonical_id",
            "display_name",
            "anti_display_name",
        ):
            if not getattr(self, field_name):
                raise ValueError(f"particle identity {field_name} must not be empty")
        for field_name in ("pdg_label", "anti_pdg_label"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"particle identity {field_name} must be an integer")
        if self.orientation not in {
            "particle",
            "antiparticle",
            "self-conjugate",
        }:
            raise ValueError(f"invalid particle orientation {self.orientation!r}")
        if self.self_conjugate != (self.canonical_id == self.anti_canonical_id):
            raise ValueError(
                "particle self-conjugacy must agree with its canonical anti relation"
            )
        if self.self_conjugate != (self.pdg_label == self.anti_pdg_label):
            raise ValueError(
                "particle self-conjugacy must agree with its PDG-label anti relation"
            )
        if (self.orientation == "self-conjugate") != self.self_conjugate:
            raise ValueError(
                "self-conjugate source orientation must agree with particle identity"
            )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "canonical_id": self.canonical_id,
            "species_id": self.species_id,
            "anti_canonical_id": self.anti_canonical_id,
            "display_name": self.display_name,
            "anti_display_name": self.anti_display_name,
            "pdg_label": self.pdg_label,
            "anti_pdg_label": self.anti_pdg_label,
            "orientation": self.orientation,
            "self_conjugate": self.self_conjugate,
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, object]) -> ParticleIdentityIR:
        return cls(
            canonical_id=str(payload["canonical_id"]),
            species_id=str(payload["species_id"]),
            anti_canonical_id=str(payload["anti_canonical_id"]),
            display_name=str(payload["display_name"]),
            anti_display_name=str(payload["anti_display_name"]),
            pdg_label=_integer(payload["pdg_label"], "particle PDG label"),
            anti_pdg_label=_integer(
                payload["anti_pdg_label"],
                "antiparticle PDG label",
            ),
            orientation=_particle_orientation(payload["orientation"]),
            self_conjugate=_boolean(
                payload["self_conjugate"],
                "particle self-conjugacy",
            ),
        )


@dataclass(frozen=True, slots=True)
class SourceStateIR:
    helicity: int
    chirality: int
    spin_state: int | tuple[int, ...]

    def __post_init__(self) -> None:
        for field_name in ("helicity", "chirality"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"source state {field_name} must be an integer")
        if isinstance(self.spin_state, bool) or not isinstance(
            self.spin_state, (int, tuple)
        ):
            raise TypeError("source spin state must be an integer or integer tuple")
        if isinstance(self.spin_state, tuple) and not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in self.spin_state
        ):
            raise TypeError("source spin-state tuples must contain only integers")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "helicity": self.helicity,
            "chirality": self.chirality,
            "spin_state": (
                list(self.spin_state)
                if isinstance(self.spin_state, tuple)
                else self.spin_state
            ),
        }


@dataclass(frozen=True, slots=True)
class CrossingIR:
    """Transform an outgoing source basis into an incoming physical leg."""

    momentum_transform: MomentumTransform = "negate-four-momentum"
    helicity_factor: int = 1
    chirality_factor: int = 1
    spin_state_factor: int = 1
    phase: tuple[float, float] = (1.0, 0.0)

    def __post_init__(self) -> None:
        if self.momentum_transform not in {"identity", "negate-four-momentum"}:
            raise ValueError(
                f"unsupported crossing momentum transform {self.momentum_transform!r}"
            )
        for field_name in (
            "helicity_factor",
            "chirality_factor",
            "spin_state_factor",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"crossing {field_name} must be an integer")
            if value not in {-1, 1}:
                raise ValueError(f"crossing {field_name} must be -1 or 1")
        if not isinstance(self.phase, (tuple, list)) or len(self.phase) != 2:
            raise ValueError("crossing phase must be a finite complex pair")
        if any(
            isinstance(value, bool) or not isinstance(value, Real)
            for value in self.phase
        ):
            raise TypeError("crossing phase components must be real numbers")
        phase = (float(self.phase[0]), float(self.phase[1]))
        if not all(math.isfinite(value) for value in phase):
            raise ValueError("crossing phase must be a finite complex pair")
        if phase == (0.0, 0.0):
            raise ValueError("crossing phase must be nonzero")
        object.__setattr__(self, "phase", phase)

    @classmethod
    def identity(cls) -> CrossingIR:
        return cls(momentum_transform="identity")

    def apply(self, state: SourceStateIR) -> SourceStateIR:
        spin_state = state.spin_state
        if self.spin_state_factor != 1:
            if not isinstance(spin_state, int):
                raise TypeError(
                    "crossing cannot multiply a structured source spin state"
                )
            spin_state *= self.spin_state_factor
        return SourceStateIR(
            helicity=state.helicity * self.helicity_factor,
            chirality=state.chirality * self.chirality_factor,
            spin_state=spin_state,
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "momentum_transform": self.momentum_transform,
            "helicity_factor": self.helicity_factor,
            "chirality_factor": self.chirality_factor,
            "spin_state_factor": self.spin_state_factor,
            "phase": list(self.phase),
        }


@dataclass(frozen=True, slots=True)
class SourceIR:
    identity: ParticleIdentityIR
    statistics: ParticleStatistics
    wavefunction_family: WavefunctionFamily
    component_dimension: int
    states: tuple[SourceStateIR, ...]
    crossing: CrossingIR
    basis: str
    mass_parameter: str | None = None
    width_parameter: str | None = None

    def __post_init__(self) -> None:
        if self.statistics not in {"boson", "fermion", "ghost", "auxiliary"}:
            raise ValueError(f"invalid source statistics {self.statistics!r}")
        if self.wavefunction_family not in {
            "scalar",
            "fermion",
            "vector",
            "spin2",
            "ghost",
            "auxiliary",
        }:
            raise ValueError(
                f"invalid source wavefunction family {self.wavefunction_family!r}"
            )
        expected_statistics: ParticleStatistics
        if self.wavefunction_family == "fermion":
            expected_statistics = "fermion"
        elif self.wavefunction_family == "ghost":
            expected_statistics = "ghost"
        elif self.wavefunction_family == "auxiliary":
            expected_statistics = "auxiliary"
        else:
            expected_statistics = "boson"
        if self.statistics != expected_statistics:
            raise ValueError(
                f"source wavefunction family {self.wavefunction_family!r} requires "
                f"statistics {expected_statistics!r}, got {self.statistics!r}"
            )
        if isinstance(self.component_dimension, bool) or not isinstance(
            self.component_dimension, int
        ):
            raise TypeError("source component dimension must be an integer")
        if self.component_dimension < 1:
            raise ValueError("source component dimension must be positive")
        if not self.states:
            raise ValueError("source metadata must declare at least one spin state")
        if not self.basis:
            raise ValueError("source basis must not be empty")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity.to_json_dict(),
            "statistics": self.statistics,
            "wavefunction_family": self.wavefunction_family,
            "component_dimension": self.component_dimension,
            "states": [state.to_json_dict() for state in self.states],
            "crossing": self.crossing.to_json_dict(),
            "basis": self.basis,
            "mass_parameter": self.mass_parameter,
            "width_parameter": self.width_parameter,
        }


@dataclass(frozen=True, slots=True)
class PropagatorIR:
    identity: ParticleIdentityIR
    chirality: int
    kind: PropagatorKind
    backend: str
    basis: str
    applies_propagator: bool
    kernel: str
    full_tensor_network_ready: bool
    mass_class: PropagatorMassClass
    gauge: PropagatorGauge | None = None
    numerator: str | None = None
    denominator: str | None = None
    mass_parameter: str | None = None
    width_parameter: str | None = None
    custom_source: str | None = None
    auxiliary_policy: str | None = None
    goldstone_policy: GoldstonePolicy = "not-applicable"
    description: str = ""

    def __post_init__(self) -> None:
        if not self.backend or not self.basis or not self.kernel:
            raise ValueError("propagator backend, basis, and kernel must not be empty")
        if self.kind not in {
            "identity",
            "scalar",
            "weyl-fermion",
            "dirac-fermion",
            "vector",
            "spin2",
            "custom",
            "unsupported",
        }:
            raise ValueError(f"invalid propagator kind {self.kind!r}")
        if self.gauge not in {
            None,
            "feynman",
            "unitary",
            "de-donder",
            "fierz-pauli",
            "model-supplied",
        }:
            raise ValueError(f"invalid propagator gauge {self.gauge!r}")
        if self.goldstone_policy not in {
            "not-applicable",
            "absorbed",
            "explicit",
            "model-supplied",
        }:
            raise ValueError(
                f"invalid propagator Goldstone policy {self.goldstone_policy!r}"
            )
        if self.mass_class not in {"massless", "massive", "not-applicable"}:
            raise ValueError(f"invalid propagator mass class {self.mass_class!r}")
        if not self.applies_propagator and self.auxiliary_policy is None:
            raise ValueError(
                "a no-propagator current must declare its auxiliary policy"
            )
        if not self.applies_propagator and self.kind != "identity":
            raise ValueError("a no-propagator current must use the identity kind")
        if self.applies_propagator and self.kind == "identity":
            raise ValueError("an identity current cannot apply a propagator")
        if (self.kind == "identity") != (self.mass_class == "not-applicable"):
            raise ValueError(
                "only identity currents may use a not-applicable mass class"
            )
        if self.kind == "vector" and self.gauge is None:
            raise ValueError("a vector propagator must declare its gauge")
        if self.kind == "custom" and (
            self.gauge != "model-supplied" or not self.custom_source
        ):
            raise ValueError(
                "a custom propagator must declare its model-supplied source"
            )
        if self.goldstone_policy == "absorbed" and not (
            self.kind == "vector" and self.gauge == "unitary"
        ):
            raise ValueError(
                "an absorbed Goldstone mode requires a unitary-gauge vector"
            )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "identity": self.identity.to_json_dict(),
            "particle_id": self.identity.pdg_label,
            "chirality": self.chirality,
            "kind": self.kind,
            "backend": self.backend,
            "basis": self.basis,
            "applies_propagator": self.applies_propagator,
            "kernel": self.kernel,
            "full_tensor_network_ready": self.full_tensor_network_ready,
            "mass_class": self.mass_class,
            "gauge": self.gauge,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "mass_parameter": self.mass_parameter,
            "width_parameter": self.width_parameter,
            "custom_source": self.custom_source,
            "auxiliary_policy": self.auxiliary_policy,
            "goldstone_policy": self.goldstone_policy,
            "description": self.description,
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, object]) -> PropagatorIR:
        identity = payload.get("identity")
        if not isinstance(identity, Mapping):
            raise TypeError("propagator identity must be a mapping")
        return cls(
            identity=ParticleIdentityIR.from_json_dict(identity),
            chirality=_integer(payload["chirality"], "propagator chirality"),
            kind=_propagator_kind(payload["kind"]),
            backend=str(payload["backend"]),
            basis=str(payload["basis"]),
            applies_propagator=_boolean(
                payload["applies_propagator"],
                "propagator application flag",
            ),
            kernel=str(payload["kernel"]),
            full_tensor_network_ready=_boolean(
                payload["full_tensor_network_ready"],
                "propagator tensor-network readiness",
            ),
            mass_class=_propagator_mass_class(payload["mass_class"]),
            gauge=_optional_propagator_gauge(payload.get("gauge")),
            numerator=_optional_string(payload.get("numerator")),
            denominator=_optional_string(payload.get("denominator")),
            mass_parameter=_optional_string(payload.get("mass_parameter")),
            width_parameter=_optional_string(payload.get("width_parameter")),
            custom_source=_optional_string(payload.get("custom_source")),
            auxiliary_policy=_optional_string(payload.get("auxiliary_policy")),
            goldstone_policy=_goldstone_policy(
                payload.get("goldstone_policy", "not-applicable")
            ),
            description=str(payload.get("description", "")),
        )


@dataclass(frozen=True, slots=True)
class ContractionIR:
    name: str
    left_basis: str
    right_basis: str
    coefficients: tuple[tuple[float, float], ...]
    chirality_relation: Literal["any", "equal", "opposite"] = "any"
    metric_signature: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("name", "left_basis", "right_basis"):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(f"contraction {field_name} must be a string")
        if not self.name or not self.left_basis or not self.right_basis:
            raise ValueError("contraction name and bases must not be empty")
        if isinstance(self.coefficients, (str, bytes)) or not isinstance(
            self.coefficients, Sequence
        ):
            raise TypeError("contraction coefficients must be a sequence")
        if not self.coefficients:
            raise ValueError("contraction must declare component coefficients")
        coefficients: list[tuple[float, float]] = []
        for index, value in enumerate(self.coefficients):
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise TypeError(
                    f"contraction coefficient {index} must be a complex pair"
                )
            if len(value) != 2:
                raise ValueError(
                    "contraction coefficients must be finite complex pairs"
                )
            if any(
                isinstance(component, bool) or not isinstance(component, Real)
                for component in value
            ):
                raise TypeError(
                    "contraction coefficient components must be real numbers"
                )
            coefficient = (float(value[0]), float(value[1]))
            if not all(math.isfinite(component) for component in coefficient):
                raise ValueError(
                    "contraction coefficients must be finite complex pairs"
                )
            coefficients.append(coefficient)
        normalized_coefficients = tuple(coefficients)
        if not any(value != (0.0, 0.0) for value in normalized_coefficients):
            raise ValueError("contraction must have a nonzero component coefficient")
        object.__setattr__(self, "coefficients", normalized_coefficients)
        if not isinstance(self.chirality_relation, str):
            raise TypeError("contraction chirality relation must be a string")
        if self.chirality_relation not in {"any", "equal", "opposite"}:
            raise ValueError(
                f"invalid contraction chirality relation {self.chirality_relation!r}"
            )
        if self.metric_signature is not None:
            if not isinstance(self.metric_signature, str):
                raise TypeError("contraction metric signature must be a string or null")
            if not self.metric_signature:
                raise ValueError("contraction metric signature must not be empty")

    def to_json_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "left_basis": self.left_basis,
            "right_basis": self.right_basis,
            "coefficients": [list(value) for value in self.coefficients],
            "chirality_relation": self.chirality_relation,
            "metric_signature": self.metric_signature,
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, object]) -> ContractionIR:
        if not isinstance(payload, Mapping):
            raise TypeError("contraction IR must be a mapping")
        required_fields = {
            "name",
            "left_basis",
            "right_basis",
            "coefficients",
            "chirality_relation",
            "metric_signature",
        }
        fields = set(payload)
        missing = required_fields - fields
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"contraction IR is missing required fields: {names}")
        unknown = fields - required_fields
        if unknown:
            names = ", ".join(sorted(str(name) for name in unknown))
            raise ValueError(f"contraction IR has unknown fields: {names}")

        coefficients = payload["coefficients"]
        if isinstance(coefficients, (str, bytes)) or not isinstance(
            coefficients, Sequence
        ):
            raise TypeError("contraction coefficients must be an array")
        parsed_coefficients: list[tuple[float, float]] = []
        for index, value in enumerate(coefficients):
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise TypeError(
                    f"contraction coefficient {index} must be a complex pair"
                )
            if len(value) != 2:
                raise ValueError(
                    f"contraction coefficient {index} must have two components"
                )
            if any(
                isinstance(component, bool) or not isinstance(component, Real)
                for component in value
            ):
                raise TypeError(
                    "contraction coefficient components must be real numbers"
                )
            parsed_coefficients.append((float(value[0]), float(value[1])))

        name = payload["name"]
        left_basis = payload["left_basis"]
        right_basis = payload["right_basis"]
        if not isinstance(name, str):
            raise TypeError("contraction name must be a string")
        if not isinstance(left_basis, str) or not isinstance(right_basis, str):
            raise TypeError("contraction bases must be strings")
        metric_signature = payload["metric_signature"]
        if metric_signature is not None and not isinstance(metric_signature, str):
            raise TypeError("contraction metric signature must be a string or null")
        chirality_relation = payload["chirality_relation"]
        if not isinstance(chirality_relation, str):
            raise TypeError("contraction chirality relation must be a string")
        if chirality_relation not in {"any", "equal", "opposite"}:
            raise ValueError(
                f"invalid contraction chirality relation {chirality_relation!r}"
            )
        return cls(
            name=name,
            left_basis=left_basis,
            right_basis=right_basis,
            coefficients=tuple(parsed_coefficients),
            chirality_relation=cast(
                Literal["any", "equal", "opposite"], chirality_relation
            ),
            metric_signature=metric_signature,
        )

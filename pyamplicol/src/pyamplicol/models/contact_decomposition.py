# SPDX-License-Identifier: 0BSD
"""Serialized proof records for explicit UFO contact decomposition."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

CONTACT_DECOMPOSITION_ALGORITHM = "ufo-four-point-contact-decomposition"
CONTACT_DECOMPOSITION_ALGORITHM_VERSION = 2
CONTACT_ORBIT_ALGORITHM = "compiler-certified-contact-orbit"
CONTACT_ORBIT_ALGORITHM_VERSION = 2
CONTACT_ORBIT_EVALUATOR_CLASS = "constant-scalar-literal-singlet-self-conjugate-boson"
HEFT_CONTACT_ORBIT_EVALUATOR_CLASS = "scalar-heft-colored-contact-tree"
CONTACT_ORBIT_EVALUATOR_CLASSES = frozenset(
    {CONTACT_ORBIT_EVALUATOR_CLASS, HEFT_CONTACT_ORBIT_EVALUATOR_CLASS}
)


def contact_orbit_evaluator_class_is_certifiable(
    *,
    color_source: str,
    color_expression: str,
    lorentz_expression: str,
    decomposition_kinds: tuple[str, ...],
    particle_contracts: tuple[tuple[int, int, str, str, bool | None], ...],
) -> bool:
    """Recognize the one compiler-owned evaluator class admitted by the proof."""

    return (
        len(particle_contracts) == 4
        and color_expression == "1"
        and lorentz_expression == "1"
        and (color_source in {"1", "UFO::{}::1"} or color_expression == "1")
        and bool(decomposition_kinds)
        and all(kind == "literal-color-singlet" for kind in decomposition_kinds)
        and all(
            spin == 1
            and color == 1
            and statistics == "boson"
            and wavefunction_family == "scalar"
            and self_conjugate is True
            for spin, color, statistics, wavefunction_family, self_conjugate in (
                particle_contracts
            )
        )
    )


class _ContactTerm(Protocol):
    @property
    def id(self) -> int: ...

    @property
    def vertex(self) -> str: ...

    @property
    def particles(self) -> tuple[str, ...]: ...

    @property
    def color_index(self) -> int: ...

    @property
    def lorentz_index(self) -> int: ...

    @property
    def color_source(self) -> str: ...

    @property
    def color_expression(self) -> str: ...

    @property
    def lorentz_name(self) -> str: ...

    @property
    def lorentz_source(self) -> str: ...

    @property
    def lorentz_expression(self) -> str: ...


@dataclass(frozen=True)
class CompiledContactUnsupportedReason:
    code: str
    message: str
    context: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.code or not self.message:
            raise ValueError(
                "contact decomposition reason must have a code and message"
            )
        if self.context != tuple(sorted(self.context)):
            raise ValueError("contact decomposition reason context must be sorted")
        names = tuple(name for name, _value in self.context)
        if any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError(
                "contact decomposition reason context names must be non-empty and "
                "unique"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "context": dict(self.context),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
    ) -> CompiledContactUnsupportedReason:
        fields = _strict_record_fields(
            payload,
            required={"code", "message", "context"},
            context="compiled contact unsupported reason",
        )
        context = _strict_mapping(fields["context"], "contact reason context")
        return cls(
            code=_strict_string(fields["code"], "contact reason code"),
            message=_strict_string(fields["message"], "contact reason message"),
            context=tuple(
                sorted(
                    (
                        _strict_string(name, "contact reason context name"),
                        _strict_string(value, "contact reason context value"),
                    )
                    for name, value in context.items()
                )
            ),
        )


@dataclass(frozen=True)
class CompiledContactDummyIndexMapping:
    source_index: int
    normalized_symbol: str
    outer_slot: int
    final_slot: int

    def __post_init__(self) -> None:
        if self.source_index >= 0:
            raise ValueError("contact dummy source index must be negative")
        if not self.normalized_symbol:
            raise ValueError("contact dummy normalized symbol must not be empty")
        if self.outer_slot not in range(3) or self.final_slot not in range(3):
            raise ValueError("contact dummy slots must address a structure constant")

    def to_dict(self) -> dict[str, object]:
        return {
            "source_index": self.source_index,
            "normalized_symbol": self.normalized_symbol,
            "outer_slot": self.outer_slot,
            "final_slot": self.final_slot,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
    ) -> CompiledContactDummyIndexMapping:
        fields = _strict_record_fields(
            payload,
            required={"source_index", "normalized_symbol", "outer_slot", "final_slot"},
            context="compiled contact dummy-index mapping",
        )
        return cls(
            source_index=_strict_integer(
                fields["source_index"], "contact dummy source index"
            ),
            normalized_symbol=_strict_string(
                fields["normalized_symbol"], "contact dummy normalized symbol"
            ),
            outer_slot=_strict_integer(
                fields["outer_slot"], "contact dummy outer slot"
            ),
            final_slot=_strict_integer(
                fields["final_slot"], "contact dummy final slot"
            ),
        )


@dataclass(frozen=True)
class CompiledContactOrientationProof:
    stage: str
    input_legs: tuple[int, int]
    permutation_parity: int
    scalar_prefactor: str

    def __post_init__(self) -> None:
        if self.stage not in {"partial", "final"}:
            raise ValueError(f"unknown contact proof orientation stage {self.stage!r}")
        if self.permutation_parity not in {-1, 1}:
            raise ValueError("contact orientation parity must be -1 or 1")
        if not self.scalar_prefactor:
            raise ValueError("contact orientation scalar prefactor must not be empty")
        auxiliary_count = self.input_legs.count(-1)
        if self.stage == "partial" and (
            auxiliary_count or any(leg < 0 for leg in self.input_legs)
        ):
            raise ValueError("contact partial orientation requires two physical legs")
        if self.stage == "final" and auxiliary_count != 1:
            raise ValueError("contact final orientation requires one auxiliary leg")

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "input_legs": list(self.input_legs),
            "permutation_parity": self.permutation_parity,
            "scalar_prefactor": self.scalar_prefactor,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
    ) -> CompiledContactOrientationProof:
        fields = _strict_record_fields(
            payload,
            required={
                "stage",
                "input_legs",
                "permutation_parity",
                "scalar_prefactor",
            },
            context="compiled contact orientation proof",
        )
        return cls(
            stage=_strict_string(fields["stage"], "contact orientation stage"),
            input_legs=_strict_int_pair(
                fields["input_legs"], "contact orientation input legs"
            ),
            permutation_parity=_strict_integer(
                fields["permutation_parity"], "contact orientation parity"
            ),
            scalar_prefactor=_strict_string(
                fields["scalar_prefactor"], "contact orientation scalar prefactor"
            ),
        )


@dataclass(frozen=True)
class CompiledContactDecompositionSplit:
    decomposition_kind: str
    result_leg: int
    pair_legs: tuple[int, int]
    remaining_leg: int
    outer_color_source: str
    final_color_source: str
    outer_color_factor: tuple[int, ...]
    final_color_factor: tuple[int, ...]
    dummy_index_mapping: CompiledContactDummyIndexMapping | None
    outer_color_normalization_power: int
    final_color_normalization_power: int
    color_coefficient: str
    auxiliary_color: int
    open_legs: tuple[int, int]
    component_axis_order: tuple[str, ...]
    component_basis_order: tuple[int, ...]
    component_expansion: tuple[tuple[int, int] | None, ...]
    assignment_multiplicity: int
    canonical_outer_parity: int
    orientations: tuple[CompiledContactOrientationProof, ...]

    def __post_init__(self) -> None:
        if self.decomposition_kind not in {
            "literal-color-singlet",
            "two-structure-constants",
        }:
            raise ValueError(
                f"unknown contact decomposition kind {self.decomposition_kind!r}"
            )
        physical_legs = (*self.pair_legs, self.remaining_leg, self.result_leg)
        if any(leg < 0 for leg in physical_legs) or len(set(physical_legs)) != 4:
            raise ValueError("contact decomposition must partition four physical legs")
        if self.open_legs != tuple(sorted((self.remaining_leg, self.result_leg))):
            raise ValueError("contact decomposition open-leg order is not canonical")
        if self.assignment_multiplicity <= 0:
            raise ValueError("contact assignment multiplicity must be positive")
        if (
            self.outer_color_normalization_power < 0
            or self.final_color_normalization_power < 0
        ):
            raise ValueError("contact color normalization powers must be non-negative")
        if self.canonical_outer_parity not in {-1, 1}:
            raise ValueError("contact canonical outer parity must be -1 or 1")
        if not self.color_coefficient:
            raise ValueError("contact color coefficient must not be empty")
        _validate_auxiliary_color(self.auxiliary_color)
        if not self.component_basis_order or not self.component_expansion:
            raise ValueError("contact component basis and expansion must not be empty")
        if len(set(self.component_basis_order)) != len(self.component_basis_order):
            raise ValueError(
                "contact component basis order must not contain duplicates"
            )
        if any(
            source < 0 or source >= len(self.component_expansion)
            for source in self.component_basis_order
        ):
            raise ValueError("contact component basis source is outside the expansion")
        for entry in self.component_expansion:
            if entry is None:
                continue
            basis_index, coefficient = entry
            if basis_index not in range(len(self.component_basis_order)):
                raise ValueError(
                    "contact component expansion has an absent basis index"
                )
            if coefficient not in {-1, 1}:
                raise ValueError("contact component expansion coefficient must be +/-1")
        for basis_index, source_component in enumerate(self.component_basis_order):
            if self.component_expansion[source_component] != (basis_index, 1):
                raise ValueError(
                    "contact component basis order disagrees with its expansion"
                )
        partials = tuple(item for item in self.orientations if item.stage == "partial")
        finals = tuple(item for item in self.orientations if item.stage == "final")
        if not partials or not finals:
            raise ValueError(
                "contact decomposition requires partial and final orientations"
            )
        allowed_partial_orders = {self.pair_legs, tuple(reversed(self.pair_legs))}
        if partials[0].input_legs != self.pair_legs or any(
            item.input_legs not in allowed_partial_orders for item in partials
        ):
            raise ValueError(
                "contact partial orientations disagree with the chosen pair"
            )
        if any(
            tuple(leg for leg in item.input_legs if leg >= 0) != (self.remaining_leg,)
            for item in finals
        ):
            raise ValueError(
                "contact final orientations disagree with the remaining leg"
            )
        if self.decomposition_kind == "two-structure-constants":
            if (
                len(self.outer_color_factor) != 3
                or len(self.final_color_factor) != 3
                or self.dummy_index_mapping is None
            ):
                raise ValueError(
                    "two-structure-constant contact proof requires two factors and a "
                    "dummy mapping"
                )
            mapping = self.dummy_index_mapping
            if (
                self.outer_color_factor[mapping.outer_slot] != mapping.source_index
                or self.final_color_factor[mapping.final_slot] != mapping.source_index
            ):
                raise ValueError(
                    "contact dummy-index mapping disagrees with the chosen factors"
                )
        elif (
            self.outer_color_factor
            or self.final_color_factor
            or self.dummy_index_mapping is not None
        ):
            raise ValueError(
                "literal singlet contact proof must not contain color factors"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "decomposition_kind": self.decomposition_kind,
            "result_leg": self.result_leg,
            "pair_legs": list(self.pair_legs),
            "remaining_leg": self.remaining_leg,
            "outer_color_source": self.outer_color_source,
            "final_color_source": self.final_color_source,
            "outer_color_factor": list(self.outer_color_factor),
            "final_color_factor": list(self.final_color_factor),
            "dummy_index_mapping": (
                None
                if self.dummy_index_mapping is None
                else self.dummy_index_mapping.to_dict()
            ),
            "outer_color_normalization_power": self.outer_color_normalization_power,
            "final_color_normalization_power": self.final_color_normalization_power,
            "color_coefficient": self.color_coefficient,
            "auxiliary_color": self.auxiliary_color,
            "open_legs": list(self.open_legs),
            "component_axis_order": list(self.component_axis_order),
            "component_basis_order": list(self.component_basis_order),
            "component_expansion": [
                None if entry is None else list(entry)
                for entry in self.component_expansion
            ],
            "assignment_multiplicity": self.assignment_multiplicity,
            "canonical_outer_parity": self.canonical_outer_parity,
            "orientations": [item.to_dict() for item in self.orientations],
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
    ) -> CompiledContactDecompositionSplit:
        fields = _strict_record_fields(
            payload,
            required={
                "decomposition_kind",
                "result_leg",
                "pair_legs",
                "remaining_leg",
                "outer_color_source",
                "final_color_source",
                "outer_color_factor",
                "final_color_factor",
                "dummy_index_mapping",
                "outer_color_normalization_power",
                "final_color_normalization_power",
                "color_coefficient",
                "auxiliary_color",
                "open_legs",
                "component_axis_order",
                "component_basis_order",
                "component_expansion",
                "assignment_multiplicity",
                "canonical_outer_parity",
                "orientations",
            },
            context="compiled contact decomposition split",
        )
        dummy_payload = fields["dummy_index_mapping"]
        return cls(
            decomposition_kind=_strict_string(
                fields["decomposition_kind"], "contact decomposition kind"
            ),
            result_leg=_strict_integer(fields["result_leg"], "contact result leg"),
            pair_legs=_strict_int_pair(fields["pair_legs"], "contact pair legs"),
            remaining_leg=_strict_integer(
                fields["remaining_leg"], "contact remaining leg"
            ),
            outer_color_source=_strict_string(
                fields["outer_color_source"], "contact outer color source"
            ),
            final_color_source=_strict_string(
                fields["final_color_source"], "contact final color source"
            ),
            outer_color_factor=_strict_int_tuple(
                fields["outer_color_factor"], "contact outer color factor"
            ),
            final_color_factor=_strict_int_tuple(
                fields["final_color_factor"], "contact final color factor"
            ),
            dummy_index_mapping=(
                None
                if dummy_payload is None
                else CompiledContactDummyIndexMapping.from_dict(
                    _strict_mapping(dummy_payload, "contact dummy-index mapping")
                )
            ),
            outer_color_normalization_power=_strict_integer(
                fields["outer_color_normalization_power"],
                "contact outer color normalization power",
            ),
            final_color_normalization_power=_strict_integer(
                fields["final_color_normalization_power"],
                "contact final color normalization power",
            ),
            color_coefficient=_strict_string(
                fields["color_coefficient"], "contact color coefficient"
            ),
            auxiliary_color=_strict_integer(
                fields["auxiliary_color"], "contact auxiliary color"
            ),
            open_legs=_strict_int_pair(fields["open_legs"], "contact open legs"),
            component_axis_order=_strict_string_tuple(
                fields["component_axis_order"], "contact component axis order"
            ),
            component_basis_order=_strict_int_tuple(
                fields["component_basis_order"], "contact component basis order"
            ),
            component_expansion=_strict_optional_int_pairs(
                fields["component_expansion"], "contact component expansion"
            ),
            assignment_multiplicity=_strict_integer(
                fields["assignment_multiplicity"], "contact assignment multiplicity"
            ),
            canonical_outer_parity=_strict_integer(
                fields["canonical_outer_parity"], "contact canonical outer parity"
            ),
            orientations=tuple(
                CompiledContactOrientationProof.from_dict(item)
                for item in _strict_mappings(
                    fields["orientations"], "contact decomposition orientations"
                )
            ),
        )


@dataclass(frozen=True)
class CompiledContactDecompositionProof:
    status: str
    algorithm: str
    algorithm_version: int
    term_id: int
    vertex: str
    particles: tuple[str, ...]
    color_index: int
    lorentz_index: int
    original_color_source: str
    normalized_color_expression: str
    lorentz_name: str
    original_lorentz_source: str
    normalized_lorentz_expression: str
    splits: tuple[CompiledContactDecompositionSplit, ...] = ()
    unsupported_reasons: tuple[CompiledContactUnsupportedReason, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {"proven", "unsupported"}:
            raise ValueError(
                f"unknown contact decomposition proof status {self.status!r}"
            )
        if (
            self.algorithm != CONTACT_DECOMPOSITION_ALGORITHM
            or self.algorithm_version != CONTACT_DECOMPOSITION_ALGORITHM_VERSION
        ):
            raise ValueError(
                "unsupported contact decomposition proof algorithm: "
                f"{self.algorithm}/v{self.algorithm_version}"
            )
        if not self.vertex or not self.particles:
            raise ValueError("contact decomposition proof requires a term identity")
        if len(self.particles) != 4:
            raise ValueError("contact decomposition proof requires four particles")
        result_legs = tuple(split.result_leg for split in self.splits)
        if len(result_legs) != len(set(result_legs)):
            raise ValueError("contact decomposition proof has duplicate result legs")
        if self.status == "proven" and (not self.splits or self.unsupported_reasons):
            raise ValueError("proven contact decomposition must contain only splits")
        if self.status == "unsupported" and (
            self.splits or not self.unsupported_reasons
        ):
            raise ValueError(
                "unsupported contact decomposition must contain only reasons"
            )

    def matches(self, term: _ContactTerm) -> bool:
        return (
            self.term_id == term.id
            and self.vertex == term.vertex
            and self.particles == term.particles
            and self.color_index == term.color_index
            and self.lorentz_index == term.lorentz_index
            and self.original_color_source == term.color_source
            and self.normalized_color_expression == term.color_expression
            and self.lorentz_name == term.lorentz_name
            and self.original_lorentz_source == term.lorentz_source
            and self.normalized_lorentz_expression == term.lorentz_expression
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "algorithm": self.algorithm,
            "algorithm_version": self.algorithm_version,
            "term_identity": {
                "term_id": self.term_id,
                "vertex": self.vertex,
                "particles": list(self.particles),
                "color_index": self.color_index,
                "lorentz_index": self.lorentz_index,
                "original_color_source": self.original_color_source,
                "normalized_color_expression": self.normalized_color_expression,
                "lorentz_name": self.lorentz_name,
                "original_lorentz_source": self.original_lorentz_source,
                "normalized_lorentz_expression": self.normalized_lorentz_expression,
            },
            "splits": [split.to_dict() for split in self.splits],
            "unsupported_reasons": [
                reason.to_dict() for reason in self.unsupported_reasons
            ],
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
    ) -> CompiledContactDecompositionProof:
        fields = _strict_record_fields(
            payload,
            required={
                "status",
                "algorithm",
                "algorithm_version",
                "term_identity",
                "splits",
                "unsupported_reasons",
            },
            context="compiled contact decomposition proof",
        )
        identity = _strict_record_fields(
            _strict_mapping(fields["term_identity"], "contact term identity"),
            required={
                "term_id",
                "vertex",
                "particles",
                "color_index",
                "lorentz_index",
                "original_color_source",
                "normalized_color_expression",
                "lorentz_name",
                "original_lorentz_source",
                "normalized_lorentz_expression",
            },
            context="compiled contact term identity",
        )
        return cls(
            status=_strict_string(fields["status"], "contact proof status"),
            algorithm=_strict_string(fields["algorithm"], "contact proof algorithm"),
            algorithm_version=_strict_integer(
                fields["algorithm_version"], "contact proof algorithm version"
            ),
            term_id=_strict_integer(identity["term_id"], "contact proof term id"),
            vertex=_strict_string(identity["vertex"], "contact proof vertex"),
            particles=_strict_string_tuple(
                identity["particles"], "contact proof particles"
            ),
            color_index=_strict_integer(
                identity["color_index"], "contact proof color index"
            ),
            lorentz_index=_strict_integer(
                identity["lorentz_index"], "contact proof Lorentz index"
            ),
            original_color_source=_strict_string(
                identity["original_color_source"], "contact proof color source"
            ),
            normalized_color_expression=_strict_string(
                identity["normalized_color_expression"],
                "contact proof normalized color expression",
            ),
            lorentz_name=_strict_string(
                identity["lorentz_name"], "contact proof Lorentz name"
            ),
            original_lorentz_source=_strict_string(
                identity["original_lorentz_source"], "contact proof Lorentz source"
            ),
            normalized_lorentz_expression=_strict_string(
                identity["normalized_lorentz_expression"],
                "contact proof normalized Lorentz expression",
            ),
            splits=tuple(
                CompiledContactDecompositionSplit.from_dict(item)
                for item in _strict_mappings(fields["splits"], "contact proof splits")
            ),
            unsupported_reasons=tuple(
                CompiledContactUnsupportedReason.from_dict(item)
                for item in _strict_mappings(
                    fields["unsupported_reasons"],
                    "contact proof unsupported reasons",
                )
            ),
        )


@dataclass(frozen=True, order=True, slots=True)
class CompiledContactOrbitCertificate:
    """Compiler proof that contact paths describe one physical assignment orbit."""

    algorithm: str
    algorithm_version: int
    term_id: int
    vertex: str
    particles: tuple[str, ...]
    color_expression: str
    lorentz_expression: str
    coupling_expression: str
    evaluator_class: str
    physical_leg_equivalence_classes: tuple[int, ...]
    reconstruction_factor: str

    def __post_init__(self) -> None:
        if (
            self.algorithm != CONTACT_ORBIT_ALGORITHM
            or self.algorithm_version != CONTACT_ORBIT_ALGORITHM_VERSION
        ):
            raise ValueError(
                "unsupported contact-orbit certificate algorithm: "
                f"{self.algorithm}/v{self.algorithm_version}"
            )
        if self.term_id < 0 or not self.vertex:
            raise ValueError("contact-orbit certificate requires a term identity")
        if len(self.particles) not in {4, 5} or any(
            not particle for particle in self.particles
        ):
            raise ValueError("contact-orbit certificate particles must be non-empty")
        if self.evaluator_class not in CONTACT_ORBIT_EVALUATOR_CLASSES:
            raise ValueError(
                f"unsupported contact-orbit evaluator class {self.evaluator_class!r}"
            )
        if self.evaluator_class == CONTACT_ORBIT_EVALUATOR_CLASS and (
            len(self.particles) != 4
            or self.color_expression != "1"
            or self.lorentz_expression != "1"
        ):
            raise ValueError(
                "contact-orbit certificate requires literal scalar-singlet tensors"
            )
        if self.evaluator_class == HEFT_CONTACT_ORBIT_EVALUATOR_CLASS and (
            not self.color_expression or not self.lorentz_expression
        ):
            raise ValueError("HEFT contact-orbit tensors must be non-empty")
        if not self.coupling_expression:
            raise ValueError(
                "contact-orbit certificate coupling expression must not be empty"
            )
        classes = self.physical_leg_equivalence_classes
        if len(classes) != len(self.particles) or any(value < 0 for value in classes):
            raise ValueError("contact-orbit equivalence classes must be non-negative")
        canonical: dict[str, int] = {}
        expected = tuple(
            canonical.setdefault(particle, len(canonical))
            for particle in self.particles
        )
        if classes != expected:
            raise ValueError(
                "contact-orbit physical-leg equivalence classes are not canonical"
            )
        if self.reconstruction_factor != "1":
            raise ValueError("contact-orbit reconstruction factor must be one")

    def to_dict(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "algorithm_version": self.algorithm_version,
            "term_id": self.term_id,
            "vertex": self.vertex,
            "particles": list(self.particles),
            "color_expression": self.color_expression,
            "lorentz_expression": self.lorentz_expression,
            "coupling_expression": self.coupling_expression,
            "evaluator_class": self.evaluator_class,
            "physical_leg_equivalence_classes": list(
                self.physical_leg_equivalence_classes
            ),
            "reconstruction_factor": self.reconstruction_factor,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
    ) -> CompiledContactOrbitCertificate:
        fields = _strict_record_fields(
            payload,
            required={
                "algorithm",
                "algorithm_version",
                "term_id",
                "vertex",
                "particles",
                "color_expression",
                "lorentz_expression",
                "coupling_expression",
                "evaluator_class",
                "physical_leg_equivalence_classes",
                "reconstruction_factor",
            },
            context="compiled contact-orbit certificate",
        )
        particles = _strict_string_tuple(fields["particles"], "contact-orbit particles")
        classes = _strict_int_tuple(
            fields["physical_leg_equivalence_classes"],
            "contact-orbit physical-leg equivalence classes",
        )
        if len(particles) not in {4, 5} or len(classes) != len(particles):
            raise ValueError(
                "contact-orbit certificate must describe four or five legs"
            )
        return cls(
            algorithm=_strict_string(fields["algorithm"], "contact-orbit algorithm"),
            algorithm_version=_strict_integer(
                fields["algorithm_version"], "contact-orbit algorithm version"
            ),
            term_id=_strict_integer(fields["term_id"], "contact-orbit term ID"),
            vertex=_strict_string(fields["vertex"], "contact-orbit vertex"),
            particles=particles,
            color_expression=_strict_string(
                fields["color_expression"], "contact-orbit color expression"
            ),
            lorentz_expression=_strict_string(
                fields["lorentz_expression"], "contact-orbit Lorentz expression"
            ),
            coupling_expression=_strict_string(
                fields["coupling_expression"], "contact-orbit coupling expression"
            ),
            evaluator_class=_strict_string(
                fields["evaluator_class"], "contact-orbit evaluator class"
            ),
            physical_leg_equivalence_classes=classes,
            reconstruction_factor=_strict_string(
                fields["reconstruction_factor"],
                "contact-orbit reconstruction factor",
            ),
        )


@dataclass(frozen=True, order=True, slots=True)
class CompiledContactOrbitStep:
    """Exact physical-leg lineage for one compiler-lowered contact kernel."""

    algorithm: str
    algorithm_version: int
    term_id: int
    stage: str
    result_leg: int
    left_covered_legs: tuple[int, ...]
    right_covered_legs: tuple[int, ...]
    source_particle_legs: tuple[int, int, int]
    reconstruction_factor: str

    def __post_init__(self) -> None:
        if (
            self.algorithm != CONTACT_ORBIT_ALGORITHM
            or self.algorithm_version != CONTACT_ORBIT_ALGORITHM_VERSION
        ):
            raise ValueError(
                "unsupported contact-orbit step algorithm: "
                f"{self.algorithm}/v{self.algorithm_version}"
            )
        if self.term_id < 0 or self.stage not in {"partial", "final"}:
            raise ValueError("contact-orbit step has an invalid term or stage")
        if self.result_leg not in range(5):
            raise ValueError("contact-orbit result leg must address supported input")
        if self.reconstruction_factor != "1":
            raise ValueError("contact-orbit step reconstruction factor must be one")
        covered = (*self.left_covered_legs, *self.right_covered_legs)
        if (
            not self.left_covered_legs
            or not self.right_covered_legs
            or any(leg not in range(5) for leg in covered)
            or len(covered) != len(set(covered))
        ):
            raise ValueError("contact-orbit step has invalid covered physical legs")
        left_source, right_source, output_source = self.source_particle_legs
        if self.stage == "partial":
            if (
                len(self.left_covered_legs) != 1
                or len(self.right_covered_legs) != 1
                or (left_source, right_source)
                != (self.left_covered_legs[0], self.right_covered_legs[0])
                or output_source != -1
                or self.result_leg in covered
            ):
                raise ValueError("contact-orbit partial step lineage is inconsistent")
        else:
            arity = len(covered) + 1
            if (
                arity not in {4, 5}
                or set(covered) != (set(range(arity)) - {self.result_leg})
                or output_source != self.result_leg
                or any(
                    len(side) not in {1, 2}
                    for side in (self.left_covered_legs, self.right_covered_legs)
                )
            ):
                raise ValueError("contact-orbit final step lineage is inconsistent")
            expected_sources = tuple(
                -1 if len(side) > 1 else side[0]
                for side in (self.left_covered_legs, self.right_covered_legs)
            )
            if (left_source, right_source) != expected_sources:
                raise ValueError("contact-orbit final source lineage is inconsistent")

    def to_dict(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "algorithm_version": self.algorithm_version,
            "term_id": self.term_id,
            "stage": self.stage,
            "result_leg": self.result_leg,
            "left_covered_legs": list(self.left_covered_legs),
            "right_covered_legs": list(self.right_covered_legs),
            "source_particle_legs": list(self.source_particle_legs),
            "reconstruction_factor": self.reconstruction_factor,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> CompiledContactOrbitStep:
        fields = _strict_record_fields(
            payload,
            required={
                "algorithm",
                "algorithm_version",
                "term_id",
                "stage",
                "result_leg",
                "left_covered_legs",
                "right_covered_legs",
                "source_particle_legs",
                "reconstruction_factor",
            },
            context="compiled contact-orbit step",
        )
        source_legs = _strict_int_tuple(
            fields["source_particle_legs"], "contact-orbit step source legs"
        )
        if len(source_legs) != 3:
            raise ValueError("contact-orbit step must describe three kernel slots")
        return cls(
            algorithm=_strict_string(fields["algorithm"], "contact-orbit algorithm"),
            algorithm_version=_strict_integer(
                fields["algorithm_version"], "contact-orbit algorithm version"
            ),
            term_id=_strict_integer(fields["term_id"], "contact-orbit term ID"),
            stage=_strict_string(fields["stage"], "contact-orbit stage"),
            result_leg=_strict_integer(
                fields["result_leg"], "contact-orbit result leg"
            ),
            left_covered_legs=_strict_int_tuple(
                fields["left_covered_legs"], "contact-orbit left covered legs"
            ),
            right_covered_legs=_strict_int_tuple(
                fields["right_covered_legs"], "contact-orbit right covered legs"
            ),
            source_particle_legs=(source_legs[0], source_legs[1], source_legs[2]),
            reconstruction_factor=_strict_string(
                fields["reconstruction_factor"],
                "contact-orbit reconstruction factor",
            ),
        )


def canonical_contact_orbit_steps(
    *groups: tuple[CompiledContactOrbitStep, ...],
) -> tuple[CompiledContactOrbitStep, ...]:
    """Return the deterministic union carried by one retained compiler kernel."""

    return tuple(sorted({step for group in groups for step in group}))


def compiled_contact_orbit_step(
    certificate: CompiledContactOrbitCertificate,
    split: CompiledContactDecompositionSplit,
    orientation: CompiledContactOrientationProof,
) -> CompiledContactOrbitStep:
    """Derive the exact compiler lineage named by one proven orientation."""

    if orientation.stage == "partial":
        left_covered = (orientation.input_legs[0],)
        right_covered = (orientation.input_legs[1],)
        source_particle_legs = (
            orientation.input_legs[0],
            orientation.input_legs[1],
            -1,
        )
    else:
        auxiliary_on_left = orientation.input_legs[0] == -1
        pair_legs = tuple(sorted(split.pair_legs))
        remaining = (split.remaining_leg,)
        left_covered, right_covered = (
            (pair_legs, remaining) if auxiliary_on_left else (remaining, pair_legs)
        )
        source_particle_legs = (
            -1 if auxiliary_on_left else split.remaining_leg,
            split.remaining_leg if auxiliary_on_left else -1,
            split.result_leg,
        )
    return CompiledContactOrbitStep(
        algorithm=CONTACT_ORBIT_ALGORITHM,
        algorithm_version=CONTACT_ORBIT_ALGORITHM_VERSION,
        term_id=certificate.term_id,
        stage=orientation.stage,
        result_leg=split.result_leg,
        left_covered_legs=left_covered,
        right_covered_legs=right_covered,
        source_particle_legs=source_particle_legs,
        reconstruction_factor=certificate.reconstruction_factor,
    )


def _validate_auxiliary_color(value: int) -> None:
    if value not in {-3, 1, 3, 8}:
        raise ValueError(
            f"contact auxiliary uses unsupported UFO color representation {value}"
        )


def _strict_record_fields(
    payload: Mapping[str, object],
    *,
    required: set[str],
    context: str,
) -> Mapping[str, object]:
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


def _strict_mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    return value


def _strict_string(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{context} must be a string")
    return value


def _strict_integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
    return value


def _strict_values(value: object, context: str) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, list | tuple):
        raise TypeError(f"{context} must be an array")
    return tuple(value)


def _strict_int_tuple(value: object, context: str) -> tuple[int, ...]:
    return tuple(
        _strict_integer(item, f"{context} entry")
        for item in _strict_values(value, context)
    )


def _strict_int_pair(value: object, context: str) -> tuple[int, int]:
    result = _strict_int_tuple(value, context)
    if len(result) != 2:
        raise ValueError(f"{context} must contain two entries")
    return result[0], result[1]


def _strict_string_tuple(value: object, context: str) -> tuple[str, ...]:
    return tuple(
        _strict_string(item, f"{context} entry")
        for item in _strict_values(value, context)
    )


def _strict_optional_int_pairs(
    value: object,
    context: str,
) -> tuple[tuple[int, int] | None, ...]:
    return tuple(
        None if item is None else _strict_int_pair(item, f"{context} entry")
        for item in _strict_values(value, context)
    )


def _strict_mappings(
    value: object,
    context: str,
) -> tuple[Mapping[str, object], ...]:
    return tuple(
        _strict_mapping(item, f"{context} entry")
        for item in _strict_values(value, context)
    )


__all__ = [
    "CONTACT_DECOMPOSITION_ALGORITHM",
    "CONTACT_DECOMPOSITION_ALGORITHM_VERSION",
    "CompiledContactDecompositionProof",
    "CompiledContactDecompositionSplit",
    "CompiledContactDummyIndexMapping",
    "CompiledContactOrientationProof",
    "CompiledContactUnsupportedReason",
]

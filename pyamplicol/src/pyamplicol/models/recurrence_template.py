# SPDX-License-Identifier: 0BSD
"""Exact, model-generic semantic templates for recurrence execution.

This module defines the Python side of ``pyamplicol-recurrence-template-v1``.
It intentionally contains no model implementation and no evaluator-building
logic.  A catalog is a content-addressed statement of model semantics that a
later recurrence builder may consume only after all references and digests
have been checked.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import ClassVar, Literal, TypeAlias

RECURRENCE_TEMPLATE_ABI = "pyamplicol-recurrence-template-v1"
RECURRENCE_TEMPLATE_CANONICALIZATION_ABI = "pyamplicol-canonical-json-v1"
RECURRENCE_TEMPLATE_EXACT_SCALAR_ABI = "pyamplicol-exact-complex-rational-v1"
_DYNAMIC_UNION_SOURCE_SPIN_STATE = -(1 << 31)

ParameterKind: TypeAlias = Literal["external", "derived", "constant"]
ParameterValueType: TypeAlias = Literal["real", "complex"]
EvaluatorContractKind: TypeAlias = Literal[
    "source",
    "vertex",
    "propagator",
    "closure",
    "model-parameter",
]
EvaluatorCallableKind: TypeAlias = Literal["prepared-kernel", "rusticol-template"]
PreparedOutputFactorSource: TypeAlias = Literal[
    "none",
    "coupling-real",
    "coupling-imag",
]
FlavourFlowV1: TypeAlias = tuple[int, ...]
QuantumNumberFlowV1: TypeAlias = tuple[tuple[str, str], ...]
FlavourFlowOperationV1: TypeAlias = Literal[
    "constant-result",
    "append-left-result",
    "append-right-result",
    "concat-left-right-result",
]
QuantumNumberFlowOperationV1: TypeAlias = Literal["particle-static-result"]
LCColorShapeKindV1: TypeAlias = Literal[
    "singlet-forest",
    "fundamental-open-string",
    "antifundamental-open-string",
    "adjoint-segment",
]
LCColorComponentKindV1: TypeAlias = Literal["open-string", "adjoint-segment", "trace"]
LCColorComponentRoleV1: TypeAlias = Literal["active", "passive", "none"]
LCColorSourceSeedOperationV1: TypeAlias = Literal["empty", "singleton"]
LCColorComponentOperationV1: TypeAlias = Literal[
    "concatenate-join",
    "concatenate-keep",
    "inherit-left",
    "inherit-right",
    "empty",
    "close",
]

SUPPORTED_SYMMETRY_PROOF_ALGORITHMS = frozenset(
    {
        "canonical-crossing-bijection-v1",
        "canonical-current-word-reversal-v1",
        "canonical-kernel-input-exchange-v1",
        "canonical-model-contract-label-equivariance-v1",
        "canonical-recurrence-replay-witness-v1",
        "canonical-recurrence-union-witness-v1",
        "canonical-source-transition-dependency-shape-v1",
        "canonical-trace-amplitude-reversal-v1",
        "exact-expression-identity-v1",
        "prepared-kernel-homogeneous-complex-linear-current-v1",
        "prepared-kernel-independent-current-block-v1",
    }
)

_PARAMETER_KINDS = frozenset({"external", "derived", "constant"})
_PARAMETER_VALUE_TYPES = frozenset({"real", "complex"})
_ORIENTATIONS = frozenset({"particle", "antiparticle", "self-conjugate"})
_STATISTICS = frozenset({"boson", "fermion"})
_WAVEFUNCTION_FAMILIES = frozenset(
    {"scalar", "fermion", "vector", "spin2", "ghost", "auxiliary"}
)
_EVALUATOR_CONTRACT_KINDS = frozenset(
    {"source", "vertex", "propagator", "closure", "model-parameter"}
)
_EVALUATOR_CALLABLE_KINDS = frozenset({"prepared-kernel", "rusticol-template"})
_OUTPUT_FACTOR_SOURCES = frozenset({"none", "coupling-real", "coupling-imag"})
_FLAVOUR_FLOW_OPERATIONS = frozenset(
    {
        "constant-result",
        "append-left-result",
        "append-right-result",
        "concat-left-right-result",
    }
)
_QUANTUM_NUMBER_FLOW_OPERATIONS = frozenset({"particle-static-result"})
_LC_COLOR_SHAPE_KINDS = frozenset(
    {
        "singlet-forest",
        "fundamental-open-string",
        "antifundamental-open-string",
        "adjoint-segment",
    }
)
_LC_COLOR_COMPONENT_KINDS = frozenset({"open-string", "adjoint-segment", "trace"})
_LC_COLOR_COMPONENT_ROLES = frozenset({"active", "passive", "none"})
_LC_COLOR_SOURCE_SEED_OPERATIONS = frozenset({"empty", "singleton"})
_LC_COLOR_COMPONENT_OPERATIONS = frozenset(
    {
        "concatenate-join",
        "concatenate-keep",
        "inherit-left",
        "inherit-right",
        "empty",
        "close",
    }
)
_HEX = frozenset("0123456789abcdef")


class RecurrenceTemplateError(ValueError):
    """Raised when a recurrence semantic catalog is not canonical or complete."""


def _validate_rusticol_runtime_template(
    contract_kind: EvaluatorContractKind,
    runtime_template: str,
    callable_signature: str,
) -> None:
    suffix = f":{callable_signature[:24]}"
    if contract_kind == "source":
        prefix = "rusticol.source-fill."
        version = ".v1"
        if not runtime_template.startswith(prefix) or not runtime_template.endswith(
            version + suffix
        ):
            raise RecurrenceTemplateError(
                "source Rusticol template must use the authenticated "
                "rusticol.source-fill.<family>.v1:<signature> contract"
            )
        family = runtime_template[len(prefix) : -len(version + suffix)]
        if family not in _WAVEFUNCTION_FAMILIES:
            raise RecurrenceTemplateError(
                f"unsupported Rusticol source-fill family {family!r}"
            )
        return
    if contract_kind == "closure":
        expected = f"rusticol.closure-reduce.v1{suffix}"
        if runtime_template != expected:
            raise RecurrenceTemplateError(
                "closure Rusticol template does not match its callable signature"
            )
        return
    raise RecurrenceTemplateError(
        f"Rusticol template callable is not registered for {contract_kind!r}"
    )


def _canonical_json(payload: object) -> str:
    try:
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise RecurrenceTemplateError(
            "recurrence template payload is not canonical JSON"
        ) from exc


def _digest(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()


def _require_sha256(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise RecurrenceTemplateError(f"{name} must be a lowercase SHA-256")
    return value


def _require_optional_sha256(name: str, value: object) -> str | None:
    if value is None:
        return None
    return _require_sha256(name, value)


def _require_nonempty(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise RecurrenceTemplateError(f"{name} must be a nonempty string")
    return value


def _require_int(name: str, value: object, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise RecurrenceTemplateError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise RecurrenceTemplateError(f"{name} must be at least {minimum}")
    return value


def _require_tuple(name: str, value: object) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        raise RecurrenceTemplateError(f"{name} must be an immutable tuple")
    return value


def _require_string_tuple(
    name: str,
    value: object,
    *,
    nonempty: bool = False,
    sorted_unique: bool = False,
) -> tuple[str, ...]:
    items = _require_tuple(name, value)
    if any(not isinstance(item, str) or not item for item in items):
        raise RecurrenceTemplateError(f"{name} must contain only nonempty strings")
    result = tuple(items)
    if nonempty and not result:
        raise RecurrenceTemplateError(f"{name} must not be empty")
    if sorted_unique and result != tuple(sorted(set(result))):
        raise RecurrenceTemplateError(f"{name} must be sorted and unique")
    return result


def _require_int_tuple(
    name: str,
    value: object,
    *,
    nonempty: bool = False,
) -> tuple[int, ...]:
    items = _require_tuple(name, value)
    if any(type(item) is not int for item in items):
        raise RecurrenceTemplateError(f"{name} must contain only integers")
    result = tuple(items)
    if nonempty and not result:
        raise RecurrenceTemplateError(f"{name} must not be empty")
    return result


def _require_concrete_spin_state(name: str, value: object) -> int:
    result = _require_int(name, value)
    if result == _DYNAMIC_UNION_SOURCE_SPIN_STATE:
        raise RecurrenceTemplateError(
            f"{name} uses the reserved dynamic-union sentinel"
        )
    return result


def _require_concrete_spin_states(
    name: str,
    value: object,
) -> tuple[int, ...]:
    result = _require_int_tuple(name, value)
    if _DYNAMIC_UNION_SOURCE_SPIN_STATE in result:
        raise RecurrenceTemplateError(
            f"{name} uses the reserved dynamic-union sentinel"
        )
    return result


def _require_permutation(name: str, value: object, arity: int) -> tuple[int, ...]:
    result = _require_int_tuple(name, value)
    if result != tuple(sorted(result)) and set(result) != set(range(arity)):
        raise RecurrenceTemplateError(f"{name} is not a permutation of its inputs")
    if len(result) != arity or set(result) != set(range(arity)):
        raise RecurrenceTemplateError(f"{name} is not a permutation of its inputs")
    return result


def _require_exact_keys(
    name: str,
    payload: Mapping[str, object],
    expected: frozenset[str],
) -> None:
    actual = frozenset(payload)
    if actual != expected:
        unknown = sorted(actual - expected)
        missing = sorted(expected - actual)
        details: list[str] = []
        if unknown:
            details.append("unknown=" + ",".join(unknown))
        if missing:
            details.append("missing=" + ",".join(missing))
        raise RecurrenceTemplateError(
            f"{name} has noncanonical fields ({'; '.join(details)})"
        )


def _require_mapping(name: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise RecurrenceTemplateError(f"{name} must be a string-keyed object")
    return value


def _decode_string_tuple(name: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise RecurrenceTemplateError(f"{name} must be a JSON string array")
    return tuple(value)


def _decode_int_tuple(name: str, value: object) -> tuple[int, ...]:
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise RecurrenceTemplateError(f"{name} must be a JSON integer array")
    return tuple(value)


def _decode_string_pairs(
    name: str,
    value: object,
) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list):
        raise RecurrenceTemplateError(f"{name} must be a JSON array")
    result: list[tuple[str, str]] = []
    for index, item in enumerate(value):
        if (
            not isinstance(item, list)
            or len(item) != 2
            or any(not isinstance(part, str) or not part for part in item)
        ):
            raise RecurrenceTemplateError(
                f"{name} row {index} must contain two nonempty strings"
            )
        result.append((item[0], item[1]))
    return tuple(result)


def _require_flavour_flow(name: str, value: object) -> FlavourFlowV1:
    result = _require_int_tuple(name, value, nonempty=True)
    return tuple(result)


def _require_quantum_number_flow(
    name: str,
    value: object,
) -> QuantumNumberFlowV1:
    items = _require_tuple(name, value)
    result: list[tuple[str, str]] = []
    for item in items:
        if not isinstance(item, tuple) or len(item) != 2:
            raise RecurrenceTemplateError(
                f"{name} must contain immutable (name, expression) pairs"
            )
        quantum_name = _require_nonempty(f"{name} name", item[0])
        expression = _require_nonempty(f"{name} expression", item[1])
        result.append((quantum_name, expression))
    normalized = tuple(result)
    names = tuple(item[0] for item in normalized)
    if names != tuple(sorted(set(names))):
        raise RecurrenceTemplateError(f"{name} names must be sorted and unique")
    return normalized


def _decode_flavour_flow(name: str, value: object) -> FlavourFlowV1:
    if not isinstance(value, list):
        raise RecurrenceTemplateError(f"{name} must be a JSON integer array")
    return _require_flavour_flow(name, tuple(value))


def _decode_flavour_flows(
    name: str,
    value: object,
) -> tuple[FlavourFlowV1, ...]:
    if not isinstance(value, list):
        raise RecurrenceTemplateError(f"{name} must be a JSON array")
    return tuple(
        _decode_flavour_flow(f"{name}[{index}]", item)
        for index, item in enumerate(value)
    )


def _decode_quantum_number_flow(
    name: str,
    value: object,
) -> QuantumNumberFlowV1:
    if not isinstance(value, list):
        raise RecurrenceTemplateError(f"{name} must be a JSON pair array")
    pairs: list[tuple[object, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, list) or len(item) != 2:
            raise RecurrenceTemplateError(
                f"{name}[{index}] must be a JSON [name, expression] pair"
            )
        pairs.append((item[0], item[1]))
    return _require_quantum_number_flow(name, tuple(pairs))


def _decode_quantum_number_flows(
    name: str,
    value: object,
) -> tuple[QuantumNumberFlowV1, ...]:
    if not isinstance(value, list):
        raise RecurrenceTemplateError(f"{name} must be a JSON array")
    return tuple(
        _decode_quantum_number_flow(f"{name}[{index}]", item)
        for index, item in enumerate(value)
    )


def _decode_optional_string(name: str, value: object) -> str | None:
    if value is None:
        return None
    return _require_nonempty(name, value)


def _canonical_decimal_integer(name: str, value: object) -> int:
    if not isinstance(value, str):
        raise RecurrenceTemplateError(f"{name} must be a decimal integer string")
    if value == "0":
        return 0
    negative = value.startswith("-")
    digits = value[1:] if negative else value
    if not digits or not digits.isascii() or not digits.isdigit():
        raise RecurrenceTemplateError(f"{name} is not a decimal integer")
    if digits.startswith("0"):
        raise RecurrenceTemplateError(f"{name} is not canonically encoded")
    parsed = int(value, 10)
    if str(parsed) != value:
        raise RecurrenceTemplateError(f"{name} is not canonically encoded")
    return parsed


@dataclass(frozen=True, order=True, slots=True)
class ExactComplexRationalV1:
    """One exact complex number represented by two reduced fractions."""

    real_numerator: int
    real_denominator: int
    imag_numerator: int
    imag_denominator: int

    def __post_init__(self) -> None:
        for name in ("real_numerator", "imag_numerator"):
            _require_int(name, getattr(self, name))
        for name in ("real_denominator", "imag_denominator"):
            _require_int(name, getattr(self, name), minimum=1)
        for component in ("real", "imag"):
            numerator = getattr(self, f"{component}_numerator")
            denominator = getattr(self, f"{component}_denominator")
            if numerator == 0 and denominator != 1:
                raise RecurrenceTemplateError(
                    f"{component} zero must be encoded as 0/1"
                )
            if math.gcd(abs(numerator), denominator) != 1:
                raise RecurrenceTemplateError(f"{component} fraction must be reduced")

    @classmethod
    def zero(cls) -> ExactComplexRationalV1:
        return cls(0, 1, 0, 1)

    @classmethod
    def one(cls) -> ExactComplexRationalV1:
        return cls(1, 1, 0, 1)

    @classmethod
    def from_fractions(
        cls,
        real: Fraction | int,
        imag: Fraction | int = 0,
    ) -> ExactComplexRationalV1:
        real_fraction = Fraction(real)
        imag_fraction = Fraction(imag)
        return cls(
            real_fraction.numerator,
            real_fraction.denominator,
            imag_fraction.numerator,
            imag_fraction.denominator,
        )

    @classmethod
    def from_binary64(
        cls,
        real: float,
        imag: float = 0.0,
    ) -> ExactComplexRationalV1:
        """Convert binary64 components without decimal or rounded arithmetic."""

        if type(real) is not float or type(imag) is not float:
            raise RecurrenceTemplateError(
                "binary64 conversion requires float components"
            )
        if not math.isfinite(real) or not math.isfinite(imag):
            raise RecurrenceTemplateError("binary64 proof coefficients must be finite")
        real_numerator, real_denominator = real.as_integer_ratio()
        imag_numerator, imag_denominator = imag.as_integer_ratio()
        return cls(
            real_numerator,
            real_denominator,
            imag_numerator,
            imag_denominator,
        )

    @classmethod
    def from_complex_binary64(cls, value: complex) -> ExactComplexRationalV1:
        if type(value) is not complex:
            raise RecurrenceTemplateError(
                "complex binary64 conversion requires a complex value"
            )
        return cls.from_binary64(float(value.real), float(value.imag))

    @property
    def real(self) -> Fraction:
        return Fraction(self.real_numerator, self.real_denominator)

    @property
    def imag(self) -> Fraction:
        return Fraction(self.imag_numerator, self.imag_denominator)

    def to_dict(self) -> dict[str, str]:
        return {
            "imag_denominator": str(self.imag_denominator),
            "imag_numerator": str(self.imag_numerator),
            "real_denominator": str(self.real_denominator),
            "real_numerator": str(self.real_numerator),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ExactComplexRationalV1:
        value = _require_mapping("exact complex rational", payload)
        _require_exact_keys(
            "exact complex rational",
            value,
            frozenset(
                {
                    "real_numerator",
                    "real_denominator",
                    "imag_numerator",
                    "imag_denominator",
                }
            ),
        )
        return cls(
            _canonical_decimal_integer("real_numerator", value["real_numerator"]),
            _canonical_decimal_integer("real_denominator", value["real_denominator"]),
            _canonical_decimal_integer("imag_numerator", value["imag_numerator"]),
            _canonical_decimal_integer("imag_denominator", value["imag_denominator"]),
        )


class _SemanticRecord:
    _record_kind: ClassVar[str]
    semantic_digest: str

    def _semantic_fields(self) -> dict[str, object]:
        raise NotImplementedError

    def canonical_payload(self) -> dict[str, object]:
        return {"record_kind": self._record_kind, **self._semantic_fields()}

    @property
    def expected_semantic_digest(self) -> str:
        return _digest(self.canonical_payload())

    def _finish_semantic_record(self) -> None:
        expected = self.expected_semantic_digest
        if self.semantic_digest:
            _require_sha256("semantic_digest", self.semantic_digest)
            if self.semantic_digest != expected:
                raise RecurrenceTemplateError(
                    f"stale semantic digest for {self._record_kind}"
                )
        else:
            object.__setattr__(self, "semantic_digest", expected)

    def to_dict(self) -> dict[str, object]:
        return {**self.canonical_payload(), "semantic_digest": self.semantic_digest}


@dataclass(frozen=True, slots=True)
class ContactOrbitCertificateV1(_SemanticRecord):
    _record_kind: ClassVar[str] = "contact-orbit-certificate"

    template_id: str
    algorithm: str
    algorithm_version: int
    term_id: int
    vertex: str
    particles: tuple[str, ...]
    evaluator_class: str
    physical_leg_equivalence_classes: tuple[int, ...]
    reconstruction_factor: ExactComplexRationalV1
    semantic_digest: str = ""

    def __post_init__(self) -> None:
        _require_nonempty("contact-orbit certificate template_id", self.template_id)
        _require_nonempty("contact-orbit certificate algorithm", self.algorithm)
        _require_int(
            "contact-orbit certificate algorithm version",
            self.algorithm_version,
            minimum=1,
        )
        _require_int("contact-orbit certificate term ID", self.term_id, minimum=0)
        _require_nonempty("contact-orbit certificate vertex", self.vertex)
        _require_string_tuple(
            "contact-orbit certificate particles", self.particles, nonempty=True
        )
        if len(self.particles) not in {4, 5}:
            raise RecurrenceTemplateError(
                "contact-orbit certificate must describe four or five physical legs"
            )
        _require_nonempty(
            "contact-orbit certificate evaluator class", self.evaluator_class
        )
        classes = self.physical_leg_equivalence_classes
        if (
            len(classes) != len(self.particles)
            or any(type(value) is not int or value < 0 for value in classes)
        ):
            raise RecurrenceTemplateError(
                "contact-orbit certificate equivalence classes are invalid"
            )
        canonical: dict[str, int] = {}
        expected = tuple(
            canonical.setdefault(particle, len(canonical))
            for particle in self.particles
        )
        if classes != expected:
            raise RecurrenceTemplateError(
                "contact-orbit certificate equivalence classes are not canonical"
            )
        if self.reconstruction_factor != ExactComplexRationalV1.one():
            raise RecurrenceTemplateError(
                "contact-orbit reconstruction factor must be exactly one"
            )
        self._finish_semantic_record()

    def _semantic_fields(self) -> dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "algorithm_version": self.algorithm_version,
            "evaluator_class": self.evaluator_class,
            "particles": list(self.particles),
            "physical_leg_equivalence_classes": list(
                self.physical_leg_equivalence_classes
            ),
            "reconstruction_factor": self.reconstruction_factor.to_dict(),
            "template_id": self.template_id,
            "term_id": self.term_id,
            "vertex": self.vertex,
        }


@dataclass(frozen=True, slots=True)
class ContactOrbitStepV1(_SemanticRecord):
    _record_kind: ClassVar[str] = "contact-orbit-step"

    template_id: str
    certificate_template_id: str
    stage: str
    result_leg: int
    left_covered_legs: tuple[int, ...]
    right_covered_legs: tuple[int, ...]
    source_particle_legs: tuple[int, int, int]
    reconstruction_factor: ExactComplexRationalV1
    semantic_digest: str = ""

    def __post_init__(self) -> None:
        _require_nonempty("contact-orbit step template_id", self.template_id)
        _require_nonempty(
            "contact-orbit step certificate", self.certificate_template_id
        )
        if self.stage not in {"partial", "final"}:
            raise RecurrenceTemplateError(
                f"unsupported contact-orbit step stage {self.stage!r}"
            )
        _require_int("contact-orbit step result leg", self.result_leg, minimum=0)
        if self.result_leg >= 5:
            raise RecurrenceTemplateError("contact-orbit result leg is outside arity")
        for name, values in (
            ("left", self.left_covered_legs),
            ("right", self.right_covered_legs),
        ):
            if (
                not isinstance(values, tuple)
                or not values
                or any(
                    type(value) is not int or value not in range(5)
                    for value in values
                )
            ):
                raise RecurrenceTemplateError(
                    f"contact-orbit {name} covered legs are invalid"
                )
        if set(self.left_covered_legs) & set(self.right_covered_legs):
            raise RecurrenceTemplateError(
                "contact-orbit left and right covered legs must be disjoint"
            )
        if len(self.source_particle_legs) != 3 or any(
            type(value) is not int or value not in {-1, 0, 1, 2, 3, 4}
            for value in self.source_particle_legs
        ):
            raise RecurrenceTemplateError(
                "contact-orbit source-particle legs are invalid"
            )
        if self.reconstruction_factor != ExactComplexRationalV1.one():
            raise RecurrenceTemplateError(
                "contact-orbit step reconstruction factor must be exactly one"
            )
        covered = (*self.left_covered_legs, *self.right_covered_legs)
        if len(covered) != len(set(covered)):
            raise RecurrenceTemplateError(
                "contact-orbit covered legs must be unique"
            )
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
                raise RecurrenceTemplateError(
                    "contact-orbit partial step lineage is inconsistent"
                )
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
                raise RecurrenceTemplateError(
                    "contact-orbit final step lineage is inconsistent"
                )
            expected_sources = tuple(
                -1 if len(side) > 1 else side[0]
                for side in (self.left_covered_legs, self.right_covered_legs)
            )
            if (left_source, right_source) != expected_sources:
                raise RecurrenceTemplateError(
                    "contact-orbit final source lineage is inconsistent"
                )
        self._finish_semantic_record()

    def _semantic_fields(self) -> dict[str, object]:
        return {
            "certificate_template_id": self.certificate_template_id,
            "left_covered_legs": list(self.left_covered_legs),
            "reconstruction_factor": self.reconstruction_factor.to_dict(),
            "result_leg": self.result_leg,
            "right_covered_legs": list(self.right_covered_legs),
            "source_particle_legs": list(self.source_particle_legs),
            "stage": self.stage,
            "template_id": self.template_id,
        }


@dataclass(frozen=True, slots=True)
class ParameterTemplateV1(_SemanticRecord):
    _record_kind: ClassVar[str] = "parameter"

    template_id: str
    name: str
    parameter_kind: ParameterKind
    value_type: ParameterValueType
    mutable: bool
    default_value: ExactComplexRationalV1 | None
    exact_expression_digest: str | None
    dependency_parameter_ids: tuple[str, ...]
    prepared_parameter_id: int | None = None
    semantic_digest: str = ""

    def __post_init__(self) -> None:
        _require_nonempty("parameter template_id", self.template_id)
        _require_nonempty("parameter name", self.name)
        if self.parameter_kind not in _PARAMETER_KINDS:
            raise RecurrenceTemplateError(
                f"unsupported parameter kind {self.parameter_kind!r}"
            )
        if self.value_type not in _PARAMETER_VALUE_TYPES:
            raise RecurrenceTemplateError(
                f"unsupported parameter value type {self.value_type!r}"
            )
        if type(self.mutable) is not bool:
            raise RecurrenceTemplateError("parameter mutable flag must be boolean")
        _require_optional_sha256(
            "parameter exact_expression_digest", self.exact_expression_digest
        )
        _require_string_tuple(
            "parameter dependencies",
            self.dependency_parameter_ids,
            sorted_unique=True,
        )
        if self.prepared_parameter_id is not None:
            _require_int(
                "prepared parameter ID",
                self.prepared_parameter_id,
                minimum=0,
            )
        if self.parameter_kind == "derived":
            if self.exact_expression_digest is None:
                raise RecurrenceTemplateError(
                    "derived parameters require an exact expression digest"
                )
        elif self.default_value is None:
            raise RecurrenceTemplateError(
                "external and constant parameters require an exact default"
            )
        if (
            self.value_type == "real"
            and self.default_value is not None
            and self.default_value.imag_numerator != 0
        ):
            raise RecurrenceTemplateError(
                "real parameter defaults cannot have an imaginary component"
            )
        self._finish_semantic_record()

    def _semantic_fields(self) -> dict[str, object]:
        return {
            "default_value": (
                None if self.default_value is None else self.default_value.to_dict()
            ),
            "dependency_parameter_ids": list(self.dependency_parameter_ids),
            "exact_expression_digest": self.exact_expression_digest,
            "mutable": self.mutable,
            "name": self.name,
            "parameter_kind": self.parameter_kind,
            "prepared_parameter_id": self.prepared_parameter_id,
            "template_id": self.template_id,
            "value_type": self.value_type,
        }


@dataclass(frozen=True, slots=True)
class CurrentStateTemplateV1(_SemanticRecord):
    _record_kind: ClassVar[str] = "current-state"

    template_id: str
    particle_id: int
    anti_particle_id: int
    species_id: str
    orientation: str
    statistics: str
    color_representation: int
    basis: str
    tensor_ordering: tuple[str, ...]
    dimension: int
    chirality: int
    lc_color_shape_kind: LCColorShapeKindV1
    auxiliary_kind: str | None
    mass_parameter_id: str | None
    width_parameter_id: str | None
    semantic_digest: str = ""

    def __post_init__(self) -> None:
        _require_nonempty("current-state template_id", self.template_id)
        _require_int("particle_id", self.particle_id)
        _require_int("anti_particle_id", self.anti_particle_id)
        _require_nonempty("species_id", self.species_id)
        if self.orientation not in _ORIENTATIONS:
            raise RecurrenceTemplateError(
                f"unsupported current orientation {self.orientation!r}"
            )
        if self.statistics not in _STATISTICS:
            raise RecurrenceTemplateError(
                f"unsupported current statistics {self.statistics!r}"
            )
        _require_int("color_representation", self.color_representation)
        _require_nonempty("current basis", self.basis)
        _require_string_tuple(
            "current tensor_ordering", self.tensor_ordering, nonempty=True
        )
        _require_int("current dimension", self.dimension, minimum=1)
        if len(self.tensor_ordering) != self.dimension:
            raise RecurrenceTemplateError(
                "current tensor ordering must name every output component"
            )
        _require_int("current chirality", self.chirality)
        if self.lc_color_shape_kind not in _LC_COLOR_SHAPE_KINDS:
            raise RecurrenceTemplateError(
                f"unsupported current LC color shape {self.lc_color_shape_kind!r}"
            )
        if self.auxiliary_kind is not None:
            _require_nonempty("current auxiliary_kind", self.auxiliary_kind)
        if self.mass_parameter_id is not None:
            _require_nonempty("current mass_parameter_id", self.mass_parameter_id)
        if self.width_parameter_id is not None:
            _require_nonempty("current width_parameter_id", self.width_parameter_id)
        self._finish_semantic_record()

    def _semantic_fields(self) -> dict[str, object]:
        return {
            "anti_particle_id": self.anti_particle_id,
            "auxiliary_kind": self.auxiliary_kind,
            "basis": self.basis,
            "chirality": self.chirality,
            "lc_color_shape_kind": self.lc_color_shape_kind,
            "color_representation": self.color_representation,
            "dimension": self.dimension,
            "mass_parameter_id": self.mass_parameter_id,
            "orientation": self.orientation,
            "particle_id": self.particle_id,
            "species_id": self.species_id,
            "statistics": self.statistics,
            "template_id": self.template_id,
            "tensor_ordering": list(self.tensor_ordering),
            "width_parameter_id": self.width_parameter_id,
        }


@dataclass(frozen=True, slots=True)
class LCColorSourceSeedV1:
    """Authenticated LC color forest seeded by one source template."""

    operation: LCColorSourceSeedOperationV1
    output_shape_kind: LCColorShapeKindV1
    component_kind: LCColorComponentKindV1 | None
    component_role: LCColorComponentRoleV1
    proof_digest: str
    provenance: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.operation not in _LC_COLOR_SOURCE_SEED_OPERATIONS:
            raise RecurrenceTemplateError(
                f"unsupported LC source seed operation {self.operation!r}"
            )
        if self.output_shape_kind not in _LC_COLOR_SHAPE_KINDS:
            raise RecurrenceTemplateError(
                f"unsupported LC source seed shape {self.output_shape_kind!r}"
            )
        if self.component_role not in _LC_COLOR_COMPONENT_ROLES:
            raise RecurrenceTemplateError(
                f"unsupported LC source seed role {self.component_role!r}"
            )
        if self.operation == "empty":
            if self.component_kind is not None or self.component_role != "none":
                raise RecurrenceTemplateError(
                    "an empty LC source seed cannot declare a component"
                )
        elif (
            self.component_kind not in _LC_COLOR_COMPONENT_KINDS
            or self.component_role == "none"
        ):
            raise RecurrenceTemplateError(
                "a singleton LC source seed requires component kind and role"
            )
        expected = {
            "singlet-forest": ("empty", None, "none"),
            "fundamental-open-string": ("singleton", "open-string", "active"),
            "antifundamental-open-string": (
                "singleton",
                "open-string",
                "active",
            ),
            "adjoint-segment": ("singleton", "adjoint-segment", "active"),
        }[self.output_shape_kind]
        if (self.operation, self.component_kind, self.component_role) != expected:
            raise RecurrenceTemplateError(
                "LC source seed does not match its certified output shape"
            )
        _require_sha256("LC source seed proof digest", self.proof_digest)
        if self.provenance != tuple(sorted(set(self.provenance))):
            raise RecurrenceTemplateError(
                "LC source seed provenance must be sorted and unique"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "component_kind": self.component_kind,
            "component_role": self.component_role,
            "operation": self.operation,
            "output_shape_kind": self.output_shape_kind,
            "proof_digest": self.proof_digest,
            "provenance": [list(item) for item in self.provenance],
        }


@dataclass(frozen=True, slots=True)
class SourceTemplateV1(_SemanticRecord):
    _record_kind: ClassVar[str] = "source"

    template_id: str
    state_template_id: str
    crossing: str
    wavefunction_family: str
    helicity: int
    spin_state: int
    flavour_flow: FlavourFlowV1
    quantum_number_flow: QuantumNumberFlowV1
    lc_color_seed: LCColorSourceSeedV1
    wavefunction_expression_digest: str
    evaluator_resolver_key: str
    mass_parameter_id: str | None = None
    width_parameter_id: str | None = None
    semantic_digest: str = ""

    def __post_init__(self) -> None:
        _require_nonempty("source template_id", self.template_id)
        _require_nonempty("source state_template_id", self.state_template_id)
        _require_nonempty("source crossing", self.crossing)
        if self.wavefunction_family not in _WAVEFUNCTION_FAMILIES:
            raise RecurrenceTemplateError(
                f"unsupported source wavefunction family {self.wavefunction_family!r}"
            )
        _require_int("source helicity", self.helicity)
        _require_concrete_spin_state("source spin_state", self.spin_state)
        _require_flavour_flow("source flavour flow", self.flavour_flow)
        _require_quantum_number_flow(
            "source quantum-number flow", self.quantum_number_flow
        )
        if not isinstance(self.lc_color_seed, LCColorSourceSeedV1):
            raise RecurrenceTemplateError(
                "source template requires an authenticated LC color seed"
            )
        _require_sha256(
            "source wavefunction_expression_digest",
            self.wavefunction_expression_digest,
        )
        _require_nonempty("source evaluator_resolver_key", self.evaluator_resolver_key)
        if self.mass_parameter_id is not None:
            _require_nonempty("source mass_parameter_id", self.mass_parameter_id)
        if self.width_parameter_id is not None:
            _require_nonempty("source width_parameter_id", self.width_parameter_id)
        self._finish_semantic_record()

    def _semantic_fields(self) -> dict[str, object]:
        return {
            "crossing": self.crossing,
            "evaluator_resolver_key": self.evaluator_resolver_key,
            "helicity": self.helicity,
            "flavour_flow": list(self.flavour_flow),
            "mass_parameter_id": self.mass_parameter_id,
            "quantum_number_flow": [list(item) for item in self.quantum_number_flow],
            "lc_color_seed": self.lc_color_seed.to_dict(),
            "spin_state": self.spin_state,
            "state_template_id": self.state_template_id,
            "template_id": self.template_id,
            "wavefunction_family": self.wavefunction_family,
            "wavefunction_expression_digest": self.wavefunction_expression_digest,
            "width_parameter_id": self.width_parameter_id,
        }


@dataclass(frozen=True, slots=True)
class QuantumFlowTemplateV1(_SemanticRecord):
    _record_kind: ClassVar[str] = "quantum-flow"

    template_id: str
    input_state_template_ids: tuple[str, ...]
    input_spin_states: tuple[int, ...]
    input_flavour_flows: tuple[FlavourFlowV1, ...]
    input_quantum_number_flows: tuple[QuantumNumberFlowV1, ...]
    flavour_flow_operation: FlavourFlowOperationV1
    quantum_number_flow_operation: QuantumNumberFlowOperationV1
    coupling_orders: tuple[tuple[str, int], ...]
    result_state_template_id: str
    result_spin_state: int
    result_flavour_flow: FlavourFlowV1
    result_quantum_number_flow: QuantumNumberFlowV1
    exact_coupling: ExactComplexRationalV1
    predicate_digest: str
    semantic_digest: str = ""

    def __post_init__(self) -> None:
        _require_nonempty("quantum-flow template_id", self.template_id)
        _require_string_tuple(
            "quantum-flow input states",
            self.input_state_template_ids,
            nonempty=True,
        )
        _require_concrete_spin_states(
            "quantum-flow spin states",
            self.input_spin_states,
        )
        flavour_flows = _require_tuple(
            "quantum-flow flavour flows", self.input_flavour_flows
        )
        for index, flow in enumerate(flavour_flows):
            _require_flavour_flow(f"quantum-flow flavour flow {index}", flow)
        quantum_number_flows = _require_tuple(
            "quantum-flow quantum-number flows",
            self.input_quantum_number_flows,
        )
        for index, flow in enumerate(quantum_number_flows):
            _require_quantum_number_flow(
                f"quantum-flow quantum-number flow {index}", flow
            )
        if self.flavour_flow_operation not in _FLAVOUR_FLOW_OPERATIONS:
            raise RecurrenceTemplateError(
                "unsupported quantum-flow flavour operation "
                f"{self.flavour_flow_operation!r}"
            )
        if self.quantum_number_flow_operation not in _QUANTUM_NUMBER_FLOW_OPERATIONS:
            raise RecurrenceTemplateError(
                "unsupported quantum-flow quantum-number operation "
                f"{self.quantum_number_flow_operation!r}"
            )
        arity = len(self.input_state_template_ids)
        if arity != 2:
            raise RecurrenceTemplateError(
                "recurrence quantum-flow templates must have binary input arity"
            )
        if not all(
            len(values) == arity
            for values in (
                self.input_spin_states,
                self.input_flavour_flows,
                self.input_quantum_number_flows,
            )
        ):
            raise RecurrenceTemplateError(
                "quantum-flow input contracts must have equal arity"
            )
        _validate_coupling_orders(self.coupling_orders)
        _require_nonempty("quantum-flow result state", self.result_state_template_id)
        _require_concrete_spin_state(
            "quantum-flow result spin state",
            self.result_spin_state,
        )
        _require_flavour_flow(
            "quantum-flow result flavour flow", self.result_flavour_flow
        )
        _require_quantum_number_flow(
            "quantum-flow result quantum-number flow",
            self.result_quantum_number_flow,
        )
        if not isinstance(self.exact_coupling, ExactComplexRationalV1):
            raise RecurrenceTemplateError(
                "quantum-flow coupling must be an exact complex rational"
            )
        _require_sha256("quantum-flow predicate_digest", self.predicate_digest)
        self._finish_semantic_record()

    def _semantic_fields(self) -> dict[str, object]:
        return {
            "coupling_orders": [list(item) for item in self.coupling_orders],
            "exact_coupling": self.exact_coupling.to_dict(),
            "flavour_flow_operation": self.flavour_flow_operation,
            "input_flavour_flows": [list(flow) for flow in self.input_flavour_flows],
            "input_quantum_number_flows": [
                [list(item) for item in flow]
                for flow in self.input_quantum_number_flows
            ],
            "input_spin_states": list(self.input_spin_states),
            "input_state_template_ids": list(self.input_state_template_ids),
            "predicate_digest": self.predicate_digest,
            "quantum_number_flow_operation": self.quantum_number_flow_operation,
            "result_flavour_flow": list(self.result_flavour_flow),
            "result_quantum_number_flow": [
                list(item) for item in self.result_quantum_number_flow
            ],
            "result_state_template_id": self.result_state_template_id,
            "result_spin_state": self.result_spin_state,
            "template_id": self.template_id,
        }


def _apply_flavour_flow_operation(
    operation: FlavourFlowOperationV1,
    left: FlavourFlowV1,
    right: FlavourFlowV1,
    result_particle: int,
) -> FlavourFlowV1:
    if operation == "constant-result":
        return (result_particle,)
    if operation == "append-left-result":
        return (
            left if left and left[-1] == result_particle else (*left, result_particle)
        )
    if operation == "append-right-result":
        return (
            right
            if right and right[-1] == result_particle
            else (*right, result_particle)
        )
    if operation == "concat-left-right-result":
        return (*left, *right, result_particle)
    raise AssertionError(f"unknown flavour-flow operation {operation!r}")


def _validate_coupling_orders(value: object) -> tuple[tuple[str, int], ...]:
    items = _require_tuple("coupling orders", value)
    normalized: list[tuple[str, int]] = []
    for item in items:
        if not isinstance(item, tuple) or len(item) != 2:
            raise RecurrenceTemplateError(
                "coupling orders must contain (name, power) tuples"
            )
        name = _require_nonempty("coupling-order name", item[0])
        power = _require_int("coupling-order power", item[1], minimum=0)
        normalized.append((name, power))
    result = tuple(normalized)
    if result != tuple(sorted(result)) or len({name for name, _ in result}) != len(
        result
    ):
        raise RecurrenceTemplateError("coupling orders must be sorted and unique")
    return result


@dataclass(frozen=True, slots=True)
class TransitionTemplateV1(_SemanticRecord):
    _record_kind: ClassVar[str] = "transition"

    template_id: str
    input_state_template_ids: tuple[str, ...]
    result_state_template_id: str
    quantum_flow_template_id: str
    evaluator_resolver_key: str
    canonical_input_order: tuple[int, ...]
    momentum_convention: tuple[str, ...]
    coupling_parameter_ids: tuple[str, ...]
    coupling_orders: tuple[tuple[str, int], ...]
    color_contraction_template_id: str
    binding_coupling: ExactComplexRationalV1
    exact_factor: ExactComplexRationalV1
    output_factor_source: PreparedOutputFactorSource
    equivalence_class: str
    input_exchange_factor: ExactComplexRationalV1 | None
    output_projection: str
    contact_orbit_step_template_ids: tuple[str, ...] = ()
    contact_orbit_step_semantic_digests: tuple[str, ...] = ()
    semantic_digest: str = ""

    def __post_init__(self) -> None:
        _require_nonempty("transition template_id", self.template_id)
        _require_string_tuple(
            "transition input states", self.input_state_template_ids, nonempty=True
        )
        _require_nonempty("transition result state", self.result_state_template_id)
        _require_nonempty(
            "transition quantum-flow template", self.quantum_flow_template_id
        )
        _require_nonempty("transition evaluator resolver", self.evaluator_resolver_key)
        _require_permutation(
            "transition canonical input order",
            self.canonical_input_order,
            len(self.input_state_template_ids),
        )
        _require_string_tuple(
            "transition momentum convention",
            self.momentum_convention,
            nonempty=True,
        )
        if len(self.momentum_convention) != len(self.input_state_template_ids):
            raise RecurrenceTemplateError(
                "transition momentum convention must cover every input"
            )
        _require_string_tuple(
            "transition coupling parameters",
            self.coupling_parameter_ids,
            sorted_unique=True,
        )
        _validate_coupling_orders(self.coupling_orders)
        _require_nonempty(
            "transition color contraction", self.color_contraction_template_id
        )
        if not isinstance(self.binding_coupling, ExactComplexRationalV1):
            raise RecurrenceTemplateError(
                "transition binding coupling must be an exact complex rational"
            )
        if not isinstance(self.exact_factor, ExactComplexRationalV1):
            raise RecurrenceTemplateError(
                "transition exact factor must be an exact complex rational"
            )
        if self.output_factor_source not in _OUTPUT_FACTOR_SOURCES:
            raise RecurrenceTemplateError(
                f"unsupported transition output factor source "
                f"{self.output_factor_source!r}"
            )
        _require_nonempty("transition equivalence class", self.equivalence_class)
        if self.input_exchange_factor is not None:
            if not isinstance(self.input_exchange_factor, ExactComplexRationalV1):
                raise RecurrenceTemplateError(
                    "transition input-exchange factor must be an exact complex rational"
                )
            if self.input_exchange_factor == ExactComplexRationalV1.zero():
                raise RecurrenceTemplateError(
                    "transition input-exchange factor must be nonzero"
                )
        _require_nonempty("transition output projection", self.output_projection)
        if len(self.contact_orbit_step_template_ids) != len(
            self.contact_orbit_step_semantic_digests
        ):
            raise RecurrenceTemplateError(
                "transition contact-orbit step IDs and digests must be paired"
            )
        _require_string_tuple(
            "transition contact-orbit steps", self.contact_orbit_step_template_ids
        )
        if self.contact_orbit_step_template_ids != tuple(
            sorted(set(self.contact_orbit_step_template_ids))
        ):
            raise RecurrenceTemplateError(
                "transition contact-orbit step IDs must be sorted and unique"
            )
        for digest in self.contact_orbit_step_semantic_digests:
            _require_sha256(
                "transition contact-orbit step semantic digest", digest
            )
        self._finish_semantic_record()

    def _semantic_fields(self) -> dict[str, object]:
        return {
            "canonical_input_order": list(self.canonical_input_order),
            "binding_coupling": self.binding_coupling.to_dict(),
            "color_contraction_template_id": self.color_contraction_template_id,
            "contact_orbit_step_semantic_digests": list(
                self.contact_orbit_step_semantic_digests
            ),
            "contact_orbit_step_template_ids": list(
                self.contact_orbit_step_template_ids
            ),
            "coupling_orders": [list(item) for item in self.coupling_orders],
            "coupling_parameter_ids": list(self.coupling_parameter_ids),
            "evaluator_resolver_key": self.evaluator_resolver_key,
            "equivalence_class": self.equivalence_class,
            "exact_factor": self.exact_factor.to_dict(),
            "input_exchange_factor": (
                None
                if self.input_exchange_factor is None
                else self.input_exchange_factor.to_dict()
            ),
            "input_state_template_ids": list(self.input_state_template_ids),
            "momentum_convention": list(self.momentum_convention),
            "output_projection": self.output_projection,
            "output_factor_source": self.output_factor_source,
            "quantum_flow_template_id": self.quantum_flow_template_id,
            "result_state_template_id": self.result_state_template_id,
            "template_id": self.template_id,
        }


@dataclass(frozen=True, slots=True)
class PropagatorTemplateV1(_SemanticRecord):
    _record_kind: ClassVar[str] = "propagator"

    template_id: str
    state_template_id: str
    applies_propagator: bool
    evaluator_resolver_key: str | None
    numerator_expression_digest: str | None
    denominator_expression_digest: str | None
    mass_parameter_id: str | None
    width_parameter_id: str | None
    gauge: str | None
    linearity_proof_template_id: str | None
    semantic_digest: str = ""

    def __post_init__(self) -> None:
        _require_nonempty("propagator template_id", self.template_id)
        _require_nonempty("propagator state", self.state_template_id)
        if type(self.applies_propagator) is not bool:
            raise RecurrenceTemplateError(
                "propagator applies_propagator must be boolean"
            )
        _require_optional_sha256(
            "propagator numerator expression", self.numerator_expression_digest
        )
        _require_optional_sha256(
            "propagator denominator expression", self.denominator_expression_digest
        )
        for name, value in (
            ("evaluator_resolver_key", self.evaluator_resolver_key),
            ("mass_parameter_id", self.mass_parameter_id),
            ("width_parameter_id", self.width_parameter_id),
            ("gauge", self.gauge),
            ("linearity_proof_template_id", self.linearity_proof_template_id),
        ):
            if value is not None:
                _require_nonempty(f"propagator {name}", value)
        required = (
            self.evaluator_resolver_key,
            self.numerator_expression_digest,
            self.denominator_expression_digest,
        )
        if self.applies_propagator and any(value is None for value in required):
            raise RecurrenceTemplateError(
                "active propagators require evaluator, numerator, and denominator"
            )
        if not self.applies_propagator and any(value is not None for value in required):
            raise RecurrenceTemplateError(
                "identity propagators cannot carry evaluator or expressions"
            )
        self._finish_semantic_record()

    def _semantic_fields(self) -> dict[str, object]:
        return {
            "applies_propagator": self.applies_propagator,
            "denominator_expression_digest": self.denominator_expression_digest,
            "evaluator_resolver_key": self.evaluator_resolver_key,
            "gauge": self.gauge,
            "linearity_proof_template_id": self.linearity_proof_template_id,
            "mass_parameter_id": self.mass_parameter_id,
            "numerator_expression_digest": self.numerator_expression_digest,
            "state_template_id": self.state_template_id,
            "template_id": self.template_id,
            "width_parameter_id": self.width_parameter_id,
        }


@dataclass(frozen=True, slots=True)
class ClosureTemplateV1(_SemanticRecord):
    _record_kind: ClassVar[str] = "closure"

    template_id: str
    input_state_template_ids: tuple[str, ...]
    result_state_template_id: str | None
    evaluator_resolver_key: str
    canonical_input_order: tuple[int, ...]
    coupling_parameter_ids: tuple[str, ...]
    coupling_orders: tuple[tuple[str, int], ...]
    eligible_quantum_flow_template_ids: tuple[str, ...]
    color_contraction_template_id: str
    binding_coupling: ExactComplexRationalV1
    exact_factor: ExactComplexRationalV1
    output_factor_source: PreparedOutputFactorSource
    equivalence_class: str
    input_exchange_factor: ExactComplexRationalV1 | None
    projection: str
    component_coefficients: tuple[ExactComplexRationalV1, ...] = ()
    chirality_relation: str = "any"
    metric_signature: str | None = None
    semantic_digest: str = ""

    def __post_init__(self) -> None:
        _require_nonempty("closure template_id", self.template_id)
        _require_string_tuple(
            "closure input states", self.input_state_template_ids, nonempty=True
        )
        if self.result_state_template_id is not None:
            _require_nonempty("closure result state", self.result_state_template_id)
        _require_nonempty("closure evaluator resolver", self.evaluator_resolver_key)
        _require_permutation(
            "closure canonical input order",
            self.canonical_input_order,
            len(self.input_state_template_ids),
        )
        _require_string_tuple(
            "closure coupling parameters",
            self.coupling_parameter_ids,
            sorted_unique=True,
        )
        _validate_coupling_orders(self.coupling_orders)
        _require_string_tuple(
            "closure eligible quantum flows",
            self.eligible_quantum_flow_template_ids,
            sorted_unique=True,
        )
        _require_nonempty(
            "closure color contraction", self.color_contraction_template_id
        )
        if not isinstance(self.binding_coupling, ExactComplexRationalV1):
            raise RecurrenceTemplateError(
                "closure binding coupling must be an exact complex rational"
            )
        if not isinstance(self.exact_factor, ExactComplexRationalV1):
            raise RecurrenceTemplateError(
                "closure exact factor must be an exact complex rational"
            )
        if self.output_factor_source not in _OUTPUT_FACTOR_SOURCES:
            raise RecurrenceTemplateError(
                f"unsupported closure output factor source "
                f"{self.output_factor_source!r}"
            )
        _require_nonempty("closure equivalence class", self.equivalence_class)
        if self.input_exchange_factor is not None:
            if not isinstance(self.input_exchange_factor, ExactComplexRationalV1):
                raise RecurrenceTemplateError(
                    "closure input-exchange factor must be an exact complex rational"
                )
            if self.input_exchange_factor == ExactComplexRationalV1.zero():
                raise RecurrenceTemplateError(
                    "closure input-exchange factor must be nonzero"
                )
        _require_nonempty("closure projection", self.projection)
        coefficients = _require_tuple(
            "closure component coefficients", self.component_coefficients
        )
        if not all(
            isinstance(coefficient, ExactComplexRationalV1)
            for coefficient in coefficients
        ):
            raise RecurrenceTemplateError(
                "closure component coefficients must be exact complex rationals"
            )
        if self.chirality_relation not in {"any", "equal", "opposite"}:
            raise RecurrenceTemplateError(
                "closure chirality relation must be any, equal, or opposite"
            )
        if self.metric_signature is not None:
            _require_nonempty("closure metric signature", self.metric_signature)
        self._finish_semantic_record()

    def _semantic_fields(self) -> dict[str, object]:
        return {
            "canonical_input_order": list(self.canonical_input_order),
            "binding_coupling": self.binding_coupling.to_dict(),
            "chirality_relation": self.chirality_relation,
            "color_contraction_template_id": self.color_contraction_template_id,
            "component_coefficients": [
                value.to_dict() for value in self.component_coefficients
            ],
            "coupling_orders": [list(item) for item in self.coupling_orders],
            "coupling_parameter_ids": list(self.coupling_parameter_ids),
            "eligible_quantum_flow_template_ids": list(
                self.eligible_quantum_flow_template_ids
            ),
            "evaluator_resolver_key": self.evaluator_resolver_key,
            "equivalence_class": self.equivalence_class,
            "exact_factor": self.exact_factor.to_dict(),
            "input_exchange_factor": (
                None
                if self.input_exchange_factor is None
                else self.input_exchange_factor.to_dict()
            ),
            "input_state_template_ids": list(self.input_state_template_ids),
            "result_state_template_id": self.result_state_template_id,
            "metric_signature": self.metric_signature,
            "projection": self.projection,
            "output_factor_source": self.output_factor_source,
            "template_id": self.template_id,
        }


@dataclass(frozen=True, slots=True)
class LCColorTransitionWitnessV1:
    """One exact executable ordered-word term in an LC color contraction."""

    input_shape_kinds: tuple[LCColorShapeKindV1, LCColorShapeKindV1]
    input_permutation: tuple[int, int]
    reverse_parent_mask: int
    component_operation: LCColorComponentOperationV1
    result_component_kind: LCColorComponentKindV1 | None
    result_component_role: LCColorComponentRoleV1
    result_shape_kind: LCColorShapeKindV1 | None
    exact_factor: ExactComplexRationalV1
    proof_digest: str
    provenance: tuple[tuple[str, str], ...] = ()
    input_port_pairings: tuple[tuple[tuple[int, int], tuple[int, int]], ...] = ()
    result_port_bindings: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if len(self.input_shape_kinds) != 2 or any(
            value not in _LC_COLOR_SHAPE_KINDS for value in self.input_shape_kinds
        ):
            raise RecurrenceTemplateError(
                "LC color witness requires two supported input shape kinds"
            )
        if self.input_permutation not in {(0, 1), (1, 0)}:
            raise RecurrenceTemplateError(
                "LC color witness input permutation must be (0, 1) or (1, 0)"
            )
        _require_int("LC color witness reverse-parent mask", self.reverse_parent_mask)
        if self.reverse_parent_mask not in range(4):
            raise RecurrenceTemplateError(
                "LC color witness reverse-parent mask must fit two inputs"
            )
        if self.component_operation not in _LC_COLOR_COMPONENT_OPERATIONS:
            raise RecurrenceTemplateError(
                f"unsupported LC color witness operation {self.component_operation!r}"
            )
        if self.result_component_kind is not None and (
            self.result_component_kind not in _LC_COLOR_COMPONENT_KINDS
        ):
            raise RecurrenceTemplateError(
                "unsupported LC color witness result component kind"
            )
        if self.result_component_role not in _LC_COLOR_COMPONENT_ROLES:
            raise RecurrenceTemplateError(
                "unsupported LC color witness result component role"
            )
        if self.component_operation == "concatenate-join":
            if (
                self.result_component_kind is None
                or self.result_component_role == "none"
            ):
                raise RecurrenceTemplateError(
                    "an LC join requires a result component kind and role"
                )
        elif self.component_operation == "close":
            if self.result_component_role != "none":
                raise RecurrenceTemplateError(
                    "an LC closure cannot declare a recurrence result component"
                )
        elif self.result_component_kind is not None or self.result_component_role != (
            "active"
            if self.component_operation in {"inherit-left", "inherit-right"}
            else "none"
        ):
            raise RecurrenceTemplateError(
                "LC non-join result role does not match its component operation"
            )
        if self.component_operation == "close":
            if self.result_shape_kind is not None:
                raise RecurrenceTemplateError(
                    "LC color closure witness cannot declare a result shape"
                )
        elif self.result_shape_kind not in _LC_COLOR_SHAPE_KINDS:
            raise RecurrenceTemplateError(
                "non-closure LC color witness requires a supported result shape"
            )
        if not isinstance(self.exact_factor, ExactComplexRationalV1):
            raise RecurrenceTemplateError(
                "LC color witness factor must be an exact complex rational"
            )
        if self.exact_factor == ExactComplexRationalV1.zero():
            raise RecurrenceTemplateError("LC color witness factor must be nonzero")
        _require_sha256("LC color witness proof digest", self.proof_digest)
        if self.provenance != tuple(sorted(set(self.provenance))):
            raise RecurrenceTemplateError(
                "LC color witness provenance must be sorted and unique"
            )
        if any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
            for key, value in self.provenance
        ):
            raise RecurrenceTemplateError(
                "LC color witness provenance requires nonempty string pairs"
            )
        consumed_ports: list[tuple[int, int]] = []
        for pairing in self.input_port_pairings:
            if not isinstance(pairing, tuple) or len(pairing) != 2:
                raise RecurrenceTemplateError(
                    "LC color witness port pairings must contain two port references"
                )
            for port in pairing:
                _validate_lc_color_port_reference(port)
                consumed_ports.append(port)
        for port in self.result_port_bindings:
            _validate_lc_color_port_reference(port)
            consumed_ports.append(port)
        if len(consumed_ports) != len(set(consumed_ports)):
            raise RecurrenceTemplateError(
                "LC color witness must consume each parent port at most once"
            )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "component_operation": self.component_operation,
            "exact_factor": self.exact_factor.to_dict(),
            "input_permutation": list(self.input_permutation),
            "input_shape_kinds": list(self.input_shape_kinds),
            "proof_digest": self.proof_digest,
            "provenance": [list(item) for item in self.provenance],
            "result_component_kind": self.result_component_kind,
            "result_component_role": self.result_component_role,
            "result_shape_kind": self.result_shape_kind,
            "reverse_parent_mask": self.reverse_parent_mask,
        }
        if self.input_port_pairings or self.result_port_bindings:
            payload["input_port_pairings"] = [
                [list(left), list(right)] for left, right in self.input_port_pairings
            ]
            payload["result_port_bindings"] = [
                list(port) for port in self.result_port_bindings
            ]
        return payload


@dataclass(frozen=True, slots=True)
class ColorContractionTemplateV1(_SemanticRecord):
    _record_kind: ClassVar[str] = "color-contraction"

    template_id: str
    rule_kind: str
    input_representations: tuple[int, ...]
    output_representation: int | None
    ordered_open_string_arity: int
    exact_coefficient: ExactComplexRationalV1
    nc_polynomial: tuple[tuple[int, ExactComplexRationalV1], ...]
    expression_digest: str
    transition_witnesses: tuple[LCColorTransitionWitnessV1, ...]
    semantic_digest: str = ""

    def __post_init__(self) -> None:
        _require_nonempty("color template_id", self.template_id)
        _require_nonempty("color rule_kind", self.rule_kind)
        _require_int_tuple("color input representations", self.input_representations)
        if self.output_representation is not None:
            _require_int("color output representation", self.output_representation)
        _require_int(
            "color ordered_open_string_arity",
            self.ordered_open_string_arity,
            minimum=0,
        )
        if not isinstance(self.exact_coefficient, ExactComplexRationalV1):
            raise RecurrenceTemplateError(
                "color coefficient must be an exact complex rational"
            )
        terms = _require_tuple("color Nc polynomial", self.nc_polynomial)
        powers: list[int] = []
        for term in terms:
            if not isinstance(term, tuple) or len(term) != 2:
                raise RecurrenceTemplateError(
                    "color Nc polynomial terms must be (power, coefficient)"
                )
            power = _require_int("color Nc power", term[0], minimum=0)
            if not isinstance(term[1], ExactComplexRationalV1):
                raise RecurrenceTemplateError(
                    "color Nc coefficients must be exact complex rationals"
                )
            if term[1] == ExactComplexRationalV1.zero():
                raise RecurrenceTemplateError(
                    "color Nc polynomial cannot retain exact zero terms"
                )
            powers.append(power)
        if powers != sorted(set(powers)):
            raise RecurrenceTemplateError(
                "color Nc polynomial powers must be sorted and unique"
            )
        _require_sha256("color expression_digest", self.expression_digest)
        if not self.transition_witnesses or any(
            not isinstance(value, LCColorTransitionWitnessV1)
            for value in self.transition_witnesses
        ):
            raise RecurrenceTemplateError(
                "color contraction requires exact LC transition witnesses"
            )
        self._finish_semantic_record()

    def _semantic_fields(self) -> dict[str, object]:
        return {
            "exact_coefficient": self.exact_coefficient.to_dict(),
            "expression_digest": self.expression_digest,
            "input_representations": list(self.input_representations),
            "nc_polynomial": [
                [power, coefficient.to_dict()]
                for power, coefficient in self.nc_polynomial
            ],
            "ordered_open_string_arity": self.ordered_open_string_arity,
            "output_representation": self.output_representation,
            "rule_kind": self.rule_kind,
            "template_id": self.template_id,
            "transition_witnesses": [
                witness.to_dict() for witness in self.transition_witnesses
            ],
        }


@dataclass(frozen=True, slots=True)
class SymmetryProofV1(_SemanticRecord):
    _record_kind: ClassVar[str] = "symmetry-proof"

    template_id: str
    proof_algorithm: str
    subject_template_ids: tuple[str, ...]
    input_permutation: tuple[int, ...]
    exact_phase: ExactComplexRationalV1
    expression_digests: tuple[str, ...]
    witness_digest: str
    semantic_digest: str = ""

    def __post_init__(self) -> None:
        _require_nonempty("symmetry proof template_id", self.template_id)
        if self.proof_algorithm not in SUPPORTED_SYMMETRY_PROOF_ALGORITHMS:
            raise RecurrenceTemplateError(
                f"unsupported recurrence proof algorithm {self.proof_algorithm!r}"
            )
        _require_string_tuple(
            "symmetry proof subjects", self.subject_template_ids, nonempty=True
        )
        _require_permutation(
            "symmetry proof input permutation",
            self.input_permutation,
            len(self.input_permutation),
        )
        if not isinstance(self.exact_phase, ExactComplexRationalV1):
            raise RecurrenceTemplateError(
                "symmetry proof phase must be an exact complex rational"
            )
        if self.exact_phase == ExactComplexRationalV1.zero():
            raise RecurrenceTemplateError("symmetry proof phase must be nonzero")
        _require_string_tuple(
            "symmetry proof expression digests",
            self.expression_digests,
            nonempty=True,
        )
        for digest in self.expression_digests:
            _require_sha256("symmetry proof expression digest", digest)
        _require_sha256("symmetry proof witness_digest", self.witness_digest)
        self._finish_semantic_record()

    def _semantic_fields(self) -> dict[str, object]:
        return {
            "exact_phase": self.exact_phase.to_dict(),
            "expression_digests": list(self.expression_digests),
            "input_permutation": list(self.input_permutation),
            "proof_algorithm": self.proof_algorithm,
            "subject_template_ids": list(self.subject_template_ids),
            "template_id": self.template_id,
            "witness_digest": self.witness_digest,
        }


@dataclass(frozen=True, slots=True)
class RecurrenceRuntimeHelicityVariantV1:
    """One exact source-wavefunction embedding into a full current state."""

    source_template_id: str
    source_state_template_id: str
    embedding_source_components: tuple[int | None, ...]
    embedding_factors: tuple[ExactComplexRationalV1, ...]
    projection_full_components: tuple[int, ...]
    proof_digest: str

    def __post_init__(self) -> None:
        _require_nonempty(
            "runtime-helicity source template ID", self.source_template_id
        )
        _require_nonempty(
            "runtime-helicity source-state template ID",
            self.source_state_template_id,
        )
        embedding = _require_tuple(
            "runtime-helicity component embedding",
            self.embedding_source_components,
        )
        if any(
            item is not None and (type(item) is not int or item < 0)
            for item in embedding
        ):
            raise RecurrenceTemplateError(
                "runtime-helicity embedding entries must be nonnegative integers "
                "or null"
            )
        factors = _require_tuple(
            "runtime-helicity embedding factors", self.embedding_factors
        )
        if len(factors) != len(embedding) or any(
            not isinstance(item, ExactComplexRationalV1) for item in factors
        ):
            raise RecurrenceTemplateError(
                "runtime-helicity embedding factors must align with full-state "
                "components"
            )
        zero = ExactComplexRationalV1.zero()
        if any(
            (component is None) != (factor == zero)
            for component, factor in zip(embedding, factors, strict=True)
        ):
            raise RecurrenceTemplateError(
                "runtime-helicity zero embedding slots must have exactly zero factor"
            )
        projection = _require_tuple(
            "runtime-helicity component projection",
            self.projection_full_components,
        )
        if any(type(item) is not int or item < 0 for item in projection):
            raise RecurrenceTemplateError(
                "runtime-helicity projection entries must be nonnegative integers"
            )
        if len(set(projection)) != len(projection):
            raise RecurrenceTemplateError(
                "runtime-helicity projection entries must be unique"
            )
        if tuple(
            sorted(component for component in embedding if component is not None)
        ) != tuple(range(len(projection))):
            raise RecurrenceTemplateError(
                "runtime-helicity embedding must cover each source component once"
            )
        if any(
            full_component >= len(embedding)
            or embedding[full_component] != source_component
            for source_component, full_component in enumerate(projection)
        ):
            raise RecurrenceTemplateError(
                "runtime-helicity projection must invert the component embedding"
            )
        _require_sha256("runtime-helicity variant proof digest", self.proof_digest)

    def to_dict(self) -> dict[str, object]:
        return {
            "embedding_factors": [item.to_dict() for item in self.embedding_factors],
            "embedding_source_components": list(self.embedding_source_components),
            "projection_full_components": list(self.projection_full_components),
            "proof_digest": self.proof_digest,
            "source_state_template_id": self.source_state_template_id,
            "source_template_id": self.source_template_id,
        }


@dataclass(frozen=True, slots=True)
class RecurrenceRuntimeHelicityContractV1(_SemanticRecord):
    """Compiler certificate for runtime source dispatch into one full state."""

    _record_kind: ClassVar[str] = "runtime-helicity-contract"

    template_id: str
    full_state_template_id: str
    variants: tuple[RecurrenceRuntimeHelicityVariantV1, ...]
    proof_algorithm: str
    proof_digest: str
    semantic_digest: str = ""

    def __post_init__(self) -> None:
        _require_nonempty("runtime-helicity template ID", self.template_id)
        _require_nonempty(
            "runtime-helicity full-state template ID", self.full_state_template_id
        )
        _require_nonempty("runtime-helicity proof algorithm", self.proof_algorithm)
        _require_sha256("runtime-helicity proof digest", self.proof_digest)
        variants = _require_tuple("runtime-helicity variants", self.variants)
        if not variants or any(
            not isinstance(item, RecurrenceRuntimeHelicityVariantV1)
            for item in variants
        ):
            raise RecurrenceTemplateError(
                "runtime-helicity contract requires typed source variants"
            )
        source_ids = tuple(item.source_template_id for item in variants)
        if source_ids != tuple(sorted(set(source_ids))):
            raise RecurrenceTemplateError(
                "runtime-helicity source variants must be sorted and unique"
            )
        self._finish_semantic_record()

    def _semantic_fields(self) -> dict[str, object]:
        return {
            "full_state_template_id": self.full_state_template_id,
            "proof_algorithm": self.proof_algorithm,
            "proof_digest": self.proof_digest,
            "template_id": self.template_id,
            "variants": [item.to_dict() for item in self.variants],
        }


@dataclass(frozen=True, slots=True)
class EvaluatorBindingV1(_SemanticRecord):
    _record_kind: ClassVar[str] = "evaluator-binding"

    resolver_key: str
    prepared_kernel_id: int | None
    contract_kind: EvaluatorContractKind
    callable_signature: str
    input_state_template_ids: tuple[str, ...]
    output_state_template_id: str | None
    input_layout: tuple[str, ...]
    output_layout: tuple[str, ...]
    exact_expression_digests: tuple[str, ...]
    semantic_template_ids: tuple[str, ...]
    callable_kind: EvaluatorCallableKind = "prepared-kernel"
    runtime_template: str | None = None
    semantic_digest: str = ""

    def __post_init__(self) -> None:
        _require_nonempty("evaluator resolver_key", self.resolver_key)
        if self.callable_kind not in _EVALUATOR_CALLABLE_KINDS:
            raise RecurrenceTemplateError(
                f"unsupported evaluator callable kind {self.callable_kind!r}"
            )
        if self.callable_kind == "prepared-kernel":
            if self.prepared_kernel_id is None:
                raise RecurrenceTemplateError(
                    "prepared-kernel evaluator requires prepared_kernel_id"
                )
            _require_int(
                "evaluator prepared_kernel_id",
                self.prepared_kernel_id,
                minimum=0,
            )
            if self.runtime_template is not None:
                raise RecurrenceTemplateError(
                    "prepared-kernel evaluator cannot name a Rusticol template"
                )
        else:
            if self.prepared_kernel_id is not None:
                raise RecurrenceTemplateError(
                    "Rusticol-template evaluator cannot name a prepared kernel"
                )
            _require_nonempty("evaluator runtime_template", self.runtime_template)
        if self.contract_kind not in _EVALUATOR_CONTRACT_KINDS:
            raise RecurrenceTemplateError(
                f"unsupported evaluator contract kind {self.contract_kind!r}"
            )
        _require_sha256("evaluator callable_signature", self.callable_signature)
        if self.callable_kind == "rusticol-template":
            assert self.runtime_template is not None
            _validate_rusticol_runtime_template(
                self.contract_kind,
                self.runtime_template,
                self.callable_signature,
            )
        _require_string_tuple("evaluator input states", self.input_state_template_ids)
        if self.output_state_template_id is not None:
            _require_nonempty("evaluator output state", self.output_state_template_id)
        _require_string_tuple(
            "evaluator input layout", self.input_layout, nonempty=True
        )
        _require_string_tuple(
            "evaluator output layout", self.output_layout, nonempty=True
        )
        _require_string_tuple(
            "evaluator exact expression digests",
            self.exact_expression_digests,
            nonempty=True,
        )
        if len(self.output_layout) != len(self.exact_expression_digests):
            raise RecurrenceTemplateError(
                "evaluator output layout and exact expressions must align"
            )
        for digest in self.exact_expression_digests:
            _require_sha256("evaluator exact expression digest", digest)
        _require_string_tuple(
            "evaluator semantic templates",
            self.semantic_template_ids,
            nonempty=True,
            sorted_unique=True,
        )
        if self.contract_kind in {"source", "vertex", "propagator"}:
            if self.output_state_template_id is None:
                raise RecurrenceTemplateError(
                    f"{self.contract_kind} evaluators require an output state"
                )
        elif (
            self.contract_kind in {"closure", "model-parameter"}
            and self.output_state_template_id is not None
        ):
            raise RecurrenceTemplateError(
                f"{self.contract_kind} evaluators cannot produce a current state"
            )
        self._finish_semantic_record()

    def _semantic_fields(self) -> dict[str, object]:
        return {
            "callable_kind": self.callable_kind,
            "callable_signature": self.callable_signature,
            "contract_kind": self.contract_kind,
            "exact_expression_digests": list(self.exact_expression_digests),
            "input_layout": list(self.input_layout),
            "input_state_template_ids": list(self.input_state_template_ids),
            "output_layout": list(self.output_layout),
            "output_state_template_id": self.output_state_template_id,
            "prepared_kernel_id": self.prepared_kernel_id,
            "resolver_key": self.resolver_key,
            "runtime_template": self.runtime_template,
            "semantic_template_ids": list(self.semantic_template_ids),
        }


@dataclass(frozen=True, slots=True)
class RecurrenceTemplateCatalogHeaderV1:
    compiled_model_digest: str
    prepared_kernel_pack_digest: str
    catalog_digest: str
    abi: str = RECURRENCE_TEMPLATE_ABI
    canonicalization_abi: str = RECURRENCE_TEMPLATE_CANONICALIZATION_ABI
    exact_scalar_abi: str = RECURRENCE_TEMPLATE_EXACT_SCALAR_ABI

    def __post_init__(self) -> None:
        if self.abi != RECURRENCE_TEMPLATE_ABI:
            raise RecurrenceTemplateError(
                f"unsupported recurrence template ABI {self.abi!r}"
            )
        if self.canonicalization_abi != RECURRENCE_TEMPLATE_CANONICALIZATION_ABI:
            raise RecurrenceTemplateError(
                "unsupported recurrence template canonicalization ABI"
            )
        if self.exact_scalar_abi != RECURRENCE_TEMPLATE_EXACT_SCALAR_ABI:
            raise RecurrenceTemplateError(
                "unsupported recurrence template exact-scalar ABI"
            )
        _require_sha256("compiled_model_digest", self.compiled_model_digest)
        _require_sha256("prepared_kernel_pack_digest", self.prepared_kernel_pack_digest)
        _require_sha256("catalog_digest", self.catalog_digest)

    def digest_payload(self) -> dict[str, object]:
        return {
            "abi": self.abi,
            "canonicalization_abi": self.canonicalization_abi,
            "compiled_model_digest": self.compiled_model_digest,
            "exact_scalar_abi": self.exact_scalar_abi,
            "prepared_kernel_pack_digest": self.prepared_kernel_pack_digest,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self.digest_payload(), "catalog_digest": self.catalog_digest}


_Record = (
    ParameterTemplateV1
    | CurrentStateTemplateV1
    | SourceTemplateV1
    | QuantumFlowTemplateV1
    | ContactOrbitCertificateV1
    | ContactOrbitStepV1
    | TransitionTemplateV1
    | PropagatorTemplateV1
    | ClosureTemplateV1
    | ColorContractionTemplateV1
    | SymmetryProofV1
    | RecurrenceRuntimeHelicityContractV1
    | EvaluatorBindingV1
)


@dataclass(frozen=True, slots=True)
class RecurrenceTemplateCatalog:
    """Complete, content-addressed recurrence semantics for one compiled model."""

    header: RecurrenceTemplateCatalogHeaderV1
    parameters: tuple[ParameterTemplateV1, ...]
    current_states: tuple[CurrentStateTemplateV1, ...]
    sources: tuple[SourceTemplateV1, ...]
    quantum_flows: tuple[QuantumFlowTemplateV1, ...]
    contact_orbit_certificates: tuple[ContactOrbitCertificateV1, ...]
    contact_orbit_steps: tuple[ContactOrbitStepV1, ...]
    transitions: tuple[TransitionTemplateV1, ...]
    propagators: tuple[PropagatorTemplateV1, ...]
    closures: tuple[ClosureTemplateV1, ...]
    color_contractions: tuple[ColorContractionTemplateV1, ...]
    symmetry_proofs: tuple[SymmetryProofV1, ...]
    runtime_helicity_contracts: tuple[RecurrenceRuntimeHelicityContractV1, ...]
    evaluator_bindings: tuple[EvaluatorBindingV1, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.header, RecurrenceTemplateCatalogHeaderV1):
            raise RecurrenceTemplateError(
                "recurrence catalog header has the wrong type"
            )
        sections = self._sections()
        for name, records in sections:
            _require_tuple(f"recurrence catalog {name}", records)
            keys = tuple(_record_identity(record) for record in records)
            if keys != tuple(sorted(keys)):
                raise RecurrenceTemplateError(
                    f"recurrence catalog {name} must be sorted by semantic identity"
                )
            if len(keys) != len(set(keys)):
                if name == "evaluator_bindings":
                    raise RecurrenceTemplateError(
                        "recurrence evaluator resolver keys must be unique"
                    )
                raise RecurrenceTemplateError(
                    f"duplicate semantic identity in recurrence catalog {name}"
                )

        all_records = tuple(record for _, records in sections for record in records)
        digests = tuple(record.semantic_digest for record in all_records)
        if len(digests) != len(set(digests)):
            raise RecurrenceTemplateError(
                "duplicate semantic records in recurrence template catalog"
            )
        template_records = tuple(
            record
            for record in all_records
            if not isinstance(record, EvaluatorBindingV1)
        )
        template_ids = tuple(record.template_id for record in template_records)
        if len(template_ids) != len(set(template_ids)):
            raise RecurrenceTemplateError(
                "recurrence template semantic identities must be globally unique"
            )
        self._validate_references()
        expected = _digest(self.digest_payload())
        if self.header.catalog_digest != expected:
            raise RecurrenceTemplateError("stale recurrence template catalog digest")

    @classmethod
    def create(
        cls,
        *,
        compiled_model_digest: str,
        prepared_kernel_pack_digest: str,
        parameters: Sequence[ParameterTemplateV1] = (),
        current_states: Sequence[CurrentStateTemplateV1] = (),
        sources: Sequence[SourceTemplateV1] = (),
        quantum_flows: Sequence[QuantumFlowTemplateV1] = (),
        contact_orbit_certificates: Sequence[ContactOrbitCertificateV1] = (),
        contact_orbit_steps: Sequence[ContactOrbitStepV1] = (),
        transitions: Sequence[TransitionTemplateV1] = (),
        propagators: Sequence[PropagatorTemplateV1] = (),
        closures: Sequence[ClosureTemplateV1] = (),
        color_contractions: Sequence[ColorContractionTemplateV1] = (),
        symmetry_proofs: Sequence[SymmetryProofV1] = (),
        runtime_helicity_contracts: Sequence[RecurrenceRuntimeHelicityContractV1] = (),
        evaluator_bindings: Sequence[EvaluatorBindingV1] = (),
    ) -> RecurrenceTemplateCatalog:
        _require_sha256("compiled_model_digest", compiled_model_digest)
        _require_sha256("prepared_kernel_pack_digest", prepared_kernel_pack_digest)
        sorted_sections = {
            "parameters": tuple(sorted(parameters, key=_record_identity)),
            "current_states": tuple(sorted(current_states, key=_record_identity)),
            "sources": tuple(sorted(sources, key=_record_identity)),
            "quantum_flows": tuple(sorted(quantum_flows, key=_record_identity)),
            "contact_orbit_certificates": tuple(
                sorted(contact_orbit_certificates, key=_record_identity)
            ),
            "contact_orbit_steps": tuple(
                sorted(contact_orbit_steps, key=_record_identity)
            ),
            "transitions": tuple(sorted(transitions, key=_record_identity)),
            "propagators": tuple(sorted(propagators, key=_record_identity)),
            "closures": tuple(sorted(closures, key=_record_identity)),
            "color_contractions": tuple(
                sorted(color_contractions, key=_record_identity)
            ),
            "symmetry_proofs": tuple(sorted(symmetry_proofs, key=_record_identity)),
            "runtime_helicity_contracts": tuple(
                sorted(runtime_helicity_contracts, key=_record_identity)
            ),
            "evaluator_bindings": tuple(
                sorted(evaluator_bindings, key=_record_identity)
            ),
        }
        provisional_header = {
            "abi": RECURRENCE_TEMPLATE_ABI,
            "canonicalization_abi": RECURRENCE_TEMPLATE_CANONICALIZATION_ABI,
            "compiled_model_digest": compiled_model_digest,
            "exact_scalar_abi": RECURRENCE_TEMPLATE_EXACT_SCALAR_ABI,
            "prepared_kernel_pack_digest": prepared_kernel_pack_digest,
        }
        digest_payload = {
            "header": provisional_header,
            **{
                name: [record.to_dict() for record in records]
                for name, records in sorted_sections.items()
            },
        }
        header = RecurrenceTemplateCatalogHeaderV1(
            compiled_model_digest=compiled_model_digest,
            prepared_kernel_pack_digest=prepared_kernel_pack_digest,
            catalog_digest=_digest(digest_payload),
        )
        return cls(header=header, **sorted_sections)

    def _sections(self) -> tuple[tuple[str, tuple[_Record, ...]], ...]:
        return (
            ("parameters", self.parameters),
            ("current_states", self.current_states),
            ("sources", self.sources),
            ("quantum_flows", self.quantum_flows),
            ("contact_orbit_certificates", self.contact_orbit_certificates),
            ("contact_orbit_steps", self.contact_orbit_steps),
            ("transitions", self.transitions),
            ("propagators", self.propagators),
            ("closures", self.closures),
            ("color_contractions", self.color_contractions),
            ("symmetry_proofs", self.symmetry_proofs),
            ("runtime_helicity_contracts", self.runtime_helicity_contracts),
            ("evaluator_bindings", self.evaluator_bindings),
        )

    def digest_payload(self) -> dict[str, object]:
        return {
            "header": self.header.digest_payload(),
            **{
                name: [record.to_dict() for record in records]
                for name, records in self._sections()
            },
        }

    @property
    def canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    @property
    def catalog_digest(self) -> str:
        return self.header.catalog_digest

    def require_complete_runtime_helicity_contracts(self) -> None:
        """Fail all-flow-union preflight unless every source is certified."""

        covered = {
            variant.source_template_id
            for contract in self.runtime_helicity_contracts
            for variant in contract.variants
        }
        missing = tuple(
            source.template_id
            for source in self.sources
            if source.template_id not in covered
        )
        if missing:
            raise RecurrenceTemplateError(
                "recurrence all-flow-union preflight requires complete certified "
                "full-state runtime-helicity contracts; missing source templates: "
                + ", ".join(missing)
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "header": self.header.to_dict(),
            **{
                name: [record.to_dict() for record in records]
                for name, records in self._sections()
            },
        }

    def _validate_references(self) -> None:
        parameters = {record.template_id: record for record in self.parameters}
        states = {record.template_id: record for record in self.current_states}
        sources = {record.template_id: record for record in self.sources}
        quantum_flows = {record.template_id: record for record in self.quantum_flows}
        contact_orbit_certificates = {
            record.template_id: record for record in self.contact_orbit_certificates
        }
        contact_orbit_steps = {
            record.template_id: record for record in self.contact_orbit_steps
        }
        colors = {record.template_id: record for record in self.color_contractions}
        proofs = {record.template_id: record for record in self.symmetry_proofs}
        templates = {
            record.template_id: record
            for _, records in self._sections()
            for record in records
            if not isinstance(record, EvaluatorBindingV1)
        }
        evaluators = {record.resolver_key: record for record in self.evaluator_bindings}
        static_quantum_flows: dict[str, QuantumNumberFlowV1] = {}

        for parameter in self.parameters:
            _require_references(
                "parameter dependencies",
                parameter.dependency_parameter_ids,
                parameters,
            )
            if parameter.template_id in parameter.dependency_parameter_ids:
                raise RecurrenceTemplateError(
                    "parameter cannot depend directly on itself"
                )
        _require_acyclic_parameter_dependencies(parameters)
        for step in self.contact_orbit_steps:
            certificate = _require_reference(
                "contact-orbit step certificate",
                step.certificate_template_id,
                contact_orbit_certificates,
            )
            arity = len(certificate.particles)
            referenced_legs = {
                step.result_leg,
                *step.left_covered_legs,
                *step.right_covered_legs,
            }
            if any(leg not in range(arity) for leg in referenced_legs):
                raise RecurrenceTemplateError(
                    "contact-orbit step exceeds its certificate arity"
                )
            if step.stage == "final" and set(
                (*step.left_covered_legs, *step.right_covered_legs)
            ) != (set(range(arity)) - {step.result_leg}):
                raise RecurrenceTemplateError(
                    "contact-orbit final step does not cover its certificate"
                )
        for state in self.current_states:
            _require_optional_reference(
                "current mass parameter", state.mass_parameter_id, parameters
            )
            _require_optional_reference(
                "current width parameter", state.width_parameter_id, parameters
            )
        for source in self.sources:
            source_state = _require_reference(
                "source state", source.state_template_id, states
            )
            if (
                source.lc_color_seed.output_shape_kind
                != source_state.lc_color_shape_kind
            ):
                raise RecurrenceTemplateError(
                    "source LC color seed shape does not match its current state"
                )
            _require_optional_reference(
                "source mass parameter", source.mass_parameter_id, parameters
            )
            _require_optional_reference(
                "source width parameter", source.width_parameter_id, parameters
            )
            evaluator = _require_reference(
                "source evaluator", source.evaluator_resolver_key, evaluators
            )
            _validate_evaluator_contract(
                evaluator,
                kind="source",
                inputs=(),
                output=source.state_template_id,
                semantic_template_id=source.template_id,
            )
        for flow in self.quantum_flows:
            _require_references(
                "quantum-flow input states", flow.input_state_template_ids, states
            )
            result_state = _require_reference(
                "quantum-flow result state", flow.result_state_template_id, states
            )
            expected_flavour = _apply_flavour_flow_operation(
                flow.flavour_flow_operation,
                flow.input_flavour_flows[0],
                flow.input_flavour_flows[1],
                result_state.particle_id,
            )
            if flow.result_flavour_flow != expected_flavour:
                raise RecurrenceTemplateError(
                    "quantum-flow flavour operation does not reproduce its stored "
                    "result witness"
                )
            previous_quantum_flow = static_quantum_flows.setdefault(
                flow.result_state_template_id,
                flow.result_quantum_number_flow,
            )
            if flow.result_quantum_number_flow != previous_quantum_flow:
                raise RecurrenceTemplateError(
                    "particle-static quantum-number flow differs for one result state"
                )
        for transition in self.transitions:
            _require_references(
                "transition input states",
                transition.input_state_template_ids,
                states,
            )
            _require_reference(
                "transition result state",
                transition.result_state_template_id,
                states,
            )
            flow = _require_reference(
                "transition quantum flow",
                transition.quantum_flow_template_id,
                quantum_flows,
            )
            if (
                flow.input_state_template_ids != transition.input_state_template_ids
                or flow.result_state_template_id != transition.result_state_template_id
                or flow.coupling_orders != transition.coupling_orders
            ):
                raise RecurrenceTemplateError(
                    "transition and quantum-flow state/coupling contracts do not match"
                )
            _require_references(
                "transition coupling parameters",
                transition.coupling_parameter_ids,
                parameters,
            )
            _require_reference(
                "transition color contraction",
                transition.color_contraction_template_id,
                colors,
            )
            evaluator = _require_reference(
                "transition evaluator",
                transition.evaluator_resolver_key,
                evaluators,
            )
            _validate_evaluator_contract(
                evaluator,
                kind="vertex",
                inputs=transition.input_state_template_ids,
                output=transition.result_state_template_id,
                semantic_template_id=transition.template_id,
            )
            for step_id, step_digest in zip(
                transition.contact_orbit_step_template_ids,
                transition.contact_orbit_step_semantic_digests,
                strict=True,
            ):
                step = _require_reference(
                    "transition contact-orbit step", step_id, contact_orbit_steps
                )
                if step_digest != step.semantic_digest:
                    raise RecurrenceTemplateError(
                        "transition contact-orbit step digest is stale"
                    )
        for propagator in self.propagators:
            _require_reference("propagator state", propagator.state_template_id, states)
            _require_optional_reference(
                "propagator mass parameter", propagator.mass_parameter_id, parameters
            )
            _require_optional_reference(
                "propagator width parameter",
                propagator.width_parameter_id,
                parameters,
            )
            _require_optional_reference(
                "propagator linearity proof",
                propagator.linearity_proof_template_id,
                proofs,
            )
            if propagator.evaluator_resolver_key is not None:
                evaluator = _require_reference(
                    "propagator evaluator",
                    propagator.evaluator_resolver_key,
                    evaluators,
                )
                _validate_evaluator_contract(
                    evaluator,
                    kind="propagator",
                    inputs=(propagator.state_template_id,),
                    output=propagator.state_template_id,
                    semantic_template_id=propagator.template_id,
                )
        for closure in self.closures:
            _require_references(
                "closure input states", closure.input_state_template_ids, states
            )
            _require_optional_reference(
                "closure result state", closure.result_state_template_id, states
            )
            eligible_flows = tuple(
                _require_reference(
                    "closure eligible quantum flow", flow_id, quantum_flows
                )
                for flow_id in closure.eligible_quantum_flow_template_ids
            )
            if any(
                flow.input_state_template_ids != closure.input_state_template_ids
                or flow.result_state_template_id != closure.result_state_template_id
                or flow.coupling_orders != closure.coupling_orders
                for flow in eligible_flows
            ):
                raise RecurrenceTemplateError(
                    "closure and eligible quantum-flow input/result/coupling "
                    "contracts do not match"
                )
            _require_references(
                "closure coupling parameters",
                closure.coupling_parameter_ids,
                parameters,
            )
            _require_reference(
                "closure color contraction",
                closure.color_contraction_template_id,
                colors,
            )
            evaluator = _require_reference(
                "closure evaluator", closure.evaluator_resolver_key, evaluators
            )
            if evaluator.callable_kind == "prepared-kernel" and not eligible_flows:
                raise RecurrenceTemplateError(
                    "prepared closure must carry at least one eligible quantum-flow "
                    "witness"
                )
            if evaluator.callable_kind == "rusticol-template" and eligible_flows:
                raise RecurrenceTemplateError(
                    "direct Rusticol closure cannot carry prepared quantum-flow "
                    "witnesses"
                )
            if (
                evaluator.callable_kind == "prepared-kernel"
                and closure.result_state_template_id is None
            ):
                raise RecurrenceTemplateError(
                    "prepared closure must carry a result-state contract"
                )
            if (
                evaluator.callable_kind == "rusticol-template"
                and closure.result_state_template_id is not None
            ):
                raise RecurrenceTemplateError(
                    "direct Rusticol closure cannot carry a result-state contract"
                )
            _validate_evaluator_contract(
                evaluator,
                kind="closure",
                inputs=closure.input_state_template_ids,
                output=None,
                semantic_template_id=closure.template_id,
            )
        for proof in self.symmetry_proofs:
            _require_references(
                "symmetry proof subjects", proof.subject_template_ids, templates
            )
        source_contract_owner: dict[str, str] = {}
        particle_contract_owner: dict[int, str] = {}
        for contract in self.runtime_helicity_contracts:
            full_state = _require_reference(
                "runtime-helicity full state",
                contract.full_state_template_id,
                states,
            )
            previous_particle_owner = particle_contract_owner.setdefault(
                full_state.particle_id, contract.template_id
            )
            if previous_particle_owner != contract.template_id:
                raise RecurrenceTemplateError(
                    "one physical source family cannot span runtime-helicity contracts"
                )
            variant_source_ids = {item.source_template_id for item in contract.variants}
            family_source_ids = {
                item.template_id
                for item in self.sources
                if states[item.state_template_id].particle_id == full_state.particle_id
            }
            if variant_source_ids != family_source_ids:
                raise RecurrenceTemplateError(
                    "runtime-helicity contract must cover its complete physical "
                    "source family"
                )
            for variant in contract.variants:
                source = _require_reference(
                    "runtime-helicity source", variant.source_template_id, sources
                )
                source_state = _require_reference(
                    "runtime-helicity source state",
                    variant.source_state_template_id,
                    states,
                )
                if source.state_template_id != source_state.template_id:
                    raise RecurrenceTemplateError(
                        "runtime-helicity source variant names the wrong source state"
                    )
                if (
                    source_state.particle_id != full_state.particle_id
                    or source_state.species_id != full_state.species_id
                    or source_state.orientation != full_state.orientation
                ):
                    raise RecurrenceTemplateError(
                        "runtime-helicity variant and full state are not one physical "
                        "source family"
                    )
                if len(variant.embedding_source_components) != full_state.dimension:
                    raise RecurrenceTemplateError(
                        "runtime-helicity embedding dimension does not match full state"
                    )
                if len(variant.projection_full_components) != source_state.dimension:
                    raise RecurrenceTemplateError(
                        "runtime-helicity projection dimension does not match source "
                        "state"
                    )
                previous = source_contract_owner.setdefault(
                    source.template_id, contract.template_id
                )
                if previous != contract.template_id:
                    raise RecurrenceTemplateError(
                        "source template belongs to multiple runtime-helicity contracts"
                    )
        for binding in self.evaluator_bindings:
            _require_references(
                "evaluator input states", binding.input_state_template_ids, states
            )
            _require_optional_reference(
                "evaluator output state", binding.output_state_template_id, states
            )
            _require_references(
                "evaluator semantic templates",
                binding.semantic_template_ids,
                templates,
            )
            if binding.output_state_template_id is not None:
                dimension = states[binding.output_state_template_id].dimension
                if len(binding.output_layout) != dimension:
                    raise RecurrenceTemplateError(
                        "evaluator output layout does not match state dimension"
                    )

        callable_by_kernel: dict[
            int,
            tuple[
                str,
                str,
                tuple[str, ...],
                tuple[str, ...],
                tuple[str, ...],
            ],
        ] = {}
        callable_by_runtime_template: dict[
            str,
            tuple[
                str,
                str,
                tuple[str, ...],
                tuple[str, ...],
                tuple[str, ...],
            ],
        ] = {}
        semantic_binding_owner: dict[str, str] = {}
        expected_template_types = {
            "source": SourceTemplateV1,
            "vertex": TransitionTemplateV1,
            "propagator": PropagatorTemplateV1,
            "closure": ClosureTemplateV1,
            "model-parameter": ParameterTemplateV1,
        }
        for binding in self.evaluator_bindings:
            callable_contract = (
                binding.contract_kind,
                binding.callable_signature,
                binding.input_layout,
                binding.output_layout,
                binding.exact_expression_digests,
            )
            if binding.prepared_kernel_id is not None:
                previous = callable_by_kernel.setdefault(
                    binding.prepared_kernel_id, callable_contract
                )
                owner = "prepared kernel ID"
            else:
                assert binding.runtime_template is not None
                previous = callable_by_runtime_template.setdefault(
                    binding.runtime_template, callable_contract
                )
                owner = "Rusticol runtime template"
            if previous != callable_contract:
                raise RecurrenceTemplateError(
                    f"{owner} has inconsistent callable bindings"
                )
            expected_type = expected_template_types[binding.contract_kind]
            for template_id in binding.semantic_template_ids:
                template = templates[template_id]
                if not isinstance(template, expected_type):
                    raise RecurrenceTemplateError(
                        "evaluator contract kind does not match semantic template"
                    )
                previous_owner = semantic_binding_owner.setdefault(
                    template_id, binding.resolver_key
                )
                if previous_owner != binding.resolver_key:
                    raise RecurrenceTemplateError(
                        "semantic template has multiple evaluator resolver keys"
                    )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RecurrenceTemplateCatalog:
        root = _require_mapping("recurrence template catalog", payload)
        section_names = frozenset(
            {
                "header",
                "parameters",
                "current_states",
                "sources",
                "quantum_flows",
                "contact_orbit_certificates",
                "contact_orbit_steps",
                "transitions",
                "propagators",
                "closures",
                "color_contractions",
                "symmetry_proofs",
                "runtime_helicity_contracts",
                "evaluator_bindings",
            }
        )
        _require_exact_keys("recurrence template catalog", root, section_names)
        header_payload = _require_mapping("catalog header", root["header"])
        _require_exact_keys(
            "catalog header",
            header_payload,
            frozenset(
                {
                    "abi",
                    "canonicalization_abi",
                    "compiled_model_digest",
                    "exact_scalar_abi",
                    "prepared_kernel_pack_digest",
                    "catalog_digest",
                }
            ),
        )
        header = RecurrenceTemplateCatalogHeaderV1(
            compiled_model_digest=_require_sha256(
                "compiled_model_digest", header_payload["compiled_model_digest"]
            ),
            prepared_kernel_pack_digest=_require_sha256(
                "prepared_kernel_pack_digest",
                header_payload["prepared_kernel_pack_digest"],
            ),
            catalog_digest=_require_sha256(
                "catalog_digest", header_payload["catalog_digest"]
            ),
            abi=_require_nonempty("catalog abi", header_payload["abi"]),
            canonicalization_abi=_require_nonempty(
                "catalog canonicalization_abi",
                header_payload["canonicalization_abi"],
            ),
            exact_scalar_abi=_require_nonempty(
                "catalog exact_scalar_abi", header_payload["exact_scalar_abi"]
            ),
        )
        decoders = {
            "parameters": _parameter_from_dict,
            "current_states": _current_state_from_dict,
            "sources": _source_from_dict,
            "quantum_flows": _quantum_flow_from_dict,
            "contact_orbit_certificates": _contact_orbit_certificate_from_dict,
            "contact_orbit_steps": _contact_orbit_step_from_dict,
            "transitions": _transition_from_dict,
            "propagators": _propagator_from_dict,
            "closures": _closure_from_dict,
            "color_contractions": _color_from_dict,
            "symmetry_proofs": _symmetry_from_dict,
            "runtime_helicity_contracts": _runtime_helicity_contract_from_dict,
            "evaluator_bindings": _evaluator_from_dict,
        }
        decoded: dict[str, tuple[_Record, ...]] = {}
        for name, decoder in decoders.items():
            rows = root[name]
            if not isinstance(rows, list):
                raise RecurrenceTemplateError(f"catalog {name} must be an array")
            decoded[name] = tuple(
                decoder(_require_mapping(f"catalog {name} record", row)) for row in rows
            )
        return cls(header=header, **decoded)  # type: ignore[arg-type]


def _record_identity(record: _Record) -> str:
    if isinstance(record, EvaluatorBindingV1):
        return record.resolver_key
    return record.template_id


def _require_reference(name: str, key: str, known: Mapping[str, _Record]):
    try:
        return known[key]
    except KeyError as exc:
        raise RecurrenceTemplateError(f"{name} references unknown {key!r}") from exc


def _require_optional_reference(
    name: str,
    key: str | None,
    known: Mapping[str, _Record],
) -> None:
    if key is not None:
        _require_reference(name, key, known)


def _require_references(
    name: str,
    keys: Sequence[str],
    known: Mapping[str, _Record],
) -> None:
    for key in keys:
        _require_reference(name, key, known)


def _require_acyclic_parameter_dependencies(
    parameters: Mapping[str, ParameterTemplateV1],
) -> None:
    indegree = {
        template_id: len(parameter.dependency_parameter_ids)
        for template_id, parameter in parameters.items()
    }
    dependents: dict[str, list[str]] = {template_id: [] for template_id in parameters}
    for template_id, parameter in parameters.items():
        for dependency_id in parameter.dependency_parameter_ids:
            dependents[dependency_id].append(template_id)
    ready = sorted(template_id for template_id, count in indegree.items() if count == 0)
    visited = 0
    while ready:
        template_id = ready.pop()
        visited += 1
        for dependent_id in dependents[template_id]:
            indegree[dependent_id] -= 1
            if indegree[dependent_id] == 0:
                ready.append(dependent_id)
    if visited != len(parameters):
        cyclic = sorted(
            template_id for template_id, count in indegree.items() if count > 0
        )
        raise RecurrenceTemplateError(
            "parameter dependency graph contains a cycle involving: "
            + ", ".join(cyclic)
        )


def _validate_evaluator_contract(
    binding: EvaluatorBindingV1,
    *,
    kind: EvaluatorContractKind,
    inputs: tuple[str, ...],
    output: str | None,
    semantic_template_id: str,
) -> None:
    if binding.contract_kind != kind:
        raise RecurrenceTemplateError(
            f"{semantic_template_id!r} uses a {binding.contract_kind!r} evaluator "
            f"where {kind!r} is required"
        )
    if binding.input_state_template_ids != inputs:
        raise RecurrenceTemplateError(
            f"{semantic_template_id!r} evaluator input states do not match"
        )
    if binding.output_state_template_id != output:
        raise RecurrenceTemplateError(
            f"{semantic_template_id!r} evaluator output state does not match"
        )
    if semantic_template_id not in binding.semantic_template_ids:
        raise RecurrenceTemplateError(
            f"{semantic_template_id!r} is missing from its evaluator binding"
        )


def _record_payload(
    name: str,
    payload: Mapping[str, object],
    fields: frozenset[str],
    record_kind: str,
) -> Mapping[str, object]:
    _require_exact_keys(name, payload, fields | {"record_kind", "semantic_digest"})
    if payload["record_kind"] != record_kind:
        raise RecurrenceTemplateError(
            f"{name} has record kind {payload['record_kind']!r}, expected "
            f"{record_kind!r}"
        )
    _require_sha256(f"{name} semantic_digest", payload["semantic_digest"])
    return payload


def _decode_ratio(name: str, value: object) -> ExactComplexRationalV1:
    return ExactComplexRationalV1.from_dict(_require_mapping(name, value))


def _decode_optional_ratio(
    name: str,
    value: object,
) -> ExactComplexRationalV1 | None:
    if value is None:
        return None
    return _decode_ratio(name, value)


def _decode_ratio_array(
    name: str,
    value: object,
) -> tuple[ExactComplexRationalV1, ...]:
    if not isinstance(value, list):
        raise RecurrenceTemplateError(f"{name} must be an array")
    return tuple(
        _decode_ratio(f"{name}[{index}]", item) for index, item in enumerate(value)
    )


def _decode_coupling_orders(name: str, value: object) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, list):
        raise RecurrenceTemplateError(f"{name} must be an array")
    result: list[tuple[str, int]] = []
    for item in value:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not isinstance(item[0], str)
            or type(item[1]) is not int
        ):
            raise RecurrenceTemplateError(f"{name} entries must be [name, power]")
        result.append((item[0], item[1]))
    return tuple(result)


def _parameter_from_dict(payload: Mapping[str, object]) -> ParameterTemplateV1:
    value = _record_payload(
        "parameter template",
        payload,
        frozenset(
            {
                "template_id",
                "name",
                "parameter_kind",
                "value_type",
                "mutable",
                "default_value",
                "exact_expression_digest",
                "dependency_parameter_ids",
                "prepared_parameter_id",
            }
        ),
        "parameter",
    )
    default = value["default_value"]
    return ParameterTemplateV1(
        template_id=_require_nonempty("parameter template_id", value["template_id"]),
        name=_require_nonempty("parameter name", value["name"]),
        parameter_kind=value["parameter_kind"],  # type: ignore[arg-type]
        value_type=value["value_type"],  # type: ignore[arg-type]
        mutable=value["mutable"],  # type: ignore[arg-type]
        default_value=None if default is None else _decode_ratio("default", default),
        exact_expression_digest=_require_optional_sha256(
            "parameter exact expression", value["exact_expression_digest"]
        ),
        dependency_parameter_ids=_decode_string_tuple(
            "parameter dependencies", value["dependency_parameter_ids"]
        ),
        prepared_parameter_id=(
            None
            if value["prepared_parameter_id"] is None
            else _require_int(
                "prepared parameter ID",
                value["prepared_parameter_id"],
                minimum=0,
            )
        ),
        semantic_digest=value["semantic_digest"],  # type: ignore[arg-type]
    )


def _current_state_from_dict(payload: Mapping[str, object]) -> CurrentStateTemplateV1:
    fields = frozenset(
        {
            "template_id",
            "particle_id",
            "anti_particle_id",
            "species_id",
            "orientation",
            "statistics",
            "color_representation",
            "basis",
            "tensor_ordering",
            "dimension",
            "chirality",
            "lc_color_shape_kind",
            "auxiliary_kind",
            "mass_parameter_id",
            "width_parameter_id",
        }
    )
    value = _record_payload("current-state template", payload, fields, "current-state")
    return CurrentStateTemplateV1(
        template_id=_require_nonempty("state template_id", value["template_id"]),
        particle_id=_require_int("particle_id", value["particle_id"]),
        anti_particle_id=_require_int("anti_particle_id", value["anti_particle_id"]),
        species_id=_require_nonempty("species_id", value["species_id"]),
        orientation=_require_nonempty("orientation", value["orientation"]),
        statistics=_require_nonempty("statistics", value["statistics"]),
        color_representation=_require_int(
            "color_representation", value["color_representation"]
        ),
        basis=_require_nonempty("basis", value["basis"]),
        tensor_ordering=_decode_string_tuple(
            "tensor_ordering", value["tensor_ordering"]
        ),
        dimension=_require_int("dimension", value["dimension"]),
        chirality=_require_int("chirality", value["chirality"]),
        lc_color_shape_kind=_require_nonempty(
            "LC color shape kind", value["lc_color_shape_kind"]
        ),  # type: ignore[arg-type]
        auxiliary_kind=_decode_optional_string(
            "auxiliary_kind", value["auxiliary_kind"]
        ),
        mass_parameter_id=_decode_optional_string(
            "mass_parameter_id", value["mass_parameter_id"]
        ),
        width_parameter_id=_decode_optional_string(
            "width_parameter_id", value["width_parameter_id"]
        ),
        semantic_digest=value["semantic_digest"],  # type: ignore[arg-type]
    )


def _source_from_dict(payload: Mapping[str, object]) -> SourceTemplateV1:
    fields = frozenset(
        {
            "template_id",
            "state_template_id",
            "crossing",
            "wavefunction_family",
            "helicity",
            "spin_state",
            "flavour_flow",
            "quantum_number_flow",
            "lc_color_seed",
            "wavefunction_expression_digest",
            "evaluator_resolver_key",
            "mass_parameter_id",
            "width_parameter_id",
        }
    )
    value = _record_payload("source template", payload, fields, "source")
    return SourceTemplateV1(
        template_id=_require_nonempty("source template_id", value["template_id"]),
        state_template_id=_require_nonempty(
            "source state_template_id", value["state_template_id"]
        ),
        crossing=_require_nonempty("source crossing", value["crossing"]),
        wavefunction_family=_require_nonempty(
            "source wavefunction family", value["wavefunction_family"]
        ),
        helicity=_require_int("source helicity", value["helicity"]),
        spin_state=_require_int("source spin_state", value["spin_state"]),
        flavour_flow=_decode_flavour_flow("source flavour flow", value["flavour_flow"]),
        quantum_number_flow=_decode_quantum_number_flow(
            "source quantum-number flow", value["quantum_number_flow"]
        ),
        lc_color_seed=_lc_color_source_seed_from_dict(value["lc_color_seed"]),
        wavefunction_expression_digest=_require_sha256(
            "source wavefunction expression", value["wavefunction_expression_digest"]
        ),
        evaluator_resolver_key=_require_nonempty(
            "source evaluator", value["evaluator_resolver_key"]
        ),
        mass_parameter_id=_decode_optional_string(
            "source mass parameter", value["mass_parameter_id"]
        ),
        width_parameter_id=_decode_optional_string(
            "source width parameter", value["width_parameter_id"]
        ),
        semantic_digest=value["semantic_digest"],  # type: ignore[arg-type]
    )


def _lc_color_source_seed_from_dict(payload: object) -> LCColorSourceSeedV1:
    value = _require_mapping("LC color source seed", payload)
    _require_exact_keys(
        "LC color source seed",
        value,
        frozenset(
            {
                "component_kind",
                "component_role",
                "operation",
                "output_shape_kind",
                "proof_digest",
                "provenance",
            }
        ),
    )
    return LCColorSourceSeedV1(
        operation=_require_nonempty(
            "LC color source seed operation", value["operation"]
        ),  # type: ignore[arg-type]
        output_shape_kind=_require_nonempty(
            "LC color source seed output shape", value["output_shape_kind"]
        ),  # type: ignore[arg-type]
        component_kind=_decode_optional_string(
            "LC color source seed component kind", value["component_kind"]
        ),  # type: ignore[arg-type]
        component_role=_require_nonempty(
            "LC color source seed component role", value["component_role"]
        ),  # type: ignore[arg-type]
        proof_digest=_require_sha256(
            "LC color source seed proof digest", value["proof_digest"]
        ),
        provenance=_decode_string_pairs(
            "LC color source seed provenance",
            value["provenance"],
        ),
    )


def _quantum_flow_from_dict(payload: Mapping[str, object]) -> QuantumFlowTemplateV1:
    fields = frozenset(
        {
            "template_id",
            "input_state_template_ids",
            "input_spin_states",
            "input_flavour_flows",
            "input_quantum_number_flows",
            "flavour_flow_operation",
            "quantum_number_flow_operation",
            "coupling_orders",
            "result_state_template_id",
            "result_spin_state",
            "result_flavour_flow",
            "result_quantum_number_flow",
            "exact_coupling",
            "predicate_digest",
        }
    )
    value = _record_payload("quantum-flow template", payload, fields, "quantum-flow")
    return QuantumFlowTemplateV1(
        template_id=_require_nonempty("flow template_id", value["template_id"]),
        input_state_template_ids=_decode_string_tuple(
            "flow input states", value["input_state_template_ids"]
        ),
        input_spin_states=_decode_int_tuple(
            "flow spin states", value["input_spin_states"]
        ),
        input_flavour_flows=_decode_flavour_flows(
            "flow flavour inputs", value["input_flavour_flows"]
        ),
        input_quantum_number_flows=_decode_quantum_number_flows(
            "flow quantum-number inputs", value["input_quantum_number_flows"]
        ),
        flavour_flow_operation=_require_nonempty(
            "flow flavour operation", value["flavour_flow_operation"]
        ),  # type: ignore[arg-type]
        quantum_number_flow_operation=_require_nonempty(
            "flow quantum-number operation",
            value["quantum_number_flow_operation"],
        ),  # type: ignore[arg-type]
        coupling_orders=_decode_coupling_orders(
            "flow coupling orders", value["coupling_orders"]
        ),
        result_state_template_id=_require_nonempty(
            "flow result state", value["result_state_template_id"]
        ),
        result_spin_state=_require_int(
            "flow result spin state", value["result_spin_state"]
        ),
        result_flavour_flow=_decode_flavour_flow(
            "flow result flavour", value["result_flavour_flow"]
        ),
        result_quantum_number_flow=_decode_quantum_number_flow(
            "flow result quantum number", value["result_quantum_number_flow"]
        ),
        exact_coupling=_decode_ratio("flow exact coupling", value["exact_coupling"]),
        predicate_digest=_require_sha256(
            "flow predicate digest", value["predicate_digest"]
        ),
        semantic_digest=value["semantic_digest"],  # type: ignore[arg-type]
    )


def _contact_orbit_certificate_from_dict(
    payload: Mapping[str, object],
) -> ContactOrbitCertificateV1:
    fields = frozenset(
        {
            "template_id",
            "algorithm",
            "algorithm_version",
            "term_id",
            "vertex",
            "particles",
            "evaluator_class",
            "physical_leg_equivalence_classes",
            "reconstruction_factor",
        }
    )
    value = _record_payload(
        "contact-orbit certificate",
        payload,
        fields,
        "contact-orbit-certificate",
    )
    particles = _decode_string_tuple(
        "contact-orbit certificate particles", value["particles"]
    )
    classes = _decode_int_tuple(
        "contact-orbit certificate equivalence classes",
        value["physical_leg_equivalence_classes"],
    )
    if len(particles) not in {4, 5} or len(classes) != len(particles):
        raise RecurrenceTemplateError(
            "contact-orbit certificate must describe four or five legs"
        )
    return ContactOrbitCertificateV1(
        template_id=_require_nonempty(
            "contact-orbit certificate template_id", value["template_id"]
        ),
        algorithm=_require_nonempty(
            "contact-orbit certificate algorithm", value["algorithm"]
        ),
        algorithm_version=_require_int(
            "contact-orbit certificate algorithm version",
            value["algorithm_version"],
            minimum=1,
        ),
        term_id=_require_int(
            "contact-orbit certificate term ID", value["term_id"], minimum=0
        ),
        vertex=_require_nonempty(
            "contact-orbit certificate vertex", value["vertex"]
        ),
        particles=particles,
        evaluator_class=_require_nonempty(
            "contact-orbit certificate evaluator class", value["evaluator_class"]
        ),
        physical_leg_equivalence_classes=classes,
        reconstruction_factor=_decode_ratio(
            "contact-orbit certificate reconstruction factor",
            value["reconstruction_factor"],
        ),
        semantic_digest=value["semantic_digest"],  # type: ignore[arg-type]
    )


def _contact_orbit_step_from_dict(
    payload: Mapping[str, object],
) -> ContactOrbitStepV1:
    fields = frozenset(
        {
            "template_id",
            "certificate_template_id",
            "stage",
            "result_leg",
            "left_covered_legs",
            "right_covered_legs",
            "source_particle_legs",
            "reconstruction_factor",
        }
    )
    value = _record_payload(
        "contact-orbit step", payload, fields, "contact-orbit-step"
    )
    source_legs = _decode_int_tuple(
        "contact-orbit step source legs", value["source_particle_legs"]
    )
    if len(source_legs) != 3:
        raise RecurrenceTemplateError(
            "contact-orbit step must describe three kernel slots"
        )
    return ContactOrbitStepV1(
        template_id=_require_nonempty(
            "contact-orbit step template_id", value["template_id"]
        ),
        certificate_template_id=_require_nonempty(
            "contact-orbit step certificate", value["certificate_template_id"]
        ),
        stage=_require_nonempty("contact-orbit step stage", value["stage"]),
        result_leg=_require_int(
            "contact-orbit step result leg", value["result_leg"], minimum=0
        ),
        left_covered_legs=_decode_int_tuple(
            "contact-orbit left covered legs", value["left_covered_legs"]
        ),
        right_covered_legs=_decode_int_tuple(
            "contact-orbit right covered legs", value["right_covered_legs"]
        ),
        source_particle_legs=(source_legs[0], source_legs[1], source_legs[2]),
        reconstruction_factor=_decode_ratio(
            "contact-orbit step reconstruction factor",
            value["reconstruction_factor"],
        ),
        semantic_digest=value["semantic_digest"],  # type: ignore[arg-type]
    )


def _transition_from_dict(payload: Mapping[str, object]) -> TransitionTemplateV1:
    fields = frozenset(
        {
            "template_id",
            "input_state_template_ids",
            "result_state_template_id",
            "quantum_flow_template_id",
            "evaluator_resolver_key",
            "canonical_input_order",
            "momentum_convention",
            "coupling_parameter_ids",
            "coupling_orders",
            "color_contraction_template_id",
            "binding_coupling",
            "exact_factor",
            "output_factor_source",
            "equivalence_class",
            "input_exchange_factor",
            "output_projection",
            "contact_orbit_step_template_ids",
            "contact_orbit_step_semantic_digests",
        }
    )
    value = _record_payload("transition template", payload, fields, "transition")
    return TransitionTemplateV1(
        template_id=_require_nonempty("transition template_id", value["template_id"]),
        input_state_template_ids=_decode_string_tuple(
            "transition input states", value["input_state_template_ids"]
        ),
        result_state_template_id=_require_nonempty(
            "transition result state", value["result_state_template_id"]
        ),
        quantum_flow_template_id=_require_nonempty(
            "transition quantum flow", value["quantum_flow_template_id"]
        ),
        evaluator_resolver_key=_require_nonempty(
            "transition evaluator", value["evaluator_resolver_key"]
        ),
        canonical_input_order=_decode_int_tuple(
            "transition input order", value["canonical_input_order"]
        ),
        momentum_convention=_decode_string_tuple(
            "transition momenta", value["momentum_convention"]
        ),
        coupling_parameter_ids=_decode_string_tuple(
            "transition coupling parameters", value["coupling_parameter_ids"]
        ),
        coupling_orders=_decode_coupling_orders(
            "transition coupling orders", value["coupling_orders"]
        ),
        color_contraction_template_id=_require_nonempty(
            "transition color", value["color_contraction_template_id"]
        ),
        binding_coupling=_decode_ratio(
            "transition binding coupling", value["binding_coupling"]
        ),
        exact_factor=_decode_ratio("transition exact factor", value["exact_factor"]),
        output_factor_source=_require_nonempty(
            "transition output factor source", value["output_factor_source"]
        ),  # type: ignore[arg-type]
        equivalence_class=_require_nonempty(
            "transition equivalence class", value["equivalence_class"]
        ),
        input_exchange_factor=_decode_optional_ratio(
            "transition input-exchange factor", value["input_exchange_factor"]
        ),
        output_projection=_require_nonempty(
            "transition projection", value["output_projection"]
        ),
        contact_orbit_step_template_ids=_decode_string_tuple(
            "transition contact-orbit steps",
            value["contact_orbit_step_template_ids"],
        ),
        contact_orbit_step_semantic_digests=_decode_string_tuple(
            "transition contact-orbit step semantic digests",
            value["contact_orbit_step_semantic_digests"],
        ),
        semantic_digest=value["semantic_digest"],  # type: ignore[arg-type]
    )


def _propagator_from_dict(payload: Mapping[str, object]) -> PropagatorTemplateV1:
    fields = frozenset(
        {
            "template_id",
            "state_template_id",
            "applies_propagator",
            "evaluator_resolver_key",
            "numerator_expression_digest",
            "denominator_expression_digest",
            "mass_parameter_id",
            "width_parameter_id",
            "gauge",
            "linearity_proof_template_id",
        }
    )
    value = _record_payload("propagator template", payload, fields, "propagator")
    return PropagatorTemplateV1(
        template_id=_require_nonempty("propagator template_id", value["template_id"]),
        state_template_id=_require_nonempty(
            "propagator state", value["state_template_id"]
        ),
        applies_propagator=value["applies_propagator"],  # type: ignore[arg-type]
        evaluator_resolver_key=_decode_optional_string(
            "propagator evaluator", value["evaluator_resolver_key"]
        ),
        numerator_expression_digest=_require_optional_sha256(
            "propagator numerator", value["numerator_expression_digest"]
        ),
        denominator_expression_digest=_require_optional_sha256(
            "propagator denominator", value["denominator_expression_digest"]
        ),
        mass_parameter_id=_decode_optional_string(
            "propagator mass", value["mass_parameter_id"]
        ),
        width_parameter_id=_decode_optional_string(
            "propagator width", value["width_parameter_id"]
        ),
        gauge=_decode_optional_string("propagator gauge", value["gauge"]),
        linearity_proof_template_id=_decode_optional_string(
            "propagator linearity proof", value["linearity_proof_template_id"]
        ),
        semantic_digest=value["semantic_digest"],  # type: ignore[arg-type]
    )


def _closure_from_dict(payload: Mapping[str, object]) -> ClosureTemplateV1:
    fields = frozenset(
        {
            "template_id",
            "input_state_template_ids",
            "result_state_template_id",
            "evaluator_resolver_key",
            "canonical_input_order",
            "coupling_parameter_ids",
            "coupling_orders",
            "eligible_quantum_flow_template_ids",
            "color_contraction_template_id",
            "binding_coupling",
            "component_coefficients",
            "chirality_relation",
            "equivalence_class",
            "exact_factor",
            "input_exchange_factor",
            "metric_signature",
            "output_factor_source",
            "projection",
        }
    )
    value = _record_payload("closure template", payload, fields, "closure")
    return ClosureTemplateV1(
        template_id=_require_nonempty("closure template_id", value["template_id"]),
        input_state_template_ids=_decode_string_tuple(
            "closure input states", value["input_state_template_ids"]
        ),
        result_state_template_id=_decode_optional_string(
            "closure result state", value["result_state_template_id"]
        ),
        evaluator_resolver_key=_require_nonempty(
            "closure evaluator", value["evaluator_resolver_key"]
        ),
        canonical_input_order=_decode_int_tuple(
            "closure input order", value["canonical_input_order"]
        ),
        coupling_parameter_ids=_decode_string_tuple(
            "closure coupling parameters", value["coupling_parameter_ids"]
        ),
        coupling_orders=_decode_coupling_orders(
            "closure coupling orders", value["coupling_orders"]
        ),
        eligible_quantum_flow_template_ids=_decode_string_tuple(
            "closure eligible quantum flows",
            value["eligible_quantum_flow_template_ids"],
        ),
        color_contraction_template_id=_require_nonempty(
            "closure color", value["color_contraction_template_id"]
        ),
        binding_coupling=_decode_ratio(
            "closure binding coupling", value["binding_coupling"]
        ),
        exact_factor=_decode_ratio("closure exact factor", value["exact_factor"]),
        output_factor_source=_require_nonempty(
            "closure output factor source", value["output_factor_source"]
        ),  # type: ignore[arg-type]
        equivalence_class=_require_nonempty(
            "closure equivalence class", value["equivalence_class"]
        ),
        input_exchange_factor=_decode_optional_ratio(
            "closure input-exchange factor", value["input_exchange_factor"]
        ),
        component_coefficients=_decode_ratio_array(
            "closure component coefficients", value["component_coefficients"]
        ),
        chirality_relation=_require_nonempty(
            "closure chirality relation", value["chirality_relation"]
        ),
        metric_signature=_decode_optional_string(
            "closure metric signature", value["metric_signature"]
        ),
        projection=_require_nonempty("closure projection", value["projection"]),
        semantic_digest=value["semantic_digest"],  # type: ignore[arg-type]
    )


def _color_from_dict(payload: Mapping[str, object]) -> ColorContractionTemplateV1:
    fields = frozenset(
        {
            "template_id",
            "rule_kind",
            "input_representations",
            "output_representation",
            "ordered_open_string_arity",
            "exact_coefficient",
            "nc_polynomial",
            "expression_digest",
            "transition_witnesses",
        }
    )
    value = _record_payload(
        "color-contraction template", payload, fields, "color-contraction"
    )
    raw_polynomial = value["nc_polynomial"]
    if not isinstance(raw_polynomial, list):
        raise RecurrenceTemplateError("color Nc polynomial must be an array")
    polynomial: list[tuple[int, ExactComplexRationalV1]] = []
    for term in raw_polynomial:
        if not isinstance(term, list) or len(term) != 2:
            raise RecurrenceTemplateError(
                "color Nc polynomial entries must be [power, coefficient]"
            )
        polynomial.append(
            (
                _require_int("color Nc power", term[0]),
                _decode_ratio("color Nc coefficient", term[1]),
            )
        )
    output_representation = value["output_representation"]
    raw_witnesses = value["transition_witnesses"]
    if not isinstance(raw_witnesses, list):
        raise RecurrenceTemplateError("LC color transition witnesses must be an array")
    return ColorContractionTemplateV1(
        template_id=_require_nonempty("color template_id", value["template_id"]),
        rule_kind=_require_nonempty("color rule kind", value["rule_kind"]),
        input_representations=_decode_int_tuple(
            "color input representations", value["input_representations"]
        ),
        output_representation=(
            None
            if output_representation is None
            else _require_int("color output representation", output_representation)
        ),
        ordered_open_string_arity=_require_int(
            "color open string arity", value["ordered_open_string_arity"]
        ),
        exact_coefficient=_decode_ratio(
            "color coefficient", value["exact_coefficient"]
        ),
        nc_polynomial=tuple(polynomial),
        expression_digest=_require_sha256(
            "color expression digest", value["expression_digest"]
        ),
        transition_witnesses=tuple(
            _lc_color_witness_from_dict(witness) for witness in raw_witnesses
        ),
        semantic_digest=value["semantic_digest"],  # type: ignore[arg-type]
    )


def _lc_color_witness_from_dict(payload: object) -> LCColorTransitionWitnessV1:
    value = _require_mapping("LC color transition witness", payload)
    legacy_fields = frozenset(
        {
            "component_operation",
            "exact_factor",
            "input_permutation",
            "input_shape_kinds",
            "proof_digest",
            "provenance",
            "result_component_kind",
            "result_component_role",
            "result_shape_kind",
            "reverse_parent_mask",
        }
    )
    optional_port_fields = frozenset({"input_port_pairings", "result_port_bindings"})
    actual_fields = frozenset(value)
    unknown = actual_fields - legacy_fields - optional_port_fields
    missing = legacy_fields - actual_fields
    if unknown or missing:
        details: list[str] = []
        if unknown:
            details.append("unknown=" + ",".join(sorted(unknown)))
        if missing:
            details.append("missing=" + ",".join(sorted(missing)))
        raise RecurrenceTemplateError(
            "LC color transition witness has noncanonical fields "
            f"({'; '.join(details)})"
        )
    input_shapes = _decode_string_tuple(
        "LC color witness input shapes", value["input_shape_kinds"]
    )
    if len(input_shapes) != 2:
        raise RecurrenceTemplateError("LC color witness requires two input shapes")
    return LCColorTransitionWitnessV1(
        input_shape_kinds=(input_shapes[0], input_shapes[1]),  # type: ignore[arg-type]
        input_permutation=_decode_int_tuple(
            "LC color witness input permutation", value["input_permutation"]
        ),  # type: ignore[arg-type]
        reverse_parent_mask=_require_int(
            "LC color witness reverse-parent mask", value["reverse_parent_mask"]
        ),
        component_operation=_require_nonempty(
            "LC color witness operation", value["component_operation"]
        ),  # type: ignore[arg-type]
        result_component_kind=_decode_optional_string(
            "LC color witness result component", value["result_component_kind"]
        ),  # type: ignore[arg-type]
        result_component_role=_require_nonempty(
            "LC color witness result component role",
            value["result_component_role"],
        ),  # type: ignore[arg-type]
        result_shape_kind=_decode_optional_string(
            "LC color witness result shape", value["result_shape_kind"]
        ),  # type: ignore[arg-type]
        exact_factor=_decode_ratio("LC color witness factor", value["exact_factor"]),
        proof_digest=_require_sha256(
            "LC color witness proof digest", value["proof_digest"]
        ),
        provenance=_decode_string_pairs(
            "LC color witness provenance",
            value["provenance"],
        ),
        input_port_pairings=_decode_lc_color_port_pairings(
            value.get("input_port_pairings", [])
        ),
        result_port_bindings=_decode_lc_color_port_references(
            value.get("result_port_bindings", [])
        ),
    )


def _validate_lc_color_port_reference(value: object) -> None:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise RecurrenceTemplateError(
            "LC color port references must be two-element integer tuples"
        )
    parent_index, local_port_index = value
    if parent_index not in (0, 1) or local_port_index < 0:
        raise RecurrenceTemplateError(
            "LC color port reference is outside the two-parent nonnegative domain"
        )


def _decode_lc_color_port_references(value: object) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, list):
        raise RecurrenceTemplateError("LC color port references must be a JSON array")
    sequence = value
    result: list[tuple[int, int]] = []
    for item in sequence:
        port = _decode_int_tuple("LC color port reference", item)
        if len(port) != 2:
            raise RecurrenceTemplateError(
                "LC color port reference must contain two integers"
            )
        result.append((port[0], port[1]))
    return tuple(result)


def _decode_lc_color_port_pairings(
    value: object,
) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    if not isinstance(value, list):
        raise RecurrenceTemplateError("LC color port pairings must be a JSON array")
    sequence = value
    result: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for item in sequence:
        if not isinstance(item, list):
            raise RecurrenceTemplateError("LC color port pairing must be a JSON array")
        pair = item
        if len(pair) != 2:
            raise RecurrenceTemplateError(
                "LC color port pairing must contain two port references"
            )
        ports = _decode_lc_color_port_references(pair)
        result.append((ports[0], ports[1]))
    return tuple(result)


def _symmetry_from_dict(payload: Mapping[str, object]) -> SymmetryProofV1:
    fields = frozenset(
        {
            "template_id",
            "proof_algorithm",
            "subject_template_ids",
            "input_permutation",
            "exact_phase",
            "expression_digests",
            "witness_digest",
        }
    )
    value = _record_payload(
        "symmetry-proof template", payload, fields, "symmetry-proof"
    )
    return SymmetryProofV1(
        template_id=_require_nonempty("proof template_id", value["template_id"]),
        proof_algorithm=_require_nonempty("proof algorithm", value["proof_algorithm"]),
        subject_template_ids=_decode_string_tuple(
            "proof subjects", value["subject_template_ids"]
        ),
        input_permutation=_decode_int_tuple(
            "proof permutation", value["input_permutation"]
        ),
        exact_phase=_decode_ratio("proof phase", value["exact_phase"]),
        expression_digests=_decode_string_tuple(
            "proof expression digests", value["expression_digests"]
        ),
        witness_digest=_require_sha256("proof witness digest", value["witness_digest"]),
        semantic_digest=value["semantic_digest"],  # type: ignore[arg-type]
    )


def _runtime_helicity_contract_from_dict(
    payload: Mapping[str, object],
) -> RecurrenceRuntimeHelicityContractV1:
    fields = frozenset(
        {
            "template_id",
            "full_state_template_id",
            "variants",
            "proof_algorithm",
            "proof_digest",
        }
    )
    value = _record_payload(
        "runtime-helicity contract", payload, fields, "runtime-helicity-contract"
    )
    raw_variants = value["variants"]
    if not isinstance(raw_variants, list):
        raise RecurrenceTemplateError("runtime-helicity variants must be an array")
    variants: list[RecurrenceRuntimeHelicityVariantV1] = []
    variant_fields = frozenset(
        {
            "source_template_id",
            "source_state_template_id",
            "embedding_source_components",
            "embedding_factors",
            "projection_full_components",
            "proof_digest",
        }
    )
    for index, raw_variant in enumerate(raw_variants):
        variant = _require_mapping(f"runtime-helicity variant {index}", raw_variant)
        _require_exact_keys(
            f"runtime-helicity variant {index}", variant, variant_fields
        )
        raw_embedding = variant["embedding_source_components"]
        if not isinstance(raw_embedding, list) or any(
            item is not None and type(item) is not int for item in raw_embedding
        ):
            raise RecurrenceTemplateError(
                "runtime-helicity component embedding must contain integers or null"
            )
        variants.append(
            RecurrenceRuntimeHelicityVariantV1(
                source_template_id=_require_nonempty(
                    "runtime-helicity source template ID",
                    variant["source_template_id"],
                ),
                source_state_template_id=_require_nonempty(
                    "runtime-helicity source-state template ID",
                    variant["source_state_template_id"],
                ),
                embedding_source_components=tuple(raw_embedding),
                embedding_factors=_decode_ratio_array(
                    "runtime-helicity embedding factors",
                    variant["embedding_factors"],
                ),
                projection_full_components=_decode_int_tuple(
                    "runtime-helicity projection",
                    variant["projection_full_components"],
                ),
                proof_digest=_require_sha256(
                    "runtime-helicity variant proof digest",
                    variant["proof_digest"],
                ),
            )
        )
    return RecurrenceRuntimeHelicityContractV1(
        template_id=_require_nonempty(
            "runtime-helicity template ID", value["template_id"]
        ),
        full_state_template_id=_require_nonempty(
            "runtime-helicity full-state template ID",
            value["full_state_template_id"],
        ),
        variants=tuple(variants),
        proof_algorithm=_require_nonempty(
            "runtime-helicity proof algorithm", value["proof_algorithm"]
        ),
        proof_digest=_require_sha256(
            "runtime-helicity proof digest", value["proof_digest"]
        ),
        semantic_digest=value["semantic_digest"],  # type: ignore[arg-type]
    )


def _evaluator_from_dict(payload: Mapping[str, object]) -> EvaluatorBindingV1:
    fields = frozenset(
        {
            "resolver_key",
            "callable_kind",
            "prepared_kernel_id",
            "runtime_template",
            "contract_kind",
            "callable_signature",
            "input_state_template_ids",
            "output_state_template_id",
            "input_layout",
            "output_layout",
            "exact_expression_digests",
            "semantic_template_ids",
        }
    )
    value = _record_payload(
        "evaluator-binding template", payload, fields, "evaluator-binding"
    )
    return EvaluatorBindingV1(
        resolver_key=_require_nonempty("evaluator resolver_key", value["resolver_key"]),
        callable_kind=_require_nonempty(
            "evaluator callable_kind", value["callable_kind"]
        ),  # type: ignore[arg-type]
        prepared_kernel_id=(
            None
            if value["prepared_kernel_id"] is None
            else _require_int("prepared_kernel_id", value["prepared_kernel_id"])
        ),
        runtime_template=_decode_optional_string(
            "evaluator runtime_template", value["runtime_template"]
        ),
        contract_kind=value["contract_kind"],  # type: ignore[arg-type]
        callable_signature=_require_sha256(
            "callable_signature", value["callable_signature"]
        ),
        input_state_template_ids=_decode_string_tuple(
            "evaluator input states", value["input_state_template_ids"]
        ),
        output_state_template_id=_decode_optional_string(
            "evaluator output state", value["output_state_template_id"]
        ),
        input_layout=_decode_string_tuple(
            "evaluator input layout", value["input_layout"]
        ),
        output_layout=_decode_string_tuple(
            "evaluator output layout", value["output_layout"]
        ),
        exact_expression_digests=_decode_string_tuple(
            "evaluator expressions", value["exact_expression_digests"]
        ),
        semantic_template_ids=_decode_string_tuple(
            "evaluator semantic templates", value["semantic_template_ids"]
        ),
        semantic_digest=value["semantic_digest"],  # type: ignore[arg-type]
    )


# Short aliases match the names used by the design document while retaining the
# explicit version in persisted-contract code.
CatalogHeader = RecurrenceTemplateCatalogHeaderV1
ParameterTemplate = ParameterTemplateV1
CurrentStateTemplate = CurrentStateTemplateV1
SourceTemplate = SourceTemplateV1
QuantumFlowTemplate = QuantumFlowTemplateV1
TransitionTemplate = TransitionTemplateV1
PropagatorTemplate = PropagatorTemplateV1
ClosureTemplate = ClosureTemplateV1
ColorContractionTemplate = ColorContractionTemplateV1
SymmetryProof = SymmetryProofV1
RecurrenceRuntimeHelicityContract = RecurrenceRuntimeHelicityContractV1
EvaluatorBinding = EvaluatorBindingV1
RecurrenceTemplateCatalogV1 = RecurrenceTemplateCatalog

__all__ = [
    "RECURRENCE_TEMPLATE_ABI",
    "RECURRENCE_TEMPLATE_CANONICALIZATION_ABI",
    "RECURRENCE_TEMPLATE_EXACT_SCALAR_ABI",
    "SUPPORTED_SYMMETRY_PROOF_ALGORITHMS",
    "CatalogHeader",
    "ClosureTemplate",
    "ClosureTemplateV1",
    "ColorContractionTemplate",
    "ColorContractionTemplateV1",
    "CurrentStateTemplate",
    "CurrentStateTemplateV1",
    "EvaluatorBinding",
    "EvaluatorBindingV1",
    "EvaluatorCallableKind",
    "ExactComplexRationalV1",
    "FlavourFlowOperationV1",
    "LCColorComponentKindV1",
    "LCColorComponentOperationV1",
    "LCColorComponentRoleV1",
    "LCColorShapeKindV1",
    "LCColorSourceSeedOperationV1",
    "LCColorSourceSeedV1",
    "LCColorTransitionWitnessV1",
    "ParameterTemplate",
    "ParameterTemplateV1",
    "PropagatorTemplate",
    "PropagatorTemplateV1",
    "QuantumFlowTemplate",
    "QuantumFlowTemplateV1",
    "QuantumNumberFlowOperationV1",
    "RecurrenceRuntimeHelicityContract",
    "RecurrenceRuntimeHelicityContractV1",
    "RecurrenceRuntimeHelicityVariantV1",
    "RecurrenceTemplateCatalog",
    "RecurrenceTemplateCatalogHeaderV1",
    "RecurrenceTemplateCatalogV1",
    "RecurrenceTemplateError",
    "SourceTemplate",
    "SourceTemplateV1",
    "SymmetryProof",
    "SymmetryProofV1",
    "TransitionTemplate",
    "TransitionTemplateV1",
]

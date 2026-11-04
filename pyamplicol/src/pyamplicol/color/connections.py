# SPDX-License-Identifier: 0BSD
"""Exact, generation-independent ordered colour connections.

Tensors are literal products of open strings and traces of generators ``tau``
with ``tr(tau[a] tau[b]) = delta[a,b]``. Physical charges use
``t = tau / sqrt(2)``. Every emission therefore contributes one common inverse
sqrt(2), kept outside the rational tensor coefficients until the final overlap.
The existing exact trace/Fierz reducer supplies the final colour-index sum.

All legs are outgoing. Quark emission prepends a generator, antiquark emission
appends its negative, and gluon emission replaces ``tau[a]`` by
``[tau[a], tau[b]]``. A gluon splitting into a quark pair contracts its adjoint
index against ``t[a]``. Connections are ordered, and later operations may act
on previously emitted legs. No kinematic splitting factor is included.

There is deliberately no AmpliCol endpoint-pairing phase in a literal tensor.
A generation adapter must attach its basis convention as a tensor coefficient.
In particular, overlap matrices are not operator action matrices and must not
be multiplied to construct higher connections.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
from typing import TypeAlias

from .contraction_trace import _simplify_trace_terms_nc_power
from .contraction_types import NC

COLOR_CONNECTION_CONVENTION = "su3-literal-tau-physical-charge-all-outgoing-v1"


def _integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    return value


def _fraction(value: int | Fraction) -> Fraction:
    if not isinstance(value, (int, Fraction)) or isinstance(value, bool):
        raise TypeError("exact colour coefficients must be integers or Fractions")
    return Fraction(value)


@dataclass(frozen=True)
class ExactColorCoefficient:
    """A complex number with exact rational real and imaginary parts."""

    real: Fraction = Fraction(0)
    imag: Fraction = Fraction(0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "real", _fraction(self.real))
        object.__setattr__(self, "imag", _fraction(self.imag))

    def __bool__(self) -> bool:
        return bool(self.real or self.imag)

    def __complex__(self) -> complex:
        return complex(float(self.real), float(self.imag))

    def __add__(self, other: ExactColorCoefficient) -> ExactColorCoefficient:
        return ExactColorCoefficient(self.real + other.real, self.imag + other.imag)

    def __mul__(self, other: ExactColorCoefficient) -> ExactColorCoefficient:
        return ExactColorCoefficient(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real,
        )

    def conjugate(self) -> ExactColorCoefficient:
        return ExactColorCoefficient(self.real, -self.imag)

    def scaled(self, factor: int | Fraction) -> ExactColorCoefficient:
        factor = _fraction(factor)
        return ExactColorCoefficient(self.real * factor, self.imag * factor)


_ONE = ExactColorCoefficient(1)


@dataclass(frozen=True, order=True)
class ColorLeg:
    label: int
    representation: int

    def __post_init__(self) -> None:
        _integer(self.label, "colour leg label")
        _integer(self.representation, "colour representation")
        if self.representation not in (1, 3, -3, 8):
            raise ValueError("colour connections support only singlets, 3, -3 and 8")


def all_outgoing_color_leg(
    label: int, representation: int, *, incoming: bool = False
) -> ColorLeg:
    """Dualize incoming fundamental representations, without changing momenta.

    ``representation`` is the physical particle's signed representation before
    crossing. Adjoint and singlet representations are self-conjugate.
    """

    leg = ColorLeg(label, representation)
    if type(incoming) is not bool:
        raise TypeError("incoming must be a boolean")
    return (
        ColorLeg(label, -representation)
        if incoming and abs(representation) == 3
        else leg
    )


@dataclass(frozen=True, order=True)
class OpenColorString:
    fundamental_label: int
    adjoint_labels: tuple[int, ...]
    antifundamental_label: int

    def __post_init__(self) -> None:
        _integer(self.fundamental_label, "fundamental label")
        _integer(self.antifundamental_label, "antifundamental label")
        labels = tuple(self.adjoint_labels)
        for label in labels:
            _integer(label, "adjoint label")
        object.__setattr__(self, "adjoint_labels", labels)


def _canonical_trace(trace: tuple[int, ...]) -> tuple[int, ...]:
    return (
        min(trace[index:] + trace[:index] for index in range(len(trace)))
        if trace
        else ()
    )


@dataclass(frozen=True, order=True)
class ColorTensor:
    """An unordered product of open strings and cyclic (not reflected) traces."""

    open_strings: tuple[OpenColorString, ...] = ()
    traces: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        strings = tuple(self.open_strings)
        if any(not isinstance(string, OpenColorString) for string in strings):
            raise TypeError("open strings must be OpenColorString records")
        traces = tuple(tuple(trace) for trace in self.traces)
        for trace in traces:
            for label in trace:
                _integer(label, "trace adjoint label")
        object.__setattr__(self, "open_strings", tuple(sorted(strings)))
        object.__setattr__(
            self, "traces", tuple(sorted(_canonical_trace(t) for t in traces))
        )


@dataclass(frozen=True)
class EmitGluon:
    emitter_label: int
    emitted_label: int

    def __post_init__(self) -> None:
        _integer(self.emitter_label, "emitter label")
        _integer(self.emitted_label, "emitted label")


@dataclass(frozen=True)
class SplitGluon:
    parent_label: int
    quark_label: int
    antiquark_label: int

    def __post_init__(self) -> None:
        _integer(self.parent_label, "splitting parent label")
        _integer(self.quark_label, "splitting quark label")
        _integer(self.antiquark_label, "splitting antiquark label")


ColorEmission: TypeAlias = EmitGluon | SplitGluon


def _canonical_legs(legs: tuple[ColorLeg, ...]) -> tuple[ColorLeg, ...]:
    if any(not isinstance(leg, ColorLeg) for leg in legs):
        raise TypeError("connection legs must be ColorLeg records")
    if len({leg.label for leg in legs}) != len(legs):
        raise ValueError("connection leg labels must be unique")
    return tuple(sorted(legs))


@dataclass(frozen=True)
class ColorConnection:
    initial_legs: tuple[ColorLeg, ...]
    emissions: tuple[ColorEmission, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "initial_legs", _canonical_legs(tuple(self.initial_legs))
        )
        object.__setattr__(self, "emissions", tuple(self.emissions))
        _ = self.output_legs  # Validate parents, representations and fresh labels once.

    @property
    def order(self) -> int:
        return len(self.emissions)

    @cached_property
    def output_legs(self) -> tuple[ColorLeg, ...]:
        active = {leg.label: leg.representation for leg in self.initial_legs}
        seen = set(active)
        for emission in self.emissions:
            if isinstance(emission, EmitGluon):
                if active.get(emission.emitter_label) not in (3, -3, 8):
                    raise ValueError("gluon emitter must be an active coloured leg")
                additions = (ColorLeg(emission.emitted_label, 8),)
            elif isinstance(emission, SplitGluon):
                if active.get(emission.parent_label) != 8:
                    raise ValueError(
                        "quark-pair splitting parent must be an active gluon"
                    )
                additions = (
                    ColorLeg(emission.quark_label, 3),
                    ColorLeg(emission.antiquark_label, -3),
                )
                del active[emission.parent_label]
            else:
                raise TypeError("unknown colour emission record")
            labels = tuple(leg.label for leg in additions)
            if len(set(labels)) != len(labels) or seen.intersection(labels):
                raise ValueError(
                    "emitted colour labels must be fresh, including retired labels"
                )
            seen.update(labels)
            active.update((leg.label, leg.representation) for leg in additions)
        return tuple(ColorLeg(label, rep) for label, rep in sorted(active.items()))


@dataclass(frozen=True)
class TensorTerm:
    coefficient: ExactColorCoefficient
    tensor: ColorTensor
    nc_power: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.coefficient, ExactColorCoefficient):
            raise TypeError("tensor coefficients must be ExactColorCoefficient records")
        if not isinstance(self.tensor, ColorTensor):
            raise TypeError("tensor terms must contain ColorTensor records")
        _integer(self.nc_power, "tensor Nc power")


def _validate_tensor_legs(tensor: ColorTensor, legs: tuple[ColorLeg, ...]) -> None:
    labels = []
    for string in tensor.open_strings:
        labels.extend(
            (
                ColorLeg(string.fundamental_label, 3),
                ColorLeg(string.antifundamental_label, -3),
            )
        )
        labels.extend(ColorLeg(label, 8) for label in string.adjoint_labels)
    labels.extend(ColorLeg(label, 8) for trace in tensor.traces for label in trace)
    if tuple(sorted(labels)) != tuple(leg for leg in legs if leg.representation != 1):
        raise ValueError(
            "tensor indices do not match the typed external colour legs exactly once"
        )


def _combine_terms(terms: tuple[TensorTerm, ...]) -> tuple[TensorTerm, ...]:
    combined: dict[tuple[ColorTensor, int], ExactColorCoefficient] = {}
    for term in terms:
        if not isinstance(term, TensorTerm):
            raise TypeError("connected tensors require TensorTerm records")
        if any(len(trace) == 1 for trace in term.tensor.traces):
            continue
        empty_traces = sum(not trace for trace in term.tensor.traces)
        tensor = ColorTensor(
            term.tensor.open_strings, tuple(t for t in term.tensor.traces if t)
        )
        key = (tensor, term.nc_power + empty_traces)
        combined[key] = combined.get(key, ExactColorCoefficient()) + term.coefficient
    return tuple(
        TensorTerm(value, tensor, power)
        for (tensor, power), value in sorted(combined.items())
        if value
    )


@dataclass(frozen=True)
class ConnectedColorTensor:
    legs: tuple[ColorLeg, ...]
    terms: tuple[TensorTerm, ...]
    normalization_power: int = 0

    def __post_init__(self) -> None:
        legs = _canonical_legs(tuple(self.legs))
        terms = tuple(self.terms)
        power = _integer(
            self.normalization_power, "inverse sqrt(2) normalization power"
        )
        if power < 0:
            raise ValueError("normalization power must be nonnegative")
        for term in terms:
            if not isinstance(term, TensorTerm):
                raise TypeError("connected tensors require TensorTerm records")
            _validate_tensor_legs(term.tensor, legs)
        object.__setattr__(self, "legs", legs)
        object.__setattr__(self, "terms", _combine_terms(terms))


def _replace_string(
    tensor: ColorTensor, index: int, replacements: tuple[OpenColorString, ...]
) -> ColorTensor:
    return ColorTensor(
        tensor.open_strings[:index] + replacements + tensor.open_strings[index + 1 :],
        tensor.traces,
    )


def _emit(
    tensor: ColorTensor, emission: EmitGluon
) -> tuple[tuple[Fraction, ColorTensor], ...]:
    emitter, emitted = emission.emitter_label, emission.emitted_label
    for index, string in enumerate(tensor.open_strings):
        q, word, qbar = (
            string.fundamental_label,
            string.adjoint_labels,
            string.antifundamental_label,
        )
        if emitter == q:
            return (
                (
                    Fraction(1),
                    _replace_string(
                        tensor, index, (OpenColorString(q, (emitted, *word), qbar),)
                    ),
                ),
            )
        if emitter == qbar:
            return (
                (
                    Fraction(-1),
                    _replace_string(
                        tensor, index, (OpenColorString(q, (*word, emitted), qbar),)
                    ),
                ),
            )
        if emitter in word:
            position = word.index(emitter)
            before, after = word[:position], word[position + 1 :]
            return tuple(
                (
                    Fraction(sign),
                    _replace_string(
                        tensor,
                        index,
                        (OpenColorString(q, before + middle + after, qbar),),
                    ),
                )
                for sign, middle in ((1, (emitter, emitted)), (-1, (emitted, emitter)))
            )
    for index, trace in enumerate(tensor.traces):
        if emitter in trace:
            position = trace.index(emitter)
            before, after = trace[:position], trace[position + 1 :]
            return tuple(
                (
                    Fraction(sign),
                    ColorTensor(
                        tensor.open_strings,
                        (
                            *tensor.traces[:index],
                            before + middle + after,
                            *tensor.traces[index + 1 :],
                        ),
                    ),
                )
                for sign, middle in ((1, (emitter, emitted)), (-1, (emitted, emitter)))
            )
    raise ValueError("emitter has no tensor index")


def _split(
    tensor: ColorTensor, emission: SplitGluon
) -> tuple[tuple[Fraction, int, ColorTensor], ...]:
    parent, q, qbar = (
        emission.parent_label,
        emission.quark_label,
        emission.antiquark_label,
    )
    for index, string in enumerate(tensor.open_strings):
        if parent not in string.adjoint_labels:
            continue
        position = string.adjoint_labels.index(parent)
        before, after = (
            string.adjoint_labels[:position],
            string.adjoint_labels[position + 1 :],
        )
        return (
            (
                Fraction(1),
                0,
                _replace_string(
                    tensor,
                    index,
                    (
                        OpenColorString(string.fundamental_label, before, qbar),
                        OpenColorString(q, after, string.antifundamental_label),
                    ),
                ),
            ),
            (
                Fraction(-1),
                -1,
                _replace_string(
                    tensor,
                    index,
                    (
                        OpenColorString(
                            string.fundamental_label,
                            before + after,
                            string.antifundamental_label,
                        ),
                        OpenColorString(q, (), qbar),
                    ),
                ),
            ),
        )
    for index, trace in enumerate(tensor.traces):
        if parent not in trace:
            continue
        position = trace.index(parent)
        before, after = trace[:position], trace[position + 1 :]
        remaining = tensor.traces[:index] + tensor.traces[index + 1 :]
        return (
            (
                Fraction(1),
                0,
                ColorTensor(
                    (*tensor.open_strings, OpenColorString(q, after + before, qbar)),
                    remaining,
                ),
            ),
            (
                Fraction(-1),
                -1,
                ColorTensor(
                    (*tensor.open_strings, OpenColorString(q, (), qbar)),
                    (*remaining, before + after),
                ),
            ),
        )
    raise ValueError("splitting parent has no tensor index")


def apply_color_connection(
    tensor: ColorTensor,
    connection: ColorConnection,
    *,
    coefficient: ExactColorCoefficient = _ONE,
) -> ConnectedColorTensor:
    """Apply an ordered connection; at most two word terms arise per operation."""

    if not isinstance(tensor, ColorTensor) or not isinstance(
        connection, ColorConnection
    ):
        raise TypeError("apply_color_connection requires a tensor and a connection")
    _validate_tensor_legs(tensor, connection.initial_legs)
    terms = (TensorTerm(coefficient, tensor),)
    for emission in connection.emissions:
        transformed = []
        for term in terms:
            changes = (
                tuple(
                    (factor, 0, image) for factor, image in _emit(term.tensor, emission)
                )
                if isinstance(emission, EmitGluon)
                else _split(term.tensor, emission)
            )
            for factor, power, image in changes:
                transformed.append(
                    TensorTerm(
                        term.coefficient.scaled(factor), image, term.nc_power + power
                    )
                )
        terms = _combine_terms(tuple(transformed))
    return ConnectedColorTensor(connection.output_legs, terms, connection.order)


def _literal_tensor_overlap_nc_terms(
    left: ColorTensor, right: ColorTensor
) -> Mapping[int, Fraction]:
    traces = list(right.traces) + [tuple(reversed(trace)) for trace in left.traces]
    right_by_q = {string.fundamental_label: string for string in right.open_strings}
    left_by_qbar = {
        string.antifundamental_label: string for string in left.open_strings
    }
    visited: set[int] = set()
    for start in sorted(right_by_q):
        if start in visited:
            continue
        current = start
        trace: list[int] = []
        while current not in visited:
            visited.add(current)
            right_string = right_by_q[current]
            left_string = left_by_qbar[right_string.antifundamental_label]
            trace.extend(right_string.adjoint_labels)
            trace.extend(reversed(left_string.adjoint_labels))
            current = left_string.fundamental_label
        traces.append(tuple(trace))
    return _simplify_trace_terms_nc_power(((Fraction(1), 0, tuple(traces)),))


def contract_connected_tensor_nc_terms(
    left: ConnectedColorTensor, right: ConnectedColorTensor
) -> dict[int, ExactColorCoefficient]:
    """Return the exact Laurent coefficients of ``left.conjugate() * right``.

    The scalar overlap is sesquilinear, not forcibly real or positive. Matching
    physical bra/ket connection orders have even combined normalization power.
    Odd powers (unlike these correlators) need an irrational coefficient type
    and are rejected instead of silently rounded.
    """

    if left.legs != right.legs:
        raise ValueError(
            "bra and ket connections must have identical typed output legs"
        )
    power = left.normalization_power + right.normalization_power
    if power % 2:
        raise ValueError("bra/ket normalization has an odd inverse sqrt(2) power")
    result: dict[int, ExactColorCoefficient] = {}
    normalization = Fraction(1, 2 ** (power // 2))
    for left_term in left.terms:
        for right_term in right.terms:
            coefficient = (
                left_term.coefficient.conjugate() * right_term.coefficient
            ).scaled(normalization)
            for nc_power, factor in _literal_tensor_overlap_nc_terms(
                left_term.tensor, right_term.tensor
            ).items():
                exponent = nc_power + left_term.nc_power + right_term.nc_power
                result[exponent] = result.get(
                    exponent, ExactColorCoefficient()
                ) + coefficient.scaled(factor)
    return {power: coefficient for power, coefficient in result.items() if coefficient}


def evaluate_color_nc_terms(
    terms: Mapping[int, ExactColorCoefficient],
) -> ExactColorCoefficient:
    """Evaluate retained colour Laurent coefficients at physical ``Nc = 3``."""

    result = ExactColorCoefficient()
    for power, coefficient in terms.items():
        result = result + coefficient.scaled(Fraction(NC) ** power)
    return result


def contract_connected_tensors(
    left: ConnectedColorTensor, right: ConnectedColorTensor
) -> ExactColorCoefficient:
    """Sum the full connected colour overlap exactly at physical ``Nc = 3``."""

    return evaluate_color_nc_terms(contract_connected_tensor_nc_terms(left, right))


def color_connection_matrix_element(
    left_tensor: ColorTensor,
    left_connection: ColorConnection,
    right_tensor: ColorTensor,
    right_connection: ColorConnection,
    *,
    left_coefficient: ExactColorCoefficient = _ONE,
    right_coefficient: ExactColorCoefficient = _ONE,
) -> ExactColorCoefficient:
    """One exact connected matrix entry, before any amplitude/helicity sum."""

    if left_connection.initial_legs != right_connection.initial_legs:
        raise ValueError("bra and ket connections must start from identical typed legs")
    return contract_connected_tensors(
        apply_color_connection(
            left_tensor, left_connection, coefficient=left_coefficient
        ),
        apply_color_connection(
            right_tensor, right_connection, coefficient=right_coefficient
        ),
    )


__all__ = [
    "COLOR_CONNECTION_CONVENTION",
    "ColorConnection",
    "ColorEmission",
    "ColorLeg",
    "ColorTensor",
    "ConnectedColorTensor",
    "EmitGluon",
    "ExactColorCoefficient",
    "OpenColorString",
    "SplitGluon",
    "TensorTerm",
    "all_outgoing_color_leg",
    "apply_color_connection",
    "color_connection_matrix_element",
    "contract_connected_tensor_nc_terms",
    "contract_connected_tensors",
    "evaluate_color_nc_terms",
]

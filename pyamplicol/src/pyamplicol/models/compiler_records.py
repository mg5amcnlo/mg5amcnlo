# SPDX-License-Identifier: 0BSD
"""Canonical UFO record parsing and expression resolution."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from fractions import Fraction
from typing import cast

from .._internal.physics.symbols import ModelSymbolRegistry
from . import compiler_symbolica as _sym
from .contracts import (
    DEFAULT_FEYNMAN_PROPAGATOR_SOURCE,
    MODEL_SUPPLIED_PROPAGATOR_SOURCE,
    PROPAGATOR_SOURCE_FIELD,
    CompiledCouplingOrder,
    CompiledCouplingRecord,
    CompiledParameterRecord,
    CompiledParticleRecord,
    CompiledPropagatorRecord,
    validate_quantum_number_flow,
)

_COMPILER_OWNED_PARTICLE_FIELDS = frozenset({"auxiliary_kind", "component_dimension"})


def cast_int_tuple3(value: object) -> tuple[int, int, int]:
    values = tuple(int(item) for item in _sequence(value))
    if len(values) != 3:
        raise ValueError("oriented kernel source legs must have length three")
    return values[0], values[1], values[2]


def _order(item: Mapping[str, object]) -> CompiledCouplingOrder:
    return CompiledCouplingOrder(
        name=str(item["name"]),
        expansion_order=int(item["expansion_order"]),
        hierarchy=int(item["hierarchy"]),
    )


def _parameter(
    item: Mapping[str, object],
    *,
    model_symbols: ModelSymbolRegistry | None = None,
) -> CompiledParameterRecord:
    expression = _optional_string(item.get("expression"))
    if expression is not None and model_symbols is not None:
        expression = model_symbols.expression_string(expression)
    value = _optional_pair(item.get("value"))
    return CompiledParameterRecord(
        name=str(item["name"]),
        nature=str(item["nature"]),
        parameter_type=str(item["parameter_type"]),
        value=value,
        expression=expression,
        resolved_expression=(
            expression
            if expression is not None
            else _numeric_expression(value or (0.0, 0.0)).to_canonical_string()
        ),
        lhablock=_optional_string(item.get("lhablock")),
        lhacode=tuple(int(value) for value in _sequence(item.get("lhacode"))),
    )


def _particle(
    item: Mapping[str, object],
) -> CompiledParticleRecord:
    name = str(item["name"])
    supplied_metadata = sorted(_COMPILER_OWNED_PARTICLE_FIELDS.intersection(item))
    if supplied_metadata:
        fields = ", ".join(supplied_metadata)
        raise ValueError(
            f"particle {name!r} supplies compiler-owned metadata in "
            f"untrusted input: {fields}"
        )
    return _particle_record(item)


def _particle_record(item: Mapping[str, object]) -> CompiledParticleRecord:
    name = str(item["name"])
    raw_charge = item.get("charge", 0.0)
    # Loader 1.0 exports exact rational strings; retain them in charge-flow
    # proofs while keeping the public numerical charge a float.
    exact_charge = Fraction(raw_charge) if isinstance(raw_charge, str) else None
    charge = float(exact_charge if exact_charge is not None else raw_charge)
    quantum_numbers = item.get("quantum_numbers")
    if quantum_numbers is None:
        charge_flow = (
            str(exact_charge)
            if exact_charge is not None
            else _exact_float_expression(charge)
        )
        quantum_numbers = (("electric_charge", charge_flow),)
    return CompiledParticleRecord(
        name=name,
        antiname=str(item["antiname"]),
        pdg_code=int(item["pdg_code"]),
        spin=int(item["spin"]),
        color=int(item["color"]),
        mass=str(item["mass"]),
        width=str(item["width"]),
        charge=charge,
        quantum_numbers=validate_quantum_number_flow(
            quantum_numbers,
            context=f"particle {name!r}",
        ),
        ghost_number=int(item.get("ghost_number", 0)),
        propagating=bool(item.get("propagating", True)),
        goldstoneboson=bool(item.get("goldstoneboson", False)),
        propagator=_optional_string(item.get("propagator")),
        component_dimension=cast(int | None, item.get("component_dimension")),
        auxiliary_kind=_optional_string(item.get("auxiliary_kind")),
    )


def _exact_float_expression(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("particle electric charge must be finite")
    numerator, denominator = value.as_integer_ratio()
    if denominator == 1:
        return str(numerator)
    return f"{numerator}/{denominator}"


def _coupling(
    item: Mapping[str, object],
    *,
    model_symbols: ModelSymbolRegistry | None = None,
) -> CompiledCouplingRecord:
    expression = str(item["expression"])
    if model_symbols is not None:
        expression = model_symbols.expression_string(expression)
    return CompiledCouplingRecord(
        name=str(item["name"]),
        expression=expression,
        resolved_expression=expression,
        value=_optional_pair(item.get("value")),
        orders=_orders(item.get("orders")),
    )


def _propagator(
    item: Mapping[str, object],
    particles: Sequence[CompiledParticleRecord],
) -> CompiledPropagatorRecord:
    particle_name = str(item["particle"])
    particle = next(
        (candidate for candidate in particles if candidate.name == particle_name),
        None,
    )
    name = str(item["name"])
    if particle is None:
        raise ValueError(
            f"propagator {name!r} refers to unknown particle {particle_name!r}"
        )
    if particle.propagator != name:
        raise ValueError(
            f"propagator {name!r} is not linked from particle {particle_name!r}"
        )
    source = str(item.get(PROPAGATOR_SOURCE_FIELD, ""))
    if source not in {
        DEFAULT_FEYNMAN_PROPAGATOR_SOURCE,
        MODEL_SUPPLIED_PROPAGATOR_SOURCE,
    }:
        raise ValueError(
            f"propagator {name!r} has no valid generation-time source metadata"
        )
    return CompiledPropagatorRecord(
        name=name,
        particle=particle_name,
        numerator=str(item["numerator"]),
        denominator=str(item["denominator"]),
        custom=source == MODEL_SUPPLIED_PROPAGATOR_SOURCE,
    )


def _orders(value: object) -> tuple[tuple[str, int], ...]:
    result = []
    for pair in _sequence(value):
        values = _sequence(pair)
        if len(values) != 2:
            raise ValueError("coupling order must be [name, value]")
        result.append((str(values[0]), int(values[1])))
    return tuple(result)


def _pair(value: object) -> tuple[float, float]:
    pair = _sequence(value)
    if len(pair) != 2:
        raise ValueError("complex value must be [real, imaginary]")
    return float(pair[0]), float(pair[1])


def _optional_pair(value: object) -> tuple[float, float] | None:
    return None if value is None else _pair(value)


def _optional_string(value: object) -> str | None:
    return None if value is None else str(value)


def _resolve_parameter_records(
    records: Sequence[CompiledParameterRecord],
    model_symbols: ModelSymbolRegistry,
) -> tuple[CompiledParameterRecord, ...]:
    by_name = {record.name: record for record in records}
    parameter_symbols = {name: model_symbols.symbol(name) for name in by_name}
    resolved: dict[str, _sym.Expression] = {}
    active: list[str] = []

    def resolve(name: str) -> _sym.Expression:
        if name in resolved:
            return resolved[name]
        if name in active:
            cycle = " -> ".join((*active, name))
            raise ValueError(f"cyclic UFO parameter definitions: {cycle}")
        record = by_name[name]
        if record.nature == "external":
            resolved[name] = parameter_symbols[name]
            return resolved[name]
        active.append(name)
        expression = (
            model_symbols.expression(record.expression)
            if record.expression is not None
            else _numeric_expression(record.value or (0.0, 0.0))
        )
        expression_symbols = set(expression.get_all_symbols(False))
        for dependency, symbol in parameter_symbols.items():
            if dependency == name or symbol not in expression_symbols:
                continue
            expression = expression.replace(symbol, resolve(dependency))
        active.pop()
        resolved[name] = expression
        return resolved[name]

    return tuple(
        replace(record, resolved_expression=resolve(record.name).to_canonical_string())
        for record in records
    )


def _resolve_coupling_records(
    records: Sequence[CompiledCouplingRecord],
    parameters: Sequence[CompiledParameterRecord],
    model_symbols: ModelSymbolRegistry,
) -> tuple[CompiledCouplingRecord, ...]:
    replacements = {
        model_symbols.symbol(parameter.name): _sym.E(parameter.resolved_expression)
        for parameter in parameters
        if parameter.nature != "external"
    }
    result = []
    for record in records:
        expression = model_symbols.expression(record.expression)
        expression_symbols = set(expression.get_all_symbols(False))
        for symbol, replacement_expression in replacements.items():
            if symbol in expression_symbols:
                expression = expression.replace(symbol, replacement_expression)
        result.append(
            replace(
                record,
                resolved_expression=expression.to_canonical_string(),
            )
        )
    return tuple(result)


def _replace_evaluator_constants(expression: _sym.Expression) -> _sym.Expression:
    # Only call at the f64 evaluator boundary. The model's symbolic records
    # must retain Pi: rounding it before resolving G and G**2 independently
    # breaks gauge identities that higher-precision evaluation cannot recover.
    return expression.replace(_sym.E("pi"), _sym.E(repr(math.pi)))


def _numeric_expression(value: tuple[float, float]) -> _sym.Expression:
    real, imaginary = value
    return _sym.E(repr(real)) + _sym.E(repr(imaginary)) * _sym.E("1𝑖")  # noqa: RUF001


def _sequence(value: object) -> list[object]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _mappings(value: object) -> list[dict[str, object]]:
    result = []
    for item in _sequence(value):
        if not isinstance(item, dict):
            raise ValueError("compiled model list entry must be an object")
        result.append({str(key): element for key, element in item.items()})
    return result

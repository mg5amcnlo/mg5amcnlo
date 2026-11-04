# SPDX-License-Identifier: 0BSD
"""Expression and input-contract helpers for prepared kernel catalogs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .._internal.physics.symbols import symbols
from . import compiler_symbolica as _sym
from .base import Model
from .expressions import _as_expression
from .prepared_catalog import (
    PreparedContractKind,
    PreparedInputRole,
    PreparedKernelCatalogError,
    PreparedKernelInput,
)


def formal_components(model: Model, role: str, count: int) -> tuple[Any, ...]:
    symbol_role = role.replace("-", "_")
    return tuple(
        symbols.model_owned(model.name, f"prepared::{symbol_role}::c{index}")
        for index in range(count)
    )


def formal_model_parameter(model: Model, name: str, index: int) -> Any:
    del name
    return symbols.model_owned(
        model.name,
        f"prepared::model_parameter::p{index}",
    )


def component_descriptors(
    model: Model,
    role: PreparedInputRole,
    values: Sequence[Any],
) -> tuple[PreparedKernelInput, ...]:
    del model
    return tuple(
        PreparedKernelInput(
            role=role,
            component=index,
            symbol=value.to_canonical_string(),
        )
        for index, value in enumerate(values)
    )


def coupling_descriptor(model: Model, value: Any, index: int) -> PreparedKernelInput:
    del model
    return PreparedKernelInput(
        role="coupling-real" if index == 0 else "coupling-imag",
        component=0,
        symbol=value.to_canonical_string(),
    )


def model_parameter_descriptor(
    model: Model,
    value: Any,
    name: str,
    index: int,
) -> PreparedKernelInput:
    del model
    return PreparedKernelInput(
        role="model-parameter",
        component=0,
        symbol=value.to_canonical_string(),
        model_parameter_name=name,
        model_parameter_index=index,
    )


def used_input_descriptors(
    expressions: Sequence[Any],
    descriptors: Sequence[PreparedKernelInput],
) -> tuple[PreparedKernelInput, ...]:
    used_symbols = {
        symbol.to_canonical_string()
        for expression in expressions
        for symbol in _as_expression(expression).get_all_symbols(False)
    }
    return tuple(
        descriptor for descriptor in descriptors if descriptor.symbol in used_symbols
    )


def canonical_expressions(
    expressions: Sequence[Any],
    *,
    context: str,
) -> tuple[str, ...]:
    result: list[str] = []
    for index, value in enumerate(expressions):
        try:
            expression = _as_expression(value).expand()
            canonical = str(expression.to_canonical_string())
        except Exception as error:
            raise PreparedKernelCatalogError(
                f"cannot canonicalize {context} output {index}: {error}"
            ) from error
        lowered = canonical.lower()
        if not canonical or any(
            marker in lowered
            for marker in ("indeterminate", "nan", "infinity", "complexinf")
        ):
            raise PreparedKernelCatalogError(
                f"{context} output {index} is not a finite exact expression"
            )
        result.append(canonical)
    if not result:
        raise PreparedKernelCatalogError(f"{context} produced no outputs")
    return tuple(result)


def canonical_json(value: Mapping[str, object]) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise PreparedKernelCatalogError(
            f"prepared kernel contract is not canonical JSON: {error}"
        ) from error


def proves_homogeneous_complex_linearity(
    expressions: Sequence[Any],
    current_components: Sequence[Any],
) -> bool:
    """Certify exact complex linearity in one prepared current argument.

    The reconstruction identity is checked symbolically, rather than inferred
    from a propagator role. This rejects affine offsets, nonlinear terms, and
    conjugate dependence while leaving unrelated momenta and parameters exact.
    """

    if not expressions or not current_components:
        return False
    _sym._ensure_symbolica()
    currents = tuple(_as_expression(value) for value in current_components)
    if len({value.to_canonical_string() for value in currents}) != len(currents):
        return False
    try:
        basis_values: list[tuple[Any, ...]] = []
        for basis_index in range(len(currents)):
            replacements = [
                _sym.Replacement(current, int(index == basis_index))
                for index, current in enumerate(currents)
            ]
            basis_values.append(tuple(replacements))

        for raw_expression in expressions:
            expression = _as_expression(raw_expression)
            residual = expression
            for current, replacements in zip(currents, basis_values, strict=True):
                basis_output = expression.replace_multiple(replacements)
                residual -= current * basis_output
            canonical = residual.expand().to_canonical_string()
            if canonical != "0":
                canonical = residual.together().cancel().expand().to_canonical_string()
            if canonical != "0":
                return False
    except Exception:
        return False
    return True


def proves_independent_current_block_eligibility(
    contract_kind: PreparedContractKind,
    expressions: Sequence[str],
    inputs: Sequence[PreparedKernelInput],
) -> bool:
    """Certify that independent calls may be packed into block lanes.

    The first block variant deliberately targets pure current-to-current vertex
    maps. Momentum-, coupling-, and model-parameter-bearing contracts retain
    their scalar evaluator until a separate layout is specified for them.
    """

    if contract_kind != "vertex" or not expressions or not inputs:
        return False
    if any(item.role not in {"left-current", "right-current"} for item in inputs):
        return False
    if not {item.role for item in inputs}.issuperset(
        {"left-current", "right-current"}
    ):
        return False
    declared = {item.symbol for item in inputs}
    if len(declared) != len(inputs):
        return False
    try:
        used = {
            symbol.to_canonical_string()
            for expression in expressions
            for symbol in _sym.E(expression).get_all_symbols(False)
            if not symbol.is_constant()
        }
    except Exception:
        return False
    return used.issubset(declared)

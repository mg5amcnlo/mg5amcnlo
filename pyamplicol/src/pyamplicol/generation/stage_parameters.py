# SPDX-License-Identifier: 0BSD
"""Stage-local parameter layouts and expression utility functions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .._internal.physics.parameters import ParamBuilder
from .._internal.physics.symbols import symbols
from ..models._physics_ir import ContractionIR
from ..models.base import Model
from ..models.expressions import _exact_complex_expression
from .contracts import StageCompilationInput
from .contracts import (
    runtime_coupling_parameter_names as _runtime_coupling_parameter_names,
)
from .dag_types import GenericDAG
from .stage_types import GenericStageInputComponent, _StageLocalInputs

_EXPRESSION_PREVIEW_LIMIT = 512
_MODEL_FUNCTION_INLINE_MAX_BYTES = 1024


def _contract_components(
    contraction_ir: ContractionIR,
    left: Sequence[Any],
    right: Sequence[Any],
) -> Any:
    dimension = len(contraction_ir.coefficients)
    if len(left) != dimension or len(right) != dimension:
        raise ValueError(
            f"contraction {contraction_ir.name!r} requires {dimension} components, "
            f"got {len(left)} and {len(right)}"
        )
    return _sum_contraction_terms(
        _apply_contraction_coefficient(
            coefficient,
            left[index] * right[index],
        )
        for index, coefficient in enumerate(contraction_ir.coefficients)
    )


def _project_components(
    contraction_ir: ContractionIR,
    components: Sequence[Any],
) -> Any:
    dimension = len(contraction_ir.coefficients)
    if len(components) != dimension:
        raise ValueError(
            f"projection {contraction_ir.name!r} requires {dimension} components, "
            f"got {len(components)}"
        )
    return _sum_contraction_terms(
        _apply_contraction_coefficient(coefficient, components[index])
        for index, coefficient in enumerate(contraction_ir.coefficients)
    )


def _apply_contraction_coefficient(
    coefficient: tuple[float, float],
    expression: Any,
) -> Any:
    if coefficient == (1.0, 0.0):
        return expression
    if coefficient == (-1.0, 0.0):
        return -expression
    if isinstance(expression, int | float | complex):
        return complex(*coefficient) * expression
    return _exact_complex_expression(complex(*coefficient)) * expression


def _sum_contraction_terms(terms: Iterable[Any]) -> Any:
    iterator = iter(terms)
    result = next(iterator)
    for term in iterator:
        result = result + term
    return result


def _stage_input_momentum_slot_ids(
    interactions: Sequence[dict[str, Any]],
) -> tuple[int, ...]:
    slot_ids: set[int] = set()
    for interaction in interactions:
        momentum_slots = _dict(interaction["momentum_slots"])
        for key in ("left", "right", "result"):
            slot_ids.add(int(momentum_slots[key]))
    return tuple(sorted(slot_ids))


def _current_stage_model_parameter_records(
    model: Model,
    model_parameter_records: Sequence[dict[str, Any]],
    *,
    dag: GenericDAG,
    interactions: Sequence[dict[str, Any]],
    interaction_ids: Sequence[int],
    output_slots_by_current: Mapping[int, Sequence[dict[str, Any]]],
    current_slots: Mapping[int, dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    used_names = _coupling_parameter_names_used_by_records(interactions)
    if not interactions:
        processed_signatures: set[tuple[int, tuple[int, ...], tuple[float, ...]]] = (
            set()
        )
        for interaction_id in interaction_ids:
            interaction = dag.interactions[int(interaction_id)]
            signature = (
                int(interaction.vertex_kind),
                interaction.vertex_particles,
                interaction.coupling,
            )
            if signature in processed_signatures:
                continue
            processed_signatures.add(signature)
            used_names.update(
                name
                for name in _runtime_coupling_parameter_names(
                    interaction.vertex_kind,
                    interaction.vertex_particles,
                    interaction.coupling,
                    model=model,
                )
                if isinstance(name, str)
            )
    for current_id, slots in output_slots_by_current.items():
        if not any(str(slot["variant"]) == "propagated" for slot in slots):
            continue
        current_slot = current_slots[int(current_id)]
        used_names.update(
            _particle_model_parameter_names(
                model,
                int(current_slot["particle_id"]),
            )
        )
    return _filter_model_parameter_records(model_parameter_records, used_names)


def _amplitude_stage_model_parameter_records(
    model_parameter_records: Sequence[dict[str, Any]],
    *,
    roots: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    used_names = _coupling_parameter_names_used_by_records(roots)
    return _filter_model_parameter_records(model_parameter_records, used_names)


def _coupling_parameter_names_used_by_records(
    records: Sequence[dict[str, Any]],
) -> set[str]:
    used_names: set[str] = set()
    for record in records:
        names = record.get("coupling_parameter_names")
        if not isinstance(names, list):
            continue
        for name in names:
            if isinstance(name, str):
                used_names.add(name)
    return used_names


def _particle_model_parameter_names(model: Model, pdg: int) -> tuple[str, ...]:
    runtime_names = getattr(model, "runtime_parameter_names_for_particle", None)
    if callable(runtime_names):
        return tuple(str(name) for name in runtime_names(int(pdg)))
    try:
        particle = model.particle(pdg)
    except KeyError:
        return ()
    return (
        f"particle.{int(particle.pdg)}.mass",
        f"particle.{int(particle.pdg)}.width",
    )


def _filter_model_parameter_records(
    model_parameter_records: Sequence[dict[str, Any]],
    used_names: set[str],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        record
        for record in sorted(
            model_parameter_records,
            key=lambda item: int(item["parameter_index"]),
        )
        if str(record.get("runtime_name", record["name"])) in used_names
        and not (
            record.get("complex_domain") == "real"
            and record.get("complex_component") == "imag"
        )
        and not (
            record.get("complex_domain") == "imaginary"
            and record.get("complex_component") == "real"
        )
    )


def _logical_model_parameter_symbols(
    model_parameter_records: Sequence[dict[str, Any]],
    slot_symbols: Mapping[str, Any],
) -> dict[str, Any]:
    logical_symbols: dict[str, Any] = {}
    complex_components: dict[str, dict[str, Any]] = {}
    complex_domains: dict[str, str] = {}
    for record in model_parameter_records:
        slot_name = str(record["name"])
        symbol = slot_symbols[slot_name]
        runtime_name = record.get("runtime_name")
        component = record.get("complex_component")
        if isinstance(runtime_name, str) and component in {"real", "imag"}:
            complex_components.setdefault(runtime_name, {})[str(component)] = symbol
            domain = str(record.get("complex_domain", "complex"))
            previous = complex_domains.setdefault(runtime_name, domain)
            if previous != domain:
                raise ValueError(
                    f"runtime model parameter {runtime_name!r} has conflicting domains"
                )
        else:
            logical_symbols[slot_name] = symbol
    for runtime_name, components in complex_components.items():
        domain = complex_domains[runtime_name]
        if "real" not in components and domain == "imaginary":
            components["real"] = 0.0
        if "imag" not in components and domain == "real":
            components["imag"] = 0.0
        if set(components) != {"real", "imag"}:
            raise ValueError(
                f"runtime model parameter {runtime_name!r} is missing a real "
                "or imaginary slot"
            )
        logical_symbols[runtime_name] = components["real"] + 1j * components["imag"]
    return logical_symbols


def _stage_local_inputs(
    *,
    value_slot_ids: Sequence[int],
    momentum_slot_ids: Sequence[int],
    value_slots: Mapping[int, dict[str, Any]],
    momentum_slots: Mapping[int, dict[str, Any]],
    global_value_component_count: int,
    global_momentum_parameter_count: int,
    model_parameter_records: Sequence[dict[str, Any]],
) -> _StageLocalInputs:
    builder = ParamBuilder()
    input_components: list[GenericStageInputComponent] = []
    value_symbols: dict[int, tuple[Any, ...]] = {}
    momentum_symbols: dict[int, tuple[Any, ...]] = {}
    model_parameter_slot_symbols: dict[str, Any] = {}

    value_spans = tuple(
        (
            int(value_slot_id),
            int(value_slots[int(value_slot_id)]["component_start"]),
            int(value_slots[int(value_slot_id)]["component_stop"]),
        )
        for value_slot_id in value_slot_ids
    )
    momentum_spans = tuple(
        (
            int(momentum_slot_id),
            int(momentum_slots[int(momentum_slot_id)]["component_start"]),
            int(momentum_slots[int(momentum_slot_id)]["component_stop"]),
        )
        for momentum_slot_id in momentum_slot_ids
    )
    sorted_model_parameter_records = tuple(
        sorted(
            model_parameter_records,
            key=lambda item: int(item["parameter_index"]),
        )
    )
    value_parameter_count = sum(stop - start for _, start, stop in value_spans)
    momentum_parameter_count = sum(stop - start for _, start, stop in momentum_spans)
    model_parameter_count = len(sorted_model_parameter_records)
    parameter_count = (
        value_parameter_count + momentum_parameter_count + model_parameter_count
    )
    parameter_symbols = (
        builder.add_parameter_list(
            ("artifact_schema_v3_stage", "inputs"),
            parameter_count,
            role="generic_stage_input_storage",
        )
        if parameter_count
        else ()
    )
    real_parameter_start = value_parameter_count
    builder.real_valued_inputs.extend(range(real_parameter_start, parameter_count))
    parameter_cursor = 0

    for value_slot_id, start, stop in value_spans:
        length = stop - start
        local_symbols = parameter_symbols[parameter_cursor : parameter_cursor + length]
        value_symbols[value_slot_id] = local_symbols
        parameter_start = len(input_components)
        for component, global_component in enumerate(range(start, stop)):
            input_components.append(
                GenericStageInputComponent(
                    kind="value",
                    source_id=int(value_slot_id),
                    component=component,
                    global_component=global_component,
                    parameter_index=parameter_start + component,
                    real_valued=False,
                )
            )
        parameter_cursor += length

    for momentum_slot_id, start, stop in momentum_spans:
        length = stop - start
        local_symbols = parameter_symbols[parameter_cursor : parameter_cursor + length]
        momentum_symbols[momentum_slot_id] = local_symbols
        parameter_start = len(input_components)
        for component, local_component in enumerate(range(start, stop)):
            input_components.append(
                GenericStageInputComponent(
                    kind="momentum",
                    source_id=int(momentum_slot_id),
                    component=component,
                    global_component=global_value_component_count + local_component,
                    parameter_index=parameter_start + component,
                    real_valued=True,
                )
            )
        parameter_cursor += length

    model_parameter_global_start = (
        global_value_component_count + global_momentum_parameter_count
    )
    for record in sorted_model_parameter_records:
        name = str(record["name"])
        parameter_index = int(record["parameter_index"])
        symbol = parameter_symbols[parameter_cursor]
        model_parameter_slot_symbols[name] = symbol
        input_components.append(
            GenericStageInputComponent(
                kind="model_parameter",
                source_id=parameter_index,
                component=0,
                global_component=model_parameter_global_start + parameter_index,
                parameter_index=len(input_components),
                real_valued=True,
            )
        )
        parameter_cursor += 1

    if parameter_cursor != parameter_count:
        raise RuntimeError("stage-local parameter layout cursor mismatch")

    model_parameter_symbols = _logical_model_parameter_symbols(
        sorted_model_parameter_records,
        model_parameter_slot_symbols,
    )

    return _StageLocalInputs(
        parameter_symbols=tuple(parameter_symbols),
        input_components=tuple(input_components),
        value_symbols=value_symbols,
        momentum_symbols=momentum_symbols,
        model_parameter_symbols=model_parameter_symbols,
        value_parameter_count=value_parameter_count,
        momentum_parameter_count=momentum_parameter_count,
        model_parameter_count=model_parameter_count,
        real_valued_inputs=tuple(int(index) for index in builder.real_valued_inputs),
    )


def _global_stage_inputs(
    *,
    parameter_symbols: Sequence[Any],
    value_symbols: Sequence[Any],
    momentum_symbols: Sequence[Any],
    model_parameter_symbols: Mapping[str, Any],
    value_parameter_count: int,
    momentum_parameter_count: int,
    model_parameter_count: int,
    real_valued_inputs: Sequence[int],
) -> _StageLocalInputs:
    return _StageLocalInputs(
        parameter_symbols=tuple(parameter_symbols),
        input_components=(),
        value_symbols=tuple(value_symbols),
        momentum_symbols=tuple(momentum_symbols),
        model_parameter_symbols=dict(model_parameter_symbols),
        value_parameter_count=int(value_parameter_count),
        momentum_parameter_count=int(momentum_parameter_count),
        model_parameter_count=int(model_parameter_count),
        real_valued_inputs=tuple(int(index) for index in real_valued_inputs),
    )


def _parameter_builder(schema: dict[str, Any]) -> ParamBuilder:
    layout = _dict(schema["parameter_layout"])
    builder = ParamBuilder()
    value_component_count = int(layout["value_component_count"])
    if value_component_count:
        builder.add_parameter_list(
            ("artifact_schema_v3", "values"),
            value_component_count,
            role="generic_value_storage",
        )
    momentum_parameter_count = int(layout["momentum_parameter_count"])
    if momentum_parameter_count:
        builder.add_parameter_list(
            ("artifact_schema_v3", "momenta"),
            momentum_parameter_count,
            role="generic_momentum_storage",
            real_valued=True,
        )
    model_parameter_count = int(layout.get("model_parameter_count", 0))
    if model_parameter_count:
        builder.add_parameter_list(
            ("artifact_schema_v3", "model_parameters"),
            model_parameter_count,
            role="generic_model_parameters",
            real_valued=True,
        )
    return builder


def _manifest_model(manifest: StageCompilationInput | GenericDAG) -> Model:
    if isinstance(manifest, StageCompilationInput):
        return manifest.model
    raise ValueError("GenericDAG stage compilation requires an explicit model")


def _model_symbolica_functions(
    model: Model,
) -> tuple[tuple[Any, tuple[Any, ...], Any], ...]:
    getter = getattr(model, "symbolica_function_definitions", None)
    if not callable(getter):
        return ()
    definitions = getter()
    if not isinstance(definitions, Mapping):
        raise TypeError("model Symbolica function definitions must be a mapping")
    return tuple(
        (function, tuple(arguments), body)
        for (function, arguments), body in definitions.items()
    )


def _specialize_stage_symbolica_functions(
    outputs: Sequence[Any],
    definitions: Sequence[tuple[Any, tuple[Any, ...], Any]],
) -> tuple[
    tuple[Any, ...],
    tuple[tuple[Any, tuple[Any, ...], Any], ...],
]:
    """Inline model-kernel calls before constructing a stage evaluator."""

    output_expressions = tuple(outputs)
    function_definitions = tuple(definitions)
    if not output_expressions or not function_definitions:
        return output_expressions, ()

    from symbolica import Replacement

    replacements: list[Any] = []
    for definition_index, (function, arguments, body) in enumerate(
        function_definitions
    ):
        if int(body.get_byte_size()) > _MODEL_FUNCTION_INLINE_MAX_BYTES:
            continue
        wildcards = tuple(
            symbols.inline_function_wildcard(
                definition_index,
                argument_index,
            )
            for argument_index in range(len(arguments))
        )
        pattern = function(*wildcards)
        replacement = body
        for argument, wildcard in zip(arguments, wildcards, strict=True):
            replacement = replacement.replace(
                argument,
                wildcard,
                allow_new_wildcards_on_rhs=True,
            )
        replacements.append(
            Replacement(
                pattern,
                replacement,
                allow_new_wildcards_on_rhs=True,
            )
        )

    rewritten = tuple(
        expression.replace_multiple(replacements, repeat=True)
        for expression in output_expressions
    )

    function_names = {function for function, _arguments, _body in function_definitions}
    required = {
        symbol
        for expression in rewritten
        for symbol in _expression_symbols(expression, enter_functions=True)
        if symbol in function_names
    }
    while True:
        dependencies = {
            symbol
            for function, _arguments, body in function_definitions
            if function in required
            for symbol in _expression_symbols(body, enter_functions=True)
            if symbol in function_names
        }
        expanded = required | dependencies
        if expanded == required:
            break
        required = expanded
    retained = tuple(
        definition for definition in function_definitions if definition[0] in required
    )
    return rewritten, retained


def _expression_symbols(
    expression: Any,
    *,
    enter_functions: bool,
) -> set[Any]:
    getter = getattr(expression, "get_all_symbols", None)
    if not callable(getter):
        return set()
    return set(getter(enter_functions))


def _value_components(
    slot: dict[str, Any],
    value_symbols: Sequence[Any] | Mapping[int, tuple[Any, ...]],
) -> tuple[Any, ...]:
    if isinstance(value_symbols, Mapping):
        return tuple(value_symbols[int(slot["value_slot_id"])])
    start = int(slot["component_start"])
    stop = int(slot["component_stop"])
    return tuple(value_symbols[index] for index in range(start, stop))


def _momentum_components(
    key: int,
    momentum_symbols: Sequence[Any] | Mapping[int, tuple[Any, ...]],
    momentum_slots: dict[int, dict[str, Any]],
    *,
    by_slot_id: bool = False,
) -> tuple[Any, ...]:
    slot = (
        momentum_slots[key]
        if by_slot_id
        else _momentum_slot_by_mask(momentum_slots, key)
    )
    if isinstance(momentum_symbols, Mapping):
        return tuple(momentum_symbols[int(slot["momentum_slot_id"])])
    start = int(slot["component_start"])
    stop = int(slot["component_stop"])
    return tuple(momentum_symbols[index] for index in range(start, stop))


def _momentum_slot_by_mask(
    momentum_slots: dict[int, dict[str, Any]],
    momentum_mask: int,
) -> dict[str, Any]:
    for slot in momentum_slots.values():
        if int(slot["momentum_mask"]) == momentum_mask:
            return slot
    raise ValueError(f"no momentum slot for mask {momentum_mask}")


def _value_slots_by_id(schema: dict[str, Any]) -> dict[int, dict[str, Any]]:
    storage = _dict(schema["value_storage"])
    return {
        int(slot["value_slot_id"]): _dict(slot)
        for slot in _list(storage["value_slots"])
    }


def _current_slots_by_id(schema: dict[str, Any]) -> dict[int, dict[str, Any]]:
    storage = _dict(schema["current_storage"])
    return {
        int(slot["current_id"]): _dict(slot) for slot in _list(storage["current_slots"])
    }


def _momentum_slots_by_id(schema: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(slot["momentum_slot_id"]): _dict(slot)
        for slot in _list(schema["momentum_slots"])
    }


def _sum_components(left: Sequence[Any], right: Sequence[Any]) -> tuple[Any, ...]:
    if len(left) != len(right):
        raise ValueError(f"component dimensions differ: {len(left)} != {len(right)}")
    return tuple(left[index] + right[index] for index in range(len(left)))


def _coupling(value: object) -> tuple[Any, Any]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError("coupling metadata must have two entries")
    return value[0], value[1]


def _runtime_coupling(
    record: Mapping[str, Any],
    model_parameter_symbols: Mapping[str, Any],
) -> tuple[Any, Any]:
    values = list(_coupling(record.get("coupling")))
    names = record.get("coupling_parameter_names")
    if isinstance(names, list):
        for index, name in enumerate(names[: len(values)]):
            if isinstance(name, str) and name in model_parameter_symbols:
                values[index] = model_parameter_symbols[name]
    return values[0], values[1]


def _expression_previews(expressions: Sequence[Any]) -> tuple[str, ...]:
    return tuple(_preview_expression(expression) for expression in expressions[:4])


def _preview_expression(expression: Any) -> str:
    text = str(expression)
    if len(text) <= _EXPRESSION_PREVIEW_LIMIT:
        return text
    return text[:_EXPRESSION_PREVIEW_LIMIT] + "...<truncated>"


def _dict(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("expected JSON object")
    return value


def _list(value: object) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError("expected JSON array")
    return value

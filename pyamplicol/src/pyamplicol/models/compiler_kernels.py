# SPDX-License-Identifier: 0BSD
"""Oriented-kernel fusion and component expression lowering."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from itertools import product

from .._internal.physics.symbols import ModelSymbolRegistry, symbols
from . import compiler_symbolica as _sym
from .compiler_records import _replace_evaluator_constants, _sequence
from .compiler_tensor_ordering import (
    OrderedComponents,
    identity_ordering_for_materialized_axes,
)
from .contracts import (
    CompiledOrientedKernel,
    CompiledParameterRecord,
    CompiledParticleRecord,
    CompiledVertexTerm,
)


def _function_arguments(source: str, head_suffix: str) -> tuple[tuple[str, ...], ...]:
    """Extract balanced canonical arguments for one fully qualified function."""

    marker = head_suffix + "("
    result: list[tuple[str, ...]] = []
    cursor = 0
    while True:
        marker_start = source.find(marker, cursor)
        if marker_start < 0:
            break
        open_index = marker_start + len(head_suffix)
        depth = 0
        brace_depth = 0
        argument_start = open_index + 1
        arguments: list[str] = []
        index = open_index
        while index < len(source):
            character = source[index]
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0:
                    arguments.append(source[argument_start:index])
                    result.append(tuple(arguments))
                    cursor = index + 1
                    break
            elif character == "{":
                brace_depth += 1
            elif character == "}":
                brace_depth -= 1
            elif character == "," and depth == 1 and brace_depth == 0:
                arguments.append(source[argument_start:index])
                argument_start = index + 1
            index += 1
        else:
            raise ValueError(f"unbalanced canonical function {head_suffix}")
    return tuple(result)


def _permutation_sign(
    actual: tuple[int, ...],
    canonical: tuple[int, ...],
) -> int:
    if sorted(actual) != sorted(canonical):
        raise ValueError(
            f"color-factor indices {actual} do not match local orientation {canonical}"
        )
    positions = {value: index for index, value in enumerate(canonical)}
    permutation = tuple(positions[value] for value in actual)
    inversions = sum(
        permutation[left] > permutation[right]
        for left in range(len(permutation))
        for right in range(left + 1, len(permutation))
    )
    return -1 if inversions % 2 else 1


def _fuse_oriented_kernels(
    kernels: Sequence[CompiledOrientedKernel],
    *,
    model_name: str,
) -> tuple[CompiledOrientedKernel, ...]:
    model_symbols = symbols.model(model_name)
    groups: dict[tuple[object, ...], list[CompiledOrientedKernel]] = {}
    for kernel in kernels:
        key = (
            kernel.vertex,
            kernel.particles,
            kernel.source_particle_legs,
            kernel.coupling_orders,
            kernel.color_source,
            kernel.color_expression,
            kernel.lc_color_normalization_power,
            kernel.input_ordering_ids,
            kernel.output_ordering_id,
        )
        groups.setdefault(key, []).append(kernel)

    fused: list[CompiledOrientedKernel] = []
    for members in groups.values():
        kind = len(fused)
        first = members[0]
        remapped_components = [
            tuple(
                _remap_kernel_symbols(
                    _sym.E(component),
                    old_kind=member.kind,
                    new_kind=kind,
                    model_symbols=model_symbols,
                )
                for component in member.component_expressions
            )
            for member in members
        ]
        components = tuple(
            _canonicalize_oriented_kernel_component(
                sum(
                    (
                        remapped[index]
                        * symbols.derived_coupling(model_name, member.term_id)
                        for member, remapped in zip(
                            members,
                            remapped_components,
                            strict=True,
                        )
                    ),
                    _sym.E("0"),
                )
            )
            for index in range(len(remapped_components[0]))
        )
        coupling_expression = "1"
        fused.append(
            CompiledOrientedKernel(
                kind=kind,
                term_id=first.term_id,
                vertex=first.vertex,
                particles=first.particles,
                source_particle_legs=first.source_particle_legs,
                component_expressions=tuple(
                    component.to_canonical_string() for component in components
                ),
                coupling_expression=coupling_expression,
                coupling_orders=first.coupling_orders,
                runtime_parameters=tuple(
                    sorted({f"derived_coupling_{member.term_id}" for member in members})
                ),
                color_source=first.color_source,
                color_expression=first.color_expression,
                lc_color_normalization_power=first.lc_color_normalization_power,
                term_ids=tuple(
                    term_id
                    for member in members
                    for term_id in (member.term_ids or (member.term_id,))
                ),
                input_ordering_ids=first.input_ordering_ids,
                output_ordering_id=first.output_ordering_id,
            )
        )
    return tuple(fused)


def _canonicalize_oriented_kernel_component(
    expression: _sym.Expression,
) -> _sym.Expression:
    """Cancel numeric sums, factor couplings, and group primitive inputs."""

    return expression.expand_num().collect_factors().collect_horner()


def _remap_kernel_symbols(
    expression: _sym.Expression,
    *,
    old_kind: int,
    new_kind: int,
    model_symbols: ModelSymbolRegistry,
    swap_sides: bool = False,
) -> _sym.Expression:
    source = expression.to_canonical_string()
    kernel_symbols = {
        (side, momentum_marker == "momentum_", int(index))
        for side, momentum_marker, index in re.findall(
            rf"kernel_{old_kind}_(left|right)_(momentum_)?([0-9]+)",
            source,
        )
    }
    replacements: list[_sym.Replacement] = []
    for side, is_momentum, index in kernel_symbols:
        target_side = ("right" if side == "left" else "left") if swap_sides else side
        if is_momentum:
            replacements.append(
                _sym.Replacement(
                    model_symbols.kernel_momentum(old_kind, side, index),
                    model_symbols.kernel_momentum(new_kind, target_side, index),
                )
            )
        else:
            replacements.append(
                _sym.Replacement(
                    model_symbols.kernel_component(old_kind, side, index),
                    model_symbols.kernel_component(new_kind, target_side, index),
                )
            )
    return expression.replace_multiple(replacements) if replacements else expression


def _replace_expression_symbols(
    expression: _sym.Expression,
    substitutions: Mapping[_sym.Expression, _sym.Expression],
) -> _sym.Expression:
    result = expression
    for source, target in substitutions.items():
        result = result.replace(source, target)
    return result


def _oriented_component_expressions(
    term: CompiledVertexTerm,
    particle_by_name: Mapping[str, CompiledParticleRecord],
    *,
    left_leg: int,
    right_leg: int,
    result_leg: int,
    kind: int,
    model_symbols: ModelSymbolRegistry,
    use_transverse_massless_yang_mills: bool = False,
) -> OrderedComponents:
    library = _sym.TensorLibrary.hep_lib_atom()
    expression = _sym.E(term.lorentz_expression)
    particles = tuple(particle_by_name[name] for name in term.particles)
    left_symbols = _component_symbols(
        kind,
        "left",
        particles[left_leg].spin,
        model_symbols=model_symbols,
    )
    right_symbols = _component_symbols(
        kind,
        "right",
        particles[right_leg].spin,
        model_symbols=model_symbols,
    )
    expression *= _input_tensor_expression(
        library,
        kind=kind,
        side="left",
        spin=particles[left_leg].spin,
        leg=left_leg + 1,
        components=left_symbols,
        model_symbols=model_symbols,
    )
    expression *= _input_tensor_expression(
        library,
        kind=kind,
        side="right",
        spin=particles[right_leg].spin,
        leg=right_leg + 1,
        components=right_symbols,
        model_symbols=model_symbols,
    )
    left_momentum = tuple(
        model_symbols.kernel_momentum(kind, "left", component) for component in range(4)
    )
    right_momentum = tuple(
        model_symbols.kernel_momentum(kind, "right", component)
        for component in range(4)
    )
    result_momentum = tuple(
        -(left_momentum[component] + right_momentum[component])
        for component in range(4)
    )
    momentum_by_leg = {
        left_leg: left_momentum,
        right_leg: right_momentum,
        result_leg: result_momentum,
    }
    minkowski = _sym.Representation.mink(4)
    for leg, momentum in momentum_by_leg.items():
        library.register(
            _sym.Tensor.dense(
                _sym.TensorName(model_symbols.ufo_momentum_tensor_name(leg + 1))(
                    minkowski
                ),
                momentum,
            )
        )
    axis_labels = _spin_axis_labels(particles[result_leg].spin, result_leg + 1)
    # Plain expressions already carry the correct empty scalar interface.
    network = _sym.TensorNetwork(
        _sym.as_tensor(expression) if axis_labels else expression, library
    )
    network.execute(library=library)
    result = network.result_tensor(library)
    ordered_result = _ordered_dense_tensor_components(
        result,
        axis_labels,
    )
    result_components = ordered_result.values
    expected_dimension = _spin_dimension(particles[result_leg].spin)
    if len(result_components) != expected_dimension:
        raise ValueError(
            f"oriented kernel {term.vertex}/{term.id} produced "
            f"{len(result_components)} "
            f"components for spin {particles[result_leg].spin}, expected "
            f"{expected_dimension}"
        )
    components = tuple(
        _replace_evaluator_constants(_as_expression(component))
        for component in result_components
    )
    if all(particle.spin == 3 for particle in particles):
        compact = _compact_yang_mills_three_vector_components(
            left_leg=left_leg,
            right_leg=right_leg,
            result_leg=result_leg,
            left_components=left_symbols,
            right_components=right_symbols,
            momentum_by_leg=momentum_by_leg,
        )
        scale = _equivalent_component_scale(components, compact)
        if scale is not None:
            if use_transverse_massless_yang_mills:
                compact = _transverse_yang_mills_three_vector_components(
                    left_components=left_symbols,
                    right_components=right_symbols,
                    left_momentum=left_momentum,
                    right_momentum=right_momentum,
                )
            components = tuple(scale * component for component in compact)
    return OrderedComponents(
        ordering=ordered_result.ordering,
        values=tuple(component.to_canonical_string() for component in components),
    )


def _compact_yang_mills_three_vector_components(
    *,
    left_leg: int,
    right_leg: int,
    result_leg: int,
    left_components: Sequence[_sym.Expression],
    right_components: Sequence[_sym.Expression],
    momentum_by_leg: Mapping[int, Sequence[_sym.Expression]],
) -> tuple[_sym.Expression, ...]:
    """Return the compact canonical Yang-Mills three-vector contraction.

    The caller retains this form only after proving algebraic equivalence to the
    fully materialized spenso tensor. This makes the optimization independent
    of UFO Lorentz names and source-expression layout.
    """

    components_by_leg = {
        left_leg: tuple(left_components),
        right_leg: tuple(right_components),
    }
    terms = (
        ((0, 1), 0, 1, 2),
        ((0, 2), 2, 0, 1),
        ((1, 2), 1, 2, 0),
    )
    outputs: list[_sym.Expression] = []
    for component in range(4):
        output = _sym.E("0")
        for (
            metric_legs,
            positive_momentum_leg,
            negative_momentum_leg,
            vector_leg,
        ) in terms:
            momentum = tuple(
                momentum_by_leg[positive_momentum_leg][index]
                - momentum_by_leg[negative_momentum_leg][index]
                for index in range(4)
            )
            if result_leg == vector_leg:
                output += momentum[component] * _minkowski_dot(
                    components_by_leg[metric_legs[0]],
                    components_by_leg[metric_legs[1]],
                )
                continue
            other_metric_leg = (
                metric_legs[1] if metric_legs[0] == result_leg else metric_legs[0]
            )
            output += components_by_leg[other_metric_leg][component] * _minkowski_dot(
                momentum,
                components_by_leg[vector_leg],
            )
        outputs.append(output)
    return tuple(outputs)


def _transverse_yang_mills_three_vector_components(
    *,
    left_components: Sequence[_sym.Expression],
    right_components: Sequence[_sym.Expression],
    left_momentum: Sequence[_sym.Expression],
    right_momentum: Sequence[_sym.Expression],
) -> tuple[_sym.Expression, ...]:
    """Return the transverse Berends-Giele massless gauge current.

    The full Yang-Mills contraction differs only by terms proportional to each
    parent current's self-momentum contraction. AmpliCol removes those terms
    for massless adjoint gauge currents. Applying the same tensor-derived
    reduction preserves its fixed-width recursion convention without relying
    on model-specific particle or Lorentz names.
    """

    left = tuple(left_components)
    right = tuple(right_components)
    left_p = tuple(left_momentum)
    right_p = tuple(right_momentum)
    dot = _minkowski_dot(left, right)
    left_dot_right_p = _minkowski_dot(left, right_p)
    right_dot_left_p = _minkowski_dot(right, left_p)
    return tuple(
        dot * (left_p[index] - right_p[index])
        + 2 * (left_dot_right_p * right[index] - right_dot_left_p * left[index])
        for index in range(4)
    )


def _equivalent_component_scale(
    materialized: Sequence[_sym.Expression],
    compact: Sequence[_sym.Expression],
) -> _sym.Expression | None:
    if len(materialized) != len(compact) or not materialized:
        return None
    if any(
        frozenset(
            symbol.to_canonical_string() for symbol in dense.get_all_symbols(False)
        )
        != frozenset(
            symbol.to_canonical_string() for symbol in candidate.get_all_symbols(False)
        )
        for dense, candidate in zip(materialized, compact, strict=True)
    ):
        return None
    for scale in (_sym.E("1"), _sym.E("-1")):
        if all(
            (dense - scale * candidate).expand().to_canonical_string() == "0"
            for dense, candidate in zip(materialized, compact, strict=True)
        ):
            return scale
    # The compact Yang-Mills convention is only certified for an exact sign.
    # Retaining the fully materialized tensor is always correct for every other
    # normalization.  In particular, do not derive a general ratio here:
    # Symbolica's polynomial cancellation aborts the process for float atoms.
    return None


def _is_compile_time_zero_parameter(
    name: str,
    parameters: Mapping[str, CompiledParameterRecord],
) -> bool:
    if name.upper() == "ZERO":
        return True
    parameter = parameters.get(name)
    if parameter is None or parameter.nature != "internal":
        return False
    expression = _sym.E(parameter.resolved_expression)
    if expression.get_all_symbols(False):
        return False
    try:
        return complex(expression.evaluate({})) == 0.0
    except (TypeError, ValueError):
        return False


def _is_single_structure_constant(expression: str) -> bool:
    return (
        expression.count("::f(") == 1
        and len(_function_arguments(expression, "::f")) == 1
        and "::t(" not in expression.lower()
        and "::d(" not in expression.lower()
    )


def _minkowski_dot(
    left: Sequence[_sym.Expression],
    right: Sequence[_sym.Expression],
) -> _sym.Expression:
    if len(left) != 4 or len(right) != 4:
        raise ValueError("Minkowski dot products require four components")
    return left[0] * right[0] - sum(
        (left[index] * right[index] for index in range(1, 4)),
        _sym.E("0"),
    )


def _input_tensor_expression(
    library: _sym.TensorLibrary,
    *,
    kind: int,
    side: str,
    spin: int,
    leg: int,
    components: Sequence[_sym.Expression],
    model_symbols: ModelSymbolRegistry,
) -> _sym.Expression:
    representations = _spin_representations(spin)
    if not representations:
        if len(components) != 1:
            raise ValueError("scalar current must have exactly one component")
        return components[0]
    name = _sym.TensorName(model_symbols.kernel_tensor_name(kind, side))
    library.register(_sym.Tensor.dense(name(*representations), components))
    slots = _spin_slots(spin, leg)
    return name(*slots).to_expression()


def _spin_representations(spin: int) -> tuple[_sym.Representation, ...]:
    minkowski = _sym.Representation.mink(4)
    if spin in {-1, 1}:
        return ()
    if spin == 2:
        return (_sym.Representation.bis(4),)
    if spin == 3:
        return (minkowski,)
    if spin == 5:
        return (minkowski, minkowski)
    raise ValueError(f"unsupported UFO spin code {spin}")


def _spin_slots(spin: int, leg: int):
    representations = _spin_representations(spin)
    if spin == 2:
        return (representations[0](f"ufo_s_1_{leg}"),)
    if spin == 3:
        return (representations[0](f"ufo_l_1_{leg}"),)
    if spin == 5:
        return (
            representations[0](f"ufo_l_1_{leg}"),
            representations[1](f"ufo_l_2_{leg}"),
        )
    return ()


def _spin_axis_labels(spin: int, leg: int) -> tuple[str, ...]:
    if spin == 2:
        return (f"ufo_s_1_{leg}",)
    if spin == 3:
        return (f"ufo_l_1_{leg}",)
    if spin == 5:
        return (f"ufo_l_1_{leg}", f"ufo_l_2_{leg}")
    return ()


def _ordered_dense_tensor_components(
    tensor: object,
    expected_axis_labels: Sequence[str],
) -> OrderedComponents:
    """Flatten a dense tensor in explicit physical-index order."""

    tensor = tensor.to_dense()
    expected = tuple(str(label) for label in expected_axis_labels)
    if not expected:
        if len(tensor) != 1:
            raise ValueError("scalar tensor result unexpectedly has open indices")
        return OrderedComponents(
            ordering=identity_ordering_for_materialized_axes((), ()),
            values=tuple(tensor[index] for index in range(len(tensor))),
        )

    # Component coordinates follow the tensor's logical axes, not its canonical
    # structure signature, which can put the same axes in a different order.
    arguments = tuple(slot.to_expression() for slot in tensor.axes)
    actual: list[str] = []
    for argument in arguments:
        source = argument.to_canonical_string()
        labels = re.findall(r"ufo_[sl]_[0-9]+_[0-9]+", source)
        if len(labels) != 1:
            raise ValueError(
                "tensor result has an unrecognized physical index: " + source
            )
        actual.append(labels[0])
    if len(set(actual)) != len(actual) or set(actual) != set(expected):
        raise ValueError(
            "tensor result index labels do not match the requested physical order: "
            f"actual={actual}, expected={list(expected)}"
        )

    positions = tuple(actual.index(label) for label in expected)
    shape = tuple(int(dimension) for dimension in tensor.shape)
    if len(shape) != len(actual) or math.prod(shape) != len(tensor):
        raise ValueError(
            "tensor result does not cover the complete Cartesian component grid"
        )
    by_coordinates: dict[tuple[int, ...], object] = {}
    for flat_index, coordinates in enumerate(product(*(range(size) for size in shape))):
        key = tuple(int(coordinates[position]) for position in positions)
        by_coordinates[key] = tensor[flat_index]
    ordered_shape = tuple(shape[position] for position in positions)
    canonical_coordinates = product(*(range(size) for size in ordered_shape))
    ordering = identity_ordering_for_materialized_axes(
        expected,
        ordered_shape,
    )
    return OrderedComponents(
        ordering=ordering,
        values=tuple(by_coordinates[key] for key in canonical_coordinates),
    )


def _spin_dimension(spin: int) -> int:
    return {-1: 1, 1: 1, 2: 4, 3: 4, 5: 16}[spin]


def _component_symbols(
    kind: int,
    side: str,
    spin: int,
    *,
    model_symbols: ModelSymbolRegistry,
) -> tuple[_sym.Expression, ...]:
    return tuple(
        model_symbols.kernel_component(kind, side, component)
        for component in range(_spin_dimension(spin))
    )


def _lc_color_normalization_power(source: str) -> int:
    """Count normalized non-Abelian tensors in one UFO color monomial."""

    expression = _sym.E(source)
    return sum(
        len(list(expression.match(_sym.E(pattern))))
        for pattern in (
            "UFO::T(a_,b_,c_)",
            "UFO::f(a_,b_,c_)",
            "UFO::d(a_,b_,c_)",
        )
    )


def _as_expression(value: object) -> _sym.Expression:
    _sym._ensure_symbolica()
    if isinstance(value, _sym.Expression):
        return value
    raise TypeError(
        f"spenso kernel component is not an Expression: {type(value).__name__}"
    )


def cast_tuple3(value: object) -> tuple[str, str, str]:
    values = tuple(str(item) for item in _sequence(value))
    if len(values) != 3:
        raise ValueError("oriented kernel particles must have length three")
    return values[0], values[1], values[2]

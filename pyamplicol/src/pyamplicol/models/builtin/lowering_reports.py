# SPDX-License-Identifier: 0BSD
"""Built-in-model diagnostics for symbolic recursion lowering."""

from __future__ import annotations

import time
from collections import Counter
from typing import Any

from .lowering_tensor import (
    _build_auxiliary_tensor_probe,
    _build_color_probe,
    _clean_symbolica_string,
    _current_dimension,
    _current_momentum_currents,
    _GraphTensorExpressionBuilder,
    _preview_expression,
    _propagator_lowering_ready,
    _register_parametric_current_momenta,
    _register_parametric_source_currents,
    _source_current_count,
    _source_currents,
)
from .lowering_types import (
    RecursionLoweringPlan,
    SymbolicLoweringReport,
    TensorNetworkBlueprint,
    VertexLoweringReport,
    VertexLoweringStep,
)
from .model import BuiltinSMModel
from .symbols import symbols

_MAX_RECURSION_EXPRESSION_PREVIEW = 4096


def build_symbolic_lowering_report(
    model: BuiltinSMModel,
    graph: Any | None = None,
) -> SymbolicLoweringReport:
    """Exercise the real Symbolica tensor hooks used by ME lowering.

    This is intentionally a small, deterministic probe: it validates that the
    model-owned auxiliary four-gluon tensors are registered in spenso and that
    symbolic color simplification is available. It is not the full ME evaluator.
    """

    tensor_probe = _build_auxiliary_tensor_probe(model)
    color_probe = _build_color_probe()
    recursion_plan = None if graph is None else _build_recursion_plan(graph)
    vertex_lowering = (
        None if graph is None else _build_vertex_lowering_report(model, graph)
    )
    tensor_network_blueprint = (
        None if graph is None else _build_tensor_network_blueprint(model, graph)
    )
    full_me_tensor_network_ready = (
        tensor_network_blueprint.full_me_tensor_network_ready
        if tensor_network_blueprint is not None
        else False
    )
    return SymbolicLoweringReport(
        tensor_library="TensorLibrary.hep_lib_atom",
        tensor_network_probe=tensor_probe,
        color_algebra_probe=color_probe,
        recursion_plan=recursion_plan,
        vertex_lowering=vertex_lowering,
        tensor_network_blueprint=tensor_network_blueprint,
        full_me_tensor_network_ready=full_me_tensor_network_ready,
    )


def _build_recursion_plan(graph: Any) -> RecursionLoweringPlan:
    assignments = tuple(
        _interaction_assignment(interaction) for interaction in graph.interactions
    )
    amplitudes = tuple(
        symbols.amplitude(
            _current_atom(left),
            _current_atom(right),
        )
        for left, right in graph.amplitudes
    )
    assignment_sum = _sum_expressions(assignments)
    amplitude_sum = _sum_expressions(amplitudes)
    plan_expression = symbols.matrix_element_plan(assignment_sum, amplitude_sum)
    vertex_kind_counts = Counter(
        int(interaction.vertex_kind) for interaction in graph.interactions
    )
    expression = _clean_symbolica_string(str(plan_expression))
    tensor_route_vertex_kinds = tuple(
        kind for kind in (1, 2, 3) if vertex_kind_counts.get(kind, 0) > 0
    )
    first_assignments = tuple(
        _clean_symbolica_string(str(assignment)) for assignment in assignments[:5]
    )
    return RecursionLoweringPlan(
        engine="symbolica",
        current_count=len(graph.currents),
        interaction_count=len(graph.interactions),
        amplitude_count=len(graph.amplitudes),
        source_current_count=_source_current_count(graph),
        vertex_kind_counts=tuple(sorted(vertex_kind_counts.items())),
        tensor_route_vertex_kinds=tensor_route_vertex_kinds,
        color_order=tuple(int(index) for index in graph.color_order),
        expression=_preview_expression(expression),
        expression_length=len(expression),
        expression_truncated=len(expression) > _MAX_RECURSION_EXPRESSION_PREVIEW,
        first_assignments=first_assignments,
    )


def _interaction_assignment(interaction: Any) -> Any:
    return symbols.assignment(
        _current_atom(interaction.result),
        symbols.vertex(
            int(interaction.vertex_kind),
            _current_atom(interaction.left),
            _current_atom(interaction.right),
            _number(interaction.coupling[0]),
            _number(interaction.coupling[1]),
        ),
    )


def _current_atom(current: Any) -> Any:
    return symbols.current(
        int(current.pdg),
        _label_atom(tuple(int(label) for label in current.external_labels)),
        int(current.chirality),
    )


def _label_atom(labels: tuple[int, ...]) -> Any:
    return symbols.label(labels)


def _number(value: float) -> Any:
    from symbolica import Expression

    return Expression.num(value)


def _sum_expressions(expressions: tuple[Any, ...]) -> Any:
    from symbolica import Expression

    total = Expression.num(0)
    for expression in expressions:
        total = total + expression
    return total


def _current_key_tuple(current: Any) -> tuple[int, tuple[int, ...], int]:
    return (
        int(current.pdg),
        tuple(int(label) for label in current.external_labels),
        int(current.chirality),
    )


def _build_vertex_lowering_report(
    model: BuiltinSMModel,
    graph: Any,
) -> VertexLoweringReport:
    steps = tuple(
        _vertex_lowering_step(model, index, interaction)
        for index, interaction in enumerate(graph.interactions, start=1)
    )
    backend_counts = Counter(step.backend for step in steps)
    ready_kind_counts = Counter(
        step.vertex_kind for step in steps if step.full_tensor_network_ready
    )
    pending_kind_counts = Counter(
        step.vertex_kind for step in steps if not step.full_tensor_network_ready
    )
    tensor_names = tuple(
        sorted({tensor_name for step in steps for tensor_name in step.tensor_names})
    )
    return VertexLoweringReport(
        total_interactions=len(steps),
        full_tensor_network_ready=bool(steps)
        and all(step.full_tensor_network_ready for step in steps),
        backend_counts=tuple(sorted(backend_counts.items())),
        ready_vertex_kind_counts=tuple(sorted(ready_kind_counts.items())),
        pending_vertex_kind_counts=tuple(sorted(pending_kind_counts.items())),
        tensor_names=tensor_names,
        first_steps=steps[:8],
    )


def _vertex_lowering_step(
    model: BuiltinSMModel,
    index: int,
    interaction: Any,
) -> VertexLoweringStep:
    rule = model.vertex_lowering_rule(int(interaction.vertex_kind))
    return VertexLoweringStep(
        index=index,
        vertex_kind=int(interaction.vertex_kind),
        backend=rule.backend,
        tensor_names=rule.tensor_names,
        expression_head=rule.expression_head,
        full_tensor_network_ready=rule.full_tensor_network_ready,
        result_current=_clean_symbolica_string(str(_current_atom(interaction.result))),
        left_current=_clean_symbolica_string(str(_current_atom(interaction.left))),
        right_current=_clean_symbolica_string(str(_current_atom(interaction.right))),
    )


def _build_tensor_network_blueprint(
    model: BuiltinSMModel,
    graph: Any,
) -> TensorNetworkBlueprint:
    from symbolica.community.tensor import TensorNetwork, as_tensor

    max_interactions_to_build = 96
    max_interactions_to_execute = 40
    vertex_lowering = _build_vertex_lowering_report(model, graph)
    ready_interactions = sum(
        count for _, count in vertex_lowering.ready_vertex_kind_counts
    )
    pending_interactions = sum(
        count for _, count in vertex_lowering.pending_vertex_kind_counts
    )
    placeholder_vertex_kinds = tuple(
        kind for kind, _ in vertex_lowering.pending_vertex_kind_counts
    )
    registered_tensor_names = vertex_lowering.tensor_names
    source_currents = _source_currents(graph)
    momentum_currents = _current_momentum_currents(graph)
    current_leaf_count = len(source_currents)
    source_parameter_count = sum(
        _current_dimension(current) for current in source_currents
    )
    momentum_parameter_count = 4 * len(momentum_currents)
    parametric_parameter_count = source_parameter_count + momentum_parameter_count
    propagator_ready = _propagator_lowering_ready(graph)
    full_ready = vertex_lowering.full_tensor_network_ready and propagator_ready

    if len(graph.interactions) > max_interactions_to_build:
        return TensorNetworkBlueprint(
            engine="spenso",
            status="size-guarded",
            current_count=len(graph.currents),
            interaction_count=len(graph.interactions),
            amplitude_count=len(graph.amplitudes),
            expression_built=False,
            expression_executed=False,
            full_me_tensor_network_ready=False,
            propagator_lowering_ready=propagator_ready,
            ready_interactions=ready_interactions,
            pending_interactions=pending_interactions,
            placeholder_vertex_kinds=placeholder_vertex_kinds,
            registered_tensor_names=registered_tensor_names,
            current_leaf_count=current_leaf_count,
            parametric_external_current_count=current_leaf_count,
            parametric_source_current_parameter_count=source_parameter_count,
            parametric_current_momentum_count=len(momentum_currents),
            parametric_momentum_parameter_count=momentum_parameter_count,
            parametric_parameter_count=parametric_parameter_count,
            expression=None,
            expression_length=None,
            expression_truncated=None,
            executed_expression=None,
            executed_expression_length=None,
            executed_expression_truncated=None,
            execution_time_s=None,
        )

    try:
        builder = _GraphTensorExpressionBuilder(model, graph)
        raw_expression = builder.matrix_element_skeleton()
        expression = (
            as_tensor(raw_expression)
            .simplify_algebra(
                gamma=False,
                color=True,
                contract="none",
                color_substitute_cof_dimension_invariants=True,
            )
            .to_expression()
        )
        expression_text = _clean_symbolica_string(str(expression))
        if len(graph.interactions) > max_interactions_to_execute:
            return TensorNetworkBlueprint(
                engine="spenso",
                status="execution-size-guarded",
                current_count=len(graph.currents),
                interaction_count=len(graph.interactions),
                amplitude_count=len(graph.amplitudes),
                expression_built=True,
                expression_executed=False,
                full_me_tensor_network_ready=False,
                propagator_lowering_ready=propagator_ready,
                ready_interactions=ready_interactions,
                pending_interactions=pending_interactions,
                placeholder_vertex_kinds=placeholder_vertex_kinds,
                registered_tensor_names=registered_tensor_names,
                current_leaf_count=current_leaf_count,
                parametric_external_current_count=current_leaf_count,
                parametric_source_current_parameter_count=source_parameter_count,
                parametric_current_momentum_count=len(momentum_currents),
                parametric_momentum_parameter_count=momentum_parameter_count,
                parametric_parameter_count=parametric_parameter_count,
                expression=_preview_expression(expression_text),
                expression_length=len(expression_text),
                expression_truncated=(
                    len(expression_text) > _MAX_RECURSION_EXPRESSION_PREVIEW
                ),
                executed_expression=None,
                executed_expression_length=None,
                executed_expression_truncated=None,
                execution_time_s=None,
            )

        start = time.perf_counter()
        library = model.build_tensor_library()
        _register_parametric_source_currents(library, graph)
        _register_parametric_current_momenta(model, library, graph)
        network = TensorNetwork(expression, library)
        network.execute(library=library)
        scalar = network.result_scalar()
        execution_time_s = time.perf_counter() - start
        executed_text = _clean_symbolica_string(str(scalar))
        dangling = as_tensor(scalar).list_dangling()
        status = "scalar-skeleton" if not dangling else "dangling-indices"
    except (RuntimeError, TypeError, ValueError) as exc:
        return TensorNetworkBlueprint(
            engine="spenso",
            status=f"failed: {exc}",
            current_count=len(graph.currents),
            interaction_count=len(graph.interactions),
            amplitude_count=len(graph.amplitudes),
            expression_built=False,
            expression_executed=False,
            full_me_tensor_network_ready=False,
            propagator_lowering_ready=propagator_ready,
            ready_interactions=ready_interactions,
            pending_interactions=pending_interactions,
            placeholder_vertex_kinds=placeholder_vertex_kinds,
            registered_tensor_names=registered_tensor_names,
            current_leaf_count=current_leaf_count,
            parametric_external_current_count=current_leaf_count,
            parametric_source_current_parameter_count=source_parameter_count,
            parametric_current_momentum_count=len(momentum_currents),
            parametric_momentum_parameter_count=momentum_parameter_count,
            parametric_parameter_count=parametric_parameter_count,
            expression=None,
            expression_length=None,
            expression_truncated=None,
            executed_expression=None,
            executed_expression_length=None,
            executed_expression_truncated=None,
            execution_time_s=None,
        )

    return TensorNetworkBlueprint(
        engine="spenso",
        status=status,
        current_count=len(graph.currents),
        interaction_count=len(graph.interactions),
        amplitude_count=len(graph.amplitudes),
        expression_built=True,
        expression_executed=True,
        full_me_tensor_network_ready=full_ready,
        propagator_lowering_ready=propagator_ready,
        ready_interactions=ready_interactions,
        pending_interactions=pending_interactions,
        placeholder_vertex_kinds=placeholder_vertex_kinds,
        registered_tensor_names=registered_tensor_names,
        current_leaf_count=current_leaf_count,
        parametric_external_current_count=current_leaf_count,
        parametric_source_current_parameter_count=source_parameter_count,
        parametric_current_momentum_count=len(momentum_currents),
        parametric_momentum_parameter_count=momentum_parameter_count,
        parametric_parameter_count=parametric_parameter_count,
        expression=_preview_expression(expression_text),
        expression_length=len(expression_text),
        expression_truncated=len(expression_text) > _MAX_RECURSION_EXPRESSION_PREVIEW,
        executed_expression=_preview_expression(executed_text),
        executed_expression_length=len(executed_text),
        executed_expression_truncated=(
            len(executed_text) > _MAX_RECURSION_EXPRESSION_PREVIEW
        ),
        execution_time_s=execution_time_s,
    )

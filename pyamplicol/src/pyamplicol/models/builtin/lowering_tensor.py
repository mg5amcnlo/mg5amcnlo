# SPDX-License-Identifier: 0BSD
"""Built-in-model tensor-network expression construction and Symbolica probes."""

from __future__ import annotations

import re
import time
from typing import Any

from ..._internal.physics.parameters import ParamBuilder, SymbolicaEvaluatorBundle
from .lowering_shared import _current_key_tuple, _number
from .lowering_types import ColorAlgebraProbe, TensorNetworkProbe
from .model import BuiltinSMModel
from .symbols import symbols

_MAX_RECURSION_EXPRESSION_PREVIEW = 4096


def build_tensor_network_scalar_bundle(
    model: BuiltinSMModel,
    graph: Any,
    *,
    name: str,
    collect_expression_metadata: bool = False,
) -> SymbolicaEvaluatorBundle:
    """Build a Symbolica evaluator bundle from the propagated tensor network."""

    from symbolica.community.tensor import TensorNetwork, as_tensor

    total_start = time.perf_counter()
    library = model.build_tensor_library()
    param_builder = ParamBuilder()
    _register_parametric_source_currents(library, graph, param_builder)
    _register_parametric_current_momenta(model, library, graph, param_builder)
    raw_expression = _GraphTensorExpressionBuilder(
        model, graph
    ).matrix_element_skeleton()
    expression = as_tensor(raw_expression).simplify_algebra(
        gamma=False,
        contract="none",
        color_substitute_cof_dimension_invariants=True,
    )
    network = TensorNetwork(expression, library)
    reduction_start = time.perf_counter()
    network.execute(library=library)
    scalar = network.result_scalar()
    reduction_time_s = time.perf_counter() - reduction_start
    evaluator_start = time.perf_counter()
    evaluator = scalar.evaluator(param_builder.parameter_symbols())
    if param_builder.real_valued_inputs:
        evaluator.set_real_params(param_builder.real_valued_inputs)
    evaluator_build_time_s = time.perf_counter() - evaluator_start
    total_build_time_s = time.perf_counter() - total_start
    expression_length = None
    executed_expression_length = None
    if collect_expression_metadata:
        expression_length = len(_clean_symbolica_string(str(expression)))
        executed_expression_length = len(_clean_symbolica_string(str(scalar)))
    return SymbolicaEvaluatorBundle(
        name=name,
        evaluator=evaluator,
        param_builder=param_builder,
        complex_inputs=True,
        metadata={
            "engine": "symbolica",
            "kernel": "tensor-network-scalar",
            "current_count": len(graph.currents),
            "interaction_count": len(graph.interactions),
            "amplitude_count": len(graph.amplitudes),
            "expression_length": expression_length,
            "executed_expression_length": executed_expression_length,
            "expression_metadata_collected": collect_expression_metadata,
            "tensor_network_reduction_s": reduction_time_s,
            "symbolica_evaluator_build_s": evaluator_build_time_s,
            "total_build_s": total_build_time_s,
        },
    )


def build_interleaved_tensor_network_scalar_bundle(
    model: BuiltinSMModel,
    graph: Any,
    *,
    name: str,
    collect_expression_metadata: bool = False,
) -> SymbolicaEvaluatorBundle:
    """Build a Symbolica evaluator by executing one aggregate TensorNetwork in steps."""

    total_start = time.perf_counter()
    library = model.build_tensor_library()
    param_builder = ParamBuilder()
    _register_parametric_source_currents(library, graph, param_builder)
    _register_parametric_current_momenta(model, library, graph, param_builder)

    build_start = time.perf_counter()
    builder = _GraphTensorExpressionBuilder(model, graph)
    network, interleaved_metadata = builder.matrix_element_interleaved_network(
        library,
        execute_between=True,
    )
    network_build_s = time.perf_counter() - build_start

    final_start = time.perf_counter()
    network.execute(library=library)
    scalar = network.result_scalar()
    final_reduction_s = time.perf_counter() - final_start

    evaluator_start = time.perf_counter()
    evaluator = scalar.evaluator(param_builder.parameter_symbols())
    if param_builder.real_valued_inputs:
        evaluator.set_real_params(param_builder.real_valued_inputs)
    evaluator_build_time_s = time.perf_counter() - evaluator_start
    total_build_time_s = time.perf_counter() - total_start
    return SymbolicaEvaluatorBundle(
        name=name,
        evaluator=evaluator,
        param_builder=param_builder,
        complex_inputs=True,
        metadata={
            "engine": "symbolica",
            "kernel": "interleaved-tensor-network-scalar",
            "strategy": "interleaved",
            "current_count": len(graph.currents),
            "interaction_count": len(graph.interactions),
            "amplitude_count": len(graph.amplitudes),
            "network_build_s": network_build_s,
            "interleaved_execution_s": interleaved_metadata["execution_s"],
            "interleaved_execution_count": interleaved_metadata["execution_count"],
            "interleaved_multiplication_count": interleaved_metadata[
                "multiplication_count"
            ],
            "interleaved_addition_count": interleaved_metadata["addition_count"],
            "tensor_network_reduction_s": final_reduction_s,
            "symbolica_evaluator_build_s": evaluator_build_time_s,
            "total_build_s": total_build_time_s,
            "executed_expression_length": (
                len(_clean_symbolica_string(str(scalar)))
                if collect_expression_metadata
                else None
            ),
            "expression_metadata_collected": collect_expression_metadata,
        },
    )


class _GraphTensorExpressionBuilder:
    def __init__(self, model: BuiltinSMModel, graph: Any) -> None:
        from symbolica.community.tensor import Representation, TensorName

        self.model = model
        self.graph = graph
        self._mink = Representation.mink(4)
        self._aux6 = Representation(symbols.antisymmetric_lorentz_pair_name, 6)
        self._weyl = Representation(symbols.weyl_spinor_name, 2)
        self._tensor_name = TensorName
        self._interactions_by_result: dict[
            tuple[int, tuple[int, ...], int], list[Any]
        ] = {}
        for interaction in graph.interactions:
            self._interactions_by_result.setdefault(
                _current_key_tuple(interaction.result),
                [],
            ).append(interaction)

    def matrix_element_skeleton(self) -> Any:
        from symbolica import Expression

        total = Expression.num(0)
        for amplitude_index, (left, right) in enumerate(self.graph.amplitudes, start=1):
            slots = self._slots_for_current(left, f"amp_{amplitude_index}")
            total = total + (
                self._current_expression(left, slots)
                * self._current_expression(right, slots)
            )
        return total

    def matrix_element_interleaved_network(
        self,
        library: Any,
        *,
        execute_between: bool = True,
    ) -> tuple[Any, dict[str, float | int]]:
        from symbolica.community.tensor import TensorNetwork

        metadata: dict[str, float | int] = {
            "execution_s": 0.0,
            "execution_count": 0,
            "multiplication_count": 0,
            "addition_count": 0,
        }
        total = None
        for amplitude_index, (left, right) in enumerate(self.graph.amplitudes, start=1):
            slots = self._slots_for_current(left, f"amp_{amplitude_index}")
            left_network = self._current_network(
                left,
                slots,
                library,
                execute_between=execute_between,
                metadata=metadata,
            )
            right_network = self._current_network(
                right,
                slots,
                library,
                execute_between=execute_between,
                metadata=metadata,
            )
            amplitude_network = self._multiply_networks(
                left_network,
                right_network,
                library,
                execute_between=execute_between,
                metadata=metadata,
            )
            total = (
                amplitude_network
                if total is None
                else self._add_networks(
                    total,
                    amplitude_network,
                    library,
                    execute_between=execute_between,
                    metadata=metadata,
                )
            )
        if total is None:
            total = TensorNetwork.zero()
        return total, metadata

    def _current_network(
        self,
        current: Any,
        output_slots: tuple[Any, ...],
        library: Any,
        *,
        execute_between: bool,
        metadata: dict[str, float | int],
    ) -> Any:
        from symbolica.community.tensor import TensorNetwork, as_tensor

        interactions = self._interactions_by_result.get(_current_key_tuple(current))
        if not interactions:
            expression = self._current_leaf(current, output_slots)
            return TensorNetwork(
                as_tensor(expression) if output_slots else expression, library
            )

        needs_propagator = _current_needs_propagator(self.graph, current)
        result_slots = (
            self._slots_for_current(
                current,
                self._propagator_dummy_prefix(current),
            )
            if needs_propagator
            else output_slots
        )
        total = None
        for interaction_index, interaction in enumerate(interactions, start=1):
            left_slots = self._slots_for_current(
                interaction.left,
                self._dummy_prefix(interaction, interaction_index, "left"),
            )
            right_slots = self._slots_for_current(
                interaction.right,
                self._dummy_prefix(interaction, interaction_index, "right"),
            )
            expression = self._vertex_tensor(
                interaction, left_slots, right_slots, result_slots
            )
            term = TensorNetwork(
                as_tensor(expression)
                if left_slots or right_slots or result_slots
                else expression,
                library,
            )
            term = self._multiply_networks(
                term,
                self._current_network(
                    interaction.left,
                    left_slots,
                    library,
                    execute_between=execute_between,
                    metadata=metadata,
                ),
                library,
                execute_between=execute_between,
                metadata=metadata,
            )
            term = self._multiply_networks(
                term,
                self._current_network(
                    interaction.right,
                    right_slots,
                    library,
                    execute_between=execute_between,
                    metadata=metadata,
                ),
                library,
                execute_between=execute_between,
                metadata=metadata,
            )
            if needs_propagator:
                expression = self._propagator_tensor(
                    current, result_slots, output_slots
                )
                term = self._multiply_networks(
                    TensorNetwork(
                        as_tensor(expression)
                        if result_slots or output_slots
                        else expression,
                        library,
                    ),
                    term,
                    library,
                    execute_between=execute_between,
                    metadata=metadata,
                )
            total = (
                term
                if total is None
                else self._add_networks(
                    total,
                    term,
                    library,
                    execute_between=execute_between,
                    metadata=metadata,
                )
            )
        if total is None:
            return TensorNetwork.zero()
        return total

    def _multiply_networks(
        self,
        left: Any,
        right: Any,
        library: Any,
        *,
        execute_between: bool,
        metadata: dict[str, float | int],
    ) -> Any:
        result = left * right
        metadata["multiplication_count"] = int(metadata["multiplication_count"]) + 1
        if execute_between:
            result = self._execute_network(result, library, metadata)
        return result

    def _add_networks(
        self,
        left: Any,
        right: Any,
        library: Any,
        *,
        execute_between: bool,
        metadata: dict[str, float | int],
    ) -> Any:
        result = left + right
        metadata["addition_count"] = int(metadata["addition_count"]) + 1
        if execute_between:
            result = self._execute_network(result, library, metadata)
        return result

    def _execute_network(
        self,
        network: Any,
        library: Any,
        metadata: dict[str, float | int],
    ) -> Any:
        from symbolica.community.tensor import TensorNetwork

        start = time.perf_counter()
        network.execute(library=library)
        metadata["execution_s"] = float(metadata["execution_s"]) + (
            time.perf_counter() - start
        )
        metadata["execution_count"] = int(metadata["execution_count"]) + 1
        return TensorNetwork.one() * network.result_tensor(library)

    def _current_expression(self, current: Any, output_slots: tuple[Any, ...]) -> Any:
        from symbolica import Expression

        key = _current_key_tuple(current)
        interactions = self._interactions_by_result.get(key)
        if not interactions:
            return self._current_leaf(current, output_slots)

        needs_propagator = _current_needs_propagator(self.graph, current)
        result_slots = (
            self._slots_for_current(
                current,
                self._propagator_dummy_prefix(current),
            )
            if needs_propagator
            else output_slots
        )
        total = Expression.num(0)
        for interaction_index, interaction in enumerate(interactions, start=1):
            left_slots = self._slots_for_current(
                interaction.left,
                self._dummy_prefix(interaction, interaction_index, "left"),
            )
            right_slots = self._slots_for_current(
                interaction.right,
                self._dummy_prefix(interaction, interaction_index, "right"),
            )
            total = total + (
                self._vertex_tensor(interaction, left_slots, right_slots, result_slots)
                * self._current_expression(interaction.left, left_slots)
                * self._current_expression(interaction.right, right_slots)
            )
        if needs_propagator:
            total = self._propagator_tensor(current, result_slots, output_slots) * total
        return total

    def _current_leaf(self, current: Any, output_slots: tuple[Any, ...]) -> Any:
        return self._tensor_name(_current_tensor_name(current))(
            *output_slots
        ).to_expression()

    def _vertex_tensor(
        self,
        interaction: Any,
        left_slots: tuple[Any, ...],
        right_slots: tuple[Any, ...],
        output_slots: tuple[Any, ...],
    ) -> Any:
        kind = int(interaction.vertex_kind)
        if kind == 0:
            return self.model.three_gluon_current_expression(
                left_slot=left_slots[0],
                right_slot=right_slots[0],
                output_slot=output_slots[0],
                left_momentum_tensor_name=_current_momentum_tensor_name(
                    interaction.left
                ),
                right_momentum_tensor_name=_current_momentum_tensor_name(
                    interaction.right
                ),
                dummy_prefix=self._dummy_prefix(interaction, 0, "three_gluon"),
            )
        if kind == 1:
            return self._tensor_name(symbols.two_gluon_to_tensor_name)(
                left_slots[0],
                right_slots[0],
                output_slots[0],
            ).to_expression()
        if kind == 2:
            return self._tensor_name(symbols.tensor_gluon_to_gluon_name)(
                left_slots[0],
                right_slots[0],
                output_slots[0],
            ).to_expression()
        if kind == 3:
            return self._tensor_name(symbols.gluon_tensor_to_gluon_name)(
                left_slots[0],
                right_slots[0],
                output_slots[0],
            ).to_expression()
        if kind == 6:
            return self._tensor_name(
                _quark_vector_weyl_tensor_name(int(interaction.result.chirality))
            )(
                left_slots[0],
                right_slots[0],
                output_slots[0],
            ).to_expression()
        if kind == 10:
            return (
                _number(
                    _weyl_coupling_for_chirality(
                        int(interaction.result.chirality),
                        _coupling_pair(interaction.coupling),
                    )
                )
                * self._tensor_name(
                    _quark_vector_weyl_tensor_name(int(interaction.result.chirality))
                )(
                    left_slots[0],
                    right_slots[0],
                    output_slots[0],
                ).to_expression()
            )
        return self._tensor_name(symbols.vertex_kind_tensor_name(kind))(
            *left_slots,
            *right_slots,
            *output_slots,
            _number(float(interaction.coupling[0])),
            _number(float(interaction.coupling[1])),
        ).to_expression()

    def _propagator_tensor(
        self,
        current: Any,
        input_slots: tuple[Any, ...],
        output_slots: tuple[Any, ...],
    ) -> Any:
        return self._tensor_name(_propagator_tensor_name(current))(
            input_slots[0],
            output_slots[0],
        ).to_expression()

    def _slots_for_current(self, current: Any, prefix: str) -> tuple[Any, ...]:
        pdg = int(current.pdg)
        if pdg == -21:
            return (self._aux6(f"{prefix}_A"),)
        if _is_lorentz_vector_current_pdg(pdg):
            return (self._mink(f"{prefix}_mu"),)
        if 1 <= abs(pdg) <= 6 or 11 <= abs(pdg) <= 16:
            return (self._weyl(f"{prefix}_alpha"),)
        return ()

    def _dummy_prefix(self, interaction: Any, index: int, side: str) -> str:
        labels = "_".join(str(label) for label in interaction.result.external_labels)
        chirality = int(interaction.result.chirality)
        return (
            f"v{int(interaction.vertex_kind)}_{labels}_"
            f"{_signed_label(chirality)}_{index}_{side}"
        )

    def _propagator_dummy_prefix(self, current: Any) -> str:
        labels = "_".join(str(label) for label in current.external_labels)
        return (
            f"prop_{_signed_label(int(current.pdg))}_{labels}_"
            f"{_signed_label(int(current.chirality))}"
        )

    def _current_output_prefix(self, current: Any) -> str:
        labels = "_".join(str(label) for label in current.external_labels)
        return (
            f"cur_{_signed_label(int(current.pdg))}_{labels}_"
            f"{_signed_label(int(current.chirality))}"
        )


def _current_tensor_name(current: Any) -> str:
    labels = "_".join(str(label) for label in current.external_labels)
    return symbols.qualified_name(
        f"current_{_signed_label(int(current.pdg))}_{labels}_"
        f"{_signed_label(int(current.chirality))}"
    )


def _signed_label(value: int) -> str:
    if value < 0:
        return f"m{abs(value)}"
    return f"p{value}"


def _quark_vector_weyl_tensor_name(chirality: int) -> str:
    if chirality == 1:
        return symbols.quark_vector_weyl_plus_name
    if chirality == -1:
        return symbols.quark_vector_weyl_minus_name
    raise ValueError(f"Weyl vector tensor requires nonzero chirality, got {chirality}")


def _weyl_coupling_for_chirality(
    chirality: int,
    coupling: tuple[float, float],
) -> float:
    if chirality == 1:
        return coupling[1]
    if chirality == -1:
        return coupling[0]
    raise ValueError(f"Weyl coupling requires nonzero chirality, got {chirality}")


def _coupling_pair(coupling: Any) -> tuple[float, float]:
    if len(coupling) != 2:
        raise ValueError(f"expected a two-component coupling, got {coupling}")
    return float(coupling[0]), float(coupling[1])


def _source_current_count(graph: Any) -> int:
    return len(_source_currents(graph))


def _source_currents(graph: Any) -> tuple[Any, ...]:
    interactions_by_result: dict[tuple[int, tuple[int, ...], int], list[Any]] = {}
    for interaction in graph.interactions:
        interactions_by_result.setdefault(
            _current_key_tuple(interaction.result),
            [],
        ).append(interaction)
    source_keys: set[tuple[int, tuple[int, ...], int]] = set()
    seen: set[tuple[int, tuple[int, ...], int]] = set()

    def visit(current: Any) -> None:
        key = _current_key_tuple(current)
        if key in seen:
            return
        seen.add(key)
        interactions = interactions_by_result.get(key)
        if not interactions:
            source_keys.add(key)
            return
        for interaction in interactions:
            visit(interaction.left)
            visit(interaction.right)

    for left, right in graph.amplitudes:
        visit(left)
        visit(right)
    return tuple(
        current
        for current in graph.currents
        if _current_key_tuple(current) in source_keys
    )


def _register_parametric_source_currents(
    library: Any,
    graph: Any,
    builder: ParamBuilder | None = None,
) -> ParamBuilder:
    from symbolica.community.tensor import Representation

    if builder is None:
        builder = ParamBuilder()
    mink = Representation.mink(4)
    aux6 = Representation(symbols.antisymmetric_lorentz_pair_name, 6)
    weyl = Representation(symbols.weyl_spinor_name, 2)
    for current in _source_currents(graph):
        pdg = int(current.pdg)
        if pdg == -21:
            representation = aux6
        elif _is_lorentz_vector_current_pdg(pdg):
            representation = mink
        elif 1 <= abs(pdg) <= 6 or 11 <= abs(pdg) <= 16:
            representation = weyl
        else:
            continue
        builder.register_rank1_tensor(
            library,
            tensor_name=_current_tensor_name(current),
            representation=representation,
            head=_current_parameter_head(current),
            length=_current_dimension(current),
            role="source_current",
        )
    return builder


def _register_parametric_current_momenta(
    model: BuiltinSMModel,
    library: Any,
    graph: Any,
    builder: ParamBuilder | None = None,
) -> ParamBuilder:
    from symbolica.community.tensor import Representation, Tensor, TensorName

    if builder is None:
        builder = ParamBuilder()
    mink = Representation.mink(4)
    weyl = Representation(symbols.weyl_spinor_name, 2)
    momentum_symbols_by_current: dict[
        tuple[int, tuple[int, ...], int], tuple[Any, ...]
    ] = {}
    for current in _current_momentum_currents(graph):
        momentum_symbols = builder.register_rank1_tensor(
            library,
            tensor_name=_current_momentum_tensor_name(current),
            representation=mink,
            head=_current_momentum_parameter_head(current),
            length=4,
            role="current_momentum",
            real_valued=True,
        )
        momentum_symbols_by_current[_current_key_tuple(current)] = momentum_symbols

    for current in _propagating_currents(graph):
        momentum_symbols = momentum_symbols_by_current[_current_key_tuple(current)]
        pdg = int(current.pdg)
        if pdg == 21:
            library.register(
                Tensor.dense(
                    TensorName(_propagator_tensor_name(current))(mink, mink),
                    model.gluon_propagator_tensor_data(momentum_symbols),
                )
            )
        elif _is_weyl_fermion_current(current):
            library.register(
                Tensor.dense(
                    TensorName(_propagator_tensor_name(current))(weyl, weyl),
                    model.quark_weyl_propagator_tensor_data(
                        momentum_symbols,
                        chirality=int(current.chirality),
                    ),
                )
            )
    return builder


def _current_momentum_currents(graph: Any) -> tuple[Any, ...]:
    currents: dict[tuple[int, tuple[int, ...], int], Any] = {}
    for interaction in graph.interactions:
        if int(interaction.vertex_kind) == 0:
            currents.setdefault(_current_key_tuple(interaction.left), interaction.left)
            currents.setdefault(
                _current_key_tuple(interaction.right), interaction.right
            )
    for current in _propagating_currents(graph):
        currents.setdefault(_current_key_tuple(current), current)
    return tuple(currents.values())


def _propagating_currents(graph: Any) -> tuple[Any, ...]:
    result_keys = {
        _current_key_tuple(interaction.result) for interaction in graph.interactions
    }
    amplitude_keys = {
        _current_key_tuple(current)
        for amplitude in graph.amplitudes
        for current in amplitude
    }
    return tuple(
        current
        for current in graph.currents
        if _current_key_tuple(current) in result_keys
        and _current_key_tuple(current) not in amplitude_keys
        and _has_supported_propagator(current)
    )


def _current_needs_propagator(graph: Any, current: Any) -> bool:
    propagating_keys = {
        _current_key_tuple(propagating_current)
        for propagating_current in _propagating_currents(graph)
    }
    return _current_key_tuple(current) in propagating_keys


def _has_supported_propagator(current: Any) -> bool:
    pdg = int(current.pdg)
    return pdg == 21 or _is_weyl_fermion_current(current)


def _is_weyl_fermion_current(current: Any) -> bool:
    pdg = int(current.pdg)
    return (1 <= abs(pdg) <= 6 or 11 <= abs(pdg) <= 16) and int(current.chirality) != 0


def _is_lorentz_vector_current_pdg(pdg: int) -> bool:
    return pdg in (21, 22, 23, 24, -24)


def _current_dimension(current: Any) -> int:
    pdg = int(current.pdg)
    if pdg == -21:
        return 6
    if _is_lorentz_vector_current_pdg(pdg):
        return 4
    if 1 <= abs(pdg) <= 6 or 11 <= abs(pdg) <= 16:
        return 2
    return 0


def _current_parameter_head(current: Any) -> tuple[str, ...]:
    labels = tuple(str(label) for label in current.external_labels)
    return (
        "current",
        _signed_label(int(current.pdg)),
        "_".join(labels) if labels else "empty",
        _signed_label(int(current.chirality)),
    )


def _current_momentum_tensor_name(current: Any) -> str:
    labels = "_".join(str(label) for label in current.external_labels)
    return symbols.qualified_name(
        f"current_momentum_{_signed_label(int(current.pdg))}_{labels}_"
        f"{_signed_label(int(current.chirality))}"
    )


def _propagator_tensor_name(current: Any) -> str:
    labels = "_".join(str(label) for label in current.external_labels)
    if int(current.pdg) == 21:
        head = "gluon_propagator"
    elif _is_weyl_fermion_current(current):
        head = "quark_weyl_propagator"
    else:
        head = "propagator"
    return symbols.qualified_name(
        f"{head}_{_signed_label(int(current.pdg))}_{labels}_"
        f"{_signed_label(int(current.chirality))}"
    )


def _current_momentum_parameter_head(current: Any) -> tuple[str, ...]:
    labels = tuple(str(label) for label in current.external_labels)
    return (
        "current_momentum",
        _signed_label(int(current.pdg)),
        "_".join(labels) if labels else "empty",
        _signed_label(int(current.chirality)),
    )


def _propagator_lowering_ready(graph: Any) -> bool:
    result_currents = {
        _current_key_tuple(interaction.result): interaction.result
        for interaction in graph.interactions
    }
    amplitude_keys = {
        _current_key_tuple(current)
        for amplitude in graph.amplitudes
        for current in amplitude
    }
    for key, current in result_currents.items():
        if key in amplitude_keys or int(current.pdg) == -21:
            continue
        if not _has_supported_propagator(current):
            return False
    return bool(graph.interactions)


def _build_auxiliary_tensor_probe(
    model: BuiltinSMModel,
) -> TensorNetworkProbe:
    from symbolica.community.tensor import (
        Representation,
        TensorName,
        TensorNetwork,
        as_tensor,
    )

    library = model.build_tensor_library()
    mink = Representation.mink(4)
    antisym = Representation(symbols.antisymmetric_lorentz_pair_name, 6)
    two_gluon_to_tensor = TensorName(symbols.two_gluon_to_tensor_name)
    tensor_gluon_to_gluon = TensorName(symbols.tensor_gluon_to_gluon_name)

    expression = (
        two_gluon_to_tensor(
            mink("mu"),
            mink("nu"),
            antisym("A"),
        ).to_expression()
        * tensor_gluon_to_gluon(
            antisym("A"),
            mink("nu"),
            mink("rho"),
        ).to_expression()
    )
    network = TensorNetwork(as_tensor(expression), library)
    network.execute(library=library)
    result = network.result_tensor(library)
    output_size = len(result)
    structure = result.structure
    entries = tuple(complex(result[i]) for i in range(output_size))
    nonzero = tuple(
        (index, value) for index, value in enumerate(entries) if abs(value) > 1.0e-15
    )
    weighted_checksum = sum((index + 1) * value for index, value in enumerate(entries))
    return TensorNetworkProbe(
        engine="spenso",
        tensor_names=(
            symbols.two_gluon_to_tensor_name,
            symbols.tensor_gluon_to_gluon_name,
        ),
        expression=_clean_symbolica_string(str(expression)),
        output_structure=_clean_symbolica_string(str(structure)),
        output_rank=2,
        output_size=output_size,
        nonzero_entries=len(nonzero),
        max_abs_entry=max((abs(value) for value in entries), default=0.0),
        weighted_checksum=(weighted_checksum.real, weighted_checksum.imag),
        first_nonzero_entries=tuple(
            (index, _format_complex(value)) for index, value in nonzero[:4]
        ),
    )


def _build_color_probe() -> ColorAlgebraProbe:
    from symbolica.community.tensor import TensorExpression

    structure_constant = TensorExpression.color_f(8)

    def color_f(i: int, j: int, k: int) -> Any:
        return structure_constant(i, j, k)

    expression = color_f(1, 2, 3) * color_f(3, 2, 1)
    simplified = expression.simplify_algebra(
        gamma=False,
        contract="none",
        color_substitute_cof_dimension_invariants=True,
    )
    return ColorAlgebraProbe(
        engine="tensor",
        input_expression=_clean_symbolica_string(str(expression)),
        simplified_expression=_clean_symbolica_string(str(simplified)),
    )


def _format_complex(value: complex) -> str:
    real = 0.0 if abs(value.real) < 1.0e-15 else value.real
    imag = 0.0 if abs(value.imag) < 1.0e-15 else value.imag
    return f"{real:.16g}{imag:+.16g}j"


def _clean_symbolica_string(value: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", value)


def _preview_expression(value: str) -> str:
    if len(value) <= _MAX_RECURSION_EXPRESSION_PREVIEW:
        return value
    suffix = "...<truncated>"
    preview_length = _MAX_RECURSION_EXPRESSION_PREVIEW - len(suffix)
    return f"{value[:preview_length]}{suffix}"

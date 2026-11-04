# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import base64
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .symbols import symbols

ParameterHead = tuple[str, ...]


@dataclass(frozen=True)
class ParameterRange:
    head: ParameterHead
    start: int
    stop: int
    role: str = "scalar"
    tensor_name: str | None = None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "head": list(self.head),
            "range": [self.start, self.stop],
            "role": self.role,
            "tensor_name": self.tensor_name,
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, object]) -> ParameterRange:
        head = data.get("head")
        span = data.get("range")
        if not isinstance(head, list) or not all(
            isinstance(item, str) for item in head
        ):
            raise ValueError("parameter range is missing a string head")
        if not isinstance(span, list) or len(span) != 2:
            raise ValueError("parameter range is missing a two-entry range")
        role = data.get("role", "scalar")
        tensor_name = data.get("tensor_name")
        return cls(
            head=tuple(head),
            start=int(span[0]),
            stop=int(span[1]),
            role=role if isinstance(role, str) else "scalar",
            tensor_name=tensor_name if isinstance(tensor_name, str) else None,
        )


class ParamBuilder:
    """Ordered runtime input builder for pyamplicol Symbolica evaluators."""

    def __init__(self) -> None:
        self.positions: dict[ParameterHead, tuple[int, int]] = {}
        self.order: list[ParameterHead] = []
        self.ranges: dict[ParameterHead, ParameterRange] = {}
        self.values: list[complex] = []
        self._symbols: list[Any] = []
        self.real_valued_inputs: list[int] = []
        self.purely_imaginary_valued_inputs: list[int] = []
        self.forced_complex_valued_inputs: list[int] = []

    def add_parameter_list(
        self,
        head: ParameterHead,
        length: int,
        *,
        role: str = "scalar",
        tensor_name: str | None = None,
        real_valued: bool = False,
    ) -> tuple[Any, ...]:
        if head in self.positions:
            raise ValueError(f"parameter head already registered: {head}")
        if length <= 0:
            raise ValueError("parameter length must be positive")
        start = len(self.values)
        stop = start + length
        self.positions[head] = (start, stop)
        self.order.append(head)
        self.ranges[head] = ParameterRange(
            head=head,
            start=start,
            stop=stop,
            role=role,
            tensor_name=tensor_name,
        )
        self.values.extend(0j for _ in range(length))
        if real_valued:
            self.real_valued_inputs.extend(range(start, stop))
        parameter_symbols = tuple(
            symbols.parameter_component(head, index) for index in range(length)
        )
        self._symbols.extend(parameter_symbols)
        return parameter_symbols

    def register_rank1_tensor(
        self,
        library: Any,
        *,
        tensor_name: str,
        representation: Any,
        head: ParameterHead,
        length: int,
        role: str,
        real_valued: bool = False,
    ) -> tuple[Any, ...]:
        from symbolica.community.tensor import Tensor, TensorName

        parameter_symbols = self.add_parameter_list(
            head,
            length,
            role=role,
            tensor_name=tensor_name,
            real_valued=real_valued,
        )
        library.register(
            Tensor.dense(
                TensorName(tensor_name)(representation),
                parameter_symbols,
            )
        )
        return parameter_symbols

    def set_parameter_values(
        self,
        head: ParameterHead,
        values: Sequence[complex | float],
        *,
        check_phase_flags: bool = True,
    ) -> None:
        if head not in self.positions:
            raise ValueError(f"unknown parameter head: {head}")
        start, stop = self.positions[head]
        if stop - start != len(values):
            raise ValueError(
                f"parameter {head} expects {stop - start} values, got {len(values)}"
            )
        complex_values = [complex(value) for value in values]
        if check_phase_flags:
            self._check_values_against_phase_flags(start, complex_values)
        self.values[start:stop] = complex_values

    def parameter_symbols(self) -> list[Any]:
        return list(self._symbols)

    def get_values(self, *, complexified: bool = False) -> list[complex] | list[float]:
        if not complexified:
            return list(self.values)
        flat: list[float] = []
        for value in self.values:
            flat.extend([float(value.real), float(value.imag)])
        return flat

    def get_complex_values(self) -> list[complex]:
        return list(self.values)

    def get_real_values(self) -> list[float]:
        """Return evaluator inputs for Symbolica real-valued evaluators."""

        real_values: list[float] = []
        for index, value in enumerate(self.values):
            if value.imag != 0.0:
                raise ValueError(
                    f"parameter index {index} has imaginary part {value.imag}; "
                    "complex inputs must be represented with an explicit real/imag "
                    "parameter packing"
                )
            real_values.append(float(value.real))
        return real_values

    def to_json_dict(self) -> dict[str, object]:
        return {
            "parameters": [self.ranges[head].to_json_dict() for head in self.order],
            "values": [[value.real, value.imag] for value in self.values],
            "real_valued_inputs": list(self.real_valued_inputs),
            "purely_imaginary_valued_inputs": list(self.purely_imaginary_valued_inputs),
            "forced_complex_valued_inputs": list(self.forced_complex_valued_inputs),
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, object]) -> ParamBuilder:
        builder = cls()
        parameters = data.get("parameters")
        values = data.get("values")
        if not isinstance(parameters, list):
            raise ValueError("param builder payload is missing parameters")
        if not isinstance(values, list):
            raise ValueError("param builder payload is missing values")
        for item in parameters:
            if not isinstance(item, dict):
                raise ValueError("malformed parameter entry")
            parameter_range = ParameterRange.from_json_dict(item)
            length = parameter_range.stop - parameter_range.start
            symbols_for_range = builder.add_parameter_list(
                parameter_range.head,
                length,
                role=parameter_range.role,
                tensor_name=parameter_range.tensor_name,
            )
            if len(symbols_for_range) != length:
                raise ValueError("internal parameter reconstruction mismatch")
        decoded_values = []
        for item in values:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError("malformed parameter value entry")
            decoded_values.append(complex(float(item[0]), float(item[1])))
        if len(decoded_values) != len(builder.values):
            raise ValueError("parameter value count does not match ranges")
        builder.values = decoded_values
        builder.real_valued_inputs = _int_list(data.get("real_valued_inputs", []))
        builder.purely_imaginary_valued_inputs = _int_list(
            data.get("purely_imaginary_valued_inputs", [])
        )
        builder.forced_complex_valued_inputs = _int_list(
            data.get("forced_complex_valued_inputs", [])
        )
        return builder

    def _check_values_against_phase_flags(
        self,
        start: int,
        values: Sequence[complex],
    ) -> None:
        for offset, value in enumerate(values):
            index = start + offset
            if index in self.real_valued_inputs and value.imag != 0.0:
                raise ValueError(f"parameter index {index} is marked real")
            if index in self.purely_imaginary_valued_inputs and value.real != 0.0:
                raise ValueError(f"parameter index {index} is marked imaginary")


@dataclass
class SymbolicaEvaluatorBundle:
    name: str
    evaluator: Any
    param_builder: ParamBuilder
    output_length: int = 1
    complex_inputs: bool = False
    metadata: dict[str, object] | None = None

    def evaluate(self) -> tuple[complex, ...]:
        if self.complex_inputs:
            result = self.evaluator.evaluate_complex(
                [self.param_builder.get_complex_values()]
            )
        else:
            result = self.evaluator.evaluate([self.param_builder.get_real_values()])
        row = result[0]
        return tuple(complex(value) for value in row[: self.output_length])

    def to_artifact_payload(self) -> dict[str, object]:
        return {
            "kind": "pyamplicol-symbolica-evaluator-bundle",
            "name": self.name,
            "output_length": self.output_length,
            "complex_inputs": self.complex_inputs,
            "param_builder": self.param_builder.to_json_dict(),
            "evaluator_state_b64": base64.b64encode(self.evaluator.save()).decode(
                "ascii"
            ),
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_artifact_payload(
        cls,
        payload: dict[str, object],
    ) -> SymbolicaEvaluatorBundle:
        from symbolica import Evaluator

        encoded = payload.get("evaluator_state_b64")
        param_builder_payload = payload.get("param_builder")
        name = payload.get("name")
        output_length = payload.get("output_length", 1)
        complex_inputs = payload.get("complex_inputs", False)
        metadata = payload.get("metadata")
        if not isinstance(encoded, str):
            raise ValueError("bundle payload is missing evaluator_state_b64")
        if not isinstance(param_builder_payload, dict):
            raise ValueError("bundle payload is missing param_builder")
        if not isinstance(name, str):
            raise ValueError("bundle payload is missing name")
        param_builder = ParamBuilder.from_json_dict(param_builder_payload)
        evaluator = Evaluator.load(base64.b64decode(encoded))
        if param_builder.real_valued_inputs:
            evaluator.set_real_params(param_builder.real_valued_inputs)
        return cls(
            name=name,
            evaluator=evaluator,
            param_builder=param_builder,
            output_length=_int_value(output_length, "output_length"),
            complex_inputs=_bool_value(complex_inputs, "complex_inputs"),
            metadata=metadata if isinstance(metadata, dict) else {},
        )


def _int_list(value: object) -> list[int]:
    if not isinstance(value, list):
        return []
    return [int(item) for item in value]


def _int_value(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, int | str | bytes | bytearray):
        return int(value)
    raise ValueError(f"{name} must be an integer")


def _bool_value(value: object, name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{name} must be a boolean")


__all__ = [
    "ParamBuilder",
    "ParameterHead",
    "ParameterRange",
    "SymbolicaEvaluatorBundle",
]

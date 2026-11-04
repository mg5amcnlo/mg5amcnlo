# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from .._internal.physics.symbols import symbols
from .contracts import (
    CompiledOrientedKernel,
)

if TYPE_CHECKING:
    pass

from . import compiler_symbolica as _sym
from .external_helpers import (
    _chirality_tag,
    _is_numeric,
    _is_zero,
    _record_default,
    _replace_symbols,
)


class ExternalModelKernelMixin:
    def runtime_parameter_type(self, name: str) -> str:
        try:
            return self._parameter_records[str(name)].parameter_type
        except KeyError as exc:
            raise KeyError(f"unknown runtime model parameter {name!r}") from exc

    def runtime_normalization_payload(self, dag: Any) -> dict[str, object]:
        initial_pdgs = tuple(int(pdg) for pdg in dag.process.initial_pdgs)
        final_pdgs = tuple(int(pdg) for pdg in dag.process.final_pdgs)
        average_factor = 1
        for pdg in initial_pdgs:
            average_factor *= len(self.source_spin_states(pdg))
            average_factor *= max(1, abs(self.color_rep(pdg)))
        identical_factor = math.prod(
            math.factorial(count) for count in Counter(final_pdgs).values()
        )
        color_factor = self._external_color_normalization_factor(
            (*initial_pdgs, *final_pdgs)
        )
        return {
            "color_accuracy": dag.process.color_accuracy,
            "color_factor": float(color_factor),
            "average_factor": float(average_factor),
            "identical_factor": float(identical_factor),
            "global_coupling_factor": 1.0,
            "couplings_in_stage_evaluators": True,
            "coupling_policy": (
                "UFO coupling expressions are fully included in generated stage "
                "evaluators; no built-in-SM global coupling factor is applied"
            ),
        }

    def _external_color_normalization_factor(self, process: Sequence[int]) -> int:
        exponent_twice = 0
        for pdg in process:
            representation = abs(self.color_rep(int(pdg)))
            if representation == 8:
                exponent_twice += 2
            elif representation == 3:
                exponent_twice += 1
            elif representation != 1:
                raise ValueError(
                    f"unsupported leading-color representation {representation}"
                )
        if exponent_twice % 2:
            raise ValueError(f"non-integer leading-color exponent for {tuple(process)}")
        return 3 ** (exponent_twice // 2)

    def _kernel(self, kind: int) -> CompiledOrientedKernel:
        try:
            return self._kernels[int(kind)]
        except KeyError as exc:
            raise KeyError(f"unknown compiled UFO kernel kind {kind}") from exc

    def _weyl_projection_is_nonzero(
        self,
        kind: int,
        left_chirality: int,
        right_chirality: int,
        result_chirality: int,
    ) -> bool:
        key = (kind, left_chirality, right_chirality, result_chirality)
        cached = self._weyl_projection_support.get(key)
        if cached is not None:
            return cached
        kernel = self._kernel(kind)
        left_pdg = self._particle_records_by_name[kernel.particles[0]].pdg_code
        right_pdg = self._particle_records_by_name[kernel.particles[1]].pdg_code
        left = tuple(
            symbols.weyl_probe(self.name, kind, "left", "component", index)
            for index in range(self.current_dimension(left_pdg, left_chirality))
        )
        right = tuple(
            symbols.weyl_probe(self.name, kind, "right", "component", index)
            for index in range(self.current_dimension(right_pdg, right_chirality))
        )
        components = self._projected_kernel_components(
            kernel,
            left,
            right,
            left_chirality=left_chirality,
            right_chirality=right_chirality,
            result_chirality=result_chirality,
            left_momentum=tuple(
                symbols.weyl_probe(
                    self.name,
                    kind,
                    "left",
                    "momentum",
                    index,
                )
                for index in range(4)
            ),
            right_momentum=tuple(
                symbols.weyl_probe(
                    self.name,
                    kind,
                    "right",
                    "momentum",
                    index,
                )
                for index in range(4)
            ),
        )
        supported = any(not _is_zero(component) for component in components)
        self._weyl_projection_support[key] = supported
        return supported

    def _projected_kernel_components(
        self,
        kernel: CompiledOrientedKernel,
        left: Sequence[Any],
        right: Sequence[Any],
        *,
        left_chirality: int,
        right_chirality: int,
        result_chirality: int,
        left_momentum: Sequence[Any],
        right_momentum: Sequence[Any],
        runtime_parameter_values: Mapping[str, Any] | None = None,
    ) -> tuple[Any, ...]:
        left_pdg = self._particle_records_by_name[kernel.particles[0]].pdg_code
        right_pdg = self._particle_records_by_name[kernel.particles[1]].pdg_code
        result_pdg = self._particle_records_by_name[kernel.particles[2]].pdg_code
        full_left = self._embed_weyl_current(left_pdg, left_chirality, left)
        full_right = self._embed_weyl_current(right_pdg, right_chirality, right)
        parameter_values = (
            {name: self._parameter_value(name) for name in kernel.runtime_parameters}
            if runtime_parameter_values is None
            else dict(runtime_parameter_values)
        )
        substitutions: dict[_sym.Expression, Any] = {}
        for index, value in enumerate(full_left):
            substitutions[
                self._model_symbols.kernel_component(kernel.kind, "left", index)
            ] = value
        for index, value in enumerate(full_right):
            substitutions[
                self._model_symbols.kernel_component(kernel.kind, "right", index)
            ] = value
        for index, value in enumerate(left_momentum):
            substitutions[
                self._model_symbols.kernel_momentum(kernel.kind, "left", index)
            ] = value
        for index, value in enumerate(right_momentum):
            substitutions[
                self._model_symbols.kernel_momentum(kernel.kind, "right", index)
            ] = value
        for name, value in parameter_values.items():
            substitutions[symbols.runtime_model_parameter(self.name, name)] = value
        templates = self._kernel_component_expressions(kernel)
        if all(_is_numeric(value) for value in substitutions.values()):
            try:
                components = tuple(
                    complex(template.evaluate(substitutions)) for template in templates
                )
            except Exception:
                components = tuple(
                    _replace_symbols(template, substitutions) for template in templates
                )
        else:
            components = tuple(
                _replace_symbols(template, substitutions) for template in templates
            )
        if not self.is_chiral_eligible(result_pdg):
            return components
        if result_chirality == 0:
            return components
        if result_chirality == 1:
            return components[2:4]
        if result_chirality == -1:
            return components[0:2]
        raise ValueError("a projected Weyl result requires chirality -1, 0, or +1")

    def _kernel_function_component_calls(
        self,
        kernel: CompiledOrientedKernel,
        left: Sequence[Any],
        right: Sequence[Any],
        *,
        left_chirality: int,
        right_chirality: int,
        result_chirality: int,
        left_momentum: Sequence[Any],
        right_momentum: Sequence[Any],
        runtime_parameter_values: Mapping[str, Any],
    ) -> tuple[Any, ...]:
        key = (
            int(kernel.kind),
            int(left_chirality),
            int(right_chirality),
            int(result_chirality),
        )
        function_specs = self._kernel_function_specs.get(key)
        if function_specs is None:
            function_specs = self._define_kernel_component_functions(
                kernel,
                left_dimension=len(left),
                right_dimension=len(right),
                left_chirality=left_chirality,
                right_chirality=right_chirality,
                result_chirality=result_chirality,
            )
            self._kernel_function_specs[key] = function_specs
        arguments = (
            *left,
            *right,
            *left_momentum,
            *right_momentum,
            *(runtime_parameter_values[name] for name in kernel.runtime_parameters),
        )
        return tuple(
            function(*(arguments[index] for index in argument_indices))
            for function, argument_indices in function_specs
        )

    def _define_kernel_component_functions(
        self,
        kernel: CompiledOrientedKernel,
        *,
        left_dimension: int,
        right_dimension: int,
        left_chirality: int,
        right_chirality: int,
        result_chirality: int,
    ) -> tuple[tuple[_sym.Expression, tuple[int, ...]], ...]:
        key_tag = "_".join(
            (
                str(int(kernel.kind)),
                _chirality_tag(left_chirality),
                _chirality_tag(right_chirality),
                _chirality_tag(result_chirality),
            )
        )
        formal_left = tuple(
            symbols.kernel_function_argument(
                self.name,
                key_tag,
                "left",
                index,
            )
            for index in range(left_dimension)
        )
        formal_right = tuple(
            symbols.kernel_function_argument(
                self.name,
                key_tag,
                "right",
                index,
            )
            for index in range(right_dimension)
        )
        formal_left_momentum = tuple(
            symbols.kernel_function_argument(
                self.name,
                key_tag,
                "left_momentum",
                index,
            )
            for index in range(4)
        )
        formal_right_momentum = tuple(
            symbols.kernel_function_argument(
                self.name,
                key_tag,
                "right_momentum",
                index,
            )
            for index in range(4)
        )
        formal_parameters = {
            name: symbols.kernel_function_argument(
                self.name,
                key_tag,
                "parameter",
                index,
            )
            for index, name in enumerate(kernel.runtime_parameters)
        }
        formal_arguments = (
            *formal_left,
            *formal_right,
            *formal_left_momentum,
            *formal_right_momentum,
            *(formal_parameters[name] for name in kernel.runtime_parameters),
        )
        components = self._projected_kernel_components(
            kernel,
            formal_left,
            formal_right,
            left_chirality=left_chirality,
            right_chirality=right_chirality,
            result_chirality=result_chirality,
            left_momentum=formal_left_momentum,
            right_momentum=formal_right_momentum,
            runtime_parameter_values=formal_parameters,
        )
        coupling_expression = self._resolved_kernel_coupling_expression(
            kernel,
            formal_parameters,
        )
        bodies = tuple(component * coupling_expression for component in components)
        functions = tuple(
            symbols.kernel_function_component(self.name, key_tag, index)
            for index in range(len(bodies))
        )
        specs: list[tuple[_sym.Expression, tuple[int, ...]]] = []
        for function, body in zip(functions, bodies, strict=True):
            body_symbols = set(body.get_all_symbols(False))
            argument_indices = tuple(
                index
                for index, argument in enumerate(formal_arguments)
                if argument in body_symbols
            )
            used_arguments = tuple(
                formal_arguments[index] for index in argument_indices
            )
            self._symbolica_kernel_functions[(function, used_arguments)] = body
            specs.append((function, argument_indices))
        return tuple(specs)

    def _kernel_component_expressions(
        self,
        kernel: CompiledOrientedKernel,
    ) -> tuple[_sym.Expression, ...]:
        cached = self._kernel_component_expression_cache.get(kernel.kind)
        if cached is None:
            cached = tuple(
                _sym.E(component) for component in kernel.component_expressions
            )
            self._kernel_component_expression_cache[kernel.kind] = cached
        return cached

    def _resolved_kernel_coupling_expression(
        self,
        kernel: CompiledOrientedKernel,
        runtime_parameter_values: Mapping[str, Any],
        *,
        numeric: bool = False,
    ) -> Any:
        template = self._kernel_coupling_expression_cache.get(kernel.kind)
        if template is None:
            template = _sym.E(kernel.coupling_expression)
            # Keep the colour-flow normalization algebraic until evaluation.
            # Putting sqrt(2) in a binary64 recurrence weight would freeze its
            # rounding error even when evaluating the process at high precision.
            power = kernel.lc_color_normalization_power
            if power:
                template *= _sym.E(f"2^(-{power}/2)")
            self._kernel_coupling_expression_cache[kernel.kind] = template
        substitutions = {
            symbols.runtime_model_parameter(self.name, name): value
            for name, value in runtime_parameter_values.items()
        }
        if numeric and all(_is_numeric(value) for value in substitutions.values()):
            try:
                return complex(template.evaluate(substitutions))
            except Exception:
                pass
        return _replace_symbols(template, substitutions)

    def _embed_weyl_current(
        self,
        particle_id: int,
        chirality: int,
        values: Sequence[Any],
    ) -> tuple[Any, ...]:
        components = tuple(values)
        if not self.is_chiral_eligible(particle_id):
            return components
        if chirality == 0:
            if len(components) != 4:
                raise ValueError("a full Dirac input must have four components")
            return components
        if len(components) != 2:
            raise ValueError("a projected Weyl input must have two components")
        if chirality == 1:
            return (components[0], components[1], 0.0, 0.0)
        if chirality == -1:
            return (0.0, 0.0, components[0], components[1])
        raise ValueError("a projected Weyl input requires chirality -1, 0, or +1")

    def _propagator_record(self, particle_id: int):
        name = self._particle_records_by_pdg[int(particle_id)].name
        return self._propagator_records_by_particle_name.get(name)

    def _parameter_default(self, name: str) -> complex:
        if name.upper() == "ZERO":
            return 0.0 + 0.0j
        record = self._parameter_records.get(name)
        if record is None:
            raise ValueError(f"model parameter {name!r} is not defined")
        return _record_default(record)

    def _parameter_value(self, name: str) -> Any:
        if name in self._runtime_parameters:
            return self._runtime_parameters[name]
        definition = self._runtime_derived_definitions.get(name)
        if definition is not None:
            return _replace_symbols(
                _sym.E(definition),
                {
                    self._model_symbols.symbol(parameter_name): value
                    for parameter_name, value in self._runtime_parameters.items()
                    if not parameter_name.startswith("derived_coupling_")
                },
            )
        return self._parameter_default(name)

    def _real_parameter_value(self, name: str, *, field: str) -> Any:
        value = self._parameter_value(name)
        if isinstance(value, int | float | complex):
            numeric = complex(value)
            if numeric.imag != 0.0:
                raise ValueError(f"particle {field} parameter {name!r} is not real")
            return numeric.real
        return value

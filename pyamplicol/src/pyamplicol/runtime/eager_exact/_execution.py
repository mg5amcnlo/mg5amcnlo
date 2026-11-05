# SPDX-License-Identifier: 0BSD
"""Correctness-oriented exact execution of a validated eager DAG plan."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import cast

from pyamplicol.api.errors import ArtifactError
from pyamplicol.generation.eager_tables import (
    EAGER_OUTPUT_FACTOR_COUPLING_IMAG,
    EAGER_OUTPUT_FACTOR_COUPLING_REAL,
    EAGER_OUTPUT_FACTOR_NONE,
    MISSING_U32,
)
from pyamplicol.models.prepared import PreparedKernelRecord
from pyamplicol.runtime.eager_exact._contracts import (
    _ZERO,
    _complex_add,
    _complex_mul,
    _complex_zero,
    _resolve_exact_factor,
)
from pyamplicol.runtime.eager_exact._plan import _EagerExactPlan
from pyamplicol.runtime.symbolica_exact import (
    _ComplexDecimal,
    _fill_momenta,
    _fill_sources,
)


def _evaluate_point(
    plan: _EagerExactPlan,
    point: tuple[tuple[Decimal, Decimal, Decimal, Decimal], ...],
    model_parameters: tuple[Decimal, ...],
    prepared_parameters: tuple[_ComplexDecimal, ...],
    precision: int,
) -> tuple[_ComplexDecimal, ...]:
    if len(model_parameters) != plan.parameter_count:
        raise ArtifactError(
            f"eager runtime has {len(model_parameters)} model parameters, "
            f"expected {plan.parameter_count}"
        )
    couplings = _resolve_couplings(plan, model_parameters)
    amplitudes = [_complex_zero() for _ in range(plan.amplitude_count)]
    selector_groups: tuple[int | None, ...] = (
        (None,) if plan.selector_group_ids is None else tuple(plan.selector_group_ids)
    )
    for selector_group_id in selector_groups:
        _evaluate_selector_group(
            plan,
            point,
            model_parameters,
            prepared_parameters,
            couplings,
            precision,
            selector_group_id,
            amplitudes,
        )
    return tuple(amplitudes)


def _evaluate_selector_group(
    plan: _EagerExactPlan,
    point: tuple[tuple[Decimal, Decimal, Decimal, Decimal], ...],
    model_parameters: tuple[Decimal, ...],
    prepared_parameters: tuple[_ComplexDecimal, ...],
    couplings: tuple[_ComplexDecimal, ...],
    precision: int,
    selector_group_id: int | None,
    amplitudes: list[_ComplexDecimal],
) -> None:
    flattened = [
        _complex_zero()
        for _ in range(plan.value_component_count + plan.momentum_component_count)
    ]
    _fill_sources(flattened, point, plan.runtime_schema, model_parameters)
    _fill_momenta(flattened, point, plan.runtime_schema)
    values = flattened[: plan.value_component_count]
    momenta = flattened[plan.value_component_count :]
    currents = [_complex_zero() for _ in range(plan.current_component_count)]
    for stage in plan.stages:
        for invocation in stage.invocations:
            if not _selector_domain_active(
                plan,
                invocation.selector_domain_id,
                selector_group_id,
            ):
                continue
            start = invocation.attachment_start
            stop = start + invocation.attachment_count
            active_attachments = tuple(
                attachment
                for attachment in stage.attachments[start:stop]
                if _selector_domain_active(
                    plan,
                    attachment.selector_domain_id,
                    selector_group_id,
                )
            )
            if not active_attachments:
                continue
            kernel = plan.kernels[invocation.kernel_id]
            inputs = _gather_inputs(
                kernel.record,
                first_current=plan.value_slots[invocation.left_value_slot_id].read(
                    values
                ),
                second_current=plan.value_slots[invocation.right_value_slot_id].read(
                    values
                ),
                first_momentum=plan.momentum_slots[
                    invocation.left_momentum_slot_id
                ].read(momenta),
                second_momentum=plan.momentum_slots[
                    invocation.right_momentum_slot_id
                ].read(momenta),
                coupling=couplings[invocation.coupling_slot_id],
                prepared_parameters=prepared_parameters,
            )
            outputs = kernel.evaluate(inputs, precision)
            output_factor = _resolved_output_factor(
                invocation.output_factor_source,
                couplings[invocation.coupling_slot_id],
            )
            for attachment in active_attachments:
                target = plan.current_slots[attachment.result_current_id]
                factor = _complex_mul(
                    _resolve_exact_factor(attachment.factor), output_factor
                )
                accumulated = target.read(currents)
                target.write(
                    currents,
                    tuple(
                        _complex_add(current, _complex_mul(factor, output))
                        for current, output in zip(accumulated, outputs, strict=True)
                    ),
                    "eager attachment",
                )

        for finalization in stage.finalizations:
            current_slot = plan.current_slots[finalization.current_id]
            current = current_slot.read(currents)
            if (
                finalization.unpropagated_value_slot_id != MISSING_U32
                and _selector_domain_active(
                    plan,
                    finalization.unpropagated_selector_domain_id,
                    selector_group_id,
                )
            ):
                plan.value_slots[finalization.unpropagated_value_slot_id].write(
                    values, current, "unpropagated current finalization"
                )
            if finalization.kernel_id == MISSING_U32 or not _selector_domain_active(
                plan,
                finalization.propagated_selector_domain_id,
                selector_group_id,
            ):
                continue
            kernel = plan.kernels[finalization.kernel_id]
            inputs = _gather_inputs(
                kernel.record,
                first_current=current,
                second_current=(),
                first_momentum=plan.momentum_slots[finalization.momentum_slot_id].read(
                    momenta
                ),
                second_momentum=(),
                coupling=None,
                prepared_parameters=prepared_parameters,
            )
            outputs = kernel.evaluate(inputs, precision)
            plan.value_slots[finalization.propagated_value_slot_id].write(
                values, outputs, "propagated current finalization"
            )

    for closure in plan.closures:
        if selector_group_id is not None and (
            closure.coherent_group_id != selector_group_id
            or not _selector_domain_active(
                plan,
                closure.selector_domain_id,
                selector_group_id,
            )
        ):
            continue
        left = plan.value_slots[closure.left_value_slot_id].read(values)
        right = plan.value_slots[closure.right_value_slot_id].read(values)
        if closure.kernel_id == MISSING_U32:
            coefficients = closure.direct_coefficients
            if coefficients is None:
                raise ArtifactError(
                    "direct eager closure has no exact contraction coefficients"
                )
            output = _complex_zero()
            for coefficient, left_value, right_value in zip(
                coefficients, left, right, strict=True
            ):
                output = _complex_add(
                    output,
                    _complex_mul(coefficient, _complex_mul(left_value, right_value)),
                )
        else:
            kernel = plan.kernels[closure.kernel_id]
            inputs = _gather_inputs(
                kernel.record,
                first_current=left,
                second_current=right,
                first_momentum=(),
                second_momentum=(),
                coupling=couplings[closure.coupling_slot_id],
                prepared_parameters=prepared_parameters,
            )
            output = kernel.evaluate(inputs, precision)[0]
        factor = _resolve_exact_factor(closure.factor)
        if closure.kernel_id != MISSING_U32:
            factor = _complex_mul(
                factor,
                _resolved_output_factor(
                    closure.output_factor_source,
                    couplings[closure.coupling_slot_id],
                ),
            )
        amplitude_id = closure.amplitude_index
        amplitudes[amplitude_id] = _complex_add(
            amplitudes[amplitude_id], _complex_mul(factor, output)
        )


def _selector_domain_active(
    plan: _EagerExactPlan,
    domain_id: int | None,
    selector_group_id: int | None,
) -> bool:
    if plan.selector_domains is None:
        return True
    if domain_id is None or selector_group_id is None:
        raise ArtifactError("compact eager exact execution lacks a selector domain")
    return selector_group_id in plan.selector_domains[domain_id]


def _resolved_output_factor(
    source: int,
    coupling: _ComplexDecimal,
) -> _ComplexDecimal:
    if source == EAGER_OUTPUT_FACTOR_NONE:
        return Decimal(1), Decimal(0)
    if source == EAGER_OUTPUT_FACTOR_COUPLING_REAL:
        return coupling[0], Decimal(0)
    if source == EAGER_OUTPUT_FACTOR_COUPLING_IMAG:
        return coupling[1], Decimal(0)
    raise ArtifactError(f"unsupported eager output factor source {source}")


def _resolve_couplings(
    plan: _EagerExactPlan,
    parameters: Sequence[Decimal],
) -> tuple[_ComplexDecimal, ...]:
    result = []
    for row in plan.couplings:
        real = (
            row.constant[0]
            if row.real_parameter_id == MISSING_U32
            else parameters[row.real_parameter_id]
        )
        imaginary = (
            row.constant[1]
            if row.imag_parameter_id == MISSING_U32
            else parameters[row.imag_parameter_id]
        )
        result.append((real, imaginary))
    return tuple(result)


def _gather_inputs(
    record: PreparedKernelRecord,
    *,
    first_current: Sequence[_ComplexDecimal],
    second_current: Sequence[_ComplexDecimal],
    first_momentum: Sequence[_ComplexDecimal],
    second_momentum: Sequence[_ComplexDecimal],
    coupling: _ComplexDecimal | None,
    prepared_parameters: Sequence[_ComplexDecimal],
) -> tuple[_ComplexDecimal, ...]:
    inputs = []
    for contract in record.input_contracts:
        role = str(contract["role"])
        component = int(cast(int, contract["component"]))
        if role == "left-current":
            value = first_current[component]
        elif role == "right-current":
            value = second_current[component]
        elif role == "current":
            value = first_current[component]
        elif role == "left-momentum":
            value = first_momentum[component]
        elif role == "right-momentum":
            value = second_momentum[component]
        elif role == "momentum":
            value = first_momentum[component]
        elif role == "coupling-real":
            if coupling is None:
                raise ArtifactError("eager kernel requires an unavailable coupling")
            value = (coupling[0], _ZERO)
        elif role == "coupling-imag":
            if coupling is None:
                raise ArtifactError("eager kernel requires an unavailable coupling")
            value = (coupling[1], _ZERO)
        else:
            parameter_id = cast(int, contract["model_parameter_index"])
            value = prepared_parameters[parameter_id]
        inputs.append(value)
    return tuple(inputs)

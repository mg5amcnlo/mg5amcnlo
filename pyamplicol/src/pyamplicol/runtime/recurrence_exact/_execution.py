# SPDX-License-Identifier: 0BSD
"""Decimal/Symbolica execution of one authenticated Direct-Arena plan."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from pyamplicol.api.errors import ArtifactError, CompatibilityError, EvaluationError
from pyamplicol.runtime.eager_exact._contracts import (
    _complex_add,
    _complex_mul,
    _complex_zero,
)
from pyamplicol.runtime.symbolica_exact import (
    _antiquark_dirac,
    _antiquark_weyl,
    _ComplexDecimal,
    _massive_vector,
    _massless_vector,
    _quark_dirac,
    _quark_weyl,
    _spin2,
)

from ._plan import _RecurrenceExactPlan, _SourceTemplate
from ._plan_v2 import (
    DIRECT_NONE_U32,
    _Closure,
    _Contribution,
    _Current,
    _Executor,
    _Finalization,
    _ReplayTarget,
    _ResolvedHelicity,
    _Source,
    _SourceDispatchVariant,
)

_ZERO = Decimal(0)
_ONE = Decimal(1)
_ROLE_SOURCE = 0
_ROLE_CONTRIBUTION = 1
_ROLE_FINALIZATION = 2
_ROLE_CLOSURE = 3
_NODE_CURRENT = 1
_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION = 1 << 0
_CONTRIBUTION_FLAG_CERTIFIED_REUSE = 1 << 1
_CERTIFIED_REUSE_FLAGS = (
    _CONTRIBUTION_FLAG_INITIALIZE_DESTINATION | _CONTRIBUTION_FLAG_CERTIFIED_REUSE
)

_Point = Sequence[tuple[Decimal, Decimal, Decimal, Decimal]]
_CurrentObserver = Callable[[int, tuple[_ComplexDecimal, ...]], None]
_FactorCache = list[_ComplexDecimal | None]

_INPUT_CURRENT = 0
_INPUT_MOMENTUM = 1
_INPUT_PARAMETER = 2
_INPUT_CONSTANT = 3


@dataclass(frozen=True, slots=True)
class _TrustedInputRecipe:
    kind: int
    index: int
    component: int
    constant: _ComplexDecimal | None = None


@dataclass(frozen=True, slots=True)
class _TrustedExecutorRecipe:
    kernel: Any
    inputs: tuple[_TrustedInputRecipe, ...]


def _evaluate_replay_point(
    plan: _RecurrenceExactPlan,
    point: _Point,
    target: _ReplayTarget,
    prepared_parameters: Sequence[_ComplexDecimal],
    precision: int,
    *,
    current_observer: _CurrentObserver | None = None,
) -> tuple[_ComplexDecimal, ...]:
    sections = plan.sections
    permutation = sections.source_permutations[
        target.source_permutation_start : target.source_permutation_start
        + target.source_permutation_count
    ]
    if len(permutation) != sections.external_source_count:
        raise ArtifactError("recurrence replay permutation has invalid width")
    momentum_signs = sections.replay_momentum_signs[
        target.source_permutation_start : target.source_permutation_start
        + target.source_permutation_count
    ]
    if len(momentum_signs) != sections.external_source_count:
        raise ArtifactError("recurrence replay momentum-sign map has invalid width")
    momenta = _momentum_forms(plan, point, permutation, momentum_signs)
    amplitudes = list(
        _execute_schedule(
            plan,
            momenta,
            prepared_parameters,
            precision,
            selected_source_variants=None,
            current_observer=current_observer,
        )
    )

    replay_factor = _complex_mul(
        _factor(plan, target.phase_factor_id),
        (Decimal(target.multiplicity), _ZERO),
    )
    for destination in sections.amplitude_destinations:
        if destination.target_sector_id == target.representative_id:
            amplitudes[destination.destination_id] = _complex_mul(
                amplitudes[destination.destination_id],
                replay_factor,
            )
    return tuple(amplitudes)


def _evaluate_union_point(
    plan: _RecurrenceExactPlan,
    point: _Point,
    helicity: _ResolvedHelicity,
    prepared_parameters: Sequence[_ComplexDecimal],
    precision: int,
    *,
    current_observer: _CurrentObserver | None = None,
) -> tuple[_ComplexDecimal, ...]:
    """Execute one all-flow union once for one runtime-selected helicity."""

    sections = plan.sections
    if sections.strategy != "all-flow-union":
        raise ArtifactError("union exact execution requires an all-flow-union plan")
    selections = sections.resolved_source_selections[
        helicity.source_selection_start : helicity.source_selection_start
        + helicity.source_selection_count
    ]
    if len(selections) != sections.external_source_count:
        raise ArtifactError(
            "all-flow-union helicity does not select every external source"
        )
    selected_source_variants = {}
    for source_slot, selection in enumerate(selections):
        if selection.source_slot != source_slot:
            raise ArtifactError(
                "all-flow-union source selections are not in source-slot order"
            )
        try:
            variant = sections.source_dispatch_variants[selection.dispatch_variant_id]
        except IndexError as exc:
            raise ArtifactError(
                "all-flow-union source selection references an absent variant"
            ) from exc
        try:
            source = sections.sources[variant.source_row_id]
        except IndexError as exc:
            raise ArtifactError(
                "all-flow-union source variant references an absent source row"
            ) from exc
        if source.source_slot != source_slot:
            raise ArtifactError(
                "all-flow-union source variant selects the wrong external source"
            )
        if variant.source_row_id in selected_source_variants:
            raise ArtifactError(
                "all-flow-union helicity selects one source row more than once"
            )
        selected_source_variants[variant.source_row_id] = variant
    momenta = _momentum_forms(
        plan,
        point,
        tuple(range(sections.external_source_count)),
    )
    return _execute_schedule(
        plan,
        momenta,
        prepared_parameters,
        precision,
        selected_source_variants=selected_source_variants,
        current_observer=current_observer,
    )


def _evaluate_contracted_point(
    plan: _RecurrenceExactPlan,
    point: _Point,
    prepared_parameters: Sequence[_ComplexDecimal],
    precision: int,
    *,
    current_observer: _CurrentObserver | None = None,
) -> tuple[_ComplexDecimal, ...]:
    """Execute the fixed-source contracted-color schedule once."""

    sections = plan.sections
    if sections.strategy != "contracted-color-union":
        raise ArtifactError(
            "contracted exact execution requires a contracted-color plan"
        )
    momenta = _momentum_forms(
        plan,
        point,
        tuple(range(sections.external_source_count)),
    )
    return _execute_schedule(
        plan,
        momenta,
        prepared_parameters,
        precision,
        selected_source_variants=None,
        current_observer=current_observer,
    )


def _execute_schedule(
    plan: _RecurrenceExactPlan,
    momenta: Sequence[Sequence[Decimal]],
    prepared_parameters: Sequence[_ComplexDecimal],
    precision: int,
    *,
    selected_source_variants: Mapping[int, _SourceDispatchVariant] | None,
    current_observer: _CurrentObserver | None = None,
) -> tuple[_ComplexDecimal, ...]:
    sections = plan.sections
    zero = _complex_zero()
    arena = [zero] * sections.current_arena_components
    amplitudes = [zero] * sections.amplitude_destination_count
    factor_cache: _FactorCache = [None] * len(sections.exact_factors)
    initialized_contribution_stage: int | None = None
    active_stage: int | None = None

    for group in sections.row_groups:
        if active_stage is not None and group.stage != active_stage:
            _observe_stage_currents(
                current_observer,
                plan,
                active_stage,
                arena,
            )
        active_stage = group.stage
        start = group.row_start
        stop = start + group.row_count
        if group.role == _ROLE_SOURCE:
            if selected_source_variants is not None:
                if group.executor_id != DIRECT_NONE_U32:
                    raise ArtifactError(
                        "all-flow-union source row group must use runtime dispatch"
                    )
            else:
                executor = plan.executors.get(group.executor_id)
                if executor is None or _role_index(executor.role) != group.role:
                    raise ArtifactError(
                        "recurrence source row group references invalid executor "
                        f"{group.executor_id}"
                    )
            for source_row_id in range(start, stop):
                source_row = sections.sources[source_row_id]
                if selected_source_variants is None:
                    _execute_source(
                        plan,
                        source_row,
                        momenta,
                        prepared_parameters,
                        arena,
                        factor_cache=factor_cache,
                    )
                    continue
                variant = selected_source_variants.get(source_row_id)
                if variant is not None:
                    _execute_union_source(
                        plan,
                        source_row,
                        variant,
                        momenta,
                        prepared_parameters,
                        arena,
                        factor_cache=factor_cache,
                    )
            continue

        if group.role == _ROLE_CONTRIBUTION:
            if initialized_contribution_stage != group.stage:
                _clear_stage(plan, arena, group.stage)
                initialized_contribution_stage = group.stage
            if group.executor_id == DIRECT_NONE_U32:
                for contribution_row in sections.contributions[start:stop]:
                    _execute_certified_reuse_row(
                        plan,
                        contribution_row,
                        arena,
                        factor_cache=factor_cache,
                    )
                continue
        executor = plan.executors.get(group.executor_id)
        if executor is None or _role_index(executor.role) != group.role:
            raise ArtifactError(
                f"recurrence row group references invalid executor {group.executor_id}"
            )
        if group.role == _ROLE_CONTRIBUTION:
            for contribution_row in sections.contributions[start:stop]:
                _execute_prepared_row(
                    plan,
                    executor,
                    contribution_row,
                    momenta,
                    prepared_parameters,
                    arena,
                    amplitudes,
                    precision,
                    factor_cache=factor_cache,
                )
        elif group.role == _ROLE_FINALIZATION:
            for finalization_row in sections.finalizations[start:stop]:
                _execute_finalization(
                    plan,
                    executor,
                    finalization_row,
                    momenta,
                    prepared_parameters,
                    arena,
                    amplitudes,
                    precision,
                    factor_cache=factor_cache,
                )
        elif group.role == _ROLE_CLOSURE:
            for closure_row in sections.closures[start:stop]:
                if executor.runtime_template is not None:
                    _execute_intrinsic_closure(
                        plan,
                        closure_row,
                        arena,
                        amplitudes,
                        factor_cache=factor_cache,
                    )
                else:
                    _execute_prepared_row(
                        plan,
                        executor,
                        closure_row,
                        momenta,
                        prepared_parameters,
                        arena,
                        amplitudes,
                        precision,
                        factor_cache=factor_cache,
                    )
        else:  # pragma: no cover - native plan validation rejects this
            raise ArtifactError(f"unsupported recurrence row role {group.role}")
    if active_stage is not None:
        _observe_stage_currents(
            current_observer,
            plan,
            active_stage,
            arena,
        )
    return tuple(amplitudes)


def _observe_stage_currents(
    observer: _CurrentObserver | None,
    plan: _RecurrenceExactPlan,
    stage: int,
    arena: Sequence[_ComplexDecimal],
) -> None:
    if observer is None:
        return
    for current in _currents_for_stage(plan, stage):
        values = tuple(
            _arena_value(arena, current.component_base, component)
            for component in range(current.component_count)
        )
        observer(current.semantic_id, values)


def _currents_for_stage(
    plan: _RecurrenceExactPlan,
    stage: int,
) -> tuple[_Current, ...]:
    cached = plan.trusted_currents_by_stage
    if not cached:
        grouped: dict[int, list[_Current]] = {}
        for current in plan.sections.currents:
            grouped.setdefault(current.stage, []).append(current)
        cached.update(
            (stage_id, tuple(currents)) for stage_id, currents in grouped.items()
        )
    return cached.get(stage, ())


def _execute_certified_reuse_row(
    plan: _RecurrenceExactPlan,
    row: _Contribution,
    arena: list[_ComplexDecimal],
    *,
    factor_cache: _FactorCache | None = None,
) -> None:
    if (
        row.flags != _CERTIFIED_REUSE_FLAGS
        or row.parent1_base != DIRECT_NONE_U32
        or row.parent1_momentum != DIRECT_NONE_U32
        or row.parent0_momentum <= 0
    ):
        raise ArtifactError("certified-reuse contribution has an invalid encoding")
    factor = _factor(plan, row.exact_factor_id, cache=factor_cache)
    component_count = row.parent0_momentum
    values = tuple(
        _complex_mul(
            _arena_value(arena, row.parent0_base, component),
            factor,
        )
        for component in range(component_count)
    )
    _write(
        arena,
        row.destination_base,
        values,
        replace=True,
    )


def _momentum_forms(
    plan: _RecurrenceExactPlan,
    point: _Point,
    permutation: Sequence[int],
    momentum_signs: Sequence[int] | None = None,
) -> tuple[tuple[Decimal, Decimal, Decimal, Decimal], ...]:
    result = []
    terms = plan.sections.momentum_terms
    for form in plan.sections.momentum_forms:
        values = [_ZERO, _ZERO, _ZERO, _ZERO]
        for term in terms[form.term_start : form.term_start + form.term_count]:
            try:
                external_slot = permutation[term.source_slot]
                source = point[external_slot]
            except IndexError as exc:
                raise ArtifactError(
                    "recurrence momentum form references an absent external source"
                ) from exc
            replay_sign = (
                1 if momentum_signs is None else momentum_signs[term.source_slot]
            )
            coefficient = Decimal(term.coefficient * replay_sign)
            for component in range(4):
                values[component] += coefficient * source[component]
        result.append((values[0], values[1], values[2], values[3]))
    return tuple(result)


def _clear_stage(
    plan: _RecurrenceExactPlan,
    arena: list[_ComplexDecimal],
    stage: int,
) -> None:
    for current in _currents_for_stage(plan, stage):
        if current.node_kind != _NODE_CURRENT:
            continue
        stop = current.component_base + current.component_count
        arena[current.component_base : stop] = [
            _complex_zero()
        ] * current.component_count


def _execute_source(
    plan: _RecurrenceExactPlan,
    row: _Source,
    momenta: Sequence[Sequence[Decimal]],
    prepared_parameters: Sequence[_ComplexDecimal],
    arena: list[_ComplexDecimal],
    *,
    factor_cache: _FactorCache | None = None,
) -> None:
    source = row
    template = plan.source_templates[source.source_template_or_dispatch_domain]
    initial = source.source_slot in plan.initial_source_slots
    helicity = template.helicity * (template.crossing_helicity_factor if initial else 1)
    chirality = template.chirality * (
        template.crossing_chirality_factor if initial else 1
    )
    spin_state = template.spin_state * (
        template.crossing_spin_state_factor if initial else 1
    )
    if spin_state != source.spin_state_class:
        raise ArtifactError("recurrence source spin-state contract is inconsistent")
    try:
        momentum = momenta[source.momentum_form_id]
    except IndexError as exc:
        raise ArtifactError("recurrence source momentum form is absent") from exc
    mass = _ZERO
    if template.mass_prepared_parameter_id is not None:
        try:
            mass_value = prepared_parameters[template.mass_prepared_parameter_id]
        except IndexError as exc:
            raise ArtifactError(
                "recurrence source mass parameter is out of range"
            ) from exc
        if mass_value[1] != _ZERO:
            raise EvaluationError("recurrence source mass must be real")
        mass = mass_value[0]
    wave = _source_wavefunction(template, momentum, helicity, chirality, mass)
    if len(wave) != template.dimension:
        raise ArtifactError("recurrence source wavefunction has the wrong dimension")
    factor = _factor(plan, source.exact_factor_id, cache=factor_cache)
    values = tuple(_complex_mul(value, factor) for value in wave)
    _write(arena, source.destination_base, values, replace=True)


def _execute_union_source(
    plan: _RecurrenceExactPlan,
    source: _Source,
    variant: _SourceDispatchVariant,
    momenta: Sequence[Sequence[Decimal]],
    prepared_parameters: Sequence[_ComplexDecimal],
    arena: list[_ComplexDecimal],
    *,
    factor_cache: _FactorCache | None = None,
) -> None:
    if (
        source.source_template_or_dispatch_domain != variant.dispatch_domain_id
        or variant.source_row_id >= len(plan.sections.sources)
        or plan.sections.sources[variant.source_row_id] != source
    ):
        raise ArtifactError(
            "all-flow-union source variant does not match its dispatch row"
        )
    try:
        template = plan.source_templates[variant.source_template_id]
    except KeyError as exc:
        raise ArtifactError(
            "all-flow-union source variant references an absent source template"
        ) from exc
    try:
        momentum = momenta[source.momentum_form_id]
    except IndexError as exc:
        raise ArtifactError("recurrence source momentum form is absent") from exc

    initial = source.source_slot in plan.initial_source_slots
    helicity = template.helicity * (template.crossing_helicity_factor if initial else 1)
    chirality = template.chirality * (
        template.crossing_chirality_factor if initial else 1
    )
    spin_state = template.spin_state * (
        template.crossing_spin_state_factor if initial else 1
    )
    if spin_state != variant.crossed_spin_state_class:
        raise ArtifactError(
            "all-flow-union source variant has inconsistent crossed spin state"
        )

    mass = _ZERO
    if template.mass_prepared_parameter_id is not None:
        try:
            mass_value = prepared_parameters[template.mass_prepared_parameter_id]
        except IndexError as exc:
            raise ArtifactError(
                "recurrence source mass parameter is out of range"
            ) from exc
        if mass_value[1] != _ZERO:
            raise EvaluationError("recurrence source mass must be real")
        mass = mass_value[0]
    wave = _source_wavefunction(template, momentum, helicity, chirality, mass)
    if len(wave) != variant.projection_count:
        raise ArtifactError(
            "all-flow-union source wavefunction has the wrong projected dimension"
        )

    embeddings = plan.sections.source_embeddings[
        variant.embedding_start : variant.embedding_start + variant.embedding_count
    ]
    if len(embeddings) != variant.embedding_count:
        raise ArtifactError("all-flow-union source embedding is out of bounds")
    source_factor = _complex_mul(
        _factor(plan, source.exact_factor_id, cache=factor_cache),
        _factor(plan, variant.crossing_exact_factor_id, cache=factor_cache),
    )
    values = []
    for full_component, embedding in enumerate(embeddings):
        if embedding.full_component != full_component:
            raise ArtifactError(
                "all-flow-union source embedding is not in component order"
            )
        if embedding.source_component == DIRECT_NONE_U32:
            values.append(_complex_zero())
            continue
        try:
            value = wave[embedding.source_component]
        except IndexError as exc:
            raise ArtifactError(
                "all-flow-union source embedding references an absent component"
            ) from exc
        values.append(
            _complex_mul(
                _complex_mul(value, source_factor),
                _factor(plan, embedding.exact_factor_id, cache=factor_cache),
            )
        )
    _write(arena, source.destination_base, values, replace=True)


def _source_wavefunction(
    template: _SourceTemplate,
    momentum: Sequence[Decimal],
    helicity: int,
    chirality: int,
    mass: Decimal,
) -> tuple[_ComplexDecimal, ...]:
    if template.dimension == 1 and template.family == "scalar":
        return ((_ONE, _ZERO),)
    if template.family == "fermion" and template.orientation == "self-conjugate":
        raise CompatibilityError(
            "self-conjugate fermion source wavefunctions are unsupported"
        )
    if template.dimension == 2 and template.family == "fermion":
        return (
            _antiquark_weyl(momentum, helicity, chirality)
            if template.orientation == "antiparticle"
            else _quark_weyl(momentum, helicity, chirality)
        )
    if template.dimension == 4 and template.family == "fermion":
        return (
            _antiquark_dirac(momentum, helicity, mass)
            if template.orientation == "antiparticle"
            else _quark_dirac(momentum, helicity, mass)
        )
    if template.dimension == 4 and template.family == "vector":
        return (
            _massless_vector(momentum, helicity)
            if mass == _ZERO
            else _massive_vector(momentum, helicity, mass)
        )
    if template.dimension == 16 and template.family == "spin2":
        return _spin2(momentum, helicity, mass)
    raise CompatibilityError(
        f"exact recurrence source {template.family!r} with dimension "
        f"{template.dimension} is unsupported"
    )


def _execute_finalization(
    plan: _RecurrenceExactPlan,
    executor: _Executor,
    row: _Finalization,
    momenta: Sequence[Sequence[Decimal]],
    prepared_parameters: Sequence[_ComplexDecimal],
    arena: list[_ComplexDecimal],
    amplitudes: list[_ComplexDecimal],
    precision: int,
    *,
    factor_cache: _FactorCache | None = None,
) -> None:
    if executor.runtime_template is not None:
        if executor.runtime_template == "rusticol.identity-finalize-in-place.v1":
            factor = _factor(plan, row.exact_factor_id, cache=factor_cache)
            stop = row.component_base + row.component_count
            arena[row.component_base : stop] = [
                _complex_mul(value, factor)
                for value in arena[row.component_base : stop]
            ]
            return
        if executor.executor_id not in plan.executor_exact_kernel_ids:
            raise CompatibilityError(
                f"unsupported exact recurrence finalization intrinsic "
                f"{executor.runtime_template!r}"
            )
    _execute_prepared_row(
        plan,
        executor,
        row,
        momenta,
        prepared_parameters,
        arena,
        amplitudes,
        precision,
        factor_cache=factor_cache,
    )


def _execute_prepared_row(
    plan: _RecurrenceExactPlan,
    executor: _Executor,
    row: _Contribution | _Finalization | _Closure,
    momenta: Sequence[Sequence[Decimal]],
    prepared_parameters: Sequence[_ComplexDecimal],
    arena: list[_ComplexDecimal],
    amplitudes: list[_ComplexDecimal],
    precision: int,
    *,
    factor_cache: _FactorCache | None = None,
) -> None:
    recipe = _trusted_executor_recipe(plan, executor)
    inputs = _trusted_kernel_inputs(
        plan,
        recipe.inputs,
        executor,
        row,
        momenta,
        prepared_parameters,
        arena,
    )
    outputs = recipe.kernel.evaluate(inputs, precision)
    factor = _factor(plan, row.exact_factor_id, cache=factor_cache)
    scaled = tuple(_complex_mul(value, factor) for value in outputs)
    if executor.role == "contribution" and isinstance(row, _Contribution):
        _write(arena, row.destination_base, scaled, replace=False)
    elif executor.role == "finalization" and isinstance(row, _Finalization):
        _write(arena, row.component_base, scaled, replace=True)
    elif executor.role == "closure" and isinstance(row, _Closure):
        if len(scaled) != executor.destination_component_count:
            raise ArtifactError("recurrence closure output width is inconsistent")
        _write(
            amplitudes,
            row.amplitude_destination_id,
            scaled,
            replace=False,
        )
    else:
        raise ArtifactError(
            f"prepared direct executor has unsupported role {executor.role!r}"
        )


def _trusted_executor_recipe(
    plan: _RecurrenceExactPlan,
    executor: _Executor,
) -> _TrustedExecutorRecipe:
    cached = plan.trusted_executor_recipes.get(executor.executor_id)
    if cached is not None:
        if not isinstance(cached, _TrustedExecutorRecipe):
            raise ArtifactError(
                f"direct executor {executor.executor_id} has an invalid "
                "trusted exact recipe"
            )
        return cached

    kernel_id = plan.executor_exact_kernel_ids.get(
        executor.executor_id,
        executor.prepared_kernel_id,
    )
    if kernel_id is None:
        raise ArtifactError(
            f"direct executor {executor.executor_id} has no prepared exact kernel"
        )
    kernel = plan.kernels.get(kernel_id)
    if kernel is None:
        raise ArtifactError(f"prepared exact kernel {kernel_id} is absent")
    recipe = _TrustedExecutorRecipe(
        kernel=kernel,
        inputs=_compile_trusted_inputs(
            plan,
            kernel.record.input_contracts,
            executor,
        ),
    )
    plan.trusted_executor_recipes[executor.executor_id] = recipe
    return recipe


def _compile_trusted_inputs(
    plan: _RecurrenceExactPlan,
    contracts: Sequence[Mapping[str, object]],
    executor: _Executor,
) -> tuple[_TrustedInputRecipe, ...]:
    result = []
    for index, contract in enumerate(contracts):
        role = contract.get("role")
        component = contract.get("component")
        if (
            isinstance(component, bool)
            or not isinstance(component, int)
            or component < 0
        ):
            raise ArtifactError(
                f"prepared kernel input {index} has an invalid component"
            )
        if role in {"left-current", "current"}:
            result.append(
                _TrustedInputRecipe(
                    _INPUT_CURRENT,
                    _prepared_parent_operand(plan, executor, 0),
                    component,
                )
            )
        elif role == "right-current":
            result.append(
                _TrustedInputRecipe(
                    _INPUT_CURRENT,
                    _prepared_parent_operand(plan, executor, 1),
                    component,
                )
            )
        elif role in {"left-momentum", "momentum"}:
            result.append(
                _TrustedInputRecipe(
                    _INPUT_MOMENTUM,
                    _prepared_parent_operand(plan, executor, 0),
                    component,
                )
            )
        elif role == "right-momentum":
            result.append(
                _TrustedInputRecipe(
                    _INPUT_MOMENTUM,
                    _prepared_parent_operand(plan, executor, 1),
                    component,
                )
            )
        elif role in {"coupling-real", "coupling-imag"}:
            coupling = plan.executor_couplings.get(executor.executor_id)
            if coupling is None:
                raise ArtifactError(
                    f"direct executor {executor.executor_id} has no exact coupling"
                )
            result.append(
                _TrustedInputRecipe(
                    _INPUT_CONSTANT,
                    0,
                    component,
                    (
                        coupling[0 if role == "coupling-real" else 1],
                        _ZERO,
                    ),
                )
            )
        elif role == "model-parameter":
            parameter_id = contract.get("model_parameter_index")
            if (
                isinstance(parameter_id, bool)
                or not isinstance(parameter_id, int)
                or parameter_id < 0
            ):
                raise ArtifactError(
                    f"prepared kernel input {index} has no model-parameter index"
                )
            result.append(
                _TrustedInputRecipe(
                    _INPUT_PARAMETER,
                    parameter_id,
                    component,
                )
            )
        else:
            raise CompatibilityError(
                f"unsupported exact recurrence kernel input role {role!r}"
            )
    return tuple(result)


def _trusted_kernel_inputs(
    plan: _RecurrenceExactPlan,
    recipes: Sequence[_TrustedInputRecipe],
    executor: _Executor,
    row: _Contribution | _Finalization | _Closure,
    momenta: Sequence[Sequence[Decimal]],
    prepared_parameters: Sequence[_ComplexDecimal],
    arena: Sequence[_ComplexDecimal],
) -> tuple[_ComplexDecimal, ...]:
    result = []
    current_bases: list[int | None] = [None, None]
    momentum_form_ids: list[int | None] = [None, None]
    for recipe in recipes:
        if recipe.kind == _INPUT_CURRENT:
            base = current_bases[recipe.index]
            if base is None:
                base = _parent_component_base(
                    executor.role,
                    row,
                    recipe.index,
                )
                current_bases[recipe.index] = base
            result.append(
                _arena_value(
                    arena,
                    base,
                    recipe.component,
                )
            )
        elif recipe.kind == _INPUT_MOMENTUM:
            form_id = momentum_form_ids[recipe.index]
            if form_id is None:
                form_id = _momentum_form_id(
                    executor.role,
                    row,
                    recipe.index,
                )
                momentum_form_ids[recipe.index] = form_id
            result.append(
                _momentum_value(
                    momenta,
                    form_id,
                    recipe.component,
                )
            )
        elif recipe.kind == _INPUT_PARAMETER:
            result.append(_parameter(prepared_parameters, recipe.index))
        elif recipe.kind == _INPUT_CONSTANT and recipe.constant is not None:
            result.append(recipe.constant)
        else:  # pragma: no cover - recipes are constructed above
            raise ArtifactError(
                f"direct executor {executor.executor_id} has an invalid "
                "trusted input recipe"
            )
    return tuple(result)


def _kernel_inputs(
    plan: _RecurrenceExactPlan,
    contracts: Sequence[Mapping[str, object]],
    executor: _Executor,
    row: _Contribution | _Finalization | _Closure,
    momenta: Sequence[Sequence[Decimal]],
    prepared_parameters: Sequence[_ComplexDecimal],
    arena: Sequence[_ComplexDecimal],
) -> tuple[_ComplexDecimal, ...]:
    result = []
    for index, contract in enumerate(contracts):
        role = contract.get("role")
        component = contract.get("component")
        if (
            isinstance(component, bool)
            or not isinstance(component, int)
            or component < 0
        ):
            raise ArtifactError(
                f"prepared kernel input {index} has an invalid component"
            )
        if role in {"left-current", "current"}:
            parent = _prepared_parent_operand(plan, executor, 0)
            result.append(
                _arena_value(
                    arena,
                    _parent_component_base(executor.role, row, parent),
                    component,
                )
            )
        elif role == "right-current":
            parent = _prepared_parent_operand(plan, executor, 1)
            result.append(
                _arena_value(
                    arena,
                    _parent_component_base(executor.role, row, parent),
                    component,
                )
            )
        elif role in {"left-momentum", "momentum"}:
            operand = _prepared_parent_operand(plan, executor, 0)
            result.append(
                _momentum_value(
                    momenta,
                    _momentum_form_id(executor.role, row, operand),
                    component,
                )
            )
        elif role == "right-momentum":
            operand = _prepared_parent_operand(plan, executor, 1)
            result.append(
                _momentum_value(
                    momenta,
                    _momentum_form_id(executor.role, row, operand),
                    component,
                )
            )
        elif role == "coupling-real":
            coupling = plan.executor_couplings.get(executor.executor_id)
            if coupling is None:
                raise ArtifactError(
                    f"direct executor {executor.executor_id} has no exact coupling"
                )
            result.append((coupling[0], _ZERO))
        elif role == "coupling-imag":
            coupling = plan.executor_couplings.get(executor.executor_id)
            if coupling is None:
                raise ArtifactError(
                    f"direct executor {executor.executor_id} has no exact coupling"
                )
            result.append((coupling[1], _ZERO))
        elif role == "model-parameter":
            parameter_id = contract.get("model_parameter_index")
            if (
                isinstance(parameter_id, bool)
                or not isinstance(parameter_id, int)
                or parameter_id < 0
            ):
                raise ArtifactError(
                    f"prepared kernel input {index} has no model-parameter index"
                )
            result.append(_parameter(prepared_parameters, parameter_id))
        else:
            raise CompatibilityError(
                f"unsupported exact recurrence kernel input role {role!r}"
            )
    return tuple(result)


def _prepared_parent_operand(
    plan: _RecurrenceExactPlan,
    executor: _Executor,
    operand: int,
) -> int:
    if executor.role != "contribution":
        return operand
    permutation = plan.executor_parent_permutations.get(executor.executor_id)
    if permutation not in {(0, 1), (1, 0)}:
        raise ArtifactError(
            f"direct contribution executor {executor.executor_id} has no "
            "authenticated parent permutation"
        )
    try:
        return permutation[operand]
    except IndexError as exc:  # pragma: no cover - kernel roles are validated
        raise ArtifactError(
            f"direct contribution executor {executor.executor_id} references "
            f"invalid parent operand {operand}"
        ) from exc


def _parent_component_base(
    role: str,
    row: _Contribution | _Finalization | _Closure,
    parent: int,
) -> int:
    if role == "finalization":
        if parent == 0 and isinstance(row, _Finalization):
            return row.component_base
    elif role in {"contribution", "closure"} and not isinstance(row, _Finalization):
        if parent == 0:
            return row.parent0_base
        if parent == 1 and row.parent1_base != DIRECT_NONE_U32:
            return row.parent1_base
    raise ArtifactError(f"recurrence {role} row has no parent current {parent}")


def _momentum_form_id(
    role: str,
    row: _Contribution | _Finalization | _Closure,
    operand: int,
) -> int:
    if role == "finalization":
        if operand == 0 and isinstance(row, _Finalization):
            return row.momentum_form_id
    elif role in {"contribution", "closure"} and not isinstance(row, _Finalization):
        if operand == 0:
            return row.parent0_momentum
        if operand == 1 and row.parent1_momentum != DIRECT_NONE_U32:
            return row.parent1_momentum
    raise ArtifactError(f"recurrence {role} row has no momentum operand {operand}")


def _execute_intrinsic_closure(
    plan: _RecurrenceExactPlan,
    row: _Closure,
    arena: Sequence[_ComplexDecimal],
    amplitudes: list[_ComplexDecimal],
    *,
    factor_cache: _FactorCache | None = None,
) -> None:
    if row.parent1_base == DIRECT_NONE_U32 or row.component_count == 0:
        raise ArtifactError("recurrence intrinsic closure has invalid parents")
    row_factor = _factor(plan, row.exact_factor_id, cache=factor_cache)
    value = _complex_zero()
    for component in range(row.component_count):
        coefficient = _factor(
            plan,
            row.component_factor_start + component,
            cache=factor_cache,
        )
        left = _arena_value(arena, row.parent0_base, component)
        right = _arena_value(arena, row.parent1_base, component)
        value = _complex_add(
            value,
            _complex_mul(_complex_mul(left, right), coefficient),
        )
    value = _complex_mul(value, row_factor)
    amplitudes[row.amplitude_destination_id] = _complex_add(
        amplitudes[row.amplitude_destination_id],
        value,
    )


def _factor(
    plan: _RecurrenceExactPlan,
    factor_id: int,
    *,
    cache: _FactorCache | None = None,
) -> _ComplexDecimal:
    if cache is not None:
        try:
            cached = cache[factor_id]
        except IndexError as exc:
            raise ArtifactError("recurrence exact factor is out of range") from exc
        if cached is not None:
            return cached
    try:
        value = plan.sections.exact_factors[factor_id]
    except IndexError as exc:
        raise ArtifactError("recurrence exact factor is out of range") from exc
    resolved = (
        Decimal(value.real_numerator) / Decimal(value.real_denominator),
        Decimal(value.imaginary_numerator) / Decimal(value.imaginary_denominator),
    )
    if cache is not None:
        cache[factor_id] = resolved
    return resolved


def _parameter(
    parameters: Sequence[_ComplexDecimal],
    parameter_id: int,
) -> _ComplexDecimal:
    try:
        return parameters[parameter_id]
    except IndexError as exc:
        raise ArtifactError("recurrence prepared parameter is out of range") from exc


def _arena_value(
    values: Sequence[_ComplexDecimal],
    base: int,
    component: int,
) -> _ComplexDecimal:
    try:
        return values[base + component]
    except IndexError as exc:
        raise ArtifactError("recurrence component reference is out of range") from exc


def _momentum_value(
    momenta: Sequence[Sequence[Decimal]],
    form_id: int,
    component: int,
) -> _ComplexDecimal:
    try:
        return momenta[form_id][component], _ZERO
    except IndexError as exc:
        raise ArtifactError("recurrence momentum reference is out of range") from exc


def _write(
    values: list[_ComplexDecimal],
    start: int,
    entries: Sequence[_ComplexDecimal],
    *,
    replace: bool,
) -> None:
    stop = start + len(entries)
    if start < 0 or stop > len(values):
        raise ArtifactError("recurrence destination range is out of bounds")
    if replace:
        values[start:stop] = entries
    else:
        values[start:stop] = [
            _complex_add(previous, value)
            for previous, value in zip(values[start:stop], entries, strict=True)
        ]


def _role_index(role: str) -> int:
    try:
        return {
            "source": _ROLE_SOURCE,
            "contribution": _ROLE_CONTRIBUTION,
            "finalization": _ROLE_FINALIZATION,
            "closure": _ROLE_CLOSURE,
        }[role]
    except KeyError as exc:
        raise ArtifactError(f"unsupported direct executor role {role!r}") from exc


__all__ = [
    "_evaluate_contracted_point",
    "_evaluate_replay_point",
    "_evaluate_union_point",
]

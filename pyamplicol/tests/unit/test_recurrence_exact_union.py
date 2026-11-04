# SPDX-License-Identifier: 0BSD
"""Exact all-flow-union execution over synthetic Direct-Arena v2 tables."""

from __future__ import annotations

import json
from dataclasses import astuple, replace
from decimal import Decimal, localcontext
from types import SimpleNamespace

import pytest

from pyamplicol.api.errors import ArtifactError, EvaluationError
from pyamplicol.runtime.recurrence_exact import _executor as executor_module
from pyamplicol.runtime.recurrence_exact._execution import (
    _clear_stage,
    _evaluate_union_point,
    _execute_certified_reuse_row,
    _execute_union_source,
    _kernel_inputs,
    _trusted_executor_recipe,
    _trusted_kernel_inputs,
)
from pyamplicol.runtime.recurrence_exact._execution import (
    _factor as _resolved_factor,
)
from pyamplicol.runtime.recurrence_exact._executor import RecurrenceExactExecutor
from pyamplicol.runtime.recurrence_exact._plan import (
    _RecurrenceExactPlan,
    _SourceTemplate,
)
from pyamplicol.runtime.recurrence_exact._plan_v2 import (
    DIRECT_NONE_U32,
    RECURRENCE_EXACT_SECTIONS_ABI,
    RECURRENCE_RUNTIME_LAYOUT_V2_ABI,
    _AmplitudeDestination,
    _Closure,
    _Contribution,
    _Current,
    _ExactFactor,
    _Executor,
    _MomentumForm,
    _MomentumTerm,
    _parse_exact_sections,
    _RecurrenceExactSectionsV1,
    _ResolvedHelicity,
    _ResolvedSourceSelection,
    _RowGroup,
    _Source,
    _SourceDispatchVariant,
    _SourceEmbedding,
    _SourceProjection,
    _SourceStateAssignment,
)
from pyamplicol.runtime.symbolica_exact import _quark_weyl

_ZERO = Decimal(0)
_ONE = Decimal(1)


def _factor(real: int, imaginary: int = 0) -> _ExactFactor:
    return _ExactFactor(real, 1, imaginary, 1)


def test_recurrence_diagnostic_projection_uses_external_source_layout() -> None:
    massive = _SourceTemplate(
        0,
        4,
        1,
        0,
        1,
        "fermion",
        "particle",
        0,
        1,
        1,
        1,
        mass_parameter_name="ufo.MT",
    )
    massless = _SourceTemplate(
        1,
        4,
        1,
        0,
        1,
        "vector",
        "self-conjugate",
        None,
        1,
        1,
        1,
    )

    class Plan:
        def __init__(self) -> None:
            self.sections = SimpleNamespace(
                external_source_count=2,
                sources=(
                    _Source(0, 0, 0, 0, 0, 0, 0),
                    _Source(1, 0, 1, 1, 0, 0, 0),
                ),
                source_dispatch_variants=(),
            )
            self.source_templates = {0: massive, 1: massless}
            # Physics/artifact order is (source slot 1, source slot 0).
            self.external_source_slots = (1, 0)

        @staticmethod
        def resolve_model_parameters(
            _runtime_parameters: object, _precision: int
        ) -> SimpleNamespace:
            return SimpleNamespace(prepared=((Decimal("7"), Decimal(0)),))

    class Native:
        @staticmethod
        def _exact_runtime_state_json() -> str:
            return json.dumps(
                {
                    "model_parameter_values": [7.0],
                    "normalization_factor": 1.0,
                }
            )

    executor = object.__new__(RecurrenceExactExecutor)
    executor._plan = Plan()
    executor._native_runtime = Native()
    executor._physics = {
        "external_particles": [
            {"label": 1, "pdg": 21, "role": "initial"},
            {"label": 2, "pdg": 6, "role": "final"},
        ]
    }
    executor._representative_physics = executor._physics
    executor._permutation = (0, 1)

    projected, metadata = executor._diagnostic_project_onshell(
        (((-5.0, 0.0, 0.0, 5.0), (24.9, 24.0, 0.0, 0.0)),),
        precision=200,
    )

    assert projected[0][0][0] == Decimal("-5.0")
    assert projected[0][1][0] == Decimal("25.0")
    gluon, top = metadata["points"][0]["legs"]
    assert gluon["mass"] == "0"
    assert top["mass"] == "7"
    assert top["mass_parameter"] == "ufo.MT"


def _scalar_union_plan() -> _RecurrenceExactPlan:
    exact_factors = (
        _factor(1),
        _factor(0),
        _factor(2),
        _factor(3),
        _factor(0),
        _factor(1),
        _factor(0),
        _factor(2),
    )
    sources = (
        _Source(0, 0, 0, 0, 0, 0, 0),
        _Source(1, 2, 1, 1, 0, 0, 0),
    )
    variants = (
        _SourceDispatchVariant(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1),
        _SourceDispatchVariant(2, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 2, 1),
    )
    sections = _RecurrenceExactSectionsV1(
        process_id="synthetic_union",
        strategy="all-flow-union",
        semantic_digest="0" * 64,
        runtime_layout_digest="1" * 64,
        current_arena_components=4,
        amplitude_destination_count=2,
        parameter_value_count=0,
        external_source_count=2,
        currents=(),
        sources=sources,
        contributions=(),
        finalizations=(),
        closures=(
            _Closure(0, 2, 0, 1, 0, 0, 4, 2, 0, 0),
            _Closure(0, 2, 0, 1, 1, 0, 6, 2, 0, 0),
        ),
        row_groups=(
            _RowGroup(0, 0, 0, DIRECT_NONE_U32, 0, 2),
            _RowGroup(1, 3, 3, 1, 0, 2),
        ),
        momentum_forms=(_MomentumForm(0, 1), _MomentumForm(1, 1)),
        momentum_terms=(_MomentumTerm(0, 1), _MomentumTerm(1, 1)),
        replay_targets=(),
        source_permutations=(),
        replay_momentum_signs=(),
        replay_helicity_map=(),
        amplitude_destinations=(
            _AmplitudeDestination(0, 0, 10, DIRECT_NONE_U32, 1, 0),
            _AmplitudeDestination(1, 1, 20, DIRECT_NONE_U32, 1, 0),
        ),
        resolved_helicities=(_ResolvedHelicity(0, 0, 0, 0, 2, 2, 2, 0),),
        source_state_assignments=(
            _SourceStateAssignment(0, 0),
            _SourceStateAssignment(1, 0),
        ),
        source_dispatch_variants=variants,
        source_embeddings=(
            _SourceEmbedding(0, DIRECT_NONE_U32, 1),
            _SourceEmbedding(1, 0, 2),
            _SourceEmbedding(0, DIRECT_NONE_U32, 1),
            _SourceEmbedding(1, 0, 3),
        ),
        source_projections=(
            _SourceProjection(0, 1),
            _SourceProjection(0, 1),
        ),
        resolved_source_selections=(
            _ResolvedSourceSelection(0, 0),
            _ResolvedSourceSelection(1, 1),
        ),
        public_helicities=(0, 0),
        exact_factors=exact_factors,
        public_flow_ids=(10, 20),
        executors=(
            _Executor(0, "source", "initialize", (), 1, 1, None, "source"),
            _Executor(
                1,
                "closure",
                "closure-add",
                (2, 2),
                1,
                2,
                None,
                "componentwise-dot",
            ),
        ),
    )
    return _RecurrenceExactPlan(
        sections=sections,
        kernels={},
        executors={row.executor_id: row for row in sections.executors},
        executor_exact_kernel_ids={},
        executor_parent_permutations={0: (0, 1), 1: (0, 1)},
        source_templates={
            0: _SourceTemplate(
                0, 1, 0, 0, 0, "scalar", "self-conjugate", None, 1, 1, 1
            ),
            1: _SourceTemplate(
                1, 1, 0, 0, 0, "scalar", "self-conjugate", None, 1, 1, 1
            ),
        },
        initial_source_slots=frozenset(),
        executor_couplings={},
        prepared_defaults=(),
        parameter_projection=(),
        parameter_derivation=None,
    )


def _point() -> tuple[tuple[Decimal, Decimal, Decimal, Decimal], ...]:
    return (
        (Decimal(10), Decimal(1), Decimal(2), Decimal(3)),
        (Decimal(20), Decimal(4), Decimal(5), Decimal(6)),
    )


def test_union_schedule_returns_independent_flow_amplitudes() -> None:
    plan = _scalar_union_plan()
    amplitudes = _evaluate_union_point(
        plan,
        _point(),
        plan.sections.resolved_helicities[0],
        (),
        50,
    )
    assert amplitudes == ((Decimal(6), _ZERO), (Decimal(12), _ZERO))


def test_union_schedule_executes_certified_reuse_as_exact_scale_copy() -> None:
    plan = _scalar_union_plan()
    sections = plan.sections
    plan.sections = replace(
        sections,
        exact_factors=(*sections.exact_factors, _factor(3, 2)),
        contributions=(
            _Contribution(
                0,
                DIRECT_NONE_U32,
                2,
                DIRECT_NONE_U32,
                2,
                len(sections.exact_factors),
                0,
                3,
            ),
        ),
        row_groups=(
            sections.row_groups[0],
            _RowGroup(1, 1, 1, DIRECT_NONE_U32, 0, 1),
            sections.row_groups[1],
        ),
    )

    amplitudes = _evaluate_union_point(
        plan,
        _point(),
        plan.sections.resolved_helicities[0],
        (),
        50,
    )

    assert amplitudes == (
        (Decimal(12), Decimal(8)),
        (Decimal(24), Decimal(16)),
    )


def test_zero_self_reuse_clears_stale_decimal_destination_before_copy() -> None:
    plan = _scalar_union_plan()
    sections = plan.sections
    zero_factor_id = len(sections.exact_factors)
    plan.sections = replace(
        sections,
        currents=(
            _Current(
                2,
                1,
                0,
                2,
                2,
                0,
                1,
                0,
                0,
                0,
                DIRECT_NONE_U32,
                DIRECT_NONE_U32,
            ),
        ),
        exact_factors=(*sections.exact_factors, _factor(0)),
    )
    arena = [
        (Decimal(1), Decimal(2)),
        (Decimal(3), Decimal(4)),
        (Decimal("NaN"), Decimal("NaN")),
        (Decimal("NaN"), Decimal("NaN")),
    ]
    row = _Contribution(
        2,
        DIRECT_NONE_U32,
        2,
        DIRECT_NONE_U32,
        2,
        zero_factor_id,
        0,
        3,
    )

    _clear_stage(plan, arena, 1)
    _execute_certified_reuse_row(plan, row, arena)

    assert arena[2:] == [(_ZERO, _ZERO), (_ZERO, _ZERO)]


def test_certified_reuse_rejects_noncanonical_encoding() -> None:
    plan = _scalar_union_plan()
    arena = [
        (Decimal(2), Decimal(3)),
        (Decimal(4), Decimal(5)),
        (Decimal(99), Decimal(101)),
        (Decimal(103), Decimal(107)),
    ]

    with pytest.raises(ArtifactError, match="invalid encoding"):
        _execute_certified_reuse_row(
            plan,
            _Contribution(
                0,
                DIRECT_NONE_U32,
                2,
                DIRECT_NONE_U32,
                2,
                3,
                0,
                1,
            ),
            arena,
        )

    assert arena == [
        (Decimal(2), Decimal(3)),
        (Decimal(4), Decimal(5)),
        (Decimal(99), Decimal(101)),
        (Decimal(103), Decimal(107)),
    ]


def test_union_source_zeroes_inactive_weyl_embedding_components() -> None:
    plan = _scalar_union_plan()
    sections = plan.sections
    source = _Source(0, 0, 0, 0, 1, 0, 0)
    variant = _SourceDispatchVariant(
        0,
        0,
        0,
        0,
        0,
        0,
        7,
        7,
        9,
        1,
        0,
        0,
        4,
        2,
    )
    plan.sections = replace(
        sections,
        sources=(source,),
        source_dispatch_variants=(variant,),
        source_embeddings=(
            _SourceEmbedding(0, 0, 0),
            _SourceEmbedding(1, DIRECT_NONE_U32, 1),
            _SourceEmbedding(2, 1, 0),
            _SourceEmbedding(3, DIRECT_NONE_U32, 1),
        ),
        source_projections=(
            _SourceProjection(0, 0),
            _SourceProjection(1, 2),
        ),
    )
    plan.source_templates = {
        7: _SourceTemplate(
            7,
            2,
            1,
            1,
            1,
            "fermion",
            "particle",
            None,
            1,
            1,
            1,
        )
    }
    momentum = (Decimal(10), Decimal(1), Decimal(2), Decimal(9))
    arena = [(_ONE * 99, _ONE * 99) for _ in range(4)]
    _execute_union_source(plan, source, variant, (momentum,), (), arena)
    expected = _quark_weyl(momentum, 1, 1)
    assert arena == [expected[0], (_ZERO, _ZERO), expected[1], (_ZERO, _ZERO)]


def test_exact_kernel_inputs_use_executor_binding_coupling() -> None:
    plan = _scalar_union_plan()
    plan.executor_couplings = {7: (Decimal("1.25"), Decimal("-0.5"))}
    plan.executor_parent_permutations = {**plan.executor_parent_permutations, 7: (0, 1)}
    executor = _Executor(
        7,
        "contribution",
        "add",
        (1, 1),
        1,
        2,
        3,
        None,
    )
    row = _Contribution(0, 0, 0, 0, 0, 0, 0, 0)

    inputs = _kernel_inputs(
        plan,
        (
            {"component": 0, "role": "coupling-real"},
            {"component": 1, "role": "coupling-imag"},
        ),
        executor,
        row,
        (),
        ((Decimal(99), Decimal(101)),),
        (),
    )

    assert inputs == (
        (Decimal("1.25"), _ZERO),
        (Decimal("-0.5"), _ZERO),
    )


@pytest.mark.parametrize(
    ("parent_permutation", "expected"),
    (
        (
            (0, 1),
            (
                (Decimal(10), _ZERO),
                (Decimal(20), _ZERO),
                (Decimal(100), _ZERO),
                (Decimal(200), _ZERO),
            ),
        ),
        (
            (1, 0),
            (
                (Decimal(20), _ZERO),
                (Decimal(10), _ZERO),
                (Decimal(200), _ZERO),
                (Decimal(100), _ZERO),
            ),
        ),
    ),
)
def test_exact_contribution_inputs_apply_authenticated_parent_permutation_once(
    parent_permutation: tuple[int, int],
    expected: tuple[tuple[Decimal, Decimal], ...],
) -> None:
    plan = _scalar_union_plan()
    plan.executor_parent_permutations = {
        **plan.executor_parent_permutations,
        7: parent_permutation,
    }
    executor = _Executor(
        7,
        "contribution",
        "add",
        (1, 1),
        1,
        2,
        3,
        None,
    )
    row = _Contribution(0, 1, 0, 1, 0, 0, 0, 0)

    inputs = _kernel_inputs(
        plan,
        (
            {"component": 0, "role": "left-current"},
            {"component": 0, "role": "right-current"},
            {"component": 0, "role": "left-momentum"},
            {"component": 0, "role": "right-momentum"},
        ),
        executor,
        row,
        (
            (Decimal(100), _ZERO, _ZERO, _ZERO),
            (Decimal(200), _ZERO, _ZERO, _ZERO),
        ),
        (),
        ((Decimal(10), _ZERO), (Decimal(20), _ZERO)),
    )

    assert inputs == expected


def test_exact_contribution_inputs_reject_unauthenticated_parent_permutation() -> None:
    plan = _scalar_union_plan()
    plan.executor_parent_permutations = {
        **plan.executor_parent_permutations,
        7: (0, 0),
    }
    executor = _Executor(
        7,
        "contribution",
        "add",
        (1, 1),
        1,
        2,
        3,
        None,
    )
    row = _Contribution(0, 1, 0, 1, 0, 0, 0, 0)

    with pytest.raises(ArtifactError, match="authenticated parent permutation"):
        _kernel_inputs(
            plan,
            ({"component": 0, "role": "left-current"},),
            executor,
            row,
            (),
            (),
            ((Decimal(10), _ZERO), (Decimal(20), _ZERO)),
        )


def test_exact_factor_cache_is_schedule_local_and_context_exact() -> None:
    plan = _scalar_union_plan()
    plan.sections = replace(
        plan.sections,
        exact_factors=(
            *plan.sections.exact_factors,
            _ExactFactor(1, 3, -2, 7),
        ),
    )
    factor_id = len(plan.sections.exact_factors) - 1
    low_precision_cache = [None] * len(plan.sections.exact_factors)
    with localcontext() as context:
        context.prec = 9
        first = _resolved_factor(
            plan,
            factor_id,
            cache=low_precision_cache,
        )
        second = _resolved_factor(
            plan,
            factor_id,
            cache=low_precision_cache,
        )
    high_precision_cache = [None] * len(plan.sections.exact_factors)
    with localcontext() as context:
        context.prec = 40
        high_precision = _resolved_factor(
            plan,
            factor_id,
            cache=high_precision_cache,
        )

    assert first is second
    assert first != high_precision
    assert first == (Decimal("0.333333333"), Decimal("-0.285714286"))


def test_trusted_executor_recipe_matches_validated_kernel_inputs() -> None:
    plan = _scalar_union_plan()
    plan.executor_couplings = {7: (Decimal("1.25"), Decimal("-0.5"))}
    plan.executor_parent_permutations = {
        **plan.executor_parent_permutations,
        7: (1, 0),
    }
    executor = _Executor(
        7,
        "contribution",
        "add",
        (1, 1),
        1,
        2,
        3,
        None,
    )
    contracts = (
        {"component": 0, "role": "left-current"},
        {"component": 0, "role": "right-current"},
        {"component": 0, "role": "left-momentum"},
        {"component": 0, "role": "right-momentum"},
        {"component": 0, "role": "coupling-real"},
        {"component": 0, "role": "coupling-imag"},
        {
            "component": 0,
            "role": "model-parameter",
            "model_parameter_index": 0,
        },
    )
    kernel = SimpleNamespace(record=SimpleNamespace(input_contracts=contracts))
    plan.kernels = {3: kernel}
    row = _Contribution(0, 1, 0, 1, 0, 0, 0, 0)
    momenta = (
        (Decimal(100), _ZERO, _ZERO, _ZERO),
        (Decimal(200), _ZERO, _ZERO, _ZERO),
    )
    parameters = ((Decimal(300), Decimal(400)),)
    arena = ((Decimal(10), _ZERO), (Decimal(20), _ZERO))

    recipe = _trusted_executor_recipe(plan, executor)
    inputs = _trusted_kernel_inputs(
        plan,
        recipe.inputs,
        executor,
        row,
        momenta,
        parameters,
        arena,
    )

    assert _trusted_executor_recipe(plan, executor) is recipe
    assert inputs == _kernel_inputs(
        plan,
        contracts,
        executor,
        row,
        momenta,
        parameters,
        arena,
    )


def test_union_exact_resolved_values_sum_incoherently_by_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _scalar_union_plan()
    executor = object.__new__(RecurrenceExactExecutor)
    executor._plan = plan
    executor._physics = {
        "extensions": {"normalization": {}},
        "helicities": [
            {
                "id": "h:0,0",
                "values": [0, 0],
                "computed": True,
                "structural_zero": False,
                "coefficient": 1,
            }
        ],
        "color_components": [
            {"id": "flow:a", "coefficient": 1},
            {"id": "flow:b", "coefficient": 1},
        ],
        "color_accuracy": "lc",
    }
    executor._permutation = (0, 1)
    executor._native_runtime = object()
    (
        executor._helicity_representative,
        executor._helicity_orbit_members,
    ) = executor._helicity_reduction_indices()
    executor._replay_by_color = ()
    executor._destination_helicities = ()
    executor._union_destination_by_color = executor._union_destinations_by_color()
    executor._union_helicity_by_physics = executor._union_helicities_by_physics()
    monkeypatch.setattr(executor_module, "_prepare_points", lambda *_: (_point(),))
    monkeypatch.setattr(
        executor_module,
        "_runtime_state",
        lambda _: {"model_parameter_values": [], "normalization_factor": 1},
    )

    resolved = executor.evaluate_resolved(
        (((10, 1, 2, 3), (20, 4, 5, 6)),),
        helicities=None,
        color_flows=None,
        precision=40,
    )
    assert resolved.values == (((Decimal(36), Decimal(144)),),)
    assert resolved.total() == (Decimal(180),)
    assert resolved.total() != (Decimal((6 + 12) ** 2),)

    selected = executor.evaluate_resolved(
        (((10, 1, 2, 3), (20, 4, 5, 6)),),
        helicities=("h:0,0",),
        color_flows=("flow:b",),
        precision=40,
    )
    assert selected.values == (((Decimal(144),),),)
    with pytest.raises(EvaluationError, match="unknown resolved helicity ID"):
        executor.evaluate_resolved(
            (((10, 1, 2, 3), (20, 4, 5, 6)),),
            helicities=("missing",),
            color_flows=None,
            precision=40,
        )
    with pytest.raises(EvaluationError, match="unknown resolved color component ID"):
        executor.evaluate_resolved(
            (((10, 1, 2, 3), (20, 4, 5, 6)),),
            helicities=None,
            color_flows=("missing",),
            precision=40,
        )


def test_union_native_section_adapter_parses_dispatch_tables() -> None:
    sections = _scalar_union_plan().sections
    raw = {
        "abi": RECURRENCE_EXACT_SECTIONS_ABI,
        "runtime_layout_abi": RECURRENCE_RUNTIME_LAYOUT_V2_ABI,
        "process_id": sections.process_id,
        "strategy": sections.strategy,
        "semantic_digest": sections.semantic_digest,
        "runtime_layout_digest": sections.runtime_layout_digest,
        "counts": (
            sections.current_arena_components,
            sections.amplitude_destination_count,
            sections.parameter_value_count,
            sections.external_source_count,
        ),
        "currents": [astuple(row) for row in sections.currents],
        "sources": [astuple(row) for row in sections.sources],
        "contributions": [astuple(row) for row in sections.contributions],
        "finalizations": [astuple(row) for row in sections.finalizations],
        "closures": [astuple(row) for row in sections.closures],
        "row_groups": [astuple(row) for row in sections.row_groups],
        "momentum_forms": [astuple(row) for row in sections.momentum_forms],
        "momentum_terms": [astuple(row) for row in sections.momentum_terms],
        "replay_targets": [],
        "source_permutations": [],
        "replay_momentum_signs": [],
        "replay_helicity_map": [],
        "amplitude_destinations": [
            astuple(row) for row in sections.amplitude_destinations
        ],
        "resolved_helicities": [astuple(row) for row in sections.resolved_helicities],
        "source_state_assignments": [
            astuple(row) for row in sections.source_state_assignments
        ],
        "source_dispatch_variants": [
            astuple(row) for row in sections.source_dispatch_variants
        ],
        "source_embeddings": [astuple(row) for row in sections.source_embeddings],
        "source_projections": [astuple(row) for row in sections.source_projections],
        "resolved_source_selections": [
            astuple(row) for row in sections.resolved_source_selections
        ],
        "public_helicities": list(sections.public_helicities),
        "exact_factors": [
            tuple(str(value) for value in astuple(row))
            for row in sections.exact_factors
        ],
        "public_flow_ids": list(sections.public_flow_ids),
        "executors": [
            (
                row.executor_id,
                row.role,
                row.destination_operation,
                row.parent_component_counts,
                row.destination_component_count,
                row.momentum_operand_count,
                row.prepared_kernel_id,
                row.runtime_template,
            )
            for row in sections.executors
        ],
    }
    parsed = _parse_exact_sections(raw, sections.process_id)
    assert parsed.strategy == "all-flow-union"
    assert parsed.source_dispatch_variants == sections.source_dispatch_variants
    assert parsed.resolved_source_selections == sections.resolved_source_selections

    raw["public_flow_ids"] = [10, 10, 20]
    parsed = _parse_exact_sections(raw, sections.process_id)
    assert parsed.public_flow_ids == (10, 10, 20)

    raw["public_flow_ids"] = [10, 10]
    with pytest.raises(ArtifactError, match="not completely covered"):
        _parse_exact_sections(raw, sections.process_id)

    raw["public_flow_ids"] = [10, 20]
    raw["resolved_source_selections"] = [(0, 0)]
    with pytest.raises(ArtifactError, match="does not cover every external source"):
        _parse_exact_sections(raw, sections.process_id)

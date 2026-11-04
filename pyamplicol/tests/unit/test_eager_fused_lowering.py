# SPDX-License-Identifier: 0BSD
"""Regression contract for single-pass eager runtime lowering."""

from __future__ import annotations

import subprocess
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

import pytest

from pyamplicol.generation import eager_lowering
from pyamplicol.generation.contracts import RuntimeExpressionSchema
from pyamplicol.generation.dag_compiler import compile_generic_dag
from pyamplicol.generation.dag_types import GenericDAG
from pyamplicol.generation.eager_lowering import (
    EagerExecutionTables,
    MappingEagerKernelResolver,
    lower_eager_execution_tables,
)
from pyamplicol.generation.eager_tables import MISSING_U32
from pyamplicol.generation.runtime_schema import build_runtime_schema
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.base import Model
from pyamplicol.models.builtin.process_ir import build_process_ir

_FUSED_ENTRY_POINT = "lower_fused_eager_execution"


@dataclass(frozen=True, slots=True)
class _LoweringCase:
    name: str
    process_id: str
    dag: GenericDAG
    model: Model
    resolver: MappingEagerKernelResolver
    reference_schema: Mapping[str, object]
    reference_tables: EagerExecutionTables


def _build_case(
    process: str,
    *,
    process_id: str,
    prepared_closure: bool,
    color_accuracy: str = "lc",
) -> _LoweringCase:
    model = BuiltinSMModel()
    dag = compile_generic_dag(
        build_process_ir(process, color_accuracy=color_accuracy),
        model=model,
    )

    # The current built-in compiler closes amplitudes directly.  Replace one
    # root with a prepared closure to exercise both closure lanes without
    # requiring an external UFO model or any backend compilation.
    closure_kernels: dict[tuple[str, int | None], int] = {}
    if prepared_closure:
        first = replace(
            dag.amplitude_roots[0],
            kind="prepared-closure",
            vertex_kind=3,
            vertex_particles=(21, 21, 21),
            coupling=(1.0, 0.0),
        )
        dag = replace(dag, amplitude_roots=(first, *dag.amplitude_roots[1:]))
        closure_kernels[("prepared-closure", 3)] = 9_000

    schema = build_runtime_schema(dag, model, process_id=process_id)
    propagated = sorted(
        {
            (int(slot["particle_id"]), int(slot["chirality"]))
            for slot in schema["value_storage"]["value_slots"]
            if slot["variant"] == "propagated"
        }
    )
    kernel_id_by_evaluation_class: dict[str, int] = {}
    vertex_kernels: dict[int, int] = {}
    for kind in sorted(dag.required_vertex_kinds):
        class_id = model.vertex_evaluation_equivalence(kind).class_id
        vertex_kernels[kind] = kernel_id_by_evaluation_class.setdefault(
            class_id,
            100 + len(kernel_id_by_evaluation_class),
        )
    resolver = MappingEagerKernelResolver(
        vertex_kernels=vertex_kernels,
        propagator_kernels={key: 1_000 + index for index, key in enumerate(propagated)},
        closure_kernels=closure_kernels,
    )
    tables = lower_eager_execution_tables(dag, model, schema, resolver)
    return _LoweringCase(
        name=process_id,
        process_id=process_id,
        dag=dag,
        model=model,
        resolver=resolver,
        reference_schema=schema,
        reference_tables=tables,
    )


@pytest.fixture(scope="module")
def lowering_cases() -> tuple[_LoweringCase, ...]:
    return (
        _build_case(
            "g g > g g",
            process_id="synthetic_shared_mixed",
            prepared_closure=True,
        ),
        _build_case(
            "d d~ > z",
            process_id="synthetic_direct_no_propagator",
            prepared_closure=False,
        ),
    )


@pytest.fixture(scope="module")
def three_line_nlc_case() -> _LoweringCase:
    return _build_case(
        "d d~ > u u~ s s~ g",
        process_id="three_line_nlc_fused_contract",
        prepared_closure=False,
        color_accuracy="nlc",
    )


def _fused_entry_point() -> Callable[..., object] | None:
    candidate = getattr(eager_lowering, _FUSED_ENTRY_POINT, None)
    return candidate if callable(candidate) else None


@pytest.fixture
def fused_lowerer() -> Callable[..., object]:
    candidate = _fused_entry_point()
    if candidate is None:
        pytest.skip(
            f"{_FUSED_ENTRY_POINT} is not implemented; the dedicated API "
            "contract test reports this missing feature"
        )
    return candidate


def _schema_mapping(value: object) -> dict[str, object]:
    if isinstance(value, RuntimeExpressionSchema):
        return value.to_mapping()
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    raise AssertionError(
        "fused eager lowering must return a runtime schema mapping or "
        "RuntimeExpressionSchema"
    )


def _unpack_fused_result(
    result: object,
) -> tuple[dict[str, object], EagerExecutionTables]:
    if isinstance(result, tuple) and len(result) == 2:
        schema, tables = result
    else:
        schema = getattr(result, "runtime_schema", None)
        tables = getattr(result, "eager_tables", None)
    if not isinstance(tables, EagerExecutionTables):
        raise AssertionError(
            "fused eager lowering must return EagerExecutionTables as the "
            "second tuple member or eager_tables attribute"
        )
    return _schema_mapping(schema), tables


def _run_fused(
    lowerer: Callable[..., object],
    case: _LoweringCase,
    *,
    resolver: MappingEagerKernelResolver | None = None,
) -> tuple[dict[str, object], EagerExecutionTables]:
    return _unpack_fused_result(
        lowerer(
            dag=case.dag,
            model=case.model,
            resolver=resolver or case.resolver,
            process_id=case.process_id,
        )
    )


def _canonical_schema(schema: Mapping[str, object]) -> str:
    return RuntimeExpressionSchema.from_mapping(schema).canonical_json


def _forbidden_backend_call(*_args: object, **_kwargs: object) -> object:
    raise AssertionError("fused eager lowering attempted backend compilation")


def test_fused_eager_lowering_entry_point_exists() -> None:
    assert _fused_entry_point() is not None, (
        "implement pyamplicol.generation.eager_lowering."
        f"{_FUSED_ENTRY_POINT}(dag=..., model=..., resolver=..., "
        "process_id=...) returning (runtime_schema, eager_tables) or an "
        "object exposing runtime_schema/eager_tables"
    )


def test_reference_fixtures_cover_the_fused_lowering_contract(
    lowering_cases: tuple[_LoweringCase, ...],
) -> None:
    mixed, direct = lowering_cases
    mixed_tables = mixed.reference_tables
    finalizations = tuple(
        row for stage in mixed_tables.stages for row in stage.finalizations
    )
    input_fanout = Counter(
        current_id
        for interaction in mixed.dag.interactions
        for current_id in (interaction.left_id, interaction.right_id)
    )

    assert any(count > 1 for count in input_fanout.values())
    assert any(
        invocation.attachment_count > 1
        for stage in mixed_tables.stages
        for invocation in stage.invocations
    )
    assert any(row.applies_kernel for row in finalizations)
    assert any(not row.applies_kernel for row in finalizations)
    assert any(row.kernel_id != MISSING_U32 for row in mixed_tables.closures)
    assert any(row.kernel_id == MISSING_U32 for row in mixed_tables.closures)
    assert mixed_tables.selector_closures is not None
    assert len(mixed_tables.selector_closures.domains) > 1

    direct_finalizations = tuple(
        row for stage in direct.reference_tables.stages for row in stage.finalizations
    )
    assert direct_finalizations
    assert all(not row.applies_kernel for row in direct_finalizations)
    assert all(row.kernel_id == MISSING_U32 for row in direct.reference_tables.closures)


@pytest.mark.parametrize(
    "case_index",
    (0, 1),
    ids=("shared-mixed", "direct-no-propagator"),
)
def test_fused_lowering_matches_reference_schema_and_tables(
    case_index: int,
    lowering_cases: tuple[_LoweringCase, ...],
    fused_lowerer: Callable[..., object],
) -> None:
    case = lowering_cases[case_index]
    schema, tables = _run_fused(fused_lowerer, case)

    assert _canonical_schema(schema) == _canonical_schema(case.reference_schema)
    assert tables == case.reference_tables
    assert tables.to_metadata() == case.reference_tables.to_metadata()
    assert tables.binary_payloads() == case.reference_tables.binary_payloads()


def test_fused_lowering_is_deterministic_across_resolver_mapping_order(
    lowering_cases: tuple[_LoweringCase, ...],
    fused_lowerer: Callable[..., object],
) -> None:
    case = lowering_cases[0]
    reverse_resolver = MappingEagerKernelResolver(
        vertex_kernels=dict(reversed(tuple(case.resolver.vertex_kernels.items()))),
        propagator_kernels=dict(
            reversed(tuple(case.resolver.propagator_kernels.items()))
        ),
        closure_kernels=dict(reversed(tuple(case.resolver.closure_kernels.items()))),
    )

    first_schema, first_tables = _run_fused(fused_lowerer, case)
    second_schema, second_tables = _run_fused(
        fused_lowerer,
        case,
        resolver=reverse_resolver,
    )

    assert _canonical_schema(first_schema) == _canonical_schema(second_schema)
    assert first_tables.to_metadata() == second_tables.to_metadata()
    assert first_tables.binary_payloads() == second_tables.binary_payloads()


def test_three_line_nlc_fused_lowering_is_byte_identical_and_deterministic(
    three_line_nlc_case: _LoweringCase,
    fused_lowerer: Callable[..., object],
) -> None:
    first_schema, first_tables = _run_fused(
        fused_lowerer,
        three_line_nlc_case,
    )
    second_schema, second_tables = _run_fused(
        fused_lowerer,
        three_line_nlc_case,
    )

    reference_schema_bytes = _canonical_schema(
        three_line_nlc_case.reference_schema
    ).encode("utf-8")
    assert _canonical_schema(first_schema).encode("utf-8") == reference_schema_bytes
    assert _canonical_schema(second_schema).encode("utf-8") == reference_schema_bytes
    assert first_tables == three_line_nlc_case.reference_tables
    assert second_tables == three_line_nlc_case.reference_tables
    assert first_tables.to_metadata() == (
        three_line_nlc_case.reference_tables.to_metadata()
    )
    assert second_tables.to_metadata() == first_tables.to_metadata()
    assert first_tables.binary_payloads() == (
        three_line_nlc_case.reference_tables.binary_payloads()
    )
    assert second_tables.binary_payloads() == first_tables.binary_payloads()


def test_fused_lowering_never_constructs_or_compiles_backend_evaluators(
    lowering_cases: tuple[_LoweringCase, ...],
    fused_lowerer: Callable[..., object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyamplicol.evaluators import symbolica_compile
    from pyamplicol.generation import stage_artifacts

    monkeypatch.setattr(
        symbolica_compile,
        "_compile_symbolica_outputs",
        _forbidden_backend_call,
    )
    monkeypatch.setattr(
        stage_artifacts,
        "_compile_stage_evaluator_artifact",
        _forbidden_backend_call,
    )
    monkeypatch.setattr(
        stage_artifacts,
        "build_and_write_generic_stage_evaluator_artifacts",
        _forbidden_backend_call,
    )
    monkeypatch.setattr(subprocess, "run", _forbidden_backend_call)
    monkeypatch.setattr(subprocess, "Popen", _forbidden_backend_call)

    # Patch every current/future Expression.evaluator* spelling exposed by the
    # installed Symbolica API.  Importing is license-free and does not build an
    # evaluator; absent Symbolica is tolerated because the low-level lowering
    # contract itself must remain dependency-light.
    try:
        from symbolica import Expression
    except ImportError:  # pragma: no cover - minimal source-only test env
        Expression = None
    if Expression is not None:
        for name in dir(Expression):
            if name == "evaluator" or name.startswith("evaluator_"):
                monkeypatch.setattr(Expression, name, _forbidden_backend_call)

    for case in lowering_cases:
        schema, tables = _run_fused(fused_lowerer, case)
        assert _canonical_schema(schema) == _canonical_schema(case.reference_schema)
        assert tables.binary_payloads() == case.reference_tables.binary_payloads()

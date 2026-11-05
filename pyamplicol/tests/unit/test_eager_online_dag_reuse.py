# SPDX-License-Identifier: 0BSD
"""Parity contract for eager-only online recursive-current reuse."""

from __future__ import annotations

import inspect
from collections.abc import Callable

import pytest

from pyamplicol.color import build_color_plan
from pyamplicol.config import EvaluatorConfig, RunConfig
from pyamplicol.generation import service as service_module
from pyamplicol.generation.contracts import runtime_coupling_parameter_names
from pyamplicol.generation.dag_algorithms import (
    prune_dag_to_amplitude_roots,
    prune_global_helicity_flip_equivalent_roots,
)
from pyamplicol.generation.dag_compiler import compile_generic_dag
from pyamplicol.generation.dag_types import GenericDAG, InteractionNode
from pyamplicol.generation.runtime_schema import build_runtime_schema
from pyamplicol.generation.service import GenerationBackend
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.base import Model
from pyamplicol.models.builtin.process_ir import build_process_ir
from pyamplicol.processes.ir import CanonicalProcessIR

_ONLINE_REUSE_PARAMETER = "online_evaluation_reuse"
_CASES = (
    ("d d~ > z g g g", "lc"),
    ("d d~ > z g g g", "nlc"),
    ("d d~ > z g g g", "full"),
    ("d d~ > u u~ s s~ g", "lc"),
    ("d d~ > u u~ s s~ g", "nlc"),
    ("d d~ > u u~ s s~ g", "full"),
)


class _RuntimeCouplingBuiltin(BuiltinSMModel):
    def runtime_parameter_names_for_vertex(self, kind: int) -> tuple[str, str]:
        return (
            f"runtime.vertex.{kind}.component_0",
            f"runtime.vertex.{kind}.component_1",
        )


def _online_reuse_parameter() -> inspect.Parameter | None:
    return inspect.signature(compile_generic_dag).parameters.get(
        _ONLINE_REUSE_PARAMETER
    )


@pytest.fixture
def online_compiler() -> Callable[..., GenericDAG]:
    if _online_reuse_parameter() is None:
        pytest.skip(
            "online-reuse parity depends on the missing "
            "compile_generic_dag(..., online_evaluation_reuse=...) API"
        )
    return compile_generic_dag


def _interaction_reuse_contract(
    interactions: tuple[InteractionNode, ...],
) -> tuple[tuple[int, int | None, tuple[float, float]], ...]:
    return tuple(
        (
            interaction.id,
            interaction.evaluation_group_id,
            interaction.evaluation_factor,
        )
        for interaction in interactions
    )


def _assert_exact_dag_parity(reference: GenericDAG, actual: GenericDAG) -> None:
    # Dataclass equality covers every GenericDAG field, including any fields added
    # after this contract.  The focused assertions below identify ordering/reuse
    # regressions without reducing the strength of that complete comparison.
    assert actual == reference
    assert actual.process == reference.process
    assert actual.color_plan == reference.color_plan
    assert actual.currents == reference.currents
    assert actual.sources == reference.sources
    assert actual.interactions == reference.interactions
    assert _interaction_reuse_contract(actual.interactions) == (
        _interaction_reuse_contract(reference.interactions)
    )
    assert actual.amplitude_roots == reference.amplitude_roots
    assert (
        actual.truncated,
        actual.helicity_coverage,
        actual.color_coverage,
        actual.selected_source_helicities,
    ) == (
        reference.truncated,
        reference.helicity_coverage,
        reference.color_coverage,
        reference.selected_source_helicities,
    )


def test_compile_generic_dag_declares_online_reuse_as_opt_in() -> None:
    parameter = _online_reuse_parameter()

    assert parameter is not None, (
        "eager online recursive-current reuse requires "
        "compile_generic_dag(..., online_evaluation_reuse=False)"
    )
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is False


@pytest.mark.parametrize(("process", "color_accuracy"), _CASES)
def test_online_reuse_exactly_matches_current_two_pass_dag(
    online_compiler: Callable[..., GenericDAG],
    process: str,
    color_accuracy: str,
) -> None:
    model = BuiltinSMModel()
    process_ir = build_process_ir(process, color_accuracy=color_accuracy)
    reference = online_compiler(process_ir, model=model)
    online_first = online_compiler(
        process_ir,
        model=model,
        online_evaluation_reuse=True,
    )
    online_second = online_compiler(
        process_ir,
        model=model,
        online_evaluation_reuse=True,
    )

    _assert_exact_dag_parity(reference, online_first)
    _assert_exact_dag_parity(reference, online_second)
    _assert_exact_dag_parity(online_first, online_second)


def test_default_compiler_path_equals_explicit_online_reuse_opt_out(
    online_compiler: Callable[..., GenericDAG],
) -> None:
    model = BuiltinSMModel()
    process_ir = build_process_ir("d d~ > z g g g")

    implicit_default = online_compiler(process_ir, model=model)
    explicit_compiled = online_compiler(
        process_ir,
        model=model,
        online_evaluation_reuse=False,
    )

    _assert_exact_dag_parity(implicit_default, explicit_compiled)


def test_prebuilt_color_plan_preserves_exact_online_dag() -> None:
    model = BuiltinSMModel()
    process_ir = build_process_ir(
        "d d~ > u u~ s s~ g",
        color_accuracy="nlc",
    )
    color_plan = build_color_plan(process_ir, color_accuracy="nlc")

    rebuilt = compile_generic_dag(
        process_ir,
        model=model,
        online_evaluation_reuse=True,
    )
    prebuilt = compile_generic_dag(
        process_ir,
        model=model,
        color_plan=color_plan,
        online_evaluation_reuse=True,
    )

    _assert_exact_dag_parity(rebuilt, prebuilt)


def test_evaluation_groups_preserve_mutable_coupling_provenance(
    online_compiler: Callable[..., GenericDAG],
) -> None:
    model = _RuntimeCouplingBuiltin()
    process_ir = build_process_ir("d d~ > z g g g", color_accuracy="nlc")
    dag = online_compiler(
        process_ir,
        model=model,
        online_evaluation_reuse=True,
    )
    provenance_by_group: dict[int, set[tuple[str | None, ...]]] = {}

    for interaction in dag.interactions:
        assert interaction.evaluation_group_id is not None
        provenance = tuple(
            runtime_coupling_parameter_names(
                interaction.vertex_kind,
                interaction.vertex_particles,
                interaction.coupling,
                model=model,
            )
        )
        provenance_by_group.setdefault(
            interaction.evaluation_group_id,
            set(),
        ).add(provenance)

    assert provenance_by_group
    assert all(len(provenance) == 1 for provenance in provenance_by_group.values())


def test_builtin_runtime_schema_exposes_only_runtime_dependent_couplings() -> None:
    model = BuiltinSMModel()
    dag = compile_generic_dag(
        build_process_ir("d d~ > z g g g", color_accuracy="nlc"),
        model=model,
        online_evaluation_reuse=True,
    )
    schema = build_runtime_schema(dag, model, process_id="ddbar-z3g")
    coupling_names = {
        str(record["name"])
        for record in schema["model_parameters"]
        if record["kind"] == "coupling_component"
    }

    assert coupling_names == {
        "coupling.10.1_23_1.component_0",
        "coupling.10.1_23_1.component_1",
    }


def test_eager_qcd_parity_preselection_matches_post_dag_physics() -> None:
    model = BuiltinSMModel()
    process_ir = build_process_ir(
        "d d~ > u u~ s s~ g",
        color_accuracy="nlc",
    )
    limits = {"QCD": 5, "QED": 0}
    complete = compile_generic_dag(
        process_ir,
        model=model,
        max_coupling_orders=limits,
    )
    post_pruned = prune_global_helicity_flip_equivalent_roots(complete, model)
    eager_preselected = compile_generic_dag(
        process_ir,
        model=model,
        max_coupling_orders=limits,
        online_evaluation_reuse=True,
    )
    eager_pruned = prune_dag_to_amplitude_roots(eager_preselected)

    assert len(eager_preselected.amplitude_roots) * 2 == len(complete.amplitude_roots)
    assert all(
        root.helicity_weight == 2.0 for root in eager_preselected.amplitude_roots
    )
    assert (
        len(eager_pruned.currents),
        len(eager_pruned.interactions),
        len(eager_pruned.amplitude_roots),
    ) == (
        len(post_pruned.currents),
        len(post_pruned.interactions),
        len(post_pruned.amplitude_roots),
    )

    reference_schema = build_runtime_schema(
        post_pruned,
        model,
        process_id="threeq-reference",
    )
    eager_schema = build_runtime_schema(
        eager_pruned,
        model,
        process_id="threeq-reference",
    )
    assert eager_schema["physics"] == reference_schema["physics"]


@pytest.mark.parametrize(
    ("execution_mode", "expected_online_reuse"),
    (("compiled", False), ("eager", True)),
)
def test_generation_service_enables_online_reuse_only_for_eager_mode(
    online_compiler: Callable[..., GenericDAG],
    monkeypatch: pytest.MonkeyPatch,
    execution_mode: str,
    expected_online_reuse: bool,
) -> None:
    model = BuiltinSMModel()
    process_ir = build_process_ir("d d~ > z g g g")
    reference = online_compiler(process_ir, model=model)
    observed: list[bool] = []

    def capture_compile(
        process: CanonicalProcessIR,
        *,
        model: Model,
        **options: object,
    ) -> GenericDAG:
        assert process is process_ir
        observed.append(bool(options.get(_ONLINE_REUSE_PARAMETER, False)))
        return reference

    monkeypatch.setattr(service_module, "compile_generic_dag", capture_compile)
    backend = GenerationBackend(
        RunConfig(
            action="generate",
            evaluator=EvaluatorConfig(execution_mode=execution_mode),
        ),
        None,
    )

    compiled, _coverage = backend._compile_concrete_process(process_ir, model)

    assert compiled.currents == reference.currents
    assert compiled.interactions == reference.interactions
    assert compiled.amplitude_roots == reference.amplitude_roots
    assert compiled.lc_topology_replay is not None
    assert observed == [expected_online_reuse]

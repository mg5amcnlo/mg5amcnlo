# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import pytest

import pyamplicol.generation.service as service_module
from pyamplicol.api import Generator, ProcessAlias, ProcessRequest, ProcessSet
from pyamplicol.api.errors import GenerationError
from pyamplicol.color.plan import build_color_plan
from pyamplicol.config import EvaluatorConfig, ProcessConfig, RunConfig
from pyamplicol.generation.dag_algorithms import (
    _infer_minimal_coupling_order_limits_from_color_plan,
    infer_minimal_coupling_order_limits,
)
from pyamplicol.generation.dag_types import ColorState, CurrentIndex
from pyamplicol.generation.progress import PhaseHandle
from pyamplicol.generation.runtime_schema import build_runtime_expression_schema
from pyamplicol.generation.service import GenerationBackend
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.builtin.process_ir import build_process_ir


def _index(*, chirality: int = 1, ordered: tuple[int, ...] = (1, 2)) -> CurrentIndex:
    return CurrentIndex(
        particle_id=1,
        external_mask=3,
        external_labels=(1, 2),
        ordered_external_labels=ordered,
        helicity_ancestry=3,
        chirality=chirality,
        spin_state=chirality,
        flavour_flow=(1,),
        quantum_number_flow=(("electric_charge", "-1/3"),),
        color_state=ColorState(
            accuracy="lc",
            sector_id=0,
            line_groups=(0,),
            basis_key=(1, 2),
        ),
        momentum_mask=3,
        coupling_orders=(("qed", 1),),
    )


def test_generation_current_identity_keeps_every_physics_field() -> None:
    reference = _index()
    assert reference == _index()
    assert reference != _index(chirality=-1)
    assert reference != _index(ordered=(2, 1))
    payload = reference.to_json_dict()
    assert payload["ordered_external_labels"] == [1, 2]
    assert payload["coupling_orders"] == [["QED", 1]]
    assert payload["color_state"]["basis_key"] == [1, 2]


def test_generation_plan_defers_dag_compilation() -> None:
    plan = Generator(
        RunConfig(
            action="generate",
            evaluator=EvaluatorConfig(execution_mode="compiled"),
        )
    ).plan("d d~ > z")
    process = plan.estimated_coverage["processes"][0]

    assert process["key"] == "d_dbar_to_z"
    assert process["dag_compilation_deferred"] is True
    assert "source_count" not in process
    assert plan.concrete_processes[0].expression == "d d~ > z"


def test_only_expansion_derived_unsupported_tree_channel_is_prunable() -> None:
    backend = GenerationBackend(None, None)
    model = BuiltinSMModel()
    process_ir = build_process_ir("g g > z g g")
    expanded = service_module._ExpandedProcess(
        request=ProcessRequest.parse(
            process_ir.process,
            name="pp_zjj_19",
        ),
        process_ir=process_ir,
        source_expansion_size=19,
    )

    outcome = backend._compile_for_generation(
        expanded,
        model,
        PhaseHandle("multiparticle-dag", None, 1),
    )

    assert isinstance(outcome, service_module._UnsupportedProcess)
    assert outcome.expanded == expanded
    assert "no model-supported tree-level amplitudes" in outcome.reason

    explicit = service_module._ExpandedProcess(
        request=ProcessRequest.parse(process_ir.process, name="explicit"),
        process_ir=process_ir,
    )
    with pytest.raises(
        GenerationError,
        match="no model-supported tree-level amplitudes",
    ):
        backend._compile_for_generation(
            explicit,
            model,
            PhaseHandle("explicit-dag", None, 1),
        )


def test_unsupported_expansion_cannot_silently_remove_a_whole_request() -> None:
    backend = GenerationBackend(None, None)
    model = BuiltinSMModel()
    supported_source = ProcessRequest.parse("d d~ > z", name="supported")
    supported_ir = build_process_ir(supported_source.expression)
    supported = backend._compile_for_generation(
        service_module._ExpandedProcess(
            request=supported_source,
            process_ir=supported_ir,
            source_request=supported_source,
        ),
        model,
        PhaseHandle("supported-dag", None, 1),
    )
    assert isinstance(supported, service_module._DagProcess)

    unsupported_source = ProcessRequest.parse("p p > z j j", name="loop-only")
    unsupported_ir = build_process_ir("g g > z g g")
    unsupported = backend._compile_for_generation(
        service_module._ExpandedProcess(
            request=ProcessRequest.parse(
                unsupported_ir.process,
                name="loop-only_1",
            ),
            process_ir=unsupported_ir,
            source_expansion_size=2,
            source_request=unsupported_source,
        ),
        model,
        PhaseHandle("unsupported-dag", None, 1),
    )
    assert isinstance(unsupported, service_module._UnsupportedProcess)

    with pytest.raises(
        GenerationError,
        match=(
            "no model-supported tree-level amplitudes for requested process: "
            "'p p > z j j'"
        ),
    ):
        service_module._partition_model_supported_processes(
            (supported, unsupported)
        )


def test_recurrence_generation_drops_expanded_processes_without_amplitudes() -> None:
    """Recurrence builders need an amplitude: unconnectable children are skipped."""

    backend = GenerationBackend(None, None)
    model = BuiltinSMModel()
    source = ProcessRequest.parse("p p > j j", name="pp_jj")

    def expanded(expression: str, index: int) -> service_module._ExpandedProcess:
        return service_module._ExpandedProcess(
            request=ProcessRequest.parse(expression, name=f"pp_jj_{index}"),
            process_ir=build_process_ir(expression),
            source_expansion_size=3,
            source_request=source,
        )

    candidates = (
        expanded("d d~ > g g", 1),
        expanded("d d > d d~", 2),
        expanded("g g > g d", 3),
    )
    kept = backend._drop_processes_without_amplitudes(candidates, model)
    assert kept == candidates[:1]

    with pytest.raises(GenerationError, match="'p p > j j'"):
        backend._drop_processes_without_amplitudes(candidates[1:], model)


def test_generation_plan_and_physics_preserve_selected_coverage() -> None:
    config = RunConfig(
        action="generate",
        process=ProcessConfig(
            selected_color_sector_ids=(0,),
            selected_source_helicities={"1": 1},
        ),
        evaluator=EvaluatorConfig(execution_mode="compiled"),
    )
    generator = Generator(config)
    plan = generator.plan("d d~ > z g g")
    planned = plan.estimated_coverage["processes"][0]
    assert planned["color_sector_count"] == 1
    assert planned["color_coverage"] == "selected"
    assert planned["helicity_coverage"] == "selected"

    backend = GenerationBackend(config, None)
    model = BuiltinSMModel()
    dag, coverage = backend._compile_concrete_process(
        build_process_ir("d d~ > z g g"),
        model,
    )
    physics = build_runtime_expression_schema(dag, model).to_mapping()["physics"]

    assert coverage["color_sector_count"] == 1
    assert coverage["color_coverage"] == "selected"
    assert coverage["helicity_coverage"] == "selected"
    assert physics["coverage"]["color"] == "selected"
    assert physics["coverage"]["helicities"] == "selected"
    assert all(record["values"][0] == 1 for record in physics["helicities"])


def test_generation_plan_rejects_missing_selected_sector() -> None:
    config = RunConfig(
        action="generate",
        process=ProcessConfig(selected_color_sector_ids=(999,)),
        evaluator=EvaluatorConfig(execution_mode="compiled"),
    )

    with pytest.raises(GenerationError, match="did not materialize requested"):
        Generator(config).plan("d d~ > z g g")


def test_generation_plan_uses_production_alias_validation() -> None:
    request = ProcessRequest.parse("d d~ > z g", name="base")
    valid = ProcessSet(
        (request,),
        aliases=(
            ProcessAlias(
                name="permuted",
                process_name="base",
                particle_permutation=(0, 1, 3, 2),
            ),
        ),
    )

    compiled_config = RunConfig(
        action="generate",
        evaluator=EvaluatorConfig(execution_mode="compiled"),
    )
    plan = Generator(compiled_config).plan(valid)

    assert plan.estimated_coverage["alias_count"] == 1
    assert plan.concrete_processes == (request,)

    both_sides = ProcessSet(
        (request,),
        aliases=(
            ProcessAlias(
                name="both_sides",
                process_name="base",
                particle_permutation=(1, 0, 3, 2),
            ),
        ),
    )
    assert Generator(compiled_config).plan(both_sides).estimated_coverage[
        "alias_count"
    ] == 1

    crossing = ProcessSet(
        (request,),
        aliases=(
            ProcessAlias(
                name="crossing",
                process_name="base",
                particle_permutation=(0, 2, 1, 3),
            ),
        ),
    )
    with pytest.raises(GenerationError, match="incoming and outgoing sides"):
        Generator(compiled_config).plan(crossing)

    invalid = ProcessSet(
        (request,),
        aliases=(
            ProcessAlias(
                name="bad",
                process_name="base",
                particle_permutation=(0, 1, 2),
            ),
        ),
    )
    with pytest.raises(GenerationError, match="permutation has length"):
        Generator(compiled_config).plan(invalid)


def test_minimal_coupling_limits_zero_nonminimal_model_orders() -> None:
    limits = infer_minimal_coupling_order_limits(
        build_process_ir("d d~ > u u~"),
        model=BuiltinSMModel(),
    )

    assert limits == {"QCD": 2, "QED": 0}


def test_minimal_coupling_limits_accept_existing_complete_color_plan() -> None:
    process = build_process_ir("d d~ > u u~")
    model = BuiltinSMModel()
    color_plan = build_color_plan(
        process,
        color_accuracy=process.color_accuracy,
        fold_trace_reflections=(
            model.lc_trace_reflection_equivalence_is_proven(process)
        ),
    )

    assert _infer_minimal_coupling_order_limits_from_color_plan(
        process,
        model=model,
        color_plan=color_plan,
    ) == infer_minimal_coupling_order_limits(process, model=model)


def test_process_preparation_reuses_complete_color_plan_for_minimal_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = build_process_ir("d d~ > u u~")
    model = BuiltinSMModel()
    reused_plans: list[object] = []
    infer_from_plan = (
        service_module._infer_minimal_coupling_order_limits_from_color_plan
    )

    def track_inference(*args: object, **kwargs: object) -> dict[str, int]:
        reused_plans.append(kwargs["color_plan"])
        return infer_from_plan(*args, **kwargs)

    def reject_rebuild(*_args: object, **_kwargs: object) -> dict[str, int]:
        pytest.fail("minimal inference rebuilt a color plan")

    monkeypatch.setattr(
        service_module,
        "_infer_minimal_coupling_order_limits_from_color_plan",
        track_inference,
    )
    monkeypatch.setattr(
        service_module,
        "infer_minimal_coupling_order_limits",
        reject_rebuild,
    )
    prepared = GenerationBackend(
        RunConfig(
            action="generate",
            evaluator=EvaluatorConfig(execution_mode="recurrence"),
        ),
        None,
    )._prepare_process_construction(process, model)

    assert len(reused_plans) == 1
    assert reused_plans[0] is prepared.complete_color_plan


def test_process_preparation_preserves_custom_color_plan_inference_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = build_process_ir("d d~ > u u~")
    model = BuiltinSMModel()
    rebuilds = 0

    def reject_reuse(*_args: object, **_kwargs: object) -> dict[str, int]:
        pytest.fail("custom color-plan inference unexpectedly reused a truncated plan")

    def track_rebuild(*_args: object, **_kwargs: object) -> dict[str, int]:
        nonlocal rebuilds
        rebuilds += 1
        return {"QCD": 2, "QED": 0}

    monkeypatch.setattr(
        service_module,
        "_infer_minimal_coupling_order_limits_from_color_plan",
        reject_reuse,
    )
    monkeypatch.setattr(
        service_module,
        "infer_minimal_coupling_order_limits",
        track_rebuild,
    )
    prepared = GenerationBackend(
        RunConfig(
            action="generate",
            process=ProcessConfig(max_color_sectors=1),
            evaluator=EvaluatorConfig(execution_mode="recurrence"),
        ),
        None,
    )._prepare_process_construction(process, model)

    assert rebuilds == 1
    assert prepared.coupling_order_limits == {"QCD": 2, "QED": 0}

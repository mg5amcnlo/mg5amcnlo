# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from pyamplicol import (
    CompiledModel,
    Generator,
    ModelSource,
    ResolvedEvaluation,
    Runtime,
)
from pyamplicol.artifacts import inspect_artifact
from pyamplicol.color.plan import build_color_plan
from pyamplicol.config import (
    ColorConfig,
    EagerEvaluatorConfig,
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationValidationConfig,
    JITConfig,
    ProcessConfig,
    RunConfig,
)
from pyamplicol.generation.phase_space import massive_rambo_final_state
from pyamplicol.models.builtin.process_ir import build_process_ir
from pyamplicol.models.builtin.validation import generic_validation_point

_RECURRENCE_KIND = "pyamplicol-runtime-recurrence-execution"
_RECURRENCE_PLAN_ABI = "pyamplicol-recurrence-plan-v2"
_SM_MODEL_ROOT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "pyamplicol"
    / "assets"
    / "models"
    / "json"
    / "sm"
)


@pytest.fixture(scope="module")
def ufo_sm_recurrence_jit_o2_model(
    tmp_path_factory: pytest.TempPathFactory,
) -> CompiledModel:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")
    if importlib.util.find_spec("symbolica") is None:
        pytest.skip("Symbolica is unavailable")
    root = tmp_path_factory.mktemp("lc-selected-flow-ufo-sm")
    model = ModelSource.from_path(
        _SM_MODEL_ROOT / "sm.json",
        restriction=_SM_MODEL_ROOT / "restrict_default.json",
    ).compile(
        cache_dir=root / "model-cache",
        use_cache=True,
        prepared_output=root / "ufo-sm-jit-o2.pyamplicol-model",
        evaluator=EvaluatorConfig(
            execution_mode="recurrence",
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(optimization_level=2),
        ),
    )
    assert model.is_prepared
    assert model.prepared_backend == "jit"
    return model


def _model_for_kind(
    model_kind: str,
    request: pytest.FixtureRequest,
) -> CompiledModel | None:
    if model_kind == "built-in":
        return None
    model = request.getfixturevalue("ufo_sm_recurrence_jit_o2_model")
    assert isinstance(model, CompiledModel)
    return model


@pytest.mark.parametrize(
    ("process", "flow_word", "final_masses"),
    (
        ("g g > g g", (1, 2, 3, 4), (0.0, 0.0)),
        ("g g > t t~", (3, 1, 2, 4), (173.0, 173.0)),
        ("d d~ > t t~", (2, 1, 3, 4), (173.0, 173.0)),
        ("d d~ > u u~ s s~", (2, 1, 3, 4, 5, 6), (0.0, 0.0, 0.0, 0.0)),
    ),
)
def test_selected_lc_artifact_matches_complete_artifact_physical_flow(
    tmp_path: Path,
    process: str,
    flow_word: tuple[int, ...],
    final_masses: tuple[float, ...],
) -> None:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")

    generation = GenerationConfig(
        emit_api_bundle=False,
        validation=GenerationValidationConfig(
            enabled=False,
            post_build_validation=False,
        ),
    )
    evaluator = EvaluatorConfig(jit=JITConfig(optimization_level=3))
    complete_path = tmp_path / "complete"
    selected_path = tmp_path / "selected"

    Generator(
        RunConfig(
            action="generate",
            generation=generation,
            evaluator=evaluator,
        )
    ).generate(process, complete_path)
    Generator(
        RunConfig(
            action="generate",
            process=ProcessConfig(
                reference_color_order=flow_word,
                selected_color_sector_ids=(0,),
            ),
            generation=generation,
            evaluator=evaluator,
        )
    ).generate(process, selected_path)

    final_state = massive_rambo_final_state(
        len(final_masses),
        sqrt_s=1000.0,
        masses=final_masses,
        seed=731,
    )
    momenta = (
        (
            (500.0, 0.0, 0.0, 500.0),
            (500.0, 0.0, 0.0, -500.0),
            *final_state,
        ),
    )
    flow_id = "flow:" + ",".join(str(label) for label in flow_word)
    complete = Runtime.load(complete_path)
    selected = Runtime.load(selected_path)
    complete_inspection = inspect_artifact(complete_path).processes[0]
    selected_inspection = inspect_artifact(selected_path).processes[0]

    assert complete_inspection.selector_provenance == (_RECURRENCE_PLAN_ABI)
    assert complete_inspection.helicity_runtime_contract == "complete-reusable"
    assert complete_inspection.color_flow_runtime_contract == "complete-reusable"
    assert complete_inspection.generation_specialized_axes == ()
    assert complete_inspection.selected_source_helicities == ()
    assert complete_inspection.selected_color_sector_ids == ()
    assert complete_inspection.execution_mode == "recurrence"
    assert complete_inspection.lc_flow_layout == "topology-replay"
    if complete_inspection.lc_physical_sector_count is not None:
        assert complete_inspection.lc_replayed_sector_count is not None
        assert complete_inspection.lc_residual_sector_count is not None
        assert (
            complete_inspection.lc_replayed_sector_count
            + complete_inspection.lc_residual_sector_count
            == complete_inspection.lc_physical_sector_count
        )
        assert complete_inspection.lc_materialized_sector_count is not None
        assert (
            complete_inspection.lc_materialized_sector_count
            <= complete_inspection.lc_physical_sector_count
        )

    assert selected_inspection.helicity_runtime_contract == "complete-reusable"
    assert selected_inspection.color_flow_runtime_contract == ("generation-specialized")
    assert selected_inspection.generation_specialized_axes == ("color_flow",)
    assert selected_inspection.selected_source_helicities == ()
    assert selected_inspection.selected_color_sector_ids == (0,)
    assert selected_inspection.lc_physical_sector_count is None

    assert flow_id in {flow.id for flow in complete.physics.color_flows}
    selected_flow_by_id = {flow.id: flow for flow in selected.physics.color_flows}
    assert flow_id in selected_flow_by_id
    assert selected_flow_by_id[flow_id].word == flow_word

    complete_component = complete.evaluate_resolved(
        momenta,
        color_flows=[flow_id],
    ).total()[0]
    complete_total = complete.evaluate(momenta, color_flows=[flow_id])[0]
    selected_total = selected.evaluate(momenta, color_flows=[flow_id])[0]
    assert complete_total == pytest.approx(
        selected_total,
        rel=1.0e-12,
        abs=1.0e-15,
    )
    assert selected_total.real == pytest.approx(
        complete_component.real,
        rel=1.0e-12,
        abs=1.0e-15,
    )
    assert selected_total.imag == pytest.approx(
        complete_component.imag,
        rel=1.0e-12,
        abs=1.0e-15,
    )
    profile = complete._backend.profile(momenta, color_flows=[flow_id])
    assert profile["resolved_materialized_component_count"] == 0
    assert profile["total_materialized_value_count"] == len(momenta)
    homogeneous_profile = complete._backend.profile(
        momenta,
        color_flow_by_point=[flow_id] * len(momenta),
    )
    assert homogeneous_profile["selector_plan_kind"] == "homogeneous"
    assert homogeneous_profile["resolved_materialized_component_count"] == 0
    assert homogeneous_profile["total_materialized_value_count"] == len(momenta)


def test_complete_pure_gluon_replay_matches_every_sector_specialization(
    tmp_path: Path,
) -> None:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")

    process = "g g > g g"
    generation = GenerationConfig(
        emit_api_bundle=False,
        validation=GenerationValidationConfig(
            enabled=False,
            post_build_validation=False,
        ),
    )
    evaluator = EvaluatorConfig(
        eager=EagerEvaluatorConfig(point_tile_size=2),
        jit=JITConfig(optimization_level=3),
    )
    complete_path = tmp_path / "complete"
    Generator(
        RunConfig(
            action="generate",
            generation=generation,
            evaluator=evaluator,
        )
    ).generate(process, complete_path)

    final_state = massive_rambo_final_state(
        2,
        sqrt_s=1000.0,
        masses=(0.0, 0.0),
        seed=731,
    )
    momenta = (
        (
            (500.0, 0.0, 0.0, 500.0),
            (500.0, 0.0, 0.0, -500.0),
            *final_state,
        ),
    )
    complete = Runtime.load(complete_path)
    resolved = complete.evaluate_resolved(momenta)
    exact = complete.evaluate_resolved(momenta, precision=32)
    total = complete.evaluate(momenta)[0]
    assert total == pytest.approx(
        resolved.total()[0],
        rel=1.0e-12,
        abs=1.0e-15,
    )
    tiled_momenta = momenta * 3
    tiled_total = complete.evaluate(tiled_momenta)
    tiled_resolved = complete.evaluate_resolved(tiled_momenta)
    assert tiled_total == pytest.approx(
        tiled_resolved.total(),
        rel=1.0e-12,
        abs=1.0e-15,
    )
    assert exact.helicity_ids == resolved.helicity_ids
    assert exact.color_ids == resolved.color_ids
    color_plan = build_color_plan(
        build_process_ir(process),
        color_accuracy="lc",
        fold_trace_reflections=True,
    )
    sector_by_flow: dict[str, int] = {}
    for sector in color_plan.sectors:
        word = tuple(sector.word_labels or sector.color_words[0])
        words = {word}
        if sector.kind == "single-trace" and len(word) > 2:
            words.add((word[0], *reversed(word[1:])))
        for physical_word in words:
            sector_by_flow[
                "flow:" + ",".join(str(label) for label in physical_word)
            ] = int(sector.id)

    for helicity_index, helicity in enumerate(complete.physics.helicities):
        if helicity.structural_zero:
            continue
        selected_helicities = {
            str(label): value for label, value in enumerate(helicity.values, start=1)
        }
        specialized_by_sector: dict[int, Runtime] = {}
        for sector in color_plan.sectors:
            path = tmp_path / f"h{helicity_index}-sector{sector.id}"
            Generator(
                RunConfig(
                    action="generate",
                    process=ProcessConfig(
                        selected_color_sector_ids=(int(sector.id),),
                        selected_source_helicities=selected_helicities,
                    ),
                    generation=generation,
                    evaluator=evaluator,
                )
            ).generate(process, path)
            specialized_by_sector[int(sector.id)] = Runtime.load(path)

        for color_index, color in enumerate(complete.physics.color_flows):
            reusable = resolved.values[0][helicity_index][color_index]
            exact_reusable = exact.values[0][helicity_index][color_index]
            specialized = specialized_by_sector[sector_by_flow[color.id]].evaluate(
                momenta,
                color_flows=(color.id,),
            )[0]
            assert reusable.real == pytest.approx(
                specialized.real,
                rel=1.0e-12,
                abs=1.0e-15,
            )
            assert reusable.imag == pytest.approx(
                specialized.imag,
                rel=1.0e-12,
                abs=1.0e-15,
            )
            assert complex(exact_reusable).real == pytest.approx(
                reusable.real,
                rel=1.0e-12,
                abs=1.0e-15,
            )
            assert complex(exact_reusable).imag == pytest.approx(
                reusable.imag,
                rel=1.0e-12,
                abs=1.0e-15,
            )

    assert complex(exact.total()[0]).real == pytest.approx(
        resolved.total()[0].real,
        rel=1.0e-12,
        abs=1.0e-15,
    )


@pytest.mark.parametrize("model_kind", ("built-in", "ufo-sm"))
def test_complete_lc_fixed_helicity_selection_composes_with_flow_replay(
    tmp_path: Path,
    model_kind: str,
    request: pytest.FixtureRequest,
) -> None:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")

    artifact = tmp_path / model_kind
    model = _model_for_kind(model_kind, request)
    Generator(
        RunConfig(
            action="generate",
            generation=GenerationConfig(
                emit_api_bundle=False,
                validation=GenerationValidationConfig(
                    enabled=False,
                    post_build_validation=False,
                ),
            ),
            evaluator=EvaluatorConfig(jit=JITConfig(optimization_level=3)),
        )
    ).generate("d d~ > z g g", artifact, model=model)

    execution = json.loads(
        (artifact / "processes/d_dbar_to_z_g_g/execution.json").read_text(
            encoding="utf-8"
        )
    )
    assert execution["kind"] == _RECURRENCE_KIND
    assert execution["recurrence_plan_abi"] == _RECURRENCE_PLAN_ABI
    assert (
        execution["plan"]["inspection_summary"]["direct_arena"]["selector_domain_count"]
        == 2
    )

    point = (
        (500.0, 0.0, 0.0, 500.0),
        (500.0, 0.0, 0.0, -500.0),
        (
            312.48956564409934,
            -193.66118273046334,
            121.57983298112629,
            192.4790061491314,
        ),
        (
            487.79620348021274,
            250.4771539987287,
            -206.7128292830862,
            -363.97271554910253,
        ),
        (
            199.71423087568797,
            -56.81597126826545,
            85.1329963019599,
            171.49370939997104,
        ),
    )
    runtime = Runtime.load(artifact)
    flow_id = "flow:2,4,5,1"
    selected_flow = runtime.evaluate((point,), color_flows=(flow_id,))[0]
    selected_flow_exact = runtime.evaluate(
        (point,),
        color_flows=(flow_id,),
        precision=32,
    )[0]
    assert selected_flow == pytest.approx(
        complex(selected_flow_exact),
        rel=1.0e-12,
        abs=1.0e-15,
    )
    assert selected_flow.real == pytest.approx(
        0.00022626601239021538,
        rel=1.0e-11,
        abs=1.0e-15,
    )

    helicity_id = "h:-1,+1,-1,+1,-1"
    complete = runtime.evaluate_resolved((point,))
    helicity_index = complete.helicity_ids.index(helicity_id)
    selected = runtime.evaluate_resolved((point,), helicities=(helicity_id,))

    assert selected.color_ids == complete.color_ids
    assert selected.values[0][0] == pytest.approx(
        complete.values[0][helicity_index],
        rel=1.0e-12,
        abs=1.0e-15,
    )
    total = runtime.evaluate((point,), helicities=(helicity_id,))[0]
    assert total == pytest.approx(
        sum(complete.values[0][helicity_index]),
        rel=1.0e-12,
        abs=1.0e-15,
    )
    assert total.real == pytest.approx(
        0.000180803517738144,
        rel=1.0e-11,
        abs=1.0e-15,
    )


@pytest.mark.parametrize(
    (
        "process",
        "expected_total",
        "expected_fixed_helicity",
        "expected_all_flow_fixed_helicity",
    ),
    (
        (
            "d d~ > u u~ s s~ g",
            8.495913119023438e-15,
            2.6043061427778106e-15,
            4.899181433766111e-14,
        ),
        (
            "d d~ > u u~ u u~ g",
            3.215288522484574e-15,
            5.102730162523396e-16,
            4.4151994948046454e-14,
        ),
    ),
)
def test_three_line_lc_publication_reload_and_union_components_match(
    process: str,
    expected_total: float,
    expected_fixed_helicity: float,
    expected_all_flow_fixed_helicity: float,
    tmp_path: Path,
) -> None:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")

    generation = GenerationConfig(
        emit_api_bundle=False,
        validation=GenerationValidationConfig(
            enabled=False,
            post_build_validation=False,
        ),
    )
    evaluator = EvaluatorConfig(jit=JITConfig(optimization_level=3))
    complete_path = tmp_path / "complete"
    union_path = tmp_path / "all-flow-union"
    Generator(
        RunConfig(
            action="generate",
            generation=generation,
            evaluator=evaluator,
        )
    ).generate(process, complete_path)
    Generator(
        RunConfig(
            action="generate",
            color=ColorConfig(lc_flow_layout="all-flow-union"),
            generation=generation,
            evaluator=evaluator,
        )
    ).generate(process, union_path)

    point = tuple(
        tuple(float(component) for component in particle.momentum)
        for particle in generic_validation_point(process)
    )
    points = (point,)
    flow_id = "flow:2,7,1,3,4,5,6"
    helicity_id = "h:-1,+1,-1,+1,-1,+1,-1"
    complete = Runtime.load(complete_path)
    union_immediate = Runtime.load(union_path)

    def flattened_values(resolved: ResolvedEvaluation) -> tuple[complex, ...]:
        return tuple(
            complex(value)
            for point_values in resolved.values
            for helicity_values in point_values
            for value in helicity_values
        )

    complete_selected = complete.evaluate_resolved(points, color_flows=(flow_id,))
    union_selected = union_immediate.evaluate_resolved(
        points,
        color_flows=(flow_id,),
    )
    assert union_selected.helicity_ids == complete_selected.helicity_ids
    assert union_selected.color_ids == complete_selected.color_ids
    assert flattened_values(union_selected) == pytest.approx(
        flattened_values(complete_selected),
        rel=1.0e-12,
        abs=1.0e-15,
    )
    assert complete.evaluate(points, color_flows=(flow_id,))[0].real == pytest.approx(
        expected_total,
        rel=1.0e-11,
        abs=1.0e-25,
    )
    assert complete.evaluate(
        points,
        color_flows=(flow_id,),
        helicities=(helicity_id,),
    )[0].real == pytest.approx(
        expected_fixed_helicity,
        rel=1.0e-11,
        abs=1.0e-25,
    )

    complete_fixed = complete.evaluate_resolved(points, helicities=(helicity_id,))
    union_fixed = union_immediate.evaluate_resolved(points, helicities=(helicity_id,))
    assert union_fixed.color_ids == complete_fixed.color_ids
    assert flattened_values(union_fixed) == pytest.approx(
        flattened_values(complete_fixed),
        rel=1.0e-12,
        abs=1.0e-15,
    )
    assert complete_fixed.total()[0].real == pytest.approx(
        expected_all_flow_fixed_helicity,
        rel=1.0e-11,
        abs=1.0e-24,
    )

    union_reloaded = Runtime.load(union_path)
    reloaded_selected = union_reloaded.evaluate_resolved(points, color_flows=(flow_id,))
    assert flattened_values(reloaded_selected) == pytest.approx(
        flattened_values(union_selected),
        rel=0.0,
        abs=0.0,
    )


@pytest.mark.parametrize("model_kind", ("built-in", "ufo-sm"))
def test_complete_pure_gluon_fixed_helicity_preserves_every_physical_flow(
    tmp_path: Path,
    model_kind: str,
    request: pytest.FixtureRequest,
) -> None:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")

    artifact = tmp_path / model_kind
    model = _model_for_kind(model_kind, request)
    Generator(
        RunConfig(
            action="generate",
            generation=GenerationConfig(
                emit_api_bundle=False,
                validation=GenerationValidationConfig(
                    enabled=False,
                    post_build_validation=False,
                ),
            ),
            evaluator=EvaluatorConfig(jit=JITConfig(optimization_level=3)),
        )
    ).generate("g g > g g", artifact, model=model)

    execution = json.loads(
        (artifact / "processes/g_g_to_g_g/execution.json").read_text(encoding="utf-8")
    )
    assert execution["kind"] == _RECURRENCE_KIND
    assert execution["recurrence_plan_abi"] == _RECURRENCE_PLAN_ABI
    assert (
        execution["plan"]["inspection_summary"]["direct_arena"]["selector_domain_count"]
        == 4
    )

    point = (
        (500.0, 0.0, 0.0, 500.0),
        (500.0, 0.0, 0.0, -500.0),
        (
            499.99999999999994,
            -306.65836769058797,
            210.51071473894038,
            334.1345305493651,
        ),
        (
            499.99999999999994,
            306.65836769058797,
            -210.51071473894038,
            -334.1345305493651,
        ),
    )
    runtime = Runtime.load(artifact)
    helicity_id = "h:-1,+1,-1,+1"
    complete = runtime.evaluate_resolved((point,))
    helicity_index = complete.helicity_ids.index(helicity_id)
    selected = runtime.evaluate_resolved((point,), helicities=(helicity_id,))
    exact_selected = runtime.evaluate_resolved(
        (point,),
        helicities=(helicity_id,),
        precision=32,
    )

    expected_by_flow = {
        "flow:1,2,3,4": 0.2420310029906875,
        "flow:1,2,4,3": 6.121124826711009,
        "flow:1,3,2,4": 8.79749514971447,
        "flow:1,3,4,2": 6.121124826711009,
        "flow:1,4,2,3": 8.79749514971447,
        "flow:1,4,3,2": 0.2420310029906875,
    }
    assert selected.color_ids == complete.color_ids
    assert selected.values[0][0] == pytest.approx(
        complete.values[0][helicity_index],
        rel=1.0e-11,
        abs=1.0e-14,
    )
    assert exact_selected.color_ids == selected.color_ids
    for color_id, value in zip(
        selected.color_ids,
        selected.values[0][0],
        strict=True,
    ):
        assert value.real == pytest.approx(
            expected_by_flow[color_id],
            rel=1.0e-10,
            abs=1.0e-13,
        )
        assert value.imag == pytest.approx(0.0, abs=1.0e-15)
    for exact_value, value in zip(
        exact_selected.values[0][0],
        selected.values[0][0],
        strict=True,
    ):
        assert complex(exact_value) == pytest.approx(
            value,
            rel=1.0e-12,
            abs=1.0e-15,
        )

    total = runtime.evaluate((point,), helicities=(helicity_id,))[0]
    assert total.real == pytest.approx(
        30.321301958832333,
        rel=1.0e-10,
        abs=1.0e-12,
    )
    assert total.imag == pytest.approx(0.0, abs=1.0e-15)


@pytest.mark.parametrize("model_kind", ("built-in", "ufo-sm"))
@pytest.mark.parametrize("execution_mode", ("compiled", "eager"))
def test_all_flow_union_supports_complete_and_selected_runtime_axes(
    tmp_path: Path,
    execution_mode: str,
    model_kind: str,
) -> None:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")

    artifact = tmp_path / f"{execution_mode}-{model_kind}"
    model = None
    if model_kind == "ufo-sm":
        model_root = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "pyamplicol"
            / "assets"
            / "models"
            / "json"
            / "sm"
        )
        model = ModelSource.from_path(
            model_root / "sm.json",
            restriction=model_root / "restrict_default.json",
        )
        if execution_mode == "eager":
            eager_evaluator = EvaluatorConfig(
                execution_mode="eager",
                jit=JITConfig(optimization_level=0),
            )
            model = model.compile(
                use_cache=True,
                prepared_output=tmp_path / "ufo-sm-jit-o0.pyamplicol-model",
                evaluator=eager_evaluator,
            )
    Generator(
        RunConfig(
            action="generate",
            color=ColorConfig(lc_flow_layout="all-flow-union"),
            generation=GenerationConfig(
                emit_api_bundle=False,
                validation=GenerationValidationConfig(
                    enabled=False,
                    post_build_validation=False,
                ),
            ),
            evaluator=EvaluatorConfig(
                execution_mode=execution_mode,
                jit=JITConfig(optimization_level=0 if execution_mode == "eager" else 3),
            ),
        )
    ).generate("d d~ > z g g", artifact, model=model)

    point = (
        (500.0, 0.0, 0.0, 500.0),
        (500.0, 0.0, 0.0, -500.0),
        (
            312.48956564409934,
            -193.66118273046334,
            121.57983298112629,
            192.4790061491314,
        ),
        (
            487.79620348021274,
            250.4771539987287,
            -206.7128292830862,
            -363.97271554910253,
        ),
        (
            199.71423087568797,
            -56.81597126826545,
            85.1329963019599,
            171.49370939997104,
        ),
    )
    inspection = inspect_artifact(artifact).processes[0]
    runtime = Runtime.load(artifact)
    resolved = runtime.evaluate_resolved((point,))
    total = runtime.evaluate((point,))[0]

    assert inspection.lc_flow_layout == "all-flow-union"
    assert inspection.lc_union_sector_count == len(resolved.color_ids)
    assert total == pytest.approx(
        resolved.total()[0],
        rel=1.0e-12,
        abs=1.0e-15,
    )

    helicity_id = "h:-1,+1,-1,+1,-1"
    helicity_index = resolved.helicity_ids.index(helicity_id)
    selected_helicity = runtime.evaluate_resolved((point,), helicities=(helicity_id,))
    assert selected_helicity.values[0][0] == pytest.approx(
        resolved.values[0][helicity_index],
        rel=1.0e-12,
        abs=1.0e-15,
    )
    assert selected_helicity.total()[0].real == pytest.approx(
        0.000180803517738144,
        rel=1.0e-10,
        abs=1.0e-15,
    )
    assert selected_helicity.total()[0].imag == pytest.approx(0.0, abs=1.0e-15)

    color_id = resolved.color_ids[-1]
    color_index = resolved.color_ids.index(color_id)
    selected_color = runtime.evaluate_resolved((point,), color_flows=(color_id,))
    assert tuple(row[0] for row in selected_color.values[0]) == pytest.approx(
        tuple(row[color_index] for row in resolved.values[0]),
        rel=1.0e-12,
        abs=1.0e-15,
    )

    exact = runtime.evaluate_resolved(
        (point,),
        helicities=(helicity_id,),
        color_flows=(color_id,),
        precision=32,
    )
    assert complex(exact.total()[0]) == pytest.approx(
        selected_helicity.values[0][0][color_index],
        rel=1.0e-12,
        abs=1.0e-15,
    )

    nonzero_helicity_indices = [
        index
        for index, helicity in enumerate(runtime.physics.helicities)
        if not helicity.structural_zero
    ][:2]
    assert len(nonzero_helicity_indices) == 2
    per_point = runtime.evaluate(
        (point, point),
        helicity_by_point=tuple(
            resolved.helicity_ids[index] for index in nonzero_helicity_indices
        ),
        color_flow_by_point=(resolved.color_ids[0], resolved.color_ids[-1]),
    )
    assert per_point == pytest.approx(
        (
            resolved.values[0][nonzero_helicity_indices[0]][0],
            resolved.values[0][nonzero_helicity_indices[1]][-1],
        ),
        rel=1.0e-12,
        abs=1.0e-15,
    )


def test_eager_complete_replay_matches_specialized_flows_and_point_selectors(
    tmp_path: Path,
) -> None:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")

    process = "d d~ > z g g"
    generation = GenerationConfig(
        emit_api_bundle=False,
        validation=GenerationValidationConfig(
            enabled=False,
            post_build_validation=False,
        ),
    )
    eager_path = tmp_path / "eager-complete"
    Generator(
        RunConfig(
            action="generate",
            generation=generation,
            evaluator=EvaluatorConfig(
                execution_mode="eager",
                jit=JITConfig(optimization_level=0),
            ),
        )
    ).generate(process, eager_path)

    final_state = massive_rambo_final_state(
        3,
        sqrt_s=1000.0,
        masses=(91.188, 0.0, 0.0),
        seed=731,
    )
    point = (
        (500.0, 0.0, 0.0, 500.0),
        (500.0, 0.0, 0.0, -500.0),
        *final_state,
    )
    eager = Runtime.load(eager_path)
    resolved = eager.evaluate_resolved((point,))
    exact = eager.evaluate_resolved((point,), precision=32)
    inspection = inspect_artifact(eager_path).processes[0]

    assert inspection.execution_mode == "eager"
    assert inspection.lc_materialized_sector_count == 1
    assert inspection.lc_physical_sector_count == 2
    assert exact.helicity_ids == resolved.helicity_ids
    assert exact.color_ids == resolved.color_ids

    color_plan = build_color_plan(build_process_ir(process), color_accuracy="lc")
    sector_by_flow = {
        "flow:" + ",".join(str(label) for label in sector.word_labels): int(sector.id)
        for sector in color_plan.sectors
    }
    specialized_by_sector = {}
    for sector in color_plan.sectors:
        selected_path = tmp_path / f"selected-{sector.id}"
        Generator(
            RunConfig(
                action="generate",
                process=ProcessConfig(
                    selected_color_sector_ids=(int(sector.id),),
                ),
                generation=generation,
                evaluator=EvaluatorConfig(jit=JITConfig(optimization_level=3)),
            )
        ).generate(process, selected_path)
        selected_runtime = Runtime.load(selected_path)
        specialized = selected_runtime.evaluate_resolved((point,))
        specialized_by_sector[int(sector.id)] = specialized

    for helicity_index, _helicity_id in enumerate(resolved.helicity_ids):
        for color_index, color_id in enumerate(resolved.color_ids):
            specialized = specialized_by_sector[sector_by_flow[color_id]]
            actual = resolved.values[0][helicity_index][color_index]
            expected = specialized.values[0][helicity_index][0]
            exact_actual = complex(exact.values[0][helicity_index][color_index])
            assert actual.real == pytest.approx(expected.real, rel=1.0e-12, abs=1.0e-15)
            assert actual.imag == pytest.approx(expected.imag, rel=1.0e-12, abs=1.0e-15)
            assert exact_actual.real == pytest.approx(
                actual.real, rel=1.0e-12, abs=1.0e-15
            )
            assert exact_actual.imag == pytest.approx(
                actual.imag, rel=1.0e-12, abs=1.0e-15
            )

    # Independent legacy generated-library oracle at this deterministic point,
    # adjusted to the exact alpha_ew = 1 / 132.507 normalization.
    helicity_sums = tuple(
        sum(
            resolved.values[0][helicity_index][color_index].real
            for helicity_index in range(len(resolved.helicity_ids))
        )
        for color_index in range(len(resolved.color_ids))
    )
    assert dict(zip(resolved.color_ids, helicity_sums, strict=True)) == pytest.approx(
        {
            "flow:2,4,5,1": 2.20775902078958e-7,
            "flow:2,5,4,1": 4.767041136002407e-2,
        },
        rel=1.0e-12,
        abs=1.0e-15,
    )

    point_batch = (point, point)
    selected = eager.evaluate(
        point_batch,
        helicity_by_point=(resolved.helicity_ids[0], resolved.helicity_ids[1]),
        color_flow_by_point=(resolved.color_ids[0], resolved.color_ids[1]),
    )
    assert selected[0].real == pytest.approx(
        resolved.values[0][0][0].real, rel=1.0e-12, abs=1.0e-15
    )
    assert selected[1].real == pytest.approx(
        resolved.values[0][1][1].real, rel=1.0e-12, abs=1.0e-15
    )


def test_external_sm_complete_three_line_flows_match_builtin(
    tmp_path: Path,
    ufo_sm_recurrence_jit_o2_model: CompiledModel,
) -> None:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")

    config = RunConfig(
        action="generate",
        generation=GenerationConfig(
            emit_api_bundle=False,
            validation=GenerationValidationConfig(
                enabled=False,
                post_build_validation=False,
            ),
        ),
        evaluator=EvaluatorConfig(jit=JITConfig(optimization_level=3)),
    )
    process = "d d~ > u u~ s s~"
    builtin_path = tmp_path / "builtin"
    external_path = tmp_path / "external"
    Generator(config).generate(process, builtin_path)
    Generator(config).generate(
        process,
        external_path,
        model=ufo_sm_recurrence_jit_o2_model,
    )

    final_state = massive_rambo_final_state(
        4,
        sqrt_s=1000.0,
        masses=(0.0, 0.0, 0.0, 0.0),
        seed=731,
    )
    momenta = (
        (
            (500.0, 0.0, 0.0, 500.0),
            (500.0, 0.0, 0.0, -500.0),
            *final_state,
        ),
    )
    builtin = Runtime.load(builtin_path).evaluate_resolved(momenta)
    external = Runtime.load(external_path).evaluate_resolved(momenta)

    assert external.helicity_ids == builtin.helicity_ids
    assert external.color_ids == builtin.color_ids
    for builtin_helicities, external_helicities in zip(
        builtin.values[0], external.values[0], strict=True
    ):
        for builtin_value, external_value in zip(
            builtin_helicities, external_helicities, strict=True
        ):
            assert external_value.real == pytest.approx(
                builtin_value.real,
                rel=1.0e-11,
                abs=1.0e-20,
            )
            assert external_value.imag == pytest.approx(
                builtin_value.imag,
                rel=1.0e-11,
                abs=1.0e-20,
            )


def test_external_sm_fixed_width_multigluon_current_matches_builtin(
    tmp_path: Path,
    ufo_sm_recurrence_jit_o2_model: CompiledModel,
) -> None:
    """Keep the proven AmpliCol transverse Yang--Mills convention."""

    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")

    config = RunConfig(
        action="generate",
        process=ProcessConfig(
            selected_source_helicities={
                "1": -1,
                "2": 1,
                "3": -1,
                "4": 1,
                "5": -1,
                "6": 1,
            }
        ),
        generation=GenerationConfig(
            emit_api_bundle=False,
            validation=GenerationValidationConfig(
                enabled=False,
                post_build_validation=False,
            ),
        ),
        evaluator=EvaluatorConfig(jit=JITConfig(optimization_level=3)),
    )
    process = "d d~ > t t~ g g"
    builtin_path = tmp_path / "builtin-fixed-width"
    external_path = tmp_path / "external-fixed-width"
    Generator(config).generate(process, builtin_path)
    Generator(config).generate(
        process,
        external_path,
        model=ufo_sm_recurrence_jit_o2_model,
    )

    final_state = massive_rambo_final_state(
        4,
        sqrt_s=1000.0,
        masses=(173.0, 173.0, 0.0, 0.0),
        seed=731,
    )
    momenta = (
        (
            (500.0, 0.0, 0.0, 500.0),
            (500.0, 0.0, 0.0, -500.0),
            *final_state,
        ),
    )
    builtin = Runtime.load(builtin_path).evaluate(momenta)[0]
    external = Runtime.load(external_path).evaluate(momenta)[0]

    assert external.real == pytest.approx(
        builtin.real,
        rel=1.0e-11,
        abs=1.0e-20,
    )
    assert external.imag == pytest.approx(
        builtin.imag,
        rel=1.0e-11,
        abs=1.0e-20,
    )

# SPDX-License-Identifier: 0BSD
"""Prepared-model oracles through public Direct-Arena recurrence artifacts."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import pytest

from pyamplicol import CompiledModel, Generator, ModelSource, Runtime
from pyamplicol.assets.prepared_models import (
    BUILTIN_SM_JIT_O2,
    packaged_prepared_model_path,
)
from pyamplicol.config import (
    Action,
    ColorAccuracy,
    ColorConfig,
    ColorContraction,
    EvaluatorConfig,
    EvaluatorExecutionMode,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationValidationConfig,
    JITConfig,
    LCFlowLayout,
    RunConfig,
)
from pyamplicol.generation.phase_space import massive_rambo_final_state

_ZG_EXPRESSION = "d d~ > z g"
_ZGG_EXPRESSION = "d d~ > z g g"
_GG_EXPRESSION = "g g > g g"
_SAME_FLAVOUR_EXPRESSION = "d d~ > d d~"
_THREE_LINE_EXPRESSION = "d d~ > u u~ s s~"
_CROSSING_HELICITY_ID = "h:-1,+1,-1,-1"
_CROSSING_FLOW_ID = "flow:2,4,1"
_RECURRENCE_KIND = "pyamplicol-runtime-recurrence-execution"
_RECURRENCE_PLAN_ABI = "pyamplicol-recurrence-plan-v2"
_RECURRENCE_RUNTIME_LAYOUT_ABI = "pyamplicol-recurrence-runtime-layout-v2"
_RECURRENCE_CAPABILITIES = {
    "rusticol.recurrence-color.lc.v1",
    "rusticol.recurrence-direct-arena.complex-f64.v1",
}
# The immutable physics-v2 oracle was captured with the former rounded default.
_HISTORICAL_REFERENCE_ALPHA_EW = 0.007546771114
_THREE_LINE_HELICITY_SUM_BY_FLOW = {
    "flow:2,1,3,4,5,6": 1.7260373034739047e-11,
    "flow:2,1,3,6,5,4": 1.6570372188601389e-11,
    "flow:2,4,3,1,5,6": 1.0167883928661888e-10,
    "flow:2,4,3,6,5,1": 1.6445040193096446e-10,
    "flow:2,6,3,1,5,4": 6.5052811045824496e-10,
    "flow:2,6,3,4,5,1": 1.2411150983558527e-10,
}
_THREE_LINE_HELICITY_SUM_TOTAL = 1.0745996067347547e-9
_UFO_SM_ROOT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "pyamplicol"
    / "assets"
    / "models"
    / "json"
    / "sm"
)
_REFERENCE_PAYLOAD = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "reference"
        / "physics-v2.json"
    ).read_text(encoding="utf-8")
)

_Point = tuple[tuple[float, ...], ...]


def _unavailable(reason: str) -> None:
    if os.environ.get("PYAMPLICOL_REQUIRE_NATIVE_TESTS") == "1":
        pytest.fail(reason)
    pytest.skip(reason)


def _require_native_direct_arena() -> None:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        _unavailable("the Rusticol extension has not been built")
    if importlib.util.find_spec("symbolica") is None:
        _unavailable("Symbolica is unavailable")
    rusticol = importlib.import_module("pyamplicol._rusticol")
    if not hasattr(rusticol, "_lower_recurrence_direct_v2"):
        _unavailable("the installed Rusticol extension lacks Direct-Arena v2")


def _generation_config(
    *,
    lc_flow_layout: LCFlowLayout = LCFlowLayout.TOPOLOGY_REPLAY,
    color_accuracy: ColorAccuracy = ColorAccuracy.LC,
    color_contraction: ColorContraction = ColorContraction.DIRECT,
) -> RunConfig:
    return RunConfig(
        action=Action.GENERATE,
        color=ColorConfig(
            accuracy=color_accuracy,
            contraction=color_contraction,
            # These structural oracles exercise trace-word reflection/replay.
            fft_basis="trace",
            lc_flow_layout=lc_flow_layout,
        ),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
            validation=GenerationValidationConfig(
                enabled=False,
                post_build_validation=False,
            ),
        ),
        evaluator=EvaluatorConfig(
            execution_mode=EvaluatorExecutionMode.RECURRENCE,
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(optimization_level=2),
        ),
    )


@pytest.fixture(scope="module")
def ufo_sm_recurrence_jit_o2_model(
    tmp_path_factory: pytest.TempPathFactory,
) -> CompiledModel:
    """Prepare one model-generic UFO-SM pack for all Direct-Arena oracles."""

    _require_native_direct_arena()
    root = tmp_path_factory.mktemp("prepared-execution-ufo-sm")
    model = ModelSource.from_path(
        _UFO_SM_ROOT / "sm.json",
        restriction=_UFO_SM_ROOT / "restrict_default.json",
    ).compile(
        cache_dir=root / "model-cache",
        use_cache=True,
        prepared_output=root / "ufo-sm-jit-o2.pyamplicol-model",
        evaluator=_generation_config().evaluator,
    )
    assert model.is_prepared
    assert model.prepared_backend == "jit"
    return model


def _assert_direct_arena_v2_artifact(
    artifact: Path,
    *,
    expected_layout: LCFlowLayout,
) -> None:
    execution_files = tuple((artifact / "processes").glob("*/execution.json"))
    assert len(execution_files) == 1
    execution = json.loads(execution_files[0].read_text(encoding="utf-8"))
    assert execution["kind"] == _RECURRENCE_KIND
    assert execution["recurrence_plan_abi"] == _RECURRENCE_PLAN_ABI
    assert execution["runtime_layout_abi"] == _RECURRENCE_RUNTIME_LAYOUT_ABI
    assert set(execution["required_runtime_capabilities"]) == (_RECURRENCE_CAPABILITIES)
    assert execution["plan"]["recurrence_plan_abi"] == _RECURRENCE_PLAN_ABI
    assert execution["plan"]["runtime_layout_abi"] == (_RECURRENCE_RUNTIME_LAYOUT_ABI)
    schedule = execution["plan"]["runtime_schedule"]
    assert schedule["path"].startswith("recurrence/schedules/")
    assert schedule["path"].endswith("/recurrence-runtime.pacbin")
    assert execution["plan"]["process_binding"]["path"] == "recurrence-binding.bin"
    assert execution["recurrence_summary"]["lc_flow_layout"] == expected_layout.value


def _generate_direct_arena_runtime(
    *,
    tmp_path: Path,
    model_source: str,
    expression: str,
    ufo_model: CompiledModel | None,
    lc_flow_layout: LCFlowLayout = LCFlowLayout.TOPOLOGY_REPLAY,
) -> Runtime:
    _require_native_direct_arena()
    slug = expression.replace("~", "bar").replace(">", "to").replace(" ", "_")
    artifact = tmp_path / f"{model_source}-{slug}-{lc_flow_layout.value}-direct-v2"
    generator = Generator(_generation_config(lc_flow_layout=lc_flow_layout))
    if model_source == "built-in":
        with packaged_prepared_model_path(BUILTIN_SM_JIT_O2) as prepared_model:
            generator.generate(
                expression,
                artifact,
                model=ModelSource.from_path(prepared_model),
            )
    else:
        assert ufo_model is not None
        generator.generate(expression, artifact, model=ufo_model)
    _assert_direct_arena_v2_artifact(
        artifact,
        expected_layout=lc_flow_layout,
    )
    runtime = Runtime.load(artifact)
    if model_source == "built-in":
        runtime.set_model_parameters(
            {"normalization.alpha_ew": _HISTORICAL_REFERENCE_ALPHA_EW}
        )
    return runtime


def _reference_case(case_id: str) -> dict[str, Any]:
    return next(case for case in _REFERENCE_PAYLOAD["cases"] if case["id"] == case_id)


def _ufo_model_for(
    model_source: str,
    request: pytest.FixtureRequest,
) -> CompiledModel | None:
    if model_source == "built-in":
        return None
    model = request.getfixturevalue("ufo_sm_recurrence_jit_o2_model")
    assert isinstance(model, CompiledModel)
    return model


def _reference_point(point_id: str) -> _Point:
    point = next(
        item for item in _REFERENCE_PAYLOAD["points"] if item["id"] == point_id
    )
    return tuple(
        tuple(float(component) for component in row) for row in point["momenta"]
    )


def _tracked_lc_reference(
    case_id: str,
) -> tuple[_Point, dict[str, float], float]:
    case = _reference_case(case_id)
    observation = case["observations"][0]
    point = _reference_point(observation["point_id"])
    color_ids = tuple(item["id"] for item in case["axes"]["colors"])
    expected_by_flow = {
        color_id: sum(float(row[color_index]) for row in observation["values"])
        for color_index, color_id in enumerate(color_ids)
    }
    return point, expected_by_flow, float(observation["total"])


def _helicity_sum_by_flow(
    runtime: Runtime,
    point: _Point,
    *,
    precision: int = 16,
) -> dict[str, float]:
    resolved = runtime.evaluate_resolved((point,), precision=precision)
    return {
        color_id: sum(
            complex(resolved.values[0][helicity_index][color_index]).real
            for helicity_index in range(len(resolved.helicity_ids))
        )
        for color_index, color_id in enumerate(resolved.color_ids)
    }


def _assert_flow_oracle(
    runtime: Runtime,
    point: _Point,
    expected_by_flow: dict[str, float],
    expected_total: float,
    *,
    relative_tolerance: float,
    precision: int = 16,
) -> None:
    actual_by_flow = _helicity_sum_by_flow(runtime, point, precision=precision)
    assert set(actual_by_flow) == set(expected_by_flow)
    assert actual_by_flow == pytest.approx(
        expected_by_flow,
        rel=relative_tolerance,
        abs=1.0e-15,
    )
    assert sum(actual_by_flow.values()) == pytest.approx(
        expected_total,
        rel=relative_tolerance,
        abs=1.0e-15,
    )
    exact_total = complex(runtime.evaluate((point,), precision=precision)[0]).real
    assert exact_total == pytest.approx(
        expected_total,
        rel=relative_tolerance,
        abs=1.0e-15,
    )

    for flow_id, expected in expected_by_flow.items():
        selected = runtime.evaluate_resolved(
            (point,),
            color_flows=(flow_id,),
            precision=precision,
        )
        assert selected.color_ids == (flow_id,)
        assert complex(selected.total()[0]).real == pytest.approx(
            expected,
            rel=relative_tolerance,
            abs=1.0e-15,
        )
        assert complex(
            runtime.evaluate(
                (point,),
                color_flows=(flow_id,),
                precision=precision,
            )[0]
        ).real == pytest.approx(
            expected,
            rel=relative_tolerance,
            abs=1.0e-15,
        )


def _assert_exact_components_match_native(
    runtime: Runtime,
    point: _Point,
) -> None:
    native = runtime.evaluate_resolved((point,))
    exact = runtime.evaluate_resolved((point,), precision=32)
    assert exact.helicity_ids == native.helicity_ids
    assert exact.color_ids == native.color_ids
    assert exact.shape == native.shape
    for native_helicities, exact_helicities in zip(
        native.values,
        exact.values,
        strict=True,
    ):
        for native_colors, exact_colors in zip(
            native_helicities,
            exact_helicities,
            strict=True,
        ):
            assert tuple(complex(value) for value in exact_colors) == pytest.approx(
                tuple(complex(value) for value in native_colors),
                rel=1.0e-12,
                abs=1.0e-15,
            )
    assert tuple(complex(value) for value in exact.total()) == pytest.approx(
        tuple(complex(value) for value in native.total()),
        rel=1.0e-12,
        abs=1.0e-15,
    )


@pytest.mark.parametrize("model_source", ["built-in", "ufo-sm"])
def test_direct_arena_crossing_matches_tracked_component_oracle(
    tmp_path: Path,
    model_source: str,
    request: pytest.FixtureRequest,
) -> None:
    """Lock initial-state crossing through a public resolved component."""

    runtime = _generate_direct_arena_runtime(
        tmp_path=tmp_path,
        model_source=model_source,
        expression=_ZG_EXPRESSION,
        ufo_model=_ufo_model_for(model_source, request),
    )
    case = _reference_case("case:sm_ddbar_zg:lc")
    observation = case["observations"][0]
    point = _reference_point(observation["point_id"])
    helicity_ids = tuple(item["id"] for item in case["axes"]["helicities"])
    color_ids = tuple(item["id"] for item in case["axes"]["colors"])
    helicity_index = helicity_ids.index(_CROSSING_HELICITY_ID)
    color_index = color_ids.index(_CROSSING_FLOW_ID)
    expected = float(observation["values"][helicity_index][color_index])
    relative_tolerance = 5.0e-12 if model_source == "ufo-sm" else 1.0e-12

    complete = runtime.evaluate_resolved((point,))
    assert complete.helicity_ids == helicity_ids
    assert complete.color_ids == color_ids
    assert complex(
        complete.values[0][helicity_index][color_index]
    ).real == pytest.approx(
        expected,
        rel=relative_tolerance,
        abs=1.0e-15,
    )

    selected = runtime.evaluate_resolved(
        (point,),
        helicities=(_CROSSING_HELICITY_ID,),
        color_flows=(_CROSSING_FLOW_ID,),
    )
    assert selected.helicity_ids == (_CROSSING_HELICITY_ID,)
    assert selected.color_ids == (_CROSSING_FLOW_ID,)
    assert complex(selected.values[0][0][0]).real == pytest.approx(
        expected,
        rel=relative_tolerance,
        abs=1.0e-15,
    )
    assert complex(
        runtime.evaluate(
            (point,),
            helicities=(_CROSSING_HELICITY_ID,),
            color_flows=(_CROSSING_FLOW_ID,),
        )[0]
    ).real == pytest.approx(
        expected,
        rel=relative_tolerance,
        abs=1.0e-15,
    )


@pytest.mark.parametrize("model_source", ["built-in", "ufo-sm"])
def test_direct_arena_two_flow_replay_matches_tracked_lc_oracle(
    tmp_path: Path,
    model_source: str,
    request: pytest.FixtureRequest,
) -> None:
    """Replay and select both physical Zgg flows from one public artifact."""

    runtime = _generate_direct_arena_runtime(
        tmp_path=tmp_path,
        model_source=model_source,
        expression=_ZGG_EXPRESSION,
        ufo_model=_ufo_model_for(model_source, request),
    )
    point, expected_by_flow, expected_total = _tracked_lc_reference(
        "case:sm_ddbar_zgg:lc"
    )
    _assert_flow_oracle(
        runtime,
        point,
        expected_by_flow,
        expected_total,
        relative_tolerance=5.0e-12 if model_source == "ufo-sm" else 1.0e-12,
    )
    _assert_flow_oracle(
        runtime,
        point,
        expected_by_flow,
        expected_total,
        relative_tolerance=5.0e-12 if model_source == "ufo-sm" else 1.0e-12,
        precision=32,
    )


@pytest.mark.parametrize("model_source", ["built-in", "ufo-sm"])
def test_direct_arena_multiline_replay_matches_tracked_lc_oracles(
    tmp_path: Path,
    model_source: str,
    request: pytest.FixtureRequest,
) -> None:
    """Cover distinct-flavour and same-flavour closure reconstruction."""

    for expression, case_id in (
        ("d d~ > u u~", "case:sm_ddbar_uubar:lc"),
        ("d d~ > d d~", "case:sm_ddbar_ddbar:lc"),
    ):
        runtime = _generate_direct_arena_runtime(
            tmp_path=tmp_path,
            model_source=model_source,
            expression=expression,
            ufo_model=_ufo_model_for(model_source, request),
        )
        point, expected_by_flow, expected_total = _tracked_lc_reference(case_id)
        _assert_flow_oracle(
            runtime,
            point,
            expected_by_flow,
            expected_total,
            relative_tolerance=(5.0e-12 if model_source == "ufo-sm" else 1.0e-12),
        )


@pytest.mark.parametrize("model_source", ["built-in", "ufo-sm"])
def test_direct_arena_pure_gluon_replay_matches_reflection_oracle(
    tmp_path: Path,
    model_source: str,
    request: pytest.FixtureRequest,
) -> None:
    """Lock reciprocal pure-gluon flow handling against tracked LC components."""

    runtime = _generate_direct_arena_runtime(
        tmp_path=tmp_path,
        model_source=model_source,
        expression=_GG_EXPRESSION,
        ufo_model=_ufo_model_for(model_source, request),
    )
    point, expected_by_flow, expected_total = _tracked_lc_reference("case:sm_gg_gg:lc")
    _assert_flow_oracle(
        runtime,
        point,
        expected_by_flow,
        expected_total,
        relative_tolerance=5.0e-12 if model_source == "ufo-sm" else 1.0e-12,
    )


def test_symmetric_group_complete_gluon_replay_lowers_all_materialized_sectors(
    tmp_path: Path,
) -> None:
    """Lower the complete-helicity FullColour replay that exercises reflection."""

    _require_native_direct_arena()
    artifact = tmp_path / "symmetric-group-complete-gluon-replay"
    config = _generation_config(
        color_accuracy=ColorAccuracy.FULL,
        color_contraction=ColorContraction.SYMMETRIC_GROUP_FFT,
    )
    with packaged_prepared_model_path(BUILTIN_SM_JIT_O2) as prepared_model:
        Generator(config).generate(
            _GG_EXPRESSION,
            artifact,
            model=ModelSource.from_path(prepared_model),
        )

    execution_path = next((artifact / "processes").glob("*/execution.json"))
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    summary = execution["recurrence_summary"]
    schedule = execution["plan"]["inspection_summary"]["schedule"]
    assert summary["lc_flow_layout"] == "contracted-color-union"
    assert schedule["replay_target_count"] == 6
    assert schedule["resolved_helicity_count"] == 3
    assert schedule["amplitude_destination_count"] == 9


@pytest.mark.parametrize("model_source", ["built-in", "ufo-sm"])
@pytest.mark.parametrize(
    ("expression", "case_id"),
    [
        pytest.param(
            _GG_EXPRESSION,
            "case:sm_gg_gg:lc",
            id="pure-gluon-folded-reflection",
        ),
        pytest.param(
            _SAME_FLAVOUR_EXPRESSION,
            "case:sm_ddbar_ddbar:lc",
            id="same-flavour-reconstruction",
        ),
        pytest.param(
            _THREE_LINE_EXPRESSION,
            None,
            id="multiple-open-line-closures",
        ),
    ],
)
def test_direct_arena_all_flow_union_difficult_semantics(
    tmp_path: Path,
    model_source: str,
    expression: str,
    case_id: str | None,
    request: pytest.FixtureRequest,
) -> None:
    """Cover difficult closure semantics through the actual union runtime."""

    runtime = _generate_direct_arena_runtime(
        tmp_path=tmp_path,
        model_source=model_source,
        expression=expression,
        ufo_model=_ufo_model_for(model_source, request),
        lc_flow_layout=LCFlowLayout.ALL_FLOW_UNION,
    )

    if case_id is not None:
        point, expected_by_flow, expected_total = _tracked_lc_reference(case_id)
    else:
        point = (
            (500.0, 0.0, 0.0, 500.0),
            (500.0, 0.0, 0.0, -500.0),
            *massive_rambo_final_state(
                4,
                sqrt_s=1000.0,
                masses=(0.0, 0.0, 0.0, 0.0),
                seed=731,
            ),
        )
        expected_by_flow = _THREE_LINE_HELICITY_SUM_BY_FLOW
        expected_total = _THREE_LINE_HELICITY_SUM_TOTAL

    relative_tolerance = 5.0e-12 if model_source == "ufo-sm" else 1.0e-12
    _assert_flow_oracle(
        runtime,
        point,
        expected_by_flow,
        expected_total,
        relative_tolerance=relative_tolerance,
    )
    _assert_flow_oracle(
        runtime,
        point,
        expected_by_flow,
        expected_total,
        relative_tolerance=relative_tolerance,
        precision=32,
    )
    _assert_exact_components_match_native(runtime, point)

# SPDX-License-Identifier: 0BSD
"""Three-open-quark-line recurrence execution against an AmpliCol oracle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pyamplicol import Generator, ModelSource, Runtime
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationValidationConfig,
    JITConfig,
    RunConfig,
)
from pyamplicol.generation.phase_space import massive_rambo_final_state
from pyamplicol.models.loading import compile_model_source
from pyamplicol.models.prepared_compile import prepare_model_bundle

_EXPRESSION = "d d~ > u u~ s s~"
_LEGACY_HELICITY_SUM_BY_FLOW = {
    "flow:2,1,3,4,5,6": 1.7260373034739047e-11,
    "flow:2,1,3,6,5,4": 1.6570372188601389e-11,
    "flow:2,4,3,1,5,6": 1.0167883928661888e-10,
    "flow:2,4,3,6,5,1": 1.6445040193096446e-10,
    "flow:2,6,3,1,5,4": 6.5052811045824496e-10,
    "flow:2,6,3,4,5,1": 1.2411150983558527e-10,
}
_LEGACY_HELICITY_SUM_TOTAL = 1.0745996067347547e-9
_UFO_SM_ROOT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "pyamplicol"
    / "assets"
    / "models"
    / "json"
    / "sm"
)


def _normalization_factor(payload: dict[str, object]) -> float:
    return (
        float(payload["global_coupling_factor"])
        * float(payload["color_factor"])
        / (float(payload["average_factor"]) * float(payload["identical_factor"]))
    )


def _model_context(model_source: str):
    if model_source == "built-in":
        compiled_model = compile_model_source("built-in-sm", use_cache=True)
        return compiled_model, None

    model_path = _UFO_SM_ROOT / "sm.json"
    restriction_path = _UFO_SM_ROOT / "restrict_default.json"
    compiled_model = compile_model_source(
        model_path,
        restriction=str(restriction_path.resolve()),
        use_cache=True,
    )
    return (
        compiled_model,
        ModelSource.from_path(model_path, restriction=restriction_path),
    )


def _compiled_helicity_sum_by_flow(
    artifact: Path,
    point: tuple[tuple[float, ...], ...],
) -> tuple[dict[str, float], dict[str, object]]:
    resolved = Runtime.load(artifact).evaluate_resolved((point,))
    values = {
        flow_id: float(
            sum(
                resolved.values[0][helicity_index][color_index].real
                for helicity_index in range(len(resolved.helicity_ids))
            )
        )
        for color_index, flow_id in enumerate(resolved.color_ids)
    }
    physics_files = tuple((artifact / "processes").glob("*/physics.json"))
    assert len(physics_files) == 1
    physics = json.loads(physics_files[0].read_text(encoding="utf-8"))
    normalization = physics["extensions"]["normalization"]
    assert isinstance(normalization, dict)
    return values, normalization


def _recurrence_helicity_sum_by_flow(
    *,
    tmp_path: Path,
    model_source: str,
    compiled_model,
    point: tuple[tuple[float, ...], ...],
) -> tuple[dict[str, float], dict[str, object], tuple[str, ...]]:
    evaluator = EvaluatorConfig(
        execution_mode="recurrence",
        optimization=EvaluatorOptimizationConfig(cores=1),
        jit=JITConfig(optimization_level=2),
    )
    prepared = prepare_model_bundle(
        compiled_model,
        tmp_path / f"{model_source}-recurrence-jit-o2.pyamplicol-model",
        evaluator=evaluator,
    )
    artifact = tmp_path / f"{model_source}-recurrence-direct-v2"
    Generator(
        RunConfig(
            action="generate",
            color=ColorConfig(accuracy="lc"),
            generation=GenerationConfig(
                workers=1,
                emit_api_bundle=False,
                validation=GenerationValidationConfig(
                    enabled=False,
                    post_build_validation=False,
                ),
            ),
            evaluator=evaluator,
        )
    ).generate(
        _EXPRESSION,
        artifact,
        model=ModelSource.from_path(prepared.output),
    )

    values, normalization = _compiled_helicity_sum_by_flow(artifact, point)
    return values, normalization, tuple(values)


def test_three_line_topology_replay_matches_amplicol_and_compiled_per_flow(
    tmp_path: Path,
) -> None:
    """Compare both SM frontends to AmpliCol and fresh compiled artifacts."""

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
    generation = GenerationConfig(
        emit_api_bundle=False,
        validation=GenerationValidationConfig(
            enabled=False,
            post_build_validation=False,
        ),
    )
    evaluator = EvaluatorConfig(
        execution_mode="compiled",
        jit=JITConfig(optimization_level=1),
    )
    by_model: dict[str, dict[str, float]] = {}
    mismatches: dict[str, dict[str, dict[str, float]]] = {}

    for model_source in ("built-in", "ufo-sm"):
        compiled_model, public_model = _model_context(model_source)
        artifact = tmp_path / f"{model_source}-compiled-jit-o1"
        Generator(
            RunConfig(
                action="generate",
                color=ColorConfig(accuracy="lc"),
                generation=generation,
                evaluator=evaluator,
            )
        ).generate(_EXPRESSION, artifact, model=public_model)
        compiled, artifact_normalization = _compiled_helicity_sum_by_flow(
            artifact,
            point,
        )
        recurrence, recurrence_normalization, recurrence_flow_ids = (
            _recurrence_helicity_sum_by_flow(
                tmp_path=tmp_path,
                model_source=model_source,
                compiled_model=compiled_model,
                point=point,
            )
        )

        assert tuple(compiled) == recurrence_flow_ids
        for key in (
            "global_coupling_factor",
            "color_factor",
            "average_factor",
            "identical_factor",
        ):
            assert float(artifact_normalization[key]) == float(
                recurrence_normalization[key]
            )
        assert _normalization_factor(artifact_normalization) == (
            _normalization_factor(recurrence_normalization)
        )
        model_mismatches: dict[str, dict[str, float]] = {}
        for flow_id, expected in compiled.items():
            obtained = recurrence[flow_id]
            absolute = abs(obtained - expected)
            relative = absolute / abs(expected) if expected else float("inf")
            if absolute > max(1e-15, 1e-12 * abs(expected)):
                model_mismatches[flow_id] = {
                    "recurrence": obtained,
                    "compiled_jit_o1": expected,
                    "absolute_difference": absolute,
                    "relative_difference": relative,
                }
        if model_mismatches:
            mismatches[model_source] = model_mismatches
        assert recurrence == pytest.approx(
            _LEGACY_HELICITY_SUM_BY_FLOW,
            rel=5e-12,
            abs=1e-15,
        )
        assert sum(recurrence.values()) == pytest.approx(
            _LEGACY_HELICITY_SUM_TOTAL,
            rel=5e-12,
            abs=1e-15,
        )
        by_model[model_source] = recurrence

    assert by_model["ufo-sm"] == pytest.approx(
        by_model["built-in"],
        rel=5e-12,
        abs=1e-15,
    )
    assert not mismatches, json.dumps(mismatches, indent=2, sort_keys=True)

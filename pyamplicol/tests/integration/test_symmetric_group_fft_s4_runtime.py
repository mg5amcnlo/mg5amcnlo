# SPDX-License-Identifier: 0BSD
"""Focused generated-runtime acceptance for the nontrivial S4 FFT channel.

Run this deliberately bounded, but expensive, acceptance under the repository
memory watchdog::

    PYAMPLICOL_RUN_FFT_S4_RUNTIME_ACCEPTANCE=1 \
      .venv/bin/python tools/ci/memory_watchdog.py --limit-gib 20 -- \
      .venv/bin/python -m pytest -q \
      tests/integration/test_symmetric_group_fft_s4_runtime.py

The test is skipped in ordinary suites because it generates direct and FFT
artifacts for an eight-particle process in both native execution lanes.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from pyamplicol import Generator, ModelSource, Runtime
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
    GenerationRelationDiscoveryConfig,
    GenerationValidationConfig,
    JITConfig,
    ProcessConfig,
    RelationDiscoveryMode,
    RunConfig,
)
from pyamplicol.models.builtin.validation import generic_validation_point

_ACCEPTANCE_ENV = "PYAMPLICOL_RUN_FFT_S4_RUNTIME_ACCEPTANCE"
_FFT_CAPABILITY = "rusticol.color-contraction.symmetric-group-fft.v1"
_PROCESS = "d d~ > u u~ g g g g"
_HELICITIES = (-1, 1, -1, 1, -1, 1, -1, 1)
_HELICITY_ID = "h:-1,+1,-1,+1,-1,+1,-1,+1"
_ALPHA_S_PARAMETER = "normalization.alpha_s_me_check"
_UPDATED_ALPHA_S = 0.13

pytestmark = pytest.mark.skipif(
    os.environ.get(_ACCEPTANCE_ENV) != "1",
    reason=f"set {_ACCEPTANCE_ENV}=1 and run under the 20 GiB watchdog",
)


def _require_native_fft_runtime() -> None:
    assert importlib.util.find_spec("symbolica") is not None, (
        "the S4 FFT runtime acceptance requires Symbolica"
    )
    assert importlib.util.find_spec("pyamplicol._rusticol") is not None, (
        "the S4 FFT runtime acceptance requires the native Rusticol extension"
    )
    rusticol = importlib.import_module("pyamplicol._rusticol")
    assert hasattr(rusticol, "_lower_recurrence_direct_v2"), (
        "the installed Rusticol extension lacks recurrence Direct-Arena v2"
    )
    assert hasattr(rusticol, "_build_on_the_fly_process_seeds_v1"), (
        "the installed Rusticol extension lacks on-the-fly generation"
    )


@pytest.fixture(scope="module")
def prepared_builtin_sm() -> Iterator[ModelSource]:
    _require_native_fft_runtime()
    override = os.environ.get("PYAMPLICOL_RECURRENCE_TEST_PREPARED_MODEL")
    if override:
        yield ModelSource.from_path(Path(override))
        return
    with packaged_prepared_model_path(BUILTIN_SM_JIT_O2) as prepared_model:
        yield ModelSource.from_path(prepared_model)


def _generation_config(
    lane: EvaluatorExecutionMode,
    contraction: ColorContraction,
) -> RunConfig:
    selected = (
        {str(label): helicity for label, helicity in enumerate(_HELICITIES, start=1)}
        if lane is EvaluatorExecutionMode.RECURRENCE
        else {}
    )
    return RunConfig(
        action=Action.GENERATE,
        process=ProcessConfig(selected_source_helicities=selected),
        color=ColorConfig(
            accuracy=ColorAccuracy.FULL,
            contraction=contraction,
        ),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
            validation=GenerationValidationConfig(
                enabled=False,
                post_build_validation=False,
            ),
            relation_discovery=GenerationRelationDiscoveryConfig(
                mode=RelationDiscoveryMode.OFF
            ),
        ),
        evaluator=EvaluatorConfig(
            execution_mode=lane,
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(optimization_level=2),
        ),
    )


def _point() -> tuple[tuple[tuple[float, ...], ...], ...]:
    return (
        tuple(
            tuple(float(component) for component in particle.momentum)
            for particle in generic_validation_point(_PROCESS, seed=101)
        ),
    )


def _assert_s4_fft_payload(artifact: Path, lane: EvaluatorExecutionMode) -> None:
    execution_paths = tuple((artifact / "processes").glob("*/execution.json"))
    assert len(execution_paths) == 1
    execution = json.loads(execution_paths[0].read_text(encoding="utf-8"))
    expected_kind = (
        "pyamplicol-runtime-recurrence-execution"
        if lane is EvaluatorExecutionMode.RECURRENCE
        else "pyamplicol-runtime-on-the-fly-execution"
    )
    assert execution["kind"] == expected_kind
    assert _FFT_CAPABILITY in execution["required_runtime_capabilities"]

    color = execution["runtime_metadata"]["color_contraction"]
    assert color["storage"] == "convolution-kernels"
    assert color["factorization"] == {
        "kind": "symmetric-group-fourier",
        "rank": 4,
        "coset_count": 10,
    }
    assert color["sector_count"] == 480
    assert color["group_count"] == 240
    assert color["fft_provenance"] == {
        "method": "symmetric-group-fourier",
        "degree": 4,
        "channel_count": 10,
        "covered_local_group_count": 240,
        "residual_group_count": 0,
        "residual_entry_count": 0,
        "raw_kernel_bytes": 21_120,
        "transformed_kernel_bytes": 10_560,
        "capability": _FFT_CAPABILITY,
    }


def _evaluate_selected(
    runtime: Runtime,
    point: tuple[tuple[tuple[float, ...], ...], ...],
) -> tuple[complex, ...]:
    return tuple(
        complex(value)
        for value in runtime.evaluate(
            point,
            helicities=(_HELICITY_ID,),
        )
    )


def _assert_repeated_warm_up_is_local(
    runtime: Runtime,
    isolated_runtime: Runtime,
    point: tuple[tuple[tuple[float, ...], ...], ...],
) -> None:
    first = runtime.warm_up(point, helicities=(_HELICITY_ID,))
    repeated = runtime.warm_up(point, helicities=(_HELICITY_ID,))
    isolated_first = isolated_runtime.warm_up(point, helicities=(_HELICITY_ID,))

    assert first.query_count > 0
    assert first.warmed_query_count > 0
    assert first.already_warm is False
    assert repeated.query_count == first.query_count
    assert repeated.warmed_query_count == 0
    assert repeated.already_warm is True
    assert isolated_first.query_count == first.query_count
    assert isolated_first.warmed_query_count > 0
    assert isolated_first.already_warm is False


@pytest.mark.parametrize(
    "lane",
    (
        EvaluatorExecutionMode.RECURRENCE,
        EvaluatorExecutionMode.ON_THE_FLY,
    ),
)
def test_generated_s4_fft_runtime_matches_direct_and_keeps_instances_independent(
    tmp_path: Path,
    prepared_builtin_sm: ModelSource,
    lane: EvaluatorExecutionMode,
) -> None:
    direct_artifact = tmp_path / f"{lane.value}-direct"
    fft_artifact = tmp_path / f"{lane.value}-fft"
    for artifact, contraction in (
        (direct_artifact, ColorContraction.DIRECT),
        (fft_artifact, ColorContraction.SYMMETRIC_GROUP_FFT),
    ):
        Generator(_generation_config(lane, contraction)).generate(
            _PROCESS,
            artifact,
            model=prepared_builtin_sm,
        )

    _assert_s4_fft_payload(fft_artifact, lane)
    point = _point()
    direct = Runtime.load(direct_artifact)
    fft = Runtime.load(fft_artifact)
    isolated_fft = Runtime.load(fft_artifact)
    runtimes = (direct, fft, isolated_fft)
    try:
        if lane is EvaluatorExecutionMode.ON_THE_FLY:
            _assert_repeated_warm_up_is_local(fft, isolated_fft, point)
            direct.warm_up(point, helicities=(_HELICITY_ID,))

        direct_default = _evaluate_selected(direct, point)
        fft_default = _evaluate_selected(fft, point)
        isolated_default = _evaluate_selected(isolated_fft, point)
        assert fft_default == pytest.approx(
            direct_default,
            rel=1.0e-10,
            abs=1.0e-18,
        )
        assert isolated_default == pytest.approx(
            fft_default,
            rel=1.0e-13,
            abs=1.0e-20,
        )
        assert any(abs(value) > 0.0 for value in fft_default)

        direct.set_model_parameters({_ALPHA_S_PARAMETER: _UPDATED_ALPHA_S})
        fft.set_model_parameters({_ALPHA_S_PARAMETER: _UPDATED_ALPHA_S})
        direct_updated = _evaluate_selected(direct, point)
        fft_updated = _evaluate_selected(fft, point)
        assert fft_updated == pytest.approx(
            direct_updated,
            rel=1.0e-10,
            abs=1.0e-18,
        )
        assert fft_updated == pytest.approx(
            _evaluate_selected(fft, point),
            rel=1.0e-13,
            abs=1.0e-20,
        )
        assert fft_updated != pytest.approx(
            fft_default,
            rel=1.0e-8,
            abs=1.0e-20,
        )

        # The second handle must retain both its default parameters and its own
        # warmed-family state after the first handle is changed and evaluated.
        assert _evaluate_selected(isolated_fft, point) == pytest.approx(
            isolated_default,
            rel=1.0e-13,
            abs=1.0e-20,
        )
    finally:
        for runtime in reversed(runtimes):
            runtime.clear()

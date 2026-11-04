# SPDX-License-Identifier: 0BSD
"""Physical precision regression, including the full-colour normalization."""

from __future__ import annotations

import importlib.util
from decimal import Decimal, localcontext

import pytest

from pyamplicol import Generator, ModelSource, Runtime
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationRelationDiscoveryConfig,
    GenerationValidationConfig,
    JITConfig,
    RunConfig,
)


def _exact_frames():
    with localcontext() as context:
        context.prec = 120
        point = [
            [Decimal(x) for x in p]
            for p in (
                (500, 0, 0, 500),
                (500, 0, 0, -500),
                (500, 300, 0, 400),
                (500, -300, 0, -400),
            )
        ]
        axis = [Decimal(x) / 7 for x in (2, 3, 6)]
        beta, gamma = Decimal(3) / 5, Decimal(5) / 4
        boosted = []
        for p in point:
            longitudinal = sum(
                (x * n for x, n in zip(p[1:], axis, strict=True)), Decimal(0)
            )
            shift = (gamma - 1) * longitudinal + gamma * beta * p[0]
            boosted.append(
                [gamma * (p[0] + beta * longitudinal)]
                + [x + shift * n for x, n in zip(p[1:], axis, strict=True)]
            )
        return [point, boosted]


@pytest.mark.parametrize("execution_mode", ("recurrence", "eager", "compiled"))
@pytest.mark.parametrize("color_accuracy", ("lc", "full"))
def test_gg_helicity_sum_has_consistent_normalization_and_precision(
    tmp_path,
    execution_mode,
    color_accuracy,
):
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")
    config = RunConfig(
        action="generate",
        color=ColorConfig(accuracy=color_accuracy),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
            relation_discovery=GenerationRelationDiscoveryConfig(mode="off"),
            validation=GenerationValidationConfig(
                enabled=False, post_build_validation=False
            ),
        ),
        evaluator=EvaluatorConfig(
            execution_mode=execution_mode,
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(optimization_level=2),
        ),
    )
    output = tmp_path / "gg"
    Generator(config).generate("g g > g g", output, model=ModelSource.built_in_sm())
    runtime = Runtime.load(
        output,
        model_parameters={"normalization.alpha_s_me_check": 0.125},
    )
    frames = _exact_frames()
    native_frames = [[[float(x) for x in p] for p in frame] for frame in frames]
    native = runtime.evaluate(native_frames)
    separate = [runtime.evaluate([frame])[0] for frame in native_frames]
    assert native == pytest.approx(separate, rel=3e-13, abs=0)
    exact = runtime.evaluate(frames, precision=80)
    with localcontext() as context:
        context.prec = 90
        assert exact[0] > 0
        assert abs(exact[0] - exact[1]) / abs(exact[0]) < Decimal("1e-70")
        # In particular, full colour must not acquire the extra LC factor81.
        for value, reference in zip(exact, native, strict=True):
            assert float(value) == pytest.approx(reference.real, rel=3e-13, abs=0)

    if execution_mode == "eager" and color_accuracy == "full":
        # Exercise both the ordinary and profiled topology-replay gathers.
        # Repeated points exposed an amplitude/point stride mix-up even when
        # each separate evaluation and the arbitrary-precision path agreed.
        for point_count in (2, 3, 65):
            batch = [native_frames[(index // 2) % 2] for index in range(point_count)]
            expected = [separate[(index // 2) % 2] for index in range(point_count)]
            assert runtime.evaluate(batch) == pytest.approx(expected, rel=3e-13, abs=0)
            resolved = runtime.evaluate_resolved(batch)
            totals = [
                sum(value for helicity in point for value in helicity)
                for point in resolved.values
            ]
            assert totals == pytest.approx(expected, rel=3e-13, abs=0)
            profiled = runtime._backend.profile(batch, include_values=True)
            assert profiled["values"] == pytest.approx(expected, rel=3e-13, abs=0)

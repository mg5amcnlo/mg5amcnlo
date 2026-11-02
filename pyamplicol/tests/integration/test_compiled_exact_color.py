# SPDX-License-Identifier: 0BSD
"""Compare compiled/eager colour contractions with independent exact recurrence."""

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


@pytest.mark.parametrize(
    "process,point",
    (
        (
            "g g > g g",
            (
                (500, 0, 0, 500),
                (500, 0, 0, -500),
                (500, 300, 0, 400),
                (500, -300, 0, -400),
            ),
        ),
        (
            "g g > g g g g",
            (
                (500, 0, 0, 500),
                (500, 0, 0, -500),
                (250, 150, 0, 200),
                (250, -150, 0, -200),
                (250, 0, 200, 150),
                (250, 0, -200, -150),
            ),
        ),
    ),
)
def test_compiled_full_colour_matches_recurrence_beyond_binary64(
    tmp_path, process, point
):
    values = []
    for mode in ("recurrence", "compiled", "eager"):
        config = RunConfig(
            action="generate",
            color=ColorConfig(accuracy="full", contraction="direct"),
            generation=GenerationConfig(
                workers=1,
                emit_api_bundle=False,
                relation_discovery=GenerationRelationDiscoveryConfig(mode="off"),
                validation=GenerationValidationConfig(
                    enabled=False, post_build_validation=False
                ),
            ),
            evaluator=EvaluatorConfig(
                execution_mode=mode,
                optimization=EvaluatorOptimizationConfig(cores=1),
                jit=JITConfig(optimization_level=2),
            ),
        )
        output = tmp_path / mode
        Generator(config).generate(process, output, model=ModelSource.built_in_sm())
        runtime = Runtime.load(
            output, model_parameters={"normalization.alpha_s_me_check": 0.125}
        )
        values.append(runtime.evaluate([[list(p) for p in point]], precision=80)[0])
    with localcontext() as context:
        context.prec = 90
        assert values[0] > 0
        for value in values[1:]:
            assert abs(values[0] - value) / abs(values[0]) < Decimal("1e-70")

# SPDX-License-Identifier: 0BSD
"""Inherited LC/NLC Born results from independently generated ordinary artifacts."""

from decimal import Decimal, localcontext

import pytest

from pyamplicol import (
    ColorCorrelator,
    CorrelatorConfig,
    Generator,
    ModelSource,
    Runtime,
)
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationValidationConfig,
    JITConfig,
    RunConfig,
)

_POINT = (
    (500, 0, 0, 500),
    (500, 0, 0, -500),
    (500, 300, 0, 400),
    (500, -300, 0, -400),
)
_THREE_BODY_POINT = (
    (900, 0, 0, 900),
    (900, 0, 0, -900),
    (500, 400, 300, 0),
    (500, 400, -300, 0),
    (800, -800, 0, 0),
)


def _configuration(accuracy):
    return RunConfig(
        action="generate",
        color=ColorConfig(accuracy=accuracy, contraction="direct"),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
            validation=GenerationValidationConfig(
                enabled=False, post_build_validation=False
            ),
        ),
        evaluator=EvaluatorConfig(
            execution_mode="compiled",
            optimization=EvaluatorOptimizationConfig(cores=1, horner_iterations=1),
            jit=JITConfig(optimization_level=0),
        ),
    )


@pytest.mark.parametrize("accuracy", ("lc", "nlc"))
@pytest.mark.parametrize(
    ("expression", "point"),
    (
        ("g g > g g", _POINT),
        ("d d~ > g g", _POINT),
        ("d d~ > u u~", _POINT),
        ("d d~ > d d~", _POINT),
        ("d d~ > u u~ g", _THREE_BODY_POINT),
        ("d d~ > d d~ g", _THREE_BODY_POINT),
    ),
    ids=(
        "gluons",
        "one-quark-pair",
        "two-quark-pairs",
        "identical-quarks",
        "two-pairs-and-gluon",
        "identical-pairs-and-gluon",
    ),
)
def test_generated_correlated_born_matches_inherited_accuracy(
    tmp_path, expression, point, accuracy
):
    pytest.importorskip("pyamplicol._rusticol")
    config = _configuration(accuracy)
    model = ModelSource.built_in_sm()
    ordinary_path = tmp_path / "ordinary"
    correlated_path = tmp_path / "correlated"
    Generator(config).generate(expression, ordinary_path, model=model)
    Generator(config).generate(
        expression,
        correlated_path,
        model=model,
        correlators=CorrelatorConfig(
            color_correlations=(ColorCorrelator.dipole("T11", 1, 1),)
        ),
    )
    ordinary = Runtime.load(ordinary_path).evaluate((point,), precision=36)[0]
    correlated = Runtime.load(correlated_path).evaluate_correlated(
        (point,), precision=32
    )[0]
    with localcontext() as context:
        context.prec = 40
        assert ordinary > 0
        assert abs(correlated.real / ordinary - 1) < Decimal("1e-27")
        assert abs(correlated.imag / ordinary) < Decimal("1e-27")

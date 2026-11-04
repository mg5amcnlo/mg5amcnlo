# SPDX-License-Identifier: 0BSD
"""Heavy real-A regression for generic-DAG current-output isolation."""

from __future__ import annotations

import os
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from pyamplicol import Generator
from pyamplicol.api.errors import GenerationError
from pyamplicol.config import (
    Action,
    ColorAccuracy,
    ColorConfig,
    EvaluatorConfig,
    GenerationConfig,
    GenerationValidationConfig,
    RunConfig,
)
from pyamplicol.generation import numerical_current_warmup

_ACCEPTANCE_ENV = "PYAMPLICOL_RUN_GENERIC_DAG_REAL_A_REGRESSION"
_PROCESS = "g g > t t~ g g g"

pytestmark = pytest.mark.skipif(
    os.environ.get(_ACCEPTANCE_ENV) != "1",
    reason=f"set {_ACCEPTANCE_ENV}=1 to run the heavy real-A regression",
)


class _RealAValidated(RuntimeError):
    pass


def test_nlc_helicity_sum_application_is_decimal_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_apply = (
        numerical_current_warmup.apply_numerical_current_relation_certificates
    )
    real_validate = (
        numerical_current_warmup._validate_applied_current_observations
    )
    applications: list[Any] = []
    validations: list[tuple[Any, Any, dict[str, object]]] = []

    def apply(*args: Any, **kwargs: Any) -> Any:
        result = real_apply(*args, **kwargs)
        if len(result.dag.currents) == 21_434:
            applications.append(result)
        return result

    def validate(reference: Any, applied: Any, **kwargs: Any) -> Any:
        result = real_validate(reference, applied, **kwargs)
        if len(reference.observations) == 21_434:
            validations.append((reference, applied, result))
            raise _RealAValidated
        return result

    monkeypatch.setattr(
        numerical_current_warmup,
        "apply_numerical_current_relation_certificates",
        apply,
    )
    monkeypatch.setattr(
        numerical_current_warmup,
        "_validate_applied_current_observations",
        validate,
    )

    config = RunConfig(
        action=Action.GENERATE,
        color=ColorConfig(accuracy=ColorAccuracy.NLC),
        evaluator=EvaluatorConfig(execution_mode="compiled"),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
            validation=GenerationValidationConfig(samples=1),
        ),
    )
    with pytest.raises(GenerationError) as captured:
        Generator(config).generate(_PROCESS, tmp_path / "real-a")

    assert isinstance(captured.value.__cause__, _RealAValidated)
    assert len(applications) == len(validations) == 1
    application = applications[0]
    reference, applied, validation = validations[0]
    assert tuple(
        certificate.current_id
        for certificate in application.report.certificates
    ) == (15, 16, 21, 23)
    assert all(
        certificate.relation_kind == "zero"
        for certificate in application.report.certificates
    )
    assert application.report.applied_relation_count == 4
    assert application.report.interaction_evaluation_count_before == 74_260
    assert application.report.interaction_evaluation_count_projected == 73_996
    assert validation["checked_current_count"] == 21_434
    assert validation["checked_component_count"] == 362_784
    assert validation["maximum_absolute_residual"] == "0"
    assert validation["maximum_relative_residual"] == "0"
    assert validation["maximum_tolerance_ratio"] == "0"
    assert (
        reference.observation_batch_sha256
        == applied.observation_batch_sha256
    )
    assert reference.observations[202][1] == applied.observations[202][1]
    assert Decimal(str(validation["maximum_tolerance_ratio"])) <= Decimal(1)

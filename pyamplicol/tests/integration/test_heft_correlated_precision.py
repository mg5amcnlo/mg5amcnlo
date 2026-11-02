# SPDX-License-Identifier: 0BSD
"""HEFT contact currents retain gauge invariance in all arithmetic modes."""

from __future__ import annotations

from decimal import Decimal, localcontext

import pytest

from pyamplicol import (
    CorrelatedRequest,
    CorrelatorConfig,
    Generator,
    ModelSource,
    Runtime,
)
from pyamplicol.api.models import _compiled_model_payload
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationValidationConfig,
    JITConfig,
    ProcessConfig,
    RunConfig,
)


def test_reduced_heft_contacts_preserve_grouped_spin_ward_and_precision(tmp_path):
    pytest.importorskip("pyamplicol._rusticol")
    # H at rest and two back-to-back gluons. All entries are exact dyadics,
    # including the 3/5--4/5 gluon direction, with MH=125 and sqrt(s)=500.
    point = tuple(
        tuple(Decimal(component) for component in momentum)
        for momentum in (
            ("250", "0", "0", "250"),
            ("250", "0", "0", "-250"),
            ("125", "0", "0", "0"),
            ("187.5", "112.5", "0", "150"),
            ("187.5", "-112.5", "0", "-150"),
        )
    )
    points = (point, point)
    # Compile model IR afresh, rather than silently testing an older shipped
    # bundle: this is the smallest process containing the Hgggg contact.
    model = ModelSource.built_in_sm_heft().compile(use_cache=False)
    assert any(
        particle.component_dimension == 6
        and (particle.auxiliary_kind or "").startswith("ufo-heft-contact-tree:")
        for particle in _compiled_model_payload(model).ir.particles
    )
    output = tmp_path / "heft-correlated"
    Generator(
        RunConfig(
            action="generate",
            color=ColorConfig(accuracy="full", contraction="direct"),
            process=ProcessConfig(
                coupling_order_policy="explicit", max_coupling_orders={"HIG": 1}
            ),
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
    ).generate(
        "g g > H g g",
        output,
        model=model,
        correlators=CorrelatorConfig(spin_correlations=((1,), (4,), (1, 4))),
    )
    runtime = Runtime.load(output)
    requests = {
        "born": CorrelatedRequest(spin_vectors={}),
        "transverse": CorrelatedRequest(spin_vectors={4: (0, 0, 1, 0)}),
        "scaled": CorrelatedRequest(spin_vectors={4: (0, 0, Decimal(2), 0)}),
        "ward-incoming": CorrelatedRequest(spin_vectors={1: point[0]}),
        "ward-outgoing": CorrelatedRequest(spin_vectors={4: point[3]}),
        "ward-joint": CorrelatedRequest(spin_vectors={1: point[0], 4: point[3]}),
    }
    oracle = runtime.evaluate_correlated_many(points, requests, precision=60)
    ordinary = runtime.evaluate(points, precision=60)
    double = runtime.evaluate_correlated_many(points, requests, precision=16)
    double_double = runtime.evaluate_correlated_many(
        points, requests, arithmetic="double-double", precision=31
    )
    with localcontext() as context:
        context.prec = 80
        for index in range(len(points)):
            born = oracle["born"][index].real
            assert born > 0
            assert abs(ordinary[index] / born - 1) < Decimal("1e-50")
            for values, accuracy, ward_accuracy in (
                (double, Decimal("1e-11"), Decimal("1e-20")),
                (double_double, Decimal("1e-25"), Decimal("1e-45")),
                (oracle, Decimal("1e-50"), Decimal("1e-90")),
            ):
                assert tuple(values) == tuple(requests)
                for name in ("born", "transverse", "scaled"):
                    value = Decimal(values[name][index].real)
                    expected = oracle[name][index].real
                    assert expected > 0
                    assert abs(value / expected - 1) < accuracy
                    assert abs(Decimal(values[name][index].imag) / expected) < accuracy
                transverse = Decimal(values["transverse"][index].real)
                assert (
                    abs(Decimal(values["scaled"][index].real) / transverse - 4)
                    < accuracy
                )
                for name, momentum_scale in (
                    ("ward-incoming", point[0][0]),
                    ("ward-outgoing", point[3][0]),
                    ("ward-joint", point[0][0] * point[3][0]),
                ):
                    # Ward amplitudes vanish: normalize their squared values
                    # to Born times the dimensions of the substituted momenta.
                    scale = born * momentum_scale**2
                    value = values[name][index]
                    assert abs(Decimal(value.real) / scale) < ward_accuracy
                    assert abs(Decimal(value.imag) / scale) < ward_accuracy

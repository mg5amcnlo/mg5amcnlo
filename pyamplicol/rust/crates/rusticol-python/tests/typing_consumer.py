# SPDX-License-Identifier: 0BSD

from collections.abc import Mapping
from typing import assert_type

from pyamplicol._rusticol import (
    ColorFlow,
    ContractedColorComponent,
    ExternalParticle,
    HelicityConfiguration,
    ModelParameter,
    ProcessPhysics,
    ResolvedEvaluation,
    Runtime,
)


def consume_runtime(runtime: Runtime) -> None:
    momenta = [[[10.0, 0.0, 0.0, 10.0]] * 3]

    assert_type(runtime.artifact_id, str)
    assert_type(runtime.evaluate(momenta), list[float])
    assert_type(runtime.evaluate(momenta, precision=16), list[float])
    assert_type(runtime.profile(momenta, precision=16), Mapping[str, object])
    assert_type(runtime.evaluate_profile(momenta, precision=16), Mapping[str, object])
    assert_type(runtime.evaluate_resolved(momenta, precision=16), ResolvedEvaluation)

    physics = assert_type(runtime.physics, ProcessPhysics)
    assert_type(physics.external_particles, list[ExternalParticle])
    assert_type(physics.helicities, list[HelicityConfiguration])
    assert_type(physics.color_flows, list[ColorFlow])
    assert_type(physics.contracted_color_components, list[ContractedColorComponent])
    assert_type(physics.model_parameters, list[ModelParameter])

    resolved = runtime.evaluate_resolved(
        momenta,
        helicities=["h0"],
        color_flows=["c0"],
        precision=16,
    )
    assert_type(resolved.values, list[list[list[float]]])
    assert_type(resolved.total(), list[float])

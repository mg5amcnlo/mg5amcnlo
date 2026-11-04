# SPDX-License-Identifier: 0BSD
"""Static consumer of the installed native-extension stub."""

from __future__ import annotations

from pathlib import Path
from typing import assert_type

import pyamplicol._rusticol as rusticol

MOMENTA: rusticol.Momenta = (((10.0, 0.0, 0.0, 10.0),),)


def exercise_native_runtime(artifact: Path) -> None:
    runtime = rusticol.Runtime.load(
        artifact,
        process="ddbar_z",
        model_parameters={"normalization.alpha_s_me_check": 0.118},
        mute_warnings=True,
    )
    assert_type(runtime, rusticol.Runtime)
    assert_type(runtime.physics, rusticol.ProcessPhysics)
    assert_type(runtime.evaluate(MOMENTA), list[float])
    assert_type(runtime.evaluate(MOMENTA, precision=16), list[float])
    resolved = runtime.evaluate_resolved(MOMENTA, precision=16)
    assert_type(resolved, rusticol.ResolvedEvaluation)
    assert_type(resolved.values, list[list[list[float]]])
    assert_type(resolved.total(), list[float])
    assert_type(runtime.physics.external_particles, list[rusticol.ExternalParticle])
    assert_type(
        runtime.physics.helicities,
        list[rusticol.HelicityConfiguration],
    )
    assert_type(runtime.physics.color_flows, list[rusticol.ColorFlow])
    assert_type(
        runtime.physics.contracted_color_components,
        list[rusticol.ContractedColorComponent],
    )
    assert_type(runtime.physics.model_parameters, list[rusticol.ModelParameter])
    runtime.set_model_parameters({"normalization.alpha_s_me_check": 0.119})
    runtime.set_model_parameter("normalization.alpha_s_me_check", 0.120)
    assert_type(runtime.clear(), None)
    assert_type(runtime.take_warnings(), list[str])


def exercise_native_otf_cache(artifact: Path, cache: Path) -> None:
    runtime = rusticol.Runtime.load(artifact)
    assert_type(runtime._on_the_fly_warm_up_f64_json(MOMENTA, n_cores=4), str)
    assert_type(runtime.save(cache), None)
    assert_type(runtime.load_cache(str(cache)), None)


def exercise_native_metadata() -> None:
    assert_type(rusticol.abi_version(), int)
    assert_type(rusticol.package_version(), str)
    target = rusticol.target_info()
    assert_type(target, rusticol.TargetInfo)
    assert_type(target.triple, str)
    assert_type(target.cpu_features, list[str])

# SPDX-License-Identifier: 0BSD
"""Small-process cache round trips through the public Python runtime API."""

from __future__ import annotations

import importlib
import importlib.util
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from pyamplicol import Generator, ModelSource, Runtime
from pyamplicol.api.errors import ArtifactError, CompatibilityError
from pyamplicol.assets.prepared_models import (
    BUILTIN_SM_JIT_O2,
    packaged_prepared_model_path,
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
from pyamplicol.models.builtin.validation import generic_validation_point

_PROCESS = "g g > g g"
_HELICITY = "h:-1,+1,-1,+1"
_OTHER_HELICITY = "h:+1,-1,+1,-1"
_ALPHA_S = "normalization.alpha_s_me_check"


def _require_native_cache() -> None:
    reason = "the native Rusticol extension with OTF cache persistence is required"
    if importlib.util.find_spec("pyamplicol._rusticol") is not None:
        native = importlib.import_module("pyamplicol._rusticol")
        if all(hasattr(native.Runtime, name) for name in ("save", "load_cache")):
            return
    if os.environ.get("PYAMPLICOL_REQUIRE_NATIVE_TESTS") == "1":
        pytest.fail(reason)
    pytest.skip(reason)


@pytest.fixture(scope="module")
def cache_artifacts(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[dict[str, Path]]:
    _require_native_cache()
    output = tmp_path_factory.mktemp("otf-cache")
    paths: dict[str, Path] = {}
    with packaged_prepared_model_path(BUILTIN_SM_JIT_O2) as prepared:
        for name, accuracy, contraction in (
            ("lc", "lc", "direct"),
            ("nlc", "nlc", "direct"),
            ("full", "full", "direct"),
            ("fft", "full", "symmetric-group-fft"),
        ):
            path = output / name
            config = RunConfig(
                action="generate",
                color=ColorConfig(accuracy=accuracy, contraction=contraction),
                generation=GenerationConfig(
                    workers=1,
                    emit_api_bundle=False,
                    validation=GenerationValidationConfig(
                        enabled=False, post_build_validation=False
                    ),
                ),
                evaluator=EvaluatorConfig(
                    execution_mode="on-the-fly",
                    optimization=EvaluatorOptimizationConfig(cores=1),
                    jit=JITConfig(optimization_level=2),
                ),
            )
            Generator(config).generate(
                _PROCESS, path, model=ModelSource.from_path(prepared)
            )
            paths[name] = path
    yield paths


def _point(seed: int) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(float(component) for component in particle.momentum)
        for particle in generic_validation_point(_PROCESS, seed=seed)
    )


def _assert_warm(runtime: Runtime, selectors: dict[str, Any]) -> None:
    warm = runtime.warm_up((_point(211),), **selectors)
    assert warm.already_warm is True
    assert warm.warmed_query_count == 0
    assert warm.query_count > 0


@pytest.mark.parametrize(
    ("kind", "selection"),
    (
        ("lc", "selected"),
        ("lc", "helicity-sum"),
        ("lc", "all"),
        ("nlc", "selected"),
        ("full", "all"),
        ("fft", "all"),
    ),
)
def test_completed_evaluation_cache_restores_without_query_reconstruction(
    cache_artifacts: dict[str, Path], tmp_path: Path, kind: str, selection: str
) -> None:
    artifact = cache_artifacts[kind]
    runtime = Runtime.load(artifact)
    selectors: dict[str, Any] = {}
    if selection == "selected":
        selectors["helicities"] = (_HELICITY,)
    if kind == "lc" and selection != "all":
        selectors["color_flows"] = (runtime.physics.color_flows[0].id,)

    # An ordinary evaluate call, not only explicit warm_up(), must be saveable.
    expected = runtime.evaluate((_point(101),), **selectors)
    assert abs(complex(expected[0])) > 0
    cache = tmp_path / "completed recursion.cache"
    assert runtime.save(cache) is None
    restored = Runtime.load(artifact)
    assert restored.load_cache(cache) is None
    _assert_warm(restored, selectors)
    assert restored.evaluate((_point(101),), **selectors) == pytest.approx(expected)

    # No momentum or SIMD-batch workspace from the saved call may leak in.
    batch = (_point(311), _point(419), _point(521))
    assert restored.evaluate(batch, **selectors) == pytest.approx(
        runtime.evaluate(batch, **selectors), rel=2e-12
    )

    # Restoring structural state must not overwrite the receiving parameters.
    changed = Runtime.load(artifact, model_parameters={_ALPHA_S: 0.13})
    changed.load_cache(cache)
    reference = Runtime.load(artifact, model_parameters={_ALPHA_S: 0.13})
    _assert_warm(changed, selectors)
    assert changed.evaluate(batch, **selectors) == pytest.approx(
        reference.evaluate(batch, **selectors), rel=2e-12
    )
    assert changed.evaluate((_point(101),), **selectors) != pytest.approx(expected)

    # An unretained selector family still takes the ordinary cold path, and
    # its completed replacement can itself be saved and restored.
    alternative = dict(selectors, helicities=(_OTHER_HELICITY,))
    alternate_expected = runtime.evaluate(batch, **alternative)
    assert restored.evaluate(batch, **alternative) == pytest.approx(
        alternate_expected, rel=2e-12
    )
    restored.save(cache)
    changed.load_cache(cache)
    _assert_warm(changed, alternative)


def test_empty_cache_round_trip_and_failed_restore_are_atomic(
    cache_artifacts: dict[str, Path], tmp_path: Path
) -> None:
    artifact = cache_artifacts["lc"]
    runtime = Runtime.load(artifact)
    cache = tmp_path / "empty.cache"
    runtime.save(cache)
    restored = Runtime.load(artifact)
    restored.load_cache(cache)
    assert (
        restored.inspect()["on_the_fly_state"] == runtime.inspect()["on_the_fly_state"]
    )

    selectors = {
        "helicities": (_HELICITY,),
        "color_flows": (runtime.physics.color_flows[0].id,),
    }
    expected = restored.evaluate((_point(101),), **selectors)
    bad = tmp_path / "broken.cache"
    bad.write_bytes(b"not a cache")
    with pytest.raises(ArtifactError):
        restored.load_cache(bad)
    _assert_warm(restored, selectors)
    assert restored.evaluate((_point(101),), **selectors) == pytest.approx(expected)

    wrong = Runtime.load(cache_artifacts["full"])
    wrong.save(cache)
    with pytest.raises((ArtifactError, CompatibilityError)):
        restored.load_cache(cache)
    _assert_warm(restored, selectors)
    assert restored.evaluate((_point(101),), **selectors) == pytest.approx(expected)


@pytest.mark.parametrize("entry_point", ("resolved", "per-point", "zero"))
def test_other_completed_evaluation_states_are_saveable(
    cache_artifacts: dict[str, Path], tmp_path: Path, entry_point: str
) -> None:
    artifact = cache_artifacts["lc"]
    runtime = Runtime.load(artifact)
    flow = runtime.physics.color_flows[0].id
    selectors: dict[str, Any] = {
        "helicities": (_HELICITY,),
        "color_flows": (flow,),
    }
    points = (_point(101), _point(211))
    if entry_point == "resolved":
        expected = runtime.evaluate_resolved(points, **selectors).total()
    elif entry_point == "per-point":
        # Per-point selectors may also produce a completed retained family.
        expected = runtime.evaluate(
            points,
            helicity_by_point=(_HELICITY, _HELICITY),
            color_flow_by_point=(flow, flow),
        )
    else:
        # Include a completed zero-current cache, which need not own a JIT
        # executor. Do not assume an incoming/outgoing helicity convention.
        for helicity in runtime.physics.helicities:
            selectors["helicities"] = (helicity.id,)
            expected = runtime.evaluate(points, **selectors)
            if all(abs(complex(value)) < 1e-20 for value in expected):
                break
        else:
            pytest.fail("four-gluon tree amplitudes must include zero helicities")

    cache = tmp_path / f"{entry_point}.cache"
    runtime.save(cache)
    restored = Runtime.load(artifact)
    restored.load_cache(cache)
    _assert_warm(restored, selectors)
    assert restored.evaluate(points, **selectors) == pytest.approx(expected, abs=1e-25)


def test_explicit_warm_up_cache_keeps_updated_parameters_of_a_warm_receiver(
    cache_artifacts: dict[str, Path], tmp_path: Path
) -> None:
    artifact = cache_artifacts["lc"]
    source = Runtime.load(artifact)
    selectors = {"color_flows": (source.physics.color_flows[0].id,)}
    points = (_point(101), _point(211))
    # Explicit LC warm-up uses its streamed construction path, independently
    # of the ordinary evaluate-created snapshots exercised above.
    warm = source.warm_up(points[:1], **selectors)
    assert warm.already_warm is False
    assert warm.warmed_query_count > 0
    cache = tmp_path / "explicit-warm-up.cache"
    source.save(cache)

    receiver = Runtime.load(artifact, model_parameters={_ALPHA_S: 0.13})
    old = receiver.evaluate(points, **selectors)
    # Leave the existing numerical workspace at 0.13, but update the public
    # parameters to 0.15 immediately before replacing the retained family.
    receiver.set_model_parameters({_ALPHA_S: 0.15})
    receiver.load_cache(cache)
    _assert_warm(receiver, selectors)
    actual = receiver.evaluate(points, **selectors)
    fresh = Runtime.load(artifact, model_parameters={_ALPHA_S: 0.15})
    assert actual == pytest.approx(fresh.evaluate(points, **selectors), rel=2e-12)
    assert actual != pytest.approx(old)

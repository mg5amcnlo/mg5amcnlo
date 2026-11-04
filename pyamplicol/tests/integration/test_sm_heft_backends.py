# SPDX-License-Identifier: 0BSD
"""Packaged scalar HEFT execution through every evaluator backend."""

from __future__ import annotations

import importlib.util
import json
import math
import shutil
import tomllib
from pathlib import Path

import pytest

from pyamplicol import Generator, ModelSource, Runtime
from pyamplicol.config import (
    ColorConfig,
    CppConfig,
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationRelationDiscoveryConfig,
    GenerationValidationConfig,
    JITConfig,
    ModelConfig,
    ProcessConfig,
    RunConfig,
)


def test_packaged_sm_heft_runs_in_every_execution_mode_and_backend(
    tmp_path: Path,
) -> None:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")
    cxx = shutil.which("c++")

    cache_dir = tmp_path / "model-cache"
    source = ModelSource.built_in_sm_heft()
    source.compile(cache_dir=cache_dir, use_cache=True)
    values: dict[str, complex] = {}
    validation_points: dict[str, object] = {}
    lanes = [
        ("backend-jit", "compiled", "jit", "g g > H", "backend"),
        ("backend-asm", "compiled", "asm", "g g > H", "backend"),
        (
            "contact-compiled-jit",
            "compiled",
            "jit",
            "g g > H g g",
            "contact",
        ),
        ("contact-eager-jit", "eager", "jit", "g g > H g g", "contact"),
        (
            "contact-recurrence-jit",
            "recurrence",
            "jit",
            "g g > H g g",
            "contact",
        ),
        (
            "contact-on-the-fly-jit",
            "on-the-fly",
            "jit",
            "g g > H g g",
            "contact",
        ),
    ]
    if cxx is not None:
        lanes.insert(2, ("backend-cpp", "compiled", "cpp", "g g > H", "backend"))
    for lane, execution_mode, backend, process, comparison_group in lanes:
        artifact = tmp_path / lane
        evaluator = EvaluatorConfig(
            backend=backend,
            execution_mode=execution_mode,
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(optimization_level=0),
            cpp=(CppConfig(compiler=cxx) if backend == "cpp" else CppConfig()),
        )
        config = RunConfig(
            action="generate",
            model=ModelConfig(
                source="built-in-sm-heft",
                cache=True,
                cache_dir=cache_dir,
            ),
            color=ColorConfig(accuracy="full"),
            process=ProcessConfig(
                coupling_order_policy="explicit",
                max_coupling_orders={"HIG": 1},
            ),
            generation=GenerationConfig(
                workers=1,
                emit_api_bundle=False,
                relation_discovery=GenerationRelationDiscoveryConfig(mode="off"),
                validation=GenerationValidationConfig(
                    enabled=False,
                    samples=1,
                    seed=101,
                    post_build_validation=False,
                ),
            ),
            evaluator=evaluator,
        )
        Generator(config).generate(process, artifact, model=source)
        runtime = Runtime.load(artifact)
        points = runtime._backend.validation_momenta()
        assert points is not None
        if comparison_group not in validation_points:
            validation_points[comparison_group] = points
        else:
            assert points == validation_points[comparison_group]
        value = complex(runtime.evaluate(points)[0])
        assert math.isfinite(value.real) and math.isfinite(value.imag)
        assert value.real != 0.0
        assert abs(value.imag) <= abs(value.real) * 1.0e-13
        values[lane] = value

        manifest = json.loads((artifact / "artifact.json").read_text(encoding="utf-8"))
        assert manifest["model"]["name"] == "built-in-sm-heft"
        assert manifest["model"]["source_kind"] == (
            "built-in-sm-heft" if execution_mode == "compiled" else "compiled-model"
        )
        effective = tomllib.loads(
            (artifact / "config/effective.toml").read_text(encoding="utf-8")
        )
        assert effective["evaluator"]["backend"] == backend
        assert effective["evaluator"]["execution_mode"] == execution_mode

    references = {
        "backend": "backend-jit",
        "contact": "contact-compiled-jit",
    }
    for lane, _execution_mode, _backend, _process, comparison_group in lanes:
        reference = references[comparison_group]
        assert values[lane] == pytest.approx(
            values[reference], rel=1.0e-12, abs=1.0e-300
        )

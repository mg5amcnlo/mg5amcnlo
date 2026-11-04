# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

import pyamplicol.generation.service as generation_service
from pyamplicol import Generator, ModelSource, Runtime
from pyamplicol.artifacts import load_manifest
from pyamplicol.config import (
    Action,
    EvaluatorConfig,
    GenerationConfig,
    GenerationValidationConfig,
    JITConfig,
    RunConfig,
)


def test_compiled_append_preserves_existing_direct_eager_process(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")

    # A stale developer environment must never revive the removed plan-v2
    # writer. Eager generation is unconditionally plan-v3.
    monkeypatch.setenv("PYAMPLICOL_EAGER_PLAN_VERSION", "v2")
    eager_evaluator = EvaluatorConfig(
        execution_mode="eager",
        jit=JITConfig(optimization_level=0),
    )
    prepared = ModelSource.built_in_sm().compile(
        use_cache=True,
        cache_dir=tmp_path / "model-cache",
        prepared_output=tmp_path / "built-in-sm-jit-o0.pyamplicol-model",
        evaluator=eager_evaluator,
    )
    generation = GenerationConfig(
        emit_api_bundle=False,
        validation=GenerationValidationConfig(
            enabled=False,
            post_build_validation=False,
        ),
    )
    artifact = tmp_path / "artifact"
    initial_configuration: list[object] = []
    write_artifact = generation_service.write_schema_v3_artifact

    def write_with_initial_configuration(*args: Any, **kwargs: Any) -> Any:
        if kwargs["mode"] == "error":
            initial_configuration.append(kwargs["configuration"])
        else:
            # This is a writer-layer heterogeneous-append regression. Public
            # generation normally partitions execution modes by requested
            # configuration, so retain the exact initial provenance here to
            # exercise the union pack rewrite itself.
            kwargs["configuration"] = initial_configuration[0]
        return write_artifact(*args, **kwargs)

    monkeypatch.setattr(
        generation_service,
        "write_schema_v3_artifact",
        write_with_initial_configuration,
    )
    Generator(
        RunConfig(
            action=Action.GENERATE,
            generation=generation,
            evaluator=eager_evaluator,
        )
    ).generate("d d~ > z", artifact, model=prepared)

    original_manifest = load_manifest(artifact)
    original_process_id = str(original_manifest.processes[0]["id"])
    original_execution = json.loads(
        (artifact / f"processes/{original_process_id}/execution.json").read_text(
            encoding="utf-8"
        )
    )
    assert original_execution["eager_plan_abi"] == "pyamplicol-eager-plan-v3"
    assert original_execution["required_runtime_capabilities"] == [
        "eager-direct-arena-v1",
        "rusticol.eager-runtime-layout.complex-f64.v1",
    ]
    point = (
        (45.594, 0.0, 0.0, 45.594),
        (45.594, 0.0, 0.0, -45.594),
        (91.188, 0.0, 0.0, 0.0),
    )
    before = Runtime.load(artifact, process=original_process_id).evaluate((point,))[0]

    Generator(
        RunConfig(
            action=Action.GENERATE,
            generation=generation,
            evaluator=EvaluatorConfig(
                execution_mode="compiled",
                # The existing eager artifact is portable-64le. Keep the
                # appended compiled evaluator inside the same canonical
                # portable O1/O2 target contract; this regression is about
                # heterogeneous eager/compiled pack preservation, not a
                # target transition.
                jit=JITConfig(optimization_level=2),
            ),
        )
    ).generate("u u~ > z", artifact, model=prepared, mode="append")

    after = Runtime.load(artifact, process=original_process_id).evaluate((point,))[0]
    assert after == pytest.approx(before, rel=1.0e-12, abs=1.0e-15)

    appended_manifest = load_manifest(artifact)
    assert {
        "eager-direct-arena-v1",
        "rusticol.eager-runtime-layout.complex-f64.v1",
    }.issubset(appended_manifest.runtime["required_runtime_capabilities"])
    kernel_pack = json.loads(
        (artifact / "model/eager-kernel-pack.json").read_text(encoding="utf-8")
    )
    assert kernel_pack["kernels"]
    assert all(
        "direct_table" in kernel["f64_evaluator_manifest"]
        for kernel in kernel_pack["kernels"]
    )

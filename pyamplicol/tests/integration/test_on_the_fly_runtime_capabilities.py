# SPDX-License-Identifier: 0BSD
"""Small genuine-artifact regression for on-the-fly capability loading."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from pyamplicol import Generator, ModelSource, ProcessRequest, Runtime
from pyamplicol._internal.versions import (
    ON_THE_FLY_LC_COLOR_RUNTIME_CAPABILITY,
    ON_THE_FLY_RUNTIME_CAPABILITY,
)
from pyamplicol.artifacts import load_manifest
from pyamplicol.assets.prepared_models import (
    BUILTIN_SM_JIT_O2,
    packaged_prepared_model_path,
)
from pyamplicol.config import (
    Action,
    ColorAccuracy,
    ColorConfig,
    EvaluatorConfig,
    EvaluatorExecutionMode,
    GenerationConfig,
    GenerationValidationConfig,
    RunConfig,
)

_PROCESS = "d d~ > z"
_PROCESS_ID = "d_dbar_to_z"
_CAPABILITIES = (
    ON_THE_FLY_RUNTIME_CAPABILITY,
    ON_THE_FLY_LC_COLOR_RUNTIME_CAPABILITY,
)


def _unavailable(reason: str) -> None:
    if os.environ.get("PYAMPLICOL_REQUIRE_NATIVE_TESTS") == "1":
        pytest.fail(reason)
    pytest.skip(reason)


def _require_native_on_the_fly() -> None:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        _unavailable("the Rusticol extension has not been built")
    rusticol = importlib.import_module("pyamplicol._rusticol")
    if not hasattr(rusticol, "_build_on_the_fly_process_seeds_v1"):
        _unavailable("the installed Rusticol extension lacks on-the-fly generation")


def _configuration(*, emit_api_bundle: bool = False) -> RunConfig:
    return RunConfig(
        action=Action.GENERATE,
        color=ColorConfig(accuracy=ColorAccuracy.LC),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=emit_api_bundle,
            validation=GenerationValidationConfig(
                enabled=False,
                post_build_validation=False,
            ),
        ),
        evaluator=EvaluatorConfig(
            execution_mode=EvaluatorExecutionMode.ON_THE_FLY,
        ),
    )


def test_fresh_builtin_on_the_fly_artifact_loads_with_its_declared_capabilities(
    tmp_path: Path,
) -> None:
    _require_native_on_the_fly()
    artifact = tmp_path / "on-the-fly-d-dbar-to-z"
    with packaged_prepared_model_path(BUILTIN_SM_JIT_O2) as prepared_model:
        Generator(_configuration()).generate(
            ProcessRequest.parse(_PROCESS, name=_PROCESS_ID),
            artifact,
            model=ModelSource.from_path(prepared_model),
        )

    manifest = load_manifest(artifact)
    assert manifest.runtime["required_runtime_capabilities"] == _CAPABILITIES
    assert manifest.processes[0]["required_runtime_capabilities"] == _CAPABILITIES

    runtime = Runtime.load(artifact, process=_PROCESS_ID)
    assert runtime.execution_mode == "on-the-fly"
    assert runtime.physics.process_id == _PROCESS_ID
    assert runtime.physics.process == _PROCESS


def _rusticol_config() -> tuple[str, ...]:
    configured = os.environ.get("RUSTICOL_CONFIG")
    if configured:
        command = tuple(shlex.split(configured))
        if command and (shutil.which(command[0]) or Path(command[0]).is_file()):
            return command
    sibling = Path(sys.executable).parent / "rusticol-config"
    if sibling.is_file():
        return (str(sibling),)
    discovered = shutil.which("rusticol-config")
    if discovered:
        return (discovered,)
    if importlib.util.find_spec("rusticol_config") is not None:
        return (sys.executable, "-m", "rusticol_config")
    _unavailable("the installed Rusticol C SDK is unavailable")
    raise AssertionError("unreachable")


def _validation_point(artifact: Path) -> tuple[tuple[float, ...], ...]:
    lines = (
        (artifact / "API/validation_points.dat")
        .read_text(encoding="ascii")
        .splitlines()
    )
    assert lines[0] == "RUSTICOL_VALIDATION_POINTS_V1"
    fields = next(
        line.split("\t")
        for line in lines[1:]
        if line and not line.startswith("#") and line.split("\t", 1)[0] == _PROCESS_ID
    )
    external_count = int(fields[1])
    components = tuple(float(value) for value in fields[2:])
    assert len(components) == 4 * external_count
    return tuple(
        components[offset : offset + 4] for offset in range(0, len(components), 4)
    )


def _real(value: complex | Decimal) -> float:
    return float(complex(value).real)


def test_generated_c_api_loads_and_evaluates_on_the_fly_artifact(
    tmp_path: Path,
) -> None:
    _require_native_on_the_fly()
    make = shutil.which("make")
    cc = shutil.which(os.environ.get("CC", "cc"))
    if make is None or cc is None:
        _unavailable("the on-the-fly C API test requires make and a C compiler")
    rusticol_config = _rusticol_config()
    artifact = tmp_path / "on-the-fly-c-api"
    with packaged_prepared_model_path(BUILTIN_SM_JIT_O2) as prepared_model:
        Generator(_configuration(emit_api_bundle=True)).generate(
            ProcessRequest.parse(_PROCESS, name=_PROCESS_ID),
            artifact,
            model=ModelSource.from_path(prepared_model),
        )

    manifest = load_manifest(artifact)
    assert manifest.runtime["api_bundle_path"] == "API"
    c_source = artifact / "API/c/check_standalone.c"
    assert "rusticol_runtime_evaluate_f64" in c_source.read_text(encoding="utf-8")
    assert "rusticol_runtime_evaluate_resolved_f64" in c_source.read_text(
        encoding="utf-8"
    )

    environment = os.environ.copy()
    environment.update(
        {
            "CC": cc,
            "RUSTICOL_CONFIG": shlex.join(rusticol_config),
        }
    )
    compiled = subprocess.run(
        [make, "-C", str(artifact / "API/c"), "all"],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert compiled.returncode == 0, (
        f"on-the-fly C API driver failed to build\n"
        f"stdout:\n{compiled.stdout}\nstderr:\n{compiled.stderr}"
    )
    driver = (
        artifact.parent / ".pyamplicol-api-build" / artifact.name / "c/check_standalone"
    )
    executed = subprocess.run(
        [str(driver), "--json", "--process", "  d   d~  >  z  "],
        cwd=artifact,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert executed.returncode == 0, (
        f"on-the-fly C API driver failed\n"
        f"stdout:\n{executed.stdout}\nstderr:\n{executed.stderr}"
    )
    payload: dict[str, Any] = json.loads(executed.stdout)
    assert payload["available"] is True
    assert payload["process_key"] == _PROCESS_ID
    assert payload["process"].casefold() == _PROCESS.casefold()

    runtime = Runtime.load(artifact, process=_PROCESS_ID)
    point = _validation_point(artifact)
    expected_total = runtime.evaluate((point,))
    expected_resolved = runtime.evaluate_resolved((point,))
    expected_values = [
        _real(value) for helicity in expected_resolved.values[0] for value in helicity
    ]
    assert payload["shape"] == [
        1,
        len(expected_resolved.helicity_ids),
        len(expected_resolved.color_ids),
    ]
    assert payload["values"] == pytest.approx(expected_values, rel=1.0e-12, abs=1.0e-15)
    assert payload["compatibility_total"] == pytest.approx(
        [_real(expected_total[0])], rel=1.0e-12, abs=1.0e-15
    )
    assert payload["resolved_sum"] == pytest.approx(
        [sum(expected_values)], rel=1.0e-12, abs=1.0e-15
    )

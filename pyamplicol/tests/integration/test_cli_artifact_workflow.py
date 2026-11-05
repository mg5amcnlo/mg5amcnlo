# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from pyamplicol import (
    Generator,
    ModelSource,
    ProcessAlias,
    ProcessRequest,
    ProcessSet,
    Runtime,
)
from pyamplicol.artifacts import load_manifest
from pyamplicol.assets.prepared_models import (
    BUILTIN_SM_JIT_O2,
    packaged_prepared_model_path,
)
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    EvaluatorExecutionMode,
    GenerationConfig,
    GenerationValidationConfig,
    JITConfig,
    RunConfig,
)

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class _Selection:
    selector: str
    process_id: str
    expression: str
    momenta: tuple[tuple[float, float, float, float], ...]


def _native_extension_available() -> bool:
    return importlib.util.find_spec("pyamplicol._rusticol") is not None


def _base_momenta() -> tuple[tuple[float, float, float, float], ...]:
    sqrt_s = 500.0
    z_mass = 91.188
    outgoing_momentum = (sqrt_s * sqrt_s - z_mass * z_mass) / (2.0 * sqrt_s)
    z_energy = (sqrt_s * sqrt_s + z_mass * z_mass) / (2.0 * sqrt_s)
    return (
        (250.0, 0.0, 0.0, 250.0),
        (250.0, 0.0, 0.0, -250.0),
        (z_energy, outgoing_momentum, 0.0, 0.0),
        (outgoing_momentum, -outgoing_momentum, 0.0, 0.0),
    )


@pytest.fixture(scope="module")
def lc_process_set_artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not _native_extension_available():
        pytest.skip("the Rusticol extension has not been built")

    artifact = tmp_path_factory.mktemp("cli-artifact-workflow") / "artifact"
    request = ProcessRequest.parse("d d~ > z g", name="ddbar_zg")
    processes = ProcessSet(
        requests=(request,),
        aliases=(
            ProcessAlias(
                name="ddbar_gz",
                process_name=request.name,
                particle_permutation=(0, 1, 3, 2),
            ),
        ),
    )
    config = RunConfig(
        action="generate",
        color=ColorConfig(accuracy="lc"),
        generation=GenerationConfig(
            emit_api_bundle=False,
            validation=GenerationValidationConfig(
                enabled=False,
                post_build_validation=False,
            ),
        ),
        evaluator=EvaluatorConfig(jit=JITConfig(optimization_level=1)),
    )

    Generator(config).generate(processes, artifact)
    return artifact


@pytest.fixture(scope="module")
def otf_process_set_artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not _native_extension_available():
        pytest.skip("the Rusticol extension has not been built")

    artifact = tmp_path_factory.mktemp("cli-otf-workflow") / "artifact"
    request = ProcessRequest.parse("d d~ > z g", name="ddbar_zg")
    processes = ProcessSet(
        requests=(request,),
        aliases=(
            ProcessAlias(
                name="ddbar_gz",
                process_name=request.name,
                particle_permutation=(0, 1, 3, 2),
            ),
        ),
    )
    config = RunConfig(
        action="generate",
        color=ColorConfig(accuracy="lc"),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
            validation=GenerationValidationConfig(
                enabled=False,
                post_build_validation=False,
            ),
        ),
        evaluator=EvaluatorConfig(
            execution_mode=EvaluatorExecutionMode.ON_THE_FLY,
        ),
    )

    with packaged_prepared_model_path(BUILTIN_SM_JIT_O2) as prepared_model:
        Generator(config).generate(
            processes,
            artifact,
            model=ModelSource.from_path(prepared_model),
        )
    return artifact


def _run_json_cli(*arguments: str, timeout: float = 30.0) -> Any:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["SYMBOLICA_HIDE_BANNER"] = "1"
    completed = subprocess.run(
        [sys.executable, "-m", "pyamplicol", *arguments],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        pytest.fail(
            f"CLI JSON stdout contains non-JSON output: {completed.stdout!r} ({exc})"
        )


def _write_momenta(
    path: Path,
    momenta: tuple[tuple[float, float, float, float], ...],
) -> None:
    path.write_text(json.dumps([momenta]) + "\n", encoding="utf-8")


def _complex_values(payload: Any) -> tuple[complex, ...]:
    assert isinstance(payload, list)
    return tuple(
        complex(float(entry["real"]), float(entry["imag"])) for entry in payload
    )


def _flatten_complex_values(payload: Any) -> tuple[complex, ...]:
    if isinstance(payload, list):
        return tuple(
            value for entry in payload for value in _flatten_complex_values(entry)
        )
    assert isinstance(payload, dict)
    return (complex(float(payload["real"]), float(payload["imag"])),)


def test_cli_inspect_evaluate_and_profile_select_processes_by_public_identity(
    lc_process_set_artifact: Path,
    tmp_path: Path,
) -> None:
    base = _base_momenta()
    alias = (base[0], base[1], base[3], base[2])
    selections = (
        _Selection("ddbar_zg", "ddbar_zg", "d d~ > z g", base),
        _Selection("d d~ > z g", "ddbar_zg", "d d~ > z g", base),
        _Selection("d d~ > g z", "ddbar_gz", "d d~ > g z", alias),
    )

    for index, selection in enumerate(selections):
        runtime = Runtime.load(
            lc_process_set_artifact,
            process=selection.selector,
        )
        points = (selection.momenta,)
        expected = runtime.evaluate(points)

        inspection = _run_json_cli(
            "inspect",
            str(lc_process_set_artifact),
            "--process",
            selection.selector,
            "--full-physics",
            "--json",
            "--progress",
            "off",
            "--color",
            "never",
            "--log-level",
            "error",
        )
        assert inspection["process_id"] == selection.process_id
        assert inspection["process"] == selection.expression
        assert inspection["color_accuracy"] == "lc"
        assert [
            particle["name"] for particle in inspection["external_particles"]
        ] == selection.expression.replace(">", " ").split()

        momenta_path = tmp_path / f"momenta-{index}.json"
        _write_momenta(momenta_path, selection.momenta)
        evaluated = _run_json_cli(
            "evaluate",
            str(lc_process_set_artifact),
            "--process",
            selection.selector,
            "--momenta",
            str(momenta_path),
            "--json",
            "--progress",
            "off",
            "--color",
            "never",
            "--log-level",
            "error",
        )
        assert _complex_values(evaluated) == pytest.approx(expected, rel=1.0e-13)

        profile = _run_json_cli(
            "profile",
            str(lc_process_set_artifact),
            "--process",
            selection.selector,
            "--momenta",
            str(momenta_path),
            "--target-runtime",
            "0.001",
            "--batch-size",
            "2",
            "--warmup-runs",
            "0",
            "--minimum-samples",
            "2",
            "--json",
            "--progress",
            "off",
            "--color",
            "never",
            "--log-level",
            "error",
        )
        assert profile["process_id"] == runtime.physics.process_id
        assert profile["process_expression"] == runtime.physics.process
        assert (
            profile["environment"]["wall_time_source"]
            == "python_outer_perf_counter_wall_time"
        )
        assert profile["effective_config"]["target_runtime"] == pytest.approx(0.001)
        assert profile["effective_config"]["batch_size"] == 2
        assert profile["sample_count"] >= 2
        assert profile["wall_time_per_point"] > 0.0


def test_otf_alias_crosses_compact_and_full_public_cli_and_runtime_surfaces(
    otf_process_set_artifact: Path,
    tmp_path: Path,
) -> None:
    selector = "d d~ > g z"
    base = _base_momenta()
    alias_momenta = (base[0], base[1], base[3], base[2])
    points = (alias_momenta,)
    manifest = load_manifest(otf_process_set_artifact)

    runtime = Runtime.load(otf_process_set_artifact, process=selector)
    cold = runtime.inspect()
    assert cold["kind"] == "pyamplicol-runtime-inspection"
    assert cold["artifact_id"] == manifest.artifact_id
    assert cold["supported_precisions"] == (16,)
    metadata = cold["runtime_metadata"]
    assert isinstance(metadata, dict)
    assert metadata["execution_mode"] == "on-the-fly"
    assert metadata["process_key"] == "ddbar_gz"
    assert metadata["representative_process_key"] == "ddbar_zg"
    assert metadata["external_permutation"] == [0, 1, 3, 2]
    assert metadata["permutation_alias_of"] == "ddbar_zg"
    cold_state = cold["on_the_fly_state"]
    assert isinstance(cold_state, dict)
    assert cold_state["family_cache_policy"] == "last-family-only"
    assert cold_state["family_cache_limit"] == 1
    assert cold_state["retained_family_count"] == 0

    expected_total = runtime.evaluate(points)
    expected_resolved = runtime.evaluate_resolved(points)
    assert expected_resolved.total() == pytest.approx(expected_total, rel=1.0e-13)
    assert all(
        math.isfinite(value.real) and math.isfinite(value.imag)
        for value in expected_total
    )
    warm_state = runtime.inspect()["on_the_fly_state"]
    assert isinstance(warm_state, dict)
    assert warm_state["retained_family_count"] == 1
    assert warm_state["retained_selection_count"] == 1

    compact = _run_json_cli(
        "inspect",
        str(otf_process_set_artifact),
        "--process",
        selector,
        "--json",
        "--progress",
        "off",
        "--color",
        "never",
        "--log-level",
        "error",
    )
    assert compact["kind"] == "pyamplicol-artifact-inspection"
    assert len(compact["processes"]) == 1
    compact_process = compact["processes"][0]
    assert compact_process["id"] == "ddbar_zg"
    assert compact_process["execution_mode"] == "on-the-fly"
    assert compact_process["selector_provenance"] == "on-the-fly-compact-seed"
    assert compact_process["lc_flow_layout"] == "compact/query-local"
    assert [alias["id"] for alias in compact_process["aliases"]] == ["ddbar_gz"]

    full = _run_json_cli(
        "inspect",
        str(otf_process_set_artifact),
        "--process",
        selector,
        "--full-physics",
        "--json",
        "--progress",
        "off",
        "--color",
        "never",
        "--log-level",
        "error",
    )
    assert full["process_id"] == "ddbar_gz"
    assert full["process"] == selector
    assert [particle["name"] for particle in full["external_particles"]] == [
        "d",
        "d~",
        "g",
        "z",
    ]

    momenta_path = tmp_path / "otf-alias-momenta.json"
    _write_momenta(momenta_path, alias_momenta)
    evaluated = _run_json_cli(
        "evaluate",
        str(otf_process_set_artifact),
        "--process",
        selector,
        "--momenta",
        str(momenta_path),
        "--json",
        "--progress",
        "off",
        "--color",
        "never",
        "--log-level",
        "error",
    )
    assert _complex_values(evaluated) == pytest.approx(expected_total, rel=1.0e-13)

    resolved = _run_json_cli(
        "evaluate",
        str(otf_process_set_artifact),
        "--process",
        selector,
        "--momenta",
        str(momenta_path),
        "--resolved",
        "--json",
        "--progress",
        "off",
        "--color",
        "never",
        "--log-level",
        "error",
    )
    assert resolved["helicity_ids"] == list(expected_resolved.helicity_ids)
    assert resolved["color_ids"] == list(expected_resolved.color_ids)
    assert resolved["color_accuracy"] == "lc"
    assert _flatten_complex_values(resolved["values"]) == pytest.approx(
        tuple(
            value
            for point in expected_resolved.values
            for helicity in point
            for value in helicity
        ),
        rel=1.0e-13,
    )

    profile = _run_json_cli(
        "profile",
        str(otf_process_set_artifact),
        "--process",
        selector,
        "--momenta",
        str(momenta_path),
        "--target-runtime",
        "0.001",
        "--batch-size",
        "2",
        "--warmup-runs",
        "0",
        "--minimum-samples",
        "2",
        "--json",
        "--progress",
        "off",
        "--color",
        "never",
        "--log-level",
        "error",
    )
    assert profile["process_id"] == "ddbar_gz"
    assert profile["process_expression"] == selector
    assert profile["environment"]["execution_mode"] == "on-the-fly"
    assert profile["environment"]["cold_warmup_runtime_freshness"] == (
        "authenticated-cold"
    )
    assert profile["environment"]["cold_warmup_runtime_state_evidence"] == (
        "authenticated-native-otf-census-v1"
    )
    before = profile["environment"]["cold_warmup_runtime_state_before"]
    after = profile["environment"]["cold_warmup_runtime_state_after"]
    assert before["family_cache_policy"] == "last-family-only"
    assert before["family_cache_limit"] == 1
    assert before["retained_family_count"] == 0
    assert after["retained_family_count"] == 1
    assert profile["sample_count"] >= 2
    assert profile["wall_time_per_point"] > 0.0

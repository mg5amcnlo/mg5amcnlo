# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import importlib
import io
import json
import tomllib
from pathlib import Path

import pytest

import pyamplicol.api.services as service_module
import pyamplicol.evaluators.symbolica_adapters as symbolica_adapters
import pyamplicol.generation.artifact_writer as artifact_writer
import pyamplicol.licensing as licensing_module
from pyamplicol.api import Generator, ProcessSet
from pyamplicol.api.results import GenerationPlan, GenerationResult
from pyamplicol.cli import run_cli
from pyamplicol.config import (
    ColorConfig,
    ConfigResolution,
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    ProcessConfig,
    ProcessEntry,
    RunConfig,
    resolve_config,
)
from pyamplicol.licensing import SymbolicaLicenseState

cli_main_module = importlib.import_module("pyamplicol.cli.main")


class _PlanningBackend:
    def __init__(self, resolution: ConfigResolution) -> None:
        self._resolution = resolution

    def plan(self, processes: ProcessSet, *, model: object = None) -> GenerationPlan:
        del model
        return GenerationPlan(
            concrete_processes=processes.requests,
            estimated_coverage={},
            requested_settings=self._resolution.requested,
            effective_settings=self._resolution.effective,
            adjustments=self._resolution.clamps,
        )

    def generate(
        self,
        processes: ProcessSet,
        output: Path,
        **_kwargs: object,
    ) -> GenerationResult:
        return GenerationResult(output, processes, "error", files=())


def _stub_candidate_artifact_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(artifact_writer, "_default_api_bundle_hook", lambda: None)
    monkeypatch.setattr(
        artifact_writer,
        "_target_metadata",
        lambda _config: ({"triple": "aarch64-apple-darwin", "cpu_features": []}, 1),
    )
    monkeypatch.setattr(
        symbolica_adapters._JITSymbolicaEvaluatorAdapter,
        "_export_symjit_application",
        lambda _self: (
            b"test-symjit-application",
            symbolica_adapters._symjit_element_layout(),
        ),
    )
    monkeypatch.setattr(
        symbolica_adapters._JITSymbolicaEvaluatorAdapter,
        "_export_symjit_plane_application",
        lambda _self, *, optimization_level: (
            b"test-symjit-plane-application",
            "0" * 64,
            {"word_bits": 64, "endianness": "little"},
        ),
    )


def test_programmatic_generator_forwards_config_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolution = resolve_config({"action": "generate"})
    received: list[object] = []

    def factory(config: object, progress: object) -> _PlanningBackend:
        received.append((config, progress))
        assert isinstance(config, ConfigResolution)
        return _PlanningBackend(config)

    monkeypatch.setattr(service_module, "_generator_factory", factory)
    monkeypatch.setattr(
        licensing_module,
        "detect_symbolica_license",
        lambda **_kwargs: SymbolicaLicenseState(licensed=True, restricted=False),
    )

    Generator(resolution).plan("d d~ > z")

    assert received == [(resolution, None)]


@pytest.mark.parametrize("contraction", ("auto", "direct"))
def test_python_generator_preserves_automatic_fft_and_direct_opt_out(
    monkeypatch: pytest.MonkeyPatch, contraction: str
) -> None:
    config = RunConfig(
        action="generate", color=ColorConfig(accuracy="full", contraction=contraction)
    )
    received = []

    def factory(resolution, _progress):
        received.append(resolution)
        return _PlanningBackend(resolution)

    monkeypatch.setattr(service_module, "_generator_factory", factory)
    monkeypatch.setattr(
        licensing_module,
        "detect_symbolica_license",
        lambda **_kwargs: SymbolicaLicenseState(licensed=True, restricted=False),
    )
    Generator(config).plan("g g > g g")
    assert len(received) == 1
    assert received[0].effective.color.contraction == contraction
    assert received[0].effective.color.fft_basis == (
        "adjoint" if contraction == "auto" else "trace"
    )


@pytest.mark.parametrize("config", (None, GenerationConfig()))
def test_python_generator_shortcuts_resolve_conditional_color_defaults(
    monkeypatch: pytest.MonkeyPatch, config
) -> None:
    received = []

    def factory(resolution, _progress):
        received.append(resolution)
        return _PlanningBackend(resolution)

    monkeypatch.setattr(service_module, "_generator_factory", factory)
    monkeypatch.setattr(
        licensing_module,
        "detect_symbolica_license",
        lambda **_kwargs: SymbolicaLicenseState(licensed=True, restricted=False),
    )
    Generator(config).plan("g g > g g")
    assert received[0].requested.color.contraction == "auto"
    assert received[0].requested.color.fft_basis == "adjoint"
    assert received[0].effective.color.accuracy == "lc"
    assert received[0].effective.color.contraction == "direct"
    assert received[0].effective.color.fft_basis == "trace"


def test_programmatic_generation_applies_restricted_resource_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = RunConfig(
        action="generate",
        process=ProcessConfig(entries=(ProcessEntry("d d~ > z"),)),
        generation=GenerationConfig(workers=4),
        evaluator=EvaluatorConfig(optimization=EvaluatorOptimizationConfig(cores=8)),
    )
    received: list[ConfigResolution] = []

    def factory(resolution: object, _progress: object) -> _PlanningBackend:
        assert isinstance(resolution, ConfigResolution)
        received.append(resolution)
        return _PlanningBackend(resolution)

    monkeypatch.setattr(service_module, "_generator_factory", factory)
    monkeypatch.setattr(
        licensing_module,
        "detect_symbolica_license",
        lambda **_kwargs: SymbolicaLicenseState(licensed=False, restricted=True),
    )

    Generator(config).generate("d d~ > z", tmp_path / "artifact")

    assert len(received) == 1
    resolution = received[0]
    assert resolution.requested.generation.workers == 4
    assert resolution.effective.generation.workers == 1
    assert resolution.requested.evaluator.optimization.cores == 8
    assert resolution.effective.evaluator.optimization.cores == 1


def test_cli_artifact_preserves_restricted_resource_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    stdout = io.StringIO()
    stderr = io.StringIO()

    monkeypatch.setattr(
        licensing_module,
        "detect_symbolica_license",
        lambda **_kwargs: SymbolicaLicenseState(licensed=False, restricted=True),
    )
    _stub_candidate_artifact_generation(monkeypatch)

    status = run_cli(
        (
            "generate",
            "p p > z",
            str(artifact),
            "--workers",
            "4",
            "--execution-mode",
            "compiled",
            "--cores",
            "8",
            "--no-jit-compress",
            "--json",
            "--progress",
            "off",
            "--set",
            "generation.validation.post_build_validation=false",
        ),
        stdout=stdout,
        stderr=stderr,
    )

    assert status == 0, stderr.getvalue()
    assert json.loads(stdout.getvalue())["output"] == str(artifact)

    requested = tomllib.loads(
        (artifact / "config/requested.toml").read_text(encoding="utf-8")
    )
    effective = tomllib.loads(
        (artifact / "config/effective.toml").read_text(encoding="utf-8")
    )
    assert requested["generation"]["workers"] == 4
    assert effective["generation"]["workers"] == 1
    assert requested["process"]["entries"] == [{"expression": "p p > z"}]
    assert effective["process"]["entries"] == [{"expression": "p p > z"}]
    assert requested["evaluator"]["optimization"]["cores"] == 8
    assert effective["evaluator"]["optimization"]["cores"] == 1
    assert requested["evaluator"]["jit"]["compress"] is False
    assert effective["evaluator"]["jit"]["compress"] is False
    assert requested["generation"]["emit_api_bundle"] is True
    assert effective["generation"]["emit_api_bundle"] is False
    assert requested != effective

    manifest = json.loads((artifact / "artifact.json").read_text(encoding="utf-8"))
    assert len(manifest["processes"]) == 5
    assert manifest["configuration"]["adjustments"] == [
        {
            "path": "generation.workers",
            "reason": ("Symbolica restricted mode permits one generation instance"),
        },
        {
            "path": "evaluator.optimization.cores",
            "reason": "Symbolica restricted mode permits one Symbolica core",
        },
        {
            "path": "generation.emit_api_bundle",
            "reason": "no root API-bundle emitter is installed",
        },
    ]
    assert manifest["runtime"]["api_bundle_path"] is None


def test_cli_licensed_multiparticle_uses_concrete_resource_partition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    stdout = io.StringIO()
    stderr = io.StringIO()
    cpu_budget = 12

    monkeypatch.setattr(
        licensing_module,
        "detect_symbolica_license",
        lambda **_kwargs: SymbolicaLicenseState(licensed=True, restricted=False),
    )
    monkeypatch.setattr(licensing_module, "_cpu_budget", lambda: cpu_budget)
    _stub_candidate_artifact_generation(monkeypatch)

    status = run_cli(
        (
            "generate",
            "p p > z",
            str(artifact),
            "--execution-mode",
            "compiled",
            "--json",
            "--progress",
            "off",
            "--set",
            "generation.validation.post_build_validation=false",
            "--set",
            "generation.relation_discovery.mode=off",
        ),
        stdout=stdout,
        stderr=stderr,
    )

    assert status == 0, stderr.getvalue()
    assert json.loads(stdout.getvalue())["output"] == str(artifact)
    requested = tomllib.loads(
        (artifact / "config/requested.toml").read_text(encoding="utf-8")
    )
    effective = tomllib.loads(
        (artifact / "config/effective.toml").read_text(encoding="utf-8")
    )
    workers = effective["generation"]["workers"]
    cores = effective["evaluator"]["optimization"]["cores"]
    assert requested["generation"]["workers"] == "auto"
    assert requested["evaluator"]["optimization"]["cores"] == "auto"
    assert workers == 5
    assert cores == 2
    assert workers * cores <= cpu_budget

    manifest = json.loads((artifact / "artifact.json").read_text(encoding="utf-8"))
    assert len(manifest["processes"]) == 5
    assert manifest["configuration"]["adjustments"] == [
        {
            "path": "generation.workers",
            "reason": (
                "shared affinity-aware CPU budget for concurrent process generation"
            ),
        },
        {
            "path": "evaluator.optimization.cores",
            "reason": "shared affinity-aware CPU budget for Symbolica evaluator work",
        },
        {
            "path": "generation.emit_api_bundle",
            "reason": "no root API-bundle emitter is installed",
        },
    ]

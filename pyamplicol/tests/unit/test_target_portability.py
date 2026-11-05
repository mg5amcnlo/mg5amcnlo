# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from types import SimpleNamespace

import pytest

from pyamplicol._internal.versions import (
    SYMBOLICA_CPP_RUNTIME_CAPABILITY,
    SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY,
)
from pyamplicol.artifacts.manifest import PORTABLE_64LE_TARGET
from pyamplicol.config import (
    CppConfig,
    EvaluatorBackend,
    EvaluatorConfig,
    GenerationConfig,
    JITConfig,
    RunConfig,
)
from pyamplicol.generation import artifact_writer


def _config(
    *,
    backend: EvaluatorBackend,
    native_arch: bool,
    jit_optimization_level: int = 2,
) -> RunConfig:
    return RunConfig(
        action="generate",
        evaluator=EvaluatorConfig(
            backend=backend,
            cpp=CppConfig(native_arch=native_arch),
            jit=JITConfig(optimization_level=jit_optimization_level),
        ),
    )


def _install_target_info(
    monkeypatch: pytest.MonkeyPatch,
    *,
    triple: str = "x86_64-unknown-linux-gnu",
    features: tuple[str, ...] = ("avx2", "fma", "sse2"),
) -> None:
    rusticol = SimpleNamespace(
        target_info=lambda: SimpleNamespace(
            triple=triple,
            cpu_features=list(features),
        ),
        abi_version=lambda: 7,
    )
    monkeypatch.setattr(
        artifact_writer.importlib,
        "import_module",
        lambda name: rusticol,
    )


@pytest.mark.parametrize("optimization_level", (1, 2))
def test_portable_jit_artifacts_declare_portable_64le_target(
    monkeypatch: pytest.MonkeyPatch,
    optimization_level: int,
) -> None:
    _install_target_info(monkeypatch)
    target, c_abi = artifact_writer._artifact_target_metadata(
        _config(
            backend=EvaluatorBackend.JIT,
            native_arch=False,
            jit_optimization_level=optimization_level,
        )
    )
    assert target == {
        "triple": PORTABLE_64LE_TARGET,
        "cpu_features": [],
    }
    assert c_abi == 7


def test_implicit_generation_config_requires_portable_jit_content_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_target_info(monkeypatch)
    concrete, _ = artifact_writer._artifact_target_metadata(GenerationConfig())
    portable, _ = artifact_writer._artifact_target_metadata(
        GenerationConfig(),
        implicit_portable_jit_evidence=True,
    )
    assert concrete == {
        "triple": "x86_64-unknown-linux-gnu",
        "cpu_features": [],
    }
    assert portable == {
        "triple": PORTABLE_64LE_TARGET,
        "cpu_features": [],
    }


def test_portable_cpp_still_declares_concrete_host_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_target_info(monkeypatch)
    target, _ = artifact_writer._artifact_target_metadata(
        _config(backend=EvaluatorBackend.CPP, native_arch=False)
    )
    assert target == {
        "triple": "x86_64-unknown-linux-gnu",
        "cpu_features": [],
    }


@pytest.mark.parametrize("optimization_level", (0, 3))
def test_nonportable_jit_artifacts_stay_on_the_concrete_host_target(
    monkeypatch: pytest.MonkeyPatch,
    optimization_level: int,
) -> None:
    _install_target_info(monkeypatch)
    target, _ = artifact_writer._artifact_target_metadata(
        _config(
            backend=EvaluatorBackend.JIT,
            native_arch=False,
            jit_optimization_level=optimization_level,
        )
    )
    assert target == {
        "triple": "x86_64-unknown-linux-gnu",
        "cpu_features": [],
    }


@pytest.mark.parametrize(
    "runtime_capability",
    (SYMBOLICA_CPP_RUNTIME_CAPABILITY, SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY),
)
def test_target_specific_capability_prevents_portable_jit_target(
    monkeypatch: pytest.MonkeyPatch,
    runtime_capability: str,
) -> None:
    _install_target_info(monkeypatch)
    target, _ = artifact_writer._artifact_target_metadata(
        _config(backend=EvaluatorBackend.JIT, native_arch=False),
        runtime_capabilities=(runtime_capability,),
    )
    assert target == {
        "triple": "x86_64-unknown-linux-gnu",
        "cpu_features": [],
    }


@pytest.mark.parametrize(
    ("kind", "optimization_level", "expected"),
    (
        ("symjit-application-evaluator", 2, (True, 1)),
        ("symjit-application-evaluator", 1, (True, 1)),
        ("symjit-application-evaluator", 0, (False, 1)),
        ("symjit-application-evaluator", 3, (False, 1)),
        ("compiled-complex-evaluator", 2, (False, 0)),
        ("jit-symbolica-evaluator", 2, (False, 0)),
    ),
)
def test_implicit_generation_evaluator_evidence_requires_portable_symjit(
    kind: str,
    optimization_level: int,
    expected: tuple[bool, int],
) -> None:
    record = {
        "evaluator": {
            "kind": kind,
            "optimization_level": optimization_level,
            "plane_application": {
                "optimization_level": optimization_level,
            },
        }
    }
    assert (
        artifact_writer._mapping_has_only_portable_symjit_evaluators(record)
        == expected
    )


def test_jit_stays_baseline_when_cpp_native_option_is_irrelevant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_target_info(monkeypatch)
    target, _ = artifact_writer._artifact_target_metadata(
        _config(backend=EvaluatorBackend.JIT, native_arch=True)
    )
    assert target["triple"] == PORTABLE_64LE_TARGET
    assert target["cpu_features"] == []


@pytest.mark.parametrize("backend", (EvaluatorBackend.ASM, EvaluatorBackend.CPP))
def test_native_compiled_evaluators_record_detected_features(
    monkeypatch: pytest.MonkeyPatch, backend: EvaluatorBackend
) -> None:
    _install_target_info(monkeypatch)
    target, _ = artifact_writer._target_metadata(
        _config(backend=backend, native_arch=True)
    )
    assert target["cpu_features"] == ["avx2", "fma", "sse2"]


def test_writer_rejects_noncanonical_rusticol_target_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_target_info(monkeypatch, features=("sse2", "avx2"))
    with pytest.raises(RuntimeError, match="non-canonical"):
        artifact_writer._target_metadata(
            _config(backend=EvaluatorBackend.CPP, native_arch=True)
        )


def test_writer_rejects_targets_outside_the_release_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_target_info(monkeypatch, triple="aarch64-unknown-linux-gnu")
    with pytest.raises(RuntimeError, match="not supported"):
        artifact_writer._target_metadata(
            _config(backend=EvaluatorBackend.JIT, native_arch=False)
        )

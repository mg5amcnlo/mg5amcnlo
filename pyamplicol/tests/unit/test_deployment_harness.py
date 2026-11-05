# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "release"))

import test_deployment as deployment  # noqa: E402


def test_installed_smoke_uses_the_v1_builtin_model_name() -> None:
    assert "from pyamplicol.models import BuiltinSMModel" in deployment._INSTALLED_SMOKE
    assert "AmplicolSMLeadingColorModel" not in deployment._INSTALLED_SMOKE
    assert 'build_info["selftest_fixture_bootstrap"] is False' in (
        deployment._INSTALLED_SMOKE
    )
    assert 'build_info["publishable"] is True' in deployment._INSTALLED_SMOKE
    assert 'build_info["schema_version"] == 1' in deployment._INSTALLED_SMOKE
    assert 'build_info["source_revision"]' in deployment._INSTALLED_SMOKE


def test_f64_deployment_smoke_hard_blocks_symbolica() -> None:
    smoke = deployment._SYMBOLICA_FREE_F64_SMOKE
    assert 'os.environ.pop("SYMBOLICA_LICENSE", None)' in smoke
    assert "builtins.__import__ = reject_symbolica" in smoke
    assert 'check.name == "physics-f64"' in smoke
    assert 'name.startswith("symbolica.")' in smoke
    assert "for name in sys.modules" in smoke


@pytest.mark.parametrize("name", ["symbolica", "symbolica.core"])
def test_f64_deployment_import_block_models_an_absent_package(name: str) -> None:
    hook = next(
        node
        for node in ast.parse(deployment._SYMBOLICA_FREE_F64_SMOKE).body
        if isinstance(node, ast.FunctionDef) and node.name == "reject_symbolica"
    )
    namespace = {"original_import": __import__}
    exec(compile(ast.Module(body=[hook], type_ignores=[]), "<hook>", "exec"), namespace)
    with pytest.raises(ModuleNotFoundError) as caught:
        namespace["reject_symbolica"](name)
    assert caught.value.name == "symbolica"


def test_f64_minimal_deployment_requires_symbolica_to_be_absent() -> None:
    smoke = deployment._SYMBOLICA_ABSENT_F64_SMOKE
    assert "find_spec(unavailable) is None" in smoke
    assert '("symbolica", "ufo_model_loader")' in smoke
    assert 'runtime.evaluate(expected["momenta"])' in smoke
    assert 'runtime.evaluate_resolved(expected["momenta"])' in smoke
    assert "resolved.total()" in smoke
    assert 'runtime.artifact_id == manifest["artifact_id"]' in smoke
    assert 'runtime.execution_mode == "compiled"' in smoke


def test_f64_minimal_deployment_installs_only_numpy_and_wheel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel = tmp_path / "pyamplicol-test.whl"
    wheel.write_bytes(b"wheel")
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        rendered = [os.fspath(item) for item in command]
        commands.append(rendered)
        return subprocess.CompletedProcess(rendered, 0, "", "")

    monkeypatch.setattr(deployment, "run", fake_run)
    deployment._symbolica_absent_f64_smoke(
        wheel,
        target_python=Path("/host/python"),
        sandbox=tmp_path / "deployment",
        numpy_version="2.4.2",
    )

    pip_installs = [
        command
        for command in commands
        if command[1:5] == ["-I", "-m", "pip", "install"]
    ]
    assert len(pip_installs) == 2
    assert pip_installs[0][-2:] == ["--only-binary=:all:", "numpy==2.4.2"]
    assert "--no-deps" in pip_installs[1]
    assert pip_installs[1][-1] == os.fspath(wheel.resolve())
    assert all("symbolica" not in command for command in pip_installs)
    assert all("ufo-model-loader" not in command for command in pip_installs)
    assert commands[-1][-2:] == ["-c", deployment._SYMBOLICA_ABSENT_F64_SMOKE]


def test_installed_backend_smoke_covers_precision_compiled_and_eager() -> None:
    smoke = deployment._INSTALLED_BACKEND_AND_PRECISION_SMOKE
    assert "EvaluatorBackend.JIT" in smoke
    assert "EvaluatorBackend.ASM" in smoke
    assert "EvaluatorBackend.CPP" in smoke
    assert "precision=80" in smoke
    assert 'Generator(config).generate("d d~ > z", artifact)' in smoke
    assert "resolved.total()[0] == total" in smoke
    assert 'runtime.artifact_id == manifest["artifact_id"]' in smoke
    assert 'runtime.execution_mode == "compiled"' in smoke
    assert "EvaluatorExecutionMode.EAGER" in smoke
    assert 'Generator(eager_config).generate("d d~ > z", eager_artifact)' in smoke
    assert '"pyamplicol-runtime-eager-execution"' in smoke
    assert '"model/eager-kernel-pack.json"' in smoke
    assert "eager_runtime.evaluate(momenta)" in smoke
    assert 'eager_runtime.artifact_id == eager_manifest["artifact_id"]' in smoke
    assert 'eager_runtime.execution_mode == "eager"' in smoke
    assert "abs(eager_reduced - eager_total)" in smoke
    assert "eager_runtime.evaluate(momenta, precision=80)" in smoke


def test_abi3_smoke_skips_duplicate_generation_and_sdk_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel = tmp_path / "pyamplicol-0.1.0-cp311-abi3-test.whl"
    wheel.write_bytes(b"wheel")
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        rendered = [os.fspath(item) for item in command]
        commands.append(rendered)
        return subprocess.CompletedProcess(rendered, 0, "", "")

    def unexpected(*_args, **_kwargs):
        raise AssertionError("full deployment check ran during abi3 smoke")

    monkeypatch.setattr(deployment, "DEPLOYMENT_ROOT", tmp_path / "deployments")
    monkeypatch.setattr(
        deployment,
        "audit_wheel",
        lambda *_args, **_kwargs: SimpleNamespace(version="0.1.0"),
    )
    monkeypatch.setattr(deployment, "run", fake_run)
    monkeypatch.setattr(deployment, "runtime_environment", lambda _venv: {})
    monkeypatch.setattr(deployment, "interpreter_tags", lambda _python: ["abi3"])
    monkeypatch.setattr(
        deployment,
        "_install_dependencies",
        lambda *_args, **_kwargs: deployment.DependencyInstallation(
            {"numpy": "2.4.2"}, {}
        ),
    )
    monkeypatch.setattr(deployment, "_symbolica_absent_f64_smoke", unexpected)
    monkeypatch.setattr(deployment, "_native_sdk_smoke", unexpected)

    sandbox = deployment.test_deployment(
        wheel,
        target_python=Path(sys.executable),
        mode="release",
        wheelhouses=(),
        keep=True,
        abi3_smoke_only=True,
    )

    assert sandbox is not None
    assert any(
        command[-2:] == ["-c", deployment._INSTALLED_SMOKE]
        for command in commands
    )
    assert any(
        command[-2:] == ["-c", deployment._SYMBOLICA_FREE_F64_SMOKE]
        for command in commands
    )
    assert not any(
        deployment._INSTALLED_BACKEND_AND_PRECISION_SMOKE in command
        for command in commands
    )
    assert not any("pyamplicol.selftest" in command for command in commands)
    assert not any(
        "self-test" in command or "examples" in command for command in commands
    )
    result = json.loads(
        (sandbox / "deployment-result.json").read_text(encoding="utf-8")
    )
    assert result["validation_scope"] == "abi3-smoke"
    assert result["native_sdk_validated"] is False


from _common import ReleaseError, clean_environment  # noqa: E402


def test_release_environment_discards_native_toolchain_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    injected = {
        "CPATH": "/opt/local/include",
        "DYLD_LIBRARY_PATH": "/opt/local/lib",
        "LDFLAGS": "-L/opt/local/lib",
        "LIBRARY_PATH": "/opt/local/lib",
        "RUSTFLAGS": "-L/opt/local/lib/libgcc -l dylib=gcc_s",
    }
    for name, value in injected.items():
        monkeypatch.setenv(name, value)

    environment = clean_environment(mode="candidate")

    assert not set(injected) & set(environment)
    assert environment["PYAMPLICOL_BUILD_MODE"] == "candidate"


def test_deployment_copies_the_complete_installed_examples_tree(tmp_path: Path) -> None:
    python = tmp_path / "venv/bin/python"
    destination = tmp_path / "examples"

    command = [
        os.fspath(item)
        for item in deployment._examples_copy_command(python, destination)
    ]

    assert command == [
        os.fspath(python),
        "-I",
        "-m",
        "pyamplicol",
        "examples",
        "copy",
        os.fspath(destination),
    ]


def test_deployment_runs_a_copied_external_ufo_example(tmp_path: Path) -> None:
    python = tmp_path / "venv/bin/python"
    card = tmp_path / "examples/external_ufo_sm.toml"

    command = [
        os.fspath(item)
        for item in deployment._copied_example_command(python, card)
    ]

    assert command == [
        os.fspath(python),
        "-I",
        "-m",
        "pyamplicol",
        os.fspath(card),
        "--set",
        "generation.mode=replace",
        "--json",
    ]


def test_candidate_deployment_builds_fresh_instead_of_reusing_stale_wheel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYAMPLICOL_BUILD_MODE", raising=False)
    retained = tmp_path / "retained"
    retained.mkdir()
    stale = retained / "pyamplicol-0.1.0.dev0+candidate.stale-cp311-abi3-test.whl"
    stale.write_bytes(b"stale")
    scratch = tmp_path / "scratch"
    selected: list[tuple[Path, bool]] = []

    @contextmanager
    def fake_temporary(_prefix: str):
        scratch.mkdir()
        yield scratch

    def fake_run(command, **_kwargs):
        outdir = Path(command[command.index("--outdir") + 1])
        fresh = outdir / "pyamplicol-0.1.0.dev0+candidate.fresh-cp311-abi3-test.whl"
        fresh.write_bytes(b"fresh")
        return subprocess.CompletedProcess(command, 0, "", "")

    def fake_select(wheels, _tags):
        candidates = list(wheels)
        assert len(candidates) == 1
        assert candidates[0].parent == scratch
        assert candidates[0] != stale
        return candidates[0]

    def fake_deployment(wheel, **_kwargs):
        selected.append((wheel, bool(_kwargs["abi3_smoke_only"])))

    monkeypatch.setattr(deployment, "check_dependency_gate", lambda *_a, **_k: None)
    monkeypatch.setattr(deployment, "external_temporary_directory", fake_temporary)
    monkeypatch.setattr(deployment, "run", fake_run)
    monkeypatch.setattr(deployment, "interpreter_tags", lambda _python: ["test"])
    monkeypatch.setattr(deployment, "select_compatible_wheel", fake_select)
    monkeypatch.setattr(deployment, "wheelhouse_directories", lambda _path: [])
    monkeypatch.setattr(deployment, "test_deployment", fake_deployment)

    assert (
        deployment.main(
            [
                "--candidate",
                "--artifact-dir",
                os.fspath(retained),
                "--abi3-smoke-only",
            ]
        )
        == 0
    )
    assert selected and selected[0][0].read_bytes() == b"fresh"
    assert selected[0][1] is True
    assert stale.read_bytes() == b"stale"


def test_release_native_sdk_smoke_requires_all_compilers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYAMPLICOL_REQUIRE_NATIVE_TESTS", raising=False)
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.delenv("FC", raising=False)
    monkeypatch.setattr(deployment.shutil, "which", lambda _name: None)

    with pytest.raises(
        ReleaseError,
        match=r"C\+\+17 and Fortran 2008 and Rust 2021",
    ):
        deployment._native_sdk_smoke(
            Path(sys.executable),
            sandbox=tmp_path,
            mode="release",
            environment={},
        )


def test_candidate_native_sdk_smoke_reports_explicit_nonvalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("PYAMPLICOL_REQUIRE_NATIVE_TESTS", raising=False)
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.delenv("FC", raising=False)
    monkeypatch.setattr(deployment.shutil, "which", lambda _name: None)

    assert (
        deployment._native_sdk_smoke(
            Path(sys.executable),
            sandbox=tmp_path,
            mode="candidate",
            environment={},
        )
        is False
    )
    assert "did not run native SDK smoke tests" in capsys.readouterr().out


def test_candidate_native_sdk_smoke_can_require_all_compilers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYAMPLICOL_REQUIRE_NATIVE_TESTS", "1")
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.delenv("FC", raising=False)
    monkeypatch.setattr(deployment.shutil, "which", lambda _name: None)

    with pytest.raises(ReleaseError, match="native SDK validation requires"):
        deployment._native_sdk_smoke(
            Path(sys.executable),
            sandbox=tmp_path,
            mode="candidate",
            environment={},
        )


def test_native_sdk_smoke_compiles_and_runs_all_four_language_drivers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    include = tmp_path / "sdk/include"
    include.mkdir(parents=True)
    library = tmp_path / "sdk/librusticol_capi.a"
    library.write_bytes(b"archive")
    fortran_module = tmp_path / "sdk/rusticol.f90"
    fortran_module.write_text(
        "module rusticol\nend module rusticol\n", encoding="utf-8"
    )
    rust_module = tmp_path / "sdk/rusticol.rs"
    rust_module.write_text("pub struct Runtime;\n", encoding="utf-8")
    sdk = {
        "cflags": [f"-I{include}"],
        "link_flags": [str(library)],
        "rust_flags": ["-C", f"link-arg={library}"],
        "fortran_source": str(fortran_module),
        "rust_source": str(rust_module),
    }
    artifact = tmp_path / "installed/artifact"
    python_driver = artifact / "API/python/check_standalone.py"
    c_driver = artifact / "API/c/check_standalone.c"
    cpp_driver = artifact / "API/cpp/check_standalone.cpp"
    fortran_driver = artifact / "API/fortran/check_standalone.f90"
    rust_driver = artifact / "API/rust/check_standalone.rs"
    python_driver.parent.mkdir(parents=True)
    c_driver.parent.mkdir(parents=True)
    cpp_driver.parent.mkdir(parents=True)
    fortran_driver.parent.mkdir(parents=True)
    rust_driver.parent.mkdir(parents=True)
    python_driver.write_text("raise SystemExit(0)\n", encoding="utf-8")
    c_driver.write_text("int main(void) { return 0; }\n", encoding="utf-8")
    cpp_driver.write_text("int main() { return 0; }\n", encoding="utf-8")
    fortran_driver.write_text("program main\nend program main\n", encoding="utf-8")
    rust_driver.write_text("fn main() {}\n", encoding="utf-8")
    fixture = deployment.NativePhysicsFixture(
        artifact=artifact,
        process_id="smoke",
        shape=(1, 1, 1),
        totals=(2.5,),
    )
    commands: list[list[str]] = []
    command_environments: list[dict[str, str]] = []

    monkeypatch.setattr(
        deployment,
        "_native_toolchain",
        lambda _mode: deployment.NativeToolchain(
            cc=("/tool/cc",),
            cxx=("/tool/c++",),
            fortran=("/tool/gfortran",),
            rustc=("/tool/rustc",),
        ),
    )
    monkeypatch.setattr(
        deployment,
        "_installed_selftest_fixture",
        lambda *_args, **_kwargs: fixture,
    )

    def fake_run(command, **_kwargs):
        rendered = [os.fspath(item) for item in command]
        commands.append(rendered)
        command_environments.append(dict(_kwargs.get("env", {})))
        if "rusticol_config" in rendered:
            return subprocess.CompletedProcess(rendered, 0, json.dumps(sdk), "")
        language = None
        if any(item.endswith("check_standalone.py") for item in rendered):
            language = "python"
        else:
            for candidate in ("c", "cpp", "fortran", "rust"):
                if rendered[0].endswith(f"check_standalone_{candidate}"):
                    language = candidate
                    break
        if language is not None:
            payload = {
                "language": language,
                "available": True,
                "precision": 16,
                "process_key": "smoke",
                "shape": [1, 1, 1],
                "values": [2.5],
                "resolved_sum": [2.5],
                "compatibility_total": [2.5],
            }
            return subprocess.CompletedProcess(rendered, 0, json.dumps(payload), "")
        return subprocess.CompletedProcess(rendered, 0, "", "")

    monkeypatch.setattr(deployment, "run", fake_run)
    assert (
        deployment._native_sdk_smoke(
            Path(sys.executable),
            sandbox=tmp_path,
            mode="release",
            environment={"PATH": "/deployment/venv/bin:/usr/bin:/bin"},
        )
        is True
    )

    assert any(command[0] == "/tool/cc" for command in commands)
    assert any(command[0] == "/tool/c++" for command in commands)
    assert any(command[0] == "/tool/gfortran" for command in commands)
    assert any(command[0] == "/tool/rustc" for command in commands)
    assert any(command[0] == os.fspath(Path(sys.executable)) for command in commands)
    assert any(command[0].endswith("check_standalone_c") for command in commands)
    assert any(command[0].endswith("check_standalone_cpp") for command in commands)
    assert any(command[0].endswith("check_standalone_fortran") for command in commands)
    assert any(command[0].endswith("check_standalone_rust") for command in commands)
    rust_compile_index = next(
        index for index, command in enumerate(commands) if command[0] == "/tool/rustc"
    )
    assert "rustc" not in command_environments[rust_compile_index]["PATH"]
    assert command_environments[rust_compile_index]["RUSTICOL_RUST_SOURCE"] == str(
        rust_module
    )


def test_five_language_result_comparison_rejects_metadata_drift() -> None:
    common = {
        "available": True,
        "precision": 16,
        "process_key": "smoke",
        "shape": [1, 1, 1],
        "values": [2.5],
        "resolved_sum": [2.5],
        "compatibility_total": [2.5],
    }
    results = {
        language: {"language": language, **common}
        for language in ("python", "c", "cpp", "fortran", "rust")
    }
    results["rust"]["process_key"] = "different"

    with pytest.raises(ReleaseError, match="rust API driver metadata"):
        deployment._compare_driver_results(results)

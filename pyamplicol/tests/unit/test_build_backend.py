# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "build_backend"))

import _pyamplicol_build as backend  # noqa: E402
import sdk  # noqa: E402


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def test_prepared_model_bootstrap_is_candidate_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYAMPLICOL_PREPARED_MODEL_BOOTSTRAP", "1")

    assert backend._prepared_model_bootstrap("candidate") is True
    with pytest.raises(RuntimeError, match="non-publishable candidate"):
        backend._prepared_model_bootstrap("release")


def test_prepared_model_bootstrap_rejects_ambiguous_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYAMPLICOL_PREPARED_MODEL_BOOTSTRAP", "true")

    with pytest.raises(RuntimeError, match="must be either '0' or '1'"):
        backend._prepared_model_bootstrap("candidate")


def test_release_prepared_model_bootstrap_requires_explicit_release_context() -> None:
    assert backend._release_prepared_model_bootstrap("release", None) is False
    with pytest.raises(RuntimeError, match="explicit producer context"):
        backend._release_prepared_model_bootstrap("release", "wrong")
    with pytest.raises(RuntimeError, match="requires release dependency mode"):
        backend._release_prepared_model_bootstrap(
            "candidate",
            backend._RELEASE_PREPARED_MODEL_BOOTSTRAP_CONTEXT,
        )
    assert (
        backend._release_prepared_model_bootstrap(
            "release",
            backend._RELEASE_PREPARED_MODEL_BOOTSTRAP_CONTEXT,
        )
        is True
    )


def test_selftest_fixture_bootstrap_requires_explicit_candidate_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYAMPLICOL_SELFTEST_FIXTURE_BOOTSTRAP", "1")

    with pytest.raises(RuntimeError, match=r"explicit.*context"):
        backend._selftest_fixture_bootstrap("candidate")

    monkeypatch.setenv(
        "PYAMPLICOL_SELFTEST_FIXTURE_BOOTSTRAP_CONTEXT",
        backend._SELFTEST_FIXTURE_BOOTSTRAP_CONTEXT,
    )
    with pytest.raises(RuntimeError, match=r"explicit.*context"):
        backend._selftest_fixture_bootstrap("candidate")
    assert (
        backend._selftest_fixture_bootstrap(
            "candidate",
            explicit_context=True,
        )
        is True
    )
    with pytest.raises(RuntimeError, match="non-publishable candidate"):
        backend._selftest_fixture_bootstrap(
            "release",
            explicit_context=True,
        )


def test_selftest_fixture_bootstrap_rejects_raw_context_without_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "PYAMPLICOL_SELFTEST_FIXTURE_BOOTSTRAP_CONTEXT",
        backend._SELFTEST_FIXTURE_BOOTSTRAP_CONTEXT,
    )

    with pytest.raises(RuntimeError, match="context requires the bootstrap flag"):
        backend._selftest_fixture_bootstrap("candidate")


def test_pep517_rejects_raw_selftest_fixture_bootstrap_escape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYAMPLICOL_BUILD_MODE", "candidate")
    monkeypatch.setenv("PYAMPLICOL_SELFTEST_FIXTURE_BOOTSTRAP", "1")
    monkeypatch.setenv(
        "PYAMPLICOL_SELFTEST_FIXTURE_BOOTSTRAP_CONTEXT",
        backend._SELFTEST_FIXTURE_BOOTSTRAP_CONTEXT,
    )

    with pytest.raises(RuntimeError, match=r"explicit.*context"):
        backend._from_overlay(lambda: None, with_sdk=False)


def test_selftest_fixture_bootstrap_rejects_ambiguous_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYAMPLICOL_SELFTEST_FIXTURE_BOOTSTRAP", "true")

    with pytest.raises(RuntimeError, match="must be either '0' or '1'"):
        backend._selftest_fixture_bootstrap(
            "candidate",
            explicit_context=True,
        )


def test_selftest_fixture_bootstrap_marks_exact_candidate_and_strips_fixture(
    tmp_path: Path,
) -> None:
    package = tmp_path / "src" / "pyamplicol"
    package.mkdir(parents=True)
    build_info_path = package / "_build_info.json"
    build_info_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "publishable": False,
                "selftest_fixture_bootstrap": False,
                "source_revision": "a" * 40,
            }
        ),
        encoding="utf-8",
    )
    fixture = package / "assets" / "selftest" / "portable-64le"
    fixture.mkdir(parents=True)
    (fixture / "expected.json").write_text("{}\n", encoding="utf-8")

    backend._mark_selftest_fixture_bootstrap(tmp_path)

    build_info = json.loads(build_info_path.read_text(encoding="utf-8"))
    assert build_info["selftest_fixture_bootstrap"] is True
    assert not fixture.exists()


def test_selftest_fixture_bootstrap_rejects_dirty_candidate(
    tmp_path: Path,
) -> None:
    package = tmp_path / "src" / "pyamplicol"
    package.mkdir(parents=True)
    (package / "_build_info.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "publishable": False,
                "selftest_fixture_bootstrap": False,
                "source_revision": None,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="exact clean source revision"):
        backend._mark_selftest_fixture_bootstrap(tmp_path)


def test_prepared_model_bootstrap_strips_only_generated_payloads(
    tmp_path: Path,
) -> None:
    root = tmp_path / "src/pyamplicol/assets/prepared_models"
    root.mkdir(parents=True)
    package = root / "__init__.py"
    package.write_text("# package\n", encoding="utf-8")
    metadata = root / "built-in-sm-jit-o2-aarch64.metadata.json"
    metadata.write_text("{}\n", encoding="utf-8")
    bundle = root / "built-in-sm-jit-o2-aarch64.pyamplicol-model"
    bundle.write_bytes(b"bundle")

    backend._strip_prepared_model_payloads(tmp_path)

    assert package.is_file()
    assert not metadata.exists()
    assert not bundle.exists()


def test_linux_release_wheel_requests_manylinux_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYAMPLICOL_BUILD_MODE", "release")
    monkeypatch.setattr(backend.sys, "platform", "linux")
    observed: dict[str, object] = {}

    def delegated(operation, *args, **kwargs):
        observed["operation"] = operation
        observed["args"] = args
        observed["kwargs"] = kwargs
        return "pyamplicol.whl"

    monkeypatch.setattr(backend, "_from_overlay", delegated)

    assert backend.build_wheel("dist") == "pyamplicol.whl"
    assert observed == {
        "operation": backend.maturin.build_wheel,
        "args": (
            "dist",
            {"maturin.build-args": ["--compatibility", "manylinux_2_28"]},
            None,
        ),
        "kwargs": {
            "with_sdk": True,
            "retain_release_build_info": True,
        },
    }


def test_candidate_and_non_linux_wheels_preserve_frontend_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supplied = {"frontend.setting": "preserved"}
    monkeypatch.setenv("PYAMPLICOL_BUILD_MODE", "candidate")
    monkeypatch.setattr(backend.sys, "platform", "linux")
    assert backend._release_wheel_config_settings(supplied) is supplied

    monkeypatch.setenv("PYAMPLICOL_BUILD_MODE", "release")
    monkeypatch.setattr(backend.sys, "platform", "darwin")
    assert backend._release_wheel_config_settings(supplied) is supplied


@pytest.mark.parametrize("mode", ("candidate", "release"))
@pytest.mark.parametrize("platform_name", ("darwin", "linux"))
def test_host_wheel_setting_is_consumed_and_only_changes_linux_tags(
    monkeypatch: pytest.MonkeyPatch, mode: str, platform_name: str
) -> None:
    monkeypatch.setenv("PYAMPLICOL_BUILD_MODE", mode)
    monkeypatch.setattr(backend.sys, "platform", platform_name)
    supplied = {"pyamplicol.host-wheel": "true", "frontend.setting": "preserved"}
    expected = {"frontend.setting": "preserved"}
    if platform_name == "linux":
        expected["maturin.build-args"] = ["--compatibility", "linux"]
    assert backend._release_wheel_config_settings(supplied) == expected
    assert supplied["pyamplicol.host-wheel"] == "true"


def test_false_host_wheel_setting_preserves_release_manylinux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYAMPLICOL_BUILD_MODE", "release")
    monkeypatch.setattr(backend.sys, "platform", "linux")
    assert backend._release_wheel_config_settings(
        {"pyamplicol.host-wheel": "false"}
    ) == {"maturin.build-args": ["--compatibility", "manylinux_2_28"]}


@pytest.mark.parametrize("value", (None, True, "", "yes", "linux", ["true"]))
def test_host_wheel_setting_rejects_invalid_values(value) -> None:
    with pytest.raises(RuntimeError, match="host-wheel must be 'true' or 'false'"):
        backend._release_wheel_config_settings({"pyamplicol.host-wheel": value})


@pytest.mark.parametrize(
    "hook_name", ("get_requires_for_build_wheel", "prepare_metadata_for_build_wheel")
)
def test_wheel_frontend_hooks_do_not_forward_backend_host_setting(
    monkeypatch: pytest.MonkeyPatch, hook_name: str
) -> None:
    observed = []
    monkeypatch.setattr(
        backend,
        "_from_overlay",
        lambda _operation, *args, **_kwargs: observed.extend(args),
    )
    args = ("metadata",) if hook_name == "prepare_metadata_for_build_wheel" else ()
    getattr(backend, hook_name)(
        *args, config_settings={"pyamplicol.host-wheel": "true"}
    )
    assert observed[-1] == {}


@pytest.mark.parametrize("mode", ("candidate", "release"))
@pytest.mark.parametrize("platform_name", ("darwin", "linux"))
@pytest.mark.parametrize("argument_key", ("maturin.build-args", "build-args"))
@pytest.mark.parametrize("host_wheel", (False, True))
def test_authenticated_wheel_rejects_frontend_maturin_overrides(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    platform_name: str,
    argument_key: str,
    host_wheel: bool,
) -> None:
    monkeypatch.setenv("PYAMPLICOL_BUILD_MODE", mode)
    monkeypatch.setattr(backend.sys, "platform", platform_name)

    with pytest.raises(RuntimeError, match="caller-supplied Maturin build arguments"):
        supplied = {argument_key: ["--profile", "dev"]}
        if host_wheel:
            supplied["pyamplicol.host-wheel"] = "true"
        backend._release_wheel_config_settings(supplied)


@pytest.mark.parametrize("platform_name", ("linux", "darwin"))
def test_release_bootstrap_wheel_uses_the_producer_platform(
    monkeypatch: pytest.MonkeyPatch,
    platform_name: str,
) -> None:
    monkeypatch.setenv("PYAMPLICOL_BUILD_MODE", "release")
    monkeypatch.setattr(backend.sys, "platform", platform_name)
    observed: dict[str, object] = {}

    def delegated(operation, *args, **kwargs):
        observed["settings"] = args[1]
        observed["context"] = kwargs["release_prepared_model_bootstrap_context"]
        return "pyamplicol.whl"

    monkeypatch.setattr(backend, "_from_overlay", delegated)
    assert backend.build_release_prepared_model_bootstrap_wheel(
        "dist",
        bootstrap_context="test",
        config_settings={"frontend.setting": "preserved"},
    ) == "pyamplicol.whl"
    expected: dict[str, object] = {"frontend.setting": "preserved"}
    if platform_name == "linux":
        expected["maturin.build-args"] = ["--compatibility", "linux"]
    assert observed == {"settings": expected, "context": "test"}


def test_release_bootstrap_wheel_rejects_frontend_maturin_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYAMPLICOL_BUILD_MODE", "release")
    monkeypatch.setattr(backend.sys, "platform", "darwin")
    with pytest.raises(RuntimeError, match="caller-supplied Maturin build arguments"):
        backend.build_release_prepared_model_bootstrap_wheel(
            "dist",
            bootstrap_context="test",
            config_settings={"build-args": "--features untracked"},
        )


def test_candidate_bootstrap_wheel_strips_stale_generated_assets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("PYAMPLICOL_BUILD_MODE", "candidate")
    monkeypatch.setenv("PYAMPLICOL_PREPARED_MODEL_BOOTSTRAP", "1")
    overlay = tmp_path / "overlay"
    package = overlay / "src/pyamplicol"
    build_info_path = package / "_build_info.json"
    build_info_path.parent.mkdir(parents=True)
    build_info_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "publishable": False,
                "candidate_fingerprint": "b" * 12,
                "native_build_inputs_sha256": "c" * 64,
                "selftest_fixture_bootstrap": False,
                "source_revision": None,
            }
        ),
        encoding="utf-8",
    )
    assets = package / "assets/prepared_models"
    assets.mkdir(parents=True)
    (assets / "__init__.py").write_text("# package\n", encoding="utf-8")
    (assets / "built-in-sm-jit-o2-aarch64.metadata.json").write_text(
        '{"producer": "stale"}\n',
        encoding="utf-8",
    )
    (assets / "built-in-sm-jit-o2-aarch64.pyamplicol-model").write_bytes(b"stale")
    fixture = package / "assets/selftest/portable-64le"
    fixture.mkdir(parents=True)
    (fixture / "expected.json").write_text('{"producer": "stale"}\n')
    target = tmp_path / "target"
    target.mkdir()

    @contextlib.contextmanager
    def fake_overlay(_mode: str):
        yield overlay, target

    def fake_sdk(_root: Path, _target: Path) -> Path:
        staging = tmp_path / "sdk"
        staging.mkdir()
        (staging / "metadata.json").write_text(
            json.dumps({"target": "aarch64-apple-darwin"}),
            encoding="utf-8",
        )
        return staging

    def delegated(*_args, **_kwargs) -> str:
        assert {path.name for path in assets.iterdir()} == {"__init__.py"}
        assert not fixture.exists()
        build_info = json.loads(build_info_path.read_text(encoding="utf-8"))
        assert build_info["source_revision"] is None
        assert build_info["selftest_fixture_bootstrap"] is True
        return "candidate-bootstrap.whl"

    monkeypatch.setattr(backend, "_overlay", fake_overlay)
    monkeypatch.setattr(backend, "_check_dependencies", lambda _mode: None)
    monkeypatch.setattr(backend, "_rust_remap_flags", lambda *_args: "")
    monkeypatch.setattr(backend, "_stage_packaged_examples", lambda _path: None)
    monkeypatch.setattr(backend, "_stage_python_stub", lambda _path: None)
    monkeypatch.setattr(backend, "_stage_runtime_resources", lambda _path: None)
    monkeypatch.setattr(
        backend,
        "stage_packaged_prepared_models",
        lambda *_args: pytest.fail("bootstrap wheel validated stale payloads"),
    )
    monkeypatch.setattr(backend, "build_sdk", fake_sdk)
    monkeypatch.setattr(
        backend,
        "_stage_selftest_fixture",
        lambda *_args: pytest.fail("bootstrap wheel staged a stale self-test fixture"),
    )
    monkeypatch.setattr(backend.maturin, "build_wheel", delegated)

    assert backend.build_wheel(str(tmp_path / "dist")) == "candidate-bootstrap.whl"


def test_release_prepared_model_recovery_strips_selftest_fixture(
    tmp_path: Path,
) -> None:
    package = tmp_path / "src/pyamplicol"
    package.mkdir(parents=True)
    build_info_path = package / "_build_info.json"
    build_info_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "publishable": False,
                "candidate_fingerprint": None,
                "release_prepared_model_bootstrap": True,
                "selftest_fixture_bootstrap": False,
            }
        ),
        encoding="utf-8",
    )
    fixture = package / "assets/selftest/portable-64le"
    fixture.mkdir(parents=True)
    (fixture / "expected.json").write_text("{}\n", encoding="utf-8")

    backend._mark_selftest_fixture_bootstrap(
        tmp_path,
        prepared_model_recovery=True,
    )

    build_info = json.loads(build_info_path.read_text(encoding="utf-8"))
    assert build_info["selftest_fixture_bootstrap"] is True
    assert not fixture.exists()


@pytest.mark.parametrize("mode", ["candidate", "release"])
def test_ordinary_wheel_build_rejects_prepared_model_digest_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
) -> None:
    monkeypatch.setenv("PYAMPLICOL_BUILD_MODE", mode)
    monkeypatch.delenv("PYAMPLICOL_PREPARED_MODEL_BOOTSTRAP", raising=False)
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    target = tmp_path / "target"
    target.mkdir()

    @contextlib.contextmanager
    def fake_overlay(
        actual_mode: str,
        *,
        project_release_prepared_models: bool = False,
    ):
        assert actual_mode == mode
        assert project_release_prepared_models is (mode == "release")
        yield overlay, target

    monkeypatch.setattr(backend, "_overlay", fake_overlay)
    monkeypatch.setattr(backend, "_check_dependencies", lambda _mode: None)
    monkeypatch.setattr(backend, "_rust_remap_flags", lambda *_args: "")
    monkeypatch.setattr(backend, "_stage_release_build_info", lambda *_args: None)
    monkeypatch.setattr(
        backend,
        "_native_build_inputs_digest",
        lambda *_args, **_kwargs: "d" * 64,
    )
    monkeypatch.setattr(
        backend,
        "canonical_package_version",
        lambda _path: "0.1.0",
    )
    monkeypatch.setattr(backend, "_stage_packaged_examples", lambda _path: None)
    monkeypatch.setattr(backend, "_stage_python_stub", lambda _path: None)
    monkeypatch.setattr(backend, "_stage_runtime_resources", lambda _path: None)
    monkeypatch.setattr(
        backend,
        "_strip_prepared_model_payloads",
        lambda _path: pytest.fail("ordinary wheel stripped stale payloads"),
    )

    def reject_stale_pack(_path: Path, _mode: str) -> None:
        raise RuntimeError("prepared model metadata is stale")

    monkeypatch.setattr(
        backend,
        "stage_packaged_prepared_models",
        reject_stale_pack,
    )

    with pytest.raises(
        RuntimeError,
        match="prepared model metadata is stale",
    ):
        backend.build_wheel(str(tmp_path / "dist"))


def _candidate_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    candidate_lock = tmp_path / "Cargo.lock"
    candidate_config = tmp_path / "config.toml"
    installer_state = tmp_path / "install-state.json"
    candidate_lock.write_bytes((ROOT / "Cargo.lock").read_bytes())
    checkout = tmp_path / "dependencies" / "checkouts"
    candidate_config.write_text(
        "[patch.crates-io]\n"
        f'graphica = {{ path = "{checkout / "symbolica/lib/graphica"}" }}\n'
        f'numerica = {{ path = "{checkout / "symbolica/lib/numerica"}" }}\n'
        f'symbolica = {{ path = "{checkout / "symbolica"}" }}\n'
        f'symjit = {{ path = "{checkout / "symjit"}" }}\n',
        encoding="utf-8",
    )
    with (ROOT / "dependencies" / "contributor-lock.toml").open("rb") as stream:
        contributor = tomllib.load(stream)
    sources = {
        "gammaloop": {
            "url": contributor["gammaloop_candidate"]["source_url"],
            "revision": contributor["gammaloop_candidate"]["revision"],
        },
        "ratatui-ffi": {
            "url": contributor["ratatui"]["ffi_repository"],
            "revision": contributor["ratatui"]["ffi_revision"],
        },
        "symbolica": {
            "url": contributor["symbolica"]["source_url"],
            "revision": contributor["symbolica"]["candidate_revision"],
        },
        "symbolica-community": {
            "url": contributor["symbolica"]["community_url"],
            "revision": contributor["symbolica"]["community_revision"],
        },
        "symjit": {
            "url": contributor["symjit"]["repository"],
            "revision": contributor["symjit"]["revision"],
        },
    }
    installer_state.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "publishable": False,
                "sources": sources,
            }
        ),
        encoding="utf-8",
    )
    return candidate_lock, candidate_config, installer_state


def test_candidate_overlay_is_versioned_without_mutating_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_cargo = (ROOT / "Cargo.toml").read_bytes()
    source_lock = (ROOT / "Cargo.lock").read_bytes()
    source_pyproject = (ROOT / "pyproject.toml").read_bytes()
    source_core = (
        ROOT / "rust" / "crates" / "rusticol-core" / "Cargo.toml"
    ).read_bytes()
    candidate_inputs = _candidate_inputs(tmp_path)
    monkeypatch.setattr(
        backend,
        "_candidate_inputs",
        lambda: candidate_inputs,
    )
    monkeypatch.setattr(
        backend,
        "_clean_source_revision",
        lambda **_kwargs: "a" * 40,
    )
    temporary_directory = tmp_path / "overlay-temporary"
    temporary_directory.mkdir()
    cargo_target_directory = tmp_path / "shared-cargo-target"
    base_version = backend.canonical_package_version(ROOT)
    native_digest = backend._native_build_inputs_digest(ROOT)
    candidate_fingerprint = native_digest[:12]
    candidate_version = f"{base_version}-dev.0+candidate.{candidate_fingerprint}"
    with backend._overlay(
        "candidate",
        temporary_directory=temporary_directory,
        cargo_target_directory=cargo_target_directory,
    ) as (overlay, target):
        assert overlay.parent.parent == temporary_directory
        assert target == cargo_target_directory
        cargo = (overlay / "Cargo.toml").read_text(encoding="utf-8")
        lock = (overlay / "Cargo.lock").read_text(encoding="utf-8")
        assert (overlay / ".cargo" / "config.toml").read_bytes() == (
            candidate_inputs[1].read_bytes()
        )
        candidate_version_line = f'version = "{candidate_version}"'
        assert candidate_version_line in cargo
        assert lock.count(candidate_version_line) == 3
        candidate_core = (
            overlay / "rust" / "crates" / "rusticol-core" / "Cargo.toml"
        ).read_text(encoding="utf-8")
        with candidate_inputs[0].open("rb") as stream:
            candidate_packages = tomllib.load(stream)["package"]
        candidate_symbolica = next(
            package["version"]
            for package in candidate_packages
            if package["name"] == "symbolica"
        )
        with (ROOT / "dependencies" / "release-lock.toml").open("rb") as stream:
            release = tomllib.load(stream)
        assert (
            f'symjit = {{ version = "={release["symjit"]["version"]}"' in candidate_core
        )
        candidate_pyproject = (overlay / "pyproject.toml").read_text(encoding="utf-8")
        assert f'"symbolica=={candidate_symbolica}"' in candidate_pyproject
        build_info = json.loads(
            (overlay / "src" / "pyamplicol" / "_build_info.json").read_text(
                encoding="utf-8"
            )
        )
        assert build_info["publishable"] is False
        assert build_info["selftest_fixture_bootstrap"] is False
        assert build_info["candidate_fingerprint"] == candidate_fingerprint
        assert build_info["native_build_inputs_sha256"] == native_digest
        assert build_info["version"] == candidate_version.replace("-dev.", ".dev")
        assert build_info["source_checkout"] == str(ROOT.resolve())
        assert build_info["source_revision"] == "a" * 40
        target = "aarch64-apple-darwin"
        backend._stage_selftest_fixture(overlay, target)
        selftest_manifest = json.loads(
            (
                overlay
                / "src"
                / "pyamplicol"
                / "assets"
                / "selftest"
                / target
                / "artifact"
                / "artifact.json"
            ).read_text(encoding="utf-8")
        )
        assert selftest_manifest["producer"]["version"] == build_info["version"]
        assert selftest_manifest["runtime"]["engine_version"] == build_info["version"]
        compiled_payloads = [
            payload
            for payload in selftest_manifest["payloads"]
            if payload["role"] == "compiled-model"
        ]
        assert len(compiled_payloads) == 1
        compiled_payload = compiled_payloads[0]
        compiled_path = (
            overlay
            / "src"
            / "pyamplicol"
            / "assets"
            / "selftest"
            / target
            / "artifact"
            / compiled_payload["path"]
        )
        compiled_data = compiled_path.read_bytes()
        compiled_model = json.loads(compiled_data)
        assert compiled_model["producer"]["pyamplicol"] == build_info["version"]
        assert compiled_payload["sha256"] == hashlib.sha256(compiled_data).hexdigest()
        assert compiled_payload["size_bytes"] == len(compiled_data)
        assert selftest_manifest["producer"]["target"] == {
            "cpu_features": [],
            "triple": target,
        }
        assert (
            selftest_manifest["extensions"]["artifact_identity"]
            == backend._ARTIFACT_IDENTITY_CONTRACT
        )
        target_payloads = [
            payload["target"]
            for payload in selftest_manifest["payloads"]
            if "target" in payload
        ]
        assert target_payloads
        assert target_payloads == [
            {"cpu_features": [], "triple": target} for _payload in target_payloads
        ]
        expected = json.loads(
            (
                overlay
                / "src"
                / "pyamplicol"
                / "assets"
                / "selftest"
                / target
                / "expected.json"
            ).read_text(encoding="utf-8")
        )
        assert expected["target"] == target
        assert expected["compatible_targets"] == sorted(
            backend._PORTABLE_SELFTEST_TARGETS
        )
        assert not (
            overlay
            / "src"
            / "pyamplicol"
            / "assets"
            / "selftest"
            / backend._PORTABLE_SELFTEST_TEMPLATE
        ).exists()
        assert selftest_manifest["artifact_id"] == backend._runtime_artifact_id(
            selftest_manifest
        )

    assert (ROOT / "Cargo.toml").read_bytes() == source_cargo
    assert (ROOT / "Cargo.lock").read_bytes() == source_lock
    assert (ROOT / "pyproject.toml").read_bytes() == source_pyproject
    assert (
        ROOT / "rust" / "crates" / "rusticol-core" / "Cargo.toml"
    ).read_bytes() == source_core


def test_candidate_overlay_reuses_one_workspace_local_build_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    cache = checkout / ".artifacts/dev-install"
    cache.mkdir(parents=True)
    monkeypatch.setattr(backend, "ROOT", checkout)
    monkeypatch.setenv("PYAMPLICOL_CANDIDATE_CACHE_ROOT", str(cache))
    digests = iter(("a" * 64, "b" * 64))
    observed: list[tuple[Path, str | None]] = []
    monkeypatch.setattr(
        backend,
        "_native_build_inputs_digest",
        lambda *_args, **_kwargs: next(digests),
    )

    def populate(
        source: Path,
        _mode: str,
        *,
        native_build_inputs_sha256: str | None,
        **_kwargs,
    ) -> None:
        source.mkdir(parents=True)
        observed.append((source, native_build_inputs_sha256))

    monkeypatch.setattr(backend, "_populate_overlay", populate)
    paths: list[tuple[Path, Path]] = []
    for _ in range(2):
        with backend._overlay("candidate") as pair:
            paths.append(pair)

    expected = (cache / "project-build/source", cache / "cargo-target")
    assert paths == [expected, expected]
    assert observed == [
        (expected[0], "a" * 64),
        (expected[0], "b" * 64),
    ]
    assert backend._candidate_cache_root("release") is None


def test_candidate_build_cache_rejects_non_workspace_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.setattr(backend, "ROOT", checkout)
    monkeypatch.setenv(
        "PYAMPLICOL_CANDIDATE_CACHE_ROOT",
        str(tmp_path / "outside"),
    )

    with pytest.raises(RuntimeError, match="workspace-local developer cache"):
        backend._candidate_cache_root("candidate")


def test_release_overlay_uses_canonical_git_locked_resolution() -> None:
    source_lock = (ROOT / "Cargo.lock").read_bytes()
    with backend._overlay("release") as (overlay, _target):
        assert (overlay / "Cargo.lock").read_bytes() == source_lock
        assert not (overlay / ".cargo" / "config.toml").exists()
        assert not (overlay / ".pyamplicol-build-dependencies").exists()
        assert not (overlay / "dependencies" / "candidate-Cargo.lock").exists()


def test_release_overlay_skips_contributor_native_digest(monkeypatch) -> None:
    def unexpected_digest(_root: Path) -> str:
        raise AssertionError("release builds must not hash contributor inputs")

    monkeypatch.setattr(backend, "_native_build_inputs_digest", unexpected_digest)
    with backend._overlay("release") as (overlay, _target):
        assert not (overlay / "src/pyamplicol/_build_info.json").exists()


def test_release_overlay_requires_source_store_in_git_checkout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_checkout = tmp_path / "checkout"
    source_checkout.mkdir()
    (source_checkout / ".git").write_text("gitdir: elsewhere\n", encoding="utf-8")

    def copy_without_store(destination: Path) -> None:
        destination.mkdir(parents=True)

    monkeypatch.setattr(backend, "ROOT", source_checkout)
    monkeypatch.setattr(backend, "_copy_allowlisted_source", copy_without_store)
    monkeypatch.setattr(backend, "_stage_cargo_inputs", lambda *_args, **_kwargs: None)

    with (
        pytest.raises(RuntimeError, match="source store is missing"),
        backend._overlay("release", project_release_prepared_models=True),
    ):
        pass


def test_release_overlay_reuses_canonical_assets_from_source_archive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_archive = tmp_path / "archive"
    source_archive.mkdir()

    def copy_canonical_release_assets(destination: Path) -> None:
        asset = destination / "src/pyamplicol/assets/prepared_models/release.bin"
        asset.parent.mkdir(parents=True)
        asset.write_bytes(b"canonical release payload")

    monkeypatch.setattr(backend, "ROOT", source_archive)
    monkeypatch.setattr(
        backend,
        "_copy_allowlisted_source",
        copy_canonical_release_assets,
    )
    monkeypatch.setattr(backend, "_stage_cargo_inputs", lambda *_args, **_kwargs: None)

    with backend._overlay(
        "release",
        project_release_prepared_models=True,
    ) as (overlay, _target):
        assert (
            overlay / "src/pyamplicol/assets/prepared_models/release.bin"
        ).read_bytes() == b"canonical release payload"
        assert not (overlay / "release_assets").exists()


def test_release_prepared_model_bootstrap_overlay_is_non_publishable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        backend,
        "_native_build_inputs_digest",
        lambda _root, **_kwargs: "c" * 64,
    )
    source_lock = (ROOT / "Cargo.lock").read_bytes()

    with backend._overlay(
        "release",
        release_prepared_model_bootstrap=True,
    ) as (overlay, _target):
        build_info = json.loads(
            (overlay / "src/pyamplicol/_build_info.json").read_text(encoding="utf-8")
        )

        assert build_info == {
            "candidate_fingerprint": None,
            "native_build_inputs_sha256": "c" * 64,
            "publishable": False,
            "release_prepared_model_bootstrap": True,
            "schema_version": 1,
            "selftest_fixture_bootstrap": False,
            "source_checkout": str(ROOT.resolve()),
            "version": backend.canonical_package_version(ROOT),
        }
        assert (overlay / "Cargo.lock").read_bytes() == source_lock
        assert not (overlay / ".cargo/config.toml").exists()


def test_release_prepared_model_bootstrap_does_not_require_git_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_revision() -> None:
        raise AssertionError("release bootstrap must not inspect Git identity")

    monkeypatch.setattr(backend, "_clean_source_revision", unexpected_revision)
    monkeypatch.setattr(
        backend,
        "_native_build_inputs_digest",
        lambda _root, **_kwargs: "c" * 64,
    )

    with backend._overlay(
        "release",
        release_prepared_model_bootstrap=True,
    ) as (overlay, _target):
        build_info = json.loads(
            (overlay / "src/pyamplicol/_build_info.json").read_text(encoding="utf-8")
        )
    assert "source_revision" not in build_info
    assert build_info["native_build_inputs_sha256"] == "c" * 64


def test_release_build_info_is_minimal_and_retained_from_sdist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    overlay = tmp_path / "checkout-overlay"
    monkeypatch.setattr(backend, "ROOT", checkout)
    monkeypatch.setattr(backend, "_clean_source_revision", lambda: "a" * 40)

    backend._stage_release_build_info(overlay, "0.1.0")
    marker = overlay / "src/pyamplicol/_build_info.json"
    expected = {
        "schema_version": 1,
        "publishable": True,
        "selftest_fixture_bootstrap": False,
        "source_revision": "a" * 40,
        "version": "0.1.0",
    }
    assert json.loads(marker.read_text(encoding="utf-8")) == expected
    retained = marker.read_bytes()

    source_archive = tmp_path / "source-archive"
    source_archive.mkdir()
    archive_overlay = tmp_path / "archive-overlay"
    archive_marker = archive_overlay / "src/pyamplicol/_build_info.json"
    archive_marker.parent.mkdir(parents=True)
    archive_marker.write_bytes(retained)
    monkeypatch.setattr(backend, "ROOT", source_archive)

    backend._stage_release_build_info(archive_overlay, "0.1.0")
    assert archive_marker.read_bytes() == retained


def test_release_build_info_rejects_an_expanded_archive_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_archive = tmp_path / "source-archive"
    source_archive.mkdir()
    overlay = tmp_path / "overlay"
    marker = overlay / "src/pyamplicol/_build_info.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "publishable": True,
                "selftest_fixture_bootstrap": False,
                "source_revision": "a" * 40,
                "source_checkout": "/private/checkout",
                "version": "0.1.0",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(backend, "ROOT", source_archive)

    with pytest.raises(RuntimeError, match="invalid build provenance"):
        backend._stage_release_build_info(overlay, "0.1.0")


def test_selftest_staging_rejects_an_unavailable_target() -> None:
    with (
        backend._overlay("release") as (overlay, _target),
        pytest.raises(RuntimeError, match="unsupported self-test target"),
    ):
        backend._stage_selftest_fixture(overlay, "x86_64-unknown-test")


def test_selftest_staging_applies_the_arena_execution_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Path] = []

    def reject(
        fixture: Path,
        manifest: object,
        *,
        source_root: Path,
    ) -> None:
        assert isinstance(manifest, dict)
        assert source_root.name == "source"
        calls.append(fixture)
        raise RuntimeError("Arena execution contract sentinel")

    monkeypatch.setattr(
        backend,
        "_validate_portable_selftest_executions",
        reject,
    )
    with (
        backend._overlay("release") as (overlay, _target),
        pytest.raises(RuntimeError, match="Arena execution contract sentinel"),
    ):
        backend._stage_selftest_fixture(overlay, "aarch64-apple-darwin")

    assert len(calls) == 1
    assert calls[0].name == "aarch64-apple-darwin"


@pytest.mark.parametrize("target", sorted(backend._PORTABLE_SELFTEST_TARGETS))
def test_portable_selftest_stages_for_every_release_target(target: str) -> None:
    with backend._overlay("release") as (overlay, _target):
        backend._stage_selftest_fixture(overlay, target)
        fixture_root = overlay / "src" / "pyamplicol" / "assets" / "selftest"
        assert tuple(path.name for path in fixture_root.iterdir()) == (target,)
        manifest = json.loads(
            (fixture_root / target / "artifact" / "artifact.json").read_text(
                encoding="utf-8"
            )
        )
        assert manifest["producer"]["target"]["triple"] == target
        assert {
            payload["target"]["triple"]
            for payload in manifest["payloads"]
            if "target" in payload
        } == {target}


def test_cargo_overlay_rejects_unknown_resolution_mode(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="unsupported Cargo input mode"):
        backend._stage_cargo_inputs(
            tmp_path,
            "mixed",
            native_build_inputs_sha256="0" * 64,
        )


def test_candidate_inputs_fail_closed_when_installer_has_not_run() -> None:
    expected = (
        ROOT / "dependencies" / "candidate-Cargo.lock",
        ROOT / "dependencies" / "candidate-cargo-config.toml",
        ROOT / "dependencies" / "install-state.json",
    )
    if all(path.is_file() for path in expected):
        pytest.skip("candidate environment is installed")
    with pytest.raises(RuntimeError, match="dev-install"):
        backend._candidate_inputs()


def test_candidate_source_distributions_are_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PYAMPLICOL_BUILD_MODE", "candidate")
    with pytest.raises(RuntimeError, match="wheel-only"):
        backend.build_sdist(str(tmp_path))


def test_overlay_excludes_managed_dependencies_and_includes_licenses() -> None:
    with backend._overlay("release") as (overlay, _target):
        assert (overlay / "licenses" / "MadGraph5_aMCatNLO.txt").is_file()
        assert (overlay / "rust-toolchain.toml").is_file()
        assert (overlay / "tools" / "typing" / "check_public_typing.py").is_file()
        assert (overlay / "dependencies" / "release-lock.toml").is_file()
        assert not (overlay / "dependencies" / "checkouts").exists()
        assert not (overlay / "dependencies" / "contributor-lock.toml").exists()
        assert not (overlay / "dependencies" / "install_dependencies.py").exists()
        assert not (overlay / "dependencies" / "install-state.json").exists()
        assert not (overlay / "dependencies" / "patches").exists()
        assert not (overlay / "dependencies" / "python-runtime-lock.toml").exists()
        assert not (overlay / "build_backend" / "python_lock.py").exists()


def test_wheel_examples_are_staged_from_the_single_canonical_tree() -> None:
    with backend._overlay("release") as (overlay, _target):
        packaged = overlay / "src" / "pyamplicol" / "_examples"
        assert not packaged.exists()
        backend._stage_packaged_examples(overlay)
        assert (packaged / "all_options.toml").is_file()
        assert (packaged / "builtin_sm_heft.toml").is_file()
        assert (packaged / "qq_z6g_recurrence_jit_o2.toml").is_file()
        assert (packaged / "qq_z6g_compiled_jit_o3.toml").is_file()
        assert (packaged / "qq_z6g_eager_jit_o2.toml").is_file()
        assert (packaged / "otf_pp_zjj.toml").is_file()
        assert (packaged / "python/otf_pp_zjj_warm_up.py").is_file()
        assert (packaged / "native" / "runtime.cpp").is_file()
        assert (packaged / "all_options.toml").read_bytes() == (
            overlay / "examples" / "all_options.toml"
        ).read_bytes()
        with pytest.raises(RuntimeError, match="already contains"):
            backend._stage_packaged_examples(overlay)


def test_wheel_stages_the_single_maintained_rusticol_stub() -> None:
    with backend._overlay("release") as (overlay, _target):
        source = (
            overlay
            / "rust"
            / "crates"
            / "rusticol-python"
            / "stubs"
            / "pyamplicol"
            / "_rusticol.pyi"
        )
        target = overlay / "src" / "pyamplicol" / "_rusticol.pyi"
        assert source.is_file()
        assert not target.exists()
        backend._stage_python_stub(overlay)
        assert target.read_bytes() == source.read_bytes()
        with pytest.raises(RuntimeError, match="already contains"):
            backend._stage_python_stub(overlay)


def test_wheel_stages_runtime_resources_inside_package_namespace(
    tmp_path: Path,
) -> None:
    overlay = tmp_path / "overlay"
    for relative in (
        Path("schemas/README.md"),
        Path("schemas/artifact-manifest-v3.schema.json"),
        Path("schemas/runtime-physics-v1.schema.json"),
    ):
        target = overlay / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())

    backend._stage_runtime_resources(overlay)
    package_assets = overlay / "src" / "pyamplicol" / "assets"
    assert not (package_assets / "release").exists()
    assert (
        package_assets / "schemas" / "artifact-manifest-v3.schema.json"
    ).read_bytes() == (
        overlay / "schemas" / "artifact-manifest-v3.schema.json"
    ).read_bytes()
    assert (package_assets / "schemas" / "runtime-physics-v1.schema.json").is_file()
    with pytest.raises(RuntimeError, match="already contains"):
        backend._stage_runtime_resources(overlay)


def test_overlay_rejects_symlinked_inputs(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_text("content", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(RuntimeError, match="symlinks"):
        backend._reject_symlinks(link)


def test_overlay_prunes_excluded_directories_before_symlink_walk(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    excluded = source / "checkouts"
    excluded.mkdir(parents=True)
    (excluded / "escape").symlink_to(tmp_path)
    backend._reject_symlinks(source)


def test_git_overlay_copies_only_tracked_files_and_prunes_ignored_symlinks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", source], check=True)
    subprocess.run(
        ["git", "-C", str(source), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(source), "config", "user.name", "Test"],
        check=True,
    )
    tracked = source / "build_backend" / "tracked.py"
    tracked.parent.mkdir()
    tracked.write_text("tracked = True\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(source), "add", "build_backend/tracked.py"],
        check=True,
    )
    subprocess.run(["git", "-C", str(source), "commit", "-qm", "tracked"], check=True)
    (tracked.parent / "untracked.py").write_text("leak = True\n", encoding="utf-8")
    excluded = source / "dependencies" / "checkouts"
    excluded.mkdir(parents=True)
    (excluded / "escape").symlink_to(tmp_path)

    monkeypatch.setattr(backend, "ROOT", source)
    destination = tmp_path / "destination"
    backend._copy_allowlisted_source(destination)
    assert (destination / "build_backend" / "tracked.py").is_file()
    assert not (destination / "build_backend" / "untracked.py").exists()
    assert not (destination / "dependencies" / "checkouts").exists()


def test_archive_overlay_without_git_history_uses_pruned_allowlist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "archive"
    retained = {
        Path("build_backend/backend.py"): "archive = True\n",
        Path("docs/user/report.tex"): "maintained TeX\n",
        Path("release_assets/prepared_models/README.md"): "release store\n",
        Path("src/pyamplicol/_sdk/config.py"): "maintained SDK config\n",
        Path("src/rusticol_config/__init__.py"): "standalone SDK config\n",
        Path("tests/fixtures/candidate-Cargo.lock"): "fixture lock\n",
    }
    excluded = (
        Path("dependencies/candidate-Cargo.lock"),
        Path("dependencies/candidate-cargo-config.toml"),
        Path("dependencies/contributor-lock.toml"),
        Path("dependencies/install_dependencies.py"),
        Path("dependencies/install-state.json"),
        Path("dependencies/patches/dependency/fix.patch"),
        Path("dependencies/python-runtime-lock.toml"),
        Path("build_backend/python_lock.py"),
        Path("docs/.result_outputs/cache.json"),
        Path("docs/archive/retired.md"),
        Path("docs/user/report.aux"),
        Path("docs/user/report.synctex.gz"),
        Path("docs/user/report.toc"),
        Path("src/pyamplicol.egg-info/PKG-INFO"),
        Path("src/pyamplicol/_rusticol.abi3.so"),
        Path("src/pyamplicol/_rusticol.dylib"),
        Path("src/pyamplicol/_rusticol.pyd"),
        Path("src/pyamplicol/_sdk/fortran/rusticol.f90"),
        Path("src/pyamplicol/_sdk/include/rusticol.h"),
        Path("src/pyamplicol/_sdk/lib/librusticol_capi.a"),
        Path("src/pyamplicol/_sdk/link.json"),
        Path("src/pyamplicol/_sdk/metadata.json"),
        Path("tests/.artifacts/build.json"),
        Path("tests/build/output.txt"),
        Path("tests/htmlcov/index.html"),
    )
    for relative, content in retained.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    for relative in excluded:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("generated\n", encoding="utf-8")
    ignored = source / "build_backend" / "__pycache__"
    ignored.mkdir()
    (ignored / "escape").symlink_to(tmp_path)
    monkeypatch.setattr(backend, "ROOT", source)

    destination = tmp_path / "destination"
    backend._copy_allowlisted_source(destination)
    for relative, content in retained.items():
        assert (destination / relative).read_text(encoding="utf-8") == content
    for relative in excluded:
        assert not (destination / relative).exists()
    assert not (destination / "build_backend" / "__pycache__").exists()

    copy_ignore = backend._copy_ignore(source)
    assert copy_ignore(
        str(source / "docs" / "arxiv"),
        ["pyAmpliCol.aux", "pyAmpliCol.tex"],
    ) == {"pyAmpliCol.aux"}
    assert copy_ignore(
        str(source / "src" / "pyamplicol" / "_sdk"),
        ["config.py", "lib", "metadata.json"],
    ) == {"lib", "metadata.json"}


def test_release_prepared_store_is_outside_native_build_identity(
    tmp_path: Path,
) -> None:
    cargo = tmp_path / "Cargo.toml"
    cargo.write_text("[workspace]\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_bytes(
        (ROOT / "pyproject.toml").read_bytes()
    )
    before = backend._native_build_inputs_digest(tmp_path)
    store = tmp_path / "release_assets" / "prepared_models"
    store.mkdir(parents=True)
    (store / "built-in-sm-jit-o2-aarch64.metadata.json").write_text(
        "{}\n",
        encoding="utf-8",
    )

    assert backend._native_build_inputs_digest(tmp_path) == before

    native = tmp_path / "rust" / "native.rs"
    native.parent.mkdir()
    native.write_text("fn changed() {}\n", encoding="utf-8")
    assert backend._native_build_inputs_digest(tmp_path) != before


@pytest.mark.parametrize(
    ("hook", "arguments", "expected", "with_sdk"),
    [
        ("build_wheel", ("wheel",), "delegated", True),
        ("build_sdist", ("sdist",), "delegated", False),
        ("get_requires_for_build_wheel", (), ["delegated"], False),
        ("get_requires_for_build_sdist", (), ["delegated"], False),
        ("prepare_metadata_for_build_wheel", ("metadata",), "delegated", False),
    ],
)
def test_retained_pep517_hooks_use_gate_overlay_and_clean_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    hook: str,
    arguments: tuple[str, ...],
    expected: str | list[str],
    with_sdk: bool,
) -> None:
    # This test exercises the publishable hook contract independently of the
    # candidate mode used by the enclosing contributor source gate.
    monkeypatch.setenv("PYAMPLICOL_BUILD_MODE", "release")
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    target = tmp_path / "cargo-target"
    gates: list[str] = []
    sdk_stages: list[Path] = []
    selftest_stages: list[tuple[Path, str]] = []
    prepared_model_stages: list[tuple[Path, str]] = []
    injected_bins = (tmp_path / "injected-bin", tmp_path / "injected-tools")
    for directory in injected_bins:
        directory.mkdir()
    injected = {
        "CARGO": "/attacker/cargo",
        "CARGO_BUILD_TARGET": "attacker-target",
        "CC": "/attacker/cc",
        "CPATH": "/opt/local/include",
        "CXX": "/attacker/c++",
        "DYLD_LIBRARY_PATH": "/opt/local/lib",
        "GIT_INDEX_FILE": "/attacker/index",
        "LIBRARY_PATH": "/opt/local/lib",
        "MATURIN_PEP517_ARGS": "--target attacker-target",
        "PYO3_PYTHON": "/attacker/python",
        "PYTHONPATH": "/attacker/pythonpath",
        "RUSTC_WRAPPER": "/attacker/wrapper",
        "RUSTFLAGS": "-Clink-arg=/attacker/library",
    }
    for name, value in injected.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv(
        "PATH",
        os.pathsep.join(str(path) for path in injected_bins)
        + os.pathsep
        + os.environ["PATH"],
    )

    @contextlib.contextmanager
    def fake_overlay(
        mode: str,
        *,
        project_release_prepared_models: bool = False,
    ):
        assert mode == "release"
        assert project_release_prepared_models is True
        yield overlay, target

    def fake_sdk(root: Path, target_dir: Path) -> Path:
        assert root == overlay and target_dir == target
        assert not (set(injected) - {"CC", "CXX"}) & set(os.environ)
        assert not {str(path) for path in injected_bins} & set(
            os.environ["PATH"].split(os.pathsep)
        )
        assert "CARGO_ENCODED_RUSTFLAGS" in os.environ
        assert os.environ["RUSTUP_TOOLCHAIN"] == "1.97.1"
        if sys.platform == "darwin":
            assert os.environ["CC"] == "/usr/bin/clang"
            assert os.environ["CXX"] == "/usr/bin/clang++"
        else:
            assert not {"CC", "CXX"} & set(os.environ)
        staging = tmp_path / "sdk"
        staging.mkdir()
        (staging / "metadata.json").write_text(
            json.dumps({"target": "aarch64-apple-darwin"}), encoding="utf-8"
        )
        sdk_stages.append(staging)
        return staging

    def delegated(*_args, **_kwargs):
        assert Path.cwd() == overlay
        assert not (set(injected) - {"CC", "CXX"}) & set(os.environ)
        assert not {str(path) for path in injected_bins} & set(
            os.environ["PATH"].split(os.pathsep)
        )
        remaps = os.environ["CARGO_ENCODED_RUSTFLAGS"].split("\x1f")
        assert len(remaps) == 5
        assert all(flag.startswith("--remap-path-prefix=") for flag in remaps)
        assert (
            f"--remap-path-prefix={(ROOT / 'dependencies' / 'checkouts').resolve()}="
            "/pyamplicol/dependencies"
        ) in remaps
        assert os.environ["CARGO_HOME"] == str(tmp_path / "cargo-home")
        assert os.environ["CARGO_TARGET_DIR"] == str(target)
        assert os.environ["PYAMPLICOL_BUILD_OVERLAY"] == str(overlay)
        assert os.environ["PYAMPLICOL_NATIVE_BUILD_INPUTS_SHA256"] == (
            backend._native_build_inputs_digest(
                ROOT,
                normalize_release_cargo_lock=True,
            )
        )
        assert os.environ["RUSTUP_TOOLCHAIN"] == "1.97.1"
        if sys.platform == "darwin":
            assert os.environ["CC"] == "/usr/bin/clang"
            assert os.environ["CXX"] == "/usr/bin/clang++"
            assert os.environ["MACOSX_DEPLOYMENT_TARGET"] == "11.0"
        else:
            assert not {"CC", "CXX"} & set(os.environ)
        if with_sdk:
            assert os.environ["PYAMPLICOL_SDK_STAGING"] == str(tmp_path / "sdk")
        return ["delegated"] if hook.startswith("get_requires") else "delegated"

    monkeypatch.setattr(backend, "_overlay", fake_overlay)
    monkeypatch.setattr(backend, "_check_dependencies", gates.append)
    build_info_stages: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        backend,
        "_stage_release_build_info",
        lambda path, version: build_info_stages.append((path, version)),
    )
    monkeypatch.setattr(
        backend,
        "canonical_package_version",
        lambda _path: "0.1.0",
    )
    monkeypatch.setattr(backend, "_stage_packaged_examples", lambda path: None)
    monkeypatch.setattr(backend, "_stage_python_stub", lambda path: None)
    monkeypatch.setattr(backend, "_stage_runtime_resources", lambda path: None)
    monkeypatch.setattr(
        backend,
        "stage_packaged_prepared_models",
        lambda path, mode: prepared_model_stages.append((path, mode)),
    )
    monkeypatch.setattr(
        backend,
        "_stage_selftest_fixture",
        lambda path, target_name: selftest_stages.append((path, target_name)),
    )
    monkeypatch.setattr(backend, "build_sdk", fake_sdk)
    monkeypatch.setattr(backend.maturin, hook, delegated)

    result = getattr(backend, hook)(*arguments)
    assert result == expected
    output = capsys.readouterr().out
    assert ("* Building native C SDK" in output) is with_sdk
    assert ("* Native C SDK built." in output) is with_sdk
    assert gates == ["release"]
    assert bool(sdk_stages) is with_sdk
    assert selftest_stages == ([(overlay, "aarch64-apple-darwin")] if with_sdk else [])
    assert prepared_model_stages == (
        [(overlay, "release")] if with_sdk or hook == "build_sdist" else []
    )
    assert build_info_stages == (
        [(overlay, "0.1.0")] if hook in {"build_wheel", "build_sdist"} else []
    )
    for name, value in injected.items():
        assert os.environ[name] == value


def test_backend_does_not_advertise_pep660_hooks() -> None:
    editable_hooks = (
        "build_editable",
        "get_requires_for_build_editable",
        "prepare_metadata_for_build_editable",
    )
    assert not [name for name in editable_hooks if hasattr(backend, name)]


def test_sdk_build_references_the_python_owned_safe_rust_wrapper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    archive = tmp_path / "librusticol_capi.a"
    archive.write_bytes(b"static archive")
    link = {
        "schema_version": 1,
        "target": "aarch64-apple-darwin",
        "system_libraries": ["System"],
        "frameworks": [],
    }
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout="",
            stderr="",
        )

    monkeypatch.setattr(sdk, "_host_target", lambda _root: link["target"])
    monkeypatch.setattr(sdk, "_requested_target", lambda host: host)
    monkeypatch.setattr(sdk, "_cargo_fetch", lambda *_args: None)
    monkeypatch.setattr(sdk.subprocess, "run", fake_run)
    monkeypatch.setattr(sdk, "_cargo_messages", lambda _stdout: ())
    monkeypatch.setattr(sdk, "_static_library", lambda _messages: archive)
    monkeypatch.setattr(sdk, "_native_tokens", lambda _stderr: ())
    monkeypatch.setattr(sdk, "_typed_link_arguments", lambda _tokens, _target: link)
    monkeypatch.setattr(sdk, "_scan_archive", lambda _archive: None)
    monkeypatch.setattr(sdk, "_scan_archive_symbols", lambda _archive: None)
    monkeypatch.setattr(
        sdk, "_validate_archive_linkage", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(sdk, "_package_version", lambda _root: "0.1.0")
    stale = tmp_path / "wheel-data/_sdk/stale.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("stale\n", encoding="utf-8")

    staging = sdk.build_sdk(ROOT, tmp_path / "target")

    assert not stale.exists()
    assert not (staging / "rust" / "rusticol.rs").exists()
    assert (ROOT / sdk.RUST_SDK_SOURCE).is_file()
    contract = backend.load_native_build_contract(ROOT)["sdk"]
    assert commands == [
        [
            "cargo",
            "rustc",
            "--locked",
            "--offline",
            "--release",
            "--package",
            contract["package"],
            "--target",
            link["target"],
            "--message-format=json-render-diagnostics",
            "--",
            "--print",
            "native-static-libs",
            *contract["rustc-codegen-arguments"],
        ]
    ]
    metadata = json.loads((staging / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["rust_source"] == "rust/rusticol.rs"


def test_python_extension_rebuilds_and_refreshes_embedded_sdk() -> None:
    build_script = (
        ROOT / "rust/crates/rusticol-python/build.rs"
    ).read_text(encoding="utf-8")

    assert "PYAMPLICOL_NATIVE_BUILD_INPUTS_SHA256" in build_script
    assert "remove_dir_all(&sdk_output)" in build_script


def test_sdk_build_restores_maturin_pruned_sdist_workspace_member(
    tmp_path: Path,
) -> None:
    cargo = tmp_path / "Cargo.toml"
    cargo.write_text(
        """\
[workspace]
members = ["rust/crates/rusticol-core", "rust/crates/rusticol-python"]
resolver = "2"

[workspace.package]
version = "0.1.0"
""",
        encoding="utf-8",
    )
    capi = tmp_path / sdk.RUST_SDK_WORKSPACE_MEMBER / "Cargo.toml"
    capi.parent.mkdir(parents=True)
    capi.write_text('[package]\nname = "rusticol-capi"\n', encoding="utf-8")

    sdk._ensure_sdk_workspace_member(tmp_path)

    with cargo.open("rb") as stream:
        members = tomllib.load(stream)["workspace"]["members"]
    assert members == [
        "rust/crates/rusticol-core",
        "rust/crates/rusticol-python",
        "rust/crates/rusticol-capi",
    ]
    restored = cargo.read_bytes()
    sdk._ensure_sdk_workspace_member(tmp_path)
    assert cargo.read_bytes() == restored


def test_dependency_gate_subprocess_ignores_inherited_build_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CARGO", "/attacker/cargo")
    monkeypatch.setenv("PYTHONPATH", "/attacker/pythonpath")
    monkeypatch.setenv("RUSTFLAGS", "-Clink-arg=/attacker/library")
    observed: dict[str, object] = {}

    def fake_run(command, *, cwd, env, check):
        observed.update(command=command, cwd=cwd, env=env, check=check)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(backend.subprocess, "run", fake_run)
    backend._check_dependencies("candidate")
    command = observed["command"]
    environment = observed["env"]
    assert isinstance(command, list) and command[1] == "-I"
    assert command[-1] == "--candidate"
    assert "--offline" not in command
    assert isinstance(environment, dict)
    assert not {"CARGO", "PYTHONPATH", "RUSTFLAGS"} & set(environment)


def test_build_tool_path_retains_isolated_interpreter_bin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    environment_bin = tmp_path / "build-env" / "bin"
    environment_bin.mkdir(parents=True)
    interpreter = environment_bin / "python"
    interpreter.symlink_to(Path(sys.executable).resolve())
    tool_bin = tmp_path / "rust" / "bin"
    tool_bin.mkdir(parents=True)

    monkeypatch.setattr(backend.sys, "executable", str(interpreter))
    monkeypatch.setattr(
        backend.shutil,
        "which",
        lambda executable, *, path: str(tool_bin / executable),
    )

    result = backend._build_tool_path("").split(os.pathsep)

    assert str(environment_bin) in result


def test_build_tool_path_retains_pip_isolated_maturin_bin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base_bin = tmp_path / "base-python" / "bin"
    base_bin.mkdir(parents=True)
    interpreter = base_bin / "python"
    interpreter.touch()
    isolated_bin = tmp_path / "pip-build-env" / "overlay" / "bin"
    isolated_bin.mkdir(parents=True)
    maturin_script = isolated_bin / "maturin"
    maturin_script.touch()
    rust_bin = tmp_path / "rust" / "bin"
    rust_bin.mkdir(parents=True)

    def locate(executable: str, *, path: str) -> str | None:
        if executable in {"cargo", "rustc"}:
            return str(rust_bin / executable)
        assert path.startswith("/usr/bin" + os.pathsep)
        return f"/usr/bin/{executable}"

    monkeypatch.setattr(backend.sys, "executable", str(interpreter))
    monkeypatch.setattr(
        backend,
        "_maturin_console_script",
        lambda: maturin_script,
    )
    monkeypatch.setattr(backend.shutil, "which", locate)

    result = backend._build_tool_path(
        os.pathsep.join(("/opt/homebrew/bin", str(isolated_bin)))
    ).split(os.pathsep)

    assert str(base_bin) in result
    assert str(isolated_bin) in result
    assert str(rust_bin) in result
    assert "/opt/homebrew/bin" not in result


def test_build_tool_path_does_not_require_git_for_unpacked_sdist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tool_bin = tmp_path / "tools"
    tool_bin.mkdir()

    def locate(executable: str, *, path: str) -> str | None:
        del path
        if executable == "git":
            return None
        if executable in {"cargo", "rustc"}:
            return str(tool_bin / executable)
        return None

    monkeypatch.setattr(backend.shutil, "which", locate)

    result = backend._build_tool_path("").split(os.pathsep)

    assert str(tool_bin) in result


def test_macos_native_build_updates_select_apple_clang(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(backend.sys, "platform", "darwin")

    assert backend._macos_native_build_updates() == {
        "CC": "/usr/bin/clang",
        "CXX": "/usr/bin/clang++",
        "MACOSX_DEPLOYMENT_TARGET": "11.0",
    }

    monkeypatch.setattr(backend.sys, "platform", "linux")
    assert backend._macos_native_build_updates() == {}


def test_rust_remap_flags_uses_repository_toolchain_in_clean_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("RUSTUP_TOOLCHAIN", "nightly")
    monkeypatch.setenv("RUSTFLAGS", "-Clink-arg=/attacker/library")
    monkeypatch.setenv("RUSTC_WRAPPER", "/attacker/wrapper")
    sysroot = tmp_path / "rustup" / "toolchains" / "1.97.1"
    observed: dict[str, object] = {}

    def fake_run(command, *, check, capture_output, text, env):
        observed.update(
            command=command,
            check=check,
            capture_output=capture_output,
            text=text,
            env=env,
        )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=f"{sysroot}\n",
            stderr="",
        )

    monkeypatch.setattr(backend.subprocess, "run", fake_run)

    flags = backend._rust_remap_flags(
        tmp_path / "overlay",
        tmp_path / "build" / "cargo-target",
    )

    environment = observed["env"]
    assert isinstance(environment, dict)
    assert observed["command"] == ["rustc", "--print", "sysroot"]
    assert environment["RUSTUP_TOOLCHAIN"] == "1.97.1"
    assert "RUSTFLAGS" not in environment
    assert "RUSTC_WRAPPER" not in environment
    assert f"{sysroot.resolve()}=/rust/sysroot" in flags
    assert (
        f"{(backend.ROOT / 'dependencies' / 'checkouts').resolve()}="
        "/pyamplicol/dependencies"
    ) in flags


def test_build_tool_path_does_not_expose_base_python_package_manager_tools(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    environment_bin = tmp_path / "build-env" / "bin"
    environment_bin.mkdir(parents=True)
    interpreter = environment_bin / "python"
    interpreter.touch()
    rust_bin = tmp_path / "rustup" / "bin"
    rust_bin.mkdir(parents=True)

    def locate(executable: str, *, path: str) -> str | None:
        if executable in {"cargo", "rustc"}:
            return str(rust_bin / executable)
        assert path.startswith("/usr/bin" + os.pathsep)
        return f"/usr/bin/{executable}"

    monkeypatch.setattr(backend.sys, "executable", str(interpreter))
    monkeypatch.setattr(backend.shutil, "which", locate)

    result = backend._build_tool_path(
        "/opt/local/bin:/opt/homebrew/bin:/usr/local/bin"
    ).split(os.pathsep)

    assert str(environment_bin) in result
    assert str(rust_bin) in result
    assert "/usr/bin" in result
    assert "/opt/local/bin" not in result
    assert "/opt/homebrew/bin" not in result
    assert "/usr/local/bin" not in result


def test_pep517_backend_rejects_recursive_delegation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("PYAMPLICOL_BUILD_MODE", "release")
    overlay = tmp_path / "overlay"
    overlay.mkdir()

    @contextlib.contextmanager
    def fake_overlay(
        _mode: str,
        *,
        project_release_prepared_models: bool = False,
    ):
        assert project_release_prepared_models is True
        yield overlay, tmp_path / "target"

    monkeypatch.setattr(backend, "_overlay", fake_overlay)
    monkeypatch.setattr(backend, "_check_dependencies", lambda _mode: None)
    monkeypatch.setattr(
        backend,
        "canonical_package_version",
        lambda _path: "0.1.0",
    )
    monkeypatch.setattr(backend, "_stage_release_build_info", lambda *_args: None)
    monkeypatch.setattr(
        backend,
        "stage_packaged_prepared_models",
        lambda _path, _mode: None,
    )
    monkeypatch.setattr(
        backend.maturin,
        "build_sdist",
        lambda *_args: backend.get_requires_for_build_sdist(),
    )
    with pytest.raises(RuntimeError, match="recursive PEP 517"):
        backend.build_sdist(str(tmp_path / "dist"))


def test_native_link_arguments_are_typed_and_allowlisted() -> None:
    macos = sdk._typed_link_arguments(
        [
            "-lgcc_s",
            "-lSystem",
            "-lc",
            "-framework",
            "Security",
            "-lSystem",
            "-framework",
            "Security",
        ],
        "aarch64-apple-darwin",
    )
    assert "gcc_s" in macos["system_libraries"]
    assert macos["system_libraries"] == ["gcc_s", "System", "c"]
    assert macos["frameworks"] == ["Security"]

    linux = sdk._typed_link_arguments(
        [
            "-lgcc_s",
            "-lutil",
            "-lrt",
            "-lpthread",
            "-lm",
            "-ldl",
            "-lc",
            "-lpthread",
            "-ldl",
            "-lc",
        ],
        "x86_64-unknown-linux-gnu",
    )
    assert linux["frameworks"] == []
    assert linux["system_libraries"] == [
        "gcc_s",
        "util",
        "rt",
        "pthread",
        "m",
        "dl",
        "c",
    ]


def test_static_archive_byte_scan_ignores_random_python_text(tmp_path: Path) -> None:
    archive = tmp_path / "librusticol_capi.a"
    archive.write_bytes(b"archive pYO3[/ rusticol_runtime_load native-static-libs")
    sdk._scan_archive(archive)


def test_static_archive_byte_scan_rejects_nonrelocatable_paths(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "librusticol_capi.a"
    archive.write_bytes(b"archive /opt/local/lib/libexample.a")
    with pytest.raises(RuntimeError, match="non-relocatable path marker"):
        sdk._scan_archive(archive)


@pytest.mark.parametrize(
    "symbol",
    ["_PyErr_SetString", "PyLong_FromLong", "_Py_Dealloc"],
)
def test_static_archive_symbol_scan_rejects_python_symbols(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    symbol: str,
) -> None:
    archive = tmp_path / "librusticol_capi.a"
    archive.write_bytes(b"archive")
    monkeypatch.setattr(sdk.shutil, "which", lambda _name: "/usr/bin/nm")
    monkeypatch.setattr(
        sdk.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["nm"], 0, stdout=f"                 U {symbol}\n", stderr=""
        ),
    )

    with pytest.raises(RuntimeError, match="undefined Python"):
        sdk._scan_archive_symbols(archive)


def test_static_archive_symbol_scan_accepts_non_python_symbols(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "librusticol_capi.a"
    archive.write_bytes(b"archive")
    monkeypatch.setattr(sdk.shutil, "which", lambda _name: "/usr/bin/nm")
    monkeypatch.setattr(
        sdk.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["nm"], 0, stdout="                 U _malloc\n", stderr=""
        ),
    )

    assert sdk._scan_archive_symbols(archive)


def test_static_archive_symbol_scan_defers_incompatible_llvm_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "librusticol_capi.a"
    archive.write_bytes(b"archive")
    monkeypatch.delenv("LLVM_NM", raising=False)
    monkeypatch.setattr(sdk.shutil, "which", lambda _name: "/usr/bin/nm")
    monkeypatch.setattr(
        sdk.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            ["nm"],
            1,
            stdout="",
            stderr=(
                "Unknown attribute kind (91) "
                "(Producer: 'LLVM20.1.7' Reader: 'LLVM APPLE_1_1600')"
            ),
        ),
    )

    assert not sdk._scan_archive_symbols(archive)


@pytest.mark.skipif(os.name == "nt", reason="native SDK targets macOS and Linux")
def test_static_archive_link_probe_rejects_missing_public_export(
    tmp_path: Path,
) -> None:
    target = (
        "aarch64-apple-darwin"
        if sys.platform == "darwin"
        else "x86_64-unknown-linux-gnu"
    )
    try:
        compiler = sdk._native_c_compiler(target)
    except RuntimeError as error:
        pytest.skip(str(error))
    archiver = shutil.which("ar")
    if archiver is None:
        pytest.skip("ar is unavailable")

    header = tmp_path / "rusticol.h"
    header.write_text(
        """\
#include <stdint.h>
#define RUSTICOL_ABI_VERSION 1u
uint32_t rusticol_abi_version(void);
int rusticol_required_export(void);
""",
        encoding="utf-8",
    )
    implementation = tmp_path / "partial.c"
    implementation.write_text(
        "#include <stdint.h>\nuint32_t rusticol_abi_version(void) { return 1u; }\n",
        encoding="utf-8",
    )
    object_file = tmp_path / "partial.o"
    archive = tmp_path / "librusticol_capi.a"
    subprocess.run(
        [*compiler, "-c", str(implementation), "-o", str(object_file)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [archiver, "rcs", str(archive), str(object_file)],
        check=True,
        capture_output=True,
        text=True,
    )

    with pytest.raises(RuntimeError, match="complete C ABI link probe"):
        sdk._validate_archive_linkage(
            archive,
            header=header,
            link={"target": target, "system_libraries": [], "frameworks": []},
            target=target,
        )


@pytest.mark.parametrize(
    "token",
    [
        "-L/opt/local/lib",
        "-Wl,-rpath,/opt/local/lib",
        "/usr/local/lib/libfoo.a",
        "-lnot_allowlisted",
    ],
)
def test_native_link_arguments_reject_nonportable_tokens(token: str) -> None:
    with pytest.raises(RuntimeError):
        sdk._typed_link_arguments([token], "aarch64-apple-darwin")

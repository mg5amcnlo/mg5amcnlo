# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import sys
import tomllib
import zipfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "developer" / "prepare_source_runtime.py"

_VERSIONS_SPEC = importlib.util.spec_from_file_location(
    "source_runtime_versions",
    ROOT / "src/pyamplicol/_internal/versions.py",
)
assert _VERSIONS_SPEC is not None and _VERSIONS_SPEC.loader is not None
versions = importlib.util.module_from_spec(_VERSIONS_SPEC)
sys.modules[_VERSIONS_SPEC.name] = versions
_VERSIONS_SPEC.loader.exec_module(versions)


def _module():
    spec = importlib.util.spec_from_file_location("prepare_source_runtime", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _native_build_identity_module():
    spec = importlib.util.spec_from_file_location(
        "source_runtime_native_build_identity",
        ROOT / "build_backend/native_build_identity.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _source_root(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    source = root / "rust/crates/example/src/lib.rs"
    source.parent.mkdir(parents=True)
    source.write_text("pub fn value() -> u32 { 1 }\n", encoding="utf-8")
    (root / "pyproject.toml").write_bytes((ROOT / "pyproject.toml").read_bytes())
    (root / "src/pyamplicol").mkdir(parents=True)
    return root


def _write_candidate_native_inputs(
    root: Path,
    *,
    checkout_root: Path | None = None,
    include_legacy: bool = False,
) -> Path:
    dependencies = root / "dependencies"
    dependencies.mkdir(exist_ok=True)
    checkout_root = checkout_root or dependencies / "checkouts"
    config = dependencies / "candidate-cargo-config.toml"
    config.write_text(
        "# generated location is deliberately volatile\n"
        "[patch.crates-io]\n"
        f'graphica = {{ path = "{checkout_root / "symbolica/lib/graphica"}" }}\n'
        f'numerica = {{ path = "{checkout_root / "symbolica/lib/numerica"}" }}\n'
        f'symbolica = {{ path = "{checkout_root / "symbolica"}" }}\n'
        f'symjit = {{ path = "{checkout_root / "symjit"}" }}\n',
        encoding="utf-8",
    )
    lock_paths = {
        "candidate": dependencies / "candidate-Cargo.lock",
        "contributor": dependencies / "contributor-lock.toml",
        "python-runtime": dependencies / "python-runtime-lock.toml",
        "release": dependencies / "release-lock.toml",
    }
    for index, path in enumerate(lock_paths.values()):
        path.write_text(f"lock {index}\n", encoding="utf-8")
    sources: dict[str, dict[str, str]] = {
        "gammaloop": {
            "revision": "1" * 40,
            "url": "https://example.invalid/gammaloop.git",
        },
        "ratatui-ffi": {
            "revision": "5" * 40,
            "url": "https://example.invalid/ratatui-ffi.git",
        },
        "symbolica": {
            "revision": "2" * 40,
            "url": "https://example.invalid/symbolica.git",
        },
        "symbolica-community": {
            "revision": "3" * 40,
            "url": "https://example.invalid/symbolica-community.git",
        },
        "symjit": {
            "revision": "4" * 40,
            "url": "https://example.invalid/symjit-crate.git",
        },
    }
    if include_legacy:
        sources["legacy-amplicol"] = {
            "branch": "historical",
            "revision": "7" * 40,
            "url": "https://example.invalid/AmpliCol.git",
        }
    lock_paths["contributor"].write_text(
        "schema_version = 1\n",
        encoding="utf-8",
    )
    state = {
        "publishable": False,
        "schema_version": 1,
        "sources": sources,
    }
    state_path = dependencies / "install-state.json"
    state_path.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return state_path


def _wheel(
    path: Path,
    *,
    native_build_inputs_sha256: str,
    mode: str = "candidate",
    target: str | None = None,
) -> None:
    if target is None:
        machine = platform.machine().lower()
        if sys.platform == "darwin" and machine in {"arm64", "aarch64"}:
            target = "aarch64-apple-darwin"
        elif sys.platform == "darwin" and machine in {"amd64", "x86_64"}:
            target = "x86_64-apple-darwin"
        else:
            target = "x86_64-unknown-linux-gnu"
    assert mode in {"candidate", "release"}
    version = "0.1.0.dev0+candidate.testsource" if mode == "candidate" else "0.1.0"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"pyamplicol-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.4\nName: pyamplicol\nVersion: {version}\n\n",
        )
        archive.writestr("pyamplicol/_rusticol.abi3.so", b"extension")
        if mode == "candidate":
            archive.writestr(
                "pyamplicol/_build_info.json",
                json.dumps(
                    {
                        "schema_version": 1,
                        "publishable": False,
                        "native_build_inputs_sha256": native_build_inputs_sha256,
                        "source_checkout": "/candidate/source",
                        "version": version,
                    }
                ),
            )
        archive.writestr(
            "pyamplicol/_sdk/metadata.json",
            json.dumps({"target": target}),
        )
        archive.writestr("pyamplicol/_sdk/link.json", "{}\n")
        archive.writestr("pyamplicol/_sdk/lib/librusticol_capi.a", b"archive")
        archive.writestr("pyamplicol/_sdk/include/rusticol.h", "/* header */\n")
        archive.writestr("pyamplicol/_sdk/fortran/rusticol.f90", "module r\n")
        archive.writestr(
            f"pyamplicol/assets/selftest/{target}/expected.json",
            "{}\n",
        )
        archive.writestr(
            f"pyamplicol/assets/selftest/{target}/artifact/artifact.json",
            "{}\n",
        )
        driver = zipfile.ZipInfo(
            f"pyamplicol/assets/selftest/{target}/artifact/API/python/"
            "check_standalone.py"
        )
        driver.external_attr = 0o100755 << 16
        archive.writestr(driver, "#!/usr/bin/env python3\n")
        archive.writestr("README.md", "must not be staged\n")
        archive.writestr("pyamplicol/_sdk/config.py", "must not overwrite source\n")


def test_source_runtime_stages_one_attested_extension(tmp_path: Path) -> None:
    module = _module()
    source_root = _source_root(tmp_path)
    native_digest = module._native_build_inputs_digest(source_root)
    wheel = tmp_path / "pyamplicol-test.whl"
    _wheel(wheel, native_build_inputs_sha256=native_digest)
    package = source_root / "src/pyamplicol"
    build_info = source_root / ".artifacts/source-runtime/_build_info.json"
    stale_extension = package / "_rusticol.cpython-312-darwin.so"
    stale_extension.write_bytes(b"stale")
    tracked_config = package / "_sdk/config.py"
    tracked_config.parent.mkdir()
    tracked_config.write_text("tracked\n", encoding="utf-8")

    report = module.stage_runtime(
        wheel,
        source_package=package,
        source_build_info=build_info,
        source_root=source_root,
        mode="candidate",
        audit=False,
    )

    assert report["version"] == "0.1.0.dev0+candidate.testsource"
    extension = package / "_rusticol.abi3.so"
    assert extension.read_bytes() == b"extension"
    assert not stale_extension.exists()
    assert tracked_config.read_text(encoding="utf-8") == "tracked\n"
    payload = json.loads(build_info.read_text(encoding="utf-8"))
    assert payload["source_runtime"] == {
        "extension_name": extension.name,
        "extension_sha256": hashlib.sha256(b"extension").hexdigest(),
        "mode": "candidate",
        "native_build_inputs_sha256": native_digest,
    }
    assert not (build_info.parent / ".staging").exists()


@pytest.mark.parametrize(
    ("mode", "bootstrap", "publishable", "release_bootstrap", "failure"),
    (
        ("candidate", True, False, False, None),
        ("candidate", False, False, False, "no self-test fixture"),
        ("candidate", True, True, False, "not marked candidate"),
        ("release", True, True, False, "no self-test fixture"),
        ("release", False, True, False, "no self-test fixture"),
        ("release", True, False, True, None),
    ),
)
def test_source_runtime_allows_missing_fixture_only_for_nonpublishable_bootstrap(
    tmp_path: Path,
    mode: str,
    bootstrap: bool,
    publishable: bool,
    release_bootstrap: bool,
    failure: str | None,
) -> None:
    module = _module()
    source_root = _source_root(tmp_path)
    native_digest = module._native_build_inputs_digest(
        source_root, normalize_release_cargo_lock=mode == "release"
    )
    wheel = tmp_path / "pyamplicol-bootstrap.whl"
    _wheel(wheel, native_build_inputs_sha256=native_digest, mode=mode)
    with zipfile.ZipFile(wheel) as archive:
        members = {
            info.filename: archive.read(info)
            for info in archive.infolist()
            if not info.filename.startswith("pyamplicol/assets/selftest/")
        }
    build_info_name = "pyamplicol/_build_info.json"
    payload = json.loads(members.get(build_info_name, b'{"version": "0.1.0"}'))
    payload.update(publishable=publishable, selftest_fixture_bootstrap=bootstrap)
    if release_bootstrap:
        payload["release_prepared_model_bootstrap"] = True
    members[build_info_name] = json.dumps(payload).encode()
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    package = source_root / "src/pyamplicol"
    build_info = source_root / ".artifacts/source-runtime/_build_info.json"
    kwargs = {
        "source_package": package,
        "source_build_info": build_info,
        "source_root": source_root,
        "mode": mode,
        "audit": False,
    }

    if failure is not None:
        with pytest.raises(module.ReleaseError, match=failure):
            module.stage_runtime(wheel, **kwargs)
        return

    module.stage_runtime(wheel, **kwargs)
    assert (package / "_rusticol.abi3.so").read_bytes() == b"extension"
    assert (package / "_sdk/lib/librusticol_capi.a").read_bytes() == b"archive"
    assert not (package / "assets/selftest").exists()
    staged = json.loads(build_info.read_text())
    assert staged["selftest_fixture_bootstrap"] is True
    if release_bootstrap:
        assert staged["publishable"] is False
        assert staged["release_prepared_model_bootstrap"] is True
        versions._verify_source_runtime(
            staged, package_root=package, source_root=source_root
        )
        return

    (source_root / "rust/crates/example/src/lib.rs").write_text("changed\n")
    with pytest.raises(module.ReleaseError, match="different native sources"):
        module.stage_runtime(wheel, **kwargs)


def test_release_source_runtime_uses_the_release_native_identity(
    tmp_path: Path,
) -> None:
    module = _module()
    source_root = _source_root(tmp_path)
    state_path = _write_candidate_native_inputs(source_root)
    candidate_digest = module._native_build_inputs_digest(source_root)
    release_digest = module._native_build_inputs_digest(
        source_root,
        normalize_release_cargo_lock=True,
    )
    assert release_digest != candidate_digest
    wheel = tmp_path / "pyamplicol-release.whl"
    _wheel(
        wheel,
        mode="release",
        native_build_inputs_sha256=release_digest,
    )
    package = source_root / "src/pyamplicol"
    build_info = source_root / ".artifacts/source-runtime/_build_info.json"

    report = module.stage_runtime(
        wheel,
        source_package=package,
        source_build_info=build_info,
        source_root=source_root,
        mode="release",
        audit=False,
    )

    assert report["version"] == "0.1.0"
    payload = json.loads(build_info.read_text(encoding="utf-8"))
    assert payload["publishable"] is True
    assert payload["native_build_inputs_sha256"] == release_digest
    assert payload["source_runtime"]["mode"] == "release"
    assert payload["source_runtime"]["native_build_inputs_sha256"] == release_digest
    versions._verify_source_runtime(
        payload,
        package_root=package,
        source_root=source_root,
    )

    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["created_utc"] = "2030-01-01T00:00:00+00:00"
    state_path.write_text(json.dumps(state) + "\n", encoding="utf-8")
    versions._verify_source_runtime(
        payload,
        package_root=package,
        source_root=source_root,
    )


def test_source_runtime_rejects_an_old_candidate_wheel(tmp_path: Path) -> None:
    module = _module()
    source_root = _source_root(tmp_path)
    wheel = tmp_path / "pyamplicol-test.whl"
    _wheel(
        wheel,
        native_build_inputs_sha256=module._native_build_inputs_digest(source_root),
    )
    (source_root / "rust/crates/example/src/lib.rs").write_text(
        "pub fn value() -> u32 { 2 }\n",
        encoding="utf-8",
    )

    with pytest.raises(module.ReleaseError, match="different native sources"):
        module.stage_runtime(
            wheel,
            source_package=source_root / "src/pyamplicol",
            source_build_info=source_root / ".artifacts/runtime/_build_info.json",
            source_root=source_root,
            mode="candidate",
            audit=False,
        )


def test_source_runtime_verification_rejects_replaced_binary_and_source(
    tmp_path: Path,
) -> None:
    source_root = _source_root(tmp_path)
    package = source_root / "src/pyamplicol"
    extension = package / "_rusticol.abi3.so"
    extension.write_bytes(b"current")
    native_digest = versions._native_build_inputs_digest(source_root)
    payload = {
        "publishable": False,
        "native_build_inputs_sha256": native_digest,
        "source_runtime": {
            "extension_name": extension.name,
            "extension_sha256": hashlib.sha256(b"current").hexdigest(),
            "mode": "candidate",
            "native_build_inputs_sha256": native_digest,
        },
    }

    versions._verify_source_runtime(
        payload,
        package_root=package,
        source_root=source_root,
    )
    extension.write_bytes(b"stale")
    with pytest.raises(RuntimeError, match="stale or was replaced"):
        versions._verify_source_runtime(
            payload,
            package_root=package,
            source_root=source_root,
        )

    extension.write_bytes(b"current")
    (source_root / "rust/crates/example/src/lib.rs").write_text(
        "pub fn value() -> u32 { 2 }\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="native build inputs changed"):
        versions._verify_source_runtime(
            payload,
            package_root=package,
            source_root=source_root,
        )


def test_installed_candidate_is_bound_to_its_checkout(tmp_path: Path) -> None:
    source_root = _source_root(tmp_path)
    payload = {
        "publishable": False,
        "source_checkout": str(source_root),
        "native_build_inputs_sha256": versions._native_build_inputs_digest(source_root),
    }

    versions._verify_candidate_install(payload)
    (source_root / "rust/crates/example/src/lib.rs").write_text(
        "pub fn value() -> u32 { 2 }\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="candidate wheel is stale"):
        versions._verify_candidate_install(payload)


def test_release_prepared_model_bootstrap_uses_release_native_identity(
    tmp_path: Path,
) -> None:
    source_root = _source_root(tmp_path)
    _write_candidate_native_inputs(source_root)
    release_digest = versions._native_build_inputs_digest(
        source_root,
        normalize_release_cargo_lock=True,
    )
    assert release_digest != versions._native_build_inputs_digest(source_root)
    payload = {
        "publishable": False,
        "release_prepared_model_bootstrap": True,
        "source_checkout": str(source_root),
        "native_build_inputs_sha256": release_digest,
    }

    versions._verify_candidate_install(payload)
    (source_root / "rust/crates/example/src/lib.rs").write_text(
        "pub fn value() -> u32 { 2 }\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="release prepared-model bootstrap is stale"):
        versions._verify_candidate_install(payload)


def test_native_module_build_id_must_match_candidate_metadata(monkeypatch) -> None:
    build_info = {
        "publishable": False,
        "native_build_inputs_sha256": "a" * 64,
    }
    monkeypatch.setattr(versions, "_active_build_info", lambda: build_info)
    native = type(
        "Native",
        (),
        {
            "__file__": "/candidate/pyamplicol/_rusticol.so",
            "package_version": staticmethod(lambda: "0.1.0-dev.0+candidate.same"),
            "native_build_inputs_sha256": staticmethod(lambda: "a" * 64),
        },
    )()
    versions.verify_native_module(
        native,
        expected_version="0.1.0.dev0+candidate.same",
    )

    build_info["native_build_inputs_sha256"] = "b" * 64
    with pytest.raises(RuntimeError, match="different source inputs"):
        versions.verify_native_module(
            native,
            expected_version="0.1.0.dev0+candidate.same",
        )


def test_publishable_native_module_does_not_require_developer_build_id(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        versions,
        "_active_build_info",
        lambda: {"publishable": True, "version": "0.1.0"},
    )
    native = type(
        "Native",
        (),
        {
            "__file__": "/wheel/pyamplicol/_rusticol.so",
            "package_version": staticmethod(lambda: "0.1.0"),
        },
    )()

    versions.verify_native_module(native, expected_version="0.1.0")


def test_publishable_native_source_identity_uses_wheel_extension(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        versions,
        "_active_build_info",
        lambda: {
            "publishable": True,
            "source_revision": "a" * 40,
            "version": "0.1.0",
        },
    )
    native = type(
        "Native",
        (),
        {"native_build_inputs_sha256": staticmethod(lambda: "b" * 64)},
    )()
    monkeypatch.setattr(
        versions.importlib,
        "import_module",
        lambda name: native if name == "pyamplicol._rusticol" else None,
    )

    assert versions.active_native_source_identity() == ("a" * 40, "b" * 64)


def test_active_source_checkout_is_limited_to_local_candidate_builds(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        versions,
        "_active_build_info",
        lambda: {
            "publishable": False,
            "source_checkout": str(tmp_path),
        },
    )
    assert versions.active_source_checkout() == tmp_path

    monkeypatch.setattr(
        versions,
        "_active_build_info",
        lambda: {
            "publishable": True,
            "source_checkout": str(tmp_path),
        },
    )
    assert versions.active_source_checkout() is None


def test_active_source_checkout_rejects_relative_path(monkeypatch) -> None:
    monkeypatch.setattr(
        versions,
        "_active_build_info",
        lambda: {
            "publishable": False,
            "source_checkout": "relative/checkout",
        },
    )
    assert versions.active_source_checkout() is None


def test_publishable_staged_source_runtime_requires_its_native_build_id(
    monkeypatch,
) -> None:
    build_info = {
        "publishable": True,
        "native_build_inputs_sha256": "a" * 64,
        "source_runtime": {"mode": "release"},
    }
    monkeypatch.setattr(versions, "_active_build_info", lambda: build_info)
    native = type(
        "Native",
        (),
        {
            "__file__": "/source/pyamplicol/_rusticol.so",
            "package_version": staticmethod(lambda: "0.1.0"),
            "native_build_inputs_sha256": staticmethod(lambda: "a" * 64),
        },
    )()

    versions.verify_native_module(native, expected_version="0.1.0")
    build_info["native_build_inputs_sha256"] = "b" * 64
    with pytest.raises(RuntimeError, match="different source inputs"):
        versions.verify_native_module(native, expected_version="0.1.0")


def test_package_version_prefers_attested_source_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root = _source_root(tmp_path)
    package = source_root / "src/pyamplicol"
    extension = package / "_rusticol.abi3.so"
    extension.write_bytes(b"extension")
    runtime = source_root / ".artifacts/source-runtime"
    runtime.mkdir(parents=True)
    build_info = runtime / "_build_info.json"
    build_info.write_text(
        json.dumps(
            {
                "publishable": False,
                "native_build_inputs_sha256": versions._native_build_inputs_digest(
                    source_root
                ),
                "source_runtime": {
                    "extension_name": extension.name,
                    "extension_sha256": hashlib.sha256(b"extension").hexdigest(),
                    "mode": "candidate",
                    "native_build_inputs_sha256": (
                        versions._native_build_inputs_digest(source_root)
                    ),
                },
                "version": "0.1.0.dev0+candidate.current",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(versions, "_SOURCE_ROOT", source_root)
    monkeypatch.setattr(versions, "_SOURCE_PACKAGE_ROOT", package)
    monkeypatch.setattr(versions, "_SOURCE_BUILD_INFO_PATH", build_info)
    monkeypatch.setattr(versions, "_SOURCE_RUNTIME_STAGING_PATH", runtime / ".staging")
    monkeypatch.setattr(versions, "_PACKAGE_BUILD_INFO_PATH", tmp_path / "missing")

    assert versions.package_version() == "0.1.0.dev0+candidate.current"


def test_package_version_fails_closed_during_staging(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root = _source_root(tmp_path)
    runtime = source_root / ".artifacts/source-runtime"
    runtime.mkdir(parents=True)
    marker = runtime / ".staging"
    marker.write_text("incomplete\n", encoding="utf-8")
    monkeypatch.setattr(versions, "_SOURCE_ROOT", source_root)
    monkeypatch.setattr(
        versions,
        "_SOURCE_PACKAGE_ROOT",
        source_root / "src/pyamplicol",
    )
    monkeypatch.setattr(versions, "_SOURCE_BUILD_INFO_PATH", runtime / "missing")
    monkeypatch.setattr(versions, "_SOURCE_RUNTIME_STAGING_PATH", marker)

    with pytest.raises(RuntimeError, match="staging is incomplete"):
        versions.package_version()


def test_source_runtime_rejects_dot_target(tmp_path: Path) -> None:
    module = _module()
    source_root = _source_root(tmp_path)
    wheel = tmp_path / "pyamplicol-test.whl"
    _wheel(
        wheel,
        native_build_inputs_sha256=module._native_build_inputs_digest(source_root),
        target="..",
    )

    with pytest.raises(module.ReleaseError, match="unsafe target"):
        module.stage_runtime(
            wheel,
            source_package=source_root / "src/pyamplicol",
            source_build_info=source_root / ".artifacts/runtime/_build_info.json",
            source_root=source_root,
            mode="candidate",
            audit=False,
        )


def test_source_runtime_rejects_a_wheel_for_another_host(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _module()
    source_root = _source_root(tmp_path)
    wheel = tmp_path / "pyamplicol-test.whl"
    _wheel(
        wheel,
        native_build_inputs_sha256=module._native_build_inputs_digest(source_root),
        target="x86_64-unknown-linux-gnu",
    )
    monkeypatch.setattr(module, "_host_target", lambda: "aarch64-apple-darwin")

    with pytest.raises(module.ReleaseError, match="this host requires"):
        module.stage_runtime(
            wheel,
            source_package=source_root / "src/pyamplicol",
            source_build_info=source_root / ".artifacts/runtime/_build_info.json",
            source_root=source_root,
            mode="candidate",
            audit=False,
        )


@pytest.mark.parametrize("mode", ("candidate", "release"))
@pytest.mark.parametrize("platform_name", ("linux", "darwin"))
def test_source_runtime_build_requests_host_wheel_only_on_linux(
    tmp_path: Path, monkeypatch, mode: str, platform_name: str
) -> None:
    module = _module()
    wheel = tmp_path / "pyamplicol-test.whl"
    wheel.touch()
    calls = []
    monkeypatch.setattr(module.sys, "platform", platform_name)
    monkeypatch.setattr(
        module, "external_temporary_directory", lambda _name: nullcontext(tmp_path)
    )
    monkeypatch.setattr(
        module, "run", lambda command, **kwargs: calls.append((command, kwargs))
    )
    monkeypatch.setattr(module, "clean_environment", lambda **_kwargs: {})
    monkeypatch.setattr(module, "stage_runtime", lambda path, **kwargs: (path, kwargs))
    assert module.build_and_stage(python=Path("python"), mode=mode) == (
        wheel,
        {"mode": mode, "audit": False},
    )
    command, _ = calls[0]
    assert ("pyamplicol.host-wheel=true" in command) == (platform_name == "linux")
    if platform_name == "linux":
        assert (
            command[command.index("--config-setting") + 1]
            == "pyamplicol.host-wheel=true"
        )


def test_source_runtime_stages_an_existing_local_wheel_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _module()
    wheel_directory = tmp_path / "wheelhouse"
    wheel_directory.mkdir()
    wheel = wheel_directory / "pyamplicol-test.whl"
    wheel.write_bytes(b"wheel")
    observed: dict[str, object] = {}

    def stage(path: Path, *, mode: str, audit: bool) -> dict[str, object]:
        observed.update(path=path, mode=mode, audit=audit)
        return {"wheel": path.name}

    monkeypatch.setattr(module, "stage_runtime", stage)
    report = module.stage_from_directory(wheel_directory, mode="candidate")

    assert report == {"wheel": wheel.name}
    assert observed == {"path": wheel, "mode": "candidate", "audit": False}


@pytest.mark.parametrize("mode", ("candidate", "release"))
def test_reuse_current_runtime_uses_existing_authoritative_verifier(
    tmp_path: Path, monkeypatch, mode: str
) -> None:
    module = _module()
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    build_info = runtime_root / "_build_info.json"
    payload = {"source_runtime": {"mode": mode}, "version": "1.0.0"}
    build_info.write_text(json.dumps(payload), encoding="utf-8")
    observed = []

    def verify(value, **kwargs):
        observed.append((value, kwargs))

    monkeypatch.setattr(module, "SOURCE_RUNTIME_ROOT", runtime_root)
    monkeypatch.setattr(module, "SOURCE_BUILD_INFO", build_info)
    monkeypatch.setattr(module, "SOURCE_RUNTIME_STAGING", runtime_root / ".staging")
    monkeypatch.setattr(
        module,
        "_runtime_versions",
        lambda: SimpleNamespace(
            _read_build_info=versions._read_build_info,
            _verify_source_runtime=verify,
        ),
    )

    assert module.reuse_current_runtime(mode=mode) == {
        "mode": mode,
        "version": "1.0.0",
        "reused": True,
    }
    assert observed == [
        (payload, {"package_root": module.SOURCE_PACKAGE, "source_root": module.ROOT})
    ]


@pytest.mark.parametrize(
    "failure", ("mode", "incomplete", "missing", "invalid", "stale")
)
def test_reuse_current_runtime_rejects_unusable_staging(
    tmp_path: Path, monkeypatch, failure: str
) -> None:
    module = _module()
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    build_info = runtime_root / "_build_info.json"
    payload = {
        "source_runtime": {"mode": "candidate" if failure == "mode" else "release"},
        "version": "1.0.0",
    }
    if failure != "missing":
        build_info.write_text(
            "not json" if failure == "invalid" else json.dumps(payload),
            encoding="utf-8",
        )
    if failure == "incomplete":
        (runtime_root / ".staging").touch()

    def verify(*args, **kwargs):
        assert failure == "stale", "invalid staging reached the identity verifier"
        raise RuntimeError("native inputs changed")

    monkeypatch.setattr(module, "SOURCE_RUNTIME_ROOT", runtime_root)
    monkeypatch.setattr(module, "SOURCE_BUILD_INFO", build_info)
    monkeypatch.setattr(module, "SOURCE_RUNTIME_STAGING", runtime_root / ".staging")
    monkeypatch.setattr(
        module,
        "_runtime_versions",
        lambda: SimpleNamespace(
            _read_build_info=versions._read_build_info,
            _verify_source_runtime=verify,
        ),
    )
    assert module.reuse_current_runtime(mode="release") is None


@pytest.mark.parametrize(
    ("reuse_requested", "reuse_available"), ((False, True), (True, True), (True, False))
)
def test_source_runtime_main_reuse_is_opt_in_with_build_fallback(
    monkeypatch, capsys, reuse_requested: bool, reuse_available: bool
) -> None:
    module = _module()
    monkeypatch.setenv("PYAMPLICOL_BUILD_MODE", "release")
    calls = []

    def reuse(*, mode):
        calls.append("reuse")
        assert mode == "release"
        return {"reused": True} if reuse_available else None

    def build(*, python, mode):
        calls.append("build")
        assert mode == "release"
        return {"wheel": "fresh.whl"}

    monkeypatch.setattr(module, "reuse_current_runtime", reuse)
    monkeypatch.setattr(module, "build_and_stage", build)
    assert module.main(["--reuse-current"] if reuse_requested else []) == 0
    reused = reuse_requested and reuse_available
    assert calls == (
        ["reuse"] if reused else ["reuse", "build"] if reuse_requested else ["build"]
    )
    assert json.loads(capsys.readouterr().out) == (
        {"reused": True} if reused else {"wheel": "fresh.whl"}
    )


def test_native_provenance_digest_implementations_match(tmp_path: Path) -> None:
    module = _module()
    spec = importlib.util.spec_from_file_location(
        "source_runtime_build_backend",
        ROOT / "build_backend/_pyamplicol_build.py",
    )
    assert spec is not None and spec.loader is not None
    backend = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "build_backend"))
    try:
        sys.modules[spec.name] = backend
        spec.loader.exec_module(backend)
    finally:
        sys.path.pop(0)
    source_root = _source_root(tmp_path)

    before = module._native_build_inputs_digest(source_root)
    native_source = source_root / "rust/crates/example/src/lib.rs"
    native_source.write_text("pub fn value() -> u32 { 2 }\n", encoding="utf-8")
    expected = module._native_build_inputs_digest(source_root)
    assert expected != before
    assert backend._native_build_inputs_digest(source_root) == expected
    assert versions._native_build_inputs_digest(source_root) == expected

    native_source.write_text("pub fn value() -> u32 { 3 }\n", encoding="utf-8")
    changed = module._native_build_inputs_digest(source_root)
    assert changed != expected
    assert backend._native_build_inputs_digest(source_root) == changed
    assert versions._native_build_inputs_digest(source_root) == changed


def test_native_digest_v2_projects_only_native_pyproject_inputs(
    tmp_path: Path,
) -> None:
    identity = _native_build_identity_module()
    source_root = _source_root(tmp_path)
    pyproject = source_root / "pyproject.toml"
    original = pyproject.read_text(encoding="utf-8")
    expected = identity.native_build_inputs_digest(source_root)
    assert versions._native_build_inputs_digest(source_root) == expected

    packaging_only = original.replace(
        'description = "Generator using current recursion',
        'description = "Packaging-only edit; generator using current recursion',
        1,
    ).replace("locked = true", "locked = false", 1)
    packaging_only = packaging_only.replace(
        '"maturin==1.15.0",\n  "mypy',
        '"maturin==9.9.9",\n  "mypy',
        1,
    )
    pyproject.write_text(packaging_only, encoding="utf-8")
    assert identity.native_build_inputs_digest(source_root) == expected
    assert versions._native_build_inputs_digest(source_root) == expected

    pyproject.write_text(
        original.replace("maturin==1.15.0", "maturin==1.15.1", 1),
        encoding="utf-8",
    )
    pin_changed = identity.native_build_inputs_digest(source_root)
    assert pin_changed != expected
    assert versions._native_build_inputs_digest(source_root) == pin_changed

    pyproject.write_text(
        original.replace(
            'features = ["extension-module", "numpy"]',
            'features = ["extension-module", "numpy", "native-change"]',
            1,
        ),
        encoding="utf-8",
    )
    feature_changed = identity.native_build_inputs_digest(source_root)
    assert feature_changed != expected
    assert versions._native_build_inputs_digest(source_root) == feature_changed

    pyproject.write_text(
        original.replace(
            "/pyamplicol/build",
            "/pyamplicol/changed-build",
            1,
        ),
        encoding="utf-8",
    )
    contract_changed = identity.native_build_inputs_digest(source_root)
    assert contract_changed != expected
    assert versions._native_build_inputs_digest(source_root) == contract_changed


@pytest.mark.parametrize(
    "replacement",
    (
        "maturin>=1.15.0",
        "maturin==1.15.0; python_version >= '3.11'",
        "packaging==26.3",
        'maturin==1.15.0",\n  "maturin==1.15.0',
    ),
)
def test_native_digest_rejects_unpinned_or_missing_build_maturin(
    tmp_path: Path,
    replacement: str,
) -> None:
    source_root = _source_root(tmp_path)
    pyproject = source_root / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace(
            "maturin==1.15.0",
            replacement,
            1,
        ),
        encoding="utf-8",
    )
    for digest in (
        _native_build_identity_module().native_build_inputs_digest,
        versions._native_build_inputs_digest,
    ):
        with pytest.raises(RuntimeError, match="Maturin"):
            digest(source_root)


def test_native_digest_rejects_unclassified_maturin_configuration(
    tmp_path: Path,
) -> None:
    source_root = _source_root(tmp_path)
    pyproject = source_root / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace(
            "[tool.maturin]\n",
            "[tool.maturin]\nfuture-native-option = true\n",
            1,
        ),
        encoding="utf-8",
    )
    for digest in (
        _native_build_identity_module().native_build_inputs_digest,
        versions._native_build_inputs_digest,
    ):
        with pytest.raises(RuntimeError, match="unclassified Maturin"):
            digest(source_root)


def test_native_digest_rejects_boolean_contract_schema_version(
    tmp_path: Path,
) -> None:
    source_root = _source_root(tmp_path)
    pyproject = source_root / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace(
            "schema-version = 1",
            "schema-version = true",
            1,
        ),
        encoding="utf-8",
    )
    for digest in (
        _native_build_identity_module().native_build_inputs_digest,
        versions._native_build_inputs_digest,
    ):
        with pytest.raises(RuntimeError, match="schema-version 1"):
            digest(source_root)


def test_native_digest_uses_explicit_test_only_inventory(tmp_path: Path) -> None:
    identity = _native_build_identity_module()
    source_root = _source_root(tmp_path)
    expected = identity.native_build_inputs_digest(source_root)

    backend_python = source_root / "build_backend/_pyamplicol_build.py"
    backend_python.parent.mkdir(parents=True)
    backend_python.write_text("PACKAGING_ONLY = True\n", encoding="utf-8")
    staging = source_root / "rust/.artifacts/profile/report.json"
    staging.parent.mkdir(parents=True)
    staging.write_text("{}\n", encoding="utf-8")
    ignored = source_root / "rust/crates/rusticol-core/src/artifact_tests.rs"
    ignored.parent.mkdir(parents=True)
    ignored.write_text("#[test] fn changed() {}\n", encoding="utf-8")
    assert identity.native_build_inputs_digest(source_root) == expected

    productive_name = source_root / "rust/crates/example/src/future_tests.rs"
    productive_name.write_text("pub fn production() {}\n", encoding="utf-8")
    changed = identity.native_build_inputs_digest(source_root)
    assert changed != expected
    assert versions._native_build_inputs_digest(source_root) == changed

    test_support = (
        source_root
        / "rust/crates/rusticol-core/src/recurrence/on_the_fly/test_support.rs"
    )
    test_support.parent.mkdir(parents=True, exist_ok=True)
    test_support.write_text("pub fn feature_api() {}\n", encoding="utf-8")
    feature_changed = identity.native_build_inputs_digest(source_root)
    assert feature_changed != changed
    assert versions._native_build_inputs_digest(source_root) == feature_changed


def test_candidate_native_digest_is_stable_across_install_and_relocation(
    tmp_path: Path,
) -> None:
    identity = _native_build_identity_module()
    staging = _module()
    first = _source_root(tmp_path / "first")
    second = _source_root(tmp_path / "second")
    _write_candidate_native_inputs(first)
    _write_candidate_native_inputs(
        second,
        checkout_root=(tmp_path / "relocated" / "dependencies" / "checkouts"),
        include_legacy=True,
    )

    expected = identity.native_build_inputs_digest(first)
    assert identity.native_build_inputs_digest(second) == expected
    assert versions._native_build_inputs_digest(first) == expected
    assert versions._native_build_inputs_digest(second) == expected
    assert staging._native_build_inputs_digest(first) == expected

    config = first / "dependencies/candidate-cargo-config.toml"
    config.write_text(
        "# a repeat install may alter comments and formatting\n"
        + config.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    assert identity.native_build_inputs_digest(first) == expected


def test_candidate_native_digest_hashes_only_native_checkout_revisions(
    tmp_path: Path,
) -> None:
    identity = _native_build_identity_module()
    source_root = _source_root(tmp_path)
    state_path = _write_candidate_native_inputs(source_root, include_legacy=True)
    expected = identity.native_build_inputs_digest(source_root)
    state = json.loads(state_path.read_text(encoding="utf-8"))

    state["sources"]["gammaloop"]["revision"] = "a" * 40
    state["sources"]["legacy-amplicol"]["revision"] = "b" * 40
    state_path.write_text(json.dumps(state, sort_keys=True) + "\n", encoding="utf-8")
    assert identity.native_build_inputs_digest(source_root) == expected
    assert versions._native_build_inputs_digest(source_root) == expected

    state["sources"]["symbolica"]["revision"] = "c" * 40
    state_path.write_text(json.dumps(state, sort_keys=True) + "\n", encoding="utf-8")
    changed = identity.native_build_inputs_digest(source_root)
    assert changed != expected
    assert versions._native_build_inputs_digest(source_root) == changed


def test_candidate_native_digest_requires_the_minimal_install_state_schema(
    tmp_path: Path,
) -> None:
    identity = _native_build_identity_module()
    source_root = _source_root(tmp_path)
    state_path = _write_candidate_native_inputs(source_root)
    expected = identity.native_build_inputs_digest(source_root)

    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["sources"]["symjit"]["worktree_sha256"] = "9" * 64
    state_path.write_text(
        json.dumps(state, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for digest in (
        identity.native_build_inputs_digest,
        versions._native_build_inputs_digest,
    ):
        with pytest.raises(RuntimeError, match="optional branch only"):
            digest(source_root)

    del state["sources"]["symjit"]["worktree_sha256"]
    state_path.write_text(
        json.dumps(state, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = source_root / "dependencies/candidate-cargo-config.toml"
    config.write_text(
        config.read_text(encoding="utf-8") + "# formatting-only rewrite\n",
        encoding="utf-8",
    )
    for digest in (
        identity.native_build_inputs_digest,
        versions._native_build_inputs_digest,
    ):
        assert digest(source_root) == expected


def test_local_symjit_inputs_invalidate_native_cache_without_extension_noise(
    tmp_path: Path,
) -> None:
    identity = _native_build_identity_module()
    source_root = _source_root(tmp_path)
    _write_candidate_native_inputs(source_root)
    config = source_root / "dependencies/candidate-cargo-config.toml"
    config.write_text(
        config.read_text(encoding="utf-8").replace(
            str(source_root / "dependencies/checkouts/symjit"),
            str(source_root / "TMP_FIXED_SYMJIT"),
        ),
        encoding="utf-8",
    )
    (source_root / "Cargo.toml").write_text(
        '[patch.crates-io]\nsymjit = { path = "./TMP_FIXED_SYMJIT" }\n',
        encoding="utf-8",
    )
    local = source_root / "TMP_FIXED_SYMJIT"
    (local / "src").mkdir(parents=True)
    compiler = local / "src/lib.rs"
    compiler.write_text("pub fn compiler() {}\n", encoding="utf-8")
    before = identity.native_build_inputs_digest(source_root)
    assert versions._native_build_inputs_digest(source_root) == before

    # Spenso is part of the separate community extension, not RustiCol.
    with config.open("a", encoding="utf-8") as stream:
        stream.write('spenso = { path = "TMP_FIXED_SPENSO/crates/spenso" }\n')
    (local / "README.md").write_text("documentation only\n", encoding="utf-8")
    (local / "tests").mkdir()
    (local / "tests/check.rs").write_text("// test only\n", encoding="utf-8")
    assert identity.native_build_inputs_digest(source_root) == before
    assert versions._native_build_inputs_digest(source_root) == before

    compiler.write_text("pub fn compiler() { let _fixed = true; }\n", encoding="utf-8")
    after = identity.native_build_inputs_digest(source_root)
    assert after != before
    assert versions._native_build_inputs_digest(source_root) == after


def test_candidate_native_digest_rejects_wrong_patch_target(tmp_path: Path) -> None:
    identity = _native_build_identity_module()
    source_root = _source_root(tmp_path)
    _write_candidate_native_inputs(source_root)
    config = source_root / "dependencies/candidate-cargo-config.toml"
    text = config.read_text(encoding="utf-8").replace(
        "dependencies/checkouts/symjit",
        "dependencies/checkouts/symbolica",
    )
    config.write_text(text, encoding="utf-8")
    for digest in (
        identity.native_build_inputs_digest,
        versions._native_build_inputs_digest,
    ):
        with pytest.raises(RuntimeError, match="symjit must resolve"):
            digest(source_root)


def test_candidate_native_digest_ignores_provenance_locks_and_hashes_cargo_lock(
    tmp_path: Path,
) -> None:
    identity = _native_build_identity_module()
    source_root = _source_root(tmp_path)
    _write_candidate_native_inputs(source_root)
    expected = identity.native_build_inputs_digest(source_root)
    dependencies = source_root / "dependencies"
    for name in (
        "contributor-lock.toml",
        "python-runtime-lock.toml",
        "release-lock.toml",
    ):
        (dependencies / name).write_text("changed provenance\n", encoding="utf-8")
        assert identity.native_build_inputs_digest(source_root) == expected
        assert versions._native_build_inputs_digest(source_root) == expected

    (source_root / "Cargo.lock").write_text(
        "unused release resolution\n",
        encoding="utf-8",
    )
    assert identity.native_build_inputs_digest(source_root) == expected
    assert versions._native_build_inputs_digest(source_root) == expected

    lock = dependencies / "candidate-Cargo.lock"
    lock.write_text("changed Cargo resolution\n", encoding="utf-8")
    changed = identity.native_build_inputs_digest(source_root)
    assert changed != expected
    assert versions._native_build_inputs_digest(source_root) == changed


def test_release_native_digest_hashes_the_registry_cargo_lock_verbatim(
    tmp_path: Path,
) -> None:
    identity = _native_build_identity_module()
    source_root = _source_root(tmp_path)
    lock = tomllib.loads(
        (ROOT / "dependencies/release-lock.toml").read_text(encoding="utf-8")
    )
    symjit = lock["symjit"]
    source = "registry+https://github.com/rust-lang/crates.io-index"
    marker = (
        "[[package]]\n"
        'name = "symjit"\n'
        f'version = "{symjit["version"]}"\n'
        f'source = "{source}"\n'
    )
    replacement = f'[[package]]\nname = "symjit"\nversion = "{symjit["version"]}"\n'
    git_lock = (ROOT / "Cargo.lock").read_text(encoding="utf-8")
    assert git_lock.count(marker) == 1
    projected_lock = git_lock.replace(marker, replacement, 1).encode("utf-8")
    cargo_lock = source_root / "Cargo.lock"

    cargo_lock.write_text(git_lock, encoding="utf-8")
    git_shared = identity.native_build_inputs_digest(
        source_root,
        normalize_release_cargo_lock=True,
    )
    git_runtime = versions._native_build_inputs_digest(
        source_root,
        normalize_release_cargo_lock=True,
    )
    assert git_runtime == git_shared
    assert identity.canonical_release_cargo_lock_bytes(
        source_root,
        git_lock.encode("utf-8"),
    ) == git_lock.encode("utf-8")

    cargo_lock.write_bytes(projected_lock)
    projected_shared = identity.native_build_inputs_digest(
        source_root,
        normalize_release_cargo_lock=True,
    )
    projected_runtime = versions._native_build_inputs_digest(
        source_root,
        normalize_release_cargo_lock=True,
    )

    assert projected_shared != git_shared
    assert projected_runtime == projected_shared

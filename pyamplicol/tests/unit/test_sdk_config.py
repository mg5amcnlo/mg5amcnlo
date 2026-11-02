# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
import os
import runpy
import shlex
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import rusticol_config as config


def _sdk(tmp_path: Path) -> Path:
    root = tmp_path / "installed path with spaces" / "_sdk"
    (root / "include").mkdir(parents=True)
    (root / "fortran").mkdir()
    (root / "lib").mkdir()
    (root / "rust").mkdir()
    (root / "include" / "rusticol.h").write_text("", encoding="utf-8")
    (root / "include" / "rusticol.hpp").write_text("", encoding="utf-8")
    (root / "fortran" / "rusticol.f90").write_text("", encoding="utf-8")
    (root / "rust" / "rusticol.rs").write_text("", encoding="utf-8")
    (root / "lib" / "librusticol_capi.a").write_bytes(b"archive")
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "abi_version": 1,
                "version": "0.1.0",
                "target": "aarch64-apple-darwin",
                "archive": "lib/librusticol_capi.a",
                "rust_source": "rust/rusticol.rs",
            }
        ),
        encoding="utf-8",
    )
    (root / "link.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "target": "aarch64-apple-darwin",
                "system_libraries": ["System", "m"],
                "frameworks": ["Security"],
            }
        ),
        encoding="utf-8",
    )
    return root


def test_sdk_info_returns_a_complete_native_link_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _sdk(tmp_path)
    monkeypatch.setattr(config, "_resource_root", lambda: root)
    monkeypatch.setattr(config, "_package_version", lambda: "0.1.0")

    info = config.load_sdk_info()
    assert info.library == root / "lib" / "librusticol_capi.a"
    assert info.rust_source == root / "rust" / "rusticol.rs"
    assert info.link_flags == (
        str(root / "lib" / "librusticol_capi.a"),
        "-lSystem",
        "-lm",
        "-framework",
        "Security",
    )
    assert info.rust_flags == tuple(
        token
        for link_flag in info.link_flags
        for token in ("-C", f"link-arg={link_flag}")
    )
    assert info.to_json()["rust_flags"] == list(info.rust_flags)
    assert info.to_json()["rust_source"] == str(info.rust_source)
    assert info.cargo_encoded_rust_flags.split("\x1f") == list(info.rust_flags)


def test_rustflags_cli_emits_shell_safe_rustc_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _sdk(tmp_path)
    monkeypatch.setattr(config, "_resource_root", lambda: root)
    monkeypatch.setattr(config, "_package_version", lambda: "0.1.0")

    assert config.main(["--rustflags"]) == 0
    output = capsys.readouterr().out.strip()
    assert shlex.split(output) == list(config.load_sdk_info().rust_flags)
    assert str(root / "lib" / "librusticol_capi.a") in output

    assert config.main(["--rust-source"]) == 0
    assert capsys.readouterr().out.strip() == str(root / "rust" / "rusticol.rs")

    assert config.main(["--cargo-rustflags"]) == 0
    encoded = capsys.readouterr().out.rstrip("\n")
    assert encoded.split("\x1f") == list(config.load_sdk_info().rust_flags)


@pytest.mark.parametrize("field", ("archive", "rust_source"))
def test_sdk_metadata_rejects_path_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    root = _sdk(tmp_path)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    metadata[field] = "../../outside"
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(config, "_resource_root", lambda: root)
    monkeypatch.setattr(config, "_package_version", lambda: "0.1.0")

    with pytest.raises(config.SdkUnavailableError, match="escapes"):
        config.load_sdk_info()


def test_sdk_metadata_requires_exact_version_and_uses_wheel_record_for_integrity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _sdk(tmp_path)
    monkeypatch.setattr(config, "_resource_root", lambda: root)
    monkeypatch.setattr(
        config,
        "_package_version",
        lambda: "0.1.0.dev0+candidate.deadbeef",
    )
    with pytest.raises(config.SdkUnavailableError, match="version"):
        config.load_sdk_info()

    monkeypatch.setattr(config, "_package_version", lambda: "0.1.0")
    (root / "lib" / "librusticol_capi.a").write_bytes(b"tampered")
    info = config.load_sdk_info()
    assert info.library.read_bytes() == b"tampered"
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    assert "archive_sha256" not in metadata


@pytest.fixture
def standalone_sdk(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """A wheel-shaped SDK whose package initializer must never be executed."""
    source_root = Path(__file__).resolve().parents[2] / "src"
    site = tmp_path / "site packages"
    package = site / "pyamplicol"
    shutil.copytree(_sdk(tmp_path), package / "_sdk")
    (package / "__init__.py").write_text(
        "raise RuntimeError('numerical package initialization is forbidden')\n",
        encoding="utf-8",
    )
    (site / "symbolica.py").write_text(
        "raise RuntimeError('Symbolica initialization is forbidden')\n",
        encoding="utf-8",
    )
    (package / "_internal").mkdir()
    shutil.copyfile(
        source_root / "pyamplicol" / "_internal" / "versions.py",
        package / "_internal" / "versions.py",
    )
    (package / "_build_info.json").write_text(
        json.dumps({"version": "0.1.0", "publishable": True}),
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(site), str(source_root)))
    return package, environment


@pytest.mark.parametrize("invocation", ("console", "module"))
def test_standalone_sdk_query_does_not_initialize_numerical_packages(
    standalone_sdk: tuple[Path, dict[str, str]], invocation: str
) -> None:
    package, environment = standalone_sdk
    command = (
        [sys.executable, "-m", "rusticol_config"]
        if invocation == "module"
        else [
            sys.executable,
            "-c",
            "import sys; from rusticol_config import main; result = main(); "
            "assert 'pyamplicol' not in sys.modules; "
            "assert 'symbolica' not in sys.modules; raise SystemExit(result)",
        ]
    )
    result = subprocess.run(
        [*command, "--json"],
        cwd=package.parent,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    payload = json.loads(result.stdout)
    assert result.stderr == ""
    assert payload["package_version"] == "0.1.0"
    assert payload["library"] == str(package / "_sdk/lib/librusticol_capi.a")
    assert payload["link_flags"] == [
        payload["library"],
        "-lSystem",
        "-lm",
        "-framework",
        "Security",
    ]


def test_standalone_sdk_retains_authoritative_version_validation(
    standalone_sdk: tuple[Path, dict[str, str]],
) -> None:
    package, environment = standalone_sdk
    (package / "_build_info.json").write_text(
        json.dumps({"version": None, "publishable": True}), encoding="utf-8"
    )
    result = subprocess.run(
        [sys.executable, "-m", "rusticol_config", "--version"],
        cwd=package.parent,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode != 0
    assert "wheel build version provenance is invalid" in result.stderr
    assert "initialization is forbidden" not in result.stderr


def test_sdk_console_entry_and_package_are_declared() -> None:
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["scripts"]["rusticol-config"] == "rusticol_config:main"
    assert "rusticol_config" in project["tool"]["maturin"]["python-packages"]


def test_existing_sdk_import_forwards_to_the_same_implementation() -> None:
    source = Path(__file__).resolve().parents[2] / "src/pyamplicol/_sdk/config.py"
    compatibility = runpy.run_path(str(source))
    assert compatibility["main"] is config.main
    assert compatibility["load_sdk_info"] is config.load_sdk_info
    assert compatibility["SdkInfo"] is config.SdkInfo

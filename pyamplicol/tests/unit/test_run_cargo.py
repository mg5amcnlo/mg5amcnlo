# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/release/run_cargo.py"


def _module() -> object:
    spec = importlib.util.spec_from_file_location("run_cargo", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_cargo_requires_subcommand() -> None:
    module = _module()
    with pytest.raises(SystemExit, match="2"):
        module.main(["--mode", "release", "--"])


def test_run_cargo_uses_clean_overlay(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setenv("CC", "/attacker/cc")
    monkeypatch.setenv("CXX", "/attacker/c++")
    overlay = tmp_path / "overlay"
    target = tmp_path / "target"
    overlay.mkdir()
    calls: list[object] = []

    class Overlay:
        def __enter__(self) -> tuple[Path, Path]:
            calls.append(("overlay", "candidate"))
            return overlay, target

        def __exit__(self, *args: object) -> None:
            del args

    monkeypatch.setattr(module.backend, "_overlay", lambda mode: Overlay())
    monkeypatch.setattr(
        module.backend,
        "_clean_environment",
        lambda updates: {"PATH": os.environ.get("PATH", ""), **updates},
    )

    class Completed:
        returncode = 7

    def run(command: list[str], **kwargs: object) -> Completed:
        calls.append((command, kwargs))
        return Completed()

    import os

    monkeypatch.setattr(module.subprocess, "run", run)

    assert module.main(["--mode", "candidate", "--", "check", "--locked"]) == 7
    command, options = calls[1]
    assert command == ["cargo", "check", "--locked"]
    assert options["cwd"] == overlay
    assert options["env"]["CARGO_HOME"] == str(tmp_path / "cargo-home")
    assert options["env"]["CARGO_TARGET_DIR"] == str(target)
    assert options["env"]["CC"] == "/usr/bin/clang"
    assert options["env"]["CXX"] == "/usr/bin/clang++"
    assert options["env"]["MACOSX_DEPLOYMENT_TARGET"] == "11.0"


def test_rust_tests_use_only_the_selected_python_loader_directories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    libdir = tmp_path / "lib"
    libpl = tmp_path / "config"
    libdir.mkdir()
    libpl.mkdir()
    values = {"LIBDIR": str(libdir), "LIBPL": str(libpl)}

    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(
        module.sysconfig,
        "get_config_var",
        lambda name: values.get(name),
    )

    assert module._python_test_loader_updates(["test", "--workspace"]) == {
        "LD_LIBRARY_PATH": f"{libdir}{module.os.pathsep}{libpl}"
    }
    assert module._python_test_loader_updates(["check", "--workspace"]) == {}

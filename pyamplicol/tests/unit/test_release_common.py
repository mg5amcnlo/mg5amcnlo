# SPDX-License-Identifier: 0BSD

from __future__ import annotations

import io
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "release"))

import _common  # noqa: E402


def _tar(path: Path, name: str, data: bytes = b"content") -> None:
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo(name)
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))


def test_clean_environment_removes_python_and_pip_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/parent/source")
    monkeypatch.setenv("PYTHONHOME", "/parent/python")
    monkeypatch.setenv("PIP_FIND_LINKS", "/parent/wheels")
    monkeypatch.setenv("PIP_INDEX_URL", "https://example.invalid/simple")
    environment = _common.clean_environment(mode="candidate")
    assert "PYTHONPATH" not in environment
    assert "PYTHONHOME" not in environment
    assert "PIP_FIND_LINKS" not in environment
    assert "PIP_INDEX_URL" not in environment
    assert environment["PIP_CONFIG_FILE"] == os.devnull
    assert environment["PYAMPLICOL_BUILD_MODE"] == "candidate"


def test_safe_sdist_extraction_rejects_traversal_and_links(tmp_path: Path) -> None:
    traversal = tmp_path / "traversal.tar.gz"
    _tar(traversal, "../outside")
    with pytest.raises(_common.ReleaseError, match="unsafe"):
        _common.safe_extract_sdist(traversal, tmp_path / "traversal-output")

    linked = tmp_path / "linked.tar.gz"
    with tarfile.open(linked, "w:gz") as archive:
        info = tarfile.TarInfo("project/link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        archive.addfile(info)
    with pytest.raises(_common.ReleaseError, match="regular files"):
        _common.safe_extract_sdist(linked, tmp_path / "linked-output")


def test_safe_sdist_extraction_returns_one_clean_root(tmp_path: Path) -> None:
    sdist = tmp_path / "valid.tar.gz"
    _tar(sdist, "project-0.1.0/pyproject.toml", b"[project]\n")
    root = _common.safe_extract_sdist(sdist, tmp_path / "output")
    assert root == tmp_path / "output" / "project-0.1.0"
    assert (root / "pyproject.toml").read_bytes() == b"[project]\n"


def test_dependency_gate_uses_candidate_and_release_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append([os.fspath(item) for item in command])
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(_common, "run", fake_run)
    _common.check_dependency_gate("candidate")
    _common.check_dependency_gate("release")
    assert commands[0][-1:] == ["--candidate"]
    assert "--candidate" not in commands[1]


def test_build_mode_rejects_conflicting_candidate_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYAMPLICOL_BUILD_MODE", "release")
    with pytest.raises(_common.ReleaseError, match="conflicts"):
        _common.build_mode(candidate=True)


def test_container_root_checkout_does_not_treat_filesystem_root_as_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_common, "ROOT", Path("/io"))
    assert _common.containing_workspace() == Path("/io")
    assert not _common.is_relative_to(Path("/tmp/release-build"), Path("/io"))


def test_release_temporary_path_accepts_only_external_or_artifact_scratch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_common, "ROOT", Path("/io/pyamplicol"))
    assert _common._release_temporary_path_is_isolated(
        Path("/io/pyamplicol/.artifacts/migration/tmp/build")
    )
    assert _common._release_temporary_path_is_isolated(Path("/tmp/release-build"))
    assert not _common._release_temporary_path_is_isolated(
        Path("/io/other-checkout/release-build")
    )

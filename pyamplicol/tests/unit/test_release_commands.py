# SPDX-License-Identifier: 0BSD

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "release"))

import _artifacts as release_artifacts  # noqa: E402
import build_from_sdist  # noqa: E402
import build_release_artifacts  # noqa: E402
import install_wheel  # noqa: E402
import publish_dry_run  # noqa: E402
import test_deployment as release_deployment  # noqa: E402
from _common import ReleaseError  # noqa: E402


def _dependency_wheel(
    directory: Path,
    distribution: str,
    version: str,
    tag: str,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    normalized = distribution.replace("-", "_")
    wheel = directory / f"{normalized}-{version}-{tag}.whl"
    dist_info = f"{normalized}-{version}.dist-info"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            f"{dist_info}/METADATA",
            f"Metadata-Version: 2.4\nName: {distribution}\nVersion: {version}\n\n",
        )
        archive.writestr(
            f"{dist_info}/WHEEL",
            f"Wheel-Version: 1.0\nTag: {tag}\n\n",
        )
    return wheel


@pytest.fixture
def candidate_dependency_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contributor_path = tmp_path / "contributor-lock.toml"
    state_path = tmp_path / "install-state.json"
    contributor_data = (ROOT / "dependencies" / "contributor-lock.toml").read_bytes()
    contributor_path.write_bytes(contributor_data)
    contributor = tomllib.loads(contributor_data.decode("utf-8"))
    symbolica = contributor["symbolica"]
    state_path.write_text(
        json.dumps(
            {
                "publishable": False,
                "schema_version": 1,
                "sources": {
                    "symbolica": {
                        "revision": symbolica["candidate_revision"],
                        "url": symbolica["source_url"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        release_artifacts,
        "_CONTRIBUTOR_DEPENDENCY_LOCK",
        contributor_path,
    )
    monkeypatch.setattr(
        release_artifacts,
        "_CANDIDATE_INSTALL_STATE",
        state_path,
    )


def test_select_compatible_abi3_wheel_uses_target_tag_order(tmp_path: Path) -> None:
    mac = tmp_path / "pyamplicol-0.1.0-cp311-abi3-macosx_11_0_arm64.whl"
    linux = tmp_path / "pyamplicol-0.1.0-cp311-abi3-manylinux_2_28_x86_64.whl"
    mac.touch()
    linux.touch()
    selected = install_wheel.select_compatible_wheel(
        [linux, mac],
        [
            "cp314-cp314-macosx_15_0_arm64",
            "cp311-abi3-macosx_11_0_arm64",
            "py3-none-any",
        ],
    )
    assert selected == mac.resolve()


@pytest.mark.parametrize(
    ("mode", "candidate_flag"),
    [("candidate", True), ("release", False)],
)
def test_install_builds_through_release_artifact_tool_when_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    candidate_flag: bool,
) -> None:
    observed: dict[str, object] = {}

    def fake_run(command, *, cwd, env, **_kwargs):
        observed.update(command=list(map(os.fspath, command)), cwd=cwd, env=env)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setenv("PYTHONPATH", "/parent/source")
    monkeypatch.setattr(install_wheel, "run", fake_run)
    install_wheel.build_if_missing(mode, tmp_path)
    command = observed["command"]
    assert isinstance(command, list)
    assert command[1].endswith("build_release_artifacts.py")
    assert ("--candidate" in command) is candidate_flag
    assert observed["cwd"] == ROOT
    environment = observed["env"]
    assert isinstance(environment, dict)
    assert "PYTHONPATH" not in environment


def test_deployment_path_guard_allows_only_the_isolated_sandbox(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    sandbox = checkout / "PYPI_DEPLOYMENT_TEST" / "release-1"
    site_packages = sandbox / "venv" / "lib" / "site-packages"
    source = checkout / "src"
    environment = {
        **os.environ,
        "PYAMPLICOL_FORBIDDEN_ROOT": str(checkout),
        "PYAMPLICOL_DEPLOYMENT_SANDBOX": str(sandbox),
    }

    def guarded_path(path: Path) -> subprocess.CompletedProcess[str]:
        script = (
            "import os\nfrom pathlib import Path\nimport sys\n"
            f"sys.path[:] = [{str(path)!r}]\n"
            + release_deployment._PATH_ISOLATION_SMOKE
        )
        return subprocess.run(
            [sys.executable, "-I", "-c", script],
            env=environment,
            capture_output=True,
            text=True,
        )

    assert guarded_path(site_packages).returncode == 0
    assert guarded_path(source).returncode != 0


def test_candidate_deployment_installs_source_dependencies_by_exact_local_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate_dependency_provenance: None,
) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    contributor = tomllib.loads(
        (ROOT / "dependencies/contributor-lock.toml").read_text(encoding="utf-8")
    )
    release = tomllib.loads(
        (ROOT / "dependencies/release-lock.toml").read_text(encoding="utf-8")
    )
    symbolica_version = contributor["symbolica"]["candidate_version"]
    loader_version = release["ufo_model_loader"]["required_version"]
    symbolica = _dependency_wheel(
        wheelhouse / "symbolica",
        "symbolica",
        symbolica_version,
        "cp311-abi3-test_platform",
    )
    release_symbolica = _dependency_wheel(
        wheelhouse / "release-symbolica",
        "symbolica",
        "2.1.0",
        "cp311-abi3-test_platform",
    )
    loader = _dependency_wheel(
        wheelhouse / "ufo-model-loader",
        "ufo-model-loader",
        loader_version,
        "py3-none-any",
    )
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append([os.fspath(item) for item in command])
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(release_deployment, "run", fake_run)
    virtual_env = tmp_path / "venv"
    installation = release_deployment._install_dependencies(
        Path(sys.executable),
        virtual_env=virtual_env,
        mode="candidate",
        wheelhouses=[wheelhouse],
        supported_tags=["cp311-abi3-test_platform", "py3-none-any"],
    )

    assert installation.local_wheels == {
        "symbolica": symbolica.resolve(),
        "ufo-model-loader": loader.resolve(),
    }
    assert installation.versions["symbolica"] == symbolica_version
    assert installation.versions["ufo-model-loader"] == loader_version
    assert len(commands) == 1
    command = commands[0]
    assert "--require-hashes" not in command
    assert "--find-links" not in command
    assert "--index-url" in command
    assert str(symbolica.resolve()) in command
    assert str(release_symbolica.resolve()) not in command
    assert str(loader.resolve()) in command
    assert not any(item.startswith("ufo-model-loader==") for item in command)
    assert "numpy==2.4.6" in command
    assert not any(item.startswith("python-utils==") for item in command)
    assert not any(item.startswith("typing-extensions==") for item in command)
    assert not any(item.startswith("wcwidth==") for item in command)
    assert not (tmp_path / "locked-requirements.txt").exists()


def test_candidate_deployment_rejects_ambiguous_local_patched_wheels(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    for directory in (first, second):
        _dependency_wheel(
            directory,
            "symbolica",
            "2.1.0",
            "cp311-abi3-test_platform",
        )
    with pytest.raises(ReleaseError, match="expected one compatible local symbolica"):
        release_deployment._candidate_dependency_wheels(
            {"symbolica": "2.1.0"},
            [first, second],
            ["cp311-abi3-test_platform", "py3-none-any"],
        )


def test_candidate_deployment_rejects_symlinked_wheelhouse(tmp_path: Path) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    linked = tmp_path / "linked-wheelhouse"
    linked.symlink_to(wheelhouse, target_is_directory=True)
    with pytest.raises(ReleaseError, match="wheelhouse may not be a symlink"):
        release_deployment._candidate_dependency_wheels(
            {"symbolica": "2.1.0"},
            [linked],
            ["py3-none-any"],
        )


def test_sdist_build_uses_clean_external_source_without_byte_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_wheel = tmp_path / "pyamplicol-0.1.0-cp311-abi3-test.whl"
    source_wheel.write_bytes(b"source wheel")
    sdist = tmp_path / "pyamplicol-0.1.0.tar.gz"
    sdist.write_bytes(b"sdist")
    scratch = tmp_path / "external"
    extracted = scratch / "unpacked" / "pyamplicol-0.1.0"
    extracted.mkdir(parents=True)
    observed: dict[str, object] = {}

    @contextlib.contextmanager
    def fake_temporary(_prefix: str):
        scratch.mkdir(exist_ok=True)
        yield scratch

    def fake_run(command, *, cwd, env, **_kwargs):
        observed["python"] = Path(command[0])
        observed["cwd"] = cwd
        observed["env"] = env
        output = Path(command[command.index("--outdir") + 1])
        output.mkdir(exist_ok=True)
        (output / source_wheel.name).write_bytes(b"independently rebuilt wheel")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setenv("PYTHONPATH", "/parent/source")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(build_from_sdist, "audit_sdist", lambda *_a, **_k: None)
    monkeypatch.setattr(
        build_from_sdist,
        "audit_wheel",
        lambda *_a, **_k: SimpleNamespace(
            version="0.1.0",
            python_tag="cp311",
            abi_tag="abi3",
            target="test",
            rust_target="test-target",
        ),
    )
    monkeypatch.setattr(
        build_from_sdist, "external_temporary_directory", fake_temporary
    )
    monkeypatch.setattr(build_from_sdist, "safe_extract_sdist", lambda *_a: extracted)
    monkeypatch.setattr(build_from_sdist, "run", fake_run)
    output = tmp_path / "retained"
    rebuilt = build_from_sdist.build_wheel_from_sdist(
        sdist,
        source_wheel,
        output,
        mode="release",
        python=Path(".venv/bin/python"),
    )
    assert rebuilt.read_bytes() == b"independently rebuilt wheel"
    assert observed["cwd"] == extracted
    assert observed["python"] == (tmp_path / ".venv/bin/python").resolve()
    assert "PYTHONPATH" not in observed["env"]


def test_release_gate_failure_prevents_artifact_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = False

    monkeypatch.setattr(
        build_release_artifacts, "require_clean_checkout", lambda **_kwargs: None
    )

    def closed_gate(*_args, **_kwargs):
        raise ReleaseError("closed by check_dependencies")

    def unexpected_build(*_args, **_kwargs):
        nonlocal built
        built = True

    monkeypatch.setattr(build_release_artifacts, "check_dependency_gate", closed_gate)
    monkeypatch.setattr(build_release_artifacts, "_build", unexpected_build)
    with pytest.raises(ReleaseError, match="check_dependencies"):
        build_release_artifacts.build_release_artifacts(
            tmp_path / "output",
            mode="release",
            python=Path(sys.executable),
            allow_dirty_candidate=False,
            sdist_only=False,
            retained_sdist_path=None,
        )
    assert built is False


def test_release_build_normalizes_python_without_inspecting_git_cleanliness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch = tmp_path / "scratch"

    @contextlib.contextmanager
    def fake_temporary(_prefix: str):
        scratch.mkdir()
        yield scratch

    def unexpected_cleanliness_check(**_kwargs) -> None:
        raise AssertionError("release builds must validate artifacts, not Git status")

    def fake_build(
        _python: Path,
        destination: Path,
        *,
        mode: str,
        wheel: bool,
        sdist: bool,
    ) -> None:
        assert _python == (tmp_path / ".venv/bin/python").resolve()
        assert mode == "release"
        assert wheel is False
        assert sdist is True
        (destination / "pyamplicol-0.1.0.tar.gz").write_bytes(b"sdist")

    monkeypatch.setattr(
        build_release_artifacts,
        "check_dependency_gate",
        lambda _mode: None,
    )
    monkeypatch.setattr(
        build_release_artifacts,
        "require_clean_checkout",
        unexpected_cleanliness_check,
    )
    monkeypatch.setattr(
        build_release_artifacts,
        "external_temporary_directory",
        fake_temporary,
    )
    monkeypatch.setattr(build_release_artifacts, "_build", fake_build)
    monkeypatch.setattr(build_release_artifacts, "audit_sdist", lambda *_a, **_k: None)

    monkeypatch.chdir(tmp_path)
    artifacts = build_release_artifacts.build_release_artifacts(
        tmp_path / "output",
        mode="release",
        python=Path(".venv/bin/python"),
        allow_dirty_candidate=False,
        sdist_only=True,
        retained_sdist_path=None,
    )

    assert [path.name for path in artifacts] == ["pyamplicol-0.1.0.tar.gz"]


def test_publish_dry_run_prints_but_never_executes_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    wheel = tmp_path / "pyamplicol-0.1.0-cp311-abi3-test.whl"
    sdist = tmp_path / "pyamplicol-0.1.0.tar.gz"
    wheel.write_bytes(b"wheel")
    sdist.write_bytes(b"sdist")
    commands: list[list[str]] = []
    clean_install_wheels: list[Path] = []

    monkeypatch.delenv("PYAMPLICOL_BUILD_MODE", raising=False)
    monkeypatch.setattr(
        publish_dry_run, "check_dependency_gate", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        publish_dry_run,
        "audit_wheel",
        lambda *_a, **_k: SimpleNamespace(target="test"),
    )
    monkeypatch.setattr(publish_dry_run, "audit_sdist", lambda *_a, **_k: None)

    def fake_run(command, **_kwargs):
        commands.append([os.fspath(item) for item in command])
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(publish_dry_run, "run", fake_run)
    monkeypatch.setattr(
        publish_dry_run,
        "_run_clean_install",
        lambda wheels: clean_install_wheels.extend(wheels),
    )
    assert (
        publish_dry_run.main(
            ["--artifact-dir", str(tmp_path), "--no-build", "--skip-twine-check"]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "twine upload" in output
    assert commands == []
    assert clean_install_wheels == [wheel]


def test_publish_dry_run_stages_plain_cross_platform_package_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incoming = tmp_path / "incoming"
    output = tmp_path / "upload"
    platforms = (
        "macosx_11_0_arm64",
        "macosx_11_0_x86_64",
        "manylinux_2_28_x86_64",
    )
    for index, platform in enumerate(platforms):
        directory = incoming / f"platform-{index}"
        directory.mkdir(parents=True)
        (directory / f"pyamplicol-0.1.0-cp311-abi3-{platform}.whl").write_bytes(
            platform.encode()
        )
    source = incoming / "source"
    source.mkdir(parents=True)
    (source / "pyamplicol-0.1.0.tar.gz").write_bytes(b"sdist")

    monkeypatch.delenv("PYAMPLICOL_BUILD_MODE", raising=False)
    monkeypatch.setattr(
        publish_dry_run, "check_dependency_gate", lambda *_a, **_k: None
    )
    monkeypatch.setattr(publish_dry_run, "audit_sdist", lambda *_a, **_k: None)

    def fake_audit_wheel(path: Path, **_kwargs):
        target = next(platform for platform in platforms if platform in path.name)
        return SimpleNamespace(target=target)

    monkeypatch.setattr(publish_dry_run, "audit_wheel", fake_audit_wheel)
    assert (
        publish_dry_run.main(
            [
                "--artifact-dir",
                str(incoming),
                "--output-dir",
                str(output),
                "--no-build",
                "--require-all-targets",
                "--skip-twine-check",
                "--skip-clean-install",
            ]
        )
        == 0
    )
    assert sorted(path.name for path in output.iterdir()) == [
        "pyamplicol-0.1.0-cp311-abi3-macosx_11_0_arm64.whl",
        "pyamplicol-0.1.0-cp311-abi3-macosx_11_0_x86_64.whl",
        "pyamplicol-0.1.0-cp311-abi3-manylinux_2_28_x86_64.whl",
        "pyamplicol-0.1.0.tar.gz",
    ]


def test_candidate_dry_run_accepts_wheel_only_and_withholds_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    wheel = tmp_path / "pyamplicol-candidate-cp311-abi3-test.whl"
    wheel.write_bytes(b"candidate wheel")
    monkeypatch.delenv("PYAMPLICOL_BUILD_MODE", raising=False)
    monkeypatch.setattr(
        publish_dry_run, "check_dependency_gate", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        publish_dry_run,
        "audit_wheel",
        lambda *_a, **_k: SimpleNamespace(target="test"),
    )
    assert (
        publish_dry_run.main(
            [
                "--candidate",
                "--artifact-dir",
                str(tmp_path),
                "--no-build",
                "--skip-twine-check",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "non-publishable" in output
    assert "twine upload" not in output

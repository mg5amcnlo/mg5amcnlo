# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "build_backend"))
sys.path.insert(0, str(ROOT / "tools" / "release"))

import package_version  # noqa: E402
import sync_version  # noqa: E402

RELEASE_VERSION = sync_version.madgraph_version()


def _copy_version_contract(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    (root / "dependencies").mkdir(parents=True)
    for relative in (
        Path("Cargo.toml"),
        Path("pyproject.toml"),
        Path("dependencies/release-lock.toml"),
        Path("dependencies/contributor-lock.toml"),
    ):
        shutil.copy2(ROOT / relative, root / relative)
    return root


def test_cargo_workspace_is_the_canonical_package_version() -> None:
    assert package_version.canonical_package_version(ROOT) == RELEASE_VERSION


def test_package_version_contract_rejects_release_lock_drift(tmp_path: Path) -> None:
    root = _copy_version_contract(tmp_path)
    lock = root / "dependencies" / "release-lock.toml"
    text = lock.read_text(encoding="utf-8")
    lock.write_text(
        text.replace(
            f'version = "{RELEASE_VERSION}"',
            'version = "9.9.9"',
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="package version mismatch"):
        package_version.canonical_package_version(root)


def test_package_version_contract_rejects_static_python_version(
    tmp_path: Path,
) -> None:
    root = _copy_version_contract(tmp_path)
    pyproject = root / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace(
            'dynamic = ["version"]',
            'version = "0.1.0"',
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match=r"project\.version as dynamic"):
        package_version.canonical_package_version(root)


def test_contributor_and_release_locks_share_compatibility_abi(
    tmp_path: Path,
) -> None:
    root = _copy_version_contract(tmp_path)
    package_version.check_contributor_lock_consistency(root)
    lock = root / "dependencies" / "contributor-lock.toml"
    lock.write_text(
        lock.read_text(encoding="utf-8").replace(
            'symbolica_serialization = "symbolica-bincode2-v1"',
            'symbolica_serialization = "incompatible-test-abi"',
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="dependency locks disagree"):
        package_version.check_contributor_lock_consistency(root)


def test_contributor_and_release_locks_share_symjit_plane_abi(
    tmp_path: Path,
) -> None:
    root = _copy_version_contract(tmp_path)
    package_version.check_contributor_lock_consistency(root)
    lock = root / "dependencies" / "contributor-lock.toml"
    lock.write_text(
        lock.read_text(encoding="utf-8").replace(
            'symjit_plane_application = "pyamplicol-symjit-plane-application-v2"',
            'symjit_plane_application = "incompatible-test-abi"',
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="plane-application ABI"):
        package_version.check_contributor_lock_consistency(root)


def test_contributor_lock_has_no_local_symjit_source_or_patch_inventory() -> None:
    contributor = (ROOT / "dependencies/contributor-lock.toml").read_text(
        encoding="utf-8"
    )

    assert "candidate_version = \"2.26.0\"" in contributor
    assert "revision = \"530304a07d1be6d5abc80291aa5e92cf28ac5546\"" in contributor
    assert "\npatches =" not in contributor
    assert not tuple((ROOT / "dependencies/patches/symjit").rglob("*.patch"))


def test_dev_install_keeps_all_build_caches_inside_the_workspace() -> None:
    justfile = (ROOT / "justfile").read_text(encoding="utf-8")
    match = re.search(
        r"(?ms)^dev-install(?: [^:\n]+)?:.*?"
        r"(?=^[A-Za-z0-9_-]+(?: [^:\n]+)?:|\Z)",
        justfile,
    )
    assert match is not None
    recipe = match.group(0)
    assert 'dev_cache := ".artifacts/dev-install"' in justfile
    for variable in (
        "TMPDIR",
        "CARGO_HOME",
        "CARGO_TARGET_DIR",
        "PIP_CACHE_DIR",
        "XDG_CACHE_HOME",
        "PYTHONPYCACHEPREFIX",
    ):
        assert recipe.count(f'{variable}="$PWD/{{{{dev_cache}}}}/') == 1
    assert recipe.count('PYAMPLICOL_CANDIDATE_CACHE_ROOT="$PWD/{{dev_cache}}"') == 1
    assert "*INSTALLER_ARGS" in recipe
    assert "dependencies/install_dependencies.py {{INSTALLER_ARGS}}" in recipe
    assert "prepare_source_runtime.py" not in recipe
    assert "PYAMPLICOL_BUILD_MODE=candidate" not in recipe


def test_dev_test_defaults_to_release_and_preserves_explicit_candidate_mode() -> None:
    justfile = (ROOT / "justfile").read_text(encoding="utf-8")
    assert (
        'build_mode := env_var_or_default("PYAMPLICOL_BUILD_MODE", "release")'
        in justfile
    )
    recipe = justfile.split("\ndev-test: _source-checkout\n", 1)[1].split("\n\n", 1)[0]
    assert "PYAMPLICOL_BUILD_MODE={{build_mode}} just source-gate" in recipe
    assert 'if [ "{{build_mode}}" = candidate ]' in recipe
    assert "PYAMPLICOL_BUILD_MODE=release just test-deployment" in recipe
    assert "PYAMPLICOL_BUILD_MODE=candidate just test-deployment-candidate" in recipe


@pytest.mark.parametrize(
    "recipe",
    (
        "dev-install",
        "dev-build",
        "dev-test",
        "legacy-physics",
        "legacy-physics-verify",
        "independent-physics-oracle",
        "test-deployment-candidate",
        "release-artifacts",
        "publish-dry-run",
    ),
)
def test_source_checkout_commands_fail_immediately_from_sdist(
    tmp_path: Path,
    recipe: str,
) -> None:
    just = shutil.which("just")
    if just is None:
        pytest.skip("just is not installed")
    source = tmp_path / "pyamplicol-0.1.0"
    source.mkdir()
    shutil.copy2(ROOT / "justfile", source / "justfile")
    marker = tmp_path / "python-was-invoked"
    fake_python = tmp_path / "python-must-not-run"
    fake_python.write_text(
        '#!/bin/sh\nprintf invoked > "$PYAMPLICOL_TEST_MARKER"\nexit 99\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = os.environ.copy()
    environment["PYTHON"] = str(fake_python)
    environment["PYAMPLICOL_TEST_MARKER"] = str(marker)

    completed = subprocess.run(
        [just, recipe],
        cwd=source,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "requires a full pyAmpliCol Git source checkout" in (
        completed.stdout + completed.stderr
    )
    assert not marker.exists()

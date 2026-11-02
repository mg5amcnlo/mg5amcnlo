# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from tools.release import prepare_release_prepared_models as producer

RELEASE_VERSION = producer.EXPECTED_VERSION


def _bootstrap_wheel(
    path: Path,
    *,
    publishable: bool = False,
    include_prepared_payload: bool = False,
    include_release_store: bool = False,
    include_selftest_fixture: bool = False,
) -> Path:
    marker = {
        "candidate_fingerprint": None,
        "native_build_inputs_sha256": "b" * 64,
        "publishable": publishable,
        "release_prepared_model_bootstrap": True,
        "schema_version": 1,
        "selftest_fixture_bootstrap": True,
        "source_checkout": str(producer.ROOT.resolve()),
        "version": RELEASE_VERSION,
    }
    metadata = (
        f"Metadata-Version: 2.4\nName: pyamplicol\nVersion: {RELEASE_VERSION}\n\n"
    ).encode()
    members = {
        "pyamplicol/_build_info.json": (
            json.dumps(marker, sort_keys=True) + "\n"
        ).encode(),
        "pyamplicol/assets/prepared_models/__init__.py": b"",
        f"pyamplicol-{RELEASE_VERSION}.dist-info/METADATA": metadata,
    }
    if include_prepared_payload:
        members[
            "pyamplicol/assets/prepared_models/built-in-sm-jit-o2-x86_64.metadata.json"
        ] = b"{}\n"
    if include_release_store:
        members[
            "release_assets/prepared_models/built-in-sm-jit-o2-x86_64.metadata.json"
        ] = b"{}\n"
    if include_selftest_fixture:
        members["pyamplicol/assets/selftest/portable-64le/expected.json"] = b"{}\n"
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return path


def test_release_bootstrap_wheel_is_explicitly_non_publishable(tmp_path: Path) -> None:
    wheel = _bootstrap_wheel(tmp_path / f"pyamplicol-{RELEASE_VERSION}.whl")

    result = producer.audit_bootstrap_wheel(wheel)

    assert result["version"] == RELEASE_VERSION
    assert result["publishable"] is False
    assert result["release_prepared_model_bootstrap"] is True
    assert result["selftest_fixture_bootstrap"] is True
    assert result["candidate_fingerprint"] is None


def test_release_bootstrap_wheel_rejects_publishable_marker(tmp_path: Path) -> None:
    wheel = _bootstrap_wheel(
        tmp_path / f"pyamplicol-{RELEASE_VERSION}.whl",
        publishable=True,
    )

    with pytest.raises(
        producer.ReleasePreparedModelError,
        match="marker publishable is invalid",
    ):
        producer.audit_bootstrap_wheel(wheel)


def test_release_bootstrap_wheel_rejects_prepared_payloads(tmp_path: Path) -> None:
    wheel = _bootstrap_wheel(
        tmp_path / f"pyamplicol-{RELEASE_VERSION}.whl",
        include_prepared_payload=True,
    )

    with pytest.raises(
        producer.ReleasePreparedModelError,
        match="stale prepared-model payloads",
    ):
        producer.audit_bootstrap_wheel(wheel)


def test_release_bootstrap_wheel_rejects_auxiliary_source_store(
    tmp_path: Path,
) -> None:
    wheel = _bootstrap_wheel(
        tmp_path / f"pyamplicol-{RELEASE_VERSION}.whl",
        include_release_store=True,
    )

    with pytest.raises(
        producer.ReleasePreparedModelError,
        match="release prepared-model source store",
    ):
        producer.audit_bootstrap_wheel(wheel)


def test_release_bootstrap_wheel_rejects_stale_selftest_fixture(
    tmp_path: Path,
) -> None:
    wheel = _bootstrap_wheel(
        tmp_path / f"pyamplicol-{RELEASE_VERSION}.whl",
        include_selftest_fixture=True,
    )

    with pytest.raises(
        producer.ReleasePreparedModelError,
        match="stale self-test fixture",
    ):
        producer.audit_bootstrap_wheel(wheel)


def test_release_bootstrap_builder_rejects_candidate_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYAMPLICOL_PREPARED_MODEL_BOOTSTRAP", "1")

    with pytest.raises(
        producer.ReleasePreparedModelError,
        match="cannot be combined",
    ):
        producer.build_bootstrap_wheel(tmp_path / "wheel")


def test_release_bootstrap_builder_rejects_candidate_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYAMPLICOL_BUILD_MODE", "candidate")

    with pytest.raises(
        producer.ReleasePreparedModelError,
        match="requires PYAMPLICOL_BUILD_MODE=release",
    ):
        producer.build_bootstrap_wheel(tmp_path / "wheel")

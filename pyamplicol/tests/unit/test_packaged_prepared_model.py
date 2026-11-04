# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from pyamplicol.assets import prepared_models
from pyamplicol.models.prepared_catalog import PREPARED_INDEPENDENT_BLOCK_PROOF
from pyamplicol.models.prepared_target import canonical_architecture

ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = ROOT / "src" / "pyamplicol" / "assets" / "prepared_models"
ASSET_STEM = f"built-in-sm-jit-o2-{canonical_architecture()}"


def _metadata() -> dict[str, object]:
    return json.loads(
        (ASSET_ROOT / f"{ASSET_STEM}.metadata.json").read_text(encoding="utf-8")
    )


def test_packaged_builtin_sm_jit_o2_is_discoverable_and_validated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyamplicol._internal.versions as versions
    import pyamplicol.models.loading as loading

    metadata = _metadata()
    producer = metadata["producer"]
    assert isinstance(producer, dict)
    monkeypatch.setattr(
        versions,
        "package_version",
        lambda: producer["package_version"],
    )
    monkeypatch.setattr(
        loading,
        "package_version",
        lambda: producer["package_version"],
    )

    assert prepared_models.available_prepared_models() == (
        prepared_models.BUILTIN_SM_JIT_O2,
        prepared_models.BUILTIN_SM_HEFT_JIT_O2,
    )
    with prepared_models.packaged_prepared_model_path(
        prepared_models.BUILTIN_SM_JIT_O2
    ) as path:
        from pyamplicol import ModelSource

        assert path.name == f"{ASSET_STEM}.pyamplicol-model"
        assert path.is_file()
        compiled = ModelSource.from_path(path).compile(use_cache=False)
        assert compiled.name == "built-in-sm"
        assert compiled.prepared_backend == "jit"
    with prepared_models.open_packaged_prepared_model() as bundle:
        assert bundle.backend == "jit"
        assert len(bundle.kernel_pack.kernels) == metadata["kernel_count"] == 53
        eligible_ids = {
            kernel.kernel_id
            for kernel in bundle.kernel_pack.kernels
            if PREPARED_INDEPENDENT_BLOCK_PROOF in kernel.proof_classes
        }
        assert len(eligible_ids) == 32
        assert {
            variant.base_kernel_id for variant in bundle.kernel_pack.kernel_variants
        } == eligible_ids
        assert all(
            variant.variant_id == "independent-block-4"
            for variant in bundle.kernel_pack.kernel_variants
        )
        assert bundle.kernel_pack.target["portable"] is True
        assert bundle.kernel_pack.target["cpu_features"] == ()


def test_packaged_builtin_sm_heft_jit_o2_is_selectable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyamplicol._internal.versions as versions
    import pyamplicol.models.loading as loading

    architecture = canonical_architecture()
    stem = f"built-in-sm-heft-jit-o2-{architecture}"
    metadata = json.loads(
        (ASSET_ROOT / f"{stem}.metadata.json").read_text(encoding="utf-8")
    )
    producer = metadata["producer"]
    assert isinstance(producer, dict)
    monkeypatch.setattr(
        versions, "package_version", lambda: producer["package_version"]
    )
    monkeypatch.setattr(loading, "package_version", lambda: producer["package_version"])

    with prepared_models.open_packaged_prepared_model(
        prepared_models.BUILTIN_SM_HEFT_JIT_O2
    ) as bundle:
        assert bundle.compiled_model["model"]["name"] == "built-in-sm-heft"
        assert bundle.compiled_model["source"]["kind"] == "built-in-sm-heft"
        assert bundle.backend == "jit"
        assert len(bundle.kernel_pack.kernels) == metadata["kernel_count"]


def test_packaged_prepared_model_materializes_stable_cached_copy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pyamplicol._internal.versions as versions
    import pyamplicol.models.loading as loading

    metadata = _metadata()
    producer = metadata["producer"]
    assert isinstance(producer, dict)
    monkeypatch.setattr(
        versions,
        "package_version",
        lambda: producer["package_version"],
    )
    monkeypatch.setattr(loading, "package_version", lambda: producer["package_version"])

    first = prepared_models.materialize_packaged_prepared_model(cache_dir=tmp_path)
    second = prepared_models.materialize_packaged_prepared_model(cache_dir=tmp_path)

    assert first == second
    assert first.is_file()
    assert first.parent.name == _metadata()["bundle_sha256"]
    assert (
        first.read_bytes()
        == (ASSET_ROOT / f"{ASSET_STEM}.pyamplicol-model").read_bytes()
    )


def test_packaged_prepared_model_rejects_unknown_identity() -> None:
    with (
        pytest.raises(
            prepared_models.PackagedPreparedModelError,
            match="unknown packaged prepared model",
        ),
        prepared_models.packaged_prepared_model_path("not-a-model"),
    ):
        pass


def test_packaged_prepared_model_rejects_unsupported_host_architecture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "pyamplicol.models.prepared_target.platform.machine",
        lambda: "sparc64",
    )
    with (
        pytest.raises(
            prepared_models.PackagedPreparedModelError,
            match="host architecture",
        ),
        prepared_models.packaged_prepared_model_path(prepared_models.BUILTIN_SM_JIT_O2),
    ):
        pass


def test_packaged_prepared_model_rejects_resource_tampering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    copied = tmp_path / "prepared_models"
    shutil.copytree(ASSET_ROOT, copied)
    bundle = copied / f"{ASSET_STEM}.pyamplicol-model"
    bundle.write_bytes(bundle.read_bytes() + b"tampered")
    monkeypatch.setattr(prepared_models.resources, "files", lambda _package: copied)

    with (
        pytest.raises(
            prepared_models.PackagedPreparedModelError,
            match="size does not match",
        ),
        prepared_models.packaged_prepared_model_path(prepared_models.BUILTIN_SM_JIT_O2),
    ):
        pass


def test_packaged_prepared_model_accepts_older_package_producer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyamplicol._internal.versions as versions

    producer = _metadata()["producer"]
    assert isinstance(producer, dict)
    recorded = str(producer["package_version"])
    release = recorded.split(".dev", maxsplit=1)[0].split("+", maxsplit=1)[0]
    major, minor, patch = (int(component) for component in release.split("."))
    active = f"{major}.{minor}.{patch + 1}"
    monkeypatch.setattr(versions, "package_version", lambda: active)

    with prepared_models.packaged_prepared_model_path(
        prepared_models.BUILTIN_SM_JIT_O2
    ) as path:
        assert path.is_file()


def test_packaged_prepared_model_accepts_candidate_fingerprint_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pyamplicol._internal.versions as versions

    metadata = _metadata()
    producer = metadata["producer"]
    assert isinstance(producer, dict)
    recorded = str(producer["package_version"])
    active = recorded.rsplit("+candidate.", maxsplit=1)[0] + ("+candidate." + "0" * 12)
    monkeypatch.setattr(versions, "package_version", lambda: active)

    with prepared_models.packaged_prepared_model_path(
        prepared_models.BUILTIN_SM_JIT_O2
    ) as path:
        assert path.is_file()


def test_packaged_prepared_model_rejects_internal_producer_version_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pyamplicol._internal.versions as versions

    copied = tmp_path / "prepared_models"
    shutil.copytree(ASSET_ROOT, copied)
    metadata = _metadata()
    producer = metadata["producer"]
    assert isinstance(producer, dict)
    recorded = str(producer["package_version"])
    producer["package_version"] = recorded.rsplit("+candidate.", maxsplit=1)[0] + (
        "+candidate." + "0" * 12
    )
    (copied / f"{ASSET_STEM}.metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        versions,
        "package_version",
        lambda: producer["package_version"],
    )
    monkeypatch.setattr(prepared_models.resources, "files", lambda _package: copied)

    with (
        pytest.raises(
            prepared_models.PackagedPreparedModelError,
            match="package version disagrees with metadata",
        ),
        prepared_models.packaged_prepared_model_path(prepared_models.BUILTIN_SM_JIT_O2),
    ):
        pass


def test_packaged_prepared_model_runtime_ignores_producer_source_fingerprints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pyamplicol._internal.versions as versions
    import pyamplicol.models.loading as loading

    copied = tmp_path / "prepared_models"
    shutil.copytree(ASSET_ROOT, copied)
    metadata = _metadata()
    producer = metadata["producer"]
    assert isinstance(producer, dict)
    for key in (
        "model_compiler_sha256",
        "model_source_digest",
        "native_build_inputs_sha256",
    ):
        producer[key] = "0" * 64
    (copied / f"{ASSET_STEM}.metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    package_version = producer["package_version"]
    monkeypatch.setattr(
        versions,
        "package_version",
        lambda: package_version,
    )
    monkeypatch.setattr(loading, "package_version", lambda: package_version)
    monkeypatch.setattr(prepared_models.resources, "files", lambda _package: copied)

    with prepared_models.packaged_prepared_model_path(
        prepared_models.BUILTIN_SM_JIT_O2
    ) as path:
        assert path.is_file()

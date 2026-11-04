# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from pyamplicol import ArtifactError
from pyamplicol.artifacts import load_manifest
from pyamplicol.artifacts.manifest import ArtifactPayloads, validate_payloads
from pyamplicol.generation.evaluator_container import PacbinReader
from pyamplicol.runtime._evaluator_payloads import ExactEvaluatorPayloadResolver
from pyamplicol.runtime.eager_exact._contracts import _PayloadIndex
from tests.unit.test_artifacts import _build_packable

_EXECUTION = "processes/dd_to_z/execution.json"
_BINDING = "processes/dd_to_z/recurrence-binding.bin"


@pytest.mark.parametrize("pack", [False, True])
def test_payloads_resolve_loose_and_packed_layouts(tmp_path: Path, pack: bool) -> None:
    root = tmp_path / "artifact"
    _build_packable(root, pack=pack)
    manifest = load_manifest(root, verify_payloads=True)
    payloads = ArtifactPayloads(manifest)

    assert payloads.is_packed(_EXECUTION) is pack
    assert (payloads.container_path is not None) is pack
    assert json.loads(payloads.read_text(_EXECUTION)) == {"plan": [1, 2, 3]}
    assert payloads.read_bytes(_BINDING) == b"binding"
    # nested containers and top-level payloads stay loose
    assert not payloads.is_packed("processes/dd_to_z/eager-runtime.pacbin")
    assert (
        payloads.read_bytes("physics/process.json")
        == (root / "physics/process.json").read_bytes()
    )
    assert _PayloadIndex.from_manifest(manifest).read_json(_EXECUTION, "execution") == {
        "plan": [1, 2, 3]
    }
    assert ExactEvaluatorPayloadResolver(manifest).payloads.read_bytes(_BINDING) == (
        b"binding"
    )


def _corrupt_member(root: Path, logical_path: str) -> None:
    container = root / "payloads.pacbin"
    with PacbinReader.open(container, verify_payloads=False) as reader:
        offset = reader.member(logical_path).offset
    data = bytearray(container.read_bytes())
    data[offset] ^= 0xFF
    container.chmod(0o644)
    container.write_bytes(bytes(data))


def test_packed_payload_reads_verify_member_digests(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _build_packable(root, pack=True)
    _corrupt_member(root, _BINDING)
    manifest = load_manifest(root)
    payloads = ArtifactPayloads(manifest)

    assert json.loads(payloads.read_text(_EXECUTION)) == {"plan": [1, 2, 3]}
    with pytest.raises(ArtifactError, match="digest mismatch"):
        payloads.read_bytes(_BINDING)
    with pytest.raises(ArtifactError, match="payload container verification"):
        validate_payloads(manifest)


def test_packed_payloads_may_not_also_be_files(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _build_packable(root, pack=True)
    shadow = root / _EXECUTION
    shadow.parent.mkdir(parents=True, exist_ok=True)
    shadow.write_text('{"plan": []}', encoding="utf-8")

    with pytest.raises(ArtifactError, match="also stored as a file"):
        load_manifest(root, verify_payloads=True)


def _pack_artifact(source: Path, root: Path) -> None:
    """Copy ``source`` and move its packable payloads into payloads.pacbin."""

    from pyamplicol.artifacts.manifest import compute_artifact_id
    from pyamplicol.artifacts.security import sha256_file
    from pyamplicol.generation.artifact_writer import _packs_payload
    from pyamplicol.generation.evaluator_container import (
        PacbinMemberKind,
        PacbinMemberSource,
        write_pacbin,
    )

    shutil.copytree(source, root)
    raw = json.loads((root / "artifact.json").read_text(encoding="utf-8"))
    packed = [
        record["path"] for record in raw["payloads"] if _packs_payload(record["path"])
    ]
    assert packed
    container = root / "payloads.pacbin"
    write_pacbin(
        container,
        [
            PacbinMemberSource(path, PacbinMemberKind.DECLARED_PAYLOAD, root / path)
            for path in packed
        ],
    )
    for path in packed:
        (root / path).unlink()
    raw["payloads"] = sorted(
        [
            *raw["payloads"],
            {
                "executable": False,
                "media_type": "application/octet-stream",
                "path": "payloads.pacbin",
                "role": "payload-container",
                "sha256": sha256_file(container),
                "size_bytes": container.stat().st_size,
            },
        ],
        key=lambda record: record["path"],
    )
    raw["artifact_id"] = compute_artifact_id(raw)
    (root / "artifact.json").write_text(json.dumps(raw), encoding="utf-8")


def test_inspection_reads_packed_payloads(tmp_path: Path) -> None:
    from pyamplicol.artifacts import inspect_artifact

    source = (
        Path(__file__).resolve().parents[2]
        / "src/pyamplicol/assets/selftest/portable-64le/artifact"
    )
    root = tmp_path / "packed"
    _pack_artifact(source, root)
    assert ArtifactPayloads(load_manifest(root)).is_packed(
        "processes/d_dbar_to_z/physics.json"
    )
    assert not (root / "processes/d_dbar_to_z/physics.json").exists()

    loose = inspect_artifact(source)
    packed = inspect_artifact(root)
    assert packed.integrity == "verified"
    assert packed.processes == loose.processes

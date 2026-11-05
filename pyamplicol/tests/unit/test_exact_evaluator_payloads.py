# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import io
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from pyamplicol.api.errors import ArtifactError
from pyamplicol.artifacts.manifest import ArtifactManifest, PayloadRecord
from pyamplicol.artifacts.security import sha256_file
from pyamplicol.generation.evaluator_container import (
    PacbinMemberKind,
    PacbinMemberSource,
    write_pacbin_atomic,
)
from pyamplicol.models.prepared import PreparedKernelRecord
from pyamplicol.runtime._evaluator_payloads import ExactEvaluatorPayloadResolver
from pyamplicol.runtime.eager_exact._contracts import _artifact_kernel_loader
from pyamplicol.runtime.symbolica_exact import _ExactEvaluator


class _LoadedEvaluator:
    def evaluate_complex_with_prec(
        self, _values: object, _precision: int
    ) -> tuple[tuple[int, int], ...]:
        return ((0, 0),)


class _EvaluatorFactory:
    loaded: ClassVar[list[bytes]] = []

    @classmethod
    def load(cls, state: bytes) -> _LoadedEvaluator:
        cls.loaded.append(state)
        return _LoadedEvaluator()


@pytest.fixture(autouse=True)
def _fake_symbolica(monkeypatch: pytest.MonkeyPatch) -> None:
    _EvaluatorFactory.loaded = []
    monkeypatch.setitem(
        sys.modules,
        "symbolica",
        SimpleNamespace(Evaluator=_EvaluatorFactory),
    )


def _payload_record(
    root: Path,
    relative: str,
    *,
    process_id: str | None,
) -> PayloadRecord:
    path = root / relative
    return PayloadRecord(
        path=relative,
        role="evaluator-state",
        media_type="application/octet-stream",
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
        executable=False,
        target=None,
        process_id=process_id,
    )


def _manifest(
    root: Path,
    payloads: tuple[PayloadRecord, ...],
    *,
    extension: dict[str, object] | None = None,
) -> ArtifactManifest:
    return ArtifactManifest(
        root=root,
        kind="pyamplicol-process",
        artifact_id="0" * 64,
        created_utc="2026-01-01T00:00:00Z",
        producer={},
        model={},
        configuration={},
        processes=(),
        default_process_id=None,
        runtime={},
        payloads=payloads,
        dependencies=(),
        extensions=(
            {}
            if extension is None
            else {"evaluator_payload_container": extension}
        ),
    )


def _packed_manifest(
    root: Path,
    logical_path: str,
    state: bytes,
) -> ArtifactManifest:
    root.mkdir()
    container = root / "evaluators.pacbin"
    index = write_pacbin_atomic(
        container,
        (
            PacbinMemberSource(
                logical_path,
                PacbinMemberKind.SYMBOLICA_EXACT_STATE,
                io.BytesIO(state),
            ),
        ),
    )
    extension = {
        "kind": "pyamplicol-evaluator-payload-container",
        "schema_version": 1,
        "storage_abi": "pacbin-v1",
        "path": "evaluators.pacbin",
        "member_count": 1,
        "unpacked_size_bytes": len(state),
        "index_sha256": index.index_sha256,
    }
    return _manifest(
        root,
        (_payload_record(root, "evaluators.pacbin", process_id=None),),
        extension=extension,
    )


def _loose_manifest(
    root: Path,
    logical_path: str,
    state: bytes,
    *,
    process_id: str | None,
) -> ArtifactManifest:
    path = root / logical_path
    path.parent.mkdir(parents=True)
    path.write_bytes(state)
    return _manifest(
        root,
        (_payload_record(root, logical_path, process_id=process_id),),
    )


@pytest.mark.parametrize("packed", [False, True])
def test_compiled_exact_evaluator_loads_loose_and_packed_state(
    tmp_path: Path,
    packed: bool,
) -> None:
    state = b"compiled-exact-state"
    logical = "processes/process/stage.evaluator.bin"
    manifest = (
        _packed_manifest(tmp_path / "artifact", logical, state)
        if packed
        else _loose_manifest(
            tmp_path / "artifact",
            logical,
            state,
            process_id="process",
        )
    )
    resolver = ExactEvaluatorPayloadResolver(manifest)
    evaluator = _ExactEvaluator.load(
        {"evaluator_state_path": "stage.evaluator.bin", "input_len": 2},
        manifest.root / "processes/process",
        state_loader=lambda relative: resolver.read_exact_state(
            f"processes/process/{relative}",
            process_id="process",
        ),
    )

    assert evaluator.input_len == 2
    assert _EvaluatorFactory.loaded == [state]


def test_packed_exact_state_enforces_process_path_ownership(tmp_path: Path) -> None:
    manifest = _packed_manifest(
        tmp_path / "artifact",
        "processes/other/stage.evaluator.bin",
        b"state",
    )
    resolver = ExactEvaluatorPayloadResolver(manifest)

    with pytest.raises(ArtifactError, match="does not belong to process"):
        resolver.require_exact_state(
            "processes/other/stage.evaluator.bin",
            process_id="requested",
        )


def _kernel() -> PreparedKernelRecord:
    return PreparedKernelRecord(
        kernel_id=7,
        contract_kind="vertex",
        canonical_signature="test:vertex",
        input_arity=2,
        output_arity=1,
        input_layout=("left-current:0", "right-current:0"),
        input_contracts=(
            {
                "role": "left-current",
                "component": 0,
                "symbol": "test::left-current::0",
                "model_parameter_name": None,
                "model_parameter_index": None,
            },
            {
                "role": "right-current",
                "component": 0,
                "symbol": "test::right-current::0",
                "model_parameter_name": None,
                "model_parameter_index": None,
            },
        ),
        output_layout=("scalar:0",),
        exact_expressions=("out",),
        exact_evaluator_state_path="kernels/000007/exact.evaluator.bin",
        f64_evaluator_manifest={"kind": "test-evaluator"},
    )


@pytest.mark.parametrize("packed", [False, True])
def test_eager_exact_kernel_loads_loose_and_packed_state(
    tmp_path: Path,
    packed: bool,
) -> None:
    state = b"eager-exact-state"
    prefix = "model/eager-kernels"
    logical = f"{prefix}/{_kernel().exact_evaluator_state_path}"
    manifest = (
        _packed_manifest(tmp_path / "artifact", logical, state)
        if packed
        else _loose_manifest(
            tmp_path / "artifact",
            logical,
            state,
            process_id=None,
        )
    )
    resolver = ExactEvaluatorPayloadResolver(manifest)
    loader = _artifact_kernel_loader(resolver, prefix)
    evaluator = loader(_kernel(), manifest.root / prefix)

    assert evaluator.input_len == 2
    assert _EvaluatorFactory.loaded == [state]

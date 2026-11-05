# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import pyamplicol.runtime.backend as backend_module
from pyamplicol.runtime.backend import RusticolRuntimeBackend


class _NativeRuntime:
    @staticmethod
    def _exact_runtime_state_json() -> str:
        return json.dumps(
            {
                "model_parameter_values": [],
                "normalization_factor": 1.0,
                "representative_process_id": "representative",
                "representative_process_key": "representative",
                "external_permutation": [1, 0, 3, 2],
            }
        )


def test_validation_momenta_uses_native_inferred_process_permutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vectors = tuple((float(index), 0.0, 0.0, 0.0) for index in range(4))
    (tmp_path / "validation.json").write_text(
        json.dumps(
            {
                "available": True,
                "points": [
                    [
                        {"pdg": pdg, "momentum": list(vector)}
                        for pdg, vector in zip(
                            (2, -2, 23, 21), vectors, strict=True
                        )
                    ]
                ],
            }
        ),
        encoding="utf-8",
    )
    process = {
        "id": "representative",
        "external_pdgs": [2, -2, 23, 21],
        "aliases": [],
    }
    manifest = SimpleNamespace(
        root=tmp_path,
        processes=(process,),
        payloads=(
            SimpleNamespace(
                role="validation-momenta",
                process_id="representative",
                path="validation.json",
            ),
        ),
    )
    monkeypatch.setattr(backend_module, "load_manifest", lambda _path: manifest)
    runtime = object.__new__(RusticolRuntimeBackend)
    runtime._artifact_path = tmp_path
    runtime._runtime = _NativeRuntime()
    runtime._physics = SimpleNamespace(process_id="representative")

    assert runtime.validation_momenta() == (
        (vectors[1], vectors[0], vectors[3], vectors[2]),
    )

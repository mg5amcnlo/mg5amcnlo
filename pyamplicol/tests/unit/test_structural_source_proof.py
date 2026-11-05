# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import io
import json
from copy import deepcopy
from pathlib import Path

import pytest

from pyamplicol.generation.evaluator_container import (
    PacbinMemberKind,
    PacbinMemberSource,
    write_pacbin_atomic,
)
from pyamplicol.generation.structural_source_proof import (
    _payload_paths,
    build_generation_structural_proof,
    validate_generation_structural_proof,
)

REVISION = "a" * 40
NATIVE_INPUTS = "b" * 64


def _fixture(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    execution: dict[str, object] = {
        "schema_version": 3,
        "kind": "pyamplicol-runtime-execution",
        "key": "process",
        "process": "d d~ > z",
        "color_accuracy": "lc",
        "external_pdg_order": [1, -1, 23],
        "compiled": {
            "stage_evaluators": {
                "application_path": "processes/process/stage.symjit",
                "settings": {"compiler_path": "/usr/bin/c++"},
            }
        },
        "runtime_schema": {
            "source_fill": {"source_count": 3},
            "amplitude_stage": {"output_count": 1},
            "stages": [{"interaction_count": 7}],
        },
        "dag_summary": {
            "source_count": 3,
            "current_count": 5,
            "interaction_count": 7,
            "interaction_evaluation_count": 6,
            "amplitude_root_count": 1,
        },
    }
    execution_path = "processes/process/execution.json"
    concrete = root / execution_path
    concrete.parent.mkdir(parents=True)
    concrete.write_text(json.dumps(execution, sort_keys=True))
    container = root / "evaluators.pacbin"
    index = write_pacbin_atomic(
        container,
        [
            PacbinMemberSource(
                "processes/process/stage.symjit",
                PacbinMemberKind.SYMJIT_APPLICATION,
                io.BytesIO(b"compiled evaluator"),
            )
        ],
    )
    proof = build_generation_structural_proof(
        artifact_root=root,
        process_id="process",
        source_revision=REVISION,
        native_build_inputs_sha256=NATIVE_INPUTS,
        execution_path=execution_path,
        execution_sha256=hashlib.sha256(concrete.read_bytes()).hexdigest(),
        execution=execution,
        evaluator_container_path="evaluators.pacbin",
        evaluator_container_index_sha256=index.index_sha256,
    )
    return execution, proof


def test_generation_proof_authenticates_semantics_lanes_and_pacbin(
    tmp_path: Path,
) -> None:
    execution, proof = _fixture(tmp_path)

    assert (
        validate_generation_structural_proof(
            proof,
            artifact_root=tmp_path,
            expected_process_id="process",
            expected_source_revision=REVISION,
            expected_native_build_inputs_sha256=NATIVE_INPUTS,
            execution=execution,
        )
        == proof
    )
    inventory = proof["physical_lane_inventory"]
    assert inventory["pacbin_members"] == [
        {
            "kind": "symjit_application",
            "length": 18,
            "logical_path": "processes/process/stage.symjit",
            "sha256": hashlib.sha256(b"compiled evaluator").hexdigest(),
        }
    ]
    assert inventory["lanes"][0]["structural_metrics"]


def test_generation_proof_rejects_semantic_and_member_tampering(
    tmp_path: Path,
) -> None:
    execution, proof = _fixture(tmp_path)
    changed = deepcopy(proof)
    changed["semantic_maps"]["current_member_map"]["rows"][0]["value"] = 99
    with pytest.raises(ValueError, match="semantic maps do not recompute"):
        validate_generation_structural_proof(
            changed,
            artifact_root=tmp_path,
            expected_process_id="process",
            expected_source_revision=REVISION,
            execution=execution,
        )

    (tmp_path / "evaluators.pacbin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="pacbin container changed"):
        validate_generation_structural_proof(
            proof,
            artifact_root=tmp_path,
            expected_process_id="process",
            expected_source_revision=REVISION,
            execution=execution,
        )


def test_generation_proof_keeps_payload_paths_confined() -> None:
    assert _payload_paths(
        {
            "application_path": "processes/process/stage.symjit",
            "compiler_path": "/usr/bin/c++",
        }
    ) == ("processes/process/stage.symjit",)
    with pytest.raises(ValueError, match="execution payload path is not confined"):
        _payload_paths({"application_path": "/outside/stage.symjit"})

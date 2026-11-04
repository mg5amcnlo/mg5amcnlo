# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pytest

from tools.developer import compiled_mode_matrix as matrix
from tools.developer import compiled_mode_matrix_x86 as x86


def test_frozen_partition_is_complete_disjoint_and_group_local() -> None:
    observed_cells: list[str] = []
    observed_groups: list[str] = []
    for shard_index in range(x86.SHARD_COUNT):
        groups = x86.shard_groups(shard_index)
        cells = x86.shard_cells(shard_index)
        assert len(groups) == 7
        assert len(cells) == 21
        assert Counter(cell.category for cell in cells) == {
            "primary": 3,
            "medium": 15,
            "color-heavy": 3,
        }
        for group in groups:
            assert tuple(cell.batch_size for cell in group) == matrix.BATCH_SIZES
            assert len({cell.artifact_group_id for cell in group}) == 1
            assert all(
                x86.artifact_group_shard(group) == shard_index for _ in group
            )
            observed_groups.append(group[0].artifact_group_id)
        observed_cells.extend(cell.cell_id for cell in cells)
    assert len(observed_groups) == len(set(observed_groups)) == 56
    assert len(observed_cells) == len(set(observed_cells)) == 168
    assert set(observed_cells) == {cell.cell_id for cell in matrix.CANONICAL_CELLS}


def test_frozen_partition_spreads_primary_and_union_work() -> None:
    primary = []
    union_counts = []
    for shard_index in range(x86.SHARD_COUNT):
        groups = x86.shard_groups(shard_index)
        primary_groups = [group for group in groups if group[0].category == "primary"]
        assert len(primary_groups) == 1
        primary.append(primary_groups[0][0].artifact_group_id)
        union_counts.append(
            sum(group[0].lc_flow_layout == "all-flow-union" for group in groups)
        )
    assert len(set(primary)) == 8
    assert max(union_counts) <= 3
    assert min(union_counts) >= 1


def test_partition_definition_is_content_stable() -> None:
    definition = x86.PARTITION_DEFINITION
    body = dict(definition)
    digest = body.pop("sha256")
    assert definition["contract"] == x86.PARTITION_CONTRACT
    assert definition["shard_count"] == 8
    assert digest == x86._canonical_sha256(body)
    assert set(definition["assignments"]) == {str(index) for index in range(8)}


def test_checked_json_rejects_duplicates_nonfinite_and_symlinks(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}\n', encoding="utf-8")
    with pytest.raises(x86.ShardError, match="duplicate"):
        x86._checked_json(duplicate, label="duplicate")
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"a":NaN}\n', encoding="utf-8")
    with pytest.raises(x86.ShardError, match="non-finite"):
        x86._checked_json(nonfinite, label="nonfinite")
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(x86.ShardError, match="non-symlink"):
        x86._checked_json(link, label="link")


def test_content_identity_rejects_semantic_tampering() -> None:
    payload = x86._attach_content_identity({"passes": True, "value": 7})
    x86._require_content_identity(payload, label="payload")
    payload["value"] = 8
    with pytest.raises(x86.ShardError, match="does not match"):
        x86._require_content_identity(payload, label="payload")


def _arguments(tmp_path: Path) -> argparse.Namespace:
    sha = "a" * 64
    git = "b" * 40
    return argparse.Namespace(
        shard_count=8,
        shard_root=tmp_path / "shards",
        output_root=tmp_path / "matrix",
        aggregate_result=tmp_path / "aggregate.json",
        workflow_run_id="12345",
        runtime_bundle_sha256=sha,
        expected_baseline_source_revision=(
            matrix.FROZEN_BASELINE_SOURCE_REVISION
        ),
        expected_current_source_revision=git,
        expected_baseline_native_inputs_sha256=(
            matrix.FROZEN_BASELINE_NATIVE_INPUTS_SHA256
        ),
        expected_current_native_inputs_sha256="c" * 64,
        expected_baseline_distribution_sha256="d" * 64,
        expected_current_distribution_sha256="e" * 64,
        expected_baseline_native_module_sha256="f" * 64,
        expected_current_native_module_sha256="1" * 64,
        baseline_python=Path("/private/tmp/arena/baseline/bin/python"),
        current_python=Path("/private/tmp/arena/current/bin/python"),
        baseline_dependency_site=Path("/private/tmp/arena/dependencies"),
        current_dependency_site=Path("/private/tmp/arena/dependencies"),
        ufo_sm_model=Path(
            "/workspace/src/pyamplicol/assets/models/json/sm/sm.json"
        ),
    )


def _fake_evidence(
    cell: matrix.MatrixCell,
    *_args: object,
    **_kwargs: object,
) -> dict[str, object]:
    return {
        "cell_id": cell.cell_id,
        "configuration": matrix.asdict(cell),
        "result_content_sha256": x86._canonical_sha256({}),
        "errors": [],
        "passes": True,
        "gain_gate_passes": cell.category == "primary",
        "relative_gain": 0.2,
        "generation_current_over_baseline": 1.0,
        "runtime_identity_sha256_by_lane": {
            "baseline": "2" * 64,
            "current": "3" * 64,
        },
        "native_module_sha256_by_lane": {
            "baseline": "f" * 64,
            "current": "1" * 64,
        },
        "artifact_identity_sha256_by_lane": {
            "baseline": "4" * 64,
            "current": "5" * 64,
        },
        "provenance_sha256": {
            "driver": "6" * 64,
            "watchdog": "7" * 64,
            "native_sample_helper": "8" * 64,
            "dependency_entry": "9" * 64,
            "interpreters": "a" * 64,
            "dependency_sites": "b" * 64,
            "model": ("c" if cell.model_kind == "built-in" else "d") * 64,
        },
    }


def _write_aggregate_fixture(
    arguments: argparse.Namespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(matrix, "_cell_evidence", _fake_evidence)
    monkeypatch.setattr(
        matrix,
        "_repository_identity",
        lambda: {
            "root": "/workspace",
            "head_revision": arguments.expected_current_source_revision,
            "clean": True,
            "dirty_entries": [],
        },
    )
    arguments.shard_root.mkdir(parents=True, exist_ok=True)
    for index in range(8):
        cells = x86.shard_cells(index)
        result_files = []
        evidence = []
        for cell in cells:
            path = arguments.output_root / "cells" / cell.cell_id / "result.json"
            x86._write_json_atomic(path, {})
            _, identity = x86._checked_json(path, label=cell.cell_id)
            result_files.append({"cell_id": cell.cell_id, **identity})
            evidence.append(_fake_evidence(cell))
        stable = {"same": True}
        payload = x86._attach_content_identity(
            {
                "kind": x86.SHARD_KIND,
                "schema_version": x86.SCHEMA_VERSION,
                "matrix_contract": matrix.MATRIX_CONTRACT,
                "partition": x86.PARTITION_DEFINITION,
                "shard_count": 8,
                "shard_index": index,
                "workflow_run_id": arguments.workflow_run_id,
                "runtime_bundle_sha256": arguments.runtime_bundle_sha256,
                "output_root": str(arguments.output_root.resolve()),
                "expected_builds": x86._expected_builds(arguments),
                "platform": {
                    "platform": f"Linux-shard-{index}",
                    "system": "Linux",
                    "machine": "x86_64",
                },
                "selected_artifact_group_ids": [
                    group[0].artifact_group_id for group in x86.shard_groups(index)
                ],
                "selected_cell_ids": [cell.cell_id for cell in cells],
                "result_files": result_files,
                "cell_evidence": evidence,
                "cell_gate": {"failures": {}, "passes": True},
                "artifact_postflight_gate": {"errors": [], "passes": True},
                "provenance": {
                    "preflight": stable,
                    "postflight": stable,
                    "all_match": True,
                    "shard_driver": {"sha256": "e" * 64},
                },
                "complete": True,
                "passes": True,
            }
        )
        x86._write_json_atomic(arguments.shard_root / f"shard-{index}.json", payload)


def test_pure_json_aggregate_accepts_exact_eight_shard_partition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _arguments(tmp_path)
    _write_aggregate_fixture(arguments, monkeypatch)
    payload = x86._aggregate(arguments)
    assert payload["complete"] is True
    assert payload["passes"] is True
    assert payload["matrix_audit"]["coverage"]["observed"] == 168
    assert payload["shard_gate"]["passes"] is True
    x86._require_content_identity(payload, label="aggregate")


def test_aggregate_rejects_missing_or_tampered_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _arguments(tmp_path)
    _write_aggregate_fixture(arguments, monkeypatch)
    (arguments.shard_root / "shard-7.json").unlink()
    with pytest.raises(x86.ShardError, match="manifest set"):
        x86._aggregate(arguments)

    _write_aggregate_fixture(arguments, monkeypatch)
    path = arguments.shard_root / "shard-3.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["selected_cell_ids"] = payload["selected_cell_ids"][:-1]
    x86._write_json_atomic(path, payload)
    with pytest.raises(x86.ShardError, match="content identity"):
        x86._aggregate(arguments)


def test_aggregate_rejects_changed_result_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments = _arguments(tmp_path)
    _write_aggregate_fixture(arguments, monkeypatch)
    cell = x86.shard_cells(0)[0]
    path = arguments.output_root / "cells" / cell.cell_id / "result.json"
    x86._write_json_atomic(path, {"tampered": True})
    payload = x86._aggregate(arguments)
    assert payload["passes"] is False
    assert any(
        "result file identity changed" in error
        for error in payload["shard_gate"]["errors"]
    )

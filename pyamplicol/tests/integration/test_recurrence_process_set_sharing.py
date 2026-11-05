# SPDX-License-Identifier: 0BSD
"""Guarded acceptance test for real recurrence process-set schedule sharing.

Run the full acceptance explicitly under the repository memory watchdog:

.. code-block:: console

   PYAMPLICOL_RUN_RECURRENCE_PROCESS_SET_ACCEPTANCE=1 \
     .venv/bin/python tools/ci/memory_watchdog.py --limit-gib 30 -- \
     .venv/bin/python -m pytest -q \
     tests/integration/test_recurrence_process_set_sharing.py

The caller must provide the normal Symbolica license environment. The test is
skipped by default because it generates the bounded one-flavour ``p p > 4j``
process set and loads representative native schedules.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from pyamplicol import Generator, ModelSource, ProcessSet, Runtime
from pyamplicol.artifacts import load_manifest
from pyamplicol.assets.prepared_models import (
    BUILTIN_SM_JIT_O2,
    packaged_prepared_model_path,
)
from pyamplicol.config import (
    Action,
    ColorAccuracy,
    ColorConfig,
    EvaluatorConfig,
    EvaluatorExecutionMode,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationRelationDiscoveryConfig,
    GenerationValidationConfig,
    JITConfig,
    LCFlowLayout,
    ProcessConfig,
    RelationDiscoveryMode,
    RunConfig,
)
from pyamplicol.generation.recurrence_schedule_sharing import (
    RECURRENCE_PROCESS_BINDING_MAGIC,
    RECURRENCE_SCHEDULE_INDEX_PATH,
    RECURRENCE_SCHEDULE_SHARING_KIND,
    RECURRENCE_SCHEDULE_SHARING_SCHEMA_VERSION,
)

_ACCEPTANCE_ENV = "PYAMPLICOL_RUN_RECURRENCE_PROCESS_SET_ACCEPTANCE"
_PROCESS_EXPRESSION = "p p > j j j j"
_PROCESS_NAME = "pp_4j"
_EXPECTED_PROCESS_COUNT = 11

pytestmark = pytest.mark.skipif(
    os.environ.get(_ACCEPTANCE_ENV) != "1",
    reason=(
        "set PYAMPLICOL_RUN_RECURRENCE_PROCESS_SET_ACCEPTANCE=1 and run "
        "under the 30 GiB watchdog"
    ),
)


def _generation_config() -> RunConfig:
    partons = ("d", "d~", "g")
    return RunConfig(
        action=Action.GENERATE,
        process=ProcessConfig(
            multiparticles={"p": partons, "j": partons},
            flavor_scheme=1,
            max_quark_lines=2,
        ),
        color=ColorConfig(
            accuracy=ColorAccuracy.LC,
            lc_flow_layout=LCFlowLayout.TOPOLOGY_REPLAY,
        ),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
            relation_discovery=GenerationRelationDiscoveryConfig(
                mode=RelationDiscoveryMode.OFF,
            ),
            validation=GenerationValidationConfig(
                enabled=False,
                post_build_validation=False,
            ),
        ),
        evaluator=EvaluatorConfig(
            execution_mode=EvaluatorExecutionMode.RECURRENCE,
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(optimization_level=2),
        ),
    )


def _require_native_direct_arena() -> None:
    assert importlib.util.find_spec("symbolica") is not None, (
        "the process-set acceptance requires Symbolica"
    )
    assert importlib.util.find_spec("pyamplicol._rusticol") is not None, (
        "the process-set acceptance requires the native Rusticol extension"
    )
    rusticol = importlib.import_module("pyamplicol._rusticol")
    assert hasattr(rusticol, "_lower_recurrence_direct_v2"), (
        "the installed Rusticol extension lacks recurrence Direct-Arena v2"
    )


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    assert isinstance(value, Mapping), f"{context} must be a mapping"
    return value


def _sequence(value: object, context: str) -> tuple[object, ...]:
    assert isinstance(value, list), f"{context} must be a JSON array"
    return tuple(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _inode(path: Path) -> tuple[int, int]:
    stat = path.stat()
    assert stat.st_nlink == 1, f"{path} must not be hard-linked"
    return stat.st_dev, stat.st_ino


def _assert_process_binding(
    artifact: Path,
    *,
    process_id: str,
    binding: Mapping[str, Any],
) -> Path:
    assert binding["process_id"] == process_id
    assert binding["path"] == "recurrence-binding.bin"
    schedule_digest = str(binding["schedule_digest"])
    binding_path = artifact / "processes" / process_id / str(binding["path"])
    payload = binding_path.read_bytes()

    assert binding_path.stat().st_size == int(binding["size_bytes"])
    assert _sha256(binding_path) == binding["sha256"]
    assert payload[:8] == RECURRENCE_PROCESS_BINDING_MAGIC

    (
        version,
        fixed_size,
        process_id_size,
        support_word_count,
        target_size,
        _cpu_feature_count,
        descriptor_count,
        catalog_executor_count,
    ) = struct.unpack_from("<8I", payload, 8)
    assert version == 4
    assert fixed_size == 344
    assert support_word_count >= 1
    assert descriptor_count >= 1
    assert catalog_executor_count == int(
        _mapping(
            _mapping(binding["remap"], "binding remap")["direct_executors"],
            "binding direct-executor remap",
        )["count"]
    )
    assert payload[88:120] == bytes.fromhex(schedule_digest)
    assert payload[fixed_size : fixed_size + process_id_size].decode("utf-8") == (
        process_id
    )
    assert len(payload) >= (
        fixed_size
        + process_id_size
        + target_size
        + 8 * support_word_count
        + 16 * descriptor_count
    )
    return binding_path


def _representative_process_ids(
    schedules: tuple[Mapping[str, Any], ...],
) -> tuple[str, ...]:
    shared = next(
        schedule
        for schedule in schedules
        if len(_sequence(schedule["process_ids"], "schedule process IDs")) > 1
    )
    shared_ids = tuple(
        str(value)
        for value in _sequence(shared["process_ids"], "shared schedule process IDs")
    )
    representatives = [shared_ids[0], shared_ids[-1]]
    distinct = next(
        (schedule for schedule in schedules if schedule["digest"] != shared["digest"]),
        None,
    )
    if distinct is not None:
        representatives.append(
            str(
                _sequence(
                    distinct["process_ids"],
                    "distinct schedule process IDs",
                )[0]
            )
        )
    return tuple(dict.fromkeys(representatives))


def _flatten(values: object) -> list[complex]:
    if isinstance(values, (tuple, list)):
        result: list[complex] = []
        for value in values:
            result.extend(_flatten(value))
        return result
    return [complex(values)]


def test_bounded_pp_four_jets_shares_root_recurrence_schedules(
    tmp_path: Path,
) -> None:
    """Generate and load a real bounded multiprocess Direct-Arena artifact."""

    _require_native_direct_arena()
    artifact = tmp_path / "recurrence-pp-four-jets"
    processes = ProcessSet.from_expressions(
        (_PROCESS_EXPRESSION,),
        names=(_PROCESS_NAME,),
    )
    with packaged_prepared_model_path(BUILTIN_SM_JIT_O2) as prepared_model:
        Generator(_generation_config()).generate(
            processes,
            artifact,
            model=ModelSource.from_path(prepared_model),
        )

    manifest = load_manifest(artifact)
    process_records = tuple(manifest.processes)
    process_ids = tuple(str(record["id"]) for record in process_records)
    assert manifest.kind == "pyamplicol-process-set"
    assert len(process_ids) == _EXPECTED_PROCESS_COUNT
    assert all(
        set(str(record["expression"]).replace(">", " ").split()) <= {"d", "d~", "g"}
        for record in process_records
    )

    index_path = artifact / RECURRENCE_SCHEDULE_INDEX_PATH
    index = _mapping(
        json.loads(index_path.read_text(encoding="utf-8")),
        "recurrence schedule index",
    )
    assert index["kind"] == RECURRENCE_SCHEDULE_SHARING_KIND
    assert index["schema_version"] == RECURRENCE_SCHEDULE_SHARING_SCHEMA_VERSION
    assert index["runtime_ownership"] == "root-schedule-plus-process-binding"
    assert index["interning_phase"] == "before-direct-lowering"

    schedules = tuple(
        _mapping(value, "recurrence schedule")
        for value in _sequence(index["schedules"], "recurrence schedules")
    )
    bindings = tuple(
        _mapping(value, "recurrence binding")
        for value in _sequence(index["bindings"], "recurrence bindings")
    )
    assert int(index["binding_count"]) == len(bindings) == len(process_ids)
    assert int(index["schedule_count"]) == len(schedules)
    assert len(schedules) < len(bindings)
    assert int(index["schedule_alias_count"]) == len(bindings) - len(schedules)

    schedule_paths: list[Path] = []
    schedule_payload_digests: set[str] = set()
    schedule_inodes: set[tuple[int, int]] = set()
    declared_owners: dict[str, set[str]] = {}
    for schedule in schedules:
        digest = str(schedule["digest"])
        path = artifact / str(schedule["path"])
        expected_path = (
            artifact / "recurrence" / "schedules" / digest / "recurrence-runtime.pacbin"
        )
        assert path == expected_path
        assert path.stat().st_size == int(schedule["size_bytes"])
        assert _sha256(path) == schedule["sha256"]
        assert str(schedule["sha256"]) not in schedule_payload_digests
        schedule_payload_digests.add(str(schedule["sha256"]))
        inode = _inode(path)
        assert inode not in schedule_inodes
        schedule_inodes.add(inode)
        schedule_paths.append(path)
        declared_owners[digest] = {
            str(value)
            for value in _sequence(
                schedule["process_ids"],
                f"schedule {digest} process IDs",
            )
        }

    assert set(artifact.rglob("recurrence-runtime.pacbin")) == set(schedule_paths)

    bindings_by_process = {str(binding["process_id"]): binding for binding in bindings}
    assert set(bindings_by_process) == set(process_ids)
    binding_paths: list[Path] = []
    binding_inodes: set[tuple[int, int]] = set()
    binding_payload_digests: set[str] = set()
    bound_owners: dict[str, set[str]] = {}
    for process_id in process_ids:
        binding = bindings_by_process[process_id]
        binding_path = _assert_process_binding(
            artifact,
            process_id=process_id,
            binding=binding,
        )
        inode = _inode(binding_path)
        assert inode not in binding_inodes
        binding_inodes.add(inode)
        assert str(binding["sha256"]) not in binding_payload_digests
        binding_payload_digests.add(str(binding["sha256"]))
        binding_paths.append(binding_path)
        bound_owners.setdefault(str(binding["schedule_digest"]), set()).add(process_id)

        execution = _mapping(
            json.loads(
                (artifact / "processes" / process_id / "execution.json").read_text(
                    encoding="utf-8"
                )
            ),
            f"process {process_id} execution",
        )
        plan = _mapping(execution["plan"], f"process {process_id} plan")
        runtime_schedule = _mapping(
            plan["runtime_schedule"],
            f"process {process_id} runtime schedule",
        )
        process_binding = _mapping(
            plan["process_binding"],
            f"process {process_id} binding",
        )
        assert runtime_schedule["path"] == next(
            schedule["path"]
            for schedule in schedules
            if schedule["digest"] == binding["schedule_digest"]
        )
        assert process_binding == binding

    assert set(artifact.rglob("recurrence-binding.bin")) == set(binding_paths)
    assert bound_owners == declared_owners
    assert schedule_inodes.isdisjoint(binding_inodes)

    for process_id in _representative_process_ids(schedules):
        runtime = Runtime.load(artifact, process=process_id)
        assert runtime.physics.process_id == process_id
        assert runtime.physics.color_accuracy == "lc"

    expressions = {
        str(record["id"]): str(record["expression"]) for record in process_records
    }
    shared_groups = tuple(
        tuple(
            str(process_id)
            for process_id in _sequence(
                schedule["process_ids"],
                "shared schedule process IDs",
            )
        )
        for schedule in schedules
        if len(_sequence(schedule["process_ids"], "schedule process IDs")) > 1
    )
    assert shared_groups
    for group_index, shared_ids in enumerate(shared_groups):
        process_id = shared_ids[-1]
        shared_runtime = Runtime.load(artifact, process=process_id)
        reference_artifact = tmp_path / f"reference-{group_index}"
        with packaged_prepared_model_path(BUILTIN_SM_JIT_O2) as prepared_model:
            Generator(_generation_config()).generate(
                ProcessSet.from_expressions(
                    (expressions[process_id],),
                    names=(f"reference_{group_index}",),
                ),
                reference_artifact,
                model=ModelSource.from_path(prepared_model),
            )
        reference_runtime = Runtime.load(reference_artifact)
        assert shared_runtime.physics.helicity_ids == (
            reference_runtime.physics.helicity_ids
        )
        assert shared_runtime.physics.color_ids == reference_runtime.physics.color_ids
        momenta = shared_runtime._backend.validation_momenta()
        assert momenta is not None
        selected_flow = shared_runtime.physics.color_ids[0]
        shared_resolved = shared_runtime.evaluate_resolved(
            momenta,
            color_flows=(selected_flow,),
        )
        reference_resolved = reference_runtime.evaluate_resolved(
            momenta,
            color_flows=(selected_flow,),
        )
        assert shared_resolved.helicity_ids == reference_resolved.helicity_ids
        assert shared_resolved.color_ids == reference_resolved.color_ids
        assert _flatten(shared_resolved.values) == pytest.approx(
            _flatten(reference_resolved.values),
            rel=1.0e-12,
            abs=1.0e-15,
        )

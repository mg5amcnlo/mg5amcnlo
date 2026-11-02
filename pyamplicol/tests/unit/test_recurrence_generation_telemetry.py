# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path

import pytest

from pyamplicol.generation import artifact_writer
from pyamplicol.generation import service as generation_service
from pyamplicol.generation.recurrence_schedule_sharing import (
    RecurrenceProcessExecutorPack,
)


def _native_profile() -> dict[str, object]:
    return {
        "schema_version": 1,
        "scope": "generation-only",
        "timings_seconds": {
            name: 0.125
            for name in generation_service._RECURRENCE_GENERATION_PROFILE_TIMINGS
        },
        "operation_counters": {
            name: 7
            for name in generation_service._RECURRENCE_GENERATION_PROFILE_COUNTERS
        },
        "serialized_bytes": {
            "plan_payload": 13,
            "container": 37,
            "unpacked_container": 19,
        },
    }


def _inspection() -> dict[str, object]:
    return {
        "runtime_container_member": {
            "size_bytes": 13,
            "container_size_bytes": 37,
        }
    }


def _recurrence_artifact(
    process_id: str,
    profile: dict[str, object],
) -> artifact_writer.RecurrenceProcessArtifact:
    return artifact_writer.RecurrenceProcessArtifact(
        process_id=process_id,
        expression="d d~ > Z g",
        color_accuracy="lc",
        external_pdgs=(1, -1, 23, 21),
        aliases=(),
        physics={},
        recurrence_schedule_path=Path("recurrence-runtime.pacbin"),
        recurrence_schedule_digest="a" * 64,
        recurrence_native_schedule_semantic_digest="b" * 64,
        recurrence_schedule_size_bytes=37,
        recurrence_schedule_sha256="c" * 64,
        recurrence_schedule_member_count=1,
        recurrence_schedule_unpacked_size_bytes=19,
        recurrence_schedule_index_sha256="d" * 64,
        builder_input_sha256="e" * 64,
        prepared_kernel_pack_digest="f" * 64,
        direct_template_catalog_digest="1" * 64,
        referenced_kernel_ids=frozenset(),
        inspection_summary={},
        runtime_metadata={},
        color_contraction_payload=None,
        color_contraction_summary=None,
        point_tile_size=1,
        workspace_mib=1,
        recurrence_summary={},
        validation_point=None,  # type: ignore[arg-type]
        generation_filters={},
        generation_profile=profile,
        recurrence_process_remap=None,  # type: ignore[arg-type]
        recurrence_process_executor_pack=RecurrenceProcessExecutorPack(
            compiled_model_digest="2" * 64,
            recurrence_template_catalog_digest="3" * 64,
            prepared_kernel_pack_digest="f" * 64,
            direct_template_catalog_digest="1" * 64,
            runtime_layout_digest="4" * 64,
            backend="jit",
            target_triple="symjit-storage-v3-portable",
            portable=True,
            cpu_features=(),
            catalog_executor_count=1,
            executor_ids=(0,),
            descriptor_payloads=(bytes.fromhex("10000000000000000000000001000000"),),
        ),
        process_digest="5" * 64,
    )


def test_generation_profile_is_strict_and_generation_only() -> None:
    profile = _native_profile()

    normalized = generation_service._validate_recurrence_generation_profile(
        profile,
        inspection=_inspection(),
        unpacked_size_bytes=19,
    )

    assert normalized == profile
    assert (
        normalized["operation_counters"]["constructed_interaction_count"]  # type: ignore[index]
        == 7
    )
    assert (
        normalized["operation_counters"]["emitted_interaction_count"]  # type: ignore[index]
        == 7
    )
    assert (
        normalized["operation_counters"]["indexed_hash_lookup_count"]  # type: ignore[index]
        == 7
    )


def test_generation_profile_is_keyed_by_shared_schedule_digest() -> None:
    wrapped = {
        "schema_version": 1,
        "native_passes": {"final": _native_profile()},
    }
    processes = (
        _recurrence_artifact("first", wrapped),
        _recurrence_artifact("second", copy.deepcopy(wrapped)),
    )

    extensions = artifact_writer._extensions(
        None,
        processes=processes,
        timings={},
        api_bundle_requested=False,
        api_bundle_path=None,
        eager_pack_identity=None,
        execution_manifest_sha256_by_process={
            "first": "2" * 64,
            "second": "3" * 64,
        },
        evaluator_payload_container=None,
        recurrence_schedule_sharing=None,
    )

    assert extensions["generation"]["recurrence_schedule_profiles"] == {  # type: ignore[index]
        "a" * 64: wrapped
    }


def test_generation_profile_rejects_inconsistent_shared_schedule_claims() -> None:
    left = {
        "schema_version": 1,
        "native_passes": {"final": _native_profile()},
    }
    right = copy.deepcopy(left)
    right["native_passes"]["final"]["operation_counters"][  # type: ignore[index]
        "emitted_current_count"
    ] = 8

    with pytest.raises(ValueError, match="inconsistent generation telemetry"):
        artifact_writer._extensions(
            None,
            processes=(
                _recurrence_artifact("first", left),
                _recurrence_artifact("second", right),
            ),
            timings={},
            api_bundle_requested=False,
            api_bundle_path=None,
            eager_pack_identity=None,
            execution_manifest_sha256_by_process={
                "first": "2" * 64,
                "second": "3" * 64,
            },
            evaluator_payload_container=None,
            recurrence_schedule_sharing=None,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda profile: profile["timings_seconds"].__setitem__(  # type: ignore[union-attr]
                "candidate-processing",
                float("nan"),
            ),
            "timing",
        ),
        (
            lambda profile: profile["operation_counters"].__setitem__(  # type: ignore[union-attr]
                "unknown",
                1,
            ),
            "fields do not match",
        ),
        (
            lambda profile: profile["serialized_bytes"].__setitem__(  # type: ignore[union-attr]
                "container",
                38,
            ),
            "disagrees",
        ),
    ],
)
def test_generation_profile_rejects_invalid_or_unlinked_claims(
    mutate: Callable[[dict[str, object]], None],
    message: str,
) -> None:
    profile = copy.deepcopy(_native_profile())
    mutate(profile)

    with pytest.raises(generation_service.GenerationError, match=message):
        generation_service._validate_recurrence_generation_profile(
            profile,
            inspection=_inspection(),
            unpacked_size_bytes=19,
        )

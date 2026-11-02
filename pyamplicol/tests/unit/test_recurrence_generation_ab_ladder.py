# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.developer import recurrence_generation_ab_ladder as ladder


def test_process_family_and_approved_timeout_policy() -> None:
    assert ladder.process_expression(1) == "d d~ > Z"
    assert ladder.process_expression(4) == "d d~ > Z g g g"
    assert ladder.generation_timeout_seconds(2, "topology-replay") == 300.0
    assert ladder.generation_timeout_seconds(7, "all-flow-union") == 900.0
    assert ladder.generation_timeout_seconds(8, "topology-replay") == 3600.0
    assert ladder.generation_timeout_seconds(8, "all-flow-union") == 7200.0
    assert ladder.generation_timeout_seconds(9, "topology-replay") == 7200.0
    assert ladder.generation_timeout_seconds(9, "all-flow-union") == 21600.0
    with pytest.raises(ladder.LadderError):
        ladder.generation_timeout_seconds(10, "topology-replay")


def test_schedule_alternates_pair_order_and_selects_runtime_cells() -> None:
    schedule = ladder.build_schedule(
        (2, 6),
        ("topology-replay",),
        2,
        frozenset({6}),
    )

    assert [sample.variant_name for sample in schedule] == [
        "baseline",
        "candidate",
        "candidate",
        "baseline",
        "baseline",
        "candidate",
        "candidate",
        "baseline",
    ]
    assert [sample.pair_index for sample in schedule] == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
    ]
    assert all(not sample.runtime_enabled for sample in schedule[:4])
    assert all(sample.runtime_enabled for sample in schedule[4:])


def _spec(*, multiplicity: int, runtime_enabled: bool) -> ladder.SampleSpec:
    return ladder.SampleSpec(
        sequence_index=0,
        pair_index=0,
        order_in_pair=0,
        repetition=0,
        multiplicity=multiplicity,
        layout="all-flow-union",
        variant_name="baseline",
        runtime_enabled=runtime_enabled,
    )


def test_command_wraps_each_capture_and_runtime_workers_in_watchdog(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    variant = ladder.Variant(
        name="baseline",
        checkout=checkout,
        python=tmp_path / "python",
        pythonpath=tmp_path / "site-packages",
        prepared_model=tmp_path / "model.pack",
    )
    settings = ladder.RunnerSettings()
    sample_root = tmp_path / "sample"

    generation_command = ladder.build_sample_command(
        _spec(multiplicity=2, runtime_enabled=False),
        variant,
        sample_root,
        settings,
    )
    assert generation_command[:6] == [
        str(variant.python),
        str(ladder.ROOT / ladder.WATCHDOG_RELATIVE_PATH),
        "--limit-gib",
        "30",
        "--",
        str(variant.python),
    ]
    assert str(ladder.ROOT / ladder.HARNESS_RELATIVE_PATH) in generation_command
    assert generation_command.count("--mode") == 1
    assert "recurrence" in generation_command
    assert "--generation-only" in generation_command
    assert "--allow-diagnostic-incomplete-success" not in generation_command
    assert generation_command[generation_command.index("--process-expression") + 1] == (
        "d d~ > Z g"
    )
    validation_index = generation_command.index("--validation-samples")
    assert generation_command[validation_index : validation_index + 4] == [
        "--validation-samples",
        "10",
        "--point-tile-size",
        "1024",
    ]

    runtime_command = ladder.build_sample_command(
        _spec(multiplicity=6, runtime_enabled=True),
        variant,
        sample_root,
        settings,
    )
    assert "--generation-only" not in runtime_command
    assert [
        runtime_command[index + 1]
        for index, value in enumerate(runtime_command)
        if value == "--batch-size"
    ] == ["1", "128", "1024"]
    assert runtime_command[runtime_command.index("--subprocess-samples") + 1] == "7"
    assert runtime_command[runtime_command.index("--warmup-runs") + 1] == "2"
    assert runtime_command[runtime_command.index("--target-runtime") + 1] == "5"

    scouting_command = ladder.build_sample_command(
        _spec(multiplicity=9, runtime_enabled=False),
        variant,
        sample_root,
        ladder.RunnerSettings(allow_diagnostic_incomplete_success=True),
    )
    assert "--allow-diagnostic-incomplete-success" in scouting_command

    _, overrides = ladder._sample_environment(
        variant,
        tmp_path / "sample-environment",
    )
    assert overrides[ladder.HARNESS_SOURCE_CHECKOUT_ENV] == str(checkout)


def test_incomplete_diagnostic_capture_is_never_classified_as_passed() -> None:
    outcome = ladder.ProcessOutcome(
        exit_code=0,
        timed_out=False,
        wall_seconds=1.0,
        error=None,
    )
    watchdog = {"terminal_record": "command-finished"}
    incomplete = {"complete": None, "passes": None}

    assert (
        ladder._sample_status(
            outcome,
            watchdog=watchdog,
            result_error=None,
            harness_summary=incomplete,
            allow_diagnostic_incomplete_success=False,
        )
        == "failed-validation"
    )
    assert (
        ladder._sample_status(
            outcome,
            watchdog=watchdog,
            result_error=None,
            harness_summary=incomplete,
            allow_diagnostic_incomplete_success=True,
        )
        == "censored"
    )
    assert (
        ladder._sample_status(
            outcome,
            watchdog=watchdog,
            result_error=None,
            harness_summary={"complete": True, "passes": True},
            allow_diagnostic_incomplete_success=False,
        )
        == "passed"
    )


def test_generation_timeout_is_censored_only_in_explicit_scouting_mode(
    tmp_path: Path,
) -> None:
    stderr = tmp_path / "stderr.log"
    stderr.write_text(
        "recurrence-z6g-benchmark: recurrence generation worker exceeded "
        "21600 seconds\n"
        "memory-watchdog: command finished exit=2 peak_rss=1.250 GiB "
        "peak_physical_footprint=1.500 GiB peak_guard=1.500 GiB "
        "peak_processes=4\n",
        encoding="utf-8",
    )
    timeout = ladder.parse_harness_generation_timeout(stderr)
    assert timeout == {"configured_seconds": 21600.0}
    outcome = ladder.ProcessOutcome(
        exit_code=2,
        timed_out=False,
        wall_seconds=21600.0,
        error=None,
    )
    watchdog = {"terminal_record": "command-finished"}

    assert (
        ladder._sample_status(
            outcome,
            watchdog=watchdog,
            result_error=None,
            harness_summary=None,
            harness_generation_timeout=timeout,
            allow_diagnostic_incomplete_success=False,
        )
        == "timeout"
    )
    assert (
        ladder._sample_status(
            outcome,
            watchdog=watchdog,
            result_error=None,
            harness_summary=None,
            harness_generation_timeout=timeout,
            allow_diagnostic_incomplete_success=True,
        )
        == "censored"
    )


def test_scouting_does_not_censor_arbitrary_or_memory_limit_failures(
    tmp_path: Path,
) -> None:
    stderr = tmp_path / "stderr.log"
    stderr.write_text(
        "recurrence-z6g-benchmark: recurrence generation failed\n",
        encoding="utf-8",
    )
    assert ladder.parse_harness_generation_timeout(stderr) is None
    outcome = ladder.ProcessOutcome(
        exit_code=2,
        timed_out=False,
        wall_seconds=1.0,
        error=None,
    )
    assert (
        ladder._sample_status(
            outcome,
            watchdog={"terminal_record": "memory-limit-exceeded"},
            result_error=None,
            harness_summary=None,
            allow_diagnostic_incomplete_success=True,
        )
        == "failed"
    )


def test_watchdog_terminal_records_are_parsed() -> None:
    finished = ladder.parse_watchdog_text(
        "memory-watchdog: command finished exit=0"
        " peak_rss=1.250 GiB"
        " peak_physical_footprint=1.500 GiB"
        " peak_guard=1.500 GiB peak_processes=4\n"
    )
    assert finished["terminal_record"] == "command-finished"
    assert finished["child_exit_code"] == 0
    assert finished["peak_rss"]["bytes_rounded_from_watchdog"] == round(1.25 * 1024**3)
    assert finished["peak_guard"]["gib"] == 1.5

    exceeded = ladder.parse_watchdog_text(
        "memory-watchdog: memory limit exceeded"
        " reason=process-tree-rss-limit"
        " observed=30.125 GiB limit=30.000 GiB"
        " rss=30.125 GiB physical_footprint=unavailable"
        " processes=9; terminating tree\n"
    )
    assert exceeded["terminal_record"] == "memory-limit-exceeded"
    assert exceeded["limit_exceeded"] is True
    assert exceeded["reason"] == "process-tree-rss-limit"
    assert exceeded["process_count_at_limit"] == 9


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_harness_result_parser_extracts_phase_and_runtime_telemetry(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    execution = artifact / "processes" / "process" / "execution.json"
    _write_json(
        execution,
        {
            "kind": ladder.EXECUTION_KIND,
            "schema_version": ladder.EXECUTION_SCHEMA_VERSION,
            "plan": {
                "inspection_summary": {
                    "generation_timings_seconds": {
                        "python_extraction": 1.0,
                        "direct_lowering": 2.0,
                        "native_total": 3.5,
                    },
                    "schedule": {"current_count": 11},
                    "schedule_digest": "a" * 64,
                }
            },
        },
    )
    result_path = tmp_path / "result.json"
    _write_json(
        result_path,
        {
            "kind": ladder.HARNESS_KIND,
            "schema_version": ladder.HARNESS_SCHEMA_VERSION,
            "complete": None,
            "passes": None,
            "process": "d d~ > Z g g g g g",
            "configuration": {"lc_flow_layout": "all-flow-union"},
            "source": {"git_revision": "1" * 40},
            "runtime_provenance": {"extension_sha256": "2" * 64},
            "provenance": {"wall_seconds": 12.0},
            "generation": {
                "recurrence": {
                    "artifact": str(artifact),
                    "artifact_identity": {"artifact_id": "3" * 64},
                    "artifact_semantic_identity_sha256": "4" * 64,
                    "generation_wall_seconds": 9.0,
                    "worker_process_record": {"wall_seconds": 8.0},
                    "peak_rss": {"observed_lower_bound_bytes": 1234},
                    "phase_timings_seconds": {
                        "model-loading": 1.5,
                        "recurrence-construction": 6.0,
                    },
                    "phase_total_seconds": 7.5,
                }
            },
            "profiles": {
                "recurrence": {
                    "process_id": "process",
                    "process_expression": "d d~ > Z g g g g g",
                    "profiles": [
                        {
                            "batch_size": 128,
                            "sample_count": 1,
                            "wall_seconds_per_point": 1.0e-6,
                            "wall_seconds_per_point_median": 1.0e-6,
                            "wall_seconds_per_point_mad": 1.0e-8,
                            "interrupted": False,
                            "subprocess_samples": [
                                {
                                    "schedule_index": 2,
                                    "round": 0,
                                    "wall_seconds_per_point": 1.0e-6,
                                    "internal_sample_count": 7,
                                    "repetitions_per_sample": 4,
                                    "evaluation_count": 28,
                                    "evaluated_point_count": 3584,
                                    "interrupted": False,
                                    "worker_process_record": {"wall_seconds": 5.0},
                                }
                            ],
                        }
                    ],
                }
            },
        },
    )

    summary = ladder.parse_harness_result(result_path)

    assert summary["generation_wall_seconds"] == 9.0
    assert summary["generation_peak_rss"]["observed_lower_bound_bytes"] == 1234
    assert summary["native_generation_timings_seconds"] == {
        "python_extraction": 1.0,
        "direct_lowering": 2.0,
        "native_total": 3.5,
    }
    assert summary["native_inspection_summary"]["schedule"]["current_count"] == 11
    measurement = summary["runtime_profile"]["measurements"][0]
    assert measurement["batch_size"] == 128
    assert measurement["subprocess_samples"][0]["worker_wall_seconds"] == 5.0


def test_generation_only_result_has_an_independent_fail_closed_acceptance(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    execution = artifact / "processes" / "process" / "execution.json"
    _write_json(
        execution,
        {
            "kind": ladder.EXECUTION_KIND,
            "schema_version": ladder.EXECUTION_SCHEMA_VERSION,
            "plan": {
                "inspection_summary": {
                    "generation_timings_seconds": {
                        "python_extraction": 1.0,
                        "direct_lowering": 2.0,
                        "native_total": 3.5,
                    }
                }
            },
        },
    )
    relative_execution = execution.relative_to(artifact).as_posix()
    payload_entry = {
        "path": relative_execution,
        "role": "evaluator-manifest",
        "process_id": "process",
        "sha256": ladder._sha256_file(execution),
        "size_bytes": execution.stat().st_size,
    }
    artifact_id = "3" * 64
    manifest_path = artifact / "artifact.json"
    _write_json(
        manifest_path,
        {
            "artifact_id": artifact_id,
            "payloads": [payload_entry],
        },
    )
    signature = {
        "kind": "pyamplicol-benchmark-generation-signature",
        "schema_version": 1,
        "mode": "recurrence",
        "lc_flow_layout": "all-flow-union",
        "validation": {
            "enabled": True,
            "post_build_validation": True,
            "samples": 10,
        },
    }
    semantic_identity = {
        "coverage": {
            "complete_physical_axes": True,
            "color": "complete",
            "helicities": "complete",
        },
        "execution_reduction_coverage": {
            "complete": True,
            "errors": [],
        },
    }
    evidence = {
        "kind": "pyamplicol-preserved-worker-result-evidence",
        "schema_version": 1,
        "payload": {
            "mode": "recurrence",
            "generation_reused": False,
            "generation_wall_seconds": 9.0,
        },
        "worker_result_record": {},
    }
    evidence["content_sha256"] = ladder._canonical_sha256(evidence)
    result_path = tmp_path / "generation-only-result.json"
    _write_json(
        result_path,
        {
            "kind": ladder.HARNESS_KIND,
            "schema_version": ladder.HARNESS_SCHEMA_VERSION,
            "complete": False,
            "passes": None,
            "process": "d d~ > Z g",
            "configuration": {
                "generation_only": True,
                "lc_flow_layout": "all-flow-union",
                "validation_samples": 10,
            },
            "source": {"git_revision": "1" * 40},
            "runtime_provenance": {"extension_sha256": "2" * 64},
            "provenance": {"wall_seconds": 12.0},
            "generation": {
                "recurrence": {
                    "mode": "recurrence",
                    "artifact": str(artifact),
                    "artifact_identity": {
                        "artifact_id": artifact_id,
                        "path": str(artifact.resolve()),
                        "manifest": {
                            "resolved_path": str(manifest_path.resolve()),
                            "sha256": ladder._sha256_file(manifest_path),
                            "size_bytes": manifest_path.stat().st_size,
                        },
                        "payloads": [payload_entry],
                    },
                    "artifact_semantic_identity": semantic_identity,
                    "artifact_semantic_identity_sha256": (
                        ladder._canonical_sha256(semantic_identity)
                    ),
                    "semantic_generation_signature": signature,
                    "semantic_generation_signature_sha256": (
                        ladder._canonical_sha256(signature)
                    ),
                    "generation_worker_result_evidence": evidence,
                    "generation_reused": False,
                    "generation_wall_seconds": 9.0,
                    "worker_process_record": {"wall_seconds": 8.0},
                    "peak_rss": {"observed_lower_bound_bytes": 1234},
                    "phase_timings_seconds": {
                        "model-loading": 1.5,
                        "recurrence-construction": 6.0,
                    },
                    "phase_total_seconds": 7.5,
                }
            },
            "profiles": {},
        },
    )

    summary = ladder.parse_harness_result(result_path)
    acceptance = summary["generation_only_acceptance"]
    assert acceptance["passes"] is True
    assert acceptance["payload_count"] == 1
    assert acceptance["post_build_validation_samples"] == 10
    assert (
        ladder._sample_status(
            ladder.ProcessOutcome(
                exit_code=0,
                timed_out=False,
                wall_seconds=12.0,
                error=None,
            ),
            watchdog={"terminal_record": "command-finished"},
            result_error=None,
            harness_summary=summary,
            allow_diagnostic_incomplete_success=True,
        )
        == "passed"
    )

    execution.write_text(execution.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ladder.LadderError, match="payload identity differs"):
        ladder.parse_harness_result(result_path)


def _write_runtime_acceptance_result(tmp_path: Path) -> Path:
    artifact = tmp_path / "runtime-artifact"
    execution = artifact / "processes" / "process" / "execution.json"
    _write_json(
        execution,
        {
            "kind": ladder.EXECUTION_KIND,
            "schema_version": ladder.EXECUTION_SCHEMA_VERSION,
            "plan": {
                "inspection_summary": {
                    "generation_timings_seconds": {"native_total": 3.5}
                }
            },
        },
    )
    relative_execution = execution.relative_to(artifact).as_posix()
    payload_entry = {
        "path": relative_execution,
        "role": "evaluator-manifest",
        "process_id": "process",
        "sha256": ladder._sha256_file(execution),
        "size_bytes": execution.stat().st_size,
    }
    artifact_id = "3" * 64
    manifest_path = artifact / "artifact.json"
    _write_json(
        manifest_path,
        {
            "artifact_id": artifact_id,
            "payloads": [payload_entry],
        },
    )
    semantic_identity = {
        "coverage": {"complete_physical_axes": True},
        "generation_specialized_axes": [],
    }
    semantic_sha256 = ladder._canonical_sha256(semantic_identity)
    artifact_identity = {
        "artifact_id": artifact_id,
        "path": str(artifact.resolve()),
        "manifest": {
            "resolved_path": str(manifest_path.resolve()),
            "sha256": ladder._sha256_file(manifest_path),
            "size_bytes": manifest_path.stat().st_size,
        },
        "payloads": [payload_entry],
        "semantic_identity": semantic_identity,
        "semantic_identity_sha256": semantic_sha256,
    }
    process = "d d~ > Z g g g g g"
    configuration = {
        "batch_sizes": [1, 128, 1024],
        "target_runtime_seconds": 5.0,
        "minimum_samples": 7,
        "subprocess_samples": 7,
        "warmup_runs": 2,
        "generation_timeout_seconds": 900.0,
        "profile_timeout_seconds": 300.0,
        "color_flow_request": "1",
        "helicity_request": "1",
        "lc_flow_layout": "all-flow-union",
        "gluon_count": 5,
        "validation_samples": 10,
        "point_tile_size": 1024,
        "jit_optimization_level": 2,
        "validation_point_artifact": str(artifact),
        "generation_only": False,
        "allow_diagnostic_incomplete_success": True,
        "modes": ["recurrence"],
        "prepared_model_path": "/workspace/model.pack",
        "model_identities": {},
        "validation_seed": 12345,
        "specialize_flow_at_generation": False,
        "external_watchdog_required_for_long_runs": True,
    }
    selector = {
        "color_flow_request": "1",
        "resolved_color_flow_id": "flow-1",
        "helicity_request": "1",
        "resolved_helicity_id": "helicity-1",
        "color_flow_count": 8,
        "helicity_count": 32,
        "structural_zero_helicity_count": 0,
        "workload": "all-flows/runtime-selected-single-helicity",
    }
    fixture = {
        "point_count": 10,
        "points_sha256": "4" * 64,
        "file_sha256": "5" * 64,
    }
    validation = {
        "passes": True,
        "fixture": {
            "point_count": fixture["point_count"],
            "points_sha256": fixture["points_sha256"],
            "file": {"sha256": fixture["file_sha256"]},
        },
    }
    recurrence_profile = {
        "mode": "recurrence",
        "process_id": "process",
        "process_expression": process.casefold(),
        "selector_contract": selector,
        "validation": validation,
        "artifact_semantic_identity": semantic_identity,
        "artifact_semantic_identity_sha256": semantic_sha256,
        "profiles": [
            {
                "batch_size": batch_size,
                "sample_count": 7,
                "wall_seconds_per_point": 1.0e-6,
                "wall_seconds_per_point_median": 1.0e-6,
                "wall_seconds_per_point_mad": 1.0e-8,
                "interrupted": False,
                "subprocess_samples": [],
            }
            for batch_size in ladder.DEFAULT_BATCH_SIZES
        ],
    }
    profile_schedule = {
        "kind": "pyamplicol-interleaved-subprocess-profile-schedule",
        "schema_version": 2,
        "entries": [],
    }
    source = {"git_revision": "1" * 40}
    runtime_provenance = {"extension_sha256": "2" * 64}
    observed_configuration = {
        "modes": ["recurrence"],
        "generation_only": False,
        "batch_sizes": [1, 128, 1024],
        "target_runtime_seconds": 5.0,
        "warmup_runs": 2,
        "minimum_internal_samples": 7,
        "subprocess_samples": 7,
        "gluon_count": 5,
        "process_expression_override": process,
        "lc_flow_layout": "all-flow-union",
        "color_flow_request": "1",
        "helicity_request": "1",
        "validation_samples": 10,
        "point_tile_size": 1024,
        "jit_optimization_level": 2,
        "specialize_flow_at_generation": False,
        "prepared_model_path": "/workspace/model.pack",
    }
    bindings = {
        "source_identity_sha256": ladder._canonical_sha256(source),
        "runtime_provenance_sha256": ladder._canonical_sha256(runtime_provenance),
        "generation_artifact_id": artifact_id,
        "generation_artifact_semantic_identity_sha256": semantic_sha256,
        "profile_schedule_sha256": ladder._canonical_sha256(profile_schedule),
        "recurrence_profile_sha256": ladder._canonical_sha256(recurrence_profile),
        "selector_contract": selector,
        "selector_contract_sha256": ladder._canonical_sha256(selector),
        "validation_fixture": fixture,
        "validation_fixture_sha256": ladder._canonical_sha256(fixture),
        "process_id": "process",
        "process_expression": process.casefold(),
        "lc_flow_layout": "all-flow-union",
        "configuration": observed_configuration,
        "configuration_sha256": ladder._canonical_sha256(observed_configuration),
    }
    acceptance = {
        "kind": ladder.RUNTIME_ACCEPTANCE_KIND,
        "schema_version": ladder.RUNTIME_ACCEPTANCE_SCHEMA_VERSION,
        "accepted": True,
        "status": "accepted",
        "errors": [],
        "required": {
            "modes": ["recurrence"],
            "generation_only": False,
            "batch_sizes": [1, 128, 1024],
            "target_runtime_seconds": 5.0,
            "warmup_runs": 2,
            "minimum_internal_samples": 7,
            "subprocess_samples_minimum": 7,
            "subprocess_samples_maximum": 21,
            "process_family": "d d~ > Z + (n-1)*g",
            "accepted_n_values": [6, 7],
            "generation_specialized_axes_allowed": False,
            "complete_physical_axes_required": True,
        },
        "contracts": {
            "measurement": {"passes": True},
            "artifact_semantics": {"passes": True},
            "lane_validation": {
                "passes": True,
                "lane_validation_passes": True,
                "selectors_match": True,
                "fixtures_match": True,
                "pairwise_validation_passes": None,
                "summary_sha256": "6" * 64,
            },
            "worker_rss": {
                "passes": True,
                "errors": [],
                "worker_count": 21,
                "maximum_cold_load_observed_lower_bound_bytes": 1024,
                "maximum_post_profile_observed_lower_bound_bytes": 2048,
            },
        },
        "bindings": bindings,
    }
    acceptance["content_sha256"] = ladder._canonical_sha256(acceptance)
    result = {
        "kind": ladder.HARNESS_KIND,
        "schema_version": ladder.HARNESS_SCHEMA_VERSION,
        "complete": False,
        "passes": None,
        "process": process,
        "configuration": configuration,
        "source": source,
        "runtime_provenance": runtime_provenance,
        "provenance": {"wall_seconds": 12.0},
        "generation": {
            "recurrence": {
                "mode": "recurrence",
                "artifact": str(artifact),
                "artifact_identity": artifact_identity,
                "artifact_semantic_identity_sha256": semantic_sha256,
                "generation_reused": False,
                "generation_wall_seconds": 9.0,
                "worker_process_record": {"wall_seconds": 8.0},
                "peak_rss": {"observed_lower_bound_bytes": 1234},
                "phase_timings_seconds": {"recurrence-construction": 6.0},
                "phase_total_seconds": 6.0,
            }
        },
        "profile_schedule": profile_schedule,
        "profiles": {"recurrence": recurrence_profile},
        "recurrence_runtime_acceptance": acceptance,
    }
    result_path = tmp_path / "runtime-result.json"
    _write_json(result_path, result)
    return result_path


def test_runtime_result_has_an_independent_fail_closed_acceptance(
    tmp_path: Path,
) -> None:
    result_path = _write_runtime_acceptance_result(tmp_path)
    summary = ladder.parse_harness_result(result_path)
    acceptance = summary["recurrence_runtime_acceptance"]
    assert acceptance["passes"] is True
    assert acceptance["worker_count"] == 21
    assert acceptance["maximum_cold_load_observed_lower_bound_bytes"] == 1024
    assert acceptance["maximum_post_profile_observed_lower_bound_bytes"] == 2048
    assert (
        ladder._sample_status(
            ladder.ProcessOutcome(
                exit_code=0,
                timed_out=False,
                wall_seconds=12.0,
                error=None,
            ),
            watchdog={"terminal_record": "command-finished"},
            result_error=None,
            harness_summary=summary,
            allow_diagnostic_incomplete_success=True,
        )
        == "passed"
    )


def test_runtime_acceptance_rejects_tampered_digest_and_configuration(
    tmp_path: Path,
) -> None:
    result_path = _write_runtime_acceptance_result(tmp_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["recurrence_runtime_acceptance"]["content_sha256"] = "0" * 64
    _write_json(result_path, result)
    with pytest.raises(ladder.LadderError, match="address is invalid"):
        ladder.parse_harness_result(result_path)

    result_path = _write_runtime_acceptance_result(tmp_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["configuration"]["lc_flow_layout"] = "topology-replay"
    _write_json(result_path, result)
    with pytest.raises(ladder.LadderError, match="bindings failed authentication"):
        ladder.parse_harness_result(result_path)


def test_runtime_n_must_be_selected_in_generation_ladder() -> None:
    with pytest.raises(ladder.LadderError, match="runtime-n"):
        ladder.build_schedule(
            (2,),
            ("topology-replay",),
            1,
            frozenset({6}),
        )

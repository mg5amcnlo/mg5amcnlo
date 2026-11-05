# SPDX-License-Identifier: 0BSD

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import statistics
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
M0_SCRIPT = ROOT / "tools" / "developer" / "eager_compiled_arena_m0.py"
Z6G_TEST = ROOT / "tests" / "unit" / "test_recurrence_z6g_benchmark.py"


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


m0 = _load("_test_eager_compiled_arena_m0", M0_SCRIPT)
z6g = _load("_test_eager_compiled_arena_m0_z6g", Z6G_TEST)
benchmark = z6g.benchmark

SOURCE_REVISION = "a" * 40
AMPLICOL_REVISION = "b" * 40
POINTS_SHA256 = "1" * 64
FLOW_ID = "flow:2,4,5,6,7,8,9,1"
FLOW_WORD = [2, 4, 5, 6, 7, 8, 9, 1]
HELICITY_ID = "h:-1,+1,-1,+1,-1,+1,-1,+1,-1"
HELICITY_VALUES = [-1, 1, -1, 1, -1, 1, -1, 1, -1]
OPPOSITE_HELICITY_ID = "h:+1,-1,+1,-1,+1,-1,+1,-1,+1"
SOURCE = {
    "checkout": "/checkout",
    "revision": SOURCE_REVISION,
    "dirty": False,
    "untracked_files_checked": True,
}
HOST = {
    "platform": "Darwin-test",
    "system": "Darwin",
    "release": "24.0.0",
    "version": "test",
    "machine": "arm64",
    "processor": "arm",
    "cpu_model": "test-cpu",
    "logical_cpu_count": 8,
}
RUNTIME = {
    "interpreter": {
        "path": "/opt/test/python",
        "resolved_path": "/opt/test/python",
        "size_bytes": 10,
        "sha256": "2" * 64,
        "python_version": "3.12.6",
        "implementation": "CPython",
    },
    "installed_distribution": {
        "package_version": "0.1.0",
        "distribution_content": {
            "algorithm": "sha256-relative-path-size-content-v1",
            "sha256": "3" * 64,
            "file_count": 10,
            "size_bytes": 100,
        },
        "native_modules": [
            {
                "relative_path": "pyamplicol/_rusticol.so",
                "path": "/opt/test/_rusticol.so",
                "resolved_path": "/opt/test/_rusticol.so",
                "size_bytes": 20,
                "sha256": "4" * 64,
            }
        ],
        "build_info_files": [
            {
                "relative_path": "pyamplicol/_build_info.json",
                "path": "/opt/test/_build_info.json",
                "resolved_path": "/opt/test/_build_info.json",
                "size_bytes": 30,
                "sha256": "a" * 64,
            }
        ],
    },
    "active_build_info": {
        "path": "/opt/test/_build_info.json",
        "resolved_path": "/opt/test/_build_info.json",
        "size_bytes": 30,
        "sha256": "a" * 64,
        "payload": {
            "source_revision": SOURCE_REVISION,
            "source_checkout": "/checkout",
            "native_build_inputs_sha256": "5" * 64,
        },
    },
    "native_extension": {
        "path": "/opt/test/_rusticol.so",
        "resolved_path": "/opt/test/_rusticol.so",
        "size_bytes": 20,
        "sha256": "4" * 64,
        "package_version": "0.1.0",
        "build_inputs_sha256": "5" * 64,
    },
    "dependencies": {
        name: {
            "present": True,
            "path": f"/checkout/{name}",
            "resolved_path": f"/checkout/{name}",
            "sha256": hashlib.sha256(name.encode()).hexdigest(),
            "size_bytes": 42,
        }
        for name in m0._RUNTIME_DEPENDENCIES
    },
}


def _write_bytes(path: Path, raw: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {
        "path": str(path.resolve()),
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _write_json(path: Path, payload: object) -> dict[str, object]:
    raw = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()
    return _write_bytes(path, raw)


def _fixture(path: Path) -> dict[str, object]:
    identity = _write_bytes(path, b'{"points":"m0-test"}\n')
    return {
        "point_count": 2,
        "points_sha256": POINTS_SHA256,
        "file": {
            **identity,
            "resolved_path": identity["path"],
        },
    }


def _selector(layout: str) -> dict[str, object]:
    union = layout == "all-flow-union"
    return {
        "color_flow_request": "1",
        "resolved_color_flow_id": None if union else FLOW_ID,
        "helicity_request": "1",
        "resolved_helicity_id": HELICITY_ID if union else None,
        "color_flow_count": 1,
        "helicity_count": 2,
        "structural_zero_helicity_count": 0,
        "workload": m0.UNION_WORKLOAD if union else m0.SELECTED_WORKLOAD,
    }


def _make_capture(
    root: Path,
    *,
    model: str,
    layout: str,
    fixture: dict[str, object],
) -> tuple[dict[str, object], dict[str, Any]]:
    argument_tokens = [
        "--jit-optimization-level",
        "3",
        "--lc-flow-layout",
        layout,
    ]
    if model != "built-in-sm":
        argument_tokens.extend(("--prepared-model", f"/prepared/{model}.pacbin"))
    arguments = z6g._arguments(*argument_tokens)
    schedule = z6g._passing_schedule(
        jit_optimization_level=3,
        lc_flow_layout=layout,
    )
    for entry in schedule["entries"]:
        entry["pre_timing_verification"]["effective_contract"][
            "jit_optimization_level"
        ] = benchmark._expected_effective_jit_optimization_level(
            arguments,
            mode=entry["mode"],
        )
    root_bindings = {
        "source_identity_sha256": m0._canonical_sha256(SOURCE),
        "runtime_provenance_sha256": m0._canonical_sha256(RUNTIME),
        "interpreter_sha256": RUNTIME["interpreter"]["sha256"],
        "native_extension_sha256": RUNTIME["native_extension"]["sha256"],
    }
    for entry in schedule["entries"]:
        verification = entry["pre_timing_verification"]
        for side in ("expected", "observed"):
            verification[side].update(root_bindings)
    semantic = z6g._artifact_semantic_identity(
        model_name="built-in-sm" if model == "built-in-sm" else "sm",
        model_content_sha256="c" * 64 if model == "built-in-sm" else "d" * 64,
        color_id=FLOW_ID,
        color_word=tuple(FLOW_WORD),
        helicity_id=HELICITY_ID,
        helicity_values=tuple(HELICITY_VALUES),
    )
    profiles = z6g._passing_profiles(
        schedule,
        semantic_identity=semantic,
        selector_contract=_selector(layout),
        fixture=fixture,
    )
    validation_preflight = benchmark._validation_preflight_contract(
        arguments,
        {
            mode: profiles[mode]
            for mode in benchmark.AUTHORITATIVE_EXECUTION_MODES
            if mode in profiles
        },
    )
    validation_preflight["worker_evidence"] = {
        mode: {"fixture": "synthetic"}
        for mode in benchmark.AUTHORITATIVE_EXECUTION_MODES
    }
    validation = benchmark._pairwise_profile_validation(profiles)
    capture = benchmark._capture_acceptance(
        arguments,
        profiles,
        validation,
        profile_schedule=schedule,
    )
    assert capture["complete"] is True
    assert capture["passes"] is True
    per_layout = benchmark._milestone0_acceptance_manifest(arguments, capture)
    if model == "built-in-sm":
        model_identities = {
            "compiled": {
                "kind": "built-in-sm-source",
                "resource_id": None,
                "source_revision": SOURCE_REVISION,
                "compile_excluded_from_generation": False,
            },
            "eager": {
                "kind": "packaged-prepared-model",
                "resource_id": m0.BUILTIN_PACKAGED_MODEL_RESOURCE_ID,
                "size_bytes": 100,
                "sha256": "7" * 64,
                "compile_excluded_from_generation": True,
            },
            "recurrence": {
                "kind": "packaged-prepared-model",
                "resource_id": m0.BUILTIN_PACKAGED_MODEL_RESOURCE_ID,
                "size_bytes": 100,
                "sha256": "7" * 64,
                "compile_excluded_from_generation": True,
            },
        }
    else:
        model_identities = {
            mode: {
                "kind": "explicit-prepared-model",
                "resource_id": None,
                "file": {
                    "size_bytes": 100,
                    "sha256": "8" * 64,
                    "path": f"/prepared/{model}.pacbin",
                    "resolved_path": f"/prepared/{model}.pacbin",
                },
                "compile_excluded_from_generation": True,
            }
            for mode in m0.MODES
        }
    generation: dict[str, dict[str, Any]] = {}
    for mode in m0.MODES:
        signature = {
            "kind": "pyamplicol-benchmark-generation-signature",
            "schema_version": 1,
            "source_revision": SOURCE_REVISION,
            "runtime_provenance_sha256": m0._canonical_sha256(RUNTIME),
            "mode": mode,
            "process": m0.PROCESS,
            "model": model_identities[mode],
            "color_accuracy": "lc",
            "lc_flow_layout": layout,
            "jit_optimization_level": 3,
        }
        generation[mode] = {
            "model_source": model_identities[mode],
            "effective_contract": {
                "execution_mode": mode,
                "backend": "jit",
                "jit_optimization_level": (
                    m0._expected_effective_jit_optimization_level(
                        model_identities[mode],
                        mode=mode,
                    )
                ),
                "color_accuracy": "lc",
                "lc_flow_layout": layout,
            },
            "semantic_generation_signature": signature,
            "semantic_generation_signature_sha256": m0._canonical_sha256(signature),
            "artifact_semantic_identity": profiles[mode]["artifact_semantic_identity"],
            "artifact_semantic_identity_sha256": profiles[mode][
                "artifact_semantic_identity_sha256"
            ],
        }
    configuration = {
        "batch_sizes": list(m0.BATCHES),
        "target_runtime_seconds": arguments.target_runtime,
        "minimum_samples": arguments.minimum_samples,
        "subprocess_samples": arguments.subprocess_samples,
        "warmup_runs": arguments.warmup_runs,
        "generation_timeout_seconds": arguments.generation_timeout,
        "profile_timeout_seconds": arguments.profile_timeout,
        "color_flow_request": arguments.color_flow,
        "helicity_request": arguments.helicity,
        "lc_flow_layout": layout,
        "gluon_count": 6,
        "validation_samples": arguments.validation_samples,
        "point_tile_size": arguments.point_tile_size,
        "jit_optimization_level": 3,
        "validation_point_artifact": fixture["file"]["path"],
        "generation_only": False,
        "allow_diagnostic_incomplete_success": False,
        "modes": list(m0.MODES),
        "prepared_model_path": (
            None if model == "built-in-sm" else f"/prepared/{model}.pacbin"
        ),
        "model_identities": model_identities,
        "validation_seed": benchmark.VALIDATION_SEED,
        "specialize_flow_at_generation": False,
        "external_watchdog_required_for_long_runs": True,
    }
    recurrence_runtime_acceptance = benchmark._recurrence_runtime_acceptance(
        arguments,
        source_identity=SOURCE,
        runtime_provenance=RUNTIME,
        generation=generation,
        profiles=profiles,
        validation_summary=validation,
        profile_schedule=schedule,
    )
    payload = {
        "kind": m0.CAPTURE_KIND,
        "schema_version": m0.CAPTURE_SCHEMA,
        "complete": True,
        "passes": True,
        "capture_acceptance": capture,
        "milestone0_acceptance": per_layout,
        "recurrence_runtime_acceptance": recurrence_runtime_acceptance,
        "source": copy.deepcopy(SOURCE),
        "runtime_provenance": copy.deepcopy(RUNTIME),
        "provenance": {
            "started_at_utc": "2026-07-24T00:00:00+00:00",
            "finished_at_utc": "2026-07-24T00:10:00+00:00",
            "wall_seconds": 600.0,
            "host": copy.deepcopy(HOST),
            "working_directory": "/checkout",
            "driver_command": {"argv": ["python", "benchmark"]},
            "post_worker_identity_rechecks": [],
            "external_watchdog_required_for_long_runs": True,
        },
        "process": "u u~ > Z g g g g g g",
        "process_name": "uubar_z6g",
        "workload": (
            m0.UNION_WORKLOAD if layout == "all-flow-union" else m0.SELECTED_WORKLOAD
        ),
        "configuration": configuration,
        "generation": generation,
        "profile_schedule": schedule,
        "validation_preflight": validation_preflight,
        "paired_profile_coordination": None,
        "profiles": profiles,
        "validation_summary": validation,
        "selector_contracts_match": validation["selectors_match"],
        "validation_fixtures_match": validation["fixtures_match"],
        "lane_comparisons": validation["comparisons"],
        "result_json": f"/results/{model}-{layout}.json",
    }
    ref = _write_json(root / f"{model}-{layout}.json", payload)
    return ref, payload


def _amplicol_evidence(
    root: Path,
    *,
    role: str,
    fixture_ref: dict[str, object],
    values: list[list[float]],
) -> tuple[dict[str, object], dict[str, Any]]:
    stem = role.replace("-", "_")
    executable = _write_bytes(root / f"{stem}.exe", b"executable")
    Path(executable["path"]).chmod(0o755)
    library = _write_bytes(root / f"{stem}.dylib", b"library")
    source = _write_bytes(root / f"{stem}.cc", b"int main() {}\n")
    samples: list[dict[str, object]] = []
    timings: list[float] = []
    position_offset = 1 if role == m0.AMPLICOL_UNION_ROLE else 0
    for index in range(m0.MIN_SAMPLES):
        seconds_per_point = 4.0e-5 * (1.0 + index * 0.01)
        evaluated_point_count = 1000
        elapsed = seconds_per_point * evaluated_point_count
        selector_argument = (
            f"--helicity-id={HELICITY_ID}"
            if role == m0.AMPLICOL_UNION_ROLE
            else f"--color-flow-id={FLOW_ID}"
        )
        command = [
            executable["path"],
            f"--workload={role}",
            f"--round={index}",
            f"--momenta={fixture_ref['path']}",
            f"--source-revision={AMPLICOL_REVISION}",
            selector_argument,
        ]
        command_sha256 = m0._canonical_sha256(command)
        stdout_payload = {
            "kind": "amplicol-m0-probe-result",
            "schema_version": 1,
            "role": role,
            "sample_index": index,
            "evaluated_point_count": evaluated_point_count,
            "elapsed_seconds": elapsed,
            "seconds_per_point": seconds_per_point,
            "selected_totals": values,
            "resolved_sums": values,
        }
        stdout = json.dumps(
            stdout_payload,
            sort_keys=True,
            separators=(",", ":"),
        )
        raw_payload = {
            "kind": "pyamplicol-amplicol-m0-raw-sample",
            "schema_version": 1,
            "role": role,
            "sample_index": index,
            "command_sha256": command_sha256,
            "evaluated_point_count": evaluated_point_count,
            "elapsed_seconds": elapsed,
            "seconds_per_point": seconds_per_point,
            "stdout": stdout,
            "stdout_sha256": hashlib.sha256(stdout.encode()).hexdigest(),
        }
        raw_payload["content_sha256"] = m0._canonical_sha256(raw_payload)
        raw_output = _write_json(
            root / f"{stem}-sample-{index}.json",
            raw_payload,
        )
        start_second = 4 * index + 2 * position_offset
        sample = {
            "sample_index": index,
            "interleave_round": index,
            "interleave_position": 2 * index + position_offset,
            "started_at_utc": f"2026-07-24T01:00:{start_second:02d}+00:00",
            "finished_at_utc": f"2026-07-24T01:00:{start_second + 1:02d}+00:00",
            "subprocess": True,
            "command": command,
            "command_sha256": command_sha256,
            "evaluated_point_count": evaluated_point_count,
            "elapsed_seconds": elapsed,
            "seconds_per_point": seconds_per_point,
            "interrupted": False,
            "raw_output_file": raw_output,
        }
        samples.append(sample)
        timings.append(seconds_per_point)
    median = statistics.median(timings)
    mad = statistics.median(abs(value - median) for value in timings)
    union = role == m0.AMPLICOL_UNION_ROLE
    comparisons = [
        {
            "point_index": index,
            "passes": True,
        }
        for index in range(len(values))
    ]
    payload = {
        "kind": m0.AMPLICOL_KIND,
        "schema_version": m0.AMPLICOL_SCHEMA,
        "complete": True,
        "evidence_scope": "authoritative-host-capture-v1",
        "workload": m0.UNION_WORKLOAD if union else m0.SELECTED_WORKLOAD,
        "source": {
            "revision": AMPLICOL_REVISION,
            "dirty": False,
            "compiler": {
                "id": "apple-clang",
                "version": "16.0.0",
                "target": "arm64-apple-darwin",
                "flags_sha256": m0._canonical_sha256(["-O3"]),
            },
            "source_tree_sha256": m0._amplicol_source_tree_sha256([source]),
        },
        "host": copy.deepcopy(HOST),
        "process": {
            "expression": "u u~ > Z g g g g g g",
            "normalized_expression": m0.PROCESS,
        },
        "physical_axes": {
            "color_flow": {
                "count": 1,
                "ordered_ids_sha256": m0._canonical_sha256([FLOW_ID]),
            },
            "helicity": {
                "count": 2,
                "ordered_ids_sha256": m0._canonical_sha256(
                    [HELICITY_ID, OPPOSITE_HELICITY_ID]
                ),
            },
        },
        "selector": {
            "color_flow_request": "1",
            "resolved_color_flow_id": None if union else FLOW_ID,
            "color_flow_word": FLOW_WORD,
            "helicity_request": "1",
            "resolved_helicity_id": HELICITY_ID if union else None,
            "helicity_values": HELICITY_VALUES,
            "sum_axis": "color_flow" if union else "helicity",
            "source_to_generated_permutation": list(range(m0.EXTERNAL_LEG_COUNT)),
            "complete_physical_axes": True,
            "generation_specialized_axes": [],
        },
        "momenta": {
            "point_count": len(values),
            "points_sha256": POINTS_SHA256,
            "raw_file": {
                "path": fixture_ref["path"],
                "size_bytes": fixture_ref["size_bytes"],
                "sha256": fixture_ref["sha256"],
            },
        },
        "normalization_sha256": z6g._artifact_semantic_identity()[
            "normalization_sha256"
        ],
        "timing": {
            "boundary": "direct-library-total" if union else "amplitude-evaluation",
            "batch_semantics": "scalar-normalized-per-point",
            "statistics_contract": "subprocess-median-and-raw-mad-v1",
            "interleave_group_sha256": m0._canonical_sha256(
                {
                    "kind": "amplicol-m0-paired-interleave",
                    "roles": [
                        m0.AMPLICOL_SELECTED_ROLE,
                        m0.AMPLICOL_UNION_ROLE,
                    ],
                    "rounds": m0.MIN_SAMPLES,
                }
            ),
            "sample_count": len(samples),
            "median_seconds_per_point": median,
            "mad_seconds_per_point": mad,
            "samples": samples,
            "samples_sha256": m0._canonical_sha256(samples),
        },
        "validation": {
            "selected_totals": copy.deepcopy(values),
            "resolved_sums": copy.deepcopy(values),
            "point_comparisons": comparisons,
            "maximum_absolute_difference": 0.0,
            "maximum_relative_difference": 0.0,
            "passes": True,
        },
        "binary_evidence": {
            "executable": executable,
            "linked_libraries": [library],
            "source_files": [source],
        },
    }
    payload["content_sha256"] = m0._canonical_sha256(payload)
    ref = _write_json(root / f"amplicol-{role}.json", payload)
    return ref, payload


def _interleave_projection(
    role: str,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "role": role,
            "round": sample["interleave_round"],
            "position": sample["interleave_position"],
            "started_at_utc": sample["started_at_utc"],
            "finished_at_utc": sample["finished_at_utc"],
            "command_sha256": sample["command_sha256"],
        }
        for sample in payload["timing"]["samples"]
    ]


def _make_corpus(root: Path) -> dict[str, Any]:
    fixture = _fixture(root / "momenta.json")
    capture_refs: dict[str, dict[str, dict[str, object]]] = {}
    captures: dict[tuple[str, str], dict[str, Any]] = {}
    for model in m0.MODELS:
        capture_refs[model] = {}
        for layout in m0.LAYOUTS:
            ref, payload = _make_capture(
                root,
                model=model,
                layout=layout,
                fixture=fixture,
            )
            capture_refs[model][layout] = ref
            captures[(model, layout)] = payload
    selected_values = captures[("built-in-sm", "topology-replay")]["profiles"][
        "compiled"
    ]["validation"]["selected_totals"]
    union_values = captures[("built-in-sm", "all-flow-union")]["profiles"]["compiled"][
        "validation"
    ]["selected_totals"]
    selected_ref, selected_payload = _amplicol_evidence(
        root,
        role=m0.AMPLICOL_SELECTED_ROLE,
        fixture_ref=fixture["file"],
        values=selected_values,
    )
    union_ref, union_payload = _amplicol_evidence(
        root,
        role=m0.AMPLICOL_UNION_ROLE,
        fixture_ref=fixture["file"],
        values=union_values,
    )
    combined_interleave = sorted(
        [
            *_interleave_projection(m0.AMPLICOL_SELECTED_ROLE, selected_payload),
            *_interleave_projection(m0.AMPLICOL_UNION_ROLE, union_payload),
        ],
        key=lambda record: record["position"],
    )
    interleave_sha256 = m0._amplicol_interleave_group_sha256(combined_interleave)
    refreshed: dict[str, dict[str, object]] = {}
    for role, payload, ref in (
        (m0.AMPLICOL_SELECTED_ROLE, selected_payload, selected_ref),
        (m0.AMPLICOL_UNION_ROLE, union_payload, union_ref),
    ):
        payload["timing"]["interleave_group_sha256"] = interleave_sha256
        without_digest = dict(payload)
        without_digest.pop("content_sha256")
        payload["content_sha256"] = m0._canonical_sha256(without_digest)
        refreshed[role] = _write_json(Path(ref["path"]), payload)
    selected_ref = refreshed[m0.AMPLICOL_SELECTED_ROLE]
    union_ref = refreshed[m0.AMPLICOL_UNION_ROLE]
    semantic_builtin = captures[("built-in-sm", "topology-replay")]["profiles"][
        "compiled"
    ]["artifact_semantic_identity"]
    semantic_ufo = captures[("ufo-sm", "topology-replay")]["profiles"]["compiled"][
        "artifact_semantic_identity"
    ]
    configuration_builtin = captures[("built-in-sm", "topology-replay")][
        "configuration"
    ]
    configuration_ufo = captures[("ufo-sm", "topology-replay")]["configuration"]
    request = {
        "kind": m0.REQUEST_KIND,
        "schema_version": m0.REQUEST_SCHEMA,
        "captures": capture_refs,
        "amplicol_evidence": {
            m0.AMPLICOL_SELECTED_ROLE: selected_ref,
            m0.AMPLICOL_UNION_ROLE: union_ref,
        },
        "expected": {
            "pyamplicol_source_revision": SOURCE_REVISION,
            "amplicol_source_revision": AMPLICOL_REVISION,
            "process": m0.PROCESS,
            "runtime_provenance_sha256": m0._canonical_sha256(
                m0._path_stripped(RUNTIME)
            ),
            "host_sha256": m0._canonical_sha256(HOST),
            "momenta_points_sha256": POINTS_SHA256,
            "normalization_sha256": semantic_builtin["normalization_sha256"],
            "model_common_physics_identity_sha256": {
                "built-in-sm": semantic_builtin["manifest_model_identity"][
                    "common_physics_identity_sha256"
                ],
                "ufo-sm": semantic_ufo["manifest_model_identity"][
                    "common_physics_identity_sha256"
                ],
            },
            "generation_model_identities_sha256": {
                "built-in-sm": m0._canonical_sha256(
                    m0._path_stripped(configuration_builtin["model_identities"])
                ),
                "ufo-sm": m0._canonical_sha256(
                    m0._path_stripped(configuration_ufo["model_identities"])
                ),
            },
            "color_flow": {"id": FLOW_ID, "word": FLOW_WORD},
            "helicity": {"id": HELICITY_ID, "values": HELICITY_VALUES},
            "external_leg_permutation": list(range(m0.EXTERNAL_LEG_COUNT)),
        },
    }
    request_ref = _write_json(root / "request.json", request)
    return {
        "request": request,
        "request_ref": request_ref,
        "captures": captures,
        "capture_refs": capture_refs,
        "amplicol": {
            m0.AMPLICOL_SELECTED_ROLE: selected_payload,
            m0.AMPLICOL_UNION_ROLE: union_payload,
        },
    }


def _run(root: Path, corpus: dict[str, Any]) -> tuple[dict[str, Any], int]:
    request_path = Path(corpus["request_ref"]["path"])
    return m0.combine(
        request_path=request_path,
        request_sha256=corpus["request_ref"]["sha256"],
        output_path=root / "decision.json",
    )


def _rewrite_capture(
    corpus: dict[str, Any],
    *,
    model: str,
    layout: str,
) -> None:
    payload = corpus["captures"][(model, layout)]
    path = Path(corpus["capture_refs"][model][layout]["path"])
    ref = _write_json(path, payload)
    corpus["capture_refs"][model][layout] = ref
    corpus["request"]["captures"][model][layout] = ref


def _rewrite_amplicol(corpus: dict[str, Any], role: str) -> None:
    payload = corpus["amplicol"][role]
    without_digest = dict(payload)
    without_digest.pop("content_sha256", None)
    payload["content_sha256"] = m0._canonical_sha256(without_digest)
    path = Path(corpus["request"]["amplicol_evidence"][role]["path"])
    ref = _write_json(path, payload)
    corpus["request"]["amplicol_evidence"][role] = ref


def _refresh_amplicol_timing(corpus: dict[str, Any], role: str) -> None:
    timing = corpus["amplicol"][role]["timing"]
    samples = timing["samples"]
    values = [sample["seconds_per_point"] for sample in samples]
    median = statistics.median(values)
    timing["sample_count"] = len(samples)
    timing["median_seconds_per_point"] = median
    timing["mad_seconds_per_point"] = statistics.median(
        abs(value - median) for value in values
    )
    timing["samples_sha256"] = m0._canonical_sha256(samples)
    _rewrite_amplicol(corpus, role)
    _rewrite_request(corpus)


def _rewrite_request(corpus: dict[str, Any]) -> None:
    path = Path(corpus["request_ref"]["path"])
    corpus["request_ref"] = _write_json(path, corpus["request"])


def _capture_with_compiled_recurrence_ratio(
    capture: m0.Capture,
    *,
    batch: int,
    ratio: float,
) -> m0.Capture:
    timings = copy.deepcopy(capture.timings)
    recurrence = timings["recurrence"][str(batch)]
    compiled = timings["compiled"][str(batch)]
    compiled_values = [value * ratio for value in recurrence["raw_seconds_per_point"]]
    compiled_median = statistics.median(compiled_values)
    compiled["sample_count"] = len(compiled_values)
    compiled["median_seconds_per_point"] = compiled_median
    compiled["mad_seconds_per_point"] = statistics.median(
        abs(value - compiled_median) for value in compiled_values
    )
    compiled["raw_seconds_per_point"] = compiled_values
    return replace(capture, timings=timings)


def test_positive_fixture_emits_content_addressed_acceptance(
    tmp_path: Path,
) -> None:
    corpus = _make_corpus(tmp_path)
    decision, code = _run(tmp_path, corpus)
    assert code == 0
    assert decision["accepted"] is True
    assert decision["status"] == "accepted"
    assert decision["errors"] == []
    digest = decision.pop("content_sha256")
    assert digest == m0._canonical_sha256(decision)
    assert set(decision["layout_captures"]) == set(m0.MODELS)
    assert decision["validation"]["amplicol_selected_flow_parity"] is True
    comparison = decision["comparisons"][m0.SELECTED_WORKLOAD]["built-in-sm"][
        "compiled"
    ]["1024"]
    assert comparison["pyamplicol_over_amplicol"] > 0.0
    assert comparison["amplicol_boundary"] == "amplitude-evaluation"
    ceiling = decision["validation"]["compiled_recurrence_runtime_ceiling"]
    assert ceiling["passes"] is True
    assert ceiling["required_batches"] == list(m0.COMPILED_RECURRENCE_BATCHES)
    assert ceiling["maximum_compiled_over_recurrence"] == (
        m0.MAX_COMPILED_OVER_RECURRENCE
    )
    assert {
        (model, layout, int(batch))
        for model, model_cells in ceiling["cells"].items()
        for layout, layout_cells in model_cells.items()
        for batch in layout_cells
    } == {
        (model, layout, batch)
        for model in m0.MODELS
        for layout in m0.LAYOUTS
        for batch in m0.COMPILED_RECURRENCE_BATCHES
    }


def test_compiled_recurrence_ceiling_is_inclusive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = _make_corpus(tmp_path)
    validate_capture = m0._validate_capture

    def at_ceiling(
        loaded: m0.LoadedJson,
        *,
        model: str,
        layout: str,
        expected: dict[str, Any],
        benchmark: ModuleType,
    ) -> m0.Capture:
        capture = validate_capture(
            loaded,
            model=model,
            layout=layout,
            expected=expected,
            benchmark=benchmark,
        )
        for batch in m0.COMPILED_RECURRENCE_BATCHES:
            capture = _capture_with_compiled_recurrence_ratio(
                capture,
                batch=batch,
                ratio=m0.MAX_COMPILED_OVER_RECURRENCE,
            )
        return capture

    monkeypatch.setattr(m0, "_validate_capture", at_ceiling)
    decision, code = _run(tmp_path, corpus)

    assert code == 0
    assert decision["accepted"] is True
    ceiling = decision["validation"]["compiled_recurrence_runtime_ceiling"]
    for model in m0.MODELS:
        for layout in m0.LAYOUTS:
            for batch in m0.COMPILED_RECURRENCE_BATCHES:
                cell = ceiling["cells"][model][layout][str(batch)]
                assert cell["passes"] is True
                assert cell["compiled_over_recurrence"] == pytest.approx(
                    m0.MAX_COMPILED_OVER_RECURRENCE
                )


@pytest.mark.parametrize(
    ("target_model", "target_layout", "target_batch"),
    [
        pytest.param(model, layout, batch, id=f"{model}-{layout}-{batch}")
        for model in m0.MODELS
        for layout in m0.LAYOUTS
        for batch in m0.COMPILED_RECURRENCE_BATCHES
    ],
)
def test_rejects_every_required_compiled_recurrence_ceiling_cell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_model: str,
    target_layout: str,
    target_batch: int,
) -> None:
    corpus = _make_corpus(tmp_path)
    validate_capture = m0._validate_capture

    def over_ceiling(
        loaded: m0.LoadedJson,
        *,
        model: str,
        layout: str,
        expected: dict[str, Any],
        benchmark: ModuleType,
    ) -> m0.Capture:
        capture = validate_capture(
            loaded,
            model=model,
            layout=layout,
            expected=expected,
            benchmark=benchmark,
        )
        if model == target_model and layout == target_layout:
            capture = _capture_with_compiled_recurrence_ratio(
                capture,
                batch=target_batch,
                ratio=m0.MAX_COMPILED_OVER_RECURRENCE + 1.0e-6,
            )
        return capture

    monkeypatch.setattr(m0, "_validate_capture", over_ceiling)
    decision, code = _run(tmp_path, corpus)

    assert code == 2
    assert decision["accepted"] is False
    assert decision["status"] == "rejected"
    assert decision["validation"] is None
    assert decision["timings"] is None
    assert decision["comparisons"] is None
    assert len(decision["errors"]) == 1
    error = decision["errors"][0]
    assert f"{target_model}/{target_layout}" in error
    assert f"batch {target_batch}" in error
    assert "compiled qq_Z6g exceeds the recurrence ceiling" in error
    assert decision["policy"]["maximum_compiled_over_recurrence"] == (
        m0.MAX_COMPILED_OVER_RECURRENCE
    )


def test_producer_shaped_builtin_model_identity_split_is_accepted(
    tmp_path: Path,
) -> None:
    corpus = _make_corpus(tmp_path)
    for layout in m0.LAYOUTS:
        capture = corpus["captures"][("built-in-sm", layout)]
        identities = capture["configuration"]["model_identities"]
        assert identities["compiled"]["kind"] == "built-in-sm-source"
        assert identities["compiled"]["compile_excluded_from_generation"] is False
        assert identities["eager"]["kind"] == "packaged-prepared-model"
        assert identities["recurrence"]["kind"] == "packaged-prepared-model"
        for mode in m0.MODES:
            generation = capture["generation"][mode]
            assert generation["model_source"] == identities[mode]
            assert (
                generation["semantic_generation_signature"]["model"]
                == (identities[mode])
            )

    decision, code = _run(tmp_path, corpus)
    assert code == 0
    assert decision["accepted"] is True


@pytest.mark.parametrize(
    ("model_kind", "mode", "expected_level"),
    [
        pytest.param("built-in-sm-source", "compiled", 3, id="builtin-compiled"),
        pytest.param(
            "packaged-prepared-model",
            "eager",
            m0.PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL,
            id="builtin-eager",
        ),
        pytest.param(
            "packaged-prepared-model",
            "recurrence",
            m0.PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL,
            id="builtin-recurrence",
        ),
        pytest.param("explicit-prepared-model", "compiled", 3, id="ufo-compiled"),
        pytest.param(
            "explicit-prepared-model",
            "eager",
            m0.PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL,
            id="ufo-eager",
        ),
        pytest.param(
            "explicit-prepared-model",
            "recurrence",
            m0.PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL,
            id="ufo-recurrence",
        ),
    ],
)
def test_effective_jit_level_is_execution_mode_aware(
    model_kind: str,
    mode: str,
    expected_level: int,
) -> None:
    assert (
        m0._expected_effective_jit_optimization_level(
            {"kind": model_kind},
            mode=mode,
        )
        == expected_level
    )


def test_producer_shaped_ufo_compiled_remains_o3(tmp_path: Path) -> None:
    corpus = _make_corpus(tmp_path)
    for layout in m0.LAYOUTS:
        generation = corpus["captures"][("ufo-sm", layout)]["generation"]
        assert generation["compiled"]["model_source"]["kind"] == (
            "explicit-prepared-model"
        )
        assert (
            generation["compiled"]["effective_contract"]["jit_optimization_level"] == 3
        )
        for mode in ("eager", "recurrence"):
            assert (
                generation[mode]["effective_contract"]["jit_optimization_level"]
                == m0.PREPARED_JIT_PORTABLE_OPTIMIZATION_LEVEL
            )

    decision, code = _run(tmp_path, corpus)
    assert code == 0
    assert decision["accepted"] is True


@pytest.mark.parametrize(
    ("model", "mode", "wrong_level"),
    [
        pytest.param("built-in-sm", "compiled", 2, id="builtin-compiled-o2"),
        pytest.param("built-in-sm", "eager", 3, id="builtin-eager-o3"),
        pytest.param("built-in-sm", "recurrence", 3, id="builtin-recurrence-o3"),
        pytest.param("ufo-sm", "compiled", 2, id="ufo-compiled-o2"),
        pytest.param("ufo-sm", "eager", 3, id="ufo-eager-o3"),
        pytest.param("ufo-sm", "recurrence", 3, id="ufo-recurrence-o3"),
    ],
)
def test_rejects_model_mode_effective_jit_level_drift(
    tmp_path: Path,
    model: str,
    mode: str,
    wrong_level: int,
) -> None:
    corpus = _make_corpus(tmp_path)
    capture = corpus["captures"][(model, "topology-replay")]
    capture["generation"][mode]["effective_contract"]["jit_optimization_level"] = (
        wrong_level
    )
    _rewrite_capture(corpus, model=model, layout="topology-replay")
    _rewrite_request(corpus)

    decision, code = _run(tmp_path, corpus)
    assert code == 2
    assert decision["accepted"] is False
    assert "wrong effective JIT optimization level" in decision["errors"][0]


def test_rejects_self_pinned_builtin_compiled_model_branch_mismatch(
    tmp_path: Path,
) -> None:
    corpus = _make_corpus(tmp_path)
    drifted_compiled_identity = {
        "kind": "packaged-prepared-model",
        "resource_id": m0.BUILTIN_PACKAGED_MODEL_RESOURCE_ID,
        "size_bytes": 100,
        "sha256": "e" * 64,
        "compile_excluded_from_generation": True,
    }
    for layout in m0.LAYOUTS:
        capture = corpus["captures"][("built-in-sm", layout)]
        identities = capture["configuration"]["model_identities"]
        identities["compiled"] = copy.deepcopy(drifted_compiled_identity)
        generation = capture["generation"]["compiled"]
        generation["model_source"] = copy.deepcopy(drifted_compiled_identity)
        signature = generation["semantic_generation_signature"]
        signature["model"] = copy.deepcopy(drifted_compiled_identity)
        generation["semantic_generation_signature_sha256"] = m0._canonical_sha256(
            signature
        )
        _rewrite_capture(corpus, model="built-in-sm", layout=layout)
    corpus["request"]["expected"]["generation_model_identities_sha256"][
        "built-in-sm"
    ] = m0._canonical_sha256(
        m0._path_stripped(
            corpus["captures"][("built-in-sm", "topology-replay")]["configuration"][
                "model_identities"
            ]
        )
    )
    _rewrite_request(corpus)

    decision, code = _run(tmp_path, corpus)
    assert code == 2
    assert decision["accepted"] is False
    assert "must use built-in-sm-source" in decision["errors"][0]


@pytest.mark.parametrize(
    ("model", "mode", "wrong_identity", "expected_kind"),
    [
        pytest.param(
            "built-in-sm",
            "compiled",
            {
                "kind": "packaged-prepared-model",
                "resource_id": m0.BUILTIN_PACKAGED_MODEL_RESOURCE_ID,
                "size_bytes": 100,
                "sha256": "e" * 64,
                "compile_excluded_from_generation": True,
            },
            "built-in-sm-source",
            id="builtin-compiled",
        ),
        *[
            pytest.param(
                "built-in-sm",
                mode,
                {
                    "kind": "explicit-prepared-model",
                    "resource_id": None,
                    "file": {
                        "path": "/wrong/model.pacbin",
                        "resolved_path": "/wrong/model.pacbin",
                        "size_bytes": 100,
                        "sha256": "e" * 64,
                    },
                    "compile_excluded_from_generation": True,
                },
                "packaged-prepared-model",
                id=f"builtin-{mode}",
            )
            for mode in ("eager", "recurrence")
        ],
        *[
            pytest.param(
                "ufo-sm",
                mode,
                {
                    "kind": "packaged-prepared-model",
                    "resource_id": m0.BUILTIN_PACKAGED_MODEL_RESOURCE_ID,
                    "size_bytes": 100,
                    "sha256": "e" * 64,
                    "compile_excluded_from_generation": True,
                },
                "explicit-prepared-model",
                id=f"ufo-{mode}",
            )
            for mode in m0.MODES
        ],
    ],
)
def test_model_family_mode_policy_matrix_rejects_supported_wrong_branches(
    tmp_path: Path,
    model: str,
    mode: str,
    wrong_identity: dict[str, object],
    expected_kind: str,
) -> None:
    corpus = _make_corpus(tmp_path)
    identities = copy.deepcopy(
        corpus["captures"][(model, "topology-replay")]["configuration"][
            "model_identities"
        ]
    )
    identities[mode] = wrong_identity
    with pytest.raises(m0.EvidenceError, match=f"must use {expected_kind}"):
        m0._validate_generation_model_identities(
            identities,
            model_family=model,
            source_revision=SOURCE_REVISION,
            label="test.model_identities",
        )


def test_cli_accepts_the_content_addressed_positive_fixture(
    tmp_path: Path,
) -> None:
    corpus = _make_corpus(tmp_path)
    output = tmp_path / "cli-decision.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(M0_SCRIPT),
            "--request",
            corpus["request_ref"]["path"],
            "--request-sha256",
            corpus["request_ref"]["sha256"],
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["accepted"] is True
    assert summary["output"] == str(output)
    decision = json.loads(output.read_text(encoding="utf-8"))
    assert decision["content_sha256"] == m0._canonical_sha256(
        {key: value for key, value in decision.items() if key != "content_sha256"}
    )


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda corpus: corpus["captures"][
                ("built-in-sm", "topology-replay")
            ].__setitem__("complete", False),
            id="incomplete-capture",
        ),
        pytest.param(
            lambda corpus: corpus["captures"][
                ("built-in-sm", "topology-replay")
            ].__setitem__("schema_version", 5),
            id="obsolete-schema-5-capture",
        ),
        pytest.param(
            lambda corpus: corpus["captures"][("built-in-sm", "topology-replay")][
                "configuration"
            ].__setitem__("jit_optimization_level", 2),
            id="o2-capture",
        ),
        pytest.param(
            lambda corpus: corpus["captures"][("built-in-sm", "topology-replay")][
                "generation"
            ]["compiled"]["effective_contract"].__setitem__(
                "jit_optimization_level", 2
            ),
            id="compiled-generation-not-o3",
        ),
        pytest.param(
            lambda corpus: corpus["captures"][("built-in-sm", "topology-replay")][
                "generation"
            ]["recurrence"]["effective_contract"].__setitem__(
                "jit_optimization_level", 3
            ),
            id="recurrence-prepared-generation-contract-drift",
        ),
        pytest.param(
            lambda corpus: corpus["captures"][("built-in-sm", "topology-replay")][
                "configuration"
            ].__setitem__("batch_sizes", [1, 128]),
            id="missing-batch",
        ),
        pytest.param(
            lambda corpus: corpus["captures"][("built-in-sm", "topology-replay")][
                "configuration"
            ].__setitem__("specialize_flow_at_generation", True),
            id="generation-fixed-headline",
        ),
        pytest.param(
            lambda corpus: corpus["captures"][("built-in-sm", "topology-replay")][
                "provenance"
            ].__setitem__("external_watchdog_required_for_long_runs", False),
            id="watchdog-provenance-disabled",
        ),
        pytest.param(
            lambda corpus: corpus["captures"][("built-in-sm", "topology-replay")][
                "validation_preflight"
            ].__setitem__("passes", False),
            id="validation-preflight-drift",
        ),
        pytest.param(
            lambda corpus: corpus["captures"][
                ("built-in-sm", "topology-replay")
            ].__setitem__(
                "paired_profile_coordination",
                {"kind": "unexpected-paired-campaign"},
            ),
            id="unexpected-paired-profile-coordination",
        ),
        pytest.param(
            lambda corpus: corpus["captures"][("built-in-sm", "topology-replay")][
                "source"
            ].__setitem__("revision", "e" * 40),
            id="source-identity-drift",
        ),
        pytest.param(
            lambda corpus: corpus["captures"][("built-in-sm", "topology-replay")][
                "provenance"
            ]["host"].__setitem__("logical_cpu_count", 99),
            id="host-identity-drift",
        ),
        pytest.param(
            lambda corpus: corpus["captures"][("built-in-sm", "topology-replay")][
                "runtime_provenance"
            ]["native_extension"].__setitem__("build_inputs_sha256", "f" * 64),
            id="runtime-identity-drift",
        ),
        pytest.param(
            lambda corpus: corpus["captures"][("built-in-sm", "topology-replay")][
                "profiles"
            ]["compiled"]["selector_contract"].__setitem__(
                "resolved_color_flow_id", "flow:forged"
            ),
            id="py-selector-drift",
        ),
        pytest.param(
            lambda corpus: corpus["captures"][("built-in-sm", "topology-replay")][
                "profiles"
            ]["compiled"]["artifact_semantic_identity"]["coverage"].__setitem__(
                "complete_physical_axes", False
            ),
            id="incomplete-physical-axes",
        ),
        pytest.param(
            lambda corpus: corpus["captures"][("built-in-sm", "topology-replay")][
                "profiles"
            ]["compiled"]["profiles"][0]["subprocess_samples"].pop(),
            id="too-few-py-subprocess-samples",
        ),
    ],
)
def test_capture_rejection_classes(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    corpus = _make_corpus(tmp_path)
    mutate(corpus)
    _rewrite_capture(corpus, model="built-in-sm", layout="topology-replay")
    _rewrite_request(corpus)
    decision, code = _run(tmp_path, corpus)
    assert code == 2
    assert decision["accepted"] is False
    assert decision["status"] == "rejected"
    assert decision["errors"]
    assert decision["comparisons"] is None


@pytest.mark.parametrize(
    ("role", "mutate"),
    [
        pytest.param(
            m0.AMPLICOL_SELECTED_ROLE,
            lambda payload: payload.__setitem__("complete", False),
            id="incomplete",
        ),
        pytest.param(
            m0.AMPLICOL_SELECTED_ROLE,
            lambda payload: payload.__setitem__("evidence_scope", "fixture"),
            id="fixture-evidence",
        ),
        pytest.param(
            m0.AMPLICOL_SELECTED_ROLE,
            lambda payload: payload["selector"].__setitem__(
                "resolved_color_flow_id", "flow:forged"
            ),
            id="selector-drift",
        ),
        pytest.param(
            m0.AMPLICOL_SELECTED_ROLE,
            lambda payload: payload["selector"].__setitem__(
                "source_to_generated_permutation", [0]
            ),
            id="truncated-source-permutation",
        ),
        pytest.param(
            m0.AMPLICOL_UNION_ROLE,
            lambda payload: payload["validation"]["resolved_sums"].__setitem__(
                1, [99.0, 0.0]
            ),
            id="resolved-sum-closure",
        ),
        pytest.param(
            m0.AMPLICOL_UNION_ROLE,
            lambda payload: payload["timing"]["samples"].pop(),
            id="too-few-samples",
        ),
        pytest.param(
            m0.AMPLICOL_UNION_ROLE,
            lambda payload: payload["source"]["compiler"].__setitem__(
                "version", "different"
            ),
            id="source-compiler-drift",
        ),
    ],
)
def test_amplicol_rejection_classes(
    tmp_path: Path,
    role: str,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    corpus = _make_corpus(tmp_path)
    mutate(corpus["amplicol"][role])
    if (
        role == m0.AMPLICOL_UNION_ROLE
        and len(corpus["amplicol"][role]["timing"]["samples"]) == 6
    ):
        samples = corpus["amplicol"][role]["timing"]["samples"]
        corpus["amplicol"][role]["timing"]["sample_count"] = len(samples)
        corpus["amplicol"][role]["timing"]["samples_sha256"] = m0._canonical_sha256(
            samples
        )
    _rewrite_amplicol(corpus, role)
    _rewrite_request(corpus)
    decision, code = _run(tmp_path, corpus)
    assert code == 2
    assert decision["accepted"] is False
    assert decision["errors"]


def test_rejects_content_hash_drift_before_parsing_evidence(tmp_path: Path) -> None:
    corpus = _make_corpus(tmp_path)
    evidence = Path(corpus["request"]["captures"]["ufo-sm"]["all-flow-union"]["path"])
    raw = evidence.read_bytes()
    evidence.write_bytes(b" " + raw[1:])
    decision, code = _run(tmp_path, corpus)
    assert code == 2
    assert "content hash drifted" in decision["errors"][0]


def test_rejects_blocked_amplicol_workloads_mislabeled_as_interleaved(
    tmp_path: Path,
) -> None:
    corpus = _make_corpus(tmp_path)
    selected = corpus["amplicol"][m0.AMPLICOL_SELECTED_ROLE]["timing"]["samples"]
    union = corpus["amplicol"][m0.AMPLICOL_UNION_ROLE]["timing"]["samples"]
    for index, sample in enumerate(selected):
        sample["interleave_position"] = index
    for index, sample in enumerate(union):
        sample["interleave_position"] = m0.MIN_SAMPLES + index
    for role in (m0.AMPLICOL_SELECTED_ROLE, m0.AMPLICOL_UNION_ROLE):
        samples = corpus["amplicol"][role]["timing"]["samples"]
        corpus["amplicol"][role]["timing"]["samples_sha256"] = m0._canonical_sha256(
            samples
        )
        _rewrite_amplicol(corpus, role)
    _rewrite_request(corpus)
    decision, code = _run(tmp_path, corpus)
    assert code == 2
    assert decision["accepted"] is False
    assert "interleav" in decision["errors"][0]


def test_rejects_amplicol_raw_momenta_file_drift(tmp_path: Path) -> None:
    corpus = _make_corpus(tmp_path)
    role = m0.AMPLICOL_UNION_ROLE
    replacement = _write_bytes(tmp_path / "different-momenta.json", b"different\n")
    corpus["amplicol"][role]["momenta"]["raw_file"] = replacement
    _rewrite_amplicol(corpus, role)
    _rewrite_request(corpus)
    decision, code = _run(tmp_path, corpus)
    assert code == 2
    assert "not bound to its evidence" in decision["errors"][0]


def test_rejects_timing_longer_than_subprocess_envelope(tmp_path: Path) -> None:
    corpus = _make_corpus(tmp_path)
    role = m0.AMPLICOL_SELECTED_ROLE
    sample = corpus["amplicol"][role]["timing"]["samples"][0]
    sample["elapsed_seconds"] = 1.0e6
    sample["seconds_per_point"] = 1.0e3
    raw_path = Path(sample["raw_output_file"]["path"])
    raw_payload = json.loads(raw_path.read_text(encoding="utf-8"))
    raw_payload["elapsed_seconds"] = sample["elapsed_seconds"]
    raw_payload["seconds_per_point"] = sample["seconds_per_point"]
    without_digest = dict(raw_payload)
    without_digest.pop("content_sha256")
    raw_payload["content_sha256"] = m0._canonical_sha256(without_digest)
    sample["raw_output_file"] = _write_json(raw_path, raw_payload)
    _refresh_amplicol_timing(corpus, role)
    decision, code = _run(tmp_path, corpus)
    assert code == 2
    assert "exceeds its subprocess envelope" in decision["errors"][0]


def test_rejects_unstructured_amplicol_raw_output(tmp_path: Path) -> None:
    corpus = _make_corpus(tmp_path)
    role = m0.AMPLICOL_SELECTED_ROLE
    sample = corpus["amplicol"][role]["timing"]["samples"][0]
    raw_path = Path(sample["raw_output_file"]["path"])
    sample["raw_output_file"] = _write_bytes(raw_path, b"garbage\n")
    _refresh_amplicol_timing(corpus, role)
    decision, code = _run(tmp_path, corpus)
    assert code == 2
    assert "is not strict JSON" in decision["errors"][0]


def test_rejects_self_pinned_empty_host_contract(tmp_path: Path) -> None:
    corpus = _make_corpus(tmp_path)
    for model in m0.MODELS:
        for layout in m0.LAYOUTS:
            corpus["captures"][(model, layout)]["provenance"]["host"] = {}
            _rewrite_capture(corpus, model=model, layout=layout)
    for role in (m0.AMPLICOL_SELECTED_ROLE, m0.AMPLICOL_UNION_ROLE):
        corpus["amplicol"][role]["host"] = {}
        _rewrite_amplicol(corpus, role)
    corpus["request"]["expected"]["host_sha256"] = m0._canonical_sha256({})
    _rewrite_request(corpus)
    decision, code = _run(tmp_path, corpus)
    assert code == 2
    assert "host keys differ" in decision["errors"][0]


def test_rejects_self_pinned_empty_runtime_contract(tmp_path: Path) -> None:
    corpus = _make_corpus(tmp_path)
    for model in m0.MODELS:
        for layout in m0.LAYOUTS:
            corpus["captures"][(model, layout)]["runtime_provenance"] = {}
            _rewrite_capture(corpus, model=model, layout=layout)
    corpus["request"]["expected"]["runtime_provenance_sha256"] = m0._canonical_sha256(
        {}
    )
    _rewrite_request(corpus)
    decision, code = _run(tmp_path, corpus)
    assert code == 2
    assert "runtime_provenance keys differ" in decision["errors"][0]


def test_rejects_self_pinned_null_model_identity_inventory(tmp_path: Path) -> None:
    corpus = _make_corpus(tmp_path)
    model = "built-in-sm"
    for layout in m0.LAYOUTS:
        corpus["captures"][(model, layout)]["configuration"]["model_identities"] = None
        _rewrite_capture(corpus, model=model, layout=layout)
    corpus["request"]["expected"]["generation_model_identities_sha256"][model] = (
        m0._canonical_sha256(None)
    )
    _rewrite_request(corpus)
    decision, code = _run(tmp_path, corpus)
    assert code == 2
    assert "model_identities must be a JSON object" in decision["errors"][0]


def test_rejects_absent_evidence(tmp_path: Path) -> None:
    corpus = _make_corpus(tmp_path)
    evidence = Path(
        corpus["request"]["amplicol_evidence"][m0.AMPLICOL_SELECTED_ROLE]["path"]
    )
    evidence.unlink()
    decision, code = _run(tmp_path, corpus)
    assert code == 2
    assert "does not exist" in decision["errors"][0]


def test_rejects_unknown_request_key(tmp_path: Path) -> None:
    corpus = _make_corpus(tmp_path)
    corpus["request"]["policy_override"] = {"minimum_samples": 1}
    _rewrite_request(corpus)
    decision, code = _run(tmp_path, corpus)
    assert code == 2
    assert "unknown=['policy_override']" in decision["errors"][0]


def test_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    corpus = _make_corpus(tmp_path)
    request_path = Path(corpus["request_ref"]["path"])
    request_path.write_text('{"kind":"a","kind":"b"}\n', encoding="utf-8")
    corpus["request_ref"] = {
        **corpus["request_ref"],
        "size_bytes": request_path.stat().st_size,
        "sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
    }
    decision, code = _run(tmp_path, corpus)
    assert code == 2
    assert "duplicate JSON key" in decision["errors"][0]

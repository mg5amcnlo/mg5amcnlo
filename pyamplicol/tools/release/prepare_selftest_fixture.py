# SPDX-License-Identifier: 0BSD
"""Prepare the portable 64-bit installed-package physics self-test fixture."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import re
import shutil
import sys
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = ROOT / "src" / "pyamplicol" / "assets" / "selftest"
PORTABLE_TEMPLATE = "portable-64le"
PORTABLE_OPTIMIZATION_LEVEL = 2
COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY = "compiled-plane-arena-v1"
COMPILED_PLANE_DIRECT_APPLICATION_ABI = "pyamplicol-compiled-plane-kernel-v2"
SYMJIT_APPLICATION_ABI = "symjit-application-storage-v3"
SYMJIT_PLANE_APPLICATION_ABI = "pyamplicol-symjit-plane-application-v2"
SYMJIT_F64_RUNTIME_CAPABILITY = "symjit.application.complex-f64.v1"
COMPATIBLE_TARGETS = (
    "aarch64-apple-darwin",
    "x86_64-apple-darwin",
    "x86_64-unknown-linux-gnu",
)
_RUNTIME_ARTIFACT_ID_PAYLOAD_ROLES = frozenset(
    {
        "compiled-model",
        "evaluator-manifest",
        "evaluator-state",
        "model-parameters",
        "runtime-physics",
    }
)
ARTIFACT_IDENTITY_CONTRACT = {
    "kind": "pyamplicol-runtime-payload-identity",
    "schema_version": 1,
}


def validate_artifact_identity_contract(
    manifest: Mapping[str, object],
    *,
    context: str,
) -> None:
    extensions = manifest.get("extensions")
    if (
        not isinstance(extensions, dict)
        or extensions.get("artifact_identity") != ARTIFACT_IDENTITY_CONTRACT
    ):
        raise RuntimeError(
            f"{context} lacks the current artifact identity contract; regenerate it"
        )


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _artifact_id(manifest: Mapping[str, object]) -> str:
    payloads = manifest.get("payloads")
    if not isinstance(payloads, list):
        raise RuntimeError("artifact payload inventory is invalid")
    records = [
        dict(payload)
        for payload in payloads
        if isinstance(payload, dict)
        and payload.get("role") in _RUNTIME_ARTIFACT_ID_PAYLOAD_ROLES
    ]
    records.sort(key=lambda payload: str(payload.get("path", "")))
    content = {
        "kind": "pyamplicol-runtime-payload-identity",
        "schema_version": 1,
        "payloads": records,
    }
    return hashlib.sha256(_canonical_json(content)).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compiled_model_payload(
    manifest: Mapping[str, object],
) -> tuple[dict[str, Any], Path]:
    payloads = manifest.get("payloads")
    if not isinstance(payloads, list):
        raise RuntimeError("artifact payload inventory is invalid")
    matches = [
        payload
        for payload in payloads
        if isinstance(payload, dict) and payload.get("role") == "compiled-model"
    ]
    if len(matches) != 1:
        raise RuntimeError("self-test artifact must contain one compiled model")
    payload = matches[0]
    relative = payload.get("path")
    if not isinstance(relative, str):
        raise RuntimeError("self-test compiled-model payload has no path")
    path = Path(relative)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeError("self-test compiled-model payload path is unsafe")
    return payload, path


def _validate_source_compiled_model(
    source: Path,
    manifest: Mapping[str, object],
) -> None:
    """Require the source artifact to match the active model compiler exactly."""

    from pyamplicol.models.loading import load_compiled_model

    _payload, relative = _compiled_model_payload(manifest)
    load_compiled_model(source / relative)


def _normalize_compiled_model_version(
    artifact: Path,
    manifest: Mapping[str, object],
    *,
    version: str,
) -> None:
    """Retarget the embedded compiler fingerprint to the release package version."""

    payload, relative = _compiled_model_payload(manifest)
    path = artifact / relative
    try:
        compiled_model = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("self-test compiled model is invalid") from error
    if not isinstance(compiled_model, dict):
        raise RuntimeError("self-test compiled model must be an object")
    producer = compiled_model.get("producer")
    if not isinstance(producer, dict):
        raise RuntimeError("self-test compiled-model producer is invalid")
    producer["pyamplicol"] = version
    path.write_bytes(_canonical_json(compiled_model))
    payload["sha256"] = _sha256(path)
    payload["size_bytes"] = path.stat().st_size


def _workspace_version() -> str:
    with (ROOT / "Cargo.toml").open("rb") as stream:
        cargo = tomllib.load(stream)
    try:
        version = cargo["workspace"]["package"]["version"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("Cargo.toml has no workspace package version") from exc
    if not isinstance(version, str) or not version:
        raise RuntimeError("Cargo workspace package version is invalid")
    return version.replace("-dev.", ".dev")


def _walk_evaluator_manifests(value: object) -> Sequence[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if value.get("kind") == "symjit-application-evaluator":
            found.append(value)
        for child in value.values():
            found.extend(_walk_evaluator_manifests(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_walk_evaluator_manifests(child))
    return found


def _walk_compiled_execution_manifests(
    value: object,
) -> Sequence[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        compiled = value.get("compiled")
        if (
            isinstance(compiled, dict)
            and compiled.get("kind") == "generic-dag-stage-blueprint"
        ):
            found.append(value)
        for child in value.values():
            found.extend(_walk_compiled_execution_manifests(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_walk_compiled_execution_manifests(child))
    return found


def _portable_nonnegative_int(
    record: Mapping[str, object],
    name: str,
    *,
    context: str,
) -> int:
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"{context} has an invalid {name.replace('_', ' ')}")
    return value


def _portable_sequence(value: object, *, context: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise RuntimeError(f"{context} must be a list")
    return value


def _validate_portable_capability_owner(
    owner: Mapping[str, object],
    *,
    context: str,
) -> None:
    capabilities = _portable_sequence(
        owner.get("required_runtime_capabilities"),
        context=f"{context} runtime capabilities",
    )
    if any(
        not isinstance(capability, str) or not capability for capability in capabilities
    ) or len(capabilities) != len(set(capabilities)):
        raise RuntimeError(f"{context} runtime capabilities are invalid")
    for capability in (
        COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY,
        SYMJIT_F64_RUNTIME_CAPABILITY,
    ):
        if capability not in capabilities:
            raise RuntimeError(f"{context} must require {capability}")


def validate_portable_artifact_capabilities(
    manifest: Mapping[str, object],
    *,
    context: str,
) -> int:
    """Require Arena and SymJIT capabilities at artifact and process scope."""

    runtime = manifest.get("runtime")
    if not isinstance(runtime, Mapping):
        raise RuntimeError(f"{context} has no artifact runtime contract")
    _validate_portable_capability_owner(
        runtime,
        context=f"{context} artifact runtime",
    )

    processes = _portable_sequence(
        manifest.get("processes"),
        context=f"{context} processes",
    )
    if not processes:
        raise RuntimeError(f"{context} has no processes")
    for process_index, process in enumerate(processes):
        if not isinstance(process, Mapping):
            raise RuntimeError(f"{context} process {process_index} is invalid")
        _validate_portable_capability_owner(
            process,
            context=f"{context} process {process_index}",
        )
    return len(processes)


def _validate_portable_symjit_source(
    evaluator: Mapping[str, object],
) -> Mapping[str, object]:
    if (
        evaluator.get("compiler_type") != "native"
        or evaluator.get("translation_mode") != "indirect"
    ):
        raise RuntimeError(
            "portable self-test requires native indirect SymJIT evaluators"
        )
    if evaluator.get("optimization_level") != PORTABLE_OPTIMIZATION_LEVEL:
        raise RuntimeError(
            "portable self-test source must use its canonical SymJIT "
            f"optimization level {PORTABLE_OPTIMIZATION_LEVEL}"
        )
    if (
        evaluator.get("runtime_capability") != SYMJIT_F64_RUNTIME_CAPABILITY
        or evaluator.get("application_abi") != SYMJIT_APPLICATION_ABI
        or evaluator.get("element_layout") != "complex-f64"
        or evaluator.get("batch_layout") != "row-major"
    ):
        raise RuntimeError("portable self-test has an incompatible SymJIT source ABI")
    application_path = evaluator.get("application_path")
    if not isinstance(application_path, str) or not application_path:
        raise RuntimeError("portable self-test has no ordinary SymJIT application path")
    input_len = _portable_nonnegative_int(
        evaluator,
        "input_len",
        context="portable self-test SymJIT evaluator",
    )
    output_len = _portable_nonnegative_int(
        evaluator,
        "output_len",
        context="portable self-test SymJIT evaluator",
    )
    plane = evaluator.get("plane_application")
    if not isinstance(plane, Mapping):
        raise RuntimeError(
            "portable self-test SymJIT evaluator has no plane application"
        )
    expected_plane = {
        "application_abi": SYMJIT_PLANE_APPLICATION_ABI,
        "storage_abi": SYMJIT_APPLICATION_ABI,
        "element_layout": "split-complex-plane-major",
        "descriptor_order": "inputs-re-im-then-outputs-re-im",
        "input_complex_count": input_len,
        "output_complex_count": output_len,
        "input_plane_count": 2 * input_len,
        "output_plane_count": 2 * output_len,
        "compiler_type": "native",
        "translation_mode": "symbolica-structured-instructions",
        "optimization_level": PORTABLE_OPTIMIZATION_LEVEL,
        "simd": True,
        "complex": True,
        "fast_math": True,
        "fast_complex": False,
        "threading": False,
        "direct_arena": True,
    }
    if any(plane.get(name) != value for name, value in expected_plane.items()):
        raise RuntimeError(
            "portable self-test SymJIT plane application contract is incompatible"
        )
    plane_path = plane.get("application_path")
    if not isinstance(plane_path, str) or not plane_path:
        raise RuntimeError(
            "portable self-test SymJIT plane application path is invalid"
        )
    source_digest = plane.get("source_digest")
    if (
        not isinstance(source_digest, str)
        or len(source_digest) != 64
        or any(character not in "0123456789abcdef" for character in source_digest)
    ):
        raise RuntimeError("portable self-test SymJIT plane source digest is invalid")
    if not isinstance(plane.get("compression"), bool):
        raise RuntimeError(
            "portable self-test SymJIT plane compression flag is invalid"
        )
    target = plane.get("target")
    if (
        not isinstance(target, Mapping)
        or target.get("word_bits") != 64
        or target.get("endianness") != "little"
    ):
        raise RuntimeError("portable self-test SymJIT plane target is incompatible")
    return plane


def _portable_arena_source_leaves(
    evaluator: Mapping[str, object],
    parent_inputs: tuple[int, ...],
    output_start: int,
    *,
    context: str,
) -> tuple[list[dict[str, object]], int]:
    kind = evaluator.get("kind")
    if kind == "chunked-symbolica-evaluator":
        input_len = evaluator.get("input_len")
        if input_len is not None and input_len != len(parent_inputs):
            raise RuntimeError(f"{context} chunk input width is inconsistent")
        chunks = _portable_sequence(
            evaluator.get("chunks"),
            context=f"{context} chunks",
        )
        if not chunks:
            raise RuntimeError(f"{context} has no evaluator leaves")
        raw_maps = evaluator.get("chunk_input_indices")
        if raw_maps is None:
            input_maps = [parent_inputs] * len(chunks)
        else:
            maps = _portable_sequence(
                raw_maps,
                context=f"{context} chunk input maps",
            )
            if len(maps) != len(chunks):
                raise RuntimeError(f"{context} chunk input maps are incomplete")
            input_maps: list[tuple[int, ...]] = []
            for raw_map in maps:
                indices = _portable_sequence(
                    raw_map,
                    context=f"{context} chunk input map",
                )
                mapped: list[int] = []
                for raw_index in indices:
                    if (
                        isinstance(raw_index, bool)
                        or not isinstance(raw_index, int)
                        or not 0 <= raw_index < len(parent_inputs)
                    ):
                        raise RuntimeError(
                            f"{context} chunk input map references an absent input"
                        )
                    mapped.append(parent_inputs[raw_index])
                input_maps.append(tuple(mapped))

        result: list[dict[str, object]] = []
        cursor = output_start
        for chunk_index, (raw_chunk, mapped) in enumerate(
            zip(chunks, input_maps, strict=True)
        ):
            if not isinstance(raw_chunk, Mapping):
                raise RuntimeError(f"{context} evaluator child is invalid")
            child, cursor = _portable_arena_source_leaves(
                raw_chunk,
                mapped,
                cursor,
                context=f"{context} chunk {chunk_index}",
            )
            result.extend(child)
        return result, cursor

    if kind != "symjit-application-evaluator":
        raise RuntimeError(f"{context} is not an all-SymJIT compiled Plane-Arena stage")
    plane = _validate_portable_symjit_source(evaluator)
    application_path = plane.get("application_path")
    if not isinstance(application_path, str) or not application_path:
        raise RuntimeError(f"{context} has no SymJIT plane application path")
    input_len = _portable_nonnegative_int(evaluator, "input_len", context=context)
    output_len = _portable_nonnegative_int(evaluator, "output_len", context=context)
    if input_len != len(parent_inputs) or output_len == 0:
        raise RuntimeError(f"{context} has inconsistent evaluator dimensions")
    output_stop = output_start + output_len
    return (
        [
            {
                "application_path": application_path,
                "source_application_abi": SYMJIT_PLANE_APPLICATION_ABI,
                "optimization_level": plane["optimization_level"],
                "direct_codegen_optimization_level": plane["optimization_level"],
                "input_len": input_len,
                "output_len": output_len,
                "input_indices": list(parent_inputs),
                "output_start": output_start,
                "output_stop": output_stop,
            }
        ],
        output_stop,
    )


def _portable_stage_input_bindings(
    stage: Mapping[str, object],
    parameter_count: int,
    *,
    context: str,
) -> list[dict[str, object]]:
    inputs = _portable_sequence(
        stage.get("input_components"),
        context=f"{context} input components",
    )
    expected: list[dict[str, object] | None] = [None] * parameter_count
    for raw_input in inputs:
        if not isinstance(raw_input, Mapping):
            raise RuntimeError(f"{context} input component is invalid")
        parameter_index = _portable_nonnegative_int(
            raw_input,
            "parameter_index",
            context=f"{context} input component",
        )
        if (
            parameter_index >= parameter_count
            or expected[parameter_index] is not None
            or raw_input.get("kind") not in {"value", "momentum", "model_parameter"}
            or not isinstance(raw_input.get("real_valued"), bool)
        ):
            raise RuntimeError(f"{context} input components are inconsistent")
        binding = {
            "parameter_index": parameter_index,
            "kind": raw_input["kind"],
            "source_id": _portable_nonnegative_int(
                raw_input,
                "source_id",
                context=f"{context} input component",
            ),
            "component": _portable_nonnegative_int(
                raw_input,
                "component",
                context=f"{context} input component",
            ),
            "global_component": _portable_nonnegative_int(
                raw_input,
                "global_component",
                context=f"{context} input component",
            ),
            "real_valued": raw_input["real_valued"],
        }
        expected[parameter_index] = binding
    if any(binding is None for binding in expected):
        raise RuntimeError(f"{context} input components are incomplete")
    return [binding for binding in expected if binding is not None]


def _portable_stage_output_bindings(
    stage: Mapping[str, object],
    output_length: int,
    *,
    arena: str,
    context: str,
) -> list[dict[str, object]]:
    slots = _portable_sequence(
        stage.get("output_slots"),
        context=f"{context} output slots",
    )
    expected: list[dict[str, object] | None] = [None] * output_length
    seen_components: set[int] = set()
    for raw_slot in slots:
        if not isinstance(raw_slot, Mapping):
            raise RuntimeError(f"{context} output slot is invalid")
        output_start = _portable_nonnegative_int(
            raw_slot,
            "output_start",
            context=f"{context} output slot",
        )
        output_stop = _portable_nonnegative_int(
            raw_slot,
            "output_stop",
            context=f"{context} output slot",
        )
        component_start = _portable_nonnegative_int(
            raw_slot,
            "component_start",
            context=f"{context} output slot",
        )
        component_stop = _portable_nonnegative_int(
            raw_slot,
            "component_stop",
            context=f"{context} output slot",
        )
        if (
            output_stop < output_start
            or output_stop > output_length
            or component_stop - component_start != output_stop - output_start
        ):
            raise RuntimeError(f"{context} output slot range is inconsistent")
        for offset in range(output_stop - output_start):
            output_index = output_start + offset
            component = component_start + offset
            if expected[output_index] is not None or component in seen_components:
                raise RuntimeError(f"{context} output slots alias")
            expected[output_index] = {
                "output_index": output_index,
                "arena": arena,
                "component": component,
            }
            seen_components.add(component)
    if any(binding is None for binding in expected):
        raise RuntimeError(f"{context} output slots are incomplete")
    return [binding for binding in expected if binding is not None]


def _validate_portable_arena_stage(
    stage: Mapping[str, object],
    *,
    arena: str,
    context: str,
) -> None:
    if stage.get("parameter_layout") != "stage-local-value-momentum":
        raise RuntimeError(f"{context} does not use stage-local parameters")
    parameter_count = _portable_nonnegative_int(
        stage,
        "parameter_count",
        context=context,
    )
    output_length = _portable_nonnegative_int(
        stage,
        "output_length",
        context=context,
    )
    if output_length == 0:
        raise RuntimeError(f"{context} has no outputs")
    evaluator = stage.get("evaluator")
    if not isinstance(evaluator, Mapping):
        raise RuntimeError(f"{context} has no compiled evaluator")
    descriptor = stage.get("compiled_plane_arena")
    if not isinstance(descriptor, Mapping):
        raise RuntimeError(
            f"{context} requires a complete compiled_plane_arena descriptor"
        )

    contract = {
        "schema_version": 1,
        "kind": "compiled-plane-arena-stage",
        "application_abi": COMPILED_PLANE_DIRECT_APPLICATION_ABI,
        "source_application_abi": SYMJIT_PLANE_APPLICATION_ABI,
        "element_layout": "split-complex-component-major",
        "output_operation": "overwrite",
        "output_factor": "identity",
        "input_output_aliasing": "forbidden",
        "output_output_aliasing": "forbidden",
    }
    if any(descriptor.get(name) != value for name, value in contract.items()):
        raise RuntimeError(f"{context} compiled_plane_arena contract is incompatible")

    expected_inputs = _portable_stage_input_bindings(
        stage,
        parameter_count,
        context=context,
    )
    input_bindings = _portable_sequence(
        descriptor.get("input_bindings"),
        context=f"{context} compiled_plane_arena input bindings",
    )
    if list(input_bindings) != expected_inputs:
        raise RuntimeError(
            f"{context} compiled_plane_arena input bindings are incomplete"
        )

    expected_outputs = _portable_stage_output_bindings(
        stage,
        output_length,
        arena=arena,
        context=context,
    )
    output_bindings = _portable_sequence(
        descriptor.get("output_bindings"),
        context=f"{context} compiled_plane_arena output bindings",
    )
    if list(output_bindings) != expected_outputs:
        raise RuntimeError(
            f"{context} compiled_plane_arena output bindings are incomplete"
        )

    expected_leaves, output_stop = _portable_arena_source_leaves(
        evaluator,
        tuple(range(parameter_count)),
        0,
        context=f"{context} evaluator",
    )
    if output_stop != output_length:
        raise RuntimeError(f"{context} evaluator leaves do not cover its outputs")
    leaves = _portable_sequence(
        descriptor.get("leaves"),
        context=f"{context} compiled_plane_arena leaves",
    )
    if len(leaves) != len(expected_leaves):
        raise RuntimeError(f"{context} compiled_plane_arena leaves are incomplete")
    for leaf_index, (raw_leaf, expected_leaf) in enumerate(
        zip(leaves, expected_leaves, strict=True)
    ):
        if not isinstance(raw_leaf, Mapping):
            raise RuntimeError(
                f"{context} compiled_plane_arena leaf {leaf_index} is invalid"
            )
        if raw_leaf.get("direct_codegen_optimization_level") != expected_leaf.get(
            "direct_codegen_optimization_level"
        ):
            raise RuntimeError(
                f"{context} compiled_plane_arena direct codegen optimization "
                "does not match its SymJIT plane application"
            )
        if any(raw_leaf.get(name) != value for name, value in expected_leaf.items()):
            raise RuntimeError(
                f"{context} compiled_plane_arena leaf {leaf_index} "
                "does not bind its SymJIT source"
            )


def _validate_portable_compiled_execution(
    execution: Mapping[str, object],
    *,
    context: str,
) -> None:
    _validate_portable_capability_owner(execution, context=context)
    compiled = execution.get("compiled")
    if not isinstance(compiled, Mapping):
        raise RuntimeError(f"{context} has no compiled DAG")
    stage_set = compiled.get("stage_evaluators")
    if not isinstance(stage_set, Mapping):
        raise RuntimeError(f"{context} has no compiled DAG stage evaluators")
    if stage_set.get("kind") != "generic-dag-stage-evaluator-artifacts":
        raise RuntimeError(f"{context} stage evaluator contract is invalid")
    _validate_portable_capability_owner(
        stage_set,
        context=f"{context} stage evaluators",
    )
    stages = _portable_sequence(
        stage_set.get("stages"),
        context=f"{context} stages",
    )
    stage_count = _portable_nonnegative_int(
        stage_set,
        "stage_count",
        context=f"{context} stage evaluator set",
    )
    if stage_count != len(stages) + 1:
        raise RuntimeError(f"{context} stage evaluator set is incomplete")
    for stage_index, stage in enumerate(stages):
        if not isinstance(stage, Mapping):
            raise RuntimeError(f"{context} stage {stage_index} is invalid")
        _validate_portable_arena_stage(
            stage,
            arena="current",
            context=f"{context} stage {stage_index}",
        )
    amplitude_stage = stage_set.get("amplitude_stage")
    if not isinstance(amplitude_stage, Mapping):
        raise RuntimeError(f"{context} amplitude stage is absent")
    _validate_portable_arena_stage(
        amplitude_stage,
        arena="amplitude",
        context=f"{context} amplitude stage",
    )


def validate_portable_execution_manifest(
    execution: Mapping[str, object],
    *,
    context: str,
) -> int:
    """Validate one root compiled execution and every nested selector execution.

    This is the shared release-facing contract used while preparing, staging,
    and auditing the installed self-test fixture. It deliberately validates the
    portable O2 ordinary SymJIT fallback together with the complete O2
    P-kernel Plane-Arena descriptor rather than treating either representation
    as sufficient alone.
    """

    evaluator_count = 0
    for evaluator in _walk_evaluator_manifests(execution):
        evaluator_count += 1
        _validate_portable_symjit_source(evaluator)
    if evaluator_count == 0:
        raise RuntimeError(f"{context} has no SymJIT evaluator manifests")

    compiled_executions = _walk_compiled_execution_manifests(execution)
    if not compiled_executions or compiled_executions[0] is not execution:
        raise RuntimeError(f"{context} has no root compiled DAG")
    for execution_index, compiled_execution in enumerate(compiled_executions):
        _validate_portable_compiled_execution(
            compiled_execution,
            context=f"{context} compiled execution {execution_index}",
        )
    return evaluator_count


def _validate_portable_evaluator_configuration(
    artifact: Path,
    manifest: Mapping[str, object],
) -> int:
    """Require architecture-neutral MIR before retargeting an evaluator state."""

    payloads = manifest.get("payloads")
    if not isinstance(payloads, list):
        raise RuntimeError("artifact payload inventory is invalid")
    evaluator_count = 0
    for payload in payloads:
        if (
            not isinstance(payload, dict)
            or payload.get("role") != "evaluator-manifest"
            or not str(payload.get("path", "")).endswith("/execution.json")
        ):
            continue
        relative = payload.get("path")
        if not isinstance(relative, str):
            raise RuntimeError("self-test evaluator manifest has no path")
        execution = json.loads((artifact / relative).read_text(encoding="utf-8"))
        if not isinstance(execution, dict):
            raise RuntimeError("self-test evaluator manifest must be an object")
        evaluator_count += validate_portable_execution_manifest(
            execution,
            context=f"portable self-test evaluator manifest {relative}",
        )
    if evaluator_count == 0:
        raise RuntimeError("self-test artifact has no SymJIT evaluator manifests")
    return PORTABLE_OPTIMIZATION_LEVEL


def _strip_symbolica_fallbacks(artifact: Path, manifest: dict[str, Any]) -> None:
    removed: set[str] = set()
    payloads = manifest.get("payloads")
    if not isinstance(payloads, list):
        raise RuntimeError("artifact payload inventory is invalid")
    execution_payloads = [
        payload
        for payload in payloads
        if isinstance(payload, dict) and payload.get("role") == "evaluator-manifest"
    ]
    for payload in execution_payloads:
        relative = payload.get("path")
        if not isinstance(relative, str) or not relative.endswith("/execution.json"):
            continue
        path = artifact / relative
        execution = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(execution, dict):
            raise RuntimeError(f"invalid evaluator manifest: {relative}")
        for evaluator in _walk_evaluator_manifests(execution):
            state = evaluator.get("evaluator_state_path")
            if isinstance(state, str):
                removed.add((Path(relative).parent / state).as_posix())
            evaluator["evaluator_state_path"] = None
            evaluator["evaluator_state_runtime_capability"] = None
        path.write_bytes(_canonical_json(execution))
        payload["sha256"] = _sha256(path)
        payload["size_bytes"] = path.stat().st_size
    if not removed:
        raise RuntimeError("source artifact has no Symbolica fallback evaluator states")
    removed_loose: set[str] = set()
    for relative in sorted(removed):
        path = artifact / relative
        if path.is_file():
            path.unlink()
            removed_loose.add(relative)
    removed_packed = removed - removed_loose
    if removed_packed:
        _strip_packed_symbolica_fallbacks(artifact, manifest, removed_packed)
    manifest["payloads"] = [
        payload
        for payload in payloads
        if not (
            isinstance(payload, dict)
            and isinstance(payload.get("path"), str)
            and payload["path"] in removed_loose
        )
    ]


def _strip_packed_symbolica_fallbacks(
    artifact: Path,
    manifest: dict[str, Any],
    removed: set[str],
) -> None:
    """Remove exact fallback states from a packed evaluator container."""

    from pyamplicol.generation.evaluator_container import (
        PacbinMemberKind,
        PacbinMemberSource,
        PacbinReader,
        write_pacbin_atomic,
    )

    extensions = manifest.get("extensions")
    if not isinstance(extensions, dict):
        raise RuntimeError("artifact extensions are invalid")
    extension = extensions.get("evaluator_payload_container")
    if not isinstance(extension, dict):
        missing = ", ".join(sorted(removed))
        raise RuntimeError(f"missing Symbolica fallback state: {missing}")
    relative = extension.get("path")
    if not isinstance(relative, str):
        raise RuntimeError("evaluator payload container has no path")
    container_path = artifact / relative

    with PacbinReader.open(container_path, verify_payloads=True) as reader:
        by_path = {member.logical_path: member for member in reader.members}
        missing = sorted(removed - by_path.keys())
        if missing:
            raise RuntimeError(
                "missing Symbolica fallback state: " + ", ".join(missing)
            )
        invalid = sorted(
            path
            for path in removed
            if by_path[path].kind is not PacbinMemberKind.SYMBOLICA_EXACT_STATE
        )
        if invalid:
            raise RuntimeError(
                "Symbolica fallback state has an invalid packed kind: "
                + ", ".join(invalid)
            )
        retained = tuple(
            PacbinMemberSource(
                member.logical_path,
                member.kind,
                reader.open_member_stream(member.logical_path),
            )
            for member in reader.members
            if member.logical_path not in removed
        )
        if not retained:
            raise RuntimeError("portable self-test evaluator container is empty")
        index = write_pacbin_atomic(container_path, retained)

    extension["member_count"] = len(index.members)
    extension["unpacked_size_bytes"] = sum(member.length for member in index.members)
    extension["index_sha256"] = index.index_sha256

    payloads = manifest.get("payloads")
    if not isinstance(payloads, list):
        raise RuntimeError("artifact payload inventory is invalid")
    matches = [
        payload
        for payload in payloads
        if isinstance(payload, dict) and payload.get("path") == relative
    ]
    if len(matches) != 1:
        raise RuntimeError("artifact must declare one evaluator payload container")
    payload = matches[0]
    payload["sha256"] = _sha256(container_path)
    payload["size_bytes"] = container_path.stat().st_size


def _sanitize_configuration_paths(artifact: Path, manifest: dict[str, Any]) -> None:
    payloads = manifest.get("payloads")
    if not isinstance(payloads, list):
        raise RuntimeError("artifact payload inventory is invalid")
    roles = {"configuration-requested", "configuration-effective"}
    found: set[str] = set()
    for payload in payloads:
        if not isinstance(payload, dict) or payload.get("role") not in roles:
            continue
        role = str(payload["role"])
        relative = payload.get("path")
        if not isinstance(relative, str):
            raise RuntimeError(f"self-test {role} payload has no path")
        path = artifact / relative
        text = path.read_text(encoding="utf-8")
        updated = _sanitize_generation_output(text, role)
        path.write_text(updated, encoding="utf-8")
        payload["sha256"] = _sha256(path)
        payload["size_bytes"] = path.stat().st_size
        found.add(role)
    if found != roles:
        raise RuntimeError(
            "self-test artifact lacks requested/effective configurations"
        )


def _sanitize_generation_output(text: str, role: str) -> str:
    pattern = r'(?m)^output\s*=\s*"[^"]*"$'
    matches = tuple(re.finditer(pattern, text))
    if len(matches) > 1:
        raise RuntimeError(f"self-test {role} has multiple generation outputs")
    if not matches:
        return text
    return re.sub(pattern, 'output = "."', text, count=1)


def _retarget_portable_manifest(
    manifest: dict[str, Any],
    *,
    source_target: str,
) -> None:
    producer = manifest.get("producer")
    if not isinstance(producer, dict):
        raise RuntimeError("self-test artifact producer metadata is invalid")
    target = producer.get("target")
    if not isinstance(target, dict) or target.get("triple") != source_target:
        raise RuntimeError("self-test artifact target differs from the active runtime")
    target["triple"] = PORTABLE_TEMPLATE
    target["cpu_features"] = []

    payloads = manifest.get("payloads")
    if not isinstance(payloads, list):
        raise RuntimeError("self-test artifact payload inventory is invalid")
    evaluator_targets = 0
    for payload in payloads:
        if not isinstance(payload, dict):
            raise RuntimeError("self-test artifact payload entry is invalid")
        payload_target = payload.get("target")
        if payload_target is None:
            continue
        if (
            not isinstance(payload_target, dict)
            or payload_target.get("triple") != source_target
        ):
            raise RuntimeError("self-test evaluator target metadata is invalid")
        payload_target["triple"] = PORTABLE_TEMPLATE
        payload_target["cpu_features"] = []
        evaluator_targets += 1
    if evaluator_targets == 0:
        raise RuntimeError("self-test artifact has no target-tagged evaluator state")


def _complex_pairs(values: Sequence[object]) -> list[list[float]]:
    return [
        [float(complex(value).real), float(complex(value).imag)] for value in values
    ]


def _complex_sequences_close(
    left: Sequence[object],
    right: Sequence[object],
) -> bool:
    if len(left) != len(right):
        return False
    for left_value, right_value in zip(left, right, strict=True):
        actual = complex(left_value)
        expected = complex(right_value)
        if not (
            math.isclose(actual.real, expected.real, rel_tol=1.0e-12, abs_tol=1.0e-14)
            and math.isclose(
                actual.imag, expected.imag, rel_tol=1.0e-12, abs_tol=1.0e-14
            )
        ):
            return False
    return True


def prepare(source: Path, destination: Path | None = None) -> Path:
    source = source.expanduser().resolve(strict=True)
    if not source.is_dir() or not (source / "artifact.json").is_file():
        raise RuntimeError("self-test source must be a schema-v3 artifact directory")

    try:
        source_manifest = json.loads(
            (source / "artifact.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("self-test source manifest is invalid") from error
    if (
        not isinstance(source_manifest, dict)
        or source_manifest.get("schema_version") != 3
    ):
        raise RuntimeError("self-test source is not a schema-v3 artifact")
    _validate_source_compiled_model(source, source_manifest)

    from pyamplicol import Runtime

    native = importlib.import_module("pyamplicol._rusticol")
    active_target = str(native.target_info().triple)
    if active_target not in COMPATIBLE_TARGETS:
        raise RuntimeError(f"unsupported portable self-test target: {active_target}")
    producer = source_manifest.get("producer")
    producer_target = producer.get("target") if isinstance(producer, dict) else None
    raw_source_target = (
        producer_target.get("triple") if isinstance(producer_target, dict) else None
    )
    if not isinstance(raw_source_target, str) or raw_source_target not in {
        active_target,
        PORTABLE_TEMPLATE,
    }:
        raise RuntimeError(
            "self-test artifact target differs from the active or portable runtime"
        )
    source_target = raw_source_target
    runtime = Runtime.load(source, mute_warnings=True)
    momentum_loader = getattr(runtime._backend, "validation_momenta", None)
    if not callable(momentum_loader):
        raise RuntimeError("artifact runtime cannot expose its validation point")
    momenta = momentum_loader()
    if momenta is None:
        raise RuntimeError("artifact has no deterministic validation point")
    total = runtime.evaluate(momenta)
    resolved = runtime.evaluate_resolved(momenta)
    if not _complex_sequences_close(resolved.total(), total):
        raise RuntimeError("resolved self-test values do not reproduce evaluate()")

    output = (destination or (FIXTURE_ROOT / PORTABLE_TEMPLATE)).expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"self-test destination already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.staging")
    if temporary.exists():
        raise RuntimeError(f"stale self-test staging directory exists: {temporary}")
    try:
        artifact = temporary / "artifact"
        shutil.copytree(source, artifact, symlinks=False)
        manifest_path = artifact / "artifact.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 3:
            raise RuntimeError("self-test source is not a schema-v3 artifact")
        validate_artifact_identity_contract(
            manifest,
            context="self-test source",
        )
        producer = manifest.get("producer")
        runtime_metadata = manifest.get("runtime")
        if not isinstance(producer, dict) or not isinstance(runtime_metadata, dict):
            raise RuntimeError(
                "self-test artifact producer/runtime metadata is invalid"
            )
        if producer.get("target", {}).get("triple") != source_target:
            raise RuntimeError(
                "self-test artifact target changed while preparing the fixture"
            )
        validate_portable_artifact_capabilities(
            manifest,
            context="portable self-test artifact",
        )
        portable_optimization_level = _validate_portable_evaluator_configuration(
            artifact,
            manifest,
        )
        _strip_symbolica_fallbacks(artifact, manifest)
        _sanitize_configuration_paths(artifact, manifest)
        _retarget_portable_manifest(manifest, source_target=source_target)
        version = _workspace_version()
        _normalize_compiled_model_version(
            artifact,
            manifest,
            version=version,
        )
        producer["version"] = version
        runtime_metadata["engine_version"] = version
        manifest["artifact_id"] = _artifact_id(manifest)
        manifest_path.write_bytes(_canonical_json(manifest))
        expected = {
            "schema_version": 1,
            "artifact_path": "artifact",
            "target": PORTABLE_TEMPLATE,
            "compatible_targets": list(COMPATIBLE_TARGETS),
            "source_generation_target": source_target,
            "serialization": {
                "kind": "symjit-application-mir-v3",
                "endianness": "little",
                "word_size_bits": 64,
                "load_behavior": "recompile-mir-for-loading-host",
                "source_optimization_level": portable_optimization_level,
            },
            "process_id": runtime.physics.process_id,
            "process": runtime.physics.process,
            "momenta": momenta,
            "resolved_shape": list(resolved.shape),
            "helicity_ids": list(resolved.helicity_ids),
            "color_ids": list(resolved.color_ids),
            "total": _complex_pairs(total),
        }
        (temporary / "expected.json").write_text(
            json.dumps(expected, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.rename(output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args(argv)
    output = prepare(args.source, args.destination)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

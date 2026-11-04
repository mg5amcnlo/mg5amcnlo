# SPDX-License-Identifier: 0BSD
"""Compilation and persistence of stage evaluator artifacts."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .._internal.physics.parameters import ParamBuilder
from .._internal.physics.symbols import symbols
from .._internal.versions import (
    COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY,
    COMPILED_PLANE_DIRECT_APPLICATION_ABI,
    COMPILED_RUNTIME_SELECTORS_CAPABILITY,
    NATIVE_COMPILED_DIRECT_APPLICATION_ABI,
    SYMBOLICA_ASM_RUNTIME_CAPABILITY,
    SYMBOLICA_CPP_RUNTIME_CAPABILITY,
    SYMJIT_APPLICATION_ABI,
    SYMJIT_F64_RUNTIME_CAPABILITY,
    SYMJIT_PLANE_APPLICATION_ABI,
)
from ..evaluators.execution_schema import (
    aggregate_runtime_capabilities,
    evaluator_runtime_capabilities,
)
from ..models.base import Model
from .contracts import StageCompilationInput
from .dag_types import GenericDAG
from .stage_parameters import _dict, _list, _logical_model_parameter_symbols
from .stage_planning import (
    _prepare_stage_for_output_chunking,
    build_generic_stage_compiler_blueprint,
)
from .stage_settings import _stage_symbolica_settings
from .stage_types import (
    GenericCompiledStageBlueprint,
    GenericStageCompilerBlueprint,
    StageBlueprintProgress,
    StageEvaluatorCompiler,
)

_EXPRESSION_PREVIEW_LIMIT = 512


def write_generic_stage_evaluator_artifacts(
    blueprint: GenericStageCompilerBlueprint,
    artifact_dir: str | Path,
    *,
    compiler: StageEvaluatorCompiler | None = None,
    symbolica_settings: Any | None = None,
    merge_evaluators_strategy: bool = False,
    verbose_evaluator_build: bool = False,
    jit_compile: bool = True,
    progress_callback: Any | None = None,
) -> dict[str, object]:
    """Serialize evaluator artifacts for a schema-v3 stage blueprint.

    The function is intentionally opt-in. Normal schema-v3 process manifest
    generation remains cheap, while this path is the bridge from the
    process-generic current DAG to concrete Symbolica evaluator artifacts.
    The native runtime can validate, load, and execute the resulting metadata
    through its generic staged runtime.
    """

    if not blueprint.expression_ready:
        raise ValueError(
            "cannot write generic evaluator artifacts with lowering blockers: "
            + "; ".join(blueprint.blockers)
        )

    output_dir = Path(artifact_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    build_started = time.perf_counter()
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "stage compile",
                "item": "start",
                "total": blueprint.stage_count,
            }
        )

    def compile_stage(stage: GenericCompiledStageBlueprint) -> dict[str, object]:
        return _compile_stage_evaluator_artifact(
            stage,
            output_dir,
            compiler=compiler,
            blueprint=blueprint,
            symbolica_settings=symbolica_settings,
            merge_evaluators_strategy=merge_evaluators_strategy,
            verbose_evaluator_build=verbose_evaluator_build,
            jit_compile=jit_compile,
            progress_callback=progress_callback,
        )

    stage_payloads = []
    stage_timings: list[dict[str, object]] = []
    for stage in blueprint.stages:
        prepared_stage = _prepare_stage_for_output_chunking(
            stage,
            blueprint=blueprint,
            symbolica_settings=symbolica_settings,
        )
        payload = prepared_stage.to_json_dict()
        payload["evaluator"] = compile_stage(prepared_stage)
        direct = _compiled_plane_arena_stage(payload)
        if direct is not None:
            payload["compiled_plane_arena"] = direct
        stage_timings.append(
            _stage_build_timing_record(
                prepared_stage.evaluator_label,
                payload["evaluator"],
            )
        )
        stage_payloads.append(payload)
        if progress_callback is not None:
            timing = stage_timings[-1]
            progress_callback(
                {
                    "stage": "stage complete",
                    "item": stage.evaluator_label,
                    "increment": 1,
                    "total": blueprint.stage_count,
                    "duration_s": timing["stage_evaluator_build_s"],
                }
            )

    prepared_amplitude_stage = _prepare_stage_for_output_chunking(
        blueprint.amplitude_stage,
        blueprint=blueprint,
        symbolica_settings=symbolica_settings,
    )
    amplitude_payload = prepared_amplitude_stage.to_json_dict()
    amplitude_payload["evaluator"] = compile_stage(prepared_amplitude_stage)
    direct = _compiled_plane_arena_stage(amplitude_payload)
    if direct is not None:
        amplitude_payload["compiled_plane_arena"] = direct
    stage_timings.append(
        _stage_build_timing_record(
            blueprint.amplitude_stage.evaluator_label,
            amplitude_payload["evaluator"],
        )
    )
    if progress_callback is not None:
        timing = stage_timings[-1]
        progress_callback(
            {
                "stage": "stage complete",
                "item": blueprint.amplitude_stage.evaluator_label,
                "increment": 1,
                "total": blueprint.stage_count,
                "duration_s": timing["stage_evaluator_build_s"],
            }
        )

    return _finalize_stage_evaluator_payload(
        blueprint,
        stage_payloads=stage_payloads,
        amplitude_payload=amplitude_payload,
        stage_timings=stage_timings,
        build_started=build_started,
    )


def write_model_parameter_evaluator_artifact(
    model: Model,
    runtime_schema: Mapping[str, object],
    artifact_dir: str | Path,
    *,
    symbolica_settings: Any | None = None,
    jit_compile: bool = True,
    progress_callback: Any | None = None,
) -> dict[str, object] | None:
    schema = _dict(runtime_schema)
    records = tuple(
        sorted(
            (_dict(item) for item in _list(schema.get("model_parameters", []))),
            key=lambda item: int(item["parameter_index"]),
        )
    )
    input_records = tuple(
        record
        for record in records
        if str(record.get("kind"))
        in {"external_parameter", "external_parameter_component"}
    )
    derived_components: dict[str, dict[str, int]] = {}
    for record in records:
        if str(record.get("kind")) != "derived_parameter_component":
            continue
        runtime_name = record.get("runtime_name")
        component = record.get("complex_component")
        if isinstance(runtime_name, str) and component in {"real", "imag"}:
            derived_components.setdefault(runtime_name, {})[str(component)] = int(
                record["parameter_index"]
            )
    requested_output_names = tuple(
        name
        for name, components in sorted(
            derived_components.items(),
            key=lambda item: min(item[1].values()),
        )
        if set(components) == {"real", "imag"}
    )
    if not requested_output_names:
        return None

    definitions_provider = getattr(
        model,
        "runtime_derived_parameter_definitions",
        None,
    )
    if not callable(definitions_provider):
        return None
    definitions_subset_provider = getattr(
        model,
        "runtime_derived_parameter_definitions_for",
        None,
    )
    definition_values = (
        definitions_subset_provider(requested_output_names)
        if callable(definitions_subset_provider)
        else definitions_provider()
    )
    definitions = {
        str(name): str(expression)
        for name, expression in definition_values.items()
        if str(name) in requested_output_names
    }
    output_names = tuple(name for name in requested_output_names if name in definitions)
    if not output_names:
        return None

    # Planning imports this module without compiling model parameters.
    from ..models.compiler_records import _replace_evaluator_constants

    builder = ParamBuilder()
    model_symbols = symbols.model(getattr(model, "name", "unnamed-model"))
    parameter_symbols = tuple(
        builder.add_parameter_list(
            ("artifact_schema_v3", "external_model_parameters"),
            len(input_records),
            role="generic_external_model_parameters",
            real_valued=True,
        )
    )
    slot_symbols = {
        str(record["name"]): parameter_symbols[index]
        for index, record in enumerate(input_records)
    }
    logical_symbols = _logical_model_parameter_symbols(
        input_records,
        slot_symbols,
    )
    outputs = []
    for name in output_names:
        expression = model_symbols.expression(definitions[name])
        for parameter_name, symbol in logical_symbols.items():
            expression = expression.replace(
                model_symbols.symbol(parameter_name),
                symbol,
            )
        outputs.append(expression)

    stage = GenericCompiledStageBlueprint(
        stage_index=0,
        stage_kind="model-parameter-derivation",
        subset_size=None,
        evaluator_label="generic_model_parameter_derivation",
        parameter_layout="external-model-parameters",
        output_length=len(outputs),
        output_slots=(),
        input_value_slot_ids=(),
        output_value_slot_ids=(),
        interaction_ids=(),
        input_components=(),
        parameter_count=len(parameter_symbols),
        value_parameter_count=0,
        momentum_parameter_count=0,
        model_parameter_count=len(parameter_symbols),
        real_valued_inputs=tuple(range(len(parameter_symbols))),
        expression_ready=True,
        blockers=(),
        first_output_previews=tuple(
            expression.to_canonical_string()[:_EXPRESSION_PREVIEW_LIMIT]
            for expression in outputs[:3]
        ),
        parameter_symbols=parameter_symbols,
        output_expressions=tuple(outputs),
    )
    parameter_evaluator_settings = (
        None
        if symbolica_settings is None
        else replace(
            symbolica_settings,
            compiled_output_chunk_size=None,
            output_chunk_strategy="uniform",
        )
    )
    evaluator = _compile_stage_evaluator_artifact(
        # Native evaluators use f64 constants, whereas the retained compiled
        # model keeps the exact definitions used by the high-precision path.
        replace(
            stage,
            output_expressions=tuple(
                _replace_evaluator_constants(expression) for expression in outputs
            ),
        ),
        Path(artifact_dir).expanduser(),
        compiler=None,
        blueprint=None,
        symbolica_settings=parameter_evaluator_settings,
        merge_evaluators_strategy=False,
        verbose_evaluator_build=False,
        jit_compile=jit_compile,
        progress_callback=progress_callback,
    )
    return {
        "kind": "generic-model-parameter-evaluator",
        "required_runtime_capabilities": list(
            evaluator_runtime_capabilities(evaluator)
        ),
        "input_parameter_indices": [
            int(record["parameter_index"]) for record in input_records
        ],
        "outputs": [
            {
                "runtime_name": name,
                "output_index": output_index,
                "real_parameter_index": derived_components[name]["real"],
                "imag_parameter_index": derived_components[name]["imag"],
            }
            for output_index, name in enumerate(output_names)
        ],
        "evaluator": evaluator,
    }


def _compiled_plane_arena_stage(
    stage: Mapping[str, object],
) -> dict[str, object] | None:
    """Freeze one fused SymJIT stage's direct plane bindings.

    The ordinary SymJIT application remains a cold source payload.  Its
    row-major interface is not the execution contract: this record fixes the
    canonical preorder leaf maps, arena inputs, and overwrite destinations
    used to lower and bind the factor-free DirectApplication at load time.
    """

    if stage.get("parameter_layout") != "stage-local-value-momentum":
        return None
    evaluator = stage.get("evaluator")
    if not isinstance(evaluator, Mapping):
        raise ValueError("compiled stage evaluator metadata is invalid")
    parameter_count = stage.get("parameter_count")
    output_length = stage.get("output_length")
    if (
        isinstance(parameter_count, bool)
        or not isinstance(parameter_count, int)
        or parameter_count < 0
        or isinstance(output_length, bool)
        or not isinstance(output_length, int)
        or output_length < 1
    ):
        raise ValueError("compiled stage dimensions are invalid")

    leaves, output_stop, application_abi, source_application_abi = (
        _compiled_plane_arena_leaves(
            evaluator,
            tuple(range(parameter_count)),
            0,
        )
    )
    if leaves is None:
        return None
    if output_stop != output_length:
        raise ValueError(
            "compiled plane-arena leaves do not cover the fused stage outputs"
        )

    raw_inputs = stage.get("input_components")
    if not isinstance(raw_inputs, Sequence) or isinstance(raw_inputs, (str, bytes)):
        raise ValueError("compiled plane-arena input bindings are absent")
    inputs: list[dict[str, object] | None] = [None] * parameter_count
    for raw in raw_inputs:
        if not isinstance(raw, Mapping):
            raise ValueError("compiled plane-arena input binding is invalid")
        parameter_index = raw.get("parameter_index")
        kind = raw.get("kind")
        if (
            isinstance(parameter_index, bool)
            or not isinstance(parameter_index, int)
            or not 0 <= parameter_index < parameter_count
            or inputs[parameter_index] is not None
            or kind not in {"value", "momentum", "model_parameter"}
        ):
            raise ValueError("compiled plane-arena input bindings are inconsistent")
        inputs[parameter_index] = {
            "parameter_index": parameter_index,
            "kind": kind,
            "source_id": _required_nonnegative_int(raw, "source_id"),
            "component": _required_nonnegative_int(raw, "component"),
            "global_component": _required_nonnegative_int(raw, "global_component"),
            "real_valued": bool(raw.get("real_valued", False)),
        }
    if any(binding is None for binding in inputs):
        raise ValueError("compiled plane-arena input bindings are incomplete")

    raw_slots = stage.get("output_slots")
    if not isinstance(raw_slots, Sequence) or isinstance(raw_slots, (str, bytes)):
        raise ValueError("compiled plane-arena output bindings are absent")
    arena = (
        "amplitude"
        if str(stage.get("stage_kind", "")).startswith("amplitude")
        else "current"
    )
    outputs: list[dict[str, object] | None] = [None] * output_length
    seen_components: set[int] = set()
    for raw in raw_slots:
        if not isinstance(raw, Mapping):
            raise ValueError("compiled plane-arena output slot is invalid")
        output_start = _required_nonnegative_int(raw, "output_start")
        output_stop = _required_nonnegative_int(raw, "output_stop")
        component_start = _required_nonnegative_int(raw, "component_start")
        component_stop = _required_nonnegative_int(raw, "component_stop")
        if (
            output_stop < output_start
            or component_stop - component_start != output_stop - output_start
            or output_stop > output_length
        ):
            raise ValueError("compiled plane-arena output slot range is invalid")
        for offset in range(output_stop - output_start):
            output_index = output_start + offset
            component = component_start + offset
            if outputs[output_index] is not None or component in seen_components:
                raise ValueError("compiled plane-arena output bindings alias")
            seen_components.add(component)
            outputs[output_index] = {
                "output_index": output_index,
                "arena": arena,
                "component": component,
            }
    if any(binding is None for binding in outputs):
        raise ValueError("compiled plane-arena output bindings are incomplete")

    return {
        "schema_version": 1,
        "kind": "compiled-plane-arena-stage",
        "application_abi": application_abi,
        "source_application_abi": source_application_abi,
        "element_layout": "split-complex-component-major",
        "output_operation": "overwrite",
        "output_factor": "identity",
        "input_output_aliasing": "forbidden",
        "output_output_aliasing": "forbidden",
        "input_bindings": inputs,
        "output_bindings": outputs,
        "leaves": leaves,
    }


def _compiled_plane_arena_leaves(
    evaluator: Mapping[str, object],
    parent_inputs: tuple[int, ...],
    output_start: int,
) -> tuple[
    list[dict[str, object]] | None,
    int,
    str | None,
    str | None,
]:
    kind = evaluator.get("kind")
    if kind == "chunked-symbolica-evaluator":
        chunks = evaluator.get("chunks")
        if not isinstance(chunks, Sequence) or isinstance(chunks, (str, bytes)):
            raise ValueError("compiled chunked evaluator has invalid children")
        input_len = evaluator.get("input_len")
        if input_len is not None and input_len != len(parent_inputs):
            raise ValueError("compiled chunked evaluator input width changed")
        raw_maps = evaluator.get("chunk_input_indices")
        if raw_maps is None:
            input_maps = [parent_inputs] * len(chunks)
        else:
            if (
                not isinstance(raw_maps, Sequence)
                or isinstance(raw_maps, (str, bytes))
                or len(raw_maps) != len(chunks)
            ):
                raise ValueError("compiled chunk input maps are invalid")
            input_maps = []
            for raw_map in raw_maps:
                if not isinstance(raw_map, Sequence) or isinstance(
                    raw_map, (str, bytes)
                ):
                    raise ValueError("compiled chunk input map is invalid")
                mapped: list[int] = []
                for raw_index in raw_map:
                    if (
                        isinstance(raw_index, bool)
                        or not isinstance(raw_index, int)
                        or not 0 <= raw_index < len(parent_inputs)
                    ):
                        raise ValueError(
                            "compiled chunk input map references an absent input"
                        )
                    mapped.append(parent_inputs[raw_index])
                input_maps.append(tuple(mapped))
        result: list[dict[str, object]] = []
        cursor = output_start
        application_abi: str | None = None
        source_application_abi: str | None = None
        for raw_chunk, mapped in zip(chunks, input_maps, strict=True):
            if not isinstance(raw_chunk, Mapping):
                raise ValueError("compiled evaluator child is invalid")
            child, cursor, child_application_abi, child_source_abi = (
                _compiled_plane_arena_leaves(raw_chunk, mapped, cursor)
            )
            if child is None:
                return None, output_start, None, None
            if application_abi is None:
                application_abi = child_application_abi
                source_application_abi = child_source_abi
            elif (
                child_application_abi != application_abi
                or child_source_abi != source_application_abi
            ):
                raise ValueError(
                    "compiled plane-arena evaluator chunks mix direct ABIs"
                )
            result.extend(child)
        return result, cursor, application_abi, source_application_abi

    if kind == "symjit-application-evaluator":
        if (
            evaluator.get("runtime_capability") != SYMJIT_F64_RUNTIME_CAPABILITY
            or evaluator.get("application_abi") != SYMJIT_APPLICATION_ABI
            or evaluator.get("element_layout") != "complex-f64"
            or evaluator.get("batch_layout") != "row-major"
        ):
            raise ValueError("compiled SymJIT leaf has an incompatible source ABI")
        plane = evaluator.get("plane_application")
        if not isinstance(plane, Mapping):
            raise ValueError(
                "compiled SymJIT leaf predates the SymJIT 2.22 plane ABI; "
                "regenerate this artifact"
            )
        if (
            plane.get("application_abi") != SYMJIT_PLANE_APPLICATION_ABI
            or plane.get("storage_abi") != SYMJIT_APPLICATION_ABI
            or plane.get("element_layout") != "split-complex-plane-major"
            or plane.get("descriptor_order") != "inputs-re-im-then-outputs-re-im"
        ):
            raise ValueError("compiled SymJIT leaf has an incompatible plane ABI")
        application_path = plane.get("application_path")
        source_application_abi = SYMJIT_PLANE_APPLICATION_ABI
        application_abi = COMPILED_PLANE_DIRECT_APPLICATION_ABI
        optimization_level = _required_nonnegative_int(
            plane,
            "optimization_level",
        )
        direct_codegen_optimization_level = optimization_level
        if optimization_level not in {0, 1, 2, 3}:
            raise ValueError(
                "compiled SymJIT plane-arena optimization level must be 0, 1, 2, or 3"
            )
    elif kind == "compiled-complex-evaluator":
        if evaluator.get("runtime_capability") not in {
            SYMBOLICA_CPP_RUNTIME_CAPABILITY,
            SYMBOLICA_ASM_RUNTIME_CAPABILITY,
        }:
            raise ValueError(
                "compiled native leaf has an incompatible runtime capability"
            )
        direct = evaluator.get("native_direct_application")
        if not isinstance(direct, Mapping):
            raise ValueError(
                "compiled native leaf has no plane-native DirectApplication"
            )
        application_abi = direct.get("application_abi")
        application_path = direct.get("library_path")
        source_application_abi = application_abi
        optimization_level = 3
        direct_codegen_optimization_level = 3
        if application_abi != NATIVE_COMPILED_DIRECT_APPLICATION_ABI:
            raise ValueError("compiled native leaf has an incompatible direct ABI")
    else:
        return None, output_start, None, None

    input_len = _required_nonnegative_int(evaluator, "input_len")
    output_len = _required_nonnegative_int(evaluator, "output_len")
    if (
        input_len != len(parent_inputs)
        or output_len < 1
        or not isinstance(application_path, str)
        or not application_path
    ):
        raise ValueError("compiled SymJIT leaf metadata is invalid")
    output_stop = output_start + output_len
    return (
        [
            {
                "application_path": application_path,
                "source_application_abi": source_application_abi,
                "optimization_level": optimization_level,
                "direct_codegen_optimization_level": (
                    direct_codegen_optimization_level
                ),
                "input_len": input_len,
                "output_len": output_len,
                "input_indices": list(parent_inputs),
                "output_start": output_start,
                "output_stop": output_stop,
            }
        ],
        output_stop,
        str(application_abi),
        str(source_application_abi),
    )


def _required_nonnegative_int(
    record: Mapping[str, object],
    name: str,
) -> int:
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"compiled plane-arena {name.replace('_', ' ')} is invalid")
    return value


def _finalize_stage_evaluator_payload(
    blueprint: GenericStageCompilerBlueprint,
    *,
    stage_payloads: list[dict[str, object]],
    amplitude_payload: dict[str, object],
    stage_timings: list[dict[str, object]],
    build_started: float,
    total_build_s_override: float | None = None,
) -> dict[str, object]:
    stage_local_layout = (
        blueprint.amplitude_stage.parameter_layout == "stage-local-value-momentum"
        and all(
            stage.parameter_layout == "stage-local-value-momentum"
            for stage in blueprint.stages
        )
    )
    total_build_s = (
        time.perf_counter() - build_started
        if total_build_s_override is None
        else float(total_build_s_override)
    )
    jit_compile_s = sum(
        float(record.get("jit_compile_s") or 0.0) for record in stage_timings
    )
    timing_totals: dict[str, object] = {}
    non_additive_stage_timing_keys = {
        "output_chunk_autotune_batch_size",
        "output_chunk_autotune_baseline_size",
        "output_chunk_autotune_selected_size",
        "output_chunk_autotune_baseline_us",
        "output_chunk_autotune_selected_us",
        "output_chunk_autotune_gain",
        "output_chunk_autotune_shared_pack_scoring",
    }
    for record in stage_timings:
        for key, value in record.items():
            if (
                key == "evaluator_label"
                or key == "jit_compile_s"
                or key in non_additive_stage_timing_keys
            ):
                continue
            if isinstance(value, (float, int)):
                timing_totals[key] = float(timing_totals.get(key, 0.0)) + float(value)
    timing_totals["stage_evaluator_build_s"] = total_build_s
    timing_totals["jit_compile_s"] = jit_compile_s
    timing_totals["jit_fraction_of_stage_evaluator_build"] = (
        None if total_build_s <= 0.0 else jit_compile_s / total_build_s
    )
    timing_totals["stages"] = stage_timings
    evaluator_manifests = [
        _dict(payload["evaluator"]) for payload in (*stage_payloads, amplitude_payload)
    ]
    required_runtime_capabilities = set(
        aggregate_runtime_capabilities(evaluator_manifests)
    )
    if any(
        slot.selector_domain_ids or slot.color_selector_domain_ids
        for stage in (*blueprint.stages, blueprint.amplitude_stage)
        for slot in stage.output_slots
    ):
        required_runtime_capabilities.add(COMPILED_RUNTIME_SELECTORS_CAPABILITY)
    direct_stage_count = sum(
        "compiled_plane_arena" in payload
        for payload in (*stage_payloads, amplitude_payload)
    )
    stage_count = len(stage_payloads) + 1
    requires_direct = bool(
        required_runtime_capabilities
        & {
            SYMJIT_F64_RUNTIME_CAPABILITY,
            SYMBOLICA_CPP_RUNTIME_CAPABILITY,
            SYMBOLICA_ASM_RUNTIME_CAPABILITY,
        }
    )
    if requires_direct and direct_stage_count != stage_count:
        raise ValueError(
            "compiled f64 artifacts require compiled-plane-arena-v1 metadata "
            "for every fused stage"
        )
    if direct_stage_count:
        if direct_stage_count != stage_count:
            raise ValueError(
                "compiled plane-arena metadata must cover every fused stage"
            )
        required_runtime_capabilities.add(COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY)
    return {
        "kind": "generic-dag-stage-evaluator-artifacts",
        "required_runtime_capabilities": sorted(required_runtime_capabilities),
        "runtime_available": True,
        "runtime_unavailable_message": None,
        "parameter_count": 0 if stage_local_layout else blueprint.parameter_count,
        "value_parameter_count": (
            0 if stage_local_layout else blueprint.value_parameter_count
        ),
        "momentum_parameter_count": (
            0 if stage_local_layout else blueprint.momentum_parameter_count
        ),
        "model_parameter_count": (
            0 if stage_local_layout else blueprint.model_parameter_count
        ),
        "real_valued_inputs": (
            [] if stage_local_layout else list(blueprint.real_valued_inputs)
        ),
        "parameter_layout": (
            "stage-local-value-momentum"
            if stage_local_layout
            else "global-value-momentum"
        ),
        "stage_count": blueprint.stage_count,
        "build_timing": timing_totals,
        "stages": stage_payloads,
        "amplitude_stage": amplitude_payload,
    }


def _compile_stage_evaluator_artifact(
    stage: GenericCompiledStageBlueprint,
    artifact_dir: Path,
    *,
    compiler: StageEvaluatorCompiler | None,
    blueprint: GenericStageCompilerBlueprint | None,
    symbolica_settings: Any | None,
    merge_evaluators_strategy: bool,
    verbose_evaluator_build: bool,
    jit_compile: bool,
    progress_callback: Any | None,
    current_stage_position: int | None = None,
    current_stage_count: int | None = None,
) -> dict[str, object]:
    if not stage.output_expressions:
        raise ValueError(
            f"generic stage {stage.evaluator_label!r} has no output expressions"
        )
    started = time.perf_counter()
    if compiler is not None:
        manifest = compiler(
            stage,
            stage.parameter_symbols,
            stage.real_valued_inputs,
        )
    else:
        manifest = _compile_default_stage_evaluator(
            stage,
            blueprint,
            artifact_dir,
            symbolica_settings=symbolica_settings,
            merge_evaluators_strategy=merge_evaluators_strategy,
            verbose_evaluator_build=verbose_evaluator_build,
            jit_compile=jit_compile,
            progress_callback=progress_callback,
            current_stage_position=current_stage_position,
            current_stage_count=current_stage_count,
        )
    if not isinstance(manifest, dict):
        raise TypeError(
            f"generic stage compiler for {stage.evaluator_label!r} "
            "did not return a manifest dictionary"
        )
    build_s = time.perf_counter() - started
    manifest.setdefault("build_timing", {})
    timing = manifest["build_timing"]
    if isinstance(timing, dict):
        previous_stage_build_s = timing.get("stage_evaluator_build_s")
        timing["stage_evaluator_build_s"] = build_s
        if previous_stage_build_s is not None:
            timing["stage_compiler_wrapper_s"] = build_s - float(previous_stage_build_s)
        timing.setdefault("symbolica_evaluator_build_s", build_s)
        if _manifest_uses_jit_evaluator(manifest):
            timing.setdefault("jit_compile_s", build_s)
    return manifest


def build_and_write_generic_stage_evaluator_artifacts(
    manifest: StageCompilationInput | GenericDAG,
    runtime_schema: Mapping[str, object],
    artifact_dir: str | Path,
    *,
    model: Model | None = None,
    enable_lc_sector_runtime_selector: bool | None = None,
    stage_local_parameter_layout: bool = True,
    compiler: StageEvaluatorCompiler | None = None,
    symbolica_settings: Any | None = None,
    merge_evaluators_strategy: bool = False,
    verbose_evaluator_build: bool = False,
    jit_compile: bool = True,
    blueprint_progress_callback: StageBlueprintProgress | None = None,
    evaluator_progress_callback: Any | None = None,
) -> tuple[GenericStageCompilerBlueprint, dict[str, object]]:
    """Lower, compile, and release one recursion stage at a time."""

    output_dir = Path(artifact_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    schema = _dict(runtime_schema)
    current_stage_count = len(_list(schema["stages"]))
    stage_count = current_stage_count + 1
    build_started = time.perf_counter()
    if evaluator_progress_callback is not None:
        evaluator_progress_callback(
            {
                "stage": "stage compile",
                "item": "start",
                "total": stage_count,
                "step": "evaluator compilation",
                "stage_total": stage_count,
            }
        )

    stage_payloads: list[dict[str, object]] = []
    amplitude_payload: dict[str, object] | None = None
    stage_timings: list[dict[str, object]] = []

    def consume_stage(
        stage: GenericCompiledStageBlueprint,
        position: int,
        reported_current_stage_count: int,
    ) -> None:
        nonlocal amplitude_payload
        if reported_current_stage_count != current_stage_count:
            raise ValueError("streamed stage count changed during blueprint lowering")
        if not stage.expression_ready:
            raise ValueError(
                "cannot write generic evaluator artifact with lowering blockers: "
                + "; ".join(stage.blockers)
            )
        prepared_stage = _prepare_stage_for_output_chunking(
            stage,
            blueprint=None,
            symbolica_settings=symbolica_settings,
            current_stage_position=position,
            current_stage_count=current_stage_count,
        )
        payload = prepared_stage.to_json_dict()
        payload["evaluator"] = _compile_stage_evaluator_artifact(
            prepared_stage,
            output_dir,
            compiler=compiler,
            blueprint=None,
            symbolica_settings=symbolica_settings,
            merge_evaluators_strategy=merge_evaluators_strategy,
            verbose_evaluator_build=verbose_evaluator_build,
            jit_compile=jit_compile,
            progress_callback=evaluator_progress_callback,
            current_stage_position=position,
            current_stage_count=current_stage_count,
        )
        direct = _compiled_plane_arena_stage(payload)
        if direct is not None:
            payload["compiled_plane_arena"] = direct
        timing = _stage_build_timing_record(
            prepared_stage.evaluator_label,
            payload["evaluator"],
        )
        stage_timings.append(timing)
        if str(prepared_stage.stage_kind).startswith("amplitude"):
            amplitude_payload = payload
        else:
            stage_payloads.append(payload)
        if evaluator_progress_callback is not None:
            evaluator_progress_callback(
                {
                    "stage": "stage complete",
                    "item": prepared_stage.evaluator_label,
                    "increment": 1,
                    "total": stage_count,
                    "duration_s": timing["stage_evaluator_build_s"],
                    "step": "stage complete",
                    "stage_index": position + 1,
                    "stage_total": stage_count,
                    "subset_size": prepared_stage.subset_size,
                    "interaction_count": len(prepared_stage.interaction_ids),
                    "input_count": prepared_stage.parameter_count,
                    "output_count": prepared_stage.output_length,
                }
            )

    blueprint = build_generic_stage_compiler_blueprint(
        manifest,
        model=model,
        enable_lc_sector_runtime_selector=enable_lc_sector_runtime_selector,
        runtime_schema=schema,
        stage_local_parameter_layout=stage_local_parameter_layout,
        progress_callback=blueprint_progress_callback,
        stage_consumer=consume_stage,
        release_consumed_expressions=True,
    )
    if amplitude_payload is None or len(stage_payloads) != current_stage_count:
        raise ValueError("streamed stage compilation produced incomplete metadata")
    return blueprint, _finalize_stage_evaluator_payload(
        blueprint,
        stage_payloads=stage_payloads,
        amplitude_payload=amplitude_payload,
        stage_timings=stage_timings,
        build_started=build_started,
        total_build_s_override=sum(
            float(timing["stage_evaluator_build_s"]) for timing in stage_timings
        ),
    )


def _stage_build_timing_record(
    evaluator_label: str,
    evaluator_manifest: object,
) -> dict[str, object]:
    manifest = evaluator_manifest if isinstance(evaluator_manifest, dict) else {}
    raw_timing = manifest.get("build_timing") if isinstance(manifest, dict) else None
    timing = raw_timing if isinstance(raw_timing, dict) else {}
    record: dict[str, object] = {
        "evaluator_label": evaluator_label,
        "stage_evaluator_build_s": float(timing.get("stage_evaluator_build_s") or 0.0),
        "symbolica_evaluator_build_s": float(
            timing.get("symbolica_evaluator_build_s") or 0.0
        ),
        "jit_compile_s": (
            None
            if timing.get("jit_compile_s") is None
            else float(timing.get("jit_compile_s") or 0.0)
        ),
    }
    for key, value in timing.items():
        if key in record:
            continue
        if isinstance(value, (float, int)):
            record[str(key)] = float(value)
    return record


def _manifest_uses_jit_evaluator(manifest: Mapping[str, object]) -> bool:
    if str(manifest.get("kind", "")) in {
        "jit-symbolica-evaluator",
        "symjit-application-evaluator",
    }:
        return True
    if str(manifest.get("kind", "")) == "chunked-symbolica-evaluator":
        chunks = manifest.get("chunks")
        if isinstance(chunks, Sequence) and chunks:
            return all(
                isinstance(chunk, Mapping) and _manifest_uses_jit_evaluator(chunk)
                for chunk in chunks
            )
    settings = manifest.get("settings")
    if isinstance(settings, Mapping):
        return str(settings.get("backend", "")) == "jit"
    return False


def _compile_default_stage_evaluator(
    stage: GenericCompiledStageBlueprint,
    blueprint: GenericStageCompilerBlueprint | None,
    artifact_dir: Path,
    *,
    symbolica_settings: Any | None,
    merge_evaluators_strategy: bool,
    verbose_evaluator_build: bool,
    jit_compile: bool,
    progress_callback: Any | None,
    current_stage_position: int | None = None,
    current_stage_count: int | None = None,
) -> dict[str, object]:
    from ..evaluators.symbolica import (
        SymbolicaEvaluatorSettings,
        _compile_symbolica_outputs,
        _symbolica_evaluator_artifact_manifest,
    )

    settings = _stage_symbolica_settings(
        stage,
        blueprint,
        symbolica_settings or SymbolicaEvaluatorSettings(),
        current_stage_position=current_stage_position,
        current_stage_count=current_stage_count,
    )
    symbolica_started = time.perf_counter()
    params = list(stage.parameter_symbols)

    def compile_with(candidate_settings: Any, candidate_label: str) -> Any:
        return _compile_symbolica_outputs(
            stage.output_expressions,
            params,
            merge_evaluators_strategy=merge_evaluators_strategy,
            verbose_evaluator_build=verbose_evaluator_build,
            real_params=stage.real_valued_inputs,
            symbolica_settings=candidate_settings,
            jit_compile=jit_compile,
            label=candidate_label,
            progress_callback=progress_callback,
            functions={
                (function, arguments): body
                for function, arguments, body in stage.symbolica_functions
            },
            output_partitions=stage.selector_output_partitions,
            native_direct_only=(
                getattr(candidate_settings, "backend", None) == "compiled-complex"
                and stage.parameter_layout == "stage-local-value-momentum"
            ),
        )

    autotune_timing: dict[str, float] = {}
    if getattr(settings, "output_chunk_strategy", "uniform") == "measured-stage":
        evaluator, autotune_timing = _compile_measured_stage_output_chunks(
            settings=settings,
            output_count=len(stage.output_expressions),
            parameter_count=len(params),
            real_params=stage.real_valued_inputs,
            label=stage.evaluator_label,
            compile_with=compile_with,
            progress_callback=progress_callback,
            jit_compile=jit_compile,
        )
    else:
        evaluator = compile_with(settings, stage.evaluator_label)
    _compile_native_stage_direct_applications(
        evaluator,
        stage,
        settings,
    )
    symbolica_build_s = time.perf_counter() - symbolica_started
    artifact_started = time.perf_counter()
    manifest = _symbolica_evaluator_artifact_manifest(evaluator, artifact_dir)
    artifact_manifest_s = time.perf_counter() - artifact_started
    timing = manifest.setdefault("build_timing", {})
    if isinstance(timing, dict):
        timing.update(autotune_timing)
        timing["symbolica_evaluator_build_s"] = symbolica_build_s
        timing["artifact_manifest_s"] = artifact_manifest_s
        timing["stage_evaluator_build_s"] = symbolica_build_s + artifact_manifest_s
    return manifest


def _compile_native_stage_direct_applications(
    evaluator: Any,
    stage: GenericCompiledStageBlueprint,
    settings: Any,
) -> None:
    """Produce the direct-only native companion before serialization."""

    if getattr(settings, "backend", None) != "compiled-complex":
        return
    if stage.parameter_layout != "stage-local-value-momentum":
        return
    from ..evaluators.native_direct_cpp import NativeDirectCppParameterKind
    from ..evaluators.symbolica_adapters import (
        compile_native_direct_applications,
    )
    from ..models.prepared_target import native_prepared_target

    components: list[object | None] = [None] * stage.parameter_count
    for component in stage.input_components:
        if (
            component.parameter_index >= len(components)
            or components[component.parameter_index] is not None
        ):
            raise ValueError(
                "native DirectApplication stage input bindings are invalid"
            )
        if component.kind == "model_parameter":
            kind = (
                NativeDirectCppParameterKind.REAL_SCALAR
                if component.real_valued
                else NativeDirectCppParameterKind.COMPLEX_SCALAR
            )
        else:
            kind = (
                NativeDirectCppParameterKind.REAL_PLANE
                if component.real_valued
                else NativeDirectCppParameterKind.COMPLEX_PLANE
            )
        components[component.parameter_index] = kind
    if any(component is None for component in components):
        missing = tuple(
            index for index, component in enumerate(components) if component is None
        )
        raise ValueError(
            "native DirectApplication stage input bindings are incomplete: "
            f"stage={stage.evaluator_label!r}, parameter_count="
            f"{stage.parameter_count}, input_component_count="
            f"{len(stage.input_components)}, missing_parameter_indices={missing!r}"
        )

    include_features = bool(getattr(settings, "compiled_native", False))
    target = native_prepared_target(include_cpu_features=include_features)
    target_triple = str(target["target_triple"])
    cpu_features = tuple(str(item) for item in target["cpu_features"])
    simd_lane_width = (
        4 if target_triple.startswith("x86_64") and "avx2" in cpu_features else 2
    )
    if not compile_native_direct_applications(
        evaluator,
        components,
        target_triple=target_triple,
        cpu_features=cpu_features,
        simd_lane_width=simd_lane_width,
    ):
        raise ValueError(
            "compiled native stage did not produce a plane-native DirectApplication"
        )


def _compile_measured_stage_output_chunks(
    *,
    settings: Any,
    output_count: int,
    parameter_count: int,
    real_params: Sequence[int],
    label: str,
    compile_with: Callable[[Any, str], Any],
    progress_callback: Any | None,
    jit_compile: bool,
) -> tuple[Any, dict[str, float]]:
    base = getattr(settings, "compiled_output_chunk_size", None)
    if base is None or getattr(settings, "backend", None) != "jit" or not jit_compile:
        uniform = replace(settings, output_chunk_strategy="uniform")
        return compile_with(uniform, label), {}

    requested_sizes = (
        int(base),
        max(1, int(base) // 2),
        max(1, 3 * int(base) // 4),
        max(1, 3 * int(base) // 2),
        int(base) * 2,
        None,
    )
    effective_sizes: list[int | None] = []
    for requested in requested_sizes:
        effective = (
            None
            if requested is None or output_count <= int(requested)
            else int(requested)
        )
        if effective not in effective_sizes:
            effective_sizes.append(effective)
    baseline_size = None if output_count <= int(base) else int(base)

    started = time.perf_counter()
    candidates: dict[int | None, Any] = {}
    for chunk_size in effective_sizes:
        suffix = "none" if chunk_size is None else str(chunk_size)
        candidate_settings = replace(
            settings,
            compiled_output_chunk_size=chunk_size,
            output_chunk_strategy="uniform",
        )
        candidate_label = (
            label if chunk_size == baseline_size else f"{label}_autotune_chunk_{suffix}"
        )
        candidates[chunk_size] = compile_with(candidate_settings, candidate_label)

    autotune_batch_size = int(
        getattr(settings, "output_chunk_autotune_batch_size", 128)
    )
    rows = np.full(
        (autotune_batch_size, parameter_count),
        complex(0.75, 0.125),
        dtype=np.complex128,
    )
    if real_params:
        rows[:, list(real_params)] = 0.75

    materialize_started = time.perf_counter()
    for evaluator in candidates.values():
        evaluator.evaluate_complex(rows)
    materialize_s = time.perf_counter() - materialize_started

    shared_pack_scoring = all(
        bool(getattr(evaluator, "supports_complex_profiled", lambda: False)())
        for evaluator in candidates.values()
    )

    def score_candidate(evaluator: Any) -> float:
        if shared_pack_scoring:
            _output, profile = evaluator.evaluate_complex_profiled(rows)
            return max(sum(float(value) for value in profile), 1.0e-9)
        probe_started = time.perf_counter()
        evaluator.evaluate_complex(rows)
        return max(time.perf_counter() - probe_started, 1.0e-6)

    benchmark_started = time.perf_counter()
    scores: dict[int | None, float] = {}
    repeats: dict[int | None, int] = {}
    for chunk_size, evaluator in candidates.items():
        probe_s = max(score_candidate(evaluator), 1.0e-6)
        repeats[chunk_size] = max(1, min(256, int(0.01 / probe_s)))
        scores[chunk_size] = probe_s

    samples: dict[int | None, list[float]] = {
        chunk_size: [] for chunk_size in candidates
    }
    ordered_sizes = list(candidates)
    for round_index in range(5):
        rotated = ordered_sizes[round_index:] + ordered_sizes[:round_index]
        for chunk_size in rotated:
            evaluator = candidates[chunk_size]
            count = repeats[chunk_size]
            if shared_pack_scoring:
                samples[chunk_size].append(
                    sum(score_candidate(evaluator) for _ in range(count)) / count
                )
            else:
                sample_started = time.perf_counter()
                for _ in range(count):
                    evaluator.evaluate_complex(rows)
                samples[chunk_size].append(
                    (time.perf_counter() - sample_started) / count
                )
    scores = {
        chunk_size: sorted(values)[len(values) // 2]
        for chunk_size, values in samples.items()
    }
    benchmark_s = time.perf_counter() - benchmark_started
    selected_size = _select_measured_chunk_candidate(
        scores,
        baseline_size=baseline_size,
        minimum_gain=0.05,
    )
    selected = candidates[selected_size]
    baseline_score = scores[baseline_size]
    selected_score = scores[selected_size]
    autotune_s = time.perf_counter() - started
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "chunk autotune",
                "item": (
                    f"{label} base={baseline_size or 'none'} "
                    f"selected={selected_size or 'none'} "
                    f"gain={(1.0 - selected_score / baseline_score):.1%}"
                    + (" shared-pack" if shared_pack_scoring else "")
                ),
            }
        )
    evaluator_timing = getattr(selected, "build_timing", None)
    if isinstance(evaluator_timing, dict):
        evaluator_timing.update(
            {
                "output_chunk_autotune_s": autotune_s,
                "output_chunk_autotune_materialize_s": materialize_s,
                "output_chunk_autotune_benchmark_s": benchmark_s,
                "output_chunk_autotune_candidate_count": float(len(candidates)),
                "output_chunk_autotune_batch_size": float(autotune_batch_size),
                "output_chunk_autotune_baseline_size": float(baseline_size or 0),
                "output_chunk_autotune_selected_size": float(selected_size or 0),
                "output_chunk_autotune_baseline_us": baseline_score * 1.0e6,
                "output_chunk_autotune_selected_us": selected_score * 1.0e6,
                "output_chunk_autotune_gain": 1.0 - selected_score / baseline_score,
                "output_chunk_autotune_shared_pack_scoring": float(shared_pack_scoring),
            }
        )
    return selected, {
        "output_chunk_autotune_s": autotune_s,
        "output_chunk_autotune_materialize_s": materialize_s,
        "output_chunk_autotune_benchmark_s": benchmark_s,
        "output_chunk_autotune_candidate_count": float(len(candidates)),
        "output_chunk_autotune_batch_size": float(autotune_batch_size),
        "output_chunk_autotune_baseline_size": float(baseline_size or 0),
        "output_chunk_autotune_selected_size": float(selected_size or 0),
        "output_chunk_autotune_baseline_us": baseline_score * 1.0e6,
        "output_chunk_autotune_selected_us": selected_score * 1.0e6,
        "output_chunk_autotune_gain": 1.0 - selected_score / baseline_score,
        "output_chunk_autotune_shared_pack_scoring": float(shared_pack_scoring),
    }


def _select_measured_chunk_candidate(
    scores: Mapping[int | None, float],
    *,
    baseline_size: int | None,
    minimum_gain: float,
) -> int | None:
    if baseline_size not in scores:
        raise ValueError("measured chunk scores do not include the baseline")
    best_size = min(scores, key=scores.__getitem__)
    if scores[best_size] <= scores[baseline_size] * (1.0 - minimum_gain):
        return best_size
    return baseline_size

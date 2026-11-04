# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
from copy import deepcopy

import pytest

from pyamplicol._internal.versions import (
    COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY,
    COMPILED_PLANE_DIRECT_APPLICATION_ABI,
    NATIVE_COMPILED_DIRECT_APPLICATION_ABI,
    SYMBOLICA_ASM_RUNTIME_CAPABILITY,
    SYMBOLICA_CPP_RUNTIME_CAPABILITY,
    SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY,
    SYMJIT_APPLICATION_ABI,
    SYMJIT_F64_RUNTIME_CAPABILITY,
    SYMJIT_PLANE_APPLICATION_ABI,
)
from pyamplicol.generation.artifact_writer import (
    _compiled_momentum_slot_ids,
    _prefix_evaluator_payload_paths,
    _stage_evaluator_set,
)
from pyamplicol.generation.stage_artifacts import (
    _compiled_plane_arena_stage,
)


def _leaf(
    path: str,
    input_len: int,
    output_len: int,
    *,
    optimization_level: int = 3,
) -> dict[str, object]:
    plane_path = path.removesuffix(".symjit") + ".plane.symjit"
    return {
        "kind": "symjit-application-evaluator",
        "runtime_capability": SYMJIT_F64_RUNTIME_CAPABILITY,
        "input_len": input_len,
        "output_len": output_len,
        "application_path": path,
        "application_abi": SYMJIT_APPLICATION_ABI,
        "plane_application": {
            "application_path": plane_path,
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
            "optimization_level": optimization_level,
            "simd": True,
            "complex": True,
            "fast_math": True,
            "fast_complex": False,
            "compression": False,
            "threading": False,
            "direct_arena": True,
            "source_digest": hashlib.sha256(
                f"instructions:{path}".encode()
            ).hexdigest(),
            "target": {"word_bits": 64, "endianness": "little"},
        },
        "element_layout": "complex-f64",
        "batch_layout": "row-major",
        "compiler_type": "native",
        "translation_mode": "indirect",
        "optimization_level": optimization_level,
        "word_bits": 64,
        "endianness": "little",
        "required_defuns": [],
        "evaluator_state_path": None,
        "evaluator_state_runtime_capability": (SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY),
    }


def _stage(*, amplitude: bool = False) -> dict[str, object]:
    evaluator = {
        "kind": "chunked-symbolica-evaluator",
        "input_len": 3,
        "chunk_input_indices": [[0, 2], [1]],
        "required_runtime_capabilities": [SYMJIT_F64_RUNTIME_CAPABILITY],
        "chunks": [
            _leaf("evaluators/left.symjit", 2, 1),
            _leaf("evaluators/right.symjit", 1, 1),
        ],
    }
    return {
        "stage_index": 2,
        "stage_kind": "amplitude-roots" if amplitude else "current-combine",
        "subset_size": None if amplitude else 3,
        "evaluator_label": "amplitude" if amplitude else "stage-2",
        "parameter_layout": "stage-local-value-momentum",
        "output_length": 2,
        "output_slots": [
            {
                "value_slot_id": -1 if amplitude else 7,
                "current_id": -1 if amplitude else 7,
                "variant": "amplitude-root" if amplitude else "propagated",
                "component_start": 0 if amplitude else 12,
                "component_stop": 2 if amplitude else 14,
                "output_start": 0,
                "output_stop": 2,
                "color_selector_domain_ids": [0],
            }
        ],
        "input_value_slot_ids": [0],
        "output_value_slot_ids": [] if amplitude else [7],
        "interaction_ids": [4],
        "input_components": [
            {
                "kind": "value",
                "source_id": 0,
                "component": 0,
                "global_component": 0,
                "parameter_index": 0,
                "real_valued": False,
            },
            {
                "kind": "momentum",
                "source_id": 3,
                "component": 2,
                "global_component": 22,
                "parameter_index": 1,
                "real_valued": True,
            },
            {
                "kind": "model_parameter",
                "source_id": 1,
                "component": 0,
                "global_component": 31,
                "parameter_index": 2,
                "real_valued": False,
            },
        ],
        "parameter_count": 3,
        "value_parameter_count": 1,
        "momentum_parameter_count": 1,
        "model_parameter_count": 1,
        "real_valued_inputs": [1],
        "expression_ready": True,
        "blockers": [],
        "evaluator": evaluator,
    }


def test_compiled_momentum_slot_selection_uses_whole_lane_leaf_inputs() -> None:
    stage = _stage()
    # The parent binding still mentions a momentum, but neither final chunk
    # consumes it. It must not keep a dead momentum sum alive.
    stage["evaluator"]["chunk_input_indices"] = [[0, 2], [0]]
    stage["compiled_plane_arena"] = _compiled_plane_arena_stage(stage)
    amplitude = _stage(amplitude=True)
    amplitude["compiled_plane_arena"] = _compiled_plane_arena_stage(amplitude)
    evaluators = {"stages": [stage], "amplitude_stage": amplitude}
    assert _compiled_momentum_slot_ids(
        evaluators, value_component_count=8, momentum_component_count=16
    ) == [3]

    # Every leaf in the prepared lane contributes, regardless of any runtime
    # helicity/colour selection. Repeated bindings are retained only once.
    stage["evaluator"]["chunk_input_indices"] = [[1, 2], [1]]
    stage["input_components"][1]["global_component"] = 10
    stage["input_components"][1]["source_id"] = 0
    stage["compiled_plane_arena"] = _compiled_plane_arena_stage(stage)
    assert _compiled_momentum_slot_ids(
        evaluators, value_component_count=8, momentum_component_count=16
    ) == [0, 3]
    assert _compiled_momentum_slot_ids(
        {"stages": [], "amplitude_stage": stage},
        value_component_count=8,
        momentum_component_count=16,
    ) == [0]


def test_compiled_momentum_slot_selection_rejects_out_of_range_input() -> None:
    stage = _stage(amplitude=True)
    stage["compiled_plane_arena"] = _compiled_plane_arena_stage(stage)
    with pytest.raises(ValueError, match="outside momentum slots"):
        _compiled_momentum_slot_ids(
            {"stages": [], "amplitude_stage": stage},
            value_component_count=8,
            momentum_component_count=12,
        )


def _native_leaf(
    path: str,
    input_len: int,
    output_len: int,
    runtime_capability: str,
) -> dict[str, object]:
    return {
        "kind": "compiled-complex-evaluator",
        "runtime_capability": runtime_capability,
        "backend": "compiled-complex",
        "number_type": "complex",
        "function_name": path.replace("/", "_").replace(".", "_"),
        "input_len": input_len,
        "output_len": output_len,
        "settings": {
            "compiled_optimization_level": 3,
            "compiled_inline_asm": (
                "none"
                if runtime_capability == SYMBOLICA_CPP_RUNTIME_CAPABILITY
                else "default"
            ),
        },
        "source_path": f"{path}.direct.cpp",
        "library_path": f"{path}.direct",
        "evaluator_state_path": f"{path}.evaluator.bin",
        "native_direct_application": {
            "application_abi": NATIVE_COMPILED_DIRECT_APPLICATION_ABI,
            "function_name": path.replace("/", "_").replace(".", "_"),
            "source_path": f"{path}.direct.cpp",
            "library_path": f"{path}.direct",
            "target": {
                "triple": "aarch64-apple-darwin",
                "cpu_features": [],
            },
            "evaluator_state_sha256": "a" * 64,
            "instruction_count": 1,
            "temporary_count": 0,
            "input_plane_count": input_len,
            "scalar_input_count": 0,
            "output_plane_count": 2 * output_len,
            "simd_lane_width": 2,
            "logical_stack_bytes": 32 * output_len,
            "output_semantics": "factor-free-overwrite",
        },
    }


def _set() -> dict[str, object]:
    stage = _stage()
    amplitude = _stage(amplitude=True)
    stage["compiled_plane_arena"] = _compiled_plane_arena_stage(stage)
    amplitude["compiled_plane_arena"] = _compiled_plane_arena_stage(amplitude)
    return {
        "kind": "generic-dag-stage-evaluator-artifacts",
        "required_runtime_capabilities": [
            COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY,
            SYMJIT_F64_RUNTIME_CAPABILITY,
        ],
        "runtime_available": True,
        "runtime_unavailable_message": None,
        "parameter_count": 0,
        "value_parameter_count": 0,
        "momentum_parameter_count": 0,
        "model_parameter_count": 0,
        "real_valued_inputs": [],
        "parameter_layout": "stage-local-value-momentum",
        "stage_count": 2,
        "stages": [stage],
        "amplitude_stage": amplitude,
    }


def test_compiled_plane_contract_freezes_leaf_and_plane_bindings() -> None:
    stage = _stage()
    direct = _compiled_plane_arena_stage(stage)

    assert direct is not None
    assert direct["application_abi"] == COMPILED_PLANE_DIRECT_APPLICATION_ABI
    assert [leaf["input_indices"] for leaf in direct["leaves"]] == [[0, 2], [1]]
    assert [leaf["output_start"] for leaf in direct["leaves"]] == [0, 1]
    assert [binding["component"] for binding in direct["output_bindings"]] == [
        12,
        13,
    ]


@pytest.mark.parametrize("optimization_level", [0, 1, 2, 3])
def test_compiled_plane_contract_covers_every_jit_optimization_level(
    optimization_level: int,
) -> None:
    stage = _stage()
    leaf = stage["evaluator"]["chunks"][0]
    leaf["optimization_level"] = optimization_level
    leaf["plane_application"]["optimization_level"] = optimization_level

    direct = _compiled_plane_arena_stage(stage)

    assert direct is not None
    assert [leaf["optimization_level"] for leaf in direct["leaves"]] == [
        optimization_level,
        3,
    ]
    assert [leaf["direct_codegen_optimization_level"] for leaf in direct["leaves"]] == [
        optimization_level,
        3,
    ]


def test_compiled_plane_contract_rejects_unknown_jit_optimization_level() -> None:
    stage = _stage()
    stage["evaluator"]["chunks"][0]["plane_application"]["optimization_level"] = 4

    with pytest.raises(ValueError, match="optimization level must be 0, 1, 2, or 3"):
        _compiled_plane_arena_stage(stage)


@pytest.mark.parametrize(
    "runtime_capability",
    [SYMBOLICA_CPP_RUNTIME_CAPABILITY, SYMBOLICA_ASM_RUNTIME_CAPABILITY],
)
def test_native_compiled_contract_publishes_direct_library_leaves(
    runtime_capability: str,
) -> None:
    stage = _stage()
    stage["evaluator"] = {
        "kind": "chunked-symbolica-evaluator",
        "input_len": 3,
        "chunk_input_indices": [[0, 2], [1]],
        "required_runtime_capabilities": [runtime_capability],
        "chunks": [
            _native_leaf("compiled/left", 2, 1, runtime_capability),
            _native_leaf("compiled/right", 1, 1, runtime_capability),
        ],
    }

    direct = _compiled_plane_arena_stage(stage)

    assert direct is not None
    assert direct["application_abi"] == NATIVE_COMPILED_DIRECT_APPLICATION_ABI
    assert direct["source_application_abi"] == (NATIVE_COMPILED_DIRECT_APPLICATION_ABI)
    assert [leaf["application_path"] for leaf in direct["leaves"]] == [
        "compiled/left.direct",
        "compiled/right.direct",
    ]


def test_stage_set_requires_complete_capability_bound_metadata() -> None:
    serialized = _stage_evaluator_set(_set())
    assert (
        COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY
        in serialized["required_runtime_capabilities"]
    )
    assert all(
        "compiled_plane_arena" in stage
        for stage in (*serialized["stages"], serialized["amplitude_stage"])
    )

    missing = deepcopy(_set())
    del missing["stages"][0]["compiled_plane_arena"]
    with pytest.raises(ValueError, match="compiled f64 artifacts require"):
        _stage_evaluator_set(missing)

    pre_arena = deepcopy(_set())
    pre_arena["required_runtime_capabilities"].remove(
        COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY
    )
    del pre_arena["stages"][0]["compiled_plane_arena"]
    del pre_arena["amplitude_stage"]["compiled_plane_arena"]
    with pytest.raises(ValueError, match="compiled f64 artifacts require"):
        _stage_evaluator_set(pre_arena)

    drift = deepcopy(_set())
    drift["amplitude_stage"]["compiled_plane_arena"]["leaves"][1]["output_start"] = 0
    with pytest.raises(ValueError, match="leaf bindings"):
        _stage_evaluator_set(drift)


def test_direct_source_paths_follow_nested_lane_prefixes() -> None:
    serialized = _stage_evaluator_set(_set())
    prefixed = _prefix_evaluator_payload_paths(serialized, "lane-7")
    stage = prefixed["stages"][0]
    assert stage["evaluator"]["chunks"][0]["application_path"] == (
        "lane-7/evaluators/left.symjit"
    )
    assert stage["evaluator"]["chunks"][0]["plane_application"][
        "application_path"
    ] == ("lane-7/evaluators/left.plane.symjit")
    assert stage["compiled_plane_arena"]["leaves"][0]["application_path"] == (
        "lane-7/evaluators/left.plane.symjit"
    )

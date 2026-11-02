# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import re
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/release/prepare_selftest_fixture.py"
FIXTURE = ROOT / "src/pyamplicol/assets/selftest/portable-64le"
ARENA_CAPABILITY = "compiled-plane-arena-v1"
SOURCE_APPLICATION_ABI = "symjit-application-storage-v3"
PLANE_APPLICATION_ABI = "pyamplicol-symjit-plane-application-v2"
DIRECT_APPLICATION_ABI = "pyamplicol-compiled-plane-kernel-v2"
SYMJIT_CAPABILITY = "symjit.application.complex-f64.v1"
SELFTEST_API_DRIVERS = (
    Path("python/check_standalone.py"),
    Path("c/check_standalone.c"),
    Path("cpp/check_standalone.cpp"),
    Path("fortran/check_standalone.f90"),
    Path("rust/check_standalone.rs"),
)


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("prepare_selftest_fixture", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _portable_stage(
    *,
    amplitude: bool,
    optimization_level: int = 2,
) -> dict[str, object]:
    application_path = (
        "evaluators/amplitude.symjit" if amplitude else "evaluators/current.symjit"
    )
    plane_application_path = application_path.replace(
        ".symjit",
        ".plane.symjit",
    )
    arena = "amplitude" if amplitude else "current"
    output_component = 0 if amplitude else 2
    evaluator = {
        "kind": "symjit-application-evaluator",
        "runtime_capability": SYMJIT_CAPABILITY,
        "application_abi": SOURCE_APPLICATION_ABI,
        "application_path": application_path,
        "plane_application": {
            "application_path": plane_application_path,
            "application_abi": PLANE_APPLICATION_ABI,
            "storage_abi": SOURCE_APPLICATION_ABI,
            "element_layout": "split-complex-plane-major",
            "descriptor_order": "inputs-re-im-then-outputs-re-im",
            "input_complex_count": 1,
            "output_complex_count": 1,
            "input_plane_count": 2,
            "output_plane_count": 2,
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
                f"instructions:{arena}".encode()
            ).hexdigest(),
            "target": {"word_bits": 64, "endianness": "little"},
        },
        "compiler_type": "native",
        "translation_mode": "indirect",
        "optimization_level": optimization_level,
        "element_layout": "complex-f64",
        "batch_layout": "row-major",
        "input_len": 1,
        "output_len": 1,
    }
    input_component = {
        "parameter_index": 0,
        "kind": "value",
        "source_id": 0,
        "component": 0,
        "global_component": 0,
        "real_valued": False,
    }
    output_slot = {
        "output_start": 0,
        "output_stop": 1,
        "component_start": output_component,
        "component_stop": output_component + 1,
    }
    return {
        "stage_kind": "amplitude-roots" if amplitude else "current-combine",
        "parameter_layout": "stage-local-value-momentum",
        "parameter_count": 1,
        "output_length": 1,
        "input_components": [input_component],
        "output_slots": [output_slot],
        "evaluator": evaluator,
        "compiled_plane_arena": {
            "schema_version": 1,
            "kind": "compiled-plane-arena-stage",
            "application_abi": DIRECT_APPLICATION_ABI,
            "source_application_abi": PLANE_APPLICATION_ABI,
            "element_layout": "split-complex-component-major",
            "output_operation": "overwrite",
            "output_factor": "identity",
            "input_output_aliasing": "forbidden",
            "output_output_aliasing": "forbidden",
            "input_bindings": [input_component],
            "output_bindings": [
                {
                    "output_index": 0,
                    "arena": arena,
                    "component": output_component,
                }
            ],
            "leaves": [
                {
                    "application_path": plane_application_path,
                    "source_application_abi": PLANE_APPLICATION_ABI,
                    "optimization_level": optimization_level,
                    "direct_codegen_optimization_level": optimization_level,
                    "input_len": 1,
                    "output_len": 1,
                    "input_indices": [0],
                    "output_start": 0,
                    "output_stop": 1,
                }
            ],
        },
    }


def _portable_execution(
    *,
    optimization_level: int = 2,
) -> dict[str, object]:
    return {
        "required_runtime_capabilities": [
            ARENA_CAPABILITY,
            SYMJIT_CAPABILITY,
        ],
        "compiled": {
            "kind": "generic-dag-stage-blueprint",
            "stage_evaluators": {
                "kind": "generic-dag-stage-evaluator-artifacts",
                "required_runtime_capabilities": [
                    ARENA_CAPABILITY,
                    SYMJIT_CAPABILITY,
                ],
                "stage_count": 2,
                "stages": [
                    _portable_stage(
                        amplitude=False,
                        optimization_level=optimization_level,
                    )
                ],
                "amplitude_stage": _portable_stage(
                    amplitude=True,
                    optimization_level=optimization_level,
                ),
            },
        },
    }


def _portable_artifact_manifest() -> dict[str, object]:
    capabilities = [ARENA_CAPABILITY, SYMJIT_CAPABILITY]
    return {
        "runtime": {"required_runtime_capabilities": list(capabilities)},
        "processes": [
            {"required_runtime_capabilities": list(capabilities)},
        ],
    }


def _write_portable_execution(
    tmp_path: Path,
    execution: dict[str, object],
) -> dict[str, object]:
    relative = Path("processes/example/execution.json")
    execution_path = tmp_path / relative
    execution_path.parent.mkdir(parents=True)
    execution_path.write_text(json.dumps(execution), encoding="utf-8")
    return {
        "payloads": [
            {
                "role": "evaluator-manifest",
                "path": relative.as_posix(),
            }
        ]
    }


def test_resolved_reduction_accepts_floating_point_roundoff() -> None:
    module = _module()

    assert module._complex_sequences_close(
        [0.3298449642209169 + 0j],
        [0.32984496422091686 + 0j],
    )
    assert not module._complex_sequences_close([1.0 + 0j], [1.001 + 0j])
    assert not module._complex_sequences_close([1.0 + 0j], [])


def test_generation_output_sanitizer_accepts_programmatic_config() -> None:
    module = _module()

    programmatic = '[generation]\nmode = "error"\n'
    assert module._sanitize_generation_output(programmatic, "x") == programmatic
    assert (
        module._sanitize_generation_output(
            '[generation]\noutput = "/private/tmp/build"\n', "x"
        )
        == '[generation]\noutput = "."\n'
    )


def test_generation_output_sanitizer_rejects_ambiguous_config() -> None:
    module = _module()

    text = 'output = "/first"\noutput = "/second"\n'
    with pytest.raises(RuntimeError, match="multiple generation outputs"):
        module._sanitize_generation_output(text, "configuration-effective")


def test_portable_manifest_retargeting_rewrites_every_target_tag() -> None:
    module = _module()
    manifest = {
        "producer": {
            "target": {"triple": "aarch64-apple-darwin", "cpu_features": ["x"]}
        },
        "payloads": [
            {"path": "metadata.json"},
            {
                "path": "stage.symjit",
                "target": {
                    "triple": "aarch64-apple-darwin",
                    "cpu_features": ["neon"],
                },
            },
        ],
    }

    module._retarget_portable_manifest(
        manifest,
        source_target="aarch64-apple-darwin",
    )

    expected_target = {"triple": "portable-64le", "cpu_features": []}
    assert manifest["producer"]["target"] == expected_target
    assert manifest["payloads"][1]["target"] == expected_target


def test_source_selftest_fixture_is_one_portable_64bit_template() -> None:
    module = _module()
    expected = json.loads((FIXTURE / "expected.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (FIXTURE / "artifact/artifact.json").read_text(encoding="utf-8")
    )

    assert expected["target"] == module.PORTABLE_TEMPLATE
    assert expected["compatible_targets"] == list(module.COMPATIBLE_TARGETS)
    assert expected["serialization"] == {
        "endianness": "little",
        "kind": "symjit-application-mir-v3",
        "load_behavior": "recompile-mir-for-loading-host",
        "source_optimization_level": 2,
        "word_size_bits": 64,
    }
    assert manifest["producer"]["target"] == {
        "cpu_features": [],
        "triple": module.PORTABLE_TEMPLATE,
    }
    assert (
        manifest["extensions"]["artifact_identity"] == module.ARTIFACT_IDENTITY_CONTRACT
    )
    assert {
        payload["target"]["triple"]
        for payload in manifest["payloads"]
        if "target" in payload
    } == {module.PORTABLE_TEMPLATE}
    assert manifest["artifact_id"] == module._artifact_id(manifest)
    assert (
        module.validate_portable_artifact_capabilities(
            manifest,
            context="tracked portable self-test artifact",
        )
        > 0
    )
    assert (
        module._validate_portable_evaluator_configuration(
            FIXTURE / "artifact",
            manifest,
        )
        > 0
    )


def test_selftest_api_drivers_match_the_public_templates() -> None:
    template_root = ROOT / "src/pyamplicol/assets/api_templates"
    artifact = FIXTURE / "artifact"
    manifest = json.loads((artifact / "artifact.json").read_text(encoding="utf-8"))
    payloads = {
        payload["path"]: payload
        for payload in manifest["payloads"]
        if payload.get("role") == "api-source"
    }
    for relative in SELFTEST_API_DRIVERS:
        expected = (template_root / relative).read_bytes()
        artifact_relative = Path("API") / relative
        actual = (artifact / artifact_relative).read_bytes()
        payload = payloads[artifact_relative.as_posix()]

        assert actual == expected
        assert payload["size_bytes"] == len(actual)
        assert payload["sha256"] == hashlib.sha256(actual).hexdigest()


@pytest.mark.parametrize("optimization_level", (0, 1, 3))
def test_portable_fixture_requires_its_canonical_o2_optimization_level(
    tmp_path: Path,
    optimization_level: int,
) -> None:
    module = _module()
    manifest = _write_portable_execution(
        tmp_path,
        _portable_execution(optimization_level=optimization_level),
    )

    with pytest.raises(RuntimeError, match="optimization level 2"):
        module._validate_portable_evaluator_configuration(tmp_path, manifest)


def test_portable_fixture_accepts_portable_o2_mir(tmp_path: Path) -> None:
    module = _module()
    manifest = _write_portable_execution(tmp_path, _portable_execution())

    assert module._validate_portable_evaluator_configuration(tmp_path, manifest) == 2


@pytest.mark.parametrize("capability_owner", ("execution", "stage_evaluators"))
@pytest.mark.parametrize(
    "capability",
    (ARENA_CAPABILITY, SYMJIT_CAPABILITY),
)
def test_portable_fixture_requires_compiled_runtime_capabilities(
    tmp_path: Path,
    capability_owner: str,
    capability: str,
) -> None:
    module = _module()
    execution = _portable_execution()
    if capability_owner == "execution":
        capabilities = execution["required_runtime_capabilities"]
    else:
        capabilities = execution["compiled"]["stage_evaluators"][
            "required_runtime_capabilities"
        ]
    capabilities.remove(capability)
    manifest = _write_portable_execution(tmp_path, execution)

    with pytest.raises(RuntimeError, match=rf"must require {re.escape(capability)}"):
        module._validate_portable_evaluator_configuration(tmp_path, manifest)


@pytest.mark.parametrize("capability_owner", ("artifact_runtime", "process"))
@pytest.mark.parametrize(
    "capability",
    (ARENA_CAPABILITY, SYMJIT_CAPABILITY),
)
def test_portable_fixture_requires_outer_runtime_capabilities(
    capability_owner: str,
    capability: str,
) -> None:
    module = _module()
    manifest = _portable_artifact_manifest()
    owner = (
        manifest["runtime"]
        if capability_owner == "artifact_runtime"
        else manifest["processes"][0]
    )
    owner["required_runtime_capabilities"].remove(capability)

    with pytest.raises(RuntimeError, match=rf"must require {re.escape(capability)}"):
        module.validate_portable_artifact_capabilities(
            manifest,
            context="portable artifact",
        )


@pytest.mark.parametrize("stage_kind", ("current", "amplitude"))
def test_portable_fixture_requires_every_arena_stage_descriptor(
    tmp_path: Path,
    stage_kind: str,
) -> None:
    module = _module()
    execution = _portable_execution()
    stage_set = execution["compiled"]["stage_evaluators"]
    stage = (
        stage_set["stages"][0]
        if stage_kind == "current"
        else stage_set["amplitude_stage"]
    )
    del stage["compiled_plane_arena"]
    manifest = _write_portable_execution(tmp_path, execution)

    with pytest.raises(RuntimeError, match="complete compiled_plane_arena"):
        module._validate_portable_evaluator_configuration(tmp_path, manifest)


def test_portable_fixture_rejects_incomplete_arena_bindings(tmp_path: Path) -> None:
    module = _module()
    execution = _portable_execution()
    descriptor = execution["compiled"]["stage_evaluators"]["stages"][0][
        "compiled_plane_arena"
    ]
    descriptor["input_bindings"] = []
    manifest = _write_portable_execution(tmp_path, execution)

    with pytest.raises(RuntimeError, match="input bindings are incomplete"):
        module._validate_portable_evaluator_configuration(tmp_path, manifest)


def test_portable_fixture_requires_plane_codegen_optimization_match(
    tmp_path: Path,
) -> None:
    module = _module()
    execution = _portable_execution()
    descriptor = execution["compiled"]["stage_evaluators"]["amplitude_stage"][
        "compiled_plane_arena"
    ]
    descriptor["leaves"][0]["direct_codegen_optimization_level"] = 3
    manifest = _write_portable_execution(tmp_path, execution)

    with pytest.raises(RuntimeError, match=r"direct codegen optimization.*plane"):
        module._validate_portable_evaluator_configuration(tmp_path, manifest)


def test_portable_fixture_binds_arena_leaf_to_o2_source(tmp_path: Path) -> None:
    module = _module()
    execution = _portable_execution()
    descriptor = execution["compiled"]["stage_evaluators"]["stages"][0][
        "compiled_plane_arena"
    ]
    descriptor["leaves"][0]["application_path"] = "evaluators/drift.symjit"
    manifest = _write_portable_execution(tmp_path, execution)

    with pytest.raises(RuntimeError, match="does not bind its SymJIT source"):
        module._validate_portable_evaluator_configuration(tmp_path, manifest)


def test_portable_fixture_validates_nested_selector_executions(
    tmp_path: Path,
) -> None:
    module = _module()
    execution = _portable_execution()
    nested = deepcopy(execution)
    descriptor = nested["compiled"]["stage_evaluators"]["amplitude_stage"][
        "compiled_plane_arena"
    ]
    descriptor["leaves"][0]["direct_codegen_optimization_level"] = 3
    execution["helicity_sum_execution"] = nested
    manifest = _write_portable_execution(tmp_path, execution)

    with pytest.raises(RuntimeError, match=r"direct codegen optimization.*plane"):
        module._validate_portable_evaluator_configuration(tmp_path, manifest)


def test_compiled_model_version_normalization_refreshes_payload(
    tmp_path: Path,
) -> None:
    module = _module()
    artifact = tmp_path / "artifact"
    compiled_path = artifact / "model" / "compiled-model.json"
    compiled_path.parent.mkdir(parents=True)
    compiled_path.write_text(
        json.dumps({"producer": {"pyamplicol": "old"}}),
        encoding="utf-8",
    )
    manifest = {
        "payloads": [
            {
                "path": "model/compiled-model.json",
                "role": "compiled-model",
                "sha256": "stale",
                "size_bytes": 0,
            }
        ]
    }

    module._normalize_compiled_model_version(
        artifact,
        manifest,
        version="0.1.0.dev0+candidate.test",
    )

    data = compiled_path.read_bytes()
    compiled = json.loads(data)
    payload = manifest["payloads"][0]
    assert compiled["producer"]["pyamplicol"] == "0.1.0.dev0+candidate.test"
    assert payload["sha256"] == hashlib.sha256(data).hexdigest()
    assert payload["size_bytes"] == len(data)


def test_symbolica_fallback_stripping_rewrites_packed_container(
    tmp_path: Path,
) -> None:
    module = _module()
    from pyamplicol.generation.evaluator_container import (
        PacbinMemberKind,
        PacbinMemberSource,
        PacbinReader,
        write_pacbin_atomic,
    )

    artifact = tmp_path / "artifact"
    execution_path = artifact / "processes/example/execution.json"
    execution_path.parent.mkdir(parents=True)
    execution_path.write_text(
        json.dumps(
            {
                "compiled": {
                    "evaluator": {
                        "kind": "symjit-application-evaluator",
                        "evaluator_state_path": "chunk.evaluator.bin",
                        "evaluator_state_runtime_capability": "symbolica",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    container_path = artifact / "evaluators.pacbin"
    index = write_pacbin_atomic(
        container_path,
        (
            PacbinMemberSource(
                "processes/example/chunk.evaluator.bin",
                PacbinMemberKind.SYMBOLICA_EXACT_STATE,
                io.BytesIO(b"exact"),
            ),
            PacbinMemberSource(
                "processes/example/chunk.symjit",
                PacbinMemberKind.SYMJIT_APPLICATION,
                io.BytesIO(b"runtime"),
            ),
        ),
    )
    manifest = {
        "extensions": {
            "evaluator_payload_container": {
                "path": "evaluators.pacbin",
                "member_count": 2,
                "unpacked_size_bytes": 12,
                "index_sha256": index.index_sha256,
            }
        },
        "payloads": [
            {
                "path": "processes/example/execution.json",
                "role": "evaluator-manifest",
                "sha256": "stale",
                "size_bytes": 0,
            },
            {
                "path": "evaluators.pacbin",
                "role": "evaluator-state",
                "sha256": "stale",
                "size_bytes": 0,
            },
        ],
    }

    module._strip_symbolica_fallbacks(artifact, manifest)

    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    evaluator = execution["compiled"]["evaluator"]
    assert evaluator["evaluator_state_path"] is None
    assert evaluator["evaluator_state_runtime_capability"] is None
    with PacbinReader.open(container_path) as reader:
        assert [member.logical_path for member in reader.members] == [
            "processes/example/chunk.symjit"
        ]
        extension = manifest["extensions"]["evaluator_payload_container"]
        assert extension["member_count"] == 1
        assert extension["unpacked_size_bytes"] == len(b"runtime")
        assert extension["index_sha256"] == reader.index.index_sha256
    payload = next(
        payload
        for payload in manifest["payloads"]
        if payload["path"] == "evaluators.pacbin"
    )
    assert payload["sha256"] == hashlib.sha256(container_path.read_bytes()).hexdigest()
    assert payload["size_bytes"] == container_path.stat().st_size


def test_source_selftest_compiled_model_matches_active_compiler_sources() -> None:
    module = _module()
    manifest = json.loads(
        (FIXTURE / "artifact/artifact.json").read_text(encoding="utf-8")
    )
    _payload, relative = module._compiled_model_payload(manifest)
    compiled = json.loads((FIXTURE / "artifact" / relative).read_text(encoding="utf-8"))

    from pyamplicol.models import loading

    expected_producer = loading.compiler_fingerprint()
    actual_producer = dict(compiled["producer"])
    # The portable source template is retargeted to the concrete candidate or
    # release version by the wheel overlay. The full compiler-source hash is
    # provenance rather than a compatibility boundary, just as it is for the
    # packaged prepared-model store.
    for provenance_field in ("pyamplicol", "model_compiler_sha256"):
        expected_producer.pop(provenance_field)
        actual_producer.pop(provenance_field)

    assert compiled["kind"] == loading.COMPILED_MODEL_KIND
    assert compiled["schema_version"] == loading.COMPILED_MODEL_SCHEMA_VERSION
    assert compiled["model_compiler_version"] == loading.MODEL_COMPILER_VERSION
    assert actual_producer == expected_producer
    assert compiled["source"]["digest"] == loading._source_digest(
        "built-in-sm",
        "built-in-sm",
    )


def test_staged_selftest_fixture_loads_with_the_current_native_runtime() -> None:
    native = pytest.importorskip("pyamplicol._rusticol")
    from pyamplicol import Runtime

    fixture = FIXTURE.parent / str(native.target_info().triple)
    if not fixture.is_dir():
        pytest.skip("the source runtime has not staged its target self-test fixture")
    expected = json.loads((fixture / "expected.json").read_text(encoding="utf-8"))
    runtime = Runtime.load(fixture / expected["artifact_path"], mute_warnings=True)

    total = runtime.evaluate(expected["momenta"])
    expected_total = [complex(real, imag) for real, imag in expected["total"]]

    assert total == pytest.approx(expected_total, rel=1.0e-12, abs=1.0e-15)

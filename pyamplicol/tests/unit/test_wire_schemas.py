# SPDX-License-Identifier: 0BSD

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jsonschema
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _schema(name: str) -> dict[str, object]:
    return json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8"))


def test_wire_schemas_are_strict_and_versioned() -> None:
    artifact = _schema("artifact-manifest-v3.schema.json")
    physics = _schema("runtime-physics-v1.schema.json")

    assert artifact["additionalProperties"] is False
    assert artifact["properties"]["schema_version"]["const"] == 3
    assert (
        artifact["$defs"]["version_set"]["properties"]["runtime_physics"]["const"] == 1
    )

    assert physics["oneOf"] == [
        {"$ref": "#/$defs/resolved_physics"},
        {"$ref": "#/$defs/on_the_fly_public_metadata"},
    ]
    resolved = physics["$defs"]["resolved_physics"]
    compact = physics["$defs"]["on_the_fly_public_metadata"]
    assert resolved["additionalProperties"] is False
    assert resolved["properties"]["schema_version"]["const"] == 1
    assert resolved["properties"]["kind"]["const"] == (
        "pyamplicol-resolved-physics"
    )
    assert compact["additionalProperties"] is False
    assert compact["properties"]["kind"]["const"] == (
        "pyamplicol-on-the-fly-public-metadata"
    )
    assert (
        physics["$defs"]["selectors"]["properties"]["contracted_color"]["const"]
        is False
    )


def test_runtime_physics_schema_accepts_only_strict_compact_otf_metadata() -> None:
    physics = _schema("runtime-physics-v1.schema.json")
    compact = {
        "schema_version": 1,
        "kind": "pyamplicol-on-the-fly-public-metadata",
        "process_id": "six_fermions",
        "process": "d d~ > u u~ u u~",
        "color_accuracy": "lc",
        "external_particles": [
            {
                "index": index,
                "label": index + 1,
                "particle": particle,
                "pdg": pdg,
                "role": "initial" if index < 2 else "final",
                "momentum_slot": index,
                "momentum_components": ["E", "px", "py", "pz"],
            }
            for index, (particle, pdg) in enumerate(
                (
                    ("d", 1),
                    ("d~", -1),
                    ("u", 2),
                    ("u~", -2),
                    ("u", 2),
                    ("u~", -2),
                )
            )
        ],
        "model_parameters": [],
    }
    validator = jsonschema.Draft202012Validator(physics)
    validator.validate(compact)

    invented = copy.deepcopy(compact)
    invented["helicities"] = []
    assert list(validator.iter_errors(invented))


def test_artifact_manifest_declares_security_critical_payload_fields() -> None:
    artifact = _schema("artifact-manifest-v3.schema.json")
    required = set(artifact["$defs"]["payload"]["required"])
    assert {
        "path",
        "role",
        "media_type",
        "size_bytes",
        "sha256",
        "executable",
    } <= required


def _valid_manifest() -> dict[str, object]:
    return {
        "schema_version": 3,
        "kind": "pyamplicol-process",
        "artifact_id": "0" * 64,
        "created_utc": "2026-07-15T14:43:42Z",
        "producer": {
            "distribution": "pyamplicol",
            "version": "0.1.0",
            "versions": {
                "python_api": 1,
                "toml": 1,
                "compiled_model": 1,
                "process_artifact": 3,
                "runtime_physics": 1,
                "symbolica_serialization": "test",
                "c_abi": 1,
            },
            "target": {
                "triple": "aarch64-apple-darwin",
                "cpu_features": [],
            },
            "git_revision": "1" * 40,
        },
        "model": {
            "name": "built-in-sm",
            "source_kind": "built-in-sm",
            "content_sha256": "2" * 64,
            "compiled_schema_version": 1,
        },
        "configuration": {
            "toml_schema_version": 1,
            "requested_path": "config/requested.toml",
            "effective_path": "config/effective.toml",
            "adjustments": [],
        },
        "processes": [
            {
                "id": "dd_to_z",
                "expression": "d d~ > z",
                "color_accuracy": "lc",
                "external_pdgs": [1, -1, 23],
                "physics_path": "physics/dd_to_z.json",
                "required_runtime_capabilities": ["symjit.application.complex-f64.v1"],
                "aliases": [],
            }
        ],
        "runtime": {
            "engine": "rusticol",
            "engine_version": "0.1.0",
            "evaluator_manifest_path": "runtime/evaluators.json",
            "api_bundle_path": None,
            "required_runtime_capabilities": ["symjit.application.complex-f64.v1"],
        },
        "payloads": [
            {
                "path": "physics/dd_to_z.json",
                "role": "runtime-physics",
                "media_type": "application/json",
                "size_bytes": 2,
                "sha256": "3" * 64,
                "executable": False,
            },
            {
                "path": "process-groups.json",
                "role": "process-groups",
                "media_type": "application/json",
                "size_bytes": 2,
                "sha256": "4" * 64,
                "executable": False,
            },
        ],
        "dependencies": [],
    }


@pytest.mark.parametrize("source_kind", ("built-in-sm", "built-in-sm-heft"))
def test_artifact_schema_accepts_a_positive_v3_manifest(source_kind: str) -> None:
    manifest = _valid_manifest()
    manifest["model"]["source_kind"] = source_kind
    jsonschema.Draft202012Validator(
        _schema("artifact-manifest-v3.schema.json")
    ).validate(manifest)


def test_artifact_schema_accepts_native_identity_without_git_revision() -> None:
    manifest = _valid_manifest()
    manifest["producer"].pop("git_revision")
    manifest["producer"]["native_build_inputs_sha256"] = "3" * 64

    jsonschema.Draft202012Validator(
        _schema("artifact-manifest-v3.schema.json")
    ).validate(manifest)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value["producer"].update(git_revision="4" * 64),
        lambda value: value["producer"].update(
            native_build_inputs_sha256="4" * 63 + "z"
        ),
        lambda value: value["payloads"][0].update(path="../escape"),
        lambda value: value["payloads"][0].update(
            role="configuration-requested",
            executable=True,
        ),
        lambda value: value["payloads"][0].update(
            role="evaluator-state",
            executable=True,
        ),
    ),
)
def test_artifact_schema_rejects_security_contract_violations(
    mutation: Callable[[dict[str, Any]], None],
) -> None:
    manifest = copy.deepcopy(_valid_manifest())
    mutation(manifest)
    validator = jsonschema.Draft202012Validator(
        _schema("artifact-manifest-v3.schema.json")
    )
    assert list(validator.iter_errors(manifest))

# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from tools.developer import reference_capture as CAPTURE

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "developer" / "reference_fixture_v2.py"


def _module():
    spec = importlib.util.spec_from_file_location("reference_fixture_v2", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


REFERENCE = _module()


def test_compatibility_facade_imports_from_outside_checkout(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    for name in (
        "ReferenceFixtureError",
        "Tolerances",
        "load_reference_fixture",
        "parse_reference_fixture",
        "physics_case_sha256",
        "reduction_plan_sha256",
    ):
        assert hasattr(REFERENCE, name)


@pytest.mark.parametrize(
    "name",
    [
        "reference-fixture-bundle-v1.schema.json",
        "reference-physics-v2.schema.json",
        "reference-oracle-evidence-v2.schema.json",
    ],
)
def test_reference_v2_json_schemas_are_valid_draft_2020_12(name: str) -> None:
    schema = json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8"))

    Draft202012Validator.check_schema(schema)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _input_sha(point: dict[str, Any]) -> str:
    payload = {
        "arithmetic_precision_bits": point["arithmetic_precision_bits"],
        "certified_decimal_digits": point["certified_decimal_digits"],
        "masses": point["masses"],
        "momenta": point["momenta"],
        "process_id": point["process_id"],
        "round_trip_decimal_digits": point["round_trip_decimal_digits"],
        "sqrt_s": point["sqrt_s"],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _point(point_id: str, point_class: str, seed: int | None) -> dict[str, Any]:
    generic_momenta = {
        "point:generic-0": [
            ["600", "0", "0", "600"],
            ["600", "0", "0", "-600"],
            ["300", "0", "0", "0"],
            ["450", "450", "0", "0"],
            ["450", "-450", "0", "0"],
        ],
        "point:generic-1": [
            ["600", "0", "0", "600"],
            ["600", "0", "0", "-600"],
            ["300", "0", "0", "0"],
            ["450", "0", "450", "0"],
            ["450", "0", "-450", "0"],
        ],
        "point:generic-2": [
            ["600", "0", "0", "600"],
            ["600", "0", "0", "-600"],
            ["300", "0", "0", "0"],
            ["450", "0", "0", "450"],
            ["450", "0", "0", "-450"],
        ],
    }
    is_stress = point_class == "stress"
    return {
        "id": point_id,
        "process_id": "process:dd-zgg",
        "class": point_class,
        "algorithm": {
            "name": "deterministic-rambo" if seed is not None else "near-collinear",
            "version": "1",
            "rng": "PCG64" if seed is not None else None,
            "seed": seed,
        },
        "sqrt_s": "1000" if is_stress else "1200",
        "momenta": (
            [
                ["500", "0", "0", "500"],
                ["500", "0", "0", "-500"],
                ["545", "455", "0", "0"],
                ["454.999999", "-454.999999", "0", "0"],
                ["0.000001", "-0.000001", "0", "0"],
            ]
            if is_stress
            else generic_momenta[point_id]
        ),
        "masses": ["0", "0", "300", "0", "0"],
        "arithmetic_precision_bits": 266 if is_stress else 128,
        "round_trip_decimal_digits": 80 if is_stress else 32,
        "certified_decimal_digits": 40 if is_stress else 24,
        "stress_metric": (
            {
                "kind": "minimum-final-energy-fraction",
                "value": "0.000000001",
            }
            if is_stress
            else None
        ),
    }


def _refresh_physics_case_hashes(
    fixture: dict[str, Any],
    evidence: dict[str, Any],
) -> None:
    case_hashes: dict[str, str] = {}
    for case in fixture["cases"]:
        case["reduction"]["plan_sha256"] = REFERENCE.reduction_plan_sha256(
            case["reduction"]
        )
        digest = REFERENCE.physics_case_sha256(fixture, case["id"])
        case["physics_case_sha256"] = digest
        case_hashes[case["id"]] = digest
    for record in evidence["records"]:
        case_id = record["case_id"]
        if case_id in case_hashes:
            record["physics_case_sha256"] = case_hashes[case_id]
        record["evidence_record_sha256"] = REFERENCE.evidence_record_sha256(record)


def _valid_payloads() -> tuple[dict[str, Any], dict[str, Any]]:
    points = [
        _point("point:generic-0", "generic", 11),
        _point("point:generic-1", "generic", 12),
        _point("point:generic-2", "generic", 13),
        _point("point:stress-0", "stress", None),
    ]
    observations = [
        {
            "point_id": point["id"],
            "arithmetic_precision_bits": point["arithmetic_precision_bits"],
            "round_trip_decimal_digits": point["round_trip_decimal_digits"],
            "certified_decimal_digits": point["certified_decimal_digits"],
            "values": [
                ["1", "2"],
                ["2", "4"],
                ["3", "6"],
                ["0", "0"],
            ],
            "total": "18",
            "evidence_refs": [f"evidence:{point['id'].split(':', 1)[1]}"],
        }
        for point in points
    ]
    fixture = {
        "fixture_schema_version": 2,
        "kind": "pyamplicol-reference-physics",
        "fixture_id": "fixture:compact-lc",
        "provenance": {
            "source_repository": "https://github.com/mg5amcnlo/pyamplicol",
            "source_revision": "1" * 40,
            "source_tree_sha256": _sha("source-tree"),
            "captured_at": "2026-07-16T12:00:00Z",
            "capture_command": ["pyamplicol", "fixture", "capture"],
            "working_tree_clean": True,
            "memory_watchdog_gb": 30,
        },
        "dependencies": [
            {
                "id": "dependency:oracle",
                "name": "independent analytic oracle",
                "version": "1.0",
                "revision": "oracle-r1",
                "content_sha256": _sha("oracle-dependency"),
                "serialization_abi": None,
                "license": "0BSD",
            }
        ],
        "models": [
            {
                "id": "model:builtin-sm",
                "name": "built-in-sm",
                "source_kind": "built-in-sm",
                "content_sha256": _sha("model-source"),
                "compiled_model_sha256": _sha("compiled-model"),
                "compiled_schema_version": 1,
                "restriction": None,
                "dependency_ids": [],
                "parameter_defaults": {
                    "normalization.alpha_s": {"real": "0.118", "imag": "0"}
                },
            }
        ],
        "processes": [
            {
                "id": "process:dd-zgg",
                "expression": "d d~ > z g g",
                "external_pdgs": [1, -1, 23, 21, 21],
                "external_labels": [1, 2, 3, 4, 5],
                "external_leg_ids": [
                    "leg:d-in",
                    "leg:dbar-in",
                    "leg:z-out",
                    "leg:g1-out",
                    "leg:g2-out",
                ],
                "external_spins": [2, 2, 1, 1, 1],
                "external_colors": [3, -3, 1, 8, 8],
                "external_masses": ["0", "0", "300", "0", "0"],
                "external_helicity_domains": [
                    [-1, 1],
                    [-1, 1],
                    [0],
                    [0],
                    [0],
                ],
                "initial_state_count": 2,
                "alias_of": None,
                "final_state_permutation": None,
            },
            {
                "id": "process:dd-ggz",
                "expression": "d d~ > g g z",
                "external_pdgs": [1, -1, 21, 21, 23],
                "external_labels": [1, 2, 3, 4, 5],
                "external_leg_ids": [
                    "leg:d-in",
                    "leg:dbar-in",
                    "leg:g1-out",
                    "leg:g2-out",
                    "leg:z-out",
                ],
                "external_spins": [2, 2, 1, 1, 1],
                "external_colors": [3, -3, 8, 8, 1],
                "external_masses": ["0", "0", "0", "0", "300"],
                "external_helicity_domains": [
                    [-1, 1],
                    [-1, 1],
                    [0],
                    [0],
                    [0],
                ],
                "initial_state_count": 2,
                "alias_of": "process:dd-zgg",
                "final_state_permutation": [1, 2, 0],
            },
        ],
        "points": points,
        "cases": [
            {
                "id": "case:dd-zgg-lc",
                "case_kind": "substantive",
                "model_id": "model:builtin-sm",
                "process_id": "process:dd-zgg",
                "color_accuracy": "lc",
                "point_policy": "standard",
                "point_ids": [point["id"] for point in points],
                "coverage": {
                    "helicities": "complete",
                    "color": "complete",
                    "color_kind": "physical-lc-flows",
                    "helicity_count": 4,
                    "color_component_count": 2,
                    "structural_zero_helicity_count": 1,
                },
                "selectors": {
                    "helicity": True,
                    "color_flow": True,
                    "omitted_helicity": "all-components",
                    "omitted_color": "all-components",
                },
                "normalization": {
                    "average_factor": "36",
                    "color_factor": "27",
                    "identical_factor": "2",
                    "global_coupling_factor": "1",
                    "quark_line_partner_factor": "1",
                    "couplings_in_stage_evaluators": True,
                },
                "topology": {
                    "currents": 17,
                    "interactions": 31,
                    "roots": 4,
                    "reduction_groups": 2,
                },
                "artifact_physics_sha256": _sha("physics-payload"),
                "artifact_execution_sha256": _sha("execution-payload"),
                "physics_case_sha256": "0" * 64,
                "axes": {
                    "helicities": [
                        {
                            "id": "helicity:active",
                            "index": 0,
                            "values": [-1, -1, 0, 0, 0],
                            "computed": True,
                            "structural_zero": False,
                            "representative_id": "helicity:active",
                            "coefficient": "1",
                        },
                        {
                            "id": "helicity:folded",
                            "index": 1,
                            "values": [-1, 1, 0, 0, 0],
                            "computed": False,
                            "structural_zero": False,
                            "representative_id": "helicity:active",
                            "coefficient": "2",
                        },
                        {
                            "id": "helicity:second-active",
                            "index": 2,
                            "values": [1, -1, 0, 0, 0],
                            "computed": True,
                            "structural_zero": False,
                            "representative_id": "helicity:second-active",
                            "coefficient": "1",
                        },
                        {
                            "id": "helicity:zero",
                            "index": 3,
                            "values": [1, 1, 0, 0, 0],
                            "computed": False,
                            "structural_zero": True,
                            "representative_id": "helicity:zero",
                            "coefficient": "0",
                        },
                    ],
                    "colors": [
                        {
                            "kind": "lc-flow",
                            "id": "flow:2,4,5,1",
                            "index": 0,
                            "word": [2, 4, 5, 1],
                            "computed": True,
                            "representative_id": "flow:2,4,5,1",
                            "coefficient": "1",
                        },
                        {
                            "kind": "lc-flow",
                            "id": "flow:2,5,4,1",
                            "index": 1,
                            "word": [2, 5, 4, 1],
                            "computed": False,
                            "representative_id": "flow:2,4,5,1",
                            "coefficient": "2",
                        },
                    ],
                },
                "reduction": {
                    "kind": "lc-diagonal",
                    "cell_semantics": "sum-all-contributing-groups",
                    "groups": [
                        {
                            "id": "reduction:0",
                            "representative_helicity_id": "helicity:active",
                            "representative_color_id": "flow:2,4,5,1",
                            "physical_helicity_ids": [
                                "helicity:active",
                                "helicity:folded",
                            ],
                            "physical_color_ids": [
                                "flow:2,4,5,1",
                                "flow:2,5,4,1",
                            ],
                        },
                        {
                            "id": "reduction:1",
                            "representative_helicity_id": ("helicity:second-active"),
                            "representative_color_id": "flow:2,4,5,1",
                            "physical_helicity_ids": ["helicity:second-active"],
                            "physical_color_ids": [
                                "flow:2,4,5,1",
                                "flow:2,5,4,1",
                            ],
                        },
                    ],
                    "plan_sha256": "0" * 64,
                },
                "observations": observations,
            }
        ],
        "evidence_sets": ["oracle:analytic"],
    }
    evidence = {
        "evidence_schema_version": 2,
        "kind": "pyamplicol-reference-oracle-evidence",
        "evidence_set_id": "oracle:analytic",
        "captured_at": "2026-07-16T12:00:00Z",
        "oracle": {
            "id": "oracle:analytic",
            "name": "independent analytic fixture oracle",
            "implementation": "closed-form test expression",
            "revision": "oracle-r1",
            "content_sha256": _sha("oracle-implementation"),
            "independence_statement": (
                "The oracle does not import or execute pyAmpliCol generation code."
            ),
            "validation_profile": "high-precision",
            "tolerance_ceiling": {
                "relative": "0.000000000000000000000001",
                "absolute": "0.000000000000000000000000000001",
            },
        },
        "dependency_ids": ["dependency:oracle"],
        "records": [
            {
                "id": f"evidence:{point['id'].split(':', 1)[1]}",
                "case_id": "case:dd-zgg-lc",
                "point_id": point["id"],
                "independent_of_pyamplicol": True,
                "arithmetic_precision_bits": point["arithmetic_precision_bits"],
                "round_trip_decimal_digits": point["round_trip_decimal_digits"],
                "certified_decimal_digits": point["certified_decimal_digits"],
                "arithmetic": "analytic",
                "coverage": "resolved",
                "helicity_ids": [
                    "helicity:active",
                    "helicity:folded",
                    "helicity:second-active",
                    "helicity:zero",
                ],
                "color_ids": ["flow:2,4,5,1", "flow:2,5,4,1"],
                "observed_total": "18",
                "observed_helicity_totals": None,
                "observed_values": [
                    ["1", "2"],
                    ["2", "4"],
                    ["3", "6"],
                    ["0", "0"],
                ],
                "process_identity": {
                    "expression": "d d~ > z g g",
                    "ordered_external_pdgs": [1, -1, 23, 21, 21],
                    "ordered_external_leg_ids": [
                        "leg:d-in",
                        "leg:dbar-in",
                        "leg:z-out",
                        "leg:g1-out",
                        "leg:g2-out",
                    ],
                    "source_to_row_permutation": [0, 1, 2, 3, 4],
                    "row_id": None,
                    "color_order_count": 2,
                    "ordered_color_legs": [
                        "leg:dbar-in",
                        "leg:g1-out",
                        "leg:g2-out",
                        "leg:d-in",
                    ],
                },
                "input_sha256": _input_sha(point),
                "physics_case_sha256": "0" * 64,
                "oracle_output_sha256": _sha(f"raw:{point['id']}"),
                "command": ["analytic-oracle", point["id"]],
                "tolerances": {"relative": "0", "absolute": "0"},
            }
            for point in points
        ],
    }
    _refresh_physics_case_hashes(fixture, evidence)
    return fixture, evidence


def _parse(
    fixture: dict[str, Any],
    evidence: dict[str, Any],
    *,
    refresh_physics_case_hashes: bool = True,
):
    if refresh_physics_case_hashes:
        _refresh_physics_case_hashes(fixture, evidence)
    return REFERENCE.parse_reference_fixture(fixture, [evidence])


def _degenerate_payloads() -> tuple[dict[str, Any], dict[str, Any]]:
    fixture, evidence = _valid_payloads()
    point = {
        "id": "point:canonical",
        "process_id": "process:dd-z",
        "class": "canonical",
        "algorithm": {
            "name": "exact-two-to-one",
            "version": "1",
            "rng": None,
            "seed": None,
        },
        "sqrt_s": "1000",
        "momenta": [
            ["500", "0", "0", "500"],
            ["500", "0", "0", "-500"],
            ["1000", "0", "0", "0"],
        ],
        "masses": ["0", "0", "1000"],
        "arithmetic_precision_bits": 128,
        "round_trip_decimal_digits": 32,
        "certified_decimal_digits": 24,
        "stress_metric": None,
    }
    fixture["processes"] = [
        {
            "id": "process:dd-z",
            "expression": "d d~ > z",
            "external_pdgs": [1, -1, 23],
            "external_labels": [1, 2, 3],
            "external_leg_ids": ["leg:d-in", "leg:dbar-in", "leg:z-out"],
            "external_spins": [2, 1, 1],
            "external_colors": [3, -3, 1],
            "external_masses": ["0", "0", "1000"],
            "external_helicity_domains": [[-1, 1], [0], [0]],
            "initial_state_count": 2,
            "alias_of": None,
            "final_state_permutation": None,
        }
    ]
    fixture["points"] = [point]
    case = fixture["cases"][0]
    case.update(
        id="case:dd-z-lc",
        process_id="process:dd-z",
        point_policy="degenerate-2to1",
        point_ids=["point:canonical"],
        observations=[
            {
                "point_id": "point:canonical",
                "arithmetic_precision_bits": 128,
                "round_trip_decimal_digits": 32,
                "certified_decimal_digits": 24,
                "values": [["2"], ["0"]],
                "total": "2",
                "evidence_refs": ["evidence:canonical"],
            }
        ],
    )
    case["axes"]["helicities"] = [
        {
            "id": "helicity:active",
            "index": 0,
            "values": [-1, 0, 0],
            "computed": True,
            "structural_zero": False,
            "representative_id": "helicity:active",
            "coefficient": "1",
        },
        {
            "id": "helicity:zero",
            "index": 1,
            "values": [1, 0, 0],
            "computed": False,
            "structural_zero": True,
            "representative_id": "helicity:zero",
            "coefficient": "0",
        },
    ]
    case["axes"]["colors"] = [case["axes"]["colors"][0]]
    case["axes"]["colors"][0].update(
        id="flow:2,1",
        word=[2, 1],
        representative_id="flow:2,1",
    )
    case["coverage"]["helicity_count"] = 2
    case["coverage"]["color_component_count"] = 1
    case["topology"]["reduction_groups"] = 1
    case["reduction"]["groups"] = [
        {
            "id": "reduction:0",
            "representative_helicity_id": "helicity:active",
            "representative_color_id": "flow:2,1",
            "physical_helicity_ids": ["helicity:active"],
            "physical_color_ids": ["flow:2,1"],
        }
    ]
    evidence["records"] = [
        {
            "id": "evidence:canonical",
            "case_id": "case:dd-z-lc",
            "point_id": "point:canonical",
            "independent_of_pyamplicol": True,
            "arithmetic_precision_bits": 128,
            "round_trip_decimal_digits": 32,
            "certified_decimal_digits": 24,
            "arithmetic": "analytic",
            "coverage": "resolved",
            "helicity_ids": ["helicity:active", "helicity:zero"],
            "color_ids": ["flow:2,1"],
            "observed_total": "2",
            "observed_helicity_totals": None,
            "observed_values": [["2"], ["0"]],
            "process_identity": {
                "expression": "d d~ > z",
                "ordered_external_pdgs": [1, -1, 23],
                "ordered_external_leg_ids": [
                    "leg:d-in",
                    "leg:dbar-in",
                    "leg:z-out",
                ],
                "source_to_row_permutation": [0, 1, 2],
                "row_id": None,
                "color_order_count": 1,
                "ordered_color_legs": ["leg:dbar-in", "leg:d-in"],
            },
            "input_sha256": _input_sha(point),
            "physics_case_sha256": "0" * 64,
            "oracle_output_sha256": _sha("raw:canonical"),
            "command": ["analytic-oracle", "point:canonical"],
            "tolerances": {"relative": "0", "absolute": "0"},
        }
    ]
    _refresh_physics_case_hashes(fixture, evidence)
    return fixture, evidence


def _contracted_payloads(
    color_accuracy: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fixture, evidence = _valid_payloads()
    case = fixture["cases"][0]
    case["color_accuracy"] = color_accuracy
    case["coverage"].update(
        color="contracted",
        color_kind="contracted-color",
        color_component_count=1,
    )
    case["selectors"].update(
        color_flow=False,
        omitted_color="contracted-component",
    )
    case["axes"]["colors"] = [
        {
            "kind": "contracted-color",
            "id": "color:contracted",
            "index": 0,
            "description": f"fully contracted {color_accuracy} color",
        }
    ]
    case["reduction"] = {
        "kind": "contracted-color",
        "cell_semantics": "fully-contracted-color",
        "groups": [
            {
                **group,
                "representative_color_id": "color:contracted",
                "physical_color_ids": ["color:contracted"],
            }
            for group in case["reduction"]["groups"]
        ],
        "plan_sha256": "0" * 64,
    }
    for observation in case["observations"]:
        observation["values"] = [["3"], ["6"], ["9"], ["0"]]
    for record in evidence["records"]:
        record["color_ids"] = ["color:contracted"]
        record["observed_values"] = [["3"], ["6"], ["9"], ["0"]]
    _refresh_physics_case_hashes(fixture, evidence)
    return fixture, evidence


def _binary64_degenerate_payloads() -> tuple[dict[str, Any], dict[str, Any]]:
    fixture, evidence = _degenerate_payloads()
    observation = fixture["cases"][0]["observations"][0]
    observation.update(
        arithmetic_precision_bits=53,
        round_trip_decimal_digits=17,
        certified_decimal_digits=15,
    )
    oracle = evidence["oracle"]
    oracle["validation_profile"] = "binary64"
    oracle["tolerance_ceiling"] = {
        "relative": "0.0000000001",
        "absolute": "0.000000000001",
    }
    record = evidence["records"][0]
    record.update(
        arithmetic="binary64",
        arithmetic_precision_bits=53,
        round_trip_decimal_digits=17,
        certified_decimal_digits=15,
    )
    _refresh_physics_case_hashes(fixture, evidence)
    return fixture, evidence


def test_valid_lc_multiflow_fixture_parses_to_frozen_decimal_dtos() -> None:
    fixture_payload, evidence_payload = _valid_payloads()

    fixture = _parse(fixture_payload, evidence_payload)

    point = fixture.point("point:stress-0")
    assert point.runtime_momenta()[0][0] == Decimal("500")
    assert isinstance(point.runtime_momenta()[0][0], Decimal)
    assert point.f64_momenta()[0][0] == 500.0
    assert point.input_sha256() == evidence_payload["records"][3]["input_sha256"]
    assert len(fixture.cases[0].colors) == 2
    assert fixture.processes[1].final_state_permutation == (1, 2, 0)
    with pytest.raises((AttributeError, TypeError)):
        point.id = "mutated"


@pytest.mark.parametrize("bad_value", [1.25, "1.250", "1e-3", "-0"])
def test_numeric_expectations_must_be_canonical_decimal_strings(
    bad_value: object,
) -> None:
    fixture, evidence = _valid_payloads()
    fixture["cases"][0]["observations"][0]["values"][0][0] = bad_value

    with pytest.raises(REFERENCE.ReferenceFixtureError, match="schema violation"):
        _parse(fixture, evidence)


def test_sparse_observation_matrix_is_rejected() -> None:
    fixture, evidence = _valid_payloads()
    fixture["cases"][0]["observations"][0]["values"][0].pop()

    with pytest.raises(REFERENCE.ReferenceFixtureError, match="dense"):
        _parse(fixture, evidence)


def test_missing_axis_is_rejected_against_declared_coverage() -> None:
    fixture, evidence = _valid_payloads()
    fixture["cases"][0]["axes"]["colors"].pop()
    for observation in fixture["cases"][0]["observations"]:
        for row in observation["values"]:
            row.pop()

    with pytest.raises(REFERENCE.ReferenceFixtureError, match="axis count"):
        _parse(fixture, evidence)


def test_structural_zero_must_be_explicit_and_exact() -> None:
    fixture, evidence = _valid_payloads()
    fixture["cases"][0]["observations"][0]["values"][3][0] = "0.1"

    with pytest.raises(REFERENCE.ReferenceFixtureError, match="structural helicity"):
        _parse(fixture, evidence)


def test_aggregate_evidence_must_preserve_structural_zero_exactly() -> None:
    fixture, evidence = _valid_payloads()
    record = evidence["records"][0]
    record.update(
        coverage="helicity-aggregate",
        observed_helicity_totals=[
            "3",
            "6",
            "9",
            "0.0000000000000000000000001",
        ],
        observed_values=None,
        observed_total="18.0000000000000000000000001",
        tolerances={
            "relative": "0",
            "absolute": "0.000000000000000000000001",
        },
    )
    evidence["oracle"]["tolerance_ceiling"]["absolute"] = "0.000000000000000000000001"

    with pytest.raises(REFERENCE.ReferenceFixtureError, match="structural zero"):
        _parse(fixture, evidence)


def test_standard_case_requires_three_generic_and_one_stress_point() -> None:
    fixture, evidence = _valid_payloads()
    fixture["points"][3]["class"] = "generic"
    fixture["points"][3]["stress_metric"] = None

    with pytest.raises(REFERENCE.ReferenceFixtureError, match="three generic"):
        _parse(fixture, evidence)


def test_declared_total_must_equal_exact_decimal_component_sum() -> None:
    fixture, evidence = _valid_payloads()
    fixture["cases"][0]["observations"][0]["total"] = "3"

    with pytest.raises(REFERENCE.ReferenceFixtureError, match="component sum"):
        _parse(fixture, evidence)


def test_component_sum_is_exact_beyond_ambient_decimal_precision() -> None:
    fixture, evidence = _contracted_payloads("full")
    observation = fixture["cases"][0]["observations"][0]
    observation["values"] = [
        ["10000000000000000000000000000"],
        ["20000000000000000000000000000"],
        ["0.0000000000000000000000000001"],
        ["0"],
    ]
    observation["total"] = "30000000000000000000000000000"

    with pytest.raises(REFERENCE.ReferenceFixtureError, match="component sum"):
        _parse(fixture, evidence)


def test_nlc_and_full_cannot_expose_lc_flow_axes() -> None:
    fixture, evidence = _valid_payloads()
    case = fixture["cases"][0]
    case["color_accuracy"] = "nlc"
    case["coverage"].update(color="contracted", color_kind="contracted-color")

    with pytest.raises(REFERENCE.ReferenceFixtureError, match="contracted color axis"):
        _parse(fixture, evidence)


@pytest.mark.parametrize("color_accuracy", ["nlc", "full"])
def test_contracted_nlc_and_full_fixtures_are_valid(color_accuracy: str) -> None:
    fixture, evidence = _contracted_payloads(color_accuracy)

    parsed = _parse(fixture, evidence)

    case = parsed.cases[0]
    assert case.color_accuracy == color_accuracy
    assert len(case.colors) == 1
    assert case.colors[0].id == "color:contracted"


def test_observation_requires_existing_independent_evidence() -> None:
    fixture, evidence = _valid_payloads()
    fixture["cases"][0]["observations"][0]["evidence_refs"] = ["evidence:missing"]

    with pytest.raises(REFERENCE.ReferenceFixtureError, match="unknown evidence"):
        _parse(fixture, evidence)


def test_total_only_evidence_cannot_back_resolved_cells() -> None:
    fixture, evidence = _valid_payloads()
    for record in evidence["records"]:
        record.update(
            coverage="total",
            helicity_ids=[],
            color_ids=[],
            observed_helicity_totals=None,
            observed_values=None,
        )

    with pytest.raises(
        REFERENCE.ReferenceFixtureError,
        match="resolved or complete-color aggregate evidence",
    ):
        _parse(fixture, evidence)


def test_complete_helicity_aggregates_can_certify_multiflow_observation() -> None:
    fixture, evidence = _valid_payloads()
    for record in evidence["records"]:
        record.update(
            coverage="helicity-aggregate",
            observed_helicity_totals=["3", "6", "9", "0"],
            observed_values=None,
        )

    parsed = _parse(fixture, evidence)

    assert len(parsed.cases[0].colors) == 2
    assert all(
        record.coverage == "helicity-aggregate"
        for record in parsed.evidence_sets[0].records
    )


def test_helicity_aggregate_must_cover_complete_color_axis() -> None:
    fixture, evidence = _valid_payloads()
    for record in evidence["records"]:
        record.update(
            coverage="helicity-aggregate",
            color_ids=["flow:2,4,5,1"],
            observed_helicity_totals=["1", "2", "3", "0"],
            observed_values=None,
            observed_total="6",
        )

    with pytest.raises(REFERENCE.ReferenceFixtureError, match="complete color axis"):
        _parse(fixture, evidence)


def test_evidence_cannot_underwrite_more_digits_than_it_certifies() -> None:
    fixture, evidence = _valid_payloads()
    evidence["records"][3]["certified_decimal_digits"] = 21

    with pytest.raises(REFERENCE.ReferenceFixtureError, match="observation claim"):
        _parse(fixture, evidence)


def test_structural_zero_requires_zero_reduction_coefficient() -> None:
    fixture, evidence = _valid_payloads()
    fixture["cases"][0]["axes"]["helicities"][3]["coefficient"] = "1"

    with pytest.raises(REFERENCE.ReferenceFixtureError, match="zero coefficient"):
        _parse(fixture, evidence)


def test_reduction_plan_hash_detects_normalized_mapping_tampering() -> None:
    fixture, evidence = _valid_payloads()
    fixture["cases"][0]["reduction"]["groups"][0]["physical_color_ids"].reverse()

    with pytest.raises(REFERENCE.ReferenceFixtureError, match="reduction plan hash"):
        REFERENCE.parse_reference_fixture(fixture, [evidence])


def test_reduction_groups_must_partition_every_nonzero_cell() -> None:
    fixture, evidence = _valid_payloads()
    fixture["cases"][0]["reduction"]["groups"][0]["physical_helicity_ids"] = [
        "helicity:active"
    ]

    with pytest.raises(REFERENCE.ReferenceFixtureError, match="do not partition"):
        _parse(fixture, evidence)


def test_contracted_reduction_allows_many_groups_to_feed_one_cell() -> None:
    fixture, evidence = _contracted_payloads("nlc")
    case = fixture["cases"][0]
    duplicate = copy.deepcopy(case["reduction"]["groups"][0])
    duplicate["id"] = "reduction:additional-contracted-contribution"
    case["reduction"]["groups"].append(duplicate)
    case["topology"]["reduction_groups"] += 1

    _parse(fixture, evidence)


def test_lc_reduction_still_rejects_overlapping_physical_cells() -> None:
    fixture, evidence = _valid_payloads()
    case = fixture["cases"][0]
    duplicate = copy.deepcopy(case["reduction"]["groups"][0])
    duplicate["id"] = "reduction:overlap"
    case["reduction"]["groups"].append(duplicate)
    case["topology"]["reduction_groups"] += 1

    with pytest.raises(REFERENCE.ReferenceFixtureError, match="overlap"):
        _parse(fixture, evidence)


def test_invalid_alias_permutation_is_rejected() -> None:
    fixture, evidence = _valid_payloads()
    fixture["processes"][1]["final_state_permutation"] = [0, 2, 1]

    with pytest.raises(
        REFERENCE.ReferenceFixtureError, match="does not match its PDGs"
    ):
        _parse(fixture, evidence)


def test_alias_case_hash_covers_transitive_source_process() -> None:
    fixture, _evidence = _valid_payloads()
    case = fixture["cases"][0]
    case["process_id"] = "process:dd-ggz"
    before = REFERENCE.physics_case_sha256(fixture, case["id"])

    fixture["processes"][0]["expression"] = "d d~ > changed-source"

    assert REFERENCE.physics_case_sha256(fixture, case["id"]) != before


def test_identical_particle_alias_must_preserve_stable_leg_identity() -> None:
    fixture, evidence = _valid_payloads()
    alias = fixture["processes"][1]
    alias["external_leg_ids"][2:4] = ["leg:g2-out", "leg:g1-out"]

    with pytest.raises(REFERENCE.ReferenceFixtureError, match="leg identities"):
        _parse(fixture, evidence)


def test_color_singlet_lc_flow_may_have_an_empty_word() -> None:
    fixture, evidence = _degenerate_payloads()
    fixture["processes"][0]["external_colors"] = [1, 1, 1]
    color = fixture["cases"][0]["axes"]["colors"][0]
    color.update(
        id="flow:singlet",
        word=[],
        representative_id="flow:singlet",
    )
    reduction_group = fixture["cases"][0]["reduction"]["groups"][0]
    reduction_group["representative_color_id"] = "flow:singlet"
    reduction_group["physical_color_ids"] = ["flow:singlet"]
    evidence["records"][0]["color_ids"] = ["flow:singlet"]
    evidence["records"][0]["process_identity"]["ordered_color_legs"] = []

    parsed = _parse(fixture, evidence)

    assert parsed.cases[0].colors[0].word == ()


@pytest.mark.parametrize("word", ([2, 4, 4, 1], [2, 4, 999, 1]))
def test_lc_flow_word_must_reference_distinct_external_labels(
    word: list[int],
) -> None:
    fixture, evidence = _valid_payloads()
    fixture["cases"][0]["axes"]["colors"][0]["word"] = word

    with pytest.raises(
        REFERENCE.ReferenceFixtureError, match="model-derived colored external"
    ):
        _parse(fixture, evidence)


def test_complete_helicity_coverage_is_not_self_declared() -> None:
    fixture, evidence = _valid_payloads()
    case = fixture["cases"][0]
    case["axes"]["helicities"].pop()
    case["coverage"]["helicity_count"] = 3
    case["coverage"]["structural_zero_helicity_count"] = 0
    for observation in case["observations"]:
        observation["values"].pop()
    for record in evidence["records"]:
        record["helicity_ids"].pop()
        record["observed_values"].pop()

    with pytest.raises(REFERENCE.ReferenceFixtureError, match="complete physical"):
        _parse(fixture, evidence)


def test_helicity_domains_are_derived_from_spin_and_mass_metadata() -> None:
    fixture, evidence = _valid_payloads()
    fixture["processes"][0]["external_spins"][2] = 3

    with pytest.raises(REFERENCE.ReferenceFixtureError, match="model-derived spin"):
        _parse(fixture, evidence)


def test_point_masses_must_match_model_derived_process_masses() -> None:
    fixture, evidence = _valid_payloads()
    fixture["points"][0]["masses"][2] = "299"

    with pytest.raises(REFERENCE.ReferenceFixtureError, match="model-derived masses"):
        _parse(fixture, evidence)


def test_dirty_or_synthetic_provenance_is_rejected() -> None:
    fixture, evidence = _valid_payloads()
    fixture["provenance"]["working_tree_clean"] = False

    with pytest.raises(REFERENCE.ReferenceFixtureError, match="clean source tree"):
        _parse(fixture, evidence)

    fixture, evidence = _valid_payloads()
    fixture["provenance"]["source_tree_sha256"] = "0" * 64

    with pytest.raises(REFERENCE.ReferenceFixtureError, match="all-zero digest"):
        _parse(fixture, evidence)


def test_synthetic_oracle_output_digest_is_rejected() -> None:
    fixture, evidence = _valid_payloads()
    evidence["records"][0]["oracle_output_sha256"] = "0" * 64

    with pytest.raises(REFERENCE.ReferenceFixtureError, match="all-zero digest"):
        _parse(fixture, evidence)


def test_evidence_record_hash_detects_normalized_record_tampering() -> None:
    fixture, evidence = _valid_payloads()
    evidence["records"][0]["command"].append("tampered=true")

    with pytest.raises(REFERENCE.ReferenceFixtureError, match="canonical hash"):
        REFERENCE.parse_reference_fixture(fixture, [evidence])


def test_bundle_manifest_commits_and_authenticates_loaded_documents(
    tmp_path: Path,
) -> None:
    fixture, evidence = _valid_payloads()
    paths = CAPTURE.atomic_write_documents(
        tmp_path,
        {
            CAPTURE.PHYSICS_FILENAME: fixture,
            CAPTURE.ANALYTIC_EVIDENCE_FILENAME: evidence,
        },
        bundle_manifest_name=CAPTURE.BUNDLE_MANIFEST_FILENAME,
    )

    parsed = REFERENCE.load_reference_fixture(paths[0], paths[1:])

    assert parsed.id == fixture["fixture_id"]
    manifest_path = tmp_path / CAPTURE.BUNDLE_MANIFEST_FILENAME
    assert manifest_path.is_file()
    paths[1].write_text("{}\n", encoding="ascii")
    with pytest.raises(REFERENCE.ReferenceFixtureError, match="digest mismatch"):
        REFERENCE.load_reference_fixture(paths[0], paths[1:])


def test_bundle_loader_rejects_documents_without_commit_marker(tmp_path: Path) -> None:
    fixture, evidence = _valid_payloads()
    fixture_path = tmp_path / CAPTURE.PHYSICS_FILENAME
    evidence_path = tmp_path / CAPTURE.ANALYTIC_EVIDENCE_FILENAME
    fixture_path.write_text(json.dumps(fixture), encoding="ascii")
    evidence_path.write_text(json.dumps(evidence), encoding="ascii")

    with pytest.raises(REFERENCE.ReferenceFixtureError, match="commit marker"):
        REFERENCE.load_reference_fixture(fixture_path, (evidence_path,))


def test_binary64_precision_metadata_is_explicit_and_valid() -> None:
    fixture, evidence = _binary64_degenerate_payloads()

    parsed = _parse(fixture, evidence)

    record = parsed.evidence_sets[0].records[0]
    assert record.arithmetic_precision_bits == 53
    assert record.round_trip_decimal_digits == 17
    assert record.certified_decimal_digits == 15


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("arithmetic_precision_bits", 54),
        ("round_trip_decimal_digits", 16),
        ("certified_decimal_digits", 16),
    ],
)
def test_binary64_precision_fields_cannot_be_conflated(
    field: str,
    bad_value: int,
) -> None:
    fixture, evidence = _binary64_degenerate_payloads()
    evidence["records"][0][field] = bad_value

    with pytest.raises(REFERENCE.ReferenceFixtureError, match="schema violation"):
        _parse(fixture, evidence)


def test_degenerate_two_to_one_policy_accepts_one_canonical_point() -> None:
    fixture, evidence = _degenerate_payloads()

    parsed = _parse(fixture, evidence)

    assert parsed.cases[0].point_policy == "degenerate-2to1"
    assert parsed.cases[0].point_ids == ("point:canonical",)


def test_degenerate_two_to_one_policy_is_rejected_for_other_topologies() -> None:
    fixture, evidence = _valid_payloads()
    fixture["cases"][0]["point_policy"] = "degenerate-2to1"

    with pytest.raises(
        REFERENCE.ReferenceFixtureError,
        match="exactly two incoming and one outgoing",
    ):
        _parse(fixture, evidence)


def test_degenerate_two_to_one_policy_rejects_multiple_points() -> None:
    fixture, evidence = _degenerate_payloads()
    extra_point = copy.deepcopy(fixture["points"][0])
    extra_point["id"] = "point:canonical-extra"
    fixture["points"].append(extra_point)
    fixture["cases"][0]["point_ids"].append("point:canonical-extra")
    extra_observation = copy.deepcopy(fixture["cases"][0]["observations"][0])
    extra_observation["point_id"] = "point:canonical-extra"
    fixture["cases"][0]["observations"].append(extra_observation)

    with pytest.raises(REFERENCE.ReferenceFixtureError, match="one canonical point"):
        _parse(fixture, evidence)

# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import json
import tomllib
from math import fsum
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = ROOT / "tests" / "fixtures" / "reference"
FIXTURE = REFERENCE_ROOT / "physics-v1.json"
CAPTURE = REFERENCE_ROOT / "CAPTURE.toml"
PHYSICS_V2 = REFERENCE_ROOT / "physics-v2.json"
BUNDLE_V2 = REFERENCE_ROOT / "reference-fixture-v2.manifest.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reference_fixture_is_compact_and_self_consistent() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert payload["fixture_schema_version"] == 1
    cases = payload["cases"]
    assert {
        "builtin_sm_ddbar_z_lc",
        "builtin_sm_ddbar_z_nlc",
        "builtin_sm_ddbar_z_full",
        "builtin_sm_ddbar_zg_lc",
        "builtin_sm_ddbar_zg_nlc",
        "builtin_sm_ddbar_zg_full",
        "scalars_2to2_lc",
        "scalar_gravity_2to2_lc",
    } == set(cases)

    for case in cases.values():
        resolved = case.get("resolved")
        if resolved is None:
            resolved = cases[case["resolved_from"]]["resolved"]
        resolved_total = fsum(
            value for colors in resolved.values() for value in colors.values()
        )
        assert resolved_total == case["total"] or abs(
            resolved_total - case["total"]
        ) <= 2.0e-15 * max(abs(case["total"]), 1.0)


def test_current_reference_capture_matches_v2_bundle() -> None:
    with CAPTURE.open("rb") as stream:
        capture = tomllib.load(stream)
    physics = json.loads(PHYSICS_V2.read_text(encoding="utf-8"))
    bundle = json.loads(BUNDLE_V2.read_text(encoding="utf-8"))

    assert capture["schema_version"] == 2
    assert capture["fixture_bundle"] == BUNDLE_V2.name
    assert capture["memory_watchdog_gb"] == 30
    assert capture["source"]["revision"] == physics["provenance"]["source_revision"]
    assert capture["source"]["tree_sha256"] == physics["provenance"][
        "source_tree_sha256"
    ]
    assert capture["source"]["working_tree_clean"] is True
    assert capture["coverage"]["case_count"] == len(physics["cases"])
    assert capture["coverage"]["point_count"] == len(physics["points"])
    assert capture["evidence"]["legacy_fortran"]["records"] == 90
    assert capture["evidence"]["analytic"]["records"] == 8
    assert {record["path"] for record in bundle["files"]} == {
        "analytic-oracles-v2.json",
        "legacy-fortran-v2.json",
        "physics-v2.json",
    }
    assert next(
        record["sha256"]
        for record in bundle["files"]
        if record["path"] == PHYSICS_V2.name
    ) == _sha256(PHYSICS_V2)

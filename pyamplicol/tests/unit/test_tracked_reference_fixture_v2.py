# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import re
from pathlib import Path

from tools.developer.reference_fixture import ReferenceFixture, load_reference_fixture

REFERENCE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "reference"


def _tracked_fixture() -> ReferenceFixture:
    return load_reference_fixture(
        REFERENCE_ROOT / "physics-v2.json",
        (
            REFERENCE_ROOT / "legacy-fortran-v2.json",
            REFERENCE_ROOT / "analytic-oracles-v2.json",
        ),
    )


def test_tracked_reference_bundle_is_complete_and_revision_bound() -> None:
    fixture = _tracked_fixture()
    sm_processes = {
        "sm_ddbar_z",
        "sm_udbar_wplus",
        "sm_ddbar_zg",
        "sm_ddbar_ee",
        "sm_ddbar_uubar",
        "sm_ddbar_ddbar",
        "sm_gg_gg",
        "sm_gg_ttbar",
        "sm_ddbar_zgg",
    }
    expected_cases = {
        *(
            f"case:{process}:{accuracy}"
            for process in sm_processes
            for accuracy in ("lc", "nlc", "full")
        ),
        "case:scalars_2to2:lc",
        "case:scalar_gravity_2to2:lc",
    }

    assert fixture.provenance.working_tree_clean
    assert fixture.provenance.memory_watchdog_gb == 30
    assert re.fullmatch(r"[0-9a-f]{40}", fixture.provenance.source_revision)
    candidate = next(
        dependency
        for dependency in fixture.dependencies
        if dependency.id == "dependency:pyamplicol-candidate"
    )
    assert candidate.revision == fixture.provenance.source_revision
    assert {case.id for case in fixture.cases} == expected_cases
    assert {evidence.id for evidence in fixture.evidence_sets} == {
        "evidence-set:legacy-fortran-amplicol",
        "evidence-set:analytic-oracles",
    }

    for case in fixture.cases:
        assert tuple(
            observation.point_id for observation in case.observations
        ) == case.point_ids
        assert all(observation.evidence_refs for observation in case.observations)


def test_tracked_stress_points_retain_high_precision_inputs() -> None:
    fixture = _tracked_fixture()
    stress_points = tuple(
        point for point in fixture.points if point.point_class == "stress"
    )

    assert stress_points
    assert all(point.arithmetic_precision_bits >= 256 for point in stress_points)
    assert all(point.certified_decimal_digits >= 80 for point in stress_points)
    assert all(point.stress_metric is not None for point in stress_points)

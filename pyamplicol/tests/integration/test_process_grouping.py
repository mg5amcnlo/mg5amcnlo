# SPDX-License-Identifier: 0BSD
"""Numerical process grouping of generated built-in SM artifacts."""

from __future__ import annotations

import importlib.util
import os
import shutil
from collections.abc import Callable, Iterator
from fractions import Fraction
from pathlib import Path

import pytest

from pyamplicol import Generator, ModelSource, Runtime
from pyamplicol.api.process_groups import group_processes
from pyamplicol.artifacts.manifest import load_manifest
from pyamplicol.artifacts.process_groups import grouping_process, read_process_groups
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationRelationDiscoveryConfig,
    GenerationValidationConfig,
    JITConfig,
    ProcessConfig,
    RunConfig,
)
from pyamplicol.generation.process_grouping import (
    GroupingBasis,
    flavor_class_key,
    profile_key,
)
from pyamplicol.generation.process_grouping_numerical import (
    _artifact_model,
    member_point,
    on_shell_point,
)


def _unavailable(reason: str) -> None:
    if os.environ.get("PYAMPLICOL_REQUIRE_NATIVE_TESTS") == "1":
        pytest.fail(reason)
    pytest.skip(reason)


def _config(
    flavor_scheme: int = 5, selected_source_helicities: dict[str, int] | None = None
) -> RunConfig:
    return RunConfig(
        action="generate",
        color=ColorConfig(accuracy="lc"),
        process=ProcessConfig(
            flavor_scheme=flavor_scheme,
            selected_source_helicities=selected_source_helicities or {},
        ),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
            relation_discovery=GenerationRelationDiscoveryConfig(mode="off"),
            validation=GenerationValidationConfig(
                enabled=False, post_build_validation=False
            ),
        ),
        evaluator=EvaluatorConfig(
            execution_mode="eager",
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(optimization_level=2),
        ),
    )


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Path]]:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        _unavailable("the Rusticol extension has not been built")
    if importlib.util.find_spec("symbolica") is None:
        _unavailable("Symbolica is unavailable")
    root = tmp_path_factory.mktemp("process-grouping")
    artifacts = {
        "jets": root / "jets",
        "photons": root / "photons",
        "selected": root / "selected",
        "cycle": root / "cycle",
    }
    try:
        Generator(_config(flavor_scheme=2)).generate(
            "p p > j j", artifacts["jets"], model=ModelSource.built_in_sm()
        )
        Generator(_config()).generate(
            ("u u~ > a a", "d d~ > a a"),
            artifacts["photons"],
            model=ModelSource.built_in_sm(),
        )
        Generator(_config(selected_source_helicities={"3": 1})).generate(
            ("u u~ > a a", "d d~ > a a"),
            artifacts["selected"],
            model=ModelSource.built_in_sm(),
        )
        Generator(_config()).generate(
            ("d g > d d d~", "d~ g > d d~ d~"),
            artifacts["cycle"],
            model=ModelSource.built_in_sm(),
        )
        yield artifacts
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def artifact(generated: dict[str, Path], tmp_path: Path) -> Callable[[str], Path]:
    def copy(name: str) -> Path:
        destination = tmp_path / f"{name}-{len(list(tmp_path.iterdir()))}"
        shutil.copytree(generated[name], destination)
        return destination

    return copy


def _partition(path: Path) -> set[frozenset[str]]:
    group_map = read_process_groups(path)
    assert group_map is not None
    return {
        frozenset(member.process_id for member in group.members)
        for group in group_map.groups.values()
    }


def _flavour_classes(
    path: Path, *, consider_charge_conjugation: bool
) -> set[frozenset[str]]:
    table = _artifact_model(path).table
    classes: dict[object, set[str]] = {}
    for process in load_manifest(path).processes:
        grouping = grouping_process(
            str(process["id"]),
            str(process["expression"]),
            process["external_pdgs"],  # type: ignore[arg-type]
        )
        key = (
            flavor_class_key(grouping, table, consider_charge_conjugation=True)
            if consider_charge_conjugation
            else profile_key(grouping, table)
        )
        classes.setdefault(key, set()).add(grouping.process_id)
    return {frozenset(members) for members in classes.values()}


@pytest.mark.parametrize("consider_charge_conjugation", (True, False))
def test_jets_group_into_flavour_profile_classes(
    artifact: Callable[[str], Path],
    consider_charge_conjugation: bool,
) -> None:
    path = artifact("jets")
    identity = load_manifest(path).artifact_id

    summary = group_processes(
        path, consider_charge_conjugation=consider_charge_conjugation
    )

    assert summary["skipped"] == []
    expected = _flavour_classes(
        path, consider_charge_conjugation=consider_charge_conjugation
    )
    assert len(expected) == (9 if consider_charge_conjugation else 12)
    assert _partition(path) == expected
    assert load_manifest(path).artifact_id == identity
    group_map = read_process_groups(path)
    assert group_map is not None
    assert group_map.analyses[-1]["consider_charge_conjugation"] is (
        consider_charge_conjugation
    )
    members = list(group_map.members())
    assert {member.factor for member in members} == {Fraction(1)}
    assert any(member.conjugated for member in members) is consider_charge_conjugation

    # Each recorded relation holds for the summed matrix element at a point
    # outside the analysed ones: ME_member(p) = factor * ME_rep(q).
    point = on_shell_point((0.0, 0.0), (0.0, 0.0), seed=11)
    assert point is not None
    for group in group_map.groups.values():
        representative = Runtime.load(path, process=group.representative.process_id)
        (expected_value,) = representative.evaluate([point])
        for member in group.members[1:]:
            assert member.basis is GroupingBasis.NUMERICAL
            runtime = Runtime.load(path, process=member.process_id)
            (value,) = runtime.evaluate([member_point(point, member.permutation)])
            assert value.real == pytest.approx(
                float(member.factor) * expected_value.real, rel=1e-10
            )


def _helicity_totals(
    path: Path, process: str, point: tuple[object, ...], legs: list[int]
) -> dict[tuple[int, ...], float]:
    """Colour-summed values keyed by the helicities of ``legs`` in order."""

    runtime = Runtime.load(path, process=process)
    values = {item.id: item.values for item in runtime.physics.helicities}
    resolved = runtime.evaluate_resolved([point])
    return {
        tuple(values[identifier][leg] for leg in legs): sum(row).real
        for identifier, row in zip(
            resolved.helicity_ids, resolved.values[0], strict=True
        )
    }


def test_conjugated_relation_with_cycled_legs_holds_per_helicity(
    artifact: Callable[[str], Path],
) -> None:
    # d~ g > d d~ d~ is the charge conjugate of d g > d d d~ with the final
    # legs cycled, a permutation that is not its own inverse.
    path = artifact("cycle")
    group_processes(path)
    group_map = read_process_groups(path)
    assert group_map is not None
    (group,) = group_map.groups.values()
    assert group.representative.process_id == "d_g_to_d_d_dbar"
    (member,) = group.members[1:]
    assert member.conjugated
    assert member.factor == 1
    assert member.permutation == (0, 1, 3, 4, 2)

    # q[r] = p[perm[r]], and helicities map unchanged under conjugation.
    point = on_shell_point((0.0, 0.0), (0.0, 0.0, 0.0), seed=11)
    assert point is not None
    member_momenta: list[object] = list(point)
    for representative_index, member_index in enumerate(member.permutation):
        member_momenta[member_index] = point[representative_index]
    expected = _helicity_totals(path, "d_g_to_d_d_dbar", point, [0, 1, 2, 3, 4])
    actual = _helicity_totals(
        path, member.process_id, tuple(member_momenta), list(member.permutation)
    )
    assert actual.keys() == expected.keys()
    scale = max(map(abs, expected.values()))
    for helicities, value in expected.items():
        assert actual[helicities] == pytest.approx(value, abs=1e-10 * scale)


def test_rational_factor_is_found_and_unit_restriction_rejects_it(
    artifact: Callable[[str], Path],
) -> None:
    path = artifact("photons")
    summary = group_processes(path)
    group_map = read_process_groups(path)
    assert group_map is not None
    (group,) = group_map.groups.values()
    # Charges (2/3)^4 / (1/3)^4: the up-quark process is 16 times the down one.
    assert group.representative.process_id == "d_dbar_to_a_a"
    (member,) = group.members[1:]
    assert member.process_id == "u_ubar_to_a_a"
    assert member.factor == Fraction(16)
    assert summary["new_relations"] == [
        {
            "process_id": "u_ubar_to_a_a",
            "process": "u u~ > a a",
            "representative": "d_dbar_to_a_a",
            "factor": "16",
            "permutation": [0, 1, 2, 3],
            "conjugated": False,
        }
    ]

    unit = artifact("photons")
    group_processes(unit, only_unit_multiplier=True)
    assert _partition(unit) == {
        frozenset({"u_ubar_to_a_a"}),
        frozenset({"d_dbar_to_a_a"}),
    }


def test_selected_helicity_coverage_is_never_grouped(
    artifact: Callable[[str], Path],
) -> None:
    path = artifact("selected")
    summary = group_processes(path)
    assert summary["new_relations"] == []
    assert summary["skipped"] == [
        {"process_id": "d_dbar_to_a_a", "reason": "helicity coverage is selected"},
        {"process_id": "u_ubar_to_a_a", "reason": "helicity coverage is selected"},
    ]
    assert _partition(path) == {
        frozenset({"u_ubar_to_a_a"}),
        frozenset({"d_dbar_to_a_a"}),
    }


def test_rerunning_the_analysis_merges_idempotently(
    artifact: Callable[[str], Path],
) -> None:
    path = artifact("jets")
    groups = path / "process-groups.json"
    first = group_processes(path, consider_charge_conjugation=False)
    stored = groups.read_bytes()

    again = group_processes(path, consider_charge_conjugation=False)
    assert again["new_relations"] == []
    assert again["groups_before"] == again["groups_after"] == first["groups_after"]
    assert groups.read_bytes() == stored
    assert group_processes(path, analysis="none")["new_relations"] == []
    assert groups.read_bytes() == stored

    # Allowing charge conjugation only merges existing groups further.
    conjugated = group_processes(path)
    assert conjugated["groups_after"] == 9
    assert all(relation["conjugated"] for relation in conjugated["new_relations"])
    group_map = read_process_groups(path)
    assert group_map is not None
    assert [entry["consider_charge_conjugation"] for entry in group_map.analyses] == [
        False,
        True,
    ]

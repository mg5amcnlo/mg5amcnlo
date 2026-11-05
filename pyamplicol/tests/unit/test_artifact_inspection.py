# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import shutil
from fractions import Fraction
from pathlib import Path

import pytest

from pyamplicol.artifacts import inspect_artifact
from pyamplicol.artifacts.inspection import (
    ArtifactProcessGroupMemberInspection,
    _computed,
    process_group_selection,
    stored_process_groups,
)
from pyamplicol.artifacts.process_groups import (
    initial_process_groups,
    write_process_groups,
)
from pyamplicol.generation.process_grouping import GroupingBasis, GroupMember

ROOT = Path(__file__).resolve().parents[2]
PORTABLE_ARTIFACT = ROOT / "src/pyamplicol/assets/selftest/portable-64le/artifact"


def test_artifact_inspection_lists_processes_without_loading_evaluators() -> None:
    inspection = inspect_artifact(PORTABLE_ARTIFACT)

    assert inspection.kind == "pyamplicol-artifact-inspection"
    assert inspection.integrity == "verified"
    assert inspection.model_name == "built-in-sm"
    assert inspection.target == "portable-64le"
    assert inspection.default_process_id == "d_dbar_to_z"
    assert inspection.payload_count > 0
    assert inspection.payload_size_bytes > 0

    assert len(inspection.processes) == 1
    process = inspection.processes[0]
    assert process.id == "d_dbar_to_z"
    assert process.expression == "d d~ > z"
    assert process.default
    assert process.physical_helicities == 12
    assert process.computed_helicities == 2
    assert process.physical_color_components == 1
    assert process.computed_color_components == 1
    assert process.helicity_coverage == "complete"
    assert process.color_coverage == "complete"
    assert process.lc_flow_layout == "topology-replay"
    assert process.execution_mode == "compiled"
    assert process.prepared_backend is None
    assert process.invocation_count is None
    assert process.native_profile_phases == (
        "source-fill",
        "momentum-setup",
        "stage-input-pack",
        "stage-evaluator-call",
        "output-assign",
        "amplitude-input-pack",
        "amplitude-evaluator-call",
        "reduction",
    )


def test_contracted_color_is_definitionally_computed() -> None:
    assert _computed(
        {
            "kind": "contracted-color",
            "id": "color:contracted",
            "index": 0,
            "description": "fully contracted color",
        },
        "color component",
    )


def _grouped_artifact(tmp_path: Path) -> Path:
    """The self-test artifact with two flavour-blind members of ``d d~ > z``."""

    root = tmp_path / "artifact"
    shutil.copytree(PORTABLE_ARTIFACT, root)
    members = [
        GroupMember(
            process_id="u_ubar_to_z",
            expression="u u~ > z",
            pdgs=(2, -2, 23),
            initial_count=2,
            factor=Fraction(16, 9),
            permutation=(0, 1, 2),
            basis=GroupingBasis.FLAVOR_BLIND,
            generated=False,
            warning="Z couplings differ between d and u",
        ),
        GroupMember(
            process_id="dbar_d_to_z",
            expression="d~ d > z",
            pdgs=(-1, 1, 23),
            initial_count=2,
            permutation=(1, 0, 2),
            conjugated=True,
            basis=GroupingBasis.FLAVOR_BLIND,
            generated=False,
        ),
    ]
    write_process_groups(
        root,
        initial_process_groups(
            [("d_dbar_to_z", "d d~ > z", (1, -1, 23))], {"d_dbar_to_z": members}
        ),
    )
    return root


def test_artifact_without_group_map_inspects_as_identity_groups() -> None:
    groups = inspect_artifact(PORTABLE_ARTIFACT).process_groups

    assert groups is not None
    assert not groups.stored
    assert groups.analyses == ()
    assert [group.representative_id for group in groups.groups] == ["d_dbar_to_z"]
    assert groups.groups[0].members == (
        ArtifactProcessGroupMemberInspection(
            id="d_dbar_to_z",
            process="d d~ > z",
            pdgs=(1, -1, 23),
            initial_count=2,
            factor="1",
            permutation=(0, 1, 2),
            conjugated=False,
            basis="identity",
            generated=True,
        ),
    )
    assert process_group_selection(PORTABLE_ARTIFACT, "d_dbar_to_z") is None
    assert stored_process_groups(PORTABLE_ARTIFACT) is None


def test_artifact_inspection_lists_stored_process_groups(tmp_path: Path) -> None:
    root = _grouped_artifact(tmp_path)

    groups = inspect_artifact(root).process_groups

    assert groups is not None
    assert groups.stored
    (group,) = groups.groups
    assert (group.representative_id, group.representative) == (
        "d_dbar_to_z",
        "d d~ > z",
    )
    representative, routed, conjugated = group.members
    assert representative.generated and representative.basis == "identity"
    assert routed == ArtifactProcessGroupMemberInspection(
        id="u_ubar_to_z",
        process="u u~ > z",
        pdgs=(2, -2, 23),
        initial_count=2,
        factor="16/9",
        permutation=(0, 1, 2),
        conjugated=False,
        basis="flavor-blind",
        generated=False,
        warning="Z couplings differ between d and u",
    )
    assert conjugated.conjugated
    assert conjugated.permutation == (1, 0, 2)
    assert conjugated.factor == "1"
    assert conjugated.warning is None
    assert (conjugated.pdgs, conjugated.initial_count) == ((-1, 1, 23), 2)
    assert stored_process_groups(root) == groups


def test_process_group_selection_describes_members_and_representatives(
    tmp_path: Path,
) -> None:
    root = _grouped_artifact(tmp_path)

    member = process_group_selection(root, "dbar_d_to_z", permutation=(0, 1, 2))
    assert member is not None
    assert member.role == "routed member"
    assert member.representative_id == "d_dbar_to_z"
    assert member.conjugated
    # the runtime permutation of the requested leg order replaces the stored one
    assert member.permutation == (0, 1, 2)
    assert (member.group_size, member.routed_count, member.conjugated_count) == (
        3,
        2,
        1,
    )

    routed = process_group_selection(root, "u_ubar_to_z")
    assert routed is not None
    assert (routed.factor, routed.basis) == ("16/9", "flavor-blind")
    assert routed.warning == "Z couplings differ between d and u"

    representative = process_group_selection(root, "d_dbar_to_z", permutation=(1, 0, 2))
    assert representative is not None
    assert representative.role == "representative"
    assert representative.factor == "1"
    # the representative requested as `d~ d > z` relates to itself by a swap
    assert representative.permutation == (1, 0, 2)
    stored = process_group_selection(root, "d_dbar_to_z")
    assert stored is not None and stored.permutation == (0, 1, 2)


def test_process_group_selection_composes_a_generated_member_permutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A generated member's runtime permutes its own legs, not the
    representative's, so the stored relation is composed with it."""

    import pyamplicol.artifacts.inspection as inspection_module

    representative = ("d_g_to_d_g", "d g > d g", (1, 21, 1, 21))
    member = GroupMember(
        process_id="g_u_to_u_g",
        expression="g u > u g",
        pdgs=(21, 2, 2, 21),
        initial_count=2,
        permutation=(1, 0, 2, 3),
        basis=GroupingBasis.NUMERICAL,
        generated=True,
    )
    group_map = initial_process_groups([representative], {"d_g_to_d_g": [member]})
    monkeypatch.setattr(
        inspection_module, "read_process_groups", lambda _artifact: group_map
    )

    stored = process_group_selection("artifact", "g_u_to_u_g")
    # `u g > u g` is `g u > u g` with its initial legs swapped
    requested = process_group_selection(
        "artifact", "g_u_to_u_g", permutation=(1, 0, 2, 3)
    )

    assert stored is not None and requested is not None
    assert stored.role == requested.role == "generated member"
    assert stored.permutation == (1, 0, 2, 3)
    assert requested.permutation == (0, 1, 2, 3)

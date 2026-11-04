# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import pyamplicol.generation.process_grouping_numerical as numerical
import pyamplicol.generation.service as service_module
from pyamplicol.config import Action, ProcessConfig, RunConfig
from pyamplicol.generation.process_grouping import GroupingBasis, GroupMember
from pyamplicol.generation.progress import GenerationPhaseReporter


def test_initial_groups_keep_the_members_recorded_at_expansion() -> None:
    member = GroupMember(
        process_id="u_ubar_to_g_g",
        expression="u u~ > g g",
        pdgs=(2, -2, 21, 21),
        initial_count=2,
        permutation=(0, 1, 2, 3),
        basis=GroupingBasis.FLAVOR_BLIND,
        generated=False,
    )
    processes = (
        SimpleNamespace(
            process_id="d_dbar_to_g_g",
            expression="d d~ > g g",
            external_pdgs=(1, -1, 21, 21),
        ),
        SimpleNamespace(
            process_id="g_g_to_g_g",
            expression="g g > g g",
            external_pdgs=(21, 21, 21, 21),
        ),
    )
    expanded = (
        SimpleNamespace(
            request=SimpleNamespace(name="d_dbar_to_g_g"), group_members=(member,)
        ),
        SimpleNamespace(request=SimpleNamespace(name="g_g_to_g_g"), group_members=()),
    )

    group_map = service_module._initial_process_groups(processes, expanded)  # type: ignore[arg-type]

    assert {
        identifier: [item.process_id for item in group.members]
        for identifier, group in group_map.groups.items()
    } == {
        "d_dbar_to_g_g": ["d_dbar_to_g_g", "u_ubar_to_g_g"],
        "g_g_to_g_g": ["g_g_to_g_g"],
    }
    assert group_map.groups["d_dbar_to_g_g"].members[1] == member


def test_numerical_grouping_runs_only_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path, dict[str, object]]] = []

    def update(artifact: Path, **options: object) -> dict[str, object]:
        calls.append((artifact, options))
        return {}

    monkeypatch.setattr(numerical, "update_artifact_process_groups", update)
    for analysis in ("none", "numerical"):
        backend = service_module.GenerationBackend(
            RunConfig(
                action=Action.GENERATE,
                process=ProcessConfig(
                    analyze_process_groups=analysis,  # type: ignore[arg-type]
                    only_unit_multiplier=True,
                ),
            ),
            None,
        )
        backend._analyze_process_groups(tmp_path, GenerationPhaseReporter(None))

    assert calls == [
        (
            tmp_path,
            {
                "analysis": "numerical",
                "only_unit_multiplier": True,
                "consider_charge_conjugation": True,
            },
        )
    ]


def test_skipped_representative_reports_its_flavor_blind_members(
    caplog: pytest.LogCaptureFixture,
) -> None:
    member = GroupMember(
        process_id="u_ubar_to_z_g_g",
        expression="u u~ > z g g",
        pdgs=(2, -2, 23, 21, 21),
        initial_count=2,
        generated=False,
    )
    expanded = SimpleNamespace(
        request=SimpleNamespace(name="d_dbar_to_z_g_g"),
        process_ir=SimpleNamespace(process="d d~ > z g g"),
        group_members=(member,),
    )

    service_module._warn_unsupported_processes(
        [SimpleNamespace(expanded=expanded, reason="no amplitudes")]  # type: ignore[list-item]
    )

    assert "d_dbar_to_z_g_g (d d~ > z g g, with 1 flavor-blind member(s))" in (
        caplog.text
    )


def test_ungenerated_members_need_a_positive_factor() -> None:
    from fractions import Fraction

    from pyamplicol.api.errors import ArtifactError
    from pyamplicol.artifacts.process_groups import (
        initial_process_groups,
        validate_process_groups,
    )

    def member(factor: Fraction) -> GroupMember:
        return GroupMember(
            process_id="u_ubar_to_g_g",
            expression="u u~ > g g",
            pdgs=(2, -2, 21, 21),
            initial_count=2,
            factor=factor,
            permutation=(0, 1, 2, 3),
            basis=GroupingBasis.FLAVOR_BLIND,
            generated=False,
        )

    stored = [("d_dbar_to_g_g", "d d~ > g g", (1, -1, 21, 21))]
    validate_process_groups(
        initial_process_groups(stored, {"d_dbar_to_g_g": [member(Fraction(4))]}),
        ["d_dbar_to_g_g"],
    )
    with pytest.raises(ArtifactError, match="must be positive"):
        validate_process_groups(
            initial_process_groups(stored, {"d_dbar_to_g_g": [member(Fraction(-1))]}),
            ["d_dbar_to_g_g"],
        )

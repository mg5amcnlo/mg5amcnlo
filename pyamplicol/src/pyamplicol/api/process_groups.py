# SPDX-License-Identifier: 0BSD
"""Process-group analysis of a generated artifact."""

from __future__ import annotations

import os
from pathlib import Path

from pyamplicol.generation.process_grouping import GroupingAnalysis
from pyamplicol.generation.process_grouping_numerical import (
    update_artifact_process_groups,
)


def group_processes(
    artifact: os.PathLike[str] | str,
    *,
    analysis: str = "numerical",
    only_unit_multiplier: bool = False,
    consider_charge_conjugation: bool = True,
) -> dict[str, object]:
    """Group the artifact's subprocesses with equal matrix elements.

    The stored representatives are compared numerically and every new
    relation (a rational factor and a leg permutation, optionally with charge
    conjugation) is merged into ``process-groups.json``, which is rewritten in
    place. UMAMI files and the artifact ID are left unchanged. ``"none"`` only
    writes the trivial map when the artifact has none. Returns a summary.
    """

    if analysis not in tuple(GroupingAnalysis):
        raise ValueError(
            "analysis must be one of "
            + ", ".join(repr(str(value)) for value in GroupingAnalysis)
        )
    for name, value in (
        ("only_unit_multiplier", only_unit_multiplier),
        ("consider_charge_conjugation", consider_charge_conjugation),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be a boolean")
    path = Path(os.fspath(artifact)).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"artifact does not exist: {path}")
    return update_artifact_process_groups(
        path,
        analysis=GroupingAnalysis(analysis),
        only_unit_multiplier=only_unit_multiplier,
        consider_charge_conjugation=consider_charge_conjugation,
    )


__all__ = ["group_processes"]

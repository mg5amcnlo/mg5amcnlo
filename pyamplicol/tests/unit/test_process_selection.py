# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import pytest

from pyamplicol.models.builtin.process_selection import (
    build_generic_process_selection_report,
    enumerate_generic_process_set,
)


def test_generic_selection_accepts_model_driven_pure_singlet_processes() -> None:
    enumeration = enumerate_generic_process_set("e- e+ > mu- mu+")

    assert tuple(entry.process for entry in enumeration.entries) == ("e- e+ > mu- mu+",)


def test_generic_selection_still_rejects_charge_violating_singlets() -> None:
    with pytest.raises(ValueError, match="no valid processes"):
        enumerate_generic_process_set("e- e- > mu- mu+")


@pytest.mark.parametrize(
    ("process", "expected_quark_lines"),
    (
        ("d d~ > u u~ s s~ c c~", 4),
        ("d d~ > u u~ s s~ c c~ b b~", 5),
    ),
)
def test_generic_selection_has_no_default_open_quark_line_ceiling(
    process: str,
    expected_quark_lines: int,
) -> None:
    report = build_generic_process_selection_report(process)

    assert report.selected_count == 1
    assert report.entries[0].process == process
    assert report.records[0].quark_lines == expected_quark_lines


def test_generic_selection_honors_an_explicit_open_quark_line_ceiling() -> None:
    process = "d d~ > u u~ s s~ c c~ b b~"
    report = build_generic_process_selection_report(
        process,
        max_quark_pairs=4,
    )

    assert report.selected_count == 0
    assert report.rejection_counts == (("max-quark-lines", 1),)
    with pytest.raises(ValueError, match="no valid processes"):
        enumerate_generic_process_set(process, max_quark_pairs=4)

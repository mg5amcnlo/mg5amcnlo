# SPDX-License-Identifier: 0BSD
"""Environment-gated scalar HEFT comparison with MadGraph standalone."""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

from tools.developer.heft_madgraph_acceptance import (
    ABSOLUTE_TOLERANCE,
    PROCESSES,
    RELATIVE_TOLERANCE,
    run_acceptance,
)

_RUN_ENV = "PYAMPLICOL_RUN_HEFT_MADGRAPH"
_INSTALLATION_ENV = "PYAMPLICOL_MADGRAPH"

pytestmark = pytest.mark.skipif(
    os.environ.get(_RUN_ENV) != "1",
    reason=f"set {_RUN_ENV}=1 and {_INSTALLATION_ENV} to run the live HEFT gate",
)


def test_scalar_heft_full_color_matches_madgraph(
    tmp_path: Path,
) -> None:
    configured = os.environ.get(_INSTALLATION_ENV)
    if not configured:
        pytest.fail(f"{_INSTALLATION_ENV} must name the MadGraph installation")

    result = run_acceptance(
        Path(configured),
        model_root=tmp_path / "heft-ufo",
        output_root=tmp_path / "acceptance",
    )

    processes = result["processes"]
    assert isinstance(processes, dict)
    assert set(processes) == {spec.process_id for spec in PROCESSES}
    for record in processes.values():
        assert isinstance(record, dict)
        pyamplicol_value = float(record["pyamplicol"])
        madgraph_value = float(record["madgraph"])
        assert math.isfinite(pyamplicol_value) and pyamplicol_value != 0.0
        assert math.isfinite(madgraph_value) and madgraph_value != 0.0
        assert math.isclose(
            pyamplicol_value,
            madgraph_value,
            rel_tol=RELATIVE_TOLERANCE,
            abs_tol=ABSOLUTE_TOLERANCE,
        )

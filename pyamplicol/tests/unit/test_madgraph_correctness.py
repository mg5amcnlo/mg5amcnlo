# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.developer.madgraph_correctness import (
    CommandResult,
    MadGraphAdapterError,
    madgraph_command_card,
    reject_failed_generation,
    set_parameter_card_values,
)

ROOT = Path(__file__).resolve().parents[2]


def _result(tmp_path: Path, *, stderr: str = "") -> CommandResult:
    return CommandResult(
        args=("mg5_aMC",),
        cwd=tmp_path,
        elapsed_seconds=0.0,
        returncode=0,
        stdout="",
        stderr=stderr,
    )


def test_heft_acceptance_driver_supports_direct_invocation(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            os.fspath(ROOT / "tools/developer/heft_madgraph_acceptance.py"),
            "--help",
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "MadGraph5_aMC installation" in completed.stdout


def test_correctness_command_card_binds_exact_ufo_and_heft_order() -> None:
    assert madgraph_command_card(
        "g g > H g g",
        model_import="/tmp/prepared-heft-ufo",
        coupling_orders={"HIG": 1},
    ) == (
        "import model /tmp/prepared-heft-ufo\n"
        "generate g g > H g g HIG=1\n"
        "output standalone standalone -f\n"
        "launch -f\n"
    )


def test_correctness_generation_guard_rejects_import_fallback(tmp_path: Path) -> None:
    cards = tmp_path / "standalone/Cards"
    cards.mkdir(parents=True)
    process_card = cards / "proc_card_mg5.dat"
    expected = "/tmp/prepared-heft-ufo"
    process_card.write_text(f"import model {expected}\n", encoding="utf-8")

    reject_failed_generation(
        _result(tmp_path),
        tmp_path / "standalone",
        expected_model_import=expected,
    )

    process_card.write_text("import model sm\n", encoding="utf-8")
    with pytest.raises(MadGraphAdapterError, match="bound exclusively"):
        reject_failed_generation(
            _result(tmp_path),
            tmp_path / "standalone",
            expected_model_import=expected,
        )

    process_card.write_text(f"import model {expected}\n", encoding="utf-8")
    with pytest.raises(MadGraphAdapterError, match="possible model fallback"):
        reject_failed_generation(
            _result(tmp_path, stderr="UFOImportError: rejected"),
            tmp_path / "standalone",
            expected_model_import=expected,
        )


def test_correctness_parameter_card_uses_packaged_heft_inputs(tmp_path: Path) -> None:
    card = tmp_path / "param_card.dat"
    card.write_text(
        "Block SMINPUTS\n"
        "  1 1.279000e+02 # aEWM1\n"
        "  3 1.184000e-01 # aS (not used with PDFs)\n"
        "Block MASS\n"
        "  25 1.200000e+02 # MH\n",
        encoding="ascii",
    )

    observed = set_parameter_card_values(
        card,
        {"aEWM1": 132.507, "aS": 0.118, "MH": 125.0},
    )

    assert observed == ("MH", "aEWM1", "aS")
    rendered = card.read_text(encoding="ascii")
    assert "1.32507000000000e+02 # aEWM1" in rendered
    assert "1.18000000000000e-01 # aS (not used with PDFs)" in rendered
    assert "1.25000000000000e+02 # MH" in rendered
    with pytest.raises(MadGraphAdapterError, match="lacks required"):
        set_parameter_card_values(card, {"MT": 173.0})

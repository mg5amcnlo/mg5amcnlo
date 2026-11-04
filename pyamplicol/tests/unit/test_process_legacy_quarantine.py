# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"


def _assert_isolated_import_does_not_load_builtin(statement: str) -> None:
    script = "\n".join(
        (
            "import sys",
            f"sys.path.insert(0, {str(SOURCE_ROOT)!r})",
            statement,
            "loaded = sorted(name for name in sys.modules ",
            "    if name == 'pyamplicol.models.builtin' ",
            "    or name.startswith('pyamplicol.models.builtin.'))",
            "assert loaded == [], loaded",
        )
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    "statement",
    (
        "import pyamplicol.processes",
        "import pyamplicol.processes.ir",
        "import pyamplicol.processes.model",
        "import pyamplicol.generation.service",
        "from pyamplicol.processes import ProcessTuple, canonical_process_key",
    ),
)
def test_generic_process_imports_do_not_load_builtin_physics(statement: str) -> None:
    _assert_isolated_import_does_not_load_builtin(statement)


@pytest.mark.parametrize(
    "name",
    (
        "QUARKS",
        "ANTIQUARKS",
        "GLUONS",
        "SINGLETS",
        "ALL_COLOURED",
        "PDGS",
        "ANTI_PARTICLE",
        "SORT_PARTICLES",
        "CHARGES3",
        "FAMILY",
        "ProcessOptions",
        "ProcessEnumeration",
        "ProcessSelectionRecord",
        "ProcessSelectionReport",
        "ProcessSetEntry",
        "ProcessSetEnumeration",
        "SubprocessRecord",
        "ProcessEnumerator",
        "build_generic_process_selection_report",
        "enumerate_generic_process_set",
        "enumerate_process_set",
        "enumerate_processes",
    ),
)
def test_generic_process_package_has_no_builtin_catalog_or_records(name: str) -> None:
    import pyamplicol.processes as processes

    assert not hasattr(processes, name)


@pytest.mark.parametrize(
    "module_name",
    (
        "pyamplicol.processes.core",
        "pyamplicol.processes.enumeration",
        "pyamplicol.processes.selection",
    ),
)
def test_builtin_process_facades_are_removed(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        __import__(module_name)


def test_builtin_process_records_are_model_scoped() -> None:
    from pyamplicol.models.builtin.process_types import BuiltinProcessOptions

    options = BuiltinProcessOptions(flavour_scheme=4, include_cc=True)

    assert options.flavour_scheme == 4
    assert options.include_cc is True


def test_generic_process_ir_has_no_builtin_construction_facade() -> None:
    import pyamplicol.processes.ir as process_ir

    assert not hasattr(process_ir, "build_process_ir")
    assert not hasattr(process_ir, "build_process_set_ir")

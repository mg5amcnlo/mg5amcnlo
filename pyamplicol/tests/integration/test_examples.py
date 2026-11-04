# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import io
import json
import runpy
import shutil
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from pyamplicol.cli import parse_cli, run_cli
from pyamplicol.config import ProcessEntry, RunConfig
from pyamplicol.reporting import ProgressSink

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
RUN_CARDS = tuple(sorted(EXAMPLES.glob("*.toml")))


def _run_example(
    script: Path, arguments: tuple[str, ...], cwd: Path
) -> subprocess.CompletedProcess[str]:
    # This test module already owns a Symbolica session through the public API.
    # Run the exact script in that session, still from an unrelated directory;
    # the serialized installed-wheel smoke covers independent Python processes.
    command = [str(script), *arguments]
    stdout, stderr = io.StringIO(), io.StringIO()
    returncode = 0
    with (
        pytest.MonkeyPatch.context() as context,
        redirect_stdout(stdout),
        redirect_stderr(stderr),
    ):
        context.chdir(cwd)
        context.setattr(sys, "argv", command)
        try:
            runpy.run_path(str(script), run_name="__main__")
        except SystemExit as error:
            if error.code is None or isinstance(error.code, int):
                returncode = error.code or 0
            else:
                print(error.code, file=sys.stderr)
                returncode = 1
    return subprocess.CompletedProcess(
        command, returncode, stdout.getvalue(), stderr.getvalue()
    )


class _RecordingServices:
    def __init__(self) -> None:
        self.config: RunConfig | None = None

    def _record(self, config: RunConfig, progress: ProgressSink) -> object:
        del progress
        self.config = config
        return {"action": config.action, "schema_version": config.schema_version}

    generate = _record
    plan = _record
    evaluate = _record
    benchmark = _record
    inspect = _record
    model_inspect = _record
    model_compile = _record
    model_processes = _record
    group_processes = _record


@pytest.mark.parametrize("card", RUN_CARDS, ids=lambda path: path.stem)
def test_cards_cross_cli_resolution_and_typed_dispatch_without_running_work(
    card: Path,
) -> None:
    stdout = io.StringIO()
    stderr = io.StringIO()
    services = _RecordingServices()
    status = run_cli(
        (str(card), "--progress", "off", "--json"),
        services=services,
        stdout=stdout,
        stderr=stderr,
    )

    assert status == 0, stderr.getvalue()
    assert services.config is not None
    assert json.loads(stdout.getvalue()) == {
        "action": services.config.action,
        "schema_version": 1,
    }
    assert stderr.getvalue() == ""


def test_copied_card_rebases_relative_paths_to_user_directory(tmp_path: Path) -> None:
    source = EXAMPLES / "evaluate_total.toml"
    copied = tmp_path / source.name
    shutil.copy2(source, copied)

    config = parse_cli((str(copied),)).resolve().effective
    assert config.evaluation.artifact == (tmp_path / "artifacts/pp_zjj")
    assert config.evaluation.process == "d d~ > g z g"
    assert config.evaluation.momenta == (tmp_path / "data/pp_zjj_momenta.json")
    assert config.evaluation.model_parameters == (
        tmp_path / "data/model_parameters.json"
    )


def test_direct_cli_example_resolves_process_set_and_ordered_override(
    tmp_path: Path,
) -> None:
    invocation = parse_cli(
        (
            "generate",
            "p p > Z j j",
            str(tmp_path / "pp_zjj"),
            "--model",
            str(tmp_path / "models/json/sm/sm.json"),
            "--restriction",
            "default",
            "--multiparticle",
            "p=d,d~,g",
            "--multiparticle",
            "j=d,d~,g",
            "--flavor-scheme",
            "2",
            "--max-quark-lines",
            "2",
            "--workers",
            "4",
            "--set",
            "generation.workers=2",
        )
    )
    config = invocation.resolve().effective
    assert config.process.entries == (ProcessEntry("p p > Z j j"),)
    assert config.process.multiparticles == {
        "p": ("d", "d~", "g"),
        "j": ("d", "d~", "g"),
    }
    assert config.process.flavor_scheme == 2
    assert config.process.max_quark_lines == 2
    assert config.model.restriction == "default"
    assert config.color.accuracy == "lc"
    assert config.generation.workers == 2


@pytest.mark.parametrize(
    "script",
    tuple(sorted((EXAMPLES / "python").glob("*.py"))),
    ids=lambda path: path.stem,
)
def test_python_examples_offer_help_without_running_generation(
    script: Path,
    tmp_path: Path,
) -> None:
    completed = _run_example(script, ("--help",), tmp_path)
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout
    assert "schema-v3 artifact writing" not in completed.stderr
    assert "no pyAmpliCol backend" not in completed.stderr


def test_packaged_model_helper_materializes_external_card_inputs(
    tmp_path: Path,
) -> None:
    script = EXAMPLES / "python/copy_packaged_models.py"
    completed = _run_example(script, (), tmp_path)
    assert completed.returncode == 0, completed.stderr
    assert Path(completed.stdout.strip()) == tmp_path / "models"

    expected = (
        tmp_path / "models/json/sm/sm.json",
        tmp_path / "models/json/scalars/scalars.json",
        tmp_path / "models/json/scalar_gravity/scalar_gravity.json",
        tmp_path / "models/ufo/sm/vertices.py",
    )
    assert all(path.is_file() for path in expected)

    copied_card = tmp_path / "generate_pp_zjj_from_ufo_sm.toml"
    shutil.copy2(EXAMPLES / copied_card.name, copied_card)
    config = parse_cli((str(copied_card),)).resolve().effective
    assert Path(config.model.source).is_file()

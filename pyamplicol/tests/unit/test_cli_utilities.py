# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from pyamplicol.cli import UtilityInvocation, parse_cli, run_cli
from pyamplicol.cli.utilities import list_examples


class _TTYStringIO(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_examples_list_is_checkout_independent_and_descriptive() -> None:
    invocation = parse_cli(("examples", "list", "--json"))
    assert isinstance(invocation, UtilityInvocation)
    assert invocation.output_format == "json"
    assert invocation.output_color == "auto"
    stdout = io.StringIO()
    assert run_cli(("examples", "list", "--json"), stdout=stdout) == 0
    entries = json.loads(stdout.getvalue())
    assert any(entry["name"] == "builtin_sm_lc" for entry in entries)
    heft = next(entry for entry in entries if entry["name"] == "builtin_sm_heft")
    assert "scalar HEFT" in heft["description"]
    assert "g g > H g g" in heft["description"]
    assert all("SPDX" not in entry["description"] for entry in entries)
    otf = next(entry for entry in entries if entry["name"] == "otf_pp_zjj")
    assert "first explicit warm_up or evaluation" in otf["description"]
    assert "complete helicity sum" in otf["description"]


def test_examples_list_presentation_options_are_explicit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    invocation = parse_cli(("examples", "list", "--color", "always"))
    assert isinstance(invocation, UtilityInvocation)
    assert invocation.output_format == "human"
    assert invocation.output_color == "always"

    machine = parse_cli(("examples", "list", "--json"))
    assert isinstance(machine, UtilityInvocation)
    assert machine.output_format == "json"

    with pytest.raises(SystemExit) as exc_info:
        parse_cli(("examples", "list", "--help"))
    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--json" in help_text
    assert "--color {auto,always,never}" in help_text
    assert "--format" not in help_text
    with pytest.raises(SystemExit):
        parse_cli(("examples", "list", "--format", "json"))


def test_utility_output_auto_colors_a_terminal_and_json_is_uncolored(
    tmp_path: Path,
) -> None:
    human = _TTYStringIO()
    assert run_cli(("examples", "list"), stdout=human) == 0
    assert "Packaged Examples" in human.getvalue()
    assert "\x1b[" in human.getvalue()
    with pytest.raises(json.JSONDecodeError):
        json.loads(human.getvalue())

    target = tmp_path / "template.toml"
    machine = _TTYStringIO()
    assert (
        run_cli(
            ("config", "template", str(target), "--json"),
            stdout=machine,
        )
        == 0
    )
    assert json.loads(machine.getvalue()) == {
        "destination": str(target.resolve()),
        "operation": "configuration template",
        "status": "complete",
    }
    assert "\x1b[" not in machine.getvalue()


def test_examples_run_help_lists_every_runnable_card(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        parse_cli(("examples", "run", "--help"))

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    runnable_names = {
        entry.name for entry in list_examples() if entry.action != "reference"
    }
    assert runnable_names
    assert all(f"  {name}\n" in help_text for name in runnable_names)
    assert "all_options" not in help_text


def test_examples_copy_requires_force_for_nonempty_destination(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "examples"
    assert run_cli(("examples", "copy", str(destination))) == 0
    assert (destination / "builtin_sm_lc.toml").is_file()
    assert (destination / "builtin_sm_heft.toml").is_file()
    assert (destination / "benchmark_z6g_single_flow_helicity_sum.toml").is_file()
    assert (destination / "benchmark_z6g_all_flows_single_helicity.toml").is_file()
    assert (destination / "otf_pp_zjj.toml").is_file()
    assert (destination / "python/otf_pp_zjj_warm_up.py").is_file()
    selected_card = (
        destination / "benchmark_z6g_single_flow_helicity_sum.toml"
    ).read_text(encoding="utf-8")
    union_card = (
        destination / "benchmark_z6g_all_flows_single_helicity.toml"
    ).read_text(encoding="utf-8")
    assert 'lc_flow_layout = "all-flow-union"' not in selected_card
    assert 'lc_flow_layout = "all-flow-union"' in union_card
    assert (destination / "models/json/sm/sm.json").is_file()
    assert (destination / "models/ufo/sm/vertices.py").is_file()
    expected_progress = {
        "evaluate_total.toml": "off",
        "evaluate_resolved.toml": "off",
        "benchmark.toml": "log",
    }
    for name, progress in expected_progress.items():
        card = parse_cli((str(destination / name),))
        assert not isinstance(card, UtilityInvocation)
        output = card.resolve().effective.output
        assert output.format == "human"
        assert output.color == "auto"
        assert output.progress == progress
    stderr = io.StringIO()
    assert run_cli(("examples", "copy", str(destination)), stderr=stderr) == 2
    assert "not empty" in stderr.getvalue()
    assert run_cli(("examples", "copy", str(destination), "--force")) == 0


def test_config_template_and_resolve(tmp_path: Path) -> None:
    target = tmp_path / "all.toml"
    assert run_cli(("config", "template", str(target))) == 0
    template = target.read_text(encoding="utf-8")
    assert "schema_version" in template
    for path in (
        "color.coverage",
        "color.flow_ids",
        "generation.validation.zero_current_filter",
        "generation.validation.current_merging",
    ):
        assert path not in template
    stdout = io.StringIO()
    assert (
        run_cli(
            (
                "config",
                "resolve",
                str(target),
                "--set",
                "generation.workers=1",
                "--json",
            ),
            stdout=stdout,
        )
        == 0
    )
    payload = json.loads(stdout.getvalue())
    assert payload["effective"]["generation"]["workers"] == 1


def test_generate_dry_run_does_not_require_output() -> None:
    invocation = parse_cli(("generate", "d d~ > z", "--dry-run"))
    assert invocation.dry_run is True
    assert invocation.resolve().effective.generation.output is None


def test_example_run_materializes_outside_package_resources(
    tmp_path: Path, monkeypatch: object
) -> None:
    monkeypatch.setenv("PYAMPLICOL_EXAMPLE_CACHE", str(tmp_path / "cache"))  # type: ignore[attr-defined]
    from pyamplicol.cli.utilities import example_card

    card = example_card("builtin_sm_lc")
    assert card == tmp_path / "cache/builtin_sm_lc.toml"
    assert (tmp_path / "cache/data/pp_zjj_momenta.json").is_file()
    assert (tmp_path / "cache/models/json/scalars/scalars.json").is_file()
    assert (tmp_path / "cache/models/ufo/sm/vertices.py").is_file()

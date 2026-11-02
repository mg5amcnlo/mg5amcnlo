# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import io
import json

import pytest

from pyamplicol.cli import LicenseRequestInvocation, parse_cli, run_cli


class _TTYStringIO(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_trial_request_parses_without_entering_run_configuration() -> None:
    invocation = parse_cli(
        [
            "request-symbolica-trial-license",
            "--name",
            "Ada",
            "--email",
            "ada@example.org",
            "--organization",
            "Institute",
            "--yes",
        ]
    )
    assert isinstance(invocation, LicenseRequestInvocation)
    assert invocation.kind == "trial"
    assert invocation.assume_yes


def test_noninteractive_request_requires_all_fields_and_yes() -> None:
    invocation = LicenseRequestInvocation(
        kind="hobbyist",
        name="Ada",
        email=None,
        assume_yes=True,
    )
    with pytest.raises(ValueError, match="missing email"):
        invocation.run(stdin=io.StringIO(), stdout=io.StringIO())


def test_license_request_uses_table_by_default_and_json_only_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "pyamplicol.cli.licensing.request_hobbyist_license",
        lambda name, email: requests.append((name, email)),
    )
    arguments = (
        "request-symbolica-hobbyist-license",
        "--name",
        "Ada",
        "--email",
        "ada@example.org",
        "--yes",
    )

    human = io.StringIO()
    assert run_cli((*arguments, "--color", "always"), stdout=human) == 0
    assert "Symbolica License Request" in human.getvalue()
    assert "accepted" in human.getvalue()
    assert "ada@example.org" not in human.getvalue()
    assert "\x1b[" in human.getvalue()

    machine = io.StringIO()
    assert run_cli((*arguments, "--json"), stdout=machine) == 0
    assert json.loads(machine.getvalue()) == {
        "next_step": "The license key will be sent by email.",
        "request": "hobbyist",
        "status": "accepted",
    }
    assert "\x1b[" not in machine.getvalue()
    assert requests == [
        ("Ada", "ada@example.org"),
        ("Ada", "ada@example.org"),
    ]


def test_json_license_request_never_prompts_on_stdout() -> None:
    stdout = _TTYStringIO()
    stderr = io.StringIO()
    status = run_cli(
        ("request-symbolica-hobbyist-license", "--json"),
        stdin=_TTYStringIO("Ada\nada@example.org\nyes\n"),
        stdout=stdout,
        stderr=stderr,
    )
    assert status == 2
    assert stdout.getvalue() == ""
    assert "all fields and --yes" in stderr.getvalue()

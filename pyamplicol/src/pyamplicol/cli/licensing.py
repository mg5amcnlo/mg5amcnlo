# SPDX-License-Identifier: 0BSD
"""Transient CLI requests for Symbolica licenses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TextIO

from pyamplicol.licensing import request_hobbyist_license, request_trial_license

LicenseKind = Literal["trial", "hobbyist"]


@dataclass(frozen=True, slots=True)
class LicenseRequestResult:
    request: LicenseKind
    status: Literal["accepted"] = "accepted"
    next_step: str = "The license key will be sent by email."


@dataclass(frozen=True, slots=True)
class LicenseRequestInvocation:
    kind: LicenseKind
    name: str | None
    email: str | None
    organization: str | None = None
    assume_yes: bool = False
    output_format: Literal["human", "json"] = "human"
    output_color: Literal["auto", "always", "never"] = "auto"

    def run(self, *, stdin: TextIO, stdout: TextIO) -> LicenseRequestResult:
        if self.output_format == "json" and (
            self.name is None
            or self.email is None
            or (self.kind == "trial" and self.organization is None)
            or not self.assume_yes
        ):
            raise ValueError(
                "--json license requests require all fields and --yes; "
                "interactive prompts are disabled to keep stdout machine-readable"
            )
        name = self.name or _prompt(stdin, stdout, "Name")
        email = self.email or _prompt(stdin, stdout, "Email")
        organization: str | None = None
        if self.kind == "trial":
            organization = self.organization or _prompt(
                stdin,
                stdout,
                "Organization",
            )
        if not self.assume_yes:
            if not bool(getattr(stdin, "isatty", lambda: False)()):
                raise ValueError(
                    "noninteractive license requests require complete fields and --yes"
                )
            answer = _prompt(
                stdin,
                stdout,
                "Submit this request to Symbolica? [y/N]",
            )
            if answer.casefold() not in {"y", "yes"}:
                raise ValueError("Symbolica license request cancelled")
        if self.kind == "trial":
            assert organization is not None
            request_trial_license(name, email, organization)
        else:
            request_hobbyist_license(name, email)
        return LicenseRequestResult(self.kind)


def _prompt(stdin: TextIO, stdout: TextIO, label: str) -> str:
    if not bool(getattr(stdin, "isatty", lambda: False)()):
        raise ValueError(f"noninteractive license request is missing {label.lower()}")
    stdout.write(f"{label}: ")
    stdout.flush()
    value = stdin.readline()
    if not value:
        raise ValueError("unexpected end of input during license request")
    value = value.strip()
    if not value:
        raise ValueError(f"{label} must not be empty")
    return value


__all__ = ["LicenseKind", "LicenseRequestInvocation", "LicenseRequestResult"]

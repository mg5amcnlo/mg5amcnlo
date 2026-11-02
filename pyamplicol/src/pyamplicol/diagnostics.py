# SPDX-License-Identifier: 0BSD
"""Fast installed-package diagnostics used by ``doctor`` and ``self-test``."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import shutil
import sys
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Literal

from . import __version__
from ._internal.versions import verify_native_module

CheckStatus = Literal["pass", "warning", "fail"]


@dataclass(frozen=True, slots=True)
class DiagnosticCheck:
    name: str
    status: CheckStatus
    detail: str


@dataclass(frozen=True, slots=True)
class DiagnosticReport:
    package_version: str
    checks: tuple[DiagnosticCheck, ...]

    @property
    def ok(self) -> bool:
        return all(check.status != "fail" for check in self.checks)


def _python_check() -> DiagnosticCheck:
    version = ".".join(str(item) for item in sys.version_info[:3])
    status: CheckStatus = "pass" if sys.version_info >= (3, 11) else "fail"
    return DiagnosticCheck("python", status, version)


def _asset_check() -> DiagnosticCheck:
    root = resources.files("pyamplicol").joinpath("assets", "models")
    try:
        lines = (
            root.joinpath("MANIFEST.sha256").read_text(encoding="utf-8").splitlines()
        )
        if not lines:
            raise ValueError("empty model manifest")
        for line in lines:
            digest, separator, relative = line.partition("  ")
            if not separator or len(digest) != 64 or not relative:
                raise ValueError(f"malformed model manifest line: {line!r}")
            payload = root.joinpath(*relative.split("/"))
            actual = hashlib.sha256(payload.read_bytes()).hexdigest()
            if actual != digest:
                raise ValueError(f"model asset digest mismatch: {relative}")
    except (FileNotFoundError, OSError, ValueError) as exc:
        return DiagnosticCheck("model-assets", "fail", str(exc))
    return DiagnosticCheck("model-assets", "pass", f"{len(lines)} verified files")


def _native_check(*, required: bool) -> DiagnosticCheck:
    try:
        native = importlib.import_module("pyamplicol._rusticol")
        verify_native_module(native)
        abi_version = int(native.abi_version())
        native_version = str(native.package_version())
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as exc:
        return DiagnosticCheck(
            "rusticol-python",
            "fail" if required else "warning",
            f"native extension unavailable: {exc}",
        )
    if abi_version != 1:
        return DiagnosticCheck(
            "rusticol-python",
            "fail",
            f"unsupported C ABI {abi_version}; expected 1",
        )
    return DiagnosticCheck(
        "rusticol-python", "pass", f"package {native_version}, C ABI {abi_version}"
    )


def _sdk_check(*, required: bool) -> DiagnosticCheck:
    try:
        from ._sdk.config import load_sdk_info

        info = load_sdk_info()
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return DiagnosticCheck(
            "native-sdk",
            "fail" if required else "warning",
            f"SDK unavailable: {exc}",
        )
    return DiagnosticCheck(
        "native-sdk",
        "pass",
        f"{info.target}; ABI {info.abi_version}; {info.library.name}",
    )


def _complex_pair(value: object, description: str) -> complex:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or isinstance(value[0], bool)
        or isinstance(value[1], bool)
        or not isinstance(value[0], (int, float))
        or not isinstance(value[1], (int, float))
    ):
        raise ValueError(f"{description} must be a [real, imaginary] pair")
    return complex(float(value[0]), float(value[1]))


def _physics_check() -> DiagnosticCheck:
    try:
        native = importlib.import_module("pyamplicol._rusticol")
        verify_native_module(native)
        target = str(native.target_info().triple)
        fixture = resources.files("pyamplicol").joinpath("assets", "selftest", target)
        expected_raw = json.loads(
            fixture.joinpath("expected.json").read_text(encoding="utf-8")
        )
        if (
            not isinstance(expected_raw, dict)
            or expected_raw.get("schema_version") != 1
        ):
            raise ValueError("self-test expectation must be a schema-v1 object")
        artifact = fixture.joinpath(str(expected_raw.get("artifact_path", "")))
        artifact_path = Path(str(artifact))
        if not artifact_path.is_dir():
            raise ValueError(f"self-test artifact is unavailable for {target}")

        from .api.services import Runtime

        runtime = Runtime.load(artifact_path, mute_warnings=True)
        process_id = str(expected_raw.get("process_id", ""))
        if runtime.physics.process_id != process_id:
            raise ValueError(
                "self-test process metadata differs from its recorded expectation"
            )
        momenta = expected_raw.get("momenta")
        if not isinstance(momenta, list):
            raise ValueError("self-test expectation has no momentum batch")
        total = runtime.evaluate(momenta)
        resolved = runtime.evaluate_resolved(momenta)
        expected_total_raw = expected_raw.get("total")
        if not isinstance(expected_total_raw, list):
            raise ValueError("self-test expectation has no total values")
        expected_total = tuple(
            _complex_pair(value, f"self-test total {index}")
            for index, value in enumerate(expected_total_raw)
        )
        if len(total) != len(expected_total):
            raise ValueError("self-test total has the wrong point count")
        for index, (actual, expected) in enumerate(
            zip(total, expected_total, strict=True)
        ):
            value = complex(actual)
            if not (
                math.isclose(
                    value.real, expected.real, rel_tol=1.0e-12, abs_tol=1.0e-14
                )
                and math.isclose(
                    value.imag, expected.imag, rel_tol=1.0e-12, abs_tol=1.0e-14
                )
            ):
                raise ValueError(
                    f"self-test total {index} is {value!r}, expected {expected!r}"
                )
        reduced = tuple(complex(value) for value in resolved.total())
        for index, (actual, expected) in enumerate(
            zip(reduced, expected_total, strict=True)
        ):
            if not (
                math.isclose(
                    actual.real, expected.real, rel_tol=1.0e-12, abs_tol=1.0e-14
                )
                and math.isclose(
                    actual.imag, expected.imag, rel_tol=1.0e-12, abs_tol=1.0e-14
                )
            ):
                raise ValueError(
                    f"resolved self-test total {index} does not reproduce evaluate()"
                )
        expected_shape = expected_raw.get("resolved_shape")
        if list(resolved.shape) != expected_shape:
            raise ValueError(
                f"resolved self-test shape {resolved.shape!r} differs from "
                f"{expected_shape!r}"
            )
    except Exception as exc:
        return DiagnosticCheck("physics-f64", "fail", str(exc))
    return DiagnosticCheck(
        "physics-f64",
        "pass",
        f"{process_id}; shape {resolved.shape}; direct SymJIT on {target}",
    )


def _symbolica_check() -> DiagnosticCheck:
    try:
        from .licensing import detect_symbolica_license

        state = detect_symbolica_license(suggest=False, json_mode=True)
    except (ImportError, RuntimeError) as exc:
        return DiagnosticCheck("symbolica", "fail", str(exc))
    mode = "licensed" if state.licensed else "restricted single-core"
    return DiagnosticCheck("symbolica", "pass", mode)


def _tool_check(name: str, *, required: bool) -> DiagnosticCheck:
    path = shutil.which(name)
    if path is None:
        return DiagnosticCheck(
            f"tool:{name}",
            "fail" if required else "warning",
            "not found on PATH",
        )
    return DiagnosticCheck(f"tool:{name}", "pass", path)


def run_doctor() -> DiagnosticReport:
    """Inspect source/install capabilities without running physics generation."""

    checks = (
        _python_check(),
        _asset_check(),
        _native_check(required=False),
        _sdk_check(required=False),
        _symbolica_check(),
        _tool_check("cc", required=False),
        _tool_check("c++", required=False),
        _tool_check("gfortran", required=False),
    )
    return DiagnosticReport(__version__, checks)


def run_self_test() -> DiagnosticReport:
    """Run the fast checks required of an installed binary distribution."""

    checks = (
        _python_check(),
        _asset_check(),
        _native_check(required=True),
        _sdk_check(required=True),
        _physics_check(),
    )
    return DiagnosticReport(__version__, checks)


__all__ = [
    "CheckStatus",
    "DiagnosticCheck",
    "DiagnosticReport",
    "run_doctor",
    "run_self_test",
]

# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import pyamplicol.diagnostics as diagnostics
from pyamplicol.api.services import Runtime
from pyamplicol.diagnostics import DiagnosticCheck, DiagnosticReport, run_doctor


def test_diagnostic_report_fails_only_for_failed_checks() -> None:
    warning = DiagnosticReport(
        "0.1.0", (DiagnosticCheck("optional", "warning", "missing"),)
    )
    failure = DiagnosticReport(
        "0.1.0", (DiagnosticCheck("required", "fail", "missing"),)
    )
    assert warning.ok
    assert not failure.ok


def test_source_doctor_verifies_packaged_model_assets() -> None:
    report = run_doctor()
    assets = next(check for check in report.checks if check.name == "model-assets")
    assert assets.status == "pass"
    assert "verified files" in assets.detail


def test_physics_selftest_uses_target_fixture_without_symbolica(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    target = "aarch64-apple-darwin"
    fixture = tmp_path / "assets" / "selftest" / target
    (fixture / "artifact").mkdir(parents=True)
    (fixture / "expected.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_path": "artifact",
                "target": target,
                "process_id": "d_dbar_to_z",
                "momenta": [[[45.594, 0.0, 0.0, 45.594]]],
                "resolved_shape": [1, 2, 1],
                "total": [[3.5, 0.0]],
            }
        ),
        encoding="utf-8",
    )

    class Resolved:
        shape = (1, 2, 1)

        @staticmethod
        def total() -> tuple[complex, ...]:
            return (3.5 + 0.0j,)

    runtime = SimpleNamespace(
        physics=SimpleNamespace(process_id="d_dbar_to_z"),
        evaluate=lambda _momenta: (3.5 + 0.0j,),
        evaluate_resolved=lambda _momenta: Resolved(),
    )
    imports: list[str] = []
    original_import = diagnostics.importlib.import_module

    def fake_import(name: str):
        imports.append(name)
        if name == "pyamplicol._rusticol":
            return SimpleNamespace(target_info=lambda: SimpleNamespace(triple=target))
        return original_import(name)

    monkeypatch.setattr(diagnostics.importlib, "import_module", fake_import)
    monkeypatch.setattr(diagnostics.resources, "files", lambda _package: tmp_path)
    monkeypatch.setattr(
        Runtime,
        "load",
        classmethod(lambda _cls, _artifact, **_kwargs: runtime),
    )

    check = diagnostics._physics_check()

    assert check.status == "pass"
    assert "direct SymJIT" in check.detail
    assert imports == ["pyamplicol._rusticol"]

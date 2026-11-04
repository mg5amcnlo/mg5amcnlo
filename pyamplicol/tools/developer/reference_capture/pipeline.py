# SPDX-License-Identifier: 0BSD
"""Capture orchestration with a validate-before-publish boundary."""

from __future__ import annotations

from datetime import UTC, datetime

from .artifacts import (
    artifact_capture_specs,
    assemble_fixture,
    materialize_artifacts,
)
from .common import (
    BUNDLE_MANIFEST_FILENAME,
    ROOT,
    WATCHDOG_GB,
    CaptureConfig,
    CaptureError,
    CaptureResult,
)
from .evidence import (
    build_analytic_evidence,
    build_fortran_evidence,
    validate_and_publish,
)
from .provenance import (
    assert_runtime_snapshot_unchanged,
    assert_source_snapshot_unchanged,
    collect_dependency_snapshot,
    collect_runtime_snapshot,
    collect_source_snapshot,
)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_capture(config: CaptureConfig) -> CaptureResult:
    """Run artifact capture, evidence, strict validation, and publication."""

    if config.external_watchdog_gb != WATCHDOG_GB:
        raise CaptureError(
            f"capture requires confirmation of an external {WATCHDOG_GB} GB watchdog"
        )
    source = collect_source_snapshot(ROOT)
    runtime = collect_runtime_snapshot(source)
    dependencies = collect_dependency_snapshot(runtime)
    specs = artifact_capture_specs(config.artifact_root)
    artifact_paths = materialize_artifacts(
        specs, config.artifact_root, config.artifact_mode
    )
    captured_at = _utc_now()
    fixture = assemble_fixture(
        specs,
        artifact_paths,
        source,
        runtime,
        dependencies,
        captured_at=captured_at,
        capture_command=config.capture_command,
    )
    analytic_evidence = build_analytic_evidence(fixture)
    fortran_evidence = build_fortran_evidence(
        fixture,
        config.legacy_repository,
        config.legacy_jobs,
    )
    assert_source_snapshot_unchanged(source, ROOT)
    assert_runtime_snapshot_unchanged(runtime, source)
    written = validate_and_publish(
        config.output_directory,
        fixture,
        fortran_evidence,
        analytic_evidence,
    )
    return CaptureResult(
        fixture_path=written[0],
        evidence_paths=(written[1], written[2]),
        bundle_manifest_path=config.output_directory / BUNDLE_MANIFEST_FILENAME,
        artifact_paths=tuple(artifact_paths[spec.id] for spec in specs),
    )

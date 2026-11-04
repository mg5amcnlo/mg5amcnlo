#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Required source-distribution inventory for release audits."""

from __future__ import annotations

from collections.abc import Collection

PREPARED_MODEL_ARCHITECTURES = ("aarch64", "x86_64")
PREPARED_MODEL_ASSET_BASENAME = "built-in-sm-jit-o2"
PREPARED_MODEL_ASSETS = {
    PREPARED_MODEL_ASSET_BASENAME: "built-in-sm",
    "built-in-sm-heft-jit-o2": "built-in-sm-heft",
}


def prepared_model_asset_members(prefix: str) -> frozenset[str]:
    """Return the exact prepared-model inventory rooted at *prefix*."""

    root = prefix.rstrip("/")
    members = {f"{root}/__init__.py"}
    for identifier in PREPARED_MODEL_ASSETS:
        for architecture in PREPARED_MODEL_ARCHITECTURES:
            stem = f"{identifier}-{architecture}"
            members.update(
                {
                    f"{root}/{stem}.metadata.json",
                    f"{root}/{stem}.pyamplicol-model",
                }
            )
    return frozenset(members)


PREPARED_MODEL_SDIST_MEMBERS = prepared_model_asset_members(
    "src/pyamplicol/assets/prepared_models"
)

REQUIRED_SDIST_MEMBERS = frozenset(
    {
        "Cargo.lock",
        "Cargo.toml",
        "rust-toolchain.toml",
        "LICENSE",
        "README.md",
        "THIRD_PARTY_NOTICES.md",
        "justfile",
        "build_backend/_pyamplicol_build.py",
        "build_backend/sdk.py",
        "dependencies/release-lock.toml",
        "docs/README.md",
        "docs/user/installation.md",
        "examples/builtin_sm_lc.toml",
        "examples/data/pp_zjj_momenta.json",
        "examples/native/Makefile",
        "examples/python/typed_generation.py",
        "pyproject.toml",
        "rust/crates/rusticol-capi/Cargo.toml",
        "rust/crates/rusticol-capi/build.rs",
        "rust/crates/rusticol-capi/fortran/rusticol.f90",
        "rust/crates/rusticol-capi/include/rusticol.h",
        "rust/crates/rusticol-capi/include/rusticol.hpp",
        "rust/crates/rusticol-capi/src/lib.rs",
        "rust/crates/rusticol-python/stubs/pyamplicol/_rusticol.pyi",
        "schemas/README.md",
        "schemas/artifact-manifest-v3.schema.json",
        "schemas/reference-oracle-evidence-v2.schema.json",
        "schemas/reference-fixture-bundle-v1.schema.json",
        "schemas/reference-physics-v2.schema.json",
        "schemas/runtime-physics-v1.schema.json",
        "src/pyamplicol/assets/api_templates/rust/Makefile",
        "src/pyamplicol/assets/api_templates/rust/check_standalone.rs",
        "src/pyamplicol/assets/api_templates/umami/Makefile",
        "src/pyamplicol/assets/api_templates/umami/umami.h",
        "src/pyamplicol/assets/api_templates/umami/umami.c",
        "src/pyamplicol/assets/api_templates/umami/umami_provider.h",
        "src/pyamplicol/assets/api_templates/umami/umami_driver.c",
        "src/pyamplicol/assets/api_templates/umami/umami.exports",
        "src/pyamplicol/assets/api_templates/umami/umami.exports.macos",
        "src/pyamplicol/assets/api_templates/umami/UMAMI_LICENSE",
        "src/pyamplicol/assets/api_templates/umami/README.md",
        "src/pyamplicol/_build_info.json",
        "src/rusticol_config/__init__.py",
        "src/rusticol_config/__main__.py",
        *PREPARED_MODEL_SDIST_MEMBERS,
        "src/pyamplicol/assets/selftest/portable-64le/expected.json",
        "src/pyamplicol/assets/selftest/portable-64le/artifact/artifact.json",
        "tests/integration/test_examples.py",
        "tests/fixtures/reference/analytic-oracles-v2.json",
        "tests/fixtures/reference/legacy-fortran-v2.json",
        "tests/fixtures/reference/physics-v2.json",
        "tests/fixtures/reference/reference-fixture-v2.manifest.json",
        "tests/release/test_artifacts.py",
        "tests/unit/test_capture_reference_fixture_v2.py",
        "tests/unit/test_reference_fixture_v2.py",
        "tests/unit/test_tracked_reference_fixture_v2.py",
        "tests/unit/test_cli_utilities.py",
        "tools/developer/analytic_oracles.py",
        "tools/developer/capture_reference_fixture_v2.py",
        "tools/developer/legacy_amplicol.py",
        "tools/developer/legacy_oracle/__init__.py",
        "tools/developer/legacy_oracle/checkout.py",
        "tools/developer/legacy_oracle/evidence.py",
        "tools/developer/legacy_oracle/model.py",
        "tools/developer/legacy_oracle/probe.py",
        "tools/developer/legacy_oracle/processes.py",
        "tools/developer/prepare_source_runtime.py",
        "tools/developer/reference_capture/__init__.py",
        "tools/developer/reference_capture/artifacts.py",
        "tools/developer/reference_capture/common.py",
        "tools/developer/reference_capture/evidence.py",
        "tools/developer/reference_capture/pipeline.py",
        "tools/developer/reference_capture/points.py",
        "tools/developer/reference_capture/provenance.py",
        "tools/developer/reference_fixture/__init__.py",
        "tools/developer/reference_fixture/api.py",
        "tools/developer/reference_fixture/codec.py",
        "tools/developer/reference_fixture/evidence.py",
        "tools/developer/reference_fixture/model.py",
        "tools/developer/reference_fixture/numerics.py",
        "tools/developer/reference_fixture/structure.py",
        "tools/developer/reference_fixture_v2.py",
        "tools/release/_artifacts.py",
        "tools/release/_common.py",
        "tools/release/audit_sdist.py",
        "tools/release/build_from_sdist.py",
        "tools/release/build_release_artifacts.py",
        "tools/release/check_dependencies.py",
        "tools/release/install_wheel.py",
        "tools/release/prepare_selftest_fixture.py",
        "tools/release/publish_dry_run.py",
        "tools/release/run_cargo.py",
        "tools/release/test_deployment.py",
        "tools/typing/check_public_typing.py",
    }
)


def missing_required_sdist_members(members: Collection[str]) -> tuple[str, ...]:
    """Return required release members absent from an extracted sdist."""

    return tuple(sorted(REQUIRED_SDIST_MEMBERS.difference(members)))


__all__ = [
    "PREPARED_MODEL_ARCHITECTURES",
    "PREPARED_MODEL_ASSETS",
    "PREPARED_MODEL_ASSET_BASENAME",
    "PREPARED_MODEL_SDIST_MEMBERS",
    "REQUIRED_SDIST_MEMBERS",
    "missing_required_sdist_members",
    "prepared_model_asset_members",
]

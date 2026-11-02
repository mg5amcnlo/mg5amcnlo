# SPDX-License-Identifier: 0BSD
"""Independent public, serialization, and evaluator runtime contracts."""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import tomllib
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path
from typing import Any

PYTHON_API_VERSION = 1
TOML_SCHEMA_VERSION = 1
COMPILED_MODEL_SCHEMA_VERSION = 9
PROCESS_ARTIFACT_SCHEMA_VERSION = 3
RUNTIME_PHYSICS_SCHEMA_VERSION = 1
C_ABI_VERSION = 1

# These are project-owned wire-format identifiers. Dependency source revisions
# live in the release and contributor locks, outside serialized artifacts.
SYMBOLICA_SERIALIZATION_ABI = "symbolica-bincode2-v1"
SYMJIT_APPLICATION_ABI = "symjit-application-storage-v3"
SYMJIT_PLANE_APPLICATION_ABI = "pyamplicol-symjit-plane-application-v2"
NATIVE_COMPILED_DIRECT_APPLICATION_ABI = (
    "pyamplicol-native-compiled-direct-application-v1"
)
NATIVE_EAGER_DIRECT_TABLE_APPLICATION_ABI = "pyamplicol-eager-native-direct-table-v1"
COMPILED_PLANE_DIRECT_APPLICATION_ABI = "pyamplicol-compiled-plane-kernel-v2"
EAGER_DIRECT_TABLE_DESCRIPTOR_ABI = "pyamplicol-eager-plane-table-descriptor-v1"
EAGER_DIRECT_TABLE_BINDING_ABI = "pyamplicol-eager-plane-table-binding-v2"
RECURRENCE_DIRECT_BINDING_ABI = "pyamplicol-recurrence-plane-binding-v2"

SYMJIT_F64_RUNTIME_CAPABILITY = "symjit.application.complex-f64.v1"
SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY = (
    "symbolica.legacy-jit-container.complex-f64.v1"
)
SYMBOLICA_CPP_RUNTIME_CAPABILITY = "symbolica.compiled-cpp.complex-f64.v1"
SYMBOLICA_ASM_RUNTIME_CAPABILITY = "symbolica.compiled-asm.complex-f64.v1"
EAGER_DAG_F64_RUNTIME_CAPABILITY = "rusticol.eager-dag.complex-f64.v1"
EAGER_RUNTIME_LAYOUT_F64_CAPABILITY = "rusticol.eager-runtime-layout.complex-f64.v1"
EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY = "eager-direct-arena-v1"
EAGER_LC_TOPOLOGY_REPLAY_RUNTIME_CAPABILITY = "rusticol.eager-dag.lc-topology-replay.v1"
RECURRENCE_BUILDER_INPUT_ABI = "pyamplicol-recurrence-builder-input-v2"
RECURRENCE_PLAN_ABI = "pyamplicol-recurrence-plan-v2"
RECURRENCE_RUNTIME_LAYOUT_ABI = "pyamplicol-recurrence-runtime-layout-v2"
RECURRENCE_DIRECT_TEMPLATE_ABI = "pyamplicol-recurrence-direct-template-v1"
RECURRENCE_DIRECT_BACKEND_ABI = "rusticol.recurrence-direct-backend.v1"
RECURRENCE_DIRECT_ARENA_RUNTIME_CAPABILITY = (
    "rusticol.recurrence-direct-arena.complex-f64.v1"
)
RECURRENCE_COLOR_RUNTIME_CAPABILITY = "rusticol.recurrence-color.lc.v1"
RECURRENCE_CONTRACTED_COLOR_RUNTIME_CAPABILITY = (
    "rusticol.recurrence-color.contracted.v1"
)
RECURRENCE_HELICITY_SELECTOR_COMPANION_RUNTIME_CAPABILITY = (
    "rusticol.recurrence-helicity-selector-companion.v2"
)
ON_THE_FLY_RUNTIME_CAPABILITY = "rusticol.on-the-fly.complex-f64.v1"
ON_THE_FLY_LC_COLOR_RUNTIME_CAPABILITY = "rusticol.on-the-fly.lc-color.v1"
ON_THE_FLY_CONTRACTED_COLOR_RUNTIME_CAPABILITY = (
    "rusticol.on-the-fly.contracted-color.v1"
)
SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY = (
    "rusticol.color-contraction.symmetric-group-fft.v1"
)
COMPILED_RUNTIME_SELECTORS_CAPABILITY = "rusticol.compiled.runtime-selectors.v1"
COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY = "compiled-plane-arena-v1"
COMPILED_HELICITY_DUAL_LANE_CAPABILITY = "rusticol.compiled.helicity-dual-lane.v1"
COMPILED_HELICITY_SELECTOR_UNION_CAPABILITY = (
    "rusticol.compiled.helicity-selector-union.v1"
)
COMPILED_HELICITY_PRIMARY_RECURRENCE_CAPABILITY = (
    "rusticol.compiled.helicity-primary-recurrence.v1"
)
COMPILED_COLOR_TOPOLOGY_LANES_CAPABILITY = "rusticol.compiled.color-topology-lanes.v1"
COMPILED_COLOR_CONTRACTION_WALSH_CAPABILITY = (
    "rusticol.compiled.color-contraction-walsh.v1"
)
COMPILED_COLOR_CONTRACTION_WALSH_C2K_CAPABILITY = (
    "rusticol.compiled.color-contraction-walsh-c2k.v1"
)
EVALUATOR_RUNTIME_CAPABILITIES = frozenset(
    {
        COMPILED_COLOR_CONTRACTION_WALSH_C2K_CAPABILITY,
        COMPILED_COLOR_CONTRACTION_WALSH_CAPABILITY,
        COMPILED_COLOR_TOPOLOGY_LANES_CAPABILITY,
        COMPILED_HELICITY_DUAL_LANE_CAPABILITY,
        COMPILED_HELICITY_PRIMARY_RECURRENCE_CAPABILITY,
        COMPILED_HELICITY_SELECTOR_UNION_CAPABILITY,
        COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY,
        COMPILED_RUNTIME_SELECTORS_CAPABILITY,
        EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY,
        EAGER_RUNTIME_LAYOUT_F64_CAPABILITY,
        RECURRENCE_COLOR_RUNTIME_CAPABILITY,
        RECURRENCE_CONTRACTED_COLOR_RUNTIME_CAPABILITY,
        RECURRENCE_HELICITY_SELECTOR_COMPANION_RUNTIME_CAPABILITY,
        RECURRENCE_DIRECT_ARENA_RUNTIME_CAPABILITY,
        ON_THE_FLY_LC_COLOR_RUNTIME_CAPABILITY,
        ON_THE_FLY_CONTRACTED_COLOR_RUNTIME_CAPABILITY,
        SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY,
        ON_THE_FLY_RUNTIME_CAPABILITY,
        SYMJIT_F64_RUNTIME_CAPABILITY,
        SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY,
        SYMBOLICA_CPP_RUNTIME_CAPABILITY,
        SYMBOLICA_ASM_RUNTIME_CAPABILITY,
    }
)
KNOWN_EVALUATOR_RUNTIME_CAPABILITIES = EVALUATOR_RUNTIME_CAPABILITIES | frozenset(
    {
        # Historical eager plan-v2 artifacts remain parseable for inspection
        # and actionable migration errors, but are not executable capabilities.
        EAGER_DAG_F64_RUNTIME_CAPABILITY,
        EAGER_LC_TOPOLOGY_REPLAY_RUNTIME_CAPABILITY,
    }
)

_SOURCE_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = Path(__file__).resolve().parents[3]
_PACKAGE_BUILD_INFO_PATH = _SOURCE_PACKAGE_ROOT / "_build_info.json"
_SOURCE_RUNTIME_ROOT = _SOURCE_ROOT / ".artifacts" / "source-runtime"
_SOURCE_BUILD_INFO_PATH = _SOURCE_RUNTIME_ROOT / "_build_info.json"
_SOURCE_RUNTIME_STAGING_PATH = _SOURCE_RUNTIME_ROOT / ".staging"

_NATIVE_BUILD_IDENTITY_DOMAIN = b"pyamplicol-native-build-inputs-v2\0"
_NATIVE_BUILD_INPUT_FILES = (
    Path("Cargo.lock"),
    Path("Cargo.toml"),
    Path("pyproject.toml"),
    Path("rust-toolchain.toml"),
    Path("dependencies/candidate-Cargo.lock"),
    Path("dependencies/candidate-cargo-config.toml"),
    Path("dependencies/install-state.json"),
)
_NATIVE_BUILD_INPUT_TREES = (Path("rust"),)
# Keep this projection identical to build_backend/native_build_identity.py
# without importing build-only modules. Release sdists omit candidate inputs.
_RELEASE_OMITTED_NATIVE_BUILD_INPUTS = frozenset(
    {
        Path("dependencies/candidate-Cargo.lock"),
        Path("dependencies/candidate-cargo-config.toml"),
        Path("dependencies/install-state.json"),
    }
)
_NATIVE_MATURIN_CONFIG_KEYS = frozenset(
    {
        "all-features",
        "bindings",
        "config",
        "features",
        "include-debuginfo",
        "manifest-path",
        "module-name",
        "no-default-features",
        "profile",
        "rustc-args",
        "strip",
        "target",
        "targets",
        "unstable-flags",
        "zig",
    }
)
_NON_OUTPUT_MATURIN_CONFIG_KEYS = frozenset(
    {
        "editable-profile",
        "frozen",
        "locked",
        "pgo-command",
        "target-dir",
        "use-base-python",
    }
)
_PACKAGING_MATURIN_CONFIG_KEYS = frozenset(
    {
        "auditwheel",
        "compatibility",
        "data",
        "exclude",
        "generate-ci",
        "generate-stubs",
        "include",
        "include-import-lib",
        "manylinux",
        "python-packages",
        "python-source",
        "sbom",
        "sdist-generator",
        "skip-auditwheel",
        "compression-enable-large-file-support",
        "compression-level",
        "compression-method",
        "sbom-include",
    }
)
_NATIVE_BUILD_IGNORED_TREE_PARTS = frozenset({".artifacts", "__pycache__", "target"})
_NON_NATIVE_RUST_PATHS = frozenset(
    {
        Path("rust/crates/rusticol-capi/tests/eager_artifact.rs"),
        Path("rust/crates/rusticol-capi/tests/runtime_selectors.rs"),
        Path("rust/crates/rusticol-core/src/artifact_tests.rs"),
        Path("rust/crates/rusticol-core/src/eager_layout_tests.rs"),
        Path("rust/crates/rusticol-core/src/eager_lowering_v3_tests.rs"),
        Path("rust/crates/rusticol-core/src/eager_plan_v3_pacbin_tests.rs"),
        Path("rust/crates/rusticol-core/src/eager_runtime/plan_v3_tests.rs"),
        Path("rust/crates/rusticol-core/src/engine/contraction_metadata_tests.rs"),
        Path("rust/crates/rusticol-core/src/engine/eager_integration_tests.rs"),
        Path("rust/crates/rusticol-core/src/engine/eager_v3_manifest_tests.rs"),
        Path("rust/crates/rusticol-core/src/engine/quantum_number_flow_tests.rs"),
        Path("rust/crates/rusticol-core/src/engine/recurrence_integration_tests.rs"),
        Path("rust/crates/rusticol-core/src/engine/source_metadata_tests.rs"),
        Path("rust/crates/rusticol-core/src/engine_tests.rs"),
        Path("rust/crates/rusticol-core/src/metadata_tests.rs"),
        Path("rust/crates/rusticol-core/src/pacbin_tests.rs"),
        Path("rust/crates/rusticol-core/src/recurrence/direct_backend_tests.rs"),
        Path("rust/crates/rusticol-core/src/recurrence/direct_codec_tests.rs"),
        Path("rust/crates/rusticol-core/src/recurrence/direct_lowering_tests.rs"),
        Path("rust/crates/rusticol-core/src/recurrence/direct_pacbin_tests.rs"),
        Path("rust/crates/rusticol-core/src/recurrence/direct_plan_tests.rs"),
        Path("rust/crates/rusticol-core/src/recurrence/direct_runtime_tests.rs"),
        Path("rust/crates/rusticol-core/src/recurrence/tests.rs"),
        Path("rust/crates/rusticol-core/tests/direct_arena_workspace_allocations.rs"),
        Path("rust/crates/rusticol-core/tests/eager_runtime.rs"),
        Path(
            "rust/crates/rusticol-core/tests/fixtures/on_the_fly_query_parity_v1.json"
        ),
        Path(
            "rust/crates/rusticol-core/tests/fixtures/recurrence_execution_hzz_full_v2.json"
        ),
        Path(
            "rust/crates/rusticol-core/tests/fixtures/recurrence_execution_hzz_lc.json"
        ),
        Path(
            "rust/crates/rusticol-core/tests/fixtures/recurrence_execution_hzz_nlc.json"
        ),
        Path("rust/crates/rusticol-core/tests/generated_artifact_odd_tails.rs"),
        Path("rust/crates/rusticol-core/tests/recurrence_direct_arena_allocations.rs"),
        Path("rust/crates/rusticol-python/stubs/pyamplicol/__init__.pyi"),
        Path("rust/crates/rusticol-python/stubs/pyamplicol/_rusticol.pyi"),
        Path("rust/crates/rusticol-python/tests/pyrightconfig.json"),
        Path("rust/crates/rusticol-python/tests/stub_contract.rs"),
        Path("rust/crates/rusticol-python/tests/typing_consumer.py"),
    }
)
_NATIVE_BUILD_CONTRACT_KEYS = frozenset(
    {"macos", "rust-path-remapping", "schema-version", "sdk"}
)
_RUST_PATH_REMAP_KEYS = frozenset(
    {"build", "candidate-checkouts", "checkout", "source", "sysroot"}
)
_MACOS_NATIVE_BUILD_KEYS = frozenset({"cc", "cxx", "deployment-target"})
_SDK_NATIVE_BUILD_KEYS = frozenset({"package", "profile", "rustc-codegen-arguments"})
_CANDIDATE_CARGO_CONFIG = Path("dependencies/candidate-cargo-config.toml")
_CANDIDATE_INSTALL_STATE = Path("dependencies/install-state.json")
_CANDIDATE_PATCH_TARGETS = {
    "graphica": "dependencies/checkouts/symbolica/lib/graphica",
    "numerica": "dependencies/checkouts/symbolica/lib/numerica",
    "symbolica": "dependencies/checkouts/symbolica",
    "symjit": "dependencies/checkouts/symjit",
}
_NATIVE_CANDIDATE_SOURCE_NAMES = frozenset({"symbolica", "symjit"})
_NATIVE_EXTENSION_SUFFIXES = (".dylib", ".pyd", ".so")


def _canonical_json_bytes(payload: Any) -> bytes:
    try:
        return (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"candidate native build identity is not canonical JSON: {error}"
        ) from error


def _native_build_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    tool = payload.get("tool")
    pyamplicol = tool.get("pyamplicol") if isinstance(tool, Mapping) else None
    contract = (
        pyamplicol.get("native-build") if isinstance(pyamplicol, Mapping) else None
    )
    if not isinstance(contract, Mapping) or set(contract) != set(
        _NATIVE_BUILD_CONTRACT_KEYS
    ):
        raise RuntimeError(
            "native pyproject contract must contain exactly schema-version, "
            "rust-path-remapping, macos, and sdk"
        )
    schema_version = contract.get("schema-version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != 1
    ):
        raise RuntimeError("native pyproject contract must use schema-version 1")
    remapping = contract.get("rust-path-remapping")
    if not isinstance(remapping, Mapping) or set(remapping) != set(
        _RUST_PATH_REMAP_KEYS
    ):
        raise RuntimeError("native Rust path-remapping contract is incomplete")
    canonical_remapping: dict[str, str] = {}
    for key in sorted(_RUST_PATH_REMAP_KEYS):
        value = remapping[key]
        if not isinstance(value, str) or not value.startswith("/"):
            raise RuntimeError(
                f"native Rust path-remapping destination {key} must be absolute"
            )
        canonical_remapping[key] = value
    if len(set(canonical_remapping.values())) != len(canonical_remapping):
        raise RuntimeError("native Rust path-remapping destinations must be unique")
    macos = contract.get("macos")
    if not isinstance(macos, Mapping) or set(macos) != set(_MACOS_NATIVE_BUILD_KEYS):
        raise RuntimeError("native macOS build contract is incomplete")
    canonical_macos: dict[str, str] = {}
    for key in sorted(_MACOS_NATIVE_BUILD_KEYS):
        value = macos[key]
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"native macOS build setting {key} is invalid")
        canonical_macos[key] = value
    if not canonical_macos["cc"].startswith("/") or not canonical_macos[
        "cxx"
    ].startswith("/"):
        raise RuntimeError("native macOS compilers must be absolute paths")
    sdk = contract.get("sdk")
    if not isinstance(sdk, Mapping) or set(sdk) != set(_SDK_NATIVE_BUILD_KEYS):
        raise RuntimeError("native SDK build contract is incomplete")
    package = sdk.get("package")
    profile = sdk.get("profile")
    codegen = sdk.get("rustc-codegen-arguments")
    if not isinstance(package, str) or not package:
        raise RuntimeError("native SDK package must be a non-empty string")
    if not isinstance(profile, str) or not profile:
        raise RuntimeError("native SDK profile must be a non-empty string")
    if (
        not isinstance(codegen, list)
        or not codegen
        or any(not isinstance(value, str) or not value for value in codegen)
    ):
        raise RuntimeError("native SDK rustc codegen arguments must be strings")
    return {
        "macos": canonical_macos,
        "rust-path-remapping": canonical_remapping,
        "schema-version": 1,
        "sdk": {
            "package": package,
            "profile": profile,
            "rustc-codegen-arguments": list(codegen),
        },
    }


def _native_pyproject_bytes(data: bytes) -> bytes:
    try:
        payload = tomllib.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise RuntimeError(
            f"invalid native pyproject configuration: {error}"
        ) from error
    build_system = payload.get("build-system")
    requirements = (
        build_system.get("requires") if isinstance(build_system, Mapping) else None
    )
    if not isinstance(requirements, list) or any(
        not isinstance(requirement, str) for requirement in requirements
    ):
        raise RuntimeError("native pyproject build-system requires must be a list")
    maturin_pins: list[str] = []
    for requirement in requirements:
        match = re.fullmatch(
            r"\s*maturin\s*==\s*([0-9][A-Za-z0-9._+-]*)\s*",
            requirement,
            flags=re.IGNORECASE,
        )
        if match is not None:
            maturin_pins.append(f"maturin=={match.group(1)}")
        elif re.match(r"\s*maturin(?:\W|$)", requirement, flags=re.IGNORECASE):
            raise RuntimeError(
                "native Maturin build requirement must be exactly pinned"
            )
    if len(maturin_pins) != 1:
        raise RuntimeError(
            "native pyproject must contain exactly one pinned Maturin requirement"
        )
    tool = payload.get("tool")
    maturin = tool.get("maturin") if isinstance(tool, Mapping) else None
    if not isinstance(maturin, Mapping):
        raise RuntimeError("native pyproject Maturin configuration must be a table")
    classified = (
        _NATIVE_MATURIN_CONFIG_KEYS
        | _NON_OUTPUT_MATURIN_CONFIG_KEYS
        | _PACKAGING_MATURIN_CONFIG_KEYS
    )
    unknown = set(maturin) - classified
    if unknown:
        rendered = ", ".join(sorted(str(key) for key in unknown))
        raise RuntimeError(f"unclassified Maturin configuration: {rendered}")
    return _canonical_json_bytes(
        {
            "build-system": {"requires": maturin_pins},
            "tool": {
                "maturin": {
                    key: maturin[key]
                    for key in sorted(_NATIVE_MATURIN_CONFIG_KEYS & set(maturin))
                },
                "pyamplicol": {"native-build": _native_build_contract(payload)},
            },
        }
    )


def _canonical_checkout_path(raw: str) -> str:
    normalized = raw.replace("\\", "/").rstrip("/")
    if normalized in {"TMP_FIXED_SYMJIT", "./TMP_FIXED_SYMJIT"} or normalized.endswith(
        "/TMP_FIXED_SYMJIT"
    ):
        return "TMP_FIXED_SYMJIT"
    marker = "/dependencies/checkouts/"
    if marker in normalized:
        suffix = normalized.rsplit(marker, maxsplit=1)[1]
        if suffix:
            return f"dependencies/checkouts/{suffix}"
    prefix = "dependencies/checkouts/"
    if normalized.startswith(prefix) and len(normalized) > len(prefix):
        return normalized
    raise RuntimeError(
        "candidate Cargo patch paths must resolve below dependencies/checkouts"
    )


def _candidate_cargo_config_bytes(data: bytes) -> bytes:
    """Return the exact path-independent candidate Cargo patch contract."""

    try:
        payload = tomllib.loads(data.decode("utf-8"))
        patch = payload["patch"]
        crates_io = patch["crates-io"]
    except (
        KeyError,
        TypeError,
        UnicodeDecodeError,
        tomllib.TOMLDecodeError,
    ) as error:
        raise RuntimeError(f"invalid candidate Cargo config: {error}") from error
    if (
        not isinstance(payload, dict)
        or set(payload) != {"patch"}
        or not isinstance(patch, dict)
        or set(patch) != {"crates-io"}
        or not isinstance(crates_io, dict)
        or not set(_CANDIDATE_PATCH_TARGETS).issubset(crates_io)
    ):
        raise RuntimeError(
            "candidate Cargo config must contain the locked native crates.io "
            "patch table"
        )
    canonical: dict[str, Any] = {"patch": {"crates-io": {}}}
    canonical_patches = canonical["patch"]["crates-io"]
    for name, expected in _CANDIDATE_PATCH_TARGETS.items():
        entry = crates_io[name]
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"path"}
            or not isinstance(entry["path"], str)
        ):
            raise RuntimeError(
                f"candidate Cargo patch {name} must contain exactly one path"
            )
        observed = _canonical_checkout_path(entry["path"])
        if observed != expected and not (
            name == "symjit" and observed == "TMP_FIXED_SYMJIT"
        ):
            raise RuntimeError(
                f"candidate Cargo patch {name} must resolve to {expected}"
            )
        canonical_patches[name] = {"path": observed}
    return _canonical_json_bytes(canonical)


def _required_string(
    payload: Mapping[str, Any],
    key: str,
    *,
    description: str,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{description} {key} must be a non-empty string")
    return value


def _required_hex(
    payload: Mapping[str, Any],
    key: str,
    *,
    length: int,
    description: str,
) -> str:
    value = _required_string(payload, key, description=description)
    if len(value) != length or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise RuntimeError(
            f"{description} {key} must be a lowercase {length}-digit hex value"
        )
    return value


def _canonical_candidate_sources(raw_sources: Any) -> dict[str, dict[str, str]]:
    if not isinstance(raw_sources, Mapping):
        raise RuntimeError("candidate installer state sources must be an object")
    source_names = set(raw_sources)
    if any(not isinstance(name, str) or not name for name in source_names):
        raise RuntimeError("candidate installer source names must be non-empty strings")
    if not _NATIVE_CANDIDATE_SOURCE_NAMES.issubset(source_names):
        raise RuntimeError(
            "candidate installer state has an incomplete native source map"
        )
    sources: dict[str, dict[str, str]] = {}
    for name in sorted(_NATIVE_CANDIDATE_SOURCE_NAMES):
        raw_source = raw_sources[name]
        description = f"candidate installer source {name}"
        if not isinstance(raw_source, Mapping):
            raise RuntimeError(f"{description} must be an object")
        allowed_fields = {"url", "revision", "branch"}
        required_fields = {"url", "revision"}
        if not required_fields.issubset(raw_source) or not set(raw_source).issubset(
            allowed_fields
        ):
            raise RuntimeError(
                f"{description} must contain url/revision and optional branch only"
            )
        source = {
            "url": _required_string(raw_source, "url", description=description),
            "revision": _required_hex(
                raw_source,
                "revision",
                length=40,
                description=description,
            ),
        }
        if "branch" in raw_source:
            source["branch"] = _required_string(
                raw_source,
                "branch",
                description=description,
            )
        sources[name] = source
    return sources


def _candidate_install_state_bytes(
    root: Path,
    data: bytes,
    *,
    cargo_config_data: bytes,
) -> bytes:
    """Validate and canonicalize the minimal immutable-source install state."""

    del root, cargo_config_data

    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid candidate installer state: {error}") from error
    schema_version = (
        payload.get("schema_version") if isinstance(payload, dict) else None
    )
    if not isinstance(payload, dict) or set(payload) != {
        "publishable",
        "schema_version",
        "sources",
    }:
        raise RuntimeError(
            "candidate installer state must contain only schema_version, "
            "publishable, and sources"
        )
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != 1
        or payload["publishable"] is not False
    ):
        raise RuntimeError(
            "candidate installer state must be a non-publishable schema-1 object"
        )
    sources = _canonical_candidate_sources(payload["sources"])
    return _canonical_json_bytes(
        {
            "publishable": False,
            "schema_version": 1,
            "sources": sources,
        }
    )


def _release_cargo_lock_bytes(root: Path, data: bytes) -> bytes:
    """Return the ordinary immutable-Git Cargo lock without projection."""

    del root
    return data


def _is_native_build_tree_input(root: Path, tree: Path, path: Path) -> bool:
    if not path.is_file():
        return False
    tree_relative = path.relative_to(tree)
    root_relative = path.relative_to(root)
    return (
        not _NATIVE_BUILD_IGNORED_TREE_PARTS.intersection(tree_relative.parts)
        and root_relative not in _NON_NATIVE_RUST_PATHS
    )


def _native_build_inputs_digest(
    root: Path,
    *,
    normalize_release_cargo_lock: bool = False,
) -> str:
    """Hash the small set of checkout inputs that determines the native build."""

    candidate_config = root / _CANDIDATE_CARGO_CONFIG
    candidate_state = root / _CANDIDATE_INSTALL_STATE
    canonical_candidate_inputs: dict[Path, bytes] = {}
    if not normalize_release_cargo_lock:
        if candidate_config.is_file() != candidate_state.is_file():
            raise RuntimeError(
                "candidate native build identity requires both "
                "candidate-cargo-config.toml and install-state.json"
            )
        if candidate_config.is_file():
            cargo_config_data = candidate_config.read_bytes()
            canonical_candidate_inputs[candidate_config] = (
                _candidate_cargo_config_bytes(cargo_config_data)
            )
            canonical_candidate_inputs[candidate_state] = (
                _candidate_install_state_bytes(
                    root,
                    candidate_state.read_bytes(),
                    cargo_config_data=cargo_config_data,
                )
            )

    paths = [
        root / relative
        for relative in _NATIVE_BUILD_INPUT_FILES
        if not (
            normalize_release_cargo_lock
            and relative in _RELEASE_OMITTED_NATIVE_BUILD_INPUTS
        )
        and not (canonical_candidate_inputs and relative == Path("Cargo.lock"))
    ]
    # Local upstream fixes have no new Git revision. Include their compiler
    # inputs in the existing cache key, but not their tests or documentation.
    manifest = root / "Cargo.toml"
    if manifest.is_file():
        cargo = tomllib.loads(manifest.read_text(encoding="utf-8"))
        local_symjit = cargo.get("patch", {}).get("crates-io", {}).get("symjit", {})
        local_path = local_symjit.get("path")
        if isinstance(local_path, str):
            local_root = (root / local_path).resolve()
            if local_root == (root / "TMP_FIXED_SYMJIT").resolve():
                paths.extend(local_root / name for name in ("Cargo.toml", "build.rs"))
                paths.extend(
                    path for path in (local_root / "src").rglob("*") if path.is_file()
                )
    for relative in _NATIVE_BUILD_INPUT_TREES:
        tree = root / relative
        if not tree.is_dir():
            continue
        paths.extend(
            path
            for path in tree.rglob("*")
            if _is_native_build_tree_input(root, tree, path)
            and not (
                normalize_release_cargo_lock
                and path.relative_to(root) in _RELEASE_OMITTED_NATIVE_BUILD_INPUTS
            )
        )
    digest = hashlib.sha256(_NATIVE_BUILD_IDENTITY_DOMAIN)
    for path in sorted(set(paths)):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        data = canonical_candidate_inputs.get(path, path.read_bytes())
        if path == root / "pyproject.toml":
            data = _native_pyproject_bytes(data)
        if normalize_release_cargo_lock and path == root / "Cargo.lock":
            data = _release_cargo_lock_bytes(root, data)
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest()


def _native_extensions(package_root: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            (
                path
                for path in package_root.glob("_rusticol.*")
                if path.is_file() and path.name.endswith(_NATIVE_EXTENSION_SUFFIXES)
            ),
            key=lambda path: path.name,
        )
    )


def _read_build_info(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"{description} is unreadable; rerun `just dev-install`"
        ) from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{description} is invalid; rerun `just dev-install`")
    return payload


def _is_source_checkout(package_root: Path, source_root: Path) -> bool:
    return (
        package_root == source_root / "src" / "pyamplicol"
        and (source_root / "pyproject.toml").is_file()
    )


def _verify_source_runtime(
    payload: dict[str, Any],
    *,
    package_root: Path | None = None,
    source_root: Path | None = None,
) -> None:
    contract = payload.get("source_runtime")
    if not isinstance(contract, dict):
        raise RuntimeError(
            "source runtime provenance is missing; rerun `just dev-install`"
        )
    extension_name = contract.get("extension_name")
    extension_sha256 = contract.get("extension_sha256")
    mode = contract.get("mode")
    native_digest = contract.get("native_build_inputs_sha256")
    if not all(
        isinstance(value, str) and value
        for value in (extension_name, extension_sha256, native_digest)
    ) or mode not in {"candidate", "release"}:
        raise RuntimeError(
            "source runtime provenance is incomplete; rerun `just dev-install`"
        )
    valid_build_mode = (
        mode == "candidate" and payload.get("publishable") is False
    ) or (
        mode == "release"
        and (
            payload.get("publishable") is True
            or (
                payload.get("publishable") is False
                and payload.get("release_prepared_model_bootstrap") is True
            )
        )
    )
    if not valid_build_mode:
        raise RuntimeError(
            "source runtime provenance has an inconsistent build mode; "
            "rerun `just dev-install`"
        )
    if (
        len(extension_sha256) != 64
        or any(character not in "0123456789abcdef" for character in extension_sha256)
        or len(native_digest) != 64
        or any(character not in "0123456789abcdef" for character in native_digest)
        or payload.get("native_build_inputs_sha256") != native_digest
    ):
        raise RuntimeError(
            "source runtime provenance is incomplete; rerun `just dev-install`"
        )
    package_root = package_root or _SOURCE_PACKAGE_ROOT
    extensions = _native_extensions(package_root)
    if len(extensions) != 1 or extensions[0].name != extension_name:
        raise RuntimeError(
            "source runtime extension inventory is ambiguous or stale; "
            "rerun `just dev-install`"
        )
    if hashlib.sha256(extensions[0].read_bytes()).hexdigest() != extension_sha256:
        raise RuntimeError(
            "source runtime extension is stale or was replaced; "
            "rerun `just dev-install`"
        )
    source_root = source_root or _SOURCE_ROOT
    if (
        _native_build_inputs_digest(
            source_root,
            normalize_release_cargo_lock=mode == "release",
        )
        != native_digest
    ):
        raise RuntimeError(
            "native build inputs changed after the source runtime was staged; "
            "rerun `just dev-install`"
        )


def _verify_candidate_install(payload: dict[str, Any]) -> None:
    raw_root = payload.get("source_checkout")
    native_digest = payload.get("native_build_inputs_sha256")
    if (
        not isinstance(raw_root, str)
        or not raw_root
        or not isinstance(native_digest, str)
        or len(native_digest) != 64
    ):
        raise RuntimeError(
            "candidate wheel provenance is incomplete; rerun `just dev-install`"
        )
    source_root = Path(raw_root)
    if not source_root.is_absolute() or not (source_root / "pyproject.toml").is_file():
        raise RuntimeError(
            "candidate wheel source checkout is unavailable; rerun `just dev-install`"
        )
    release_prepared_model_bootstrap = (
        payload.get("release_prepared_model_bootstrap") is True
    )
    if (
        _native_build_inputs_digest(
            source_root,
            normalize_release_cargo_lock=release_prepared_model_bootstrap,
        )
        != native_digest
    ):
        install_kind = (
            "release prepared-model bootstrap"
            if release_prepared_model_bootstrap
            else "candidate wheel"
        )
        raise RuntimeError(
            f"installed {install_kind} is stale for this checkout; "
            "rerun `just dev-install`"
        )


def _active_build_info() -> dict[str, Any] | None:
    if (
        _is_source_checkout(_SOURCE_PACKAGE_ROOT, _SOURCE_ROOT)
        and _SOURCE_BUILD_INFO_PATH.exists()
    ):
        return _read_build_info(
            _SOURCE_BUILD_INFO_PATH,
            "source runtime provenance",
        )
    if _PACKAGE_BUILD_INFO_PATH.exists():
        return _read_build_info(_PACKAGE_BUILD_INFO_PATH, "wheel build provenance")
    return None


def _native_build_inputs_from_build_info(build_info: Mapping[str, Any]) -> str:
    native_digest = build_info.get("native_build_inputs_sha256")
    if native_digest is None and build_info.get("publishable") is True:
        try:
            native = importlib.import_module("pyamplicol._rusticol")
        except ImportError as error:
            raise RuntimeError(
                "active native build-input identity is unavailable; reinstall "
                "pyAmpliCol"
            ) from error
        operation = getattr(native, "native_build_inputs_sha256", None)
        native_digest = operation() if callable(operation) else None
    if (
        not isinstance(native_digest, str)
        or len(native_digest) != 64
        or any(character not in "0123456789abcdef" for character in native_digest)
    ):
        raise RuntimeError(
            "active native build-input identity is incomplete; reinstall pyAmpliCol"
        )
    return native_digest


def active_native_build_inputs_sha256() -> str:
    """Return the authenticated native build-input digest, if installed."""

    build_info = _active_build_info()
    if build_info is None:
        raise RuntimeError(
            "active native build provenance is unavailable; reinstall pyAmpliCol"
        )
    return _native_build_inputs_from_build_info(build_info)


def active_native_source_identity() -> tuple[str, str]:
    """Return the authenticated source revision and native-input digest."""

    build_info = _active_build_info()
    if build_info is None:
        raise RuntimeError(
            "active native build provenance is unavailable; reinstall pyAmpliCol"
        )
    source_revision = build_info.get("source_revision")
    if (
        not isinstance(source_revision, str)
        or len(source_revision) != 40
        or any(character not in "0123456789abcdef" for character in source_revision)
    ):
        raise RuntimeError(
            "active native source identity is incomplete; reinstall pyAmpliCol"
        )
    return source_revision, _native_build_inputs_from_build_info(build_info)


def active_source_revision() -> str | None:
    """Return the revision bound to the active wheel/source runtime, if any."""

    build_info = _active_build_info()
    if build_info is None:
        return None
    revision = build_info.get("source_revision")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        return None
    return revision


def active_source_checkout() -> Path | None:
    """Return the checkout backing a non-publishable local build, if available."""

    build_info = _active_build_info()
    if build_info is None or build_info.get("publishable") is not False:
        return None
    raw_root = build_info.get("source_checkout")
    if not isinstance(raw_root, str) or not raw_root:
        return None
    root = Path(raw_root)
    return root if root.is_absolute() else None


def verify_native_module(module: Any, *, expected_version: str | None = None) -> None:
    """Reject a stale native extension in contributor builds.

    Published wheels retain the normal package-manager import path; the extra
    build-ID check is limited to non-publishable candidate wheels and staged
    source runtimes.
    """

    # Unit tests and external adapters may provide a duck-typed stand-in. Real
    # extension modules always expose an on-disk module path.
    if not getattr(module, "__file__", None):
        return
    operation = getattr(module, "package_version", None)
    if not callable(operation):
        raise RuntimeError(
            "native runtime has no package-version contract; reinstall pyAmpliCol"
        )
    observed = operation()
    expected = expected_version or package_version()
    if not isinstance(observed, str) or observed.replace("-dev.", ".dev") != expected:
        raise RuntimeError(
            "native runtime version does not match the Python package "
            f"({observed!r} != {expected!r}); rerun `just dev-install`"
        )
    build_info = _active_build_info()
    if build_info is None:
        return
    source_runtime = build_info.get("source_runtime")
    if build_info.get("publishable") is not False and not isinstance(
        source_runtime, dict
    ):
        return
    native_digest = build_info.get("native_build_inputs_sha256")
    native_operation = getattr(module, "native_build_inputs_sha256", None)
    if not isinstance(native_digest, str) or not callable(native_operation):
        raise RuntimeError(
            "native runtime build-ID contract is incomplete; rerun `just dev-install`"
        )
    if native_operation() != native_digest:
        raise RuntimeError(
            "native runtime was built from different source inputs; "
            "rerun `just dev-install`"
        )


def package_version(default: str = "0.2.1") -> str:
    """Return the wheel/source-runtime version without importing heavy modules."""

    if _is_source_checkout(_SOURCE_PACKAGE_ROOT, _SOURCE_ROOT):
        source_runtime_present = (
            _SOURCE_BUILD_INFO_PATH.exists()
            or _SOURCE_RUNTIME_STAGING_PATH.exists()
            or bool(_native_extensions(_SOURCE_PACKAGE_ROOT))
        )
        if source_runtime_present:
            if _SOURCE_RUNTIME_STAGING_PATH.exists():
                raise RuntimeError(
                    "source runtime staging is incomplete; rerun `just dev-install`"
                )
            payload = _read_build_info(
                _SOURCE_BUILD_INFO_PATH,
                "source runtime provenance",
            )
            _verify_source_runtime(payload)
            value = payload.get("version")
            if not isinstance(value, str) or not value:
                raise RuntimeError(
                    "source runtime version provenance is invalid; "
                    "rerun `just dev-install`"
                )
            return value

    if _PACKAGE_BUILD_INFO_PATH.exists():
        payload = _read_build_info(
            _PACKAGE_BUILD_INFO_PATH,
            "wheel build provenance",
        )
        value = payload.get("version")
        if not isinstance(value, str) or not value:
            raise RuntimeError(
                "wheel build version provenance is invalid; reinstall pyAmpliCol"
            )
        if payload.get("publishable") is False:
            _verify_candidate_install(payload)
        return value
    try:
        return metadata.version("pyamplicol")
    except metadata.PackageNotFoundError:
        return default


__all__ = [
    "COMPILED_COLOR_CONTRACTION_WALSH_C2K_CAPABILITY",
    "COMPILED_COLOR_CONTRACTION_WALSH_CAPABILITY",
    "COMPILED_COLOR_TOPOLOGY_LANES_CAPABILITY",
    "COMPILED_HELICITY_DUAL_LANE_CAPABILITY",
    "COMPILED_HELICITY_PRIMARY_RECURRENCE_CAPABILITY",
    "COMPILED_HELICITY_SELECTOR_UNION_CAPABILITY",
    "COMPILED_MODEL_SCHEMA_VERSION",
    "COMPILED_PLANE_ARENA_RUNTIME_CAPABILITY",
    "COMPILED_PLANE_DIRECT_APPLICATION_ABI",
    "COMPILED_RUNTIME_SELECTORS_CAPABILITY",
    "C_ABI_VERSION",
    "EAGER_DAG_F64_RUNTIME_CAPABILITY",
    "EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY",
    "EAGER_DIRECT_TABLE_BINDING_ABI",
    "EAGER_DIRECT_TABLE_DESCRIPTOR_ABI",
    "EAGER_LC_TOPOLOGY_REPLAY_RUNTIME_CAPABILITY",
    "EAGER_RUNTIME_LAYOUT_F64_CAPABILITY",
    "EVALUATOR_RUNTIME_CAPABILITIES",
    "KNOWN_EVALUATOR_RUNTIME_CAPABILITIES",
    "ON_THE_FLY_CONTRACTED_COLOR_RUNTIME_CAPABILITY",
    "ON_THE_FLY_LC_COLOR_RUNTIME_CAPABILITY",
    "ON_THE_FLY_RUNTIME_CAPABILITY",
    "PROCESS_ARTIFACT_SCHEMA_VERSION",
    "PYTHON_API_VERSION",
    "RECURRENCE_BUILDER_INPUT_ABI",
    "RECURRENCE_COLOR_RUNTIME_CAPABILITY",
    "RECURRENCE_DIRECT_ARENA_RUNTIME_CAPABILITY",
    "RECURRENCE_DIRECT_BACKEND_ABI",
    "RECURRENCE_DIRECT_BINDING_ABI",
    "RECURRENCE_DIRECT_TEMPLATE_ABI",
    "RECURRENCE_HELICITY_SELECTOR_COMPANION_RUNTIME_CAPABILITY",
    "RECURRENCE_PLAN_ABI",
    "RECURRENCE_RUNTIME_LAYOUT_ABI",
    "RUNTIME_PHYSICS_SCHEMA_VERSION",
    "SYMBOLICA_ASM_RUNTIME_CAPABILITY",
    "SYMBOLICA_CPP_RUNTIME_CAPABILITY",
    "SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY",
    "SYMBOLICA_SERIALIZATION_ABI",
    "SYMJIT_APPLICATION_ABI",
    "SYMJIT_F64_RUNTIME_CAPABILITY",
    "SYMJIT_PLANE_APPLICATION_ABI",
    "SYMMETRIC_GROUP_FFT_COLOR_RUNTIME_CAPABILITY",
    "TOML_SCHEMA_VERSION",
    "active_native_build_inputs_sha256",
    "active_native_source_identity",
    "active_source_checkout",
    "active_source_revision",
    "package_version",
    "verify_native_module",
]

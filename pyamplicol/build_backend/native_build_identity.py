# SPDX-License-Identifier: 0BSD
"""Canonical source identity for the native pyAmpliCol build."""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

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
# Release sdists omit candidate-only resolution state.
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
# This explicit inventory contains only separately compiled test modules,
# integration-test inputs, and Python/stub packaging files. Additions default
# to native inputs. Mixed production files with inline #[cfg(test)] modules are
# deliberately hashed at file granularity; stripping Rust syntax here would be
# a second, incomplete compiler frontend.
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
    """Validate the one declarative contract consumed by native build code."""

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


def canonical_native_pyproject_bytes(data: bytes) -> bytes:
    """Project only semantic inputs to the authenticated native build."""

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


def load_native_build_contract(root: Path) -> dict[str, Any]:
    """Load the canonical backend knobs from the hashed pyproject input."""

    path = root / "pyproject.toml"
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise RuntimeError(f"invalid native build contract: {path}: {error}") from error
    return _native_build_contract(payload)


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


def canonical_candidate_cargo_config_bytes(data: bytes) -> bytes:
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


def canonical_candidate_install_state_bytes(
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


def canonical_release_cargo_lock_bytes(root: Path, data: bytes) -> bytes:
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


def native_build_inputs_digest(
    root: Path,
    *,
    normalize_release_cargo_lock: bool = False,
) -> str:
    """Hash every checked-out input that can change the native runtime build."""

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
                canonical_candidate_cargo_config_bytes(cargo_config_data)
            )
            canonical_candidate_inputs[candidate_state] = (
                canonical_candidate_install_state_bytes(
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
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        data = canonical_candidate_inputs.get(path, path.read_bytes())
        if path == root / "pyproject.toml":
            data = canonical_native_pyproject_bytes(data)
        if normalize_release_cargo_lock and path == root / "Cargo.lock":
            data = canonical_release_cargo_lock_bytes(root, data)
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest()


__all__ = [
    "canonical_candidate_cargo_config_bytes",
    "canonical_candidate_install_state_bytes",
    "canonical_native_pyproject_bytes",
    "canonical_release_cargo_lock_bytes",
    "load_native_build_contract",
    "native_build_inputs_digest",
]

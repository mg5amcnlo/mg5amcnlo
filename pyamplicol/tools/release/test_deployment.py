#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Test a wheel in an isolated PyPI-style deployment environment."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import sys
import tempfile
import tomllib
import zipfile
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Any

try:
    from packaging.utils import InvalidWheelFilename, parse_wheel_filename
except ModuleNotFoundError:  # pragma: no cover - pip vendors the build fallback
    from pip._vendor.packaging.utils import (  # type: ignore[no-redef]
        InvalidWheelFilename,
        parse_wheel_filename,
    )

from _artifacts import (
    _candidate_dependency_overrides,
    audit_wheel,
    canonicalize_name,
)
from _common import (
    CANDIDATE_ARTIFACTS,
    DEPENDENCY_WHEELHOUSE,
    DEPLOYMENT_ROOT,
    DIST,
    ROOT,
    ReleaseError,
    build_mode,
    check_dependency_gate,
    clean_environment,
    exactly_one,
    external_temporary_directory,
    is_relative_to,
    run,
    runtime_environment,
    sha256,
)
from install_wheel import (
    interpreter_tags,
    select_compatible_wheel,
    wheelhouse_directories,
)

_LOCK = ROOT / "dependencies" / "release-lock.toml"

_SELFTEST_RESOURCE = r"""
import importlib.resources
import json

import pyamplicol._rusticol as native

target = str(native.target_info().triple)
fixture = importlib.resources.files("pyamplicol").joinpath(
    "assets", "selftest", target
)
expected = json.loads(fixture.joinpath("expected.json").read_text(encoding="utf-8"))
artifact = fixture.joinpath(str(expected["artifact_path"]))
print(json.dumps({"artifact": str(artifact), "expected": expected}, sort_keys=True))
"""

_PATH_ISOLATION_SMOKE = r"""
import os
from pathlib import Path
import sys

forbidden_root = Path(os.environ["PYAMPLICOL_FORBIDDEN_ROOT"]).resolve()
deployment_sandbox = Path(
    os.environ["PYAMPLICOL_DEPLOYMENT_SANDBOX"]
).resolve()

for entry in sys.path:
    if not entry:
        continue
    resolved = Path(entry).resolve()
    inside_checkout = (
        resolved == forbidden_root or forbidden_root in resolved.parents
    )
    inside_sandbox = (
        resolved == deployment_sandbox or deployment_sandbox in resolved.parents
    )
    assert not inside_checkout or inside_sandbox, entry
"""

_INSTALLED_SMOKE = (
    _PATH_ISOLATION_SMOKE
    + r"""
import importlib.metadata
import importlib.resources
import json
import os
from pathlib import Path
import re
import shutil
import sys
import urllib.parse

expected = json.loads(os.environ["PYAMPLICOL_EXPECTED_DEPENDENCIES"])
local_sources = json.loads(os.environ["PYAMPLICOL_EXPECTED_LOCAL_SOURCES"])
mode = os.environ["PYAMPLICOL_DEPLOYMENT_MODE"]
version = os.environ["PYAMPLICOL_EXPECTED_VERSION"]

assert "PYTHONPATH" not in os.environ
assert shutil.which("cargo") is None
assert shutil.which("rustc") is None

distribution = importlib.metadata.distribution("pyamplicol")
assert distribution.metadata["Name"] == "pyamplicol"
assert distribution.version == version
assert distribution.metadata["Requires-Python"] == ">=3.11"
direct = re.compile(r"(?i)(?:\s@\s|(?:file|git\+[^:]+|https?)://)")
assert not [item for item in (distribution.requires or ()) if direct.search(item)]

for name, required_version in expected.items():
    assert importlib.metadata.version(name) == required_version, name
    dependency = importlib.metadata.distribution(name)
    direct_text = dependency.read_text("direct_url.json")
    if name not in local_sources:
        assert direct_text is None, name
        continue
    assert mode == "candidate", name
    assert direct_text is not None, name
    direct_url = json.loads(direct_text)
    parsed = urllib.parse.urlsplit(direct_url["url"])
    assert parsed.scheme == "file" and parsed.netloc in {"", "localhost"}, name
    installed_from = Path(urllib.parse.unquote(parsed.path)).resolve()
    source = local_sources[name]
    assert installed_from == Path(source["path"]).resolve(), name
    archive_info = direct_url.get("archive_info", {})
    expected_hash = source["sha256"]
    assert (
        archive_info.get("hash") == f"sha256={expected_hash}"
        or archive_info.get("hashes", {}).get("sha256") == expected_hash
    ), name

import pyamplicol
import pyamplicol._rusticol
from pyamplicol._sdk.config import load_sdk_info
from pyamplicol.models import BuiltinSMModel

assert pyamplicol.__version__ == version
package = importlib.resources.files("pyamplicol")
build_info_resource = package.joinpath("_build_info.json")
if mode == "candidate":
    build_info = json.loads(build_info_resource.read_text(encoding="utf-8"))
    assert build_info["publishable"] is False
    assert build_info["selftest_fixture_bootstrap"] is False
    assert build_info["version"] == version
else:
    build_info = json.loads(build_info_resource.read_text(encoding="utf-8"))
    assert set(build_info) == {
        "publishable",
        "schema_version",
        "selftest_fixture_bootstrap",
        "source_revision",
        "version",
    }
    assert build_info["schema_version"] == 1
    assert build_info["publishable"] is True
    assert build_info["selftest_fixture_bootstrap"] is False
    assert build_info["version"] == version
    assert re.fullmatch(r"[0-9a-f]{40}", build_info["source_revision"])

sdk = load_sdk_info()
metadata = json.loads(
    package.joinpath("_sdk", "metadata.json").read_text(encoding="utf-8")
)

assert sdk.abi_version == 1
assert sdk.package_version == version

model = BuiltinSMModel()
assert model.name == "built-in-sm-leading-color"
assert model.particles and model.vertices

files = [str(item).replace("\\", "/") for item in (distribution.files or ())]
assert any("example" in item.lower() for item in files)
installed_scripts = {item for item in files if item.startswith("../../../bin/")}
assert installed_scripts == {
    "../../../bin/pyamplicol",
    "../../../bin/rusticol-config",
}, sorted(installed_scripts)
package_files = [item for item in files if not item.startswith("../../../bin/")]
roots = {item.split("/", 1)[0] for item in package_files}
assert all(
    root == "pyamplicol"
    or root == "rusticol_config"
    or root == "pyamplicol.libs"
    or root.endswith(".dist-info")
    for root in roots
), sorted(roots)
for relative in (
    ("assets", "schemas", "README.md"),
    ("assets", "schemas", "artifact-manifest-v3.schema.json"),
    ("assets", "schemas", "runtime-physics-v1.schema.json"),
):
    assert package.joinpath(*relative).is_file(), relative

print(json.dumps({"mode": mode, "version": version, "sdk_target": sdk.target}))
"""
)

_SYMBOLICA_FREE_F64_SMOKE = (
    _PATH_ISOLATION_SMOKE
    + r"""
import builtins
import json
import os
import sys

os.environ.pop("SYMBOLICA_LICENSE", None)
os.environ["SYMBOLICA_HIDE_BANNER"] = "1"

original_import = builtins.__import__

def reject_symbolica(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "symbolica" or name.startswith("symbolica."):
        raise ModuleNotFoundError(
            "Symbolica import blocked by the f64 deployment gate", name="symbolica"
        )
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = reject_symbolica
from pyamplicol.diagnostics import run_self_test

report = run_self_test()
physics = [check for check in report.checks if check.name == "physics-f64"]
assert report.ok and len(physics) == 1 and physics[0].status == "pass"
assert not any(
    name == "symbolica" or name.startswith("symbolica.")
    for name in sys.modules
)
print(json.dumps({"ok": True, "physics": physics[0].detail}, sort_keys=True))
"""
)

_SYMBOLICA_ABSENT_F64_SMOKE = (
    _PATH_ISOLATION_SMOKE
    + r"""
import importlib.resources
import importlib.util
import json
import sys

for unavailable in ("symbolica", "ufo_model_loader"):
    assert importlib.util.find_spec(unavailable) is None, unavailable

import pyamplicol._rusticol as native
from pyamplicol import Runtime

target = str(native.target_info().triple)
fixture = importlib.resources.files("pyamplicol").joinpath(
    "assets", "selftest", target
)
expected = json.loads(fixture.joinpath("expected.json").read_text(encoding="utf-8"))
artifact = fixture.joinpath(str(expected["artifact_path"]))
manifest = json.loads(artifact.joinpath("artifact.json").read_text(encoding="utf-8"))
runtime = Runtime.load(
    str(artifact),
    process=str(expected["process_id"]),
    mute_warnings=True,
)
assert runtime.artifact_id == manifest["artifact_id"]
assert runtime.execution_mode == "compiled"
total = tuple(complex(value) for value in runtime.evaluate(expected["momenta"]))
resolved = runtime.evaluate_resolved(expected["momenta"])
reduced = tuple(complex(value) for value in resolved.total())
expected_total = tuple(complex(*value) for value in expected["total"])
assert resolved.shape == tuple(expected["resolved_shape"])
assert len(total) == len(reduced) == len(expected_total)
for actual, explicit, reference in zip(total, reduced, expected_total, strict=True):
    scale = max(abs(reference), 1.0)
    assert abs(actual - reference) <= 1.0e-12 * scale
    assert abs(explicit - reference) <= 1.0e-12 * scale
assert not any(
    name == "symbolica"
    or name.startswith("symbolica.")
    or name == "ufo_model_loader"
    or name.startswith("ufo_model_loader.")
    for name in sys.modules
)
print(json.dumps({"ok": True, "process": runtime.physics.process_id, "target": target}))
"""
)

_INSTALLED_BACKEND_AND_PRECISION_SMOKE = (
    _PATH_ISOLATION_SMOKE
    + r"""
import faulthandler
import json
import math
import os
from pathlib import Path

faulthandler.enable()

from pyamplicol import Generator, Runtime
from pyamplicol.config import (
    ColorConfig,
    EvaluatorBackend,
    EvaluatorConfig,
    EvaluatorExecutionMode,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationValidationConfig,
    JITConfig,
    RunConfig,
    SymbolicaConfig,
)

root = Path(os.environ["PYAMPLICOL_DEPLOYMENT_SANDBOX"]) / "backend-smoke"
totals = {}
for backend in (
    EvaluatorBackend.JIT,
    EvaluatorBackend.ASM,
    EvaluatorBackend.CPP,
):
    artifact = root / backend.value
    config = RunConfig(
        action="generate",
        color=ColorConfig(accuracy="lc"),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
            validation=GenerationValidationConfig(samples=10),
        ),
        evaluator=EvaluatorConfig(
            backend=backend,
            execution_mode=EvaluatorExecutionMode.COMPILED,
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(optimization_level=3),
        ),
        symbolica=SymbolicaConfig(suggest_license=False),
    )
    print(f"deployment: {backend.value} generation", flush=True)
    Generator(config).generate("d d~ > z", artifact)
    manifest = json.loads((artifact / "artifact.json").read_text(encoding="utf-8"))
    process_id = manifest["processes"][0]["id"]
    validation = json.loads(
        (
            artifact
            / "processes"
            / process_id
            / "validation-momenta.json"
        ).read_text(encoding="utf-8")
    )
    momenta = [
        [
            [float(component) for component in particle["momentum"]]
            for particle in validation["points"][0]
        ]
    ]
    print(f"deployment: {backend.value} load and f64 evaluation", flush=True)
    runtime = Runtime.load(artifact)
    assert runtime.artifact_id == manifest["artifact_id"]
    assert runtime.execution_mode == "compiled"
    total = runtime.evaluate(momenta)[0]
    resolved = runtime.evaluate_resolved(momenta)
    assert resolved.total()[0] == total
    assert math.isclose(total.imag, 0.0, abs_tol=1.0e-15)
    totals[backend.value] = total.real
    if backend is EvaluatorBackend.JIT:
        print("deployment: jit arbitrary-precision evaluation", flush=True)
        precise = runtime.evaluate(momenta, precision=80)[0]
        assert math.isclose(
            float(precise),
            total.real,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        )

reference = totals[EvaluatorBackend.JIT.value]
assert all(
    math.isclose(value, reference, rel_tol=1.0e-12, abs_tol=1.0e-15)
    for value in totals.values()
)

eager_artifact = root / "eager-jit"
eager_config = RunConfig(
    action="generate",
    color=ColorConfig(accuracy="lc"),
    generation=GenerationConfig(
        workers=1,
        emit_api_bundle=False,
        validation=GenerationValidationConfig(samples=10),
    ),
    evaluator=EvaluatorConfig(
        backend=EvaluatorBackend.JIT,
        execution_mode=EvaluatorExecutionMode.EAGER,
        optimization=EvaluatorOptimizationConfig(cores=1),
        jit=JITConfig(optimization_level=3),
    ),
    symbolica=SymbolicaConfig(suggest_license=False),
)
print("deployment: eager-jit generation", flush=True)
Generator(eager_config).generate("d d~ > z", eager_artifact)
eager_manifest = json.loads(
    (eager_artifact / "artifact.json").read_text(encoding="utf-8")
)
eager_process_id = eager_manifest["processes"][0]["id"]
eager_execution = json.loads(
    (
        eager_artifact
        / "processes"
        / eager_process_id
        / "execution.json"
    ).read_text(encoding="utf-8")
)
assert eager_execution["kind"] == "pyamplicol-runtime-eager-execution"
assert (eager_artifact / "model/eager-kernel-pack.json").is_file()
print("deployment: eager-jit load and f64 evaluation", flush=True)
eager_runtime = Runtime.load(eager_artifact)
assert eager_runtime.artifact_id == eager_manifest["artifact_id"]
assert eager_runtime.execution_mode == "eager"
eager_total = eager_runtime.evaluate(momenta)[0]
eager_resolved = eager_runtime.evaluate_resolved(momenta)
eager_reduced = eager_resolved.total()[0]
assert abs(eager_reduced - eager_total) <= 1.0e-12 * max(abs(eager_total), 1.0)
assert math.isclose(
    eager_total.real,
    reference,
    rel_tol=1.0e-12,
    abs_tol=1.0e-15,
)
assert math.isclose(eager_total.imag, 0.0, abs_tol=1.0e-15)
print("deployment: eager-jit arbitrary-precision evaluation", flush=True)
eager_precise = eager_runtime.evaluate(momenta, precision=80)[0]
assert math.isclose(
    float(eager_precise),
    eager_total.real,
    rel_tol=1.0e-12,
    abs_tol=1.0e-15,
)

print(
    json.dumps(
        {
            "backends": sorted(totals),
            "eager_execution": True,
            "total": reference,
        },
        sort_keys=True,
    )
)
"""
)


@dataclass(frozen=True)
class DependencyInstallation:
    versions: dict[str, str]
    local_wheels: dict[str, Path]


@dataclass(frozen=True)
class NativeToolchain:
    cc: tuple[str, ...]
    cxx: tuple[str, ...]
    fortran: tuple[str, ...]
    rustc: tuple[str, ...]


@dataclass(frozen=True)
class NativePhysicsFixture:
    artifact: Path
    process_id: str
    shape: tuple[int, int, int]
    totals: tuple[float, ...]


def _release_lock() -> dict[str, Any]:
    with _LOCK.open("rb") as stream:
        payload = tomllib.load(stream)
    if payload.get("schema_version") != 1:
        raise ReleaseError("dependencies/release-lock.toml must use schema_version = 1")
    return payload


def exact_dependencies() -> dict[str, str]:
    raw = _release_lock().get("python_dependencies")
    if not isinstance(raw, list):
        raise ReleaseError("release lock has no exact Python dependency list")
    dependencies: dict[str, str] = {}
    for entry in raw:
        if not isinstance(entry, dict):
            raise ReleaseError("release lock Python dependencies must be tables")
        distribution = entry.get("distribution")
        version = entry.get("version")
        if not isinstance(distribution, str) or not isinstance(version, str):
            raise ReleaseError("release dependency needs distribution/version")
        name = canonicalize_name(distribution)
        if name in dependencies:
            raise ReleaseError(f"release lock repeats Python dependency {name}")
        dependencies[name] = version
    return dependencies


def _wheel_identity(path: Path) -> tuple[str, str, set[str]]:
    try:
        filename_name, filename_version, _build, filename_tags = parse_wheel_filename(
            path.name
        )
    except InvalidWheelFilename as error:
        raise ReleaseError(
            f"invalid candidate dependency wheel filename {path.name}: {error}"
        ) from error
    try:
        with zipfile.ZipFile(path) as archive:
            metadata_names = [
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_names) != 1:
                raise ReleaseError(
                    f"candidate dependency wheel must contain one METADATA: {path}"
                )
            metadata = BytesParser(policy=policy.default).parsebytes(
                archive.read(metadata_names[0])
            )
    except (OSError, KeyError, zipfile.BadZipFile) as error:
        raise ReleaseError(
            f"invalid candidate dependency wheel {path}: {error}"
        ) from error
    name = canonicalize_name(str(metadata.get("Name", "")))
    version = str(metadata.get("Version", ""))
    if not name or not version:
        raise ReleaseError(f"candidate dependency wheel has no identity: {path}")
    if canonicalize_name(filename_name) != name or str(filename_version) != version:
        raise ReleaseError(
            f"candidate dependency wheel filename disagrees with metadata: {path.name}"
        )
    return name, version, {str(tag) for tag in filename_tags}


def _candidate_dependency_wheels(
    requirements: dict[str, str],
    wheelhouses: Sequence[Path],
    supported_tags: Sequence[str],
) -> dict[str, Path]:
    if not wheelhouses:
        raise ReleaseError(
            "candidate deployment requires a local dependency wheelhouse"
        )
    wheels: set[Path] = set()
    for raw_root in wheelhouses:
        if raw_root.is_symlink():
            raise ReleaseError(
                f"candidate dependency wheelhouse may not be a symlink: {raw_root}"
            )
        root = raw_root.resolve()
        if not root.is_dir():
            raise ReleaseError(f"candidate dependency wheelhouse is missing: {root}")
        for raw_wheel in root.rglob("*.whl"):
            if raw_wheel.is_symlink():
                raise ReleaseError(
                    f"candidate dependency wheel may not be a symlink: {raw_wheel}"
                )
            wheel = raw_wheel.resolve()
            if not is_relative_to(wheel, root):
                raise ReleaseError(
                    f"candidate dependency wheel escapes its wheelhouse: {raw_wheel}"
                )
            wheels.add(wheel)

    supported = set(supported_tags)
    matches: dict[str, list[Path]] = {name: [] for name in requirements}
    seen_versions: dict[str, set[str]] = {name: set() for name in requirements}
    for wheel in sorted(wheels):
        name, version, tags = _wheel_identity(wheel)
        if name not in requirements:
            continue
        seen_versions[name].add(version)
        if version == requirements[name] and tags & supported:
            matches[name].append(wheel)

    selected: dict[str, Path] = {}
    for name, expected_version in sorted(requirements.items()):
        candidates = matches[name]
        if len(candidates) != 1:
            raise ReleaseError(
                f"expected one compatible local {name}=={expected_version} wheel, "
                f"found {len(candidates)}; available versions="
                f"{sorted(seen_versions[name])}"
            )
        selected[name] = candidates[0]
    return selected


def _venv_python(virtual_env: Path) -> Path:
    return virtual_env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _examples_copy_command(python: Path, destination: Path) -> list[Path | str]:
    return [
        python,
        "-I",
        "-m",
        "pyamplicol",
        "examples",
        "copy",
        destination,
    ]


def _copied_example_command(python: Path, card: Path) -> list[Path | str]:
    return [
        python,
        "-I",
        "-m",
        "pyamplicol",
        card,
        "--set",
        "generation.mode=replace",
        "--json",
    ]


def _native_command(variable: str, defaults: Sequence[str]) -> tuple[str, ...] | None:
    configured = os.environ.get(variable)
    candidates = (configured,) if configured else tuple(defaults)
    for candidate in candidates:
        if not candidate:
            continue
        try:
            command = tuple(shlex.split(candidate))
        except ValueError as error:
            raise ReleaseError(f"invalid {variable} command: {error}") from error
        if not command:
            continue
        executable = shutil.which(command[0])
        if executable is None and Path(command[0]).is_file():
            executable = str(Path(command[0]).resolve())
        if executable is not None:
            return (executable, *command[1:])
    return None


def _native_toolchain(mode: str) -> NativeToolchain | None:
    cc = _native_command("CC", ("cc", "clang", "gcc"))
    cxx = _native_command("CXX", ("c++", "clang++", "g++"))
    fortran = _native_command("FC", ("gfortran", "flang-new", "flang", "ifx"))
    rustc = _native_command("RUSTC", ("rustc",))
    missing = tuple(
        name
        for name, command in (
            ("C11", cc),
            ("C++17", cxx),
            ("Fortran 2008", fortran),
            ("Rust 2021", rustc),
        )
        if command is None
    )
    if missing:
        message = (
            "installed-wheel native SDK validation requires "
            + " and ".join(missing)
            + " compilers"
        )
        if mode == "release" or os.environ.get(
            "PYAMPLICOL_REQUIRE_NATIVE_TESTS"
        ) == "1":
            raise ReleaseError(message)
        print(f"Candidate deployment did not run native SDK smoke tests: {message}")
        return None
    assert all(command is not None for command in (cc, cxx, fortran, rustc))
    assert cc is not None and cxx is not None
    assert fortran is not None and rustc is not None
    return NativeToolchain(cc=cc, cxx=cxx, fortran=fortran, rustc=rustc)


def _sdk_payload(python: Path, environment: dict[str, str]) -> dict[str, Any]:
    completed = run(
        [python, "-I", "-m", "rusticol_config", "--json"],
        env=environment,
        capture_output=True,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ReleaseError(
            f"rusticol-config emitted invalid SDK JSON: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ReleaseError("rusticol-config SDK metadata must be an object")
    for key in (
        "cflags",
        "link_flags",
        "rust_flags",
        "fortran_source",
        "rust_source",
    ):
        if key not in payload:
            raise ReleaseError(f"rusticol-config SDK metadata is missing {key}")
    if not isinstance(payload["cflags"], list) or not all(
        isinstance(item, str) for item in payload["cflags"]
    ):
        raise ReleaseError("rusticol-config cflags must be strings")
    if not isinstance(payload["link_flags"], list) or not all(
        isinstance(item, str) for item in payload["link_flags"]
    ):
        raise ReleaseError("rusticol-config link flags must be strings")
    if not isinstance(payload["rust_flags"], list) or not all(
        isinstance(item, str) for item in payload["rust_flags"]
    ):
        raise ReleaseError("rusticol-config rust flags must be strings")
    return payload


def _installed_selftest_fixture(
    python: Path,
    *,
    sandbox: Path,
    environment: dict[str, str],
) -> NativePhysicsFixture:
    completed = run(
        [python, "-I", "-c", _SELFTEST_RESOURCE],
        env=environment,
        capture_output=True,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ReleaseError(
            f"installed self-test resource query returned invalid JSON: {error}"
        ) from error
    if not isinstance(payload, dict) or not isinstance(payload.get("expected"), dict):
        raise ReleaseError("installed self-test resource query is incomplete")
    artifact_raw = payload.get("artifact")
    if not isinstance(artifact_raw, str) or not artifact_raw:
        raise ReleaseError("installed self-test resource query has no artifact path")
    artifact = Path(artifact_raw).resolve(strict=True)
    if not artifact.is_dir() or not is_relative_to(artifact, sandbox):
        raise ReleaseError(
            "installed self-test artifact escapes the deployment sandbox"
        )
    expected = payload["expected"]
    process_id = expected.get("process_id")
    shape_raw = expected.get("resolved_shape")
    totals_raw = expected.get("total")
    if (
        not isinstance(process_id, str)
        or not process_id
        or not isinstance(shape_raw, list)
        or len(shape_raw) != 3
        or not all(isinstance(value, int) and value > 0 for value in shape_raw)
        or not isinstance(totals_raw, list)
        or not totals_raw
    ):
        raise ReleaseError("installed self-test expectation is invalid")
    totals: list[float] = []
    for index, value in enumerate(totals_raw):
        if (
            not isinstance(value, list)
            or len(value) != 2
            or isinstance(value[0], bool)
            or isinstance(value[1], bool)
            or not isinstance(value[0], (int, float))
            or not isinstance(value[1], (int, float))
            or not math.isclose(float(value[1]), 0.0, abs_tol=1.0e-15)
        ):
            raise ReleaseError(
                f"installed native self-test total[{index}] is not real f64"
            )
        totals.append(float(value[0]))
    return NativePhysicsFixture(
        artifact=artifact,
        process_id=process_id,
        shape=tuple(shape_raw),
        totals=tuple(totals),
    )


def _driver_result(
    command: Sequence[str | os.PathLike[str]],
    *,
    language: str,
    fixture: NativePhysicsFixture,
    environment: dict[str, str],
) -> dict[str, Any]:
    completed = run(
        [*command, "--json", "--process", fixture.process_id],
        cwd=fixture.artifact,
        env=environment,
        capture_output=True,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ReleaseError(
            f"installed {language} API driver returned invalid JSON: {error}"
        ) from error
    if (
        not isinstance(payload, dict)
        or payload.get("language") != language
        or payload.get("available") is not True
        or payload.get("precision") != 16
        or payload.get("process_key") != fixture.process_id
        or payload.get("shape") != list(fixture.shape)
    ):
        raise ReleaseError(
            f"installed {language} API driver returned inconsistent metadata"
        )
    values = payload.get("values")
    resolved = payload.get("resolved_sum")
    compatibility = payload.get("compatibility_total")
    expected_value_count = math.prod(fixture.shape)
    numeric_arrays = (values, resolved, compatibility)
    if not all(
        isinstance(items, list)
        and all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in items
        )
        for items in numeric_arrays
    ):
        raise ReleaseError(
            f"installed {language} API driver returned nonnumeric values"
        )
    assert isinstance(values, list)
    assert isinstance(resolved, list)
    assert isinstance(compatibility, list)
    if (
        len(values) != expected_value_count
        or len(resolved) != len(fixture.totals)
        or len(compatibility) != len(fixture.totals)
    ):
        raise ReleaseError(
            f"installed {language} API driver returned inconsistent result sizes"
        )
    for index, expected in enumerate(fixture.totals):
        scale = max(abs(expected), 1.0)
        tolerance = 1.0e-12 * scale
        if (
            abs(float(resolved[index]) - expected) > tolerance
            or abs(float(compatibility[index]) - expected) > tolerance
        ):
            raise ReleaseError(
                f"installed {language} API result[{index}] disagrees with "
                "the packaged physics expectation"
            )
    if abs(sum(map(float, values)) - sum(fixture.totals)) > (
        1.0e-12 * max(abs(sum(fixture.totals)), 1.0)
    ):
        raise ReleaseError(
            f"installed {language} resolved components do not reproduce the total"
        )
    return payload


def _compare_driver_results(results: dict[str, dict[str, Any]]) -> None:
    expected_languages = {"python", "c", "cpp", "fortran", "rust"}
    if set(results) != expected_languages:
        raise ReleaseError(
            "installed API comparison did not exercise all five languages"
        )
    reference = results["python"]
    result_keys = ("values", "resolved_sum", "compatibility_total")
    reference_fields = set(reference) - {"language", *result_keys}
    for language, payload in results.items():
        fields = set(payload) - {"language", *result_keys}
        if fields != reference_fields or any(
            payload.get(key) != reference.get(key) for key in reference_fields
        ):
            raise ReleaseError(
                f"installed {language} API driver metadata disagrees with Python"
            )
        for key in result_keys:
            reference_values = list(map(float, reference[key]))
            values = list(map(float, payload[key]))
            if len(values) != len(reference_values) or any(
                not math.isclose(left, right, rel_tol=1.0e-12, abs_tol=1.0e-15)
                for left, right in zip(values, reference_values, strict=True)
            ):
                raise ReleaseError(
                    f"installed {language} API driver disagrees with Python for {key}"
                )


def _native_sdk_smoke(
    python: Path,
    *,
    sandbox: Path,
    mode: str,
    environment: dict[str, str],
) -> bool:
    toolchain = _native_toolchain(mode)
    if toolchain is None:
        return False
    sdk = _sdk_payload(python, environment)
    fixture = _installed_selftest_fixture(
        python,
        sandbox=sandbox,
        environment=environment,
    )
    native = sandbox / "native-sdk-smoke"
    native.mkdir(parents=True, exist_ok=False)
    python_source = fixture.artifact / "API" / "python" / "check_standalone.py"
    c_source = fixture.artifact / "API" / "c" / "check_standalone.c"
    cpp_source = fixture.artifact / "API" / "cpp" / "check_standalone.cpp"
    fortran_source = fixture.artifact / "API" / "fortran" / "check_standalone.f90"
    rust_source = fixture.artifact / "API" / "rust" / "check_standalone.rs"
    missing_sources = [
        source
        for source in (python_source, c_source, cpp_source, fortran_source, rust_source)
        if not source.is_file()
    ]
    if missing_sources:
        raise ReleaseError(
            "installed self-test artifact has no complete five-language API bundle: "
            + ", ".join(
                str(path.relative_to(fixture.artifact)) for path in missing_sources
            )
        )
    c_binary = native / "check_standalone_c"
    cpp_binary = native / "check_standalone_cpp"
    fortran_binary = native / "check_standalone_fortran"
    rust_binary = native / "check_standalone_rust"

    cflags = list(map(str, sdk["cflags"]))
    link_flags = list(map(str, sdk["link_flags"]))
    rust_flags = list(map(str, sdk["rust_flags"]))
    packaged_fortran = Path(str(sdk["fortran_source"])).resolve(strict=True)
    packaged_rust = Path(str(sdk["rust_source"])).resolve(strict=True)
    run(
        [
            *toolchain.cc,
            "-std=c11",
            *cflags,
            c_source,
            "-o",
            c_binary,
            *link_flags,
        ],
        cwd=native,
        env=environment,
        capture_output=True,
    )
    run(
        [
            *toolchain.cxx,
            "-std=c++17",
            *cflags,
            cpp_source,
            "-o",
            cpp_binary,
            *link_flags,
        ],
        cwd=native,
        env=environment,
        capture_output=True,
    )
    fortran_flags = ["-std=f2008"]
    if "gfortran" in Path(toolchain.fortran[0]).name:
        fortran_flags.append("-ffree-line-length-none")
    run(
        [
            *toolchain.fortran,
            *fortran_flags,
            packaged_fortran,
            fortran_source,
            "-o",
            fortran_binary,
            *link_flags,
        ],
        cwd=native,
        env=environment,
        capture_output=True,
    )
    run(
        [
            *toolchain.rustc,
            "--edition=2021",
            "-C",
            "opt-level=2",
            rust_source,
            "-o",
            rust_binary,
            *rust_flags,
        ],
        cwd=native,
        env={**environment, "RUSTICOL_RUST_SOURCE": str(packaged_rust)},
        capture_output=True,
    )
    results = {
        "python": _driver_result(
            (python, "-I", python_source),
            language="python",
            fixture=fixture,
            environment=environment,
        ),
        "c": _driver_result(
            (c_binary,),
            language="c",
            fixture=fixture,
            environment=environment,
        ),
        "cpp": _driver_result(
            (cpp_binary,),
            language="cpp",
            fixture=fixture,
            environment=environment,
        ),
        "fortran": _driver_result(
            (fortran_binary,),
            language="fortran",
            fixture=fixture,
            environment=environment,
        ),
        "rust": _driver_result(
            (rust_binary,),
            language="rust",
            fixture=fixture,
            environment=environment,
        ),
    }
    _compare_driver_results(results)
    return True


def _symbolica_absent_f64_smoke(
    wheel: Path,
    *,
    target_python: Path,
    sandbox: Path,
    numpy_version: str,
) -> None:
    root = sandbox / "symbolica-absent-f64"
    root.mkdir(parents=True, exist_ok=False)
    virtual_env = root / "venv"
    run(
        [target_python, "-I", "-m", "venv", virtual_env],
        env=clean_environment(),
    )
    python = _venv_python(virtual_env)
    environment = clean_environment(virtual_env=virtual_env)
    run(
        [
            python,
            "-I",
            "-m",
            "pip",
            "install",
            "--only-binary=:all:",
            f"numpy=={numpy_version}",
        ],
        env=environment,
    )
    run(
        [
            python,
            "-I",
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--force-reinstall",
            wheel.resolve(),
        ],
        env=environment,
    )
    smoke_environment = runtime_environment(virtual_env)
    smoke_environment.update(
        {
            "PYAMPLICOL_FORBIDDEN_ROOT": str(ROOT),
            "PYAMPLICOL_DEPLOYMENT_SANDBOX": str(sandbox),
        }
    )
    run(
        [python, "-I", "-c", _SYMBOLICA_ABSENT_F64_SMOKE],
        env=smoke_environment,
    )


def _build_fresh_candidate_wheel(artifact_directory: Path) -> Path:
    artifact_directory.mkdir(parents=True, exist_ok=True)
    if list(artifact_directory.iterdir()):
        raise ReleaseError(
            "candidate deployment build directory must start empty: "
            f"{artifact_directory}"
        )
    run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--outdir",
            artifact_directory,
        ],
        cwd=ROOT,
        env=clean_environment(mode="candidate"),
    )
    return exactly_one(
        list(artifact_directory.glob("pyamplicol-*.whl")),
        "fresh candidate deployment wheel",
    )


def _build_release_if_needed(artifact_directory: Path) -> None:
    if list(artifact_directory.glob("pyamplicol-*.whl")):
        return
    run(
        [
            sys.executable,
            ROOT / "tools" / "release" / "build_release_artifacts.py",
        ],
        cwd=ROOT,
        env=clean_environment(mode="release"),
    )


def _install_dependencies(
    python: Path,
    *,
    virtual_env: Path,
    mode: str,
    wheelhouses: Sequence[Path],
    supported_tags: Sequence[str] | None = None,
) -> DependencyInstallation:
    lock = _release_lock()
    dependencies = exact_dependencies()
    base_command: list[str | os.PathLike[str]] = [
        python,
        "-I",
        "-m",
        "pip",
        "install",
        "--only-binary=:all:",
        "--index-url",
        "https://pypi.org/simple",
    ]
    if mode == "candidate":
        candidate_requirements = _candidate_dependency_overrides()
        dependencies.update(candidate_requirements)
        local_wheels = _candidate_dependency_wheels(
            candidate_requirements,
            wheelhouses,
            supported_tags or interpreter_tags(python),
        )
    elif wheelhouses:
        raise ReleaseError("release deployment cannot consume a local wheelhouse")
    else:
        local_wheels = {}
    distributions = {
        canonicalize_name(str(entry["distribution"])): str(entry["distribution"])
        for entry in lock["python_dependencies"]
    }
    requirements: list[str | os.PathLike[str]] = []
    for name, version in sorted(dependencies.items()):
        local = local_wheels.get(name)
        requirements.append(
            local if local is not None else f"{distributions[name]}=={version}"
        )
    command = [*base_command, *requirements]
    run(command, env=clean_environment(virtual_env=virtual_env))
    return DependencyInstallation(dependencies, local_wheels)


def test_deployment(
    wheel: Path,
    *,
    target_python: Path,
    mode: str,
    wheelhouses: Sequence[Path],
    keep: bool,
    abi3_smoke_only: bool = False,
) -> Path | None:
    report = audit_wheel(wheel, mode=mode)
    DEPLOYMENT_ROOT.mkdir(parents=True, exist_ok=True)
    sandbox = Path(tempfile.mkdtemp(prefix=f"{mode}-", dir=DEPLOYMENT_ROOT)).resolve()
    try:
        virtual_env = sandbox / "venv"
        run(
            [target_python, "-I", "-m", "venv", virtual_env],
            env=clean_environment(),
        )
        python = _venv_python(virtual_env)
        installation = _install_dependencies(
            python,
            virtual_env=virtual_env,
            mode=mode,
            wheelhouses=wheelhouses,
            supported_tags=interpreter_tags(python),
        )
        install_environment = clean_environment(virtual_env=virtual_env)
        run(
            [
                python,
                "-I",
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--force-reinstall",
                wheel.resolve(),
            ],
            env=install_environment,
        )
        run([python, "-I", "-m", "pip", "check"], env=install_environment)

        smoke_environment = runtime_environment(virtual_env)
        smoke_environment.update(
            {
                "PYAMPLICOL_DEPLOYMENT_MODE": mode,
                "PYAMPLICOL_EXPECTED_DEPENDENCIES": json.dumps(
                    installation.versions, sort_keys=True
                ),
                "PYAMPLICOL_EXPECTED_LOCAL_SOURCES": json.dumps(
                    {
                        name: {"path": str(path), "sha256": sha256(path)}
                        for name, path in installation.local_wheels.items()
                    },
                    sort_keys=True,
                ),
                "PYAMPLICOL_EXPECTED_VERSION": report.version,
                "PYAMPLICOL_FORBIDDEN_ROOT": str(ROOT),
                "PYAMPLICOL_DEPLOYMENT_SANDBOX": str(sandbox),
            }
        )
        run([python, "-I", "-c", _INSTALLED_SMOKE], env=smoke_environment)
        if not abi3_smoke_only:
            run(
                [python, "-I", "-m", "pyamplicol.selftest"],
                env=smoke_environment,
            )
        run(
            [python, "-I", "-c", _SYMBOLICA_FREE_F64_SMOKE],
            env=smoke_environment,
        )
        native_sdk_validated = False
        if not abi3_smoke_only:
            numpy_version = installation.versions.get("numpy")
            if numpy_version is None:
                raise ReleaseError(
                    "deployment dependency closure has no NumPy version"
                )
            _symbolica_absent_f64_smoke(
                wheel,
                target_python=target_python,
                sandbox=sandbox,
                numpy_version=numpy_version,
            )
            backend_environment = dict(smoke_environment)
            backend_environment["SYMBOLICA_HIDE_BANNER"] = "1"
            run(
                [python, "-I", "-c", _INSTALLED_BACKEND_AND_PRECISION_SMOKE],
                cwd=sandbox,
                env=backend_environment,
            )
            run(
                [
                    python,
                    "-I",
                    "-m",
                    "pyamplicol",
                    "self-test",
                    "--json",
                ],
                env=smoke_environment,
            )
            copied_examples = sandbox / "examples"
            run(
                _examples_copy_command(python, copied_examples),
                env=smoke_environment,
            )
            if not any(path.is_file() for path in copied_examples.rglob("*")):
                raise ReleaseError("installed example-copy command produced no files")
            example_artifact = copied_examples / "artifacts/external_ufo_sm"
            run(
                _copied_example_command(
                    python,
                    copied_examples / "external_ufo_sm.toml",
                ),
                cwd=copied_examples,
                env=smoke_environment,
            )
            if not (example_artifact / "artifact.json").is_file():
                raise ReleaseError(
                    "installed external-UFO example did not generate its artifact"
                )
            native_sdk_validated = _native_sdk_smoke(
                python,
                sandbox=sandbox,
                mode=mode,
                environment=smoke_environment,
            )
        (sandbox / "deployment-result.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "mode": mode,
                    "python": str(target_python.resolve()),
                    "wheel": wheel.name,
                    "wheel_sha256": sha256(wheel),
                    "version": report.version,
                    "validation_scope": ("abi3-smoke" if abi3_smoke_only else "full"),
                    "native_sdk_validated": native_sdk_validated,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Deployment validation passed in {sandbox}")
        return sandbox if keep else None
    finally:
        if not keep:
            shutil.rmtree(sandbox, ignore_errors=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="store_true")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--wheelhouse", type=Path, action="append", default=[])
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--keep", action="store_true")
    parser.add_argument(
        "--abi3-smoke-only",
        action="store_true",
        help=(
            "validate installed abi3/import/direct-runtime compatibility without "
            "repeating generation and native SDK tests"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    mode = build_mode(candidate=args.candidate)
    check_dependency_gate(mode)
    artifact_directory = args.artifact_dir or (
        CANDIDATE_ARTIFACTS if mode == "candidate" else DIST
    )
    with ExitStack() as stack:
        if args.wheel is not None:
            wheels = [args.wheel]
        elif mode == "candidate" and not args.no_build:
            scratch = stack.enter_context(
                external_temporary_directory("pyamplicol-candidate-deployment-build-")
            )
            wheels = [_build_fresh_candidate_wheel(scratch)]
        else:
            if not args.no_build:
                _build_release_if_needed(artifact_directory)
            wheels = sorted(artifact_directory.glob("pyamplicol-*.whl"))
        wheel = select_compatible_wheel(wheels, interpreter_tags(args.python))
        wheelhouses = [path.resolve() for path in args.wheelhouse]
        if mode == "candidate" and not wheelhouses:
            wheelhouses = wheelhouse_directories(DEPENDENCY_WHEELHOUSE)
        test_deployment(
            wheel,
            target_python=args.python,
            mode=mode,
            wheelhouses=wheelhouses,
            keep=args.keep,
            abi3_smoke_only=args.abi3_smoke_only,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReleaseError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error

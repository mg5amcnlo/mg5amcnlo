#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Stage a freshly built wheel's native resources for source-tree tests."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import sys
import zipfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from email import policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
RELEASE_TOOLS = ROOT / "tools" / "release"
sys.path.insert(0, str(RELEASE_TOOLS))

from _artifacts import audit_wheel  # noqa: E402
from _common import (  # noqa: E402
    ReleaseError,
    build_mode,
    clean_environment,
    exactly_one,
    external_temporary_directory,
    is_relative_to,
    run,
)

SOURCE_PACKAGE = ROOT / "src" / "pyamplicol"
SOURCE_RUNTIME_ROOT = ROOT / ".artifacts" / "source-runtime"
SOURCE_BUILD_INFO = SOURCE_RUNTIME_ROOT / "_build_info.json"
SOURCE_RUNTIME_STAGING = SOURCE_RUNTIME_ROOT / ".staging"
_SDK_PREFIXES = (
    "pyamplicol/_sdk/fortran/",
    "pyamplicol/_sdk/include/",
    "pyamplicol/_sdk/lib/",
)
_SDK_FILES = {
    "pyamplicol/_sdk/link.json",
    "pyamplicol/_sdk/metadata.json",
}
_NATIVE_EXTENSION_SUFFIXES = (".dylib", ".pyd", ".so")

sys.path.insert(0, str(ROOT / "build_backend"))
from native_build_identity import native_build_inputs_digest  # noqa: E402

sys.path.pop(0)


def _host_target() -> str | None:
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"} and sys.platform == "darwin":
        return "aarch64-apple-darwin"
    if machine in {"amd64", "x86_64"} and sys.platform == "darwin":
        return "x86_64-apple-darwin"
    if machine in {"amd64", "x86_64"} and sys.platform.startswith("linux"):
        return "x86_64-unknown-linux-gnu"
    return None


@contextmanager
def _publication_lock(directory: Path):
    """Serialize the developer-only publication of source runtime files."""

    try:
        import fcntl
    except ImportError as error:  # pragma: no cover - unsupported build host
        raise ReleaseError(
            "source runtime publication requires a POSIX advisory lock"
        ) from error
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".publication.lock").open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _native_build_inputs_digest(
    root: Path,
    *,
    normalize_release_cargo_lock: bool = False,
) -> str:
    return native_build_inputs_digest(
        root,
        normalize_release_cargo_lock=normalize_release_cargo_lock,
    )


def _wheel_identity(members: Mapping[str, bytes]) -> tuple[str, str]:
    metadata_names = [name for name in members if name.endswith(".dist-info/METADATA")]
    if len(metadata_names) != 1:
        raise ReleaseError("source runtime wheel must contain one METADATA file")
    metadata = BytesParser(policy=policy.default).parsebytes(members[metadata_names[0]])
    name = str(metadata.get("Name", ""))
    version = str(metadata.get("Version", ""))
    if name != "pyamplicol" or not version:
        raise ReleaseError("source runtime wheel has an invalid distribution identity")
    return name, version


def _safe_destination(source_package: Path, member: str) -> Path:
    relative = PurePosixPath(member).relative_to("pyamplicol")
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ReleaseError(f"unsafe source runtime wheel member: {member}")
    destination = source_package.joinpath(*relative.parts).resolve()
    if not is_relative_to(destination, source_package.resolve()):
        raise ReleaseError(f"source runtime wheel member escapes package: {member}")
    return destination


def _write_atomic(path: Path, data: bytes, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.staging-{os.getpid()}")
    temporary.write_bytes(data)
    temporary.chmod(mode)
    os.replace(temporary, path)


def _native_extension_inventory(source_package: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            (
                path
                for path in source_package.glob("_rusticol.*")
                if path.is_file() and path.name.endswith(_NATIVE_EXTENSION_SUFFIXES)
            ),
            key=lambda path: path.name,
        )
    )


def _stage_selftest_tree(
    source_package: Path,
    members: Mapping[str, bytes],
    member_modes: Mapping[str, int],
    *,
    prefix: str,
    target: str,
) -> None:
    root = source_package / "assets" / "selftest"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / target
    staging = root / f".{target}.staging-{os.getpid()}"
    previous = root / f".{target}.previous-{os.getpid()}"
    if staging.exists() or previous.exists():
        raise ReleaseError("stale source-runtime self-test staging directory exists")

    staging.mkdir()
    try:
        for member in sorted(name for name in members if name.startswith(prefix)):
            relative = PurePosixPath(member).relative_to(prefix)
            if (
                relative.is_absolute()
                or not relative.parts
                or any(part in {"", ".", ".."} for part in relative.parts)
            ):
                raise ReleaseError(f"unsafe source-runtime self-test member: {member}")
            _write_atomic(
                staging.joinpath(*relative.parts),
                members[member],
                mode=member_modes[member] or 0o644,
            )

        moved_previous = False
        if destination.exists():
            if destination.is_symlink() or not destination.is_dir():
                raise ReleaseError(
                    "source-runtime self-test destination is not a plain directory"
                )
            os.replace(destination, previous)
            moved_previous = True
        try:
            os.replace(staging, destination)
        except BaseException:
            if moved_previous:
                os.replace(previous, destination)
            raise
        if moved_previous:
            shutil.rmtree(previous)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _stage_runtime_locked(
    wheel: Path,
    *,
    source_package: Path = SOURCE_PACKAGE,
    source_build_info: Path = SOURCE_BUILD_INFO,
    source_root: Path = ROOT,
    mode: str,
    audit: bool = True,
) -> dict[str, object]:
    """Stage one wheel and publish its provenance only after all payloads."""

    if mode not in {"candidate", "release"}:
        raise ReleaseError(f"unsupported source runtime mode: {mode}")
    marker = source_build_info.parent / ".staging"
    _write_atomic(marker, b"incomplete\n", mode=0o600)
    wheel = wheel.resolve(strict=True)
    if audit:
        audit_wheel(wheel, mode=mode)
    with zipfile.ZipFile(wheel) as archive:
        member_modes = {
            info.filename: (info.external_attr >> 16) & 0o777
            for info in archive.infolist()
            if not info.is_dir()
        }
        members = {
            info.filename: archive.read(info)
            for info in archive.infolist()
            if not info.is_dir()
        }
    _, version = _wheel_identity(members)

    extension_names = [
        name
        for name in members
        if name.startswith("pyamplicol/_rusticol.")
        and name.rsplit(".", 1)[-1] in {"dylib", "pyd", "so"}
    ]
    if len(extension_names) != 1:
        raise ReleaseError(
            "source runtime wheel must contain one Rusticol extension, found "
            f"{len(extension_names)}"
        )
    sdk_metadata_name = "pyamplicol/_sdk/metadata.json"
    try:
        sdk_metadata = json.loads(members[sdk_metadata_name])
        target = str(sdk_metadata["target"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ReleaseError("source runtime wheel has invalid SDK metadata") from error
    if not target or target in {".", ".."} or "/" in target or "\\" in target:
        raise ReleaseError(f"source runtime wheel has an unsafe target: {target!r}")
    host_target = _host_target()
    if host_target is not None and target != host_target:
        raise ReleaseError(
            f"source runtime wheel targets {target}, but this host requires "
            f"{host_target}"
        )

    build_info_name = "pyamplicol/_build_info.json"
    if build_info_name in members:
        try:
            build_info = json.loads(members[build_info_name])
        except (TypeError, ValueError) as error:
            raise ReleaseError(
                "source runtime wheel has invalid build metadata"
            ) from error
        if not isinstance(build_info, dict):
            raise ReleaseError("source runtime wheel build metadata must be an object")
    else:
        build_info = {
            "schema_version": 1,
            "publishable": mode == "release",
            "version": version,
        }
    if build_info.get("version") != version:
        raise ReleaseError(
            "source runtime wheel build metadata and METADATA versions differ"
        )
    native_digest = _native_build_inputs_digest(
        source_root,
        normalize_release_cargo_lock=mode == "release",
    )
    if mode == "candidate":
        if build_info.get("publishable") is not False:
            raise ReleaseError("candidate source runtime wheel is not marked candidate")
        if build_info.get("native_build_inputs_sha256") != native_digest:
            raise ReleaseError(
                "candidate source runtime wheel was built from different native "
                "sources; rerun `just dev-install`"
            )
    elif not (
        build_info.get("publishable") is True
        or (
            build_info.get("publishable") is False
            and build_info.get("release_prepared_model_bootstrap") is True
        )
    ):
        raise ReleaseError(
            "release source runtime wheel has an inconsistent release marker"
        )

    selftest_prefix = f"pyamplicol/assets/selftest/{target}/"
    selected = {
        name
        for name in members
        if name in extension_names
        or name in _SDK_FILES
        or name.startswith(_SDK_PREFIXES)
        or name.startswith(selftest_prefix)
    }
    has_selftest = any(name.startswith(selftest_prefix) for name in selected)
    fixture_bootstrap = (
        build_info.get("publishable") is False
        and build_info.get("selftest_fixture_bootstrap") is True
    )
    if not has_selftest and not fixture_bootstrap:
        raise ReleaseError(
            f"source runtime wheel has no self-test fixture for {target}"
        )
    for member in sorted(
        name for name in selected if not name.startswith(selftest_prefix)
    ):
        _write_atomic(
            _safe_destination(source_package, member),
            members[member],
            mode=member_modes[member] or 0o644,
        )
    if has_selftest:
        _stage_selftest_tree(
            source_package,
            members,
            member_modes,
            prefix=selftest_prefix,
            target=target,
        )

    expected_extension = PurePosixPath(extension_names[0]).name
    for path in _native_extension_inventory(source_package):
        if path.name != expected_extension:
            path.unlink()
    final_extensions = _native_extension_inventory(source_package)
    if len(final_extensions) != 1 or final_extensions[0].name != expected_extension:
        raise ReleaseError("source runtime extension publication was not exact")

    build_info["native_build_inputs_sha256"] = native_digest
    build_info["source_runtime"] = {
        "extension_name": expected_extension,
        "extension_sha256": hashlib.sha256(members[extension_names[0]]).hexdigest(),
        "mode": mode,
        "native_build_inputs_sha256": native_digest,
    }
    _write_atomic(
        source_build_info,
        (json.dumps(build_info, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    marker.unlink()
    return {
        "mode": mode,
        "target": target,
        "version": version,
        "wheel": wheel.name,
        "staged_file_count": len(selected) + 1,
    }


def stage_runtime(
    wheel: Path,
    *,
    source_package: Path = SOURCE_PACKAGE,
    source_build_info: Path = SOURCE_BUILD_INFO,
    source_root: Path = ROOT,
    mode: str,
    audit: bool = True,
) -> dict[str, object]:
    """Stage one wheel while excluding concurrent developer publications."""

    with _publication_lock(source_build_info.parent):
        return _stage_runtime_locked(
            wheel,
            source_package=source_package,
            source_build_info=source_build_info,
            source_root=source_root,
            mode=mode,
            audit=audit,
        )


def build_and_stage(*, python: Path, mode: str) -> dict[str, object]:
    with external_temporary_directory("pyamplicol-source-runtime-") as temporary:
        run(
            [
                python,
                "-m",
                "build",
                "--wheel",
                "--outdir",
                temporary,
                *(
                    ["--config-setting", "pyamplicol.host-wheel=true"]
                    if sys.platform == "linux"
                    else []
                ),
            ],
            cwd=ROOT,
            env=clean_environment(mode=mode),
        )
        wheel = exactly_one(
            list(temporary.glob("pyamplicol-*.whl")),
            "fresh source-test wheel",
        )
        # Source staging consumes the backend's unrepaired local wheel. Release
        # artifact auditing, including final manylinux tags, runs separately.
        return stage_runtime(wheel, mode=mode, audit=False)


def stage_from_directory(directory: Path, *, mode: str) -> dict[str, object]:
    """Stage the one candidate wheel already retained in ``directory``."""

    directory = directory.expanduser().resolve(strict=True)
    if not directory.is_dir():
        raise ReleaseError(
            f"source runtime wheel directory is not a directory: {directory}"
        )
    wheel = exactly_one(
        list(directory.glob("pyamplicol-*.whl")),
        "existing source-test wheel",
    )
    return stage_runtime(wheel, mode=mode, audit=False)


def _runtime_versions():
    # Import the lightweight verifier directly: importing pyamplicol would
    # itself reject a stale runtime before we can rebuild it.
    spec = importlib.util.spec_from_file_location(
        "_source_runtime_versions", SOURCE_PACKAGE / "_internal" / "versions.py"
    )
    if spec is None or spec.loader is None:
        raise ReleaseError("cannot load the source runtime identity verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reuse_current_runtime(*, mode: str) -> dict[str, object] | None:
    """Reuse only the current, completely staged runtime in the requested mode."""

    with _publication_lock(SOURCE_RUNTIME_ROOT):
        if SOURCE_RUNTIME_STAGING.exists() or not SOURCE_BUILD_INFO.is_file():
            return None
        versions = _runtime_versions()
        try:
            payload = versions._read_build_info(
                SOURCE_BUILD_INFO, "source runtime provenance"
            )
            contract = payload.get("source_runtime")
            if not isinstance(contract, dict) or contract.get("mode") != mode:
                return None
            version = payload.get("version")
            if not isinstance(version, str) or not version:
                return None
            versions._verify_source_runtime(
                payload, package_root=SOURCE_PACKAGE, source_root=ROOT
            )
        except (OSError, RuntimeError):
            return None
        return {"mode": mode, "version": version, "reused": True}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="store_true")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--wheel-directory",
        type=Path,
        help="stage an already built local wheel instead of rebuilding it",
    )
    source.add_argument(
        "--reuse-current",
        action="store_true",
        help="reuse a matching staged runtime; otherwise build and stage normally",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    mode = build_mode(candidate=args.candidate)
    report = reuse_current_runtime(mode=mode) if args.reuse_current else None
    if report is None:
        report = (
            stage_from_directory(args.wheel_directory, mode=mode)
            if args.wheel_directory is not None
            else build_and_stage(python=args.python, mode=mode)
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReleaseError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error

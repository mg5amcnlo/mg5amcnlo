# SPDX-License-Identifier: 0BSD
"""Report native SDK metadata without initializing pyAmpliCol or Symbolica."""

from __future__ import annotations

import argparse
import importlib.util
import json
import runpy
import shlex
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class SdkUnavailableError(RuntimeError):
    """Raised when a source tree has no staged native SDK."""


@dataclass(frozen=True)
class SdkInfo:
    abi_version: int
    package_version: str
    target: str
    include_dir: Path
    library: Path
    fortran_source: Path
    rust_source: Path
    system_libraries: tuple[str, ...]
    frameworks: tuple[str, ...]

    @property
    def cflags(self) -> tuple[str, ...]:
        return (f"-I{self.include_dir}",)

    @property
    def link_flags(self) -> tuple[str, ...]:
        libraries = tuple(f"-l{name}" for name in self.system_libraries)
        frameworks = tuple(
            token for name in self.frameworks for token in ("-framework", name)
        )
        return (str(self.library), *libraries, *frameworks)

    @property
    def rust_flags(self) -> tuple[str, ...]:
        return tuple(
            token
            for link_flag in self.link_flags
            for token in ("-C", f"link-arg={link_flag}")
        )

    @property
    def cargo_encoded_rust_flags(self) -> str:
        return "\x1f".join(self.rust_flags)

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["include_dir"] = str(self.include_dir)
        payload["library"] = str(self.library)
        payload["fortran_source"] = str(self.fortran_source)
        payload["rust_source"] = str(self.rust_source)
        payload["cflags"] = list(self.cflags)
        payload["link_flags"] = list(self.link_flags)
        payload["rust_flags"] = list(self.rust_flags)
        return payload


def _resource_root() -> Path:
    # Finding the top-level package does not execute its __init__.py. SDK
    # compiler/linker queries must not acquire a numerical-runtime licence.
    spec = importlib.util.find_spec("pyamplicol")
    if spec is None or spec.origin is None or not Path(spec.origin).is_file():
        raise SdkUnavailableError(
            "Rusticol's static SDK requires a normally installed, unpacked "
            "pyamplicol wheel"
        )
    return Path(spec.origin).resolve().parent / "_sdk"


def _package_version() -> str:
    # Reuse the authoritative, stdlib-only version/source-runtime validation.
    # Importing it through pyamplicol would first initialize Symbolica.
    versions = runpy.run_path(
        str(_resource_root().parent / "_internal" / "versions.py")
    )
    return str(versions["package_version"]())


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SdkUnavailableError(
            "Rusticol's native SDK is unavailable in this source tree; "
            "install a pyamplicol binary wheel or build one with 'just wheel'"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SdkUnavailableError(f"invalid SDK metadata object: {path}")
    return payload


def _confined(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise SdkUnavailableError(f"SDK metadata escapes its resource root: {relative}")
    if not path.is_file():
        raise SdkUnavailableError(f"missing SDK resource: {relative}")
    return path


def load_sdk_info() -> SdkInfo:
    root = _resource_root()
    metadata = _load_json(root / "metadata.json")
    link = _load_json(root / "link.json")
    if metadata.get("schema_version") != 1 or link.get("schema_version") != 1:
        raise SdkUnavailableError("unsupported Rusticol SDK metadata schema")
    if metadata.get("target") != link.get("target"):
        raise SdkUnavailableError("Rusticol SDK target metadata is inconsistent")

    package_version = _package_version()
    metadata_version = str(metadata.get("version"))
    if package_version != metadata_version:
        raise SdkUnavailableError(
            "Rusticol SDK metadata version does not match pyamplicol"
        )
    library = _confined(root, str(metadata["archive"]))
    for required in (
        "include/rusticol.h",
        "include/rusticol.hpp",
        "fortran/rusticol.f90",
        "rust/rusticol.rs",
    ):
        _confined(root, required)

    return SdkInfo(
        abi_version=int(metadata["abi_version"]),
        package_version=package_version,
        target=str(metadata["target"]),
        include_dir=(root / "include").resolve(),
        library=library,
        fortran_source=_confined(root, "fortran/rusticol.f90"),
        rust_source=_confined(
            root, str(metadata.get("rust_source", "rust/rusticol.rs"))
        ),
        system_libraries=tuple(map(str, link.get("system_libraries", []))),
        frameworks=tuple(map(str, link.get("frameworks", []))),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rusticol-config",
        description="Report paths and typed linker arguments for Rusticol's SDK.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--abi-version", action="store_true")
    group.add_argument("--version", action="store_true")
    group.add_argument("--target", action="store_true")
    group.add_argument("--include-dir", action="store_true")
    group.add_argument("--library", action="store_true")
    group.add_argument("--fortran-source", action="store_true")
    group.add_argument("--rust-source", action="store_true")
    group.add_argument("--cflags", action="store_true")
    group.add_argument("--libs", action="store_true")
    group.add_argument("--rustflags", action="store_true")
    group.add_argument("--cargo-rustflags", action="store_true")
    group.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        info = load_sdk_info()
    except SdkUnavailableError as error:
        _parser().error(str(error))

    if args.abi_version:
        print(info.abi_version)
    elif args.version:
        print(info.package_version)
    elif args.target:
        print(info.target)
    elif args.include_dir:
        print(info.include_dir)
    elif args.library:
        print(info.library)
    elif args.fortran_source:
        print(info.fortran_source)
    elif args.rust_source:
        print(info.rust_source)
    elif args.cflags:
        print(shlex.join(info.cflags))
    elif args.libs:
        print(shlex.join(info.link_flags))
    elif args.rustflags:
        print(shlex.join(info.rust_flags))
    elif args.cargo_rustflags:
        print(info.cargo_encoded_rust_flags)
    elif args.json:
        print(json.dumps(info.to_json(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Pack, unpack and fetch the single prepared-model release archive.

Prepared-model packs are not tracked in Git. Each MadGraph7 release attaches
one archive holding the metadata of every architecture pair and each distinct
bundle once (named by its SHA-256). Unpacking restores the files in both the
package asset directory and the release store.

    prepared_models_archive.py pack --output DIR
    prepared_models_archive.py unpack ARCHIVE
    prepared_models_archive.py fetch [--version V] [--output DIR]
    prepared_models_archive.py regenerate --work DIR

``regenerate`` repeats the release-prepared-models workflow on this host: it
builds the bootstrap wheel, produces the packs with it and, as the packs are
portable, installs them for both architectures.

Stdlib only, so that installers can run it before any build dependency exists.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import venv
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
ASSET_DIRECTORY = ROOT / "src" / "pyamplicol" / "assets" / "prepared_models"
RELEASE_STORE = ROOT / "release_assets" / "prepared_models"
ARCHIVE_NAME = "pyamplicol-prepared-models.tar.gz"
RELEASES_REPOSITORY = "MadGraphTeam/MadGraph7"
IDENTIFIERS = ("built-in-sm-jit-o2", "built-in-sm-heft-jit-o2")
ARCHITECTURES = ("aarch64", "x86_64")
ARCHIVE_SCHEMA = 1
_MAX_MEMBER_BYTES = 512 * 1024 * 1024


class ArchiveError(RuntimeError):
    """The prepared-model archive is missing, malformed or inconsistent."""


def _stems() -> list[str]:
    return [f"{ident}-{arch}" for ident in IDENTIFIERS for arch in ARCHITECTURES]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o644
    info.mtime = 0
    info.uname = info.gname = ""
    tar.addfile(info, io.BytesIO(data))


def pack(source: Path, output: Path) -> Path:
    """Write a deterministic archive from one directory of 8 pack files."""

    metadata: dict[str, bytes] = {}
    bundles: dict[str, bytes] = {}
    for stem in _stems():
        raw = (source / f"{stem}.metadata.json").read_bytes()
        record = json.loads(raw)
        data = (source / record["bundle"]).read_bytes()
        if _sha256(data) != record["bundle_sha256"]:
            raise ArchiveError(f"{record['bundle']} does not match its metadata")
        metadata[stem] = raw
        bundles[record["bundle_sha256"]] = data
    manifest = {
        "schema_version": ARCHIVE_SCHEMA,
        "metadata": sorted(metadata),
        "bundles": sorted(bundles),
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / ARCHIVE_NAME
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
        _add(tar, "manifest.json", json.dumps(manifest, indent=2).encode() + b"\n")
        for stem in sorted(metadata):
            _add(tar, f"metadata/{stem}.metadata.json", metadata[stem])
        for digest in sorted(bundles):
            _add(tar, f"bundles/{digest}.pyamplicol-model", bundles[digest])
    with (
        path.open("wb") as raw_stream,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw_stream, mtime=0) as stream,
    ):
        stream.write(buffer.getvalue())
    return path


def _read_members(archive: Path) -> dict[str, bytes]:
    members: dict[str, bytes] = {}
    with tarfile.open(archive, mode="r:gz") as tar:
        for info in tar:
            name = PurePosixPath(info.name)
            if not info.isfile() or name.is_absolute() or ".." in name.parts:
                raise ArchiveError(f"unsafe archive member: {info.name}")
            if info.size > _MAX_MEMBER_BYTES:
                raise ArchiveError(f"archive member is too large: {info.name}")
            stream = tar.extractfile(info)
            assert stream is not None
            members[info.name] = stream.read()
    return members


def unpack(archive: Path, destinations: tuple[Path, ...] | None = None) -> None:
    """Verify the archive and write the 8 pack files into each destination."""

    destinations = destinations or (ASSET_DIRECTORY, RELEASE_STORE)
    members = _read_members(archive)
    try:
        manifest = json.loads(members["manifest.json"])
    except (KeyError, ValueError) as error:
        raise ArchiveError("archive has no valid manifest") from error
    if manifest.get("schema_version") != ARCHIVE_SCHEMA:
        raise ArchiveError("unsupported prepared-model archive schema")
    files: dict[str, bytes] = {}
    for stem in _stems():
        raw = members.get(f"metadata/{stem}.metadata.json")
        if raw is None:
            raise ArchiveError(f"archive has no metadata for {stem}")
        record = json.loads(raw)
        data = members.get(f"bundles/{record['bundle_sha256']}.pyamplicol-model")
        if data is None or _sha256(data) != record["bundle_sha256"]:
            raise ArchiveError(f"archive bundle for {stem} is missing or corrupt")
        files[f"{stem}.metadata.json"] = raw
        files[record["bundle"]] = data
    for destination in destinations:
        destination.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            (destination / name).write_bytes(data)


def _download(url: str, target: Path) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            target.write_bytes(response.read())
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return False
        raise
    return True


def _release_tags(version: str | None) -> list[str]:
    tags = [f"v{version}"] if version else []
    api = f"https://api.github.com/repos/{RELEASES_REPOSITORY}/releases?per_page=30"
    try:
        with urllib.request.urlopen(api, timeout=60) as response:
            releases = json.load(response)
    except (urllib.error.URLError, ValueError):
        return tags
    for release in releases:
        if release.get("draft") or not any(
            asset.get("name") == ARCHIVE_NAME for asset in release.get("assets", ())
        ):
            continue
        if release["tag_name"] not in tags:
            tags.append(release["tag_name"])
    return tags


def fetch(output: Path, version: str | None = None) -> Path:
    """Download the archive of release ``version`` or else the latest one."""

    output.mkdir(parents=True, exist_ok=True)
    target = output / ARCHIVE_NAME
    base = f"https://github.com/{RELEASES_REPOSITORY}/releases/download"
    for tag in _release_tags(version):
        if _download(f"{base}/{tag}/{ARCHIVE_NAME}", target):
            print(f"Fetched {ARCHIVE_NAME} from release {tag}")
            return target
    raise ArchiveError(f"no MadGraph7 release provides {ARCHIVE_NAME}")


def _run(command: list[str | os.PathLike[str]]) -> None:
    print("$ " + " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def regenerate(work: Path) -> None:
    """Produce the packs on this host and install them into both stores."""

    machine = platform.machine()
    host = {"arm64": "aarch64", "aarch64": "aarch64", "x86_64": "x86_64"}.get(machine)
    if host is None:
        raise ArchiveError(f"cannot produce prepared models on {machine}")
    for path in (work / "wheel", work / "output"):
        if path.exists():
            shutil.rmtree(path)
    tools = ROOT / "tools"
    _run(
        [
            sys.executable,
            tools / "release" / "prepare_release_prepared_models.py",
            "bootstrap-wheel",
            work / "wheel",
        ]
    )
    runtime = work / "runtime"
    if not runtime.exists():
        venv.create(runtime, with_pip=True)
    python = runtime / "bin" / "python"
    wheels = sorted((work / "wheel").glob("pyamplicol-*.whl"))
    _run([python, "-m", "pip", "install", "--force-reinstall", *wheels])
    _run(
        [
            python,
            tools / "ci" / "eager_portability.py",
            "produce",
            work / "output",
            "--python",
            python,
            "--asset-mode",
            "release",
            "--expected-system",
            platform.system(),
            "--expected-machine",
            machine,
        ]
    )
    produced = work / "output" / "release_assets" / "prepared_models"
    for destination in (ASSET_DIRECTORY, RELEASE_STORE):
        for arch in ARCHITECTURES:
            for ident in IDENTIFIERS:
                stem = f"{ident}-{host}"
                record = json.loads((produced / f"{stem}.metadata.json").read_text())
                data = (produced / record["bundle"]).read_bytes()
                record["bundle"] = f"{ident}-{arch}.pyamplicol-model"
                (destination / record["bundle"]).write_bytes(data)
                (destination / f"{ident}-{arch}.metadata.json").write_text(
                    json.dumps(record, indent=2, sort_keys=True) + "\n"
                )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    pack_parser = commands.add_parser("pack", help="create the archive")
    pack_parser.add_argument("--source", type=Path, default=ASSET_DIRECTORY)
    pack_parser.add_argument("--output", type=Path, required=True)
    unpack_parser = commands.add_parser("unpack", help="restore the pack files")
    unpack_parser.add_argument("archive", type=Path)
    fetch_parser = commands.add_parser("fetch", help="download and unpack")
    fetch_parser.add_argument("--version")
    fetch_parser.add_argument("--output", type=Path)
    fetch_parser.add_argument(
        "--no-unpack", action="store_true", help="only download the archive"
    )
    regenerate_parser = commands.add_parser(
        "regenerate", help="produce the packs on this host"
    )
    regenerate_parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "pack":
            print(pack(args.source, args.output))
        elif args.command == "unpack":
            unpack(args.archive)
        elif args.command == "regenerate":
            regenerate(args.work)
        elif args.output is not None:
            archive = fetch(args.output, args.version)
            if not args.no_unpack:
                unpack(archive)
        else:
            with tempfile.TemporaryDirectory() as temporary:
                unpack(fetch(Path(temporary), args.version))
    except (
        ArchiveError,
        OSError,
        urllib.error.URLError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

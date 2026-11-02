# SPDX-License-Identifier: 0BSD
"""Locked transactional directory updates for generated artifacts."""

from __future__ import annotations

import ctypes
import errno
import fcntl
import os
import shutil
import sys
import uuid
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, Literal

from pyamplicol.api.errors import ArtifactError

from .security import fsync_directory

ArtifactWriteMode = Literal["error", "append", "replace"]

_AT_FDCWD = -100
_RENAME_EXCHANGE = 0x2
_LINUX_X86_64_RENAMEAT2 = 316


def _raise_exchange_error(source: Path, destination: Path) -> None:
    error_number = ctypes.get_errno() or errno.EIO
    raise OSError(
        error_number,
        (f"atomic artifact directory exchange failed: {os.strerror(error_number)}"),
        f"{source} <-> {destination}",
    )


def _exchange_directories(source: Path, destination: Path) -> None:
    """Atomically exchange two existing directories on release platforms."""

    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    libc = ctypes.CDLL(None, use_errno=True)
    ctypes.set_errno(0)

    if sys.platform == "darwin":
        try:
            renameatx_np = libc.renameatx_np
        except AttributeError as error:
            raise OSError(
                errno.ENOTSUP,
                "this macOS runtime does not provide atomic directory exchange",
            ) from error
        renameatx_np.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameatx_np.restype = ctypes.c_int
        result = renameatx_np(
            _AT_FDCWD,
            source_bytes,
            _AT_FDCWD,
            destination_bytes,
            _RENAME_EXCHANGE,
        )
    elif sys.platform.startswith("linux"):
        try:
            renameat2 = libc.renameat2
        except AttributeError:
            if os.uname().machine != "x86_64":
                raise OSError(
                    errno.ENOTSUP,
                    "atomic directory exchange is unsupported on this Linux target",
                ) from None
            syscall = libc.syscall
            syscall.restype = ctypes.c_long
            result = syscall(
                _LINUX_X86_64_RENAMEAT2,
                _AT_FDCWD,
                source_bytes,
                _AT_FDCWD,
                destination_bytes,
                _RENAME_EXCHANGE,
            )
        else:
            renameat2.argtypes = (
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            )
            renameat2.restype = ctypes.c_int
            result = renameat2(
                _AT_FDCWD,
                source_bytes,
                _AT_FDCWD,
                destination_bytes,
                _RENAME_EXCHANGE,
            )
    else:
        raise OSError(
            errno.ENOTSUP,
            "atomic artifact replacement is supported only on macOS and Linux",
        )

    if result != 0:
        _raise_exchange_error(source, destination)


class ArtifactTransaction:
    """Prepare a sibling directory and publish it only after successful work."""

    def __init__(
        self,
        destination: str | Path,
        *,
        mode: ArtifactWriteMode = "error",
    ) -> None:
        if mode not in ("error", "append", "replace"):
            raise ValueError("artifact mode must be 'error', 'append', or 'replace'")
        self.destination = Path(destination).expanduser().resolve(strict=False)
        self.mode = mode
        self.staging = self.destination.with_name(
            f".{self.destination.name}.staging-{uuid.uuid4().hex}"
        )
        self._lock_path = self.destination.with_name(f".{self.destination.name}.lock")
        self._lock_stream: BinaryIO | None = None
        self._entered = False
        self._committed = False

    def __enter__(self) -> Path:
        if self._entered:
            raise RuntimeError("artifact transaction cannot be entered twice")
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        lock_stream = self._lock_path.open("a+b")
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        self._lock_stream = lock_stream
        self._entered = True
        try:
            if self.destination.exists() and not self.destination.is_dir():
                raise ArtifactError(
                    f"artifact destination is not a directory: {self.destination}"
                )
            if self.mode == "error" and self.destination.exists():
                raise FileExistsError(f"artifact already exists: {self.destination}")
            if self.mode == "append":
                if not self.destination.is_dir():
                    raise FileNotFoundError(
                        f"cannot append to missing artifact: {self.destination}"
                    )
                shutil.copytree(self.destination, self.staging)
            else:
                self.staging.mkdir()
            return self.staging
        except BaseException:
            self._release()
            raise

    def commit(self) -> None:
        if not self._entered:
            raise RuntimeError("artifact transaction has not been entered")
        if self._committed:
            raise RuntimeError("artifact transaction is already committed")
        if not self.staging.is_dir():
            raise ArtifactError("artifact staging directory disappeared before commit")
        fsync_directory(self.staging)
        try:
            if self.destination.exists():
                _exchange_directories(self.staging, self.destination)
            else:
                os.replace(self.staging, self.destination)
            fsync_directory(self.destination.parent)
            self._committed = True
        except BaseException:
            # Atomic exchange leaves both directory names untouched on failure.
            raise

    def _release(self) -> None:
        if self.staging.exists():
            shutil.rmtree(self.staging)
        if self._lock_stream is not None:
            stream = self._lock_stream
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            stream.close()
            self._lock_stream = None

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        del exception, traceback
        try:
            if exception_type is None and not self._committed:
                self.commit()
        finally:
            self._release()
        return False


__all__ = ["ArtifactTransaction", "ArtifactWriteMode"]

# SPDX-License-Identifier: 0BSD
"""Prepared-kernel target identities and host compatibility checks."""

from __future__ import annotations

import importlib
import platform
import struct
import sys
from collections.abc import Mapping, Sequence

from pyamplicol._internal.versions import verify_native_module

SYMJIT_STORAGE_V3_ABI = "symjit-application-storage-v3"
SYMJIT_STORAGE_V3_TARGET_PREFIX = "symjit-storage-v3"
SYMJIT_STORAGE_V3_PORTABLE_TARGET = f"{SYMJIT_STORAGE_V3_TARGET_PREFIX}-portable"


class PreparedTargetError(ValueError):
    """Raised when a prepared kernel pack cannot execute on this host."""


def canonical_architecture(machine: str | None = None) -> str:
    """Return the architecture identity used by prepared JIT storage v3."""

    value = platform.machine() if machine is None else machine
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "amd64": "x86_64",
        "x64": "x86_64",
        "x86_64": "x86_64",
        "aarch64": "aarch64",
        "arm64": "aarch64",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise PreparedTargetError(
            f"prepared JIT kernels do not support host architecture {value!r}"
        ) from exc


def symjit_storage_v3_target(
    *,
    machine: str | None = None,
    word_bits: int | None = None,
    endianness: str | None = None,
) -> dict[str, object]:
    """Describe a portable O2 SymJIT storage-v3 prepared-kernel pack."""

    bits = struct.calcsize("P") * 8 if word_bits is None else word_bits
    byte_order = sys.byteorder if endianness is None else endianness
    if bits != 64 or byte_order != "little":
        raise PreparedTargetError(
            "prepared JIT kernels require a 64-bit little-endian host"
        )
    # Fail early on unsupported hosts while keeping the stored target
    # architecture-neutral. SymJIT storage-v3 applications compiled at O2 are
    # portable between the supported x86-64 and AArch64 hosts.
    canonical_architecture(machine)
    return {
        "portable": True,
        "word_bits": 64,
        "endianness": "little",
        "target_triple": SYMJIT_STORAGE_V3_PORTABLE_TARGET,
        "cpu_features": [],
    }


def native_prepared_target(*, include_cpu_features: bool) -> dict[str, object]:
    """Describe a target-native C++ or ASM prepared pack."""

    try:
        rusticol = importlib.import_module("pyamplicol._rusticol")
        verify_native_module(rusticol)
        info = rusticol.target_info()
    except (AttributeError, ImportError, OSError) as exc:
        raise PreparedTargetError(
            "native prepared packs require Rusticol target introspection"
        ) from exc
    features = (
        sorted(str(item) for item in info.cpu_features) if include_cpu_features else []
    )
    return {
        "portable": False,
        "word_bits": struct.calcsize("P") * 8,
        "endianness": sys.byteorder,
        "target_triple": str(info.triple),
        "cpu_features": features,
    }


def validate_prepared_target(
    target: Mapping[str, object],
    *,
    backend: str,
    symjit_application_abi: str | None = None,
    machine: str | None = None,
    word_bits: int | None = None,
    endianness: str | None = None,
) -> None:
    """Fail before DAG construction when a prepared pack cannot run here."""

    if backend == "jit":
        if symjit_application_abi != SYMJIT_STORAGE_V3_ABI:
            raise PreparedTargetError(
                "prepared JIT pack declares unsupported SymJIT application ABI "
                f"{symjit_application_abi!r}"
            )
        expected = symjit_storage_v3_target(
            machine=machine,
            word_bits=word_bits,
            endianness=endianness,
        )
        actual = _plain_target(target)
        if actual != expected:
            raise PreparedTargetError(
                "prepared JIT pack target is incompatible with this host: "
                f"pack={actual.get('target_triple')!r}, expected the portable "
                f"target {expected['target_triple']!r}; recompile the prepared "
                "model with SymJIT optimization level 2"
            )
        return

    if backend not in {"asm", "cpp"}:
        raise PreparedTargetError(f"unsupported prepared backend {backend!r}")
    expected = native_prepared_target(include_cpu_features=False)
    actual = _plain_target(target)
    for field in ("portable", "word_bits", "endianness", "target_triple"):
        if actual.get(field) != expected[field]:
            raise PreparedTargetError(
                "native prepared pack target is incompatible with this host: "
                f"pack={actual.get('target_triple')!r}, "
                f"host={expected['target_triple']!r}"
            )
    required_features = set(_strings(actual.get("cpu_features")))
    try:
        rusticol = importlib.import_module("pyamplicol._rusticol")
        verify_native_module(rusticol)
        available_features = {str(item) for item in rusticol.target_info().cpu_features}
    except (AttributeError, ImportError, OSError) as exc:
        raise PreparedTargetError(
            "native prepared packs require Rusticol target introspection"
        ) from exc
    missing = sorted(required_features - available_features)
    if missing:
        raise PreparedTargetError(
            "native prepared pack requires unavailable CPU features: "
            + ", ".join(missing)
        )


def _plain_target(target: Mapping[str, object]) -> dict[str, object]:
    return {
        "portable": target.get("portable"),
        "word_bits": target.get("word_bits"),
        "endianness": target.get("endianness"),
        "target_triple": target.get("target_triple"),
        "cpu_features": list(_strings(target.get("cpu_features"))),
    }


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise PreparedTargetError("prepared target cpu_features must be an array")
    if not all(isinstance(item, str) for item in value):
        raise PreparedTargetError(
            "prepared target cpu_features must contain only strings"
        )
    return tuple(value)


__all__ = [
    "SYMJIT_STORAGE_V3_ABI",
    "SYMJIT_STORAGE_V3_PORTABLE_TARGET",
    "SYMJIT_STORAGE_V3_TARGET_PREFIX",
    "PreparedTargetError",
    "canonical_architecture",
    "native_prepared_target",
    "symjit_storage_v3_target",
    "validate_prepared_target",
]

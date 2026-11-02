# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import pytest

from pyamplicol.models.prepared_target import (
    SYMJIT_STORAGE_V3_ABI,
    PreparedTargetError,
    canonical_architecture,
    symjit_storage_v3_target,
    validate_prepared_target,
)


@pytest.mark.parametrize(
    ("machine", "expected"),
    (
        ("arm64", "aarch64"),
        ("aarch64", "aarch64"),
        ("AMD64", "x86_64"),
        ("x86_64", "x86_64"),
    ),
)
def test_canonical_architecture_aliases(machine: str, expected: str) -> None:
    assert canonical_architecture(machine) == expected


def test_symjit_storage_v3_target_is_cross_architecture_portable() -> None:
    target = symjit_storage_v3_target(machine="arm64")

    assert target == {
        "portable": True,
        "word_bits": 64,
        "endianness": "little",
        "target_triple": "symjit-storage-v3-portable",
        "cpu_features": [],
    }


def test_symjit_storage_v3_target_rejects_unsupported_host() -> None:
    with pytest.raises(PreparedTargetError, match="architecture"):
        symjit_storage_v3_target(machine="sparc64")
    with pytest.raises(PreparedTargetError, match="64-bit little-endian"):
        symjit_storage_v3_target(machine="arm64", word_bits=32)


def test_prepared_jit_target_accepts_same_architecture_across_os() -> None:
    target = symjit_storage_v3_target(machine="AMD64")

    validate_prepared_target(
        target,
        backend="jit",
        symjit_application_abi=SYMJIT_STORAGE_V3_ABI,
        machine="x86_64",
        word_bits=64,
        endianness="little",
    )


def test_prepared_jit_target_accepts_cross_architecture() -> None:
    target = symjit_storage_v3_target(machine="arm64")

    validate_prepared_target(
        target,
        backend="jit",
        symjit_application_abi=SYMJIT_STORAGE_V3_ABI,
        machine="x86_64",
        word_bits=64,
        endianness="little",
    )


def test_prepared_jit_target_rejects_architecture_scoped_legacy_target() -> None:
    target = symjit_storage_v3_target(machine="arm64")
    target["portable"] = False
    target["target_triple"] = "symjit-storage-v3-aarch64"

    with pytest.raises(PreparedTargetError, match="optimization level 2"):
        validate_prepared_target(
            target,
            backend="jit",
            symjit_application_abi=SYMJIT_STORAGE_V3_ABI,
            machine="arm64",
        )


def test_prepared_jit_target_rejects_unknown_application_abi() -> None:
    with pytest.raises(PreparedTargetError, match="unsupported SymJIT"):
        validate_prepared_target(
            symjit_storage_v3_target(machine="arm64"),
            backend="jit",
            symjit_application_abi="symjit-application-storage-v4",
            machine="arm64",
        )

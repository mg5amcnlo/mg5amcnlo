# SPDX-License-Identifier: 0BSD

from __future__ import annotations

import ctypes
import hashlib
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from pyamplicol.evaluators.native_eager_direct_cpp import (
    NATIVE_EAGER_DIRECT_TABLE_APPLICATION_ABI,
    render_native_eager_direct_table_cpp,
)


class _AddEvaluator:
    def get_instructions(self) -> tuple[list[object], int, list[object]]:
        return (
            [("add", ("out", 0), [("param", 0), ("param", 1)], 0)],
            0,
            [],
        )

    def save(self) -> bytes:
        return b"native-eager-direct-table-test-state"


class _DirectPlane(ctypes.Structure):
    _fields_ = [
        ("values", ctypes.POINTER(ctypes.c_double)),
        ("length", ctypes.c_size_t),
    ]


class _DirectScalar(ctypes.Structure):
    _fields_ = [("value", ctypes.POINTER(ctypes.c_double))]


class _DirectTableCallView(ctypes.Structure):
    _fields_ = [
        ("invocations", ctypes.POINTER(ctypes.c_uint8)),
        ("invocation_count", ctypes.c_uint32),
        ("invocation_stride", ctypes.c_uint32),
        ("attachments", ctypes.POINTER(ctypes.c_uint8)),
        ("attachment_count", ctypes.c_uint32),
        ("attachment_stride", ctypes.c_uint32),
        ("planes", ctypes.POINTER(_DirectPlane)),
        ("plane_count", ctypes.c_uint32),
        ("scalar_count", ctypes.c_uint32),
        ("scalars", ctypes.POINTER(_DirectScalar)),
        ("factor_re", ctypes.POINTER(ctypes.c_double)),
        ("factor_im", ctypes.POINTER(ctypes.c_double)),
        ("factor_count", ctypes.c_uint32),
        ("point_start", ctypes.c_uint32),
        ("point_count", ctypes.c_uint32),
    ]


class _NativeEagerDirectTableMetadata(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("invocation_stride", ctypes.c_uint32),
        ("attachment_stride", ctypes.c_uint32),
        ("input_complex_count", ctypes.c_uint32),
        ("output_complex_count", ctypes.c_uint32),
        ("simd_lane_width", ctypes.c_uint32),
        ("application_abi", ctypes.c_char_p),
        ("function_name", ctypes.c_char_p),
        ("target_triple", ctypes.c_char_p),
        ("evaluator_state_sha256", ctypes.c_char_p),
    ]


@pytest.mark.skipif(
    sys.platform not in {"darwin", "linux"},
    reason="native table smoke needs a POSIX shared-library compiler",
)
def test_native_eager_table_is_plane_native_and_preserves_fanout_order(
    tmp_path: Path,
) -> None:
    rendered = render_native_eager_direct_table_cpp(
        _AddEvaluator(),
        kernel_id=7,
        input_complex_count=2,
        output_complex_count=1,
        target_triple="unit-test-target",
        evaluator_state_bytes=b"packaged-native-eager-state",
    )
    assert "_complexf64(" not in rendered.source
    assert "_realf64(" not in rendered.source
    assert rendered.invocation_stride == 24
    assert rendered.attachment_stride == 16
    assert rendered.evaluator_state_sha256 == hashlib.sha256(
        b"packaged-native-eager-state"
    ).hexdigest()

    source = tmp_path / "eager_direct.cpp"
    library = tmp_path / (
        "libeager_direct.dylib" if sys.platform == "darwin" else "libeager_direct.so"
    )
    source.write_text(rendered.source, encoding="utf-8")
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-shared",
            "-O3",
            "-fPIC",
            str(source),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    points = 5  # exercises two SIMD bundles plus an odd scalar tail.
    arrays = [
        (ctypes.c_double * points)(1.0, 2.0, 3.0, 4.0, 5.0),
        (ctypes.c_double * points)(0.5, 1.0, 1.5, 2.0, 2.5),
        (ctypes.c_double * points)(10.0, 20.0, 30.0, 40.0, 50.0),
        (ctypes.c_double * points)(-1.0, -2.0, -3.0, -4.0, -5.0),
        (ctypes.c_double * points)(*([99.0] * points)),
        (ctypes.c_double * points)(*([99.0] * points)),
    ]
    planes = (_DirectPlane * len(arrays))(
        *(_DirectPlane(values, points) for values in arrays)
    )
    invocation = (ctypes.c_uint8 * rendered.invocation_stride).from_buffer_copy(
        struct.pack("<6I", 0, 1, 2, 3, 0, 2)
    )
    attachments = (
        ctypes.c_uint8 * (2 * rendered.attachment_stride)
    ).from_buffer_copy(
        struct.pack("<4I", 4, 5, 0, 0)
        + struct.pack("<4I", 4, 5, 1, 1)
    )
    factor_re = (ctypes.c_double * 2)(1.0, 2.0)
    factor_im = (ctypes.c_double * 2)(0.0, 0.0)
    view = _DirectTableCallView(
        invocation,
        1,
        rendered.invocation_stride,
        attachments,
        2,
        rendered.attachment_stride,
        planes,
        len(arrays),
        0,
        None,
        factor_re,
        factor_im,
        2,
        0,
        points,
    )
    loaded = ctypes.CDLL(str(library))
    metadata_call = getattr(loaded, f"{rendered.function_name}_metadata_v1")
    metadata_call.argtypes = []
    metadata_call.restype = ctypes.POINTER(_NativeEagerDirectTableMetadata)
    metadata = metadata_call().contents
    assert metadata.struct_size == ctypes.sizeof(_NativeEagerDirectTableMetadata)
    assert metadata.abi_version == 1
    assert metadata.flags == 0x1F
    assert metadata.invocation_stride == rendered.invocation_stride
    assert metadata.attachment_stride == rendered.attachment_stride
    assert metadata.input_complex_count == 2
    assert metadata.output_complex_count == 1
    assert metadata.simd_lane_width == rendered.simd_lane_width
    assert metadata.application_abi.decode() == (
        NATIVE_EAGER_DIRECT_TABLE_APPLICATION_ABI
    )
    assert metadata.function_name.decode() == rendered.function_name
    assert metadata.target_triple.decode() == "unit-test-target"
    assert (
        metadata.evaluator_state_sha256.decode()
        == rendered.evaluator_state_sha256
    )
    call = getattr(loaded, rendered.function_name)
    call.argtypes = [ctypes.POINTER(_DirectTableCallView)]
    call.restype = ctypes.c_int
    assert call(ctypes.byref(view)) == 0

    expected_re = [
        3.0 * (arrays[0][point] + arrays[2][point])
        for point in range(points)
    ]
    expected_im = [
        3.0 * (arrays[1][point] + arrays[3][point])
        for point in range(points)
    ]
    assert list(arrays[4]) == expected_re
    assert list(arrays[5]) == expected_im

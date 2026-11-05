# SPDX-License-Identifier: 0BSD
"""Emit the wheel-independent API examples bundled with process artifacts."""

from __future__ import annotations

import importlib.resources
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal

from .manifest import PayloadRecord
from .writer import ArtifactBuilder

Scalar = float | int | str | Decimal
ValidationPoint = Sequence[Sequence[Scalar]]

_TEMPLATES: tuple[tuple[str, str, str, bool], ...] = (
    (
        "python/check_standalone.py",
        "text/x-python",
        "api-source",
        True,
    ),
    (
        "c/check_standalone.c",
        "text/x-csrc",
        "api-source",
        False,
    ),
    ("c/Makefile", "text/x-makefile", "api-build-file", False),
    (
        "cpp/check_standalone.cpp",
        "text/x-c++src",
        "api-source",
        False,
    ),
    ("cpp/Makefile", "text/x-makefile", "api-build-file", False),
    (
        "fortran/check_standalone.f90",
        "text/x-fortran",
        "api-source",
        False,
    ),
    ("fortran/Makefile", "text/x-makefile", "api-build-file", False),
    (
        "rust/check_standalone.rs",
        "text/x-rust",
        "api-source",
        False,
    ),
    ("rust/Makefile", "text/x-makefile", "api-build-file", False),
    ("umami/Makefile", "text/x-makefile", "api-build-file", False),
    ("umami/umami.h", "text/x-chdr", "api-source", False),
    ("umami/umami_provider.h", "text/x-chdr", "api-source", False),
    ("umami/umami.c", "text/x-csrc", "api-source", False),
    ("umami/umami_driver.c", "text/x-csrc", "api-source", False),
    ("umami/umami.exports", "text/plain", "api-build-file", False),
    ("umami/umami.exports.macos", "text/plain", "api-build-file", False),
    ("umami/UMAMI_LICENSE", "text/plain", "sdk-metadata", False),
    ("umami/README.md", "text/markdown", "sdk-metadata", False),
)


@dataclass(frozen=True, slots=True)
class ApiBundlePayload:
    path: str
    content: bytes
    role: str
    media_type: str
    executable: bool = False


def _template_bytes(relative: str) -> bytes:
    root = importlib.resources.files("pyamplicol.assets.api_templates")
    resource = root.joinpath(*relative.split("/"))
    if not resource.is_file():
        raise RuntimeError(f"packaged API template is missing: {relative}")
    return resource.read_bytes()


def _scalar_text(value: Scalar) -> str:
    if isinstance(value, bool):
        raise TypeError("validation momenta must not contain booleans")
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("validation momenta must be finite")
        return str(value)
    if isinstance(value, str):
        parsed = Decimal(value)
        if not parsed.is_finite():
            raise ValueError("validation momenta must be finite")
        return value
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("validation momenta must be finite")
    return format(numeric, ".17g")


def format_validation_points(
    points: Mapping[str, ValidationPoint],
) -> bytes:
    """Serialize deterministic f64-compatible points in a language-neutral format."""

    lines = ["RUSTICOL_VALIDATION_POINTS_V1"]
    for process_id in sorted(points):
        if not process_id or any(character.isspace() for character in process_id):
            raise ValueError("validation-point process IDs must be non-empty tokens")
        momenta = tuple(tuple(momentum) for momentum in points[process_id])
        if not momenta:
            raise ValueError(f"validation point for {process_id!r} is empty")
        if any(len(momentum) != 4 for momentum in momenta):
            raise ValueError(
                f"validation point for {process_id!r} must use four-vectors"
            )
        components = [
            _scalar_text(component) for momentum in momenta for component in momentum
        ]
        lines.append("\t".join((process_id, str(len(momenta)), *components)))
    return ("\n".join(lines) + "\n").encode("ascii")


def api_bundle_payloads(
    validation_points: Mapping[str, ValidationPoint] | None = None,
) -> tuple[ApiBundlePayload, ...]:
    payloads = [
        ApiBundlePayload(
            path=f"API/{relative}",
            content=_template_bytes(relative),
            role=role,
            media_type=media_type,
            executable=executable,
        )
        for relative, media_type, role, executable in _TEMPLATES
    ]
    payloads.append(
        ApiBundlePayload(
            path="API/validation_points.dat",
            content=format_validation_points(validation_points or {}),
            role="validation-momenta",
            media_type="text/tab-separated-values",
        )
    )
    return tuple(payloads)


def emit_api_bundle(
    builder: ArtifactBuilder,
    validation_points: Mapping[str, ValidationPoint] | None = None,
) -> tuple[PayloadRecord, ...]:
    """Write exactly one root API bundle through an active artifact builder."""

    return tuple(
        builder.add_bytes(
            payload.path,
            payload.content,
            role=payload.role,
            media_type=payload.media_type,
            executable=payload.executable,
        )
        for payload in api_bundle_payloads(validation_points)
    )


__all__ = [
    "ApiBundlePayload",
    "ValidationPoint",
    "api_bundle_payloads",
    "emit_api_bundle",
    "format_validation_points",
]

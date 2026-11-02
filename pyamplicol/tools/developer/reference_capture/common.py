# SPDX-License-Identifier: 0BSD
"""Shared contracts and canonical serialization for reference capture."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal, cast

ROOT = Path(__file__).resolve().parents[3]
DEPENDENCIES = ROOT / "dependencies"
RELEASE_LOCK = DEPENDENCIES / "release-lock.toml"
CONTRIBUTOR_LOCK = DEPENDENCIES / "contributor-lock.toml"
INSTALL_STATE = DEPENDENCIES / "install-state.json"
MODEL_ASSETS = ROOT / "src" / "pyamplicol" / "assets" / "models" / "json"

PHYSICS_FILENAME = "physics-v2.json"
FORTRAN_EVIDENCE_FILENAME = "legacy-fortran-v2.json"
ANALYTIC_EVIDENCE_FILENAME = "analytic-oracles-v2.json"
BUNDLE_MANIFEST_FILENAME = "reference-fixture-v2.manifest.json"
FORTRAN_EVIDENCE_SET_ID = "evidence-set:legacy-fortran-amplicol"
ANALYTIC_EVIDENCE_SET_ID = "evidence-set:analytic-oracles"
WATCHDOG_GB = 30
OBSERVATION_PRECISION = 80
FORTRAN_CERTIFIED_DIGITS = 10
ANALYTIC_CERTIFIED_DIGITS = 12


class CaptureError(RuntimeError):
    """The capture cannot proceed without violating its provenance contract."""


Momentum = tuple[Decimal, Decimal, Decimal, Decimal]


@dataclass(frozen=True, slots=True)
class StressMetric:
    """One quantified singularity measure derived from captured momenta."""

    kind: Literal[
        "minimum-final-energy-fraction",
        "minimum-final-transverse-momentum-squared-fraction",
        "relative-excess-energy",
    ]
    value: Decimal

    def as_payload(self) -> dict[str, str]:
        return {"kind": self.kind, "value": canonical_decimal(self.value)}


@dataclass(frozen=True, slots=True)
class CapturePoint:
    """One exact wire-level phase-space input."""

    id: str
    process_id: str
    point_class: Literal["canonical", "generic", "stress"]
    algorithm_name: str
    algorithm_version: str
    rng: str | None
    seed: int | None
    sqrt_s: Decimal
    momenta: tuple[Momentum, ...]
    masses: tuple[Decimal, ...]
    arithmetic_precision_bits: int
    round_trip_decimal_digits: int
    certified_decimal_digits: int
    stress_metric: StressMetric | None

    def as_payload(self) -> dict[str, object]:
        return {
            "id": self.id,
            "process_id": self.process_id,
            "class": self.point_class,
            "algorithm": {
                "name": self.algorithm_name,
                "version": self.algorithm_version,
                "rng": self.rng,
                "seed": self.seed,
            },
            "sqrt_s": canonical_decimal(self.sqrt_s),
            "momenta": [
                [canonical_decimal(component) for component in momentum]
                for momentum in self.momenta
            ],
            "masses": [canonical_decimal(mass) for mass in self.masses],
            "arithmetic_precision_bits": self.arithmetic_precision_bits,
            "round_trip_decimal_digits": self.round_trip_decimal_digits,
            "certified_decimal_digits": self.certified_decimal_digits,
            "stress_metric": (
                None if self.stress_metric is None else self.stress_metric.as_payload()
            ),
        }


@dataclass(frozen=True, slots=True)
class ProcessCaptureSpec:
    id: str
    expression: str
    model_id: str
    expected_external_masses: tuple[str, ...]
    point_policy: Literal[
        "exact-2to1",
        "seeded-two-body",
        "one-massive-two-massless",
    ]
    point_seeds: tuple[int, int, int] = ()


@dataclass(frozen=True, slots=True)
class ArtifactCaptureSpec:
    id: str
    directory_name: str
    model_id: str
    model_source: Path | None
    color_accuracy: Literal["lc", "nlc", "full"]
    processes: tuple[ProcessCaptureSpec, ...]


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    repository_uri: str
    revision: str
    tree_sha256: str


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    """Identity of the installed candidate used for generation and replay."""

    version: str
    candidate_fingerprint: str
    source_revision: str
    distribution_sha256: str
    build_info_sha256: str


@dataclass(frozen=True, slots=True)
class DependencySnapshot:
    payloads: tuple[dict[str, object], ...]
    release_lock: Mapping[str, object]
    contributor_lock: Mapping[str, object]
    install_state: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    output_directory: Path
    artifact_root: Path
    legacy_repository: Path
    legacy_jobs: int
    artifact_mode: Literal["generate", "reuse"]
    capture_command: tuple[str, ...]
    external_watchdog_gb: int = WATCHDOG_GB


@dataclass(frozen=True, slots=True)
class CaptureResult:
    fixture_path: Path
    evidence_paths: tuple[Path, ...]
    bundle_manifest_path: Path
    artifact_paths: tuple[Path, ...]


def canonical_decimal(value: Decimal | float | int | str) -> str:
    """Return the schema's canonical fixed-point decimal spelling."""

    if isinstance(value, bool):
        raise CaptureError("boolean values are not decimal physics values")
    try:
        if isinstance(value, Decimal):
            decimal = value
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise CaptureError(f"non-finite binary64 value {value!r}")
            decimal = Decimal(format(value, ".17g"))
        elif isinstance(value, (int, str)):
            decimal = Decimal(value)
        else:
            raise CaptureError(f"unsupported decimal value {value!r}")
    except InvalidOperation as error:
        raise CaptureError(f"invalid decimal value {value!r}") from error
    if not decimal.is_finite():
        raise CaptureError(f"non-finite decimal value {value!r}")
    if decimal == 0:
        return "0"
    rendered = format(decimal, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def decimal_digits_to_bits(decimal_digits: int) -> int:
    """Return the smallest binary precision able to carry ``decimal_digits``."""

    if isinstance(decimal_digits, bool) or decimal_digits < 1:
        raise CaptureError("decimal precision must be a positive integer")
    numerator = decimal_digits * 1_000_000_000_000_000_000
    denominator = 301_029_995_663_981_195
    return (numerator + denominator - 1) // denominator


def as_mapping(value: object, where: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CaptureError(f"{where} must be an object")
    return cast(Mapping[str, object], value)


def as_sequence(value: object, where: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise CaptureError(f"{where} must be an array")
    return cast(Sequence[object], value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def developer_module(name: str) -> Any:
    root_text = str(ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return importlib.import_module(f"tools.developer.{name}")


def json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")

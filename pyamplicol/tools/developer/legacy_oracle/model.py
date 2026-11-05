# SPDX-License-Identifier: 0BSD
"""Shared paths, constants, and value objects for the legacy oracle."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LOCK = ROOT / "dependencies" / "contributor-lock.toml"
DEFAULT_REPOSITORY = ROOT / "dependencies" / "checkouts" / "legacy-amplicol"
DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "reference" / "physics-v2.json"
PINNED_REFERENCE_REVISION = "dcb3efbcbe89c51969818cddf48113fba19ad763"
PINNED_REFERENCE_BRANCH = "amplicol_with_patches"
PINNED_SOURCE_URL = "https://github.com/rikkert-frederix/AmpliCol.git"

FORTRAN_RTOL = 1.0e-8
FORTRAN_ATOL = 1.0e-15
FORTRAN_EVIDENCE_RTOL_DECIMAL = Decimal("0.0000000001")
FORTRAN_EVIDENCE_ATOL_DECIMAL = Decimal("0.000000000001")
FORTRAN_CERTIFIED_DECIMAL_DIGITS = 10

MAX_SUPPORTED_QUARK_LINES = 2
MAX_DIRECT_COLOR_PROBE_QUARK_LINES = 3
MAX_DIRECT_COLOR_PROBE_COLOR_ORDERS = 5_000


class LegacyOracleError(RuntimeError):
    """The independent Fortran oracle could not be prepared or evaluated."""


@dataclass(frozen=True)
class ProcessEntry:
    group: int
    integral: int
    process_pdgs: tuple[int, ...]
    color_order: tuple[int, ...]


@dataclass(frozen=True)
class LcRowPartition:
    row: int
    value: float
    permutation: tuple[int, ...]
    decimal_value: Decimal | None = field(default=None, compare=False)


@dataclass(frozen=True)
class CompilerProvenance:
    identity: str
    version: str
    flags: tuple[str, ...]
    target: str
    executable_sha256: str


@dataclass(frozen=True)
class ProbeResult:
    value: float
    components: tuple[float, float, float]
    currents: int
    vertices: int
    amplitudes: int
    color_orders: int
    lc_row_partitions: tuple[LcRowPartition, ...] = ()
    lc_partition_sum: float | None = None
    value_decimal: Decimal | None = field(default=None, compare=False)
    component_decimals: tuple[Decimal, ...] = field(default=(), compare=False)
    lc_partition_sum_decimal: Decimal | None = field(default=None, compare=False)


@dataclass(frozen=True)
class SelectedFlowProbeResult:
    """Normalized scalar and row metadata from a generated LC library."""

    value: float
    group: int
    integral: int
    process_pdgs: tuple[int, ...]
    color_order: tuple[int, ...]
    amplitudes: int
    color_factor: int
    identical_factor: int
    singlet_vertices: int
    normalization: float
    value_decimal: Decimal = field(compare=False)
    normalization_decimal: Decimal = field(compare=False)
    helicities: tuple[int, ...] = ()
    fixed_helicity_value: float | None = None
    fixed_helicity_value_decimal: Decimal | None = field(
        default=None,
        compare=False,
    )

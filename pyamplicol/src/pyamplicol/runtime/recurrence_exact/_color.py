# SPDX-License-Identifier: 0BSD
"""Authenticated compact color contraction for exact recurrence execution."""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path, PurePosixPath

from pyamplicol.api.errors import ArtifactError, CompatibilityError
from pyamplicol.artifacts.manifest import ArtifactManifest, PayloadRecord
from pyamplicol.artifacts.security import confined_path, normalize_relative_path
from pyamplicol.color.symmetric_group import (
    _inverse_permutation,
    _lexicographic_permutation_rank,
    _LexicographicPermutations,
    _relative_permutation,
)
from pyamplicol.runtime.eager_exact._contracts import _mapping

RECURRENCE_COLOR_CONTRACTION_CODEC_ABI = "pyamplicol-recurrence-color-contraction-v3"
RECURRENCE_CONTRACTED_COLOR_CAPABILITY = "rusticol.recurrence-color.contracted.v1"

_MAGIC = b"PACRCLR3"
_VERSION = 3
_STORAGE_EXPANDED = 1
_STORAGE_REPEATED = 2
_STORAGE_CONVOLUTION_KERNELS = 3
_ACCURACY = {1: "nlc", 2: "full"}
_FACTOR_NONE = 0
_FACTOR_KLEIN_FOUR = 1
_FACTOR_ELEMENTARY_ABELIAN = 2
_FACTOR_SYMMETRIC_GROUP_FOURIER = 3
_FLAG_INCLUDES_COLOR_FACTOR = 1 << 0
_KNOWN_FLAGS = _FLAG_INCLUDES_COLOR_FACTOR
_MAX_FACTOR_RANK = 31  # 2**rank must fit the u32 local-group count.
_ZERO_SECTOR_OWNER = 0xFFFF_FFFF
_HEADER = struct.Struct("<8s14I7Q")
_ENTRY = struct.Struct("<IIdddI")
_EXACT_FACTOR_BYTES = 64
_U32 = struct.Struct("<I")


@dataclass(frozen=True, slots=True)
class _RawColorEntry:
    left_group_id: int
    right_group_id: int
    weight_re: float
    weight_im: float
    symmetry_factor: float
    exact_factor_id: int


@dataclass(frozen=True, slots=True)
class _ExactColorFactor:
    real_numerator: int
    real_denominator: int
    imag_numerator: int
    imag_denominator: int

    def decimal(self) -> tuple[Decimal, Decimal]:
        return (
            Decimal(self.real_numerator) / Decimal(self.real_denominator),
            Decimal(self.imag_numerator) / Decimal(self.imag_denominator),
        )


@dataclass(frozen=True, slots=True)
class _ExactColorEntry:
    left_group_id: int
    right_group_id: int
    left_destination_id: int
    right_destination_id: int
    component_id: int
    coefficient_re: Decimal
    coefficient_im: Decimal


@dataclass(frozen=True, slots=True)
class _RecurrenceColorContraction:
    color_accuracy: str
    storage: str
    includes_color_factor: bool
    group_count: int
    sector_count: int
    component_count: int
    local_group_count: int
    destination_count: int
    entries: tuple[_RawColorEntry, ...]
    exact_factors: tuple[_ExactColorFactor, ...]
    ordered_group_ids: tuple[int, ...]
    destination_by_group: tuple[int, ...]
    group_sector_ids: tuple[int, ...]
    group_component_ids: tuple[int, ...]
    owner_by_sector: tuple[int, ...]
    logical_entry_count: int
    factorization_kind: str | None
    factorization_rank: int
    factorization_coset_count: int

    def runtime_entries(self) -> Iterator[_ExactColorEntry]:
        """Yield the symmetry-folded rows represented by the compact payload."""

        if self.storage == "expanded":
            for entry in self.entries:
                yield self._runtime_entry(
                    entry,
                    entry.left_group_id,
                    entry.right_group_id,
                )
            return

        group_by_local_component = [0] * self.group_count
        for local_group_id in range(self.local_group_count):
            start = local_group_id * self.component_count
            for group_id in self.ordered_group_ids[
                start : start + self.component_count
            ]:
                component_id = self.group_component_ids[group_id]
                group_by_local_component[
                    local_group_id * self.component_count + component_id
                ] = group_id
        if self.storage == "convolution-kernels":
            # Exact coefficients are retained in the FFT payload. Expand only
            # the iteration, not a dense matrix or a floating-point transform.
            for entry, left, right in self._convolution_entries():
                for component in range(self.component_count):
                    yield self._runtime_entry(
                        entry,
                        group_by_local_component[
                            left * self.component_count + component
                        ],
                        group_by_local_component[
                            right * self.component_count + component
                        ],
                    )
            return
        for component in range(self.component_count):
            for entry in self.entries:
                left_index = entry.left_group_id * self.component_count + component
                right_index = entry.right_group_id * self.component_count + component
                yield self._runtime_entry(
                    entry,
                    group_by_local_component[left_index],
                    group_by_local_component[right_index],
                )

    def _convolution_entries(self) -> Iterator[tuple[_RawColorEntry, int, int]]:
        permutations = _LexicographicPermutations(self.factorization_rank)
        order = len(permutations)
        offset = 0
        for left_channel in range(self.factorization_coset_count):
            for right_channel in range(left_channel, self.factorization_coset_count):
                for left_index, left in enumerate(permutations):
                    for right_index, right in enumerate(permutations):
                        relative = _lexicographic_permutation_rank(
                            _relative_permutation(left, right)
                        )
                        # Same-channel kernels have weight one and use every
                        # ordered pair. Cross-channel kernels already include
                        # weight two for their implicit Hermitian partner.
                        yield (
                            self.entries[offset + relative],
                            left_channel * order + left_index,
                            right_channel * order + right_index,
                        )
                offset += order
        for entry in self.entries[offset:]:
            yield entry, entry.left_group_id, entry.right_group_id

    def _runtime_entry(
        self,
        entry: _RawColorEntry,
        left_group_id: int,
        right_group_id: int,
    ) -> _ExactColorEntry:
        left_component = self.group_component_ids[left_group_id]
        right_component = self.group_component_ids[right_group_id]
        if left_component != right_component:
            raise ArtifactError(
                "recurrence color contraction mixes resolved components"
            )
        try:
            coefficient_re, coefficient_im = self.exact_factors[
                entry.exact_factor_id
            ].decimal()
        except IndexError as exc:
            raise ArtifactError(
                "recurrence color entry references an unknown exact coefficient"
            ) from exc
        return _ExactColorEntry(
            left_group_id=left_group_id,
            right_group_id=right_group_id,
            left_destination_id=self.destination_by_group[left_group_id],
            right_destination_id=self.destination_by_group[right_group_id],
            component_id=left_component,
            coefficient_re=coefficient_re,
            coefficient_im=coefficient_im,
        )


def _load_recurrence_color_contraction(
    *,
    artifact_root: Path,
    process_id: str,
    execution_path: str,
    execution: Mapping[str, object],
    manifest: ArtifactManifest,
) -> _RecurrenceColorContraction | None:
    metadata = _mapping(execution.get("runtime_metadata"), "runtime metadata")
    raw_reference = metadata.get("color_contraction")
    if raw_reference is None:
        return None
    reference = _mapping(raw_reference, "recurrence color-contraction reference")
    if reference.get("abi") != RECURRENCE_COLOR_CONTRACTION_CODEC_ABI:
        raise CompatibilityError(
            "unsupported recurrence color-contraction payload contract"
        )
    if reference.get("path") != "recurrence-color.bin":
        raise CompatibilityError(
            "unsupported recurrence color-contraction payload path"
        )

    logical_path = normalize_relative_path(
        (
            PurePosixPath(normalize_relative_path(execution_path)).parent
            / "recurrence-color.bin"
        ).as_posix()
    )
    record = _payload_record(manifest, logical_path, process_id)
    declared_size = _nonnegative_integer(
        reference.get("size_bytes"), "color-contraction payload size"
    )
    declared_sha = _digest(reference.get("sha256"), "color-contraction payload SHA-256")
    semantic_digest = _digest(
        reference.get("semantic_digest"),
        "color-contraction semantic digest",
    )
    if (
        record.role != "evaluator-state"
        or record.media_type != "application/octet-stream"
        or record.process_id != process_id
        or record.size_bytes != declared_size
        or record.sha256 != declared_sha
        or semantic_digest != declared_sha
    ):
        raise ArtifactError(
            "recurrence color-contraction payload disagrees with execution metadata"
        )
    try:
        payload = confined_path(artifact_root, logical_path).read_bytes()
    except OSError as exc:
        raise ArtifactError(
            f"could not read recurrence color-contraction payload: {exc}"
        ) from exc
    if (
        len(payload) != declared_size
        or hashlib.sha256(payload).hexdigest() != declared_sha
    ):
        raise ArtifactError(
            "recurrence color-contraction payload authentication failed"
        )

    contraction = _decode_recurrence_color_contraction(payload)
    expected = {
        "color_accuracy": contraction.color_accuracy,
        "storage": contraction.storage,
        "includes_color_factor": contraction.includes_color_factor,
        "group_count": contraction.group_count,
        "sector_count": contraction.sector_count,
        "active_sector_count": sum(
            owner == sector for sector, owner in enumerate(contraction.owner_by_sector)
        ),
        "component_count": contraction.component_count,
        "destination_count": contraction.destination_count,
        "entry_count": len(contraction.entries),
        "logical_entry_count": contraction.logical_entry_count,
    }
    for name, value in expected.items():
        if reference.get(name) != value:
            raise ArtifactError(
                "recurrence color-contraction payload disagrees with its "
                f"bounded summary field {name!r}"
            )
    factorization = reference.get("factorization")
    expected_factorization: object
    if contraction.factorization_kind is None:
        expected_factorization = None
    else:
        expected_factorization = {
            "kind": contraction.factorization_kind,
            "rank": contraction.factorization_rank,
            "coset_count": contraction.factorization_coset_count,
        }
    if factorization != expected_factorization:
        raise ArtifactError(
            "recurrence color-contraction factorization disagrees with its "
            "bounded summary"
        )
    if not contraction.includes_color_factor:
        raise ArtifactError(
            "NLC/full recurrence color contraction must include the color factor"
        )
    return contraction


def _decode_recurrence_color_contraction(
    payload: bytes,
) -> _RecurrenceColorContraction:
    if len(payload) < _HEADER.size:
        raise ArtifactError("recurrence color-contraction payload is truncated")
    (
        magic,
        version,
        header_size,
        storage_id,
        accuracy_id,
        flags,
        group_count,
        sector_count,
        component_count,
        local_group_count,
        destination_count,
        factor_kind,
        factor_rank,
        entry_stride,
        exact_factor_stride,
        entry_count,
        exact_factor_count,
        coset_count,
        coset_index_count,
        logical_entry_count,
        owner_map_count,
        payload_size,
    ) = _HEADER.unpack_from(payload)
    if magic != _MAGIC or version != _VERSION or header_size != _HEADER.size:
        raise CompatibilityError(
            "unsupported recurrence color-contraction binary header"
        )
    if entry_stride != _ENTRY.size or exact_factor_stride != _EXACT_FACTOR_BYTES:
        raise ArtifactError(
            "recurrence color-contraction fixed-width header is inconsistent"
        )
    if flags & ~_KNOWN_FLAGS:
        raise ArtifactError("recurrence color-contraction payload has unknown flags")
    try:
        accuracy = _ACCURACY[accuracy_id]
    except KeyError as exc:
        raise ArtifactError(
            "recurrence color-contraction payload has invalid color accuracy"
        ) from exc
    if storage_id not in {
        _STORAGE_EXPANDED,
        _STORAGE_REPEATED,
        _STORAGE_CONVOLUTION_KERNELS,
    }:
        raise ArtifactError("recurrence color-contraction payload has invalid storage")
    storage = {
        _STORAGE_EXPANDED: "expanded",
        _STORAGE_REPEATED: "repeated",
        _STORAGE_CONVOLUTION_KERNELS: "convolution-kernels",
    }[storage_id]
    if min(group_count, sector_count, component_count, destination_count) <= 0:
        raise ArtifactError("recurrence color-contraction dimensions must be positive")
    if owner_map_count != sector_count:
        raise ArtifactError(
            "recurrence color-contraction owner map count disagrees with sector_count"
        )
    expected_payload_size = (
        entry_count * _ENTRY.size
        + exact_factor_count * _EXACT_FACTOR_BYTES
        + 4 * group_count * _U32.size
        + owner_map_count * _U32.size
        + coset_index_count * _U32.size
    )
    if (
        payload_size != expected_payload_size
        or len(payload) != _HEADER.size + payload_size
    ):
        raise ArtifactError("recurrence color-contraction payload size is inconsistent")

    entry_domain = group_count if storage == "expanded" else local_group_count
    entries = []
    seen_pairs = set()
    offset = _HEADER.size
    for index in range(entry_count):
        raw = _RawColorEntry(*_ENTRY.unpack_from(payload, offset))
        offset += _ENTRY.size
        pair = (raw.left_group_id, raw.right_group_id)
        if (
            raw.left_group_id >= entry_domain
            or raw.right_group_id >= entry_domain
            or raw.left_group_id > raw.right_group_id
            or pair in seen_pairs
        ):
            raise ArtifactError(
                f"recurrence color-contraction entry {index} is not canonical"
            )
        seen_pairs.add(pair)
        if not all(
            math.isfinite(value)
            for value in (raw.weight_re, raw.weight_im, raw.symmetry_factor)
        ) or not all(
            math.isfinite(raw.symmetry_factor * value)
            for value in (raw.weight_re, raw.weight_im)
        ):
            raise ArtifactError(
                f"recurrence color-contraction entry {index} is not finite"
            )
        entries.append(raw)

    exact_factors = []
    for index in range(exact_factor_count):
        values = []
        for _ in range(4):
            stop = offset + 16
            if stop > len(payload):
                raise ArtifactError(
                    "recurrence color-contraction exact factor catalog is truncated"
                )
            values.append(
                int.from_bytes(payload[offset:stop], byteorder="little", signed=True)
            )
            offset = stop
        factor = _ExactColorFactor(*values)
        _validate_exact_factor(factor, index)
        exact_factors.append(factor)
    if any(entry.exact_factor_id >= len(exact_factors) for entry in entries):
        raise ArtifactError(
            "recurrence color entry references an out-of-bounds exact coefficient"
        )
    for index, entry in enumerate(entries):
        factor = exact_factors[entry.exact_factor_id]
        expected_re = factor.real_numerator / factor.real_denominator
        expected_im = factor.imag_numerator / factor.imag_denominator
        for component, actual, expected in (
            (
                "real",
                entry.weight_re * entry.symmetry_factor,
                expected_re,
            ),
            (
                "imaginary",
                entry.weight_im * entry.symmetry_factor,
                expected_im,
            ),
        ):
            if not math.isclose(
                actual,
                expected,
                rel_tol=0.0,
                abs_tol=max(math.ulp(actual), math.ulp(expected)),
            ):
                raise ArtifactError(
                    f"recurrence color entry {index} {component} f64 coefficient "
                    "disagrees with its exact factor"
                )

    ordered_groups, offset = _read_u32_array(
        payload, offset, group_count, "ordered group map"
    )
    destinations, offset = _read_u32_array(
        payload, offset, group_count, "destination map"
    )
    group_sector_ids, offset = _read_u32_array(
        payload, offset, group_count, "group sector map"
    )
    group_component_ids, offset = _read_u32_array(
        payload, offset, group_count, "group component map"
    )
    owner_by_sector, offset = _read_u32_array(
        payload, offset, owner_map_count, "physical sector owner map"
    )
    coset_indices, offset = _read_u32_array(
        payload, offset, coset_index_count, "factorization cosets"
    )
    if offset != len(payload):
        raise ArtifactError(
            "recurrence color-contraction payload contains trailing bytes"
        )
    _validate_permutation(ordered_groups, group_count, "ordered group map")
    if len(set(destinations)) != len(destinations) or any(
        value >= destination_count for value in destinations
    ):
        raise ArtifactError(
            "recurrence color-contraction destination map is not injective"
        )
    _validate_group_coordinates(
        group_count=group_count,
        sector_count=sector_count,
        component_count=component_count,
        group_sector_ids=group_sector_ids,
        group_component_ids=group_component_ids,
    )
    _validate_sector_owners(
        sector_count=sector_count,
        owner_by_sector=owner_by_sector,
        active_sector_ids=set(group_sector_ids),
    )

    factorization_kind = None
    if storage == "expanded":
        if (
            local_group_count != 0
            or factor_kind != _FACTOR_NONE
            or factor_rank != 0
            or coset_count != 0
            or coset_indices
            or logical_entry_count != entry_count
        ):
            raise ArtifactError(
                "expanded recurrence color storage carries repeated metadata"
            )
        if any(
            group_component_ids[entry.left_group_id]
            != group_component_ids[entry.right_group_id]
            for entry in entries
        ):
            raise ArtifactError(
                "expanded recurrence color entry mixes helicity components"
            )
    else:
        if (
            (component_count < 2 and storage == "repeated")
            or local_group_count * component_count != group_count
            or logical_entry_count != entry_count * component_count
        ):
            raise ArtifactError("repeated recurrence color dimensions are inconsistent")
        _validate_repeated_group_coordinates(
            ordered_group_ids=ordered_groups,
            group_sector_ids=group_sector_ids,
            group_component_ids=group_component_ids,
            sector_count=sector_count,
            component_count=component_count,
            local_group_count=local_group_count,
        )
        if storage == "convolution-kernels":
            _validate_symmetric_group_convolution(
                factor_kind,
                factor_rank,
                coset_count,
                coset_indices,
                local_group_count,
                entries,
                exact_factors,
            )
            factorization_kind = "symmetric-group-fourier"
        else:
            _validate_factorization(
                factor_kind=factor_kind,
                factor_rank=factor_rank,
                coset_count=coset_count,
                coset_indices=coset_indices,
                local_group_count=local_group_count,
                entries=entries,
            )
            factorization_kind = (
                "klein-four-walsh"
                if factor_kind == _FACTOR_KLEIN_FOUR
                else "elementary-abelian-walsh"
                if factor_kind == _FACTOR_ELEMENTARY_ABELIAN
                else None
            )

    return _RecurrenceColorContraction(
        color_accuracy=accuracy,
        storage=storage,
        includes_color_factor=bool(flags & _FLAG_INCLUDES_COLOR_FACTOR),
        group_count=group_count,
        sector_count=sector_count,
        component_count=component_count,
        local_group_count=local_group_count,
        destination_count=destination_count,
        entries=tuple(entries),
        exact_factors=tuple(exact_factors),
        ordered_group_ids=ordered_groups,
        destination_by_group=destinations,
        group_sector_ids=group_sector_ids,
        group_component_ids=group_component_ids,
        owner_by_sector=owner_by_sector,
        logical_entry_count=logical_entry_count,
        factorization_kind=factorization_kind,
        factorization_rank=factor_rank,
        factorization_coset_count=coset_count,
    )


def _contract_color_amplitudes(
    contraction: _RecurrenceColorContraction,
    amplitudes: Sequence[tuple[Decimal, Decimal]],
    destination_physics_helicity: Sequence[int],
    selected_helicities: set[int] | None = None,
) -> dict[int, Decimal]:
    if (
        len(amplitudes) != contraction.destination_count
        or len(destination_physics_helicity) != contraction.destination_count
    ):
        raise ArtifactError(
            "recurrence color contraction does not match amplitude destinations"
        )
    result: dict[int, Decimal] = {}
    physics_helicity_by_component: dict[int, int] = {}
    for entry in contraction.runtime_entries():
        left_helicity = destination_physics_helicity[entry.left_destination_id]
        right_helicity = destination_physics_helicity[entry.right_destination_id]
        if left_helicity != right_helicity:
            raise ArtifactError("recurrence color contraction mixes public helicities")
        known_helicity = physics_helicity_by_component.setdefault(
            entry.component_id, left_helicity
        )
        if known_helicity != left_helicity:
            raise ArtifactError(
                "recurrence color component maps to inconsistent public helicities"
            )
        if selected_helicities is not None and left_helicity not in selected_helicities:
            continue
        left_re, left_im = amplitudes[entry.left_destination_id]
        right_re, right_im = amplitudes[entry.right_destination_id]
        product_re = left_re * right_re + left_im * right_im
        product_im = left_im * right_re - left_re * right_im
        value = entry.coefficient_re * product_re - entry.coefficient_im * product_im
        result[left_helicity] = result.get(left_helicity, Decimal(0)) + value
    return result


def _payload_record(
    manifest: ArtifactManifest,
    path: str,
    process_id: str,
) -> PayloadRecord:
    matches = tuple(record for record in manifest.payloads if record.path == path)
    if len(matches) != 1:
        raise ArtifactError(
            "recurrence color-contraction payload is not uniquely declared"
        )
    record = matches[0]
    if record.process_id != process_id:
        raise ArtifactError(
            "recurrence color-contraction payload selects the wrong process"
        )
    return record


def _read_u32_array(
    payload: bytes,
    offset: int,
    count: int,
    label: str,
) -> tuple[tuple[int, ...], int]:
    stop = offset + count * _U32.size
    if stop > len(payload):
        raise ArtifactError(f"recurrence color-contraction {label} is truncated")
    return (
        tuple(
            _U32.unpack_from(payload, offset + index * _U32.size)[0]
            for index in range(count)
        ),
        stop,
    )


def _validate_permutation(values: Sequence[int], count: int, label: str) -> None:
    if len(values) != count or sorted(values) != list(range(count)):
        raise ArtifactError(
            f"recurrence color-contraction {label} is not a complete permutation"
        )


def _validate_group_coordinates(
    *,
    group_count: int,
    sector_count: int,
    component_count: int,
    group_sector_ids: Sequence[int],
    group_component_ids: Sequence[int],
) -> None:
    if (
        len(group_sector_ids) != group_count
        or len(group_component_ids) != group_count
        or any(value >= sector_count for value in group_sector_ids)
        or any(value >= component_count for value in group_component_ids)
    ):
        raise ArtifactError(
            "recurrence color-contraction group coordinates are out of bounds"
        )
    coordinates = tuple(zip(group_sector_ids, group_component_ids, strict=True))
    if len(set(coordinates)) != group_count:
        raise ArtifactError(
            "recurrence color-contraction group coordinates are not unique"
        )


def _validate_exact_factor(factor: _ExactColorFactor, index: int) -> None:
    for label, numerator, denominator in (
        ("real", factor.real_numerator, factor.real_denominator),
        ("imaginary", factor.imag_numerator, factor.imag_denominator),
    ):
        if denominator <= 0:
            raise ArtifactError(
                f"recurrence exact color factor {index} has a non-positive "
                f"{label} denominator"
            )
        if numerator == 0 and denominator != 1:
            raise ArtifactError(
                f"recurrence exact color factor {index} has a non-canonical "
                f"zero {label} component"
            )
        if math.gcd(abs(numerator), denominator) != 1:
            raise ArtifactError(
                f"recurrence exact color factor {index} has a reducible "
                f"{label} component"
            )


def _validate_sector_owners(
    *,
    sector_count: int,
    owner_by_sector: Sequence[int],
    active_sector_ids: set[int],
) -> None:
    if len(owner_by_sector) != sector_count:
        raise ArtifactError("recurrence physical-sector owner map has the wrong length")
    fixed_points: set[int] = set()
    for sector_id, owner_id in enumerate(owner_by_sector):
        if owner_id == _ZERO_SECTOR_OWNER:
            continue
        if (
            owner_id >= sector_count
            or owner_id > sector_id
            or owner_by_sector[owner_id] != owner_id
        ):
            raise ArtifactError(
                f"recurrence physical sector {sector_id} has an invalid owner"
            )
        if owner_id == sector_id:
            fixed_points.add(sector_id)
    if fixed_points != active_sector_ids:
        raise ArtifactError(
            "recurrence active sectors do not match the authenticated owner map"
        )


def _validate_repeated_group_coordinates(
    *,
    ordered_group_ids: Sequence[int],
    group_sector_ids: Sequence[int],
    group_component_ids: Sequence[int],
    sector_count: int,
    component_count: int,
    local_group_count: int,
) -> None:
    expected_components = set(range(component_count))
    row_sectors: list[int] = []
    for local_group_id in range(local_group_count):
        start = local_group_id * component_count
        row = ordered_group_ids[start : start + component_count]
        sectors = {group_sector_ids[group_id] for group_id in row}
        components = {group_component_ids[group_id] for group_id in row}
        if len(sectors) != 1 or components != expected_components:
            raise ArtifactError(
                "repeated recurrence color row does not cover one physical "
                "sector and every component exactly once"
            )
        row_sectors.append(next(iter(sectors)))
    if len(set(row_sectors)) != len(row_sectors) or any(
        sector_id >= sector_count for sector_id in row_sectors
    ):
        raise ArtifactError(
            "repeated recurrence color rows do not identify unique physical sectors"
        )


def _validate_symmetric_group_convolution(
    factor_kind: int,
    degree: int,
    channel_count: int,
    channel_indices: Sequence[int],
    local_group_count: int,
    entries: Sequence[_RawColorEntry],
    factors: Sequence[_ExactColorFactor],
) -> None:
    if factor_kind != _FACTOR_SYMMETRIC_GROUP_FOURIER or not 2 <= degree <= 10:
        raise ArtifactError(
            "convolution color storage requires symmetric-group metadata"
        )
    order = math.factorial(degree)
    eligible = channel_count * order
    if (
        channel_count <= 0
        or eligible > local_group_count
        or tuple(channel_indices) != tuple(range(eligible))
    ):
        raise ArtifactError("symmetric-group channel coordinates are inconsistent")
    kernel_count = channel_count * (channel_count + 1) // 2 * order
    residual_count = (
        local_group_count * (local_group_count + 1) - eligible * (eligible + 1)
    ) // 2
    if len(entries) != kernel_count + residual_count:
        raise ArtifactError("symmetric-group kernel/residual row count is inconsistent")
    offset = 0
    for left in range(channel_count):
        for right in range(left, channel_count):
            for relative, permutation in enumerate(_LexicographicPermutations(degree)):
                entry = entries[offset + relative]
                if (entry.left_group_id, entry.right_group_id) != (
                    left * order,
                    right * order + relative,
                ) or entry.symmetry_factor != (1.0 if left == right else 2.0):
                    raise ArtifactError("symmetric-group kernel rows are not canonical")
                if left == right:
                    inverse = _lexicographic_permutation_rank(
                        _inverse_permutation(permutation)
                    )
                    if (
                        factors[entry.exact_factor_id]
                        != factors[entries[offset + inverse].exact_factor_id]
                    ):
                        raise ArtifactError(
                            "symmetric-group diagonal kernel is not Hermitian"
                        )
            offset += order
    for left in range(local_group_count):
        for right in range(max(left, eligible), local_group_count):
            entry = entries[offset]
            if (entry.left_group_id, entry.right_group_id) != (
                left,
                right,
            ) or entry.symmetry_factor != (1.0 if left == right else 2.0):
                raise ArtifactError("symmetric-group residual rows are not canonical")
            offset += 1
    if any(factors[entry.exact_factor_id].imag_numerator != 0 for entry in entries):
        raise ArtifactError("symmetric-group color kernels require real coefficients")


def _validate_factorization(
    *,
    factor_kind: int,
    factor_rank: int,
    coset_count: int,
    coset_indices: Sequence[int],
    local_group_count: int,
    entries: Sequence[_RawColorEntry],
) -> None:
    if factor_kind == _FACTOR_NONE:
        if factor_rank != 0 or coset_count != 0 or coset_indices:
            raise ArtifactError("recurrence color factorization-none carries metadata")
        return
    if factor_kind == _FACTOR_KLEIN_FOUR:
        if factor_rank != 2:
            raise ArtifactError("Klein-four color factorization must have rank two")
    elif factor_kind == _FACTOR_ELEMENTARY_ABELIAN:
        if not 3 <= factor_rank <= _MAX_FACTOR_RANK:
            raise ArtifactError(
                "elementary-Abelian color factorization has invalid rank"
            )
    else:
        raise ArtifactError("recurrence color factorization kind is unsupported")
    subgroup_order = 1 << factor_rank
    if (
        coset_count <= 0
        or coset_count * subgroup_order != len(coset_indices)
        or len(coset_indices) != local_group_count
    ):
        raise ArtifactError(
            "recurrence color factorization coset shape is inconsistent"
        )
    _validate_permutation(coset_indices, local_group_count, "factorization cosets")
    matrix = {}
    for entry in entries:
        if entry.weight_im != 0.0:
            raise ArtifactError(
                "factorized recurrence color contraction requires real weights"
            )
        coefficient = entry.weight_re * entry.symmetry_factor
        if entry.left_group_id != entry.right_group_id:
            coefficient *= 0.5
        matrix[(entry.left_group_id, entry.right_group_id)] = coefficient

    def matrix_value(left: int, right: int) -> float:
        return matrix.get((min(left, right), max(left, right)), 0.0)

    for left_coset_index in range(coset_count):
        left_start = left_coset_index * subgroup_order
        left_coset = coset_indices[left_start : left_start + subgroup_order]
        for right_coset_index in range(coset_count):
            right_start = right_coset_index * subgroup_order
            right_coset = coset_indices[right_start : right_start + subgroup_order]
            for left_index in range(subgroup_order):
                for right_index in range(subgroup_order):
                    if matrix_value(
                        left_coset[left_index],
                        right_coset[right_index],
                    ) != matrix_value(
                        left_coset[0],
                        right_coset[left_index ^ right_index],
                    ):
                        raise ArtifactError(
                            "recurrence color factorization is not Walsh-invariant"
                        )


def _digest(value: object, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ArtifactError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _nonnegative_integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ArtifactError(f"{context} must be a non-negative integer")
    return value


__all__ = [
    "RECURRENCE_CONTRACTED_COLOR_CAPABILITY",
    "_RecurrenceColorContraction",
    "_contract_color_amplitudes",
    "_decode_recurrence_color_contraction",
    "_load_recurrence_color_contraction",
]

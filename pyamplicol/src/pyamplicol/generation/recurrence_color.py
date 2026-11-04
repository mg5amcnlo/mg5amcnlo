# SPDX-License-Identifier: 0BSD
"""Bounded binary codec for process-owned recurrence color contraction."""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from hashlib import sha256

from pyamplicol.color import (
    ColorContractionEntry,
    ColorContractionPlan,
    ColorContractionTemplateEntry,
    FactorizedColorContractionBlock,
    SymmetricGroupColorContractionBlock,
)
from pyamplicol.generation.recurrence_columnar import ExactComplexRationalV1

RECURRENCE_COLOR_CONTRACTION_CODEC_ABI = "pyamplicol-recurrence-color-contraction-v3"
RECURRENCE_COLOR_CONTRACTION_MAGIC = b"PACRCLR3"

_VERSION = 3
_STORAGE_EXPANDED = 1
_STORAGE_REPEATED = 2
_STORAGE_CONVOLUTION_KERNELS = 3
_ACCURACY_NLC = 1
_ACCURACY_FULL = 2
_FACTOR_NONE = 0
_FACTOR_KLEIN_FOUR = 1
_FACTOR_ELEMENTARY_ABELIAN = 2
_FACTOR_SYMMETRIC_GROUP_FOURIER = 3
_FLAG_INCLUDES_COLOR_FACTOR = 1 << 0
_KNOWN_FLAGS = _FLAG_INCLUDES_COLOR_FACTOR
_MAX_FACTOR_RANK = 31  # 2**rank must fit the u32 local-group count.
_MAX_SYMMETRIC_GROUP_DEGREE = 10
_ZERO_SECTOR_OWNER = 0xFFFF_FFFF

# magic; 14 u32 fields; 7 u64 fields
_HEADER = struct.Struct("<8s14I7Q")
# left/right group or local-group IDs; raw complex weight; raw symmetry factor;
# exact, symmetry-folded coefficient catalog ID.
_ENTRY = struct.Struct("<IIdddI")
_EXACT_FACTOR_BYTES = 64
_U32 = struct.Struct("<I")


class RecurrenceColorCodecError(ValueError):
    """The color-contraction plan cannot be represented by the fixed ABI."""


def encode_recurrence_color_contraction(
    plan: ColorContractionPlan,
    *,
    sector_count: int,
    component_count: int,
    ordered_group_ids: Sequence[int],
    destination_by_group: Sequence[int],
    group_sector_ids: Sequence[int],
    group_component_ids: Sequence[int],
    sector_owner_ids: Sequence[int],
    exact_coefficients: Sequence[ExactComplexRationalV1],
    destination_count: int,
) -> bytes:
    """Encode one supported NLC/full color-contraction plan.

    The caller owns payload hashing and container authentication. The codec
    deliberately preserves raw weights and symmetry factors as distinct f64
    fields while binding every row to a canonical exact coefficient.

    >>> plan = ColorContractionPlan(
    ...     color_accuracy="nlc",
    ...     supported=True,
    ...     reason=None,
    ...     group_count=2,
    ...     entries=(
    ...         ColorContractionEntry(0, 0, 3.0),
    ...         ColorContractionEntry(0, 1, 2.0, symmetry_factor=2.0),
    ...     ),
    ... )
    >>> payload = encode_recurrence_color_contraction(
    ...     plan,
    ...     sector_count=2,
    ...     component_count=1,
    ...     ordered_group_ids=(0, 1),
    ...     destination_by_group=(7, 9),
    ...     group_sector_ids=(0, 1),
    ...     group_component_ids=(0, 0),
    ...     sector_owner_ids=(0, 1),
    ...     exact_coefficients=(
    ...         ExactComplexRationalV1(3),
    ...         ExactComplexRationalV1(4),
    ...     ),
    ...     destination_count=10,
    ... )
    >>> payload[:8] == RECURRENCE_COLOR_CONTRACTION_MAGIC
    True
    >>> len(payload) == (
    ...     _HEADER.size
    ...     + 2 * _ENTRY.size
    ...     + 2 * _EXACT_FACTOR_BYTES
    ...     + 10 * _U32.size
    ... )
    True
    """

    if not isinstance(plan, ColorContractionPlan):
        raise RecurrenceColorCodecError(
            "recurrence color payload requires a ColorContractionPlan"
        )
    if not plan.supported:
        raise RecurrenceColorCodecError(
            "unsupported color-contraction plans cannot be encoded"
        )
    if plan.reason is not None:
        raise RecurrenceColorCodecError(
            "a supported color-contraction plan cannot carry a failure reason"
        )
    accuracy = _encode_accuracy(plan.color_accuracy)
    group_count = _checked_u32("group_count", plan.group_count)
    if group_count == 0:
        raise RecurrenceColorCodecError("group_count must be positive")
    sector_count = _checked_u32("sector_count", sector_count)
    component_count = _checked_u32("component_count", component_count)
    if sector_count == 0 or component_count == 0:
        raise RecurrenceColorCodecError(
            "sector_count and component_count must be positive"
        )
    ordered_groups = tuple(
        _checked_u32("ordered group ID", value) for value in ordered_group_ids
    )
    destinations = tuple(
        _checked_u32("Direct-Arena destination ID", value)
        for value in destination_by_group
    )
    sectors = tuple(
        _checked_u32("group sector ID", value) for value in group_sector_ids
    )
    components = tuple(
        _checked_u32("group component ID", value) for value in group_component_ids
    )
    owners = tuple(
        _checked_u32("physical sector owner ID", value) for value in sector_owner_ids
    )
    destination_count = _checked_u32("destination_count", destination_count)
    if destination_count == 0:
        raise RecurrenceColorCodecError("destination_count must be positive")
    if len(ordered_groups) != group_count:
        raise RecurrenceColorCodecError(
            "ordered group map count does not match group_count"
        )
    if sorted(ordered_groups) != list(range(group_count)):
        raise RecurrenceColorCodecError(
            "ordered group map is not a permutation of all group IDs"
        )
    if len(destinations) != group_count:
        raise RecurrenceColorCodecError(
            "destination map count does not match group_count"
        )
    if len(set(destinations)) != len(destinations):
        raise RecurrenceColorCodecError(
            "destination map contains duplicate Direct-Arena destinations"
        )
    if any(value >= destination_count for value in destinations):
        raise RecurrenceColorCodecError(
            "destination map references an out-of-bounds Direct-Arena destination"
        )
    _validate_group_coordinates(
        group_count=group_count,
        sector_count=sector_count,
        component_count=component_count,
        group_sector_ids=sectors,
        group_component_ids=components,
    )
    _validate_sector_owners(
        sector_count=sector_count,
        owner_by_sector=owners,
        active_sector_ids=set(sectors),
    )

    flags = _FLAG_INCLUDES_COLOR_FACTOR if plan.includes_color_factor else 0
    if flags & ~_KNOWN_FLAGS:
        raise AssertionError("internal recurrence color flag drift")

    repeated = plan.repeated_block
    symmetric_group = plan.symmetric_group_block
    if repeated is None and symmetric_group is None:
        storage = _STORAGE_EXPANDED
        local_group_count = 0
        entries: Sequence[ColorContractionEntry | ColorContractionTemplateEntry] = (
            plan.entries
        )
        factor_kind = _FACTOR_NONE
        factor_rank = 0
        cosets: tuple[tuple[int, ...], ...] = ()
        _validate_entries(entries, group_count=group_count, label="expanded")
        _validate_expanded_components(
            entries,
            group_component_ids=components,
        )
        logical_entry_count = len(entries)
    elif repeated is not None:
        if plan.entries:
            raise RecurrenceColorCodecError(
                "expanded and repeated color-contraction storage cannot be mixed"
            )
        storage = _STORAGE_REPEATED
        repeated_component_count = _checked_u32(
            "repeated component_count", repeated.component_count
        )
        if repeated_component_count < 2:
            raise RecurrenceColorCodecError(
                "repeated storage requires at least two components"
            )
        if repeated_component_count != component_count:
            raise RecurrenceColorCodecError(
                "repeated component_count does not match the explicit component_count"
            )
        repeated_group_map = tuple(
            _checked_u32("repeated component group ID", value)
            for value in repeated.component_group_ids
        )
        if len(repeated_group_map) != group_count:
            raise RecurrenceColorCodecError(
                "repeated group map count does not match group_count"
            )
        if repeated_group_map != ordered_groups:
            raise RecurrenceColorCodecError(
                "ordered group map does not match repeated local-color-major/"
                "component-minor storage"
            )
        if len(set(repeated_group_map)) != len(repeated_group_map):
            raise RecurrenceColorCodecError(
                "repeated group map contains duplicate group IDs"
            )
        if any(value >= group_count for value in repeated_group_map):
            raise RecurrenceColorCodecError(
                "repeated group map references an out-of-bounds group"
            )
        if len(repeated_group_map) % component_count:
            raise RecurrenceColorCodecError("repeated group map is not rectangular")
        local_group_count = len(repeated_group_map) // component_count
        if local_group_count == 0:
            raise RecurrenceColorCodecError(
                "repeated storage requires at least one local color group"
            )
        _validate_repeated_group_coordinates(
            ordered_group_ids=ordered_groups,
            group_sector_ids=sectors,
            group_component_ids=components,
            sector_count=sector_count,
            component_count=component_count,
            local_group_count=local_group_count,
        )
        entries = repeated.entries
        _validate_entries(
            entries,
            group_count=local_group_count,
            label="repeated",
        )
        factor_kind, factor_rank, cosets = _encode_factorization(
            repeated.factorized_block,
            local_group_count=local_group_count,
            entries=repeated.entries,
        )
        logical_entry_count = _checked_product(
            "logical entry count", component_count, len(entries)
        )
    else:
        if symmetric_group is None:
            raise AssertionError("unreachable symmetric-group storage branch")
        if plan.entries or repeated is not None:
            raise RecurrenceColorCodecError(
                "convolution-kernel storage cannot be mixed with expanded or "
                "repeated color contraction"
            )
        storage = _STORAGE_CONVOLUTION_KERNELS
        if symmetric_group.component_count != component_count:
            raise RecurrenceColorCodecError(
                "symmetric-group component_count does not match the explicit "
                "component_count"
            )
        symmetric_group_map = tuple(
            _checked_u32("symmetric-group component group ID", value)
            for value in symmetric_group.component_group_ids
        )
        if symmetric_group_map != ordered_groups:
            raise RecurrenceColorCodecError(
                "ordered group map does not match symmetric-group local-color-major/"
                "component-minor storage"
            )
        if len(symmetric_group_map) % component_count:
            raise RecurrenceColorCodecError(
                "symmetric-group component map is not rectangular"
            )
        local_group_count = len(symmetric_group_map) // component_count
        if local_group_count == 0:
            raise RecurrenceColorCodecError(
                "convolution-kernel storage requires at least one local color group"
            )
        if _checked_u32("symmetric-group local_group_count", local_group_count) != (
            symmetric_group.local_group_count
        ):
            raise RecurrenceColorCodecError(
                "symmetric-group local group count is inconsistent"
            )
        _validate_repeated_group_coordinates(
            ordered_group_ids=ordered_groups,
            group_sector_ids=sectors,
            group_component_ids=components,
            sector_count=sector_count,
            component_count=component_count,
            local_group_count=local_group_count,
        )
        local_sector_ids = tuple(
            sectors[symmetric_group_map[index * component_count]]
            for index in range(local_group_count)
        )
        if local_sector_ids != symmetric_group.local_sector_ids:
            raise RecurrenceColorCodecError(
                "symmetric-group local sector order disagrees with the group map"
            )
        entries = (*symmetric_group.kernel_entries, *symmetric_group.residual_entries)
        _validate_entries(
            entries,
            group_count=local_group_count,
            label="convolution-kernel",
        )
        factor_kind, factor_rank, cosets = _encode_symmetric_group_convolution(
            symmetric_group,
            local_group_count=local_group_count,
            entries=entries,
        )
        expected_exact_coefficients = _symmetric_group_exact_coefficients(
            symmetric_group
        )
        if tuple(exact_coefficients) != expected_exact_coefficients:
            raise RecurrenceColorCodecError(
                "symmetric-group exact color coefficients disagree with the "
                "certified convolution plan"
            )
        logical_entry_count = _checked_product(
            "logical entry count", component_count, len(entries)
        )

    entry_count = _checked_u64("entry_count", len(entries))
    if len(exact_coefficients) != len(entries):
        raise RecurrenceColorCodecError(
            "exact color coefficient count does not match entry_count"
        )
    exact_catalog: list[ExactComplexRationalV1] = []
    exact_id_by_key: dict[tuple[int, int, int, int], int] = {}
    exact_factor_ids: list[int] = []
    for index, (entry, factor) in enumerate(
        zip(entries, exact_coefficients, strict=True)
    ):
        if not isinstance(factor, ExactComplexRationalV1):
            raise RecurrenceColorCodecError(
                f"exact color coefficient {index} is not canonical"
            )
        _validate_exact_matches_f64(entry, factor, index)
        factor_id = exact_id_by_key.get(factor.canonical_key)
        if factor_id is None:
            factor_id = len(exact_catalog)
            _checked_u32("exact color factor ID", factor_id)
            exact_id_by_key[factor.canonical_key] = factor_id
            exact_catalog.append(factor)
        exact_factor_ids.append(factor_id)
    exact_factor_count = _checked_u64("exact_factor_count", len(exact_catalog))
    coset_count = _checked_u64("coset_count", len(cosets))
    flattened_cosets = tuple(value for coset in cosets for value in coset)
    coset_index_count = _checked_u64("coset_index_count", len(flattened_cosets))
    logical_entry_count_u64 = _checked_u64("logical_entry_count", logical_entry_count)
    owner_map_count = _checked_u64("owner_map_count", len(owners))

    payload_size = _checked_payload_size(
        entry_count=entry_count,
        exact_factor_count=exact_factor_count,
        group_count=group_count,
        sector_count=sector_count,
        coset_index_count=coset_index_count,
    )
    header = _HEADER.pack(
        RECURRENCE_COLOR_CONTRACTION_MAGIC,
        _VERSION,
        _HEADER.size,
        storage,
        accuracy,
        flags,
        group_count,
        sector_count,
        component_count,
        local_group_count,
        destination_count,
        factor_kind,
        factor_rank,
        _ENTRY.size,
        _EXACT_FACTOR_BYTES,
        entry_count,
        exact_factor_count,
        coset_count,
        coset_index_count,
        logical_entry_count_u64,
        owner_map_count,
        payload_size,
    )
    output = bytearray(_HEADER.size + payload_size)
    output[: _HEADER.size] = header
    offset = _HEADER.size
    for entry, exact_factor_id in zip(entries, exact_factor_ids, strict=True):
        _ENTRY.pack_into(
            output,
            offset,
            _entry_left(entry),
            _entry_right(entry),
            float(entry.weight_re),
            float(entry.weight_im),
            float(entry.symmetry_factor),
            exact_factor_id,
        )
        offset += _ENTRY.size
    for factor in exact_catalog:
        offset = _pack_exact_factor(output, offset, factor)
    for value in ordered_groups:
        _U32.pack_into(output, offset, value)
        offset += _U32.size
    for value in destinations:
        _U32.pack_into(output, offset, value)
        offset += _U32.size
    for value in sectors:
        _U32.pack_into(output, offset, value)
        offset += _U32.size
    for value in components:
        _U32.pack_into(output, offset, value)
        offset += _U32.size
    for value in owners:
        _U32.pack_into(output, offset, value)
        offset += _U32.size
    for value in flattened_cosets:
        _U32.pack_into(output, offset, value)
        offset += _U32.size
    if offset != len(output):
        raise AssertionError("internal recurrence color payload size drift")
    return bytes(output)


def _encode_accuracy(value: str) -> int:
    if value == "nlc":
        return _ACCURACY_NLC
    if value == "full":
        return _ACCURACY_FULL
    raise RecurrenceColorCodecError(
        f"recurrence color payload requires nlc or full accuracy, got {value!r}"
    )


def _entry_left(
    entry: ColorContractionEntry | ColorContractionTemplateEntry,
) -> int:
    if isinstance(entry, ColorContractionEntry):
        return entry.left_group_id
    return entry.left_group_index


def _entry_right(
    entry: ColorContractionEntry | ColorContractionTemplateEntry,
) -> int:
    if isinstance(entry, ColorContractionEntry):
        return entry.right_group_id
    return entry.right_group_index


def _validate_entries(
    entries: Sequence[ColorContractionEntry | ColorContractionTemplateEntry],
    *,
    group_count: int,
    label: str,
) -> None:
    seen: set[tuple[int, int]] = set()
    for index, entry in enumerate(entries):
        left = _checked_u32(f"{label} entry {index} left ID", _entry_left(entry))
        right = _checked_u32(f"{label} entry {index} right ID", _entry_right(entry))
        if left >= group_count or right >= group_count:
            raise RecurrenceColorCodecError(
                f"{label} entry {index} references an out-of-bounds group"
            )
        if left > right:
            raise RecurrenceColorCodecError(
                f"{label} entry {index} is not canonical upper triangular"
            )
        pair = (left, right)
        if pair in seen:
            raise RecurrenceColorCodecError(
                f"{label} entries contain duplicate pair {pair}"
            )
        seen.add(pair)
        values = (
            float(entry.weight_re),
            float(entry.weight_im),
            float(entry.symmetry_factor),
        )
        if not all(math.isfinite(value) for value in values):
            raise RecurrenceColorCodecError(
                f"{label} entry {index} contains a non-finite f64"
            )
        if not all(
            math.isfinite(float(entry.symmetry_factor) * value)
            for value in (float(entry.weight_re), float(entry.weight_im))
        ):
            raise RecurrenceColorCodecError(
                f"{label} entry {index} overflows after symmetry folding"
            )


def _validate_expanded_components(
    entries: Sequence[ColorContractionEntry | ColorContractionTemplateEntry],
    *,
    group_component_ids: Sequence[int],
) -> None:
    for index, entry in enumerate(entries):
        if (
            group_component_ids[_entry_left(entry)]
            != group_component_ids[_entry_right(entry)]
        ):
            raise RecurrenceColorCodecError(
                f"expanded entry {index} couples groups from different components"
            )


def _validate_exact_matches_f64(
    entry: ColorContractionEntry | ColorContractionTemplateEntry,
    factor: ExactComplexRationalV1,
    index: int,
) -> None:
    actual = (
        float(entry.weight_re) * float(entry.symmetry_factor),
        float(entry.weight_im) * float(entry.symmetry_factor),
    )
    expected = (
        factor.real_numerator / factor.real_denominator,
        factor.imag_numerator / factor.imag_denominator,
    )
    for component, actual_value, expected_value in zip(
        ("real", "imaginary"),
        actual,
        expected,
        strict=True,
    ):
        tolerance = max(math.ulp(actual_value), math.ulp(expected_value))
        if not math.isclose(
            actual_value,
            expected_value,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise RecurrenceColorCodecError(
                f"entry {index} {component} f64 coefficient disagrees with its "
                "exact color factor"
            )


def _validate_group_coordinates(
    *,
    group_count: int,
    sector_count: int,
    component_count: int,
    group_sector_ids: Sequence[int],
    group_component_ids: Sequence[int],
) -> None:
    if len(group_sector_ids) != group_count or len(group_component_ids) != group_count:
        raise RecurrenceColorCodecError(
            "group coordinate map count does not match group_count"
        )
    if any(value >= sector_count for value in group_sector_ids):
        raise RecurrenceColorCodecError(
            "group sector map references an out-of-bounds physical sector"
        )
    if any(value >= component_count for value in group_component_ids):
        raise RecurrenceColorCodecError(
            "group component map references an out-of-bounds resolved component"
        )
    coordinates = tuple(zip(group_sector_ids, group_component_ids, strict=True))
    if len(set(coordinates)) != group_count:
        raise RecurrenceColorCodecError(
            "group coordinate maps contain a duplicate sector/component pair"
        )


def _validate_sector_owners(
    *,
    sector_count: int,
    owner_by_sector: Sequence[int],
    active_sector_ids: set[int],
) -> None:
    if len(owner_by_sector) != sector_count:
        raise RecurrenceColorCodecError(
            "physical sector owner map count does not match sector_count"
        )
    fixed_points: set[int] = set()
    for sector_id, owner_id in enumerate(owner_by_sector):
        if owner_id == _ZERO_SECTOR_OWNER:
            continue
        if owner_id >= sector_count:
            raise RecurrenceColorCodecError(
                f"physical sector {sector_id} has an out-of-bounds owner"
            )
        if owner_id > sector_id:
            raise RecurrenceColorCodecError(
                f"physical sector {sector_id} does not use its minimal owner"
            )
        if owner_by_sector[owner_id] != owner_id:
            raise RecurrenceColorCodecError(
                f"physical sector {sector_id} has a non-idempotent owner"
            )
        if owner_id == sector_id:
            fixed_points.add(sector_id)
    if fixed_points != active_sector_ids:
        raise RecurrenceColorCodecError(
            "active recurrence sectors are not exactly the authenticated "
            "physical-sector owners"
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
    row_sectors: list[int] = []
    for local_group_id in range(local_group_count):
        start = local_group_id * component_count
        row = ordered_group_ids[start : start + component_count]
        sectors = {group_sector_ids[group_id] for group_id in row}
        components = tuple(group_component_ids[group_id] for group_id in row)
        if len(sectors) != 1 or components != tuple(range(component_count)):
            raise RecurrenceColorCodecError(
                "repeated local color row does not cover one physical sector "
                "and every component exactly once"
            )
        row_sectors.append(next(iter(sectors)))
    if len(set(row_sectors)) != len(row_sectors) or any(
        sector_id >= sector_count for sector_id in row_sectors
    ):
        raise RecurrenceColorCodecError(
            "repeated local color rows do not identify unique physical sectors"
        )


def _encode_factorization(
    factorization: FactorizedColorContractionBlock | None,
    *,
    local_group_count: int,
    entries: Sequence[ColorContractionTemplateEntry],
) -> tuple[int, int, tuple[tuple[int, ...], ...]]:
    if factorization is None:
        return _FACTOR_NONE, 0, ()
    if factorization.kind == "klein-four-walsh":
        factor_kind = _FACTOR_KLEIN_FOUR
        rank = 2
    elif factorization.kind == "elementary-abelian-walsh":
        factor_kind = _FACTOR_ELEMENTARY_ABELIAN
        if factorization.rank is None:
            raise RecurrenceColorCodecError(
                "elementary-Abelian factorization is missing its rank"
            )
        rank = _checked_u32("factorization rank", factorization.rank)
        if rank < 3 or rank > _MAX_FACTOR_RANK:
            raise RecurrenceColorCodecError(
                f"elementary-Abelian factorization rank must be in "
                f"[3, {_MAX_FACTOR_RANK}]"
            )
    else:
        raise RecurrenceColorCodecError(
            f"unknown color-contraction factorization {factorization.kind!r}"
        )
    subgroup_order = 1 << rank
    cosets = tuple(
        tuple(_checked_u32("factorization local group ID", value) for value in coset)
        for coset in factorization.cosets
    )
    if not cosets or any(len(coset) != subgroup_order for coset in cosets):
        raise RecurrenceColorCodecError(
            "factorization coset sizes are inconsistent with its rank"
        )
    flattened = tuple(value for coset in cosets for value in coset)
    if len(flattened) != local_group_count:
        raise RecurrenceColorCodecError(
            "factorization cosets do not cover every local group"
        )
    if sorted(flattened) != list(range(local_group_count)):
        raise RecurrenceColorCodecError(
            "factorization cosets are not a partition of local groups"
        )
    _validate_walsh_invariance(cosets, entries)
    return factor_kind, rank, cosets


def _encode_symmetric_group_convolution(
    block: SymmetricGroupColorContractionBlock,
    *,
    local_group_count: int,
    entries: Sequence[ColorContractionTemplateEntry],
) -> tuple[int, int, tuple[tuple[int, ...], ...]]:
    """Validate and encode canonical ``S_m`` convolution-kernel storage.

    For every channel pair ``c <= d`` and relative permutation ``r``, the
    fixed-width entry IDs are ``channel[c][identity]`` and ``channel[d][r]``.
    Kernel rows precede ordinary upper-triangle rows touching the consecutive
    residual suffix.  Zero kernel and residual coefficients remain explicit so
    the payload authenticates the complete convolution/direct decomposition.
    """

    degree = _checked_u32("symmetric-group degree", block.degree)
    if degree < 2 or degree > _MAX_SYMMETRIC_GROUP_DEGREE:
        raise RecurrenceColorCodecError(
            "symmetric-group convolution degree must be in "
            f"[2, {_MAX_SYMMETRIC_GROUP_DEGREE}]"
        )
    group_order = math.factorial(degree)
    channels = tuple(
        tuple(
            _checked_u32("symmetric-group channel local group ID", value)
            for value in channel
        )
        for channel in block.channel_cosets
    )
    if not channels:
        raise RecurrenceColorCodecError(
            "symmetric-group convolution requires at least one complete channel"
        )
    eligible_group_count = _checked_product(
        "symmetric-group eligible group count", len(channels), group_order
    )
    if eligible_group_count > local_group_count:
        raise RecurrenceColorCodecError(
            "symmetric-group channels exceed the local group domain"
        )
    flattened_channels = tuple(value for channel in channels for value in channel)
    if flattened_channels != tuple(range(eligible_group_count)):
        raise RecurrenceColorCodecError(
            "symmetric-group channels are not canonical channel-major/"
            "permutation-major local indices"
        )
    if tuple(block.residual_local_group_indices) != tuple(
        range(eligible_group_count, local_group_count)
    ):
        raise RecurrenceColorCodecError(
            "symmetric-group residual groups are not the consecutive local suffix"
        )

    kernel_entry_count = _checked_product(
        "symmetric-group kernel entry count",
        len(channels) * (len(channels) + 1) // 2,
        group_order,
    )
    if kernel_entry_count != len(block.kernel_entries):
        raise RecurrenceColorCodecError(
            "symmetric-group kernel rows do not cover every channel pair"
        )
    if tuple(entries[:kernel_entry_count]) != block.kernel_entries:
        raise RecurrenceColorCodecError(
            "symmetric-group kernel rows are not the leading entry section"
        )
    offset = 0
    for left_channel, left_coset in enumerate(channels):
        for right_channel in range(left_channel, len(channels)):
            right_coset = channels[right_channel]
            symmetry = 1.0 if left_channel == right_channel else 2.0
            for relative_index in range(group_order):
                entry = entries[offset]
                if (
                    _entry_left(entry) != left_coset[0]
                    or _entry_right(entry) != right_coset[relative_index]
                    or entry.symmetry_factor != symmetry
                ):
                    raise RecurrenceColorCodecError(
                        "symmetric-group kernel rows are not canonical "
                        "(channel, channel, relative-permutation) records"
                    )
                offset += 1
            if left_channel == right_channel:
                kernel = block.kernel_exact_weights[offset - group_order : offset]
                for relative_index, exact in enumerate(kernel):
                    inverse_index = _inverse_lexicographic_permutation_index(
                        degree, relative_index
                    )
                    if exact != kernel[inverse_index]:
                        raise RecurrenceColorCodecError(
                            "symmetric-group diagonal kernel violates inverse "
                            "Hermiticity"
                        )

    residual_entries = tuple(entries[kernel_entry_count:])
    if residual_entries != block.residual_entries:
        raise RecurrenceColorCodecError(
            "symmetric-group residual rows are not the trailing entry section"
        )
    residual_pairs = tuple(
        (_entry_left(entry), _entry_right(entry)) for entry in residual_entries
    )
    expected_residual_pairs = (
        tuple(
            (left, right)
            for left in range(local_group_count)
            for right in range(left, local_group_count)
            if right >= eligible_group_count
        )
        if block.residual_local_group_indices
        else ()
    )
    if residual_pairs != expected_residual_pairs:
        raise RecurrenceColorCodecError(
            "symmetric-group residual rows do not exhaust the canonical pairs "
            "touching the residual suffix"
        )
    for entry in residual_entries:
        if entry.weight_im != 0.0:
            raise RecurrenceColorCodecError(
                "symmetric-group residual row contains a complex coefficient"
            )
        expected_symmetry = 1.0 if _entry_left(entry) == _entry_right(entry) else 2.0
        if entry.symmetry_factor != expected_symmetry:
            raise RecurrenceColorCodecError(
                "symmetric-group residual row has a noncanonical symmetry factor"
            )
    return _FACTOR_SYMMETRIC_GROUP_FOURIER, degree, channels


def _symmetric_group_exact_coefficients(
    block: SymmetricGroupColorContractionBlock,
) -> tuple[ExactComplexRationalV1, ...]:
    entries = (*block.kernel_entries, *block.residual_entries)
    exact_weights = (*block.kernel_exact_weights, *block.residual_exact_weights)
    result = []
    for entry, exact in zip(entries, exact_weights, strict=True):
        symmetry = int(entry.symmetry_factor)
        if float(symmetry) != entry.symmetry_factor or symmetry not in {1, 2}:
            raise RecurrenceColorCodecError(
                "symmetric-group entry has a noncanonical symmetry factor"
            )
        folded = exact * symmetry
        result.append(
            ExactComplexRationalV1(
                real_numerator=folded.numerator,
                real_denominator=folded.denominator,
            )
        )
    return tuple(result)


def _inverse_lexicographic_permutation_index(degree: int, index: int) -> int:
    """Return the lexicographic rank of a ranked permutation's inverse."""

    remaining = list(range(degree))
    permutation: list[int] = []
    rank = index
    for width in range(degree, 0, -1):
        stride = math.factorial(width - 1)
        quotient, rank = divmod(rank, stride)
        permutation.append(remaining.pop(quotient))
    inverse = [0] * degree
    for position, value in enumerate(permutation):
        inverse[value] = position
    remaining = list(range(degree))
    inverse_rank = 0
    for position, value in enumerate(inverse):
        offset = remaining.index(value)
        inverse_rank += offset * math.factorial(degree - position - 1)
        remaining.pop(offset)
    return inverse_rank


def _validate_walsh_invariance(
    cosets: Sequence[Sequence[int]],
    entries: Sequence[ColorContractionTemplateEntry],
) -> None:
    matrix: dict[tuple[int, int], float] = {}
    for index, entry in enumerate(entries):
        if entry.weight_im != 0.0:
            raise RecurrenceColorCodecError(
                "factorized color contraction requires real weights"
            )
        left = entry.left_group_index
        right = entry.right_group_index
        coefficient = float(entry.symmetry_factor) * float(entry.weight_re)
        if left != right:
            coefficient *= 0.5
        if not math.isfinite(coefficient):
            raise RecurrenceColorCodecError(
                f"factorized entry {index} has a non-finite matrix coefficient"
            )
        matrix[(left, right)] = coefficient

    def value(left: int, right: int) -> float:
        return matrix.get((min(left, right), max(left, right)), 0.0)

    subgroup_order = len(cosets[0])
    for left_coset in cosets:
        for right_coset in cosets:
            for left_index in range(subgroup_order):
                for right_index in range(subgroup_order):
                    if value(left_coset[left_index], right_coset[right_index]) != value(
                        left_coset[0],
                        right_coset[left_index ^ right_index],
                    ):
                        raise RecurrenceColorCodecError(
                            "factorization cosets are inconsistent with the "
                            "canonical color matrix"
                        )


def _checked_u32(label: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecurrenceColorCodecError(f"{label} must be an integer")
    if value < 0 or value > 0xFFFF_FFFF:
        raise RecurrenceColorCodecError(f"{label} exceeds the u32 domain")
    return value


def _checked_u64(label: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecurrenceColorCodecError(f"{label} must be an integer")
    if value < 0 or value > 0xFFFF_FFFF_FFFF_FFFF:
        raise RecurrenceColorCodecError(f"{label} exceeds the u64 domain")
    return value


def _checked_product(label: str, left: int, right: int) -> int:
    result = left * right
    return _checked_u64(label, result)


def _checked_payload_size(
    *,
    entry_count: int,
    exact_factor_count: int,
    group_count: int,
    sector_count: int,
    coset_index_count: int,
) -> int:
    size = (
        entry_count * _ENTRY.size
        + exact_factor_count * _EXACT_FACTOR_BYTES
        + 2 * group_count * _U32.size
        + 2 * group_count * _U32.size
        + sector_count * _U32.size
        + coset_index_count * _U32.size
    )
    return _checked_u64("recurrence color payload byte count", size)


def _pack_exact_factor(
    output: bytearray,
    offset: int,
    factor: ExactComplexRationalV1,
) -> int:
    for value in factor.canonical_key:
        if value < -(1 << 127) or value >= 1 << 127:
            raise RecurrenceColorCodecError(
                "exact color coefficient exceeds the signed i128 domain"
            )
        output[offset : offset + 16] = value.to_bytes(
            16,
            byteorder="little",
            signed=True,
        )
        offset += 16
    return offset


def recurrence_color_contraction_digest(payload: bytes) -> str:
    """Return the caller-owned deterministic SHA-256 payload digest."""

    return sha256(payload).hexdigest()


__all__ = [
    "RECURRENCE_COLOR_CONTRACTION_CODEC_ABI",
    "RECURRENCE_COLOR_CONTRACTION_MAGIC",
    "RecurrenceColorCodecError",
    "encode_recurrence_color_contraction",
    "recurrence_color_contraction_digest",
]

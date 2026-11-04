# SPDX-License-Identifier: 0BSD
"""Generic symmetric-group color-contraction planning.

The planner knows only canonical external color roles and color-sector
structures.  It discovers complete regular permutation orbits, stores exact
relative color kernels for those orbits, and leaves every uncertified group in
an ordinary direct residual.  Model names, particle identities, and process
families never enter the construction.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import cast, overload

from .contraction_factors import exact_color_contraction_factor
from .contraction_types import (
    ColorContractionPlan,
    ColorContractionTemplateEntry,
    ColorGroupDescriptor,
    SymmetricGroupColorContractionBlock,
)
from .plan_types import GenericColorPlan, LCColorSector, LCOpenColorLine

_MAX_SYMMETRIC_GROUP_DEGREE = 10
_ZERO_SECTOR_OWNER = 0xFFFF_FFFF
_SYMMETRIC_GROUP_ORDERS = tuple(
    math.factorial(degree) for degree in range(_MAX_SYMMETRIC_GROUP_DEGREE + 1)
)


def _symmetric_group_order(degree: int) -> int:
    if (
        not isinstance(degree, int)
        or isinstance(degree, bool)
        or not 0 <= degree <= _MAX_SYMMETRIC_GROUP_DEGREE
    ):
        raise ValueError("symmetric-group degree is unsupported")
    return _SYMMETRIC_GROUP_ORDERS[degree]


def _lexicographic_permutation_rank(
    permutation: Sequence[int],
    *,
    degree: int | None = None,
) -> int:
    """Return the checked ``itertools.permutations`` index in ``O(m**2)``."""

    expected_degree = len(permutation) if degree is None else degree
    _symmetric_group_order(expected_degree)
    if len(permutation) != expected_degree:
        raise ValueError("permutation degree is inconsistent")
    available = list(range(expected_degree))
    rank = 0
    for position, value in enumerate(permutation):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("permutation contains a non-integer label")
        try:
            available_index = available.index(value)
        except ValueError as exc:
            raise ValueError("permutation labels are not a bijection") from exc
        rank += (
            available_index * _SYMMETRIC_GROUP_ORDERS[expected_degree - position - 1]
        )
        del available[available_index]
    return rank


def _lexicographic_permutation_unrank(
    degree: int,
    rank: int,
) -> tuple[int, ...]:
    """Return one lexicographic permutation without materializing its group."""

    group_order = _symmetric_group_order(degree)
    if (
        not isinstance(rank, int)
        or isinstance(rank, bool)
        or not 0 <= rank < group_order
    ):
        raise ValueError("permutation rank is out of bounds")
    available = list(range(degree))
    permutation: list[int] = []
    remainder = rank
    for position in range(degree):
        suffix_order = _SYMMETRIC_GROUP_ORDERS[degree - position - 1]
        available_index, remainder = divmod(remainder, suffix_order)
        permutation.append(available.pop(available_index))
    return tuple(permutation)


@dataclass(frozen=True, slots=True)
class _LexicographicPermutations(Sequence[tuple[int, ...]]):
    """Constant-space sequence view of canonical symmetric-group elements."""

    degree: int

    def __post_init__(self) -> None:
        _symmetric_group_order(self.degree)

    def __len__(self) -> int:
        return _SYMMETRIC_GROUP_ORDERS[self.degree]

    @overload
    def __getitem__(self, index: int) -> tuple[int, ...]: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[tuple[int, ...], ...]: ...

    def __getitem__(
        self,
        index: int | slice,
    ) -> tuple[int, ...] | tuple[tuple[int, ...], ...]:
        if isinstance(index, slice):
            return tuple(
                _lexicographic_permutation_unrank(self.degree, rank)
                for rank in range(*index.indices(len(self)))
            )
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("permutation index must be an integer or slice")
        normalized_index = index if index >= 0 else len(self) + index
        if not 0 <= normalized_index < len(self):
            raise IndexError("permutation index is out of range")
        return _lexicographic_permutation_unrank(self.degree, normalized_index)

    def __iter__(self) -> Iterator[tuple[int, ...]]:
        for rank in range(len(self)):
            yield _lexicographic_permutation_unrank(self.degree, rank)


@dataclass(frozen=True)
class CertifiedSymmetricGroupOrbit:
    """One structurally certified regular ``S_m`` color-sector orbit."""

    channel_key: tuple[object, ...]
    sector_ids: tuple[int, ...]
    permutations: Sequence[tuple[int, ...]]

    def __post_init__(self) -> None:
        if not self.permutations:
            raise ValueError("symmetric-group orbit has no permutations")
        if isinstance(self.permutations, _LexicographicPermutations):
            degree = self.permutations.degree
        else:
            degree = len(self.permutations[0])
            _symmetric_group_order(degree)
            if any(
                _lexicographic_permutation_rank(permutation, degree=degree) != index
                for index, permutation in enumerate(self.permutations)
            ):
                raise ValueError(
                    "symmetric-group orbit permutations are not lexicographic"
                )
        if len(self.permutations) != _SYMMETRIC_GROUP_ORDERS[degree]:
            raise ValueError("symmetric-group orbit has the wrong group order")
        if len(self.sector_ids) != len(self.permutations):
            raise ValueError(
                "symmetric-group orbit sectors do not match the group order"
            )
        if len(set(self.sector_ids)) != len(self.sector_ids) or any(
            sector_id < 0 for sector_id in self.sector_ids
        ):
            raise ValueError(
                "symmetric-group orbit sector IDs must be unique and nonnegative"
            )

    @property
    def degree(self) -> int:
        if isinstance(self.permutations, _LexicographicPermutations):
            return self.permutations.degree
        return len(self.permutations[0])


@dataclass(frozen=True)
class SymmetricGroupOrbitPartition:
    """Certified orbit partition of one active structural color domain."""

    degree: int
    permuted_adjoint_labels: tuple[int, ...]
    fixed_adjoint_labels: tuple[int, ...]
    orbits: tuple[CertifiedSymmetricGroupOrbit, ...]
    residual_sector_ids: tuple[int, ...]
    sector_owner_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.degree != len(self.permuted_adjoint_labels):
            raise ValueError(
                "symmetric-group degree does not match permuted adjoint labels"
            )
        if self.degree < 0 or self.degree > _MAX_SYMMETRIC_GROUP_DEGREE:
            raise ValueError("symmetric-group orbit degree is unsupported")
        labels = (*self.fixed_adjoint_labels, *self.permuted_adjoint_labels)
        if len(set(labels)) != len(labels):
            raise ValueError("symmetric-group adjoint label partition overlaps")
        if tuple(sorted(self.residual_sector_ids)) != self.residual_sector_ids:
            raise ValueError(
                "symmetric-group residual sectors are not canonically ordered"
            )
        covered: set[int] = set()
        for orbit in self.orbits:
            previous_count = len(covered)
            covered.update(orbit.sector_ids)
            if len(covered) != previous_count + len(orbit.sector_ids):
                raise ValueError("symmetric-group certified orbits overlap")
        if covered.intersection(self.residual_sector_ids):
            raise ValueError("symmetric-group orbit and residual sectors overlap")
        if any(orbit.degree != self.degree for orbit in self.orbits):
            raise ValueError("symmetric-group orbit degrees are inconsistent")
        if self.orbits:
            # Internally this is one constant-space canonical sequence view;
            # retain compatibility with explicitly constructed canonical orbits.
            shared_permutations = self.orbits[0].permutations
            if any(
                orbit.permutations is not shared_permutations
                for orbit in self.orbits[1:]
            ):
                raise ValueError(
                    "symmetric-group channels do not share their canonical view"
                )

    @property
    def ordered_sector_ids(self) -> tuple[int, ...]:
        return (
            *(sector_id for orbit in self.orbits for sector_id in orbit.sector_ids),
            *self.residual_sector_ids,
        )


def certify_symmetric_group_orbits(
    color_plan: GenericColorPlan,
    sector_ids: Sequence[int],
    *,
    sector_owner_ids: Sequence[int] | None = None,
) -> SymmetricGroupOrbitPartition:
    """Partition active sectors into complete free adjoint-label orbits.

    Certification is structural: every channel contains exactly one canonical
    sector for every lexicographic permutation, and each sector differs from
    the identity only by simultaneous relabelling of the declared adjoint
    labels.  This is the invariance used by ``exact_color_contraction_factor``;
    no dense color matrix is constructed here.
    """

    if not isinstance(color_plan, GenericColorPlan):
        raise TypeError("symmetric-group planning requires a GenericColorPlan")
    requested_sector_ids = tuple(sorted(int(value) for value in sector_ids))
    if len(set(requested_sector_ids)) != len(requested_sector_ids):
        raise ValueError("symmetric-group active sectors contain duplicates")
    sectors_by_id = {int(sector.id): sector for sector in color_plan.sectors}
    if any(sector_id not in sectors_by_id for sector_id in requested_sector_ids):
        raise ValueError("symmetric-group planning references an unknown sector")

    # The caller supplies the schedule-authenticated alias certificate.  Orbit
    # discovery operates only on its canonical owners; traversal replicas are
    # destinations of those owners, never additional FFT channels.
    owners = _validated_sector_owner_map(color_plan, sector_owner_ids)
    _validate_owner_structures(color_plan, owners)
    active_owners = {owners[sector_id] for sector_id in requested_sector_ids}
    if _ZERO_SECTOR_OWNER in active_owners:
        raise ValueError(
            "symmetric-group active sector is certified as structural zero"
        )
    active_sector_ids = tuple(sorted(active_owners))
    permuted_labels, fixed_labels = _adjoint_action_labels(color_plan)
    degree = len(permuted_labels)
    if degree > _MAX_SYMMETRIC_GROUP_DEGREE:
        raise ValueError(
            "symmetric-group degree exceeds the supported maximum "
            f"{_MAX_SYMMETRIC_GROUP_DEGREE}"
        )
    group_order = _symmetric_group_order(degree)
    lexicographic_permutations = _LexicographicPermutations(degree)

    buckets: dict[
        tuple[object, ...],
        dict[int, int | list[int]],
    ] = {}
    residual: set[int] = set()
    for sector_id in active_sector_ids:
        candidate = _sector_orbit_coordinate(
            sectors_by_id[sector_id],
            permuted_labels=permuted_labels,
            fixed_labels=fixed_labels,
            basis=color_plan.basis,
        )
        if candidate is None or degree < 2:
            residual.add(sector_id)
            continue
        channel_key, coordinate = candidate
        try:
            coordinate_index = _lexicographic_permutation_rank(
                coordinate,
                degree=degree,
            )
        except ValueError:
            residual.add(sector_id)
            continue
        sectors_by_coordinate = buckets.setdefault(channel_key, {})
        existing_sector_ids = sectors_by_coordinate.get(coordinate_index)
        if existing_sector_ids is None:
            sectors_by_coordinate[coordinate_index] = sector_id
        elif isinstance(existing_sector_ids, int):
            sectors_by_coordinate[coordinate_index] = [
                existing_sector_ids,
                sector_id,
            ]
        else:
            existing_sector_ids.append(sector_id)

    certified_orbit_rows: list[tuple[tuple[object, ...], tuple[int, ...]]] = []
    for channel_key in sorted(buckets):
        sectors_by_coordinate = buckets[channel_key]
        if len(sectors_by_coordinate) != group_order or any(
            isinstance(sector_ids_at_coordinate, list)
            for sector_ids_at_coordinate in sectors_by_coordinate.values()
        ):
            for sector_ids_at_coordinate in sectors_by_coordinate.values():
                if isinstance(sector_ids_at_coordinate, int):
                    residual.add(sector_ids_at_coordinate)
                else:
                    residual.update(sector_ids_at_coordinate)
            continue
        orbit_sector_ids = tuple(
            cast(int, sectors_by_coordinate[coordinate_index])
            for coordinate_index in range(group_order)
        )
        certified_orbit_rows.append(
            (
                channel_key,
                orbit_sector_ids,
            )
        )

    if buckets:
        del sectors_by_coordinate
    buckets.clear()
    orbits = tuple(
        CertifiedSymmetricGroupOrbit(
            channel_key=channel_key,
            sector_ids=orbit_sector_ids,
            permutations=lexicographic_permutations,
        )
        for channel_key, orbit_sector_ids in certified_orbit_rows
    )
    covered_count = sum(len(orbit.sector_ids) for orbit in orbits)
    if covered_count + len(residual) != len(active_sector_ids):
        raise AssertionError("symmetric-group orbit partition lost active sectors")
    return SymmetricGroupOrbitPartition(
        degree=degree,
        permuted_adjoint_labels=permuted_labels,
        fixed_adjoint_labels=fixed_labels,
        orbits=orbits,
        residual_sector_ids=tuple(sorted(residual)),
        sector_owner_ids=owners,
    )


def build_symmetric_group_color_contraction_plan(
    color_plan: GenericColorPlan,
    groups: Sequence[ColorGroupDescriptor],
    *,
    sector_owner_ids: Sequence[int] | None = None,
) -> ColorContractionPlan:
    """Build exact convolution kernels and a direct residual color metric."""

    accuracy = color_plan.color_accuracy
    descriptors = tuple(groups)
    if accuracy not in {"nlc", "full"}:
        return _unsupported(
            accuracy,
            len(descriptors),
            "symmetric-group FFT requires NLC or full color",
        )
    if color_plan.truncated or not color_plan.sectors:
        return _unsupported(
            accuracy,
            len(descriptors),
            "symmetric-group FFT requires a complete nonempty color plan",
        )
    if not descriptors:
        return _unsupported(
            accuracy,
            0,
            "symmetric-group FFT requires color-contraction groups",
        )

    try:
        owners = _validated_sector_owner_map(color_plan, sector_owner_ids)
        _validate_owner_structures(color_plan, owners)
    except ValueError as exc:
        return _unsupported(accuracy, len(descriptors), str(exc))
    rectangular = _rectangular_components(
        color_plan,
        descriptors,
        owners=owners,
    )
    if isinstance(rectangular, str):
        return _unsupported(accuracy, len(descriptors), rectangular)
    components, component_sector_ids, helicity_weight = rectangular
    try:
        partition = certify_symmetric_group_orbits(
            color_plan,
            component_sector_ids,
            sector_owner_ids=owners,
        )
    except ValueError as exc:
        return _unsupported(accuracy, len(descriptors), str(exc))

    if not partition.orbits and partition.degree >= 2:
        return _unsupported(
            accuracy,
            len(descriptors),
            "symmetric-group FFT found no certified orbit at nontrivial "
            f"degree {partition.degree}",
        )

    # S_0 and S_1 have no useful transform.  They are intentionally represented
    # by the established exact direct residual rather than an empty FFT payload.
    if not partition.orbits:
        from .contraction import build_color_contraction_plan

        normalized_descriptors, destination_by_group = _normalized_descriptors(
            components,
            partition.ordered_sector_ids,
        )
        direct = build_color_contraction_plan(color_plan, normalized_descriptors)
        if direct is None:
            return _unsupported(
                accuracy,
                len(descriptors),
                "could not build the symmetric-group direct residual",
            )
        return replace(direct, destination_by_group=destination_by_group)

    local_sector_ids = partition.ordered_sector_ids
    _, destination_by_group = _normalized_descriptors(
        components,
        local_sector_ids,
    )
    component_group_ids = tuple(range(len(local_sector_ids) * len(components)))
    group_order = _symmetric_group_order(partition.degree)
    channel_cosets = tuple(
        tuple(
            range(
                channel_index * group_order,
                (channel_index + 1) * group_order,
            )
        )
        for channel_index in range(len(partition.orbits))
    )
    residual_local_group_indices = tuple(
        range(len(channel_cosets) * group_order, len(local_sector_ids))
    )
    sector_by_id = {int(sector.id): sector for sector in color_plan.sectors}
    weight_fraction = Fraction.from_float(helicity_weight)

    kernel_entries: list[ColorContractionTemplateEntry] = []
    kernel_exact_weights: list[Fraction] = []
    kernel_exact_by_key: dict[tuple[int, int, int], Fraction] = {}
    diagonal_trace_exact: dict[tuple[int, tuple[int, ...]], Fraction] = {}
    for left_channel, left_orbit in enumerate(partition.orbits):
        left_local_index = channel_cosets[left_channel][0]
        left_sector = sector_by_id[left_orbit.sector_ids[0]]
        for right_channel in range(left_channel, len(partition.orbits)):
            right_orbit = partition.orbits[right_channel]
            symmetry = 1.0 if left_channel == right_channel else 2.0
            quotient_diagonal_trace = (
                left_channel == right_channel
                and color_plan.basis == "trace"
                and color_plan.process.color_endpoints.pair_count == 0
                and left_sector.kind == "single-trace"
                and left_orbit.channel_key[:1] == ("single-trace",)
            )
            for relative_index, right_sector_id in enumerate(right_orbit.sector_ids):
                right_local_index = channel_cosets[right_channel][relative_index]
                cache_key = None
                if quotient_diagonal_trace:
                    cache_key = (
                        left_channel,
                        _single_trace_diagonal_kernel_representative(
                            _lexicographic_permutation_unrank(
                                partition.degree,
                                relative_index,
                            )
                        ),
                    )
                exact = (
                    None if cache_key is None else diagonal_trace_exact.get(cache_key)
                )
                if exact is None:
                    exact = weight_fraction * exact_color_contraction_factor(
                        color_plan,
                        left_sector,
                        sector_by_id[right_sector_id],
                        accuracy=accuracy,
                        full_col_acc=20,
                    )
                    if cache_key is not None:
                        diagonal_trace_exact[cache_key] = exact
                kernel_entries.append(
                    ColorContractionTemplateEntry(
                        left_group_index=left_local_index,
                        right_group_index=right_local_index,
                        weight_re=float(exact),
                        symmetry_factor=symmetry,
                    )
                )
                kernel_exact_weights.append(exact)
                kernel_exact_by_key[(left_channel, right_channel, relative_index)] = (
                    exact
                )

    hermiticity_relative_indices = _hermiticity_relative_indices(
        partition.degree,
        channel_count=len(partition.orbits),
    )
    _certify_cross_channel_hermiticity(
        color_plan,
        partition,
        sector_by_id=sector_by_id,
        accuracy=accuracy,
        weight_fraction=weight_fraction,
        kernel_exact_by_key=kernel_exact_by_key,
        relative_indices=hermiticity_relative_indices,
    )
    _certify_sampled_equivariance(
        color_plan,
        partition,
        sector_by_id=sector_by_id,
        accuracy=accuracy,
        weight_fraction=weight_fraction,
        kernel_exact_by_key=kernel_exact_by_key,
    )
    hermiticity_check_mode = (
        "vacuous"
        if len(partition.orbits) < 2
        else "full"
        if partition.degree <= 4
        else "deterministic-samples"
    )

    residual_entries: list[ColorContractionTemplateEntry] = []
    residual_exact_weights: list[Fraction] = []
    if residual_local_group_indices:
        residual_set = set(residual_local_group_indices)
        for left_local_index, left_sector_id in enumerate(local_sector_ids):
            for right_local_index in range(left_local_index, len(local_sector_ids)):
                if not residual_set.intersection((left_local_index, right_local_index)):
                    continue
                exact = weight_fraction * exact_color_contraction_factor(
                    color_plan,
                    sector_by_id[left_sector_id],
                    sector_by_id[local_sector_ids[right_local_index]],
                    accuracy=accuracy,
                    full_col_acc=20,
                )
                residual_entries.append(
                    ColorContractionTemplateEntry(
                        left_group_index=left_local_index,
                        right_group_index=right_local_index,
                        weight_re=float(exact),
                        symmetry_factor=(
                            1.0 if left_local_index == right_local_index else 2.0
                        ),
                    )
                )
                residual_exact_weights.append(exact)

    block = SymmetricGroupColorContractionBlock(
        degree=partition.degree,
        component_count=len(components),
        component_group_ids=component_group_ids,
        local_sector_ids=local_sector_ids,
        channel_cosets=channel_cosets,
        kernel_entries=tuple(kernel_entries),
        kernel_exact_weights=tuple(kernel_exact_weights),
        residual_entries=tuple(residual_entries),
        residual_exact_weights=tuple(residual_exact_weights),
        residual_local_group_indices=residual_local_group_indices,
        hermiticity_check_mode=hermiticity_check_mode,
        hermiticity_relative_indices=hermiticity_relative_indices,
    )
    return ColorContractionPlan(
        color_accuracy=accuracy,
        supported=True,
        reason=None,
        group_count=len(component_group_ids),
        entries=(),
        symmetric_group_block=block,
        destination_by_group=destination_by_group,
    )


def reconstruct_symmetric_group_dense_exact(
    block: SymmetricGroupColorContractionBlock,
) -> dict[tuple[int, int], Fraction]:
    """Expand a small plan for tests and generation-time certification tools."""

    kernel: dict[tuple[int, int, int], Fraction] = {}
    offset = 0
    for left_channel in range(block.channel_count):
        for right_channel in range(left_channel, block.channel_count):
            for relative_index in range(block.group_order):
                kernel[(left_channel, right_channel, relative_index)] = (
                    block.kernel_exact_weights[offset]
                )
                offset += 1

    dense: dict[tuple[int, int], Fraction] = {}
    for left_channel, left_coset in enumerate(block.channel_cosets):
        for right_channel in range(left_channel, block.channel_count):
            right_coset = block.channel_cosets[right_channel]
            for left_permutation_index, left_local in enumerate(left_coset):
                left_permutation = _lexicographic_permutation_unrank(
                    block.degree,
                    left_permutation_index,
                )
                for right_permutation_index, right_local in enumerate(right_coset):
                    if left_local > right_local:
                        continue
                    relative = _relative_permutation(
                        left_permutation,
                        _lexicographic_permutation_unrank(
                            block.degree,
                            right_permutation_index,
                        ),
                    )
                    dense[(left_local, right_local)] = kernel[
                        (
                            left_channel,
                            right_channel,
                            _lexicographic_permutation_rank(
                                relative,
                                degree=block.degree,
                            ),
                        )
                    ]
    dense.update(
        (
            (entry.left_group_index, entry.right_group_index),
            exact,
        )
        for entry, exact in zip(
            block.residual_entries,
            block.residual_exact_weights,
            strict=True,
        )
    )
    return dense


def _unsupported(accuracy: str, group_count: int, reason: str) -> ColorContractionPlan:
    return ColorContractionPlan(
        color_accuracy=accuracy,
        supported=False,
        reason=reason,
        group_count=group_count,
        entries=(),
    )


def _rectangular_components(
    color_plan: GenericColorPlan,
    descriptors: tuple[ColorGroupDescriptor, ...],
    *,
    owners: tuple[int, ...],
) -> (
    tuple[
        tuple[tuple[ColorGroupDescriptor, ...], ...],
        tuple[int, ...],
        float,
    ]
    | str
):
    if len({descriptor.group_id for descriptor in descriptors}) != len(descriptors):
        return "symmetric-group color groups contain duplicate group IDs"
    components_by_helicity: dict[tuple[object, ...], list[ColorGroupDescriptor]] = {}
    for descriptor in descriptors:
        components_by_helicity.setdefault(descriptor.helicity_key, []).append(
            descriptor
        )
    original_components = tuple(
        sorted(
            (
                tuple(sorted(component, key=lambda descriptor: descriptor.group_id))
                for component in components_by_helicity.values()
            ),
            key=lambda component: component[0].group_id,
        )
    )
    if any(
        descriptor.sector_id < 0 or descriptor.sector_id >= len(owners)
        for descriptor in descriptors
    ):
        return "symmetric-group color group references an unknown sector"
    if any(
        owners[descriptor.sector_id] == _ZERO_SECTOR_OWNER
        for descriptor in descriptors
    ):
        return (
            "symmetric-group color group references a certified structural-zero sector"
        )
    reference_owner_ids = tuple(
        sorted({owners[descriptor.sector_id] for descriptor in original_components[0]})
    )
    if not reference_owner_ids:
        return "symmetric-group helicity component has no canonical owners"
    reference_weight = float(original_components[0][0].helicity_weight)
    if not math.isfinite(reference_weight):
        return "symmetric-group helicity weight is not finite"
    reference_weight_bits = struct.pack(">d", reference_weight)
    components: list[tuple[ColorGroupDescriptor, ...]] = []
    sectors_by_id = {int(sector.id): sector for sector in color_plan.sectors}
    for component in original_components:
        by_sector: dict[int, ColorGroupDescriptor] = {}
        for descriptor in component:
            if descriptor.sector_id in by_sector:
                return "symmetric-group component contains duplicate color sectors"
            by_sector[descriptor.sector_id] = descriptor
        descriptors_by_owner: dict[int, list[ColorGroupDescriptor]] = {}
        for sector_id, descriptor in by_sector.items():
            descriptors_by_owner.setdefault(owners[sector_id], []).append(descriptor)
        owner_ids = tuple(sorted(descriptors_by_owner))
        if owner_ids != reference_owner_ids:
            return "symmetric-group helicity components do not share one owner domain"
        if any(
            not math.isfinite(float(descriptor.helicity_weight))
            or struct.pack(">d", float(descriptor.helicity_weight))
            != reference_weight_bits
            for descriptor in component
        ):
            return "symmetric-group helicity components have inconsistent weights"
        canonical_component: list[ColorGroupDescriptor] = []
        for owner_id in owner_ids:
            aliases = descriptors_by_owner[owner_id]
            source = next(
                (
                    descriptor
                    for descriptor in aliases
                    if descriptor.sector_id == owner_id
                ),
                min(aliases, key=lambda descriptor: descriptor.group_id),
            )
            owner_sector = sectors_by_id[owner_id]
            canonical_component.append(
                replace(
                    source,
                    sector_id=owner_id,
                    word=owner_sector.color_words[0],
                )
            )
        components.append(tuple(canonical_component))

    reference_sector_ids = reference_owner_ids
    if len(set(reference_sector_ids)) != len(reference_sector_ids):
        return "symmetric-group component contains duplicate color sectors"
    return tuple(components), reference_sector_ids, reference_weight


def _normalized_descriptors(
    components: tuple[tuple[ColorGroupDescriptor, ...], ...],
    local_sector_ids: tuple[int, ...],
) -> tuple[tuple[ColorGroupDescriptor, ...], tuple[int, ...]]:
    by_component_and_sector = tuple(
        {descriptor.sector_id: descriptor for descriptor in component}
        for component in components
    )
    normalized: list[ColorGroupDescriptor] = []
    destination_by_group = [0] * (len(local_sector_ids) * len(components))
    for component_index, descriptors_by_sector in enumerate(by_component_and_sector):
        for local_index, sector_id in enumerate(local_sector_ids):
            source = descriptors_by_sector[sector_id]
            group_id = local_index * len(components) + component_index
            normalized.append(
                ColorGroupDescriptor(
                    group_id=group_id,
                    helicity_key=source.helicity_key,
                    sector_id=source.sector_id,
                    word=source.word,
                    helicity_weight=source.helicity_weight,
                )
            )
            destination_by_group[group_id] = source.group_id
    return tuple(normalized), tuple(destination_by_group)


def _adjoint_action_labels(
    color_plan: GenericColorPlan,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    adjoint_labels = tuple(
        sorted(int(label) for label in color_plan.process.adjoint_labels)
    )
    if not adjoint_labels:
        return (), ()
    if color_plan.process.fundamental_labels:
        return adjoint_labels, ()
    if color_plan.basis == "adjoint":
        return adjoint_labels[1:-1], (adjoint_labels[0], adjoint_labels[-1])
    anchor = adjoint_labels[0]
    return adjoint_labels[1:], (anchor,)


def _sector_orbit_coordinate(
    sector: LCColorSector,
    *,
    permuted_labels: tuple[int, ...],
    fixed_labels: tuple[int, ...],
    basis: str = "trace",
) -> tuple[tuple[object, ...], tuple[int, ...]] | None:
    rank_by_label = {label: rank for rank, label in enumerate(permuted_labels)}
    if basis == "adjoint":
        word = tuple(int(label) for label in sector.trace_labels)
        if (
            sector.kind != "single-trace"
            or len(fixed_labels) != 2
            or len(word) != len(permuted_labels) + 2
            or word[0] != fixed_labels[0]
            or word[-1] != fixed_labels[1]
            or set(word[1:-1]) != set(permuted_labels)
        ):
            return None
        return (
            ("adjoint", fixed_labels, tuple(sector.singlet_labels)),
            tuple(rank_by_label[x] for x in word[1:-1]),
        )
    if sector.kind == "single-trace":
        if len(fixed_labels) != 1:
            return None
        anchor = fixed_labels[0]
        trace = tuple(int(label) for label in sector.trace_labels)
        if len(trace) != len(permuted_labels) + 1 or set(trace) != {
            anchor,
            *permuted_labels,
        }:
            return None
        anchor_index = trace.index(anchor)
        rotated = trace[anchor_index:] + trace[:anchor_index]
        try:
            coordinate = tuple(rank_by_label[label] for label in rotated[1:])
        except KeyError:
            return None
        return (
            ("single-trace", tuple(int(label) for label in sector.singlet_labels)),
            coordinate,
        )
    if sector.kind != "open-lines" or fixed_labels:
        return None

    ordered_lines = tuple(sorted(sector.open_color_lines, key=_open_line_sort_key))
    flattened = tuple(
        int(label) for line in ordered_lines for label in line.adjoint_labels
    )
    if len(flattened) != len(permuted_labels) or set(flattened) != set(permuted_labels):
        return None
    if _open_line_block_order(sector) is None:
        return None
    try:
        coordinate = tuple(rank_by_label[label] for label in flattened)
    except KeyError:
        return None
    line_skeleton = tuple(
        (
            int(line.fundamental_label),
            int(line.antifundamental_label),
            len(line.adjoint_labels),
            tuple(int(label) for label in line.singlet_labels),
        )
        for line in ordered_lines
    )
    return (
        (
            "open-lines",
            line_skeleton,
            tuple(int(label) for label in sector.singlet_labels),
        ),
        coordinate,
    )


def _open_line_sort_key(line: LCOpenColorLine) -> tuple[object, ...]:
    return (
        int(line.fundamental_label),
        int(line.antifundamental_label),
        tuple(int(label) for label in line.singlet_labels),
    )


def _open_line_block_order(
    sector: LCColorSector,
) -> tuple[tuple[int, int, tuple[int, ...]], ...] | None:
    lines = tuple(sector.open_color_lines)
    word = tuple(int(label) for label in sector.color_words[0])
    remaining = list(lines)
    order: list[tuple[int, int, tuple[int, ...]]] = []
    offset = 0
    while offset < len(word):
        matching = [
            line
            for line in remaining
            if word[offset : offset + len(line.coloured_labels)] == line.coloured_labels
        ]
        if len(matching) != 1:
            return None
        line = matching[0]
        order.append(
            (
                int(line.fundamental_label),
                int(line.antifundamental_label),
                tuple(int(label) for label in line.singlet_labels),
            )
        )
        offset += len(line.coloured_labels)
        remaining.remove(line)
    if remaining:
        return None
    return tuple(order)


def _validated_sector_owner_map(
    color_plan: GenericColorPlan,
    sector_owner_ids: Sequence[int] | None,
) -> tuple[int, ...]:
    sector_count = len(color_plan.sectors)
    owners = (
        tuple(range(sector_count))
        if sector_owner_ids is None
        else tuple(int(value) for value in sector_owner_ids)
    )
    if len(owners) != sector_count:
        raise ValueError("symmetric-group sector owner map has the wrong size")
    for sector_id, owner_id in enumerate(owners):
        if owner_id == _ZERO_SECTOR_OWNER:
            continue
        if owner_id < 0 or owner_id >= sector_count:
            raise ValueError("symmetric-group sector owner map is out of bounds")
        if owners[owner_id] != owner_id:
            raise ValueError("symmetric-group sector owner map is not idempotent")
        if owner_id > sector_id:
            raise ValueError("symmetric-group sector owner is not minimal")
    return owners


def _validate_owner_structures(
    color_plan: GenericColorPlan,
    owners: tuple[int, ...],
) -> None:
    sectors_by_id = {int(sector.id): sector for sector in color_plan.sectors}
    if set(sectors_by_id) != set(range(len(color_plan.sectors))):
        raise ValueError("symmetric-group color sectors are not densely numbered")
    for sector_id, owner_id in enumerate(owners):
        if owner_id == _ZERO_SECTOR_OWNER:
            continue
        if _owner_tensor_key(sectors_by_id[sector_id]) != _owner_tensor_key(
            sectors_by_id[owner_id]
        ):
            raise ValueError(
                "symmetric-group sector owner changes the canonical color tensor"
            )


def _owner_tensor_key(sector: LCColorSector) -> tuple[object, ...]:
    if sector.kind == "open-lines":
        return (
            "open-lines",
            tuple(
                sorted(
                    (
                        int(line.fundamental_label),
                        tuple(int(label) for label in line.adjoint_labels),
                        int(line.antifundamental_label),
                        tuple(int(label) for label in line.singlet_labels),
                    )
                    for line in sector.open_color_lines
                )
            ),
            tuple(int(label) for label in sector.singlet_labels),
        )
    if sector.kind == "single-trace":
        return (
            "single-trace",
            tuple(int(label) for label in sector.trace_labels),
            tuple(int(label) for label in sector.singlet_labels),
        )
    return (sector.kind, tuple(int(label) for label in sector.singlet_labels))


def _relative_permutation(
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> tuple[int, ...]:
    inverse = [0] * len(left)
    for position, label in enumerate(left):
        inverse[label] = position
    return tuple(inverse[label] for label in right)


def _inverse_permutation(permutation: tuple[int, ...]) -> tuple[int, ...]:
    inverse = [0] * len(permutation)
    for position, label in enumerate(permutation):
        inverse[label] = position
    return tuple(inverse)


def _single_trace_diagonal_kernel_representative(
    permutation: tuple[int, ...],
) -> tuple[int, ...]:
    """Canonicalize one anchored trace under cyclic relabelling and inversion."""

    trace_length = len(permutation) + 1
    anchored_word = (0, *(value + 1 for value in permutation))
    candidates: list[tuple[int, ...]] = []
    for shift in range(trace_length):
        relabelled = tuple((value + shift) % trace_length for value in anchored_word)
        anchor_index = relabelled.index(0)
        rotated = relabelled[anchor_index:] + relabelled[:anchor_index]
        coordinate = tuple(value - 1 for value in rotated[1:])
        candidates.extend((coordinate, _inverse_permutation(coordinate)))
    return min(candidates)


def _hermiticity_relative_indices(
    degree: int,
    *,
    channel_count: int,
) -> tuple[int, ...]:
    if channel_count < 2:
        return ()
    group_order = _symmetric_group_order(degree)
    if degree <= 4:
        return tuple(range(group_order))
    samples: list[tuple[int, ...]] = [tuple(range(degree))]
    for left_position in range(degree - 1):
        adjacent = list(range(degree))
        adjacent[left_position], adjacent[left_position + 1] = (
            adjacent[left_position + 1],
            adjacent[left_position],
        )
        samples.append(tuple(adjacent))
    non_involution = list(range(degree))
    non_involution[:3] = (1, 2, 0)
    samples.append(tuple(non_involution))
    return tuple(
        dict.fromkeys(
            _lexicographic_permutation_rank(sample, degree=degree) for sample in samples
        )
    )


def _certify_cross_channel_hermiticity(
    color_plan: GenericColorPlan,
    partition: SymmetricGroupOrbitPartition,
    *,
    sector_by_id: dict[int, LCColorSector],
    accuracy: str,
    weight_fraction: Fraction,
    kernel_exact_by_key: dict[tuple[int, int, int], Fraction],
    relative_indices: tuple[int, ...],
) -> None:
    if not relative_indices:
        return
    inverse_indices = tuple(
        (
            relative_index,
            _lexicographic_permutation_rank(
                _inverse_permutation(
                    _lexicographic_permutation_unrank(
                        partition.degree,
                        relative_index,
                    )
                ),
                degree=partition.degree,
            ),
        )
        for relative_index in relative_indices
    )
    for left_channel, left_orbit in enumerate(partition.orbits):
        for right_channel in range(left_channel + 1, len(partition.orbits)):
            right_orbit = partition.orbits[right_channel]
            for relative_index, inverse_index in inverse_indices:
                reverse = weight_fraction * exact_color_contraction_factor(
                    color_plan,
                    sector_by_id[right_orbit.sector_ids[0]],
                    sector_by_id[left_orbit.sector_ids[inverse_index]],
                    accuracy=accuracy,
                    full_col_acc=20,
                )
                if (
                    kernel_exact_by_key[(left_channel, right_channel, relative_index)]
                    != reverse
                ):
                    raise ValueError(
                        "symmetric-group cross-channel kernel failed its exact "
                        "Hermiticity certificate"
                    )


def _certify_sampled_equivariance(
    color_plan: GenericColorPlan,
    partition: SymmetricGroupOrbitPartition,
    *,
    sector_by_id: dict[int, LCColorSector],
    accuracy: str,
    weight_fraction: Fraction,
    kernel_exact_by_key: dict[tuple[int, int, int], Fraction],
) -> None:
    """Check exact samples of the structurally proved convolution identity.

    Orbit discovery strips every permuted adjoint label to its rank while
    retaining the complete non-adjoint trace/open-line skeleton.  A complete
    coordinate bucket is therefore a bijective simultaneous relabelling by
    ``S_m``.  Existing SU(3) contractions are invariant under that dummy-label
    relabelling, which proves dependence on ``p^-1 q``.  These deterministic
    non-identity samples guard the coordinate/composition implementation
    without an ``O(channel_count**2 * |S_m|**2)`` dense certificate.
    """

    left_action = list(range(partition.degree))
    left_action[0], left_action[1] = left_action[1], left_action[0]
    left_action_tuple = tuple(left_action)
    left_action_index = _lexicographic_permutation_rank(
        left_action_tuple,
        degree=partition.degree,
    )
    relative_indices = _equivariance_sample_indices(partition.degree)
    right_indices = tuple(
        (
            relative_index,
            _lexicographic_permutation_rank(
                tuple(
                    left_action_tuple[position]
                    for position in _lexicographic_permutation_unrank(
                        partition.degree,
                        relative_index,
                    )
                ),
                degree=partition.degree,
            ),
        )
        for relative_index in relative_indices
    )
    for left_channel, left_orbit in enumerate(partition.orbits):
        left_sector = sector_by_id[left_orbit.sector_ids[left_action_index]]
        for right_channel in range(left_channel, len(partition.orbits)):
            right_orbit = partition.orbits[right_channel]
            for relative_index, right_permutation_index in right_indices:
                exact = weight_fraction * exact_color_contraction_factor(
                    color_plan,
                    left_sector,
                    sector_by_id[right_orbit.sector_ids[right_permutation_index]],
                    accuracy=accuracy,
                    full_col_acc=20,
                )
                if (
                    kernel_exact_by_key[(left_channel, right_channel, relative_index)]
                    != exact
                ):
                    raise ValueError(
                        "symmetric-group kernel failed its exact sampled "
                        "equivariance certificate"
                    )


def _equivariance_sample_indices(degree: int) -> tuple[int, ...]:
    _symmetric_group_order(degree)
    samples: list[tuple[int, ...]] = [tuple(range(degree))]
    for left_position in range(degree - 1):
        adjacent = list(range(degree))
        adjacent[left_position], adjacent[left_position + 1] = (
            adjacent[left_position + 1],
            adjacent[left_position],
        )
        samples.append(tuple(adjacent))
    if degree >= 3:
        non_involution = list(range(degree))
        non_involution[:3] = (1, 2, 0)
        samples.append(tuple(non_involution))
    return tuple(
        dict.fromkeys(
            _lexicographic_permutation_rank(sample, degree=degree) for sample in samples
        )
    )


__all__ = [
    "CertifiedSymmetricGroupOrbit",
    "SymmetricGroupOrbitPartition",
    "build_symmetric_group_color_contraction_plan",
    "certify_symmetric_group_orbits",
    "reconstruct_symmetric_group_dense_exact",
]

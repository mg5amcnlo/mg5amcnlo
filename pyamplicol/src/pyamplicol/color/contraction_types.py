# SPDX-License-Identifier: 0BSD
"""Frozen color-contraction records."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction

NC = 3


@dataclass(frozen=True)
class ColorContractionEntry:
    left_group_id: int
    right_group_id: int
    weight_re: float
    weight_im: float = 0.0
    symmetry_factor: float = 1.0
    exact_weight: tuple[Fraction, Fraction] | None = None

    def to_json_dict(self) -> dict[str, object]:
        result = {
            "left_group_id": self.left_group_id,
            "right_group_id": self.right_group_id,
            "weight": [self.weight_re, self.weight_im],
            "symmetry_factor": self.symmetry_factor,
        }
        if self.exact_weight is not None:
            result["exact_weight"] = _exact_weight_payload(self.exact_weight)
        return result


@dataclass(frozen=True)
class ColorContractionTemplateEntry:
    left_group_index: int
    right_group_index: int
    weight_re: float
    weight_im: float = 0.0
    symmetry_factor: float = 1.0
    exact_weight: tuple[Fraction, Fraction] | None = None

    def to_json_dict(self) -> dict[str, object]:
        result = {
            "left_group_index": self.left_group_index,
            "right_group_index": self.right_group_index,
            "weight": [self.weight_re, self.weight_im],
            "symmetry_factor": self.symmetry_factor,
        }
        if self.exact_weight is not None:
            result["exact_weight"] = _exact_weight_payload(self.exact_weight)
        return result


def _exact_weight_payload(weight: tuple[Fraction, Fraction]) -> list[str]:
    # Strings keep arbitrarily large integers exact in every JSON consumer.
    return [
        str(part) for value in weight for part in (value.numerator, value.denominator)
    ]


@dataclass(frozen=True)
class FactorizedColorContractionBlock:
    """A compact transform plan for one repeated color matrix.

    ``klein-four-walsh`` records a free action of a Klein-four subgroup.  Each
    coset is ordered as identity, first generator, second generator, and their
    product.  The runtime validates the matrix invariance before using the
    corresponding four-point Walsh transform.

    ``elementary-abelian-walsh`` generalizes the same representation to a
    free ``C2**rank`` action.  Each coset is ordered by generator bitmask, so
    group multiplication is bitwise XOR.
    """

    kind: str
    cosets: tuple[tuple[int, ...], ...]
    rank: int | None = None

    def __post_init__(self) -> None:
        if not self.cosets:
            raise ValueError("factorized color contraction coset map is empty")
        if self.kind == "klein-four-walsh":
            if self.rank is not None:
                raise ValueError("Klein-four color factorization cannot declare rank")
            expected_coset_size = 4
        elif self.kind == "elementary-abelian-walsh":
            if self.rank is None or self.rank < 3:
                raise ValueError(
                    "elementary-Abelian color factorization requires rank >= 3"
                )
            expected_coset_size = len(self.cosets[0])
            if (
                expected_coset_size & (expected_coset_size - 1)
                or expected_coset_size.bit_length() - 1 != self.rank
            ):
                raise ValueError(
                    "elementary-Abelian color factorization rank is inconsistent"
                )
        else:
            raise ValueError(f"unknown color contraction factorization {self.kind!r}")
        if any(len(coset) != expected_coset_size for coset in self.cosets):
            raise ValueError(
                "factorized color contraction coset size does not match its rank"
            )
        flattened = tuple(index for coset in self.cosets for index in coset)
        if any(index < 0 for index in flattened):
            raise ValueError(
                "factorized color contraction coset map has a negative index"
            )
        if len(set(flattened)) != len(flattened):
            raise ValueError(
                "factorized color contraction coset map contains duplicate indices"
            )

    def to_json_dict(self) -> dict[str, object]:
        cosets = [list(coset) for coset in self.cosets]
        if self.rank is None:
            return {"kind": self.kind, "cosets": cosets}
        return {"kind": self.kind, "rank": self.rank, "cosets": cosets}


@dataclass(frozen=True)
class RepeatedColorContractionBlock:
    component_count: int
    component_group_ids: tuple[int, ...]
    entries: tuple[ColorContractionTemplateEntry, ...]
    factorized_block: FactorizedColorContractionBlock | None = None

    def __post_init__(self) -> None:
        if self.component_count < 2:
            raise ValueError(
                "repeated color contraction requires at least two components"
            )
        if not self.component_group_ids:
            raise ValueError("repeated color contraction group map is empty")
        if len(self.component_group_ids) % self.component_count != 0:
            raise ValueError(
                "repeated color contraction group IDs do not form a rectangular map"
            )
        if len(set(self.component_group_ids)) != len(self.component_group_ids):
            raise ValueError("repeated color contraction group IDs are not unique")
        local_group_count = self.local_group_count
        if any(
            entry.left_group_index < 0
            or entry.left_group_index >= local_group_count
            or entry.right_group_index < 0
            or entry.right_group_index >= local_group_count
            for entry in self.entries
        ):
            raise ValueError(
                "repeated color contraction entry references an unknown local group"
            )
        if self.factorized_block is not None:
            flattened = tuple(
                index for coset in self.factorized_block.cosets for index in coset
            )
            if sorted(flattened) != list(range(local_group_count)):
                raise ValueError(
                    "factorized color contraction cosets do not partition local groups"
                )

    @property
    def local_group_count(self) -> int:
        return len(self.component_group_ids) // self.component_count

    def to_json_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "component_count": self.component_count,
            "component_group_ids": list(self.component_group_ids),
            "entries": [entry.to_json_dict() for entry in self.entries],
        }
        if self.factorized_block is not None:
            result["factorized_block"] = self.factorized_block.to_json_dict()
        return result


@dataclass(frozen=True)
class SymmetricGroupColorContractionBlock:
    """Certified regular symmetric-group orbits plus exact residual metric.

    Local groups are ordered channel-major and permutation-major.  Every
    channel coset contains the lexicographic permutations of ``range(degree)``
    with the identity at offset zero.  Groups outside certified ``S_degree``
    orbits form one consecutive residual suffix.  ``kernel_entries`` stores
    exactly one complete relative-permutation row for every upper-triangular
    channel pair; it is not an expanded color matrix.

    The component map follows the existing local-group-major/component-minor
    convention.  Exact weights are parallel to their binary64 template entries
    and exclude the entry's upper-triangle symmetry factor.
    """

    degree: int
    component_count: int
    component_group_ids: tuple[int, ...]
    local_sector_ids: tuple[int, ...]
    channel_cosets: tuple[tuple[int, ...], ...]
    kernel_entries: tuple[ColorContractionTemplateEntry, ...]
    kernel_exact_weights: tuple[Fraction, ...]
    residual_entries: tuple[ColorContractionTemplateEntry, ...]
    residual_exact_weights: tuple[Fraction, ...]
    residual_local_group_indices: tuple[int, ...]
    hermiticity_check_mode: str = "vacuous"
    hermiticity_relative_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.degree, bool) or not isinstance(self.degree, int):
            raise ValueError("symmetric-group degree must be an integer")
        if self.degree < 2:
            raise ValueError("symmetric-group degree must be at least two")
        if self.degree > 10:
            raise ValueError("symmetric-group degree exceeds the supported maximum 10")
        if self.component_count < 1:
            raise ValueError("symmetric-group color contraction requires a component")
        if not self.component_group_ids:
            raise ValueError("symmetric-group component group map is empty")
        if len(self.component_group_ids) % self.component_count:
            raise ValueError(
                "symmetric-group component group IDs do not form a rectangular map"
            )
        if len(set(self.component_group_ids)) != len(self.component_group_ids):
            raise ValueError("symmetric-group component group IDs contain duplicates")
        if any(group_id < 0 for group_id in self.component_group_ids):
            raise ValueError(
                "symmetric-group component group IDs contain a negative ID"
            )

        local_group_count = self.local_group_count
        if len(self.local_sector_ids) != local_group_count:
            raise ValueError(
                "symmetric-group local sector map does not match local groups"
            )
        if len(set(self.local_sector_ids)) != len(self.local_sector_ids) or any(
            sector_id < 0 for sector_id in self.local_sector_ids
        ):
            raise ValueError(
                "symmetric-group local sector map must contain unique nonnegative IDs"
            )
        group_order = math.factorial(self.degree)
        if not self.channel_cosets:
            raise ValueError("symmetric-group color contraction has no channels")
        eligible_group_count = len(self.channel_cosets) * group_order
        if eligible_group_count > local_group_count:
            raise ValueError(
                "symmetric-group channel cosets exceed the local group domain"
            )
        for channel_index, coset in enumerate(self.channel_cosets):
            start = channel_index * group_order
            if coset != tuple(range(start, start + group_order)):
                raise ValueError(
                    "symmetric-group channel cosets must be consecutive and "
                    "channel-major"
                )
        expected_residual = tuple(range(eligible_group_count, local_group_count))
        if self.residual_local_group_indices != expected_residual:
            raise ValueError(
                "symmetric-group residual groups must be the consecutive "
                "complement after channel cosets"
            )

        expected_kernel_count = (
            len(self.channel_cosets) * (len(self.channel_cosets) + 1) // 2 * group_order
        )
        if len(self.kernel_entries) != expected_kernel_count:
            raise ValueError(
                "symmetric-group kernel rows do not cover every channel pair"
            )
        if len(self.kernel_exact_weights) != len(self.kernel_entries):
            raise ValueError(
                "symmetric-group exact kernel weights do not match kernel entries"
            )
        kernel_offset = 0
        for left_channel, left_coset in enumerate(self.channel_cosets):
            for right_channel in range(left_channel, len(self.channel_cosets)):
                right_coset = self.channel_cosets[right_channel]
                expected_symmetry = 1.0 if left_channel == right_channel else 2.0
                for relative_index in range(group_order):
                    entry = self.kernel_entries[kernel_offset]
                    if (
                        entry.left_group_index != left_coset[0]
                        or entry.right_group_index != right_coset[relative_index]
                        or entry.symmetry_factor != expected_symmetry
                    ):
                        raise ValueError(
                            "symmetric-group kernel rows are not in canonical "
                            "channel/relative-permutation order"
                        )
                    kernel_offset += 1
        self._validate_exact_entries(
            self.kernel_entries,
            self.kernel_exact_weights,
            local_group_count=local_group_count,
            label="kernel",
        )

        if len(self.residual_exact_weights) != len(self.residual_entries):
            raise ValueError(
                "symmetric-group exact residual weights do not match residual entries"
            )
        expected_residual_pairs = (
            tuple(
                (left_index, right_index)
                for left_index in range(local_group_count)
                for right_index in range(left_index, local_group_count)
                if right_index >= eligible_group_count
            )
            if self.residual_local_group_indices
            else ()
        )
        residual_pairs = tuple(
            (entry.left_group_index, entry.right_group_index)
            for entry in self.residual_entries
        )
        if residual_pairs != expected_residual_pairs:
            raise ValueError(
                "symmetric-group residual rows must exhaust every canonical pair "
                "touching the residual suffix"
            )
        for entry in self.residual_entries:
            pair = (entry.left_group_index, entry.right_group_index)
            expected_symmetry = 1.0 if pair[0] == pair[1] else 2.0
            if entry.symmetry_factor != expected_symmetry:
                raise ValueError(
                    "symmetric-group residual entry has an invalid symmetry factor"
                )
        self._validate_exact_entries(
            self.residual_entries,
            self.residual_exact_weights,
            local_group_count=local_group_count,
            label="residual",
        )

        cross_channel_count = self.channel_count * (self.channel_count - 1) // 2
        if cross_channel_count == 0:
            if (
                self.hermiticity_check_mode != "vacuous"
                or self.hermiticity_relative_indices
            ):
                raise ValueError(
                    "single-channel symmetric-group Hermiticity provenance must "
                    "be vacuous"
                )
        elif self.degree <= 4:
            if (
                self.hermiticity_check_mode != "full"
                or self.hermiticity_relative_indices != tuple(range(group_order))
            ):
                raise ValueError(
                    "small symmetric-group cross-channel kernels require a full "
                    "Hermiticity certificate"
                )
        elif (
            self.hermiticity_check_mode != "deterministic-samples"
            or not self.hermiticity_relative_indices
            or self.hermiticity_relative_indices[0] != 0
            or len(set(self.hermiticity_relative_indices))
            != len(self.hermiticity_relative_indices)
            or any(
                index < 0 or index >= group_order
                for index in self.hermiticity_relative_indices
            )
        ):
            raise ValueError(
                "large symmetric-group cross-channel kernels require canonical "
                "sampled Hermiticity provenance"
            )

    @staticmethod
    def _validate_exact_entries(
        entries: tuple[ColorContractionTemplateEntry, ...],
        exact_weights: tuple[Fraction, ...],
        *,
        local_group_count: int,
        label: str,
    ) -> None:
        for index, (entry, exact) in enumerate(
            zip(entries, exact_weights, strict=True)
        ):
            if not isinstance(exact, Fraction):
                raise ValueError(
                    f"symmetric-group {label} exact weight {index} is not a Fraction"
                )
            if (
                entry.left_group_index < 0
                or entry.left_group_index >= local_group_count
                or entry.right_group_index < 0
                or entry.right_group_index >= local_group_count
            ):
                raise ValueError(
                    f"symmetric-group {label} entry references an unknown local group"
                )
            if entry.weight_im != 0.0:
                raise ValueError(
                    f"symmetric-group {label} entry has a complex color weight"
                )
            if not math.isfinite(entry.weight_re) or float(exact) != entry.weight_re:
                raise ValueError(
                    f"symmetric-group {label} binary64 and exact weights disagree"
                )

    @property
    def local_group_count(self) -> int:
        return len(self.component_group_ids) // self.component_count

    @property
    def group_order(self) -> int:
        return math.factorial(self.degree)

    @property
    def channel_count(self) -> int:
        return len(self.channel_cosets)

    @property
    def stored_entry_count(self) -> int:
        return len(self.kernel_entries) + len(self.residual_entries)

    @property
    def hermiticity_check_count(self) -> int:
        return (
            self.channel_count
            * (self.channel_count - 1)
            // 2
            * len(self.hermiticity_relative_indices)
        )

    def to_json_dict(self) -> dict[str, object]:
        def exact_payload(value: Fraction) -> list[int]:
            return [value.numerator, value.denominator]

        return {
            "kind": "symmetric-group-fourier",
            "degree": self.degree,
            "component_count": self.component_count,
            "component_group_ids": list(self.component_group_ids),
            "local_sector_ids": list(self.local_sector_ids),
            "channel_cosets": [list(coset) for coset in self.channel_cosets],
            "kernel_entries": [entry.to_json_dict() for entry in self.kernel_entries],
            "kernel_exact_weights": [
                exact_payload(value) for value in self.kernel_exact_weights
            ],
            "residual_entries": [
                entry.to_json_dict() for entry in self.residual_entries
            ],
            "residual_exact_weights": [
                exact_payload(value) for value in self.residual_exact_weights
            ],
            "residual_local_group_indices": list(self.residual_local_group_indices),
            "hermiticity_certificate": {
                "mode": self.hermiticity_check_mode,
                "relative_indices": list(self.hermiticity_relative_indices),
                "check_count": self.hermiticity_check_count,
            },
        }


@dataclass(frozen=True)
class ColorContractionPlan:
    color_accuracy: str
    supported: bool
    reason: str | None
    group_count: int
    entries: tuple[ColorContractionEntry, ...]
    repeated_block: RepeatedColorContractionBlock | None = None
    symmetric_group_block: SymmetricGroupColorContractionBlock | None = None
    destination_by_group: tuple[int, ...] | None = None
    includes_color_factor: bool = True

    def __post_init__(self) -> None:
        storage_count = sum(
            (
                bool(self.entries),
                self.repeated_block is not None,
                self.symmetric_group_block is not None,
            )
        )
        if storage_count > 1:
            raise ValueError(
                "color contraction cannot mix expanded, repeated, and "
                "symmetric-group storage"
            )
        if (
            self.repeated_block is not None
            and len(self.repeated_block.component_group_ids) != self.group_count
        ):
            raise ValueError(
                "repeated color contraction group map does not match group count"
            )
        if (
            self.symmetric_group_block is not None
            and len(self.symmetric_group_block.component_group_ids) != self.group_count
        ):
            raise ValueError(
                "symmetric-group color contraction group map does not match group count"
            )
        if self.destination_by_group is not None:
            if len(self.destination_by_group) != self.group_count:
                raise ValueError(
                    "color contraction destination projection does not match "
                    "group count"
                )
            if len(set(self.destination_by_group)) != len(
                self.destination_by_group
            ) or any(value < 0 for value in self.destination_by_group):
                raise ValueError(
                    "color contraction destination projection must contain "
                    "unique nonnegative IDs"
                )

    @property
    def logical_entry_count(self) -> int:
        if self.repeated_block is None:
            if self.symmetric_group_block is None:
                return len(self.entries)
            return (
                self.symmetric_group_block.component_count
                * self.symmetric_group_block.stored_entry_count
            )
        return self.repeated_block.component_count * len(self.repeated_block.entries)

    def iter_logical_entries(self) -> Iterator[ColorContractionEntry]:
        if self.symmetric_group_block is not None:
            raise ValueError(
                "symmetric-group kernel rows are not an expanded color matrix"
            )
        if self.repeated_block is None:
            yield from self.entries
            return
        block = self.repeated_block
        for component_index in range(block.component_count):
            for entry in block.entries:
                yield ColorContractionEntry(
                    left_group_id=block.component_group_ids[
                        entry.left_group_index * block.component_count + component_index
                    ],
                    right_group_id=block.component_group_ids[
                        entry.right_group_index * block.component_count
                        + component_index
                    ],
                    weight_re=entry.weight_re,
                    weight_im=entry.weight_im,
                    symmetry_factor=entry.symmetry_factor,
                    exact_weight=entry.exact_weight,
                )

    def to_json_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "kind": "pyamplicol-color-contraction-plan",
            "color_accuracy": self.color_accuracy,
            "supported": self.supported,
            "reason": self.reason,
            "group_count": self.group_count,
            "includes_color_factor": self.includes_color_factor,
            "entry_count": len(self.entries),
            "logical_entry_count": self.logical_entry_count,
            "storage": "upper-triangular sparse metric over coherent amplitude groups",
            "entries": [entry.to_json_dict() for entry in self.entries],
        }
        if self.repeated_block is not None:
            result["repeated_block"] = self.repeated_block.to_json_dict()
        if self.symmetric_group_block is not None:
            result["symmetric_group_block"] = self.symmetric_group_block.to_json_dict()
        if self.destination_by_group is not None:
            result["destination_by_group"] = list(self.destination_by_group)
        return result


@dataclass(frozen=True)
class ColorGroupDescriptor:
    group_id: int
    helicity_key: tuple[object, ...]
    sector_id: int
    word: tuple[int, ...]
    helicity_weight: float

# SPDX-License-Identifier: 0BSD
"""Color-contraction planning and public factor selection."""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from fractions import Fraction

from .contraction_factors import (
    _coloured_word,
    _common_helicity_weight,
    _is_open_line_pair,
    _multi_open_line_color_factors,
    _one_open_line_color_factors,
    _pure_adjoint_color_factor,
    _pure_adjoint_color_factors,
    _two_open_line_color_factors,
    exact_color_contraction_factor,
    exact_color_contraction_factors,
)
from .contraction_types import (
    ColorContractionEntry,
    ColorContractionPlan,
    ColorContractionTemplateEntry,
    ColorGroupDescriptor,
    FactorizedColorContractionBlock,
    RepeatedColorContractionBlock,
    SymmetricGroupColorContractionBlock,
)
from .plan import GenericColorPlan, LCColorSector


def build_color_contraction_plan(
    color_plan: GenericColorPlan,
    groups: Sequence[ColorGroupDescriptor],
) -> ColorContractionPlan | None:
    """Build the final colour contraction over coherent amplitude groups.

    LC artifacts keep the historical Rusticol diagonal reduction path.  NLC and
    full-colour artifacts attach an explicit sparse colour matrix whose entries
    include AmpliCol's colour factors, so no leading-colour scalar factor should
    be applied again at runtime.
    """

    accuracy = color_plan.color_accuracy
    if accuracy == "lc":
        return None
    if accuracy not in {"nlc", "full"}:
        return ColorContractionPlan(
            color_accuracy=accuracy,
            supported=False,
            reason=f"unknown colour accuracy {accuracy!r}",
            group_count=len(groups),
            entries=(),
        )
    descriptors = tuple(groups)
    sector_by_id = {sector.id: sector for sector in color_plan.sectors}
    descriptors_by_helicity: dict[tuple[object, ...], list[ColorGroupDescriptor]] = {}
    for descriptor in descriptors:
        descriptors_by_helicity.setdefault(descriptor.helicity_key, []).append(
            descriptor
        )
    components = tuple(
        tuple(component) for component in descriptors_by_helicity.values()
    )
    repeated_block = _build_repeated_color_contraction_block(
        color_plan,
        components,
        sector_by_id=sector_by_id,
        accuracy=accuracy,
    )
    if repeated_block is not None:
        return ColorContractionPlan(
            color_accuracy=accuracy,
            supported=True,
            reason=None,
            group_count=len(groups),
            entries=(),
            repeated_block=repeated_block,
        )

    entries: list[ColorContractionEntry] = []
    factor_cache: dict[tuple[int, int], Fraction] = {}
    for helicity_descriptors in descriptors_by_helicity.values():
        for left_offset, left in enumerate(helicity_descriptors):
            left_sector = sector_by_id.get(left.sector_id)
            if left_sector is None:
                return ColorContractionPlan(
                    color_accuracy=accuracy,
                    supported=False,
                    reason=f"missing colour sector {left.sector_id}",
                    group_count=len(groups),
                    entries=(),
                )
            for right in helicity_descriptors[left_offset:]:
                right_sector = sector_by_id.get(right.sector_id)
                if right_sector is None:
                    return ColorContractionPlan(
                        color_accuracy=accuracy,
                        supported=False,
                        reason=f"missing colour sector {right.sector_id}",
                        group_count=len(groups),
                        entries=(),
                    )
                sector_pair = (left.sector_id, right.sector_id)
                weight = factor_cache.get(sector_pair)
                if weight is None:
                    weight = exact_color_contraction_factor(
                        color_plan,
                        left_sector,
                        right_sector,
                        accuracy=accuracy,
                        full_col_acc=20,
                    )
                    factor_cache[sector_pair] = weight
                if abs(weight) <= 0.0:
                    continue
                symmetry = 1.0 if left.group_id == right.group_id else 2.0
                helicity_weight = _common_helicity_weight(left, right)
                entries.append(
                    ColorContractionEntry(
                        left_group_id=left.group_id,
                        right_group_id=right.group_id,
                        weight_re=helicity_weight * float(weight),
                        symmetry_factor=symmetry,
                        exact_weight=(Fraction(helicity_weight) * weight, Fraction(0)),
                    )
                )
    return ColorContractionPlan(
        color_accuracy=accuracy,
        supported=True,
        reason=None,
        group_count=len(groups),
        entries=tuple(entries),
    )


def _build_repeated_color_contraction_block(
    color_plan: GenericColorPlan,
    components: Sequence[Sequence[ColorGroupDescriptor]],
    *,
    sector_by_id: dict[int, LCColorSector],
    accuracy: str,
) -> RepeatedColorContractionBlock | None:
    if len(components) < 2 or not components[0]:
        return None
    sorted_sector_ids = tuple(
        sorted(descriptor.sector_id for descriptor in components[0])
    )
    if len(set(sorted_sector_ids)) != len(sorted_sector_ids):
        return None
    reference_weight_bits = _binary64_bits(components[0][0].helicity_weight)
    descriptors_by_component_and_sector: list[dict[int, ColorGroupDescriptor]] = []
    for component in components:
        component_sector_ids = tuple(
            sorted(descriptor.sector_id for descriptor in component)
        )
        if (
            component_sector_ids != sorted_sector_ids
            or len(set(component_sector_ids)) != len(component_sector_ids)
            or any(
                _binary64_bits(descriptor.helicity_weight) != reference_weight_bits
                for descriptor in component
            )
        ):
            return None
        descriptors_by_component_and_sector.append(
            {descriptor.sector_id: descriptor for descriptor in component}
        )

    component_group_ids = tuple(
        descriptors_by_sector[sector_id].group_id
        for sector_id in sorted_sector_ids
        for descriptors_by_sector in descriptors_by_component_and_sector
    )
    base_entries: list[ColorContractionTemplateEntry] = []
    helicity_weight = components[0][0].helicity_weight
    for left_group_index, left_sector_id in enumerate(sorted_sector_ids):
        left_sector = sector_by_id.get(left_sector_id)
        if left_sector is None:
            return None
        for right_group_index in range(left_group_index, len(sorted_sector_ids)):
            right_sector_id = sorted_sector_ids[right_group_index]
            right_sector = sector_by_id.get(right_sector_id)
            if right_sector is None:
                return None
            weight = exact_color_contraction_factor(
                color_plan,
                left_sector,
                right_sector,
                accuracy=accuracy,
                full_col_acc=20,
            )
            if abs(weight) <= 0.0:
                continue
            base_entries.append(
                ColorContractionTemplateEntry(
                    left_group_index=left_group_index,
                    right_group_index=right_group_index,
                    weight_re=helicity_weight * float(weight),
                    symmetry_factor=(
                        1.0 if left_group_index == right_group_index else 2.0
                    ),
                    exact_weight=(Fraction(helicity_weight) * weight, Fraction(0)),
                )
            )
    return RepeatedColorContractionBlock(
        component_count=len(components),
        component_group_ids=component_group_ids,
        entries=tuple(base_entries),
        factorized_block=_build_walsh_color_contraction_block(
            tuple(
                descriptors_by_component_and_sector[0][sector_id].word
                for sector_id in sorted_sector_ids
            ),
            tuple(base_entries),
        ),
    )


def _build_walsh_color_contraction_block(
    words: Sequence[Sequence[int]],
    entries: Sequence[ColorContractionTemplateEntry],
) -> FactorizedColorContractionBlock | None:
    """Recognize an exact free elementary-Abelian action on one color matrix."""

    if not words:
        return None
    canonical_words = tuple(tuple(word) for word in words)
    canonical_word = canonical_words[0]
    if len(set(canonical_word)) != len(canonical_word):
        return None
    if any(
        len(word) != len(canonical_word)
        or len(set(word)) != len(word)
        or set(word) != set(canonical_word)
        for word in canonical_words
    ):
        return None

    matrix: dict[tuple[int, int], float] = {}
    for entry in entries:
        if (
            entry.weight_im != 0.0
            or not math.isfinite(entry.weight_re)
            or not math.isfinite(entry.symmetry_factor)
        ):
            return None
        left, right = sorted((entry.left_group_index, entry.right_group_index))
        if (left, right) in matrix:
            return None
        weight = (
            entry.symmetry_factor * entry.weight_re
            if left == right
            else 0.5 * entry.symmetry_factor * entry.weight_re
        )
        if not math.isfinite(weight):
            return None
        matrix[(left, right)] = weight

    def matrix_value(left: int, right: int) -> float:
        return matrix.get((min(left, right), max(left, right)), 0.0)

    def matrix_is_invariant(action: tuple[int, ...]) -> bool:
        return all(
            matrix_value(action[left], action[right]) == weight
            for (left, right), weight in matrix.items()
        )

    def action_cosets(
        actions: Sequence[tuple[int, ...]],
    ) -> tuple[tuple[int, ...], ...] | None:
        if len(actions) < 2:
            return None
        expected_indices = list(range(len(canonical_words)))
        for action in actions:
            if sorted(action) != expected_indices or any(
                action[action[index]] != index for index in expected_indices
            ):
                return None
        for first_offset, first_action in enumerate(actions):
            for second_action in actions[first_offset + 1 :]:
                if any(
                    first_action[second_action[index]]
                    != second_action[first_action[index]]
                    for index in expected_indices
                ):
                    return None

        group_actions: list[tuple[int, ...]] = []
        for mask in range(1 << len(actions)):
            action = tuple(expected_indices)
            for generator_index, generator in enumerate(actions):
                if mask & (1 << generator_index):
                    action = tuple(generator[index] for index in action)
            group_actions.append(action)

        remaining = set(expected_indices)
        cosets: list[tuple[int, ...]] = []
        while remaining:
            representative = min(remaining)
            coset = tuple(action[representative] for action in group_actions)
            if len(set(coset)) != len(group_actions) or not set(coset).issubset(
                remaining
            ):
                return None
            cosets.append(coset)
            remaining.difference_update(coset)
        return tuple(cosets)

    def matrix_has_walsh_blocks(
        cosets: Sequence[tuple[int, ...]],
    ) -> bool:
        if not cosets:
            return False
        group_size = len(cosets[0])
        if group_size < 4 or any(len(coset) != group_size for coset in cosets):
            return False
        return all(
            matrix_value(
                left_coset[left_subgroup_index],
                right_coset[right_subgroup_index],
            )
            == matrix_value(
                left_coset[0],
                right_coset[left_subgroup_index ^ right_subgroup_index],
            )
            for left_coset in cosets
            for right_coset in cosets
            for left_subgroup_index in range(group_size)
            for right_subgroup_index in range(group_size)
        )

    def factorized_block(
        actions: Sequence[tuple[int, ...]],
    ) -> FactorizedColorContractionBlock | None:
        # H8 is the largest physically measured transform. Preserve K4 as the
        # fallback and do not emit wider transforms until realistic multi-coset
        # workloads demonstrate that their locality/transform cost is useful.
        for rank in range(min(len(actions), 3), 1, -1):
            cosets = action_cosets(actions[:rank])
            if cosets is None or not matrix_has_walsh_blocks(cosets):
                continue
            if rank == 2:
                return FactorizedColorContractionBlock(
                    kind="klein-four-walsh",
                    cosets=cosets,
                )
            return FactorizedColorContractionBlock(
                kind="elementary-abelian-walsh",
                rank=rank,
                cosets=cosets,
            )
        return None

    # Preserve the established plan for a complete scalar permutation orbit.
    positions = {
        label: tuple(word.index(label) for word in canonical_words)
        for label in canonical_word
    }
    variable_labels = tuple(
        label for label in canonical_word if len(set(positions[label])) > 1
    )
    variable_count = len(variable_labels)
    if variable_count >= 4 and len(canonical_words) == math.factorial(variable_count):
        rank_by_label = {
            label: rank for rank, label in enumerate(sorted(variable_labels))
        }
        permutations = tuple(
            tuple(rank_by_label[label] for label in word if label in rank_by_label)
            for word in canonical_words
        )
        expected_ranks = set(range(variable_count))
        index_by_permutation = {
            permutation: index for index, permutation in enumerate(permutations)
        }
        if all(
            set(permutation) == expected_ranks for permutation in permutations
        ) and len(index_by_permutation) == math.factorial(variable_count):
            substitutions = tuple(
                tuple(
                    pair_start + 1
                    if rank == pair_start
                    else pair_start
                    if rank == pair_start + 1
                    else rank
                    for rank in range(variable_count)
                )
                for pair_start in range(0, variable_count - 1, 2)
            )
            actions = tuple(
                tuple(
                    index_by_permutation[
                        tuple(substitution[rank] for rank in permutation)
                    ]
                    for permutation in permutations
                )
                for substitution in substitutions
            )
            block = factorized_block(actions)
            if block is not None:
                return block

    # Multi-open-line bases are often unions of permutation orbits rather than
    # one full symmetric-group orbit.  Discover exact label-swap automorphisms
    # of the actual word set and matrix; no particle or process identity is
    # assumed, and any incomplete or weight-asymmetric orbit falls back.
    index_by_word = {word: index for index, word in enumerate(canonical_words)}
    if len(index_by_word) != len(canonical_words):
        return None
    invariant_swaps: list[tuple[tuple[int, int], tuple[int, ...]]] = []
    ordered_labels = sorted(canonical_word)
    for left_offset, left_label in enumerate(ordered_labels):
        for right_label in ordered_labels[left_offset + 1 :]:
            substitution = {left_label: right_label, right_label: left_label}
            try:
                action = tuple(
                    index_by_word[
                        tuple(substitution.get(label, label) for label in word)
                    ]
                    for word in canonical_words
                )
            except KeyError:
                continue
            if matrix_is_invariant(action):
                invariant_swaps.append(((left_label, right_label), action))

    selected_actions: list[tuple[int, ...]] = []
    selected_labels: set[int] = set()
    for labels, action in invariant_swaps:
        if selected_labels.isdisjoint(labels):
            selected_actions.append(action)
            selected_labels.update(labels)
    return factorized_block(selected_actions)


def _binary64_bits(value: float) -> bytes:
    return struct.pack(">d", float(value))


def color_contraction_factors(
    color_plan: GenericColorPlan,
    left: LCColorSector,
    right: LCColorSector,
    *,
    full_col_acc: int = 20,
) -> tuple[float, float, float]:
    """Return reference-normalized (LC, NLC, full) colour factors."""

    if color_plan.basis == "adjoint":
        return tuple(
            float(value)
            for value in exact_color_contraction_factors(
                color_plan, left, right, full_col_acc=full_col_acc
            )
        )
    open_line_count = color_plan.process.color_endpoints.pair_count
    n_ord = len(_coloured_word(left))
    if len(_coloured_word(right)) != n_ord:
        return (0.0, 0.0, 0.0)
    if open_line_count == 0:
        return _pure_adjoint_color_factors(left, right, n_ord, full_col_acc)
    if open_line_count == 1:
        return _one_open_line_color_factors(left, right, n_ord)
    if open_line_count == 2:
        return _two_open_line_color_factors(color_plan, left, right, n_ord)
    if open_line_count >= 3 and _is_open_line_pair(left, right):
        return _multi_open_line_color_factors(color_plan, left, right)
    return (0.0, 0.0, 0.0)


def color_contraction_factor(
    color_plan: GenericColorPlan,
    left: LCColorSector,
    right: LCColorSector,
    *,
    accuracy: str,
    full_col_acc: int = 20,
) -> float:
    """Return only the requested reference-normalized colour factor."""

    if color_plan.basis == "adjoint":
        return float(
            exact_color_contraction_factor(
                color_plan, left, right, accuracy=accuracy, full_col_acc=full_col_acc
            )
        )
    open_line_count = color_plan.process.color_endpoints.pair_count
    n_ord = len(_coloured_word(left))
    if len(_coloured_word(right)) != n_ord:
        return 0.0
    if open_line_count == 0:
        return _pure_adjoint_color_factor(
            left,
            right,
            n_ord,
            accuracy=accuracy,
            full_col_acc=full_col_acc,
        )
    values = color_contraction_factors(
        color_plan,
        left,
        right,
        full_col_acc=full_col_acc,
    )
    if accuracy == "lc":
        return values[0]
    if accuracy == "nlc":
        return values[1]
    if accuracy == "full":
        return values[2]
    return 0.0


__all__ = [
    "ColorContractionEntry",
    "ColorContractionPlan",
    "ColorContractionTemplateEntry",
    "ColorGroupDescriptor",
    "FactorizedColorContractionBlock",
    "RepeatedColorContractionBlock",
    "SymmetricGroupColorContractionBlock",
    "build_color_contraction_plan",
    "color_contraction_factor",
    "color_contraction_factors",
    "exact_color_contraction_factor",
    "exact_color_contraction_factors",
]

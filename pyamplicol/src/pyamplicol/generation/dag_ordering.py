# SPDX-License-Identifier: 0BSD
"""Color-order and closure-mask algorithms for generic DAG generation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from ..color.plan import GenericColorPlan, LCColorSector
from ..models._physics_ir import ContractionIR
from ..models.base import Model
from ..processes.ir import CanonicalProcessIR
from .dag_types import CurrentIndex

if TYPE_CHECKING:
    from .dag_color import ColorEngine


def _known_color_representation(model: Model, particle_id: int) -> int | None:
    """Return explicit model colour metadata without guessing from identity."""

    try:
        return int(model.color_rep(int(particle_id)))
    except (KeyError, NotImplementedError, TypeError, ValueError):
        return None


def _known_fermion_statistics(model: Model, particle_id: int) -> bool | None:
    """Return explicit model statistics, or ``None`` when they are unavailable."""

    try:
        return bool(model.is_fermion(int(particle_id)))
    except (KeyError, NotImplementedError, TypeError, ValueError):
        return None


def _direct_contraction_ir(
    model: Model,
    left: CurrentIndex,
    right: CurrentIndex,
) -> ContractionIR | None:
    return model.direct_contraction_ir(
        left.particle_id,
        right.particle_id,
        left_chirality=left.chirality,
        right_chirality=right.chirality,
    )


def _sector_group_indices_for_label(
    sector: LCColorSector,
    label: int,
) -> tuple[int, ...]:
    return tuple(
        index
        for index, group in enumerate(sector.line_label_groups)
        if label in set(group)
    )


def _line_positions(labels: Iterable[int], order: tuple[int, ...]) -> tuple[int, ...]:
    positions = {label: index for index, label in enumerate(order)}
    return tuple(positions[label] for label in labels if label in positions)


def _labels_projected_to_word(
    labels: Iterable[int],
    word: tuple[int, ...],
) -> tuple[int, ...]:
    word_labels = set(word)
    return tuple(label for label in labels if label in word_labels)


def _positions_contiguous(positions: tuple[int, ...]) -> bool:
    if not positions:
        return True
    return positions == tuple(range(positions[0], positions[-1] + 1))


def _word_contains_ordered_segment(
    word: tuple[int, ...],
    segment: tuple[int, ...],
) -> bool:
    if not segment or len(segment) > len(word):
        return False
    width = len(segment)
    return any(
        tuple(word[start : start + width]) == segment
        for start in range(len(word) - width + 1)
    )


def _ordered_combination_matches_word(
    left_labels: Iterable[int],
    right_labels: Iterable[int],
    word: tuple[int, ...],
) -> bool:
    return _ordered_combination_segment(left_labels, right_labels, word) is not None


def _closure_combination_matches_word(
    left_labels: Iterable[int],
    right_labels: Iterable[int],
    word: tuple[int, ...],
) -> bool:
    """Return whether two endpoint label sets close one colour word.

    Intermediate current construction must preserve the actual line order because
    it fixes where colour-singlet insertions may attach.  Final closure is
    different: the sink current can be oriented opposite to the sector's primary
    colour word, while still representing the same contiguous endpoint segment.
    Closure therefore checks contiguity of the projected coloured label sets, not
    their internal iteration order.
    """

    left_positions = tuple(sorted(_line_positions(left_labels, word)))
    right_positions = tuple(sorted(_line_positions(right_labels, word)))
    if not left_positions or not right_positions:
        return False
    if set(left_positions) & set(right_positions):
        return False
    union = tuple(sorted((*left_positions, *right_positions)))
    return (
        _positions_contiguous(left_positions)
        and _positions_contiguous(right_positions)
        and _positions_contiguous(union)
    )


def _shared_single_trace_closure_matches_word(
    left_labels: Iterable[int],
    right_labels: Iterable[int],
    word: tuple[int, ...],
) -> bool:
    left_projected = _labels_projected_to_word(left_labels, word)
    right_projected = _labels_projected_to_word(right_labels, word)
    return (*left_projected, *right_projected) == word


def _shared_lc_closure_matches_word(
    left_labels: Iterable[int],
    right_labels: Iterable[int],
    word: tuple[int, ...],
) -> bool:
    left_projected = _labels_projected_to_word(left_labels, word)
    right_projected = _labels_projected_to_word(right_labels, word)
    if not left_projected or not right_projected:
        return False
    return (*left_projected, *right_projected) == word


def _ordered_combination_segment(
    left_labels: Iterable[int],
    right_labels: Iterable[int],
    word: tuple[int, ...],
) -> tuple[int, ...] | None:
    left_positions = _line_positions(left_labels, word)
    right_positions = _line_positions(right_labels, word)
    if not left_positions and not right_positions:
        return ()
    if not left_positions or not right_positions:
        positions = left_positions or right_positions
        if not _positions_contiguous(positions):
            return None
        return tuple(word[positions[0] : positions[-1] + 1])
    union = tuple(sorted((*left_positions, *right_positions)))
    if (
        _positions_contiguous(left_positions)
        and _positions_contiguous(right_positions)
        and _positions_contiguous(union)
        and max(left_positions) < min(right_positions)
    ):
        return tuple(word[union[0] : union[-1] + 1])
    return None


def _lc_all_adjoint_symmetry_order_variants(
    left: tuple[int, ...],
    right: tuple[int, ...],
    *,
    left_all_adjoint: bool,
    right_all_adjoint: bool,
    result_reflection_proven: bool,
    fold_result_reflections: bool = True,
) -> tuple[tuple[tuple[int, ...], tuple[float, float]], ...]:
    """Return signed pure-adjoint LC symmetry variants."""

    n1 = len(left)
    n2 = len(right)
    switch1 = 2 if left_all_adjoint and n1 >= 2 else 1
    switch2 = 2 if right_all_adjoint and n2 >= 2 else 1
    switch3 = 2 if result_reflection_proven else 1
    variants: list[tuple[tuple[int, ...], tuple[float, float]]] = []
    for i in range(1, switch1 + 1):
        for j in range(1, switch2 + 1):
            for k in range(1, switch3 + 1):
                invert = 0
                if (i == 2 and k == 1) or (j == 2 and k == 2):
                    invert |= 1
                if (i == 2 and k == 2) or (j == 2 and k == 1):
                    invert |= 2
                if k == 1:
                    first = tuple(reversed(left)) if invert & 1 else left
                    second = tuple(reversed(right)) if invert & 2 else right
                else:
                    first = tuple(reversed(right)) if invert & 1 else right
                    second = tuple(reversed(left)) if invert & 2 else left
                proposed = (*first, *second)
                if (
                    result_reflection_proven
                    and fold_result_reflections
                    and not _lc_all_adjoint_symmetry_order_kept(proposed)
                ):
                    continue
                negative = (
                    (k == 2) ^ (j == 2 and n2 % 2 == 0) ^ (i == 2 and n1 % 2 == 0)
                )
                variants.append((proposed, (-1.0, 0.0) if negative else (1.0, 0.0)))
    return tuple(variants)


def _lc_all_adjoint_symmetry_order_kept(labels: tuple[int, ...]) -> bool:
    if not labels:
        return False
    min_label = min(labels)
    max_label = max(labels)
    return labels.index(min_label) < labels.index(max_label)


def _complex_weight_mul(
    left: tuple[float, float],
    right: tuple[float, float],
) -> tuple[float, float]:
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _canonical_lc_ordered_labels(
    labels: Iterable[int],
    sector: LCColorSector,
) -> tuple[int, ...]:
    label_tuple = tuple(labels)
    label_set = set(label_tuple)
    for word in _sector_intermediate_order_words(sector):
        positions = _line_positions(label_tuple, word)
        if not positions:
            continue
        if not _positions_contiguous(positions):
            continue
        word_labels = set(word)
        extras = tuple(sorted(label for label in label_set if label not in word_labels))
        segment = tuple(word[positions[0] : positions[-1] + 1])
        if not _line_local_singlet_extras_allowed(segment, extras, sector):
            continue
        return (*segment, *extras)
    return tuple(sorted(label_set))


def _sector_intermediate_order_words(
    sector: LCColorSector,
    *,
    include_compatible_traversals: bool = True,
    process: CanonicalProcessIR | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Return LC words used for intermediate current construction.

    A physical multi-line sector admits several block traversals.  Cyclically
    equivalent traversals describe the same closure, so normalize every word
    to the physical sector's authenticated closure sink and deduplicate it.
    These are private compatibility words, not public legacy amplitude-order
    representatives.  This retains the distinct cyclic classes needed by
    three or more open lines without creating duplicate physical sectors.

    Non-LC recursion keeps one sector-private traversal.  For open lines this
    is the authenticated canonical closure traversal, so intermediate currents
    and the final sink use the same orientation without changing the public
    physical colour word.
    """

    physical_words = sector.color_words
    if not include_compatible_traversals:
        if sector.kind == "open-lines" and process is not None:
            canonical_word = sector.canonical_closure_traversal_word(process)
            return (canonical_word,) if canonical_word else physical_words
        return physical_words or sector.admissible_traversal_words
    if sector.kind != "open-lines" or not physical_words or not physical_words[0]:
        return physical_words or sector.admissible_traversal_words

    sink_label = (
        int(physical_words[0][-1])
        if process is None
        else sector.canonical_closure_sink_label(process)
    )
    if sink_label is None:
        return physical_words
    words: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for raw_word in sector.admissible_traversal_words:
        normalized = tuple(int(label) for label in raw_word)
        if sink_label in normalized:
            sink_index = normalized.index(sink_label)
            word = (
                *normalized[sink_index + 1 :],
                *normalized[: sink_index + 1],
            )
        else:
            word = normalized
        if word in seen:
            continue
        seen.add(word)
        words.append(word)
    return tuple(words)


def _lc_word_with_sink_last(
    word: Iterable[int],
    sink_label: int | None,
) -> tuple[int, ...]:
    normalized = tuple(int(label) for label in word)
    if sink_label is None or sink_label not in normalized:
        return normalized
    sink_index = normalized.index(sink_label)
    rotated = (
        *normalized[sink_index + 1 :],
        *normalized[: sink_index + 1],
    )
    current_word = rotated[:-1]
    if current_word and not _lc_all_adjoint_symmetry_order_kept(current_word):
        current_word = tuple(reversed(current_word))
    return (*current_word, rotated[-1])


def _line_local_singlet_extras_allowed(
    colored_segment: Iterable[int],
    extras: Iterable[int],
    sector: LCColorSector,
) -> bool:
    extra_set = set(extras)
    if not extra_set:
        return True
    if sector.kind != "open-lines":
        return extra_set.issubset(set(sector.singlet_labels))
    return extra_set.issubset(set(sector.singlet_labels))


def _labels_mask(labels: Iterable[int]) -> int:
    mask = 0
    for label in labels:
        mask |= 1 << (label - 1)
    return mask


def _mask_labels(mask: int) -> tuple[int, ...]:
    return tuple(index + 1 for index in range(mask.bit_length()) if mask & (1 << index))


def _canonical_sink_mask(process_ir: CanonicalProcessIR, model: Model) -> int:
    fermion_labels = [
        leg.label
        for leg in process_ir.legs
        if leg.outgoing_pdg is not None
        and _known_fermion_statistics(model, int(leg.outgoing_pdg)) is True
    ]
    if fermion_labels:
        return 1 << (min(fermion_labels) - 1)
    labels = [leg.label for leg in process_ir.legs if leg.outgoing_pdg is not None]
    if not labels:
        return 0
    return 1 << (min(labels) - 1)


def _closure_candidate_splits(
    process_ir: CanonicalProcessIR,
    model: Model,
    color_engine: ColorEngine,
    *,
    reference_color_order: tuple[int, ...] | None = None,
) -> tuple[tuple[int, int], ...]:
    full_mask = _labels_mask(leg.label for leg in process_ir.legs)
    splits: list[tuple[int, int]] = []
    split_seen: set[tuple[int, int]] = set()
    sink_labels: list[int] = []
    if color_engine.shared_lc_fixed_sink_label is not None:
        sink_labels.append(color_engine.shared_lc_fixed_sink_label)
    sectors = color_engine.color_plan.sectors
    if not sink_labels and (
        color_engine.color_plan.color_accuracy == "lc"
        or color_engine.shared_single_trace
        or any(sector.kind == "open-lines" for sector in sectors)
    ):
        if reference_color_order is not None:
            selected_sectors = tuple(
                sector
                for sector in sectors
                if reference_color_order in sector.admissible_traversal_words
            )
            if selected_sectors:
                sectors = selected_sectors
        for sector in sectors:
            # A public reference word selects a physical sector, but its
            # terminal label is not a private recursion sink.  Final closure
            # always uses that sector's authenticated canonical traversal.
            sink_label = sector.canonical_closure_sink_label(process_ir)
            if sink_label is not None:
                sink_labels.append(sink_label)
    if not sink_labels and reference_color_order:
        leg_by_label = {leg.label: leg for leg in process_ir.legs}
        colored_reference_labels: list[int] = []
        ordered_reference_labels: list[int] = []
        for raw_label in reference_color_order:
            label = int(raw_label)
            if not (full_mask & (1 << (label - 1))):
                continue
            ordered_reference_labels.append(label)
            leg = leg_by_label.get(label)
            if leg is None or leg.outgoing_pdg is None:
                continue
            representation = _known_color_representation(
                model,
                int(leg.outgoing_pdg),
            )
            if representation is not None and representation != 1:
                colored_reference_labels.append(label)
        if colored_reference_labels:
            sink_labels.append(colored_reference_labels[-1])
        elif ordered_reference_labels:
            sink_labels.append(ordered_reference_labels[-1])
    if not sink_labels:
        fallback_sink_mask = _canonical_sink_mask(process_ir, model)
        fallback_sink_labels = _mask_labels(fallback_sink_mask)
        if fallback_sink_labels:
            sink_labels.append(fallback_sink_labels[0])
    seen: set[int] = set()
    for label in sink_labels:
        if label in seen:
            continue
        seen.add(label)
        sink_mask = 1 << (label - 1)
        if not (sink_mask & full_mask):
            continue
        split = (full_mask ^ sink_mask, sink_mask)
        if split in split_seen:
            continue
        split_seen.add(split)
        splits.append(split)
    return tuple(splits)


def _closure_side_reachable_masks(
    full_mask: int,
    candidate_splits: Iterable[tuple[int, int]],
) -> frozenset[int]:
    """Return current masks that can feed one configured amplitude closure.

    The generic forward sweep otherwise builds every locally valid current for
    every proper subset, then removes dead currents after closure.  AmpliCol's
    library generation is faster because it knows which endpoint side each
    current can ultimately feed.  This helper applies the same idea without
    naming a process family: once the generic closure splitter has selected the
    possible amplitude endpoint masks, any useful intermediate current must be
    a submask of one of those endpoint masks.
    """

    allowed: set[int] = set()
    for left_mask, right_mask in candidate_splits:
        for side_mask in (left_mask, right_mask):
            submask = side_mask & full_mask
            while submask:
                allowed.add(submask)
                submask = (submask - 1) & side_mask
    return frozenset(allowed)


def _lc_color_order_reachable_masks(
    process_ir: CanonicalProcessIR,
    color_plan: GenericColorPlan,
    model: Model,
) -> frozenset[int] | None:
    """Return subset masks compatible with at least one LC colour word.

    This is process-generic pruning.  It only uses model colour
    representations and the LC words produced by the colour planner.  Any
    useful coloured current in a colour-ordered recursion must cover a
    contiguous segment of one compatibility word.  Colour singlets are left as
    attachments to those segments because their allowed positions are governed
    by ordinary model vertices and the existing singlet-order rule.
    """

    if not color_plan.sectors:
        return None

    full_mask = _labels_mask(leg.label for leg in process_ir.legs)
    colored_labels: set[int] = set()
    singlet_labels: set[int] = set()
    for leg in process_ir.legs:
        if leg.outgoing_pdg is None:
            return None
        representation = _known_color_representation(model, int(leg.outgoing_pdg))
        if representation is None:
            return None
        if representation != 1:
            colored_labels.add(leg.label)
        else:
            singlet_labels.add(leg.label)

    allowed: set[int] = set()

    if not colored_labels:
        return frozenset(_nonzero_submasks(full_mask))

    for sector in color_plan.sectors:
        sector_colored_labels = {
            label for group in sector.coloured_label_groups for label in group
        }
        if sector_colored_labels != colored_labels:
            return None
        if set(sector.singlet_labels) != singlet_labels:
            return None
        if sector.kind == "open-lines":
            for singlet_submask in _submasks_for_labels(sector.singlet_labels):
                if singlet_submask:
                    allowed.add(singlet_submask)
        for raw_word in sector.admissible_traversal_words:
            word = tuple(label for label in raw_word if label in colored_labels)
            for start in range(len(word)):
                segment_mask = 0
                for stop in range(start, len(word)):
                    segment_mask |= 1 << (word[stop] - 1)
                    allowed.add(segment_mask)
                    segment_labels = tuple(word[start : stop + 1])
                    line_singlets = _line_local_singlet_labels_for_segment(
                        sector,
                        segment_labels,
                    )
                    for singlet_submask in _submasks_for_labels(line_singlets):
                        allowed.add(segment_mask | singlet_submask)

    for label in colored_labels:
        allowed.add(1 << (label - 1))
    for label in singlet_labels:
        allowed.add(1 << (label - 1))
    return frozenset(mask for mask in allowed if mask & full_mask)


def _line_local_singlet_labels_for_segment(
    sector: LCColorSector,
    colored_segment: Iterable[int],
) -> tuple[int, ...]:
    if sector.kind != "open-lines":
        return sector.singlet_labels
    return sector.singlet_labels


def _submasks_for_labels(labels: Iterable[int]) -> tuple[int, ...]:
    label_tuple = tuple(labels)
    masks: list[int] = []
    count = len(label_tuple)
    for bits in range(1 << count):
        mask = 0
        for index, label in enumerate(label_tuple):
            if bits & (1 << index):
                mask |= 1 << (label - 1)
        masks.append(mask)
    return tuple(masks)


def _nonzero_submasks(mask: int) -> tuple[int, ...]:
    submasks: list[int] = []
    submask = mask
    while submask:
        submasks.append(submask)
        submask = (submask - 1) & mask
    return tuple(submasks)

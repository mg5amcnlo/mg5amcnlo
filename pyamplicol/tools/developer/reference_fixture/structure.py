# SPDX-License-Identifier: 0BSD
"""Structural invariants for parsed reference fixtures."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from decimal import Decimal
from itertools import product
from typing import TypeVar

from .model import (
    ContractedColorComponent,
    HelicityAxis,
    LCColorFlow,
    Observation,
    Process,
    ReferenceCase,
    ReferenceFixtureError,
)
from .numerics import _as_fraction

_Item = TypeVar("_Item")


def _unique(
    items: Iterable[_Item],
    kind: str,
    *,
    key: str = "id",
) -> dict[str, _Item]:
    result: dict[str, _Item] = {}
    for item in items:
        item_id = getattr(item, key)
        if item_id in result:
            raise ReferenceFixtureError(f"duplicate {kind} id: {item_id}")
        result[item_id] = item
    return result


def _validate_digest(value: str, where: str) -> None:
    if value == "0" * 64:
        raise ReferenceFixtureError(f"{where} must not use an all-zero digest")


def _physical_helicity_domain(spin: int, mass: Decimal) -> tuple[int, ...]:
    if spin < 1:
        raise ReferenceFixtureError("external UFO spin codes must be positive")
    if mass < 0:
        raise ReferenceFixtureError("external particle masses must be non-negative")
    if spin == 1:
        return (0,)
    if spin % 2 == 0:
        return tuple(range(-(spin - 1), spin, 2))
    physical_spin = (spin - 1) // 2
    if mass == 0:
        return (-physical_spin, physical_spin)
    return tuple(range(-physical_spin, physical_spin + 1))


def _validate_processes(processes: Mapping[str, object]) -> None:
    typed = {
        key: value for key, value in processes.items() if isinstance(value, Process)
    }
    for process in typed.values():
        width = len(process.external_pdgs)
        if (
            len(process.external_labels) != width
            or len(process.external_leg_ids) != width
            or len(process.external_spins) != width
            or len(process.external_colors) != width
            or len(process.external_masses) != width
            or len(process.external_helicity_domains) != width
        ):
            raise ReferenceFixtureError(
                f"process {process.id} external identity/helicity metadata has "
                "the wrong width"
            )
        if len(set(process.external_labels)) != width or any(
            label < 1 for label in process.external_labels
        ):
            raise ReferenceFixtureError(
                f"process {process.id} external labels must be distinct and positive"
            )
        unsupported_colors = set(process.external_colors) - {1, 3, -3, 8}
        if unsupported_colors:
            raise ReferenceFixtureError(
                f"process {process.id} uses unsupported external color "
                f"representations {sorted(unsupported_colors)}"
            )
        expected_domains = tuple(
            _physical_helicity_domain(spin, mass)
            for spin, mass in zip(
                process.external_spins, process.external_masses, strict=True
            )
        )
        if process.external_helicity_domains != expected_domains:
            raise ReferenceFixtureError(
                f"process {process.id} helicity domains do not follow its "
                "model-derived spin and mass metadata"
            )
        if process.initial_state_count >= len(process.external_pdgs):
            raise ReferenceFixtureError(
                f"process {process.id} has no final-state particles"
            )
        if process.alias_of is None:
            if process.final_state_permutation is not None:
                raise ReferenceFixtureError(
                    f"primary process {process.id} must not define a final-state "
                    "permutation"
                )
            continue
        source = typed.get(process.alias_of)
        if source is None:
            raise ReferenceFixtureError(
                f"process {process.id} references unknown alias source "
                f"{process.alias_of}"
            )
        if process.id == source.id:
            raise ReferenceFixtureError(f"process {process.id} aliases itself")
        if process.initial_state_count != source.initial_state_count:
            raise ReferenceFixtureError(
                f"process {process.id} alias changes the initial-state width"
            )
        if len(process.external_pdgs) != len(source.external_pdgs):
            raise ReferenceFixtureError(
                f"process {process.id} alias changes the external-state width"
            )
        if process.external_labels != source.external_labels:
            raise ReferenceFixtureError(
                f"process {process.id} alias changes canonical external labels"
            )
        if (
            process.external_pdgs[: process.initial_state_count]
            != source.external_pdgs[: source.initial_state_count]
        ):
            raise ReferenceFixtureError(
                f"process {process.id} alias changes initial-state PDGs"
            )
        if (
            process.external_leg_ids[: process.initial_state_count]
            != source.external_leg_ids[: source.initial_state_count]
        ):
            raise ReferenceFixtureError(
                f"process {process.id} alias changes initial-state leg identities"
            )
        if (
            process.external_helicity_domains[: process.initial_state_count]
            != source.external_helicity_domains[: source.initial_state_count]
        ):
            raise ReferenceFixtureError(
                f"process {process.id} alias changes initial-state helicity domains"
            )
        for name, values, source_values in (
            ("spin", process.external_spins, source.external_spins),
            ("color", process.external_colors, source.external_colors),
            ("mass", process.external_masses, source.external_masses),
        ):
            if (
                values[: process.initial_state_count]
                != source_values[: source.initial_state_count]
            ):
                raise ReferenceFixtureError(
                    f"process {process.id} alias changes initial-state {name} metadata"
                )
        permutation = process.final_state_permutation
        final_count = len(source.external_pdgs) - source.initial_state_count
        if permutation is None or sorted(permutation) != list(range(final_count)):
            raise ReferenceFixtureError(
                f"process {process.id} has an invalid final-state permutation"
            )
        source_final = source.external_pdgs[source.initial_state_count :]
        expected_final = tuple(source_final[position] for position in permutation)
        if process.external_pdgs[process.initial_state_count :] != expected_final:
            raise ReferenceFixtureError(
                f"process {process.id} final-state permutation does not match its PDGs"
            )
        source_final_legs = source.external_leg_ids[source.initial_state_count :]
        expected_final_legs = tuple(
            source_final_legs[position] for position in permutation
        )
        if process.external_leg_ids[process.initial_state_count :] != (
            expected_final_legs
        ):
            raise ReferenceFixtureError(
                f"process {process.id} final-state permutation does not match "
                "its leg identities"
            )
        source_final_helicities = source.external_helicity_domains[
            source.initial_state_count :
        ]
        expected_final_helicities = tuple(
            source_final_helicities[position] for position in permutation
        )
        if process.external_helicity_domains[process.initial_state_count :] != (
            expected_final_helicities
        ):
            raise ReferenceFixtureError(
                f"process {process.id} final-state permutation does not match "
                "its helicity domains"
            )
        for name, values, source_values in (
            ("spin", process.external_spins, source.external_spins),
            ("color", process.external_colors, source.external_colors),
            ("mass", process.external_masses, source.external_masses),
        ):
            expected_values = tuple(
                source_values[source.initial_state_count :][position]
                for position in permutation
            )
            if values[process.initial_state_count :] != expected_values:
                raise ReferenceFixtureError(
                    f"process {process.id} final-state permutation does not match "
                    f"its {name} metadata"
                )

    for process in typed.values():
        seen: set[str] = set()
        current = process
        while current.alias_of is not None:
            if current.id in seen:
                raise ReferenceFixtureError(
                    f"process alias cycle includes {current.id}"
                )
            seen.add(current.id)
            current = typed[current.alias_of]


def _validate_case_axes(case: ReferenceCase, process: Process) -> None:
    helicity_ids = _unique(case.helicities, f"case {case.id} helicity")
    color_ids = _unique(case.colors, f"case {case.id} color")
    if [axis.index for axis in case.helicities] != list(range(len(case.helicities))):
        raise ReferenceFixtureError(
            f"case {case.id} helicity indices must be contiguous and ordered"
        )
    if [axis.index for axis in case.colors] != list(range(len(case.colors))):
        raise ReferenceFixtureError(
            f"case {case.id} color indices must be contiguous and ordered"
        )
    if case.coverage.helicity_count != len(case.helicities):
        raise ReferenceFixtureError(
            f"case {case.id} helicity axis count does not match coverage"
        )
    if case.coverage.color_component_count != len(case.colors):
        raise ReferenceFixtureError(
            f"case {case.id} color axis count does not match coverage"
        )
    if case.case_kind == "substantive":
        if case.coverage.helicities != "complete":
            raise ReferenceFixtureError(
                f"substantive case {case.id} requires complete helicity coverage"
            )
        expected_color_coverage = (
            "complete" if case.color_accuracy == "lc" else "contracted"
        )
        if case.coverage.color != expected_color_coverage:
            raise ReferenceFixtureError(
                f"substantive case {case.id} requires complete color coverage"
            )
    elif (
        case.coverage.helicities == "selected" or case.coverage.color == "selected"
    ) and case.case_kind != "diagnostic":
        raise ReferenceFixtureError(
            f"selected coverage is restricted to diagnostic case {case.id}"
        )
    helicity_values = [axis.values for axis in case.helicities]
    if len(set(helicity_values)) != len(helicity_values):
        raise ReferenceFixtureError(f"case {case.id} has duplicate helicity vectors")
    physical_helicities = set(product(*process.external_helicity_domains))
    if not set(helicity_values) <= physical_helicities:
        raise ReferenceFixtureError(
            f"case {case.id} contains helicities outside the process domains"
        )
    if case.coverage.helicities == "complete" and set(helicity_values) != (
        physical_helicities
    ):
        raise ReferenceFixtureError(
            f"case {case.id} does not contain the complete physical helicity axis"
        )
    for helicity in case.helicities:
        if len(helicity.values) != len(process.external_pdgs):
            raise ReferenceFixtureError(
                f"case {case.id} helicity {helicity.id} has the wrong width"
            )
        if helicity.representative_id not in helicity_ids:
            raise ReferenceFixtureError(
                f"case {case.id} helicity {helicity.id} has an unknown representative"
            )
        representative = helicity_ids[helicity.representative_id]
        assert isinstance(representative, HelicityAxis)
        if helicity.structural_zero:
            if (
                helicity.computed
                or helicity.coefficient != 0
                or helicity.representative_id != helicity.id
            ):
                raise ReferenceFixtureError(
                    f"case {case.id} structural-zero helicity {helicity.id} "
                    "must be uncomputed, self-represented, and have zero coefficient"
                )
        elif helicity.computed:
            if helicity.representative_id != helicity.id or helicity.coefficient != 1:
                raise ReferenceFixtureError(
                    f"case {case.id} computed helicity {helicity.id} must be its "
                    "own unit-coefficient representative"
                )
        elif (
            representative.structural_zero
            or not representative.computed
            or helicity.representative_id == helicity.id
            or helicity.coefficient <= 0
        ):
            raise ReferenceFixtureError(
                f"case {case.id} folded helicity {helicity.id} must reference a "
                "computed representative with a positive physical weight"
            )
    structural_zeros = sum(axis.structural_zero for axis in case.helicities)
    if structural_zeros != case.coverage.structural_zero_helicity_count:
        raise ReferenceFixtureError(
            f"case {case.id} structural-zero coverage count is inconsistent"
        )
    if not case.selectors.helicity:
        raise ReferenceFixtureError(
            f"case {case.id} must expose physical helicity selection"
        )

    if case.color_accuracy == "lc":
        if not all(isinstance(color, LCColorFlow) for color in case.colors):
            raise ReferenceFixtureError(
                f"LC case {case.id} must use only physical LC flow axes"
            )
        if (
            case.coverage.color_kind != "physical-lc-flows"
            or case.coverage.color
            not in {
                "complete",
                "selected",
            }
        ):
            raise ReferenceFixtureError(f"LC case {case.id} has invalid color coverage")
        if (
            not case.selectors.color_flow
            or case.selectors.omitted_color != "all-components"
        ):
            raise ReferenceFixtureError(
                f"LC case {case.id} has invalid color-flow selector semantics"
            )
        if (
            case.reduction.kind != "lc-diagonal"
            or case.reduction.cell_semantics != "sum-all-contributing-groups"
        ):
            raise ReferenceFixtureError(
                f"LC case {case.id} has invalid reduction semantics"
            )
        color_words = [
            color.word for color in case.colors if isinstance(color, LCColorFlow)
        ]
        if len(set(color_words)) != len(color_words):
            raise ReferenceFixtureError(f"case {case.id} has duplicate LC flow words")
        colored_labels = {
            label
            for label, color in zip(
                process.external_labels, process.external_colors, strict=True
            )
            if color != 1
        }
        for word in color_words:
            if len(set(word)) != len(word) or set(word) != colored_labels:
                raise ReferenceFixtureError(
                    f"case {case.id} LC flow word does not contain exactly the "
                    "model-derived colored external labels"
                )
        for color in case.colors:
            assert isinstance(color, LCColorFlow)
            if color.representative_id not in color_ids:
                raise ReferenceFixtureError(
                    f"case {case.id} color {color.id} has an unknown representative"
                )
            color_representative = color_ids[color.representative_id]
            assert isinstance(color_representative, LCColorFlow)
            if color.computed:
                if color.representative_id != color.id or color.coefficient != 1:
                    raise ReferenceFixtureError(
                        f"case {case.id} computed color {color.id} must be its "
                        "own unit-coefficient representative"
                    )
            elif (
                not color_representative.computed
                or color.representative_id == color.id
                or color.coefficient <= 0
            ):
                raise ReferenceFixtureError(
                    f"case {case.id} folded color {color.id} must reference a "
                    "computed representative with a positive physical weight"
                )
    else:
        if len(case.colors) != 1 or not isinstance(
            case.colors[0], ContractedColorComponent
        ):
            raise ReferenceFixtureError(
                f"{case.color_accuracy.upper()} case {case.id} must have exactly one "
                "contracted color axis"
            )
        if case.coverage.color != "contracted" or case.coverage.color_kind != (
            "contracted-color"
        ):
            raise ReferenceFixtureError(
                f"{case.color_accuracy.upper()} case {case.id} has invalid "
                "color coverage"
            )
        if case.selectors.color_flow or case.selectors.omitted_color != (
            "contracted-component"
        ):
            raise ReferenceFixtureError(
                f"{case.color_accuracy.upper()} case {case.id} has invalid "
                "selector semantics"
            )
        if (
            case.reduction.kind != "contracted-color"
            or case.reduction.cell_semantics != "fully-contracted-color"
        ):
            raise ReferenceFixtureError(
                f"{case.color_accuracy.upper()} case {case.id} has invalid "
                "reduction semantics"
            )
    _validate_reduction_groups(case, helicity_ids, color_ids)


def _validate_reduction_groups(
    case: ReferenceCase,
    helicity_ids: Mapping[str, object],
    color_ids: Mapping[str, object],
) -> None:
    groups = _unique(case.reduction.groups, f"case {case.id} reduction group")
    if len(groups) != case.topology.reduction_groups:
        raise ReferenceFixtureError(
            f"case {case.id} reduction group count does not match topology"
        )

    expected_cells = {
        (helicity.id, color.id)
        for helicity in case.helicities
        if not helicity.structural_zero
        for color in case.colors
    }
    covered_cells: set[tuple[str, str]] = set()
    for group in case.reduction.groups:
        if (
            not group.physical_helicity_ids
            or not group.physical_color_ids
            or len(set(group.physical_helicity_ids)) != len(group.physical_helicity_ids)
            or len(set(group.physical_color_ids)) != len(group.physical_color_ids)
        ):
            raise ReferenceFixtureError(
                f"case {case.id} reduction group {group.id} has empty or duplicate "
                "physical members"
            )
        if (
            group.representative_helicity_id not in group.physical_helicity_ids
            or group.representative_color_id not in group.physical_color_ids
        ):
            raise ReferenceFixtureError(
                f"case {case.id} reduction group {group.id} excludes its representative"
            )
        unknown_helicities = set(group.physical_helicity_ids) - helicity_ids.keys()
        unknown_colors = set(group.physical_color_ids) - color_ids.keys()
        if unknown_helicities or unknown_colors:
            raise ReferenceFixtureError(
                f"case {case.id} reduction group {group.id} references unknown axes"
            )

        representative_helicity = helicity_ids[group.representative_helicity_id]
        assert isinstance(representative_helicity, HelicityAxis)
        if not representative_helicity.computed:
            raise ReferenceFixtureError(
                f"case {case.id} reduction group {group.id} helicity "
                "representative is not computed"
            )
        for identifier in group.physical_helicity_ids:
            helicity = helicity_ids[identifier]
            assert isinstance(helicity, HelicityAxis)
            if helicity.structural_zero or (
                helicity.representative_id != group.representative_helicity_id
            ):
                raise ReferenceFixtureError(
                    f"case {case.id} reduction group {group.id} has an invalid "
                    f"helicity member {identifier}"
                )

        representative_color = color_ids[group.representative_color_id]
        if isinstance(representative_color, LCColorFlow) and not (
            representative_color.computed
        ):
            raise ReferenceFixtureError(
                f"case {case.id} reduction group {group.id} color representative "
                "is not computed"
            )
        for identifier in group.physical_color_ids:
            color = color_ids[identifier]
            if isinstance(color, LCColorFlow) and (
                color.representative_id != group.representative_color_id
            ):
                raise ReferenceFixtureError(
                    f"case {case.id} reduction group {group.id} has an invalid "
                    f"color member {identifier}"
                )

        group_cells = set(
            product(group.physical_helicity_ids, group.physical_color_ids)
        )
        # LC groups expand disjoint representative cells. Contracted NLC/full
        # outputs are many-to-one: several coherent groups can contribute to
        # the same public (helicity, contracted-color) cell.
        if case.color_accuracy == "lc" and covered_cells & group_cells:
            raise ReferenceFixtureError(
                f"case {case.id} reduction groups overlap on physical cells"
            )
        covered_cells.update(group_cells)

    if covered_cells != expected_cells:
        relation = "partition" if case.color_accuracy == "lc" else "cover"
        raise ReferenceFixtureError(
            f"case {case.id} reduction groups do not {relation} every nonzero "
            "physical cell"
        )


def _validate_observation_reduction_relations(
    case: ReferenceCase,
    observation: Observation,
) -> None:
    helicity_indices = {axis.id: index for index, axis in enumerate(case.helicities)}
    color_indices = {axis.id: index for index, axis in enumerate(case.colors)}
    for helicity in case.helicities:
        if helicity.computed or helicity.structural_zero:
            continue
        member_index = helicity_indices[helicity.id]
        representative_index = helicity_indices[helicity.representative_id]
        coefficient = _as_fraction(helicity.coefficient)
        for member, representative in zip(
            observation.values[member_index],
            observation.values[representative_index],
            strict=True,
        ):
            if _as_fraction(member) != coefficient * _as_fraction(representative):
                raise ReferenceFixtureError(
                    f"observation {case.id}/{observation.point_id} violates "
                    f"helicity representative relation for {helicity.id}"
                )
    for color in case.colors:
        if not isinstance(color, LCColorFlow) or color.computed:
            continue
        member_index = color_indices[color.id]
        representative_index = color_indices[color.representative_id]
        coefficient = _as_fraction(color.coefficient)
        for row in observation.values:
            if _as_fraction(row[member_index]) != (
                coefficient * _as_fraction(row[representative_index])
            ):
                raise ReferenceFixtureError(
                    f"observation {case.id}/{observation.point_id} violates "
                    f"color representative relation for {color.id}"
                )

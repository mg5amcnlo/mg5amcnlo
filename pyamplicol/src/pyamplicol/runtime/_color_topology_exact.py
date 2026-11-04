# SPDX-License-Identifier: 0BSD
"""Exact replay of authenticated NLC/full-colour topology quotients."""

from __future__ import annotations

import struct
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import cast

from pyamplicol.api.errors import ArtifactError, EvaluationError
from pyamplicol.runtime._color_weight_exact import exact_color_weight

_ZERO = Decimal(0)
_ONE = Decimal(1)
_TWO = Decimal(2)
_ComplexDecimal = tuple[Decimal, Decimal]


@dataclass(frozen=True, slots=True)
class ExactColorReplayRoute:
    source_output_indices: tuple[int, ...]
    target_group_index: int
    factor: _ComplexDecimal


@dataclass(frozen=True, slots=True)
class ExactColorReplayMapping:
    input_mapping: tuple[tuple[int, int], ...]
    routes: tuple[ExactColorReplayRoute, ...]


@dataclass(frozen=True, slots=True)
class _ExactColorContractionEntry:
    left_group_index: int
    right_group_index: int
    weight: _ComplexDecimal
    symmetry_factor: Decimal


@dataclass(frozen=True, slots=True)
class _ExactRepeatedColorContraction:
    component_count: int
    component_group_indices: tuple[int, ...]
    entries: tuple[_ExactColorContractionEntry, ...]


@dataclass(frozen=True, slots=True)
class ExactColorReplayPlan:
    mappings: tuple[ExactColorReplayMapping, ...]
    physical_group_members: tuple[tuple[tuple[int, Decimal], ...], ...]
    expanded_entries: tuple[_ExactColorContractionEntry, ...]
    repeated_contraction: _ExactRepeatedColorContraction | None
    helicity_ids: tuple[str, ...]
    color_ids: tuple[str, ...]

    @property
    def physical_group_count(self) -> int:
        return len(self.physical_group_members)


def parse_exact_color_topology_replay(
    execution: Mapping[str, object],
    physics: Mapping[str, object],
    public_permutation: tuple[int, ...] | None,
) -> ExactColorReplayPlan | None:
    """Validate and bind one exact colour-topology replay plan.

    The amplitude replay and physical colour contraction live in the runtime
    amplitude stage.  They are authenticated artifact data; this parser binds
    their stable group IDs to materialized root outputs and public helicities.
    """

    runtime_schema = _mapping(execution.get("runtime_schema"), "runtime schema")
    amplitude_stage = _mapping(
        runtime_schema.get("amplitude_stage"), "runtime amplitude stage"
    )
    raw_replay = amplitude_stage.get("color_topology_replay")
    proof = runtime_schema.get("color_topology_replay")
    if proof is None:
        compiled = execution.get("compiled")
        if isinstance(compiled, Mapping):
            proof = compiled.get("color_topology_replay")
    if raw_replay is None:
        if isinstance(proof, Mapping) and proof.get("enabled") is True:
            raise ArtifactError(
                "enabled color topology replay has no exact amplitude gather"
            )
        return None
    replay = _mapping(raw_replay, "color topology replay amplitude gather")
    if not isinstance(proof, Mapping) or proof.get("enabled") is not True:
        raise ArtifactError(
            "color topology replay amplitude gather has no enabled proof"
        )

    particles = _records(physics, "external_particles", "external particles")
    external_count = len(particles)
    if public_permutation is not None and (
        len(public_permutation) != external_count
        or set(public_permutation) != set(range(external_count))
    ):
        raise ArtifactError("color topology replay has an invalid public permutation")
    if proof.get("contract_version") != 3:
        raise ArtifactError("color topology replay proof has an unsupported contract")
    if proof.get("mode") != "external-label-permutation":
        raise ArtifactError("color topology replay proof has an unsupported mode")
    accuracy = str(physics.get("color_accuracy"))
    if accuracy not in {"nlc", "full"} or proof.get("color_accuracy") != accuracy:
        raise ArtifactError("color topology replay has inconsistent colour accuracy")
    if _integer(replay.get("contract_version"), "color replay contract") != 1:
        raise ArtifactError("color topology replay amplitude contract is unsupported")

    roots = _records(amplitude_stage, "roots", "amplitude roots")
    output_count = _integer(amplitude_stage.get("output_count"), "amplitude count")
    outputs_by_group: dict[int, list[int]] = {}
    seen_outputs: set[int] = set()
    for root_index, root in enumerate(roots):
        output_index = _integer(root.get("output_index"), "amplitude output index")
        group_id = _integer(root.get("coherent_group_id"), "coherent group ID")
        if output_index >= output_count or output_index in seen_outputs:
            raise ArtifactError("color replay amplitude outputs are not unique")
        seen_outputs.add(output_index)
        outputs_by_group.setdefault(group_id, []).append(output_index)
        root_id = _integer(root.get("root_id"), "amplitude root ID")
        if root_id != root_index:
            raise ArtifactError("color replay amplitude roots are not contiguous")
    if seen_outputs != set(range(output_count)):
        raise ArtifactError("color replay amplitude outputs are incomplete")

    physical_group_count = _integer(
        replay.get("physical_group_count"), "physical color group count"
    )
    raw_groups = _records(replay, "physical_groups", "physical color groups")
    if physical_group_count == 0 or len(raw_groups) != physical_group_count:
        raise ArtifactError("color replay physical group count is inconsistent")

    helicities = _records(physics, "helicities", "physical helicities")
    colors = _records(physics, "color_components", "physical color components")
    if len(colors) != 1:
        raise ArtifactError("color topology replay requires contracted colour")
    helicity_ids = tuple(str(item.get("id")) for item in helicities)
    color_ids = tuple(str(item.get("id")) for item in colors)
    if len(set(helicity_ids)) != len(helicity_ids) or not color_ids[0]:
        raise ArtifactError("color replay public axes contain duplicate IDs")
    physical_by_values: dict[tuple[int, ...], tuple[int, Decimal]] = {}
    for index, helicity in enumerate(helicities):
        values = _integer_sequence(
            helicity.get("values"), external_count, "physical helicity values"
        )
        coefficient = _decimal(
            helicity.get("coefficient", 1), "physical helicity coefficient"
        )
        if coefficient < _ZERO or values in physical_by_values:
            raise ArtifactError("physical helicity metadata is inconsistent")
        physical_by_values[values] = (index, coefficient)

    physical_group_members = []
    for index, group in enumerate(raw_groups):
        if _integer(group.get("group_id"), "physical group ID") != index:
            raise ArtifactError("color replay physical group IDs are not contiguous")
        if _integer(group.get("color_sector_id"), "physical color sector") < 0:
            raise ArtifactError("color replay physical color sector is invalid")
        color_word = _nonempty_integer_sequence(
            group.get("color_word"), "physical color word"
        )
        if any(label < 1 or label > external_count for label in color_word):
            raise ArtifactError("color replay physical color word is out of range")
        representative = _integer_sequence(
            group.get("helicities"), external_count, "physical group helicity"
        )
        if public_permutation is not None:
            representative = _permute_to_public(representative, public_permutation)
        reuse_weight = _decimal(
            group.get("helicity_weight"), "physical group helicity weight"
        )
        if reuse_weight <= _ZERO:
            raise ArtifactError("color replay helicity weight must be positive")
        members = [representative]
        if reuse_weight > _ONE:
            if reuse_weight != _TWO:
                raise ArtifactError(
                    "color replay has an unsupported helicity reuse weight"
                )
            flipped = tuple(-value for value in representative)
            if flipped != representative:
                members.append(flipped)
        indexed = []
        total_weight = _ZERO
        for values in members:
            try:
                physical_index, weight = physical_by_values[values]
            except KeyError as exc:
                raise ArtifactError(
                    "replayed physical helicity is absent from public metadata"
                ) from exc
            total_weight += weight
            indexed.append((physical_index, weight))
        if total_weight <= _ZERO:
            raise ArtifactError("replayed physical helicity has no positive weight")
        physical_group_members.append(
            tuple((member, weight / total_weight) for member, weight in indexed)
        )

    raw_mappings = _records(replay, "mappings", "color replay mappings")
    if not raw_mappings:
        raise ArtifactError("color replay amplitude gather has no mappings")
    mappings = []
    covered_targets: set[int] = set()
    input_mappings: list[tuple[tuple[int, int], ...]] = []
    for raw_mapping in raw_mappings:
        input_mapping = _label_permutation(
            raw_mapping.get("label_permutation"), external_count
        )
        input_mappings.append(input_mapping)
        raw_routes = _records(raw_mapping, "group_routes", "color replay routes")
        if not raw_routes:
            raise ArtifactError("color replay mapping has no group routes")
        routes = []
        mapping_sources: set[int] = set()
        for raw_route in raw_routes:
            source_id = _integer(
                raw_route.get("source_group_id"), "color replay source group"
            )
            try:
                source_outputs = tuple(outputs_by_group[source_id])
            except KeyError as exc:
                raise ArtifactError(
                    "color replay route references an unknown materialized group"
                ) from exc
            target = _integer(
                raw_route.get("target_group_id"), "color replay target group"
            )
            if target >= physical_group_count:
                raise ArtifactError("color replay route target is out of range")
            factor = _complex_pair(raw_route.get("factor"), "color replay factor")
            if (
                factor == (_ZERO, _ZERO)
                or source_id in mapping_sources
                or target in covered_targets
            ):
                raise ArtifactError("color replay routes are not a bijection")
            mapping_sources.add(source_id)
            covered_targets.add(target)
            routes.append(ExactColorReplayRoute(source_outputs, target, factor))
        routes.sort(key=lambda route: route.target_group_index)
        mappings.append(ExactColorReplayMapping(input_mapping, tuple(routes)))
    if covered_targets != set(range(physical_group_count)):
        raise ArtifactError("color replay routes do not cover every physical group")
    if input_mappings != sorted(input_mappings) or len(set(input_mappings)) != len(
        input_mappings
    ):
        raise ArtifactError("color replay input mappings are not canonical")
    _validate_proof_mapping_catalog(proof, tuple(input_mappings), external_count)

    contraction = _mapping(
        amplitude_stage.get("color_contraction"),
        "color replay physical contraction",
    )
    if contraction.get("supported") is not True:
        raise ArtifactError("color replay physical contraction is unsupported")
    if contraction.get("includes_color_factor") is not True:
        raise ArtifactError("color replay contraction omits the colour factor")
    if (
        _integer(contraction.get("group_count"), "color contraction group count")
        != physical_group_count
    ):
        raise ArtifactError("color replay contraction group count is inconsistent")
    expanded = tuple(
        _parse_expanded_entry(entry, physical_group_count)
        for entry in _records(contraction, "entries", "color contraction entries")
    )
    raw_repeated = contraction.get("repeated_block")
    repeated = (
        None
        if raw_repeated is None
        else _parse_repeated_contraction(raw_repeated, physical_group_count)
    )
    if (bool(expanded) == (repeated is not None)) or (
        not expanded and repeated is None
    ):
        raise ArtifactError(
            "color replay contraction must use exactly one storage layout"
        )
    plan = ExactColorReplayPlan(
        mappings=tuple(mappings),
        physical_group_members=tuple(physical_group_members),
        expanded_entries=expanded,
        repeated_contraction=repeated,
        helicity_ids=helicity_ids,
        color_ids=color_ids,
    )
    for entry in _contraction_entries(plan):
        if (
            plan.physical_group_members[entry.left_group_index]
            != plan.physical_group_members[entry.right_group_index]
        ):
            raise ArtifactError(
                "color contraction mixes distinct replayed physical helicities"
            )
    return plan


def apply_exact_color_replay_input_mapping(
    point: tuple[tuple[Decimal, Decimal, Decimal, Decimal], ...],
    mapping: Sequence[tuple[int, int]],
) -> tuple[tuple[Decimal, Decimal, Decimal, Decimal], ...]:
    mapped = list(point)
    for representative, sector in mapping:
        mapped[representative] = point[sector]
    return tuple(mapped)


def reduce_exact_color_topology_replay(
    amplitudes: Sequence[Sequence[_ComplexDecimal]],
    plan: ExactColorReplayPlan,
    point_count: int,
    normalization: Decimal,
    selected_helicities: Sequence[str] | None,
    selected_colors: Sequence[str] | None,
) -> tuple[
    tuple[tuple[tuple[Decimal, ...], ...], ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    """Gather materialized amplitudes and apply the physical colour matrix."""

    if selected_colors is not None:
        raise EvaluationError(
            "LC color-flow selection is unavailable for NLC/full artifacts"
        )
    selected_h = _selected_indices(plan.helicity_ids, selected_helicities, "helicity")
    expected = point_count * len(plan.mappings)
    if len(amplitudes) != expected:
        raise ArtifactError("color replay amplitude point count is inconsistent")
    full_points = []
    for point_index in range(point_count):
        physical = [(_ZERO, _ZERO) for _ in range(plan.physical_group_count)]
        for mapping_index, mapping in enumerate(plan.mappings):
            raw = amplitudes[mapping_index * point_count + point_index]
            for route in mapping.routes:
                value = (_ZERO, _ZERO)
                for output_index in route.source_output_indices:
                    if output_index >= len(raw):
                        raise ArtifactError(
                            "color replay materialized amplitude is too short"
                        )
                    value = _complex_add(value, raw[output_index])
                physical[route.target_group_index] = _complex_mul(value, route.factor)
        full = [_ZERO for _ in plan.helicity_ids]
        for entry in _contraction_entries(plan):
            left = physical[entry.left_group_index]
            right = physical[entry.right_group_index]
            product_real = left[0] * right[0] + left[1] * right[1]
            product_imaginary = left[1] * right[0] - left[0] * right[1]
            contribution = (
                normalization
                * entry.symmetry_factor
                * (entry.weight[0] * product_real - entry.weight[1] * product_imaginary)
            )
            for helicity_index, weight in plan.physical_group_members[
                entry.left_group_index
            ]:
                full[helicity_index] += contribution * weight
        full_points.append(
            tuple((full[helicity_index],) for helicity_index in selected_h)
        )
    return (
        tuple(full_points),
        tuple(plan.helicity_ids[index] for index in selected_h),
        plan.color_ids,
    )


def _parse_expanded_entry(
    raw: Mapping[str, object], group_count: int
) -> _ExactColorContractionEntry:
    left = _integer(raw.get("left_group_id"), "color contraction left group")
    right = _integer(raw.get("right_group_id"), "color contraction right group")
    if left >= group_count or right >= group_count:
        raise ArtifactError("color contraction group ID is out of range")
    return _ExactColorContractionEntry(
        left,
        right,
        exact_color_weight(raw)
        or _complex_pair(raw.get("weight"), "color contraction weight"),
        _decimal(raw.get("symmetry_factor", 1), "color contraction symmetry"),
    )


def _parse_repeated_contraction(
    raw: object, group_count: int
) -> _ExactRepeatedColorContraction:
    block = _mapping(raw, "repeated color contraction")
    component_count = _integer(
        block.get("component_count"), "repeated color component count"
    )
    group_ids = _integer_sequence(
        block.get("component_group_ids"),
        group_count,
        "repeated color group map",
    )
    if (
        component_count < 2
        or len(group_ids) != group_count
        or len(group_ids) % component_count
        or set(group_ids) != set(range(group_count))
    ):
        raise ArtifactError("repeated color contraction group map is invalid")
    groups_per_component = group_count // component_count
    entries = []
    for raw_entry in _records(block, "entries", "repeated color entries"):
        left = _integer(raw_entry.get("left_group_index"), "repeated left group")
        right = _integer(raw_entry.get("right_group_index"), "repeated right group")
        if left >= groups_per_component or right >= groups_per_component:
            raise ArtifactError("repeated color contraction index is out of range")
        entries.append(
            _ExactColorContractionEntry(
                left,
                right,
                exact_color_weight(raw_entry)
                or _complex_pair(raw_entry.get("weight"), "repeated color weight"),
                _decimal(
                    raw_entry.get("symmetry_factor", 1),
                    "repeated color symmetry",
                ),
            )
        )
    if not entries:
        raise ArtifactError("repeated color contraction is empty")
    return _ExactRepeatedColorContraction(component_count, group_ids, tuple(entries))


def _contraction_entries(
    plan: ExactColorReplayPlan,
) -> Iterator[_ExactColorContractionEntry]:
    if plan.repeated_contraction is None:
        yield from plan.expanded_entries
        return
    block = plan.repeated_contraction
    for component in range(block.component_count):
        for entry in block.entries:
            yield _ExactColorContractionEntry(
                block.component_group_indices[
                    entry.left_group_index * block.component_count + component
                ],
                block.component_group_indices[
                    entry.right_group_index * block.component_count + component
                ],
                entry.weight,
                entry.symmetry_factor,
            )


def _validate_proof_mapping_catalog(
    proof: Mapping[str, object],
    actual: tuple[tuple[tuple[int, int], ...], ...],
    external_count: int,
) -> None:
    groups = _records(proof, "groups", "color replay proof groups")
    expected: set[tuple[tuple[int, int], ...]] = set()
    for group in groups:
        permutations = _records(
            group, "sector_permutations", "color replay proof permutations"
        )
        for permutation in permutations:
            expected.add(
                _label_permutation(permutation.get("label_permutation"), external_count)
            )
    residual = _integer_members(
        proof.get("residual_sector_ids"), "color replay residual sectors"
    )
    if residual:
        expected.add(())
    if tuple(sorted(expected)) != actual:
        raise ArtifactError(
            "color replay proof mappings do not match the amplitude gather"
        )


def _selected_indices(
    available: Sequence[str], requested: Sequence[str] | None, kind: str
) -> tuple[int, ...]:
    if requested is None:
        return tuple(range(len(available)))
    if not requested:
        raise EvaluationError(f"{kind} selection must not be empty")
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise EvaluationError(f"unknown resolved {kind} ID {unknown[0]!r}")
    selected = set(requested)
    return tuple(
        index for index, identifier in enumerate(available) if identifier in selected
    )


def _label_permutation(raw: object, external_count: int) -> tuple[tuple[int, int], ...]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ArtifactError("color replay label permutation is invalid")
    representatives: set[int] = set()
    sectors: set[int] = set()
    result = []
    for item in raw:
        record = _mapping(item, "color replay label permutation entry")
        representative = _integer(
            record.get("representative_label"), "representative label"
        )
        sector = _integer(record.get("sector_label"), "sector label")
        if (
            not 1 <= representative <= external_count
            or not 1 <= sector <= external_count
        ):
            raise ArtifactError("color replay label permutation is out of range")
        representative -= 1
        sector -= 1
        if representative in representatives or sector in sectors:
            raise ArtifactError("color replay label permutation is not one-to-one")
        representatives.add(representative)
        sectors.add(sector)
        if representative != sector:
            result.append((representative, sector))
    if representatives != sectors:
        raise ArtifactError("color replay label permutation support is inconsistent")
    return tuple(sorted(result))


def _permute_to_public(
    representative: tuple[int, ...], permutation: tuple[int, ...]
) -> tuple[int, ...]:
    public = [0] * len(representative)
    for representative_index, public_index in enumerate(permutation):
        public[public_index] = representative[representative_index]
    return tuple(public)


def _complex_add(left: _ComplexDecimal, right: _ComplexDecimal) -> _ComplexDecimal:
    return left[0] + right[0], left[1] + right[1]


def _complex_mul(left: _ComplexDecimal, right: _ComplexDecimal) -> _ComplexDecimal:
    return (
        left[0] * right[0] - left[1] * right[1],
        left[0] * right[1] + left[1] * right[0],
    )


def _complex_pair(value: object, context: str) -> _ComplexDecimal:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ArtifactError(f"{context} is not complex data")
    if len(value) != 2:
        raise ArtifactError(f"{context} is not complex data")
    return _decimal(value[0], context), _decimal(value[1], context)


def _decimal(value: object, context: str) -> Decimal:
    try:
        if isinstance(value, str) and value.startswith("binary64:"):
            bits = value.removeprefix("binary64:")
            if len(bits) != 16:
                raise ValueError("binary64 payload must contain 16 hexadecimal digits")
            result = Decimal.from_float(struct.unpack(">d", bytes.fromhex(bits))[0])
        else:
            result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ArtifactError(f"{context} is not a decimal scalar") from exc
    if not result.is_finite():
        raise ArtifactError(f"{context} must be finite")
    return result


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ArtifactError(f"{context} is not an object")
    return cast(Mapping[str, object], value)


def _records(
    mapping: Mapping[str, object], key: str, context: str
) -> tuple[Mapping[str, object], ...]:
    value = mapping.get(key)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ArtifactError(f"{context} are invalid")
    return tuple(_mapping(item, f"{context} entry") for item in value)


def _integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ArtifactError(f"{context} is not a non-negative integer")
    return value


def _integer_sequence(
    value: object, expected_length: int, context: str
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ArtifactError(f"{context} are invalid")
    result = tuple(
        int(cast(int | float | str, item))
        for item in value
        if not isinstance(item, bool)
    )
    if len(result) != expected_length or len(result) != len(value):
        raise ArtifactError(f"{context} have an inconsistent length")
    return result


def _nonempty_integer_sequence(value: object, context: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ArtifactError(f"{context} is invalid")
    try:
        result = tuple(int(cast(int | float | str, item)) for item in value)
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"{context} is invalid") from exc
    if not result:
        raise ArtifactError(f"{context} is empty")
    return result


def _integer_members(value: object, context: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ArtifactError(f"{context} are invalid")
    result = tuple(_integer(item, f"{context} entry") for item in value)
    if len(set(result)) != len(result):
        raise ArtifactError(f"{context} contain duplicates")
    return result


__all__ = [
    "ExactColorReplayMapping",
    "ExactColorReplayPlan",
    "apply_exact_color_replay_input_mapping",
    "parse_exact_color_topology_replay",
    "reduce_exact_color_topology_replay",
]

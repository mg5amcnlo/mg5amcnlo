# SPDX-License-Identifier: 0BSD
"""Public exact executor for compact recurrence artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from pathlib import Path
from typing import Any, cast

from pyamplicol.api.errors import ArtifactError, EvaluationError
from pyamplicol.api.protocols import Momenta
from pyamplicol.api.results import ResolvedEvaluation
from pyamplicol.artifacts import load_manifest
from pyamplicol.artifacts.security import confined_path
from pyamplicol.runtime._evaluator_payloads import ExactEvaluatorPayloadResolver
from pyamplicol.runtime._native_selection import (
    native_physics_axes,
    native_process_selection,
    representative_vector_to_public,
)
from pyamplicol.runtime._normalization_exact import exact_normalization
from pyamplicol.runtime.eager_exact._contracts import (
    _KernelLoader,
    _mapping,
    _PayloadIndex,
    _read_json,
    _sequence,
)
from pyamplicol.runtime.symbolica_exact import (
    _decimal,
    _diagnostic_project_onshell_points,
    _DiagnosticMassBinding,
    _prepare_points,
    _runtime_state,
    _selected_indices,
    _working_precision,
)

from ._color import _contract_color_amplitudes
from ._execution import (
    _evaluate_contracted_point,
    _evaluate_replay_point,
    _evaluate_union_point,
)
from ._plan import _RecurrenceExactPlan
from ._plan_v2 import (
    DIRECT_NONE_U32,
    _AmplitudeDestination,
    _NativeExactSectionsLoader,
    _ReplayTarget,
    _ResolvedHelicity,
)

_ZERO = Decimal(0)


def _diagnostic_recurrence_mass_bindings(
    plan: _RecurrenceExactPlan,
    prepared_parameters: Sequence[tuple[Decimal, Decimal]],
) -> tuple[_DiagnosticMassBinding, ...]:
    """Resolve one unambiguous authenticated mass for every external source."""

    template_ids_by_slot: dict[int, set[int]] = {
        slot: set() for slot in range(plan.sections.external_source_count)
    }
    if plan.sections.source_dispatch_variants:
        for variant in plan.sections.source_dispatch_variants:
            try:
                source = plan.sections.sources[variant.source_row_id]
            except IndexError as exc:
                raise ArtifactError(
                    "recurrence diagnostic source variant is out of range"
                ) from exc
            template_ids_by_slot.setdefault(source.source_slot, set()).add(
                variant.source_template_id
            )
    else:
        for source in plan.sections.sources:
            template_ids_by_slot.setdefault(source.source_slot, set()).add(
                source.source_template_or_dispatch_domain
            )

    if len(plan.external_source_slots) != plan.sections.external_source_count or set(
        plan.external_source_slots
    ) != set(range(plan.sections.external_source_count)):
        raise ArtifactError("recurrence diagnostic external source layout is invalid")
    by_slot: dict[int, _DiagnosticMassBinding] = {}
    for source_slot in range(plan.sections.external_source_count):
        candidates = set()
        for template_id in template_ids_by_slot.get(source_slot, set()):
            try:
                template = plan.source_templates[template_id]
            except KeyError as exc:
                raise ArtifactError(
                    "recurrence diagnostic source template is absent"
                ) from exc
            parameter_id = template.mass_prepared_parameter_id
            if parameter_id is None:
                binding = _DiagnosticMassBinding(
                    mass=_ZERO,
                    authority="authenticated-recurrence-static-massless-source",
                    parameter_name=None,
                )
            else:
                try:
                    real, imaginary = prepared_parameters[parameter_id]
                except IndexError as exc:
                    raise ArtifactError(
                        "recurrence diagnostic mass parameter is out of range"
                    ) from exc
                if imaginary != _ZERO:
                    raise EvaluationError(
                        "recurrence diagnostic external mass must be real"
                    )
                if not real.is_finite():
                    raise EvaluationError(
                        "recurrence diagnostic external mass must be finite"
                    )
                if not template.mass_parameter_name:
                    raise ArtifactError(
                        "recurrence diagnostic mass parameter name is absent"
                    )
                binding = _DiagnosticMassBinding(
                    mass=real,
                    authority="authenticated-current-recurrence-prepared-parameter",
                    parameter_name=template.mass_parameter_name,
                )
            candidates.add(binding)
        if len(candidates) != 1:
            raise ArtifactError(
                f"recurrence diagnostic source slot {source_slot} does not have "
                "one unambiguous mass binding"
            )
        by_slot[source_slot] = candidates.pop()
    return tuple(by_slot[slot] for slot in plan.external_source_slots)


class RecurrenceExactExecutor:
    """Execute one authenticated compact recurrence through exact kernels."""

    def __init__(
        self,
        artifact: Path,
        process_id: str,
        native_runtime: Any,
        *,
        kernel_loader: _KernelLoader | None = None,
        native_sections_loader: _NativeExactSectionsLoader | None = None,
    ) -> None:
        self._artifact = Path(artifact).expanduser().resolve(strict=True)
        self._native_runtime = native_runtime
        manifest = load_manifest(self._artifact)
        exact_payloads = ExactEvaluatorPayloadResolver(manifest)
        selection = native_process_selection(native_runtime, manifest.processes)
        process = selection.process
        permutation = selection.external_permutation
        representative_id = selection.representative_process_id
        execution_records = tuple(
            record
            for record in manifest.payloads
            if record.role == "evaluator-manifest"
            and record.process_id == representative_id
        )
        if len(execution_records) != 1:
            raise ArtifactError(
                f"process {representative_id!r} must declare one recurrence "
                "execution manifest"
            )
        physics_path = process.get("physics_path")
        if not isinstance(physics_path, str):
            raise ArtifactError(f"process {representative_id!r} has no physics path")
        payloads = _PayloadIndex.from_manifest(manifest)
        payloads.require(
            physics_path,
            role="runtime-physics",
            process_id=representative_id,
        )
        execution = _read_json(
            confined_path(self._artifact, execution_records[0].path),
            "recurrence execution metadata",
        )
        representative_physics = _read_json(
            confined_path(self._artifact, physics_path),
            "runtime physics metadata",
        )
        axes = native_physics_axes(native_runtime, representative_physics)
        if execution.get("key") != selection.representative_process_key:
            raise ArtifactError(
                "recurrence exact execution disagrees with Rusticol's representative "
                "process key"
            )
        self._plan = _RecurrenceExactPlan.load(
            artifact_root=self._artifact,
            process_id=representative_id,
            execution_path=execution_records[0].path,
            execution=execution,
            manifest=manifest,
            kernel_loader=kernel_loader,
            exact_payloads=exact_payloads,
            native_sections_loader=native_sections_loader,
        )
        self._permutation = permutation
        self._representative_physics = representative_physics
        self._physics = axes.public_physics
        (
            self._helicity_representative,
            self._helicity_orbit_members,
        ) = self._helicity_reduction_indices()
        self._replay_by_color: tuple[_ReplayTarget, ...] = ()
        self._destination_helicities: tuple[tuple[int, ...], ...] = ()
        self._union_destination_by_color: tuple[_AmplitudeDestination, ...] = ()
        self._union_helicity_by_physics: tuple[_ResolvedHelicity | None, ...] = ()
        self._contracted_destination_helicity: tuple[int, ...] = ()
        if self._plan.sections.strategy == "topology-replay":
            self._replay_by_color = self._replay_targets_by_color()
            self._destination_helicities = self._destination_helicity_maps()
        elif self._plan.sections.strategy == "all-flow-union":
            self._union_destination_by_color = self._union_destinations_by_color()
            self._union_helicity_by_physics = self._union_helicities_by_physics()
        else:
            self._contracted_destination_helicity = (
                self._contracted_destination_helicity_map()
            )

    def _diagnostic_project_onshell(
        self,
        momenta: Momenta,
        *,
        precision: int,
    ) -> tuple[
        tuple[tuple[tuple[Decimal, Decimal, Decimal, Decimal], ...], ...],
        dict[str, object],
    ]:
        """Build diagnostic-only on-shell kinematics from current parameters."""

        working_precision = _working_precision(precision)
        state = _runtime_state(self._native_runtime)
        runtime_parameters = tuple(
            _decimal(value, "runtime model parameter")
            for value in state["model_parameter_values"]
        )
        with localcontext() as context:
            context.prec = working_precision
            context.rounding = ROUND_HALF_EVEN
            parameters = self._plan.resolve_model_parameters(
                runtime_parameters,
                working_precision,
            )
            bindings = _diagnostic_recurrence_mass_bindings(
                self._plan,
                parameters.prepared,
            )
        return _diagnostic_project_onshell_points(
            momenta,
            physics=self._representative_physics,
            artifact_mass_bindings=bindings,
            permutation=self._permutation,
            precision=precision,
        )

    def evaluate_resolved(
        self,
        momenta: Momenta,
        *,
        helicities: Sequence[str] | None,
        color_flows: Sequence[str] | None,
        precision: int,
    ) -> ResolvedEvaluation:
        if (
            isinstance(precision, bool)
            or not isinstance(precision, int)
            or precision < 1
        ):
            raise EvaluationError(
                "precision must be a positive integer number of decimal digits"
            )
        working_precision = _working_precision(precision)
        points = _prepare_points(momenta, self._physics, self._permutation)
        state = _runtime_state(self._native_runtime)
        runtime_parameters = tuple(
            _decimal(value, "runtime model parameter")
            for value in state["model_parameter_values"]
        )
        helicity_records = tuple(
            _mapping(value, f"physics helicity {index}")
            for index, value in enumerate(
                _sequence(self._physics.get("helicities"), "physics helicities")
            )
        )
        color_records = tuple(
            _mapping(value, f"physics color {index}")
            for index, value in enumerate(
                _sequence(
                    self._physics.get("color_components"),
                    "physics color components",
                )
            )
        )
        helicity_ids = tuple(str(record["id"]) for record in helicity_records)
        color_ids = tuple(str(record["id"]) for record in color_records)
        selected_helicities = _selected_indices(
            helicity_ids,
            helicities,
            "helicity",
        )
        if (
            self._plan.sections.strategy == "contracted-color-union"
            and color_flows is not None
        ):
            raise EvaluationError(
                "contracted NLC/full recurrence does not expose a color-flow selector"
            )
        selected_colors = _selected_indices(color_ids, color_flows, "color component")
        helicity_positions = {
            index: position for position, index in enumerate(selected_helicities)
        }

        with localcontext() as context:
            context.prec = working_precision
            context.rounding = ROUND_HALF_EVEN
            normalization = exact_normalization(
                self._physics,
                runtime_parameters,
                working_precision,
                tuple(
                    {"name": row.runtime_name, "parameter_index": row.runtime_slot}
                    for row in self._plan.runtime_parameter_schema
                ),
            )
            parameters = self._plan.resolve_model_parameters(
                runtime_parameters,
                working_precision,
            )
            values = []
            for point in points:
                point_values = [
                    [_ZERO for _ in selected_colors] for _ in selected_helicities
                ]
                if self._plan.sections.strategy == "topology-replay":
                    self._evaluate_replay_resolved_point(
                        point,
                        selected_colors,
                        helicity_positions,
                        helicity_records,
                        color_records,
                        parameters.prepared,
                        working_precision,
                        normalization,
                        point_values,
                    )
                elif self._plan.sections.strategy == "all-flow-union":
                    self._evaluate_union_resolved_point(
                        point,
                        selected_helicities,
                        selected_colors,
                        helicity_records,
                        color_records,
                        parameters.prepared,
                        working_precision,
                        normalization,
                        point_values,
                    )
                else:
                    self._evaluate_contracted_resolved_point(
                        point,
                        selected_helicities,
                        helicity_records,
                        parameters.prepared,
                        working_precision,
                        normalization,
                        point_values,
                    )
                values.append(tuple(tuple(colors) for colors in point_values))

        with localcontext() as context:
            context.prec = precision
            context.rounding = ROUND_HALF_EVEN
            rounded = tuple(
                tuple(tuple(+entry for entry in colors) for colors in point)
                for point in values
            )
        return ResolvedEvaluation(
            values=rounded,
            helicity_ids=tuple(helicity_ids[index] for index in selected_helicities),
            color_ids=tuple(color_ids[index] for index in selected_colors),
            color_accuracy=cast(Any, str(self._physics["color_accuracy"])),
        )

    def _evaluate_replay_resolved_point(
        self,
        point: object,
        selected_colors: Sequence[int],
        helicity_positions: dict[int, int],
        helicity_records: Sequence[Mapping[str, object]],
        color_records: Sequence[Mapping[str, object]],
        prepared_parameters: Sequence[tuple[Decimal, Decimal]],
        working_precision: int,
        normalization: Decimal,
        point_values: list[list[Decimal]],
    ) -> None:
        for color_position, color_index in enumerate(selected_colors):
            target = self._replay_by_color[color_index]
            amplitudes = _evaluate_replay_point(
                self._plan,
                cast(Any, point),
                target,
                prepared_parameters,
                working_precision,
            )
            destination_helicities = self._destination_helicities[color_index]
            color_weight = _decimal(
                color_records[color_index].get("coefficient", 1),
                "color coefficient",
            )
            for destination in self._plan.sections.amplitude_destinations:
                if destination.target_sector_id != target.representative_id:
                    continue
                physics_helicity = destination_helicities[destination.destination_id]
                helicity = helicity_records[physics_helicity]
                if (
                    helicity.get("computed") is not True
                    or helicity.get("structural_zero") is True
                ):
                    continue
                amplitude = amplitudes[destination.destination_id]
                squared = amplitude[0] * amplitude[0] + amplitude[1] * amplitude[1]
                for physical_helicity in self._helicity_orbit_members[physics_helicity]:
                    helicity_position = helicity_positions.get(physical_helicity)
                    if helicity_position is None:
                        continue
                    helicity_weight = _decimal(
                        helicity_records[physical_helicity].get("coefficient", 1),
                        "helicity coefficient",
                    )
                    point_values[helicity_position][color_position] += (
                        normalization * color_weight * helicity_weight * squared
                    )

    def _evaluate_union_resolved_point(
        self,
        point: object,
        selected_helicities: Sequence[int],
        selected_colors: Sequence[int],
        helicity_records: Sequence[Mapping[str, object]],
        color_records: Sequence[Mapping[str, object]],
        prepared_parameters: Sequence[tuple[Decimal, Decimal]],
        working_precision: int,
        normalization: Decimal,
        point_values: list[list[Decimal]],
    ) -> None:
        for helicity_position, physics_helicity in enumerate(selected_helicities):
            helicity_record = helicity_records[physics_helicity]
            if (
                helicity_record.get("computed") is not True
                or helicity_record.get("structural_zero") is True
            ):
                continue
            direct_helicity = self._union_helicity_by_physics[physics_helicity]
            if direct_helicity is None:
                raise ArtifactError(
                    "all-flow-union plan is missing a retained physical helicity"
                )
            amplitudes = _evaluate_union_point(
                self._plan,
                cast(Any, point),
                direct_helicity,
                prepared_parameters,
                working_precision,
            )
            helicity_weight = _decimal(
                helicity_record.get("coefficient", 1),
                "helicity coefficient",
            )
            for color_position, color_index in enumerate(selected_colors):
                destination = self._union_destination_by_color[color_index]
                color_weight = _decimal(
                    color_records[color_index].get("coefficient", 1),
                    "color coefficient",
                )
                amplitude = amplitudes[destination.destination_id]
                point_values[helicity_position][color_position] += (
                    normalization
                    * color_weight
                    * helicity_weight
                    * (amplitude[0] * amplitude[0] + amplitude[1] * amplitude[1])
                )

    def _evaluate_contracted_resolved_point(
        self,
        point: object,
        selected_helicities: Sequence[int],
        helicity_records: Sequence[Mapping[str, object]],
        prepared_parameters: Sequence[tuple[Decimal, Decimal]],
        working_precision: int,
        normalization: Decimal,
        point_values: list[list[Decimal]],
    ) -> None:
        contraction = self._plan.color_contraction
        if contraction is None:
            raise ArtifactError(
                "contracted exact execution has no color-contraction payload"
            )
        amplitudes = self._contracted_amplitudes(
            point,
            prepared_parameters,
            working_precision,
        )
        selected = {
            self._helicity_representative[physical] for physical in selected_helicities
        }
        contracted = _contract_color_amplitudes(
            contraction,
            amplitudes,
            self._contracted_destination_helicity,
            selected,
        )
        for helicity_position, physics_helicity in enumerate(selected_helicities):
            helicity_record = helicity_records[physics_helicity]
            if helicity_record.get("structural_zero") is True:
                continue
            representative = self._helicity_representative[physics_helicity]
            helicity_weight = _decimal(
                helicity_record.get("coefficient", 1),
                "helicity coefficient",
            )
            point_values[helicity_position][0] = (
                normalization * helicity_weight * contracted.get(representative, _ZERO)
            )

    def _contracted_amplitudes(
        self,
        point: object,
        prepared_parameters: Sequence[tuple[Decimal, Decimal]],
        working_precision: int,
    ) -> tuple[tuple[Decimal, Decimal], ...]:
        sections = self._plan.sections
        contraction = self._plan.color_contraction
        if contraction is None:
            raise ArtifactError(
                "contracted exact execution has no color-contraction payload"
            )
        if not sections.replay_targets:
            return _evaluate_contracted_point(
                self._plan,
                cast(Any, point),
                prepared_parameters,
                working_precision,
            )

        destination_by_coordinate: dict[tuple[int, int], int] = {}
        for group_id, destination_id in enumerate(contraction.destination_by_group):
            coordinate = (
                contraction.group_sector_ids[group_id],
                contraction.group_component_ids[group_id],
            )
            if coordinate in destination_by_coordinate:
                raise ArtifactError(
                    "contracted replay color coordinates are not unique"
                )
            destination_by_coordinate[coordinate] = destination_id

        amplitudes = [(_ZERO, _ZERO)] * contraction.destination_count
        covered: set[int] = set()
        for target in sections.replay_targets:
            helicity_map = sections.replay_helicity_map[
                target.helicity_map_start : target.helicity_map_start
                + target.helicity_map_count
            ]
            if len(helicity_map) != len(sections.resolved_helicities):
                raise ArtifactError(
                    "contracted replay helicity mapping has incomplete coverage"
                )
            replayed = _evaluate_replay_point(
                self._plan,
                cast(Any, point),
                target,
                prepared_parameters,
                working_precision,
            )
            for destination in sections.amplitude_destinations:
                if destination.target_sector_id != target.representative_id:
                    continue
                direct_helicity = destination.target_helicity_id
                if direct_helicity >= len(helicity_map):
                    raise ArtifactError(
                        "contracted replay destination helicity is not mapped"
                    )
                coordinate = (
                    target.public_flow_id,
                    helicity_map[direct_helicity],
                )
                try:
                    physical_destination = destination_by_coordinate[coordinate]
                    amplitude = replayed[destination.destination_id]
                except (IndexError, KeyError) as exc:
                    raise ArtifactError(
                        "contracted replay route is outside the physical "
                        "color/helicity domain"
                    ) from exc
                if physical_destination in covered:
                    raise ArtifactError(
                        "contracted replay routes repeat a physical destination"
                    )
                covered.add(physical_destination)
                amplitudes[physical_destination] = amplitude
        if len(covered) != contraction.destination_count:
            raise ArtifactError(
                "contracted replay routes do not cover every physical destination"
            )
        return tuple(amplitudes)

    def _helicity_reduction_indices(
        self,
    ) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
        records = tuple(
            _mapping(value, f"physics helicity {index}")
            for index, value in enumerate(
                _sequence(self._physics.get("helicities"), "physics helicities")
            )
        )
        index_by_id = {str(record["id"]): index for index, record in enumerate(records)}
        if len(index_by_id) != len(records):
            raise ArtifactError("physics helicities repeat a public ID")
        representatives = []
        members: list[list[int]] = [[] for _ in records]
        for index, record in enumerate(records):
            representative_id = str(record.get("representative_id", record["id"]))
            try:
                representative = index_by_id[representative_id]
            except KeyError as exc:
                raise ArtifactError(
                    f"physics helicity {index} references absent representative "
                    f"{representative_id!r}"
                ) from exc
            representative_record = records[representative]
            if record.get("structural_zero") is not True and (
                representative_record.get("computed") is not True
                or representative_record.get("structural_zero") is True
                or str(
                    representative_record.get(
                        "representative_id", representative_record["id"]
                    )
                )
                != str(representative_record["id"])
            ):
                raise ArtifactError(
                    f"physics helicity {index} has an invalid computed representative"
                )
            representatives.append(representative)
            if record.get("structural_zero") is not True:
                members[representative].append(index)
        return tuple(representatives), tuple(tuple(group) for group in members)

    def _replay_targets_by_color(self) -> tuple[_ReplayTarget, ...]:
        by_public_id = {
            target.public_flow_id: target
            for target in self._plan.sections.replay_targets
        }
        if len(by_public_id) != len(self._plan.sections.replay_targets):
            raise ArtifactError("recurrence replay targets repeat a public flow")
        try:
            return tuple(
                by_public_id[public_id]
                for public_id in self._plan.sections.public_flow_ids
            )
        except KeyError as exc:
            raise ArtifactError(
                "recurrence public color axis references an absent replay target"
            ) from exc

    def _destination_helicity_maps(self) -> tuple[tuple[int, ...], ...]:
        sections = self._plan.sections
        helicity_records = tuple(
            _mapping(value, f"physics helicity {index}")
            for index, value in enumerate(
                _sequence(self._physics.get("helicities"), "physics helicities")
            )
        )
        physics_by_values = {
            tuple(
                _signed_integer(component, "physics helicity component")
                for component in _sequence(
                    record.get("values"), "physics helicity values"
                )
            ): index
            for index, record in enumerate(helicity_records)
        }
        direct_to_physics = {}
        for descriptor in sections.resolved_helicities:
            start = descriptor.public_helicity_start
            stop = start + descriptor.public_helicity_count
            vector = sections.public_helicities[start:stop]
            if len(vector) != sections.external_source_count:
                raise ArtifactError(
                    "recurrence direct helicity has invalid source coverage"
                )
            try:
                direct_to_physics[descriptor.helicity_id] = physics_by_values[
                    representative_vector_to_public(vector, self._permutation)
                ]
            except KeyError as exc:
                raise ArtifactError(
                    "recurrence direct helicity is absent from public physics coverage"
                ) from exc

        result = []
        for target in self._replay_by_color:
            start = target.helicity_map_start
            stop = start + target.helicity_map_count
            helicity_map = sections.replay_helicity_map[start:stop]
            if len(helicity_map) != len(sections.resolved_helicities):
                raise ArtifactError(
                    "recurrence replay helicity mapping has incomplete coverage"
                )
            destinations = [DIRECT_NONE_U32] * sections.amplitude_destination_count
            for destination in sections.amplitude_destinations:
                direct_id = destination.target_helicity_id
                if direct_id == DIRECT_NONE_U32:
                    raise ArtifactError(
                        "topology-replay destination has no resolved helicity"
                    )
                if direct_id >= len(helicity_map):
                    raise ArtifactError(
                        "recurrence destination helicity is absent from replay mapping"
                    )
                mapped_id = helicity_map[direct_id]
                try:
                    destinations[destination.destination_id] = direct_to_physics[
                        mapped_id
                    ]
                except KeyError as exc:
                    raise ArtifactError(
                        "recurrence replay maps to an absent direct helicity"
                    ) from exc
            if any(value == DIRECT_NONE_U32 for value in destinations):
                raise ArtifactError(
                    "recurrence amplitude destination helicity map is incomplete"
                )
            result.append(tuple(destinations))
        return tuple(result)

    def _union_destinations_by_color(
        self,
    ) -> tuple[_AmplitudeDestination, ...]:
        sections = self._plan.sections
        by_sector = {
            destination.target_sector_id: destination
            for destination in sections.amplitude_destinations
        }
        if len(by_sector) != len(sections.amplitude_destinations):
            raise ArtifactError(
                "all-flow-union repeats an amplitude destination sector"
            )
        try:
            return tuple(by_sector[sector_id] for sector_id in sections.public_flow_ids)
        except KeyError as exc:
            raise ArtifactError(
                "all-flow-union public color axis references an absent destination"
            ) from exc

    def _union_helicities_by_physics(
        self,
    ) -> tuple[_ResolvedHelicity | None, ...]:
        sections = self._plan.sections
        helicity_records = tuple(
            _mapping(value, f"physics helicity {index}")
            for index, value in enumerate(
                _sequence(self._physics.get("helicities"), "physics helicities")
            )
        )
        physics_by_values = {
            tuple(
                _signed_integer(component, "physics helicity component")
                for component in _sequence(
                    record.get("values"), "physics helicity values"
                )
            ): index
            for index, record in enumerate(helicity_records)
        }
        result: list[_ResolvedHelicity | None] = [None] * len(helicity_records)
        for descriptor in sections.resolved_helicities:
            start = descriptor.public_helicity_start
            stop = start + descriptor.public_helicity_count
            vector = representative_vector_to_public(
                sections.public_helicities[start:stop], self._permutation
            )
            if len(vector) != sections.external_source_count:
                raise ArtifactError(
                    "all-flow-union helicity has incomplete source coverage"
                )
            try:
                physics_index = physics_by_values[vector]
            except KeyError as exc:
                raise ArtifactError(
                    "all-flow-union helicity is outside public coverage"
                ) from exc
            if result[physics_index] is not None:
                raise ArtifactError("all-flow-union repeats a public helicity")
            result[physics_index] = descriptor
        for index, (record, resolved_descriptor) in enumerate(
            zip(helicity_records, result, strict=True)
        ):
            if (
                record.get("computed") is True
                and record.get("structural_zero") is not True
                and resolved_descriptor is None
            ):
                raise ArtifactError(f"all-flow-union omits computed helicity {index}")
        return tuple(result)

    def _contracted_destination_helicity_map(self) -> tuple[int, ...]:
        sections = self._plan.sections
        contraction = self._plan.color_contraction
        if contraction is None:
            raise ArtifactError(
                "contracted exact execution has no color-contraction payload"
            )
        replayed = bool(sections.replay_targets)
        expected_destination_count = (
            contraction.destination_count
            if replayed
            else sections.amplitude_destination_count
        )
        if (
            contraction.component_count != len(sections.resolved_helicities)
            or contraction.destination_count != expected_destination_count
            or len(contraction.destination_by_group) != contraction.group_count
            or len(contraction.group_sector_ids) != contraction.group_count
            or len(contraction.group_component_ids) != contraction.group_count
        ):
            raise ArtifactError(
                "contracted color dimensions disagree with the recurrence plan"
            )
        direct_to_physics = self._direct_helicity_to_physics()
        result = [DIRECT_NONE_U32] * contraction.destination_count
        if replayed:
            expected_coordinates = {
                (target.public_flow_id, direct_helicity)
                for target in sections.replay_targets
                for direct_helicity in range(len(sections.resolved_helicities))
            }
            actual_coordinates = set(
                zip(
                    contraction.group_sector_ids,
                    contraction.group_component_ids,
                    strict=True,
                )
            )
            if actual_coordinates != expected_coordinates:
                raise ArtifactError(
                    "contracted replay color coordinates disagree with the "
                    "recurrence plan"
                )
            for group_id, destination_id in enumerate(contraction.destination_by_group):
                try:
                    physics_helicity = direct_to_physics[
                        contraction.group_component_ids[group_id]
                    ]
                    previous = result[destination_id]
                except IndexError as exc:
                    raise ArtifactError(
                        "contracted replay color mapping references an absent "
                        "destination"
                    ) from exc
                if previous != DIRECT_NONE_U32:
                    raise ArtifactError(
                        "contracted replay color mapping repeats a destination"
                    )
                result[destination_id] = physics_helicity
            if any(value == DIRECT_NONE_U32 for value in result):
                raise ArtifactError(
                    "contracted replay color mapping does not cover every destination"
                )
            return tuple(result)

        for group_id, destination_id in enumerate(contraction.destination_by_group):
            expected_sector = contraction.group_sector_ids[group_id]
            direct_helicity = contraction.group_component_ids[group_id]
            try:
                destination = sections.amplitude_destinations[destination_id]
                physics_helicity = direct_to_physics[direct_helicity]
            except IndexError as exc:
                raise ArtifactError(
                    "contracted color mapping references an absent destination"
                ) from exc
            if (
                destination.destination_id != destination_id
                or destination.target_sector_id != expected_sector
                or destination.target_helicity_id != direct_helicity
                or result[destination_id] != DIRECT_NONE_U32
            ):
                raise ArtifactError(
                    "contracted color mapping disagrees with recurrence destinations"
                )
            result[destination_id] = physics_helicity
        if any(value == DIRECT_NONE_U32 for value in result):
            raise ArtifactError(
                "contracted color mapping does not cover every destination"
            )
        return tuple(result)

    def _direct_helicity_to_physics(self) -> tuple[int, ...]:
        sections = self._plan.sections
        helicity_records = tuple(
            _mapping(value, f"physics helicity {index}")
            for index, value in enumerate(
                _sequence(self._physics.get("helicities"), "physics helicities")
            )
        )
        physics_by_values = {
            tuple(
                _signed_integer(component, "physics helicity component")
                for component in _sequence(
                    record.get("values"), "physics helicity values"
                )
            ): index
            for index, record in enumerate(helicity_records)
        }
        result = [DIRECT_NONE_U32] * len(sections.resolved_helicities)
        for descriptor in sections.resolved_helicities:
            start = descriptor.public_helicity_start
            stop = start + descriptor.public_helicity_count
            vector = representative_vector_to_public(
                sections.public_helicities[start:stop], self._permutation
            )
            if len(
                vector
            ) != sections.external_source_count or descriptor.helicity_id >= len(
                result
            ):
                raise ArtifactError(
                    "contracted recurrence helicity has invalid source coverage"
                )
            try:
                physics_helicity = physics_by_values[vector]
            except KeyError as exc:
                raise ArtifactError(
                    "contracted recurrence helicity is outside public coverage"
                ) from exc
            if result[descriptor.helicity_id] != DIRECT_NONE_U32:
                raise ArtifactError("contracted recurrence repeats a direct helicity")
            result[descriptor.helicity_id] = physics_helicity
        if any(value == DIRECT_NONE_U32 for value in result):
            raise ArtifactError(
                "contracted recurrence does not map every direct helicity"
            )
        return tuple(result)


def _signed_integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactError(f"{context} must be an integer")
    return value


__all__ = ["RecurrenceExactExecutor"]

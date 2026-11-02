# SPDX-License-Identifier: 0BSD
"""Public facade for exact replay of a prepared eager process artifact."""

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
from pyamplicol.runtime._color_topology_exact import (
    apply_exact_color_replay_input_mapping,
    parse_exact_color_topology_replay,
    reduce_exact_color_topology_replay,
)
from pyamplicol.runtime._evaluator_payloads import ExactEvaluatorPayloadResolver
from pyamplicol.runtime._native_selection import (
    native_physics_axes,
    native_process_selection,
    remap_reduction_group,
)
from pyamplicol.runtime._normalization_exact import exact_normalization
from pyamplicol.runtime.eager_exact._contracts import (
    _KernelLoader,
    _mapping,
    _PayloadIndex,
    _read_json,
)
from pyamplicol.runtime.eager_exact._execution import _evaluate_point
from pyamplicol.runtime.eager_exact._plan import _EagerExactPlan
from pyamplicol.runtime.eager_exact._plan_v3 import _NativeExactSectionsLoader
from pyamplicol.runtime.symbolica_exact import (
    _apply_lc_replay_input_mapping,
    _apply_lc_replay_resolved,
    _decimal,
    _diagnostic_project_onshell_points,
    _diagnostic_schema_mass_bindings,
    _lc_replay_plan,
    _prepare_points,
    _reduce_resolved,
    _runtime_state,
    _working_precision,
)


class EagerExactExecutor:
    """Replay one schema-v3 eager process through retained exact kernel states."""

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
                f"process {representative_id!r} must declare one eager "
                "execution manifest"
            )
        physics_path = process.get("physics_path")
        if not isinstance(physics_path, str):
            raise ArtifactError(f"process {representative_id!r} has no physics path")
        payloads = _PayloadIndex.from_manifest(manifest)
        payloads.require(
            physics_path, role="runtime-physics", process_id=representative_id
        )
        execution = _read_json(
            confined_path(self._artifact, execution_records[0].path),
            "eager execution metadata",
        )
        representative_physics = _read_json(
            confined_path(self._artifact, physics_path), "runtime physics metadata"
        )
        axes = native_physics_axes(native_runtime, representative_physics)
        physics = axes.public_physics
        if execution.get("key") != selection.representative_process_key:
            raise ArtifactError(
                "eager exact execution disagrees with Rusticol's representative "
                "process key"
            )
        process_root = self._artifact / "processes" / representative_id
        self._execution = execution
        self._permutation = permutation
        self._representative_physics = representative_physics
        self._plan = _EagerExactPlan.load_for_execution(
            artifact_root=self._artifact,
            process_root=process_root,
            process_id=representative_id,
            execution=execution,
            manifest=manifest,
            kernel_loader=kernel_loader,
            exact_payloads=exact_payloads,
            native_sections_loader=native_sections_loader,
        )
        if self._plan.physics_reduction_groups is not None:
            physics = dict(physics)
            reduction = dict(_mapping(physics.get("reduction"), "physics reduction"))
            reduction["groups"] = [
                remap_reduction_group(group, axes, index=index)
                for index, group in enumerate(self._plan.physics_reduction_groups)
            ]
            physics["reduction"] = reduction
        self._physics = physics
        if "runtime_schema" in execution:
            exact_execution = execution
        else:
            exact_execution = {
                "runtime_schema": self._plan.runtime_schema,
                "lc_topology_replay": self._plan.runtime_schema.get(
                    "lc_topology_replay"
                ),
            }
        self._exact_execution = exact_execution
        self._lc_replay = _lc_replay_plan(exact_execution, physics, permutation)
        self._color_replay = parse_exact_color_topology_replay(
            exact_execution, physics, permutation
        )
        if self._lc_replay is not None and self._color_replay is not None:
            raise ArtifactError(
                "eager exact execution cannot combine LC and full-colour replay"
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

        state = _runtime_state(self._native_runtime)
        parameters = tuple(
            _decimal(value, "runtime model parameter")
            for value in state["model_parameter_values"]
        )
        schema = self._exact_execution.get("runtime_schema")
        if not isinstance(schema, Mapping):
            raise ArtifactError("eager execution has no runtime schema")
        bindings = _diagnostic_schema_mass_bindings(schema, self._physics, parameters)
        return _diagnostic_project_onshell_points(
            momenta,
            physics=self._physics,
            artifact_mass_bindings=bindings,
            # Both public physics and its PDG-derived bindings already follow
            # the caller's selected ordering.
            permutation=None,
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
        parameters = tuple(
            _decimal(value, "runtime model parameter")
            for value in state["model_parameter_values"]
        )
        with localcontext() as context:
            context.prec = working_precision
            context.rounding = ROUND_HALF_EVEN
            if self._color_replay is not None:
                self._color_replay = parse_exact_color_topology_replay(
                    self._exact_execution, self._physics, self._permutation
                )
            normalization = exact_normalization(
                self._physics,
                parameters,
                working_precision,
                cast(Any, self._plan.runtime_schema["model_parameters"]),
            )
            exact_parameters = self._plan.resolve_model_parameters(
                parameters,
                working_precision,
            )
            if self._lc_replay is not None:
                evaluation_points = tuple(
                    _apply_lc_replay_input_mapping(point, entry.input_mapping)
                    for entry in self._lc_replay.entries
                    for point in points
                )
            elif self._color_replay is not None:
                evaluation_points = tuple(
                    apply_exact_color_replay_input_mapping(point, mapping.input_mapping)
                    for mapping in self._color_replay.mappings
                    for point in points
                )
            else:
                evaluation_points = points
            amplitudes = tuple(
                _evaluate_point(
                    self._plan,
                    point,
                    exact_parameters.runtime,
                    exact_parameters.prepared,
                    working_precision,
                )
                for point in evaluation_points
            )
            if self._color_replay is not None:
                values, helicity_ids, color_ids = reduce_exact_color_topology_replay(
                    amplitudes,
                    self._color_replay,
                    len(points),
                    normalization,
                    helicities,
                    color_flows,
                )
            else:
                values, helicity_ids, color_ids = _reduce_resolved(
                    amplitudes,
                    self._exact_execution,
                    self._physics,
                    normalization,
                    helicities if self._lc_replay is None else None,
                    color_flows if self._lc_replay is None else None,
                )
            if self._lc_replay is not None:
                values, helicity_ids, color_ids = _apply_lc_replay_resolved(
                    values,
                    self._lc_replay,
                    len(points),
                    helicity_ids,
                    color_ids,
                    helicities,
                    color_flows,
                )
        with localcontext() as context:
            context.prec = precision
            context.rounding = ROUND_HALF_EVEN
            values = tuple(
                tuple(tuple(+entry for entry in colors) for colors in helicity_values)
                for helicity_values in values
            )
        return ResolvedEvaluation(
            values=values,
            helicity_ids=helicity_ids,
            color_ids=color_ids,
            color_accuracy=cast(Any, str(self._physics["color_accuracy"])),
        )

# SPDX-License-Identifier: 0BSD
"""Opt-in generation of complete coherent amplitudes and colour operators.

The ordinary generator never imports this module.  This first implementation
uses the existing generic compiled pipeline, but bypasses reductions whose
proof applies only to physical-helicity squared amplitudes.  In particular,
arbitrary external vector sources must not inherit physical-helicity zeros.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast

from pyamplicol.api.errors import GenerationError
from pyamplicol.api.models import CompiledModel
from pyamplicol.api.requests import ModelSource, ProcessSet
from pyamplicol.api.results import GenerationResult
from pyamplicol.artifacts import ArtifactBuilder
from pyamplicol.color.correlator_matrices import (
    build_color_correlator_matrices,
    process_color_legs,
)
from pyamplicol.color.plan import GenericColorPlan, build_color_plan
from pyamplicol.config import (
    Action,
    ColorContraction,
    ColorFFTBasis,
    ConfigClamp,
    ConfigResolution,
    EvaluatorExecutionMode,
    GenerationConfig,
    LCFlowLayout,
    RelationDiscoveryMode,
    RunConfig,
)
from pyamplicol.correlators import CorrelatorConfig
from pyamplicol.models.base import Model
from pyamplicol.processes.ir import CanonicalProcessIR
from pyamplicol.reporting import ProgressSink

from .artifact_writer import ApiBundleHook, ArtifactWriteResult
from .contracts import RuntimeExpressionSchema
from .dag_algorithms import _infer_minimal_coupling_order_limits_from_color_plan
from .dag_compiler import compile_generic_dag
from .dag_types import GenericDAG
from .numerical_current_warmup import generic_dag_numerical_current_opt_out_report
from .progress import PhaseHandle
from .service import (
    GenerationBackend,
    _CompiledProcess,
    _DagProcess,
    _EvaluatorProcess,
    _NoModelSupportedAmplitudes,
    _PreparedProcessConstruction,
    _ProcessSelection,
)
from .validation import build_validation_point

CORRELATOR_PAYLOAD_PATH = "correlators.json"


def correlated_configuration(
    config: GenerationConfig | RunConfig | ConfigResolution | None,
) -> ConfigResolution:
    """Retain requested colour accuracy with direct/compiled source execution."""

    if isinstance(config, ConfigResolution):
        requested, effective, clamps = config.requested, config.effective, config.clamps
    elif isinstance(config, RunConfig):
        requested = effective = config
        clamps = ()
    elif config is None or isinstance(config, GenerationConfig):
        requested = effective = RunConfig(
            action=Action.GENERATE,
            generation=GenerationConfig() if config is None else config,
        )
        clamps = ()
    else:
        raise TypeError("invalid correlated generation configuration")
    if (
        effective.color.contraction is ColorContraction.SYMMETRIC_GROUP_FFT
        and effective.color.fft_basis is ColorFFTBasis.ADJOINT
    ):
        raise ValueError(
            "adjoint FFT is not supported for correlated matrix elements; "
            "use the direct colour contraction"
        )
    result = replace(
        effective,
        color=replace(
            effective.color,
            contraction=ColorContraction.DIRECT,
            fft_basis=ColorFFTBasis.TRACE,
            lc_flow_layout=LCFlowLayout.TOPOLOGY_REPLAY,
        ),
        evaluator=replace(
            effective.evaluator, execution_mode=EvaluatorExecutionMode.COMPILED
        ),
        generation=replace(
            effective.generation,
            relation_discovery=replace(
                effective.generation.relation_discovery, mode=RelationDiscoveryMode.OFF
            ),
        ),
    )
    paths = (
        "color.contraction",
        "color.fft_basis",
        "color.lc_flow_layout",
        "evaluator.execution_mode",
        "generation.relation_discovery.mode",
    )

    def field(config: RunConfig, path: str) -> object:
        value: object = config
        for name in path.split("."):
            value = getattr(value, name)
        return value

    adjustments = tuple(
        ConfigClamp(
            path=path,
            requested=field(effective, path),
            effective=field(result, path),
            reason="correlations require complete coherent generic compiled amplitudes",
        )
        for path in paths
        if field(effective, path) != field(result, path)
    )
    # The source configuration may already have resolved recurrence's auto
    # compression to false. Re-select the default for this compiled lane, but
    # retain explicit user or prepared-pack choices.
    evaluator = result.evaluator
    if requested.evaluator.jit.compress == "auto" and not any(
        clamp.path == "evaluator.jit.compress" for clamp in clamps
    ):
        evaluator = replace(evaluator, jit=replace(evaluator.jit, compress="auto"))
    result = replace(
        result,
        evaluator=replace(
            evaluator,
            jit=replace(evaluator.jit, compress=evaluator.resolved_jit_compress),
        ),
    )
    return ConfigResolution(requested, result, (*clamps, *adjustments))


@dataclass(frozen=True)
class CorrelatedProcessMetadata:
    """Generation-local inputs; no new serialized copy of the DAG or model."""

    process: CanonicalProcessIR
    color_plan: GenericColorPlan
    runtime_schema: RuntimeExpressionSchema
    aliases: tuple[Mapping[str, object], ...]


class CorrelatedGenerationBackend(GenerationBackend):
    """Safety lane sharing model resolution, stage compilation and publishing."""

    def __init__(
        self,
        config: GenerationConfig | RunConfig | ConfigResolution | None,
        progress: ProgressSink | None,
        *,
        declarations: CorrelatorConfig,
        api_bundle_hook: ApiBundleHook | None = None,
        process_selection: _ProcessSelection | None = None,
    ) -> None:
        if not isinstance(declarations, CorrelatorConfig):
            raise TypeError("correlators must be a CorrelatorConfig")
        super().__init__(
            correlated_configuration(config),
            progress,
            api_bundle_hook=api_bundle_hook,
            process_selection=process_selection,
        )
        self.declarations = declarations
        self._correlated_processes: dict[str, CorrelatedProcessMetadata] = {}
        selection = self._process_selection
        partial = tuple(
            name
            for name in (
                "max_color_sectors",
                "reference_color_order",
                "selected_color_sector_ids",
                "selected_source_helicities",
            )
            if getattr(selection, name) is not None
        )
        if partial:
            raise GenerationError(
                "correlated generation requires complete colour and helicity coverage; "
                "remove " + ", ".join(f"process.{name}" for name in partial)
            )

    @property
    def _color_accuracy(self) -> str:
        # The inherited setting selects the correlated metric, not the source
        # basis or native ordinary consumer. All amplitudes remain full colour.
        return "full"

    @property
    def process_metadata(self) -> Mapping[str, CorrelatedProcessMetadata]:
        return MappingProxyType(self._correlated_processes)

    def generate(
        self,
        processes: ProcessSet,
        output: PathLike[str] | str,
        *,
        model: ModelSource | CompiledModel | None = None,
        mode: str = "error",
    ) -> GenerationResult:
        if mode == "append":
            raise GenerationError("correlated generation does not support append mode")
        for alias in processes.aliases:
            permutation = tuple(alias.particle_permutation)
            if permutation and permutation != tuple(range(len(permutation))):
                raise GenerationError(
                    "correlated generation currently requires identity process aliases"
                )
        self._correlated_processes.clear()
        return super().generate(processes, output, model=model, mode=mode)

    def _prepare_process_construction(
        self, process: CanonicalProcessIR, model: Model
    ) -> _PreparedProcessConstruction:
        if process.color_accuracy != "full":
            raise GenerationError("correlated process expansion must use full colour")
        by_label = {leg.label: leg for leg in process.legs}
        for label in self.declarations.spin_legs:
            leg = by_label.get(label)
            if leg is None or leg.outgoing_pdg is None:
                raise GenerationError(
                    f"spin-correlated leg {label} is absent from {process.process!r}"
                )
            source = model._source_ir(int(leg.outgoing_pdg))
            if (
                source.wavefunction_family != "vector"
                or source.component_dimension != 4
            ):
                raise GenerationError(
                    f"spin-correlated leg {label} requires a full "
                    "four-component vector source"
                )
        plan = build_color_plan(
            process, color_accuracy="full", fold_trace_reflections=False
        )
        if not plan.sectors or plan.truncated:
            raise GenerationError(
                f"process {process.process!r} has no complete colour plan"
            )
        limits = self._coupling_order_limits
        run = self._run_config
        if run is not None and run.process.coupling_order_policy == "minimal":
            limits = (
                _infer_minimal_coupling_order_limits_from_color_plan(
                    process,
                    model=model,
                    color_plan=plan,
                    max_coupling_orders=limits or None,
                )
                or limits
            )
        return _PreparedProcessConstruction(
            process=process,
            complete_color_plan=plan,
            restricted_color_plan=plan,
            materialized_sector_ids=None,
            coupling_order_limits=MappingProxyType(dict(limits)),
            topology_replay=None,
        )

    def _compile_concrete_process(
        self,
        process: CanonicalProcessIR,
        model: Model,
        *,
        progress_callback: Callable[[Mapping[str, str | int]], None] | None = None,
    ) -> tuple[GenericDAG, dict[str, object]]:
        prepared = self._prepare_process_construction(process, model)
        limits = dict(prepared.coupling_order_limits)
        dag = compile_generic_dag(
            process,
            model=model,
            color_plan=prepared.complete_color_plan,
            max_coupling_orders=limits or None,
            max_quark_pairs=self._max_quark_pairs,
            selected_source_helicities=None,
            selected_color_sector_ids=None,
            online_evaluation_reuse=False,
            backward_live_planning=False,
            lc_all_ordering_symmetry=False,
            closure_side_mask_pruning=False,
            color_order_mask_pruning=False,
            species_reachability_pruning=False,
            progress_callback=progress_callback,
        )
        if dag.truncated:
            raise GenerationError(
                f"process {process.process!r} DAG was unexpectedly truncated"
            )
        if not dag.has_amplitudes:
            raise _NoModelSupportedAmplitudes(
                f"process {process.process!r} has no model-supported "
                "tree-level amplitudes"
            )
        return dag, {
            "key": process.key,
            "process": process.process,
            "color_sector_count": dag.color_plan.sector_count,
            "color_coverage": dag.color_coverage,
            "helicity_coverage": dag.helicity_coverage,
            "source_count": len(dag.sources),
            "current_count": len(dag.currents),
            "interaction_count": len(dag.interactions),
            "interaction_evaluation_count": dag.interaction_evaluation_count,
            "amplitude_root_count": len(dag.amplitude_roots),
            "coupling_order_limits": limits,
        }

    def _prepare_warmup_process_inner(
        self,
        process: _DagProcess,
        model: Model,
        *,
        index: int,
        progress: PhaseHandle,
    ) -> _CompiledProcess:
        # No parity reduction, source-zero discovery, replay or numerical reuse.
        # Even exact identities proved only on physical sources are inapplicable.
        dag = process.dag
        validation = self._generation_config.validation
        sample_count = (
            validation.samples
            if validation.enabled and validation.post_build_validation
            else 1
        )
        points = tuple(
            build_validation_point(
                dag,
                model,
                process_id=process.expanded.request.name,
                seed=validation.seed + index * sample_count + sample_index,
            )
            for sample_index in range(sample_count)
        )
        relation = generic_dag_numerical_current_opt_out_report(
            dag, execution_mode="compiled"
        )
        filters = {
            "structural_helicity_reduction": {
                "applied": False,
                "before_amplitude_roots": len(dag.amplitude_roots),
                "after_amplitude_roots": len(dag.amplitude_roots),
                "mode": "complete correlated source basis",
            },
            "relation_discovery": relation,
        }
        progress.update(2, message="complete correlated source basis retained")
        return _CompiledProcess(
            expanded=process.expanded,
            dag=dag,
            helicity_sum_dag=None,
            helicity_selector_union_dag=None,
            coverage={**process.coverage, "relation_discovery": relation},
            filters=filters,
            validation_points=points,
        )

    def _construct_evaluator(
        self, process: _CompiledProcess, model: Model, phase: PhaseHandle
    ) -> _EvaluatorProcess:
        result = super()._construct_evaluator(process, model, phase)
        if (
            result.helicity_sum_runtime_schema is not None
            or result.helicity_selector_lanes
            or result.color_selector_lanes
        ):
            raise GenerationError(
                "correlated generation cannot contain companion or replay lanes"
            )
        # Full source coverage does not promise the separate native per-point
        # selector ABI.  This lane deliberately has no selector/recurrence
        # schedules; the correlated executor selects its coherent groups itself.
        payload = result.runtime_schema.to_mapping()
        physics = dict(cast(Mapping[str, object], payload["physics"]))
        extensions = dict(cast(Mapping[str, object], physics.get("extensions", {})))
        extensions.pop("runtime_selectors", None)
        physics["extensions"] = extensions
        payload["physics"] = physics
        schema = RuntimeExpressionSchema.from_mapping(payload)
        result = replace(
            result,
            runtime_schema=schema,
            stage_input=replace(result.stage_input, runtime_schema=schema),
        )
        self._correlated_processes[process.expanded.request.name] = (
            CorrelatedProcessMetadata(
                process=process.dag.process,
                color_plan=process.dag.color_plan,
                runtime_schema=result.runtime_schema,
                aliases=process.expanded.aliases,
            )
        )
        return result

    def correlator_payload(self) -> dict[str, object]:
        """Build the declared exact matrices once, at the ordinary write boundary."""

        processes: dict[str, object] = {}
        for process_id, metadata in self._correlated_processes.items():
            declarations = self.declarations._resolve_color(
                process_color_legs(metadata.process)
            )
            amplitude = cast(
                Mapping[str, object],
                metadata.runtime_schema.to_mapping()["amplitude_stage"],
            )
            groups = cast(Sequence[Mapping[str, object]], amplitude["coherent_groups"])
            basis_by_helicity: dict[tuple[int, ...], set[int]] = {}
            for group in groups:
                helicities = tuple(cast(Sequence[int], group["helicities"]))
                sector_id = cast(int, group["color_sector_id"])
                basis = basis_by_helicity.setdefault(helicities, set())
                basis.add(sector_id)
            if not basis_by_helicity:
                raise GenerationError("correlated amplitudes have no coherent groups")
            owners = next(iter(basis_by_helicity.values()))
            if any(basis != owners for basis in basis_by_helicity.values()):
                raise GenerationError(
                    "correlated colour basis differs between retained helicities"
                )
            # Open-line colour plans retain aliases whose amplitudes already
            # belong to canonical owners.  Never contract those aliases twice.
            matrices = build_color_correlator_matrices(
                metadata.color_plan,
                declarations.color_requests,
                sector_ids=sorted(owners),
                color_accuracy=cast(
                    Literal["lc", "nlc", "full"],
                    self._run_config.color.accuracy
                    if self._run_config is not None
                    else "lc",
                ),
            )
            processes[process_id] = {
                "declarations": declarations.to_json_dict(),
                "spin_legs": list(declarations.spin_legs),
                "matrices": [matrix.to_json_dict() for matrix in matrices],
                # The native execution schema keeps root group IDs but omits
                # these compiler descriptors, needed only by this consumer.
                "coherent_groups": groups,
            }
        return {
            "schema_version": 1,
            "complete_source_basis": True,
            "declarations": self.declarations.to_json_dict(),
            "processes": processes,
        }

    def _write_correlator_payload(
        self, builder: ArtifactBuilder
    ) -> Mapping[str, object]:
        builder.add_json(
            CORRELATOR_PAYLOAD_PATH, self.correlator_payload(), role="runtime-physics"
        )
        return {"correlators": {"schema_version": 1, "path": CORRELATOR_PAYLOAD_PATH}}

    def _write_artifact(self, *args: object, **kwargs: object) -> ArtifactWriteResult:
        return super()._write_artifact(
            *args, **kwargs, payload_hook=self._write_correlator_payload
        )


def generate_correlated(
    processes: ProcessSet,
    destination: str | Path,
    *,
    declarations: CorrelatorConfig,
    config: GenerationConfig | RunConfig | ConfigResolution | None = None,
    progress: ProgressSink | None = None,
    model: ModelSource | CompiledModel | None = None,
    mode: Literal["error", "append", "replace"] = "error",
) -> GenerationResult:
    return CorrelatedGenerationBackend(
        config, progress, declarations=declarations
    ).generate(processes, destination, model=model, mode=mode)

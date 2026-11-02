# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, TypeVar, cast

from pyamplicol.api.errors import ConfigurationError
from pyamplicol.config import ConfigResolution, EvaluatorConfig, RunConfig
from pyamplicol.reporting import (
    ProgressEnd,
    ProgressSink,
    ProgressStart,
    ProgressUpdate,
)

if TYPE_CHECKING:
    from pyamplicol.api import ModelSource, ProcessSet
    from pyamplicol.artifacts.inspection import (
        ArtifactInspection,
        ArtifactProcessInspection,
    )
    from pyamplicol.correlators import CorrelatorConfig
    from pyamplicol.models.loading import CompiledModel as _CompiledModelPayload

_T = TypeVar("_T")


class CliServices(Protocol):
    """Command-level service boundary used by CLI handlers."""

    def generate(self, config: RunConfig, progress: ProgressSink) -> object: ...

    def plan(self, config: RunConfig, progress: ProgressSink) -> object: ...

    def evaluate(self, config: RunConfig, progress: ProgressSink) -> object: ...

    def benchmark(self, config: RunConfig, progress: ProgressSink) -> object: ...

    def inspect(
        self,
        config: RunConfig,
        progress: ProgressSink,
        *,
        full_physics: bool = False,
    ) -> object: ...

    def model_inspect(self, config: RunConfig, progress: ProgressSink) -> object: ...

    def model_compile(self, config: RunConfig, progress: ProgressSink) -> object: ...

    def model_processes(self, config: RunConfig, progress: ProgressSink) -> object: ...

    def group_processes(self, config: RunConfig, progress: ProgressSink) -> object: ...


def dispatch(
    config: RunConfig,
    services: CliServices,
    progress: ProgressSink,
    *,
    dry_run: bool = False,
    full_physics: bool = False,
) -> object:
    """Dispatch a typed config without importing a concrete domain backend."""

    if config.action == "generate":
        return (
            services.plan(config, progress)
            if dry_run
            else services.generate(config, progress)
        )
    if config.action == "evaluate":
        return services.evaluate(config, progress)
    if config.action == "benchmark":
        return services.benchmark(config, progress)
    if config.action == "inspect":
        if full_physics:
            return services.inspect(config, progress, full_physics=True)
        return services.inspect(config, progress)
    if config.action == "model-inspect":
        return services.model_inspect(config, progress)
    if config.action == "model-compile":
        return services.model_compile(config, progress)
    if config.action == "model-processes":
        return services.model_processes(config, progress)
    if config.action == "group-processes":
        return services.group_processes(config, progress)
    raise ConfigurationError(f"unsupported action {config.action!r}")


def _model_source(config: RunConfig) -> ModelSource:
    from pyamplicol.api import ModelSource

    return ModelSource.from_config(config.model)


def _process_set(config: RunConfig) -> ProcessSet:
    from pyamplicol.api import ProcessRequest, ProcessSet

    requests: list[ProcessRequest] = []
    errors: list[str] = []
    for index, entry in enumerate(config.process.entries):
        try:
            requests.append(ProcessRequest.parse(entry.expression, name=entry.name))
        except (TypeError, ValueError) as exc:
            errors.append(f"process.entries[{index}]: {exc}")
    if errors:
        raise ConfigurationError(errors)
    try:
        return ProcessSet(tuple(requests))
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(str(exc)) from exc


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigurationError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"invalid JSON in {path}: {exc}") from exc


def _load_process_output(
    path: Path,
    progress: ProgressSink,
    callback: Callable[[], _T],
) -> _T:
    resolved = path.expanduser().resolve(strict=False)
    task_id = "process-output-load"
    started = time.perf_counter()
    progress.emit(
        ProgressStart(
            task_id,
            f"Loading process output {resolved}",
            details={"step": "loading process output", "path": str(resolved)},
        )
    )
    try:
        result = callback()
    except Exception as exc:
        progress.emit(
            ProgressEnd(
                task_id,
                success=False,
                message=str(exc),
                elapsed_seconds=time.perf_counter() - started,
            )
        )
        raise
    progress.emit(
        ProgressEnd(
            task_id,
            elapsed_seconds=time.perf_counter() - started,
        )
    )
    return result


def _select_compact_process_inspection(
    inspection: ArtifactInspection,
    selector: str,
) -> ArtifactInspection:
    """Select one representative without loading its executable runtime."""

    def select(matches: list[ArtifactProcessInspection]) -> ArtifactInspection:
        representatives = {item.id: item for item in matches}
        if len(representatives) > 1:
            identifiers = ", ".join(sorted(representatives))
            raise ConfigurationError(
                f"inspect process selector {selector!r} is ambiguous across "
                f"representatives: {identifiers}"
            )
        if representatives:
            selected = next(iter(representatives.values()))
            groups = inspection.process_groups
            if groups is not None:
                groups = replace(
                    groups,
                    groups=tuple(
                        group
                        for group in groups.groups
                        if any(member.id == selected.id for member in group.members)
                    ),
                )
            return replace(inspection, processes=(selected,), process_groups=groups)
        groups = inspection.process_groups
        for group in () if groups is None else groups.groups:
            member = next(
                (
                    item
                    for item in group.members[1:]
                    if not item.generated and selector in {item.id, item.process}
                ),
                None,
            )
            if member is not None:
                raise ConfigurationError(
                    f"inspect process {member.id} ({member.process}) is a routed "
                    f"member of process group {group.representative_id} "
                    f"({group.representative}) with no runtime of its own; "
                    f"--process {group.representative_id} lists the group, and "
                    "--full-physics shows the member's relation and physics"
                )
        raise ConfigurationError(
            f"inspect process selector {selector!r} did not match a stable "
            "representative ID, alias ID, or exact process expression; add "
            "--full-physics to resolve permutations and process-group members"
        )

    identifier_matches = [
        process
        for process in inspection.processes
        if process.id == selector
        or any(alias.id == selector for alias in process.aliases)
    ]
    if identifier_matches:
        return select(identifier_matches)
    return select(
        [
            process
            for process in inspection.processes
            if process.expression == selector
            or any(alias.expression == selector for alias in process.aliases)
        ]
    )


def _full_physics_inspection(artifact: Path, selector: str | None) -> object:
    """Return the selected process physics and, when grouped, its relation.

    Rusticol owns process selection, so side permutations and non-generated
    process-group members resolve as in `evaluate`. A grouped selection keeps
    every physics field at the top level and adds ``process_group``.
    """

    from pyamplicol.api import Runtime
    from pyamplicol.api.results import ProcessPhysics
    from pyamplicol.artifacts.inspection import process_group_selection

    runtime = Runtime.load(artifact, process=selector)
    physics = runtime.physics
    if not isinstance(physics, ProcessPhysics):
        return physics
    group = process_group_selection(
        artifact,
        physics.process_id,
        permutation=runtime.external_permutation,
    )
    if group is None:
        return physics
    return {
        **{item.name: getattr(physics, item.name) for item in fields(physics)},
        "process_group": group,
    }


def _compile_configured_model(
    config: RunConfig,
    *,
    require_supported: bool,
) -> _CompiledModelPayload:
    from pyamplicol.api.requests import ModelSource, _compile_model_source_payload

    payload = _compile_model_source_payload(
        ModelSource.from_config(config.model),
        cache_dir=config.model.cache_dir,
        use_cache=config.model.cache,
        require_supported=require_supported,
    )
    return cast("_CompiledModelPayload", payload)


def _model_inspection_payload(compiled: _CompiledModelPayload) -> dict[str, object]:
    return {
        "name": compiled.name,
        "supported": compiled.supported,
        "source": dict(compiled.source),
        "producer": dict(compiled.producer),
        "capabilities": dict(compiled.capabilities),
        "issues": [issue.to_dict() for issue in compiled.issues],
        "conversion_seconds": compiled.conversion_seconds,
        "phase_timings": dict(compiled.phase_timings),
        "contents": {
            "orders": [order.name for order in compiled.ir.orders],
            "particles": [particle.to_dict() for particle in compiled.ir.particles],
            "runtime_parameter_names": sorted(compiled.parameter_defaults),
        },
    }


class DefaultCliServices:
    """Thin adapters over the public API; backend imports remain first-use only."""

    def __init__(
        self,
        *,
        resolution: ConfigResolution | None = None,
        correlators: CorrelatorConfig | None = None,
    ) -> None:
        self._resolution = resolution
        self._correlators = correlators

    def _generation_config(self, config: RunConfig) -> RunConfig | ConfigResolution:
        if self._resolution is None:
            return config
        if self._resolution.effective != config:
            raise ConfigurationError(
                "CLI generation config does not match its resolved provenance"
            )
        return self._resolution

    def generate(self, config: RunConfig, progress: ProgressSink) -> object:
        from pyamplicol.api import Generator

        if not config.process.entries:
            raise ConfigurationError("generate requires at least one process request")
        if config.generation.output is None:
            raise ConfigurationError("generate requires generation.output")
        processes = _process_set(config)
        return Generator(
            config=self._generation_config(config), progress=progress
        ).generate(
            processes,
            config.generation.output,
            model=_model_source(config),
            mode=cast(
                Literal["error", "append", "replace"],
                str(config.generation.mode),
            ),
            **(
                {"correlators": self._correlators}
                if self._correlators is not None
                else {}
            ),
        )

    def plan(self, config: RunConfig, progress: ProgressSink) -> object:
        from pyamplicol.api import Generator

        if not config.process.entries:
            raise ConfigurationError("generate --dry-run requires a process request")
        processes = _process_set(config)
        return Generator(
            config=self._generation_config(config), progress=progress
        ).plan(
            processes,
            model=_model_source(config),
        )

    def evaluate(self, config: RunConfig, progress: ProgressSink) -> object:
        from pyamplicol.api import Runtime

        if config.evaluation.artifact is None:
            raise ConfigurationError("evaluate requires evaluation.artifact")
        if config.evaluation.momenta is None:
            raise ConfigurationError("evaluate requires evaluation.momenta")
        parameters: (
            Mapping[
                str,
                complex
                | float
                | int
                | list[float | int]
                | tuple[float | int, float | int],
            ]
            | None
        ) = None
        if config.evaluation.model_parameters is not None:
            raw_parameters = _read_json(config.evaluation.model_parameters)
            if not isinstance(raw_parameters, Mapping):
                raise ConfigurationError("model parameter JSON must be an object")
            parameters = raw_parameters
        momenta = _read_json(config.evaluation.momenta)
        if not isinstance(momenta, list):
            raise ConfigurationError("momenta JSON must be a point list")
        runtime = _load_process_output(
            config.evaluation.artifact,
            progress,
            lambda: Runtime.load(
                config.evaluation.artifact,
                process=config.evaluation.process,
                model_parameters=parameters,
            ),
        )
        selectors = {
            "helicities": config.evaluation.helicity_ids or None,
            "color_flows": config.evaluation.color_flow_ids or None,
        }
        if config.evaluation.resolved:
            return runtime.evaluate_resolved(
                momenta,
                precision=config.evaluation.precision,
                **selectors,
            )
        return runtime.evaluate(
            momenta,
            precision=config.evaluation.precision,
            **selectors,
        )

    def benchmark(self, config: RunConfig, progress: ProgressSink) -> object:
        from pyamplicol.api import BenchmarkRunner

        if config.evaluation.artifact is None:
            raise ConfigurationError("benchmark requires an artifact target")
        points = None
        if config.evaluation.momenta is not None:
            raw_points = _read_json(config.evaluation.momenta)
            if not isinstance(raw_points, list):
                raise ConfigurationError("momenta JSON must be a point list")
            points = raw_points
        return BenchmarkRunner(config=config, progress=progress).run(
            config.evaluation.artifact,
            points=points,
        )

    def inspect(
        self,
        config: RunConfig,
        progress: ProgressSink,
        *,
        full_physics: bool = False,
    ) -> object:
        if config.evaluation.artifact is None:
            raise ConfigurationError("inspect requires evaluation.artifact")
        if full_physics and config.evaluation.process is None:
            raise ConfigurationError("inspect --full-physics requires --process")

        selector = config.evaluation.process
        if full_physics:
            artifact = config.evaluation.artifact
            return _load_process_output(
                artifact,
                progress,
                lambda: _full_physics_inspection(artifact, selector),
            )

        from pyamplicol.artifacts import inspect_artifact

        inspection = _load_process_output(
            config.evaluation.artifact,
            progress,
            lambda: inspect_artifact(config.evaluation.artifact),
        )
        if selector is None:
            return inspection
        return _select_compact_process_inspection(inspection, selector)

    def model_inspect(self, config: RunConfig, progress: ProgressSink) -> object:
        del progress
        return _model_inspection_payload(
            _compile_configured_model(config, require_supported=False)
        )

    def model_compile(self, config: RunConfig, progress: ProgressSink) -> object:
        if config.generation.output is None:
            raise ConfigurationError("model compile requires generation.output")
        requested_output = config.generation.output
        prepared_output = requested_output.name.lower().endswith(".pyamplicol-model")
        default_evaluator = EvaluatorConfig()
        resolved_default_evaluator = replace(
            default_evaluator,
            jit=replace(
                default_evaluator.jit, compress=default_evaluator.resolved_jit_compress
            ),
        )
        if not prepared_output and config.evaluator not in (
            default_evaluator,
            resolved_default_evaluator,
        ):
            raise ConfigurationError(
                "model compile evaluator settings require an output ending "
                "with '.pyamplicol-model'; IR-only outputs do not compile kernels"
            )
        compiled = _compile_configured_model(config, require_supported=True)
        result: dict[str, object] = {
            "model": compiled.name,
            "supported": compiled.supported,
            "conversion_seconds": compiled.conversion_seconds,
            "phase_timings": dict(compiled.phase_timings),
            "source": dict(compiled.source),
            "capabilities": dict(compiled.capabilities),
        }
        if not prepared_output:
            output = compiled.write(requested_output)
            result["output"] = str(output)
            return result

        from pyamplicol.models.prepared_compile import prepare_model_bundle

        task_id = "model-prepare"
        started = False
        task_started_at = time.perf_counter()
        backend = str(config.evaluator.backend).upper()

        def report_preparation(label: str, completed: int, total: int) -> None:
            nonlocal started
            if not started:
                progress.emit(
                    ProgressStart(
                        task_id=task_id,
                        description="Prepare model evaluator kernels",
                        total=total,
                        unit="kernels",
                        details={
                            "step": "prepared kernel compilation",
                            "backend": backend,
                            "kernel_total": total,
                        },
                    )
                )
                started = True
            progress.emit(
                ProgressUpdate(
                    task_id=task_id,
                    completed=completed,
                    total=total,
                    message=label,
                    details={
                        "step": label,
                        "backend": backend,
                        "kernel_index": min(completed + 1, total),
                        "kernel_total": total,
                    },
                )
            )

        try:
            prepared = prepare_model_bundle(
                compiled,
                requested_output,
                evaluator=config.evaluator,
                progress=report_preparation,
            )
        except BaseException:
            if started:
                progress.emit(
                    ProgressEnd(
                        task_id=task_id,
                        success=False,
                        message="Prepared model compilation failed",
                        elapsed_seconds=time.perf_counter() - task_started_at,
                    )
                )
            raise
        if started:
            progress.emit(
                ProgressEnd(
                    task_id=task_id,
                    message="Prepared model compilation complete",
                    elapsed_seconds=time.perf_counter() - task_started_at,
                )
            )
        result.update(
            {
                "output": str(prepared.output),
                "prepared_backend": prepared.bundle.backend,
                "kernel_count": prepared.kernel_count,
                "preparation_phase_timings": dict(prepared.phase_timings_seconds),
            }
        )
        return result

    def model_processes(self, config: RunConfig, progress: ProgressSink) -> object:
        del progress
        if not config.process.entries:
            raise ConfigurationError("model processes requires a process request")

        from pyamplicol.processes.conservation import conserved_charges
        from pyamplicol.processes.model import (
            ModelParticleCatalog,
            build_model_process_ir,
            expand_model_processes,
        )

        compiled = _compile_configured_model(config, require_supported=True)
        charges = conserved_charges(compiled.ir)
        catalog = ModelParticleCatalog(
            compiled.ir.name,
            compiled.ir.particles,
            compiled.ir.parameters,
        )
        multiparticles = {
            **catalog.default_multiparticles(),
            **config.process.multiparticles,
        }
        entries: dict[str, dict[str, object]] = {}
        for entry in config.process.entries:
            request = entry.expression
            for process in expand_model_processes(
                request,
                catalog,
                multiparticles=multiparticles,
                charges=charges,
            ):
                process_ir = build_model_process_ir(
                    process,
                    compiled.ir,
                    color_accuracy=config.color.accuracy,
                )
                entries.setdefault(
                    process_ir.key,
                    {
                        "key": process_ir.key,
                        "process": process_ir.process,
                        "ir": process_ir.to_json_dict(),
                    },
                )
        concrete = list(entries.values())
        return {
            "available": True,
            "model": compiled.name,
            "model_source": dict(compiled.source),
            "requests": [entry.expression for entry in config.process.entries],
            "default_key": concrete[0]["key"] if concrete else None,
            "n_entries": len(concrete),
            "multiparticles": {
                name: list(values) for name, values in sorted(multiparticles.items())
            },
            "entries": concrete,
        }

    def group_processes(self, config: RunConfig, progress: ProgressSink) -> object:
        del progress
        if config.evaluation.artifact is None:
            raise ConfigurationError("group-processes requires an artifact")

        from pyamplicol.api.process_groups import group_processes

        return group_processes(
            config.evaluation.artifact,
            analysis=str(config.process.analyze_process_groups),
            only_unit_multiplier=config.process.only_unit_multiplier,
            consider_charge_conjugation=config.process.consider_charge_conjugation,
        )


__all__ = ["CliServices", "DefaultCliServices", "dispatch"]

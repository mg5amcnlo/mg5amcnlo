#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Run the bounded compiled-versus-eager acceptance matrix.

This driver is intentionally independent of the publication-report tooling.
It generates one complete compiled artifact
and one complete eager artifact per cell, then records native Rusticol wall
timings in one machine-readable result. LC workloads additionally generate the
conventional specialized compiled references used by the report so reusable
eager selection is not mistaken for a speedup over a differently specialized
baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
WATCHDOG = ROOT / "tools" / "ci" / "memory_watchdog.py"
COMPARISON_MODULE = "tools.developer.eager_artifact_compare"
SELECTOR_PROFILE_MODULE = "tools.developer.runtime_selector_profile"
DEFAULT_GENERATION_TIMEOUT = 300.0
DEFAULT_SUITE_TIMEOUT = 300.0
DEFAULT_MEMORY_LIMIT_GIB = 30.0
DEFAULT_BATCH_SIZES = (1, 128, 1024)
SMOKE_BATCH_SIZES = (128, 1024)
DEFAULT_COLORS = ("lc", "nlc", "full")
HARD_GENERATION_SPEEDUP = 7.0
SOFT_GENERATION_SPEEDUP = 10.0
SOFT_RUNTIME_RATIO = 1.5
HARD_REUSABLE_RUNTIME_RATIO = 1.25
_TOPOLOGY_FIELDS = (
    "physical_helicities",
    "computed_helicities",
    "physical_color_components",
    "computed_color_components",
    "invocation_count",
    "attachment_count",
    "evaluation_alias_count",
    "maximum_fanout",
    "finalization_count",
    "closure_count",
)
_OPTIONAL_TOPOLOGY_FIELDS = frozenset(("maximum_fanout",))


@dataclass(frozen=True, slots=True)
class ProcessCase:
    key: str
    process: str
    n_final: int
    feature: str
    compiled_limit_seconds: float
    smoke: bool = False


PROCESS_CASES = (
    ProcessCase(
        "dd_z_3g",
        "d d~ > z g g g",
        4,
        "color singlet with mixed QCD/electroweak recursion",
        3.3,
        smoke=True,
    ),
    ProcessCase(
        "gg_tt_2g",
        "g g > t t~ g g",
        4,
        "gluon-initiated massive fermions and two adjoints",
        20.6,
    ),
    ProcessCase(
        "gg_4g",
        "g g > g g g g",
        4,
        "pure Yang-Mills dense-color recursion",
        128.3,
    ),
    ProcessCase(
        "dd_3q_1g",
        "d d~ > u u~ s s~ g",
        5,
        "three open quark lines and nontrivial color contraction",
        8.4,
        smoke=True,
    ),
    ProcessCase(
        "dd_tt_3g",
        "d d~ > t t~ g g g",
        5,
        "high-fanout massive-QCD recursion",
        108.3,
    ),
)


class MatrixError(RuntimeError):
    """Raised when a benchmark command or acceptance invariant fails."""


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _csv_values(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("must contain at least one value")
    return values


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in _csv_values(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be a comma-separated integer list"
        ) from error
    if any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("batch sizes must be positive")
    return values


def _selected_cases(suite: str, requested: Sequence[str]) -> tuple[ProcessCase, ...]:
    allowed = {case.key: case for case in PROCESS_CASES}
    if requested:
        unknown = sorted(set(requested) - allowed.keys())
        if unknown:
            raise MatrixError(f"unknown process case(s): {', '.join(unknown)}")
        return tuple(allowed[key] for key in requested)
    if suite == "smoke":
        return tuple(case for case in PROCESS_CASES if case.smoke)
    return PROCESS_CASES


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _run_json(
    command: Sequence[str],
    *,
    timeout: float,
    memory_limit_gib: float,
    environment: Mapping[str, str],
) -> tuple[dict[str, Any], float, str]:
    guarded = (
        sys.executable,
        str(WATCHDOG),
        "--limit-gib",
        str(memory_limit_gib),
        "--",
        *command,
    )
    started = time.monotonic()
    process = subprocess.Popen(
        guarded,
        cwd=ROOT,
        env=dict(environment),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        _terminate_process_group(process)
        raise MatrixError(
            f"command exceeded {timeout:.1f}s: {' '.join(command)}"
        ) from error
    elapsed = time.monotonic() - started
    if process.returncode != 0:
        output_sections: list[str] = []
        if stdout.strip():
            output_sections.append(f"stdout (tail):\n{stdout.strip()[-8000:]}")
        if stderr.strip():
            output_sections.append(f"stderr (tail):\n{stderr.strip()[-8000:]}")
        detail = "\n".join(output_sections) or f"exit code {process.returncode}"
        raise MatrixError(f"command failed: {' '.join(command)}\n{detail}")
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise MatrixError(
            f"command did not emit JSON: {' '.join(command)}\n{stdout[-2000:]}"
        ) from error
    if not isinstance(payload, dict):
        raise MatrixError("command JSON result must be an object")
    return payload, elapsed, stderr


def _artifact_process(artifact: Path) -> tuple[str, dict[str, Any]]:
    try:
        outer = json.loads((artifact / "artifact.json").read_text(encoding="utf-8"))
        processes = outer["processes"]
        if not isinstance(processes, list) or len(processes) != 1:
            raise TypeError
        process_id = str(processes[0]["id"])
        physics = json.loads(
            (artifact / "processes" / process_id / "physics.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise MatrixError(f"invalid generated artifact at {artifact}") from error
    if not isinstance(physics, dict):
        raise MatrixError(f"invalid physics payload in {artifact}")
    return process_id, physics


def _first_computed_record(records: object, *, label: str) -> dict[str, Any]:
    if not isinstance(records, list):
        raise MatrixError(f"physics {label} must be a list")
    for record in records:
        if not isinstance(record, dict):
            continue
        if record.get("computed") is True and record.get("structural_zero") is not True:
            identifier = record.get("id")
            if isinstance(identifier, str) and identifier:
                return record
    raise MatrixError(f"physics payload has no computed {label}")


def _workloads(color: str, physics: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    if color != "lc":
        return ({"name": "summed", "selectors": {}},)
    color_record = _first_computed_record(
        physics.get("color_components"), label="color flow"
    )
    helicity_record = _first_computed_record(
        physics.get("helicities"), label="helicity"
    )
    color_id = str(color_record["id"])
    helicity_id = str(helicity_record["id"])
    color_word = color_record.get("word")
    helicity_values = helicity_record.get("values")
    external_particles = physics.get("external_particles")
    if (
        not isinstance(color_word, list)
        or not color_word
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in color_word
        )
    ):
        raise MatrixError("computed LC flow has no valid source-label word")
    if (
        not isinstance(external_particles, list)
        or not isinstance(helicity_values, list)
        or len(external_particles) != len(helicity_values)
    ):
        raise MatrixError("computed helicity does not match external particles")
    source_helicities: dict[int, int] = {}
    for particle, helicity in zip(external_particles, helicity_values, strict=True):
        if not isinstance(particle, dict):
            raise MatrixError("external particle metadata must be an object")
        label = particle.get("label")
        if (
            isinstance(label, bool)
            or not isinstance(label, int)
            or label < 1
            or isinstance(helicity, bool)
            or not isinstance(helicity, int)
        ):
            raise MatrixError("computed helicity has invalid source-label metadata")
        source_helicities[label] = helicity
    return (
        {
            "name": "single-flow-helicity-sum",
            "selectors": {"color_flow": color_id},
            "compiled_specialization": {
                "reference_color_order": tuple(color_word),
                "selected_color_sector_ids": (0,),
            },
        },
        {
            "name": "all-flow-single-helicity",
            "selectors": {"helicity": helicity_id},
            "compiled_specialization": {
                "selected_source_helicities": source_helicities,
            },
        },
    )


def _toml_override_literal(value: object) -> str:
    if isinstance(value, tuple):
        if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
            raise MatrixError("process override tuples must contain integers")
        return "[" + ",".join(str(item) for item in value) + "]"
    if isinstance(value, Mapping):
        entries: list[str] = []
        for key, item in value.items():
            if (
                isinstance(key, bool)
                or not isinstance(key, int)
                or isinstance(item, bool)
                or not isinstance(item, int)
            ):
                raise MatrixError("process override mappings must contain integers")
            entries.append(f"{json.dumps(str(key))}={item}")
        return "{" + ",".join(entries) + "}"
    raise MatrixError(f"unsupported process override value: {value!r}")


def _generation_command(
    python: Path,
    *,
    process: str,
    artifact: Path,
    model: Path | str,
    color: str,
    execution_mode: str,
    process_overrides: Mapping[str, object] | None = None,
) -> tuple[str, ...]:
    command = [
        str(python),
        "-m",
        "pyamplicol",
        "generate",
        process,
        str(artifact),
        "--model",
        str(model),
        "--execution-mode",
        execution_mode,
        "--backend",
        "jit",
        "--jit-optimization-level",
        "3",
        "--color-accuracy",
        color,
        "--mode",
        "replace",
        "--validation",
        "--validation-samples",
        "1",
        "--validation-seed",
        "20260719",
        "--no-post-build-validation",
        "--no-emit-api-bundle",
        "--progress",
        "off",
        "--json",
    ]
    for name, value in (process_overrides or {}).items():
        command.extend(
            ("--set", f"process.{name}={_toml_override_literal(value)}")
        )
    return tuple(command)


def _profile_command(
    python: Path,
    *,
    artifact: Path,
    process_id: str,
    batch_size: int,
    target_runtime: float,
    minimum_samples: int,
    selectors: Mapping[str, str],
) -> tuple[str, ...]:
    command = [
        str(python),
        "-m",
        "pyamplicol",
        "profile",
        str(artifact),
        "--process",
        process_id,
        "--target-runtime",
        str(target_runtime),
        "--batch-size",
        str(batch_size),
        "--warmup-runs",
        "2",
        "--minimum-samples",
        str(minimum_samples),
        "--progress",
        "off",
        "--json",
    ]
    if "color_flow" in selectors:
        command.extend(("--color-flow", selectors["color_flow"]))
    if "helicity" in selectors:
        command.extend(("--helicity", selectors["helicity"]))
    return tuple(command)


def _selector_profile_command(
    python: Path,
    *,
    artifact: Path,
    process_id: str,
    batch_size: int,
    target_runtime: float,
    minimum_samples: int,
    seed: int,
) -> tuple[str, ...]:
    return (
        str(python),
        "-m",
        SELECTOR_PROFILE_MODULE,
        str(artifact),
        "--process",
        process_id,
        "--axis",
        "both",
        "--batch-size",
        str(batch_size),
        "--target-runtime",
        str(target_runtime),
        "--minimum-samples",
        str(minimum_samples),
        "--seed",
        str(seed),
    )


def _comparison_command(
    python: Path,
    *,
    eager_artifact: Path,
    compiled_artifact: Path,
    process_id: str,
    selectors: Mapping[str, str],
    compiled_selectors: str = "same",
) -> tuple[str, ...]:
    command = [
        str(python),
        "-m",
        COMPARISON_MODULE,
        "--eager-artifact",
        str(eager_artifact),
        "--compiled-artifact",
        str(compiled_artifact),
        "--process",
        process_id,
        "--compiled-selectors",
        compiled_selectors,
    ]
    if "color_flow" in selectors:
        command.extend(("--color-flow", selectors["color_flow"]))
    if "helicity" in selectors:
        command.extend(("--helicity", selectors["helicity"]))
    return tuple(command)


def _geometric_mean(values: Sequence[float]) -> float:
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise MatrixError("geometric mean requires positive finite values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _generation_assessment(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, bool], dict[str, bool]]:
    non_lc_speedups: list[float] = []
    lc_speedups: list[float] = []
    grouped: dict[tuple[str, str], dict[str, float]] = {}
    for record in records:
        ratio = float(record["compiled_over_eager_core_generation"])
        color = str(record["color"])
        case = record["case"]
        if not isinstance(case, Mapping):
            raise MatrixError("matrix record has invalid case metadata")
        group = (str(case["key"]), str(record["model"]))
        grouped.setdefault(group, {})[color] = ratio
        if color == "lc":
            lc_speedups.append(ratio)
        else:
            non_lc_speedups.append(ratio)
    complete_groups = bool(grouped) and all(
        set(colors) == set(DEFAULT_COLORS) for colors in grouped.values()
    )
    geometric_means = tuple(
        _geometric_mean(tuple(colors.values())) for colors in grouped.values()
    )
    hard_gates = {
        "nlc_full_each_at_least_7x": bool(non_lc_speedups)
        and all(value >= HARD_GENERATION_SPEEDUP for value in non_lc_speedups),
        "lc_no_generation_regression": bool(lc_speedups)
        and all(value >= 1.0 for value in lc_speedups),
        "per_process_geometric_mean_at_least_7x": complete_groups
        and all(value >= HARD_GENERATION_SPEEDUP for value in geometric_means),
    }
    soft_targets = {
        "nlc_full_each_at_least_10x": bool(non_lc_speedups)
        and all(value >= SOFT_GENERATION_SPEEDUP for value in non_lc_speedups),
        "per_process_geometric_mean_at_least_10x": complete_groups
        and all(value >= SOFT_GENERATION_SPEEDUP for value in geometric_means),
    }
    return hard_gates, soft_targets


def _reusable_runtime_gate(records: Sequence[Mapping[str, Any]]) -> bool:
    """Apply the reusable/specialized bound only to n=4 and n=5 profiles."""

    ratios = tuple(
        float(ratio)
        for record in records
        if int(record["case"]["n_final"]) in {4, 5}
        for workload in record["workloads"]
        for profile in workload["profiles"]
        if int(profile["batch_size"]) in {128, 1024}
        for ratio in [profile["compiled_reusable_over_specialized_wall"]]
        if ratio is not None
    )
    return bool(ratios) and all(
        ratio <= HARD_REUSABLE_RUNTIME_RATIO for ratio in ratios
    )


def _scope_gate(arguments: argparse.Namespace, cases: Sequence[ProcessCase]) -> bool:
    expected_cases = {
        case.key
        for case in PROCESS_CASES
        if arguments.suite == "milestone" or case.smoke
    }
    expected_models = (
        {"built-in", "ufo-sm"} if arguments.suite == "milestone" else {"built-in"}
    )
    expected_batch_sizes = (
        DEFAULT_BATCH_SIZES
        if arguments.suite == "milestone"
        else SMOKE_BATCH_SIZES
    )
    return (
        {case.key for case in cases} == expected_cases
        and set(arguments.models) == expected_models
        and set(arguments.colors) == set(DEFAULT_COLORS)
        and set(arguments.batch_sizes) == set(expected_batch_sizes)
    )


def _eager_topology(artifact: Path) -> dict[str, int]:
    from pyamplicol.artifacts import inspect_artifact

    inspection = inspect_artifact(artifact)
    if len(inspection.processes) != 1:
        raise MatrixError("benchmark artifacts must contain exactly one process")
    process = inspection.processes[0]
    if getattr(process, "execution_mode", None) != "eager":
        raise MatrixError(f"artifact is not eager: {artifact}")
    result: dict[str, int] = {}
    for field in _TOPOLOGY_FIELDS:
        value = getattr(process, field, None)
        if value is None and field in _OPTIONAL_TOPOLOGY_FIELDS:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise MatrixError(f"eager inspection has invalid {field}: {artifact}")
        result[field] = value
    return result


def _topology_gate(records: Sequence[Mapping[str, Any]], *, require_ufo: bool) -> bool:
    if not require_ufo:
        return True
    grouped: dict[tuple[str, str], dict[str, Mapping[str, int]]] = {}
    for record in records:
        case = record.get("case")
        topology = record.get("eager_topology")
        if not isinstance(case, Mapping) or not isinstance(topology, Mapping):
            return False
        key = (str(case.get("key")), str(record.get("color")))
        grouped.setdefault(key, {})[str(record.get("model"))] = topology
    return bool(grouped) and all(
        set(models) == {"built-in", "ufo-sm"}
        and dict(models["built-in"]) == dict(models["ufo-sm"])
        for models in grouped.values()
    )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _generation_phases(artifact: Path) -> dict[str, float]:
    try:
        outer = json.loads((artifact / "artifact.json").read_text(encoding="utf-8"))
        raw = outer["extensions"]["generation"]["phase_timings_seconds"]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise MatrixError(
            f"artifact has no valid generation phases: {artifact}"
        ) from error
    if not isinstance(raw, dict):
        raise MatrixError(f"artifact generation phases are not an object: {artifact}")
    phases: dict[str, float] = {}
    for name, value in raw.items():
        if not isinstance(name, str) or isinstance(value, bool):
            raise MatrixError(f"artifact has invalid generation phase: {artifact}")
        seconds = float(value)
        if not math.isfinite(seconds) or seconds < 0:
            raise MatrixError(f"artifact has invalid generation phase time: {artifact}")
        phases[name] = seconds
    return phases


def _core_generation_seconds(phases: Mapping[str, float]) -> float:
    excluded = {"model-loading", "process-expansion"}
    result = sum(value for name, value in phases.items() if name not in excluded)
    if result <= 0 or not math.isfinite(result):
        raise MatrixError("core generation phase total must be positive and finite")
    return result


def _artifact_size_bytes(artifact: Path) -> int:
    return sum(path.stat().st_size for path in artifact.rglob("*") if path.is_file())


def _watchdog_peak_gib(stderr: str) -> float | None:
    matches = re.findall(r"peak_rss=([0-9]+(?:\.[0-9]+)?) GiB", stderr)
    return float(matches[-1]) if matches else None


def _git_head() -> str | None:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _model_specs(
    arguments: argparse.Namespace,
) -> tuple[tuple[str, str | Path, Path], ...]:
    requested = set(arguments.models)
    specs: list[tuple[str, str | Path, Path]] = []
    if "built-in" in requested:
        specs.append(("built-in", "built-in-sm", arguments.builtin_pack.resolve()))
    if "ufo-sm" in requested:
        if arguments.ufo_source is None or arguments.ufo_pack is None:
            raise MatrixError("ufo-sm requires both --ufo-source and --ufo-pack")
        specs.append(
            ("ufo-sm", arguments.ufo_source.resolve(), arguments.ufo_pack.resolve())
        )
    return tuple(specs)


def run_matrix(arguments: argparse.Namespace) -> dict[str, Any]:
    cases = _selected_cases(arguments.suite, arguments.case)
    colors = tuple(arguments.colors)
    unknown_colors = sorted(set(colors) - set(DEFAULT_COLORS))
    if unknown_colors:
        raise MatrixError(f"unknown color accuracy: {', '.join(unknown_colors)}")
    models = _model_specs(arguments)
    for _, _, pack in models:
        if not pack.is_file():
            raise MatrixError(f"prepared model bundle does not exist: {pack}")

    output_root = arguments.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault("SYMBOLICA_HIDE_BANNER", "1")
    environment.setdefault("PYTHONFAULTHANDLER", "1")
    environment.setdefault("RUST_BACKTRACE", "full")
    started = time.monotonic()
    records: list[dict[str, Any]] = []

    for case in cases:
        for model_name, compiled_source, prepared_pack in models:
            for color in colors:
                cell_root = output_root / case.key / model_name / color
                artifacts = {
                    "compiled": cell_root / "compiled",
                    "eager": cell_root / "eager",
                }
                generation: dict[str, dict[str, Any]] = {}
                for mode, model_source in (
                    ("compiled", compiled_source),
                    ("eager", prepared_pack),
                ):
                    command = _generation_command(
                        arguments.python,
                        process=case.process,
                        artifact=artifacts[mode],
                        model=model_source,
                        color=color,
                        execution_mode=mode,
                    )
                    payload, elapsed, stderr = _run_json(
                        command,
                        timeout=arguments.generation_timeout,
                        memory_limit_gib=arguments.memory_limit_gib,
                        environment=environment,
                    )
                    generation[mode] = {
                        "elapsed_seconds": elapsed,
                        "output": payload.get("output"),
                        "schema_version": payload.get("schema_version"),
                        "peak_rss_gib": _watchdog_peak_gib(stderr),
                        "watchdog": stderr.strip().splitlines()[-1]
                        if stderr.strip()
                        else None,
                    }
                process_id, eager_physics = _artifact_process(artifacts["eager"])
                compiled_process_id, _ = _artifact_process(artifacts["compiled"])
                if compiled_process_id != process_id:
                    raise MatrixError(
                        "compiled/eager process IDs differ: "
                        f"{compiled_process_id!r} != {process_id!r}"
                    )
                for mode in ("compiled", "eager"):
                    phases = _generation_phases(artifacts[mode])
                    generation[mode]["phase_timings_seconds"] = phases
                    generation[mode]["core_phase_seconds"] = _core_generation_seconds(
                        phases
                    )
                    generation[mode]["artifact_size_bytes"] = _artifact_size_bytes(
                        artifacts[mode]
                    )
                eager_topology = _eager_topology(artifacts["eager"])

                workload_results: list[dict[str, Any]] = []
                for workload in _workloads(color, eager_physics):
                    selectors = workload["selectors"]
                    assert isinstance(selectors, dict)
                    specialized_artifact: Path | None = None
                    specialized_generation: dict[str, Any] | None = None
                    specialization = workload.get("compiled_specialization")
                    if isinstance(specialization, Mapping):
                        specialized_artifact = (
                            cell_root
                            / "compiled-specialized"
                            / str(workload["name"])
                        )
                        payload, elapsed, stderr = _run_json(
                            _generation_command(
                                arguments.python,
                                process=case.process,
                                artifact=specialized_artifact,
                                model=compiled_source,
                                color=color,
                                execution_mode="compiled",
                                process_overrides=specialization,
                            ),
                            timeout=arguments.generation_timeout,
                            memory_limit_gib=arguments.memory_limit_gib,
                            environment=environment,
                        )
                        specialized_process_id, specialized_physics = (
                            _artifact_process(specialized_artifact)
                        )
                        if specialized_process_id != process_id:
                            raise MatrixError(
                                "specialized/complete process IDs differ: "
                                f"{specialized_process_id!r} != {process_id!r}"
                            )
                        coverage = specialized_physics.get("coverage")
                        if not isinstance(coverage, Mapping):
                            raise MatrixError(
                                "specialized compiled artifact has no coverage metadata"
                            )
                        expected_selected_axis = (
                            "color"
                            if "reference_color_order" in specialization
                            else "helicities"
                        )
                        if coverage.get(expected_selected_axis) != "selected":
                            raise MatrixError(
                                "specialized compiled artifact did not select "
                                f"{expected_selected_axis} coverage"
                            )
                        phases = _generation_phases(specialized_artifact)
                        specialized_generation = {
                            "elapsed_seconds": elapsed,
                            "output": payload.get("output"),
                            "schema_version": payload.get("schema_version"),
                            "peak_rss_gib": _watchdog_peak_gib(stderr),
                            "phase_timings_seconds": phases,
                            "core_phase_seconds": _core_generation_seconds(phases),
                            "artifact_size_bytes": _artifact_size_bytes(
                                specialized_artifact
                            ),
                            "coverage": dict(coverage),
                        }
                    correctness, _, _ = _run_json(
                        _comparison_command(
                            arguments.python,
                            eager_artifact=artifacts["eager"],
                            compiled_artifact=artifacts["compiled"],
                            process_id=process_id,
                            selectors=selectors,
                        ),
                        timeout=arguments.generation_timeout,
                        memory_limit_gib=arguments.memory_limit_gib,
                        environment=environment,
                    )
                    if specialized_artifact is not None:
                        specialized_correctness, _, _ = _run_json(
                            _comparison_command(
                                arguments.python,
                                eager_artifact=artifacts["eager"],
                                compiled_artifact=specialized_artifact,
                                process_id=process_id,
                                selectors=selectors,
                                compiled_selectors="none",
                            ),
                            timeout=arguments.generation_timeout,
                            memory_limit_gib=arguments.memory_limit_gib,
                            environment=environment,
                        )
                        correctness["specialized_compiled"] = (
                            specialized_correctness
                        )
                        correctness["passes"] = bool(correctness["passes"]) and bool(
                            specialized_correctness["passes"]
                        )
                    profiles: list[dict[str, Any]] = []
                    for batch_size in arguments.batch_sizes:
                        timings: dict[str, dict[str, Any]] = {}
                        profile_targets: list[
                            tuple[str, Path, Mapping[str, str]]
                        ] = [
                            ("compiled_complete", artifacts["compiled"], selectors),
                            ("eager_complete", artifacts["eager"], selectors),
                        ]
                        if specialized_artifact is not None:
                            profile_targets.append(
                                ("compiled_specialized", specialized_artifact, {})
                            )
                        for (
                            mode,
                            profile_artifact,
                            profile_selectors,
                        ) in profile_targets:
                            payload, elapsed, _ = _run_json(
                                _profile_command(
                                    arguments.python,
                                    artifact=profile_artifact,
                                    process_id=process_id,
                                    batch_size=batch_size,
                                    target_runtime=arguments.target_runtime,
                                    minimum_samples=arguments.minimum_samples,
                                    selectors=profile_selectors,
                                ),
                                timeout=max(60.0, arguments.target_runtime * 10.0),
                                memory_limit_gib=arguments.memory_limit_gib,
                                environment=environment,
                            )
                            timings[mode] = {
                                "command_elapsed_seconds": elapsed,
                                "wall_seconds_per_point": payload.get(
                                    "wall_time_per_point"
                                ),
                                "evaluator_seconds_per_point": payload.get(
                                    "evaluator_time_per_point"
                                ),
                                "result": payload,
                            }
                        compiled_complete_wall = float(
                            timings["compiled_complete"]["wall_seconds_per_point"]
                        )
                        eager_wall = float(
                            timings["eager_complete"]["wall_seconds_per_point"]
                        )
                        baseline_name = (
                            "compiled_specialized"
                            if "compiled_specialized" in timings
                            else "compiled_complete"
                        )
                        baseline_wall = float(
                            timings[baseline_name]["wall_seconds_per_point"]
                        )
                        profiles.append(
                            {
                                "batch_size": batch_size,
                                **timings,
                                "runtime_baseline": baseline_name,
                                "eager_over_compiled_complete_wall": (
                                    eager_wall / compiled_complete_wall
                                ),
                                "eager_over_compiled_specialized_wall": (
                                    eager_wall / baseline_wall
                                    if baseline_name == "compiled_specialized"
                                    else None
                                ),
                                "compiled_reusable_over_specialized_wall": (
                                    compiled_complete_wall / baseline_wall
                                    if baseline_name == "compiled_specialized"
                                    else None
                                ),
                                "batch_1024_soft_target_passes": (
                                    batch_size != 1024
                                    or eager_wall
                                    <= SOFT_RUNTIME_RATIO * baseline_wall
                                ),
                            }
                        )
                    workload_results.append(
                        {
                            "name": workload["name"],
                            "selectors": selectors,
                            "compiled_specialization": specialization,
                            "compiled_specialized_generation": (
                                specialized_generation
                            ),
                            "correctness": correctness,
                            "profiles": profiles,
                        }
                    )
                selector_pattern_profiles: dict[str, Any] | None = None
                if color == "lc" and 1024 in arguments.batch_sizes:
                    selector_pattern_profiles = {}
                    for mode in ("compiled", "eager"):
                        payload, profile_elapsed, _ = _run_json(
                            _selector_profile_command(
                                arguments.python,
                                artifact=artifacts[mode],
                                process_id=process_id,
                                batch_size=1024,
                                target_runtime=arguments.selector_target_runtime,
                                minimum_samples=arguments.minimum_samples,
                                seed=arguments.selector_seed,
                            ),
                            timeout=max(
                                120.0,
                                arguments.selector_target_runtime * 40.0,
                            ),
                            memory_limit_gib=arguments.memory_limit_gib,
                            environment=environment,
                        )
                        selector_pattern_profiles[mode] = {
                            "command_elapsed_seconds": profile_elapsed,
                            "result": payload,
                        }
                compiled_generation = generation["compiled"]["elapsed_seconds"]
                eager_generation = generation["eager"]["elapsed_seconds"]
                compiled_core = generation["compiled"]["core_phase_seconds"]
                eager_core = generation["eager"]["core_phase_seconds"]
                records.append(
                    {
                        "case": asdict(case),
                        "model": model_name,
                        "color": color,
                        "process_id": process_id,
                        "eager_topology": eager_topology,
                        "generation": generation,
                        "compiled_over_eager_command_elapsed": (
                            compiled_generation / eager_generation
                        ),
                        "compiled_over_eager_core_generation": (
                            compiled_core / eager_core
                        ),
                        "compiled_generation_under_hard_limit": (
                            compiled_generation <= arguments.generation_timeout
                        ),
                        "workloads": workload_results,
                        "selector_pattern_profiles": selector_pattern_profiles,
                    }
                )
                partial = {
                    "kind": "pyamplicol-eager-benchmark-matrix",
                    "schema_version": 3,
                    "complete": False,
                    "records": records,
                }
                _write_json_atomic(output_root / "result.json", partial)

    elapsed = time.monotonic() - started
    generation_gates, generation_soft_targets = _generation_assessment(records)
    runtime_soft_target = 1024 in arguments.batch_sizes and all(
        profile["batch_1024_soft_target_passes"]
        for record in records
        for workload in record["workloads"]
        for profile in workload["profiles"]
    )
    reusable_runtime_gate = _reusable_runtime_gate(records)
    gates = {
        "matrix_scope_complete": _scope_gate(arguments, cases),
        "correctness": all(
            workload["correctness"]["passes"]
            for record in records
            for workload in record["workloads"]
        ),
        "per_point_selector_patterns": all(
            record["selector_pattern_profiles"] is None
            or all(
                bool(mode_profile["result"]["passes"])
                for mode_profile in record["selector_pattern_profiles"].values()
            )
            for record in records
        ),
        "builtin_ufo_topology_parity": _topology_gate(
            records, require_ufo=arguments.suite == "milestone"
        ),
        "smoke_under_five_minutes": (
            arguments.suite != "smoke" or elapsed <= DEFAULT_SUITE_TIMEOUT
        ),
        "n4_n5_reusable_compiled_at_most_1_25x_specialized": (
            reusable_runtime_gate
        ),
        **generation_gates,
    }
    result = {
        "kind": "pyamplicol-eager-benchmark-matrix",
        "schema_version": 3,
        "complete": True,
        "source_revision": _git_head(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "suite": arguments.suite,
        "elapsed_seconds": elapsed,
        "configuration": {
            "batch_sizes": list(arguments.batch_sizes),
            "colors": list(colors),
            "generation_timeout": arguments.generation_timeout,
            "memory_limit_gib": arguments.memory_limit_gib,
            "minimum_samples": arguments.minimum_samples,
            "target_runtime": arguments.target_runtime,
            "selector_target_runtime": arguments.selector_target_runtime,
            "selector_seed": arguments.selector_seed,
        },
        "gates": gates,
        "soft_targets": {
            **generation_soft_targets,
            "batch_1024_runtime_at_most_1_5x_specialized_compiled": (
                runtime_soft_target
            ),
        },
        "passes": all(gates.values()),
        "records": records,
    }
    _write_json_atomic(output_root / "result.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--suite", choices=("smoke", "milestone"), default="smoke")
    result.add_argument("--case", action="append", default=[], help="case key; repeat")
    result.add_argument(
        "--models",
        type=_csv_values,
        default=("built-in",),
        help="comma-separated built-in,ufo-sm",
    )
    result.add_argument(
        "--colors", type=_csv_values, default=DEFAULT_COLORS, help="lc,nlc,full"
    )
    result.add_argument(
        "--batch-sizes",
        type=_csv_ints,
        default=None,
        help=(
            "comma-separated batch sizes; defaults to 128,1024 for smoke and "
            "1,128,1024 for milestone"
        ),
    )
    result.add_argument("--builtin-pack", type=Path, required=True)
    result.add_argument("--ufo-source", type=Path)
    result.add_argument("--ufo-pack", type=Path)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--python", type=Path, default=Path(sys.executable))
    result.add_argument(
        "--generation-timeout", type=_positive_float, default=DEFAULT_GENERATION_TIMEOUT
    )
    result.add_argument("--target-runtime", type=_positive_float, default=5.0)
    result.add_argument(
        "--selector-target-runtime",
        type=_positive_float,
        default=1.0,
        help="target time for each per-point selector-pattern profile",
    )
    result.add_argument("--selector-seed", type=int, default=0xC0FFEE)
    result.add_argument("--minimum-samples", type=_positive_int, default=5)
    result.add_argument(
        "--memory-limit-gib", type=_positive_float, default=DEFAULT_MEMORY_LIMIT_GIB
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    if arguments.batch_sizes is None:
        arguments.batch_sizes = (
            DEFAULT_BATCH_SIZES
            if arguments.suite == "milestone"
            else SMOKE_BATCH_SIZES
        )
    try:
        result = run_matrix(arguments)
    except (MatrixError, OSError, ValueError) as error:
        print(f"eager-benchmark-matrix: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

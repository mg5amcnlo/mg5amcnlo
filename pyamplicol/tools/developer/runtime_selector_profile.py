#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Profile stable per-point selector grouping on one generated artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

_PATTERNS = ("homogeneous", "pre-pooled", "alternating", "seeded-random")
_MAX_SAMPLE_COUNT = 80
_TARGET_SAMPLE_SECONDS = 0.25


class SelectorProfileError(RuntimeError):
    """Raised when selector profiling cannot establish a valid contract."""


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


def selector_pattern(
    name: str,
    selector_ids: Sequence[str],
    point_count: int,
    *,
    seed: int,
) -> tuple[str, ...]:
    """Build one deterministic selector sequence for the native planner."""

    selectors = tuple(selector_ids)
    if len(selectors) < 2:
        raise SelectorProfileError("selector-pattern profiling needs two selectors")
    if point_count < len(selectors):
        raise SelectorProfileError(
            "selector-pattern batch must contain at least one point per selector"
        )
    if name == "homogeneous":
        return (selectors[0],) * point_count
    if name == "pre-pooled":
        quotient, remainder = divmod(point_count, len(selectors))
        return tuple(
            selector
            for index, selector in enumerate(selectors)
            for _ in range(quotient + (index < remainder))
        )
    if name == "alternating":
        return tuple(selectors[index % len(selectors)] for index in range(point_count))
    if name == "seeded-random":
        generator = random.Random(seed)
        result = [generator.choice(selectors) for _ in range(point_count)]
        for index, selector in enumerate(selectors):
            result[index] = selector
        return tuple(result)
    raise SelectorProfileError(f"unknown selector pattern {name!r}")


def _selector_ids(
    runtime: Any,
    axis: Literal["color", "helicity"],
    count: int,
) -> tuple[str, ...]:
    physics = runtime.physics
    if axis == "color":
        candidates = tuple(flow.id for flow in physics.color_flows)
    else:
        candidates = tuple(
            helicity.id
            for helicity in physics.helicities
            if not helicity.structural_zero
        )
    selected = candidates[:count]
    if len(selected) < 2:
        raise SelectorProfileError(
            f"artifact exposes fewer than two usable {axis} selectors"
        )
    return selected


def _selector_kwargs(
    axis: Literal["color", "helicity"], selectors: Sequence[str]
) -> dict[str, tuple[str, ...]]:
    key = "color_flow_by_point" if axis == "color" else "helicity_by_point"
    return {key: tuple(selectors)}


def _global_selector_kwargs(
    axis: Literal["color", "helicity"], selector: str
) -> dict[str, tuple[str, ...]]:
    key = "color_flows" if axis == "color" else "helicities"
    return {key: (selector,)}


def _assert_scatter_correct(
    runtime: Any,
    batch: Sequence[Sequence[Sequence[float]]],
    axis: Literal["color", "helicity"],
    selectors: Sequence[str],
) -> dict[str, object]:
    actual = runtime.evaluate(batch, **_selector_kwargs(axis, selectors))
    expected_by_selector = {
        selector: runtime.evaluate(batch, **_global_selector_kwargs(axis, selector))
        for selector in dict.fromkeys(selectors)
    }
    maximum_absolute = 0.0
    maximum_relative = 0.0
    passes = True
    for point_index, (selector, actual_value) in enumerate(
        zip(selectors, actual, strict=True)
    ):
        expected_value = expected_by_selector[selector][point_index]
        absolute = abs(complex(actual_value) - complex(expected_value))
        relative = absolute / max(
            abs(complex(actual_value)), abs(complex(expected_value)), 1.0e-300
        )
        maximum_absolute = max(maximum_absolute, absolute)
        maximum_relative = max(maximum_relative, relative)
        passes = passes and (absolute <= 1.0e-15 or relative <= 1.0e-12)
    return {
        "maximum_absolute_difference": maximum_absolute,
        "maximum_relative_difference": maximum_relative,
        "passes": passes,
    }


def _mean_statistics(samples: Sequence[float]) -> dict[str, float]:
    mean = statistics.fmean(samples)
    deviation = statistics.stdev(samples) if len(samples) > 1 else 0.0
    error = deviation / math.sqrt(len(samples)) if samples else 0.0
    return {
        "mean_seconds_per_point": mean,
        "standard_deviation_seconds_per_point": deviation,
        "standard_error_seconds_per_point": error,
        "relative_standard_error": error / mean if mean > 0 else 0.0,
    }


def _profile_pattern(
    runtime: Any,
    batch: Sequence[Sequence[Sequence[float]]],
    axis: Literal["color", "helicity"],
    pattern: str,
    selectors: tuple[str, ...],
    *,
    target_runtime: float,
    minimum_samples: int,
    warmup_runs: int,
    seed: int,
) -> dict[str, object]:
    sequence = selector_pattern(pattern, selectors, len(batch), seed=seed)
    selector_kwargs = _selector_kwargs(axis, sequence)
    correctness = _assert_scatter_correct(runtime, batch, axis, sequence)
    for _ in range(warmup_runs):
        runtime.evaluate(batch, **selector_kwargs)

    probe = runtime._backend.profile_repeated(batch, 1, **selector_kwargs)
    probe_seconds = float(probe["wall_time_s"])
    if not math.isfinite(probe_seconds) or probe_seconds <= 0:
        raise SelectorProfileError("native selector profile returned no wall time")
    target_sample = min(
        _TARGET_SAMPLE_SECONDS,
        target_runtime / max(minimum_samples, 1),
    )
    repetitions = max(1, math.ceil(target_sample / probe_seconds))
    sample_count = max(
        minimum_samples,
        math.ceil(target_runtime / (probe_seconds * repetitions)),
    )
    sample_count = min(sample_count, _MAX_SAMPLE_COUNT)

    profiles = tuple(
        runtime._backend.profile_repeated(
            batch,
            repetitions,
            **selector_kwargs,
        )
        for _ in range(sample_count)
    )
    measured_points = len(batch) * repetitions
    wall_samples = tuple(
        float(profile["wall_time_s"]) / measured_points for profile in profiles
    )
    first = profiles[0]
    stable_fields = (
        "selector_plan_kind",
        "selector_group_sizes",
        "selector_reordered_point_count",
        "selector_simd_lane_width",
        "selector_simd_occupancy",
    )
    for profile in profiles[1:]:
        if any(profile[field] != first[field] for field in stable_fields):
            raise SelectorProfileError("native selector plan changed between samples")
    expected_kind = {
        "homogeneous": "homogeneous",
        "pre-pooled": "contiguous",
        "alternating": "stable-grouped",
        "seeded-random": "stable-grouped",
    }[pattern]
    if first["selector_plan_kind"] != expected_kind:
        raise SelectorProfileError(
            f"{pattern} selector plan used {first['selector_plan_kind']!r}, "
            f"expected {expected_kind!r}"
        )

    component_fields = (
        "selector_planner_time_s",
        "selector_gather_time_s",
        "selector_scatter_time_s",
    )
    timing = {
        field.removesuffix("_time_s") + "_seconds_per_point": statistics.fmean(
            float(profile[field]) / measured_points for profile in profiles
        )
        for field in component_fields
    }
    population = Counter(sequence)
    digest = hashlib.sha256("\0".join(sequence).encode("utf-8")).hexdigest()
    return {
        "pattern": pattern,
        "selector_axis": axis,
        "selector_ids": list(selectors),
        "selector_population": dict(population),
        "selector_sequence_sha256": digest,
        "seed": seed if pattern == "seeded-random" else None,
        "batch_size": len(batch),
        "warmup_runs": warmup_runs,
        "sample_count": sample_count,
        "repetitions_per_sample": repetitions,
        "measured_point_count": measured_points * sample_count,
        "selector_plan_kind": first["selector_plan_kind"],
        "selector_group_sizes": list(first["selector_group_sizes"]),
        "selector_reordered_point_count": int(
            first["selector_reordered_point_count"]
        ),
        "selector_simd_lane_width": int(first["selector_simd_lane_width"]),
        "selector_simd_occupancy": float(first["selector_simd_occupancy"]),
        "correctness": correctness,
        "wall": _mean_statistics(wall_samples),
        **timing,
    }


def profile_artifact(arguments: argparse.Namespace) -> dict[str, object]:
    from pyamplicol import Runtime

    runtime = Runtime.load(arguments.artifact, process=arguments.process)
    validation = runtime._backend.validation_momenta()
    if validation is None or len(validation) == 0:
        raise SelectorProfileError("artifact has no deterministic validation point")
    source = tuple(validation)
    batch = tuple(source[index % len(source)] for index in range(arguments.batch_size))
    axes = ("color", "helicity") if arguments.axis == "both" else (arguments.axis,)
    results: list[dict[str, object]] = []
    for axis in axes:
        selector_ids = _selector_ids(runtime, axis, arguments.selector_count)
        for pattern in _PATTERNS:
            results.append(
                _profile_pattern(
                    runtime,
                    batch,
                    axis,
                    pattern,
                    selector_ids,
                    target_runtime=arguments.target_runtime,
                    minimum_samples=arguments.minimum_samples,
                    warmup_runs=arguments.warmup_runs,
                    seed=arguments.seed,
                )
            )
    return {
        "kind": "pyamplicol-runtime-selector-pattern-profile",
        "schema_version": 1,
        "artifact": str(arguments.artifact.resolve()),
        "process": runtime.physics.process,
        "process_id": runtime.physics.process_id,
        "execution_mode": str(getattr(runtime._backend, "execution_mode", "unknown")),
        "complete": True,
        "passes": all(bool(result["correctness"]["passes"]) for result in results),
        "profiles": results,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("artifact", type=Path)
    result.add_argument("--process")
    result.add_argument("--axis", choices=("color", "helicity", "both"), default="both")
    result.add_argument("--batch-size", type=_positive_int, default=1024)
    result.add_argument("--selector-count", type=_positive_int, default=4)
    result.add_argument("--seed", type=int, default=0xC0FFEE)
    result.add_argument("--target-runtime", type=_positive_float, default=1.0)
    result.add_argument("--minimum-samples", type=_positive_int, default=5)
    result.add_argument("--warmup-runs", type=int, default=2)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    if arguments.warmup_runs < 0:
        parser().error("--warmup-runs must be non-negative")
    try:
        result = profile_artifact(arguments)
    except (OSError, SelectorProfileError, ValueError) as error:
        print(f"runtime-selector-profile: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

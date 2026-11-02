# SPDX-License-Identifier: 0BSD
# ruff: noqa: E501
from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import shlex
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from pyamplicol import EvaluationError, Generator, ProcessSet, Runtime
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    JITConfig,
    RunConfig,
)
from pyamplicol.generation.phase_space import massive_rambo_final_state

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_POINT_COUNT = 8
_RANDOM_SELECTOR_SEED = 0xC0FFEE


@dataclass(frozen=True, slots=True)
class _NativeSdk:
    cc: str
    cxx: str
    fortran: str
    rustc: str
    config: dict[str, Any]


def _source_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (str((_PROJECT_ROOT / "src").resolve()), environment.get("PYTHONPATH", ""))
    ).rstrip(os.pathsep)
    return environment


def _unavailable(reason: str) -> None:
    if os.environ.get("PYAMPLICOL_REQUIRE_NATIVE_TESTS") == "1":
        pytest.fail(reason)
    pytest.skip(reason)


def _configured_tool(variable: str, default: str) -> str | None:
    command = os.environ.get(variable, default).split()[0]
    return shutil.which(command) or (command if Path(command).is_file() else None)


def _rusticol_config_command() -> tuple[str, ...] | None:
    configured = os.environ.get("RUSTICOL_CONFIG")
    if configured:
        return tuple(shlex.split(configured))
    if importlib.util.find_spec("rusticol_config") is not None:
        return (sys.executable, "-m", "rusticol_config")
    sibling = Path(sys.executable).parent / "rusticol-config"
    if sibling.is_file():
        return (str(sibling),)
    return None


@pytest.fixture(scope="module")
def native_sdk() -> _NativeSdk:
    tools = {
        "cc": _configured_tool("CC", "cc"),
        "cxx": _configured_tool("CXX", "c++"),
        "fortran": _configured_tool("FC", "gfortran"),
        "rustc": _configured_tool("RUSTC", "rustc"),
    }
    missing = tuple(name for name, path in tools.items() if path is None)
    if missing:
        _unavailable("runtime selector parity requires " + ", ".join(missing))
    command = _rusticol_config_command()
    if command is None:
        _unavailable("runtime selector parity requires rusticol-config")
    completed = subprocess.run(
        [*command, "--json"],
        env=_source_environment(),
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if completed.returncode != 0:
        _unavailable("Rusticol SDK configuration failed: " + completed.stderr.strip())
    return _NativeSdk(
        cc=tools["cc"],  # type: ignore[arg-type]
        cxx=tools["cxx"],  # type: ignore[arg-type]
        fortran=tools["fortran"],  # type: ignore[arg-type]
        rustc=tools["rustc"],  # type: ignore[arg-type]
        config=json.loads(completed.stdout),
    )


@pytest.fixture(scope="module")
def selector_artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        _unavailable("the Rusticol extension has not been built")
    artifact = tmp_path_factory.mktemp("runtime-selector-parity") / "dd-zgg-lc"
    config = RunConfig(
        action="generate",
        color=ColorConfig(accuracy="lc"),
        generation=GenerationConfig(workers=1),
        evaluator=EvaluatorConfig(
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(optimization_level=1),
        ),
    )
    Generator(config).generate(
        ProcessSet.from_expressions(("d d~ > z g g",), names=("dd_zgg",)), artifact
    )
    return artifact


@pytest.fixture(scope="module")
def contracted_selector_artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        _unavailable("the Rusticol extension has not been built")
    artifact = tmp_path_factory.mktemp("runtime-selector-contracted") / "dd-zgg-nlc"
    config = RunConfig(
        action="generate",
        color=ColorConfig(accuracy="nlc"),
        generation=GenerationConfig(workers=1),
        evaluator=EvaluatorConfig(
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(optimization_level=1),
        ),
    )
    Generator(config).generate(
        ProcessSet.from_expressions(("d d~ > z g g",), names=("dd_zgg",)),
        artifact,
    )
    return artifact


def _validation_point(artifact: Path) -> tuple[tuple[float, ...], ...]:
    payload = json.loads(
        (artifact / "processes/dd_zgg/validation-momenta.json").read_text(
            encoding="utf-8"
        )
    )
    return tuple(
        tuple(float(component) for component in leg["momentum"])
        for leg in payload["points"][0]
    )


def _phase_space_points(
    artifact: Path,
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    validation = _validation_point(artifact)
    sqrt_s = validation[0][0] + validation[1][0]
    z_momentum = validation[2]
    z_mass = math.sqrt(
        max(
            0.0,
            z_momentum[0] ** 2 - sum(component**2 for component in z_momentum[1:]),
        )
    )
    points = tuple(
        (
            validation[0],
            validation[1],
            *massive_rambo_final_state(
                3,
                sqrt_s=sqrt_s,
                masses=(z_mass, 0.0, 0.0),
                seed=7000 + index,
            ),
        )
        for index in range(_POINT_COUNT)
    )
    assert len(set(points)) == _POINT_COUNT
    return points


def _selector_patterns() -> dict[str, tuple[tuple[int, ...], tuple[int, ...]]]:
    random_generator = random.Random(_RANDOM_SELECTOR_SEED)
    seeded_random = (
        tuple(random_generator.randrange(2) for _ in range(_POINT_COUNT)),
        tuple(random_generator.randrange(2) for _ in range(_POINT_COUNT)),
    )
    return {
        "homogeneous": ((0,) * _POINT_COUNT, (0,) * _POINT_COUNT),
        "pooled": ((0,) * 4 + (1,) * 4, (1,) * 4 + (0,) * 4),
        "alternating": ((0, 1) * 4, (1, 0) * 4),
        "seeded-random": seeded_random,
    }


def _assert_close(actual: complex, expected: complex) -> None:
    assert actual.real == pytest.approx(expected.real, rel=1.0e-12, abs=1.0e-15)
    assert actual.imag == pytest.approx(expected.imag, rel=1.0e-12, abs=1.0e-15)


def test_python_global_and_per_point_selector_contract(selector_artifact: Path) -> None:
    runtime = Runtime.load(selector_artifact)
    momenta = _phase_space_points(selector_artifact)
    resolved = runtime.evaluate_resolved(momenta)
    assert len(resolved.helicity_ids) >= 2
    assert len(resolved.color_ids) >= 2

    patterns = _selector_patterns()
    helicities, colors = patterns["alternating"]
    default_values = runtime.evaluate(momenta)
    global_values = runtime.evaluate(
        momenta,
        helicities=(resolved.helicity_ids[0],),
        color_flows=(resolved.color_ids[0],),
    )
    global_helicity_only = runtime.evaluate(
        momenta,
        helicities=(resolved.helicity_ids[0],),
    )
    global_color_only = runtime.evaluate(
        momenta,
        color_flows=(resolved.color_ids[0],),
    )
    selected_by_pattern = {
        name: runtime.evaluate(
            momenta,
            helicity_by_point=tuple(
                resolved.helicity_ids[index] for index in pattern_helicities
            ),
            color_flow_by_point=tuple(
                resolved.color_ids[index] for index in pattern_colors
            ),
        )
        for name, (pattern_helicities, pattern_colors) in patterns.items()
    }
    helicity_only = runtime.evaluate(
        momenta,
        helicity_by_point=tuple(resolved.helicity_ids[index] for index in helicities),
    )
    color_only = runtime.evaluate(
        momenta,
        color_flow_by_point=tuple(resolved.color_ids[index] for index in colors),
    )

    for point in range(_POINT_COUNT):
        _assert_close(
            default_values[point],
            sum(sum(row) for row in resolved.values[point]),
        )
        _assert_close(global_values[point], resolved.values[point][0][0])
        _assert_close(global_helicity_only[point], sum(resolved.values[point][0]))
        _assert_close(
            global_color_only[point], sum(row[0] for row in resolved.values[point])
        )
        for name, (pattern_helicities, pattern_colors) in patterns.items():
            _assert_close(
                selected_by_pattern[name][point],
                resolved.values[point][pattern_helicities[point]][
                    pattern_colors[point]
                ],
            )
        _assert_close(
            helicity_only[point], sum(resolved.values[point][helicities[point]])
        )
        _assert_close(
            color_only[point],
            sum(row[colors[point]] for row in resolved.values[point]),
        )

    with pytest.raises(ValueError, match="helicities and helicity_by_point"):
        runtime.evaluate(
            momenta,
            helicities=(resolved.helicity_ids[0],),
            helicity_by_point=(resolved.helicity_ids[0],) * _POINT_COUNT,
        )
    with pytest.raises(ValueError, match="color_flows and color_flow_by_point"):
        runtime.evaluate(
            momenta,
            color_flows=(resolved.color_ids[0],),
            color_flow_by_point=(resolved.color_ids[0],) * _POINT_COUNT,
        )
    with pytest.raises(
        EvaluationError, match="expected one selector for each of 8 points"
    ):
        runtime.evaluate(
            momenta,
            helicity_by_point=(resolved.helicity_ids[0],),
        )
    with pytest.raises(EvaluationError, match=r"color_flow_by_point\[1\]"):
        runtime.evaluate(
            momenta,
            color_flow_by_point=(
                resolved.color_ids[0],
                "missing",
                "missing",
                "missing",
                "missing",
                "missing",
                "missing",
                "missing",
            ),
        )


def test_native_profile_reports_stable_grouping_and_pooled_bypass(
    selector_artifact: Path,
) -> None:
    runtime = Runtime.load(selector_artifact)
    momenta = _phase_space_points(selector_artifact)
    resolved = runtime.evaluate_resolved(momenta)
    patterns = _selector_patterns()

    pooled_helicities, pooled_colors = patterns["pooled"]
    pooled = runtime._backend.profile_repeated(
        momenta,
        2,
        helicity_by_point=tuple(
            resolved.helicity_ids[index] for index in pooled_helicities
        ),
        color_flow_by_point=tuple(
            resolved.color_ids[index] for index in pooled_colors
        ),
    )
    assert pooled["selector_plan_kind"] == "contiguous"
    assert pooled["selector_group_sizes"] == [4, 4]
    assert pooled["selector_reordered_point_count"] == 0
    assert pooled["selector_gather_time_s"] == 0.0

    alternating_helicities, alternating_colors = patterns["alternating"]
    alternating = runtime._backend.profile_repeated(
        momenta,
        2,
        helicity_by_point=tuple(
            resolved.helicity_ids[index] for index in alternating_helicities
        ),
        color_flow_by_point=tuple(
            resolved.color_ids[index] for index in alternating_colors
        ),
    )
    assert alternating["selector_plan_kind"] == "stable-grouped"
    assert alternating["selector_group_sizes"] == [4, 4]
    assert alternating["selector_reordered_point_count"] == 6
    assert 0.0 < float(alternating["selector_simd_occupancy"]) <= 1.0
    assert float(alternating["selector_planner_time_s"]) > 0.0
    assert float(alternating["selector_gather_time_s"]) > 0.0


def test_python_color_selectors_reject_contracted_axis(
    contracted_selector_artifact: Path,
) -> None:
    runtime = Runtime.load(contracted_selector_artifact)
    momenta = (_validation_point(contracted_selector_artifact),)

    for selectors in (
        {"color_flows": (runtime.physics.color_ids[0],)},
        {"color_flow_by_point": (runtime.physics.color_ids[0],)},
    ):
        with pytest.raises(
            EvaluationError,
            match="LC color-flow selection is unavailable",
        ):
            runtime.evaluate(momenta, **selectors)


_C_PROBE = r"""
#include <rusticol.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(call) do { \
    int status_ = (call); \
    if (status_ != RUSTICOL_STATUS_OK) { \
        size_t required_ = 0; \
        rusticol_last_error_message(NULL, 0, &required_); \
        char *message_ = required_ > 0 ? malloc(required_) : NULL; \
        if (message_ != NULL && rusticol_last_error_message( \
                message_, required_, &required_) == RUSTICOL_STATUS_OK) \
            fprintf(stderr, "Rusticol error: %s\\n", message_); \
        free(message_); \
        return 10; \
    } \
} while (0)

typedef int (*id_getter)(const RusticolRuntimeHandle *, size_t, char *, size_t, size_t *);

static char *get_id(RusticolRuntimeHandle *runtime, id_getter getter, size_t index) {
    size_t required = 0;
    if (getter(runtime, index, NULL, 0, &required) != RUSTICOL_STATUS_OK) return NULL;
    char *value = malloc(required);
    if (value == NULL || getter(runtime, index, value, required, &required) != RUSTICOL_STATUS_OK) {
        free(value);
        return NULL;
    }
    return value;
}

static int close_value(double actual, double expected) {
    const double scale = fmax(fabs(actual), fabs(expected));
    return fabs(actual - expected) <= 1.0e-15 + 1.0e-12 * scale;
}

static double component(const double *values, size_t point, size_t helicity, size_t color,
                        size_t helicity_count, size_t color_count) {
    return values[(point * helicity_count + helicity) * color_count + color];
}

static int check_pattern(RusticolRuntimeHandle *runtime, const double *momenta,
                         size_t momenta_count, size_t point_count,
                         const uint32_t *helicities, const uint32_t *colors,
                         const double *resolved, size_t helicity_count,
                         size_t color_count, int failure_code) {
    double selected[8];
    if (rusticol_runtime_evaluate_selected_f64(
            runtime, momenta, momenta_count, point_count,
            NULL, 0, NULL, 0, helicities, point_count, colors, point_count,
            selected, point_count) != RUSTICOL_STATUS_OK) return 10;
    for (size_t point = 0; point < point_count; ++point) {
        const double expected = component(
            resolved, point, helicities[point], colors[point], helicity_count, color_count);
        if (!close_value(selected[point], expected)) return failure_code;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 4) return 2;
    const size_t point_count = 8;
    const size_t momenta_count = (size_t)(argc - 3);
    if (momenta_count % point_count != 0) return 2;
    double *momenta = malloc(momenta_count * sizeof(double));
    if (momenta == NULL) return 3;
    for (size_t index = 0; index < momenta_count; ++index)
        momenta[index] = strtod(argv[index + 3], NULL);

    RusticolRuntimeHandle *runtime = NULL;
    CHECK(rusticol_runtime_load(argv[1], argv[2], NULL, &runtime));
    size_t helicity_count = 0, color_count = 0;
    CHECK(rusticol_runtime_helicity_count(runtime, &helicity_count));
    CHECK(rusticol_runtime_color_count(runtime, &color_count));
    if (getenv("RUSTICOL_EXPECT_CONTRACTED_COLOR") != NULL) {
        char *contracted_color_id = get_id(runtime, rusticol_runtime_color_id, 0);
        const char *contracted_color_ids[] = {contracted_color_id};
        const uint32_t contracted_colors[] = {0, 0, 0, 0, 0, 0, 0, 0};
        double contracted_output[8];
        if (contracted_color_id == NULL) return 40;
        if (rusticol_runtime_evaluate_selected_f64(
                runtime, momenta, momenta_count, point_count,
                NULL, 0, contracted_color_ids, 1, NULL, 0, NULL, 0,
                contracted_output, point_count) != RUSTICOL_STATUS_INVALID_ARGUMENT) return 41;
        if (rusticol_runtime_evaluate_selected_f64(
                runtime, momenta, momenta_count, point_count,
                NULL, 0, NULL, 0, NULL, 0, contracted_colors, point_count,
                contracted_output, point_count) != RUSTICOL_STATUS_INVALID_ARGUMENT) return 42;
        free(contracted_color_id);
        CHECK(rusticol_runtime_free(runtime));
        free(momenta);
        puts("ok");
        return 0;
    }
    if (helicity_count < 2 || color_count < 2) return 4;
    char *helicity_id = get_id(runtime, rusticol_runtime_helicity_id, 0);
    char *color_id = get_id(runtime, rusticol_runtime_color_id, 0);
    if (helicity_id == NULL || color_id == NULL) return 5;
    const char *helicity_ids[] = {helicity_id};
    const char *color_ids[] = {color_id};
    const uint32_t homogeneous_helicities[] = {0, 0, 0, 0, 0, 0, 0, 0};
    const uint32_t homogeneous_colors[] = {0, 0, 0, 0, 0, 0, 0, 0};
    const uint32_t pooled_helicities[] = {0, 0, 0, 0, 1, 1, 1, 1};
    const uint32_t pooled_colors[] = {1, 1, 1, 1, 0, 0, 0, 0};
    const uint32_t helicities[] = {0, 1, 0, 1, 0, 1, 0, 1};
    const uint32_t colors[] = {1, 0, 1, 0, 1, 0, 1, 0};
    const uint32_t random_helicities[] = {0, 0, 0, 1, 0, 1, 1, 1};
    const uint32_t random_colors[] = {1, 1, 0, 1, 1, 1, 0, 1};
    const uint32_t short_selectors[] = {0};
    const uint32_t invalid_selectors[] = {
        UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX,
        UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX
    };

    double *resolved = malloc(point_count * helicity_count * color_count * sizeof(double));
    double selected[8];
    size_t returned_helicities = 0, returned_colors = 0;
    if (resolved == NULL) return 6;
    CHECK(rusticol_runtime_evaluate_resolved_f64(
        runtime, momenta, momenta_count, point_count,
        NULL, 0, NULL, 0, resolved, point_count * helicity_count * color_count,
        &returned_helicities, &returned_colors));
    if (returned_helicities != helicity_count || returned_colors != color_count) return 7;

    CHECK(rusticol_runtime_evaluate_selected_f64(
        runtime, momenta, momenta_count, point_count,
        NULL, 0, NULL, 0, NULL, 0, NULL, 0, selected, point_count));
    for (size_t point = 0; point < point_count; ++point) {
        double expected = 0.0;
        for (size_t helicity = 0; helicity < helicity_count; ++helicity)
            for (size_t color = 0; color < color_count; ++color)
                expected += component(resolved, point, helicity, color, helicity_count, color_count);
        if (!close_value(selected[point], expected)) return 19;
    }

    CHECK(rusticol_runtime_evaluate_selected_f64(
        runtime, momenta, momenta_count, point_count,
        helicity_ids, 1, color_ids, 1, NULL, 0, NULL, 0, selected, point_count));
    for (size_t point = 0; point < point_count; ++point)
        if (!close_value(selected[point], component(resolved, point, 0, 0, helicity_count, color_count))) return 20;

    CHECK(rusticol_runtime_evaluate_selected_f64(
        runtime, momenta, momenta_count, point_count,
        helicity_ids, 1, NULL, 0, NULL, 0, NULL, 0, selected, point_count));
    for (size_t point = 0; point < point_count; ++point) {
        double expected = 0.0;
        for (size_t color = 0; color < color_count; ++color)
            expected += component(resolved, point, 0, color, helicity_count, color_count);
        if (!close_value(selected[point], expected)) return 25;
    }

    CHECK(rusticol_runtime_evaluate_selected_f64(
        runtime, momenta, momenta_count, point_count,
        NULL, 0, color_ids, 1, NULL, 0, NULL, 0, selected, point_count));
    for (size_t point = 0; point < point_count; ++point) {
        double expected = 0.0;
        for (size_t helicity = 0; helicity < helicity_count; ++helicity)
            expected += component(resolved, point, helicity, 0, helicity_count, color_count);
        if (!close_value(selected[point], expected)) return 26;
    }

    int pattern_status = check_pattern(
        runtime, momenta, momenta_count, point_count,
        homogeneous_helicities, homogeneous_colors, resolved,
        helicity_count, color_count, 21);
    if (pattern_status != 0) return pattern_status;
    pattern_status = check_pattern(
        runtime, momenta, momenta_count, point_count,
        pooled_helicities, pooled_colors, resolved,
        helicity_count, color_count, 22);
    if (pattern_status != 0) return pattern_status;
    pattern_status = check_pattern(
        runtime, momenta, momenta_count, point_count,
        helicities, colors, resolved, helicity_count, color_count, 23);
    if (pattern_status != 0) return pattern_status;
    pattern_status = check_pattern(
        runtime, momenta, momenta_count, point_count,
        random_helicities, random_colors, resolved,
        helicity_count, color_count, 24);
    if (pattern_status != 0) return pattern_status;

    CHECK(rusticol_runtime_evaluate_selected_f64(
        runtime, momenta, momenta_count, point_count,
        NULL, 0, NULL, 0, helicities, point_count, NULL, 0, selected, point_count));
    for (size_t point = 0; point < point_count; ++point) {
        double expected = 0.0;
        for (size_t color = 0; color < color_count; ++color)
            expected += component(resolved, point, helicities[point], color, helicity_count, color_count);
        if (!close_value(selected[point], expected)) return 25;
    }

    CHECK(rusticol_runtime_evaluate_selected_f64(
        runtime, momenta, momenta_count, point_count,
        NULL, 0, NULL, 0, NULL, 0, colors, point_count, selected, point_count));
    for (size_t point = 0; point < point_count; ++point) {
        double expected = 0.0;
        for (size_t helicity = 0; helicity < helicity_count; ++helicity)
            expected += component(resolved, point, helicity, colors[point], helicity_count, color_count);
        if (!close_value(selected[point], expected)) return 26;
    }

    if (rusticol_runtime_evaluate_selected_f64(runtime, momenta, momenta_count,
            point_count, helicity_ids, 1, NULL, 0, homogeneous_helicities, point_count, NULL, 0, selected, point_count)
        != RUSTICOL_STATUS_INVALID_ARGUMENT) return 30;
    if (rusticol_runtime_evaluate_selected_f64(runtime, momenta, momenta_count,
            point_count, NULL, 0, color_ids, 1, NULL, 0, homogeneous_colors, point_count, selected, point_count)
        != RUSTICOL_STATUS_INVALID_ARGUMENT) return 31;
    if (rusticol_runtime_evaluate_selected_f64(runtime, momenta, momenta_count,
            point_count, NULL, 0, NULL, 0, short_selectors, 1, NULL, 0, selected, point_count)
        != RUSTICOL_STATUS_INVALID_ARGUMENT) return 32;
    if (rusticol_runtime_evaluate_selected_f64(runtime, momenta, momenta_count,
            point_count, NULL, 0, NULL, 0, NULL, 0, short_selectors, 1, selected, point_count)
        != RUSTICOL_STATUS_INVALID_ARGUMENT) return 33;
    if (rusticol_runtime_evaluate_selected_f64(runtime, momenta, momenta_count,
            point_count, NULL, 0, NULL, 0, invalid_selectors, point_count, NULL, 0, selected, point_count)
        != RUSTICOL_STATUS_INVALID_ARGUMENT) return 34;
    if (rusticol_runtime_evaluate_selected_f64(runtime, momenta, momenta_count,
            point_count, NULL, 0, NULL, 0, NULL, 0, invalid_selectors, point_count, selected, point_count)
        != RUSTICOL_STATUS_INVALID_ARGUMENT) return 35;

    CHECK(rusticol_runtime_free(runtime));
    free(resolved);
    free(color_id);
    free(helicity_id);
    free(momenta);
    puts("ok");
    return 0;
}
"""


_CPP_PROBE = r"""
#include <rusticol.hpp>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

static bool close_value(double actual, double expected) {
    const double scale = std::max(std::abs(actual), std::abs(expected));
    return std::abs(actual - expected) <= 1.0e-15 + 1.0e-12 * scale;
}

int main(int argc, char **argv) {
    if (argc < 4) return 2;
    constexpr std::size_t point_count = 8;
    std::vector<double> momenta;
    for (int index = 3; index < argc; ++index) momenta.push_back(std::stod(argv[index]));
    if (momenta.size() % point_count != 0) return 2;
    rusticol::Runtime runtime(argv[1], argv[2]);
    const auto helicities_metadata = runtime.helicities();
    const auto colors_metadata = runtime.colors();
    if (std::getenv("RUSTICOL_EXPECT_CONTRACTED_COLOR") != nullptr) {
        const std::vector<std::uint32_t> point_colors(point_count, 0);
        const auto rejects = [&](const std::vector<std::string> &global_colors,
                                 const std::vector<std::uint32_t> &per_point_colors) {
            try {
                (void)runtime.evaluate_selected(
                    momenta, point_count, {}, global_colors, {}, per_point_colors);
            } catch (const rusticol::Error &) {
                return true;
            }
            return false;
        };
        if (!rejects({colors_metadata[0].id}, {})) return 40;
        if (!rejects({}, point_colors)) return 41;
        std::cout << "ok\n";
        return 0;
    }
    if (helicities_metadata.size() < 2 || colors_metadata.size() < 2) return 3;
    const auto resolved = runtime.evaluate_resolved(momenta, point_count);
    const std::vector<std::uint32_t> homogeneous(point_count, 0);
    const std::vector<std::uint32_t> pooled_helicities{0, 0, 0, 0, 1, 1, 1, 1};
    const std::vector<std::uint32_t> pooled_colors{1, 1, 1, 1, 0, 0, 0, 0};
    const std::vector<std::uint32_t> helicities{0, 1, 0, 1, 0, 1, 0, 1};
    const std::vector<std::uint32_t> colors{1, 0, 1, 0, 1, 0, 1, 0};
    const std::vector<std::uint32_t> random_helicities{0, 0, 0, 1, 0, 1, 1, 1};
    const std::vector<std::uint32_t> random_colors{1, 1, 0, 1, 1, 1, 0, 1};

    const auto all = runtime.evaluate_selected(momenta, point_count);

    const auto global = runtime.evaluate_selected(
        momenta, point_count, {helicities_metadata[0].id}, {colors_metadata[0].id});
    const auto global_helicity_only = runtime.evaluate_selected(
        momenta, point_count, {helicities_metadata[0].id}, {});
    const auto global_color_only = runtime.evaluate_selected(
        momenta, point_count, {}, {colors_metadata[0].id});
    const auto homogeneous_values = runtime.evaluate_selected(
        momenta, point_count, {}, {}, homogeneous, homogeneous);
    const auto pooled = runtime.evaluate_selected(
        momenta, point_count, {}, {}, pooled_helicities, pooled_colors);
    const auto alternating = runtime.evaluate_selected(
        momenta, point_count, {}, {}, helicities, colors);
    const auto seeded_random = runtime.evaluate_selected(
        momenta, point_count, {}, {}, random_helicities, random_colors);
    const auto helicity_only = runtime.evaluate_selected(
        momenta, point_count, {}, {}, helicities, {});
    const auto color_only = runtime.evaluate_selected(
        momenta, point_count, {}, {}, {}, colors);
    for (std::size_t point = 0; point < point_count; ++point) {
        double expected_all = 0.0;
        for (std::size_t helicity = 0; helicity < helicities_metadata.size(); ++helicity)
            for (std::size_t color = 0; color < colors_metadata.size(); ++color)
                expected_all += resolved(point, helicity, color);
        if (!close_value(all[point], expected_all)) return 19;
        if (!close_value(global[point], resolved(point, 0, 0))) return 20;
        double expected_global_helicity = 0.0;
        for (std::size_t color = 0; color < colors_metadata.size(); ++color)
            expected_global_helicity += resolved(point, 0, color);
        if (!close_value(global_helicity_only[point], expected_global_helicity)) return 25;
        double expected_global_color = 0.0;
        for (std::size_t helicity = 0; helicity < helicities_metadata.size(); ++helicity)
            expected_global_color += resolved(point, helicity, 0);
        if (!close_value(global_color_only[point], expected_global_color)) return 26;
        if (!close_value(homogeneous_values[point], resolved(point, 0, 0))) return 21;
        if (!close_value(
                pooled[point],
                resolved(point, pooled_helicities[point], pooled_colors[point]))) return 27;
        if (!close_value(alternating[point], resolved(point, helicities[point], colors[point]))) return 22;
        if (!close_value(
                seeded_random[point],
                resolved(point, random_helicities[point], random_colors[point]))) return 28;
        double expected_helicity = 0.0;
        for (std::size_t color = 0; color < colors_metadata.size(); ++color)
            expected_helicity += resolved(point, helicities[point], color);
        if (!close_value(helicity_only[point], expected_helicity)) return 23;
        double expected_color = 0.0;
        for (std::size_t helicity = 0; helicity < helicities_metadata.size(); ++helicity)
            expected_color += resolved(point, helicity, colors[point]);
        if (!close_value(color_only[point], expected_color)) return 24;
    }

    const auto rejects = [&](const std::vector<std::string> &global_helicities,
                             const std::vector<std::string> &global_colors,
                             const std::vector<std::uint32_t> &point_helicities,
                             const std::vector<std::uint32_t> &point_colors) {
        try {
            (void)runtime.evaluate_selected(
                momenta, point_count, global_helicities, global_colors, point_helicities, point_colors);
        } catch (const rusticol::Error &) {
            return true;
        }
        return false;
    };
    if (!rejects({helicities_metadata[0].id}, {}, homogeneous, {})) return 30;
    if (!rejects({}, {colors_metadata[0].id}, {}, homogeneous)) return 31;
    if (!rejects({}, {}, {0}, {})) return 32;
    if (!rejects({}, {}, {}, {0})) return 33;
    const std::vector<std::uint32_t> invalid(point_count, std::numeric_limits<std::uint32_t>::max());
    if (!rejects({}, {}, invalid, {})) return 34;
    if (!rejects({}, {}, {}, invalid)) return 35;
    std::cout << "ok\n";
    return 0;
}
"""


_RUST_PROBE = r"""
#[allow(dead_code)]
mod rusticol {
    include!(env!("RUSTICOL_RUST_SOURCE"));
}

use rusticol::{ErrorKind, Runtime, Selectors};

fn close_value(actual: f64, expected: f64) -> bool {
    (actual - expected).abs() <= 1.0e-15 + 1.0e-12 * actual.abs().max(expected.abs())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = std::env::args().collect::<Vec<_>>();
    if arguments.len() < 4 {
        return Err("missing artifact, process, or momenta".into());
    }
    const POINT_COUNT: usize = 8;
    let momenta = arguments[3..]
        .iter()
        .map(|value| value.parse::<f64>())
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(momenta.len() % POINT_COUNT, 0);
    let mut runtime = Runtime::load(&arguments[1], Some(&arguments[2]), None)?;
    let metadata_helicities = runtime.helicities()?;
    let metadata_colors = runtime.colors()?;
    if std::env::var_os("RUSTICOL_EXPECT_CONTRACTED_COLOR").is_some() {
        let point_colors = [0_u32; POINT_COUNT];
        let global_color = Selectors::all().with_colors([metadata_colors[0].id.clone()]);
        for result in [
            runtime.evaluate_selected_f64(&momenta, POINT_COUNT, &global_color, None, None),
            runtime.evaluate_selected_f64(
                &momenta,
                POINT_COUNT,
                &Selectors::all(),
                None,
                Some(&point_colors),
            ),
        ] {
            assert_eq!(result.unwrap_err().kind(), ErrorKind::InvalidArgument);
        }
        println!("ok");
        return Ok(());
    }
    assert!(metadata_helicities.len() >= 2 && metadata_colors.len() >= 2);
    let resolved = runtime.evaluate_resolved_f64(&momenta, POINT_COUNT, &Selectors::all())?;
    let homogeneous = [0_u32; POINT_COUNT];
    let pooled_helicities = [0_u32, 0, 0, 0, 1, 1, 1, 1];
    let pooled_colors = [1_u32, 1, 1, 1, 0, 0, 0, 0];
    let helicities = [0_u32, 1, 0, 1, 0, 1, 0, 1];
    let colors = [1_u32, 0, 1, 0, 1, 0, 1, 0];
    let random_helicities = [0_u32, 0, 0, 1, 0, 1, 1, 1];
    let random_colors = [1_u32, 1, 0, 1, 1, 1, 0, 1];
    let global_selectors = Selectors::all()
        .with_helicities([metadata_helicities[0].id.clone()])
        .with_colors([metadata_colors[0].id.clone()]);
    let global_helicity = Selectors::all().with_helicities([metadata_helicities[0].id.clone()]);
    let global_color = Selectors::all().with_colors([metadata_colors[0].id.clone()]);

    let all = runtime.evaluate_selected_f64(
        &momenta, POINT_COUNT, &Selectors::all(), None, None)?;
    let global = runtime.evaluate_selected_f64(
        &momenta, POINT_COUNT, &global_selectors, None, None)?;
    let global_helicity_only =
        runtime.evaluate_selected_f64(&momenta, POINT_COUNT, &global_helicity, None, None)?;
    let global_color_only =
        runtime.evaluate_selected_f64(&momenta, POINT_COUNT, &global_color, None, None)?;
    let homogeneous_values = runtime.evaluate_selected_f64(
        &momenta, POINT_COUNT, &Selectors::all(), Some(&homogeneous), Some(&homogeneous))?;
    let pooled = runtime.evaluate_selected_f64(
        &momenta,
        POINT_COUNT,
        &Selectors::all(),
        Some(&pooled_helicities),
        Some(&pooled_colors),
    )?;
    let alternating = runtime.evaluate_selected_f64(
        &momenta, POINT_COUNT, &Selectors::all(), Some(&helicities), Some(&colors))?;
    let seeded_random = runtime.evaluate_selected_f64(
        &momenta,
        POINT_COUNT,
        &Selectors::all(),
        Some(&random_helicities),
        Some(&random_colors),
    )?;
    let helicity_only = runtime.evaluate_selected_f64(
        &momenta, POINT_COUNT, &Selectors::all(), Some(&helicities), None)?;
    let color_only = runtime.evaluate_selected_f64(
        &momenta, POINT_COUNT, &Selectors::all(), None, Some(&colors))?;
    for point in 0..POINT_COUNT {
        let mut expected_all = 0.0;
        for helicity in 0..metadata_helicities.len() {
            for color in 0..metadata_colors.len() {
                expected_all += resolved.get(point, helicity, color).unwrap();
            }
        }
        assert!(close_value(all[point], expected_all));
        assert!(close_value(global[point], resolved.get(point, 0, 0).unwrap()));
        let expected_global_helicity = (0..metadata_colors.len())
            .map(|color| resolved.get(point, 0, color).unwrap())
            .sum();
        assert!(close_value(global_helicity_only[point], expected_global_helicity));
        let expected_global_color = (0..metadata_helicities.len())
            .map(|helicity| resolved.get(point, helicity, 0).unwrap())
            .sum();
        assert!(close_value(global_color_only[point], expected_global_color));
        assert!(close_value(homogeneous_values[point], resolved.get(point, 0, 0).unwrap()));
        assert!(close_value(
            pooled[point],
            resolved
                .get(
                    point,
                    pooled_helicities[point] as usize,
                    pooled_colors[point] as usize,
                )
                .unwrap()
        ));
        assert!(close_value(
            alternating[point],
            resolved.get(point, helicities[point] as usize, colors[point] as usize).unwrap()
        ));
        assert!(close_value(
            seeded_random[point],
            resolved
                .get(
                    point,
                    random_helicities[point] as usize,
                    random_colors[point] as usize,
                )
                .unwrap()
        ));
        let expected_helicity = (0..metadata_colors.len())
            .map(|color| resolved.get(point, helicities[point] as usize, color).unwrap())
            .sum();
        assert!(close_value(helicity_only[point], expected_helicity));
        let expected_color = (0..metadata_helicities.len())
            .map(|helicity| resolved.get(point, helicity, colors[point] as usize).unwrap())
            .sum();
        assert!(close_value(color_only[point], expected_color));
    }

    let invalid = [u32::MAX; POINT_COUNT];
    for result in [
        runtime.evaluate_selected_f64(
            &momenta, POINT_COUNT, &global_helicity, Some(&homogeneous), None),
        runtime.evaluate_selected_f64(
            &momenta, POINT_COUNT, &global_color, None, Some(&homogeneous)),
        runtime.evaluate_selected_f64(
            &momenta, POINT_COUNT, &Selectors::all(), Some(&[0]), None),
        runtime.evaluate_selected_f64(
            &momenta, POINT_COUNT, &Selectors::all(), None, Some(&[0])),
        runtime.evaluate_selected_f64(
            &momenta, POINT_COUNT, &Selectors::all(), Some(&invalid), None),
        runtime.evaluate_selected_f64(
            &momenta, POINT_COUNT, &Selectors::all(), None, Some(&invalid)),
    ] {
        assert_eq!(result.unwrap_err().kind(), ErrorKind::InvalidArgument);
    }
    println!("ok");
    Ok(())
}
"""


_FORTRAN_PROBE = r"""
program selector_probe
  use, intrinsic :: iso_c_binding
  use rusticol
  implicit none
  type(rusticol_runtime) :: runtime
  type(rusticol_helicity_configuration), allocatable :: helicity_metadata(:)
  type(rusticol_color_component), allocatable :: color_metadata(:)
  character(len=4096) :: artifact, process, argument
  character(len=1) :: contracted_mode
  character(len=256) :: helicity_ids(1), color_ids(1)
  real(c_double), allocatable, target :: momenta(:)
  real(c_double), allocatable, target :: resolved(:, :, :), selected(:)
  integer(c_int32_t), target :: homogeneous(8), pooled_helicities(8), pooled_colors(8)
  integer(c_int32_t), target :: helicities(8), colors(8), random_helicities(8), random_colors(8)
  integer(c_int32_t), target :: short_selector(1), invalid(8)
  integer(c_int) :: status, environment_status
  integer, parameter :: point_count = 8
  integer :: argument_count, point, index, helicity, color
  real(c_double) :: expected, scale

  argument_count = command_argument_count()
  if (argument_count < 3) stop 2
  call get_command_argument(1, artifact)
  call get_command_argument(2, process)
  allocate(momenta(argument_count - 2))
  if (mod(size(momenta), point_count) /= 0) stop 2
  do index = 3, argument_count
    call get_command_argument(index, argument)
    read(argument, *) momenta(index - 2)
  end do

  call runtime%load(trim(artifact), trim(process), ierr=status)
  if (status /= RUSTICOL_STATUS_OK) stop 10
  helicity_metadata = runtime%helicities(ierr=status)
  if (status /= RUSTICOL_STATUS_OK) stop 11
  color_metadata = runtime%colors(ierr=status)
  if (status /= RUSTICOL_STATUS_OK) stop 12
  if (size(color_metadata) < 1) stop 13
  color_ids(1) = color_metadata(1)%id
  homogeneous = 0_c_int32_t
  call get_environment_variable("RUSTICOL_EXPECT_CONTRACTED_COLOR", contracted_mode, &
      status=environment_status)
  if (environment_status == 0) then
    call runtime%evaluate_selected(momenta, int(point_count, c_size_t), selected, &
        color_ids=color_ids, ierr=status)
    if (status /= RUSTICOL_STATUS_INVALID_ARGUMENT) stop 40
    call runtime%evaluate_selected(momenta, int(point_count, c_size_t), selected, &
        color_flow_by_point=homogeneous, ierr=status)
    if (status /= RUSTICOL_STATUS_INVALID_ARGUMENT) stop 41
    call runtime%close()
    write(*, '(A)') 'ok'
    stop
  end if
  if (size(helicity_metadata) < 2 .or. size(color_metadata) < 2) stop 13
  helicity_ids(1) = helicity_metadata(1)%id
  pooled_helicities = [0_c_int32_t, 0_c_int32_t, 0_c_int32_t, 0_c_int32_t, &
      1_c_int32_t, 1_c_int32_t, 1_c_int32_t, 1_c_int32_t]
  pooled_colors = [1_c_int32_t, 1_c_int32_t, 1_c_int32_t, 1_c_int32_t, &
      0_c_int32_t, 0_c_int32_t, 0_c_int32_t, 0_c_int32_t]
  helicities = [0_c_int32_t, 1_c_int32_t, 0_c_int32_t, 1_c_int32_t, &
      0_c_int32_t, 1_c_int32_t, 0_c_int32_t, 1_c_int32_t]
  colors = [1_c_int32_t, 0_c_int32_t, 1_c_int32_t, 0_c_int32_t, &
      1_c_int32_t, 0_c_int32_t, 1_c_int32_t, 0_c_int32_t]
  random_helicities = [0_c_int32_t, 0_c_int32_t, 0_c_int32_t, 1_c_int32_t, &
      0_c_int32_t, 1_c_int32_t, 1_c_int32_t, 1_c_int32_t]
  random_colors = [1_c_int32_t, 1_c_int32_t, 0_c_int32_t, 1_c_int32_t, &
      1_c_int32_t, 1_c_int32_t, 0_c_int32_t, 1_c_int32_t]
  short_selector = [0_c_int32_t]
  invalid = huge(0_c_int32_t)

  call runtime%evaluate_resolved(momenta, int(point_count, c_size_t), resolved, ierr=status)
  if (status /= RUSTICOL_STATUS_OK) stop 14
  call runtime%evaluate_selected(momenta, int(point_count, c_size_t), selected, ierr=status)
  if (status /= RUSTICOL_STATUS_OK) stop 18
  do point = 1, point_count
    expected = sum(resolved(:, :, point))
    if (.not. close_value(selected(point), expected)) stop 19
  end do
  call runtime%evaluate_selected(momenta, int(point_count, c_size_t), selected, &
      helicity_ids=helicity_ids, color_ids=color_ids, ierr=status)
  if (status /= RUSTICOL_STATUS_OK) stop 20
  do point = 1, point_count
    if (.not. close_value(selected(point), resolved(1, 1, point))) stop 21
  end do

  call runtime%evaluate_selected(momenta, int(point_count, c_size_t), selected, &
      helicity_ids=helicity_ids, ierr=status)
  if (status /= RUSTICOL_STATUS_OK) stop 40
  do point = 1, point_count
    expected = sum(resolved(:, 1, point))
    if (.not. close_value(selected(point), expected)) stop 41
  end do

  call runtime%evaluate_selected(momenta, int(point_count, c_size_t), selected, &
      color_ids=color_ids, ierr=status)
  if (status /= RUSTICOL_STATUS_OK) stop 42
  do point = 1, point_count
    expected = sum(resolved(1, :, point))
    if (.not. close_value(selected(point), expected)) stop 43
  end do

  call runtime%evaluate_selected(momenta, int(point_count, c_size_t), selected, &
      helicity_by_point=homogeneous, color_flow_by_point=homogeneous, ierr=status)
  if (status /= RUSTICOL_STATUS_OK) stop 22
  do point = 1, point_count
    if (.not. close_value(selected(point), resolved(1, 1, point))) stop 23
  end do

  call runtime%evaluate_selected(momenta, int(point_count, c_size_t), selected, &
      helicity_by_point=pooled_helicities, color_flow_by_point=pooled_colors, ierr=status)
  if (status /= RUSTICOL_STATUS_OK) stop 24
  do point = 1, point_count
    if (.not. close_value(selected(point), &
        resolved(pooled_colors(point) + 1, pooled_helicities(point) + 1, point))) stop 25
  end do

  call runtime%evaluate_selected(momenta, int(point_count, c_size_t), selected, &
      helicity_by_point=helicities, color_flow_by_point=colors, ierr=status)
  if (status /= RUSTICOL_STATUS_OK) stop 26
  do point = 1, point_count
    if (.not. close_value(selected(point), &
        resolved(colors(point) + 1, helicities(point) + 1, point))) stop 27
  end do

  call runtime%evaluate_selected(momenta, int(point_count, c_size_t), selected, &
      helicity_by_point=random_helicities, color_flow_by_point=random_colors, ierr=status)
  if (status /= RUSTICOL_STATUS_OK) stop 28
  do point = 1, point_count
    if (.not. close_value(selected(point), &
        resolved(random_colors(point) + 1, random_helicities(point) + 1, point))) stop 29
  end do

  call runtime%evaluate_selected(momenta, int(point_count, c_size_t), selected, &
      helicity_by_point=helicities, ierr=status)
  if (status /= RUSTICOL_STATUS_OK) stop 50
  do point = 1, point_count
    expected = 0.0_c_double
    do color = 1, size(color_metadata)
      expected = expected + resolved(color, helicities(point) + 1, point)
    end do
    if (.not. close_value(selected(point), expected)) stop 51
  end do

  call runtime%evaluate_selected(momenta, int(point_count, c_size_t), selected, &
      color_flow_by_point=colors, ierr=status)
  if (status /= RUSTICOL_STATUS_OK) stop 52
  do point = 1, point_count
    expected = 0.0_c_double
    do helicity = 1, size(helicity_metadata)
      expected = expected + resolved(colors(point) + 1, helicity, point)
    end do
    if (.not. close_value(selected(point), expected)) stop 53
  end do

  call runtime%evaluate_selected(momenta, int(point_count, c_size_t), selected, &
      helicity_ids=helicity_ids, helicity_by_point=homogeneous, ierr=status)
  if (status /= RUSTICOL_STATUS_INVALID_ARGUMENT) stop 30
  call runtime%evaluate_selected(momenta, int(point_count, c_size_t), selected, &
      color_ids=color_ids, color_flow_by_point=homogeneous, ierr=status)
  if (status /= RUSTICOL_STATUS_INVALID_ARGUMENT) stop 31
  call runtime%evaluate_selected(momenta, int(point_count, c_size_t), selected, &
      helicity_by_point=short_selector, ierr=status)
  if (status /= RUSTICOL_STATUS_INVALID_ARGUMENT) stop 32
  call runtime%evaluate_selected(momenta, int(point_count, c_size_t), selected, &
      color_flow_by_point=short_selector, ierr=status)
  if (status /= RUSTICOL_STATUS_INVALID_ARGUMENT) stop 33
  call runtime%evaluate_selected(momenta, int(point_count, c_size_t), selected, &
      helicity_by_point=invalid, ierr=status)
  if (status /= RUSTICOL_STATUS_INVALID_ARGUMENT) stop 34
  call runtime%evaluate_selected(momenta, int(point_count, c_size_t), selected, &
      color_flow_by_point=invalid, ierr=status)
  if (status /= RUSTICOL_STATUS_INVALID_ARGUMENT) stop 35

  call runtime%close()
  write(*, '(A)') 'ok'

contains
  logical function close_value(actual, wanted)
    real(c_double), intent(in) :: actual, wanted
    scale = max(abs(actual), abs(wanted))
    close_value = abs(actual - wanted) <= 1.0e-15_c_double + 1.0e-12_c_double * scale
  end function close_value
end program selector_probe
"""


def _compile_probe(
    language: str,
    source_text: str,
    native_sdk: _NativeSdk,
    directory: Path,
) -> Path:
    suffix = {"c": ".c", "cpp": ".cpp", "fortran": ".f90", "rust": ".rs"}[language]
    source = directory / f"selector_probe{suffix}"
    source.write_text(textwrap.dedent(source_text).lstrip(), encoding="utf-8")
    binary = directory / f"selector_probe_{language}"
    sdk = native_sdk.config
    if language == "c":
        command = [
            native_sdk.cc,
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-pedantic",
            *sdk["cflags"],
            str(source),
            "-o",
            str(binary),
            *sdk["link_flags"],
        ]
    elif language == "cpp":
        command = [
            native_sdk.cxx,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-pedantic",
            *sdk["cflags"],
            str(source),
            "-o",
            str(binary),
            *sdk["link_flags"],
        ]
    elif language == "fortran":
        command = [
            native_sdk.fortran,
            "-std=f2008",
            "-ffree-line-length-none",
            f"-J{directory}",
            sdk["fortran_source"],
            str(source),
            "-o",
            str(binary),
            *sdk["link_flags"],
        ]
    else:
        command = [
            native_sdk.rustc,
            "--edition=2021",
            "-Dwarnings",
            str(source),
            "-o",
            str(binary),
            *sdk["rust_flags"],
        ]
    environment = _source_environment()
    if language == "rust":
        environment["RUSTICOL_RUST_SOURCE"] = sdk["rust_source"]
    completed = subprocess.run(
        command,
        cwd=directory,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, (
        f"{language} selector probe failed to compile\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    return binary


@pytest.mark.parametrize(
    ("language", "source"),
    (
        ("c", _C_PROBE),
        ("cpp", _CPP_PROBE),
        ("fortran", _FORTRAN_PROBE),
        ("rust", _RUST_PROBE),
    ),
)
def test_native_selector_api_parity(
    language: str,
    source: str,
    selector_artifact: Path,
    native_sdk: _NativeSdk,
    tmp_path: Path,
) -> None:
    binary = _compile_probe(language, source, native_sdk, tmp_path)
    points = _phase_space_points(selector_artifact)
    command = [
        str(binary),
        str(selector_artifact),
        "dd_zgg",
        *(
            format(component, ".17g")
            for point in points
            for leg in point
            for component in leg
        ),
    ]
    completed = subprocess.run(
        command,
        env=_source_environment(),
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, (
        f"{language} selector probe failed\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    assert completed.stdout.strip() == "ok"


@pytest.mark.parametrize(
    ("language", "source"),
    (
        ("c", _C_PROBE),
        ("cpp", _CPP_PROBE),
        ("fortran", _FORTRAN_PROBE),
        ("rust", _RUST_PROBE),
    ),
)
def test_native_color_selectors_reject_contracted_axis(
    language: str,
    source: str,
    contracted_selector_artifact: Path,
    native_sdk: _NativeSdk,
    tmp_path: Path,
) -> None:
    binary = _compile_probe(language, source, native_sdk, tmp_path)
    points = _phase_space_points(contracted_selector_artifact)
    command = [
        str(binary),
        str(contracted_selector_artifact),
        "dd_zgg",
        *(
            format(component, ".17g")
            for point in points
            for leg in point
            for component in leg
        ),
    ]
    environment = _source_environment()
    environment["RUSTICOL_EXPECT_CONTRACTED_COLOR"] = "1"
    completed = subprocess.run(
        command,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, (
        f"{language} contracted-color selector probe failed\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    assert completed.stdout.strip() == "ok"

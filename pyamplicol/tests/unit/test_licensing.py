# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import io
import os
from types import ModuleType
from typing import Literal

import pytest

from pyamplicol.config import (
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    ProcessConfig,
    ProcessEntry,
    RunConfig,
)
from pyamplicol.licensing import (
    SymbolicaLicenseState,
    detect_symbolica_license,
    request_hobbyist_license,
    request_trial_license,
    reset_suggestion_state_for_tests,
    resolve_symbolica_resource_config,
    symbolica_resource_clamps,
)


def _module(*, licensed: bool) -> ModuleType:
    module = ModuleType("fake_symbolica")
    module.is_licensed = lambda: licensed  # type: ignore[attr-defined]
    return module


def test_unlicensed_detection_suggests_once_and_hides_json_banner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_suggestion_state_for_tests()
    monkeypatch.setenv("SYMBOLICA_HIDE_BANNER", "0")
    stream = io.StringIO()
    state = detect_symbolica_license(
        stream=stream,
        loader=lambda: _module(licensed=False),
    )
    detect_symbolica_license(stream=stream, loader=lambda: _module(licensed=False))
    assert state == SymbolicaLicenseState(licensed=False, restricted=True)
    assert stream.getvalue().count("restricted mode") == 1

    detect_symbolica_license(
        json_mode=True,
        loader=lambda: _module(licensed=False),
    )
    assert os.environ["SYMBOLICA_HIDE_BANNER"] == "1"


def test_restricted_mode_clamps_both_generation_axes() -> None:
    config = RunConfig(
        action="generate",
        process=ProcessConfig(
            entries=(ProcessEntry("d d~ > z"), ProcessEntry("d d~ > z g"))
        ),
    )
    clamps = symbolica_resource_clamps(
        config,
        SymbolicaLicenseState(licensed=False, restricted=True),
        cpu_budget=16,
    )
    assert [(item.path, item.effective) for item in clamps] == [
        ("generation.workers", 1),
        ("evaluator.optimization.cores", 1),
    ]


def test_licensed_mode_partitions_concrete_processes_across_affinity_budget() -> None:
    config = RunConfig(
        action="generate",
        process=ProcessConfig(
            entries=(
                ProcessEntry("p1 > p2"),
                ProcessEntry("p1 > p3"),
                ProcessEntry("p1 > p4"),
            )
        ),
    )
    clamps = symbolica_resource_clamps(
        config,
        SymbolicaLicenseState(licensed=True, restricted=False),
        cpu_budget=12,
        process_count=3,
    )
    assert [(item.path, item.effective) for item in clamps] == [
        ("generation.workers", 3),
        ("evaluator.optimization.cores", 4),
    ]


def test_licensed_mode_defers_until_concrete_process_count_is_known() -> None:
    config = RunConfig(
        action="generate",
        process=ProcessConfig(
            entries=(
                ProcessEntry("p1 > p2"),
                ProcessEntry("p1 > p3"),
                ProcessEntry("p1 > p4"),
            )
        ),
    )

    clamps = symbolica_resource_clamps(
        config,
        SymbolicaLicenseState(licensed=True, restricted=False),
        cpu_budget=12,
    )

    assert clamps == ()


def test_programmatic_process_count_controls_resource_partition() -> None:
    config = RunConfig(action="generate")
    clamps = symbolica_resource_clamps(
        config,
        SymbolicaLicenseState(licensed=True, restricted=False),
        cpu_budget=12,
        process_count=3,
    )
    assert [(item.path, item.effective) for item in clamps] == [
        ("generation.workers", 3),
        ("evaluator.optimization.cores", 4),
    ]

    with pytest.raises(ValueError, match="process_count"):
        symbolica_resource_clamps(
            config,
            SymbolicaLicenseState(licensed=True, restricted=False),
            process_count=0,
        )


@pytest.mark.parametrize(
    ("requested_workers", "requested_cores", "process_count", "cpu_budget"),
    (
        ("auto", "auto", 50, 8),
        (3, "auto", 20, 8),
        (6, 4, 20, 10),
        (2, 20, 20, 9),
    ),
)
def test_licensed_partition_never_oversubscribes_cpu_budget(
    requested_workers: int | Literal["auto"],
    requested_cores: int | Literal["auto"],
    process_count: int,
    cpu_budget: int,
) -> None:
    config = RunConfig(
        action="generate",
        generation=GenerationConfig(workers=requested_workers),
        evaluator=EvaluatorConfig(
            optimization=EvaluatorOptimizationConfig(cores=requested_cores)
        ),
    )

    resolution = resolve_symbolica_resource_config(
        config,
        SymbolicaLicenseState(licensed=True, restricted=False),
        cpu_budget=cpu_budget,
        process_count=process_count,
    )
    workers = resolution.effective.generation.workers
    cores = resolution.effective.evaluator.optimization.cores

    assert isinstance(workers, int)
    assert isinstance(cores, int)
    assert workers <= process_count
    assert workers * cores <= cpu_budget


def test_resource_repartition_replaces_stale_clamps_from_requested_values() -> None:
    state = SymbolicaLicenseState(licensed=True, restricted=False)
    provisional = resolve_symbolica_resource_config(
        RunConfig(action="generate"),
        state,
        cpu_budget=12,
        process_count=1,
    )

    final = resolve_symbolica_resource_config(
        provisional,
        state,
        cpu_budget=12,
        process_count=5,
    )

    assert final.requested.generation.workers == "auto"
    assert final.requested.evaluator.optimization.cores == "auto"
    assert final.effective.generation.workers == 5
    assert final.effective.evaluator.optimization.cores == 2
    assert [item.reason for item in final.clamps] == [
        "shared affinity-aware CPU budget for concurrent process generation",
        "shared affinity-aware CPU budget for Symbolica evaluator work",
    ]


def test_licensed_partition_uses_process_affinity_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        os,
        "sched_getaffinity",
        lambda _pid: {2, 4, 8},
        raising=False,
    )

    resolution = resolve_symbolica_resource_config(
        RunConfig(action="generate"),
        SymbolicaLicenseState(licensed=True, restricted=False),
        process_count=10,
    )

    assert resolution.effective.generation.workers == 3
    assert resolution.effective.evaluator.optimization.cores == 1


def test_request_helpers_forward_only_after_validation() -> None:
    calls: list[tuple[object, ...]] = []
    module = _module(licensed=True)
    module.request_trial_license = lambda *args: calls.append(args)  # type: ignore[attr-defined]
    module.request_hobbyist_license = lambda *args: calls.append(args)  # type: ignore[attr-defined]

    request_trial_license(
        "Ada",
        "ada@example.org",
        "Institute",
        loader=lambda: module,
    )
    request_hobbyist_license("Grace", "grace@example.org", loader=lambda: module)
    assert calls == [
        ("Ada", "ada@example.org", "Institute"),
        ("Grace", "grace@example.org"),
    ]
    with pytest.raises(ValueError, match="valid email"):
        request_hobbyist_license("Grace", "invalid", loader=lambda: module)

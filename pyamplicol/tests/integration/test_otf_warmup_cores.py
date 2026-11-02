# SPDX-License-Identifier: 0BSD
"""Small nonzero HEFT families exercise call-local parallel OTF preparation."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from pyamplicol import Generator, ModelSource, Runtime
from pyamplicol.assets.prepared_models import (
    BUILTIN_SM_HEFT_JIT_O2,
    packaged_prepared_model_path,
)
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationValidationConfig,
    JITConfig,
    ProcessConfig,
    RunConfig,
)
from pyamplicol.models.builtin.validation import generic_validation_point
from pyamplicol.reporting import ProgressEvent, ProgressUpdate


class _Progress:
    def __init__(self) -> None:
        self.events: list[ProgressEvent] = []

    def emit(self, event: ProgressEvent) -> None:
        self.events.append(event)

    def workers(self) -> set[int]:
        return {
            int(event.details["workers"])
            for event in self.events
            if isinstance(event, ProgressUpdate) and "workers" in event.details
        }


@pytest.mark.parametrize("process", ("g g > H g", "g g > H g g"))
@pytest.mark.parametrize(
    ("accuracy", "contraction"),
    (("lc", "direct"), ("full", "direct"), ("full", "symmetric-group-fft")),
)
def test_heft_warm_up_core_override_preserves_values_and_cache(
    tmp_path: Path, process: str, accuracy: str, contraction: str
) -> None:
    native = pytest.importorskip("pyamplicol._rusticol")
    output = tmp_path / "process"
    config = RunConfig(
        action="generate",
        color=ColorConfig(accuracy=accuracy, contraction=contraction),
        process=ProcessConfig(
            coupling_order_policy="explicit", max_coupling_orders={"HIG": 1}
        ),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
            validation=GenerationValidationConfig(
                enabled=False, post_build_validation=False
            ),
        ),
        evaluator=EvaluatorConfig(
            execution_mode="on-the-fly",
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(optimization_level=2),
        ),
    )
    with packaged_prepared_model_path(BUILTIN_SM_HEFT_JIT_O2) as prepared:
        Generator(config).generate(
            process, output, model=ModelSource.from_path(prepared)
        )
    points = tuple(
        tuple(tuple(p.momentum) for p in generic_validation_point(process, seed=seed))
        for seed in (211, 419)
    )
    # Callers of the native Python binding get the same validation, before
    # constructing any currents, even if they bypass the public facade.
    native_runtime = native.Runtime.load(output)
    for invalid in (True, False, 0, -1, 1.5, "4", 2**100):
        with pytest.raises((TypeError, ValueError), match="n_cores"):
            native_runtime._on_the_fly_warm_up_f64_json(points[:1], n_cores=invalid)
    del native_runtime
    expected = None
    expected_state = None
    observations = []
    for n_cores in (1, 2, 4):
        runtime = Runtime.load(output)
        progress = _Progress()
        started = time.perf_counter()
        warm = runtime.warm_up(points[:1], n_cores=n_cores, progress=progress)
        elapsed = time.perf_counter() - started
        assert not warm.already_warm
        assert warm.warmed_query_count > 0
        assert len(progress.workers()) == 1
        workers = next(iter(progress.workers()))
        assert 1 <= workers <= n_cores  # Hosts may restrict available cores.
        values = runtime.evaluate(points)
        assert all(abs(complex(value)) > 0 for value in values)
        state = runtime.inspect()["on_the_fly_state"]
        if expected is None:
            expected, expected_state = values, state
        else:
            assert values == pytest.approx(expected, rel=2e-12)
            assert state == expected_state

        # Neither a different override nor omission rebuilds a warm family.
        repeated = runtime.warm_up(points[:1], n_cores=3)
        assert repeated.already_warm and repeated.warmed_query_count == 0
        default_progress = _Progress()
        default = runtime.warm_up(points[:1], progress=default_progress)
        assert default.already_warm and default.warmed_query_count == 0
        assert default_progress.workers() == {1}
        observations.append(
            {
                "n_cores": n_cores,
                "effective_workers": workers,
                "warm_up_seconds": elapsed,
                "queries": warm.query_count,
                "cumulative_seconds_at_stage_end": {
                    str(event.details["stage"]): event.details["elapsed_seconds"]
                    for event in progress.events
                    if isinstance(event, ProgressUpdate)
                    and "elapsed_seconds" in event.details
                },
            }
        )
        if n_cores == 4:
            cache = tmp_path / "parallel.cache"
            runtime.save(cache)
            restored = Runtime.load(output)
            restored.load_cache(cache)
            cached = restored.warm_up(points[:1], n_cores=2)
            assert cached.already_warm and cached.warmed_query_count == 0
            assert restored.evaluate(points) == pytest.approx(expected, rel=2e-12)

    # Record the observation, but do not make noisy wall time a correctness gate.
    (tmp_path / "warm-up-profile.json").write_text(
        json.dumps(
            {
                "process": process,
                "accuracy": accuracy,
                "contraction": contraction,
                "observations": observations,
            },
            indent=2,
        )
        + "\n"
    )

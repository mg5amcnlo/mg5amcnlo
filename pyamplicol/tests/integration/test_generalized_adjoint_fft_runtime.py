# SPDX-License-Identifier: 0BSD
"""HEFT singlet attachments retain exact helicity-resolved DDM contractions."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path

import pytest

from pyamplicol import Generator, ModelSource, Runtime
from pyamplicol.assets.prepared_models import (
    BUILTIN_SM_HEFT_JIT_O2,
    BUILTIN_SM_JIT_O2,
    packaged_prepared_model_path,
)
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationRelationDiscoveryConfig,
    GenerationValidationConfig,
    JITConfig,
    ProcessConfig,
    RunConfig,
)
from pyamplicol.models.builtin.validation import generic_validation_point
from tests.integration.test_on_the_fly_runtime_capabilities import (
    _require_native_on_the_fly,
)


@pytest.fixture(scope="module")
def prepared_heft() -> Iterator[ModelSource]:
    _require_native_on_the_fly()
    with packaged_prepared_model_path(BUILTIN_SM_HEFT_JIT_O2) as path:
        yield ModelSource.from_path(path)


@pytest.fixture(scope="module")
def prepared_sm() -> Iterator[ModelSource]:
    _require_native_on_the_fly()
    with packaged_prepared_model_path(BUILTIN_SM_JIT_O2) as path:
        yield ModelSource.from_path(path)


def _configuration(
    lane: str, basis: str, *, coupling_orders: dict[str, int] | None = None
) -> RunConfig:
    return RunConfig(
        action="generate",
        process=ProcessConfig(
            coupling_order_policy="explicit" if coupling_orders else "minimal",
            max_coupling_orders=coupling_orders or {},
        ),
        color=ColorConfig(
            accuracy="full", contraction="symmetric-group-fft", fft_basis=basis
        ),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
            relation_discovery=GenerationRelationDiscoveryConfig(mode="off"),
            validation=GenerationValidationConfig(
                enabled=False, post_build_validation=False
            ),
        ),
        evaluator=EvaluatorConfig(
            execution_mode=lane,
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(optimization_level=2),
        ),
    )


def _assert_color_domain(artifact: Path, *, basis: str, gluons: int) -> None:
    execution_paths = tuple((artifact / "processes").glob("*/execution.json"))
    assert len(execution_paths) == 1
    execution = json.loads(execution_paths[0].read_text(encoding="utf-8"))
    color = execution["runtime_metadata"]["color_contraction"]
    degree = gluons - (2 if basis == "adjoint" else 1)
    # The singlet is not a third colour endpoint. This also detects silently
    # retaining all trace words after an adjoint request.
    assert color["sector_count"] == math.factorial(degree)
    if "selector_policy" in execution:
        assert execution["selector_policy"].get("color_basis", "trace") == basis
    if degree >= 2:
        assert color["storage"] == "convolution-kernels"
        assert color["factorization"]["kind"] == "symmetric-group-fourier"
        assert color["factorization"]["rank"] == degree
        assert color["fft_provenance"]["residual_group_count"] == 0
    # S_0 and S_1 contain only one element and legitimately use the direct
    # scalar contraction rather than a nontrivial Fourier transform.
    if basis == "adjoint":
        selection = _basis_selection(artifact)
        assert selection["requested"] == "adjoint"
        assert selection["actual_basis"] == "adjoint"
        assert selection["name"] == "ddm"
        assert selection["tensor_count"] == math.factorial(degree)


def _basis_selection(artifact: Path) -> dict[str, object]:
    manifest = json.loads((artifact / "artifact.json").read_text(encoding="utf-8"))
    concrete = manifest["extensions"]["generation"]["concrete_processes"]
    assert len(concrete) == 1
    return concrete[0]["filters"]["fft_basis_selection"]


@pytest.mark.parametrize("lane", ("recurrence", "on-the-fly"))
@pytest.mark.parametrize("gluons", (2, 3, 4, 5))
def test_single_insertion_heft_adjoint_matches_trace_for_every_helicity(
    tmp_path: Path, prepared_heft: ModelSource, lane: str, gluons: int
) -> None:
    process = "g g > H" + " g" * (gluons - 2)
    # The two-gluon process is on-shell 2->1 at sqrt(s)=MH. For larger
    # multiplicities use two independent nonexceptional massive RAMBO points.
    seeds = (101,) if gluons == 2 else (101, 203)
    points = tuple(
        tuple(tuple(float(x) for x in leg.momentum) for leg in point)
        for point in (generic_validation_point(process, seed=seed) for seed in seeds)
    )
    runtimes = {}
    try:
        for basis in ("trace", "adjoint"):
            artifact = tmp_path / basis
            Generator(_configuration(lane, basis, coupling_orders={"HIG": 1})).generate(
                process, artifact, model=prepared_heft
            )
            _assert_color_domain(artifact, basis=basis, gluons=gluons)
            runtimes[basis] = Runtime.load(artifact)

        resolved = {
            basis: runtime.evaluate_resolved(points)
            for basis, runtime in runtimes.items()
        }
        assert resolved["adjoint"].helicity_ids == resolved["trace"].helicity_ids
        assert len(resolved["trace"].helicity_ids) == 2**gluons
        totals = {
            basis: tuple(complex(value) for value in runtime.evaluate(points))
            for basis, runtime in runtimes.items()
        }
        for values in totals.values():
            assert all(math.isfinite(value.real) and value.real > 0 for value in values)
            assert all(abs(value.imag) < value.real * 1e-12 for value in values)
        assert totals["adjoint"] == pytest.approx(totals["trace"], rel=2e-11)
        for basis, result in resolved.items():
            assert tuple(complex(value) for value in result.total()) == pytest.approx(
                totals[basis], rel=2e-12
            )
        for index, (trace_point, adjoint_point) in enumerate(
            zip(resolved["trace"].values, resolved["adjoint"].values, strict=True)
        ):
            trace_values = tuple(
                complex(v) for helicity in trace_point for v in helicity
            )
            adjoint_values = tuple(
                complex(v) for helicity in adjoint_point for v in helicity
            )
            assert any(abs(value) > 0 for value in trace_values)
            assert adjoint_values == pytest.approx(
                trace_values, rel=2e-11, abs=totals["trace"][index].real * 2e-13
            )

        if gluons == 4:
            # Exercise parameter invalidation once per runtime lane, including
            # the lowest process that contains the effective Hgggg contact.
            for runtime in runtimes.values():
                runtime.set_model_parameters({"aS": 0.13})
            changed = {
                basis: tuple(complex(value) for value in runtime.evaluate(points))
                for basis, runtime in runtimes.items()
            }
            assert changed["trace"] != pytest.approx(totals["trace"], rel=1e-6)
            assert changed["adjoint"] == pytest.approx(changed["trace"], rel=2e-11)
    finally:
        runtimes.clear()


@pytest.mark.parametrize("lane", ("recurrence", "on-the-fly"))
@pytest.mark.parametrize(
    "process,strategy,coupling_orders",
    (
        ("d d~ > z g g", "fundamental-chain", None),
        ("d d~ > d d~ g g", "fundamental-chain-products", {"QED": 0}),
        ("e- e+ > mu- mu+", "singlet", None),
    ),
)
def test_adaptive_adjoint_preserves_fundamental_and_singlet_amplitudes(
    tmp_path: Path,
    prepared_sm: ModelSource,
    lane: str,
    process: str,
    strategy: str,
    coupling_orders: dict[str, int] | None,
) -> None:
    points = tuple(
        tuple(tuple(float(x) for x in leg.momentum) for leg in point)
        for point in (
            generic_validation_point(process, seed=seed) for seed in (101, 203)
        )
    )
    evaluations = {}
    for basis in ("trace", "adjoint"):
        artifact = tmp_path / basis
        Generator(
            _configuration(lane, basis, coupling_orders=coupling_orders)
        ).generate(process, artifact, model=prepared_sm)
        if basis == "adjoint":
            selection = _basis_selection(artifact)
            assert selection["requested"] == "adjoint"
            assert selection["actual_basis"] == "trace"
            assert selection["name"] == strategy
        runtime = Runtime.load(artifact)
        try:
            resolved = runtime.evaluate_resolved(points)
            total = tuple(complex(value) for value in runtime.evaluate(points))
            assert all(value.real > 0 and math.isfinite(value.real) for value in total)
            assert tuple(complex(value) for value in resolved.total()) == pytest.approx(
                total, rel=2e-12
            )
            evaluations[basis] = (
                resolved.helicity_ids,
                tuple(
                    complex(value)
                    for point in resolved.values
                    for helicity in point
                    for value in helicity
                ),
                total,
            )
        finally:
            del runtime
    assert evaluations["adjoint"][0] == evaluations["trace"][0]
    assert evaluations["adjoint"][1] == pytest.approx(
        evaluations["trace"][1], rel=2e-11, abs=1e-300
    )
    assert evaluations["adjoint"][2] == pytest.approx(
        evaluations["trace"][2], rel=2e-11
    )

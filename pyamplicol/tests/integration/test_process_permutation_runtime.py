# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import importlib.util
import shutil
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from pyamplicol import Generator, Runtime
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationRelationDiscoveryConfig,
    GenerationValidationConfig,
    JITConfig,
    RunConfig,
)
from pyamplicol.models.builtin.validation import generic_validation_point

_REPRESENTATIVE = "d d~ > z g"
_PUBLIC = "d~ d > g z"
_REPRESENTATIVE_TO_PUBLIC = (1, 0, 3, 2)
_MODES = ("compiled", "eager", "recurrence")


def _require_native_runtime() -> None:
    if importlib.util.find_spec("pyamplicol._rusticol") is None:
        pytest.skip("the Rusticol extension has not been built")
    if importlib.util.find_spec("symbolica") is None:
        pytest.skip("Symbolica is unavailable")


def _generation_config(execution_mode: str) -> RunConfig:
    return RunConfig(
        action="generate",
        color=ColorConfig(accuracy="lc"),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
            relation_discovery=GenerationRelationDiscoveryConfig(mode="off"),
            validation=GenerationValidationConfig(
                enabled=False,
                post_build_validation=False,
            ),
        ),
        evaluator=EvaluatorConfig(
            execution_mode=execution_mode,
            optimization=EvaluatorOptimizationConfig(cores=1),
            jit=JITConfig(optimization_level=2),
        ),
    )


@pytest.fixture(scope="module")
def permutation_artifacts(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[dict[str, Path]]:
    _require_native_runtime()
    root = tmp_path_factory.mktemp("process-permutation-runtime")
    artifacts = {mode: root / mode for mode in _MODES}
    try:
        for mode, artifact in artifacts.items():
            Generator(_generation_config(mode)).generate(_REPRESENTATIVE, artifact)
        yield artifacts
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _representative_point() -> tuple[tuple[float, float, float, float], ...]:
    return tuple(
        tuple(float(component) for component in particle.momentum)
        for particle in generic_validation_point(_REPRESENTATIVE)
    )


def _public_point(
    representative: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], ...]:
    public: list[tuple[float, ...] | None] = [None] * len(representative)
    for representative_index, public_index in enumerate(
        _REPRESENTATIVE_TO_PUBLIC
    ):
        public[public_index] = tuple(representative[representative_index])
    assert all(momentum is not None for momentum in public)
    return tuple(momentum for momentum in public if momentum is not None)


def _flatten(values: object) -> tuple[complex, ...]:
    flattened: list[complex] = []

    def visit(value: object) -> None:
        if isinstance(value, tuple | list):
            for item in value:
                visit(item)
        else:
            flattened.append(complex(value))

    visit(values)
    return tuple(flattened)


def _assert_same_values(actual: object, expected: object) -> None:
    assert _flatten(actual) == pytest.approx(
        _flatten(expected),
        rel=1.0e-12,
        abs=1.0e-15,
    )


@pytest.mark.parametrize("execution_mode", _MODES)
def test_inferred_both_side_permutation_preserves_every_execution_lane(
    permutation_artifacts: dict[str, Path],
    execution_mode: str,
) -> None:
    artifact = permutation_artifacts[execution_mode]
    representative = Runtime.load(artifact, process=_REPRESENTATIVE)
    public = Runtime.load(artifact, process=_PUBLIC)
    representative_point = _representative_point()
    public_point = _public_point(representative_point)

    assert public.physics.process == _PUBLIC
    assert public.physics.process_id == representative.physics.process_id
    assert tuple(item.pdg_id for item in public.physics.external_particles) == (
        -1,
        1,
        21,
        23,
    )
    for representative_helicity, public_helicity in zip(
        representative.physics.helicities,
        public.physics.helicities,
        strict=True,
    ):
        expected: list[int | None] = [None] * len(_REPRESENTATIVE_TO_PUBLIC)
        for representative_index, public_index in enumerate(
            _REPRESENTATIVE_TO_PUBLIC
        ):
            expected[public_index] = representative_helicity.values[
                representative_index
            ]
        assert public_helicity.values == tuple(expected)
    for representative_flow, public_flow in zip(
        representative.physics.color_flows,
        public.physics.color_flows,
        strict=True,
    ):
        assert public_flow.word == tuple(
            _REPRESENTATIVE_TO_PUBLIC[label - 1] + 1
            for label in representative_flow.word
        )

    for count in (1, 5):
        representative_batch = (representative_point,) * count
        public_batch = (public_point,) * count
        representative_resolved = representative.evaluate_resolved(
            representative_batch
        )
        public_resolved = public.evaluate_resolved(public_batch)
        _assert_same_values(public_resolved.values, representative_resolved.values)
        _assert_same_values(public_resolved.total(), representative_resolved.total())
        _assert_same_values(
            public.evaluate(public_batch),
            representative.evaluate(representative_batch),
        )

    representative_resolved = representative.evaluate_resolved(
        (representative_point,)
    )
    public_resolved = public.evaluate_resolved((public_point,))
    representative_helicity = next(
        item for item in representative.physics.helicities if not item.structural_zero
    )
    helicity_index = representative.physics.helicities.index(
        representative_helicity
    )
    public_helicity = public.physics.helicities[helicity_index]
    representative_flow = representative.physics.color_flows[0]
    public_flow = public.physics.color_flows[0]

    representative_selected = representative.evaluate_resolved(
        (representative_point,),
        helicities=(representative_helicity.id,),
        color_flows=(representative_flow.id,),
    )
    public_selected = public.evaluate_resolved(
        (public_point,),
        helicities=(public_helicity.id,),
        color_flows=(public_flow.id,),
    )
    assert representative_selected.helicity_ids == (representative_helicity.id,)
    assert representative_selected.color_ids == (representative_flow.id,)
    assert public_selected.helicity_ids == (public_helicity.id,)
    assert public_selected.color_ids == (public_flow.id,)
    _assert_same_values(public_selected.values, representative_selected.values)
    _assert_same_values(public_selected.total(), representative_selected.total())
    _assert_same_values(
        public.evaluate(
            (public_point,),
            helicities=(public_helicity.id,),
            color_flows=(public_flow.id,),
        ),
        representative.evaluate(
            (representative_point,),
            helicities=(representative_helicity.id,),
            color_flows=(representative_flow.id,),
        ),
    )
    assert public_resolved.shape == representative_resolved.shape

    representative_exact_full = representative.evaluate_resolved(
        (representative_point,), precision=32
    )
    public_exact_full = public.evaluate_resolved((public_point,), precision=32)
    assert public_exact_full.helicity_ids == tuple(
        item.id for item in public.physics.helicities
    )
    assert public_exact_full.color_ids == tuple(
        item.id for item in public.physics.color_flows
    )
    _assert_same_values(
        public_exact_full.values,
        representative_exact_full.values,
    )
    _assert_same_values(
        public_exact_full.total(),
        representative_exact_full.total(),
    )

    representative_exact = representative.evaluate_resolved(
        (representative_point,),
        helicities=(representative_helicity.id,),
        color_flows=(representative_flow.id,),
        precision=32,
    )
    public_exact = public.evaluate_resolved(
        (public_point,),
        helicities=(public_helicity.id,),
        color_flows=(public_flow.id,),
        precision=32,
    )
    _assert_same_values(public_exact.values, representative_exact.values)
    _assert_same_values(public_exact.total(), representative_exact.total())
    _assert_same_values(
        public.evaluate(
            (public_point,),
            helicities=(public_helicity.id,),
            color_flows=(public_flow.id,),
            precision=32,
        ),
        representative.evaluate(
            (representative_point,),
            helicities=(representative_helicity.id,),
            color_flows=(representative_flow.id,),
            precision=32,
        ),
    )


def test_runtime_load_accepts_ufo_style_parameter_pairs_directly(
    permutation_artifacts: dict[str, Path],
) -> None:
    artifact = permutation_artifacts["compiled"]
    point = _representative_point()
    pair = Runtime.load(
        artifact,
        process=_REPRESENTATIVE,
        model_parameters={"normalization.alpha_ew": [0.008, 0.0]},
    )
    scalar = Runtime.load(
        artifact,
        process=_REPRESENTATIVE,
        model_parameters={"normalization.alpha_ew": complex(0.008, 0.0)},
    )

    _assert_same_values(pair.evaluate((point,)), scalar.evaluate((point,)))
    _assert_same_values(
        pair.evaluate_resolved((point,)).values,
        scalar.evaluate_resolved((point,)).values,
    )

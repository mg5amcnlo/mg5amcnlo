# SPDX-License-Identifier: 0BSD
"""Source injection and reduction mechanics; no Symbolica session required."""

from __future__ import annotations

import copy
import json
from decimal import Decimal, getcontext, localcontext

import pytest

from pyamplicol.api.errors import ArtifactError, CompatibilityError, EvaluationError
from pyamplicol.runtime.correlated_exact import (
    CorrelatedExactExecutor,
    _fill_correlated_sources,
    _prepare_spin_vectors,
)


def _source(leg=1, *, start=0, helicity=1, initial=False, phase=(1, 0)):
    crossing = {
        "momentum_transform": "negate-four-momentum" if initial else "identity",
        "helicity_factor": -1 if initial else 1,
        "chirality_factor": 1,
        "spin_state_factor": 1,
        "phase": list(phase),
    }
    return {
        "leg_label": leg,
        "source_helicity": helicity,
        "chirality": 0,
        "spin_state": 0,
        "side": "initial" if initial else "final",
        "particle_id": 21,
        "anti_particle_id": 21,
        "source_orientation": "self-conjugate",
        "wavefunction_kind": "vector",
        "dimension": 4,
        "source_basis": "lorentz",
        "crossing": "negate-incoming-momentum" if initial else "identity",
        "value_slot": {"component_start": start, "component_stop": start + 4},
        "source_ir": {
            "basis": "lorentz",
            "component_dimension": 4,
            "wavefunction_family": "vector",
            "identity": {
                "pdg_label": 21,
                "anti_pdg_label": 21,
                "orientation": "self-conjugate",
            },
            "crossing": crossing,
            "states": [
                {"helicity": -1, "chirality": 0, "spin_state": 0},
                {"helicity": 1, "chirality": 0, "spin_state": 0},
            ],
        },
        "applied_crossing": crossing,
    }


def test_clear_releases_evaluators_without_mutating_the_plan():
    executor = object.__new__(CorrelatedExactExecutor)
    executor._stage_evaluators = (object(),)
    executor._amplitude_evaluator = object()
    executor._execution = {"source_plan": "retained"}
    executor.clear()
    assert executor._stage_evaluators is None
    assert executor._amplitude_evaluator is None
    assert executor._execution == {"source_plan": "retained"}


def test_literal_momentum_vector_preserves_nonzero_temporal_component():
    momentum = (Decimal(5), Decimal(3), Decimal(0), Decimal(4))
    overrides = _prepare_spin_vectors({1: momentum}, allowed_legs=(1,), point_count=1)
    state = [(Decimal(0), Decimal(0))] * 4
    _fill_correlated_sources(
        state, (momentum,), {"source_fill": {"sources": [_source()]}}, (), overrides[0]
    )
    assert state == [(component, Decimal(0)) for component in momentum]
    assert state[0][0] == 5  # v=p has NOT been projected onto transverse helicities.


def test_initial_crossing_phase_applied_once_without_conjugating_vector():
    overrides = _prepare_spin_vectors(
        {1: (2 + 3j, 5 - 7j, 11, 13)}, allowed_legs=(1,), point_count=1
    )
    state = [(Decimal(0), Decimal(0))] * 4
    _fill_correlated_sources(
        state,
        ((Decimal(5), Decimal(3), Decimal(0), Decimal(4)),),
        {"source_fill": {"sources": [_source(initial=True, phase=(0, 1))]}},
        (),
        overrides[0],
    )
    assert state == [
        (Decimal(-3), Decimal(2)),
        (Decimal(7), Decimal(5)),
        (Decimal(0), Decimal(11)),
        (Decimal(0), Decimal(13)),
    ]


def test_multiple_legs_and_point_specific_vectors_are_copied():
    batch = [[1, 2, 3, 4], [5, 6, 7, 8]]
    first = _prepare_spin_vectors(
        {1: batch, 2: (9, 10, 11, 12)}, allowed_legs=(1, 2), point_count=2
    )
    batch[0][0] = 100
    assert first[0][1][0] == (Decimal(1), Decimal(0))
    assert first[1][1][0] == (Decimal(5), Decimal(0))
    assert first[0][2] == first[1][2]
    assert _prepare_spin_vectors({}, allowed_legs=(1, 2), point_count=2) == ({}, {})
    assert _prepare_spin_vectors(None, allowed_legs=(1, 2), point_count=1) == ({},)
    assert first[0][1][0] == (Decimal(1), Decimal(0))


def test_real_decimal_vectors_preserve_all_digits_before_source_arithmetic():
    first = Decimal("1.000000000000000000000000000000000000000000000000000000000001")
    second = Decimal("1.000000000000000000000000000000000000000000000000000000000002")
    with localcontext() as context:
        context.prec = 7
        vectors = _prepare_spin_vectors(
            {1: (first, 0, 0, 0), 2: ((second, 0, 0, 0), (first, 0, 0, 0))},
            allowed_legs=(1, 2),
            point_count=2,
        )
    assert vectors[0][1][0][0] is first
    assert vectors[1][1][0][0] is first
    assert vectors[0][2][0][0] is second
    assert vectors[1][2][0][0] is first
    assert vectors[0][1] != vectors[0][2]


def test_complex_decimal_pairs_broadcast_and_batch_without_float_conversion():
    tiny = Decimal("1.00000000000000000000000000000000000000001")
    wave = ((tiny, tiny), 0, 0, 0)
    prepared = _prepare_spin_vectors(
        {1: wave, 2: (wave, wave)}, allowed_legs=(1, 2), point_count=2
    )
    assert prepared[0][1][0] == (tiny, tiny)
    assert prepared[1][2][0] == (tiny, tiny)


def test_multiple_vector_sources_write_independent_full_slots():
    overrides = _prepare_spin_vectors(
        {1: (1, 2, 3, 4), 2: (5j, 6, 7, 8)},
        allowed_legs=(1, 2),
        point_count=1,
    )[0]
    state = [(Decimal(0), Decimal(0))] * 8
    _fill_correlated_sources(
        state,
        ((Decimal(5), Decimal(3), Decimal(0), Decimal(4)),) * 2,
        {"source_fill": {"sources": [_source(), _source(2, start=4)]}},
        (),
        overrides,
    )
    assert state[:4] == list(overrides[1])
    assert state[4:] == list(overrides[2])


@pytest.mark.parametrize(
    "vectors, match",
    [
        ({2: (1, 2, 3, 4)}, "not declared"),
        ({True: (1, 2, 3, 4)}, "not declared"),
        ({1: [[1, 2, 3, 4]]}, "batch"),
        ({1: (float("nan"), 2, 3, 4)}, "finite"),
        ({1: (True, 2, 3, 4)}, "numeric"),
    ],
)
def test_invalid_vector_input(vectors, match):
    with pytest.raises(EvaluationError, match=match):
        _prepare_spin_vectors(vectors, allowed_legs=(1,), point_count=2)


def test_compressed_source_is_rejected():
    source = _source()
    source["value_slot"]["component_stop"] = 3
    with pytest.raises(CompatibilityError, match="four-component"):
        _fill_correlated_sources(
            [(Decimal(0), Decimal(0))] * 4,
            ((Decimal(5), Decimal(0), Decimal(0), Decimal(5)),),
            {"source_fill": {"sources": [source]}},
            (),
            _prepare_spin_vectors({1: (1, 2, 3, 4)}, allowed_legs=(1,), point_count=1)[
                0
            ],
        )


class _NativeState:
    def _exact_runtime_state_json(self):
        return json.dumps({"model_parameter_values": [], "normalization_factor": 99})


class _AmplitudeEvaluator:
    def __init__(self):
        self.inputs = []

    def evaluate(self, values, precision):
        self.inputs.append(tuple(values))
        # Two roots in the same coherent group; their sum must precede squaring.
        return (values[0], values[1], values[4], values[5])


class _StubExecutor(CorrelatedExactExecutor):
    def _load_evaluators(self):
        pass

    def _derive_model_parameters(self, parameters, precision):
        return parameters


def _executor():
    executor = object.__new__(_StubExecutor)
    executor._permutation = (0,)
    executor._native_runtime = _NativeState()
    executor._stage_evaluators = ()
    executor._amplitude_evaluator = _AmplitudeEvaluator()
    executor._physics = {
        "color_accuracy": "full",
        "external_particles": [{"label": 1, "pdg": 21, "role": "final"}],
        "extensions": {"normalization": {"average_factor": 2, "identical_factor": 1}},
        "helicities": [
            {
                "id": f"h:{h}",
                "values": [h],
                "computed": True,
                "structural_zero": False,
                "representative_id": f"h:{h}",
                "coefficient": 1,
            }
            for h in (-1, 1)
        ],
    }
    schema = {
        "parameter_layout": {
            "value_component_count": 8,
            "momentum_parameter_count": 0,
            "parameter_count_if_flattened": 8,
        },
        "external_particles": executor._physics["external_particles"],
        "momentum_slots": [],
        "model": {"particles": [{"pdg": 21, "mass": 0, "mass_parameter": None}]},
        "source_fill": {
            "sources": [_source(start=0, helicity=-1), _source(start=4, helicity=1)]
        },
        "amplitude_stage": {
            "roots": [
                {
                    "output_index": index,
                    "coherent_group_id": index // 2,
                    "helicity_weight": 1,
                    "all_sector_weight": 1,
                }
                for index in range(4)
            ],
            "coherent_groups": [
                {
                    "group_id": index,
                    "helicities": [h],
                    "color_word": [1],
                    "color_sector_id": 0,
                }
                for index, h in enumerate((-1, 1))
            ],
        },
    }
    executor._execution = {
        "runtime_schema": schema,
        "compiled": {
            "stage_evaluators": {
                "stages": [],
                "amplitude_stage": {
                    "parameter_count": 8,
                    "output_length": 4,
                    "output_slots": [
                        {
                            "output_start": 0,
                            "output_stop": 4,
                            "component_start": 0,
                            "component_stop": 4,
                        }
                    ],
                },
            }
        },
    }
    executor._initialize_correlated_plan((1,))
    return executor


def test_coherent_group_descriptors_can_come_from_the_correlated_sidecar():
    executor = _executor()
    amplitude = executor._execution["runtime_schema"]["amplitude_stage"]
    descriptors = amplitude.pop("coherent_groups")
    expected = executor._coherent_groups
    executor._initialize_correlated_plan((1,), descriptors)
    assert executor._coherent_groups == expected


def test_coherent_amplitudes_preserve_phase_and_deduplicate_replaced_helicity():
    executor = _executor()
    batch = executor.coherent_amplitudes(
        [[(5, 3, 0, 4)]], spin_vectors={1: (2 + 3j, 5 - 7j, 11, 13)}
    )
    assert batch.values == (((Decimal(7), Decimal(-4)),),)
    assert len(batch.groups) == 1
    assert batch.groups[0].helicities == (-1,)
    assert batch.normalization_factor == Decimal("0.5")
    assert batch.spin_correlated_legs == (1,)
    assert executor._amplitude_evaluator.inputs[0][0] == (Decimal(2), Decimal(3))
    assert executor._amplitude_evaluator.inputs[0][4] == (Decimal(2), Decimal(3))


def test_no_overrides_reset_and_separate_instances_remain_independent():
    first, second = _executor(), _executor()
    momenta = [[(5, 3, 0, 4)]]
    first.coherent_amplitudes(momenta, spin_vectors={1: (9, 2, 3, 4)})
    ordinary = first.coherent_amplitudes(momenta, spin_vectors={})
    independent = second.coherent_amplitudes(momenta)
    assert ordinary == independent
    assert len(ordinary.groups) == 2
    assert ordinary.spin_correlated_legs == ()
    assert first._amplitude_evaluator.inputs[-1][0] == (Decimal(0), Decimal(0))


def test_point_vectors_and_helicity_selector():
    executor = _executor()
    result = executor.coherent_amplitudes(
        [[(5, 3, 0, 4)], [(5, 0, 3, 4)]],
        spin_vectors={1: [(1, 2, 3, 4), (5, 6, 7, 8)]},
        helicities=["h:1"],
        precision=32,
    )
    assert result.values == (((Decimal(3), Decimal(0)),), ((Decimal(11), Decimal(0)),))
    assert result.groups[0].helicity_id == "h:1"
    with pytest.raises(EvaluationError, match="unknown"):
        executor.coherent_amplitudes([[(5, 3, 0, 4)]], helicities=["missing"])


def _executor_with_intrinsic_zeros():
    executor = _executor()
    executor._permutation = (0, 1)
    executor._physics["external_particles"].append(
        {"label": 2, "pdg": 1, "role": "final"}
    )
    for record in executor._physics["helicities"]:
        record["values"].append(-1)
    amplitude = executor._execution["runtime_schema"]["amplitude_stage"]
    for descriptor in amplitude["coherent_groups"]:
        descriptor["helicities"].append(-1)
    # A spectator chirality gives no roots, independently of the vector source.
    executor._physics["helicities"].extend(
        {
            "id": f"zero:{h}",
            "values": [h, 1],
            "computed": False,
            "structural_zero": True,
            "representative_id": f"zero:{h}",
            "coefficient": 0,
        }
        for h in (-1, 1)
    )
    executor._initialize_correlated_plan((1,))
    return executor


def test_intrinsic_zero_helicities_retain_explicit_zero_selectors():
    executor = _executor_with_intrinsic_zeros()
    point = ((5, 3, 0, 4), (5, -3, 0, -4))
    batch = executor.coherent_amplitudes(
        (point, point), helicities=("zero:-1", "zero:1")
    )
    assert batch.groups == ()
    assert batch.values == ((), ())
    assert batch.normalization_factor == Decimal("0.5")
    assert executor._known_helicity_ids == {"h:-1", "h:1", "zero:-1", "zero:1"}


def test_intrinsic_zero_spectators_do_not_duplicate_replaced_spin_states():
    executor = _executor_with_intrinsic_zeros()
    point = ((5, 3, 0, 4), (5, -3, 0, -4))
    ordinary = executor.coherent_amplitudes((point,))
    assert len(ordinary.groups) == 2
    projected = executor.coherent_amplitudes(
        (point,), spin_vectors={1: (2 + 3j, 5 - 7j, 11, 13)}
    )
    assert projected.values == (((Decimal(7), Decimal(-4)),),)
    assert projected.groups[0].helicities == (-1, -1)
    zero = executor.coherent_amplitudes(
        (point,), spin_vectors={1: (2 + 3j, 5 - 7j, 11, 13)}, helicities=("zero:1",)
    )
    assert zero.groups == ()
    assert zero.values == ((),)


@pytest.mark.parametrize("defect", ("unmarked", "alias", "weight"))
def test_intrinsic_zero_metadata_must_be_explicit_unaliased_and_zero_weight(defect):
    executor = _executor_with_intrinsic_zeros()
    zero = executor._physics["helicities"][-1]
    if defect == "unmarked":
        zero["structural_zero"] = False
    elif defect == "alias":
        zero["representative_id"] = "h:1"
    else:
        zero["coefficient"] = 2
    with pytest.raises(CompatibilityError):
        executor._initialize_correlated_plan((1,))


def test_raw_groups_in_one_physical_coefficient_are_summed_with_weights_once():
    executor = _executor()
    amplitude = executor._execution["runtime_schema"]["amplitude_stage"]
    # Production separates raw groups by internal basis keys and colour weight.
    # The mock outputs stand for already-weighted complex root expressions.
    amplitude["roots"][1]["coherent_group_id"] = 2
    amplitude["roots"][1]["color_weight"] = [3, -2]
    extra = {**amplitude["coherent_groups"][0], "group_id": 2}
    amplitude["coherent_groups"].append(extra)
    executor._initialize_correlated_plan((1,))
    assert len(executor._coherent_groups) == 2
    assert executor._coherent_root_indices[0] == (0, 1)
    batch = executor.coherent_amplitudes(
        [[(5, 3, 0, 4)]], spin_vectors={1: (2 + 3j, 5 - 7j, 11, 13)}
    )
    assert batch.values == (((Decimal(7), Decimal(-4)),),)
    assert len(batch.groups) == 1


def test_raw_groups_cannot_assign_different_words_to_one_physical_sector():
    executor = _executor()
    amplitude = executor._execution["runtime_schema"]["amplitude_stage"]
    amplitude["roots"][1]["coherent_group_id"] = 2
    amplitude["coherent_groups"].append(
        {**amplitude["coherent_groups"][0], "group_id": 2, "color_word": [2]}
    )
    with pytest.raises(ArtifactError, match="disagree"):
        executor._initialize_correlated_plan((1,))


def test_partial_colour_coverage_is_still_rejected():
    executor = _executor_with_intrinsic_zeros()
    amplitude = executor._execution["runtime_schema"]["amplitude_stage"]
    amplitude["coherent_groups"][1]["color_sector_id"] = 1
    with pytest.raises(ArtifactError, match="incomplete helicity/color"):
        executor._initialize_correlated_plan((1,))


class _CountingEvaluator:
    def __init__(self, factor=1):
        self.factor = factor
        self.inputs = []

    def evaluate(self, inputs, precision):
        self.inputs.append(inputs)
        if self.factor == 1:
            return inputs
        return tuple((a * self.factor, b * self.factor) for a, b in inputs)


def _executor_with_stages():
    executor = _executor()
    schema = executor._execution["runtime_schema"]
    schema["parameter_layout"]["parameter_count_if_flattened"] = 10
    schema["parameter_layout"]["value_component_count"] = 10
    stages = []
    for source, target in ((2, 8), (1, 9)):
        stages.append(
            {
                "parameter_count": 1,
                "input_components": [
                    {"parameter_index": 0, "global_component": source}
                ],
                "output_slots": [
                    {
                        "output_start": 0,
                        "output_stop": 1,
                        "component_start": target,
                        "component_stop": target + 1,
                    }
                ],
            }
        )
    stage_set = executor._execution["compiled"]["stage_evaluators"]
    stage_set["stages"] = stages
    stage_set["amplitude_stage"]["parameter_count"] = 4
    stage_set["amplitude_stage"]["input_components"] = [
        {"parameter_index": index, "global_component": source}
        for index, source in enumerate((8, 9, 8, 9))
    ]
    executor._stage_evaluators = (_CountingEvaluator(2), _CountingEvaluator(3))
    executor._amplitude_evaluator = _CountingEvaluator()
    return executor


def test_many_reuses_exact_stages_and_restores_slots_with_point_outer_order():
    executor = _executor_with_stages()
    points = [[(5, 3, 0, 4)], [(5, 0, 3, 4)]]
    vectors = ({1: (2, 5, 11, 13)}, {1: (2, 7, 11, 13)})
    batches = list(
        executor.iter_coherent_amplitudes_many(points, spin_vector_sets=vectors)
    )
    assert [(point, spin) for point, spin, _ in batches] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    ]
    assert [batch.values[0][0] for _, _, batch in batches] == [
        (Decimal(37), Decimal(0)),
        (Decimal(43), Decimal(0)),
        (Decimal(37), Decimal(0)),
        (Decimal(43), Decimal(0)),
    ]
    unchanged, changed = executor._stage_evaluators
    assert len(unchanged.inputs) == 2  # One evaluation per point, despite two spins.
    assert len(changed.inputs) == 4
    assert len(executor._amplitude_evaluator.inputs) == 4
    # No numerical cache survives a call or a change to runtime parameters.
    list(executor.iter_coherent_amplitudes_many(points, spin_vector_sets=vectors))
    assert len(unchanged.inputs) == 4
    assert len(changed.inputs) == 8


def test_many_stage_reuse_matches_separate_calls_and_reuses_amplitude_stage():
    executor = _executor_with_stages()
    points = [[(5, 3, 0, 4)], [(5, 0, 3, 4)]]
    vectors = ({1: (2, 5, 11, 13)}, {1: (8, 5, 11, 13)})
    batches = list(
        executor.iter_coherent_amplitudes_many(
            points, spin_vector_sets=vectors, precision=70
        )
    )
    for point, spin, batch in batches:
        separate = _executor_with_stages().coherent_amplitudes(
            [points[point]], spin_vectors=vectors[spin], precision=70
        )
        assert batch == separate
    assert len(executor._amplitude_evaluator.inputs) == 2
    assert all(len(stage.inputs) == 2 for stage in executor._stage_evaluators)


def test_stage_cache_preserves_signed_zero_and_never_reuses_failed_results():
    executor = _executor()
    evaluator = _CountingEvaluator()
    memo = {}
    plus = ((Decimal("0"), Decimal(0)),)
    minus = ((Decimal("-0"), Decimal(0)),)
    assert executor._evaluate_stage_cached(evaluator, plus, 40, memo, 0) == plus
    executor._evaluate_stage_cached(evaluator, plus, 40, memo, 0)
    returned = executor._evaluate_stage_cached(evaluator, minus, 40, memo, 0)
    assert returned[0][0].is_signed()
    assert len(evaluator.inputs) == 2

    class FailingEvaluator:
        def evaluate(self, inputs, precision):
            raise EvaluationError("test failure")

    with pytest.raises(EvaluationError, match="test failure"):
        executor._evaluate_stage_cached(FailingEvaluator(), plus, 40, memo, 1)
    assert 1 not in memo


def test_many_prepares_parameters_once_and_does_not_leak_decimal_context(monkeypatch):
    executor = _executor_with_stages()
    derivations = []

    def derive(parameters, precision):
        derivations.append((parameters, precision))
        return parameters

    monkeypatch.setattr(executor, "_derive_model_parameters", derive)
    with localcontext() as context:
        context.prec = 7
        iterator = executor.iter_coherent_amplitudes_many(
            [[(5, 3, 0, 4)], [(5, 0, 3, 4)]],
            spin_vector_sets=({1: (2, 5, 11, 13)}, {1: (8, 5, 11, 13)}),
            precision=70,
        )
        next(iterator)
        assert getcontext().prec == 7
        list(iterator)
        assert getcontext().prec == 7
    assert len(derivations) == 1


def test_many_retains_explicit_intrinsic_zero_selectors():
    executor = _executor_with_intrinsic_zeros()
    point = ((5, 3, 0, 4), (5, -3, 0, -4))
    batches = list(
        executor.iter_coherent_amplitudes_many(
            (point, point),
            spin_vector_sets=({}, {1: (1, 2, 3, 4)}),
            helicities=("zero:1",),
        )
    )
    assert len(batches) == 4
    assert all(batch.groups == () and batch.values == ((),) for _, _, batch in batches)


@pytest.mark.parametrize(
    "defect, match",
    [
        ("replay", "unreplayed"),
        ("alias", "aliases"),
        ("pruned", "zero-pruned"),
        ("missing_group", "cover"),
        ("weight", "folded"),
        ("output", "mapping"),
    ],
)
def test_unsafe_or_incomplete_plans_fail_closed(defect, match):
    executor = _executor()
    executor._execution = copy.deepcopy(executor._execution)
    amplitude = executor._execution["runtime_schema"]["amplitude_stage"]
    if defect == "replay":
        executor._helicity_plan = object()
    elif defect == "alias":
        executor._physics["helicities"][1]["representative_id"] = "h:-1"
    elif defect == "pruned":
        executor._physics["helicities"][1]["structural_zero"] = True
    elif defect == "missing_group":
        amplitude["coherent_groups"].pop()
    elif defect == "weight":
        amplitude["roots"][0]["helicity_weight"] = 2
    elif defect == "output":
        amplitude["roots"][0]["output_index"] = 5
    with pytest.raises((ArtifactError, CompatibilityError), match=match):
        executor._initialize_correlated_plan((1,))

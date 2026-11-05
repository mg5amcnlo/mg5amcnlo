# SPDX-License-Identifier: 0BSD
"""Controller and directed reducer checks with no numerical evaluator session."""

from __future__ import annotations

import copy
import json
from decimal import ROUND_UP, Decimal, localcontext
from fractions import Fraction
from types import SimpleNamespace

import pytest

from pyamplicol.api.errors import ArtifactError, CompatibilityError, EvaluationError
from pyamplicol.api.results import HelicityConfiguration
from pyamplicol.api.services import Runtime
from pyamplicol.color.connections import COLOR_CONNECTION_CONVENTION
from pyamplicol.correlators import (
    ColorCorrelator,
    CorrelatedRequest,
    CorrelatedValue,
    CorrelatorConfig,
)
from pyamplicol.runtime import correlations
from pyamplicol.runtime.correlated_exact import (
    CoherentAmplitudeBatch,
    CoherentAmplitudeGroup,
    _prepare_spin_vectors,
)
from pyamplicol.runtime.correlations import CorrelatorEvaluator, _contract, _matrix


def _matrix_record(identifier="born", weights=None):
    weights = weights or ((-16, -16), (20, 2))
    return {
        "id": identifier,
        "convention": COLOR_CONNECTION_CONVENTION,
        "color_accuracy": "full",
        "storage": "sparse-directed",
        "includes_color_factor": True,
        "sector_ids": [0, 1],
        "entries": [
            {
                "left_sector_id": left,
                "right_sector_id": right,
                "weight": {"real": [value, 3], "imag": [0, 1]},
            }
            for left, row in enumerate(weights)
            for right, value in enumerate(row)
        ],
    }


def _group(helicity, sector, index):
    return CoherentAmplitudeGroup(
        index, f"h:{helicity}", (helicity,), (sector,), sector
    )


def _amplitude(value):
    if isinstance(value, complex):
        return Decimal(str(value.real)), Decimal(str(value.imag))
    return Decimal(str(value)), Decimal(0)


def _batch(points, *, helicities=(1,), normalization="1"):
    return CoherentAmplitudeBatch(
        tuple(tuple(_amplitude(value) for value in point) for point in points),
        tuple(
            _group(helicity, sector, index * 2 + sector)
            for index, helicity in enumerate(helicities)
            for sector in (0, 1)
        ),
        Decimal(normalization),
        60,
        (),
    )


def test_directed_two_by_two_matrix_preserves_complex_interference():
    matrix = _matrix(_matrix_record())
    result = _contract(_batch([(1, 1j)], normalization="1.5"), matrix, precision=40)
    # conj((1,i)) K (1,i) = -14/3 - 12i, with one factor of 3/2.
    assert result == (CorrelatedValue(Decimal(-7), Decimal(-18)),)


def test_spectator_helicities_are_summed_incoherently():
    matrix = _matrix(_matrix_record())
    result = _contract(
        _batch([(1, 1j, 2, -1j)], helicities=(-1, 1), normalization="1.5"),
        matrix,
        precision=40,
    )
    assert result == (CorrelatedValue(Decimal(-38), Decimal(18)),)
    # Coherently adding helicities would give (-72, 0), a different observable.


def test_batched_and_single_point_contractions_agree():
    points = [(1, 1j), (2, 2j), (3 - 1j, 2 + 4j)]
    matrix = _matrix(_matrix_record())
    batched = _contract(_batch(points, normalization="1.5"), matrix, precision=50)
    single = tuple(
        _contract(_batch([point], normalization="1.5"), matrix, precision=50)[0]
        for point in points
    )
    assert batched == single
    assert batched[1] == CorrelatedValue(Decimal(-28), Decimal(-72))


def test_exact_rationals_and_rounding_ignore_ambient_decimal_context():
    record = _matrix_record()
    record["entries"] = [
        {
            "left_sector_id": 0,
            "right_sector_id": 1,
            "weight": {"real": [1, 3], "imag": [2, 7]},
        }
    ]
    with localcontext() as context:
        context.prec = 4
        context.rounding = ROUND_UP
        result = _contract(_batch([(1, 1)]), _matrix(record), precision=40)[0]
    assert result.real == Decimal("0." + "3" * 40)
    assert result.imag == Decimal("0.2857142857142857142857142857142857142857")
    assert _matrix(record).entries[0].real == Fraction(1, 3)


@pytest.mark.parametrize(
    "defect",
    [
        "float",
        "bool",
        "string",
        "zero_denominator",
        "bool_sector",
        "duplicate",
        "outside",
        "empty",
        "convention",
    ],
)
def test_matrix_catalogue_rejects_malformed_exact_data(defect):
    record = _matrix_record()
    if defect in {"float", "bool", "string"}:
        record["entries"][0]["weight"]["real"][0] = {
            "float": 1.5,
            "bool": True,
            "string": "16.0",
        }[defect]
    elif defect == "zero_denominator":
        record["entries"][0]["weight"]["imag"][1] = 0
    elif defect == "bool_sector":
        record["entries"][0]["left_sector_id"] = False
    elif defect == "duplicate":
        record["entries"].append(copy.deepcopy(record["entries"][0]))
    elif defect == "outside":
        record["entries"][0]["right_sector_id"] = 7
    elif defect == "empty":
        record["sector_ids"] = []
    elif defect == "convention":
        record["storage"] = "upper-triangle"
    with pytest.raises(ArtifactError, match="matrix"):
        _matrix(record)


def test_matrix_reads_generation_integer_string_coefficients_exactly():
    record = _matrix_record()
    record["entries"][0]["weight"]["real"] = [10**60 + 7, 10**45 + 3]
    expected = _matrix(record)
    for entry in record["entries"]:
        for component in ("real", "imag"):
            entry["weight"][component] = [
                str(value) for value in entry["weight"][component]
            ]
    assert _matrix(record) == expected


@pytest.mark.parametrize("defect", ["missing_sector", "duplicate_group", "point_shape"])
def test_reducer_checks_amplitude_matrix_shape(defect):
    batch = _batch([(1, 1j)])
    if defect == "missing_sector":
        batch = CoherentAmplitudeBatch(
            ((batch.values[0][0],),), batch.groups[:1], Decimal(1), 40, ()
        )
    elif defect == "duplicate_group":
        batch = CoherentAmplitudeBatch(
            batch.values, (batch.groups[0], batch.groups[0]), Decimal(1), 40, ()
        )
    else:
        batch = CoherentAmplitudeBatch(
            ((batch.values[0][0],),), batch.groups, Decimal(1), 40, ()
        )
    with pytest.raises(ArtifactError):
        _contract(batch, _matrix(_matrix_record()), precision=40)


class _Executor:
    def __init__(
        self, artifact, process_id, native, *, spin_correlated_legs, coherent_groups
    ):
        self.legs = spin_correlated_legs
        self.groups = coherent_groups
        self.calls = []
        self.many_calls = []

    def coherent_amplitudes(self, momenta, **kwargs):
        self.calls.append((momenta, copy.deepcopy(kwargs)))
        return _batch(
            [(point[0][0], complex(0, point[0][0])) for point in momenta],
            normalization="1.5",
        )

    def iter_coherent_amplitudes_many(self, momenta, **kwargs):
        self.many_calls.append((momenta, copy.deepcopy(kwargs)))
        assignments = tuple(
            _prepare_spin_vectors(
                vectors, allowed_legs=self.legs, point_count=len(momenta)
            )
            for vectors in kwargs["spin_vector_sets"]
        )
        for point_index, point in enumerate(momenta):
            for assignment, vectors in enumerate(assignments):
                vector = vectors[point_index].get(1, ((Decimal(0), Decimal(0)),) * 4)
                energy = Decimal(str(point[0][0]))
                x, y = energy + vector[0][0], energy + vector[1][0]
                batch = _batch([(0, 0)], normalization="1.5")
                yield (
                    point_index,
                    assignment,
                    CoherentAmplitudeBatch(
                        (((x, Decimal(0)), (Decimal(0), y)),),
                        batch.groups,
                        batch.normalization_factor,
                        60,
                        tuple(sorted(vectors[point_index])),
                    ),
                )


@pytest.fixture
def controller_factory(monkeypatch, tmp_path):
    declarations = CorrelatorConfig(
        color_correlations=(ColorCorrelator.dipole("ordered", 1, 2),),
        spin_correlations=((1,), (1, 2)),
    )
    payload = {
        "schema_version": 1,
        "complete_source_basis": True,
        "declarations": declarations.to_json_dict(),
        "processes": {
            "p": {
                "declarations": declarations.to_json_dict(),
                "spin_legs": [1, 2],
                "matrices": [_matrix_record("born"), _matrix_record("ordered")],
                "coherent_groups": [{"group_id": 0}],
            }
        },
    }
    extension = {"schema_version": 1, "path": "correlators.json"}
    manifest = SimpleNamespace(extensions={"correlators": extension}, processes=[])
    reads = []

    def read_manifest(path):
        reads.append("manifest")
        return manifest

    def read_sidecar():
        reads.append("sidecar")
        return json.dumps(payload)

    monkeypatch.setattr(correlations, "load_manifest", read_manifest)
    monkeypatch.setattr(
        correlations,
        "confined_path",
        lambda artifact, path: SimpleNamespace(read_text=read_sidecar),
    )
    monkeypatch.setattr(
        correlations,
        "native_process_selection",
        lambda runtime, processes: SimpleNamespace(
            representative_process_id="p",
            external_permutation=(0, 1),
            process_group=None,
        ),
    )
    monkeypatch.setattr(correlations, "CorrelatedExactExecutor", _Executor)
    backend = SimpleNamespace(_artifact_path=tmp_path, _runtime=object())
    return SimpleNamespace(
        create=lambda: CorrelatorEvaluator(backend),
        payload=payload,
        manifest=manifest,
        backend=backend,
        reads=reads,
    )


def test_public_catalogue_is_typed_lazy_and_reused_for_evaluation(
    controller_factory, monkeypatch
):
    factory = controller_factory
    runtime = object.__new__(Runtime)
    runtime._backend = factory.backend
    constructed = []
    validated = []
    matrix_reader = correlations._matrix

    def executor(*args, **kwargs):
        constructed.append(True)
        return _Executor(*args, **kwargs)

    def matrix(record):
        validated.append(record["id"])
        return matrix_reader(record)

    monkeypatch.setattr(correlations, "CorrelatedExactExecutor", executor)
    monkeypatch.setattr(correlations, "_matrix", matrix)
    expected = (ColorCorrelator("born"), ColorCorrelator.dipole("ordered", 1, 2))
    available = runtime.available_color_correlations()
    assert available == expected
    assert runtime.available_color_correlations() == expected
    assert all(isinstance(item, ColorCorrelator) for item in available)
    assert constructed == []
    assert not hasattr(runtime, "_correlated_evaluator")
    assert factory.reads == ["manifest", "sidecar"]
    assert validated == ["born", "ordered"]

    # Evaluating after discovery consumes the same validated metadata.
    runtime.evaluate_correlated([[(1, 0, 0, 1)]], color_correlation="ordered")
    assert constructed == [True]
    assert factory.reads == ["manifest", "sidecar"]
    assert validated == ["born", "ordered"]
    assert runtime.available_color_correlations() == expected


def test_public_catalogue_uses_selected_process_not_global_declarations(
    controller_factory, monkeypatch
):
    factory = controller_factory
    factory.payload["declarations"] = {"all_color_through_order": 3}
    process = copy.deepcopy(factory.payload["processes"]["p"])
    declarations = CorrelatorConfig((ColorCorrelator.dipole("selected", 2, 1),))
    process.update(
        declarations=declarations.to_json_dict(),
        spin_legs=[],
        matrices=[_matrix_record("born"), _matrix_record("selected")],
    )
    factory.payload["processes"]["q"] = process
    monkeypatch.setattr(
        correlations,
        "native_process_selection",
        lambda native, processes: SimpleNamespace(
            representative_process_id="q",
            external_permutation=(0, 1),
            process_group=None,
        ),
    )
    runtime = object.__new__(Runtime)
    runtime._backend = factory.backend
    assert runtime.available_color_correlations() == declarations.color_requests
    assert not hasattr(runtime, "_correlated_evaluator")


@pytest.mark.parametrize("defect", ["missing", "unresolved", "matrix_ids"])
def test_public_catalogue_requires_resolved_consistent_process_declarations(
    controller_factory, defect
):
    process = controller_factory.payload["processes"]["p"]
    if defect == "missing":
        del process["declarations"]
    elif defect == "unresolved":
        process["declarations"]["all_color_through_order"] = 1
    else:
        process["declarations"]["color_correlations"] = []
    runtime = object.__new__(Runtime)
    runtime._backend = controller_factory.backend
    with pytest.raises(ArtifactError):
        runtime.available_color_correlations()
    assert not hasattr(runtime, "_correlated_catalogue")
    assert not hasattr(runtime, "_correlated_evaluator")


def test_public_catalogue_requires_generation_opt_in(controller_factory):
    controller_factory.manifest.extensions = {}
    runtime = object.__new__(Runtime)
    runtime._backend = controller_factory.backend
    with pytest.raises(CompatibilityError, match="declared at generation"):
        runtime.available_color_correlations()


@pytest.mark.parametrize("operation", ["list", "evaluate"])
def test_public_catalogue_rejects_permuted_alias_before_executor(
    controller_factory, monkeypatch, operation
):
    monkeypatch.setattr(
        correlations,
        "native_process_selection",
        lambda native, processes: SimpleNamespace(
            representative_process_id="p",
            external_permutation=(1, 0),
            process_group=None,
        ),
    )

    def unexpected_executor(*args, **kwargs):
        raise AssertionError("alias rejection must precede amplitude loading")

    monkeypatch.setattr(correlations, "CorrelatedExactExecutor", unexpected_executor)
    runtime = object.__new__(Runtime)
    runtime._backend = controller_factory.backend
    with pytest.raises(CompatibilityError, match="non-identity process aliases"):
        if operation == "list":
            runtime.available_color_correlations()
        else:
            runtime.evaluate_correlated([[(1, 0, 0, 1)]])
    assert not hasattr(runtime, "_correlated_catalogue")
    assert not hasattr(runtime, "_correlated_evaluator")


def test_vectors_instance_isolation_copy_failed_setter_atomic_and_reset(
    controller_factory,
):
    first, second = controller_factory.create(), controller_factory.create()
    assert (
        first._executor.groups
        == controller_factory.payload["processes"]["p"]["coherent_groups"]
    )
    vector = [1, 2, 3, 4]
    first.set_spin_correlation_vectors({1: vector})
    vector[0] = 999
    assert first._vectors == {1: (1, 2, 3, 4)}
    with pytest.raises(EvaluationError, match="at least one"):
        first.set_spin_correlation_vectors({1: []})
    assert first._vectors == {1: (1, 2, 3, 4)}
    assert second._vectors == {}
    with pytest.raises(EvaluationError, match="finite"):
        first.set_spin_correlation_vectors(
            {1: (5, 6, 7, 8), 2: (float("nan"), 1, 2, 3)}
        )
    assert first._vectors == {1: (1, 2, 3, 4)}
    with pytest.raises(EvaluationError, match="class"):
        first.set_spin_correlation_vectors({2: (1, 2, 3, 4)})
    assert first._vectors == {1: (1, 2, 3, 4)}
    first.set_spin_correlation_vectors(None)
    assert first._vectors == {}
    first.set_spin_correlation_vectors({1: (1, 2, 3, 4)})
    first.set_spin_correlation_vectors({})
    assert first._vectors == {}


def test_point_vectors_copied_and_controller_forwards_requests(controller_factory):
    evaluator = controller_factory.create()
    vectors = [[1, 2, 3, 4], [5, 6, 7, 8]]
    evaluator.set_spin_correlation_vectors({1: vectors})
    vectors[1][0] = 99
    momenta = [[(1, 0, 0, 1)], [(2, 0, 0, 2)]]
    result = evaluator.evaluate(
        momenta, color_correlation="ordered", helicities=["h:1"], precision=50
    )
    assert result == (
        CorrelatedValue(Decimal(-7), Decimal(-18)),
        CorrelatedValue(Decimal(-28), Decimal(-72)),
    )
    received = evaluator._executor.calls[0][1]
    assert received["spin_vectors"] == {1: ((1, 2, 3, 4), (5, 6, 7, 8))}
    assert received["helicities"] == ["h:1"]
    assert received["precision"] > 50
    with pytest.raises(EvaluationError, match="unknown"):
        evaluator.evaluate(momenta, color_correlation="absent")
    for precision in (True, 0, "40"):
        with pytest.raises(EvaluationError, match="precision"):
            evaluator.evaluate(momenta, precision=precision)


def test_explicit_generation_opt_in_required(controller_factory):
    controller_factory.manifest.extensions = {}
    with pytest.raises(CompatibilityError, match="declared at generation"):
        controller_factory.create()
    with pytest.raises(CompatibilityError, match="does not support"):
        CorrelatorEvaluator(object())


def test_many_keeps_distinct_request_and_point_axes(controller_factory):
    evaluator = controller_factory.create()
    requests = {
        "spin-a": CorrelatedRequest("ordered", {1: ((1, 0, 0, 0), (2, 1, 0, 0))}),
        "spin-b": CorrelatedRequest("ordered", {1: (4, 2, 0, 0)}),
    }
    result = evaluator.evaluate_many(
        [[(1, 0, 0, 1)], [(3, 0, 0, 3)]], requests, precision=40
    )
    assert tuple(result) == ("spin-a", "spin-b")
    assert result == {
        "spin-a": (
            CorrelatedValue(Decimal(-31), Decimal(-36)),
            CorrelatedValue(Decimal(-184), Decimal(-360)),
        ),
        "spin-b": (
            CorrelatedValue(Decimal(-191), Decimal(-270)),
            CorrelatedValue(Decimal(-367), Decimal(-630)),
        ),
    }
    assert {label: values[1] for label, values in result.items()} == {
        "spin-a": CorrelatedValue(Decimal(-184), Decimal(-360)),
        "spin-b": CorrelatedValue(Decimal(-367), Decimal(-630)),
    }
    assert len(evaluator._executor.many_calls[0][1]["spin_vector_sets"]) == 2


def test_many_groups_inherited_explicit_and_broadcast_vectors(
    controller_factory, monkeypatch
):
    evaluator = controller_factory.create()
    evaluator.set_spin_correlation_vectors({1: (1, 2, 3, 4)})
    requests = {
        "inherited": CorrelatedRequest("born"),
        "another-colour": CorrelatedRequest("ordered", {1: ((1, 2, 3, 4),) * 2}),
        "same-again": CorrelatedRequest("born", {1: (1, 2, 3, 4)}),
        "physical": CorrelatedRequest("born", {}),
    }
    prepared = []
    original = correlations._prepare_contraction

    def count_plan(*args, **kwargs):
        prepared.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(correlations, "_prepare_contraction", count_plan)
    result = evaluator.evaluate_many([[(1, 0, 0, 1)], [(2, 0, 0, 2)]], requests)
    assert result["inherited"] == result["another-colour"] == result["same-again"]
    assert result["physical"] != result["inherited"]
    assert len(evaluator._executor.many_calls) == 1
    assert len(evaluator._executor.many_calls[0][1]["spin_vector_sets"]) == 2
    assert len(prepared) == 3  # Two colour IDs for one spin state; one for physical.
    assert evaluator._vectors == {1: (1, 2, 3, 4)}


def test_many_scalar_batch_equivalence_and_exact_vector_grouping(controller_factory):
    evaluator = controller_factory.create()
    tiny = Decimal("1e-60")
    requests = {
        "one": CorrelatedRequest("born", {1: (tiny, 2, 3, 4)}),
        "two": CorrelatedRequest("born", {1: (2 * tiny, 2, 3, 4)}),
    }
    points = [[(1, 0, 0, 1)], [(2, 0, 0, 2)]]
    with localcontext() as context:
        context.prec = 90
        batch = evaluator.evaluate_many(points, requests, precision=70)
        scalar = [
            evaluator.evaluate_many([point], requests, precision=70) for point in points
        ]
    assert batch == {
        label: tuple(point[label][0] for point in scalar) for label in requests
    }
    assert batch["one"] != batch["two"]
    assert len(evaluator._executor.many_calls[0][1]["spin_vector_sets"]) == 2


def test_decimal_setter_and_grouped_broadcast_preserve_sub_binary64_differences(
    controller_factory,
):
    runtime = object.__new__(Runtime)
    runtime._backend = controller_factory.backend
    first = Decimal("1.000000000000000000000000000000000000000000000000000000000001")
    second = Decimal("1.000000000000000000000000000000000000000000000000000000000002")
    vector = [first, Decimal(0), Decimal(0), Decimal(0)]
    points = [[(1, 0, 0, 1)], [(2, 0, 0, 2)]]
    runtime.set_spin_correlation_vectors({1: vector})
    vector[0] = second
    runtime.evaluate_correlated(points, precision=80)
    evaluator = runtime._correlated_evaluator
    assert evaluator._vectors[1][0] is first

    assert evaluator._executor.calls[0][1]["spin_vectors"][1][0] == first
    runtime.evaluate_correlated_many(
        points,
        {
            "inherited": CorrelatedRequest(),
            "same-broadcast": CorrelatedRequest(spin_vectors={1: (first, 0, 0, 0)}),
            "same-batch": CorrelatedRequest(spin_vectors={1: ((first, 0, 0, 0),) * 2}),
            "different-second-point": CorrelatedRequest(
                spin_vectors={1: ((first, 0, 0, 0), (second, 0, 0, 0))}
            ),
        },
        precision=80,
    )
    assignments = evaluator._executor.many_calls[0][1]["spin_vector_sets"]
    assert len(assignments) == 2
    assert assignments[0][1][0] == first
    assert assignments[1][1][0][0] == first
    assert assignments[1][1][1][0] == second
    assert evaluator._vectors[1][0] is first


@pytest.mark.parametrize("point_count", (2, 4))
def test_complex_decimal_pairs_are_copied_and_not_confused_with_point_batches(
    controller_factory, point_count
):
    runtime = object.__new__(Runtime)
    runtime._backend = controller_factory.backend
    real = Decimal("1.000000000000000000000000000000001")
    imaginary = Decimal("0.000000000000000000000000000000001")
    component = [real, imaginary]
    runtime.set_spin_correlation_vectors({1: [component, 0, 0, 0]})
    component[0] = Decimal(9)
    points = [[(1, 0, 0, 1)]] * point_count
    result = runtime.evaluate_correlated_many(
        points,
        {
            "inherited": CorrelatedRequest(),
            "batch": CorrelatedRequest(
                spin_vectors={1: (((real, imaginary), 0, 0, 0),) * point_count}
            ),
        },
        precision=60,
    )
    assert result["batch"] == result["inherited"]
    evaluator = runtime._correlated_evaluator
    assert evaluator._vectors[1][0] == (real, imaginary)
    assert len(evaluator._executor.many_calls[0][1]["spin_vector_sets"]) == 1


@pytest.mark.parametrize("defect", ("id", "class", "shape", "label", "type"))
def test_many_validates_all_requests_before_evaluation_without_setter_mutation(
    controller_factory, defect
):
    evaluator = controller_factory.create()
    evaluator.set_spin_correlation_vectors({1: (1, 2, 3, 4)})
    requests = {"good": CorrelatedRequest(), "bad": CorrelatedRequest()}
    if defect == "id":
        requests["bad"] = CorrelatedRequest("missing")
    elif defect == "class":
        requests["bad"] = CorrelatedRequest(spin_vectors={2: (1, 2, 3, 4)})
    elif defect == "shape":
        requests["bad"] = CorrelatedRequest(spin_vectors={1: ((1, 2, 3, 4),)})
    elif defect == "label":
        requests[None] = requests.pop("bad")
    else:
        requests["bad"] = {}
    with pytest.raises(EvaluationError):
        evaluator.evaluate_many([[(1, 0, 0, 1)], [(2, 0, 0, 2)]], requests)
    assert evaluator._executor.many_calls == []
    assert evaluator._vectors == {1: (1, 2, 3, 4)}


def test_many_empty_requests_and_signed_zero_assignments(controller_factory):
    evaluator = controller_factory.create()
    assert evaluator.evaluate_many([[(1, 0, 0, 1)]], {}) == {}
    assert evaluator._executor.many_calls == []
    evaluator.evaluate_many(
        [[(1, 0, 0, 1)]],
        {
            "positive": CorrelatedRequest(spin_vectors={1: (Decimal("0"), 0, 0, 0)}),
            "negative": CorrelatedRequest(spin_vectors={1: (Decimal("-0"), 0, 0, 0)}),
        },
    )
    assert len(evaluator._executor.many_calls[0][1]["spin_vector_sets"]) == 2


def test_many_public_typed_selectors_and_result_shape(controller_factory):
    runtime = object.__new__(Runtime)
    runtime._backend = controller_factory.backend
    result = runtime.evaluate_correlated_many(
        [[(1, 0, 0, 1)], [(2, 0, 0, 2)]],
        {"dipole": CorrelatedRequest("ordered")},
        helicities=[HelicityConfiguration("h:1", 0, (1,), True, False, "h:1", 1)],
        precision=50,
    )
    assert result["dipole"] == (
        CorrelatedValue(Decimal(-7), Decimal(-18)),
        CorrelatedValue(Decimal(-28), Decimal(-72)),
    )
    assert runtime._correlated_evaluator._executor.many_calls[0][1]["helicities"] == (
        "h:1",
    )


@pytest.mark.parametrize("accuracy", ("lc", "nlc", "full"))
def test_matrix_reader_accepts_declared_colour_accuracy(accuracy):
    record = _matrix_record()
    record["color_accuracy"] = accuracy
    assert _matrix(record).id == "born"


@pytest.mark.parametrize(
    "defect",
    [
        "extension_version",
        "payload_version",
        "basis",
        "legs",
        "bool_leg",
        "missing_id",
        "duplicate_id",
        "missing_process",
    ],
)
def test_catalogue_validation(controller_factory, defect):
    factory = controller_factory
    process = factory.payload["processes"]["p"]
    if defect == "extension_version":
        factory.manifest.extensions["correlators"]["schema_version"] = True
    elif defect == "payload_version":
        factory.payload["schema_version"] = True
    elif defect == "basis":
        factory.payload["complete_source_basis"] = False
    elif defect == "legs":
        process["spin_legs"] = [1]
    elif defect == "bool_leg":
        process["spin_legs"] = [True, 2]
    elif defect == "missing_id":
        process["matrices"].pop()
    elif defect == "duplicate_id":
        process["matrices"][1]["id"] = "born"
    elif defect == "missing_process":
        factory.payload["processes"] = {}
    with pytest.raises(ArtifactError, match="catalogue"):
        factory.create()


def test_public_runtime_forwards_typed_helicity_id_and_validates_precision():
    calls = []
    recorder = SimpleNamespace(
        evaluate=lambda *args, **kwargs: calls.append((args, kwargs)) or (),
        set_spin_correlation_vectors=lambda value: calls.append(value),
    )
    runtime = object.__new__(Runtime)
    runtime._correlated_evaluator = recorder
    helicity = HelicityConfiguration("h:1", 0, (1,), True, False, "h:1", 1)
    assert (
        runtime.evaluate_correlated(
            [[(1, 0, 0, 1)]], helicities=[helicity, "h:-1"], precision=40
        )
        == ()
    )
    assert calls[-1][1]["helicities"] == ("h:1", "h:-1")
    assert calls[-1][1]["precision"] == 40
    runtime.set_spin_correlation_vectors(None)
    assert calls[-1] is None
    with pytest.raises((ValueError, TypeError)):
        runtime.evaluate_correlated([[(1, 0, 0, 1)]], precision=True)

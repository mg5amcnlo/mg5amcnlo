# SPDX-License-Identifier: 0BSD
"""Generated tree amplitudes: Ward identities, Born recovery and colour coherence."""

from decimal import Decimal, localcontext

import pytest

from pyamplicol import (
    ColorCorrelator,
    CorrelatedRequest,
    CorrelatorConfig,
    EmitGluon,
    Generator,
    ModelSource,
    Runtime,
    SplitGluon,
)
from pyamplicol.config import (
    ColorConfig,
    EvaluatorConfig,
    EvaluatorOptimizationConfig,
    GenerationConfig,
    GenerationValidationConfig,
    JITConfig,
    RunConfig,
)

POINT = (
    (500, 0, 0, 500),
    (500, 0, 0, -500),
    (500, 300, 0, 400),
    (500, -300, 0, -400),
)
THREE_BODY_POINT = (
    (900, 0, 0, 900),
    (900, 0, 0, -900),
    (500, 400, 300, 0),
    (500, 400, -300, 0),
    (800, -800, 0, 0),
)


def _configuration():
    return RunConfig(
        action="generate",
        color=ColorConfig(accuracy="full"),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
            validation=GenerationValidationConfig(
                enabled=False, post_build_validation=False
            ),
        ),
        evaluator=EvaluatorConfig(
            execution_mode="compiled",
            optimization=EvaluatorOptimizationConfig(cores=1, horner_iterations=1),
            jit=JITConfig(optimization_level=0),
        ),
    )


@pytest.fixture(scope="module")
def correlated_gg(tmp_path_factory):
    pytest.importorskip("pyamplicol._rusticol")
    output = tmp_path_factory.mktemp("correlated-gg") / "process"
    # Three successive physical colour charges on one adjoint line give C_A^3.
    triple = tuple(EmitGluon(1, -i) for i in range(1, 4))
    declarations = CorrelatorConfig(
        color_correlations=(
            *(ColorCorrelator.dipole(f"T1{leg}", 1, leg) for leg in range(1, 5)),
            ColorCorrelator("triple-self", triple, triple),
        ),
        spin_correlations=((1,), (3,), (1, 3)),
    )
    Generator(_configuration()).generate(
        "g g > g g", output, model=ModelSource.built_in_sm(), correlators=declarations
    )
    return output


def _value(runtime, **kwargs):
    return runtime.evaluate_correlated((POINT,), precision=32, **kwargs)[0]


def test_generated_automatic_nnlo_catalogue_all_operators_in_one_point_batch(tmp_path):
    pytest.importorskip("pyamplicol._rusticol")
    output = tmp_path / "automatic-pair"
    manual = ColorCorrelator.dipole("manual-T34", 3, 4)
    Generator(_configuration()).generate(
        "e+ e- > d d~",
        output,
        model=ModelSource.built_in_sm(),
        correlators=CorrelatorConfig(
            color_correlations=(manual,), all_color_through_order=2
        ),
    )
    runtime = Runtime.load(output)
    catalogue = runtime.available_color_correlations()
    # Two coloured hard legs: 4 NLO + 108 NNLO, identity, and one manual alias.
    assert len(catalogue) == 114
    assert sum(item.order == 2 for item in catalogue) == 108
    assert not hasattr(runtime, "_correlated_evaluator")
    automatic = {
        (item.bra, item.ket): item for item in catalogue if item.id.startswith("N")
    }
    alias = automatic[(manual.bra, manual.ket)]
    points = (
        POINT,
        (POINT[0], POINT[1], (500, 400, 0, 300), (500, -400, 0, -300)),
    )
    requests = {item.id: CorrelatedRequest(item.id, {}) for item in catalogue}
    values = runtime.evaluate_correlated_many(points, requests, precision=40)
    assert tuple(values) == tuple(requests)
    assert all(len(batch) == 2 for batch in values.values())
    assert values[manual.id] == values[alias.id]
    assert values["born"][0] != values["born"][1]
    ordinary = runtime.evaluate(points, precision=40)
    pair = next(
        item
        for item in catalogue
        if item.order == 2
        and item.bra == item.ket
        and isinstance(item.bra[0], EmitGluon)
        and item.bra[0].emitter_label == 3
        and isinstance(item.bra[1], SplitGluon)
    )
    with localcontext() as context:
        context.prec = 50
        for point_index, born in enumerate(values["born"]):
            tolerance = abs(born.real) * Decimal("1e-35")
            assert born.real > 0
            assert abs(born.real - ordinary[point_index]) < tolerance
            # A quark emitter followed by g→q qbar gives C_F T_R = 2/3,
            # with neither an additional flavour sum nor extra kinematics.
            assert (
                abs(values[pair.id][point_index].real - born.real * 2 / 3) < tolerance
            )
            assert (
                abs(values[manual.id][point_index].real + born.real * 4 / 3) < tolerance
            )
            for (bra, ket), item in automatic.items():
                partner = automatic[(ket, bra)]
                left, right = (
                    values[item.id][point_index],
                    values[partner.id][point_index],
                )
                assert abs(left.real - right.real) < tolerance
                assert abs(left.imag + right.imag) < tolerance


def test_generated_many_colour_spin_and_point_axes_match_separate_calls(correlated_gg):
    runtime = Runtime.load(correlated_gg)
    runtime.set_spin_correlation_vectors({1: (0, 1, 0, 0)})
    points = (
        POINT,
        (POINT[0], POINT[1], (500, 400, 0, 300), (500, -400, 0, -300)),
    )
    vectors = {3: ((0, 0, 1, 0), (0, 0, 2, 0))}
    requests = {
        "born": CorrelatedRequest("born", {}),
        **{f"dipole1{leg}": CorrelatedRequest(f"T1{leg}", {}) for leg in range(1, 5)},
        "spin-born": CorrelatedRequest("born", vectors),
        "spin-dipole13": CorrelatedRequest("T13", vectors),
        "inherited-spin": CorrelatedRequest("born"),
    }
    combined = runtime.evaluate_correlated_many(points, requests, precision=40)
    reference = Runtime.load(correlated_gg)
    for label, request in requests.items():
        reference.set_spin_correlation_vectors(
            {1: (0, 1, 0, 0)} if request.spin_vectors is None else request.spin_vectors
        )
        expected = reference.evaluate_correlated(
            points, color_correlation=request.color_correlation, precision=40
        )
        assert combined[label] == expected
        assert len(combined[label]) == 2
    assert tuple(combined) == tuple(requests)
    assert combined["spin-dipole13"][0] != combined["spin-dipole13"][1]
    assert (
        runtime.evaluate_correlated(points, precision=40) == combined["inherited-spin"]
    )


def test_generated_arb_preserves_decimal_spin_vectors_in_single_and_many_calls(
    correlated_gg,
):
    runtime = Runtime.load(correlated_gg)
    first = Decimal("1.00000000000000000000000000000000000000000000000001")
    second = Decimal("1.00000000000000000000000000000000000000000000000002")
    zero = Decimal(0)
    vector = [zero, zero, first, zero]
    points = (
        POINT,
        (POINT[0], POINT[1], (500, 400, 0, 300), (500, -400, 0, -300)),
    )
    runtime.set_spin_correlation_vectors({3: vector})
    vector[2] = second  # The setter owns a copy, without rounding its Decimals.
    single = runtime.evaluate_correlated(points, precision=100)
    combined = runtime.evaluate_correlated_many(
        points,
        {
            "inherited": CorrelatedRequest(),
            "same-broadcast": CorrelatedRequest(
                spin_vectors={3: (zero, zero, first, zero)}
            ),
            "per-point": CorrelatedRequest(
                spin_vectors={
                    3: ((zero, zero, first, zero), (zero, zero, second, zero))
                }
            ),
            "unit": CorrelatedRequest(spin_vectors={3: (zero, zero, Decimal(1), zero)}),
        },
        precision=100,
    )
    assert single == combined["inherited"] == combined["same-broadcast"]
    assert combined["per-point"][0] == single[0]
    assert combined["per-point"][1] != single[1]
    with localcontext() as context:
        context.prec = 110
        for index, scale in enumerate((first, second)):
            unit = combined["unit"][index].real
            assert unit > 0
            assert abs(single[index].real / unit - first**2) < Decimal("1e-90")
            assert abs(combined["per-point"][index].real / unit - scale**2) < Decimal(
                "1e-90"
            )
            assert single[index].real != unit


def test_generated_double_double_decimal_spin_and_ward_identities(correlated_gg):
    runtime = Runtime.load(correlated_gg)
    first = Decimal("1.0000000000000000000000001")
    second = Decimal("1.0000000000000000000000002")
    zero = Decimal(0)
    points = (POINT, POINT)
    runtime.set_spin_correlation_vectors({3: (zero, zero, first, zero)})
    before = runtime.evaluate_correlated(points, precision=70)
    single = runtime.evaluate_correlated(
        points, arithmetic="double-double", precision=31
    )
    grouped = runtime.evaluate_correlated_many(
        points,
        {
            "inherited": CorrelatedRequest(),
            "unit": CorrelatedRequest(spin_vectors={3: (0, 0, 1, 0)}),
            "pointwise": CorrelatedRequest(
                spin_vectors={3: ((0, 0, first, 0), (0, 0, second, 0))}
            ),
            "complex": CorrelatedRequest(spin_vectors={3: (0, 0, (zero, first), 0)}),
            "ward": CorrelatedRequest(
                spin_vectors={3: tuple(Decimal(value) for value in POINT[2])}
            ),
        },
        arithmetic="double-double",
        precision=31,
    )
    assert single == grouped["inherited"] == grouped["complex"]
    assert grouped["pointwise"][0] == single[0]
    assert grouped["pointwise"][1] != single[1]
    with localcontext() as context:
        context.prec = 80
        for index, scale in enumerate((first, second)):
            unit = grouped["unit"][index].real
            assert abs(single[index].real / unit - first**2) < Decimal("1e-28")
            assert abs(grouped["pointwise"][index].real / unit - scale**2) < Decimal(
                "1e-28"
            )
            assert abs(grouped["ward"][index].real / unit) < Decimal("1e-48")
            assert abs(single[index].real / before[index].real - 1) < Decimal("1e-28")
    assert runtime.evaluate_correlated(points, precision=70) == before


def test_generated_born_recovery_and_n3lo_colour_coherence(correlated_gg):
    runtime = Runtime.load(correlated_gg)
    born = _value(runtime)
    ordinary = runtime.evaluate((POINT,), precision=40)[0]
    with localcontext() as context:
        context.prec = 40
        assert ordinary > 0
        assert abs(born.real / ordinary - 1) < Decimal("1e-28")
        assert abs(born.imag / ordinary) < Decimal("1e-28")
        dipoles = [_value(runtime, color_correlation=f"T1{leg}") for leg in range(1, 5)]
        assert abs(dipoles[0].real / born.real - 3) < Decimal("1e-28")
        assert abs(sum(item.real for item in dipoles) / born.real) < Decimal("1e-28")
        assert abs(sum(item.imag for item in dipoles) / born.real) < Decimal("1e-28")
        triple = _value(runtime, color_correlation="triple-self")
        assert abs(triple.real / born.real - 27) < Decimal("1e-28")


@pytest.mark.parametrize("legs", ((1,), (3,), (1, 3)))
def test_generated_single_and_joint_ward_identities(correlated_gg, legs):
    runtime = Runtime.load(correlated_gg)
    born = _value(runtime).real
    runtime.set_spin_correlation_vectors({leg: POINT[leg - 1] for leg in legs})
    for identifier in ("born", "T13"):
        ward = _value(runtime, color_correlation=identifier)
        # Compensate the dimensionful magnitude of each substituted momentum.
        scale = born * Decimal(500) ** (2 * len(legs))
        assert abs(ward.real / scale) < Decimal("1e-28")
        assert abs(ward.imag / scale) < Decimal("1e-28")
    runtime.set_spin_correlation_vectors(None)
    assert _value(runtime).real == born


def test_generated_polarization_completeness_and_vector_homogeneity(correlated_gg):
    runtime = Runtime.load(correlated_gg)
    born = _value(runtime).real
    # For outgoing p=(500,300,0,400), these form a real transverse basis.
    polarizations = ((0, Decimal("0.8"), 0, Decimal("-0.6")), (0, 0, 1, 0))
    with localcontext() as context:
        context.prec = 40
        projections = []
        for vector in polarizations:
            runtime.set_spin_correlation_vectors({3: vector})
            projections.append(_value(runtime).real)
        assert abs(sum(projections) / born - 1) < Decimal("1e-28")
        runtime.set_spin_correlation_vectors({3: (0, 0, 2j, 0)})
        assert abs(_value(runtime).real / projections[1] - 4) < Decimal("1e-28")
        runtime.set_spin_correlation_vectors({3: list(polarizations)})
        batch = runtime.evaluate_correlated((POINT, POINT), precision=32)
        assert tuple(item.real for item in batch) == tuple(projections)


@pytest.fixture(
    scope="module",
    params=(
        (
            "d d~ > g g",
            POINT,
            (3, 4),
            3,
            ((0, Decimal("0.8"), 0, Decimal("-0.6")), (0, 0, 1, 0)),
        ),
        (
            "d g > d g",
            POINT,
            (2, 4),
            2,
            ((0, 1, 0, 0), (0, 0, 1, 0)),
        ),
        (
            "d d~ > u u~ g",
            THREE_BODY_POINT,
            (5,),
            5,
            ((0, 0, 1, 0), (0, 0, 0, 1)),
        ),
    ),
    ids=("quark-annihilation", "crossed-quark-gluon", "two-quark-lines"),
)
def correlated_quarks(request, tmp_path_factory):
    pytest.importorskip("pyamplicol._rusticol")
    expression, point, gluons, projected_leg, polarizations = request.param
    output = tmp_path_factory.mktemp("correlated-quarks") / "process"
    declarations = CorrelatorConfig(
        color_correlations=tuple(
            ColorCorrelator.dipole(f"T1{leg}", 1, leg)
            for leg in range(1, len(point) + 1)
        ),
        spin_correlations=tuple((leg,) for leg in gluons),
    )
    Generator(_configuration()).generate(
        expression,
        output,
        model=ModelSource.built_in_sm(),
        correlators=declarations,
    )
    return output, point, gluons, projected_leg, polarizations


def test_generated_quark_born_and_colour_coherence(correlated_quarks):
    output, point, _gluons, _projected_leg, _polarizations = correlated_quarks
    runtime = Runtime.load(output)
    born = runtime.evaluate_correlated((point,), precision=32)[0]
    ordinary = runtime.evaluate((point,), precision=40)[0]
    with localcontext() as context:
        context.prec = 40
        assert ordinary > 0
        assert abs(born.real / ordinary - 1) < Decimal("1e-28")
        assert abs(born.imag / ordinary) < Decimal("1e-28")
        dipoles = [
            runtime.evaluate_correlated(
                (point,), color_correlation=f"T1{leg}", precision=32
            )[0]
            for leg in range(1, len(point) + 1)
        ]
        assert abs(dipoles[0].real / born.real - Decimal(4) / 3) < Decimal("1e-28")
        assert abs(sum(item.real for item in dipoles) / born.real) < Decimal("1e-28")
        assert abs(sum(item.imag for item in dipoles) / born.real) < Decimal("1e-28")


def test_generated_quark_ward_and_polarization_completeness(correlated_quarks):
    output, point, gluons, projected_leg, polarizations = correlated_quarks
    runtime = Runtime.load(output)
    born = runtime.evaluate_correlated((point,), precision=32)[0].real
    with localcontext() as context:
        context.prec = 40
        for leg in gluons:
            runtime.set_spin_correlation_vectors({leg: point[leg - 1]})
            scale = born * Decimal(point[leg - 1][0]) ** 2
            for identifier in ("born", f"T1{leg}"):
                ward = runtime.evaluate_correlated(
                    (point,), color_correlation=identifier, precision=32
                )[0]
                assert abs(ward.real / scale) < Decimal("1e-28")
                assert abs(ward.imag / scale) < Decimal("1e-28")
        projections = []
        for vector in polarizations:
            runtime.set_spin_correlation_vectors({projected_leg: vector})
            projections.append(
                runtime.evaluate_correlated((point,), precision=32)[0].real
            )
        assert abs(sum(projections) / born - 1) < Decimal("1e-28")
    runtime.set_spin_correlation_vectors(None)
    assert runtime.evaluate_correlated((point,), precision=32)[0].real == born

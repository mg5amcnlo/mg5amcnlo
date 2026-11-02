# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import pytest

from pyamplicol.models.compiler_entry import (
    _annotate_oriented_kernel_color_projections,
)
from pyamplicol.models.contracts import (
    CompiledOrientedKernel,
    CompiledParticleRecord,
    CompiledVertexTerm,
)
from pyamplicol.models.tensors import (
    _materialized_color_components,
    classify_trilinear_color_expression,
    normalize_color_expression,
    normalize_lorentz_expression,
    project_trilinear_color_expression,
)
from pyamplicol.processes.model import ModelParticleCatalog


def _particle(name: str, color: int) -> CompiledParticleRecord:
    return CompiledParticleRecord(
        name=name,
        antiname=name,
        pdg_code=9000001,
        spin=1,
        color=color,
        mass="ZERO",
        width="ZERO",
        charge=0.0,
        quantum_numbers=(("electric_charge", "0"),),
        ghost_number=0,
        propagating=True,
        goldstoneboson=False,
        propagator=None,
    )


def test_model_tensors_reject_unsupported_colored_representations() -> None:
    with pytest.raises(ValueError, match="unsupported UFO color representation 6"):
        normalize_color_expression("1", [6, 1, 1])

    with pytest.raises(ValueError, match="particle 'sextet'"):
        ModelParticleCatalog("synthetic", (_particle("sextet", 6),))


def test_model_tensors_keep_supported_singlet_projection() -> None:
    normalized = normalize_color_expression("1", [1, 1, 1])
    assert normalized.expression == "1"


@pytest.mark.parametrize(
    ("source", "spins", "factory", "indices"),
    (
        (
            "UFO::Metric(UFO::idx(1,1),UFO::idx(1,2))",
            (3, 3),
            "metric",
            ("ufo_l_1_1", "ufo_l_1_2"),
        ),
        (
            "UFO::Gamma(UFO::idx(1,1),UFO::idx(1,2),UFO::idx(1,3))",
            (3, 2, 2),
            "dirac_gamma",
            ("ufo_s_1_2", "ufo_s_1_3", "ufo_l_1_1"),
        ),
        (
            "UFO::Gamma5(UFO::idx(1,1),UFO::idx(1,2))",
            (2, 2),
            "gamma5",
            ("ufo_s_1_1", "ufo_s_1_2"),
        ),
        (
            "UFO::ProjM(UFO::idx(1,1),UFO::idx(1,2))",
            (2, 2),
            "projp",
            ("ufo_s_1_1", "ufo_s_1_2"),
        ),
        (
            "UFO::ProjP(UFO::idx(1,1),UFO::idx(1,2))",
            (2, 2),
            "projm",
            ("ufo_s_1_1", "ufo_s_1_2"),
        ),
        (
            "UFO::Identity(UFO::idx(1,1),UFO::idx(1,2))",
            (2, 2),
            "identity",
            ("ufo_s_1_2", "ufo_s_1_1"),
        ),
    ),
)
def test_lorentz_tensor_factories_preserve_ufo_index_conventions(
    source: str,
    spins: tuple[int, ...],
    factory: str,
    indices: tuple[str, ...],
) -> None:
    from symbolica import E
    from symbolica.community.tensor import Representation, TensorExpression

    if factory == "metric":
        tensor = TensorExpression.g(Representation.mink(4))
    elif factory == "identity":
        tensor = TensorExpression.g(Representation.bis(4))
    else:
        tensor = getattr(TensorExpression, factory)(4)
    expected = tensor(*indices).simplify_algebra(color=False).to_expression()

    normalized = normalize_lorentz_expression(source, spins)

    assert E(normalized.expression) == expected


def test_vertex_spinor_slots_follow_the_standard_ufo_fermion_order() -> None:
    """Vertices index a fermion pair as MadGraph's SM, Gamma(3,2,1) for FFV1;
    propagators keep their own order."""
    from symbolica import E

    def gamma(first: int, second: int, *, vertex: bool = True) -> object:
        return E(
            normalize_lorentz_expression(
                f"UFO::Gamma(UFO::idx(1,3),UFO::idx(1,{first}),UFO::idx(1,{second}))",
                (2, 2, 3),
                vertex=vertex,
            ).expression
        )

    assert gamma(2, 1) == gamma(1, 2, vertex=False)
    assert gamma(2, 1) != gamma(2, 1, vertex=False)


def test_lorentz_clifford_contraction_preserves_4d_normalization() -> None:
    from symbolica import E

    normalized = normalize_lorentz_expression(
        "UFO::Gamma(UFO::dummy(1),UFO::idx(1,1),UFO::dummy(2))*"
        "UFO::Gamma(UFO::dummy(1),UFO::dummy(2),UFO::idx(1,2))",
        (2, 2),
    )
    identity = normalize_lorentz_expression(
        "UFO::Identity(UFO::idx(1,1),UFO::idx(1,2))",
        (2, 2),
    )

    assert E(normalized.expression) == 4 * E(identity.expression)


def test_color_contraction_substitutes_su3_casimir() -> None:
    normalized = normalize_color_expression(
        "UFO::f(1,-1,-2)*UFO::f(2,-1,-2)",
        (8, 8),
    )

    assert _materialized_color_components(normalized.expression, (1, 2)) == {
        (left, right): complex(3 * (left == right))
        for left in range(8)
        for right in range(8)
    }


@pytest.mark.parametrize(
    ("right", "expected"), (("-1,-2,-3", "24"), ("-3,-2,-1", "-24"))
)
def test_closed_color_contraction_preserves_structure_constant_sign(
    right: str, expected: str
) -> None:
    normalized = normalize_color_expression(
        f"UFO::f(-1,-2,-3)*UFO::f({right})",
        (),
    )

    assert normalized.expression == expected


def test_materialized_adjoint_identity_preserves_physical_axis_order() -> None:
    normalized = normalize_color_expression("UFO::Identity(1,2)", (8, 8))

    for order in ((1, 2), (2, 1)):
        actual = _materialized_color_components(normalized.expression, order)
        assert actual == {
            (left, right): complex(left == right)
            for left in range(8)
            for right in range(8)
        }


@pytest.mark.parametrize("representations", ((3, -3), (-3, 3), (3, 3), (-3, -3)))
def test_color_identity_preserves_physical_and_oriented_index_variance(
    representations: tuple[int, int],
) -> None:
    from symbolica import E
    from symbolica.community.tensor import Representation, as_tensor

    normalized = normalize_color_expression("UFO::Identity(1,2)", representations)
    fundamental = Representation.cof(3)
    expected = {
        (fundamental if color == 3 else fundamental.dual())(f"ufo_c_{leg}")
        .to_expression()
        .format_plain()
        for leg, color in enumerate(representations, start=1)
    }
    actual = {
        slot.to_expression().format_plain()
        for slot in as_tensor(E(normalized.expression)).axes
    }

    assert actual == expected


def test_materialized_color_generator_reorders_logical_axes_once() -> None:
    normalized = normalize_color_expression("UFO::T(3,1,2)", (3, -3, 8))

    original = _materialized_color_components(normalized.expression, (1, 2, 3))
    permuted = _materialized_color_components(normalized.expression, (3, 1, 2))

    assert len(original) == 72
    assert any(value != 0 for value in original.values())
    assert permuted == {
        (adjoint, fundamental, antifundamental): value
        for (fundamental, antifundamental, adjoint), value in original.items()
    }


def test_color_identity_rejects_unpaired_representations() -> None:
    with pytest.raises(ValueError, match="identity requires dual representations"):
        normalize_color_expression("UFO::Identity(1,2)", (3, 8))


def test_materialized_color_scalar_has_no_open_axes() -> None:
    normalize_color_expression("2", (1,))

    assert _materialized_color_components("2", ()) == {(): 2 + 0j}


def test_trilinear_color_projection_proves_scaled_permuted_generator() -> None:
    normalized = normalize_color_expression("2*UFO::T(3,1,2)", [3, -3, 8])

    assert project_trilinear_color_expression(
        normalized.expression,
        [3, -3, 8],
    ) == ("fundamental-generator", 2.0 + 0.0j)


def test_trilinear_color_projection_does_not_trust_ufo_source_spelling() -> None:
    assert classify_trilinear_color_expression(
        "1",
        "UFO::T(3,1,2)",
        [3, -3, 8],
    ) == ("generic-tensor", 1.0 + 0.0j)


def test_external_compiler_rejects_unimplemented_symmetric_color_tensor() -> None:
    particles = tuple(
        CompiledParticleRecord(
            name=f"adjoint_{index}",
            antiname=f"adjoint_{index}",
            pdg_code=8_100_000 + index,
            spin=3,
            color=8,
            mass="ZERO",
            width="ZERO",
            charge=0.0,
            quantum_numbers=(("electric_charge", "0"),),
            ghost_number=0,
            propagating=True,
            goldstoneboson=False,
            propagator=None,
        )
        for index in range(3)
    )
    color = normalize_color_expression("UFO::d(1,2,3)", [8, 8, 8])
    term = CompiledVertexTerm(
        id=0,
        vertex="symmetric-adjoint",
        particles=tuple(particle.name for particle in particles),
        color_index=0,
        lorentz_index=0,
        color_source="UFO::d(1,2,3)",
        color_expression=color.expression,
        lorentz_name="L",
        lorentz_source="1",
        lorentz_expression="1",
        coupling="GC",
        coupling_expression="1",
        coupling_orders=(),
    )
    kernel = CompiledOrientedKernel(
        kind=0,
        term_id=0,
        vertex=term.vertex,
        particles=tuple(particle.name for particle in particles),
        source_particle_legs=(0, 1, 2),
        component_expressions=("1",),
        coupling_expression="1",
        coupling_orders=(),
        runtime_parameters=(),
        color_source=term.color_source,
        color_expression=term.color_expression,
    )

    with pytest.raises(ValueError, match="unsupported trilinear color tensor"):
        _annotate_oriented_kernel_color_projections((kernel,), particles, (term,))


def test_oriented_identity_retains_the_exact_vertex_color_proof() -> None:
    particles = (
        _particle("anti", -3),
        _particle("singlet", 1),
        _particle("fund", 3),
    )
    color = normalize_color_expression("UFO::Identity(1,3)", [-3, 1, 3])
    term = CompiledVertexTerm(
        id=0,
        vertex="identity",
        particles=("anti", "singlet", "fund"),
        color_index=0,
        lorentz_index=0,
        color_source="UFO::Identity(1,3)",
        color_expression=color.expression,
        lorentz_name="L",
        lorentz_source="1",
        lorentz_expression="1",
        coupling="GC",
        coupling_expression="1",
        coupling_orders=(),
    )
    kernel = CompiledOrientedKernel(
        kind=0,
        term_id=0,
        vertex=term.vertex,
        particles=("fund", "singlet", "fund"),
        source_particle_legs=(2, 1, 0),
        component_expressions=("1",),
        coupling_expression="1",
        coupling_orders=(),
        runtime_parameters=(),
        color_source="oriented-identity",
        color_expression="1",
    )

    (annotated,) = _annotate_oriented_kernel_color_projections(
        (kernel,), particles, (term,)
    )

    assert annotated.color_projection_structure == "color-identity"
    assert len(annotated.lc_color_transition_terms) == 1
    witness = annotated.lc_color_transition_terms[0]
    assert witness.component_operation == "inherit-left"
    assert witness.input_permutation == (0, 1)

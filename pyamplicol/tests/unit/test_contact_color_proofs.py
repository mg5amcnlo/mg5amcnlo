# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
from dataclasses import replace
from itertools import product
from pathlib import Path

import pytest

from pyamplicol._internal.physics.symbols import ModelSymbolRegistry, symbols
from pyamplicol.models import compiler_contacts
from pyamplicol.models import compiler_symbolica as _sym
from pyamplicol.models.compiler_contact_trees import _deduplicate_contact_partials
from pyamplicol.models.compiler_contacts import (
    CONTACT_DECOMPOSITION_ALGORITHM,
    CONTACT_DECOMPOSITION_ALGORITHM_VERSION,
    _build_contact_orbit_certificate,
    _four_point_contact_color_split,
    _fuse_contact_finals,
    _record_contact_decomposition_proofs,
)
from pyamplicol.models.compiler_entry import (
    _compile_four_point_contact_kernels,
    _contact_recurrence_color_contract,
)
from pyamplicol.models.compiler_tensor_ordering import (
    compile_tensor_ordering_metadata,
)
from pyamplicol.models.contact_decomposition import (
    CONTACT_ORBIT_ALGORITHM,
    CONTACT_ORBIT_ALGORITHM_VERSION,
    CONTACT_ORBIT_EVALUATOR_CLASS,
    CompiledContactOrbitCertificate,
    CompiledContactOrbitStep,
)
from pyamplicol.models.contracts import (
    CompiledModelIR,
    CompiledOrientedKernel,
    CompiledParticleRecord,
    CompiledVertexTerm,
)
from pyamplicol.models.loading import (
    CompiledModel,
    compiler_fingerprint,
    load_compiled_model,
)


def _adjoint(name: str, pdg: int, *, spin: int = 3) -> CompiledParticleRecord:
    return CompiledParticleRecord(
        name=name,
        antiname=name,
        pdg_code=pdg,
        spin=spin,
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


def _scalar(
    name: str,
    pdg: int,
    *,
    antiname: str | None = None,
    spin: int = 1,
    color: int = 1,
) -> CompiledParticleRecord:
    return CompiledParticleRecord(
        name=name,
        antiname=name if antiname is None else antiname,
        pdg_code=pdg,
        spin=spin,
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


def _term(*, color_source: str, color_expression: str) -> CompiledVertexTerm:
    return CompiledVertexTerm(
        id=901,
        vertex="V_adversarial_contact",
        particles=("a", "b", "c", "d"),
        color_index=0,
        lorentz_index=0,
        color_source=color_source,
        color_expression=color_expression,
        lorentz_name="L_contact",
        lorentz_source="1",
        lorentz_expression="1",
        coupling="GC_contact",
        coupling_expression="1",
        coupling_orders=(),
    )


def _proof_term(
    term: CompiledVertexTerm,
    particles: tuple[CompiledParticleRecord, ...],
    *,
    model_name: str,
) -> CompiledVertexTerm:
    return _record_contact_decomposition_proofs(
        (term,),
        particles,
        model_symbols=symbols.model(model_name),
    )[0]


@pytest.mark.parametrize(
    ("particle_names", "expected_classes"),
    (
        (("a", "a", "a", "a"), (0, 0, 0, 0)),
        (("a", "a", "b", "c"), (0, 0, 1, 2)),
    ),
)
def test_constant_scalar_contact_issues_deterministic_orbit_contract(
    particle_names: tuple[str, str, str, str],
    expected_classes: tuple[int, int, int, int],
) -> None:
    particle_by_name = {
        name: _scalar(name, 9_510_000 + index)
        for index, name in enumerate(dict.fromkeys(particle_names))
    }
    particles = tuple(particle_by_name.values())
    term = replace(
        _term(color_source="1", color_expression="1"),
        particles=particle_names,
    )

    proved = _proof_term(term, particles, model_name="scalar-contact-orbit")
    repeated = _proof_term(term, particles, model_name="scalar-contact-orbit")

    certificate = proved.contact_orbit_certificate
    assert certificate is not None
    assert certificate == repeated.contact_orbit_certificate
    assert certificate.algorithm == CONTACT_ORBIT_ALGORITHM
    assert certificate.algorithm_version == CONTACT_ORBIT_ALGORITHM_VERSION
    assert certificate.evaluator_class == CONTACT_ORBIT_EVALUATOR_CLASS
    assert certificate.physical_leg_equivalence_classes == expected_classes
    assert certificate.reconstruction_factor == "1"
    assert CompiledContactOrbitCertificate.from_dict(
        certificate.to_dict()
    ) == certificate

    _auxiliaries, kernels = _compile_four_point_contact_kernels(
        (proved,),
        particles,
        start_kind=0,
        model_symbols=symbols.model("scalar-contact-orbit"),
    )
    steps = tuple(
        step
        for kernel in kernels
        for step in kernel.contact_orbit_steps
    )
    assert steps
    assert {step.stage for step in steps} == {"partial", "final"}
    assert all(step.term_id == term.id for step in steps)
    assert all(step.reconstruction_factor == "1" for step in steps)
    assert all(
        CompiledContactOrbitStep.from_dict(step.to_dict()) == step for step in steps
    )
    for kernel in kernels:
        if kernel.contact_orbit_steps:
            assert len(kernel.contact_orbit_steps) == 1
            assert (
                kernel.contact_orbit_steps[0].source_particle_legs
                == kernel.source_particle_legs
            )


def test_contact_orbit_contract_rejects_tampering_and_uncertified_classes() -> None:
    scalar = _scalar("a", 9_520_000)
    term = replace(
        _term(color_source="1", color_expression="1"),
        particles=("a", "a", "a", "a"),
    )
    proved = _proof_term(term, (scalar,), model_name="scalar-contact-tamper")
    certificate = proved.contact_orbit_certificate
    assert certificate is not None

    payload = certificate.to_dict()
    payload["physical_leg_equivalence_classes"] = [0, 1, 2, 3]
    with pytest.raises(ValueError, match="not canonical"):
        CompiledContactOrbitCertificate.from_dict(payload)
    payload = certificate.to_dict()
    payload["reconstruction_factor"] = "2"
    with pytest.raises(ValueError, match="factor must be one"):
        CompiledContactOrbitCertificate.from_dict(payload)
    payload = certificate.to_dict()
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        CompiledContactOrbitCertificate.from_dict(payload)

    vector = _scalar("v", 9_520_001, spin=3)
    colored = _scalar("g", 9_520_002, color=8)
    charged = _scalar("s", 9_520_003, antiname="s~")
    anti = _scalar("s~", -9_520_003, antiname="s")
    proof = proved.contact_decomposition_proof
    assert proof is not None
    assert _build_contact_orbit_certificate(term, (vector,) * 4, proof) is None
    assert _build_contact_orbit_certificate(term, (colored,) * 4, proof) is None
    assert (
        _build_contact_orbit_certificate(
            replace(term, particles=("s", "s~", "s", "s~")),
            (charged, anti, charged, anti),
            proof,
        )
        is None
    )
    momentum = replace(term, lorentz_expression="ufo_momentum_1_0")
    assert _record_contact_decomposition_proofs(
        (momentum,),
        (scalar,),
        model_symbols=symbols.model("contact-orbit-excluded-momentum"),
    )[0].contact_orbit_certificate is None


def test_single_lineage_contact_orbit_metadata_preserves_numerical_fusion() -> None:
    particle_names = ("a", "a", "a", "a")
    particle_by_name = {
        name: _scalar(name, 9_519_000 + index)
        for index, name in enumerate(dict.fromkeys(particle_names))
    }
    particles = tuple(particle_by_name.values())
    term = replace(
        _term(color_source="1", color_expression="1"),
        particles=particle_names,
    )
    certified = _proof_term(term, particles, model_name="orbit-fusion-certified")
    plain = replace(certified, contact_orbit_certificate=None)

    def finalized(
        value: CompiledVertexTerm,
        model_name: str,
    ) -> tuple[
        tuple[CompiledParticleRecord, ...],
        tuple[CompiledOrientedKernel, ...],
    ]:
        registry = symbols.model(model_name)
        auxiliaries, kernels = _compile_four_point_contact_kernels(
            (value,), particles, start_kind=0, model_symbols=registry
        )
        auxiliaries, kernels = _deduplicate_contact_partials(
            auxiliaries, kernels, (value,), model_symbols=registry
        )
        return auxiliaries, _fuse_contact_finals(
            kernels, (value,), model_symbols=registry
        )

    certified_auxiliaries, certified_kernels = finalized(
        certified, "orbit-fusion-invariant"
    )
    plain_auxiliaries, plain_kernels = finalized(
        plain, "orbit-fusion-invariant"
    )

    assert certified_auxiliaries == plain_auxiliaries
    assert tuple(
        replace(kernel, contact_orbit_steps=()) for kernel in certified_kernels
    ) == plain_kernels
    assert len(certified_kernels) == len(plain_kernels)
    actual_steps = tuple(
        step for kernel in certified_kernels for step in kernel.contact_orbit_steps
    )
    proof = certified.contact_decomposition_proof
    assert proof is not None
    assert len(actual_steps) == sum(
        len(split.orientations) for split in proof.splits
    )
    assert len(actual_steps) == len(set(actual_steps))


def _mixed_scalar_contact_partials(
    *,
    certified: bool,
    model_name: str,
) -> tuple[
    tuple[CompiledParticleRecord, ...],
    tuple[CompiledOrientedKernel, ...],
    CompiledVertexTerm,
    ModelSymbolRegistry,
]:
    particles = tuple(
        _scalar(name, 9_519_100 + index)
        for index, name in enumerate(("a", "b", "c"))
    )
    term = replace(
        _term(color_source="1", color_expression="1"),
        particles=("a", "a", "b", "c"),
    )
    term = _proof_term(term, particles, model_name=model_name)
    if not certified:
        term = replace(term, contact_orbit_certificate=None)
    registry = symbols.model(model_name)
    auxiliaries, kernels = _compile_four_point_contact_kernels(
        (term,), particles, start_kind=0, model_symbols=registry
    )
    return auxiliaries, kernels, term, registry


def test_certified_contact_partials_keep_distinct_compiler_lineages() -> None:
    auxiliaries, kernels, term, registry = _mixed_scalar_contact_partials(
        certified=True,
        model_name="orbit-lineage-distinct",
    )

    retained_auxiliaries, retained_kernels = _deduplicate_contact_partials(
        auxiliaries,
        kernels,
        (term,),
        model_symbols=registry,
    )

    lineages = {
        particle.auxiliary_kind
        for particle in retained_auxiliaries
        if particle.auxiliary_kind in {
            "ufo-contact:901:result-2:2,3",
            "ufo-contact:901:result-3:2,3",
        }
    }
    assert lineages == {
        "ufo-contact:901:result-2:2,3",
        "ufo-contact:901:result-3:2,3",
    }
    same_input_partials = tuple(
        kernel
        for kernel in retained_kernels
        if kernel.vertex.endswith("::contact-partial")
        and kernel.particles[:2] == ("a", "a")
    )
    assert len(same_input_partials) == 2
    assert all(len(kernel.contact_orbit_steps) == 1 for kernel in same_input_partials)
    assert {
        kernel.contact_orbit_steps[0].result_leg for kernel in same_input_partials
    } == {2, 3}


def test_certified_contact_partial_aliases_with_same_lineage_still_deduplicate(
) -> None:
    auxiliaries, kernels, term, registry = _mixed_scalar_contact_partials(
        certified=True,
        model_name="orbit-lineage-alias",
    )
    auxiliary_by_name = {particle.name: particle for particle in auxiliaries}
    original_kernel = next(
        kernel
        for kernel in kernels
        if kernel.vertex.endswith("::contact-partial")
        and kernel.particles[:2] == ("a", "a")
    )
    original_auxiliary = auxiliary_by_name[original_kernel.particles[2]]
    alias_name = f"{original_auxiliary.name}_alias"
    alias_auxiliary = replace(
        original_auxiliary,
        name=alias_name,
        antiname=alias_name,
        pdg_code=original_auxiliary.pdg_code + 100,
    )
    alias_kernel = replace(
        original_kernel,
        particles=(*original_kernel.particles[:2], alias_name),
    )

    retained_auxiliaries, retained_kernels = _deduplicate_contact_partials(
        (original_auxiliary, alias_auxiliary),
        (original_kernel, alias_kernel),
        (term,),
        model_symbols=registry,
    )

    assert retained_auxiliaries == (original_auxiliary,)
    assert retained_kernels == (original_kernel,)


def test_uncertified_contact_partial_numeric_deduplication_is_unchanged() -> None:
    auxiliaries, kernels, term, registry = _mixed_scalar_contact_partials(
        certified=False,
        model_name="orbit-lineage-uncertified",
    )

    retained_auxiliaries, retained_kernels = _deduplicate_contact_partials(
        auxiliaries,
        kernels,
        (term,),
        model_symbols=registry,
    )

    assert len(retained_auxiliaries) == len(auxiliaries) - 1
    same_input_partials = tuple(
        kernel
        for kernel in retained_kernels
        if kernel.vertex.endswith("::contact-partial")
        and kernel.particles[:2] == ("a", "a")
    )
    assert len(same_input_partials) == 1
    assert same_input_partials[0].contact_orbit_steps == ()


def test_compiled_ir_requires_orbit_certificate_exactly_for_certifiable_class() -> None:
    scalar = _scalar("s", 9_520_010)
    term = replace(
        _term(color_source="1", color_expression="1"),
        particles=("s", "s", "s", "s"),
    )
    proved = _proof_term(term, (scalar,), model_name="scalar-orbit-required")
    auxiliaries, kernels = _compile_four_point_contact_kernels(
        (proved,),
        (scalar,),
        start_kind=0,
        model_symbols=symbols.model("scalar-orbit-required"),
    )
    particles = (scalar, *auxiliaries)
    terms, kernels, orderings, current_orderings = compile_tensor_ordering_metadata(
        (proved,),
        particles,
        kernels,
        (),
        (),
    )
    missing_term = replace(terms[0], contact_orbit_certificate=None)
    missing_steps = tuple(replace(kernel, contact_orbit_steps=()) for kernel in kernels)
    with pytest.raises(ValueError, match="has no orbit certificate"):
        CompiledModelIR(
            name="scalar-orbit-missing",
            orders=(),
            parameters=(),
            particles=particles,
            couplings=(),
            propagators=(),
            vertex_terms=(missing_term,),
            oriented_kernels=missing_steps,
            direct_contractions=(),
            closure_contractions=(),
            tensor_orderings=orderings,
            current_orderings=current_orderings,
        )

    colored = _scalar("g", 9_520_011, color=8)
    colored_term = replace(term, particles=("g", "g", "g", "g"))
    colored_proved = _proof_term(
        colored_term,
        (colored,),
        model_name="colored-orbit-excluded",
    )
    certificate = proved.contact_orbit_certificate
    assert certificate is not None
    forged = replace(
        colored_proved,
        contact_orbit_certificate=replace(
            certificate,
            term_id=colored_proved.id,
            vertex=colored_proved.vertex,
            particles=("g", "g", "g", "g"),
        ),
    )
    forged_terms, _kernels, forged_orderings, forged_current_orderings = (
        compile_tensor_ordering_metadata(
            (forged,),
            (colored,),
            (),
            (),
            (),
        )
    )
    with pytest.raises(ValueError, match="excluded evaluator class"):
        CompiledModelIR(
            name="vector-orbit-forged",
            orders=(),
            parameters=(),
            particles=(colored,),
            couplings=(),
            propagators=(),
            vertex_terms=forged_terms,
            oriented_kernels=(),
            direct_contractions=(),
            closure_contractions=(),
            tensor_orderings=forged_orderings,
            current_orderings=forged_current_orderings,
        )


def test_contact_tree_uses_its_compiler_owned_singlet_contract() -> None:
    kernel = CompiledOrientedKernel(
        kind=1,
        term_id=25,
        vertex="V_5_SCALAR_00000::contact-tree-partial",
        particles=("scalar_0", "scalar_0", "__pyamplicol_contact_tree_25_r0_1_2"),
        source_particle_legs=(1, 2, -1),
        component_expressions=("1",),
        coupling_expression="1",
        coupling_orders=(),
        runtime_parameters=(),
        color_source="1",
        color_expression="1",
    )

    assert _contact_recurrence_color_contract(kernel, {}) is None


def test_unproved_colored_four_point_contact_fails_closed() -> None:
    term = _term(
        color_source="UFO::{}::T(1,2,3)",
        color_expression="model_adversarial::T(1,2,3)",
    )
    particles = tuple(
        _adjoint(name, 9_300_000 + index) for index, name in enumerate(term.particles)
    )

    proved_term = _proof_term(term, particles, model_name="adversarial-contact")
    proof = proved_term.contact_decomposition_proof

    assert _four_point_contact_color_split(term, 0) is None
    assert proof is not None
    assert proof.status == "unsupported"
    assert proof.splits == ()
    assert {reason.code for reason in proof.unsupported_reasons} == {
        "unsupported-color-factor-count"
    }
    assert (
        dict(proof.unsupported_reasons[0].context)["normalized_color_expression"]
        == term.color_expression
    )
    auxiliaries, kernels = _compile_four_point_contact_kernels(
        (proved_term,),
        particles,
        start_kind=0,
        model_symbols=symbols.model("adversarial-contact"),
    )

    assert auxiliaries == ()
    assert kernels == ()


def test_literal_color_singlet_keeps_generic_contact_split() -> None:
    term = _term(color_source="1", color_expression="1")

    split = _four_point_contact_color_split(term, 2)

    assert split is not None
    pair, remaining, *_metadata = split
    assert pair == (0, 1)
    assert remaining == 3


def test_structure_constant_contact_preserves_exact_color_coefficient() -> None:
    unit_expression = (
        "spenso::f(ufo_c_2,ufo_c_dummy_7_adjoint,ufo_c_1)"
        "*spenso::f(ufo_c_dummy_7_adjoint,ufo_c_3,ufo_c_4)"
    )
    scaled = _term(
        color_source=("-3/2*UFO::{}::f(2,-7,1)*UFO::{}::f(-7,3,4)"),
        color_expression=f"-3/2*{unit_expression}",
    )
    unit = _term(
        color_source="UFO::{}::f(2,-7,1)*UFO::{}::f(-7,3,4)",
        color_expression=unit_expression,
    )
    particles = tuple(
        _adjoint(name, 9_400_000 + index, spin=1)
        for index, name in enumerate(scaled.particles)
    )
    scaled = _proof_term(
        scaled,
        particles,
        model_name="contact-color-coefficient-scaled",
    )
    unit = _proof_term(
        unit,
        particles,
        model_name="contact-color-coefficient-unit",
    )

    split = _four_point_contact_color_split(scaled, 2)
    assert split is not None
    assert split[-1] == "-3/2"
    proof = scaled.contact_decomposition_proof
    assert proof is not None
    assert proof.status == "proven"
    assert proof.algorithm == CONTACT_DECOMPOSITION_ALGORITHM
    assert proof.algorithm_version == CONTACT_DECOMPOSITION_ALGORITHM_VERSION
    assert proof.original_color_source == scaled.color_source
    assert proof.normalized_color_expression == scaled.color_expression
    assert proof.original_lorentz_source == scaled.lorentz_source
    assert proof.normalized_lorentz_expression == scaled.lorentz_expression
    chosen = next(item for item in proof.splits if item.result_leg == 2)
    assert chosen.decomposition_kind == "two-structure-constants"
    assert chosen.pair_legs == (1, 0)
    assert chosen.remaining_leg == 3
    assert chosen.outer_color_factor == (2, -7, 1)
    assert chosen.final_color_factor == (-7, 3, 4)
    assert chosen.color_coefficient == "-3/2"
    assert chosen.component_axis_order == ()
    assert chosen.component_basis_order == (0,)
    assert chosen.component_expansion == ((0, 1),)
    assert chosen.dummy_index_mapping is not None
    assert chosen.dummy_index_mapping.source_index == -7
    assert chosen.dummy_index_mapping.normalized_symbol == "ufo_c_dummy_7_adjoint"
    assert chosen.dummy_index_mapping.outer_slot == 1
    assert chosen.dummy_index_mapping.final_slot == 0
    partials = tuple(item for item in chosen.orientations if item.stage == "partial")
    finals = tuple(item for item in chosen.orientations if item.stage == "final")
    assert tuple(item.input_legs for item in partials) == ((1, 0), (0, 1))
    assert tuple(item.permutation_parity for item in partials) == (-1, 1)
    assert tuple(item.scalar_prefactor for item in partials) == ("1", "-1")
    assert tuple(item.input_legs for item in finals) == ((-1, 3), (3, -1))
    assert tuple(item.permutation_parity for item in finals) == (-1, 1)
    assert tuple(item.scalar_prefactor for item in finals) == ("-3/2", "3/2")

    model_symbols = symbols.model("contact-color-coefficient")
    _scaled_auxiliaries, scaled_kernels = _compile_four_point_contact_kernels(
        (scaled,),
        particles,
        start_kind=0,
        model_symbols=model_symbols,
    )
    _unit_auxiliaries, unit_kernels = _compile_four_point_contact_kernels(
        (unit,),
        particles,
        start_kind=0,
        model_symbols=model_symbols,
    )
    scaled_finals = tuple(
        kernel for kernel in scaled_kernels if kernel.vertex.endswith("::contact-final")
    )
    unit_finals = tuple(
        kernel for kernel in unit_kernels if kernel.vertex.endswith("::contact-final")
    )

    _sym._ensure_symbolica()
    assert len(scaled_finals) == len(unit_finals) > 0
    for scaled_kernel, unit_kernel in zip(
        scaled_finals,
        unit_finals,
        strict=True,
    ):
        assert scaled_kernel.particles == unit_kernel.particles
        for scaled_component, unit_component in zip(
            scaled_kernel.component_expressions,
            unit_kernel.component_expressions,
            strict=True,
        ):
            difference = (
                _sym.E(scaled_component) + _sym.E("3/2") * _sym.E(unit_component)
            ).expand()
            assert difference == _sym.E("0")


def test_structure_constant_contact_does_not_duplicate_normalization_sign() -> None:
    source = "UFO::{}::f(-1,1,3)*UFO::{}::f(2,4,-1)"
    normalized = (
        "-spenso::f(ufo_c_1,ufo_c_3,ufo_c_dummy_1_adjoint)"
        "*spenso::f(ufo_c_2,ufo_c_4,ufo_c_dummy_1_adjoint)"
    )
    term = _term(color_source=source, color_expression=normalized)

    split = _four_point_contact_color_split(term, 0)

    assert split is not None
    assert split[-1] == "1"


@pytest.mark.parametrize("spelling", ("plain", "canonical", "permuted"))
def test_structure_constant_contact_preserves_source_orientations(
    spelling: str,
) -> None:
    from pyamplicol.models.tensors import normalize_color_expression

    source = "-3/2*UFO::{}::f(2,-7,1)*UFO::{}::f(-7,3,4)"
    normalized = normalize_color_expression(source, (8, 8, 8, 8)).expression
    _sym._ensure_symbolica()
    if spelling == "canonical":
        normalized = _sym.E(normalized).to_canonical_string()
    elif spelling == "permuted":
        # An odd permutation changes the normalized scalar, not the source
        # coefficient or the source tensor orientations used for decomposition.
        normalized = (
            "3/2*spenso::f(ufo_c_dummy_7_adjoint,ufo_c_3,ufo_c_4)"
            "*spenso::f(ufo_c_dummy_7_adjoint,ufo_c_2,ufo_c_1)"
        )
    term = _term(color_source=source, color_expression=normalized)
    particles = tuple(
        _adjoint(name, 9_450_000 + index, spin=1)
        for index, name in enumerate(term.particles)
    )
    proved = _proof_term(term, particles, model_name="source-oriented-contact")
    reference = _proof_term(
        replace(
            term,
            color_expression=(
                "-3/2*spenso::f(ufo_c_2,ufo_c_dummy_7_adjoint,ufo_c_1)"
                "*spenso::f(ufo_c_dummy_7_adjoint,ufo_c_3,ufo_c_4)"
            ),
        ),
        particles,
        model_name="source-oriented-contact",
    )
    assert proved.contact_decomposition_proof is not None
    assert reference.contact_decomposition_proof is not None
    assert proved.contact_decomposition_proof.status == "proven"
    assert proved.contact_decomposition_proof.splits == (
        reference.contact_decomposition_proof.splits
    )

    def epsilon(a: int, b: int, c: int) -> int:
        return (a - b) * (b - c) * (c - a) // 2

    # Independently reconstruct the colour product in the SU(2) subalgebra,
    # including every result orientation and all external colour assignments.
    for split in proved.contact_decomposition_proof.splits:
        final = next(
            item
            for item in split.orientations
            if item.stage == "final" and item.input_legs[0] == -1
        )
        coefficient = _sym.E(final.scalar_prefactor)
        for indices in product(range(3), repeat=4):
            expected = _sym.E("-3/2") * sum(
                epsilon(indices[1], dummy, indices[0])
                * epsilon(dummy, indices[2], indices[3])
                for dummy in range(3)
            )
            actual = coefficient * sum(
                epsilon(dummy, indices[split.pair_legs[0]], indices[split.pair_legs[1]])
                * epsilon(
                    indices[split.result_leg], dummy, indices[split.remaining_leg]
                )
                for dummy in range(3)
            )
            assert actual == expected


def test_structure_constant_contact_rejects_residual_color_tensor() -> None:
    term = _term(
        color_source=("UFO::{}::f(-1,1,2)*UFO::{}::f(3,4,-1)*UFO::{}::T(1,2,3)"),
        color_expression=(
            "spenso::f(ufo_c_dummy_1_adjoint,ufo_c_1,ufo_c_2)"
            "*spenso::f(ufo_c_3,ufo_c_4,ufo_c_dummy_1_adjoint)"
            "*spenso::t(ufo_c_1,ufo_c_2,ufo_c_3)"
        ),
    )

    particles = tuple(
        _adjoint(name, 9_500_000 + index) for index, name in enumerate(term.particles)
    )
    proved_term = _proof_term(
        term,
        particles,
        model_name="residual-contact-color",
    )
    proof = proved_term.contact_decomposition_proof

    assert _four_point_contact_color_split(term, 0) is None
    assert proof is not None
    assert proof.status == "unsupported"
    assert {reason.code for reason in proof.unsupported_reasons} == {
        "non-scalar-color-prefactor"
    }


def test_contact_proof_round_trips_and_lowering_does_not_rediscover(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    expression = (
        "spenso::f(ufo_c_2,ufo_c_dummy_11_adjoint,ufo_c_1)"
        "*spenso::f(ufo_c_dummy_11_adjoint,ufo_c_3,ufo_c_4)"
    )
    term = _term(
        color_source="UFO::{}::f(2,-11,1)*UFO::{}::f(-11,3,4)",
        color_expression=expression,
    )
    particles = tuple(
        _adjoint(name, 9_600_000 + index, spin=1)
        for index, name in enumerate(term.particles)
    )
    proved_term = _proof_term(term, particles, model_name="serialized-contact-proof")
    proved_terms, _kernels, orderings, current_orderings = (
        compile_tensor_ordering_metadata(
            (proved_term,),
            particles,
            (),
            (),
            (),
        )
    )
    proved_term = proved_terms[0]
    model = CompiledModelIR(
        name="serialized-contact-proof",
        orders=(),
        parameters=(),
        particles=particles,
        couplings=(),
        propagators=(),
        vertex_terms=(proved_term,),
        oriented_kernels=(),
        direct_contractions=(),
        closure_contractions=(),
        tensor_orderings=orderings,
        current_orderings=current_orderings,
    )
    compiled = CompiledModel(
        source={"kind": "json"},
        producer=compiler_fingerprint(),
        model={"name": model.name},
        ir=model,
        parameter_defaults={},
        capabilities={},
        issues=(),
        phase_timings={},
        conversion_seconds=0.0,
    )
    serialized_path = compiled.write(tmp_path / "serialized-contact-proof")
    payload = json.loads(serialized_path.read_text(encoding="utf-8"))
    term_payload = payload["ir"]["vertex_terms"]
    assert isinstance(term_payload, list)
    assert term_payload[0]["contact_decomposition_proof"]["status"] == "proven"
    restored = load_compiled_model(serialized_path).ir.vertex_terms[0]

    invalid_payload = json.loads(serialized_path.read_text(encoding="utf-8"))
    invalid_payload["ir"]["vertex_terms"][0]["contact_decomposition_proof"][
        "algorithm_version"
    ] = CONTACT_DECOMPOSITION_ALGORITHM_VERSION + 1
    invalid_path = tmp_path / "invalid-contact-proof.pyAmplicol-model.json"
    invalid_path.write_text(json.dumps(invalid_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported contact decomposition proof"):
        load_compiled_model(invalid_path)

    def forbidden_discovery(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("contact split was rediscovered during lowering")

    monkeypatch.setattr(
        compiler_contacts,
        "_four_point_contact_color_split",
        forbidden_discovery,
    )
    auxiliaries, kernels = _compile_four_point_contact_kernels(
        (restored,),
        particles,
        start_kind=0,
        model_symbols=symbols.model("serialized-contact-proof"),
    )

    assert (
        restored.contact_decomposition_proof == proved_term.contact_decomposition_proof
    )
    assert auxiliaries
    assert any(kernel.vertex.endswith("::contact-final") for kernel in kernels)


def test_contact_proof_identity_and_algorithm_fail_closed() -> None:
    expression = (
        "spenso::f(ufo_c_2,ufo_c_dummy_13_adjoint,ufo_c_1)"
        "*spenso::f(ufo_c_dummy_13_adjoint,ufo_c_3,ufo_c_4)"
    )
    term = _term(
        color_source="UFO::{}::f(2,-13,1)*UFO::{}::f(-13,3,4)",
        color_expression=expression,
    )
    particles = tuple(
        _adjoint(name, 9_700_000 + index, spin=1)
        for index, name in enumerate(term.particles)
    )
    proved_term = _proof_term(term, particles, model_name="closed-contact-proof")
    proof = proved_term.contact_decomposition_proof
    assert proof is not None

    forged_split = replace(
        proof.splits[0],
        component_axis_order=("ufo_l_1_1",),
    )
    forged_proof = replace(
        proof,
        splits=(forged_split, *proof.splits[1:]),
    )
    with pytest.raises(ValueError, match="non-canonical component axes"):
        compile_tensor_ordering_metadata(
            (replace(proved_term, contact_decomposition_proof=forged_proof),),
            particles,
            (),
            (),
            (),
        )

    stale = replace(proved_term, color_expression="1")
    with pytest.raises(ValueError, match="proof identity mismatch"):
        _compile_four_point_contact_kernels(
            (stale,),
            particles,
            start_kind=0,
            model_symbols=symbols.model("closed-contact-proof"),
        )

    with pytest.raises(ValueError, match="unsupported contact decomposition proof"):
        replace(
            proof,
            algorithm_version=CONTACT_DECOMPOSITION_ALGORITHM_VERSION + 1,
        )

    with pytest.raises(ValueError, match="has no contact decomposition proof"):
        CompiledModelIR(
            name="missing-contact-proof",
            orders=(),
            parameters=(),
            particles=particles,
            couplings=(),
            propagators=(),
            vertex_terms=(term,),
            oriented_kernels=(),
            direct_contractions=(),
            closure_contractions=(),
        )

    assert compiler_fingerprint()["contact_decomposition_policy"] == (
        f"{CONTACT_DECOMPOSITION_ALGORITHM}-v{CONTACT_DECOMPOSITION_ALGORITHM_VERSION}"
    )

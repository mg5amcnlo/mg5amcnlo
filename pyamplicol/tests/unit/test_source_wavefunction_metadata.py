# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pytest

from pyamplicol.api.errors import ArtifactError, CompatibilityError
from pyamplicol.generation.artifact_writer import _execution_plan
from pyamplicol.generation.dag_compiler import compile_generic_dag
from pyamplicol.generation.runtime_schema import build_runtime_schema
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models._physics_ir import CrossingIR, PropagatorIR
from pyamplicol.models.base import Model, Particle
from pyamplicol.models.builtin.process_ir import build_process_ir
from pyamplicol.models.contracts import CompiledParticleRecord
from pyamplicol.models.external_catalog import ExternalModelCatalogMixin
from pyamplicol.runtime.symbolica_exact import (
    _antiquark_weyl,
    _massive_vector,
    _particle_mass,
    _quark_weyl,
    _source_wavefunction,
)


class _ExternalCatalog(ExternalModelCatalogMixin, Model):
    def _propagator_record(self, particle_id: int) -> None:
        del particle_id
        return None

    def _parameter_default(self, name: str) -> float:
        if name.casefold() == "zero":
            return 0.0
        raise KeyError(name)


def _compiled_particle(
    name: str,
    antiname: str,
    pdg: int,
    *,
    spin: int = 2,
) -> CompiledParticleRecord:
    return CompiledParticleRecord(
        name=name,
        antiname=antiname,
        pdg_code=pdg,
        spin=spin,
        color=1,
        mass="ZERO",
        width="ZERO",
        charge=0.0,
        quantum_numbers=(("electric_charge", "0"),),
        ghost_number=0,
        propagating=True,
        goldstoneboson=False,
        propagator=None,
    )


def _external_catalog(*records: CompiledParticleRecord) -> _ExternalCatalog:
    by_name = {record.name: record for record in records}
    model = _ExternalCatalog(name="relabelled-source-model")
    model._particle_records_by_name = by_name
    model._particle_records_by_pdg = {record.pdg_code: record for record in records}
    model.particles = {
        record.pdg_code: Particle(
            pdg=record.pdg_code,
            anti_pdg=by_name[record.antiname].pdg_code,
            spin=record.spin,
            dimension=4,
            color_rep=record.color,
        )
        for record in records
    }
    return model


def _exact_source(
    *,
    particle_id: int,
    anti_particle_id: int,
    source_orientation: str,
    dimension: int = 2,
) -> dict[str, object]:
    identity = {
        "canonical_id": f"model:test:state:{particle_id}",
        "species_id": f"model:test:species:{abs(particle_id)}",
        "anti_canonical_id": f"model:test:state:{anti_particle_id}",
        "display_name": f"state_{particle_id}",
        "anti_display_name": f"state_{anti_particle_id}",
        "pdg_label": particle_id,
        "anti_pdg_label": anti_particle_id,
        "orientation": source_orientation,
        "self_conjugate": particle_id == anti_particle_id,
    }
    crossing = {
        "momentum_transform": "identity",
        "helicity_factor": 1,
        "chirality_factor": 1,
        "spin_state_factor": 1,
        "phase": [1.0, 0.0],
    }
    return {
        "leg_label": 1,
        "side": "final",
        "crossing": "identity",
        "particle_id": particle_id,
        "anti_particle_id": anti_particle_id,
        "wavefunction_kind": "fermion",
        "source_orientation": source_orientation,
        "source_basis": "weyl-chiral" if dimension == 2 else "dirac",
        "source_ir": {
            "identity": identity,
            "statistics": "fermion",
            "wavefunction_family": "fermion",
            "component_dimension": dimension,
            "states": [{"helicity": 1, "chirality": 1, "spin_state": 1}],
            "crossing": crossing,
            "basis": "weyl-chiral" if dimension == 2 else "dirac",
            "mass_parameter": None,
            "width_parameter": None,
        },
        "applied_crossing": crossing,
        "source_helicity": 1,
        "chirality": 1,
        "spin_state": 1,
        "dimension": dimension,
    }


def test_external_relabelled_pdg_source_metadata() -> None:
    particle = _compiled_particle("chi", "chi_bar", 810_001)
    antiparticle = _compiled_particle("chi_bar", "chi", -810_001)
    model = _external_catalog(particle, antiparticle)

    assert model.source_wavefunction_kind(810_001) == "fermion"
    assert model.source_orientation(810_001) == "particle"
    assert model.source_orientation(-810_001) == "antiparticle"
    particle_source = model._source_ir(810_001)
    antiparticle_source = model._source_ir(-810_001)
    assert particle_source.identity.canonical_id.endswith(":chi")
    assert particle_source.identity.species_id.endswith(":chi")
    assert antiparticle_source.identity.canonical_id.endswith(":chi_bar")
    assert (
        antiparticle_source.identity.species_id == particle_source.identity.species_id
    )
    assert particle_source.identity.anti_canonical_id.endswith(":chi_bar")
    assert particle_source.basis == "weyl-chiral"


def test_builtin_source_crossing_is_explicit_and_state_preserving() -> None:
    model = BuiltinSMModel()

    fermion = model._source_ir(1)
    assert fermion.basis == "weyl-chiral"
    assert fermion.crossing.to_json_dict() == {
        "momentum_transform": "negate-four-momentum",
        "helicity_factor": 1,
        "chirality_factor": -1,
        "spin_state_factor": -1,
        "phase": [1.0, 0.0],
    }
    assert {
        (state.helicity, state.chirality, state.spin_state) for state in fermion.states
    } == {(-1, -1, -1), (1, 1, 1)}
    assert {
        (state.helicity, state.chirality, state.spin_state)
        for state in map(fermion.crossing.apply, fermion.states)
    } == {(-1, 1, 1), (1, -1, -1)}

    adjoint = model._source_ir(21)
    assert adjoint.basis == "lorentz-vector"
    assert adjoint.crossing.helicity_factor == -1
    assert adjoint.crossing.chirality_factor == 1
    assert adjoint.crossing.spin_state_factor == -1

    singlet_vector = model._source_ir(23)
    assert singlet_vector.crossing.helicity_factor == -1
    assert singlet_vector.crossing.spin_state_factor == -1
    assert tuple(
        (state.helicity, state.spin_state)
        for state in map(singlet_vector.crossing.apply, singlet_vector.states)
    ) == ((1, 1), (0, 0), (-1, -1))


def test_external_scalar_and_spin2_sources_are_metadata_driven() -> None:
    model = _external_catalog(
        _compiled_particle("phi", "phi", 700_001, spin=1),
        _compiled_particle("graviton", "graviton", 700_002, spin=5),
    )

    scalar = model._source_ir(700_001)
    assert scalar.wavefunction_family == "scalar"
    assert scalar.basis == "scalar"
    assert scalar.component_dimension == 1
    assert tuple(state.helicity for state in scalar.states) == (0,)

    spin2 = model._source_ir(700_002)
    assert spin2.wavefunction_family == "spin2"
    assert spin2.basis == "lorentz-rank-2"
    assert spin2.component_dimension == 16
    assert tuple(state.helicity for state in spin2.states) == (-2, 2)
    assert spin2.crossing.momentum_transform == "negate-four-momentum"
    assert tuple(spin2.crossing.apply(state).helicity for state in spin2.states) == (
        2,
        -2,
    )


@pytest.mark.parametrize("helicity", [-1, 0, 1])
def test_massive_vector_negative_energy_obeys_crossing_convention(
    helicity: int,
) -> None:
    momentum = tuple(Decimal(value) for value in ("13", "4", "8", "8"))
    crossed = tuple(-component for component in momentum)
    mass = Decimal("5")

    assert _massive_vector(crossed, -helicity, mass) == _massive_vector(
        momentum,
        helicity,
        mass,
    )


@pytest.mark.parametrize("helicity", [-1, 0, 1])
def test_massive_vector_is_transverse_for_both_energy_signs(helicity: int) -> None:
    momentum = tuple(Decimal(value) for value in ("13", "4", "8", "8"))
    mass = Decimal("5")

    for candidate, source_helicity in (
        (momentum, helicity),
        (tuple(-component for component in momentum), -helicity),
    ):
        wave = _massive_vector(candidate, source_helicity, mass)
        contraction = (
            candidate[0] * wave[0][0]
            - candidate[1] * wave[1][0]
            - candidate[2] * wave[2][0]
            - candidate[3] * wave[3][0],
            candidate[0] * wave[0][1]
            - candidate[1] * wave[1][1]
            - candidate[2] * wave[2][1]
            - candidate[3] * wave[3][1],
        )
        assert abs(contraction[0]) < Decimal("1e-25")
        assert abs(contraction[1]) < Decimal("1e-25")


def test_source_ir_rejects_inconsistent_statistics() -> None:
    source = BuiltinSMModel()._source_ir(1)

    with pytest.raises(ValueError, match="requires statistics 'fermion'"):
        replace(source, statistics="boson")


def test_source_ir_is_canonical_per_oriented_particle() -> None:
    model = BuiltinSMModel()

    assert model._source_ir(1) is model._source_ir(1)
    assert model._source_ir(-1) is model._source_ir(-1)
    assert model._source_ir(1) is not model._source_ir(-1)


@pytest.mark.parametrize("factor", [True, 1.0, "1"])
def test_crossing_ir_rejects_non_integer_factors(factor: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        CrossingIR(helicity_factor=factor)  # type: ignore[arg-type]


def test_crossing_ir_normalizes_and_rejects_zero_sequence_phase() -> None:
    crossing = CrossingIR(phase=[0, 1])  # type: ignore[arg-type]
    assert crossing.phase == (0.0, 1.0)

    with pytest.raises(ValueError, match="must be nonzero"):
        CrossingIR(phase=[0, 0])  # type: ignore[arg-type]


@pytest.mark.parametrize("dimension", [True, 1.5, "2"])
def test_source_ir_rejects_non_integer_component_dimensions(
    dimension: object,
) -> None:
    source = BuiltinSMModel()._source_ir(1)

    with pytest.raises(TypeError, match="must be an integer"):
        replace(source, component_dimension=dimension)


def test_propagator_ir_records_basis_gauge_and_formula() -> None:
    model = BuiltinSMModel()

    massless_vector = model._propagator_ir(21)
    assert massless_vector.basis == "lorentz-vector"
    assert massless_vector.kind == "vector"
    assert massless_vector.mass_class == "massless"
    assert massless_vector.gauge == "feynman"
    assert massless_vector.numerator == "-i*metric"
    assert massless_vector.denominator == "momentum_squared"

    massive_vector = model._propagator_ir(23)
    assert massive_vector.gauge == "unitary"
    assert massive_vector.mass_class == "massive"
    assert massive_vector.goldstone_policy == "absorbed"
    assert massive_vector.mass_parameter is None
    assert massive_vector.denominator == ("momentum_squared-mass_squared+i*mass*width")

    auxiliary = model._propagator_ir(-21)
    assert auxiliary.applies_propagator is False
    assert auxiliary.kind == "identity"
    assert auxiliary.mass_class == "not-applicable"
    assert auxiliary.basis == "auxiliary:antisymmetric-tensor"
    assert auxiliary.auxiliary_policy == "antisymmetric-tensor"
    assert PropagatorIR.from_json_dict(massive_vector.to_json_dict()) == massive_vector


def test_external_self_conjugate_fermion_source_fails_clearly() -> None:
    majorana = _compiled_particle("neutral_fermion", "neutral_fermion", 810_003)
    model = _external_catalog(majorana)

    with pytest.raises(ValueError, match="unsupported self-conjugate fermion source"):
        model.source_orientation(810_003)


def test_schema_v3_projects_source_relation() -> None:
    model = BuiltinSMModel()
    schema = build_runtime_schema(
        compile_generic_dag(build_process_ir("d d~ > z"), model=model),
        model,
        process_id="source_orientation",
    )
    plan = _execution_plan(schema)

    sources = plan["source_fill"]["sources"]
    by_particle = {source["particle_id"]: source for source in sources}
    assert by_particle[-1]["anti_particle_id"] == 1
    assert by_particle[-1]["source_orientation"] == "antiparticle"
    assert by_particle[1]["anti_particle_id"] == -1
    assert by_particle[1]["source_orientation"] == "particle"
    assert by_particle[23]["anti_particle_id"] == 23
    assert by_particle[23]["source_orientation"] == "self-conjugate"
    assert by_particle[1]["source_basis"] == "weyl-chiral"
    assert by_particle[1]["source_ir"]["identity"]["display_name"] == "d"
    assert by_particle[1]["applied_crossing"] == {
        "momentum_transform": "negate-four-momentum",
        "helicity_factor": 1,
        "chirality_factor": -1,
        "spin_state_factor": -1,
        "phase": [1.0, 0.0],
    }


def test_schema_v3_projects_antiparticle_mass_metadata() -> None:
    model = BuiltinSMModel()
    schema = build_runtime_schema(
        compile_generic_dag(build_process_ir("d u~ > w-"), model=model),
        model,
        process_id="antiparticle_mass",
    )
    plan = _execution_plan(schema)

    source = next(
        item for item in plan["source_fill"]["sources"] if item["particle_id"] == -24
    )
    mass_record = next(item for item in plan["model"]["particles"] if item["pdg"] == 24)

    assert source["anti_particle_id"] == 24
    assert source["source_orientation"] == "antiparticle"
    assert mass_record["mass"] == pytest.approx(80.41900244575616)


@pytest.mark.parametrize(
    ("particle_id", "source_orientation", "expected"),
    (
        (810_001, "antiparticle", _antiquark_weyl),
        (-810_001, "particle", _quark_weyl),
    ),
)
def test_exact_sources_use_orientation_instead_of_pdg_sign(
    particle_id: int,
    source_orientation: str,
    expected: object,
) -> None:
    point = ((Decimal(5), Decimal(3), Decimal(4), Decimal(0)),)
    source = _exact_source(
        particle_id=particle_id,
        anti_particle_id=-particle_id,
        source_orientation=source_orientation,
    )
    schema = {"model": {"particles": []}, "model_parameters": []}

    wave = _source_wavefunction(source, point, schema, ())

    assert callable(expected)
    assert wave == expected(point[0], 1, 1)


def test_exact_source_applies_declared_crossing_phase() -> None:
    point = ((Decimal(5), Decimal(3), Decimal(4), Decimal(0)),)
    source = _exact_source(
        particle_id=810_001,
        anti_particle_id=-810_001,
        source_orientation="particle",
    )
    crossing = {
        "momentum_transform": "negate-four-momentum",
        "helicity_factor": 1,
        "chirality_factor": 1,
        "spin_state_factor": 1,
        "phase": [0.0, 1.0],
    }
    source["side"] = "initial"
    source["crossing"] = "negate-incoming-momentum"
    source["source_ir"]["crossing"] = crossing
    source["applied_crossing"] = crossing
    schema = {"model": {"particles": []}, "model_parameters": []}

    wave = _source_wavefunction(source, point, schema, ())

    crossed_momentum = tuple(-component for component in point[0])
    unphased = _quark_weyl(crossed_momentum, 1, 1)
    assert wave == tuple((-imaginary, real) for real, imaginary in unphased)


def test_exact_source_mass_uses_explicit_antiparticle_relation() -> None:
    schema = {
        "model": {
            "particles": [
                {"pdg": -810_001, "mass": "99"},
                {"pdg": 910_002, "mass_parameter": "MX", "mass": "5"},
            ]
        },
        "model_parameters": [{"name": "MX", "parameter_index": 0}],
    }

    mass = _particle_mass(schema, 810_001, 910_002, (Decimal(7),))

    assert mass == Decimal(7)


def test_exact_self_conjugate_fermion_source_fails_clearly() -> None:
    source = _exact_source(
        particle_id=810_003,
        anti_particle_id=810_003,
        source_orientation="self-conjugate",
    )
    point = ((Decimal(5), Decimal(3), Decimal(4), Decimal(0)),)
    schema = {"model": {"particles": []}, "model_parameters": []}

    with pytest.raises(CompatibilityError, match="self-conjugate fermion source"):
        _source_wavefunction(source, point, schema, ())


def test_exact_source_rejects_inconsistent_orientation_relation() -> None:
    source = _exact_source(
        particle_id=810_003,
        anti_particle_id=810_004,
        source_orientation="self-conjugate",
    )
    point = ((Decimal(5), Decimal(3), Decimal(4), Decimal(0)),)
    schema = {"model": {"particles": []}, "model_parameters": []}

    with pytest.raises(ArtifactError, match="inconsistent with its antiparticle"):
        _source_wavefunction(source, point, schema, ())

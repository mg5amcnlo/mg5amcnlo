# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import pytest

from pyamplicol.color.plan import build_color_plan
from pyamplicol.generation.dag_color import ColorEngine
from pyamplicol.generation.dag_compiler import compile_generic_dag
from pyamplicol.models.base import Model, Particle, QuantumNumberFlow, Vertex
from pyamplicol.models.contracts import (
    CompiledModelIR,
    CompiledParameterRecord,
    CompiledParticleRecord,
)
from pyamplicol.processes.model import (
    ModelParticleCatalog,
    build_model_process_ir,
    expand_model_processes,
)


def _particle(
    name: str,
    antiname: str,
    pdg: int,
    *,
    spin: int,
    color: int,
    mass: str = "ZERO",
    charge: float = 0.0,
    exact_charge: str = "0",
    component_dimension: int | None = None,
    auxiliary_kind: str | None = None,
    ghost_number: int = 0,
    propagating: bool = True,
    goldstoneboson: bool = False,
) -> CompiledParticleRecord:
    return CompiledParticleRecord(
        name=name,
        antiname=antiname,
        pdg_code=pdg,
        spin=spin,
        color=color,
        mass=mass,
        width="ZERO",
        charge=charge,
        quantum_numbers=(("electric_charge", exact_charge),),
        ghost_number=ghost_number,
        propagating=propagating,
        goldstoneboson=goldstoneboson,
        propagator=None,
        component_dimension=component_dimension,
        auxiliary_kind=auxiliary_kind,
    )


def _model(*particles: CompiledParticleRecord) -> CompiledModelIR:
    return CompiledModelIR(
        name="renamed-gauge-model",
        orders=(),
        parameters=(),
        particles=particles,
        couplings=(),
        propagators=(),
        vertex_terms=(),
        oriented_kernels=(),
        direct_contractions=(),
        closure_contractions=(),
    )


class _RecordBackedBoundaryModel(Model):
    def __init__(
        self,
        records: tuple[CompiledParticleRecord, ...],
        *,
        vertices: tuple[Vertex, ...] = (),
        trace_reflection_proven: bool = False,
    ) -> None:
        records_by_name = {record.name: record for record in records}
        self._records = {record.pdg_code: record for record in records}
        self._trace_reflection_proven = trace_reflection_proven
        particles = {
            record.pdg_code: Particle(
                pdg=record.pdg_code,
                anti_pdg=records_by_name[record.antiname].pdg_code,
                spin=record.spin,
                dimension=record.component_dimension
                or {1: 1, 2: 4, 3: 4, 5: 16}.get(record.spin, 1),
                color_rep=record.color,
                charge=record.charge,
            )
            for record in records
        }
        super().__init__(
            name="record-backed-external-model",
            particles=particles,
            vertices=vertices,
        )

    def color_rep(self, pdg: int) -> int:
        return self._records[int(pdg)].color

    def is_fermion(self, pdg: int) -> bool:
        return self._records[int(pdg)].statistics == "fermion"

    def is_chiral_eligible(self, pdg: int) -> bool:
        del pdg
        return False

    def is_fundamental_colored_fermion(self, pdg: int) -> bool:
        record = self._records[int(pdg)]
        return record.statistics == "fermion" and abs(record.color) == 3

    def is_massless_adjoint_vector(self, pdg: int) -> bool:
        record = self._records[int(pdg)]
        return (
            record.wavefunction_family == "vector"
            and record.color_role == "adjoint"
            and record.mass.casefold() == "zero"
        )

    def auxiliary_kind(self, particle_id: int) -> str | None:
        return self._records[int(particle_id)].auxiliary_kind

    def quantum_number_flow(self, particle_id: int) -> QuantumNumberFlow:
        return self._records[int(particle_id)].quantum_numbers

    def lc_trace_reflection_equivalence_is_proven(self, process: object) -> bool:
        del process
        return self._trace_reflection_proven


def test_external_process_roles_do_not_depend_on_sm_names_or_pdgs() -> None:
    model = _model(
        _particle("chi", "chi_bar", 810_001, spin=2, color=3),
        _particle("chi_bar", "chi", -810_001, spin=2, color=-3),
        _particle("octet_vector", "octet_vector", 910_101, spin=3, color=8),
        _particle("neutral_scalar", "neutral_scalar", 710_001, spin=1, color=1),
    )

    process = build_model_process_ir(
        "chi chi_bar > neutral_scalar octet_vector",
        model,
    )

    assert process.outgoing_pdgs == (-810_001, 810_001, 710_001, 910_101)
    assert process.fundamental_labels == (2,)
    assert process.antifundamental_labels == (1,)
    assert process.adjoint_labels == (4,)
    assert process.singlet_labels == (3,)
    assert process.color_endpoints.pair_count == 1
    assert tuple(leg.statistics for leg in process.legs) == (
        "fermion",
        "fermion",
        "boson",
        "boson",
    )
    assert tuple(leg.wavefunction_family for leg in process.legs) == (
        "fermion",
        "fermion",
        "scalar",
        "vector",
    )
    assert tuple(leg.color_role for leg in process.legs) == (
        "antifundamental",
        "fundamental",
        "singlet",
        "adjoint",
    )
    assert tuple(leg.source_orientation for leg in process.legs) == (
        "antiparticle",
        "particle",
        "self-conjugate",
        "self-conjugate",
    )
    by_name = {particle.name: particle for particle in model.particles}
    assert by_name["chi"].statistics == "fermion"
    assert by_name["chi"].color_role == "fundamental"
    assert by_name["chi"].source_orientation == "particle"
    assert by_name["chi_bar"].source_orientation == "antiparticle"
    assert by_name["octet_vector"].wavefunction_family == "vector"
    assert by_name["octet_vector"].self_conjugate is True


def test_default_parton_aliases_are_derived_from_particle_metadata() -> None:
    particles = (
        _particle("octet_vector", "octet_vector", 910_101, spin=3, color=8),
        _particle("chi", "chi_bar", 810_001, spin=2, color=3),
        _particle("chi_bar", "chi", -810_001, spin=2, color=-3),
        _particle(
            "heavy_colored",
            "heavy_colored_bar",
            810_002,
            spin=2,
            color=3,
            mass="MHEAVY",
        ),
        _particle(
            "restricted_massless",
            "restricted_massless_bar",
            810_003,
            spin=2,
            color=3,
            mass="MRESTRICTED",
        ),
        _particle(
            "mutable_zero",
            "mutable_zero_bar",
            810_004,
            spin=2,
            color=3,
            mass="MMUTABLE",
        ),
        _particle("singlet", "singlet", 710_001, spin=1, color=1),
    )
    parameters = (
        CompiledParameterRecord(
            name="MRESTRICTED",
            nature="internal",
            parameter_type="real",
            value=None,
            expression="0",
            resolved_expression="0",
            lhablock=None,
            lhacode=(),
        ),
        CompiledParameterRecord(
            name="MMUTABLE",
            nature="external",
            parameter_type="real",
            value=(0.0, 0.0),
            expression=None,
            resolved_expression="0",
            lhablock="MASS",
            lhacode=(810_004,),
        ),
    )

    aliases = ModelParticleCatalog(
        "arbitrary-model-name",
        particles,
        parameters,
    ).default_multiparticles()

    assert aliases == {
        "p": ("octet_vector", "chi", "chi_bar", "restricted_massless"),
        "j": ("octet_vector", "chi", "chi_bar", "restricted_massless"),
        "all": (
            "octet_vector",
            "chi",
            "chi_bar",
            "heavy_colored",
            "restricted_massless",
            "mutable_zero",
            "singlet",
        ),
    }


def test_all_alias_keeps_only_physical_external_particles_in_order() -> None:
    physical = _particle("physical", "physical", 710_101, spin=1, color=1)
    ghost = _particle("ghost", "ghost", 710_102, spin=1, color=1, ghost_number=1)
    goldstone = _particle(
        "goldstone", "goldstone", 710_103, spin=1, color=1, goldstoneboson=True
    )
    nonpropagating = _particle(
        "nonpropagating",
        "nonpropagating",
        710_104,
        spin=1,
        color=1,
        propagating=False,
    )
    auxiliary = _particle(
        "auxiliary",
        "auxiliary",
        710_105,
        spin=1,
        color=1,
        auxiliary_kind="synthetic-contact-current",
    )
    second = _particle("second", "second", 710_106, spin=1, color=1)

    catalog = ModelParticleCatalog(
        "external-filter-model",
        (physical, ghost, goldstone, nonpropagating, auxiliary, second),
    )

    assert tuple(particle.name for particle in catalog.external_particles) == (
        "physical",
        "second",
    )
    assert catalog.default_multiparticles() == {"all": ("physical", "second")}


def test_explicit_multiparticles_extend_defaults_and_can_override_all() -> None:
    particles = (
        _particle("parton", "parton", 810_101, spin=3, color=8),
        _particle("first", "first", 710_101, spin=1, color=1),
        _particle("second", "second", 710_102, spin=1, color=1),
    )
    catalog = ModelParticleCatalog("override-model", particles)

    assert expand_model_processes(
        "beam beam > all",
        catalog,
        multiparticles={"beam": ("parton",)},
    ) == (
        "parton parton > parton",
        "parton parton > first",
        "parton parton > second",
    )
    assert expand_model_processes(
        "p p > all",
        catalog,
        multiparticles={"all": ("second",)},
    ) == ("parton parton > second",)


@pytest.mark.parametrize(
    ("particle", "process"),
    (
        (
            _particle(
                "fundamental_scalar",
                "fundamental_scalar_bar",
                710_011,
                spin=1,
                color=3,
            ),
            "fundamental_scalar fundamental_scalar_bar > neutral",
        ),
        (
            _particle("adjoint_scalar", "adjoint_scalar", 710_012, spin=1, color=8),
            "adjoint_scalar adjoint_scalar > neutral",
        ),
    ),
)
def test_colored_spin_does_not_imply_an_sm_particle_family(
    particle: CompiledParticleRecord,
    process: str,
) -> None:
    antiparticle = (
        particle
        if particle.name == particle.antiname
        else _particle(
            particle.antiname,
            particle.name,
            -particle.pdg_code,
            spin=particle.spin,
            color=-particle.color,
        )
    )
    model = _model(
        particle,
        *(() if antiparticle is particle else (antiparticle,)),
        _particle("neutral", "neutral", 710_013, spin=1, color=1),
    )

    with pytest.raises(ValueError, match="unsupported colored external-state role"):
        build_model_process_ir(process, model)


def test_sm_like_pdgs_and_fractional_charges_use_only_external_metadata() -> None:
    fundamental = _particle(
        "fundamental_21",
        "antifundamental_21",
        21,
        spin=2,
        color=3,
        charge=0.2,
        exact_charge="1/5",
    )
    antifundamental = _particle(
        "antifundamental_21",
        "fundamental_21",
        -21,
        spin=2,
        color=-3,
        charge=-0.2,
        exact_charge="-1/5",
    )
    adjoint = _particle("adjoint_1", "adjoint_1", 1, spin=3, color=8)
    scalar = _particle(
        "scalar_24",
        "antiscalar_24",
        24,
        spin=1,
        color=1,
        charge=0.2,
        exact_charge="1/5",
    )
    antiscalar = _particle(
        "antiscalar_24",
        "scalar_24",
        -24,
        spin=1,
        color=1,
        charge=-0.2,
        exact_charge="-1/5",
    )
    unrelated_auxiliary = _particle(
        "unrelated_auxiliary_99",
        "unrelated_auxiliary_99",
        99,
        spin=-1,
        color=1,
        charge=0.2,
    )
    model = _model(
        fundamental,
        antifundamental,
        adjoint,
        scalar,
        antiscalar,
        unrelated_auxiliary,
    )
    catalog = ModelParticleCatalog(model.name, model.particles)

    process = build_model_process_ir(
        "fundamental_21 antifundamental_21 > adjoint_1 scalar_24",
        model,
    )

    assert process.outgoing_pdgs == (-21, 21, 1, 24)
    assert process.antifundamental_labels == (1,)
    assert process.fundamental_labels == (2,)
    assert process.adjoint_labels == (3,)
    assert process.singlet_labels == (4,)
    assert fundamental.charge == pytest.approx(0.2)
    assert scalar.charge == pytest.approx(0.2)
    assert catalog.default_multiparticles() == {
        "p": ("fundamental_21", "antifundamental_21", "adjoint_1"),
        "j": ("fundamental_21", "antifundamental_21", "adjoint_1"),
        "all": (
            "fundamental_21",
            "antifundamental_21",
            "adjoint_1",
            "scalar_24",
            "antiscalar_24",
        ),
    }
    assert unrelated_auxiliary.statistics == "auxiliary"
    assert unrelated_auxiliary not in catalog.external_particles
    with pytest.raises(ValueError, match="is not an external state"):
        catalog.resolve(unrelated_auxiliary.name)

    auxiliary_vertex = Vertex(701, (21, -21, 99))
    boundary_model = _RecordBackedBoundaryModel(
        model.particles,
        vertices=(auxiliary_vertex,),
    )
    color_engine = ColorEngine(build_color_plan(process), boundary_model)

    assert color_engine._particle_has_colour(99) is False
    assert color_engine.vertex_allowed(auxiliary_vertex) is True


def test_pure_adjoint_lc_reflection_folding_requires_explicit_model_proof() -> None:
    adjoint = _particle(
        "external_adjoint",
        "external_adjoint",
        910_101,
        spin=3,
        color=8,
    )
    model_ir = _model(adjoint)
    process = build_model_process_ir(
        "external_adjoint external_adjoint > "
        "external_adjoint external_adjoint external_adjoint",
        model_ir,
    )
    default_plan = build_color_plan(process)

    unproven = compile_generic_dag(
        process,
        model=_RecordBackedBoundaryModel(model_ir.particles),
    )
    proven = compile_generic_dag(
        process,
        model=_RecordBackedBoundaryModel(
            model_ir.particles,
            trace_reflection_proven=True,
        ),
    )

    assert default_plan.trace_reflections_folded is False
    assert default_plan.sector_count == 24
    assert unproven.color_plan.trace_reflections_folded is False
    assert unproven.color_plan.sector_count == 24
    assert proven.color_plan.trace_reflections_folded is True
    assert proven.color_plan.sector_count == 12

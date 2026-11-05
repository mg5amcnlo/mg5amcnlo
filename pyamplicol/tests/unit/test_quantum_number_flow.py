# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from pyamplicol.generation.artifact_writer import _current_storage
from pyamplicol.generation.dag_types import ColorState, CurrentIndex, CurrentNode
from pyamplicol.generation.runtime_schema import _current_slots
from pyamplicol.models.base import Model, QuantumFlow
from pyamplicol.models.builtin.model import BuiltinSMModel
from pyamplicol.models.compiler_records import _particle
from pyamplicol.models.contracts import (
    CompiledModelIR,
    CompiledParticleRecord,
    validate_quantum_number_flow,
)
from pyamplicol.models.external_catalog import ExternalModelCatalogMixin


def _compiled_particle(
    *,
    name: str,
    antiname: str,
    pdg_code: int,
    quantum_numbers: tuple[tuple[str, str], ...],
) -> CompiledParticleRecord:
    return CompiledParticleRecord(
        name=name,
        antiname=antiname,
        pdg_code=pdg_code,
        spin=1,
        color=1,
        mass="ZERO",
        width="ZERO",
        charge=0.0,
        quantum_numbers=quantum_numbers,
        ghost_number=0,
        propagating=True,
        goldstoneboson=False,
        propagator=None,
    )


def _compiled_model(
    *particles: CompiledParticleRecord,
) -> CompiledModelIR:
    return CompiledModelIR(
        name="quantum-number-test",
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


def test_loader_charge_uses_the_exact_float_ratio() -> None:
    record = _particle(
        {
            "name": "fractional",
            "antiname": "anti-fractional",
            "pdg_code": 101,
            "spin": 1,
            "color": 1,
            "mass": "ZERO",
            "width": "ZERO",
            "charge": 0.2,
        }
    )
    numerator, denominator = (0.2).as_integer_ratio()
    assert record.quantum_numbers == (
        ("electric_charge", f"{numerator}/{denominator}"),
    )


@pytest.mark.parametrize(
    ("flow", "message"),
    [
        (
            (("second", "0"), ("first", "0")),
            "sorted and unique",
        ),
        (
            (("duplicate", "0"), ("duplicate", "0")),
            "sorted and unique",
        ),
        ((("symbolic", "x"),), "symbol-free"),
        ((("symbolic-function", "unknown(1)"),), "symbol-free"),
        ((("complex", "sqrt(-1)"),), "must be real"),
        # Symbolica identifies infinity as non-real before numeric evaluation.
        ((("infinite", "log(0)"),), "must be real"),
    ],
)
def test_quantum_number_metadata_rejects_noncanonical_constants(
    flow: tuple[tuple[str, str], ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_quantum_number_flow(flow)


def test_quantum_number_metadata_accepts_large_finite_exact_constants() -> None:
    flow = (("large_exact_number", "exp(1000000)"),)
    assert validate_quantum_number_flow(flow) == flow


@pytest.mark.parametrize("expression", ("pi", "sqrt(2)"))
def test_quantum_number_metadata_accepts_builtin_exact_constants(
    expression: str,
) -> None:
    flow = (("exact_number", expression),)
    assert validate_quantum_number_flow(flow) == flow


def test_compiled_particle_quantum_numbers_obey_anti_relations() -> None:
    particle = _compiled_particle(
        name="fractional",
        antiname="anti-fractional",
        pdg_code=101,
        quantum_numbers=(("electric_charge", "1/5"),),
    )
    antiparticle = _compiled_particle(
        name="anti-fractional",
        antiname="fractional",
        pdg_code=-101,
        quantum_numbers=(("electric_charge", "-1/5"),),
    )
    model = _compiled_model(particle, antiparticle)
    assert CompiledModelIR.from_dict(model.to_dict()) == model

    inconsistent = _compiled_particle(
        name="anti-fractional",
        antiname="fractional",
        pdg_code=-101,
        quantum_numbers=(("electric_charge", "-1/4"),),
    )
    with pytest.raises(ValueError, match="exactly negated"):
        _compiled_model(particle, inconsistent)


def test_self_conjugate_particles_require_zero_quantum_numbers() -> None:
    with pytest.raises(ValueError, match="must declare exact electric_charge"):
        _compiled_particle(
            name="missing-charge",
            antiname="missing-charge",
            pdg_code=99,
            quantum_numbers=(),
        )

    neutral = _compiled_particle(
        name="neutral",
        antiname="neutral",
        pdg_code=100,
        quantum_numbers=(("electric_charge", "0"),),
    )
    assert _compiled_model(neutral).particles == (neutral,)

    charged = _compiled_particle(
        name="charged-self-conjugate",
        antiname="charged-self-conjugate",
        pdg_code=102,
        quantum_numbers=(("electric_charge", "1"),),
    )
    with pytest.raises(ValueError, match="must have zero"):
        _compiled_model(charged)


def test_model_flow_fails_closed_and_builtin_sm_is_exact() -> None:
    with pytest.raises(NotImplementedError, match="exact quantum-number flow"):
        Model(name="unspecified").quantum_number_flow(1)

    model = BuiltinSMModel()
    assert model.quantum_number_flow(2) == (("electric_charge", "2/3"),)
    assert model.quantum_number_flow(-2) == (("electric_charge", "-2/3"),)
    assert model.quantum_number_flow(1) == (("electric_charge", "-1/3"),)
    assert model.quantum_number_flow(-1) == (("electric_charge", "1/3"),)
    assert model.quantum_number_flow(-12) == (("electric_charge", "0"),)
    assert model.quantum_number_flow(21) == (("electric_charge", "0"),)
    assert model.quantum_number_flow(26) == (("electric_charge", "1"),)
    assert model.quantum_number_flow(-26) == (("electric_charge", "-1"),)
    assert model.quantum_number_flow(125) == (("electric_charge", "0"),)


def test_external_catalog_consumes_compiled_quantum_numbers() -> None:
    record = _compiled_particle(
        name="fractional",
        antiname="anti-fractional",
        pdg_code=101,
        quantum_numbers=(("electric_charge", "1/5"),),
    )
    catalog = ExternalModelCatalogMixin()
    catalog._particle_records_by_pdg = {101: record}
    assert catalog.quantum_number_flow(101) == record.quantum_numbers


def test_current_identity_and_diagnostics_retain_exact_flow() -> None:
    first = CurrentIndex(
        particle_id=101,
        external_mask=1,
        external_labels=(1,),
        helicity_ancestry=1,
        chirality=0,
        spin_state=0,
        flavour_flow=(101,),
        quantum_number_flow=(("electric_charge", "1/5"),),
        color_state=ColorState("lc"),
        momentum_mask=1,
    )
    second = CurrentIndex(
        particle_id=101,
        external_mask=1,
        external_labels=(1,),
        helicity_ancestry=1,
        chirality=0,
        spin_state=0,
        flavour_flow=(101,),
        quantum_number_flow=(("electric_charge", "1/7"),),
        color_state=ColorState("lc"),
        momentum_mask=1,
    )
    assert len({first, second}) == 2
    assert first.to_json_dict()["quantum_number_flow"] == [["electric_charge", "1/5"]]
    flow = QuantumFlow(
        chirality=0,
        spin_state=0,
        flavour_flow=(101,),
        quantum_number_flow=(("electric_charge", "1/5"),),
        coupling=(1.0, 0.0),
    )
    assert flow.quantum_number_flow == first.quantum_number_flow

    current = CurrentNode(id=0, index=first, dimension=1, is_source=True)
    full_storage = {
        "component_count": 1,
        "number_type": "complex",
        "current_slots": _current_slots(SimpleNamespace(currents=(current,))),
    }
    assert full_storage["current_slots"][0]["quantum_number_flow"] == [
        ["electric_charge", "1/5"]
    ]
    assert full_storage["current_slots"][0]["helicity_ancestry"] == "0x1"
    wide_ancestry = 10**4300
    wide_current = replace(
        current,
        index=replace(first, helicity_ancestry=wide_ancestry),
    )
    wide_slot = _current_slots(SimpleNamespace(currents=(wide_current,)))[0]
    assert int(wide_slot["helicity_ancestry"], 16) == wide_ancestry
    compact_storage = _current_storage(full_storage)
    assert "quantum_number_flow" not in compact_storage["current_slots"][0]
    assert "charge_flow" not in compact_storage["current_slots"][0]

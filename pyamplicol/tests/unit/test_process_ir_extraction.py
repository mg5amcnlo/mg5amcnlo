# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from pyamplicol.models.builtin.process_ir import build_process_ir
from pyamplicol.models.contracts import CompiledModelIR, CompiledParticleRecord
from pyamplicol.processes.model import build_model_process_ir


def _record(
    name: str,
    antiname: str,
    pdg: int,
    spin: int,
    color: int,
) -> CompiledParticleRecord:
    return CompiledParticleRecord(
        name=name,
        antiname=antiname,
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


def test_process_ir_preserves_physical_and_all_outgoing_order() -> None:
    ir = build_process_ir("d d~ > Z g")

    assert ir.process == "d d~ > z g"
    assert ir.key == "d_dbar_to_z_g"
    assert tuple(leg.particle for leg in ir.legs) == ("d", "d~", "z", "g")
    assert tuple(leg.outgoing_particle for leg in ir.legs) == (
        "d~",
        "d",
        "z",
        "g",
    )
    assert ir.initial_pdgs == (1, -1)
    assert ir.final_pdgs == (23, 21)
    assert ir.outgoing_pdgs == (-1, 1, 23, 21)
    assert tuple(leg.statistics for leg in ir.legs) == (
        "fermion",
        "fermion",
        "boson",
        "boson",
    )
    assert tuple(leg.wavefunction_family for leg in ir.legs) == (
        "fermion",
        "fermion",
        "vector",
        "vector",
    )
    assert tuple(leg.color_role for leg in ir.legs) == (
        "antifundamental",
        "fundamental",
        "singlet",
        "adjoint",
    )
    assert tuple(leg.source_orientation for leg in ir.legs) == (
        "antiparticle",
        "particle",
        "self-conjugate",
        "self-conjugate",
    )
    assert ir.fundamental_labels == (2,)
    assert ir.antifundamental_labels == (1,)
    assert ir.adjoint_labels == (4,)
    assert ir.singlet_labels == (3,)
    assert ir.color_endpoints.fundamental_count == 1
    assert ir.color_endpoints.antifundamental_count == 1
    assert ir.color_endpoints.pair_count == 1


def test_process_external_model_uses_the_same_crossing_contract() -> None:
    particles = (
        _record("state_f", "state_af", 810_001, 2, 3),
        _record("state_af", "state_f", -810_001, 2, -3),
        _record("state_s", "state_s", 710_001, 3, 1),
        _record("state_a", "state_a", 910_101, 3, 8),
    )
    model = CompiledModelIR(
        name="opaque-role-model",
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

    ir = build_model_process_ir(
        "state_f state_af > state_s state_a",
        model,
    )

    assert ir.process == "state_f state_af > state_s state_a"
    assert ir.outgoing_pdgs == (-810_001, 810_001, 710_001, 910_101)
    assert tuple(leg.label for leg in ir.legs) == (1, 2, 3, 4)
    assert tuple(leg.statistics for leg in ir.legs) == (
        "fermion",
        "fermion",
        "boson",
        "boson",
    )
    assert tuple(leg.wavefunction_family for leg in ir.legs) == (
        "fermion",
        "fermion",
        "vector",
        "vector",
    )
    assert tuple(leg.color_role for leg in ir.legs) == (
        "antifundamental",
        "fundamental",
        "singlet",
        "adjoint",
    )
    assert tuple(leg.source_orientation for leg in ir.legs) == (
        "antiparticle",
        "particle",
        "self-conjugate",
        "self-conjugate",
    )
    assert ir.fundamental_labels == (2,)
    assert ir.antifundamental_labels == (1,)
    assert ir.adjoint_labels == (4,)
    assert ir.singlet_labels == (3,)
    assert ir.color_endpoints.pair_count == 1


def test_process_json_exposes_only_structural_roles() -> None:
    payload = build_process_ir("d d~ > Z g").to_json_dict()

    assert payload["color_endpoints"] == {
        "fundamental_count": 1,
        "antifundamental_count": 1,
        "pair_count": 1,
        "balanced": True,
    }
    assert payload["color_role_labels"] == {
        "fundamental": [2],
        "antifundamental": [1],
        "adjoint": [4],
        "singlet": [3],
    }
    assert "quark_lines" not in payload
    assert "labels" not in payload
    assert all("particle_class" not in leg for leg in payload["legs"])
    assert all(
        {
            "statistics",
            "wavefunction_family",
            "color_role",
            "source_orientation",
        }
        <= leg.keys()
        for leg in payload["legs"]
    )

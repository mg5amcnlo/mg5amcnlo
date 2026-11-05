# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from fractions import Fraction

import pytest

from pyamplicol._internal.physics.symbols import symbols
from pyamplicol.models.builtin.adapters import build_model_payload
from pyamplicol.models.builtin.compiler import compile_model_ir as compile_builtin_ir
from pyamplicol.models.compiler_contractions import compile_contraction_records
from pyamplicol.models.compiler_entry import (
    _compile_four_point_contact_kernels,
    compile_ufo_model_ir,
)
from pyamplicol.models.compiler_records import _particle
from pyamplicol.models.contracts import (
    CompiledModelIR,
    CompiledParticleRecord,
    CompiledVertexTerm,
    compiled_particle_component_dimension,
)


def _particle_payload(name: str = "phi") -> dict[str, object]:
    return {
        "name": name,
        "antiname": name,
        "pdg_code": 9_500_001,
        "spin": 1,
        "color": 1,
        "mass": "ZERO",
        "width": "ZERO",
        "charge": 0.0,
        "quantum_numbers": [["electric_charge", "0"]],
        "ghost_number": 0,
        "propagating": True,
        "goldstoneboson": False,
        "propagator": None,
    }


def _compiled_particle(
    *,
    name: str = "phi",
    pdg_code: int = 9_500_001,
    component_dimension: object = None,
) -> CompiledParticleRecord:
    return CompiledParticleRecord(
        name=name,
        antiname=name,
        pdg_code=pdg_code,
        spin=1,
        color=1,
        mass="ZERO",
        width="ZERO",
        charge=0.0,
        quantum_numbers=(("electric_charge", "0"),),
        ghost_number=0,
        propagating=True,
        goldstoneboson=False,
        propagator=None,
        component_dimension=component_dimension,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("charge", "exact"),
    [("2/3", "2/3"), ("-2/6", "-1/3"), ("0", "0"), (1, "1"), (0.5, "1/2")],
)
def test_loader_particle_charge_accepts_exact_rationals_and_numbers(
    charge: object, exact: str,
) -> None:
    payload = _particle_payload()
    payload.pop("quantum_numbers")
    payload.update(charge=charge, y_charge="-1", y_charge_right="-2")

    particle = _particle(payload)

    assert particle.charge == pytest.approx(float(Fraction(exact)))
    # Chiral hypercharges are not conserved charges of the whole Dirac particle.
    assert particle.quantum_numbers == (("electric_charge", exact),)


def test_loader_particle_charge_preserves_explicit_flow_metadata() -> None:
    payload = _particle_payload()
    payload.update(charge="2/3", quantum_numbers=[["electric_charge", "2/3"]])

    particle = _particle(payload)

    assert particle.charge == pytest.approx(2 / 3)
    assert particle.quantum_numbers == (("electric_charge", "2/3"),)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("component_dimension", 6),
        ("component_dimension", None),
        ("auxiliary_kind", "antisymmetric-tensor"),
        ("auxiliary_kind", None),
    ],
)
def test_untrusted_particle_rejects_compiler_owned_metadata(
    field: str,
    value: object,
) -> None:
    payload = _particle_payload()
    payload[field] = value

    with pytest.raises(ValueError, match=rf"compiler-owned metadata.*{field}"):
        _particle(payload)


def test_untrusted_particle_cannot_self_mark_metadata_as_builtin() -> None:
    payload = _particle_payload()
    payload.update(
        {
            "component_dimension": 6,
            "auxiliary_kind": "antisymmetric-tensor",
        }
    )
    model_payload = {
        "name": "spoofed-builtin-marker",
        "builtin_model": True,
        "particles": [payload],
    }

    with pytest.raises(ValueError, match="compiler-owned metadata in untrusted input"):
        compile_ufo_model_ir(model_payload)


@pytest.mark.parametrize("dimension", [True, False, 1.0, "1", object()])
def test_compiled_particle_rejects_non_integer_component_dimension(
    dimension: object,
) -> None:
    with pytest.raises(TypeError, match="component dimension must be an integer"):
        _compiled_particle(component_dimension=dimension)


@pytest.mark.parametrize("dimension", [0, -1])
def test_compiled_particle_rejects_non_positive_component_dimension(
    dimension: int,
) -> None:
    with pytest.raises(ValueError, match="component dimension must be positive"):
        _compiled_particle(component_dimension=dimension)


def test_trusted_builtin_particle_metadata_is_preserved() -> None:
    model_payload, _parameter_defaults = build_model_payload()
    compiled = compile_builtin_ir(model_payload)
    particle = next(item for item in compiled.particles if item.pdg_code == -21)

    assert particle.component_dimension == 6
    assert particle.auxiliary_kind == "antisymmetric-tensor"
    assert compiled_particle_component_dimension(particle) == 6
    assert particle.statistics == "auxiliary"
    direct = tuple(
        record
        for record in compiled.direct_contractions
        if record.left_particle == particle.name
        or record.right_particle == particle.name
    )
    assert tuple(record.contraction_ir.name for record in direct) == (
        "antisymmetric-tensor",
    )

    round_trip = CompiledModelIR.from_dict(compiled.to_dict())
    restored = next(item for item in round_trip.particles if item.pdg_code == -21)
    assert restored.component_dimension == 6
    assert restored.auxiliary_kind == "antisymmetric-tensor"
    assert round_trip.direct_contractions == compiled.direct_contractions


def test_generic_contractions_ignore_builtin_auxiliary_tags() -> None:
    particle = CompiledParticleRecord(
        name="model_auxiliary",
        antiname="model_auxiliary",
        pdg_code=9_700_001,
        spin=-1,
        color=1,
        mass="ZERO",
        width="ZERO",
        charge=0.0,
        quantum_numbers=(("electric_charge", "0"),),
        ghost_number=0,
        propagating=False,
        goldstoneboson=False,
        propagator=None,
        component_dimension=6,
        auxiliary_kind="antisymmetric-tensor",
    )

    direct, closure = compile_contraction_records((particle,), (), ())

    assert direct == ()
    assert closure == ()


def test_compiler_generated_contact_auxiliary_keeps_owned_metadata() -> None:
    names = ("a", "b", "c", "d")
    particles = tuple(
        _compiled_particle(name=name, pdg_code=9_600_000 + index)
        for index, name in enumerate(names)
    )
    term = CompiledVertexTerm(
        id=902,
        vertex="V_scalar_contact",
        particles=names,
        color_index=0,
        lorentz_index=0,
        color_source="1",
        color_expression="1",
        lorentz_name="L_scalar_contact",
        lorentz_source="1",
        lorentz_expression="1",
        coupling="GC_scalar_contact",
        coupling_expression="1",
        coupling_orders=(),
    )

    auxiliaries, kernels = _compile_four_point_contact_kernels(
        (term,),
        particles,
        start_kind=0,
        model_symbols=symbols.model("trusted-contact-metadata"),
    )

    assert auxiliaries
    assert kernels
    assert all(particle.component_dimension == 1 for particle in auxiliaries)
    assert all(
        compiled_particle_component_dimension(particle) == 1 for particle in auxiliaries
    )
    assert all(
        particle.auxiliary_kind is not None
        and particle.auxiliary_kind.startswith("ufo-contact:")
        for particle in auxiliaries
    )

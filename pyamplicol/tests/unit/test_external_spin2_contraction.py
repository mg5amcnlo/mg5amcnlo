# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from pyamplicol.generation.stage_expressions import _amplitude_root_expression
from pyamplicol.models.base import Model
from pyamplicol.models.compiler_contractions import compile_contraction_records
from pyamplicol.models.contracts import CompiledParticleRecord


def _spin2_particle() -> CompiledParticleRecord:
    return CompiledParticleRecord(
        name="tensor",
        antiname="tensor",
        pdg_code=910_005,
        spin=5,
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


def test_external_spin2_direct_contraction_uses_rank_two_lorentz_metric() -> None:
    direct, closure = compile_contraction_records((_spin2_particle(),), (), ())

    assert closure == ()
    assert len(direct) == 1
    record = direct[0]
    assert record.selector == ("tensor", 0, "tensor", 0)
    assert record.contraction_ir.name == "lorentz-rank-2"
    assert record.contraction_ir.left_basis == "lorentz-rank-2"
    assert record.contraction_ir.right_basis == "lorentz-rank-2"
    assert record.contraction_ir.metric_signature == (
        "mostly-minus-tensor-product"
    )

    metric = (1.0, -1.0, -1.0, -1.0)
    expected_coefficients = tuple(
        (metric[mu] * metric[nu], 0.0)
        for mu in range(4)
        for nu in range(4)
    )
    assert record.contraction_ir.coefficients == expected_coefficients

    left = tuple(float(index + 1) for index in range(16))
    right = (1.0,) * 16
    root = {
        "kind": "direct-contraction",
        "left_value_slot": {
            "value_slot_id": 0,
            "component_start": 0,
            "component_stop": 16,
            "dimension": 16,
        },
        "right_value_slot": {
            "value_slot_id": 1,
            "component_start": 16,
            "component_stop": 32,
            "dimension": 16,
        },
        "coupling": [1.0, 0.0],
        "color_weight": [1.0, 0.0],
        "contraction": record.contraction_ir.name,
        "contraction_ir": record.contraction_ir.to_json_dict(),
    }
    value_slots = {
        0: {
            "value_slot_id": 0,
            "dimension": 16,
            "chirality": 0,
            "propagator": {"basis": "lorentz-rank-2"},
        },
        1: {
            "value_slot_id": 1,
            "dimension": 16,
            "chirality": 0,
            "propagator": {"basis": "lorentz-rank-2"},
        },
    }

    result = _amplitude_root_expression(
        Model(name="external-spin2-contraction-probe"),
        root,
        value_symbols={0: left, 1: right},
        model_parameter_symbols={},
        value_slots=value_slots,
    )

    assert result == sum(
        coefficient[0] * component
        for coefficient, component in zip(
            expected_coefficients,
            left,
            strict=True,
        )
    )

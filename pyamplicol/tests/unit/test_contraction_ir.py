# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import FrozenInstanceError, replace
from typing import Any

import pytest
from symbolica import S

from pyamplicol.generation.artifact_writer import _execution_plan
from pyamplicol.generation.dag_algorithms import prune_dag_to_amplitude_roots
from pyamplicol.generation.dag_compiler import compile_generic_dag
from pyamplicol.generation.runtime_schema import build_runtime_schema
from pyamplicol.generation.stage_compiler import (
    build_generic_stage_compiler_blueprint,
)
from pyamplicol.generation.stage_expressions import _amplitude_root_expression
from pyamplicol.generation.stage_parameters import _contract_components
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models._physics_ir import ContractionIR
from pyamplicol.models.base import Model, Particle
from pyamplicol.models.builtin.process_ir import build_process_ir
from pyamplicol.models.compiler_contractions import compile_contraction_records
from pyamplicol.models.contracts import CompiledModelIR, CompiledParticleRecord
from pyamplicol.models.external import CompiledUFOModel
from pyamplicol.models.loading import CompiledModel


def test_contraction_ir_json_round_trip_and_strict_decoder() -> None:
    contraction_ir = ContractionIR(
        name="weyl",
        left_basis="weyl-chiral",
        right_basis="weyl-chiral",
        coefficients=((1.0, 0.0), (1.0, 0.0)),
        chirality_relation="opposite",
        metric_signature=None,
    )
    payload = contraction_ir.to_json_dict()

    assert ContractionIR.from_json_dict(payload) == contraction_ir
    with pytest.raises(FrozenInstanceError):
        contraction_ir.name = "mutated"  # type: ignore[misc]

    unknown = {**payload, "inferred_from_dimension": True}
    with pytest.raises(ValueError, match="unknown fields"):
        ContractionIR.from_json_dict(unknown)

    missing_nullable = dict(payload)
    del missing_nullable["metric_signature"]
    with pytest.raises(ValueError, match="missing required fields"):
        ContractionIR.from_json_dict(missing_nullable)

    malformed_pair = {**payload, "coefficients": [[1.0], [1.0, 0.0]]}
    with pytest.raises(ValueError, match="must have two components"):
        ContractionIR.from_json_dict(malformed_pair)

    all_zero = {**payload, "coefficients": [[0.0, 0.0], [0.0, 0.0]]}
    with pytest.raises(ValueError, match="nonzero component"):
        ContractionIR.from_json_dict(all_zero)

    nonfinite = {**payload, "coefficients": [[float("nan"), 0.0]]}
    with pytest.raises(ValueError, match="finite complex pairs"):
        ContractionIR.from_json_dict(nonfinite)


def test_builtin_model_owns_exact_contraction_coefficients_and_bases() -> None:
    model = BuiltinSMModel()

    scalar = model.direct_contraction_ir(25, 25)
    assert scalar == ContractionIR(
        "scalar",
        "scalar",
        "scalar",
        ((1.0, 0.0),),
        "any",
        None,
    )

    weyl = model.direct_contraction_ir(
        1,
        -1,
        left_chirality=1,
        right_chirality=-1,
    )
    assert weyl == ContractionIR(
        "weyl",
        "weyl-chiral",
        "weyl-chiral",
        ((1.0, 0.0), (1.0, 0.0)),
        "opposite",
        None,
    )
    assert (
        model.direct_contraction_ir(
            1,
            -1,
            left_chirality=1,
            right_chirality=-1,
        )
        is weyl
    )
    assert (
        model.direct_contraction_ir(
            1,
            -1,
            left_chirality=1,
            right_chirality=1,
        )
        is None
    )

    dirac = model.direct_contraction_ir(6, -6)
    assert dirac is not None
    assert dirac.name == "dirac"
    assert dirac.left_basis == dirac.right_basis == "dirac"
    assert dirac.coefficients == ((1.0, 0.0),) * 4
    assert dirac.chirality_relation == "any"

    lorentz = model.direct_contraction_ir(22, 22)
    assert lorentz is not None
    assert lorentz.name == "lorentz"
    assert lorentz.left_basis == lorentz.right_basis == "lorentz-vector"
    assert lorentz.coefficients == (
        (1.0, 0.0),
        (-1.0, 0.0),
        (-1.0, 0.0),
        (-1.0, 0.0),
    )
    assert lorentz.metric_signature == "mostly-minus"

    auxiliary_vector = model.direct_contraction_ir(99, 99)
    assert auxiliary_vector is not None
    assert auxiliary_vector.name == "lorentz"
    assert auxiliary_vector.left_basis == ("auxiliary:u1-subtraction-color-flow-vector")
    assert auxiliary_vector.coefficients == lorentz.coefficients

    antisymmetric = model.direct_contraction_ir(-21, -21)
    assert antisymmetric is not None
    assert antisymmetric.name == "antisymmetric-tensor"
    assert antisymmetric.left_basis == "auxiliary:antisymmetric-tensor"
    assert antisymmetric.right_basis == "auxiliary:antisymmetric-tensor"
    assert antisymmetric.coefficients == ((1.0, 0.0),) * 6

    closure = model.closure_contraction_ir(125)
    assert closure == ContractionIR(
        "scalar",
        "scalar",
        "scalar",
        ((1.0, 0.0),),
        "any",
        None,
    )


def test_external_massless_fermions_record_weyl_and_dirac_contractions() -> None:
    particle = CompiledParticleRecord(
        name="psi",
        antiname="psi_bar",
        pdg_code=900_001,
        spin=2,
        color=1,
        mass="ZERO",
        width="ZERO",
        charge=0.0,
        quantum_numbers=(("electric_charge", "0"),),
        ghost_number=0,
        propagating=True,
        goldstoneboson=False,
        propagator=None,
        component_dimension=4,
    )
    antiparticle = replace(
        particle,
        name="psi_bar",
        antiname="psi",
        pdg_code=-900_001,
        source_orientation="antiparticle",
    )

    direct, closure = compile_contraction_records(
        (particle, antiparticle),
        (),
        (),
    )

    by_selector = {record.selector: record.contraction_ir for record in direct}
    assert by_selector[("psi", -1, "psi_bar", 1)].name == "weyl"
    assert by_selector[("psi", 0, "psi_bar", 0)].name == "dirac"
    assert by_selector[("psi", 1, "psi_bar", -1)].name == "weyl"
    assert closure == ()


def test_reachability_remains_coarse_and_dimension_16_is_not_contractible() -> None:
    model = BuiltinSMModel()

    assert model.direct_contraction_possible(1, -1) is True
    assert (
        model.direct_contraction_ir(
            1,
            -1,
            left_chirality=1,
            right_chirality=1,
        )
        is None
    )

    spin_two = 990_016
    spin_two_model = Model(
        name="spin-two-probe",
        particles={
            spin_two: Particle(
                spin_two,
                spin_two,
                spin=5,
                dimension=16,
                color_rep=1,
            )
        },
    )
    assert spin_two_model.direct_contraction_possible(spin_two, spin_two) is False
    assert spin_two_model.direct_contraction_ir(spin_two, spin_two) is None
    assert spin_two_model.closure_contraction_ir(spin_two) is None


def test_generic_model_never_infers_contractions_from_component_dimension() -> None:
    scalar = 990_001
    model = Model(
        name="scalar-probe",
        particles={
            scalar: Particle(
                scalar,
                scalar,
                spin=1,
                dimension=1,
                color_rep=1,
            )
        },
    )

    assert model.direct_contraction_ir(scalar, scalar) is None
    assert model.closure_contraction_ir(scalar) is None
    assert model.direct_contraction_possible(scalar, scalar) is False


def test_external_model_with_empty_records_has_no_dimension_fallback() -> None:
    particle = CompiledParticleRecord(
        name="external_scalar",
        antiname="external_scalar",
        pdg_code=990_002,
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
        component_dimension=1,
    )
    ir = CompiledModelIR(
        name="external-empty-contractions",
        orders=(),
        parameters=(),
        particles=(particle,),
        couplings=(),
        propagators=(),
        vertex_terms=(),
        oriented_kernels=(),
        direct_contractions=(),
        closure_contractions=(),
    )
    model = CompiledUFOModel(
        CompiledModel(
            source={"kind": "json"},
            producer={},
            model={"name": ir.name},
            ir=ir,
            parameter_defaults={},
            capabilities={},
            issues=(),
            phase_timings={},
            conversion_seconds=0.0,
        )
    )

    assert model.dimension(particle.pdg_code) == 1
    assert model.direct_contraction_ir(particle.pdg_code, particle.pdg_code) is None
    assert model.closure_contraction_ir(particle.pdg_code) is None
    assert model.direct_contraction_possible(
        particle.pdg_code,
        particle.pdg_code,
    ) is False


def test_component_contraction_is_coefficient_driven_and_never_truncates() -> None:
    contraction_ir = ContractionIR(
        name="lorentz",
        left_basis="model-left",
        right_basis="model-right",
        coefficients=((2.0, 0.0), (0.0, 1.0)),
        chirality_relation="any",
        metric_signature=None,
    )

    assert _contract_components(contraction_ir, (2.0, 3.0), (5.0, 7.0)) == (
        20.0 + 21.0j
    )
    renamed = replace(contraction_ir, name="model-defined-opaque-contraction")
    assert _contract_components(renamed, (2.0, 3.0), (5.0, 7.0)) == (20.0 + 21.0j)
    with pytest.raises(ValueError, match="requires 2 components, got 3 and 2"):
        _contract_components(contraction_ir, (2.0, 3.0, 11.0), (5.0, 7.0))

    unit_ir = replace(
        contraction_ir,
        coefficients=((1.0, 0.0), (1.0, 0.0)),
    )
    assert _contract_components(
        unit_ir,
        (S("left_0"), S("left_1")),
        (S("right_0"), S("right_1")),
    ) == S("left_0") * S("right_0") + S("left_1") * S("right_1")


class _ClosureChiralityProbe(Model):
    observed_chiralities: tuple[int, int] | None = None

    def vertex_component_expression(
        self,
        kind: int,
        left: Sequence[Any],
        right: Sequence[Any],
        *,
        result_particle_id: int,
        result_chirality: int,
        left_chirality: int = 0,
        right_chirality: int = 0,
        coupling: tuple[Any, Any] = (1.0, 0.0),
        left_momentum: Sequence[Any] | None = None,
        right_momentum: Sequence[Any] | None = None,
    ) -> tuple[Any, ...]:
        del (
            kind,
            result_particle_id,
            result_chirality,
            coupling,
            left_momentum,
            right_momentum,
        )
        self.observed_chiralities = (left_chirality, right_chirality)
        return (left[0] + right[0],)


def test_vertex_closure_forwards_current_chiralities_and_scalar_projection() -> None:
    model = _ClosureChiralityProbe(name="closure-chirality-probe")
    contraction_ir = ContractionIR(
        "scalar",
        "scalar",
        "scalar",
        ((1.0, 0.0),),
        "any",
        None,
    )
    root = {
        "kind": "vertex-closure",
        "left_value_slot": {
            "value_slot_id": 0,
            "component_start": 0,
            "component_stop": 1,
            "dimension": 1,
        },
        "right_value_slot": {
            "value_slot_id": 1,
            "component_start": 1,
            "component_stop": 2,
            "dimension": 1,
        },
        "vertex_kind": 41,
        "vertex_particles": [101, 202, 303],
        "coupling": [1.0, 0.0],
        "color_weight": [1.0, 0.0],
        "contraction": contraction_ir.name,
        "contraction_ir": contraction_ir.to_json_dict(),
    }
    value_slots = {
        0: {"value_slot_id": 0, "dimension": 1, "chirality": -1},
        1: {"value_slot_id": 1, "dimension": 1, "chirality": 1},
    }

    result = _amplitude_root_expression(
        model,
        root,
        value_symbols={0: (2.0,), 1: (3.0,)},
        model_parameter_symbols={},
        value_slots=value_slots,
    )

    assert result == 5.0
    assert model.observed_chiralities == (-1, 1)

    malformed_projection = replace(
        contraction_ir,
        right_basis="lorentz-vector",
    )
    root["contraction_ir"] = malformed_projection.to_json_dict()
    with pytest.raises(ValueError, match="scalar projection IR"):
        _amplitude_root_expression(
            model,
            root,
            value_symbols={0: (2.0,), 1: (3.0,)},
            model_parameter_symbols={},
            value_slots=value_slots,
        )


def test_runtime_and_execution_dtos_emit_full_contraction_ir() -> None:
    model = BuiltinSMModel()
    dag = compile_generic_dag(build_process_ir("d d~ > z"), model=model)
    runtime_schema = build_runtime_schema(dag, model)
    execution_plan = _execution_plan(runtime_schema)
    value_slots = {
        int(slot["value_slot_id"]): slot
        for slot in execution_plan["value_storage"]["value_slots"]
    }

    assert execution_plan["amplitude_stage"]["roots"]
    for root in execution_plan["amplitude_stage"]["roots"]:
        contraction_ir = ContractionIR.from_json_dict(root["contraction_ir"])
        assert root["contraction"] == contraction_ir.name
        assert len(contraction_ir.coefficients) == root["left_value_slot"]["dimension"]
        assert len(contraction_ir.coefficients) == root["right_value_slot"]["dimension"]
        left_value = value_slots[root["left_value_slot"]["value_slot_id"]]
        right_value = value_slots[root["right_value_slot"]["value_slot_id"]]
        assert contraction_ir.left_basis == left_value["propagator"]["basis"]
        assert contraction_ir.right_basis == right_value["propagator"]["basis"]

    blueprint = build_generic_stage_compiler_blueprint(
        dag,
        model=model,
        runtime_schema=runtime_schema,
    )
    assert blueprint.amplitude_stage.expression_ready is True
    assert blueprint.amplitude_stage.blockers == ()


class _OpaqueContractionModel(BuiltinSMModel):
    def direct_contraction_ir(
        self,
        left_particle_id: int,
        right_particle_id: int,
        left_chirality: int = 0,
        right_chirality: int = 0,
    ) -> ContractionIR | None:
        contraction_ir = super().direct_contraction_ir(
            left_particle_id,
            right_particle_id,
            left_chirality=left_chirality,
            right_chirality=right_chirality,
        )
        if contraction_ir is None:
            return None
        return replace(contraction_ir, name="model-owned-opaque")


def test_model_owned_ir_preserves_topology_and_all_root_copy_paths() -> None:
    process = build_process_ir("d d~ > z")
    default = compile_generic_dag(process, model=BuiltinSMModel())
    unpruned = compile_generic_dag(
        process,
        model=BuiltinSMModel(),
        species_reachability_pruning=False,
    )
    opaque = compile_generic_dag(process, model=_OpaqueContractionModel())

    assert (
        len(default.currents),
        len(default.interactions),
        len(default.amplitude_roots),
    ) == (13, 6, 6)
    assert (
        len(unpruned.currents),
        len(unpruned.interactions),
        len(unpruned.amplitude_roots),
    ) == (13, 6, 6)
    assert (
        len(opaque.currents),
        len(opaque.interactions),
        len(opaque.amplitude_roots),
    ) == (13, 6, 6)
    assert all(
        root.contraction == "model-owned-opaque" for root in opaque.amplitude_roots
    )
    assert [
        (root.kind, root.left_id, root.right_id, root.color_sector_id)
        for root in default.amplitude_roots
    ] == [
        (root.kind, root.left_id, root.right_id, root.color_sector_id)
        for root in opaque.amplitude_roots
    ]

    selected = replace(default, amplitude_roots=(default.amplitude_roots[0],))
    copied = prune_dag_to_amplitude_roots(selected)
    assert copied is not selected
    assert copied.amplitude_roots[0].contraction_ir is (
        default.amplitude_roots[0].contraction_ir
    )

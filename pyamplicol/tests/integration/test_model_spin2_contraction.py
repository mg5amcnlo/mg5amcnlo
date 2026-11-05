# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from pathlib import Path

import pytest

from pyamplicol.generation.dag_algorithms import (
    infer_minimal_coupling_order_limits,
)
from pyamplicol.generation.dag_compiler import compile_generic_dag
from pyamplicol.generation.runtime_schema import build_runtime_schema
from pyamplicol.generation.stage_compiler import (
    build_generic_stage_compiler_blueprint,
)
from pyamplicol.models import CompiledUFOModel, compile_model_source
from pyamplicol.processes.model import build_model_process_ir

MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "pyamplicol"
    / "assets"
    / "models"
    / "json"
    / "scalar_gravity"
    / "scalar_gravity.json"
)


@pytest.fixture(scope="module")
def scalar_gravity_model():
    compiled = compile_model_source(MODEL_PATH, use_cache=True)
    return compiled, CompiledUFOModel(compiled)


def test_scalar_gravity_compiles_a_model_owned_spin2_contraction(
    scalar_gravity_model,
) -> None:
    compiled, model = scalar_gravity_model
    graviton = next(
        particle.pdg_code
        for particle in compiled.ir.particles
        if particle.name == "graviton"
    )

    contraction = model.direct_contraction_ir(graviton, graviton)

    assert contraction is not None
    assert contraction.name == "lorentz-rank-2"
    assert contraction.left_basis == contraction.right_basis == "lorentz-rank-2"
    assert len(contraction.coefficients) == 16
    assert model.closure_contraction_ir(graviton) is None


@pytest.mark.parametrize(
    ("process", "external_count", "coupling_order"),
    (
        ("graviton graviton > graviton", 3, 1),
        ("graviton graviton > graviton graviton", 4, 2),
        ("graviton graviton > graviton graviton graviton", 5, 3),
    ),
)
def test_scalar_gravity_pure_graviton_ladder_has_complete_direct_roots(
    scalar_gravity_model,
    process: str,
    external_count: int,
    coupling_order: int,
) -> None:
    compiled, model = scalar_gravity_model
    process_ir = build_model_process_ir(process, compiled.ir)
    limits = infer_minimal_coupling_order_limits(process_ir, model=model)

    dag = compile_generic_dag(
        process_ir,
        model=model,
        max_coupling_orders=limits,
    )

    assert limits == {"GRAV": coupling_order, "QCD": 0, "QED": 0}
    assert dag.interactions
    assert len(dag.amplitude_roots) == 2**external_count
    assert all(root.kind == "direct-contraction" for root in dag.amplitude_roots)
    assert all(
        root.contraction_ir.name == "lorentz-rank-2"
        for root in dag.amplitude_roots
    )
    if external_count == 3:
        runtime_schema = build_runtime_schema(dag, model)
        blueprint = build_generic_stage_compiler_blueprint(
            dag,
            model=model,
            runtime_schema=runtime_schema,
        )
        assert blueprint.expression_ready is True
        assert blueprint.blockers == ()
        assert blueprint.amplitude_stage.expression_ready is True
        assert blueprint.amplitude_stage.blockers == ()

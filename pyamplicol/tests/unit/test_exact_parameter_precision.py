# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from decimal import Decimal, localcontext

import pytest
from symbolica import E

from pyamplicol._internal.physics.symbols import symbols
from pyamplicol.api.errors import ArtifactError
from pyamplicol.models import compiler_symbolica
from pyamplicol.models.compiler_records import (
    _coupling,
    _parameter,
    _resolve_coupling_records,
    _resolve_parameter_records,
)
from pyamplicol.models.prepared_catalog_builder import _candidate_from_payload
from pyamplicol.runtime._normalization_exact import _pi, exact_normalization
from pyamplicol.runtime.symbolica_exact import (
    SymbolicaExactExecutor,
    _ExactExpressionEvaluator,
)


def _resolved_qcd_definitions():
    compiler_symbolica._ensure_symbolica()
    registry = symbols.model("exact_parameter_test")
    records = (
        {
            "name": "aS",
            "nature": "external",
            "parameter_type": "real",
            "value": [0.125, 0],
        },
        {
            "name": "G",
            "nature": "internal",
            "parameter_type": "real",
            "expression": "sqrt(4*pi*UFO::aS)",
        },
    )
    parameters = _resolve_parameter_records(
        tuple(_parameter(record, model_symbols=registry) for record in records),
        registry,
    )
    couplings = _resolve_coupling_records(
        tuple(
            _coupling({"name": name, "expression": expression}, model_symbols=registry)
            for name, expression in (("cubic", "UFO::G"), ("quartic", "UFO::G^2"))
        ),
        parameters,
        registry,
    )
    return registry, parameters, couplings


@pytest.mark.parametrize("precision", (80, 1000))
def test_parameter_derivation_keeps_pi_and_dependent_couplings_exact(precision):
    registry, _, couplings = _resolved_qcd_definitions()
    cubic, quartic = (E(record.resolved_expression) for record in couplings)
    assert (cubic**2 - quartic).expand() == E("0")
    assert E("pi") in quartic.get_all_symbols(False)
    evaluator = _ExactExpressionEvaluator(
        tuple(record.resolved_expression for record in couplings),
        (registry.symbol("aS").to_canonical_string(),),
    )
    with localcontext() as context:
        context.prec = precision
        values = evaluator.evaluate(((Decimal("0.125"), Decimal(0)),), precision)
        assert abs(values[0][0] ** 2 - values[1][0]) < Decimal(10) ** (-precision + 5)
        assert abs(values[1][0] - _pi(precision) / 2) < Decimal(10) ** (-precision + 5)
        assert abs(values[1][0] - Decimal("1.5707963267948966")) > Decimal("1e-18")


def test_compiled_exact_executor_rederives_current_parameters_not_native_cache():
    _registry, parameters, couplings = _resolved_qcd_definitions()
    executor = object.__new__(SymbolicaExactExecutor)
    executor._exact_compiled_model = {
        "name": "exact_parameter_test",
        "parameters": [
            {
                "name": record.name,
                "nature": record.nature,
                "resolved_expression": record.resolved_expression,
            }
            for record in parameters
        ],
        "vertex_terms": [
            {"id": 7, "coupling_expression": couplings[1].resolved_expression}
        ],
    }
    executor._execution = {
        "runtime_schema": {
            "model_parameters": [
                {"name": "aS", "kind": "external_parameter", "parameter_index": 0},
            ]
        },
        "compiled": {
            "model_parameter_evaluator": {
                "outputs": [
                    {
                        "runtime_name": "G",
                        "real_parameter_index": 1,
                        "imag_parameter_index": 2,
                    },
                    {
                        "runtime_name": "derived_coupling_7",
                        "real_parameter_index": 3,
                        "imag_parameter_index": 4,
                    },
                ]
            }
        },
    }
    with localcontext() as context:
        context.prec = 80
        result = executor._derive_model_parameters(
            (Decimal("0.25"), Decimal(999), Decimal(999), Decimal(999), Decimal(999)),
            80,
        )
        assert result[0] == Decimal("0.25")
        assert result[2] == result[4] == 0
        assert abs(result[1] ** 2 - result[3]) < Decimal("1e-75")
        assert abs(result[3] - _pi(80)) < Decimal("1e-75")


@pytest.mark.parametrize("precision", (80, 1000))
def test_real_born_normalization_uses_one_exact_current_coupling(precision):
    schema = ({"name": "normalization.alpha_s_me_check", "parameter_index": 0},)
    # Remove the public final-state symmetry factors exactly, as needed when
    # comparing one local collinear limit with its lower-multiplicity Born ME.
    real = {
        "extensions": {
            "normalization": {
                "average_factor": 256,
                "identical_factor": 24,
                "color_factor": 729,
                "qcd_coupling_power": 4,
                "global_coupling_factor": 999,
            }
        }
    }
    born = {
        "extensions": {
            "normalization": {
                "average_factor": 256,
                "identical_factor": 6,
                "color_factor": 243,
                "qcd_coupling_power": 3,
                "global_coupling_factor": 999,
            }
        }
    }
    with localcontext() as context:
        context.prec = precision
        n_real = exact_normalization(real, (Decimal("0.125"),), precision, schema)
        n_born = exact_normalization(born, (Decimal("0.125"),), precision, schema)
        ratio = n_real * 24 / (n_born * 6)
        assert abs(ratio - _pi(precision) / 2) < Decimal(10) ** (-precision + 5)


@pytest.mark.parametrize("accuracy,expected", (("lc", 81), ("nlc", 1), ("full", 1)))
def test_exact_normalization_counts_color_metric_once(accuracy, expected):
    physics = {
        "color_accuracy": accuracy,
        "extensions": {"normalization": {"color_factor": 81}},
    }
    assert exact_normalization(physics, (), 80) == expected


def test_exact_normalization_does_not_silently_use_missing_metadata():
    with pytest.raises(ArtifactError, match="metadata is absent"):
        exact_normalization({}, (), 80)


def test_prepared_contract_accepts_intrinsic_pi_without_a_runtime_input():
    candidate = _candidate_from_payload(
        "model-parameter", {}, ("pi",), (), ("constant",), proof_class=None
    )
    assert candidate.exact_expressions == ("pi",)
